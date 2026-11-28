from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from omegaconf import open_dict

from dexmani_policy.common.checkpoint import Checkpoint
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.training.data import (
    fit_action_normalizer, fit_observation_normalizer,
)

if TYPE_CHECKING:
    from dexmani_policy.agents.policies.base import Policy


def resolve_rgb_architecture(agent, source):
    name = agent.get("rgb_backbone_name")
    if name not in ("dino", "siglip"):
        return
    rgb = dict(agent.get("rgb_backbone_config") or {})
    if rgb.get("architecture") is not None:
        return
    if source is None:
        raise ValueError("Fresh HF RGB requires initialization.rgb or an explicit architecture")
    from transformers import AutoConfig, SiglipVisionConfig

    config_class = SiglipVisionConfig if name == "siglip" else AutoConfig
    rgb["architecture"] = config_class.from_pretrained(source).to_dict()
    with open_dict(agent):
        agent.rgb_backbone_config = rgb


def inherit_rgb_architecture(agent, saved_agent):
    name = agent.get("rgb_backbone_name")
    saved_name = saved_agent.get("rgb_backbone_name")
    if name != saved_name:
        raise ValueError(f"RGB backbone differs from initialization policy: current={name!r}, checkpoint={saved_name!r}")
    if name not in ("dino", "dinov3", "siglip"):
        return
    rgb = dict(agent.get("rgb_backbone_config") or {})
    if rgb.get("architecture") is not None:
        return
    rgb["architecture"] = saved_agent["rgb_backbone_config"]["architecture"]
    with open_dict(agent):
        agent.rgb_backbone_config = rgb


def build_policy(cfg, dataset, field_dims, *, checkpoint: Checkpoint | None = None, resuming: bool = False):
    if checkpoint is not None and field_dims != cfg.normalization_dims:
        raise ValueError("Checkpoint data/model mismatch: normalization_dims")
    cls = cast("type[Policy]", hydra.utils.get_class(cfg.agent._target_))
    model = cls.build_for_training(cfg, dataset, field_dims, checkpoint=checkpoint, resuming=resuming)
    return model.to(device=cfg.training.device, dtype=torch.float32)


def create_policy(cfg, dataset, *, checkpoint=None):
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    if dataset.episode_store[dataset.action_key].shape[-1] != model.control_action_dim:
        raise ValueError("Dataset primary action width must match policy control output")
    if checkpoint is not None:
        return model

    rgb_encoder = getattr(model.obs_encoder, "rgb_encoder", None)
    if rgb_encoder is None:
        return model
    rgb_source = cfg.initialization.get("rgb")
    if rgb_source is not None:
        rgb_encoder.initialize_weights(rgb_source)
    elif getattr(rgb_encoder, "tune_mode", None) in {"freeze", "lora"}:
        raise ValueError(
            "Fresh RGB training with tune_mode=freeze or lora would freeze a random backbone; "
            "set initialization.rgb to pretrained weights or "
            "agent.rgb_backbone_config.tune_mode=full to train from scratch"
        )
    return model


def load_encoder_prior(cfg, encoder, fields):
    source = cfg.initialization.encoder
    payload = torch.load(Path(source).expanduser(), map_location="cpu", weights_only=True)
    encoder.load_state_dict(payload["encoder_state"], strict=True)
    with open_dict(cfg.normalization):
        for key in fields:
            cfg.normalization[key] = payload["normalization"][key]
    inherited_fields = tuple(key for key in fields if cfg.normalization[key] != "identity")
    return payload["normalizer_state"], inherited_fields


def initialize_policy(model, cfg, dataset, field_dims, *, checkpoint=None,
                      hand_affine=None, pretrained_state=None, inherited_fields=()):
    feature_shapes = normalization_feature_shapes(
        field_dims, cfg.normalization, required_identity_fields=model.required_identity_fields,
        required_limits_fields=model.required_limits_fields,
    )
    model.configure_normalizer(field_dims, feature_shapes)
    if checkpoint is not None:
        model.load_weights(checkpoint.model_state, "cpu")
        return model

    fit_observation_normalizer(
        dataset, model.normalizer, cfg.normalization,
        initial_state=pretrained_state, inherited_fields=inherited_fields,
    )
    fit_action_normalizer(
        dataset, model.normalizer, hand_affine=hand_affine, center_rotvec=model.center_rotvec,
    )
    return model
