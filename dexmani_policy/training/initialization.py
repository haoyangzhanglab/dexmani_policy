from pathlib import Path

import hydra
import torch
from omegaconf import open_dict

from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.training.data import (
    fit_action_normalizer, fit_observation_normalizer, inherit_training_data, iter_validated_actions,
)


def resolve_rgb_architecture(agent, source):
    name = agent.rgb_backbone_name
    if name not in ("dino", "siglip"):
        return
    rgb = dict(agent.get("rgb_backbone_config") or {})
    if rgb.get("architecture") is None:
        if source is None:
            raise ValueError("Fresh HF RGB requires initialization.rgb or an explicit architecture")
        from transformers import AutoConfig, SiglipVisionConfig

        config_class = SiglipVisionConfig if name == "siglip" else AutoConfig
        rgb["architecture"] = config_class.from_pretrained(source).to_dict()
    with open_dict(agent):
        agent.rgb_backbone_config = rgb


def inherit_rgb_architecture(agent, saved_agent):
    if agent.rgb_backbone_name != saved_agent["rgb_backbone_name"]:
        raise ValueError("RGB backbone differs from initialization policy")
    rgb = dict(agent.get("rgb_backbone_config") or {})
    if rgb.get("architecture") is None:
        architecture = saved_agent["rgb_backbone_config"].get("architecture")
        if architecture is not None:
            rgb["architecture"] = architecture
    with open_dict(agent):
        agent.rgb_backbone_config = rgb


def prepare_warm_start(method, agent, saved_agent):
    if method in {"dp", "sat", "dqrise", "dexlatent"}:
        inherit_rgb_architecture(agent, saved_agent)
    if method == "dqrise" and agent.get("num_codes") is None:
        with open_dict(agent):
            agent.num_codes = saved_agent["num_codes"]
    if method == "adaptac" and agent.tactile_encoder._target_ != saved_agent["tactile_encoder"]["_target_"]:
        raise ValueError("Tactile encoder differs from initialization policy")


def build_policy(cfg, dataset, field_dims, *, checkpoint=None):
    method, initialization = cfg.policy_name, cfg.initialization
    pretrained_state, hand_affine = None, None
    inherited_fields: tuple[str, ...] = ()
    if checkpoint is not None and field_dims != cfg.normalization_dims:
        raise ValueError("Checkpoint data/model mismatch: normalization_dims")
    if checkpoint is None and method == "dqrise":
        from dexmani_policy.agents.hand_prior.codebook import load_codebook

        if initialization.codebook is None:
            raise ValueError("Fresh DQ-RISE training requires initialization.codebook")
        codebook, hand_affine, records = load_codebook(initialization.codebook)
        inherit_training_data(cfg, dataset, records, initialization.codebook)
        with open_dict(cfg.agent):
            cfg.agent.num_codes = codebook.num_codes

    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    if dataset.episode_store[dataset.action_key].shape[-1] != model.control_action_dim:
        raise ValueError("Dataset primary action width must match policy control output")
    if checkpoint is None:
        if method in {"dp", "sat", "dqrise", "dexlatent"}:
            model.obs_encoder.rgb_encoder.initialize_weights(initialization.rgb)
        elif method == "r3d":
            model.obs_encoder.pc_encoder.initialize_weights(initialization.uni3d)
        elif method in {"afro", "cordvip"} and initialization.encoder is not None:
            payload = torch.load(
                Path(initialization.encoder).expanduser(), map_location="cpu", weights_only=True,
            )
            inherit_training_data(cfg, dataset, payload["training_data"], initialization.encoder)
            fields: tuple[str, ...]
            if method == "afro":
                encoder, fields = model.obs_encoder.pc_encoder, ("point_cloud",)
            else:
                encoder = model.obs_encoder.interaction_encoder
                fields = ("hand_pointcloud", "object_pointcloud", "joint_state")
            encoder.load_state_dict(payload["encoder_state"], strict=True)
            with open_dict(cfg.normalization):
                for key in fields:
                    cfg.normalization[key] = payload["normalization"][key]
            inherited_fields = tuple(key for key in fields if cfg.normalization[key] != "identity")
            pretrained_state = payload["normalizer_state"]
            del payload
        if method == "dqrise":
            model.hand_codebook = codebook
        elif method == "dexlatent":
            hand_affine = model.hand_prior.initialize_weights(initialization.dexlatent)

    feature_shapes = normalization_feature_shapes(
        field_dims, cfg.normalization, required_identity_fields=model.required_identity_fields,
        required_limits_fields=model.required_limits_fields,
    )
    model.configure_normalizer(field_dims, feature_shapes)
    if checkpoint is not None:
        if method == "dqrise":
            for _ in iter_validated_actions(dataset):
                pass
        model.load_weights(checkpoint.model_state, cfg.training.device)
        return model

    fit_observation_normalizer(
        dataset, model.normalizer, cfg.normalization,
        initial_state=pretrained_state, inherited_fields=inherited_fields,
    )
    fit_action_normalizer(dataset, model.normalizer, hand_affine=hand_affine)
    if method == "dexlatent":
        # Latent limits depend on the hand affine inherited from its prior.
        latents = (model._encode_action_parts(chunk)[1] for chunk in dataset.iter_normalization_data("action"))
        model.normalizer.fit_field("hand_latent", latents)
    model.to(device=cfg.training.device, dtype=torch.float32)
    return model
