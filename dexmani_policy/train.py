"""Train a policy on one device, optionally restoring the full training state."""

import copy
import os
from contextlib import closing
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import set_project_root
from dexmani_policy.common.normalizer import normalization_feature_shapes, validate_normalizer_state
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.common.config import (
    assemble_policy_config, prepare_training, register_resolvers,
    validate_normalization, validate_config,
)
from dexmani_policy.training.common.data import build_dataset, fit_normalizer
from dexmani_policy.training.trainer import PolicyTrainer


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    cfg = assemble_policy_config(cfg)
    run_training(cfg, max_updates=cfg.get("max_updates"))


def _prepare_policy(cfg, dataset, field_dims, *, checkpoint=None):
    method = cfg.policy_name
    if checkpoint is not None and field_dims != cfg.normalization_dims:
        raise ValueError("Checkpoint data/model mismatch: normalization_dims")

    codebook = hand_affine = None
    if checkpoint is None and method == "dqrise":
        from dexmani_policy.agents.hand_prior.codebook import load_codebook

        source = cfg.initialization.codebook
        if source is None:
            raise ValueError("Fresh DQ-RISE training requires initialization.codebook")
        codebook, hand_affine = load_codebook(source)
        with open_dict(cfg.agent):
            cfg.agent.num_codes = codebook.num_codes
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    if dataset.episode_store[dataset.action_key].shape[-1] != model.control_action_dim:
        raise ValueError("Dataset primary action width must match policy control output")

    if checkpoint is None:
        if method in {"dp", "sat", "dqrise", "dexlatent"}:
            model.obs_encoder.rgb_encoder.initialize_weights(cfg.initialization.rgb)
        elif method == "r3d":
            model.obs_encoder.pc_encoder.initialize_weights(cfg.initialization.uni3d)
        pretrained_state = None
        inherited_fields: tuple[str, ...] = ()
        if method == "dqrise":
            model.hand_codebook = codebook
        elif method == "dexlatent":
            hand_affine = model.hand_prior.initialize_weights(cfg.initialization.dexlatent)
        elif method in {"afro", "cordvip"}:
            source = cfg.initialization.encoder
            if source is not None:
                payload = torch.load(Path(source).expanduser(), map_location="cpu", weights_only=True)
                encoder = model.obs_encoder.pc_encoder if method == "afro" else model.obs_encoder.interaction_encoder
                encoder.load_state_dict(payload["encoder_state"], strict=True)
                pretrained_state = payload["normalizer_state"]
                inherited_fields = (("point_cloud",) if method == "afro" else
                                    ("hand_pointcloud", "object_pointcloud", "joint_state"))
                with open_dict(cfg.normalization):
                    for key in inherited_fields:
                        cfg.normalization[key] = payload["normalization"][key]
                inherited_fields = tuple(key for key in inherited_fields if cfg.normalization[key] == "limits")
                del payload
    validate_normalization(cfg, field_dims)
    feature_shapes = normalization_feature_shapes(field_dims, cfg.normalization)
    model.configure_normalizer(field_dims, feature_shapes)
    if checkpoint is None:
        fit_normalizer(
            dataset, model.normalizer, feature_shapes, initial_state=pretrained_state,
            inherited_fields=inherited_fields, hand_affine=hand_affine,
        )
        print(f"[Normalizer] train source frames; inherited fields={list(inherited_fields)}")
        if method == "dexlatent":
            model.fit_latent_normalizer(dataset)
    else:
        # Assignment replaces Parameters; the optimizer must be constructed afterwards.
        model.load_state_dict(checkpoint.model_state, strict=True, assign=True)
    model.to(device=cfg.training.device, dtype=torch.float32)
    if checkpoint is not None:
        validate_normalizer_state(model.normalizer)
    return model


def build_trainer(cfg, *, checkpoint=None):
    cfg = copy.deepcopy(cfg)
    source = cfg.initialization.policy
    if checkpoint is not None and source is not None:
        raise ValueError("initialization.policy and resume are mutually exclusive")
    model_checkpoint = checkpoint
    if source is not None:
        model_checkpoint = load_checkpoint(Path(source).expanduser())
        saved = model_checkpoint.effective_config
        if cfg.agent._target_ != saved["agent"]["_target_"] or cfg.action_key != saved["action_key"]:
            raise ValueError("Policy initialization requires the same policy and action semantics")
        with open_dict(cfg):
            cfg.normalization = saved["normalization"]
            cfg.normalization_dims = saved["normalization_dims"]
            if cfg.policy_name in {"dp", "sat", "dqrise", "dexlatent"}:
                if cfg.agent.rgb_backbone_name != saved["agent"]["rgb_backbone_name"]:
                    raise ValueError("RGB backbone differs from initialization policy")
                rgb = dict(cfg.agent.get("rgb_backbone_config") or {})
                if rgb.get("architecture") is None:
                    architecture = saved["agent"]["rgb_backbone_config"].get("architecture")
                    if architecture is not None:
                        rgb["architecture"] = architecture
                cfg.agent.rgb_backbone_config = rgb
            if cfg.policy_name == "dqrise" and cfg.agent.get("num_codes") is None:
                cfg.agent.num_codes = saved["agent"]["num_codes"]
            if cfg.policy_name == "adaptac":
                tactile = cfg.agent.tactile_encoder
                saved_tactile = saved["agent"]["tactile_encoder"]
                if tactile._target_ != saved_tactile["_target_"]:
                    raise ValueError("Tactile encoder differs from initialization policy")
                if "region_ids" in saved_tactile and tactile.get("region_ids") is None:
                    tactile.region_ids = saved_tactile["region_ids"]
            cfg.initialization.policy = None
    if checkpoint is None:
        prepare_training(cfg)
    else:
        validate_config(cfg)
    set_seed(cfg.training.seed + (checkpoint.global_step if checkpoint is not None else 0))
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    dataset, data_description = build_dataset(cfg)
    if checkpoint is not None:
        data_dt = data_description["deployment_metadata"]["control_dt_s"]
        policy_dt = cfg.deployment_metadata.control_dt_s
        if not np.isclose(data_dt, policy_dt, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Resume control_dt_s mismatch: dataset dt={data_dt} s, "
                f"checkpoint control_dt_s={policy_dt} s"
            )
    with open_dict(cfg):
        cfg.zarr_path = dataset.episode_store.path
        cfg.dataset.zarr_path = cfg.zarr_path
        if checkpoint is None:
            cfg.deployment_metadata = data_description["deployment_metadata"]
    model = _prepare_policy(cfg, dataset, data_description["field_dims"], checkpoint=model_checkpoint)
    if checkpoint is None:
        with open_dict(cfg):
            cfg.pop("max_updates", None)
            cfg.normalization_dims = model.normalization_dims
            cfg.agent.update(model.effective_config)
        OmegaConf.resolve(cfg)
    return PolicyTrainer(cfg, model, dataset, checkpoint=checkpoint)


def run_trainer(trainer, *, max_updates=None):
    with closing(trainer):
        trainer.compile_models()
        trainer.train(max_updates=max_updates)
        return trainer.global_step


def run_training(cfg, *, max_updates=None):
    return run_trainer(build_trainer(cfg), max_updates=max_updates)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    register_resolvers()
    set_project_root()
    _train()


if __name__ == "__main__":
    main()
