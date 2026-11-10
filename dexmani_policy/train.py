"""Train a policy on one device, optionally restoring the full training state."""

import copy
import os
from contextlib import closing
from pathlib import Path

import hydra
import torch
from omegaconf import open_dict

from dexmani_policy.common.config import set_project_root
from dexmani_policy.common.normalizer import normalization_feature_shapes, validate_normalizer_state
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.common.config import (
    assemble_effective_config, assemble_policy_config, prepare_training, register_resolvers,
    validate_normalization,
)
from dexmani_policy.training.common.data import build_dataset, fit_normalizer
from dexmani_policy.training.trainer import PolicyTrainer


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    cfg = assemble_policy_config(cfg)
    run_training(cfg, max_updates=cfg.get("max_updates"))


def _prepare_policy(cfg, dataset, data_description, *, checkpoint=None):
    method = cfg.policy_name
    if checkpoint is not None:
        current_hoi = data_description["deployment_metadata"].get("hoi_pcd")
        saved_hoi = checkpoint.effective_config["deployment_metadata"].get("hoi_pcd")
        if current_hoi != saved_hoi:
            raise ValueError("Resume data/model mismatch: HOI task/template")
        if data_description["field_dims"] != cfg.normalization_dims:
            raise ValueError("Resume data/model mismatch: normalization_dims")

    codebook = hand_affine = None
    if checkpoint is None and method == "dqrise":
        from dexmani_policy.agents.hand_prior.codebook import load_codebook, validate_hand_coordinates

        source = cfg.initialization.codebook
        if source is None:
            raise ValueError("Fresh DQ-RISE training requires initialization.codebook")
        validate_hand_coordinates(dataset.episode_store.attrs)
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
        inherited_fields = ()
        if method == "dqrise":
            model.hand_codebook = codebook
        elif method == "dexlatent":
            hand_affine = model.hand_prior.initialize_weights(cfg.initialization.dexlatent)
        elif method in {"afro", "cordvip"}:
            source = cfg.initialization.encoder
            if source is None and method == "afro" and model.freeze_visual:
                raise ValueError("AFRO frozen visual requires initialization.encoder; scratch requires freeze_visual=false")
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
    validate_normalization(cfg)
    model.configure_normalizer(
        data_description["field_dims"],
        normalization_feature_shapes(data_description["field_dims"], cfg.normalization),
    )
    if checkpoint is None:
        fit_normalizer(
            dataset, cfg.normalization, model.normalizer, initial_state=pretrained_state,
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
    if checkpoint is None:
        cfg = prepare_training(cfg)
    set_seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    dataset, data_description = build_dataset(cfg)
    model = _prepare_policy(cfg, dataset, data_description, checkpoint=checkpoint)
    cfg = assemble_effective_config(cfg, data_description, model)
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
