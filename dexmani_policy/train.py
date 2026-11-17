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
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.training.config import (
    assemble_policy_config,
    check_fresh_output,
    normalize_output_dir,
    register_resolvers,
    validate_config,
)
from dexmani_policy.training.data import build_dataset, record_data_processing
from dexmani_policy.training.policy_init.common import (
    configure_normalizer,
    fit_normalizers,
    instantiate_policy,
)
from dexmani_policy.training.trainer import PolicyTrainer


@hydra.main(version_base=None, config_path="configs")
def _train(cfg):
    cfg = assemble_policy_config(cfg)
    run_training(cfg, max_updates=cfg.get("max_updates"))


def _build_fresh_policy(cfg, dataset, field_dims):
    method = cfg.policy_name
    if method == "dp":
        from dexmani_policy.training.policy_init.dp import build_fresh_policy
    elif method == "sat":
        from dexmani_policy.training.policy_init.sat import build_fresh_policy
    elif method == "dqrise":
        from dexmani_policy.training.policy_init.dqrise import build_fresh_policy
    elif method == "dexlatent":
        from dexmani_policy.training.policy_init.dexlatent import build_fresh_policy
    elif method == "r3d":
        from dexmani_policy.training.policy_init.r3d import build_fresh_policy
    elif method == "afro":
        from dexmani_policy.training.policy_init.afro import build_fresh_policy
    elif method == "cordvip":
        from dexmani_policy.training.policy_init.cordvip import build_fresh_policy
    else:
        model = instantiate_policy(cfg.agent, dataset)
        fit_normalizers(model, dataset, cfg.normalization, field_dims)
        return model
    return build_fresh_policy(
        cfg.agent, cfg.initialization, cfg.normalization, dataset, field_dims,
    )


def _prepare_warm_start(method, agent, saved_agent):
    if method == "dqrise":
        from dexmani_policy.training.policy_init.dqrise import prepare_warm_start
    elif method == "adaptac":
        from dexmani_policy.training.policy_init.adaptac import prepare_warm_start
    elif method in {"dp", "sat", "dexlatent"}:
        from dexmani_policy.training.policy_init.rgb import (
            inherit_rgb_architecture as prepare_warm_start,
        )
    else:
        return
    prepare_warm_start(agent, saved_agent)


def _prepare_policy(cfg, dataset, field_dims, *, checkpoint=None):
    if checkpoint is None:
        model = _build_fresh_policy(cfg, dataset, field_dims)
    else:
        if field_dims != cfg.normalization_dims:
            raise ValueError("Checkpoint data/model mismatch: normalization_dims")
        model = instantiate_policy(cfg.agent, dataset)
        configure_normalizer(model, cfg.normalization, field_dims)
        # Assignment replaces Parameters; the optimizer must be constructed afterwards.
        model.load_state_dict(checkpoint.model_state, strict=True, assign=True)
    model.to(device=cfg.training.device, dtype=torch.float32)
    if checkpoint is not None:
        model.normalizer.validate()
    return model


def build_trainer(cfg, *, checkpoint=None):
    cfg = copy.deepcopy(cfg)
    normalize_output_dir(cfg)
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
            cfg.initialization.policy = None
        _prepare_warm_start(cfg.policy_name, cfg.agent, saved["agent"])
    validate_config(cfg)
    if checkpoint is None:
        check_fresh_output(cfg.workspace.output_dir)
    if model_checkpoint is None and cfg.policy_name in {"dp", "sat", "dqrise", "dexlatent"}:
        from dexmani_policy.training.policy_init.rgb import resolve_rgb_architecture

        resolve_rgb_architecture(cfg.agent, cfg.initialization.rgb)
    set_seed(cfg.training.seed + (checkpoint.global_step if checkpoint is not None else 0))
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    dataset, data_description = build_dataset(cfg)
    record_data_processing(cfg, dataset)
    if checkpoint is not None:
        data_dt = data_description["control_dt_s"]
        policy_dt = cfg.control_dt_s
        if not np.isclose(data_dt, policy_dt, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Resume control_dt_s mismatch: dataset dt={data_dt} s, "
                f"checkpoint control_dt_s={policy_dt} s"
            )
    with open_dict(cfg):
        cfg.zarr_path = dataset.episode_store.path
        cfg.dataset.zarr_path = cfg.zarr_path
        if checkpoint is None:
            cfg.control_dt_s = data_description["control_dt_s"]
    model = _prepare_policy(cfg, dataset, data_description["field_dims"], checkpoint=model_checkpoint)
    if checkpoint is None:
        with open_dict(cfg):
            cfg.pop("max_updates", None)
            cfg.normalization_dims = model.normalization_dims
            cfg.agent.update(model.effective_config)
        OmegaConf.resolve(cfg)
    trainer = PolicyTrainer(cfg, model, dataset, checkpoint=checkpoint)
    if cfg.policy_name == "afro":
        from dexmani_policy.training.policy_init.afro import log_initialization

        log_initialization(
            model, cfg.initialization.encoder, source,
            checkpoint.global_step if checkpoint is not None else None,
        )
    return trainer


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
