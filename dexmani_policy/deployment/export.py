"""Export a selected checkpoint as a self-contained deployment artifact.

The checkpoint owns model construction, preprocessing, normalization and tensor
ordering. Experiment configuration supplies identity and inference defaults.
Export validates plain metadata and reloads safely before atomic publication;
model restoration and warmup belong to the deployment runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.obs_encoder.rgb.image_processor import (
    IMAGE_PROCESSOR_PRESETS,
)
from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
    TrainCheckpoint,
    make_normalization_contract,
    parse_normalization_contract,
)
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.datasets.base_dataset import DEFAULT_RGB_KEEP_UINT8
from dexmani_policy.datasets.real_policy_contract import (
    RealPolicyContractError,
    validate_observation_field_list,
)
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DeploymentContractError,
    parse_deployment_contract,
)


class DeploymentExportError(RuntimeError):
    """Base error for an invalid or failed deployment export."""


class InvalidExperimentError(DeploymentExportError):
    pass


class InvalidCheckpointError(DeploymentExportError):
    pass


class UnsupportedPolicyError(DeploymentExportError):
    pass


class ArtifactPublicationError(DeploymentExportError):
    pass


class ArtifactVerificationError(DeploymentExportError):
    pass


@dataclass(frozen=True)
class ExportReceipt:
    checkpoint_path: Path
    selector_path: Path
    checkpoint_selector: str


@dataclass(frozen=True)
class _CheckpointDeploymentSource:
    """The trained deployment semantics owned by one training checkpoint."""

    agent: dict[str, Any]
    agent_config: dict[str, Any]
    dataset: dict[str, Any]
    normalization_contract: dict[str, Any]
    observation_fields: tuple[str, ...]
    deployment_data_semantics: dict[str, Any]


@dataclass(frozen=True)
class _SelectedInferenceSettings:
    use_ema: bool
    denoise_steps: int


def _require_checkpoint_under_directory(candidate: Path, checkpoint_dir: Path) -> Path:
    try:
        resolved_directory = checkpoint_dir.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise InvalidCheckpointError(
            f"cannot resolve checkpoint path: {candidate}"
        ) from exc
    try:
        resolved_candidate.relative_to(resolved_directory)
    except ValueError as exc:
        raise InvalidCheckpointError(
            f"checkpoint resolves outside experiment/checkpoints: {candidate}"
        ) from exc
    if not resolved_candidate.is_file():
        raise InvalidCheckpointError(f"checkpoint is not a file: {candidate}")
    return resolved_candidate


def _resolve_checkpoint(experiment_dir: Path, selector: str) -> Path:
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise InvalidExperimentError(
            f"checkpoint directory not found: {checkpoint_dir}"
        )

    candidate: Path
    if selector == "best":
        from dexmani_policy.training.eval_utils import read_best_ckpt_json

        try:
            record = read_best_ckpt_json(experiment_dir)
        except (OSError, ValueError) as exc:
            raise InvalidCheckpointError(
                f"invalid best_ckpt.json: {experiment_dir / 'best_ckpt.json'}"
            ) from exc
        candidate = experiment_dir / record["ckpt_relpath"]
    elif selector == "latest":
        candidate = checkpoint_dir / "latest.pt"
    elif selector.endswith("pct"):
        from dexmani_policy.training.eval_utils import discover_milestone_checkpoints

        try:
            percentage = int(selector.removesuffix("pct"))
        except ValueError as exc:
            raise InvalidCheckpointError(
                f"invalid milestone selector: {selector!r}"
            ) from exc
        matches = [
            item.path
            for item in discover_milestone_checkpoints(experiment_dir)
            if item.pct == percentage
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"milestone selector {selector!r} resolved {len(matches)} checkpoints"
            )
        candidate = matches[0]
    else:
        raw = Path(selector)
        candidate = raw if raw.is_absolute() else checkpoint_dir / raw

    return _require_checkpoint_under_directory(candidate, checkpoint_dir)


def _load_config(experiment_dir: Path) -> dict[str, Any]:
    """Load the resolved experiment config for identity and the inference recipe.

    Deployment reads only ``policy_name`` / ``task_name`` and ``eval`` from
    here.  The ``agent`` / ``dataset`` / ``normalization`` sections are
    deliberately *not* consulted, so editing them after training cannot change
    what an existing checkpoint deploys as — that semantics belongs to
    :func:`_parse_checkpoint_deployment_source`.
    """
    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise InvalidExperimentError(
            f"resolved experiment config not found: {config_path}"
        )
    register_resolvers()
    try:
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        plain = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except Exception as exc:
        raise InvalidExperimentError("experiment config is not fully resolved") from exc
    if type(plain) is not dict:
        raise InvalidExperimentError("experiment config must be a resolved mapping")
    return cast(dict[str, Any], plain)


def _load_training_checkpoint(path: Path) -> TrainCheckpoint:
    try:
        checkpoint = CheckpointStore(path.parent).load(path)
    except Exception as exc:
        raise InvalidCheckpointError(
            f"cannot load simple.v3 checkpoint: {path}"
        ) from exc
    if checkpoint.epoch < 0 or checkpoint.global_step < 0:
        raise InvalidCheckpointError(
            "checkpoint epoch/global_step must be non-negative"
        )
    return checkpoint


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise InvalidExperimentError(f"{label} must be a positive integer")
    return value


def _checkpoint_observation_fields(
    dataset: dict[str, Any], agent_config: dict[str, Any]
) -> list[str]:
    """Resolve the artifact's observation modalities from checkpoint-saved semantics."""
    if any(key in agent_config for key in ("text_encoder_model", "task_texts")):
        raise UnsupportedPolicyError("dynamic task-text deployment is unsupported")
    modalities = dataset.get("sensor_modalities")
    if type(modalities) is not list or any(
        type(item) is not str for item in modalities
    ):
        raise UnsupportedPolicyError(
            "checkpoint dataset.sensor_modalities must be an explicit string list"
        )
    try:
        return validate_observation_field_list(modalities)
    except RealPolicyContractError as exc:
        raise UnsupportedPolicyError(str(exc)) from exc


def _build_policy_spec(source: _CheckpointDeploymentSource) -> dict[str, Any]:
    from dexmani_policy.deployment.contract import PolicySpec

    saved = source.deployment_data_semantics
    observations = {}
    for name in source.observation_fields:
        field = saved["observation_fields"][name]
        observations[name] = {
            "shape": [None, None, 3] if name == "rgb" else field["shape"],
            "dtype": field["dtype"],
            "ordering": field.get("ordering", {}),
        }
    public = {
        "observations": observations,
        "joint_names": saved.get("joint_names"),
        "pointcloud_config": saved.get("pointcloud_config"),
        "n_obs_steps": source.agent["n_obs_steps"],
        "n_action_steps": source.agent["n_action_steps"],
        "control_dt_s": saved["dt"],
        "action_mode": "eef" if source.agent["action_key"] == "action_ee" else "joint",
    }
    return PolicySpec.from_dict(public).to_dict()


def _parse_checkpoint_deployment_source(
    checkpoint: TrainCheckpoint,
) -> _CheckpointDeploymentSource:
    resume = checkpoint.resume_contract
    if type(resume) is not dict:
        raise InvalidCheckpointError("checkpoint resume_contract must be a plain dict")
    agent = resume.get("agent")
    agent_config = resume.get("agent_config")
    dataset = resume.get("dataset")
    for label, value in (
        ("agent", agent),
        ("agent_config", agent_config),
        ("dataset", dataset),
    ):
        if type(value) is not dict or not value:
            raise InvalidCheckpointError(
                f"checkpoint resume_contract.{label} must be a non-empty plain dict"
            )
    train = _parse_checkpoint_action_contract(agent)
    observation_fields = _checkpoint_observation_fields(dataset, agent_config)
    normalization_contract = _parse_checkpoint_normalization_contract(
        agent, observation_fields
    )
    if "deployment_data_semantics" not in resume:
        raise InvalidCheckpointError(
            "checkpoint predates deployment_data_semantics and cannot be safely "
            "exported for Real deployment"
        )
    data_semantics = resume["deployment_data_semantics"]
    if data_semantics is None:
        raise UnsupportedPolicyError(
            "checkpoint dataset has no single zarr_path; dynamic/multi-task "
            "datasets are unsupported"
        )
    if type(data_semantics) is not dict or not data_semantics:
        raise InvalidCheckpointError(
            "checkpoint resume_contract.deployment_data_semantics must be a "
            "non-empty plain dict"
        )
    return _CheckpointDeploymentSource(
        agent=_require_plain_metadata(train, "resume_contract.agent"),
        agent_config=_require_plain_metadata(
            agent_config, "resume_contract.agent_config"
        ),
        dataset=_require_plain_metadata(dataset, "resume_contract.dataset"),
        normalization_contract=normalization_contract,
        observation_fields=tuple(observation_fields),
        deployment_data_semantics=_require_plain_metadata(
            data_semantics, "resume_contract.deployment_data_semantics"
        ),
    )


def _parse_checkpoint_action_contract(agent: dict[str, Any]) -> dict[str, Any]:
    """Strictly parse the checkpoint's own action/window contract for self-consistency."""
    required = (
        "n_obs_steps",
        "n_action_steps",
        "action_dim",
        "horizon",
        "action_key",
        "tcp_dim",
        "hand_dim",
        "control_action_dim",
        "use_aux_ee",
    )
    missing = sorted(name for name in required if name not in agent)
    if missing:
        raise InvalidCheckpointError(
            f"checkpoint resume_contract.agent is missing {missing}"
        )
    action_key = agent["action_key"]
    if action_key not in {"action", "action_ee"}:
        raise InvalidCheckpointError(
            f"checkpoint action_key is unsupported: {action_key!r}"
        )
    use_aux_ee = agent["use_aux_ee"]
    if type(use_aux_ee) is not bool:
        raise InvalidCheckpointError("checkpoint use_aux_ee must be bool")
    train = {
        name: _require_checkpoint_positive_int(agent[name], f"checkpoint {name}")
        for name in ("n_obs_steps", "n_action_steps", "action_dim", "horizon")
    }
    control_action_dim = _require_checkpoint_positive_int(
        agent["control_action_dim"], "checkpoint control_action_dim"
    )
    if train["n_obs_steps"] - 1 + train["n_action_steps"] > train["horizon"]:
        raise InvalidCheckpointError(
            "checkpoint observation/action window exceeds horizon"
        )

    expected_control = 21 if action_key == "action_ee" else 19
    if use_aux_ee:
        # The only supported auxiliary layout is joint19 + ee9 predicted
        # together while control stays joint-only.
        if (
            action_key != "action"
            or train["action_dim"] != 28
            or control_action_dim != 19
        ):
            raise InvalidCheckpointError(
                "checkpoint use_aux_ee requires the joint19_ee9 action layout"
            )
    else:
        if train["action_dim"] != expected_control:
            raise InvalidCheckpointError(
                "checkpoint action_dim does not match action_key"
            )
        if control_action_dim != train["action_dim"]:
            raise InvalidCheckpointError(
                "checkpoint control_action_dim must equal action_dim without use_aux_ee"
            )

    tcp_dim = agent["tcp_dim"]
    hand_dim = agent["hand_dim"]
    if tcp_dim is None or hand_dim is None:
        if tcp_dim is not None or hand_dim is not None:
            raise InvalidCheckpointError(
                "checkpoint tcp_dim/hand_dim must both be set or both be null"
            )
    else:
        tcp_dim = _require_checkpoint_positive_int(tcp_dim, "checkpoint tcp_dim")
        hand_dim = _require_checkpoint_positive_int(hand_dim, "checkpoint hand_dim")
        if tcp_dim + hand_dim != train["action_dim"]:
            raise InvalidCheckpointError(
                "checkpoint tcp_dim + hand_dim must equal action_dim"
            )
    return {
        **train,
        "action_key": action_key,
        "tcp_dim": tcp_dim,
        "hand_dim": hand_dim,
        "control_action_dim": control_action_dim,
        "use_aux_ee": use_aux_ee,
    }


def _parse_checkpoint_normalization_contract(
    agent: dict[str, Any], observation_fields: list[str]
) -> dict[str, Any]:
    """Parse the checkpoint's versioned normalization contract, nothing else.

    The fitted ``scale``/``offset`` in this checkpoint were produced under this
    saved spec, so it is the only correct normalization metadata for the
    resulting artifact.  Exact field coverage is enforced against the
    checkpoint-derived observation fields.
    """
    saved_contract = agent.get("normalization")
    if type(saved_contract) is not dict:
        raise InvalidCheckpointError(
            "checkpoint resume_contract.agent.normalization must be a plain dict"
        )
    try:
        saved_spec = parse_normalization_contract(
            saved_contract, observation_fields=observation_fields
        )
    except ValueError as exc:
        raise InvalidCheckpointError(
            f"checkpoint normalization contract is invalid: {exc}"
        ) from exc
    return make_normalization_contract(saved_spec)


def _require_checkpoint_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise InvalidCheckpointError(f"{label} must be a positive integer")
    return value


def _canonicalize_state_dict(value: Any, label: str) -> dict[str, torch.Tensor]:
    # Eager nn.Module.state_dict() returns OrderedDict; compiled/DDP key
    # canonicalization may produce dict. Both are current training outputs.
    if type(value) not in (dict, OrderedDict) or not value:
        raise InvalidCheckpointError(f"{label} must be a non-empty state_dict")
    result: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if type(raw_key) is not str or not raw_key or type(tensor) is not torch.Tensor:
            raise InvalidCheckpointError(
                f"{label} must map non-empty strings to tensors"
            )
        key = raw_key.replace("_orig_mod.", "")
        key = key.removeprefix("module.")
        if not key or key.startswith("module.") or "_orig_mod." in key:
            raise InvalidCheckpointError(
                f"{label} contains a non-canonical key: {raw_key!r}"
            )
        if key in result:
            raise InvalidCheckpointError(
                f"{label} canonicalization collides at {key!r}"
            )
        result[key] = tensor.detach().cpu()
    return result


def _sanitize_agent_config(
    agent_config: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    train: dict[str, Any],
) -> dict[str, Any]:
    sanitized = _require_plain_metadata(agent_config, "agent config")
    if "codebook_path" in sanitized:
        required_suffixes = (
            "codebook_manager.sorted_hand_poses",
            "codebook_manager.pca_permutation",
            "codebook_manager.layer_weights",
            "codebook_manager.hand_normalizer_scale",
            "codebook_manager.hand_normalizer_offset",
            "codebook_manager.hand_min",
            "codebook_manager.hand_max",
        )
        found: dict[str, torch.Tensor] = {}
        for suffix in required_suffixes:
            matches = [
                tensor for key, tensor in model_state.items() if key.endswith(suffix)
            ]
            if len(matches) != 1 or matches[0].numel() == 0:
                raise UnsupportedPolicyError(
                    "DQ-RISE checkpoint has no complete persistent runtime codebook state"
                )
            found[suffix] = matches[0]
        poses = found["codebook_manager.sorted_hand_poses"]
        permutation = found["codebook_manager.pca_permutation"]
        layer_weights = found["codebook_manager.layer_weights"]
        hand_normalizer_scale = found["codebook_manager.hand_normalizer_scale"]
        hand_normalizer_offset = found["codebook_manager.hand_normalizer_offset"]
        hand_min = found["codebook_manager.hand_min"]
        hand_max = found["codebook_manager.hand_max"]
        normalizer_suffixes = (
            "normalizer.params_dict.action.scale",
            "normalizer.params_dict.action.offset",
        )
        normalizer_state: dict[str, torch.Tensor] = {}
        for suffix in normalizer_suffixes:
            matches = [
                tensor for key, tensor in model_state.items() if key.endswith(suffix)
            ]
            if len(matches) != 1 or matches[0].numel() == 0:
                raise UnsupportedPolicyError(
                    "DQ-RISE checkpoint has no complete policy action normalizer state"
                )
            normalizer_state[suffix] = matches[0]
        action_scale = normalizer_state["normalizer.params_dict.action.scale"]
        action_offset = normalizer_state["normalizer.params_dict.action.offset"]
        groups = sanitized.get("codebook_num_groups")
        size = sanitized.get("codebook_size")
        hand_dim = train["hand_dim"]
        if (
            type(groups) is not int
            or type(size) is not int
            or type(hand_dim) is not int
            or poses.ndim != 2
            or poses.shape[0] != size**groups
            or poses.shape[1] != hand_dim
            or permutation.ndim != 1
            or permutation.numel() != poses.shape[0]
            or permutation.dtype
            not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
            or not torch.equal(
                torch.sort(permutation.detach().cpu().to(torch.int64)).values,
                torch.arange(poses.shape[0], dtype=torch.int64),
            )
            or layer_weights.ndim != 1
            or layer_weights.numel() != groups
            or hand_normalizer_scale.ndim != 1
            or hand_normalizer_scale.numel() != hand_dim
            or hand_normalizer_offset.ndim != 1
            or hand_normalizer_offset.numel() != hand_dim
            or hand_min.numel() != 1
            or hand_max.numel() != 1
            or action_scale.ndim != 1
            or action_scale.numel() != train["action_dim"]
            or action_offset.ndim != 1
            or action_offset.numel() != train["action_dim"]
            or not bool(torch.isfinite(poses).all())
            or not bool(torch.isfinite(permutation).all())
            or not bool(torch.isfinite(layer_weights).all())
            or not bool(torch.isfinite(hand_normalizer_scale).all())
            or not bool(torch.isfinite(hand_normalizer_offset).all())
            or bool(torch.any(hand_normalizer_scale == 0))
            or not bool(torch.isfinite(hand_min).all())
            or not bool(torch.isfinite(hand_max).all())
            or not bool(torch.isfinite(action_scale).all())
            or not bool(torch.isfinite(action_offset).all())
            or bool(torch.any(action_scale == 0))
            or not bool(hand_max.item() > hand_min.item())
            or train["tcp_dim"] != sanitized.get("tcp_dim")
            or train["action_key"] != "action_ee"
        ):
            raise InvalidCheckpointError(
                "DQ-RISE codebook/config/action metadata conflict"
            )
        try:
            torch.testing.assert_close(
                action_scale[-hand_dim:].detach().cpu(),
                hand_normalizer_scale.detach().cpu(),
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                action_offset[-hand_dim:].detach().cpu(),
                hand_normalizer_offset.detach().cpu(),
                rtol=1e-5,
                atol=1e-6,
            )
        except AssertionError as exc:
            raise InvalidCheckpointError(
                "DQ-RISE codebook hand normalizer conflicts with policy action normalizer"
            ) from exc
        sanitized["codebook_path"] = None
    pc_encoder = sanitized.get("pc_encoder_config")
    if type(pc_encoder) is dict and "use_pretrained_weights" in pc_encoder:
        pc_encoder["use_pretrained_weights"] = False
    return sanitized


def _build_inference_config(
    task_name: str,
    agent_config: dict[str, Any],
    source: _CheckpointDeploymentSource,
    selected: _SelectedInferenceSettings,
) -> dict[str, Any]:
    """Assemble the artifact inference config from checkpoint-owned semantics.

    Only ``task_name`` and the selected inference recipe come from outside the
    checkpoint; everything model-facing is ``source``.
    """
    train = source.agent
    inference = {
        "task_name": task_name,
        "action_key": train["action_key"],
        "action_dim": train["action_dim"],
        "horizon": train["horizon"],
        "use_aux_ee": train["use_aux_ee"],
        "normalization": source.normalization_contract,
        "agent": agent_config,
        "eval": {
            "use_ema": selected.use_ema,
            "denoise_steps": selected.denoise_steps,
        },
    }
    return _require_plain_metadata(inference, "inference_config")


def _resolve_selected_inference_settings(
    experiment_dir: Path,
    checkpoint_selector: str,
    cfg_plain: Mapping[str, Any],
) -> _SelectedInferenceSettings:
    """Resolve the selected checkpoint's inference recipe.

    The recipe is the one piece of model-facing configuration the experiment
    config still owns: ``best_ckpt.json["inference"]`` for the ``best``
    selector, otherwise the current ``config.eval`` (the live inference recipe
    a researcher is tuning).  Everything else deployment needs comes from the
    checkpoint.  ``denoise_steps`` is an ablation knob, not architecture, so
    overriding it per run stays legitimate.
    """
    if checkpoint_selector == "best":
        from dexmani_policy.training.eval_utils import read_best_ckpt_json

        try:
            inference = read_best_ckpt_json(experiment_dir)["inference"]
        except (OSError, ValueError) as exc:
            raise InvalidCheckpointError(
                f"invalid best_ckpt.json: {experiment_dir / 'best_ckpt.json'}"
            ) from exc
        prefix = "best_ckpt.json inference"
    else:
        eval_config = cfg_plain.get("eval")
        if type(eval_config) is not dict:
            raise InvalidExperimentError("config.eval must be a mapping")
        if eval_config.get("denoise_timesteps_list") is not None:
            raise UnsupportedPolicyError(
                "eval.denoise_timesteps_list is unsupported for deployment"
            )
        inference = {
            "use_ema": eval_config.get("use_ema"),
            "denoise_steps": eval_config.get("denoise_steps"),
        }
        prefix = "config"

    use_ema = inference["use_ema"]
    if type(use_ema) is not bool:
        raise InvalidExperimentError(f"{prefix}.use_ema must be bool")
    denoise_steps = _require_positive_int(
        inference["denoise_steps"], f"{prefix}.denoise_steps"
    )
    return _SelectedInferenceSettings(use_ema, denoise_steps)


def _rgb_preprocessing(source: _CheckpointDeploymentSource) -> dict[str, Any]:
    """Record the exact validation-spatial and ImageProcessor RGB chain.

    Both stages are read from checkpoint-saved semantics, so later edits to the
    experiment's dataset/agent RGB settings cannot change an old artifact.
    """
    agent = source.agent_config
    dataset = source.dataset
    if dataset.get("_target_") != "dexmani_policy.datasets.rgb_dataset.RGBDataset":
        raise UnsupportedPolicyError(
            "RGB deployment requires the current RGBDataset validation contract"
        )
    if "rgb_preprocess_size" not in dataset or "rgb_random_crop_size" not in dataset:
        raise InvalidCheckpointError(
            "RGB export requires explicit checkpoint validation preprocessing"
        )
    resize_hw = _optional_hw(dataset["rgb_preprocess_size"], "rgb_preprocess_size")
    center_crop_hw = _optional_hw(
        dataset["rgb_random_crop_size"], "rgb_random_crop_size"
    )
    if resize_hw is None and center_crop_hw is not None:
        raise InvalidCheckpointError(
            "dataset.rgb_random_crop_size requires rgb_preprocess_size"
        )
    keep_uint8 = dataset.get("rgb_keep_uint8", DEFAULT_RGB_KEEP_UINT8)
    if type(keep_uint8) is not bool:
        raise InvalidCheckpointError("dataset.rgb_keep_uint8 must be bool")
    validation_keeps_uint8 = (
        resize_hw is not None and keep_uint8 and dataset.get("rgb_color_aug") is None
    )

    backbone_name = agent.get("rgb_backbone_name")
    backbone_config = agent.get("rgb_backbone_config")
    if type(backbone_name) is not str or backbone_name not in IMAGE_PROCESSOR_PRESETS:
        raise UnsupportedPolicyError("RGB deployment requires a supported backbone")
    if backbone_config is None:
        backbone_config = {}
    if type(backbone_config) is not dict:
        raise InvalidCheckpointError("agent.rgb_backbone_config must be a mapping")
    if any(
        key in backbone_config
        for key in ("center_crop_size", "image_mean", "image_std")
    ):
        raise UnsupportedPolicyError(
            "RGB deployment does not support unowned ImageProcessor overrides"
        )

    preset = IMAGE_PROCESSOR_PRESETS[backbone_name]
    image_size = _processor_hw(
        (
            backbone_config["image_size"]
            if backbone_config.get("image_size") is not None
            else preset.get("image_size")
        ),
        "image_size",
    )
    interpolation = (
        backbone_config["interpolation"]
        if backbone_config.get("interpolation") is not None
        else preset.get("interpolation")
    )
    if type(interpolation) is str:
        interpolation = interpolation.lower()
    if interpolation not in {"nearest", "bilinear", "bicubic"}:
        raise InvalidCheckpointError("agent RGB interpolation is unsupported")
    mean = _processor_rgb_vector(preset.get("image_mean"), "image_mean")
    std = _processor_rgb_vector(preset.get("image_std"), "image_std")

    if resize_hw is None:
        output_layout = "HWC"
        output_dtype = "uint8"
        output_value_range = [0, 255]
        scale = 1.0
    elif validation_keeps_uint8:
        output_layout = "CHW"
        output_dtype = "uint8"
        output_value_range = [0, 255]
        scale = 1.0
    else:
        output_layout = "CHW"
        output_dtype = "float32"
        output_value_range = [0, 1]
        scale = 1.0 / 255.0

    return {
        "input_layout": "HWC",
        "input_dtype": "uint8",
        "input_color_order": "rgb",
        "input_value_range": [0, 255],
        "execution_device": "cpu",
        "resize_hw": None if resize_hw is None else list(resize_hw),
        "center_crop_hw": (None if center_crop_hw is None else list(center_crop_hw)),
        "interpolation": "bilinear",
        "antialias": True,
        "output_layout": output_layout,
        "output_dtype": output_dtype,
        "scale": scale,
        "output_value_range": output_value_range,
        "processor_image_size_hw": (None if image_size is None else list(image_size)),
        "processor_interpolation": interpolation,
        "normalize_mean": list(mean),
        "normalize_std": list(std),
    }


def _processor_hw(value: Any, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is int and value > 0:
        return value, value
    if (
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value)
    ):
        return value
    if (
        type(value) is list
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value)
    ):
        return value[0], value[1]
    raise InvalidCheckpointError(f"agent RGB {label} must be positive [H, W] or null")


def _processor_rgb_vector(value: Any, label: str) -> tuple[float, float, float]:
    if (
        type(value) not in {tuple, list}
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise InvalidCheckpointError(f"agent RGB {label} must be three finite numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise InvalidCheckpointError(f"agent RGB {label} must be three finite numbers")
    return result  # type: ignore[return-value]


def _optional_hw(value: Any, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise InvalidCheckpointError(f"dataset.{label} must be [H, W] or null")
    return value[0], value[1]


def _require_plain_metadata(value: Any, label: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise InvalidCheckpointError(f"{label} contains a non-finite float")
        return value
    if type(value) is list:
        return [_require_plain_metadata(item, label) for item in value]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise InvalidCheckpointError(f"{label} contains a non-string key")
            result[key] = _require_plain_metadata(item, label)
        return result
    raise InvalidCheckpointError(
        f"{label} contains non-plain metadata: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidCheckpointError("metadata must be canonical finite JSON") from exc


def _validate_payload(payload: Any) -> None:
    try:
        parse_deployment_contract(payload)
    except (DeploymentContractError, TypeError, KeyError) as exc:
        raise ArtifactVerificationError("invalid deployment metadata") from exc
    _canonicalize_state_dict(payload["weights"], "weights")
    _require_plain_metadata(payload["contract"], "contract")


def _write_checkpoint_temp(directory: Path, payload: dict[str, Any]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".deployment-checkpoint-", suffix=".tmp", dir=directory
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_deployment_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactVerificationError(
            "cannot safely reload deployment checkpoint"
        ) from exc
    _validate_payload(payload)
    return payload


def _replace_relative_symlink(selector_path: Path, target: str) -> None:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".deployment-selector-", dir=selector_path.parent
    )
    os.close(descriptor)
    temp = Path(raw_path)
    temp.unlink()
    try:
        temp.symlink_to(target)
        os.replace(temp, selector_path)
    finally:
        temp.unlink(missing_ok=True)


def _publish_deployment_selector(
    selector_path: Path,
    checkpoint_path: Path,
) -> None:
    if selector_path.exists() and not selector_path.is_symlink():
        raise ArtifactPublicationError(
            "refusing to replace a non-symlink deployment_latest.pt selector"
        )
    _replace_relative_symlink(selector_path, checkpoint_path.name)


def cleanup_candidate_artifact(checkpoint_path: Path) -> None:
    """Remove one unpublished candidate artifact so a same-name retry succeeds.

    Never silent: a cleanup that itself fails raises
    ``ArtifactPublicationError`` because the leftover candidate would otherwise
    block the obvious retry with a confusing ``FileExistsError``, and the
    operator needs to know the artifact directory may require manual
    inspection.
    """
    try:
        if checkpoint_path.is_symlink() or checkpoint_path.exists():
            checkpoint_path.unlink()
    except OSError as exc:
        raise ArtifactPublicationError(
            "failed to remove the unpublished deployment candidate "
            f"{checkpoint_path}; inspect the checkpoint directory manually before "
            "retrying"
        ) from exc


def _build_deployment_payload(
    experiment: Path,
    checkpoint_selector: str,
) -> tuple[dict[str, Any], Path]:
    repo_root = Path(__file__).resolve().parents[2]
    selected_path = _resolve_checkpoint(experiment, checkpoint_selector)
    cfg_plain = _load_config(experiment)
    task_name = cfg_plain.get("task_name")
    if type(task_name) is not str or not task_name:
        raise InvalidExperimentError(
            "experiment config task_name must be a non-empty string"
        )
    selected_inference = _resolve_selected_inference_settings(
        experiment, checkpoint_selector, cfg_plain
    )
    checkpoint = _load_training_checkpoint(selected_path)
    source = _parse_checkpoint_deployment_source(checkpoint)
    # A config edit must not re-label the training task.
    if source.deployment_data_semantics.get("task_name") != task_name:
        raise InvalidExperimentError(
            "experiment config task_name does not match the checkpoint's "
            "training task identity"
        )

    # Selected weights first: decide raw vs EMA, then canonicalize, validate and
    # sanitize exactly one state.  The unselected state is never processed.
    selected_weights = "ema_model" if selected_inference.use_ema else "model"
    selected_raw = (
        checkpoint.ema_model_state
        if selected_inference.use_ema
        else checkpoint.model_state
    )
    if selected_raw is None:
        raise InvalidCheckpointError(
            f"eval.use_ema={selected_inference.use_ema!r} requires checkpoint "
            f"{selected_weights} weights"
        )
    selected_state = _canonicalize_state_dict(
        selected_raw, f"weights.{selected_weights}"
    )

    agent_config = _sanitize_agent_config(
        source.agent_config, selected_state, source.agent
    )
    inference = _build_inference_config(
        task_name, agent_config, source, selected_inference
    )
    public_spec = _build_policy_spec(source)
    if "rgb" in source.observation_fields:
        inference["rgb_preprocessing"] = _rgb_preprocessing(source)
        inference["warmup_rgb_hw"] = source.deployment_data_semantics[
            "observation_fields"
        ]["rgb"]["shape"][:2]
    producer = {
        "source_checkpoint": selected_path.name,
        "selected_weights": selected_weights,
    }
    source_commit = _source_commit(repo_root)
    if source_commit is not None:
        producer["source_commit"] = source_commit
    deployment_inference = {
        **inference,
        "eval": {"denoise_steps": inference["eval"]["denoise_steps"]},
    }
    payload = {
        "_format": DEPLOYMENT_FORMAT,
        "contract": {
            "inference_config": deployment_inference,
            "policy_spec": public_spec,
            "producer": producer,
        },
        "weights": selected_state,
    }
    _validate_payload(payload)
    return payload, selected_path


def _source_commit(repo_root: Path) -> str | None:
    """Best-effort git HEAD for paper-experiment traceability only.

    This is never a runtime compatibility gate; an unavailable revision simply
    omits the field rather than blocking an export.

    ``git -C <dir> rev-parse HEAD`` resolves the *nearest enclosing* repository,
    so a copied or installed package tree could otherwise record an unrelated
    outer repository's commit — indistinguishable from a correct sha, and worse
    than no provenance at all.  The revision is therefore accepted only when the
    package root is itself the repository top level.
    """
    import subprocess

    def _git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *arguments],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    if _git("rev-parse", "--show-toplevel") != str(repo_root):
        return None
    commit = _git("rev-parse", "HEAD")
    if commit is None or len(commit) != 40:
        return None
    if any(char not in "0123456789abcdef" for char in commit):
        return None
    return commit


def _resolve_artifact_paths(
    experiment: Path, selected_path: Path, output_path: Path | None
) -> tuple[Path, Path, Path]:
    checkpoint_dir = experiment / "checkpoints"
    if output_path is None:
        final_path = checkpoint_dir / f"{selected_path.stem}-deployment.pt"
    else:
        requested = Path(output_path)
        final_path = (
            requested if requested.is_absolute() else checkpoint_dir / requested
        )
    final_path = final_path.absolute()
    if (
        final_path.parent.resolve() != checkpoint_dir.resolve()
        or final_path.suffix != ".pt"
    ):
        raise ArtifactPublicationError(
            "output_path must be a .pt file in experiment/checkpoints"
        )
    return checkpoint_dir, final_path, checkpoint_dir / "deployment_latest.pt"


def _require_experiment_directory(experiment_dir: Path) -> Path:
    try:
        experiment = Path(experiment_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise InvalidExperimentError(
            f"experiment directory not found: {experiment_dir}"
        ) from exc
    if not experiment.is_dir():
        raise InvalidExperimentError(
            f"experiment path is not a directory: {experiment}"
        )
    return experiment


def export_deployment_artifact(
    experiment_dir: Path,
    checkpoint_selector: str = "best",
    output_path: Path | None = None,
) -> ExportReceipt:
    """Validate and safely reload metadata, then atomically publish the artifact.

    Failed publication removes this call's candidate and preserves the previous
    selector. Strict model restoration and warmup run before Real policy_ready.
    """
    experiment = _require_experiment_directory(experiment_dir)
    payload, selected_path = _build_deployment_payload(experiment, checkpoint_selector)
    checkpoint_dir, final_path, selector_path = _resolve_artifact_paths(
        experiment, selected_path, output_path
    )
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite deployment artifact: {final_path}"
        )
    checkpoint_temp: Path | None = None
    try:
        checkpoint_temp = _write_checkpoint_temp(checkpoint_dir, payload)
        os.replace(checkpoint_temp, final_path)
        checkpoint_temp = None
    except BaseException as exc:
        if checkpoint_temp is not None:
            checkpoint_temp.unlink(missing_ok=True)
        if isinstance(exc, DeploymentExportError):
            raise
        raise ArtifactPublicationError(
            "deployment candidate artifact write failed"
        ) from exc
    try:
        _load_deployment_payload(final_path)
        _publish_deployment_selector(selector_path, final_path)
    except BaseException:
        # The candidate already carries its final name; without this cleanup
        # the failed candidate would survive and block the identical retry.
        cleanup_candidate_artifact(final_path)
        raise
    return ExportReceipt(
        checkpoint_path=final_path,
        selector_path=selector_path,
        checkpoint_selector=checkpoint_selector,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = export_deployment_artifact(
        args.experiment_dir,
        checkpoint_selector=args.checkpoint,
        output_path=args.output,
    )
    print(
        _canonical_json(
            {
                "checkpoint_path": str(receipt.checkpoint_path),
                "selector_path": str(receipt.selector_path),
                "checkpoint_selector": receipt.checkpoint_selector,
            }
        )
    )


if __name__ == "__main__":
    main()
