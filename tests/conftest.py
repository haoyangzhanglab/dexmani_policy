"""Shared deployment-integrity fixtures.

These build *real* artifacts — a resolved ``config.yaml``, a genuine
``simple.v3`` training checkpoint carrying a ``deployment_data_semantics``
snapshot computed by the production extractor, and a Real Policy Zarr with the
exact semantic attrs the shared Real Policy contract requires — so the
deployment tests exercise the production code paths rather than mocks.
Everything is deliberately tiny so a full export/restore/predict round trip
runs on CPU in seconds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint_io import (
    CheckpointStore,
    TrainCheckpoint,
    make_normalization_contract,
)
from dexmani_policy.datasets.real_policy_contract import (
    _EEF_POSE_SEMANTICS,
    _FINGERTIP_SEMANTICS,
    _POINT_SEMANTICS,
    _TACTILE_FORCE_SEMANTICS,
    build_real_policy_data_semantics,
)

# A ManiFlow point-cloud policy is the smallest agent that needs no pretrained
# asset, no network, and no codebook, and whose ``n_head`` is a genuine
# forward-semantics constructor argument that leaves state-dict keys unchanged.
CHECKPOINT_N_HEAD = 8
DRIFTED_N_HEAD = 12
TASK_NAME = "unit_task"
NUM_POINTS = 1024  # smallest count the deployment point-cloud contract allows
CONTROL_DT = 0.1
ACTION_EE_COMPONENTS = "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)"

# The full semantic attr set a Real producer writes, minus the per-Zarr facts
# (task_name/dt) and the point-cloud attrs assembled in write_zarr below.
_PRODUCER_ATTRS: dict[str, Any] = {
    "schema_name": "dexmani-real-policy-zarr",
    "schema_version": 13,
    "domain": "real",
    "obs_alignment": "obs[t]_before_action[t]",
    "observation_alignment": "control_step_latest_causal",
    "state_alignment": "control_step",
    "contact_force_source": "raw_hand_contact_control_step",
    "contact_force_unit": "xhand_sdk_native_unknown_si",
    "contact_force_frame": "xhand_sensor_native_axes_per_finger",
    "contact_force_si_verified": False,
    "action_semantics": "teleop_published_joint_target",
    "action_ee_frame": "xarm_base",
    "action_ee_components": ACTION_EE_COMPONENTS,
    "camera_extrinsic_semantics": (
        "T_xarm_base_from_color;native_color_optical_to_xarm_base"
    ),
    "fingertip_config_json": json.dumps(
        {
            "fingertip_link_names": [
                "right_hand_thumb_rota_tip",
                "right_hand_index_rota_tip",
                "right_hand_mid_tip",
                "right_hand_ring_tip",
                "right_hand_pinky_tip",
            ],
            "handbase_position_eef_m": [-0.015, 0.0, 0.0],
            "handbase_quat_eef_wxyz": [0.707107, 0.0, 0.707107, 0.0],
        },
        separators=(",", ":"),
    ),
    "tactile_force_unit": "xhand_sdk_native_unknown_si",
    "tactile_force_si_verified": False,
    "tactile_force_spatial_geometry_verified": False,
    **_FINGERTIP_SEMANTICS,
    **_EEF_POSE_SEMANTICS,
    **_TACTILE_FORCE_SEMANTICS,
}


def agent_config(*, n_head: int = CHECKPOINT_N_HEAD) -> dict[str, Any]:
    """One resolved ManiFlow agent constructor mapping."""
    return {
        "_target_": "dexmani_policy.agents.core.maniflow.ManiFlowAgent",
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "action_dim": 19,
        "encoder_type": "pointnet_dense",
        "pc_dim": 6,
        "state_dim": 19,
        "num_points": NUM_POINTS,
        "state_out_dim": 8,
        "pc_encoder_config": {
            "out_channels": 16,
            "num_points": NUM_POINTS,
            "hidden_dims": [8, 16],
        },
        "fps_random_config": {
            "use_random": False,
            "use_random_start": False,
            "random_noise_scale": 0.0,
            "use_shuffle_output": False,
        },
        "n_layers": 1,
        "hidden_dim": 16,
        "n_head": n_head,
        "mlp_ratio": 1.0,
        "p_drop_attn": 0.0,
        "timestep_embed_dim": 8,
        "target_t_embed_dim": 8,
        "num_inference_steps": 2,
    }


def dataset_config(**overrides: Any) -> dict[str, Any]:
    """One resolved point-cloud dataset config."""
    config = {
        "_target_": "dexmani_policy.datasets.pc_dataset.PCDataset",
        "action_key": "action",
        "zarr_path": f"robot_data/{TASK_NAME}.zarr",
        "seed": 0,
        "horizon": 4,
        "pad_before": 1,
        "pad_after": 1,
        "val_ratio": 0.0,
        "sensor_modalities": ["joint_state", "point_cloud"],
        "obs_horizon": 2,
    }
    config.update(overrides)
    return config


NORMALIZATION_FIELDS = {
    "joint_state": "limits",
    "action": "auto",
    "point_cloud": "limits",
}


def agent_contract(
    *,
    normalization_fields: dict[str, str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """One ``resume_contract.agent`` action/window/normalization contract."""
    fields = (
        dict(NORMALIZATION_FIELDS)
        if normalization_fields is None
        else dict(normalization_fields)
    )
    contract = {
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "action_dim": 19,
        "horizon": 4,
        "action_key": "action",
        "tcp_dim": None,
        "hand_dim": None,
        "control_action_dim": 19,
        "use_aux_ee": False,
        "normalization": make_normalization_contract(fields),
    }
    contract.update(overrides)
    return contract


def write_zarr(
    path: Path,
    *,
    task_name: str = TASK_NAME,
    length: int = 8,
    dt: float = CONTROL_DT,
    point_count: int = NUM_POINTS,
    state_dtype: Any = np.float32,
    attrs_override: dict[str, Any] | None = None,
    drop_attrs: Sequence[str] = (),
) -> Path:
    """Write a minimal Real Policy Zarr carrying the full producer attr set.

    Every semantic attr the shared Real Policy contract may read is present, so
    fixture Zarrs exercise the same extractor path as real producer output.
    The override parameters exist for drift tests: each changes exactly one
    physical fact while everything else stays semantic-equivalent.
    """
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    rng = np.random.default_rng(0)
    data["joint_state"] = rng.uniform(-1.0, 1.0, (length, 19)).astype(state_dtype)
    data["action"] = rng.uniform(-1.0, 1.0, (length, 19)).astype(np.float32)
    data["action_ee"] = rng.uniform(-1.0, 1.0, (length, 21)).astype(np.float32)
    data["point_cloud"] = rng.uniform(
        -1.0, 1.0, (length, point_count, 6)
    ).astype(np.float32)
    attrs: dict[str, Any] = {
        **_PRODUCER_ATTRS,
        **_POINT_SEMANTICS,
        "task_name": task_name,
        "dt": dt,
        "point_cloud_table_plane_abcd_json": json.dumps(
            [0.0, 0.0, 1.0, -0.02], separators=(",", ":")
        ),
        "processing_config_json": json.dumps(
            {"pointcloud": {"depth_max_m": 1.5}, "table_plane_abcd": None}
        ),
    }
    attrs.update(attrs_override or {})
    for key in drop_attrs:
        attrs.pop(key, None)
    root.attrs.update(attrs)
    return path


def add_rgb(path: Path, *, height: int = 48, width: int = 48, length: int = 8) -> Path:
    """Add an rgb array plus camera semantics to an existing fixture Zarr."""
    root = zarr.open_group(str(path), mode="a")
    root["data"]["rgb"] = np.random.default_rng(3).integers(
        0, 255, (length, height, width, 3), dtype=np.uint8
    )
    root.attrs["camera_extrinsic_semantics"] = (
        "T_xarm_base_from_color;native_color_optical_to_xarm_base"
    )
    return path


def deployment_semantics_for(
    zarr_file: Path,
    *,
    agent_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Run the production extractor exactly the way training does.

    The inputs mirror ``training.resume._deployment_data_semantics``: the
    physical Zarr, the dataset's modalities, the saved agent constructor and
    the contract's action key — so a fixture checkpoint is indistinguishable
    from a real training checkpoint at the export boundary.
    """
    attrs = dict(zarr.open_group(str(zarr_file), mode="r").attrs)
    return build_real_policy_data_semantics(
        zarr_file,
        task_name=str(attrs["task_name"]),
        observation_fields=list(
            dataset_cfg.get("sensor_modalities", ["joint_state"])
        ),
        agent_config=agent_cfg,
        action_key=contract.get("action_key", "action"),
    )


def normalization_fields_for(modalities) -> dict[str, str]:
    """The normalization spec matching one modality list (rgb is always identity)."""
    return {
        name: ("identity" if name == "rgb" else "limits") for name in modalities
    } | {"action": "auto"}


def build_agent(agent_cfg: dict[str, Any], modalities=None) -> Any:
    """Instantiate one agent and fit a normalizer for exactly these modalities."""
    import hydra

    from dexmani_policy.common.normalizer import LinearNormalizer

    fields = list(modalities) if modalities is not None else ["joint_state", "point_cloud"]
    agent = hydra.utils.instantiate(OmegaConf.create(agent_cfg))
    normalizer = LinearNormalizer()
    rng = np.random.default_rng(1)
    for name in fields:
        if name == "rgb":  # identity: no fitted params
            continue
        if name == "joint_state":
            data = rng.uniform(-1.0, 1.0, (16, 19)).astype(np.float32)
        elif name == "point_cloud":
            data = rng.uniform(-1.0, 1.0, (16, NUM_POINTS, 6)).astype(np.float32)
        else:
            raise AssertionError(f"fixture has no data generator for {name!r}")
        normalizer.fit_field(name, data)
    normalizer.fit_field("action", rng.uniform(-1.0, 1.0, (16, 19)).astype(np.float32))
    agent.load_normalizer_from_dataset(normalizer)
    agent.action_key = "action"
    agent.normalization_spec = normalization_fields_for(fields)
    return agent


def write_config(
    experiment: Path,
    *,
    agent_cfg: dict[str, Any] | None = None,
    dataset_cfg: dict[str, Any] | None = None,
    normalization: dict[str, str] | None = None,
    task_name: str = TASK_NAME,
    use_ema: bool = False,
    denoise_steps: int = 2,
) -> Path:
    """Write a resolved experiment ``config.yaml``.

    Deployment must read only ``policy_name``/``task_name`` and the ``eval``
    recipe from this file; the agent/dataset/normalization sections exist here
    precisely so the tests can drift them and prove they are ignored.
    """
    config = {
        "policy_name": "maniflow",
        "task_name": task_name,
        "zarr_path": f"robot_data/{task_name}.zarr",
        "action_key": "action",
        "action_dim": 19,
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "normalization": (
            dict(NORMALIZATION_FIELDS) if normalization is None else dict(normalization)
        ),
        "agent": agent_config() if agent_cfg is None else agent_cfg,
        "dataset": dataset_config() if dataset_cfg is None else dataset_cfg,
        "eval": {
            "use_ema": use_ema,
            "denoise_steps": denoise_steps,
            "denoise_timesteps_list": None,
        },
    }
    path = experiment / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    return path


def write_checkpoint(
    experiment: Path,
    *,
    filename: str = "latest.pt",
    agent_cfg: dict[str, Any] | None = None,
    dataset_cfg: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
    with_ema: bool = True,
    state_dict: dict[str, torch.Tensor] | None = None,
    weights_from: dict[str, Any] | None = None,
    zarr_file: Path | None = None,
    deployment_data_semantics: Any = "auto",
) -> tuple[Path, Any]:
    """Write one genuine ``simple.v3`` training checkpoint and return its agent.

    ``agent_cfg`` is what the checkpoint *saves* as its constructor source.
    ``weights_from`` optionally builds the state dict from a different
    constructor, which is how the tests prove deployment reads the saved
    mapping rather than whatever happens to be instantiable.

    ``deployment_data_semantics`` controls the frozen training data snapshot:
    ``"auto"`` (default) runs the production extractor against ``zarr_file``
    (falling back to the saved dataset config's ``zarr_path``), an explicit
    dict/``None`` is stored verbatim, and ``"absent"`` omits the key to model
    a legacy checkpoint written before the snapshot existed.
    """
    resolved_agent_cfg = agent_config() if agent_cfg is None else agent_cfg
    resolved_dataset_cfg = dataset_config() if dataset_cfg is None else dataset_cfg
    resolved_contract = agent_contract() if contract is None else contract
    modalities = resolved_dataset_cfg.get("sensor_modalities", ["joint_state"])
    # An explicitly supplied state dict is used verbatim; only build an agent
    # when the weights still have to come from somewhere.
    agent = (
        None
        if state_dict is not None
        else build_agent(
            resolved_agent_cfg if weights_from is None else weights_from,
            modalities,
        )
    )
    model_state = (
        state_dict
        if state_dict is not None
        else {key: value.detach().clone() for key, value in agent.state_dict().items()}
    )
    if deployment_data_semantics == "auto":
        physical = zarr_file
        if physical is None:
            physical = Path(resolved_dataset_cfg["zarr_path"]).expanduser()
        if not physical.exists():
            raise AssertionError(
                "fixture Zarr for deployment_data_semantics not found: "
                f"{physical}; pass zarr_file=... to write_checkpoint"
            )
        deployment_data_semantics = deployment_semantics_for(
            physical,
            agent_cfg=resolved_agent_cfg,
            dataset_cfg=resolved_dataset_cfg,
            contract=resolved_contract,
        )
    resume_contract: dict[str, Any] = {
        "agent": resolved_contract,
        "agent_config": resolved_agent_cfg,
        "dataset": resolved_dataset_cfg,
        "loader": {"batch_size": 2, "shuffle": True, "drop_last": False},
        "dataset_length": 8,
        "batches_per_epoch": 4,
        "world_size": 1,
        "optimizer": {},
        "ema": {},
        "training": {},
    }
    if deployment_data_semantics != "absent":
        resume_contract["deployment_data_semantics"] = deployment_data_semantics
    checkpoint = TrainCheckpoint(
        epoch=1,
        global_step=10,
        next_micro_step=0,
        model_state=model_state,
        ema_model_state=(
            {key: value.detach().clone() for key, value in model_state.items()}
            if with_ema
            else None
        ),
        optimizer_state={},
        scheduler_state={},
        monitor={},
        resume_contract=resume_contract,
        ema_updater_step=10,
        ema_decay=0.999,
        rng_states=[{}],
    )
    path = CheckpointStore(experiment / "checkpoints").save(filename, checkpoint)
    return path, agent


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """A ready-to-export experiment: Zarr, config, and a matching checkpoint."""
    experiment_dir = tmp_path / "experiments" / "maniflow" / TASK_NAME / "run"
    (experiment_dir / "checkpoints").mkdir(parents=True)
    dataset = write_zarr(tmp_path / f"{TASK_NAME}.zarr")
    write_config(experiment_dir)
    write_checkpoint(experiment_dir, zarr_file=dataset)
    return experiment_dir


@pytest.fixture
def zarr_path(tmp_path: Path) -> Path:
    """The physical dataset location matching the ``experiment`` fixture."""
    return tmp_path / f"{TASK_NAME}.zarr"
