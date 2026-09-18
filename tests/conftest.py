"""Shared deployment-integrity fixtures.

These build *real* artifacts — a resolved ``config.yaml``, a genuine
``simple.v3`` training checkpoint, and a Real Policy Zarr with the exact
semantic attrs the exporter requires — so the deployment tests exercise the
production code paths rather than mocks.  Everything is deliberately tiny so a
full export/restore/predict round trip runs on CPU in seconds.
"""

from __future__ import annotations

import json
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

# A ManiFlow point-cloud policy is the smallest agent that needs no pretrained
# asset, no network, and no codebook, and whose ``n_head`` is a genuine
# forward-semantics constructor argument that leaves state-dict keys unchanged.
CHECKPOINT_N_HEAD = 8
DRIFTED_N_HEAD = 12
TASK_NAME = "unit_task"
NUM_POINTS = 1024  # smallest count the deployment point-cloud contract allows
CONTROL_DT = 0.1

_POINT_SEMANTICS = {
    "point_cloud_frame": "xarm_base",
    "point_cloud_color_source": "mean_rgb_of_aligned_depth_pixels_per_voxel",
    "point_cloud_policy_id": (
        "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9"
    ),
    "point_cloud_sampling": (
        "deterministic_coarse_voxel_stratified_hash_or_cyclic_pad"
    ),
    "point_cloud_transform": (
        "depth_gate_and_cardinal_edge_support;depth_to_color_deprojection;"
        "table_plane_height_hysteresis_crop_in_color_frame_before_deprojection;"
        "xarm_base_transform;workspace_crop;mean_voxel_xyz_and_rgb;"
        "single_radius_graph_density_and_component_outlier;spatial_candidate_cap;"
        "coarse_voxel_stratified_hash_or_cyclic_pad"
    ),
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
        "denoise_timesteps": 2,
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


def write_zarr(path: Path, *, task_name: str = TASK_NAME, length: int = 8) -> Path:
    """Write a minimal Real Policy Zarr carrying the required semantic attrs."""
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    rng = np.random.default_rng(0)
    data["joint_state"] = rng.uniform(-1.0, 1.0, (length, 19)).astype(np.float32)
    data["action"] = rng.uniform(-1.0, 1.0, (length, 19)).astype(np.float32)
    data["action_ee"] = rng.uniform(-1.0, 1.0, (length, 21)).astype(np.float32)
    points = rng.uniform(-1.0, 1.0, (length, NUM_POINTS, 6)).astype(np.float32)
    data["point_cloud"] = points
    root.attrs.update(
        {
            "schema_name": "dexmani-real-policy-zarr",
            "schema_version": 13,
            "domain": "real",
            "task_name": task_name,
            "dt": CONTROL_DT,
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_alignment": "control_step_latest_causal",
            "state_alignment": "control_step",
            "contact_force_source": "raw_hand_contact_control_step",
            "action_semantics": "teleop_published_joint_target",
            "point_cloud_table_plane_abcd_json": json.dumps(
                [0.0, 0.0, 1.0, -0.02], separators=(",", ":")
            ),
            "processing_config_json": json.dumps(
                {"pointcloud": {"depth_max_m": 1.5}, "table_plane_abcd": None}
            ),
            **_POINT_SEMANTICS,
        }
    )
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
) -> tuple[Path, Any]:
    """Write one genuine ``simple.v3`` training checkpoint and return its agent.

    ``agent_cfg`` is what the checkpoint *saves* as its constructor source.
    ``weights_from`` optionally builds the state dict from a different
    constructor, which is how the tests prove deployment reads the saved
    mapping rather than whatever happens to be instantiable.
    """
    resolved_agent_cfg = agent_config() if agent_cfg is None else agent_cfg
    resolved_dataset_cfg = dataset_config() if dataset_cfg is None else dataset_cfg
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
        resume_contract={
            "agent": agent_contract() if contract is None else contract,
            "agent_config": resolved_agent_cfg,
            "dataset": resolved_dataset_cfg,
            "loader": {"batch_size": 2, "shuffle": True, "drop_last": False},
            "dataset_length": 8,
            "batches_per_epoch": 4,
            "world_size": 1,
            "optimizer": {},
            "ema": {},
            "training": {},
        },
        ema_updater_step=10,
        ema_decay=0.999,
        rng_states=[{}],
    )
    path = CheckpointStore(experiment / "checkpoints").save(filename, checkpoint)
    return path, agent


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """A ready-to-export experiment: config, checkpoint, and a matching Zarr."""
    experiment_dir = tmp_path / "experiments" / "maniflow" / TASK_NAME / "run"
    (experiment_dir / "checkpoints").mkdir(parents=True)
    write_config(experiment_dir)
    write_checkpoint(experiment_dir)
    write_zarr(tmp_path / f"{TASK_NAME}.zarr")
    return experiment_dir


@pytest.fixture
def zarr_path(tmp_path: Path) -> Path:
    """The physical dataset location matching the ``experiment`` fixture."""
    return tmp_path / f"{TASK_NAME}.zarr"
