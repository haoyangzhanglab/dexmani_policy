"""Small, self-contained deployment-v2 fixtures for Real restore tests.

The fixture deliberately uses a tiny CPU DP3 model rather than an experiment
checkpoint.  It exercises the frozen checkpoint/sidecar contract without a
dataset, camera, simulator, or hardware process.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


POINT_CLOUD_POLICY_ID = "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9"
POINT_CLOUD_COLOR_SOURCE = "mean_rgb_of_aligned_depth_pixels_per_voxel"
POINT_CLOUD_SAMPLING = "deterministic_coarse_voxel_stratified_hash_or_cyclic_pad"
POINT_CLOUD_TRANSFORM = (
    "depth_gate_and_cardinal_edge_support;depth_to_color_deprojection;"
    "table_plane_height_hysteresis_crop_in_color_frame_before_deprojection;"
    "xarm_base_transform;workspace_crop;mean_voxel_xyz_and_rgb;"
    "single_radius_graph_density_and_component_outlier;spatial_candidate_cap;"
    "coarse_voxel_stratified_hash_or_cyclic_pad"
)


def clone_clean_policy_repository(source: Path, destination: Path) -> str:
    """Create an independent clean checkout and return its exact HEAD."""
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("temporary Policy checkout is unexpectedly dirty")
    head = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise RuntimeError("temporary Policy checkout did not produce a full git commit")
    return head


def build_tiny_dp3_artifact(
    root: Path,
    *,
    producer_commit: str,
    normalizer_point_cloud_dim: int = 6,
) -> Path:
    """Write a valid deployment-v2 DP3 artifact rooted at ``root``.

    ``normalizer_point_cloud_dim`` is intentionally configurable so the tests
    can prove Real rejects a checkpoint whose normalizer is fitted but has the
    wrong model-space width.
    """
    import torch
    from dexmani_policy.agents.core.dp3 import DP3Agent
    from dexmani_policy.common.normalizer import LinearNormalizer

    if len(producer_commit) != 40:
        raise ValueError("producer_commit must be a full git commit")
    if normalizer_point_cloud_dim <= 0 or normalizer_point_cloud_dim > 6:
        raise ValueError("normalizer_point_cloud_dim must be in [1, 6]")

    experiment = root / "tiny-dp3-experiment"
    checkpoint_dir = experiment / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (experiment / "config.yaml").write_text("agent: {}\n", encoding="utf-8")

    agent_config: dict[str, Any] = {
        "_target_": "dexmani_policy.agents.core.dp3.DP3Agent",
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "action_dim": 19,
        "encoder_type": "dp3",
        "pc_dim": 6,
        "pc_out_dim": 8,
        "state_dim": 19,
        "num_points": 1024,
        "state_out_dim": 8,
        "diffusion_step_embed_dim": 16,
        "down_dims": [8],
        "kernel_size": 3,
        "n_groups": 8,
        "num_training_steps": 2,
        "num_inference_steps": 1,
        "prediction_type": "sample",
        "cond_predict_scale": True,
    }
    # Keep the committed fixture byte-reproducible without consuming or
    # replacing the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        agent = DP3Agent(
            **{key: value for key, value in agent_config.items() if key != "_target_"}
        )
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.arange(2 * 16 * 19, dtype=torch.float32).reshape(2, 16, 19),
            "joint_state": torch.arange(2 * 2 * 19, dtype=torch.float32).reshape(2, 2, 19),
            "point_cloud": torch.arange(2 * 2 * 1024 * 6, dtype=torch.float32).reshape(
                2, 2, 1024, 6
            ),
        }
    )
    agent.load_normalizer_from_dataset(normalizer)
    # ``nn.Module.state_dict`` returns an OrderedDict; deployment-v2 freezes
    # this boundary as a plain dict so Real's safe loader can reject arbitrary
    # mapping subclasses.
    model_state = dict(agent.state_dict())
    if normalizer_point_cloud_dim != 6:
        for suffix in ("scale", "offset"):
            key = f"normalizer.params_dict.point_cloud.{suffix}"
            model_state[key] = model_state[key][:normalizer_point_cloud_dim].clone()

    train_params = {
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "action_dim": 19,
        "horizon": 16,
        "action_key": "action",
        "tcp_dim": None,
        "hand_dim": None,
        "control_action_dim": 19,
        "num_training_steps": 2,
        "use_aux_ee": False,
    }
    inference_config = {
        "task_name": "tiny_real_dp3",
        "action_key": "action",
        "action_dim": 19,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "use_aux_ee": False,
        "agent": agent_config,
        "eval": {"use_ema": False, "denoise_steps": 1},
    }
    data_contract = {
        "schema_name": "dexmani-real-policy-zarr",
        "schema_version": 5,
        "domain": "real",
        "profile": "pointcloud",
        "task_name": "tiny_real_dp3",
        "dt": 0.0625,
        "episode_start_policy": "full_history",
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_reference": "camera_source_monotonic_ns",
        "state_alignment": "camera_source_aligned_state",
        "max_observation_skew_s": 0.1,
        "action_semantics": "deployment_grid_rate_limited_target",
        "arm_max_delta_rad_per_tick": 0.1,
        "hand_max_delta_rad_per_tick": 0.3,
        "endpoint_delta_tolerance_rad": 1e-12,
        "deployment_equivalent": True,
        "point_cloud_frame": "xarm_base",
        "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
        "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
        "point_cloud_config_sha256": "a" * 64,
        "point_cloud_table_plane_abcd_json": "null",
        "point_cloud_sampling": POINT_CLOUD_SAMPLING,
        "point_cloud_transform": POINT_CLOUD_TRANSFORM,
        "sensor_modalities": ["joint_state", "point_cloud"],
        "point_cloud_num_points": 1024,
        "point_cloud_feature_dim": 6,
        "action_key": "action",
        "model_action_dim": 19,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "pad_before": 1,
        "pad_after": 7,
        "padding_semantics": "repeat_edge",
        "use_aux_ee": False,
    }
    producer = {
        "repository": "haoyangzhanglab/dexmani_policy",
        "commit": producer_commit,
        "metadata_provenance": "native",
        "retrofitted_train_params_fields": [],
    }
    deployment_contract = {
        "schema_version": 1,
        "inference_config": inference_config,
        "data_contract": data_contract,
        "train_params": train_params,
        "producer": producer,
        "retrofitted_train_params_fields": [],
    }
    payload = {
        "_format": "dexmani.deployment.v2",
        "state": {
            "epoch": 0,
            "global_step": 0,
            "train_params": train_params,
            "inference_config": inference_config,
            "data_contract": data_contract,
            "producer": producer,
            "deployment_contract": deployment_contract,
        },
        "weights": {"model": model_state, "ema_model": None},
    }
    checkpoint_path = checkpoint_dir / "tiny-dp3-deployment-v2.pt"
    torch.save(payload, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    embedded_contract_sha256 = _canonical_sha256(deployment_contract)
    allocation = {
        "task_name": "tiny_real_dp3",
        "action_key": "action",
        "action_dim": 19,
        "control_action_dim": 19,
        "auxiliary_action_layout": "none",
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "horizon": 16,
        "required_action_steps": 15,
        "control_dt_s": 0.0625,
        "sensor_modalities": ["joint_state", "point_cloud"],
        "observation_fields": ["arm_qpos", "hand_qpos", "point_cloud"],
        "requires_hand": True,
        "point_cloud_num_points": 1024,
        "point_cloud_feature_dim": 6,
        "rgb_shape": None,
        "rgb_color_order": None,
        "rgb_value_range": None,
    }
    sidecar = {
        "schema_version": 2,
        "checkpoint": {
            "filename": checkpoint_path.name,
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "embedded_contract_sha256": embedded_contract_sha256,
        "allocation": allocation,
        "producer": {
            "repository": "haoyangzhanglab/dexmani_policy",
            "commit": producer_commit,
            "metadata_provenance": "native",
        },
    }
    sidecar_path = checkpoint_dir / f"{checkpoint_path.name}.deployment.json"
    sidecar_path.write_bytes(_canonical_json(sidecar).encode("utf-8"))
    selector = checkpoint_dir / "deployment_latest.pt"
    selector.symlink_to(checkpoint_path.name)
    return experiment


def make_policy_source_dirty(policy_root: Path) -> None:
    """Dirty only a disposable cloned Policy repository."""
    init_path = policy_root / "dexmani_policy" / "__init__.py"
    with init_path.open("a", encoding="utf-8") as stream:
        stream.write("\n# temporary Real restore qualification dirt marker\n")
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
