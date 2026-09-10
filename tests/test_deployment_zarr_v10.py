"""Focused regression tests for the Real Policy Zarr v10 data boundary.

These exercise ``deployment/export.py``'s Zarr ingestion — the ``contact_force``
units contract and the schema-version gate — against a minimal temp Zarr, with
no trained checkpoint and no hardware.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from dexmani_policy.deployment.contract import DEPLOYMENT_FORMAT, parse_deployment_contract
from dexmani_policy.deployment.export import (
    InvalidZarrError,
    _POINT_SEMANTICS,
    _build_observation_contract,
    _validate_fingertip_points,
    _validate_point_cloud,
)

_SCHEMA_NAME = "dexmani-real-policy-zarr"


def _cfg() -> dict:
    return {
        "action_key": "action",
        "task_name": "t",
        "dt": 0.0625,
        "dataset": {"dt": 0.0625},
        "agent": {},
    }


def _write_zarr(
    path: Path,
    *,
    schema_version: int = 10,
    profile: str = "joint",
    contact_force_unit: str = "xhand_sdk_native_unknown_si",
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        {
            "schema_name": _SCHEMA_NAME,
            "schema_version": schema_version,
            "domain": "real",
            "profile": profile,
            "task_name": "t",
            "dt": 0.0625,
            "episode_start_policy": "full_history",
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_alignment": "control_step_latest_causal",
            "state_alignment": "control_step",
            "contact_force_source": "raw_hand_contact_control_step",
            "action_semantics": "teleop_published_joint_target",
            "contact_force_unit": contact_force_unit,
            "contact_force_frame": "xhand_sensor_native_axes_per_finger",
            "contact_force_si_verified": False,
        }
    )
    data = root.create_group("data")
    for name, shape in (
        ("joint_state", (1, 19)),
        ("action", (1, 19)),
        ("action_ee", (1, 21)),
        ("contact_force", (1, 5, 3)),
    ):
        data.create_dataset(name, shape=shape, dtype=np.float32)
    if profile in {"rgb", "rgb_pc"}:
        data.create_dataset("rgb", shape=(1, 8, 8, 3), dtype=np.uint8)
        root.attrs["camera_extrinsic_semantics"] = (
            "T_xarm_base_from_color;native_color_optical_to_xarm_base"
        )
    if profile in {"pointcloud", "rgb_pc"}:
        data.create_dataset("point_cloud", shape=(1, 1024, 6), dtype=np.float32)
        root.attrs.update(_POINT_SEMANTICS)
        root.attrs["point_cloud_table_plane_abcd_json"] = "null"


class TestPolicyZarrV10Boundary(unittest.TestCase):
    def test_accepts_v10_native_contact_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.zarr"
            _write_zarr(
                path,
                schema_version=10,
                contact_force_unit="xhand_sdk_native_unknown_si",
            )
            contract = _build_observation_contract(
                path, _cfg(), ["joint_state", "contact_force"]
            )
        self.assertEqual(contract["schema_version"], 10)
        self.assertEqual(contract["observation_alignment"], "control_step_latest_causal")
        self.assertEqual(contract["state_alignment"], "control_step")
        self.assertEqual(contract["contact_force_source"], "raw_hand_contact_control_step")
        self.assertNotIn("observation_reference", contract)
        semantics = contract["observation_fields"]["contact_force"]["semantics"]
        self.assertEqual(semantics["representation"], "per_finger_sensor_axes")
        self.assertEqual(semantics["units"], "xhand_sdk_native_unknown_si")
        self.assertEqual(semantics["frame"], "xhand_sensor_native_axes_per_finger")
        self.assertIs(semantics["si_verified"], False)

    def test_v10_data_contract_parses_with_unchanged_runtime_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.zarr"
            _write_zarr(path, profile="pointcloud")
            contract = _build_observation_contract(
                path, _cfg(), ["joint_state", "point_cloud", "contact_force"]
            )
        for action_key, action_dim in (("action", 19), ("action_ee", 21)):
            with self.subTest(action_key=action_key):
                spec = parse_deployment_contract({
                    "_format": DEPLOYMENT_FORMAT,
                    "weights": {"synthetic": 1},
                    "contract": {
                        "producer": {},
                        "data_contract": contract,
                        "inference_config": {
                            "action_key": action_key,
                            "action_dim": action_dim,
                            "horizon": 16,
                            "n_obs_steps": 2,
                            "n_action_steps": 8,
                            "eval": {"denoise_steps": 10},
                        },
                    },
                })
                self.assertEqual(spec.control_action_dim, action_dim)
                self.assertEqual(spec.control_dt_s, 0.0625)
                self.assertEqual(spec.chunk_size, 15)
                self.assertEqual(
                    [field.name for field in spec.observation_fields],
                    ["joint_state", "point_cloud", "contact_force"],
                )

    def test_rejects_noncurrent_schema(self) -> None:
        for version in (8, 9, 11, 10.0, "10", True):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "t.zarr"
                _write_zarr(path, schema_version=version)
                with self.assertRaisesRegex(
                    InvalidZarrError, "invalid Real Policy Zarr semantics"
                ):
                    _build_observation_contract(path, _cfg(), ["joint_state"])

    def test_all_profiles_use_the_same_control_step_contract(self) -> None:
        for profile, visual in (
            ("joint", []),
            ("rgb", ["rgb"]),
            ("pointcloud", ["point_cloud"]),
            ("rgb_pc", ["rgb", "point_cloud"]),
        ):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "t.zarr"
                _write_zarr(path, profile=profile)
                fields = ["joint_state", "contact_force", *visual]
                contract = _build_observation_contract(path, _cfg(), fields)
                self.assertEqual(list(contract["observation_fields"]), fields)
                self.assertEqual(contract["state_alignment"], "control_step")
                self.assertNotIn("observation_reference", contract)

    def test_rejects_wrong_control_step_semantics(self) -> None:
        for key, value in (
            ("observation_alignment", "camera_source_monotonic_ns"),
            ("state_alignment", "camera_source_aligned_state"),
            ("state_alignment", "control_grid_state"),
            ("contact_force_source", "previous_raw_row"),
        ):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "t.zarr"
                _write_zarr(path)
                zarr.open_group(str(path), mode="a").attrs[key] = value
                with self.assertRaisesRegex(InvalidZarrError, "control-step contract"):
                    _build_observation_contract(
                        path, _cfg(), ["joint_state", "contact_force"]
                    )

    def test_old_reference_is_not_a_fallback_for_missing_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.zarr"
            _write_zarr(path)
            root = zarr.open_group(str(path), mode="a")
            del root.attrs["observation_alignment"]
            root.attrs["observation_reference"] = "grid_anchor_monotonic_ns"
            with self.assertRaisesRegex(InvalidZarrError, "missing semantic attrs"):
                _build_observation_contract(path, _cfg(), ["joint_state"])

    def test_task_dt_and_action_gates_remain_strict(self) -> None:
        for key, value in (
            ("task_name", "other"),
            ("dt", 0.02),
            ("action_semantics", "candidate_joint_target"),
            ("episode_start_policy", "padded_history"),
            ("contact_force_si_verified", True),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "t.zarr"
                _write_zarr(path)
                zarr.open_group(str(path), mode="a").attrs[key] = value
                with self.assertRaises(InvalidZarrError):
                    _build_observation_contract(
                        path, _cfg(), ["joint_state", "contact_force"]
                    )

    def test_rejects_scaled_contact_force_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.zarr"
            _write_zarr(
                path,
                schema_version=10,
                contact_force_unit="sdk_scaled_unknown_si",
            )
            with self.assertRaisesRegex(
                InvalidZarrError, "xhand_sdk_native_unknown_si"
            ):
                _build_observation_contract(
                    path, _cfg(), ["joint_state", "contact_force"]
                )

    def test_point_cloud_accepts_without_config_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = zarr.open_group(str(Path(tmp) / "t.zarr"), mode="w")
            point_cloud = root.create_dataset(
                "point_cloud", shape=(1, 1024, 6), dtype=np.float32
            )
            attrs = {
                "point_cloud_frame": "xarm_base",
                "point_cloud_color_source": (
                    "mean_rgb_of_aligned_depth_pixels_per_voxel"
                ),
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
                    "single_radius_graph_density_and_component_outlier;"
                    "spatial_candidate_cap;coarse_voxel_stratified_hash_or_cyclic_pad"
                ),
                "point_cloud_table_plane_abcd_json": "null",
            }
            tail, semantics = _validate_point_cloud(point_cloud, attrs, {"agent": {}})
        self.assertEqual(tail, (1024, 6))
        self.assertNotIn("config_sha256", semantics)
        self.assertEqual(
            semantics["policy_id"],
            "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9",
        )

    def test_fingertip_points_accepts_without_geometry_sha256(self) -> None:
        attrs = {
            "fingertip_points_frame": "xarm_base",
            "fingertip_points_unit": "m",
            "fingertip_points_derivation": "fk_from_processed_joint_state",
            "fingertip_points_policy_id": "arm_hand_fk_from_joint_state_v1",
        }
        semantics = _validate_fingertip_points(attrs)
        self.assertNotIn("geometry_sha256", semantics)
        self.assertEqual(semantics["policy_id"], "arm_hand_fk_from_joint_state_v1")


if __name__ == "__main__":
    unittest.main()
