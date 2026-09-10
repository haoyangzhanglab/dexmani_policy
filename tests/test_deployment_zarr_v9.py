"""Focused regression tests for the Real Policy Zarr v9 data boundary.

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

from dexmani_policy.deployment.export import (
    InvalidZarrError,
    _build_observation_contract,
    _validate_fingertip_points,
    _validate_point_cloud,
)

_SCHEMA_NAME = "dexmani-real-policy-zarr"


def _cfg() -> dict:
    return {
        "action_key": "action",
        "task_name": "t",
        "dt": 0.02,
        "dataset": {"dt": 0.02},
    }


def _write_zarr(
    path: Path,
    *,
    schema_version: int = 9,
    contact_force_unit: str = "xhand_sdk_native_unknown_si",
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        {
            "schema_name": _SCHEMA_NAME,
            "schema_version": schema_version,
            "domain": "real",
            "profile": "joint",
            "task_name": "t",
            "dt": 0.02,
            "episode_start_policy": "full_history",
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_reference": "grid_anchor_monotonic_ns",
            "state_alignment": "control_grid_state",
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


class TestPolicyZarrV9Boundary(unittest.TestCase):
    def test_accepts_v9_native_contact_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.zarr"
            _write_zarr(
                path,
                schema_version=9,
                contact_force_unit="xhand_sdk_native_unknown_si",
            )
            contract = _build_observation_contract(
                path, _cfg(), ["joint_state", "contact_force"]
            )
        self.assertEqual(contract["schema_version"], 9)
        semantics = contract["observation_fields"]["contact_force"]["semantics"]
        self.assertEqual(semantics["representation"], "per_finger_sensor_axes")
        self.assertEqual(semantics["units"], "xhand_sdk_native_unknown_si")
        self.assertEqual(semantics["frame"], "xhand_sensor_native_axes_per_finger")
        self.assertIs(semantics["si_verified"], False)

    def test_rejects_v8_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.zarr"
            _write_zarr(path, schema_version=8)
            with self.assertRaisesRegex(
                InvalidZarrError, "invalid Real Policy Zarr semantics"
            ):
                _build_observation_contract(
                    path, _cfg(), ["joint_state", "contact_force"]
                )

    def test_rejects_scaled_contact_force_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.zarr"
            _write_zarr(
                path,
                schema_version=9,
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
