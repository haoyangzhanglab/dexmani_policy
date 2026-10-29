"""训练数据身份与公开部署配方的无设备回归。"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from dexmani_policy.training.common.data import build_dataset


class CanonicalResumeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def create_data(self, name="data.zarr", *, source_id="source-a", canonical=True):
        path = self.directory / name
        root = zarr.open_group(str(path), mode="w")
        if canonical:
            root.attrs["format"] = "dexmani.real.canonical"
        if source_id is not None:
            root.attrs["source_id"] = source_id
        root.attrs["episode_ids"] = ["episode_001"]
        data = root.create_group("data")
        for key in ("joint_state", "action"):
            data.create_dataset(key, data=np.zeros((4, 19), dtype=np.float32))
        root.create_group("meta").create_dataset("episode_ends", data=np.array([4]))
        return path

    def config(self, path, *, resume=False, manifest=None):
        return OmegaConf.create({
            "resume_from": "checkpoint.pt" if resume else None,
            "normalization": {"joint_state": "limits", "action": "auto"},
            "agent": {"action_dim": 19},
            "dataset": {
                "_target_": "dexmani_policy.datasets.common.robot_dataset.RobotDataset",
                "zarr_path": str(path), "sensor_modalities": ["joint_state"],
                "horizon": 1, "obs_horizon": 1, "split_manifest": manifest,
            },
        })

    def test_same_identity_moves_but_reexport_does_not_resume(self):
        path = self.create_data()
        _, description = build_dataset(self.config(path), verbose=False)
        manifest = description["split_manifest"]
        moved = self.directory / "moved.zarr"
        shutil.copytree(path, moved)
        build_dataset(self.config(moved, resume=True, manifest=manifest), verbose=False)
        replacement = self.create_data("replacement.zarr", source_id="source-b")
        with self.assertRaisesRegex(InstantiationException, "source identity mismatch"):
            build_dataset(self.config(replacement, resume=True, manifest=manifest), verbose=False)

    def test_missing_identity_rejects_resume_even_at_the_same_path(self):
        path = self.create_data(source_id=None)
        _, description = build_dataset(self.config(path), verbose=False)
        manifest = description["split_manifest"]
        # A supplied legacy split remains valid for fresh training and evaluation.
        build_dataset(self.config(path, manifest=manifest), verbose=False)
        with self.assertRaisesRegex(ValueError, "Canonical checkpoint resume requires source_id"):
            build_dataset(self.config(path, resume=True, manifest=manifest), verbose=False)
        for source_id, saved_id in (("source-a", None), (None, "source-a")):
            root = zarr.open_group(str(path), mode="a")
            if source_id is None:
                root.attrs.pop("source_id", None)
            else:
                root.attrs["source_id"] = source_id
            manifest["source_id"] = saved_id
            with self.assertRaisesRegex(InstantiationException, "source identity mismatch"):
                build_dataset(self.config(path, resume=True, manifest=manifest), verbose=False)

    def test_other_formats_keep_their_existing_identity_contract(self):
        path = self.create_data(source_id=None, canonical=False)
        _, description = build_dataset(self.config(path), verbose=False)
        build_dataset(
            self.config(path, resume=True, manifest=description["split_manifest"]), verbose=False,
        )


if __name__ == "__main__":
    unittest.main()
