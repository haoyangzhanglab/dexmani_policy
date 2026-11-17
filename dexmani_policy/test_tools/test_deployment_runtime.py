"""用当前 CPU 模型验证研究记录变化不阻止恢复与推理。"""

from contextlib import closing

import numpy as np
import torch

from dexmani_policy.common.checkpoint import load_checkpoint, save_checkpoint
from dexmani_policy.deployment import load_policy
from dexmani_policy.test_tools.smoke_test import check_cordvip, check_synthetic


def test_policy_training_resume_and_optional_processing(tmp_path):
    check_synthetic(tmp_path, "cpu", max_updates=1, resume_updates=1)
    saved = load_checkpoint(tmp_path / "run/checkpoints/latest.pt")
    observation = {
        "point_cloud": np.zeros((2, 8, 6), dtype=np.float32),
        "joint_state": np.zeros((2, 19), dtype=np.float32),
    }
    checkpoint = tmp_path / "policy.pt"
    for missing in (False, True):
        if missing:
            del saved.effective_config["data_processing"]
        else:
            saved.effective_config["data_processing"] = None
        save_checkpoint(checkpoint, saved)
        with closing(load_policy(checkpoint, device="cpu")) as runtime:
            assert runtime.info.data_processing is None
            prediction = runtime.predict(observation)
            assert prediction.shape == (3, 19) and np.isfinite(prediction).all()


def test_cordvip_policy_allows_geometry_changes_and_pools_point_order(tmp_path):
    check_cordvip(tmp_path / "cordvip", "cpu", max_updates=1, resume_updates=1)
    with closing(load_policy(tmp_path / "cordvip/run", device="cpu")) as runtime:
        assert runtime.info.data_processing["object_pointcloud"]["geometry_sha256"] is None
        assert not any("contact_head" in name for name, _ in runtime.agent.named_parameters())
        encoder = runtime.agent.obs_encoder.interaction_encoder.encoder_object
        torch.manual_seed(9)
        points = torch.randn(2, 31, 3)
        with torch.no_grad():
            expected = encoder(points)
            actual = encoder(points[:, torch.randperm(31)])
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
