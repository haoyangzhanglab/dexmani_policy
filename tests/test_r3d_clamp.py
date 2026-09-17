import torch
import torch.nn as nn

from dexmani_policy.agents.obs_encoder.pointcloud.r3d_obs_encoder import R3DObsEncoder


def _encoder():
    return R3DObsEncoder(
        state_dim=19,
        n_obs_steps=2,
        pc_encoder_config={
            "pc_model": "eva02_tiny_patch14_224",
            "embed_dim": 64,
            "num_group": 8,
            "group_size": 8,
            "pc_in_channels": 6,
            "use_pretrained_weights": False,
        },
        state_out_dim=32,
    )


def test_r3d_clamps_only_xyz():
    enc = _encoder()
    captured = {}

    class _Stub(nn.Module):
        def forward(self, pc, inference_mode=False):
            captured["pc"] = pc
            B, N, C = pc.shape
            return (
                torch.zeros(B, enc.num_pc_tokens, 64),
                torch.zeros(B, enc.num_pc_tokens, 64),
            )

    enc.pc_encoder = _Stub()
    enc.eval()

    B, T, N = 1, 2, 32
    pc = torch.randn(B * T, N, 6)
    pc[..., :3] *= 5.0   # XYZ far outside [-1, 1]
    pc[..., 3:] = 5.0    # RGB value that would be clamped if a full-tensor clamp ran
    original = pc.clone()

    enc({"point_cloud": pc, "joint_state": torch.randn(B * T, 19)})

    clamped = captured["pc"]
    assert (clamped[..., :3] <= 1 + 1e-6).all() and (clamped[..., :3] >= -1 - 1e-6).all()
    assert (clamped[..., 3:] == 5.0).all()  # RGB untouched
    # original input tensor is not mutated in place
    assert not torch.equal(original[..., :3], clamped[..., :3])
    assert torch.equal(original[..., 3:], clamped[..., 3:])
