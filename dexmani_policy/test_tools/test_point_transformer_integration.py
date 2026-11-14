import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint import restore_policy
from dexmani_policy.training.common.config import load_config


RGB_CONFIG = dict(
    tune_mode="freeze", image_size=[32, 32],
    architecture=dict(model_name="vit_small_patch16_dinov3", embed_dim=48, depth=1, num_heads=3),
)


def make_observation(device, channels=3):
    points = torch.rand(2, 1024, channels, device=device)
    points[..., :2] = (points[..., :2] - 0.5) * 0.5
    points[..., 2] += 1
    intrinsic = torch.tensor(
        [[24.0, 0.0, 16.0], [0.0, 24.0, 16.0], [0.0, 0.0, 1.0]], device=device,
    )
    return dict(
        point_cloud=points, rgb=torch.rand(2, 3, 32, 32, device=device),
        joint_state=torch.randn(2, 19, device=device),
        camera_intrinsic=intrinsic.expand(2, -1, -1),
        camera_extrinsic=torch.eye(4, device=device)[:3].expand(2, -1, -1),
        finger_tactile_force=torch.rand(2, 5, 3, device=device),
    )


@pytest.mark.parametrize("name", ["sat", "dqrise", "dexlatent"])
def test_rgb_policy_integration(device, name):
    config = OmegaConf.to_container(load_config(name).agent, resolve=True)
    config.update(rgb_backbone_config=RGB_CONFIG, n_layers=1, hidden_dim=48, n_head=4)
    if name == "dqrise":
        config["num_codes"] = 8
    policy = instantiate(config, _convert_="all").to(device).eval()
    encoder = policy.obs_encoder
    shapes = {}

    def record(name):
        def hook(module, args, output):
            shapes[name] = output["tokens"].shape if isinstance(output, dict) else output.shape
        return hook

    stages = {
        "patch": encoder.patch_encoder,
        "fusion": encoder.rgb_fusion,
        "spatial": encoder.spatial_encoder,
    }
    handles = [module.register_forward_hook(record(name)) for name, module in stages.items()]
    obs = make_observation(device)
    output = encoder(obs)
    for handle in handles:
        handle.remove()
    assert shapes == dict(patch=(2, 128, 192), fusion=(2, 128, 384), spatial=(2, 128, 384))
    assert output.shape == (1, 258, 384)
    assert torch.isfinite(output).all()
    (output * torch.randn_like(output)).mean().backward()
    for parameter in encoder.patch_encoder.geometry_embedding.context_mlp.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    effective = encoder.effective_config
    assert effective["pc_encoder_config"]["channels"] == 192
    assert effective["pc_encoder_config"]["local_hidden_dim"] == 64
    policy.configure_normalizer({"action": 19}, {"action": (19,)})
    config.update(policy.effective_config)
    restored = restore_policy(
        {"agent": config, "normalization_dims": {"action": 19},
         "normalization": {"action": "limits"}},
        policy.state_dict(), device,
    )
    with torch.no_grad():
        history = {key: value.unsqueeze(0) for key, value in obs.items()}
        torch.testing.assert_close(restored._encode_obs(history), output, atol=0, rtol=0)
        action_dim = getattr(policy, "flow_action_dim", policy.action_dim)
        actions = torch.randn(1, policy.horizon, action_dim, device=device)
        prediction = policy.action_decoder.model(actions, torch.zeros(1, device=device), output)
    assert prediction.shape == actions.shape
    assert torch.isfinite(prediction).all()


def test_adaptac_policy_integration(device):
    config = OmegaConf.to_container(load_config("adaptac").agent, resolve=True)
    config.update(
        down_dims=[16, 32], diffusion_step_embed_dim=16,
        force_predictor_config=dict(depth=1, embed_dim=32, num_heads=4, dropout=0.0),
    )
    policy = instantiate(config, _convert_="all").to(device).eval()
    encoder = policy.obs_encoder
    obs = make_observation(device, channels=6)
    output = encoder(obs)
    assert [x.shape for x in output] == [(2, 512), (2, 512), (2, 64)]
    assert all(torch.isfinite(x).all() for x in output)
    sum((x * torch.randn_like(x)).mean() for x in output).backward()
    for parameter in encoder.pc_encoder.patch_encoder.geometry_embedding.context_mlp.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    feature_dims = {"action": 19, "finger_tactile_force": 15}
    policy.configure_normalizer(feature_dims, {"action": (19,)})
    config.update(policy.effective_config)
    restored = restore_policy(
        {
            "agent": config, "normalization_dims": feature_dims,
            "normalization": {"action": "limits", "finger_tactile_force": "identity"},
        },
        policy.state_dict(), device,
    )
    with torch.no_grad():
        torch.testing.assert_close(restored.obs_encoder(obs), output, atol=0, rtol=0)
