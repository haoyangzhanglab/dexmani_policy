"""Check point encoding and policy interfaces with real operators and synthetic inputs."""

import argparse

import torch
from transformers import Dinov2Config

from dexmani_policy.agents.obs_encoder.pointcloud.feature_alignment import FeatureAlignment, build_patch_correspondence
from dexmani_policy.agents.obs_encoder.pointcloud.projection import PointProjection
from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.agents.obs_encoder.pointcloud.point_transformer import PointTransformerEncoder
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandForceEncoder
from dexmani_policy.agents.policies.adaptac import AdapTac
from dexmani_policy.agents.policies.dexlatent import DexLatent
from dexmani_policy.agents.policies.dqrise import DQRISE
from dexmani_policy.agents.policies.interact_dex import InterActDex
from dexmani_policy.agents.policies.sat import SAT
from dexmani_policy.training.common.config import load_policy_config


def check_gradients(model):
    modules = (model.patch_embedding.point_mlp, model.patch_embedding.projection,
               model.position_embedding, *model.blocks)
    for module in modules:
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().max() > 0 for g in gradients)


def check_encoder(device):
    for channels in (3, 6):
        points = torch.rand(2, 1024, channels, device=device)
        points[..., :3] *= 0.3
        model = PointTransformerEncoder(input_channels=channels).to(device).eval()
        patches = model.encode_patches(points)
        shapes = {
            'tokens': (2, 128, 192), 'xyz': (2, 128, 3),
            'neighbor_indices': (2, 128, 32), 'relative_xyz': (2, 128, 32, 3),
            'rope_cos': (2, 1, 128, 24), 'rope_sin': (2, 1, 128, 24),
        }
        assert set(patches) == set(shapes)
        for name, shape in shapes.items():
            assert patches[name].shape == shape, name
        neighbors = patches['neighbor_indices']
        assert neighbors.dtype == torch.long and neighbors.min() >= 0 and neighbors.max() < 1024
        torch.testing.assert_close(
            patches['relative_xyz'],
            index_points(points[..., :3], neighbors) - patches['xyz'].unsqueeze(2),
        )
        distance2 = (patches['xyz'][:, :, None] - points[:, None, :, :3]).square().sum(-1)
        assert (distance2.amin(-1) == 0).all()
        torch.testing.assert_close(
            distance2.gather(2, neighbors).sort(-1).values,
            distance2.topk(32, largest=False).values,
        )
        attributes = index_points(points[..., 3:], neighbors)
        # Member order must not affect the pooled token, including its RGB association.
        torch.testing.assert_close(
            model.patch_embedding(patches['relative_xyz'], attributes),
            model.patch_embedding(patches['relative_xyz'].flip(2), attributes.flip(2)),
        )
        result = model(points)
        assert set(result) == {'tokens', 'xyz'}
        torch.testing.assert_close(result['xyz'], patches['xyz'], rtol=0, atol=0)
        torch.testing.assert_close(
            result['tokens'], model.encode_global(patches['tokens'], patches['rope_cos'], patches['rope_sin']),
        )
        torch.testing.assert_close(result['tokens'], model(points)['tokens'], rtol=0, atol=0)
        (result['tokens'] * torch.randn_like(result['tokens'])).mean().backward()
        check_gradients(model)
        clone = PointTransformerEncoder(input_channels=channels, **model.effective_config).to(device).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(clone(points)['tokens'], result['tokens'])
        model.train()
        torch.manual_seed(18)
        first = model.encode_patches(points)
        torch.manual_seed(18)
        repeat = model.encode_patches(points)
        torch.testing.assert_close(first['xyz'], repeat['xyz'], rtol=0, atol=0)
        torch.manual_seed(19)
        assert not torch.equal(first['xyz'], model.encode_patches(points)['xyz'])
        if device.type == 'cuda':
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(points)['tokens']
                loss = (output * torch.randn_like(output)).float().mean()
            loss.backward()
            assert torch.isfinite(output).all()
            check_gradients(model)
        print(f'PASS encoder: {device}, {channels}D forward/backward, spatial alignment, FPS, reconstruction', flush=True)


def check_configs():
    expected = dict(channels=192, num_patches=128, group_size=32, depth=2, num_heads=8,
                    mlp_ratio=2.0, geometry_scale=0.10, rope_scale=0.05, rope_base=100.0)
    assert PointTransformerEncoder().effective_config == expected
    for name in ('sat', 'dqrise', 'dexlatent', 'adaptac', 'interact_dex'):
        config = load_policy_config(name).agent
        assert config['pc_encoder_config'] == expected
        assert config['num_points'] == 1024
        encoder = PointTransformerEncoder(input_channels=config['pc_dim'], **config['pc_encoder_config'])
        assert encoder.out_dim == 192 and len(encoder.blocks) == 2
        assert all(block.num_heads == 8 and block.head_dim == 24 for block in encoder.blocks)
    print('PASS Hydra/default/effective_config agreement', flush=True)


@torch.no_grad()
def check_alignment(device):
    # Include image borders, repeated members and invalid projections.
    coordinates = torch.tensor(
        [[[-0.5, -0.5], [0., 0.], [0.99, 0.99], [-1., -1.], [2., 0.], [float("nan"), 0.]]],
        device=device,
    )
    confidence = torch.tensor([[1., 0.5, 1., 1., 0., 0.]], device=device)
    neighbors = torch.tensor([[[0, 1, 1], [2, 3, 0], [4, 5, 4]]], device=device)
    image = torch.randn(1, 2, 2, 8, device=device)
    tokens = torch.randn(1, 3, 192, device=device)
    alignment = FeatureAlignment(8).to(device).eval()
    projection = PointProjection(coordinates, confidence)
    sampled = torch.nn.functional.grid_sample(
        image.permute(0, 3, 1, 2), coordinates.nan_to_num().unsqueeze(1),
        padding_mode="border", align_corners=False,
    ).squeeze(2).transpose(1, 2)
    for weights in (None, torch.tensor([[[1., 2., 3.], [2., 1., 1.], [1., 0., 1.]]], device=device)):
        correspondence = build_patch_correspondence(neighbors, projection, (2, 2), member_weight=weights)
        valid = index_points(confidence > 0, neighbors)
        supported = valid.float() if weights is None else weights * valid
        prior = supported / supported.sum(-1, keepdim=True).clamp_min(1)
        expected = (index_points(sampled, neighbors) * prior[..., None]).sum(2)
        pooled = correspondence["pool_weight"] @ image.flatten(1, 2)
        torch.testing.assert_close(pooled, expected)
        fused = alignment(tokens, image, projection, neighbors, member_weight=weights)
        semantic = alignment.proj(alignment.image_norm(expected))
        torch.testing.assert_close(fused[:, :2], (tokens + 0.1 * semantic)[:, :2])
        torch.testing.assert_close(fused[:, 2], tokens[:, 2], rtol=0, atol=0)
    unsupported = PointProjection(coordinates, torch.zeros_like(confidence))
    torch.testing.assert_close(
        alignment(tokens, torch.full_like(image, float("nan")), unsupported, neighbors),
        tokens, rtol=0, atol=0,
    )
    print("PASS RGB alignment: grid_sample reference, weighted pooling, unsupported patches", flush=True)


def make_obs(device, time, channels):
    points = torch.rand(1, time, 1024, channels, device=device)
    points[..., :2] = (points[..., :2] - 0.5) * 0.3
    points[..., 2] = points[..., 2] * 0.2 + 0.8
    return dict(point_cloud=points, joint_state=torch.rand(1, time, 19, device=device))


@torch.no_grad()
def check_policies(device):
    # Small real RGB/action networks keep this interface check independent of pretrained weights.
    rgb_config = dict(image_size=[56, 56], architecture=Dinov2Config(
        hidden_size=48, num_hidden_layers=1, num_attention_heads=4,
        image_size=56, patch_size=14, mlp_ratio=2,
    ).to_dict())
    base = dict(horizon=8, n_obs_steps=2, n_action_steps=2, action_dim=19)
    dit = dict(n_layers=1, hidden_dim=96, n_head=4, timestep_embed_dim=32)
    for policy_class, extra in ((SAT, {}), (DQRISE, {'num_codes': 4}), (DexLatent, {})):
        policy = policy_class(**base, **dit, **extra, rgb_backbone_config=rgb_config).to(device).eval()
        obs = make_obs(device, 2, 3)
        obs['rgb'] = torch.rand(1, 2, 3, 56, 56, device=device)
        obs['camera_intrinsic'] = torch.tensor(
            [[40., 0., 27.5], [0., 40., 27.5], [0., 0., 1.]], device=device,
        ).expand(1, 2, 3, 3)
        obs['camera_extrinsic'] = torch.eye(3, 4, device=device).expand(1, 2, 3, 4)
        for stage in ('pre_global', 'post_global'):
            for sigma in (0.04, None):
                policy.obs_encoder.fusion_stage = stage
                policy.obs_encoder.semantic_pool_sigma = sigma
                condition = policy._encode_obs(obs)
                assert condition.shape == (1, 258, 256) and torch.isfinite(condition).all()
                action_dim = getattr(policy, 'flow_action_dim', 19)
                output = policy.action_decoder.model(
                    torch.randn(1, 8, action_dim, device=device), torch.tensor([0.5], device=device), condition,
                )
                assert output.shape == (1, 8, action_dim) and torch.isfinite(output).all()
        print(f'PASS {policy_class.__name__}: RGB alignment, observation and action forward', flush=True)

    policy = InterActDex(
        **{**base, 'n_obs_steps': 4}, **dit, tactile_encoder=XHandForceEncoder(),
    ).to(device).eval()
    obs = make_obs(device, 4, 6)
    obs['eef_pose'] = torch.tensor([0., 0., 1., 1., 0., 0., 0., 1., 0.], device=device).expand(1, 4, 9)
    obs['fingertip_xyz'] = torch.rand(1, 4, 5, 3, device=device) * 0.1 + torch.tensor([0., 0., 1.], device=device)
    obs['finger_tactile_force'] = torch.rand(1, 4, 5, 3, device=device)
    features = policy._encode_features(obs)
    assert features.scene.shape == (1, 4, 12, 192)
    assert features.interaction.shape == (1, 4, 6, 192)
    condition = policy._pack_action_context(features)
    assert condition.shape == (1, 72, 192) and torch.isfinite(condition).all()
    output = policy.action_decoder.model(torch.randn(1, 8, 19, device=device), torch.tensor([0.5], device=device), condition)
    assert output.shape == (1, 8, 19) and torch.isfinite(output).all()
    print('PASS InterActDex: SceneTokenizer, HandSceneFusion, history and action context', flush=True)

    policy = AdapTac(
        **base, tactile_encoder=XHandForceEncoder(), down_dims=(32, 64),
        diffusion_step_embed_dim=32, force_predictor_config=dict(depth=1, embed_dim=64, num_heads=4),
    ).to(device).eval()
    obs = make_obs(device, 2, 6)
    obs['finger_tactile_force'] = torch.rand(1, 2, 5, 3, device=device)
    features, current_force, force_condition = policy._encode_force_inputs(obs)
    assert tuple(x.shape for x in features) == ((1, 1024), (1, 1024), (1, 128))
    future_force = policy.force_predictor.model(
        torch.randn(1, policy.future_steps, 15, device=device), torch.tensor([1], device=device), force_condition,
    )
    condition = policy._action_condition(features, current_force, future_force)
    assert condition.shape == (1, 640) and torch.isfinite(condition).all()
    output = policy.action_decoder.model(torch.randn(1, 8, 19, device=device), torch.tensor([1], device=device), condition)
    assert output.shape == (1, 8, 19) and torch.isfinite(output).all()
    print('PASS AdapTac: visual/tactile fusion, force predictor and action context', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    device = torch.device(args.device)
    check_configs()
    check_encoder(device)
    check_alignment(device)
    check_policies(device)


if __name__ == '__main__':
    main()
