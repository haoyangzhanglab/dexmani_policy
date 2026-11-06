"""Check point encoding and policy interfaces with real operators and synthetic inputs."""

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest import TestCase

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from transformers import Dinov2Config

from dexmani_policy.agents.obs_encoder.pointcloud.camera_projection import (
    ImageProjection, pixels_to_grid, project_to_image,
)
from dexmani_policy.agents.obs_encoder.pointcloud.ops import index_points
from dexmani_policy.agents.obs_encoder.pointcloud.point_transformer import (
    GlobalPointTransformer, PointPatchEncoder, PointTransformerEncoder,
)
from dexmani_policy.agents.obs_encoder.pointcloud.rgb_patch_fusion import RGBPatchFusion
from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
from dexmani_policy.agents.obs_encoder.spatial import SpatialObservationEncoder
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandForceEncoder
from dexmani_policy.agents.policies.adaptac import AdapTac
from dexmani_policy.agents.policies.dexlatent import DexLatent
from dexmani_policy.agents.policies.dqrise import DQRISE
from dexmani_policy.agents.policies.interact_dex import InterActDex
from dexmani_policy.agents.policies.sat import SAT
from dexmani_policy.common.checkpoint import (
    Checkpoint, load_checkpoint, policy_state_dict, restore_policy, save_checkpoint,
)
from dexmani_policy.training.common.config import assemble_effective_config, load_policy_config


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
        # Local max pooling must be invariant to the order of XYZ/RGB neighbors.
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
    print('PASS Hydra/default/effective_config agreement', flush=True)


def check_projection(device):
    pixels = torch.tensor([[[0., 0.], [3.5, 1.5], [-0.5, -0.5], [7., 3.],
                            [7.5, 0.], [-0.6, 0.], [float("nan"), 0.],
                            [float("inf"), 0.]]], device=device)
    valid = torch.ones(1, 8, dtype=torch.bool, device=device)
    projection = pixels_to_grid(pixels, valid, (4, 8))
    assert projection.valid.tolist() == [[True, True, True, True, False, False, False, False]]
    assert projection.grid_xy.dtype == torch.float32 and projection.valid.dtype == torch.bool
    torch.testing.assert_close(projection.grid_xy[:, :4], torch.tensor(
        [[[-0.875, -0.75], [0., 0.], [-1., -1.], [0.875, 0.75]]], device=device,
    ))
    assert torch.isfinite(projection.grid_xy).all()

    # Construct world points from known camera coordinates using a nontrivial R,t.
    camera = torch.cat((pixels[:, :6], torch.ones(1, 6, 1, device=device)), -1)
    camera = torch.cat((camera, torch.tensor(
        [[[0., 0., -1.], [0., 0., 0.], [float("nan"), 0., 1.],
          [float("inf"), 0., 1.]]], device=device)), 1)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
    translation = torch.tensor([0.3, -0.2, 0.5], device=device)
    world = (camera - translation) @ rotation
    extrinsic = torch.cat((rotation, translation[:, None]), -1)[None]
    processor = ImageProcessor(image_size=(8, 24)).to(device)
    rgb, transform = processor.process_images(torch.zeros(1, 3, 4, 8, device=device), return_transform=True)
    projected = project_to_image(world, torch.eye(3, device=device)[None], extrinsic, transform, rgb.shape[-2:])
    assert projected.valid.tolist() == [[True, True, True, True, False, False, False, False, False, False]]
    torch.testing.assert_close(projected.grid_xy[:, :4], projection.grid_xy[:, :4])
    assert torch.isfinite(projected.grid_xy).all()
    invalid_camera = extrinsic.clone()
    invalid_camera[0, 0, 0] = float("nan")
    assert not project_to_image(world, torch.eye(3, device=device)[None], invalid_camera,
                                transform, rgb.shape[-2:]).valid.any()
    print("PASS camera projection: world extrinsic, resize, pixel bounds, invalid geometry", flush=True)


def check_fusion(device):
    # Analytic bilinear values: pixel centers, center average, asymmetric interior, border.
    grid = torch.tensor([[[[0., 2., 4.], [4., 1., 2.]],
                          [[8., 3., 1.], [12., 6., 0.]]]], device=device, requires_grad=True)
    coords = torch.tensor([[[-0.5, -0.5], [0., 0.], [-0.25, 0.25], [-1., -1.], [0., 0.]]], device=device)
    valid = torch.tensor([[True, True, True, True, False]], device=device)
    geometry = torch.randn(1, 5, 192, device=device, requires_grad=True)
    fusion = RGBPatchFusion(3).to(device)
    with torch.no_grad():
        fusion.semantic_projection.bias.fill_(0.7)
    expected = torch.stack((grid[0, 0, 0], grid.mean((1, 2))[0],
                            0.1875 * grid[0, 0, 0] + 0.0625 * grid[0, 0, 1]
                            + 0.5625 * grid[0, 1, 0] + 0.1875 * grid[0, 1, 1],
                            grid[0, 0, 0], grid.mean((1, 2))[0]))[None]
    captured = []
    hook = fusion.rgb_norm.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    output = fusion(geometry, grid, ImageProjection(coords, valid))
    hook.remove()
    torch.testing.assert_close(captured[0], expected)
    torch.testing.assert_close(output[..., :192], fusion.geometry_norm(geometry))
    semantic = fusion.semantic_projection(fusion.rgb_norm(expected))
    torch.testing.assert_close(output[..., 192:], torch.where(valid[..., None], semantic, 0.))
    assert torch.count_nonzero(output[:, -1, 192:]) == 0
    (output * torch.randn_like(output)).mean().backward()
    assert grid.grad is not None and grid.grad.abs().sum() > 0
    assert geometry.grad is not None and geometry.grad.abs().sum() > 0
    assert fusion.semantic_projection.weight.grad.abs().sum() > 0

    for mixed in ([False, True] if device.type == "cuda" else [False]):
        patch_encoder = PointPatchEncoder(input_channels=3).to(device).eval()
        fusion = RGBPatchFusion(384).to(device)
        global_encoder = GlobalPointTransformer().to(device)
        points = torch.rand(2, 1024, 3, device=device)
        rgb = torch.randn(2, 16, 16, 384, device=device, requires_grad=True)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if mixed else nullcontext()
        with context:
            patches = patch_encoder(points)
            assert patches["tokens"].shape == (2, 128, 192)
            # Same center ordering is used for semantic lookup and global attention.
            projection = ImageProjection(patches["xyz"][..., :2] * 2 - 1,
                                         torch.ones(2, 128, dtype=torch.bool, device=device))
            tokens = fusion(patches["tokens"], rgb, projection)
            assert tokens.shape == (2, 128, 384)
            output = global_encoder(tokens, patches["xyz"])
            assert output.shape == tokens.shape
            assert all(block.head_dim == 48 and block.num_heads == 8 for block in global_encoder.blocks)
            assert global_encoder.rope(patches["xyz"])[0].shape == (2, 1, 128, 48)
            loss = (output * torch.randn_like(output)).float().mean()
        loss.backward()
        for module in (patch_encoder, fusion, global_encoder):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().max() > 0 for g in gradients)
        assert rgb.grad is not None and torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0
        permutation = torch.randperm(128, device=device)
        with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if mixed else nullcontext()):
            torch.testing.assert_close(
                fusion(patches["tokens"][:, permutation], rgb,
                       ImageProjection(projection.grid_xy[:, permutation], projection.valid[:, permutation])),
                fusion(patches["tokens"], rgb, projection)[:, permutation],
            )
    precision = "FP32/BF16" if device.type == "cuda" else "FP32"
    print(f"PASS RGB fusion/global: analytic sampling, masking, shapes, gradients, {precision}", flush=True)


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
        name = {SAT: "sat", DQRISE: "dqrise", DexLatent: "dexlatent"}[policy_class]
        cfg = load_policy_config(name)
        with open_dict(cfg):
            cfg.agent.update({**base, **dit, **extra, "rgb_backbone_config": rgb_config})
        policy = hydra.utils.instantiate(cfg.agent, _convert_="all")
        feature_dims = dict(point_cloud=3, rgb=3, camera_intrinsic=9,
                            camera_extrinsic=12, joint_state=19, action=19)
        feature_shapes = {key: (width,) for key, width in feature_dims.items()}
        feature_shapes["rgb"] = (1, 1, 1)
        policy.configure_normalizer(feature_dims, feature_shapes)
        # Nonidentity proprioception exercises restoration of saved affine values.
        policy.normalizer["joint_state"].scale.fill_(2.)
        policy.normalizer["joint_state"].offset.fill_(-0.5)
        policy = policy.to(device).eval()
        obs = make_obs(device, 2, 3)
        obs['rgb'] = torch.rand(1, 2, 3, 56, 56, device=device)
        obs['camera_intrinsic'] = torch.tensor(
            [[40., 0., 27.5], [0., 40., 27.5], [0., 0., 1.]], device=device,
        ).expand(1, 2, 3, 3).reshape(1, 2, 9)
        obs['camera_extrinsic'] = torch.eye(3, 4, device=device).expand(1, 2, 3, 4).reshape(1, 2, 12)
        condition = policy._encode_obs(obs)
        assert condition.shape == (1, 258, 256) and torch.isfinite(condition).all()
        encoder = policy.obs_encoder
        assert encoder.patch_encoder.out_dim == 192 and encoder.global_encoder.out_dim == 384
        assert not hasattr(encoder.patch_encoder, "blocks")
        action_dim = getattr(policy, 'flow_action_dim', 19)
        action_input = torch.randn(1, 8, action_dim, device=device)
        time_input = torch.tensor([0.5], device=device)
        output = policy.action_decoder.model(action_input, time_input, condition)
        assert output.shape == (1, 8, action_dim) and torch.isfinite(output).all()
        with tempfile.TemporaryDirectory(prefix="fusion-checkpoint-") as directory:
            cfg.workspace.output_dir = directory
            effective = assemble_effective_config(cfg, {"deployment_metadata": {}}, policy)
            config = OmegaConf.to_container(effective, resolve=True)
            checkpoint = Checkpoint(
                global_step=0, model_state=policy_state_dict(policy), ema_model_state=None,
                optimizer_state={}, scheduler_state={}, rng_state={},
                effective_config=config,
            )
            path = save_checkpoint(Path(directory) / "model.pt", checkpoint)
            saved = load_checkpoint(path)
            restored = restore_policy(saved.effective_config, saved.model_state, device)
            torch.testing.assert_close(restored.normalizer.state_dict(), policy.normalizer.state_dict())
            restored_condition = restored._encode_obs(obs)
            torch.testing.assert_close(restored_condition, condition)
            torch.testing.assert_close(
                restored.action_decoder.model(action_input, time_input, restored_condition), output,
            )
        print(f'PASS {policy_class.__name__}: fusion, observation/action forward, Hydra and strict checkpoint restore', flush=True)

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


def check_rgb_gradients(device):
    for mode in ("freeze", "full"):
        encoder = SpatialObservationEncoder(rgb_backbone_config=dict(
            tune_mode=mode, image_size=[56, 56],
            architecture=Dinov2Config(hidden_size=48, num_hidden_layers=1,
                num_attention_heads=4, image_size=56, patch_size=14, mlp_ratio=2).to_dict(),
        )).to(device).eval()
        obs = make_obs(device, 2, 3)
        obs.update(rgb=torch.rand(1, 2, 3, 56, 56, device=device),
                   camera_intrinsic=torch.tensor([[40., 0., 27.5], [0., 40., 27.5], [0., 0., 1.]], device=device).expand(1, 2, 3, 3),
                   camera_extrinsic=torch.eye(3, 4, device=device).expand(1, 2, 3, 4))
        obs = {key: value.flatten(0, 1) for key, value in obs.items()}
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            output = encoder(obs)
            loss = (output * torch.randn_like(output)).float().mean()
        loss.backward()
        for module in (encoder.patch_encoder, encoder.rgb_fusion, encoder.global_encoder,
                       encoder.output_projection, encoder.state_encoder):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        rgb_gradients = [p.grad for p in encoder.rgb_encoder.parameters()]
        if mode == "freeze":
            assert all(g is None for g in rgb_gradients)
        else:
            assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in rgb_gradients)
    with torch.no_grad():
        encoder.image_processor.image_size = (56, 70)
        assert encoder(obs).shape == (1, 258, 256)
        encoder.image_processor.image_size = (57, 56)
        with TestCase().assertRaisesRegex(ValueError, "divisible"):
            encoder(obs)
    print("PASS spatial encoder: RGB gradients, rectangular grid, partial-patch rejection", flush=True)


@torch.no_grad()
def check_config_widths(device):
    rgb_config = dict(architecture=Dinov2Config(
        hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
        image_size=224, patch_size=14,
    ).to_dict())
    encoder = SpatialObservationEncoder(rgb_backbone_config=rgb_config).to(device).eval()
    rgb = torch.rand(2, 3, 224, 224, device=device)
    assert encoder.rgb_encoder(rgb)["patch_tokens"].shape == (2, 256, 384)
    custom = dict(channels=96, num_patches=64, group_size=16, depth=1, num_heads=4,
                  mlp_ratio=3., geometry_scale=0.2, rope_scale=0.1, rope_base=50.)
    encoder = SpatialObservationEncoder(
        pc_encoder_config=custom, rgb_backbone_config=dict(architecture=Dinov2Config(
            hidden_size=48, num_hidden_layers=1, num_attention_heads=4,
        ).to_dict()),
    ).to(device).eval()
    assert encoder.effective_config["pc_encoder_config"] == custom
    assert encoder.patch_encoder.out_dim == 96 and encoder.global_encoder.out_dim == 192
    assert encoder.global_encoder.blocks[0].ffn[0].out_features == 576
    assert encoder.global_encoder.rope.scale == 0.1
    clone = SpatialObservationEncoder(**encoder.effective_config).to(device).eval()
    clone.load_state_dict(encoder.state_dict(), strict=True)
    patches = encoder.patch_encoder(torch.rand(2, 1024, 3, device=device))
    assert patches["tokens"].shape == (2, 64, 96)
    tokens = encoder.rgb_fusion(
        patches["tokens"], torch.rand(2, 4, 4, 48, device=device),
        ImageProjection(torch.zeros(2, 64, 2, device=device),
                        torch.ones(2, 64, dtype=torch.bool, device=device)),
    )
    assert encoder.global_encoder(tokens, patches["xyz"]).shape == (2, 64, 192)
    print("PASS default DINOv2-Small grid and nondefault config width reconstruction", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    device = torch.device(args.device)
    check_configs()
    check_config_widths(device)
    check_encoder(device)
    check_projection(device)
    check_fusion(device)
    check_rgb_gradients(device)
    check_policies(device)


if __name__ == '__main__':
    main()
