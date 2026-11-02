"""Numerical, sampling and synthetic policy checks: python -m dexmani_policy.test_tools.agent_checks."""

import argparse
import copy
import io
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import zarr

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import ConsistencyDiTX, CoupledDiTX, DiTX
from dexmani_policy.agents.action_decoders.backbones.one_way_transformer import OneWayTransformer
from dexmani_policy.agents.action_decoders.sampling.time_sampling import TimeSampler, euler_time_grid
from dexmani_policy.agents.hand_prior.action_io import SATActionIO
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, load_codebook
from dexmani_policy.agents.hand_prior.vqvae import EMAVectorQuantizer, HandVQVAE
from dexmani_policy.agents.obs_encoder.interaction.future_tactile import (
    ForceDiffusion, ForceGuidedFusion, ForceTransformer, VisuoTactileAttention,
)
from dexmani_policy.agents.obs_encoder.interaction.hand_interaction import (
    HandHistoryEncoder, HandSceneFusion, SceneTokenizer, _sample_region_seeds,
)
from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandForceEncoder, XHandPatchEncoder
from dexmani_policy.agents.policies.dp import DP
from dexmani_policy.agents.policies.dp3 import DP3
from dexmani_policy.agents.policies.dqrise import DQRISE
from dexmani_policy.common.checkpoint import Checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.config import validate_observation_fields
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.sampler import SequenceSampler, window_indices
from dexmani_policy.training.common.optimization import build_optimizer
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.pretrain.hand_prior.codebook import export_codebook
from dexmani_policy.training.pretrain.hand_prior.vqvae import evaluate_vq


def check_sampler(_device):
    ends = np.array([1, 4, 9], dtype=np.int64)
    mask = np.array([True, False, True])
    expected = np.array([
        [0, 1, 3], [0, 1, 2], [0, 1, 1], [0, 1, 0],
        [4, 5, 3], [4, 6, 2], [4, 7, 1], [4, 8, 0],
        [5, 9, 0], [6, 9, 0], [7, 9, 0], [8, 9, 0],
    ], dtype=np.int64)
    actual = window_indices(ends, 4, mask, 3, 3)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(window_indices(ends[:1], 1, mask[:1]), [[0, 1, 0]])
    np.testing.assert_array_equal(
        window_indices(ends, 4, np.ones(3, dtype=bool)), [[4, 8, 0], [5, 9, 0]],
    )
    for episode_ends, horizon, selected in (
        (ends, 4, np.zeros(3, dtype=bool)),
        (ends, 32, mask),
        (ends[:0], 1, mask[:0]),
    ):
        empty = window_indices(episode_ends, horizon, selected)
        assert empty.shape == (0, 3) and empty.dtype == np.int64

    rows = np.array([
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        [4, 4, 4, 4], [4, 4, 4, 5], [4, 4, 5, 6], [4, 5, 6, 7],
        [5, 6, 7, 8], [6, 7, 8, 8], [7, 8, 8, 8], [8, 8, 8, 8],
    ])
    values = np.arange(9 * 4, dtype=np.float32).reshape(9, 4)
    with tempfile.TemporaryDirectory(prefix="dexmani-sampler-check-") as directory:
        group = zarr.open_group(directory, mode="w")
        group.create_dataset("meta/episode_ends", data=ends)
        for key in ("action", "state"):
            group.create_dataset(f"data/{key}", data=values, chunks=(3, 4))
        sampler = SequenceSampler(
            EpisodeStore(directory), 4, 3, 3, episode_mask=mask,
            fields={"action": slice(1, 4, 2), "state": None}, field_horizons={"state": 2},
        )
        assert len(sampler) == len(rows)
        for index, expected_rows in enumerate(rows):
            np.testing.assert_array_equal(sampler.sample_source_rows(index), expected_rows)
            sample = sampler.sample_sequence(index)
            np.testing.assert_array_equal(sample["action"], values[expected_rows, 1:4:2])
            np.testing.assert_array_equal(sample["state"], values[expected_rows[:2]])


def check_time(device):
    grid = euler_time_grid(1024, torch.empty((), device=device, dtype=torch.bfloat16))
    assert grid.dtype == torch.float32 and grid.unique().numel() == 1024
    assert grid[0] == 0 and grid[-1] < 1 and (grid[1:] > grid[:-1]).all()
    sampler = TimeSampler(10)
    rng = torch.random.get_rng_state() if device.type == "cpu" else torch.cuda.get_rng_state(device)
    expected = 0.999 * torch.distributions.Beta(
        torch.tensor(1.0, device=device), torch.tensor(1.5, device=device),
    ).sample(torch.Size((128,)))
    if device.type == "cpu":
        torch.random.set_rng_state(rng)
    else:
        torch.cuda.set_rng_state(rng, device)
    assert torch.equal(expected, sampler.sample(128, "beta", device))
    for value in (0, -1, float("nan"), float("inf"), 1e-99, 1e99):
        try:
            TimeSampler(10, beta_alpha=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid FP32 beta concentration: {value}")


def check_precision(device):
    models = (
        (DiTX(16, 19, 2, 256, hidden_dim=32, n_head=4, n_layers=1, p_drop_attn=0.0,
              qkv_bias=True, qk_norm=True, action_io=SATActionIO(16, 32)),
         "ditx_blocks", (2, 16, 19), (2, 194, 256)),
        (DiT(4, 3, 16, n_emb=32, num_heads=4, n_layers=1), "blocks", (2, 4, 3), (2, 16)),
        (ConsistencyDiTX(4, 3, 2, 16, hidden_dim=32, n_head=4, n_layers=1),
         "ditx_blocks", (2, 4, 3), (2, 4, 16)),
    )
    for model, blocks, action_shape, context_shape in models:
        model.to(device)
        seen = []
        hook = getattr(model, blocks)[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].dtype))
        kwargs = dict(x=torch.randn(action_shape, device=device), timestep=torch.rand(2, device=device),
                      context=torch.randn(context_shape, device=device))
        if isinstance(model, ConsistencyDiTX):
            kwargs["target_t"] = torch.rand(2, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = model(**kwargs)
        hook.remove()
        assert seen == [torch.bfloat16] and torch.isfinite(output).all()


def check_decoder_precision(device):
    from dexmani_policy.agents.action_decoders.diffusion import Diffusion
    from dexmani_policy.agents.action_decoders.flow import ConsistencyFlow, RectifiedFlow

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.1, device=device))

        def prepare_context_kv(self, context):
            return None

        def forward(self, x, timestep, context, **kwargs):
            self.input = x.detach()
            return x * self.gain.to(x.dtype)

    cond = torch.zeros(2, 4, device=device, dtype=torch.bfloat16)
    shape = (2, 4, 3)
    for kind in ("sample", "epsilon", "v_prediction"):
        model = Head()
        decoder = Diffusion(model, prediction_type=kind)
        for dtype in (torch.float32, torch.bfloat16):
            actions = torch.zeros(shape, device=device, dtype=dtype)
            noise = torch.ones_like(actions)
            t = torch.zeros(2, device=device, dtype=torch.long)
            with patch("torch.randn_like", return_value=noise), patch("torch.randint", return_value=t):
                loss, _ = decoder.compute_loss(cond, actions)
            a = decoder.noise_scheduler.alphas_cumprod[0].float()
            expected = ((1 - a).sqrt() * noise).to(dtype)
            torch.testing.assert_close(model.input, expected, rtol=0, atol=0)
            assert model.input.abs().min() > 0
            target = actions if kind == "sample" else noise if kind == "epsilon" else (a.sqrt() * noise).to(dtype)
            torch.testing.assert_close(loss, F.mse_loss((expected * model.gain.to(dtype)).float(), target.float()))
            loss.backward()
            assert torch.isfinite(model.gain.grad)

    for decoder in (Diffusion(Head()), RectifiedFlow(Head()), ConsistencyFlow(Head())):
        decoder.eval()
        def sample(**kwargs):
            return decoder.predict_action(cond, shape, 4,
                generator=torch.Generator(device=device).manual_seed(10), **kwargs)
        with torch.inference_mode():
            ordinary = sample()
            assert ordinary.dtype == torch.float32
            torch.testing.assert_close(ordinary, sample(dtype=torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(ordinary, sample(rtc_prefix=ordinary[:, :2], rtc_guidance_cap=0), rtol=0, atol=0)
            assert sample(dtype=torch.bfloat16).dtype == torch.bfloat16
            guided = sample(rtc_prefix=ordinary[:, :2], rtc_guidance_cap=2)
            assert guided.dtype == torch.float32 and torch.isfinite(guided).all()


def check_tactile(device):
    encoder = XHandPatchEncoder().to(device)
    for mode in ("random", "zero", "ties"):
        x = torch.randn(2, 5, 120, 3, device=device)
        if mode == "zero":
            x.zero_()
        elif mode == "ties":
            x = x.sign()
        x.requires_grad_()
        mask = (encoder.region_ids[:, None] == torch.arange(5, device=device)[None, :, None])[None]
        magnitude = torch.linalg.vector_norm(x, dim=-1)
        weights = magnitude / (magnitude + encoder.activity_scale)
        weights = weights[:, :, None] * mask
        mass = weights.sum(-1)
        mean = torch.einsum("bfpt,bftc->bfpc", weights, x) / torch.where(mass > 0, mass, 1)[..., None]
        expected = torch.cat((
            mean, (x.abs()[:, :, None] * mask[..., None]).amax(-2),
            (mass / mask.sum(-1))[..., None],
            (magnitude[:, :, None] * mask).amax(-1)[..., None],
        ), -1)
        actual = encoder._patch_descriptors(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        probe = torch.randn_like(actual)
        reference_grad, = torch.autograd.grad((expected * probe).mean(), x)
        new_grad, = torch.autograd.grad((actual * probe).mean(), x)
        torch.testing.assert_close(reference_grad, new_grad, rtol=1e-5, atol=1e-6)
    with patch.object(XHandPatchEncoder, "__init__", autospec=True,
                      side_effect=XHandPatchEncoder.__init__) as constructor:
        policy = hydra.utils.instantiate({
            "_target_": "dexmani_policy.agents.policies.adaptac.AdapTac",
            "horizon": 4, "n_obs_steps": 1, "n_action_steps": 2, "action_dim": 19,
            "tactile_source": "real", "tactile_encoder": encoder.effective_config,
            "pc_dim": 3, "num_points": 16, "feature_dim": 16, "state_out_dim": 8,
            "pc_encoder_config": dict(channels=24, num_patches=4, group_size=4, depth=1, num_heads=4),
            "down_dims": [16, 32], "diffusion_step_embed_dim": 32,
            "force_predictor_config": dict(depth=1, num_heads=4, embed_dim=16),
            "fusion_num_heads": 4,
        }, _convert_="all")
    assert type(constructor.call_args.kwargs["region_ids"]) is list
    restored = policy.obs_encoder.tactile_encoder.to(device)
    assert isinstance(restored, XHandPatchEncoder)
    restored.load_state_dict(encoder.state_dict(), strict=True)


def check_interaction(device):
    scene = SceneTokenizer(24).to(device)
    for patch_count in (12, 24):
        points = torch.zeros(2, patch_count, 3, device=device)
        seeds = _sample_region_seeds(points, 12)
        assert all(row.unique().numel() == 12 for row in seeds)
        tokens = torch.randn(2, patch_count, 24, device=device, requires_grad=True)
        for bf16 in (False, True):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                output = scene(tokens, points)
            assert output.shape == (2, 12, 24) and torch.isfinite(output).all()
            grad, = torch.autograd.grad((output * torch.randn_like(output)).sum(), tokens)
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0

    model = HandSceneFusion(24, XHandForceEncoder()).to(device).eval()
    points = torch.randn(2, 64, 3, device=device) * 0.03
    anchors = torch.randn(2, 6, 3, device=device) * 0.02
    patches = dict(tokens=torch.randn(2, 12, 24, device=device), xyz=points[:, :12])
    hand = torch.randn(2, 6, 192, device=device)
    rotation = torch.eye(3, device=device).expand(2, -1, -1)
    context = torch.randn(2, 192, device=device)
    tactile = torch.randn(2, 5, 3, device=device)
    kwargs = dict(rotation_world_from_eef=rotation, query_context=context, tactile=tactile)
    output = model(points, patches, anchors, hand, **kwargs)
    changed = tactile.clone()
    changed[:, 2] += 2
    finger_output = model(points, patches, anchors, hand, **{**kwargs, "tactile": changed})
    torch.testing.assert_close(output[:, [1, 2, 4, 5]], finger_output[:, [1, 2, 4, 5]])
    assert (output[:, [0, 3]] - finger_output[:, [0, 3]]).abs().sum() > 0

    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
    shift = torch.tensor([0.2, -0.1, 0.3], device=device)
    rotated = model(
        points @ rotation.T + shift,
        {**patches, "xyz": patches["xyz"] @ rotation.T + shift},
        anchors @ rotation.T + shift, hand,
        **{**kwargs, "rotation_world_from_eef": rotation.expand(2, -1, -1)},
    )
    torch.testing.assert_close(output, rotated, rtol=1e-5, atol=1e-6)

    geometry_inputs = []
    hook = model.geometry_projection.register_forward_pre_hook(
        lambda _, args: geometry_inputs.append(args[0].detach())
    )
    for bf16 in (False, True):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            empty = model(points + 10, patches, anchors, hand, **kwargs)
        assert empty.dtype == torch.float32 and torch.isfinite(empty).all()
        assert torch.count_nonzero(geometry_inputs[-1][..., :32]) == 0
        (empty * torch.randn_like(empty)).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    hook.remove()

    history = HandHistoryEncoder(4).to(device).eval()
    tokens = torch.randn(2, 4, 6, 192, device=device)
    changed = tokens.clone()
    changed[:, 2:] += torch.randn_like(changed[:, 2:])
    for bf16 in (False, True):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            original, perturbed = history(tokens), history(changed)
        torch.testing.assert_close(original[:, :2], perturbed[:, :2], rtol=0, atol=0)


def check_force_interaction(device):
    interaction = VisuoTactileAttention(24, 64, 5).to(device).eval()
    visual = torch.randn(2, 12, 24, device=device, requires_grad=True)
    tactile = torch.randn(2, 5, 64, device=device, requires_grad=True)
    v, t = interaction(visual, tactile)
    vg, = torch.autograd.grad((t * torch.randn_like(t)).mean(), visual, retain_graph=True)
    tg, = torch.autograd.grad((v * torch.randn_like(v)).mean(), tactile)
    assert vg.abs().sum() > 0 and tg.abs().sum() > 0

    predictor = ForceDiffusion(ForceTransformer(4, 32, 15, depth=1, embed_dim=32, dropout=0)).to(device)
    fusion = ForceGuidedFusion(16, 16, 75, hidden_dim=32, num_heads=4, dropout=0).to(device)
    noisy = torch.randn(2, 4, 15, device=device)
    changed = noisy.clone()
    changed[:, 2:] += 10
    condition = torch.randn(2, 1, 32, device=device)
    torch.testing.assert_close(
        predictor.model(noisy, 3, condition)[:, :2],
        predictor.model(changed, 3, condition)[:, :2], rtol=0, atol=0,
    )
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]], device=device)
    force = torch.zeros(2, 4, 15, device=device)
    for bf16 in (False, True):
        predictor.zero_grad(set_to_none=True)
        fusion.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            loss, prediction = predictor.compute_loss_and_prediction(
                torch.randn(2, 1, 32, device=device), force, valid,
            )
            guide = torch.cat((torch.zeros(2, 1, 15, device=device), prediction), 1).flatten(1)
            output = fusion(torch.randn(2, 16, device=device), torch.randn(2, 16, device=device), guide)
        torch.testing.assert_close(loss, prediction[valid].square().mean())
        target_grad, = torch.autograd.grad(loss, prediction, retain_graph=True)
        assert torch.count_nonzero(target_grad[~valid]) == 0
        assert target_grad[valid].abs().sum() > 0
        (output * torch.randn_like(output)).sum().backward()
        assert predictor.model.head.weight.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())


def check_token_decoders(device):
    coupled = CoupledDiTX(
        horizon=8, action_dim=19, n_obs_steps=2, obs_token_dim=16,
        hidden_dim=32, n_head=4, n_layers=2, p_drop_attn=0, interaction_horizon=3,
    ).to(device).eval()
    # Zero-initialized gates and readout would hide a broken attention mask.
    for block in coupled.ditx_blocks:
        torch.nn.init.normal_(block.adaln_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(coupled.final_layer.ffn_final.fc2.weight, std=0.1)
    actions = torch.randn(2, 8, 19, device=device)
    context = torch.randn(2, 18, 16, device=device)
    timestep = torch.rand(2, device=device)
    outside = actions.clone()
    outside[:, [0, 4, 5, 6, 7]] += 3
    changed_body = context.clone()
    changed_body.unflatten(1, (2, 9))[:, -1, :6] += 2
    auxiliary = OneWayTransformer(
        horizon=8, action_dim=28, n_obs_steps=2, obs_token_dim=48, pc_pe_dim=32,
        embedding_dim=32, depth=2, num_heads=4, mlp_dim=64,
        use_aux_ee=True, joint_dim=19, ee_dim=9,
    ).to(device).eval()
    joint_ee = torch.randn(2, 8, 28, device=device)
    changed_ee = joint_ee.clone()
    changed_ee[..., 19:] += 3
    observations = torch.randn(2, 8, 48, device=device)
    for bf16 in (False, True):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            velocity, features = coupled(actions, timestep, context, return_features=True)
            perturbed, perturbed_features = coupled(outside, timestep, context, return_features=True)
            cached_kv = coupled.prepare_context_kv(context)
            cached = coupled(actions, timestep, context, context_kv=cached_kv, return_features=True)
            dynamic = coupled(actions, timestep, changed_body, context_kv=cached_kv, return_features=True)
            prediction = auxiliary(joint_ee, timestep, observations)
            perturbed_ee = auxiliary(changed_ee, timestep, observations)
        torch.testing.assert_close(velocity[:, 1:4], perturbed[:, 1:4], rtol=0, atol=0)
        torch.testing.assert_close(features[:, 8:], perturbed_features[:, 8:], rtol=0, atol=0)
        assert (velocity[:, 4:] - perturbed[:, 4:]).abs().sum() > 0
        torch.testing.assert_close(cached, (velocity, features), rtol=0, atol=0)
        assert (dynamic[1][:, 8:] - features[:, 8:]).abs().sum() > 0
        torch.testing.assert_close(prediction[..., :19], perturbed_ee[..., :19], rtol=0, atol=0)
        assert (prediction[..., 19:] - perturbed_ee[..., 19:]).abs().sum() > 0


def check_policy_roundtrips(device):
    for method, auxiliary in (("dp", False), ("afro", False), ("maniflow", False),
                              ("dexlatent", False), ("cordvip", False), ("adaptac", False), ("interact_dex", False),
                              ("sat", False), ("r3d", False), ("r3d", True)):
        overrides = ["horizon=8", "n_action_steps=2", "agent.num_inference_steps=2"]
        if method not in ("r3d", "cordvip", "afro", "dp"):
            overrides += ["agent.pc_dim=3"]
        if method in ("interact_dex", "sat", "dexlatent", "maniflow"):
            overrides += ["agent.n_layers=1", "agent.hidden_dim=32", "agent.n_head=4",
                          "agent.timestep_embed_dim=32"]
        elif method == "r3d":
            overrides += [f"use_aux_ee={str(auxiliary).lower()}", "agent.embedding_dim=32",
                          "agent.depth=1", "agent.num_heads=4", "agent.mlp_dim=64",
                          "agent.timestep_embed_dim=32", "agent.num_training_steps=10",
                          "agent.pc_encoder_config.embed_dim=32", "agent.pc_encoder_config.num_group=4",
                          "agent.pc_encoder_config.group_size=8", "agent.pc_encoder_config.drop_path_rate=0",
                          "agent.fps_random_config.use_random=false"]
        else:
            overrides += ["agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=32",
                          "agent.num_training_steps=10"]
        if method not in ("cordvip", "r3d", "afro", "dp", "maniflow"):
            overrides += ["agent.num_points=32", "agent.pc_encoder_config.channels=24",
                          "agent.pc_encoder_config.num_patches=12", "agent.pc_encoder_config.group_size=8",
                          "agent.pc_encoder_config.depth=1", "agent.pc_encoder_config.num_heads=4",
                          "agent.fps_random_config.use_random=false"]
        if method == "adaptac":
            overrides += ["agent.feature_dim=32", "agent.state_out_dim=8", "agent.force_num_inference_steps=2",
                          "agent.force_predictor_config.depth=1", "agent.force_predictor_config.embed_dim=32"]
        if method in ("sat", "dexlatent"):
            overrides += ["agent.rgb_backbone_config.image_size=[28,28]",
                          "+agent.rgb_backbone_config.architecture={model_type:dinov2,hidden_size:24,"
                          "intermediate_size:48,num_hidden_layers:1,num_attention_heads:4,image_size:28,patch_size:14}"]
        if method == "dp":
            overrides += ["agent.rgb_backbone_name=resnet"]
        if method == "maniflow":
            overrides += ["agent.num_points=32",
                          "agent.pc_encoder_config.out_channels=32", "agent.state_out_dim=8",
                          "agent.fps_random_config.use_random=false"]
        cfg = load_policy_config(method, overrides)
        if method == "dp":
            cfg.agent.rgb_backbone_config = {}
        agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
        policy = hydra.utils.instantiate(agent_cfg, _convert_="all")
        validate_observation_fields(policy, cfg.dataset.sensor_modalities)
        shapes = dict(joint_state=(19,), point_cloud=(32, 3), hand_pointcloud=(32, 3),
                      object_pointcloud=(32, 3), finger_tactile_force=(5, 3),
                      eef_pose=(9,), fingertip_xyz=(5, 3), rgb=(3, 28, 28),
                      camera_intrinsic=(9,), camera_extrinsic=(12,))
        if method == "r3d":
            shapes["point_cloud"] = (32, 6)
        obs = {key: torch.randn(2, cfg.n_obs_steps, *shapes[key], device=device)
               for key in cfg.dataset.sensor_modalities}
        if "eef_pose" in obs:
            obs["eef_pose"][..., 3:] = torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
        if method in ("sat", "dexlatent"):
            obs["rgb"] = torch.randint(256, obs["rgb"].shape, device=device, dtype=torch.uint8)
            obs["point_cloud"][..., 2] = obs["point_cloud"][..., 2].abs() + 1
            intrinsic = torch.tensor([[20., 0., 13.5], [0., 20., 13.5], [0., 0., 1.]], device=device)
            obs["camera_intrinsic"] = intrinsic.flatten().expand(2, cfg.n_obs_steps, -1)
            obs["camera_extrinsic"] = torch.eye(4, device=device)[:3].flatten().expand(2, cfg.n_obs_steps, -1)
        dims = {key: (3 if key == "rgb" else value.shape[-1]) for key, value in obs.items()}
        dims["action"] = cfg.action_dim
        normalizer_shapes = {key: shape for key, shape in normalization_feature_shapes(dims).items()
                             if cfg.normalization[key] != "identity"}
        if method == "dexlatent":
            normalizer_shapes["hand_latent"] = (32,)
        policy.configure_normalizer(dims, normalizer_shapes)
        policy.to(device)
        optimizer = build_optimizer(policy, **cfg.optimizer)
        batch = dict(obs=obs, action=torch.randn(2, cfg.horizon, cfg.action_dim, device=device))
        if method in ("adaptac", "interact_dex"):
            count = 1 if method == "interact_dex" else policy.future_steps
            valid = torch.ones(2, count, dtype=torch.bool, device=device)
            valid[1, -1] = False
            batch.update(future_force=torch.randn(2, count, 5, 3, device=device), future_force_valid=valid)
        ema = copy.deepcopy(policy).eval().requires_grad_(False) if method == "maniflow" else None
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss, metrics = policy.compute_loss(batch, **policy.get_training_loss_kwargs(ema))
            if policy.loss_groups is not None:
                expected = sum(metrics[f"loss_{name}"] * weight
                               for name, (_, _, weight) in policy.loss_groups.items())
                torch.testing.assert_close(metrics["loss_action"], expected)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
            optimizer.step()
        policy.eval()
        saved_agent = {**agent_cfg, **policy.effective_config}
        with tempfile.TemporaryDirectory(prefix="dexmani-policy-check-") as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, Checkpoint(
                2, policy.state_dict(), None, optimizer.state_dict(), {}, get_rng_state(device),
                dict(agent=saved_agent, normalization_dims=dims, normalizer_shapes=normalizer_shapes),
            ))
            # Deployment restores all parameters in FP32, including the frozen RGB encoder.
            policy.float()
            with torch.no_grad():
                expected = policy.predict_action(obs, generator=torch.Generator(device=device).manual_seed(11))
            assert expected["future_actions"].shape == (2, policy.future_steps, policy.control_action_dim)
            restored = restore_policy(path, use_ema=False, device=device)
            torch.testing.assert_close(restored.state_dict(), policy.state_dict(), rtol=0, atol=0)
            with torch.no_grad():
                actual = restored.predict_action(obs, generator=torch.Generator(device=device).manual_seed(11))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if "rtc" in restored.supported_execution_modes:
                restored.zero_grad(set_to_none=True)
                prefix = expected["future_actions"][:, :3]
                with torch.inference_mode():
                    unguided = restored.predict_action(
                        obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=0,
                        generator=torch.Generator(device=device).manual_seed(11),
                    )
                    guided = restored.predict_action(
                        obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=2,
                        generator=torch.Generator(device=device).manual_seed(11),
                    )["future_actions"]
                torch.testing.assert_close(unguided, expected, rtol=0, atol=0)
                assert guided.shape == expected["future_actions"].shape
                assert torch.isfinite(guided).all() and not guided.requires_grad
                assert all(p.grad is None for p in restored.parameters())
        print(f"PASS policy roundtrip: {method}, auxiliary={auxiliary} ({device})", flush=True)



def check_output_boundaries(device):
    from types import SimpleNamespace
    from dexmani_policy.deployment.runtime import PolicyRuntime
    from dexmani_policy.env_runner.sim_runner import SimRunner

    class OutputPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), device=device))
            self.output = torch.zeros(1, 3, 19, device=device)

        def predict_action(self, *args, **kwargs):
            return {"future_actions": self.output}

    agent = OutputPolicy()
    runtime = PolicyRuntime(agent, SimpleNamespace(inference_steps=2), seed=0)
    runner = SimRunner("test", 2, [], {})
    with patch.object(runtime, "prepare_observations", return_value={}):
        assert runtime.predict({}).shape == (3, 19)
        assert runner.predict_action_chunk({}, agent, action_steps=1).shape == (1, 19)
        for value in (float("nan"), float("inf")):
            agent.output[0, -1, 0] = value
            for predict in (lambda: runtime.predict({}),
                            lambda: runner.predict_action_chunk({}, agent, action_steps=1)):
                try:
                    predict()
                except ValueError as error:
                    assert "Nonfinite physical" in str(error)
                else:
                    raise AssertionError("Nonfinite future actions escaped the CPU boundary")

def check_rgb(device):
    architecture = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, image_size=32, patch_size=16)
    for mode in ("freeze", "full", "lora"):
        model = ViTEncoder("siglip", tune_mode=mode, out_dim=16,
                           architecture=architecture).to(device)
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert not hasattr(model.backbone.vision_model, "head")
        assert model.effective_config["architecture"]["vision_use_head"] is False
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        rgb = torch.randn(2, 3, 32, 32, device=device)
        model.train()
        for amp in (False, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model(rgb)["global_token"].float().square().mean()
            loss.backward()
            assert model.proj.weight.grad is not None and model.proj.weight.grad.abs().sum() > 0
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if mode == "freeze":
                assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())
            optimizer.step()
        restored = ViTEncoder("siglip", **model.effective_config)
        snapshot = {name: value.clone() for name, value in model.state_dict().items()}
        restored.load_state_dict(snapshot, strict=True, assign=True)
        restored.to(device)
        restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-3)
        restored_optimizer.load_state_dict(optimizer.state_dict())
        assert list(dict(model.named_parameters())) == list(dict(restored.named_parameters()))
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(rgb)["global_token"],
                                       restored.eval()(rgb)["global_token"], rtol=0, atol=0)
    pooler = ViTEncoder("siglip", global_token_type="pooler", architecture=architecture).to(device)
    assert hasattr(pooler.backbone.vision_model, "head")
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        assert torch.isfinite(pooler(rgb)["global_token"]).all()
    assert "attn_implementation" not in architecture

    from transformers import Dinov2Config, Dinov2Model, SiglipVisionConfig, SiglipVisionModel
    from dexmani_policy.training.common.config import load_policy_config, prepare_training

    for kind, config_class, model_class in (
        ("dino", Dinov2Config, Dinov2Model), ("siglip", SiglipVisionConfig, SiglipVisionModel),
    ):
        for mode in ("freeze", "full", "lora"):
            config = config_class(**architecture)
            if kind == "siglip":
                config.vision_use_head = False
            source_model = model_class(config)
            with tempfile.TemporaryDirectory(prefix="dexmani-rgb-source-") as directory:
                source_model.save_pretrained(directory)
                cfg = load_policy_config("dp", [
                    "horizon=8", "n_action_steps=3", "agent.down_dims=[16,32]",
                    "agent.diffusion_step_embed_dim=32",
                ])
                cfg.initialization.rgb = directory
                cfg.agent.rgb_backbone_name = kind
                cfg.agent.rgb_backbone_config = {"tune_mode": mode, "out_dim": 16, "image_size": [32, 32]}
                prepare_training(cfg)
                plain = cast(dict[str, Any], OmegaConf.to_container(cfg.agent, resolve=True))
                plain.pop("_target_")
                rng = torch.get_rng_state()
                reference = DP(**plain)
                initialized_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                with patch.object(ViTEncoder, "__init__", autospec=True,
                                  side_effect=ViTEncoder.__init__) as constructor:
                    policy = hydra.utils.instantiate(cfg.agent, _convert_="all")
                assert type(constructor.call_args.kwargs["architecture"]) is dict
                assert torch.equal(torch.get_rng_state(), initialized_rng)
                expected_state, actual_state = reference.state_dict(), policy.state_dict()
                assert list(actual_state) == list(expected_state)
                for key in expected_state:
                    torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
                model = policy.obs_encoder.rgb_encoder
                model.initialize_weights(directory)
                assert all(p.dtype == torch.float32 for p in model.parameters())
                base = model.backbone.get_base_model() if mode == "lora" else model.backbone
                source_state = source_model.state_dict()
                for key, value in base.state_dict().items():
                    if ".lora_" not in key:
                        torch.testing.assert_close(value, source_state[key.replace(".base_layer.", ".")], rtol=0, atol=0)
            # The source directory is gone: structure and parameters alone must restore offline.
            restored = ViTEncoder(model.kind, **model.effective_config)
            restored.load_state_dict(model.state_dict(), strict=True)
            model.to(device).eval()
            restored.to(device).eval()
            expected = model(rgb)["global_token"]
            actual = restored(rgb)["global_token"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.float().square().mean().backward()
            actual.float().square().mean().backward()
            for (name, parameter), (other_name, other) in zip(model.named_parameters(), restored.named_parameters()):
                assert name == other_name
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


def check_rgb_deployment(device):
    from peft import PeftModel
    from dexmani_policy.common.checkpoint import load_checkpoint

    for kind in ("dino", "siglip", "dinov3"):
        architecture = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, image_size=32, patch_size=16)
        if kind == "dino":
            architecture["model_type"] = "dinov2"
        elif kind == "dinov3":
            architecture = dict(embed_dim=32, depth=1, num_heads=4)
        kwargs = dict(horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19, state_dim=19,
                      rgb_backbone_name=kind, state_out_dim=8, down_dims=[16, 32],
                      diffusion_step_embed_dim=32, num_training_steps=10, num_inference_steps=2,
                      rgb_backbone_config=dict(tune_mode="lora", architecture=architecture,
                                               out_dim=16, image_size=[32, 32]))
        policy = DP(**kwargs)
        dims = dict(joint_state=19, rgb=3, action=19)
        shapes = dict(joint_state=(19,), action=(19,))
        policy.configure_normalizer(dims, shapes)
        policy.to(device)
        obs = dict(joint_state=torch.randn(2, 2, 19, device=device),
                   rgb=torch.rand(2, 2, 3, 32, 32, device=device))
        batch = dict(obs=obs, action=torch.randn(2, 8, 19, device=device))
        optimizer = build_optimizer(policy, lr=1e-3, weight_decay=0)
        for amp in (False, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, _ = policy.compute_loss(batch)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
            optimizer.step()
        assert any(p.detach().abs().sum() > 0 for n, p in policy.named_parameters() if ".lora_B." in n)
        ema = copy.deepcopy(policy)
        with torch.no_grad():
            for name, param in ema.named_parameters():
                if ".lora_B." in name:
                    param.mul_(0.8)
        config = dict(agent={"_target_": "dexmani_policy.agents.policies.dp.DP", **kwargs},
                      normalization_dims=dims, normalizer_shapes=shapes)
        with tempfile.TemporaryDirectory(prefix="dexmani-lora-restore-") as directory:
            raw, averaged = copy.deepcopy(policy), copy.deepcopy(ema)
            for model in (raw, averaged):
                model.eval()
            path = save_checkpoint(Path(directory) / "model.pt", Checkpoint(
                2, raw.state_dict(), averaged.state_dict(), {}, {}, {}, config,
            ))
            for use_ema, reference in ((False, raw), (True, averaged)):
                reference.float()
                restored = restore_policy(path, use_ema=use_ema, device=device)
                assert not isinstance(restored.obs_encoder.rgb_encoder.backbone, PeftModel)
                assert not any("lora_" in name for name in restored.state_dict())
                assert all(p.dtype == torch.float32 for p in restored.parameters())
                with torch.no_grad():
                    torch.testing.assert_close(restored._encode_obs(obs), reference._encode_obs(obs),
                                               atol=2e-6, rtol=2e-6)
                for rtc in (False, True):
                    options = dict(rtc_prefix=batch["action"][:, :3], delay_steps=1,
                                   rtc_guidance_cap=2) if rtc else {}
                    # Isolate FP32 merge error from TF32 convolution rounding.
                    with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
                        expected = reference.predict_action(obs, **options,
                            generator=torch.Generator(device=device).manual_seed(31))
                        actual = restored.predict_action(obs, **options,
                            generator=torch.Generator(device=device).manual_seed(31))
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5,
                        msg=lambda message: f"{kind}, EMA={use_ema}, RTC={rtc}: {message}")
            saved = load_checkpoint(path)
            assert any("lora_" in name for name in saved.model_state)
            resumed = DP(**kwargs)
            resumed.configure_normalizer(dims, shapes)
            resumed.load_state_dict(saved.model_state, strict=True, assign=True)
            assert isinstance(resumed.obs_encoder.rgb_encoder.backbone, PeftModel)
            assert all(p.dtype == torch.float32 for p in resumed.parameters())


def check_dinov3(device):
    from safetensors.torch import save_file
    from dexmani_policy.agents.obs_encoder.spatial import SpatialObservationEncoder
    from dexmani_policy.training.common.config import prepare_training

    architecture = dict(embed_dim=32, depth=1, num_heads=4)
    source = ViTEncoder("dinov3", architecture=architecture).backbone.state_dict()
    rgb = torch.randn(2, 3, 32, 48, device=device)
    for mode in ("freeze", "full", "lora"):
        model = ViTEncoder("dinov3", tune_mode=mode, architecture=architecture, out_dim=16)
        with tempfile.TemporaryDirectory(prefix="dexmani-dinov3-source-") as directory:
            save_file(source, str(Path(directory) / "model.safetensors"))
            model.initialize_weights(directory)
        base = model.backbone.get_base_model() if mode == "lora" else model.backbone
        for name, value in base.state_dict().items():
            if ".lora_" not in name:
                torch.testing.assert_close(value, source[name.replace(".base_layer.", ".")], rtol=0, atol=0)
        assert model.num_prefix_tokens == 5 and model.patch_size == 16
        assert all(p.dtype == torch.float32 for p in model.parameters())
        model.to(device).eval()
        with torch.no_grad():
            hidden = model.backbone.forward_features(rgb)
            output = model(rgb)
            assert output["patch_tokens"].shape == (2, 6, 16)
            torch.testing.assert_close(output["patch_tokens"], model.proj(hidden[:, 5:]))
            torch.testing.assert_close(output["global_token"], output["patch_tokens"].mean(1))
            model.global_token_type = "cls"
            torch.testing.assert_close(model(rgb)["global_token"], model.proj(hidden[:, 0]))
            model.global_token_type = "avg"
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        for amp in (False, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model(rgb)["global_token"].float().square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            if mode == "freeze":
                assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())
            else:
                gradients = [p.grad for p in model.backbone.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            if mode == "lora":
                qkv = model.backbone.get_base_model().blocks[0].attn.qkv
                assert qkv.lora_B["default"].weight.grad.abs().sum() > 0
            optimizer.step()
        restored = ViTEncoder("dinov3", **model.effective_config).to(device)
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored.eval()(rgb), model.eval()(rgb), rtol=0, atol=0)

    for method in ("dp", "sat", "dqrise", "dexlatent"):
        cfg = load_policy_config(method, ["agent.rgb_backbone_name=dinov3"])
        with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("unexpected HF config")):
            prepare_training(cfg)
        assert cfg.agent.rgb_backbone_name == "dinov3"

    for kind in ("dino", "siglip", "dinov3"):
        arch = architecture if kind == "dinov3" else dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, image_size=32, patch_size=16,
        )
        if kind == "dino":
            arch = {"model_type": "dinov2", **arch}
        encoder = SpatialObservationEncoder(
            num_points=32, n_obs_steps=2, rgb_backbone_name=kind,
            rgb_backbone_config=dict(architecture=arch, image_size=[32, 32]),
            pc_encoder_config=dict(channels=24, num_patches=12, group_size=8, depth=1, num_heads=4),
            fps_random_config=dict(use_random=False, use_random_start=False),
        ).to(device).eval()
        intrinsic = torch.tensor([[20., 0., 15.5], [0., 20., 15.5], [0., 0., 1.]], device=device)
        points = torch.rand(2, 32, 3, device=device) * 0.2
        points[..., 2] += 1
        obs = dict(point_cloud=points, joint_state=torch.randn(2, 19, device=device),
                   rgb=torch.randint(256, (2, 3, 32, 32), dtype=torch.uint8, device=device),
                   camera_intrinsic=intrinsic.expand(2, -1, -1),
                   camera_extrinsic=torch.eye(4, device=device)[:3].expand(2, -1, -1))
        output = encoder(obs)
        assert output.shape == (1, 26, 256) and torch.isfinite(output).all()
        output.square().mean().backward()
        assert encoder.alignment.proj.weight.grad.abs().sum() > 0
        restored = SpatialObservationEncoder(num_points=32, n_obs_steps=2, **encoder.effective_config).to(device).eval()
        restored.load_state_dict(encoder.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(obs), output, rtol=0, atol=0)


def check_vq(device):
    quantizer = EMAVectorQuantizer(12, 4).to(device)
    for _ in range(4):
        x = torch.randn(32, 12, device=device, requires_grad=True)
        state = {k: v.clone() for k, v in quantizer.state_dict().items()}
        expected_indices = (x.detach()[:, None] - state["embed"]).square().sum(-1).argmin(-1)
        assignment = F.one_hot(expected_indices, 4).float()
        counts = state["cluster_size"].lerp(assignment.sum(0), 1 - quantizer.decay)
        sums = state["embed_avg"].lerp(assignment.T @ x.detach(), 1 - quantizer.decay)
        smoothed = (counts + quantizer.eps) / (counts.sum() + 4 * quantizer.eps) * counts.sum()
        output, indices, loss = quantizer(x)
        assert torch.equal(indices, expected_indices)
        torch.testing.assert_close(quantizer.cluster_size, counts)
        torch.testing.assert_close(quantizer.embed_avg, sums, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(quantizer.embed, sums / smoothed[:, None], rtol=1e-5, atol=1e-6)
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        assert torch.isfinite(loss)

    quantizer.eval()
    with torch.no_grad():
        quantizer.embed.fill_(3)
        quantizer.embed[0].fill_(1)
        quantizer.embed[0, 0] += 1e-4
        quantizer.embed[1].fill_(1)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        assert quantizer(torch.ones(1, 12, device=device))[1].item() == 1

    initialized = EMAVectorQuantizer(3, 4, kmeans_init=True, kmeans_iters=2).to(device)
    initialized(torch.zeros(8, 3, device=device))
    assert initialized.is_initialized and initialized.cluster_size.sum() > 0
    restored = EMAVectorQuantizer(3, 4, kmeans_init=True).to(device)
    restored.load_state_dict(initialized.state_dict(), strict=True)
    assert restored.is_initialized
    restored.eval()(torch.zeros(8, 3, device=device))

    dead = EMAVectorQuantizer(3, 4, threshold_ema_dead_code=2).to(device)
    dead.update_codebook(torch.ones(8, 3, device=device), torch.zeros(8, device=device, dtype=torch.long))
    assert (dead.cluster_size >= 2).all()
    torch.testing.assert_close(dead.embed_avg[1:], dead.embed[1:] * 2)

    vae = HandVQVAE(12, [1] * 12, latent_dim=8, hidden_dim=16, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3, fused=device.type == "cuda")
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, commitment, _ = vae(torch.randn(16, 12, device=device))
        (reconstruction + commitment).backward()
        assert vae.quantizer.layer_weights.grad is not None and vae.quantizer.layer_weights.grad.abs().sum() > 0
        optimizer.step()
    loaded = HandVQVAE.from_checkpoint({
        "model_config": vae.model_config,
        "model_state_dict": {name: value.cpu() for name, value in vae.state_dict().items()},
    }, device=device)
    assert all(torch.isfinite(p).all() for p in loaded.parameters())


def check_codebook(device):
    for size in (256, 512):
        table = HandStateCodebook(12, num_groups=1, codebook_size=size).to(device)
        table.set_prototypes(torch.randn(size, 12, device=device))
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            _, index = table.decode(torch.tensor([-2, -1, 1, 2], device=device, dtype=dtype))
            assert index.tolist() == [0, 0, size - 1, size - 1]
        for invalid in (float("nan"), float("inf"), -float("inf")):
            try:
                table.decode(torch.tensor([invalid], device=device))
            except ValueError:
                pass
            else:
                raise AssertionError("Nonfinite code index was accepted")
        with torch.no_grad():
            table.sorted_hand_poses.fill_(3)
            table.sorted_hand_poses[0].fill_(1)
            table.sorted_hand_poses[0, 0] += 1e-4
            table.sorted_hand_poses[1].fill_(1)
        for shape in ((12,), (2, 12), (2, 3, 12)):
            continuous, indices = table.encode(torch.ones(shape, device=device))
            assert indices.shape == shape[:-1] and (indices == 1).all()
            assert continuous.shape == (*shape[:-1], 1)
        restored = HandStateCodebook(12, num_groups=1, codebook_size=size).to(device)
        restored.load_state_dict(table.state_dict(), strict=True)
        assert set(restored.state_dict()) == {"sorted_hand_poses"}


def check_codebook_pipeline(device):
    model = HandVQVAE(12, [1] * 12, latent_dim=8, hidden_dim=16, num_layers=1).to(device)
    rows = torch.randn(19, 12).clamp(-1, 1)
    loader = DataLoader(TensorDataset(rows), batch_size=7)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = evaluate_vq(model, loader, device)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    manifest = dict(source_id="agent-check", episode_ends=[10, 19], episode_ids=["train", "val"],
                    train_indices=[0], val_indices=[1])
    with tempfile.TemporaryDirectory(prefix="dexmani-codebook-check-") as directory:
        vq_checkpoint, artifact = Path(directory) / "vq.pt", Path(directory) / "codebook.npz"
        torch.save(dict(
            model_config=model.model_config, model_state_dict=model.state_dict(), epoch=1,
            metrics={"val_lookup_mse": metrics["lookup_mse"]},
            normalizer_state_dict={"fields.hand.scale": torch.ones(12), "fields.hand.offset": torch.zeros(12)},
            split_metadata=dict(split_manifest=manifest, selection_split="validation", selection_metric="val_lookup_mse"),
        ), vq_checkpoint)
        table, metadata = export_codebook(vq_checkpoint, artifact)
        loaded_table = HandStateCodebook(
            model.hand_dim, num_groups=model.num_groups, codebook_size=model.codebook_size,
        )
        loaded_metadata, _ = load_codebook(artifact, loaded_table)
        assert loaded_metadata == metadata
        for name, value in table.state_dict().items():
            torch.testing.assert_close(value, loaded_table.state_dict()[name], rtol=0, atol=0)
        lookup_mse = (rows[:, None] - table.sorted_hand_poses).square().sum(-1).amin(-1).mean() / model.hand_dim
        torch.testing.assert_close(torch.tensor(metrics["lookup_mse"]),
                                   lookup_mse, rtol=1e-5, atol=1e-7)
        policy_config: dict[str, Any] = dict(
            horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
            n_layers=1, hidden_dim=32, n_head=4,
            num_inference_steps=2, num_points=32,
            pc_encoder_config=dict(channels=48, num_patches=8, group_size=4, depth=2, num_heads=4),
            fps_random_config=dict(use_random=False),
            # Tiny random DINO only checks the codebook/policy pipeline.
            rgb_backbone_config=dict(
                tune_mode="freeze",
                image_size=(28, 28), architecture=dict(
                    model_type="dinov2", hidden_size=24, intermediate_size=48,
                    num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14,
                ),
            ),
        )
        policy = DQRISE(**policy_config)
        spec = dict(joint_state="limits", point_cloud="identity", rgb="identity",
                    camera_intrinsic="identity", camera_extrinsic="identity", action="limits")
        dims = dict(joint_state=19, point_cloud=3, rgb=3,
                    camera_intrinsic=9, camera_extrinsic=12, action=19)
        policy.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
        description = dict(
            attrs=dict(hand_joint_order=["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"],
                       hand_units="radians"),
            action_key="action", split_manifest=manifest,
        )
        try:
            policy.initialize_training(
                {**description, "split_manifest": {**manifest, "train_indices": [1]}}, artifact,
            )
        except ValueError as error:
            assert "split mismatch" in str(error)
        else:
            raise AssertionError("DQ accepted a different VQ training split")
        imported_affine = policy.initialize_training(description, artifact)
        policy.normalizer["action"].scale[-12:].copy_(imported_affine["scale"])
        policy.normalizer["action"].offset[-12:].copy_(imported_affine["offset"])
        policy.to(device)
        xyz = torch.rand(2, 2, 32, 3, device=device) * 0.2 - 0.1
        xyz[..., 2] += 1
        intrinsics = torch.tensor([[40., 0., 15.5], [0., 40., 11.5], [0., 0., 1.]], device=device)
        extrinsics = torch.eye(4, device=device)[:3]
        obs = dict(
            joint_state=torch.randn(2, 2, 19, device=device), point_cloud=xyz,
            rgb=torch.randint(256, (2, 2, 24, 32, 3), dtype=torch.uint8, device=device).permute(0, 1, 4, 2, 3).contiguous(),
            camera_intrinsic=intrinsics.flatten().expand(2, 2, -1).clone(),
            camera_extrinsic=extrinsics.flatten().expand(2, 2, -1).clone(),
        )
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = policy.compute_loss(dict(obs=obs, action=torch.rand(2, 4, 19, device=device)))
        loss.backward()
        assert torch.isfinite(loss)
        prediction = policy.eval().predict_action(obs)["future_actions"]
        assert prediction.shape == (2, 3, 19) and torch.isfinite(prediction).all()
        try:
            policy._predict_from_condition(
                torch.zeros(2, 256, device=device), rtc_prefix=prediction[:, :1],
            )
        except NotImplementedError:
            pass
        else:
            raise AssertionError("DQ-RISE accepted an unsupported RTC prefix")

        ema = copy.deepcopy(policy)
        with torch.no_grad():
            next(ema.parameters()).add_(0.01)
        saved = Path(directory) / "policy.pt"
        effective = dict(
            agent={"_target_": "dexmani_policy.agents.policies.dqrise.DQRISE",
                   **policy_config, **policy.effective_config},
            normalization_dims=dims,
            normalizer_shapes={key: tuple(field.scale.shape) for key, field in policy.normalizer.fields.items()},
        )
        checkpoint = Checkpoint(
            global_step=1, model_state=policy.state_dict(), ema_model_state=ema.state_dict(),
            optimizer_state={}, scheduler_state={}, rng_state={}, effective_config=effective,
        )
        save_checkpoint(saved, checkpoint)
        artifact.unlink()
        for use_ema, expected in ((False, policy), (True, ema)):
            before_rng = get_rng_state(device)
            restored = restore_policy(saved, use_ema=use_ema, device=device)
            after_rng = get_rng_state(device)
            assert before_rng["python"] == after_rng["python"]
            assert before_rng["numpy"][0] == after_rng["numpy"][0]
            assert (before_rng["numpy"][1] == after_rng["numpy"][1]).all()
            assert before_rng["numpy"][2:] == after_rng["numpy"][2:]
            torch.testing.assert_close(before_rng["torch"], after_rng["torch"], rtol=0, atol=0)
            if device.type == "cuda":
                torch.testing.assert_close(before_rng["torch_cuda"], after_rng["torch_cuda"], rtol=0, atol=0)
            expected.float()
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            expected_prediction = expected.predict_action(
                obs, generator=torch.Generator(device=device).manual_seed(1024),
            )["future_actions"]
            actual_prediction = restored.predict_action(
                obs, generator=torch.Generator(device=device).manual_seed(1024),
            )["future_actions"]
            torch.testing.assert_close(actual_prediction, expected_prediction, rtol=0, atol=0)
        assert checkpoint.ema_model_state is not None
        key = next(iter(checkpoint.model_state))
        del checkpoint.model_state[key]
        del checkpoint.ema_model_state[key]
        invalid = Path(directory) / "incomplete.pt"
        save_checkpoint(invalid, checkpoint)
        for use_ema in (False, True):
            try:
                restore_policy(invalid, use_ema=use_ema, device=device)
            except RuntimeError as error:
                assert "Missing key(s)" in str(error)
            else:
                raise AssertionError("Inference accepted an incomplete policy state")


def check_policies(device):
    kwargs: dict[str, Any] = dict(horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19, encoder_type="dp3",
                  pc_dim=3, pc_out_dim=16, state_dim=19, num_points=64, state_out_dim=8,
                  down_dims=(32, 64), diffusion_step_embed_dim=32, n_groups=8,
                  num_training_steps=10, num_inference_steps=2)
    policy = DP3(**kwargs)
    spec = dict(joint_state="limits", point_cloud="identity", action="limits")
    dims = dict(joint_state=19, point_cloud=3, action=19)
    policy.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
    policy.to(device)
    with torch.no_grad():
        policy.normalizer["action"].scale.copy_(torch.linspace(0.5, 1.5, 19, device=device))
        policy.normalizer["action"].offset.copy_(torch.linspace(-0.2, 0.2, 19, device=device))
    obs = dict(joint_state=torch.randn(2, 2, 19, device=device),
               point_cloud=torch.rand(2, 2, 64, 3, device=device))
    optimizer = build_optimizer(policy, lr=1e-4, weight_decay=1e-6)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = policy.compute_loss(dict(obs=obs, action=torch.randn(2, 8, 19, device=device)))
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
    policy.eval()
    with torch.inference_mode():
        ordinary = policy.predict_action(
            obs, generator=torch.Generator(device=device).manual_seed(17),
        )["future_actions"]
        zero = policy.predict_action(
            obs, rtc_prefix=ordinary[:, :3], rtc_guidance_cap=0,
            generator=torch.Generator(device=device).manual_seed(17),
        )["future_actions"]
        guided = policy.predict_action(obs, rtc_prefix=ordinary[:, :3], delay_steps=1,
                                       rtc_guidance_cap=2)["future_actions"]
    assert ordinary.shape == zero.shape == guided.shape == (2, 7, 19)
    torch.testing.assert_close(ordinary, zero, rtol=0, atol=0)
    assert torch.isfinite(guided).all()
    storage = io.BytesIO()
    torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict()), storage)
    storage.seek(0)
    saved = torch.load(storage, map_location=device, weights_only=True)
    restored = DP3(**kwargs)
    restored.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
    restored.load_state_dict(saved["model"], strict=True, assign=True)
    restored.to(device)
    new_optimizer = build_optimizer(restored, lr=1e-4, weight_decay=1e-6)
    assert [g["param_names"] for g in optimizer.param_groups] == [g["param_names"] for g in new_optimizer.param_groups]
    new_optimizer.load_state_dict(saved["optimizer"])
    actions = torch.randn(2, 8, 19, device=device)
    for model, opt in ((policy, optimizer), (restored, new_optimizer)):
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(29)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            resumed_loss, _ = model.compute_loss(dict(obs=obs, action=actions))
        resumed_loss.backward()
        opt.step()
    for value, replay in zip(policy.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(value, replay, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    for check in (check_sampler, check_time, check_precision, check_decoder_precision, check_tactile, check_interaction,
                  check_force_interaction, check_token_decoders, check_policy_roundtrips, check_rgb,
                  check_rgb_deployment, check_dinov3, check_vq,
                  check_codebook, check_codebook_pipeline, check_policies, check_output_boundaries):
        check(device)
        print(f"PASS {check.__name__} ({device})", flush=True)


if __name__ == "__main__":
    main()
