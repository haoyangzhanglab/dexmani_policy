import copy
import math
from contextlib import ExitStack, closing
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate

from dexmani_policy.agents.action_decoders.backbones import ditx
from dexmani_policy.agents.action_decoders.backbones.interaction_ditx import InteractionDiTX
from dexmani_policy.agents.action_decoders.backbones.modules import FlowTimestepEmbedding
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.obs_encoder.interaction.geometric_relation import GeoAwareRelationAttention
from dexmani_policy.agents.obs_encoder.interaction.hand_kinematics import HandKinematicEncoder
from dexmani_policy.agents.obs_encoder.interaction.hand_object_interaction import HandObjectInteractionEncoder
from dexmani_policy.agents.obs_encoder.interaction.tactile import FingerTactileEncoder, TactileFusion
from dexmani_policy.agents.obs_encoder.pointcloud import point_transformer
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.agents.policies.interaction_flow import InteractionRectifiedFlow
from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.deployment import load_policy, predict_actions, prepare_observations
from dexmani_policy.training import trainer as policy_training
from dexmani_policy.training.config import load_policy_config
from dexmani_policy.training.optimization import backward_microbatches
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer

from tests.helpers import (
    check_flow_recipe_defaults, check_observation_identity_constraints, check_recipe_composition,
    save_tiny_hf_rgb, tiny_hf_rgb_config,
)


def make_hand_scene_observation(point_channels=3):
    point_cloud = torch.rand(2, 160, point_channels)
    point_cloud[..., :3] = 0.20 * point_cloud[..., :3] - 0.1
    point_cloud[..., 2] -= 0.2
    camera_extrinsic = torch.eye(4)[:3].repeat(2, 1, 1)
    camera_extrinsic[0, 2, 3] = 1
    return {
        "point_cloud": point_cloud,
        "rgb": torch.randint(256, (2, 3, 8, 8), dtype=torch.uint8),
        "camera_intrinsic": torch.tensor([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]).expand(2, -1, -1),
        "camera_extrinsic": camera_extrinsic,
        "eef_pose": torch.tensor([[0.1, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).expand(2, -1),
        "fingertip_xyz": torch.rand(2, 5, 3) * 0.20,
        "finger_tactile_force": torch.randn(2, 5, 3),
    }


def test_interaction_flow_default_shapes_and_inference():
    from dexmani_policy.agents.policies.interaction_flow import InteractionObservationEncoder

    class Tokens(nn.Module):
        out_dim = 192

        def encode_scene(self, obs):
            return obs["scene"], obs["scene"][..., :3], None

        def encode_interaction(self, obs, scene_tokens, scene_points):
            assert len(scene_tokens) == len(obs["interaction"]) == 4
            return obs["interaction"]

    encoder = InteractionObservationEncoder(Tokens(), 19, 2)
    obs = {"scene": torch.randn(4, 128, 192), "interaction": torch.randn(4, 6, 192),
           "joint_state": torch.randn(4, 19)}
    packed = encoder(obs)
    assert packed.shape == (2, 270, 192)
    torch.testing.assert_close(packed[:, :256], obs["scene"].reshape(2, 256, 192))
    torch.testing.assert_close(packed[:, 258:], obs["interaction"].reshape(2, 12, 192))
    torch.testing.assert_close(packed[:, 256:258], encoder.state_encoder(obs["joint_state"]).reshape(2, 2, 192))
    changed = {**obs, "interaction": obs["interaction"].clone()}
    changed["interaction"][::2] += 10
    updated = encoder(changed)
    torch.testing.assert_close(updated[:, :258], packed[:, :258])
    assert not torch.equal(updated[:, 258:], packed[:, 258:])
    model = InteractionDiTX(
        horizon=16, action_dim=19, n_obs_steps=2, num_scene_tokens=128, obs_token_dim=192,
        p_drop_attn=0,
    ).eval()
    assert len(model.ditx_blocks) == 8
    global_context, interaction = model._encode_condition(packed)
    assert global_context.shape == (2, 258, 768) and interaction.shape == (2, 12, 768)
    x = torch.randn(2, 16, 19)
    with torch.no_grad():
        velocity = model(x, 0.5, packed)
        assert velocity.shape == x.shape
        assert RectifiedFlow(model, 5).eval().predict_action(packed, x.shape).shape == x.shape


@pytest.mark.parametrize("n_obs_steps", [1, 2, 3])
def test_interaction_observation_matches_full_history(n_obs_steps):
    from dexmani_policy.agents.policies.interaction_flow import InteractionObservationEncoder

    torch.manual_seed(5)
    interaction = HandObjectInteractionEncoder(
        patch_config={"num_patches": 4, "group_size": 4},
        rgb_backbone_config={"architecture": tiny_hf_rgb_config("dino").to_dict(), "image_size": 8},
    ).eval()
    encoder = InteractionObservationEncoder(interaction, 19, n_obs_steps)
    obs = {key: value.repeat_interleave(n_obs_steps, 0) for key, value in make_hand_scene_observation().items()}
    obs["point_cloud"] = obs["point_cloud"] + torch.arange(2 * n_obs_steps)[:, None, None] * 0.01
    obs["fingertip_xyz"] = torch.rand(2 * n_obs_steps, 5, 3)
    obs["joint_state"] = torch.randn(2 * n_obs_steps, 19)
    obs["finger_tactile_force"] = torch.randn(2 * n_obs_steps, 5, 3)
    scene, centers, _ = interaction.encode_scene(obs)
    full = interaction.encode_interaction(obs, scene, centers)
    scene_count = n_obs_steps * scene.shape[1]
    interaction_start = scene_count + n_obs_steps
    expected = torch.cat((
        scene.reshape(2, -1, 192),
        encoder.state_encoder(obs["joint_state"].reshape(2, n_obs_steps, 19)),
        full.reshape(2, n_obs_steps * 6, 192),
    ), dim=1)
    capture = Mock(return_value=None)
    with (
        interaction.hand_encoder.register_forward_pre_hook(capture),
        patch.object(encoder.state_encoder, "forward", wraps=encoder.state_encoder.forward) as state_encode,
        patch.object(interaction, "encode_scene", wraps=interaction.encode_scene) as encode_scene,
        patch.object(interaction, "encode_interaction", wraps=interaction.encode_interaction) as encode_interaction,
    ):
        actual = encoder(obs)
    assert encode_scene.call_count == encode_interaction.call_count == state_encode.call_count == 1
    torch.testing.assert_close(state_encode.call_args.args[0], obs["joint_state"])
    per_frame = torch.cat([interaction({key: value[i:i+1] for key, value in obs.items()})
                           for i in range(2 * n_obs_steps)])
    torch.testing.assert_close(actual[:, interaction_start:], per_frame.reshape(2, -1, 192))
    assert capture.call_args.args[1][0].shape[0] == 2 * n_obs_steps
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameters, probe),
        torch.autograd.grad(expected, parameters, probe), atol=2e-5, rtol=2e-4,
    )
    for frame in range(n_obs_steps):
        changed = {**obs, "joint_state": obs["joint_state"].clone()}
        changed["joint_state"][frame::n_obs_steps] += 1
        updated = encoder(changed)
        token = scene_count + frame
        assert not torch.allclose(updated[:, token], actual[:, token])
        torch.testing.assert_close(updated[:, :token], actual[:, :token], rtol=0, atol=0)
        torch.testing.assert_close(updated[:, token + 1:], actual[:, token + 1:], rtol=0, atol=0)


@pytest.mark.parametrize("use_ema", [False, True])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_interaction_flow_training_routing_and_restore(tmp_path, make_demo, use_ema, device):
    source, directory = tmp_path / "rgb", tmp_path / "interaction"
    pretrained = save_tiny_hf_rgb(source, "dino")
    path = make_demo(auxiliary=False, episode_ends=(32,))
    data = zarr.open_group(str(path), mode="r+")["data"]
    frames = make_hand_scene_observation()
    frames["rgb"] = frames["rgb"].movedim(1, -1)
    for name in ("camera_intrinsic", "camera_extrinsic"):
        frames[name] = frames[name].flatten(1)
    for name, value in frames.items():
        data[name] = value.numpy()[np.arange(len(data["action"])) % len(value)]
    cfg = load_policy_config("interaction_flow", [
        f"zarr_path='{path}'", "task_name=test", f"workspace.output_dir='{directory}'",
        "workspace.wandb_cfg=null", f"training.device={device}", "training.use_compile=false",
        f"training.use_ema={str(use_ema).lower()}",
        "dataloader.num_workers=0", "dataloader.batch_size=2", "training.lr_warmup_steps=0",
        f"initialization.rgb='{source}'", "agent.rgb_backbone_config.image_size=[8,8]",
        "agent.patch_config.num_patches=4", "agent.patch_config.group_size=4",
        "agent.n_layers=1", "agent.hidden_dim=32", "agent.n_head=2",
    ])
    with closing(build_trainer(cfg)) as fresh:
        model = fresh.model
        encoder = model.obs_encoder
        torch.testing.assert_close(
            encoder.rgb_encoder.backbone.state_dict(), pretrained.state_dict(), check_device=False,
        )
        assert not any(p.requires_grad for p in encoder.rgb_encoder.parameters())
        assert isinstance(model.action_decoder, RectifiedFlow) and model.action_decoder.model.action_io is None
        fresh.train(max_updates=2)
        batch = default_collate([fresh.train_loader.dataset[i] for i in (1, 2)])
        assert set(batch) == {"obs", "action"}
        batch = dict_apply(batch, lambda value: value.to(device))
        assert not model.requires_ema_for_loss
        loss, metrics = model(batch)
        assert set(metrics) == {"loss"}
        torch.testing.assert_close(loss.detach(), metrics["loss"])
        assert model(batch, log_metrics=False)[1] == {}
        assert torch.isfinite(loss)
        interaction = encoder.interaction_encoder
        gradients = torch.autograd.grad(loss, (
            encoder.state_encoder.mlp[0].weight, interaction.hand_encoder.finger_mlp[0].weight,
            interaction.tactile_encoder.force_encoder[0].weight,
            interaction.patch_encoder.geometry_embedding.projection.weight,
            model.action_decoder.model.context_embedder.weight, model.action_decoder.model.state_projection.weight,
            model.action_decoder.model.ditx_blocks[0].scene_cross_attn.kv.weight,
            model.action_decoder.model.ditx_blocks[0].interaction_cross_attn.kv.weight,
        ))
        assert all(g.isfinite().all() and g.abs().max() > 0 for g in gradients)
        if use_ema:
            torch.testing.assert_close(fresh.ema_model.state_dict(), model.state_dict())
            assert all(p.grad is None for p in fresh.ema_model.parameters())
        else:
            assert fresh.ema_model is None and fresh.ema_updater is None
        with ExitStack() as stack:
            student_compile = stack.enter_context(patch.object(model, "compile_backbone"))
            if use_ema:
                ema_compile = stack.enter_context(patch.object(fresh.ema_model, "compile_backbone"))
            fresh.cfg.training.use_compile = True
            fresh.compile_models()
            student_compile.assert_called_once()
            if use_ema:
                ema_compile.assert_not_called()
            fresh.cfg.training.use_compile = False

        model.eval()
        obs = batch["obs"]
        with torch.no_grad():
            scene_out, hand_out = Mock(return_value=None), Mock(return_value=None)
            with (
                interaction.spatial_encoder.register_forward_hook(scene_out),
                interaction.geometric_relation.register_forward_hook(hand_out),
            ):
                packed = model._encode_obs(obs)
            assert scene_out.call_count == hand_out.call_count == 1
            torch.testing.assert_close(packed[:, :8], scene_out.call_args.args[2].unflatten(0, (2, 2)).flatten(1, 2))
            torch.testing.assert_close(packed[:, 10:], hand_out.call_args.args[2].unflatten(0, (2, 2)).flatten(1, 2))
            joint_tokens = encoder.state_encoder(model.normalizer["joint_state"].normalize(obs["joint_state"]))
            torch.testing.assert_close(packed[:, 8:10], joint_tokens)
            for field in ("finger_tactile_force", "fingertip_xyz"):
                changed = {**obs, field: obs[field] + 0.1}
                updated = model._encode_obs(changed)
                torch.testing.assert_close(updated[:, :10], packed[:, :10], rtol=0, atol=0)
                assert not torch.allclose(updated[:, 10:], packed[:, 10:])
        with pytest.raises(ValueError, match="fingertip_xyz"):
            model._encode_obs({**obs, "fingertip_xyz": obs["fingertip_xyz"].flatten(0, 1).unsqueeze(0)})

        physical_obs = {key: data[key][:2] for key in cfg.dataset.sensor_modalities}
        inputs = prepare_observations(
            physical_obs, normalization_dims=model.normalization_dims,
            n_obs_steps=2, rgb_preprocessing=rgb_preprocessing_kwargs(cfg.dataset), device=device,
        )
        expected = predict_actions(model, inputs, generator=torch.Generator(device).manual_seed(17))
        if use_ema:
            ema_expected = predict_actions(
                fresh.ema_model, inputs, generator=torch.Generator(device).manual_seed(17),
            )

    source.rename(tmp_path / "detached_rgb_source")
    latest = directory / "checkpoints/latest.pt"
    with closing(load_policy(latest, weights="raw", device=device, seed=17)) as runtime:
        result = runtime.predict(physical_obs)
        assert result.shape == (15, 19)
        np.testing.assert_allclose(result, expected)
    if use_ema:
        with closing(load_policy(latest, weights="ema", device=device, seed=17)) as runtime:
            np.testing.assert_allclose(runtime.predict(physical_obs), ema_expected)
    with closing(prepare_resume(directory)) as resumed:
        saved = load_checkpoint(latest)
        torch.testing.assert_close(resumed.model.state_dict(), saved.model_state, check_device=False)
        if use_ema:
            torch.testing.assert_close(resumed.ema_model.state_dict(), saved.ema_model_state, check_device=False)
        else:
            assert saved.ema_model_state is None and resumed.ema_model is None
        assert resumed.global_step == 2
        resumed.train(max_updates=1)
        assert resumed.global_step == 3


@pytest.mark.parametrize("batch,steps", [(1, 1), (2, 1), (1, 5), (2, 5), (2, 7), (2, 10), (2, 11)])
def test_interaction_ditx_frame_layout_and_cache(monkeypatch, batch, steps):
    torch.manual_seed(5)
    model = InteractionDiTX(
        horizon=4, action_dim=19, n_obs_steps=2, num_scene_tokens=3, obs_token_dim=8,
        hidden_dim=16, n_head=2, n_layers=2, p_drop_attn=0,
    )
    # A zero velocity head or closed adaLN gates would make cache equality vacuous.
    with torch.no_grad():
        assert model.adaln_modulation.projection.weight.count_nonzero() == 0
        assert model.adaln_modulation.projection.bias.count_nonzero() == 0
        model.final_layer.ffn_final.fc2.weight.normal_(std=0.1)
        model.adaln_modulation.projection.weight.normal_(std=0.1)
    packed = torch.randn(batch, 20, 8)
    assert model.prepare_context_kv(packed) is None
    flow = InteractionRectifiedFlow(model).eval()

    def sample(context, decoder=flow):
        return decoder.predict_action(
            context, (batch, 4, 19), inference_steps=steps, generator=torch.Generator().manual_seed(9),
        )

    scene_input, state_input, modulation = (Mock(return_value=None) for _ in range(3))
    with (
        ExitStack() as stack,
        model.context_embedder.register_forward_pre_hook(scene_input),
        model.state_projection.register_forward_pre_hook(state_input),
        model.adaln_modulation.register_forward_pre_hook(modulation),
    ):
        captures, queries = [], []
        for block in model.ditx_blocks:
            for attention in (block.scene_cross_attn, block.interaction_cross_attn):
                capture = Mock(return_value=None)
                stack.enter_context(attention.kv.register_forward_pre_hook(capture))
                captures.append(capture)
                query = Mock(return_value=None)
                stack.enter_context(attention.q.register_forward_pre_hook(query))
                queries.append(query)
        projection = Mock(return_value=None)
        stack.enter_context(model.interaction_projection.register_forward_pre_hook(projection))
        first = sample(packed)
    assert all(c.call_count == 1 for c in [projection, *captures])
    assert all(q.call_count == steps for q in queries)
    for capture in (scene_input, state_input):
        assert capture.call_count == 1
    assert modulation.call_count == math.ceil(steps / flow.num_inference_steps)
    torch.testing.assert_close(scene_input.call_args.args[1][0], packed[:, :6])
    torch.testing.assert_close(state_input.call_args.args[1][0], packed[:, 6:8])

    changed = packed.clone()
    changed[:, -1, 0] += 1
    second = sample(changed)
    assert not torch.allclose(first, second)
    reference = RectifiedFlow(model).eval()
    torch.testing.assert_close(sample(packed, reference), first)
    torch.testing.assert_close(sample(changed, reference), second)
    monkeypatch.setattr(model, "prepare_context_kv", lambda context: None)
    torch.testing.assert_close(sample(packed, reference), first)
    torch.testing.assert_close(sample(changed, reference), second)


@pytest.mark.parametrize("qkv_bias,qk_norm", [(False, False), (True, True)])
def test_interaction_serial_queries_and_gradients(qkv_bias, qk_norm):
    from dexmani_policy.agents.action_decoders.backbones.interaction_ditx import InteractionDiTXBlock
    from dexmani_policy.agents.action_decoders.backbones.modules import modulate

    torch.manual_seed(13)
    block = InteractionDiTXBlock(16, 2, qkv_bias=qkv_bias, qk_norm=qk_norm)
    scene, interaction = torch.randn(2, 7, 16), torch.randn(2, 12, 16)
    x = torch.randn(2, 4, 16)
    modulation = torch.randn(2, 12, 16) * 0.2
    groups = modulation.unsqueeze(1).unbind(2)
    sa = modulate(block.norm1(x), *groups[:2])
    x1 = x + groups[2] * block.self_attn(sa, sa, sa, need_weights=False)[0]
    scene_query = modulate(block.norm2(x1), *groups[3:5])
    x2 = x1 + groups[5] * block.scene_cross_attn(scene_query, scene)
    inter_query = modulate(block.norm3(x2), *groups[6:8])
    assert not torch.allclose(inter_query, modulate(block.norm3(x1), *groups[6:8]))
    capture_scene, capture_inter = Mock(return_value=None), Mock(return_value=None)
    with (block.scene_cross_attn.register_forward_pre_hook(capture_scene),
          block.interaction_cross_attn.register_forward_pre_hook(capture_inter)):
        output = block(x, modulation, (scene, interaction))
    torch.testing.assert_close(capture_scene.call_args.args[1], (scene_query, scene))
    torch.testing.assert_close(capture_inter.call_args.args[1], (inter_query, interaction))
    x3 = x2 + groups[8] * block.interaction_cross_attn(inter_query, interaction)
    torch.testing.assert_close(output, x3 + groups[11] * block.mlp(modulate(block.norm4(x3), *groups[9:11])))
    left, right = block.scene_cross_attn, block.interaction_cross_attn
    assert not {id(p) for p in left.parameters()} & {id(p) for p in right.parameters()}
    grads = torch.autograd.grad(output.square().mean(), (*left.parameters(), *right.parameters()))
    assert all(g.isfinite().all() and g.abs().max() > 0 for g in grads)


@pytest.mark.parametrize("frames", [1, 2, 3])
@pytest.mark.parametrize("qkv_bias,qk_norm", [(False, False), (True, True)])
def test_interaction_embeddings_and_forward_cache(frames, qkv_bias, qk_norm):
    torch.manual_seed(17)
    model = InteractionDiTX(
        horizon=4, action_dim=19, n_obs_steps=frames, num_scene_tokens=3, obs_token_dim=8,
        hidden_dim=16, n_layers=2, n_head=2, p_drop_attn=0, qkv_bias=qkv_bias, qk_norm=qk_norm,
    ).eval()
    assert model.adaln_modulation.layer_shape == (2, 12, 16)
    assert model.interaction_frame_pos_embed.count_nonzero() == 0
    assert model.final_layer.ffn_final.fc2.weight.count_nonzero() == 0
    for block in model.ditx_blocks:
        assert block.scene_cross_attn.proj.weight.count_nonzero() > 0
        assert block.interaction_cross_attn.proj.weight.count_nonzero() > 0
    with torch.no_grad():
        model.adaln_modulation.projection.weight.normal_(std=0.1)
        model.final_layer.ffn_final.fc2.weight.normal_(std=0.1)
        model.interaction_frame_pos_embed.normal_()
        model.context_frame_pos_embed.normal_()
    packed = torch.randn(2, frames * 10, 8, requires_grad=True)
    global_context, interaction = model._encode_condition(packed)
    assert not any("state_mlp" in name for name, _ in model.named_modules())
    assert global_context.shape == (2, frames * 4, 16)
    scene_raw, state_raw, _ = packed.split(model.condition_counts, dim=1)
    scene_reference = model.scene_norm(
        model.context_embedder(scene_raw).reshape(2, frames, 3, 16)
        + model.context_frame_pos_embed[:, :, None]
    ).flatten(1, 2)
    state_reference = model.state_norm(model.state_projection(state_raw) + model.context_frame_pos_embed)
    torch.testing.assert_close(global_context, torch.cat((scene_reference, state_reference), dim=1))
    raw = packed[:, frames * 4:].reshape(2, frames, 6, 8)
    reference = model.interaction_norm(
        model.interaction_projection(raw)
        + model.interaction_frame_pos_embed[:, :, None]
        + model.role_embeddings[:, None]
    ).flatten(1, 2)
    torch.testing.assert_close(interaction, reference)
    actions = torch.randn(2, 4, 19, requires_grad=True)

    def gradients(cache):
        velocity = model(actions, 0.4, packed, context_kv=cache)
        loss = velocity.square().mean()
        return velocity, torch.autograd.grad(loss, (actions, packed, *model.parameters()))

    torch.testing.assert_close(gradients(model.prepare_context_kv(packed)), gradients(None))
    expected = model(actions, 0.4, packed)
    for index in (0, *range(frames * 3, frames * 4), frames * 4):
        changed = packed.clone()
        changed[:, index] += torch.randn(8)
        assert not torch.allclose(model(actions, 0.4, changed), expected)


@pytest.mark.parametrize("batch,steps", [(1, 1), (3, 1), (1, 5), (3, 5), (3, 7), (1, 10), (3, 10), (3, 11)])
def test_interaction_time_only_modulations(batch, steps):
    torch.manual_seed(29)
    model = InteractionDiTX(
        horizon=4, action_dim=19, n_obs_steps=2, num_scene_tokens=3, obs_token_dim=8,
        hidden_dim=16, n_head=2, n_layers=2, p_drop_attn=0,
    ).eval()
    with torch.no_grad():
        model.adaln_modulation.projection.weight.normal_(std=0.1)
        model.final_layer.ffn_final.fc2.weight.normal_(std=0.1)
    context = torch.randn(batch, 20, 8)
    actions = torch.randn(batch, 4, 19)
    times = torch.arange(steps, dtype=torch.float32) / steps
    cache = model.prepare_context_kv(context)
    assert len(cache) == len(model.ditx_blocks)
    assert all(len(layer_kv) == 2 for layer_kv in cache)
    precomputed = list(model.sampling_modulations(times, 10))
    assert len(precomputed) == steps
    for time, modulation in zip(times, precomputed):
        assert modulation.shape == (1, 2, 12, 16)
        expected = model.adaln_modulation(model.timestep_embedder(time.expand(batch)))
        torch.testing.assert_close(modulation.expand(batch, -1, -1, -1), expected)
        torch.testing.assert_close(
            model.forward_with_modulations(actions, context, modulation, context_kv=cache),
            model(actions, time, context),
        )
    changed = context.clone()
    changed[:, 6:8] += torch.randn(batch, 2, 8)
    captured = Mock(return_value=None)
    with model.adaln_modulation.register_forward_pre_hook(captured):
        first = model(actions, 0.4, context)
        second = model(actions, 0.4, changed)
    expected_time = model.timestep_embedder(torch.full((batch,), 0.4))
    for call in captured.call_args_list:
        torch.testing.assert_close(call.args[1][0], expected_time, rtol=0, atol=0)
    assert not torch.allclose(first, second)
    # Training retains one time per sample instead of broadcasting the sampling grid.
    model.train()
    training_times = torch.linspace(0.1, 0.9, batch)
    with model.adaln_modulation.register_forward_pre_hook(captured):
        model(actions, training_times, context)
    torch.testing.assert_close(captured.call_args.args[1][0], model.timestep_embedder(training_times))


def test_hand_token_construction():
    encoder = HandKinematicEncoder()
    fingertips_wrist = torch.arange(1, 31, dtype=torch.float32).reshape(2, 5, 3) / 256
    wrist_xyz_1 = torch.tensor([[0.125, 0.25, 0.375], [-0.25, 0.125, 0.5]])
    wrist_xyz_2 = torch.tensor([[-0.125, 0.375, 0.25], [0.25, -0.125, 0.375]])
    rotation_1 = torch.eye(3).expand(2, -1, -1)
    rotation_2 = torch.tensor([
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    ])
    capture_wrist, capture_fingers = Mock(return_value=None), Mock(return_value=None)
    finger_token_sets = []
    for wrist_xyz_base, rotation in ((wrist_xyz_1, rotation_1), (wrist_xyz_2, rotation_2)):
        rotation_6d = torch.cat((rotation[:, :, 0], rotation[:, :, 1]), dim=-1)
        eef_pose = torch.cat((wrist_xyz_base, rotation_6d), dim=-1).requires_grad_()
        fingertips_base = (fingertips_wrist @ rotation.transpose(-1, -2) + wrist_xyz_base[:, None]).requires_grad_()
        with (
            encoder.wrist_mlp.register_forward_pre_hook(capture_wrist),
            encoder.finger_mlp.register_forward_pre_hook(capture_fingers),
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            hand_tokens, hand_landmarks_base = encoder(eef_pose, fingertips_base)

        assert hand_tokens.shape == (2, 6, 192)
        assert hand_landmarks_base.shape == (2, 6, 3)
        expected_landmarks = torch.cat((wrist_xyz_base[:, None], fingertips_base), dim=1)
        assert torch.equal(hand_landmarks_base, expected_landmarks)
        _, (wrist_input,) = capture_wrist.call_args.args
        _, (finger_inputs,) = capture_fingers.call_args.args
        expected_wrist_input = torch.cat((wrist_xyz_base / 0.25, rotation_6d), dim=-1)
        torch.testing.assert_close(wrist_input, expected_wrist_input)
        torch.testing.assert_close(finger_inputs, fingertips_wrist / 0.10)
        finger_token_sets.append(hand_tokens[:, 1:])

        pose_grad, fingertips_grad = torch.autograd.grad(finger_inputs.sum(), (eef_pose, fingertips_base))
        assert pose_grad.isfinite().all() and torch.count_nonzero(pose_grad[:, 3:]) > 0
        assert fingertips_grad.isfinite().all() and torch.count_nonzero(fingertips_grad) > 0

    torch.testing.assert_close(finger_token_sets[0], finger_token_sets[1])

    with pytest.raises(ValueError, match="fingertip_xyz"):
        encoder(eef_pose, fingertips_base[:, :0])
    with pytest.raises(RuntimeError, match="split_sizes"):
        encoder(torch.cat((eef_pose, eef_pose[:, :1]), dim=-1), fingertips_base)


def test_multi_finger_tactile_encoding():
    encoder = FingerTactileEncoder()
    force = torch.randn(2, 5, 3)
    features = encoder(force)
    assert features["local_tokens"].shape == (2, 5, 64)
    assert features["global_token"].shape == (2, 64)
    assert torch.equal(features["global_token"], features["local_tokens"].mean(dim=1))
    torch.testing.assert_close(encoder(force.double()), features)

    zero_features = encoder(torch.zeros_like(force))
    assert torch.count_nonzero(zero_features["local_tokens"]) == 0
    assert torch.count_nonzero(zero_features["global_token"]) == 0


def test_tactile_fusion_zero_force_invariance():
    encoder, fusion = FingerTactileEncoder(), TactileFusion()
    hand_tokens = torch.randn(2, 6, 192)
    force = torch.zeros(2, 5, 3)
    assert torch.equal(fusion(hand_tokens, encoder(force)), hand_tokens)

    force[:, 2, 0] = 1
    features = encoder(force)
    residual = fusion(torch.zeros_like(hand_tokens), features)
    assert torch.equal(residual[:, [1, 2, 4, 5]], residual[:, :1].expand(-1, 4, -1))
    assert torch.count_nonzero(residual[:, 0]) > 0
    assert torch.count_nonzero(residual[:, 3] - residual[:, 0]) > 0

    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        fusion(hand_tokens, {**features, "global_token": features["global_token"][:1]})
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        fusion(hand_tokens, {**features, "local_tokens": features["local_tokens"][:, :1]})

    fixed_features = {key: torch.ones_like(value) for key, value in features.items()}
    hand_states = torch.zeros_like(hand_tokens)
    hand_states[1, 0, 0] = 4
    hand_states[1, 3, 0] = 2
    with torch.no_grad():
        for gate in (fusion.local_gate, fusion.contact_gate):
            torch.testing.assert_close(gate.bias, torch.full_like(gate.bias, -1.5))
            gate.weight.zero_()
            gate.weight[:, [0, 64]] = 1

    capture_local, capture_contact = Mock(return_value=None), Mock(return_value=None)
    with (
        fusion.local_gate.register_forward_hook(capture_local),
        fusion.contact_gate.register_forward_hook(capture_contact),
    ):
        fused = fusion(hand_states, fixed_features)

    _, (local_input,), local_logits = capture_local.call_args.args
    _, (contact_input,), contact_logits = capture_contact.call_args.args
    torch.testing.assert_close(local_input, torch.cat((fixed_features["local_tokens"], hand_states[:, 1:]), dim=-1))
    torch.testing.assert_close(contact_input, torch.cat((fixed_features["global_token"], hand_states.mean(dim=1)), dim=-1))
    expected_local_logits = torch.full((2, 5, 1), -0.5)
    expected_local_logits[1, 2] = 1.5
    expected_contact_logits = torch.tensor([[-0.5], [0.5]])
    torch.testing.assert_close(local_logits, expected_local_logits)
    torch.testing.assert_close(contact_logits, expected_contact_logits)
    expected = hand_states.clone()
    expected[:, 1:] += (
        expected_contact_logits.sigmoid().unsqueeze(1)
        * expected_local_logits.sigmoid()
        * fusion.local_projection(fixed_features["local_tokens"])
    )
    expected += (expected_contact_logits.sigmoid() * fusion.global_projection(fixed_features["global_token"]))[:, None]
    torch.testing.assert_close(fused, expected)

    optimizer = torch.optim.SGD([*encoder.parameters(), *fusion.parameters()], lr=0.1)
    fusion(hand_tokens, features).square().mean().backward()
    optimizer.step()
    assert torch.equal(fusion(hand_tokens, encoder(torch.zeros_like(force))), hand_tokens)


def test_tactile_contact_gate_scales_local_update():
    fusion = TactileFusion()
    local_tokens = torch.ones(2, 5, 64)
    features = {"local_tokens": local_tokens, "global_token": local_tokens.mean(dim=1)}
    hand_tokens = torch.zeros(2, 6, 192)
    hand_tokens[1, 0, 0] = 6
    with torch.no_grad():
        fusion.local_gate.weight.zero_()
        fusion.local_gate.bias.zero_()
        fusion.contact_gate.weight.zero_()
        fusion.contact_gate.bias.fill_(-3)
        fusion.contact_gate.weight[:, 64] = 6

    output = fusion(hand_tokens, features)
    residual = output - hand_tokens
    # Subtracting the wrist residual isolates the finger-local path.
    contact_weight = torch.tensor([[-3.0], [3.0]]).sigmoid()
    expected_local = contact_weight.unsqueeze(1) * 0.5 * fusion.local_projection(local_tokens)
    torch.testing.assert_close(residual[:, 1:] - residual[:, :1], expected_local)


def test_hand_scene_attention_shared_rope_origin():
    attention = GeoAwareRelationAttention()
    hand_tokens = torch.randn(2, 6, 192)
    scene_tokens = torch.randn(2, 128, 192)
    hand_landmarks_base = torch.rand(2, 6, 3) * 0.2
    scene_points_base = torch.rand(2, 128, 3) * 0.2
    shift = torch.tensor([0.25, -0.5, 0.75])

    tokens = attention(hand_tokens, scene_tokens, hand_landmarks_base, scene_points_base)
    translated = attention(hand_tokens, scene_tokens, hand_landmarks_base + shift, scene_points_base + shift)
    torch.testing.assert_close(translated, tokens)
    shifted_hand = attention(hand_tokens, scene_tokens, hand_landmarks_base + shift, scene_points_base)
    assert not torch.allclose(shifted_hand, tokens)

    capture_rope = Mock(return_value=None)
    with (
        attention.rope.register_forward_hook(capture_rope, with_kwargs=True),
        patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa,
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        tokens = attention(hand_tokens, scene_tokens, hand_landmarks_base, scene_points_base)

    assert tokens.shape == (2, 6, 192)
    assert capture_rope.call_count == 2
    for call, xyz in zip(capture_rope.call_args_list, (hand_landmarks_base, scene_points_base)):
        _, (positions,), kwargs, (cos, sin) = call.args
        torch.testing.assert_close(positions, xyz)
        torch.testing.assert_close(kwargs["origin"], scene_points_base.mean(dim=1, keepdim=True))
        assert cos.dtype == sin.dtype == torch.float32
    query, key, value = sdpa.call_args.args[:3]
    assert query.shape == (2, 4, 6, 48)
    assert key.shape == value.shape == (2, 4, 128, 48)
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16

    with pytest.raises(ValueError, match="batch"):
        attention(hand_tokens, scene_tokens[:1], hand_landmarks_base, scene_points_base[:1])
    with pytest.raises(ValueError, match="batch"):
        attention(hand_tokens, scene_tokens, hand_landmarks_base[:, :1], scene_points_base)


def test_local_geometry_matches_concatenated_patch_projection():
    torch.manual_seed(3)
    geometry = point_transformer.LocalGeometryEmbedding(3, 8, 0.1, local_hidden_dim=4)
    points = torch.randn(2, 7, 3, requires_grad=True)
    centers = points[:, :3]
    neighbors = torch.randint(7, (2, 3, 4))
    grouped = points[torch.arange(2)[:, None, None], neighbors]
    features = geometry.point_mlp((grouped - centers.unsqueeze(2)) / geometry.geometry_scale)
    pooled = features.amax(dim=2, keepdim=True)
    weight = torch.cat((geometry.point_projection.weight, geometry.context_projection.weight), dim=1)
    joined = torch.cat((features, pooled.expand_as(features)), dim=-1)
    fused = F.linear(joined, weight, geometry.context_projection.bias)
    expected = geometry.projection(geometry.context_mlp(fused).amax(dim=2))
    actual = geometry(points, centers, neighbors)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    inputs = (points, *geometry.parameters())
    torch.testing.assert_close(
        torch.autograd.grad(actual, inputs, probe), torch.autograd.grad(expected, inputs, probe),
    )


@pytest.mark.parametrize("scene_config,point_channels,samples_per_patch", [
    pytest.param({}, 3, 1, id="center-xyz"),
    pytest.param({"rgb_semantic_sampling": "group_mean", "pc_dim": 6}, 6, 32, id="group-mean-xyzrgb"),
])
def test_hand_scene_encoder(scene_config, point_channels, samples_per_patch):
    encoder = HandObjectInteractionEncoder(
        rgb_backbone_config={"architecture": tiny_hf_rgb_config("dino").to_dict(), "image_size": 8},
        **scene_config,
    ).eval()

    obs = make_hand_scene_observation(point_channels)
    point_cloud = obs["point_cloud"]
    camera_extrinsic = obs["camera_extrinsic"]
    capture_scene = Mock(return_value=None)
    with encoder.spatial_encoder.register_forward_pre_hook(capture_scene):
        tokens = encoder(obs)
    assert tokens.shape == (2, 6, 192)
    assert capture_scene.call_args.args[1][0].shape == (2, 128, 192)
    scene, centers, valid = encoder.encode_scene({
        **obs,
        "camera_intrinsic": obs["camera_intrinsic"].flatten(1),
        "camera_extrinsic": camera_extrinsic.flatten(1),
    })
    torch.testing.assert_close(encoder.encode_interaction(obs, scene, centers), tokens)
    assert scene.shape == (2, 128, 192) and scene.requires_grad
    assert centers.shape == (2, 128, 3)
    patch_center_in_cloud = (centers[:, :, None] == point_cloud[:, None, :, :3]).all(dim=-1)
    assert patch_center_in_cloud.any(dim=-1).all()
    assert valid.shape == (2, 128, samples_per_patch) and valid.dtype == torch.bool
    assert valid[0].all() and not valid[1].any()
    with pytest.raises(ValueError, match="extrinsics"):
        encoder({**obs, "camera_extrinsic": camera_extrinsic.transpose(-1, -2)})


def test_hand_scene_encoder_backward():
    torch.manual_seed(0)
    encoder = HandObjectInteractionEncoder(
        rgb_backbone_name="dino",
        rgb_backbone_config={"architecture": tiny_hf_rgb_config("dino").to_dict(), "image_size": 8},
    )
    obs = make_hand_scene_observation()
    tokens = encoder(obs)
    assert tokens.shape == (2, 6, 192)
    probe = torch.randn_like(tokens)
    (tokens * probe).mean().backward()

    for name in (
        "patch_encoder.geometry_embedding.point_mlp.0.weight",
        "rgb_fusion.semantic_projection.weight",
        "rgb_fusion.fusion_projection.weight",
        "spatial_encoder.blocks.0.qkv.weight",
        "hand_encoder.wrist_mlp.0.weight",
        "hand_encoder.finger_mlp.0.weight",
        "hand_encoder.slot_embeddings",
        "tactile_encoder.force_encoder.0.weight",
        "tactile_fusion.local_projection.weight",
        "tactile_fusion.global_projection.weight",
        "tactile_fusion.local_gate.weight",
        "tactile_fusion.local_gate.bias",
        "tactile_fusion.contact_gate.weight",
        "tactile_fusion.contact_gate.bias",
        "geometric_relation.hand_attention.in_proj_weight",
        "geometric_relation.query_projection.weight",
        "geometric_relation.scene_kv.weight",
        "geometric_relation.ffn.0.weight",
    ):
        grad = encoder.get_parameter(name).grad
        assert grad is not None, name
        assert grad.isfinite().all(), name
        assert grad.count_nonzero() > 0, name


@pytest.mark.parametrize("batch_sizes", [(2, 3), (5,)])
@pytest.mark.parametrize("log_metrics", [False, True])
def test_interaction_flow_accumulation_matches_combined(monkeypatch, batch_sizes, log_metrics):
    from dexmani_policy.agents.policies.interaction_flow import InteractionFlow

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.velocity = nn.Parameter(torch.tensor(0.6))

        def forward(self, x, timestep, context):
            return self.velocity * x

    torch.manual_seed(7)
    model = InteractionFlow.__new__(InteractionFlow)
    Policy.__init__(model, nn.Identity(), RectifiedFlow(Backbone()), 1, 1, 1, 1)
    model.normalizer = LinearNormalizer({"action": (1,)})
    monkeypatch.setattr(model, "_encode_obs", lambda obs: obs["latent"])
    monkeypatch.setattr(model.action_decoder, "sample_training_pair", lambda actions: (
        actions, actions[:, 0, 0], torch.zeros_like(actions),
    ))
    dataset = [
        {"obs": {"latent": torch.randn(2, 4)}, "action": torch.tensor([[time]])}
        for time in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]
    expected_loss, expected_metrics = model.compute_loss(
        default_collate(dataset), log_metrics=log_metrics,
    )
    parameters = tuple(model.parameters())
    expected_gradients = torch.autograd.grad(expected_loss, parameters)
    batches, start = [], 0
    for size in batch_sizes:
        batches.append(default_collate(dataset[start:start + size]))
        start += size

    def forward(item):
        batch, options = item
        loss, metrics = model.compute_loss(batch, log_metrics=log_metrics, **options)
        return loss, metrics, len(batch["action"])

    finite, metrics = backward_microbatches(
        model.training_microbatches(iter(batches), None), len(batches), parameters, forward,
        device="cpu", log_metrics=log_metrics,
    )
    assert finite
    torch.testing.assert_close([p.grad for p in parameters], list(expected_gradients))
    # Metric aggregation deliberately accumulates in FP64.
    torch.testing.assert_close(metrics, {key: value.double() for key, value in expected_metrics.items()})
    assert model.action_decoder.model.velocity.grad.abs() > 0


def test_recipe_composition():
    cfg = check_recipe_composition("interaction_flow")
    check_flow_recipe_defaults("interaction_flow", cfg)
    assert cfg.n_action_steps == 8
    assert cfg.training.use_ema
    check_observation_identity_constraints("interaction_flow", ['camera_extrinsic', 'camera_intrinsic', 'eef_pose', 'fingertip_xyz', 'point_cloud'])


def test_flow_backbone_time_and_qk_defaults_interactionditx():
    layout = {"horizon": 4, "action_dim": 19}
    layout["num_scene_tokens"] = 3
    model = InteractionDiTX(n_obs_steps=2, obs_token_dim=8, hidden_dim=16, n_layers=2, n_head=2, **layout)
    assert isinstance(model.timestep_embedder.net[0], FlowTimestepEmbedding)
    attentions = [module for module in model.ditx_blocks.modules() if isinstance(module, ditx.CrossAttention)]
    assert len(attentions) == (4)
    for attention in attentions:
        assert isinstance(attention.q_norm, nn.LayerNorm)
        assert isinstance(attention.k_norm, nn.LayerNorm)
        assert attention.q_norm.normalized_shape == attention.k_norm.normalized_shape == (8,)


@pytest.mark.parametrize("kind,preprocessing", [("dino", {"image_size": 8}), ("siglip", {})])
def test_rgb_resolution_and_external_weights(tmp_path, monkeypatch, kind, preprocessing):
    from transformers import AutoConfig, SiglipVisionConfig
    from dexmani_policy.training.initialization import inherit_rgb_architecture, resolve_rgb_architecture

    directory = tmp_path / "rgb"
    source = save_tiny_hf_rgb(directory, kind)
    agent = OmegaConf.create({
        "rgb_backbone_name": kind, "rgb_backbone_config": {"tune_mode": "full", **preprocessing},
    })
    resolve_rgb_architecture(agent, str(directory))
    assert agent.rgb_backbone_config.architecture.hidden_size == 16
    encoder = HandObjectInteractionEncoder(**OmegaConf.to_container(agent))
    config = copy.deepcopy(encoder.effective_config)
    encoder = HandObjectInteractionEncoder(**config)
    encoder.rgb_encoder.initialize_weights(str(directory))
    torch.testing.assert_close(encoder.rgb_encoder.backbone.state_dict(), source.state_dict())
    larger_config = copy.deepcopy(config)
    larger_config["rgb_backbone_config"]["architecture"]["num_hidden_layers"] += 1
    larger_encoder = HandObjectInteractionEncoder(**larger_config)
    with pytest.raises(ValueError, match="missing configured parameters"):
        larger_encoder.rgb_encoder.initialize_weights(str(directory))
    assert config == encoder.effective_config
    assert config["rgb_backbone_config"]["architecture"]["hidden_size"] == 16
    assert config["rgb_backbone_config"]["image_size"] == (8, 8)
    images = encoder.image_processor.process_images(torch.rand(2, 3, 16, 16))
    result = encoder.rgb_encoder(images)
    result["patch_tokens"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.rgb_encoder.backbone.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("resolved or inherited architecture must not read a source")

    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(SiglipVisionConfig, "from_pretrained", forbidden)
    resolve_rgb_architecture(agent, "missing-source")
    warm = OmegaConf.create({"rgb_backbone_name": kind, "rgb_backbone_config": {}})
    saved_agent = OmegaConf.to_container(agent)
    inherit_rgb_architecture(warm, saved_agent)
    assert warm.rgb_backbone_name == kind
    assert warm.rgb_backbone_config.architecture == agent.rgb_backbone_config.architecture
    warm.rgb_backbone_config.architecture.hidden_size = 32
    inherit_rgb_architecture(warm, saved_agent)
    assert warm.rgb_backbone_config.architecture.hidden_size == 32
    warm.rgb_backbone_name = "resnet"
    with pytest.raises(ValueError, match="RGB backbone"):
        inherit_rgb_architecture(warm, saved_agent)


def test_dinov3_timm_architecture_and_local_weights(tmp_path, monkeypatch):
    import timm
    from safetensors.torch import save_file
    from transformers import AutoConfig, SiglipVisionConfig

    def forbidden(*args, **kwargs):
        pytest.fail("DINOv3 uses a timm architecture")

    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(SiglipVisionConfig, "from_pretrained", forbidden)
    agent = OmegaConf.create({"rgb_backbone_name": "dinov3"})
    policy_training.initialization.resolve_rgb_architecture(agent, str(tmp_path))
    assert "rgb_backbone_config" not in agent

    overrides = {"img_size": 32, "embed_dim": 24, "depth": 1, "num_heads": 3}
    architecture = {"model_name": "vit_small_patch16_dinov3", **overrides}
    source = timm.create_model(**architecture, pretrained=False)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    with patch.object(timm, "create_model", wraps=timm.create_model) as create_model:
        encoder = HandObjectInteractionEncoder(rgb_backbone_name="dinov3", rgb_backbone_config={
            "architecture": overrides, "tune_mode": "full",
        })
    create_model.assert_called_once_with(**architecture, pretrained=False)
    config = copy.deepcopy(encoder.effective_config)
    encoder.rgb_encoder.initialize_weights(str(tmp_path))
    torch.testing.assert_close(encoder.rgb_encoder.backbone.state_dict(), source.state_dict())
    assert config == encoder.effective_config
    assert config["rgb_backbone_config"]["architecture"] == architecture
    assert config["rgb_backbone_config"]["image_size"] == (32, 32)
    assert encoder(make_hand_scene_observation()).shape == (2, 6, 192)
    invalid_weights = source.state_dict()
    invalid_weights["cls_token"] = invalid_weights["cls_token"] + 1
    invalid_weights["unexpected"] = torch.zeros(1)
    save_file(invalid_weights, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="configured architecture"):
        encoder.rgb_encoder.initialize_weights(str(tmp_path))
    torch.testing.assert_close(encoder.rgb_encoder.backbone.state_dict(), source.state_dict())
