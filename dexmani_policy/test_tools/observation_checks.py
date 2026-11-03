"""Run observation checks: python -m dexmani_policy.test_tools.observation_checks."""

import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from dexmani_policy.agents.obs_encoder.interaction.future_tactile import ForceDiffusion, ForceGuidedFusion, ForceTransformer, VisuoTactileAttention
from dexmani_policy.agents.obs_encoder.interaction.hand_interaction import HandHistoryEncoder, HandSceneFusion, SceneTokenizer, _sample_region_seeds
from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandForceEncoder, XHandPatchEncoder
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.agents.policies.dp import DP
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.training.common.config import load_policy_config, prepare_training
from dexmani_policy.training.common.optimization import build_optimizer


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
            "tactile_encoder": encoder.effective_config,
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

    from dexmani_policy.env_runner.sim_runner import SimRunner

    runner = SimRunner("unused", 1, ["joint_state", "point_cloud", "finger_tactile_force", "taxel_tactile_force"], {})
    with patch.object(runner, "make_env", side_effect=AssertionError("Created environment for dense tactile")):
        try:
            next(runner.run(policy, seeds=[1], inference_steps=2, action_steps=2))
        except ValueError as error:
            assert "taxel_tactile_force" in str(error) and "no dense" in str(error)
        else:
            raise AssertionError("SimRunner accepted dense tactile")


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


def check_adaptac(device):
    from dexmani_policy.env_runner.sim_runner import SimRunner
    from dexmani_policy.evaluation import rollout

    cfg = load_policy_config("adaptac", [
        "horizon=8", "n_action_steps=8", "agent.num_points=32",
        "agent.pc_encoder_config.channels=24", "agent.pc_encoder_config.num_patches=12",
        "agent.pc_encoder_config.group_size=8", "agent.pc_encoder_config.depth=1",
        "agent.pc_encoder_config.num_heads=4", "agent.feature_dim=32", "agent.state_out_dim=8",
        "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=32",
        "agent.force_predictor_config.depth=1", "agent.force_predictor_config.embed_dim=32",
    ])
    policy = hydra.utils.instantiate(cfg.agent, _convert_="all")
    dims = dict(joint_state=19, point_cloud=6, finger_tactile_force=3, action=19)
    policy.configure_normalizer(dims, normalization_feature_shapes(dims))
    policy.to(device)
    obs = dict(joint_state=torch.randn(2, 1, 19, device=device),
               point_cloud=torch.randn(2, 1, 48, 6, device=device),
               finger_tactile_force=torch.randn(2, 1, 5, 3, device=device))
    valid = torch.ones(2, 8, dtype=torch.bool, device=device)
    valid[1, -2:] = False
    batch = dict(
        obs=obs,
        action=torch.randn(2, 8, 19, device=device),
        future_force=torch.randn(2, 8, 5, 3, device=device),
        future_force_valid=valid,
    )

    for training, steps in ((True, None), (False, None), (False, 10)):
        policy.train(training)
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        generator = torch.Generator(device=device).manual_seed(456)
        force_method = "compute_loss_and_prediction" if training else "predict_action"
        with patch.object(policy, "_preprocess_obs", wraps=policy._preprocess_obs) as preprocess, \
             patch.object(policy.obs_encoder, "forward", wraps=policy.obs_encoder.forward) as encoder, \
             patch.object(policy, "_action_condition", wraps=policy._action_condition) as condition, \
             patch.object(policy.force_predictor, force_method,
                          wraps=getattr(policy.force_predictor, force_method)) as force, \
             patch.object(policy.force_predictor.model, "forward",
                          wraps=policy.force_predictor.model.forward) as force_model, \
             patch.object(policy.action_decoder.model, "forward",
                          wraps=policy.action_decoder.model.forward) as action_model, \
             torch.set_grad_enabled(training):
            if training:
                # Isolate the action objective's gradient through the predicted force.
                policy.force_loss_weight = 0
                loss, metrics = policy.compute_loss(batch)
                assert torch.isfinite(loss) and metrics["loss_force_weighted"] == 0
                loss.backward()
                policy.force_loss_weight = cfg.agent.force_loss_weight
            else:
                output = policy.predict_action(obs, inference_steps=steps, generator=generator)
                assert output["future_actions"].shape == (2, 8, 19)
                assert torch.isfinite(output["future_actions"]).all()
            assert preprocess.call_count == encoder.call_count == 1
            features, current_force, predicted_force = condition.call_args.args
            torch.testing.assert_close(current_force,
                                       encoder.call_args.args[0]["finger_tactile_force"].reshape(2, 1, 15))
            torch.testing.assert_close(force.call_args.args[0], torch.cat(features[:2], -1).unsqueeze(1))
            assert predicted_force.shape == (2, 8, 15)
            assert force_model.call_count == (1 if training else 20)
            assert action_model.call_count == (1 if training else steps or 20)
        if training:
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in policy.force_predictor.parameters())

    policy.eval()
    config: dict[str, Any] = {key: OmegaConf.to_container(cfg[key], resolve=True)
                              for key in ("agent", "eval", "env_runner", "dataset", "normalization")}
    config["agent"].update(policy.effective_config)
    config["normalization_dims"] = dims
    with tempfile.TemporaryDirectory(prefix="dexmani-adaptac-check-") as temporary:
        root = Path(temporary)
        checkpoint = save_checkpoint(root / "model.pt", Checkpoint(
            1, policy.state_dict(), None, {}, {}, {}, config))
        restored = restore_policy(checkpoint, use_ema=False, device=device)
        for steps, force_steps in ((None, 20), (10, 20), (10, 7)):
            restored.force_predictor.num_inference_steps = force_steps
            def run(runner, agent, **options):
                assert agent is restored
                assert options["inference_steps"] == (steps or 20)
                assert options["action_steps"] == 8
                with patch.object(agent.force_predictor.model, "forward",
                                  wraps=agent.force_predictor.model.forward) as force_model, \
                     patch.object(agent.action_decoder.model, "forward",
                                  wraps=agent.action_decoder.model.forward) as action_model:
                    agent.predict_action(obs, inference_steps=options["inference_steps"],
                                         generator=torch.Generator(device=device).manual_seed(456))
                assert force_model.call_count == force_steps
                assert action_model.call_count == (steps or 20)
                yield dict(seed=1, success=True, steps=8, total_steps=8)
            with patch.object(rollout, "_restore_policy", return_value=restored), \
                 patch.object(SimRunner, "run", autospec=True, side_effect=run):
                record, path = rollout.evaluate_policy(
                    checkpoint, weights="raw", inference_steps=steps, seeds=[1], device=str(device),
                    video=False, output_dir=root / f"eval-{steps}-{force_steps}")
            assert record["inference_steps"] == (steps or 20)
            assert record["force_inference_steps"] == force_steps
            assert record["action_steps"] == 8
            assert json.loads(path.read_text()) == json.loads(json.dumps(record))


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
                cfg.workspace.output_dir = str(Path(directory) / "run")
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
        architecture: dict[str, Any] = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, image_size=32, patch_size=16)
        if kind == "dino":
            architecture["model_type"] = "dinov2"
        elif kind == "dinov3":
            architecture = dict(embed_dim=32, depth=1, num_heads=4)
        kwargs: dict[str, Any] = dict(horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19, state_dim=19,
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
        batch: dict[str, Any] = dict(obs=obs, action=torch.randn(2, 8, 19, device=device))
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
                      normalization_dims=dims,
                      normalization=dict(joint_state="limits", rgb="identity", action="limits"))
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
                    options: dict[str, Any] = dict(rtc_prefix=batch["action"][:, :3], delay_steps=1,
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

    with tempfile.TemporaryDirectory(prefix="dexmani-dinov3-config-") as directory:
        for method in ("dp", "sat", "dqrise", "dexlatent"):
            cfg = load_policy_config(method, ["agent.rgb_backbone_name=dinov3",
                                              f"workspace.output_dir={directory}"])
            with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("unexpected HF config")):
                prepare_training(cfg)
            assert cfg.agent.rgb_backbone_name == "dinov3"

    for kind in ("dino", "siglip", "dinov3"):
        arch: dict[str, Any] = architecture if kind == "dinov3" else dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, image_size=32, patch_size=16,
        )
        if kind == "dino":
            arch = {"model_type": "dinov2", **arch}
        spatial_encoder = SpatialObservationEncoder(
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
        output = spatial_encoder(obs)
        assert output.shape == (1, 26, 256) and torch.isfinite(output).all()
        output.square().mean().backward()
        gradient = spatial_encoder.alignment.proj.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0
        restored_spatial = SpatialObservationEncoder(num_points=32, n_obs_steps=2, **spatial_encoder.effective_config).to(device).eval()
        restored_spatial.load_state_dict(spatial_encoder.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored_spatial(obs), output, rtol=0, atol=0)


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


def check_sim_observations(device):
    from dexmani_policy.common.config import POINT_CLOUD_FIELDS
    from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
    from dexmani_policy.deployment.runtime import PolicyRuntime
    from dexmani_policy.env_runner.sim_runner import SimRunner

    fields = sorted(POINT_CLOUD_FIELDS)
    runner = SimRunner("unused", 2, fields, {})
    dims = dict.fromkeys(fields, 3)
    agent = Policy(torch.nn.Linear(1, 1), torch.nn.Identity(), 2, 2, 1, 19).to(device)
    agent.normalization_dims = dims
    info = SimpleNamespace(observation_fields=fields, n_obs_steps=2, pc_dim=3)
    runtime = PolicyRuntime(agent, info, seed=0)
    observations = []
    tensor = torch.as_tensor
    for index in range(3):
        observation = {key: np.arange(8 * 6, dtype=np.float32).reshape(8, 6) + index * 100
                       for key in fields}
        observations.append(observation)
        runner.update_obs(observation)
        with patch("dexmani_policy.env_runner.sim_runner.torch.as_tensor", wraps=tensor) as transfer:
            actual = runner.get_obs_batch(device, normalization_dims=dims)
        assert len(transfer.call_args_list) == len(fields)
        for call in transfer.call_args_list:
            assert call.args[0].shape == (2, 8, 3)
            assert call.args[0].flags.c_contiguous
        for key in fields:
            history = np.stack([observations[max(0, index - 1)][key], observations[index][key]])
            old_device_slice = tensor(history, device=device)[..., :3].unsqueeze(0)
            torch.testing.assert_close(actual[key], old_device_slice, rtol=0, atol=0)
        external = runtime.prepare_observations(runner.get_stacked_obs())
        torch.testing.assert_close(external, actual, rtol=0, atol=0)
    dims["object_pointcloud"] = 6
    history = {key: np.stack([obs[key] for obs in observations]).astype(np.float64) for key in fields}
    external = runtime.prepare_observations(history)
    torch.testing.assert_close(external, runner.get_obs_batch(device, normalization_dims=dims), rtol=0, atol=0)
    for key, channels in dims.items():
        assert external[key].shape == (1, 2, 8, channels)
        assert external[key].dtype == torch.float32 and external[key].is_contiguous()
        try:
            runtime.prepare_observations({**history, key: history[key][..., :channels - 1]})
        except ValueError as error:
            assert key in str(error) and "fewer channels" in str(error)
        else:
            raise AssertionError(f"Runtime accepted too few {key} channels")
    invalid = history["object_pointcloud"].copy()
    invalid[-1, 0, -1] = np.finfo(np.float64).max
    try:
        runtime.prepare_observations({**history, "object_pointcloud": invalid})
    except ValueError as error:
        assert "Nonfinite" in str(error)
    else:
        raise AssertionError("Nonfinite point cloud escaped FP32 conversion")
    runner.reset()
    assert runner.get_obs_batch(device, normalization_dims={}) == {}
    kwargs = rgb_preprocessing_kwargs({"rgb_preprocess_size": (10, 12), "rgb_color_aug": {"present": True}})
    runner = SimRunner("unused", 2, ["rgb"], kwargs)
    rgb = np.random.default_rng(3).integers(256, size=(7, 9, 3), dtype=np.uint8)
    runner.update_obs({"rgb": rgb})
    simulated = runner.get_obs_batch(device, normalization_dims={"rgb": 3})["rgb"]
    info = SimpleNamespace(observation_fields=("rgb",), n_obs_steps=2, rgb_preprocessing=kwargs)
    runtime = PolicyRuntime(torch.nn.Linear(1, 1).to(device), info, seed=0)
    external = runtime.prepare_observations({"rgb": np.stack((rgb, rgb))})["rgb"]
    assert simulated.dtype == external.dtype == torch.uint8
    assert simulated.is_contiguous() and external.is_contiguous()
    torch.testing.assert_close(simulated, external, rtol=0, atol=0)


def check_image_geometry(device):
    from dexmani_policy.agents.obs_encoder.pointcloud.projection import project_points
    from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor

    height, width = 6, 8
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((x / (width - 1), y / (height - 1), (x + y) / (height + width - 2)))[None]
    processor = ImageProcessor((10, 14), image_mean=(0, 0, 0), image_std=(1, 1, 1)).to(device)
    resized, transform = processor.process_images(image, return_transform=True)
    pixels = torch.tensor([[[2.2, 1.7], [4.1, 3.2]]], device=device)
    points = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    projection = project_points(points, torch.eye(3, device=device)[None],
                                torch.eye(4, device=device)[None, :3], transform, (resized.shape[-2], resized.shape[-1]))
    sampled = F.grid_sample(resized, projection.grid_xy[:, :, None], align_corners=False)[:, :, :, 0].transpose(1, 2)
    expected = torch.stack((pixels[..., 0] / (width - 1), pixels[..., 1] / (height - 1),
                            pixels.sum(-1) / (height + width - 2)), -1)
    assert (projection.confidence == 1).all()
    torch.testing.assert_close(sampled, expected, rtol=2e-6, atol=2e-6)
    quantized = image.mul(255).round().to(torch.uint8)
    torch.testing.assert_close(processor.process_images(quantized),
                               processor.process_images(quantized.float() / 255), rtol=0, atol=0)


def check_fps(device):
    from pytorch3d.ops import sample_farthest_points
    from dexmani_policy.agents.obs_encoder.pointcloud.ops import farthest_point_sample, preprocess_point_cloud

    points = torch.randn(2, 17, 6, device=device)
    for random in (False, True):
        torch.manual_seed(29)
        _, expected_indices = sample_farthest_points(points[..., :3], K=7, random_start_point=random)
        if random:
            order = torch.rand(2, 7, device=device).argsort(dim=1)
            expected_indices = expected_indices.gather(1, order)
        expected_next = torch.rand(4, device=device)
        for channels in (3, 6):
            cloud = points[..., :channels]
            torch.manual_seed(29)
            sampled, indices = farthest_point_sample(cloud, 7, use_random=random)
            torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
            torch.testing.assert_close(sampled, cloud.gather(1, indices[..., None].expand(-1, -1, channels)), rtol=0, atol=0)
            torch.testing.assert_close(torch.rand(4, device=device), expected_next, rtol=0, atol=0)
            torch.manual_seed(31)
            first = preprocess_point_cloud(cloud, 7, training=False)
            torch.manual_seed(47)
            second = preprocess_point_cloud(cloud, 7, training=False)
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(4)
    for check in (check_fps, check_tactile, check_interaction, check_force_interaction, check_adaptac, check_rgb, check_rgb_deployment, check_dinov3, check_output_boundaries, check_sim_observations, check_image_geometry,):
        check(args.device)
        print(f"PASS {check.__name__} ({args.device})", flush=True)


if __name__ == "__main__":
    main()
