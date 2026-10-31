"""Focused numerical and synthetic policy checks: python -m dexmani_policy.test_tools.agent_checks."""

import argparse
import copy
import io
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import ConsistencyDiTX
from dexmani_policy.agents.action_decoders.backbones.sat import SAT
from dexmani_policy.agents.action_decoders.sampling.time_sampling import TimeSampler, euler_time_grid
from dexmani_policy.agents.hand_prior.artifacts import load_codebook
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
from dexmani_policy.agents.hand_prior.vqvae import EMAVectorQuantizer, HandVQVAE
from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandPatchEncoder
from dexmani_policy.agents.policies.dp3 import DP3Policy
from dexmani_policy.agents.policies.dqrise import DQRISEPolicy
from dexmani_policy.common.checkpoint import Checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.training.common.optimization import build_optimizer
from dexmani_policy.training.pretrain.hand_prior.codebook import deployment_diagnostics, export_codebook
from dexmani_policy.training.pretrain.hand_prior.vqvae import evaluate_vq


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
        (SAT(16, 19, 2, 256, hidden_dim=32, n_head=4, n_layers=1, p_drop_attn=0.0),
         "blocks", (2, 16, 19), (2, 194, 256)),
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
    config = encoder.effective_config.copy()
    config.pop("_target_")
    restored = XHandPatchEncoder(**config).to(device)
    restored.load_state_dict(encoder.state_dict(), strict=True)


def check_rgb(device):
    architecture = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, image_size=32, patch_size=16)
    for mode in ("freeze", "full", "lora"):
        model = ViTEncoder("siglip", tune_mode=mode, out_dim=16,
                           architecture=architecture).to(device)
        assert not hasattr(model.backbone.vision_model, "head")
        assert model.effective_config["architecture"]["vision_use_head"] is False
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        rgb = torch.randn(2, 3, 32, 32, device=device)
        model.train()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = model(rgb)["global_token"].float().square().mean()
        loss.backward()
        assert model.proj.weight.grad is not None and model.proj.weight.grad.abs().sum() > 0
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

    from omegaconf import OmegaConf
    from transformers import Dinov2Config, Dinov2Model, SiglipVisionConfig, SiglipVisionModel
    from dexmani_policy.agents.obs_encoder.rgb.build import prepare_initialization

    for kind, config_class, model_class in (
        ("dino", Dinov2Config, Dinov2Model), ("siglip", SiglipVisionConfig, SiglipVisionModel),
    ):
        for mode in ("freeze", "full", "lora"):
            config = config_class(**architecture)
            if kind == "siglip":
                config.vision_use_head = False
            source_model = model_class(config).to(dtype=torch.bfloat16)
            with tempfile.TemporaryDirectory(prefix="dexmani-rgb-source-") as directory:
                source_model.save_pretrained(directory)
                cfg = OmegaConf.create(dict(
                    policy_name="dp", initialization={"rgb": directory},
                    agent={"rgb_backbone_name": kind, "rgb_backbone_config": {"tune_mode": mode}},
                ))
                prepare_initialization(cfg)
                model = ViTEncoder(kind, **cfg.agent.rgb_backbone_config, out_dim=16)
                model.initialize_weights(directory)
                base = model.backbone.get_base_model() if mode == "lora" else model.backbone
                source_state = source_model.state_dict()
                for key, value in base.state_dict().items():
                    if ".lora_" not in key:
                        torch.testing.assert_close(value, source_state[key.replace(".base_layer.", ".")], rtol=0, atol=0)
            # The source directory is gone: structure and parameters alone must restore offline.
            restored = ViTEncoder(kind, **model.effective_config)
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
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, commitment, _, _ = vae(torch.randn(16, 12, device=device))
        (reconstruction + commitment).backward()
        assert vae.quantizer.layer_weights.grad is not None and vae.quantizer.layer_weights.grad.abs().sum() > 0
        optimizer.step()
    loaded = HandVQVAE(**vae.model_config).to(device)
    loaded.load_state_dict(vae.state_dict(), strict=True)
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
    metrics = evaluate_vq(model, loader, device, reconstruction=False, chunk_size=5)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    manifest = dict(source_id="agent-check", episode_ends=[10, 19], episode_ids=["train", "val"],
                    train_indices=[0], val_indices=[1])
    with tempfile.TemporaryDirectory(prefix="dexmani-codebook-check-") as directory:
        checkpoint, artifact = Path(directory) / "vq.pt", Path(directory) / "codebook.npz"
        torch.save(dict(
            model_config=model.model_config, model_state_dict=model.state_dict(), epoch=1,
            metrics={"val_lookup_mse": metrics["lookup_mse"]},
            normalizer_state_dict={"fields.hand.scale": torch.ones(12), "fields.hand.offset": torch.zeros(12)},
            split_metadata=dict(normalization_spec={"action": "limits"}, hand_data={"action_key": "action"},
                                split_manifest=manifest, selection_split="validation", selection_metric="val_lookup_mse"),
        ), checkpoint)
        table, metadata = export_codebook(checkpoint, artifact)
        unsupported_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        unsupported_state["split_metadata"]["normalization_spec"]["action"] = "auto"
        unsupported = Path(directory) / "unsupported.pt"
        torch.save(unsupported_state, unsupported)
        try:
            export_codebook(unsupported, Path(directory) / "unsupported.npz")
        except ValueError as error:
            assert "normalization limits" in str(error)
        else:
            raise AssertionError("VQ export accepted an obsolete normalization recipe")
        loaded_table = HandStateCodebook(
            model.hand_dim, num_groups=model.num_groups, codebook_size=model.codebook_size,
        )
        loaded_metadata, hand_affine = load_codebook(artifact, loaded_table)
        assert loaded_metadata == metadata
        for name, value in table.state_dict().items():
            torch.testing.assert_close(value, loaded_table.state_dict()[name], rtol=0, atol=0)
        report = deployment_diagnostics(
            table, {"validation": rows.numpy()}, metadata=metadata,
            hand_scale=hand_affine["scale"], chunk_size=5,
        )
        torch.testing.assert_close(torch.tensor(metrics["lookup_mse"]),
                                   torch.tensor(report["splits"]["validation"]["lookup_mse"]), rtol=1e-5, atol=1e-7)
        assert sum(report["splits"]["validation"]["prototype_counts"]) == len(rows)
        policy_config = dict(
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
        policy = DQRISEPolicy(**policy_config)
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
            rgb=torch.randint(256, (2, 2, 24, 32, 3), dtype=torch.uint8, device=device).permute(0, 1, 4, 2, 3).float().div_(255),
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
            policy.predict_action_from_condition(
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
            agent={"_target_": "dexmani_policy.agents.policies.dqrise.DQRISEPolicy",
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
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            expected_prediction = expected.predict_action(
                obs, generator=torch.Generator(device=device).manual_seed(1024),
            )["future_actions"]
            actual_prediction = restored.predict_action(
                obs, generator=torch.Generator(device=device).manual_seed(1024),
            )["future_actions"]
            torch.testing.assert_close(actual_prediction, expected_prediction, rtol=0, atol=0)
        checkpoint.model_state = dict(checkpoint.model_state)
        checkpoint.ema_model_state = dict(checkpoint.ema_model_state)
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
    policy = DP3Policy(**kwargs)
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
    restored = DP3Policy(**kwargs)
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
    for check in (check_time, check_precision, check_tactile, check_rgb, check_vq,
                  check_codebook, check_codebook_pipeline, check_policies):
        check(device)
        print(f"PASS {check.__name__} ({device})", flush=True)


if __name__ == "__main__":
    main()
