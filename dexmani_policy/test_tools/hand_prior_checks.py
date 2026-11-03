"""Run hand prior checks: python -m dexmani_policy.test_tools.hand_prior_checks."""

import argparse
import copy
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import zarr

from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, load_codebook
from dexmani_policy.agents.hand_prior.vqvae import EMAVectorQuantizer, HandVQVAE
from dexmani_policy.agents.policies.dqrise import DQRISE
from dexmani_policy.common.checkpoint import Checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.split import split_episodes
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.pretrain.hand_prior.codebook import build_hand_prototypes, export_codebook
from dexmani_policy.training.pretrain.hand_prior.vqvae import evaluate_vq
from dexmani_policy.training.pretrain.hand_prior import vqvae as vq_training


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
        expected_output = state["embed"][expected_indices]
        output, loss = quantizer(x)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(loss, F.mse_loss(expected_output, x))
        commitment_grad, = torch.autograd.grad(loss, x, retain_graph=True)
        torch.testing.assert_close(commitment_grad, 2 * (x.detach() - expected_output) / x.numel())
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
        torch.testing.assert_close(quantizer(torch.ones(1, 12, device=device))[0], quantizer.embed[1:2])

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
    for name, value in vae.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def check_vq_objective(device):
    model = HandVQVAE(12, [1] * 8 + [0.5] * 4, latent_dim=8, hidden_dim=16, num_layers=1).to(device)
    with torch.no_grad():
        model.quantizer.layer_weights.copy_(torch.tensor([-1., 2.], device=device))
    reference = copy.deepcopy(model)
    hand = torch.randn(5, 12, device=device)
    actual = model(hand)
    residual = reference.encoder(hand)
    latent = torch.zeros_like(residual)
    losses = []
    for weight, layer in zip(reference.quantizer.layer_weights.softmax(0), reference.quantizer.layers):
        index = (residual.detach()[:, None] - layer.embed).square().sum(-1).argmin(-1)
        selected = layer.embed[index]
        losses.append(F.mse_loss(selected, residual))
        quantized = residual + (selected - residual).detach()
        latent = latent + weight * quantized
        residual = residual - quantized.detach()
    decoded = reference.decoder(latent)
    expected = (
        ((hand - decoded).abs() * reference.loss_weight).mean(),
        torch.stack(losses).sum(), F.mse_loss(hand, decoded),
    )
    torch.testing.assert_close(actual, expected)
    (3 * actual[0] + 5 * actual[1]).backward()
    (3 * expected[0] + 5 * expected[1]).backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    weights = model.quantizer.layer_weights.softmax(0)
    latents = torch.stack([weights[0] * a + weights[1] * b for a in model.codebooks[0] for b in model.codebooks[1]])
    torch.testing.assert_close(build_hand_prototypes(model), model.decoder(latents).float().clamp(-1, 1))


def check_vq_tail_batches(device):
    for count in (1, 6, 8):
        with tempfile.TemporaryDirectory(prefix="dexmani-vq-tail-") as directory:
            root = Path(directory)
            group = zarr.open_group(str(root / "data.zarr"), mode="w")
            group.create_dataset("meta/episode_ends", data=np.array([count]))
            group.create_dataset("data/action", data=np.arange(count * 19, dtype=np.float32).reshape(count, 19))
            cfg = load_policy_config("pretrain/hand_prior/vqvae", [
                f"zarr_path={root / 'data.zarr'}", f"workspace.output_dir={root / 'run'}",
                "vq_vae.num_epochs=2", "vq_vae.batch_size=4", "vq_vae.save_epochs=1",
                "vq_vae.latent_dim=8", "vq_vae.hidden_dim=16", "vq_vae.num_layers=1",
                "vq_vae.num_groups=1", "vq_vae.codebook_size=1", "vq_vae.warmup_steps=1",
                f"vq_vae.device={device}",
            ])
            expected_rows, _, _ = vq_training.prepare_hand_data(cfg)
            seen = []
            observed = []
            forward = HandVQVAE.forward
            def record_forward(model, hand):
                seen.append(hand.detach().cpu().numpy())
                values = forward(model, hand)
                observed.append((len(hand), [float(v.detach()) for v in values]))
                return values
            with (
                patch.object(HandVQVAE, "forward", record_forward),
                patch.object(vq_training, "evaluate_vq", side_effect=AssertionError("No validation requested")),
                patch.object(vq_training, "build_scheduler", wraps=vq_training.build_scheduler) as scheduler,
            ):
                vq_training.train(cfg)
            updates = (count + 3) // 4
            assert scheduler.call_args.kwargs["num_training_steps"] == 2 * updates
            assert len(observed) == 2 * updates
            records = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
            for epoch, epoch_metrics in enumerate(records):
                batches = observed[epoch * updates:(epoch + 1) * updates]
                assert sum(n for n, _ in batches) == count
                actual_rows = np.concatenate(seen[epoch * updates:(epoch + 1) * updates])
                np.testing.assert_array_equal(actual_rows[np.argsort(actual_rows[:, 0])], expected_rows)
                expected = np.sum([np.array(v) * n for n, v in batches], axis=0) / count
                np.testing.assert_allclose([epoch_metrics[k] for k in ("train_enc", "train_vq", "train_mse")], expected)
            table, _ = load_codebook(root / "run/codebook.npz")
            assert table.num_codes == 1


def check_codebook(device):
    for size in (256, 512):
        table = HandStateCodebook(12, num_codes=size).to(device)
        table.sorted_hand_poses.copy_(torch.randn(size, 12, device=device))
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            decoded = table.decode(torch.tensor([-2, -1, 1, 2], device=device, dtype=dtype))
            torch.testing.assert_close(decoded, table.sorted_hand_poses[[0, 0, size - 1, size - 1]])
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
            continuous = table.encode(torch.ones(shape, device=device))
            assert continuous.shape == (*shape[:-1], 1)
            torch.testing.assert_close(continuous, torch.full_like(continuous, 2 / (size - 1) - 1))
        restored = HandStateCodebook(12, num_codes=size).to(device)
        restored.load_state_dict(table.state_dict(), strict=True)
        assert set(restored.state_dict()) == {"sorted_hand_poses"}
    single = HandStateCodebook(12, num_codes=1).to(device)
    single.sorted_hand_poses.fill_(1)
    continuous = single.encode(torch.ones(2, 12, device=device))
    assert (continuous == -1).all()
    torch.testing.assert_close(single.decode(torch.tensor([-3., 0., 3.], device=device)),
                               torch.ones(3, 12, device=device))
    odd = HandStateCodebook(12, num_codes=3).to(device)
    odd.sorted_hand_poses.copy_(torch.arange(3, device=device)[:, None].expand(-1, 12))
    torch.testing.assert_close(odd.decode(torch.tensor([-0.5, 0., 0.5], device=device)),
                               odd.sorted_hand_poses[[1, 1, 2]])


def check_codebook_loading(device):
    with tempfile.TemporaryDirectory(prefix="dexmani-codebook-loading-") as directory:
        path = Path(directory) / "codebook.npz"
        valid = dict(sorted_hand_poses=np.zeros((2, 12), dtype=np.float32),
                     hand_normalizer_scale=np.ones(12, dtype=np.float32),
                     hand_normalizer_offset=np.zeros(12, dtype=np.float32))
        invalid = [
            ("sorted_hand_poses", np.zeros((2, 11), dtype=np.float32)),
            ("sorted_hand_poses", np.empty((0, 12), dtype=np.float32)),
            ("sorted_hand_poses", np.zeros((2, 12), dtype=np.complex64)),
            ("sorted_hand_poses", np.zeros((2, 12), dtype=bool)),
            ("sorted_hand_poses", np.full((2, 12), np.nan)),
            ("sorted_hand_poses", np.full((2, 12), 1e100)),
            ("hand_normalizer_scale", np.zeros(12)),
            ("hand_normalizer_scale", np.full(12, 1e100)),
            ("hand_normalizer_offset", np.full(12, np.inf)),
        ]
        for key, value in invalid:
            np.savez(path, **{**valid, key: value})
            try:
                load_codebook(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid {key} was accepted: {value.dtype}, {value.shape}")
        np.savez(path, **valid)
        table, affine = load_codebook(path)
        torch.testing.assert_close(table.sorted_hand_poses, torch.zeros(2, 12))
        torch.testing.assert_close(affine["scale"], torch.ones(12))


def check_codebook_ordering(device):
    direction = torch.tensor([0.5, -0.75, 1.] + [0.] * 9, device=device)
    with tempfile.TemporaryDirectory(prefix="dexmani-codebook-ordering-") as directory:
        root = Path(directory)
        for size in (1, 16, 512):
            model = HandVQVAE(12, [1] * 12, latent_dim=12, hidden_dim=12, num_layers=1,
                              num_groups=1, codebook_size=size).to(device)
            positions = torch.linspace(-0.5, 0.5, size, device=device)
            if size > 1:
                positions[size // 2] = positions[size // 2 - 1]
            expected = positions[:, None] * direction
            with torch.no_grad():
                # The decoder is identity on these poses; its ReLU input stays positive.
                model.decoder[0].weight.copy_(torch.eye(12, device=device))
                model.decoder[0].bias.fill_(1)
                model.decoder[2].weight.copy_(torch.eye(12, device=device))
                model.decoder[2].bias.fill_(-1)
                model.quantizer.layers[0].embed.copy_(expected[torch.randperm(size, device=device)])
            checkpoint, artifact = root / f"vq-{size}.pt", root / f"codebook-{size}.npz"
            scale, offset = torch.linspace(0.5, 2., 12), torch.linspace(-0.2, 0.3, 12)
            torch.save(dict(model_config=model.model_config, model_state_dict=model.state_dict(),
                            normalizer_state_dict={"fields.hand.scale": scale, "fields.hand.offset": offset}),
                       checkpoint)
            assert export_codebook(checkpoint, artifact) == size
            table, affine = load_codebook(artifact)
            # The largest component of the known first principal axis is positive.
            torch.testing.assert_close(table.sorted_hand_poses, expected.cpu(), atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(affine["scale"], scale, rtol=0, atol=0)
            torch.testing.assert_close(affine["offset"], offset, rtol=0, atol=0)
            unsorted = build_hand_prototypes(model)
            actual = table.sorted_hand_poses.to(device)
            for column in range(12):
                torch.testing.assert_close(actual[:, column].sort().values, unsorted[:, column].sort().values,
                                           atol=1e-7, rtol=1e-5)
            queries = torch.randn(7, 12, device=device)
            before = (queries[:, None] - unsorted).square().sum(-1).amin(-1).mean()
            after = (queries[:, None] - actual).square().sum(-1).amin(-1).mean()
            torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)


def check_vq_training(device):
    with tempfile.TemporaryDirectory(prefix="dexmani-vq-training-check-") as directory:
        source = Path(directory) / "commands.zarr"
        ends = np.cumsum([7, 9, 11, 13])
        action = np.arange(int(ends[-1]) * 19, dtype=np.float32).reshape(-1, 19) / 100
        group = zarr.open_group(str(source), mode="w")
        group.create_dataset("meta/episode_ends", data=ends)
        group.create_dataset("data/action", data=action, chunks=(5, 19))
        for val_episodes in (0, 1):
            output = Path(directory) / f"val{val_episodes}"
            cfg = load_policy_config("pretrain/hand_prior/vqvae", [
                f"dataset.zarr_path={source}", "dataset.max_train_episodes=2",
                f"dataset.val_episodes={val_episodes}", f"workspace.output_dir={output}",
                "vq_vae.num_epochs=2", "vq_vae.batch_size=4", "vq_vae.save_epochs=1",
                "vq_vae.latent_dim=8", "vq_vae.hidden_dim=16", "vq_vae.num_layers=1",
                "vq_vae.num_groups=1", "vq_vae.codebook_size=3", "vq_vae.warmup_steps=0",
                f"vq_vae.device={device}",
            ])
            normalized, val, normalizer = vq_training.prepare_hand_data(cfg)
            train_mask, val_mask = split_episodes(4, seed=cfg.dataset.seed, max_train_episodes=2, val_episodes=val_episodes)
            rows = np.repeat(train_mask, np.diff(np.r_[0, ends]))
            np.testing.assert_allclose(normalized, normalizer["hand"].normalize(action[rows, 7:19]).numpy())
            assert (val is None) == (val_episodes == 0)
            if val is not None:
                rows = np.repeat(val_mask, np.diff(np.r_[0, ends]))
                np.testing.assert_allclose(val, normalizer["hand"].normalize(action[rows, 7:19]).numpy())
            evaluation = nullcontext() if val_episodes else patch.object(
                vq_training, "evaluate_vq", side_effect=AssertionError("V=0 must not evaluate training data"),
            )
            with evaluation:
                vq_training.train(cfg)
            try:
                vq_training.train(cfg)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Fresh VQ overwrote an existing run")
            selected = "best" if val_episodes else "last"
            checkpoint = torch.load(output / f"checkpoints/{selected}.pt", weights_only=True)
            if val_episodes:
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                assert checkpoint["metrics"]["val_lookup_mse"] == min(row["val_lookup_mse"] for row in records)
            else:
                assert checkpoint["epoch"] == cfg.vq_vae.num_epochs
            selected_model = HandVQVAE.from_checkpoint(checkpoint).eval()
            expected = build_hand_prototypes(selected_model).numpy()
            exported, _ = load_codebook(output / "codebook.npz")
            actual = exported.sorted_hand_poses.numpy()
            np.testing.assert_allclose(actual[np.lexsort(actual.T)], expected[np.lexsort(expected.T)])
            assert (output / "checkpoints/last.pt").exists()
            assert not (output / "checkpoints/epoch_0002.pt").exists()
            assert (output / "checkpoints/best.pt").exists() == bool(val_episodes)
            periodic = torch.load(output / "checkpoints/epoch_0001.pt", weights_only=True)
            assert ("val_lookup_mse" in periodic["metrics"]) == bool(val_episodes)
            # Export needs neither selection information nor the training epoch.
            for key in ("metrics", "epoch"):
                del periodic[key]
            minimal = output / "minimal.pt"
            torch.save(periodic, minimal)
            assert export_codebook(minimal, output / "reexported.npz") == 3
            assert load_codebook(output / "reexported.npz")[0].num_codes == 3


def check_codebook_pipeline(device):
    model = HandVQVAE(12, [1] * 12, latent_dim=8, hidden_dim=16, num_layers=1,
                      num_groups=2, codebook_size=3).to(device)
    with torch.no_grad():
        model.quantizer.layer_weights.copy_(torch.tensor([-1., 2.], device=device))
    rows = torch.randn(19, 12).clamp(-1, 1)
    loader = DataLoader(TensorDataset(rows), batch_size=7, generator=torch.Generator().manual_seed(2))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    lookup_values = []
    for training in (True, False):
        model.train(training)
        model.encoder.train(not training)
        modes = [module.training for module in model.modules()]
        before_rng = get_rng_state(device)
        lookup_values.append(evaluate_vq(model, loader, device))
        after_rng = get_rng_state(device)
        assert modes == [module.training for module in model.modules()]
        assert before_rng["python"] == after_rng["python"]
        assert before_rng["numpy"][0] == after_rng["numpy"][0]
        np.testing.assert_array_equal(before_rng["numpy"][1], after_rng["numpy"][1])
        assert before_rng["numpy"][2:] == after_rng["numpy"][2:]
        torch.testing.assert_close(before_rng["torch"], after_rng["torch"], rtol=0, atol=0)
        if device.type == "cuda":
            torch.testing.assert_close(before_rng["torch_cuda"], after_rng["torch_cuda"], rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert lookup_values[0] == lookup_values[1]
    with tempfile.TemporaryDirectory(prefix="dexmani-codebook-check-") as directory:
        vq_checkpoint, artifact = Path(directory) / "vq.pt", Path(directory) / "codebook.npz"
        hand_scale = torch.linspace(0.5, 2., 12)
        hand_offset = torch.linspace(-0.2, 0.3, 12)
        torch.save(dict(
            model_config=model.model_config, model_state_dict=model.state_dict(),
            normalizer_state_dict={"fields.hand.scale": hand_scale, "fields.hand.offset": hand_offset},
        ), vq_checkpoint)
        assert export_codebook(vq_checkpoint, artifact) == 9
        with np.load(artifact) as payload:
            assert set(payload.files) == {"sorted_hand_poses", "hand_normalizer_scale", "hand_normalizer_offset"}
        original_bytes = artifact.read_bytes()
        try:
            export_codebook(vq_checkpoint, artifact)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Export overwrote an existing file")
        assert artifact.read_bytes() == original_bytes
        assert export_codebook(vq_checkpoint, artifact, overwrite=True) == 9
        loaded_table, affine = load_codebook(artifact)
        torch.testing.assert_close(affine["scale"], hand_scale)
        torch.testing.assert_close(affine["offset"], hand_offset)
        poses = build_hand_prototypes(model).cpu().numpy()
        actual_poses = loaded_table.sorted_hand_poses.numpy()
        np.testing.assert_allclose(actual_poses[np.lexsort(actual_poses.T)], poses[np.lexsort(poses.T)],
                                   rtol=1e-5, atol=1e-6)
        lookup_mse = (rows[:, None] - loaded_table.sorted_hand_poses).square().sum(-1).amin(-1).mean() / model.hand_dim
        torch.testing.assert_close(torch.tensor(lookup_values[0]),
                                   lookup_mse, rtol=1e-5, atol=1e-7)
        policy_config: dict[str, Any] = dict(
            horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
            num_codes=loaded_table.num_codes,
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
            hand_coordinates=dict(hand_joint_order=["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"],
                       hand_units="radians"),
            action_key="action",
        )
        policy.initialize_training(description, loaded_table)
        policy.normalizer["action"].scale[-12:].copy_(affine["scale"])
        policy.normalizer["action"].offset[-12:].copy_(affine["offset"])
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
        for rtc in ({"rtc_prefix": prediction[:, :1]}, {"delay_steps": 1}, {"rtc_guidance_cap": 2}):
            try:
                policy.predict_action(obs, **rtc)
            except NotImplementedError:
                pass
            else:
                raise AssertionError("DQ-RISE accepted unsupported RTC arguments")

        ema = copy.deepcopy(policy)
        with torch.no_grad():
            next(ema.parameters()).add_(0.01)
        saved = Path(directory) / "policy.pt"
        effective = dict(
            agent={"_target_": "dexmani_policy.agents.policies.dqrise.DQRISE",
                   **policy_config, **policy.effective_config},
            normalization_dims=dims,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(4)
    for check in (check_vq, check_vq_objective, check_vq_tail_batches, check_codebook,
                  check_codebook_loading, check_codebook_ordering, check_vq_training, check_codebook_pipeline):
        check(args.device)
        print(f"PASS {check.__name__} ({args.device})", flush=True)


if __name__ == "__main__":
    main()
