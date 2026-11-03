"""Run hand prior checks: python -m dexmani_policy.test_tools.hand_prior_checks."""

import argparse
import copy
import tempfile
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
from dexmani_policy.training.common.config import load_policy_config
from dexmani_policy.training.pretrain.hand_prior.codebook import export_codebook
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
        table = HandStateCodebook(12, num_codes=size).to(device)
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
        restored = HandStateCodebook(12, num_codes=size).to(device)
        restored.load_state_dict(table.state_dict(), strict=True)
        assert set(restored.state_dict()) == {"sorted_hand_poses"}
    single = HandStateCodebook(12, num_codes=1).to(device)
    single.set_prototypes(torch.ones(1, 12, device=device))
    continuous, indices = single.encode(torch.ones(2, 12, device=device))
    assert (continuous == -1).all() and (indices == 0).all()
    assert (single.decode(torch.tensor([-3., 0., 3.], device=device))[1] == 0).all()
    odd = HandStateCodebook(12, num_codes=3).to(device)
    odd.set_prototypes(torch.arange(3, device=device)[:, None].expand(-1, 12))
    assert odd.decode(torch.tensor([-0.5, 0., 0.5], device=device))[1].tolist() == [1, 1, 2]


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
                f"dataset.val_episodes={val_episodes}", f"vq_vae.output_dir={output}",
                "vq_vae.num_epochs=2", "vq_vae.batch_size=4", "vq_vae.save_epochs=1",
                "vq_vae.latent_dim=8", "vq_vae.hidden_dim=16", "vq_vae.num_layers=1",
                "vq_vae.num_groups=1", "vq_vae.codebook_size=3", "vq_vae.warmup_steps=0",
                f"vq_vae.device={device}",
            ])
            with patch.object(vq_training, "load_hand_array", wraps=vq_training.load_hand_array) as read:
                _, val, _ = vq_training.prepare_hand_data(cfg)
                assert read.call_count == 1 + val_episodes
                assert (val is None) == (val_episodes == 0)
            evaluation = ({"wraps": evaluate_vq} if val_episodes else
                          {"side_effect": AssertionError("V=0 must not evaluate training data")})
            with patch.object(vq_training, "evaluate_vq", **evaluation) as evaluate, \
                 patch.object(vq_training, "DataLoader", wraps=DataLoader) as loaders, \
                 patch.object(vq_training, "export_codebook", wraps=export_codebook) as export:
                vq_training.train(cfg)
                assert evaluate.call_count == 2 * val_episodes
                assert loaders.call_count == 1 + val_episodes
                selected = "best" if val_episodes else "last"
                assert export.call_args.args[0] == output / f"vqvae_hand_{selected}.pt"
            assert (output / "vqvae_hand_last.pt").exists()
            assert not (output / "vqvae_hand_epoch=0002.pt").exists()
            assert (output / "vqvae_hand_best.pt").exists() == bool(val_episodes)
            periodic = torch.load(output / "vqvae_hand_epoch=0001.pt", weights_only=True)
            assert "split_metadata" not in periodic
            assert ("val_lookup_mse" in periodic["metrics"]) == bool(val_episodes)
            # Export needs neither selection information nor the training epoch.
            for key in ("metrics", "epoch"):
                del periodic[key]
            minimal = output / "minimal.pt"
            torch.save(periodic, minimal)
            table = export_codebook(minimal, output / "reexported.npz")
            assert table.num_codes == 3


def check_codebook_pipeline(device):
    model = HandVQVAE(12, [1] * 12, latent_dim=8, hidden_dim=16, num_layers=1,
                      num_groups=1, codebook_size=3).to(device)
    rows = torch.randn(19, 12).clamp(-1, 1)
    loader = DataLoader(TensorDataset(rows), batch_size=7)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = evaluate_vq(model, loader, device)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with tempfile.TemporaryDirectory(prefix="dexmani-codebook-check-") as directory:
        vq_checkpoint, artifact = Path(directory) / "vq.pt", Path(directory) / "codebook.npz"
        hand_scale = torch.linspace(0.5, 2., 12)
        hand_offset = torch.linspace(-0.2, 0.3, 12)
        torch.save(dict(
            model_config=model.model_config, model_state_dict=model.state_dict(),
            normalizer_state_dict={"fields.hand.scale": hand_scale, "fields.hand.offset": hand_offset},
        ), vq_checkpoint)
        table = export_codebook(vq_checkpoint, artifact)
        with np.load(artifact) as payload:
            assert set(payload.files) == {"sorted_hand_poses", "hand_normalizer_scale", "hand_normalizer_offset"}
        loaded_table, affine = load_codebook(artifact)
        torch.testing.assert_close(affine["scale"], hand_scale)
        torch.testing.assert_close(affine["offset"], hand_offset)
        for name, value in table.state_dict().items():
            torch.testing.assert_close(value, loaded_table.state_dict()[name], rtol=0, atol=0)
        lookup_mse = (rows[:, None] - table.sorted_hand_poses).square().sum(-1).amin(-1).mean() / model.hand_dim
        torch.testing.assert_close(torch.tensor(metrics["lookup_mse"]),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(4)
    for check in (check_vq, check_codebook, check_vq_training, check_codebook_pipeline,):
        check(args.device)
        print(f"PASS {check.__name__} ({args.device})", flush=True)


if __name__ == "__main__":
    main()
