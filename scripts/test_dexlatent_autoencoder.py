#!/usr/bin/env python3
"""Phase 1 smoke test: DexLatentHandVAE standalone autoencoder.

Verifies:
  1. load_pretrained() loads Phase 0 checkpoint
  2. encode/decode shapes are correct
  3. Tanh output bounds [-1, 1]
  4. All parameters frozen (requires_grad=False)
  5. Roundtrip fidelity vs original DexLatent model
  6. Batch-dimension handling (B), (B,T), arbitrary leading dims
  7. transform_action / inverse_transform_action / transform_joint_state

Usage:
    python scripts/test_dexlatent_autoencoder.py
    python scripts/test_dexlatent_autoencoder.py --ckpt <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dexmani_policy.common.dexlatent_autoencoder import DexLatentHandVAE

DEFAULT_CKPT = str(
    _PROJECT_ROOT / "pretrained_models" / "dexlatent_autoencoders.pt"
)

PASS = 0
FAIL = 0


def check(condition: bool, msg: str, fatal: bool = False) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {msg}")
    else:
        FAIL += 1
        print(f"  ❌ {msg}")
        if fatal:
            sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════


def test_load_pretrained(ckpt_path: str) -> DexLatentHandVAE:
    """Test 1: load_pretrained succeeds."""
    print("\n── Test 1: load_pretrained ──")
    vae = DexLatentHandVAE.load_pretrained(ckpt_path, hand_name="xarm7_xhand_right")
    check(isinstance(vae, nn.Module), "is nn.Module")
    check(vae.hand_dim == 12, f"hand_dim=12 (got {vae.hand_dim})")
    check(vae.latent_dim == 32, f"latent_dim=32 (got {vae.latent_dim})")
    check(vae.hidden_dims == (64, 128, 64), f"hidden_dims=(64,128,64) (got {vae.hidden_dims})")
    return vae


def test_encode_decode_shapes(vae: DexLatentHandVAE) -> None:
    """Test 2: encode/decode shapes."""
    print("\n── Test 2: encode/decode shapes ──")

    # 2D batch
    x = torch.randn(4, 12)
    latent = vae.encode(x)
    recon = vae.decode(latent)
    check(latent.shape == (4, 32), f"encode(B,12)→(B,32): got {latent.shape}")
    check(recon.shape == (4, 12), f"decode(B,32)→(B,12): got {recon.shape}")

    # 3D batch (B, horizon, D) — DexMani action shape
    x3d = torch.randn(2, 16, 12)
    latent3d = vae.encode(x3d)
    recon3d = vae.decode(latent3d)
    check(latent3d.shape == (2, 16, 32), f"encode(B,T,12)→(B,T,32): got {latent3d.shape}")
    check(recon3d.shape == (2, 16, 12), f"decode(B,T,32)→(B,T,12): got {recon3d.shape}")

    # 4D batch (B, n_obs_steps, N, D)
    x4d = torch.randn(2, 2, 4096, 12)
    latent4d = vae.encode(x4d)
    recon4d = vae.decode(latent4d)
    check(latent4d.shape == (2, 2, 4096, 32), f"encode 4D: got {latent4d.shape}")
    check(recon4d.shape == (2, 2, 4096, 12), f"decode 4D: got {recon4d.shape}")


def test_output_bounds(vae: DexLatentHandVAE) -> None:
    """Test 3: decoder Tanh bounds."""
    print("\n── Test 3: output bounds ──")
    # Test with extreme latent values
    latent_big = torch.randn(1024, 32) * 3.0  # 3-sigma
    recon = vae.decode(latent_big)
    check(recon.min() >= -1.0, f"decode min={recon.min():.4f} >= -1.0")
    check(recon.max() <= 1.0, f"decode max={recon.max():.4f} <= 1.0")

    # Test with very extreme values
    latent_huge = torch.randn(256, 32) * 10.0
    recon_huge = vae.decode(latent_huge)
    check(recon_huge.min() >= -1.0, f"extreme min={recon_huge.min():.4f} >= -1.0")
    check(recon_huge.max() <= 1.0, f"extreme max={recon_huge.max():.4f} <= 1.0")


def test_frozen_params(vae: DexLatentHandVAE) -> None:
    """Test 4: all parameters frozen."""
    print("\n── Test 4: frozen parameters ──")
    n_params = sum(1 for _ in vae.parameters())
    n_frozen = sum(1 for p in vae.parameters() if not p.requires_grad)
    check(n_frozen == n_params, f"{n_frozen}/{n_params} params frozen")
    check(vae.training is False, "model in eval mode")


def test_roundtrip_fidelity(vae: DexLatentHandVAE, ckpt_path: str) -> None:
    """Test 5: encode→decode matches original DexLatent model."""
    print("\n── Test 5: roundtrip fidelity vs original DexLatent ──")

    # Load original DexLatent model for comparison
    _DEXLATENT_ROOT = Path.home() / "Desktop" / "DexLatent"
    sys.path.insert(0, str(_DEXLATENT_ROOT))

    try:
        from HandLatent.model import CrossEmbodimentTrainer, TrainingConfig

        config = TrainingConfig(device=torch.device("cpu"))
        trainer = CrossEmbodimentTrainer(
            ["xarm7_xhand_right"], config
        )
        orig_ckpt = torch.load(
            str(_DEXLATENT_ROOT / "Checkpoints" / "20260311_225425" / "checkpoint_epoch_1000.pt"),
            map_location="cpu", weights_only=True,
        )
        trainer.load_autoencoders_from_payload(orig_ckpt)
        orig_ae = trainer.autoencoders["xarm7_xhand_right"]
    except Exception as e:
        print(f"  ⚠️  Cannot load original DexLatent model: {e}")
        print("     Skipping fidelity comparison")
        return

    torch.manual_seed(42)
    x = torch.rand(256, 12) * 2 - 1  # [-1, 1]

    with torch.no_grad():
        # Original: encode via backbone + mean_head
        orig_latent = orig_ae.hand_mean_head(orig_ae.hand_encoder_backbone(x))
        orig_recon = orig_ae.hand_decoder(orig_latent)

        # Ours
        our_latent = vae.encode(x)
        our_recon = vae.decode(our_latent)

    latent_diff = (orig_latent - our_latent).abs().max().item()
    recon_diff = (orig_recon - our_recon).abs().max().item()

    check(latent_diff < 1e-6, f"latent max diff={latent_diff:.1e} < 1e-6")
    check(recon_diff < 1e-6, f"recon max diff={recon_diff:.1e} < 1e-6")


def test_action_helpers(vae: DexLatentHandVAE) -> None:
    """Test 6: transform_action / inverse / transform_joint_state."""
    print("\n── Test 6: action/joint_state convenience methods ──")

    # transform_action: (B, 19) → (B, 39)
    action_native = torch.randn(4, 19)  # 7 arm + 12 hand
    action_latent = vae.transform_action(action_native, arm_dim=7)
    check(
        action_latent.shape == (4, 39),
        f"transform_action (B,19)→(B,39): got {action_latent.shape}",
    )

    # inverse_transform_action: (B, 39) → (B, 19)
    action_back = vae.inverse_transform_action(action_latent, arm_dim=7)
    check(
        action_back.shape == (4, 19),
        f"inverse_transform (B,39)→(B,19): got {action_back.shape}",
    )

    # Arm pass-through check: arm portion should be unchanged
    arm_diff = (action_native[:, :7] - action_back[:, :7]).abs().max().item()
    check(arm_diff < 1e-6, f"arm pass-through identity: diff={arm_diff:.1e}")

    # transform_joint_state: (B, 19) → (B, 39)  (arm always 7D)
    js_native = torch.randn(4, 19)
    js_latent = vae.transform_joint_state(js_native)
    check(
        js_latent.shape == (4, 39),
        f"transform_joint_state (B,19)→(B,39): got {js_latent.shape}",
    )
    js_arm_diff = (js_native[:, :7] - js_latent[:, :7]).abs().max().item()
    check(js_arm_diff < 1e-6, f"joint_state arm pass-through: diff={js_arm_diff:.1e}")

    # 3D action (B, horizon, D) — DexMani standard shape
    action3d = torch.randn(2, 16, 19)
    latent3d = vae.transform_action(action3d, arm_dim=7)
    back3d = vae.inverse_transform_action(latent3d, arm_dim=7)
    check(latent3d.shape == (2, 16, 39), f"3D transform: got {latent3d.shape}")
    check(back3d.shape == (2, 16, 19), f"3D inverse: got {back3d.shape}")

    # action_ee mode: arm_dim=9 (tcp_pos(3)+rot6d(6))
    action_ee = torch.randn(4, 21)  # 9 arm_ee + 12 hand
    latent_ee = vae.transform_action(action_ee, arm_dim=9)
    back_ee = vae.inverse_transform_action(latent_ee, arm_dim=9)
    check(
        latent_ee.shape == (4, 41),
        f"EE transform (B,21)→(B,41): got {latent_ee.shape}",
    )
    check(
        back_ee.shape == (4, 21),
        f"EE inverse (B,41)→(B,21): got {back_ee.shape}",
    )


def test_state_dict_roundtrip(vae: DexLatentHandVAE, tmp_path: str) -> None:
    """Test 7: save/load preserves weights."""
    print("\n── Test 7: state dict roundtrip ──")

    # Get reference output
    x = torch.randn(4, 12)
    with torch.no_grad():
        ref_latent = vae.encode(x).clone()
        ref_recon = vae.decode(ref_latent).clone()

    # Save
    torch.save({"vae": vae.state_dict(), "meta": {"hand_dim": vae.hand_dim}}, tmp_path)

    # Load into fresh instance
    payload = torch.load(tmp_path, map_location="cpu", weights_only=True)
    vae2 = DexLatentHandVAE(
        hand_dim=payload["meta"]["hand_dim"],
        latent_dim=32,
        hidden_dims=(64, 128, 64),
    )
    vae2.load_state_dict(payload["vae"])
    vae2.eval()

    with torch.no_grad():
        new_latent = vae2.encode(x)
        new_recon = vae2.decode(new_latent)

    latent_diff = (ref_latent - new_latent).abs().max().item()
    recon_diff = (ref_recon - new_recon).abs().max().item()
    check(latent_diff < 1e-6, f"save/load latent diff={latent_diff:.1e}")
    check(recon_diff < 1e-6, f"save/load recon diff={recon_diff:.1e}")

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


def test_multi_hand_loading(ckpt_path: str) -> None:
    """Test 8: all 4 hands loadable."""
    print("\n── Test 8: multi-hand loading ──")
    hand_names = [
        "xarm7_xhand_right",
        "xarm7_ability_right",
        "xarm7_inspire_right",
        "xarm7_paxini_right",
    ]
    expected_dofs = {"xarm7_xhand_right": 12, "xarm7_ability_right": 6,
                     "xarm7_inspire_right": 6, "xarm7_paxini_right": 16}

    for name in hand_names:
        vae = DexLatentHandVAE.load_pretrained(ckpt_path, hand_name=name)
        expected = expected_dofs[name]
        check(vae.hand_dim == expected,
              f"{name}: hand_dim={vae.hand_dim} (expected {expected})")

        # Quick sanity: encode random input
        x = torch.randn(2, vae.hand_dim)
        latent = vae.encode(x)
        recon = vae.decode(latent)
        check(latent.shape == (2, 32), f"{name} encode shape={latent.shape}")
        check(recon.shape == (2, vae.hand_dim), f"{name} decode shape={recon.shape}")
        check(recon.min() >= -1.0 and recon.max() <= 1.0, f"{name} Tanh bounds ok")


# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test DexLatentHandVAE autoencoder"
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Path to dexlatent_autoencoders.pt")
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        print("  Run Phase 0 first: python scripts/extract_dexlatent_weights.py")
        sys.exit(1)

    print("=" * 60)
    print("Phase 1 Smoke Test: DexLatentHandVAE")
    print("=" * 60)

    vae = test_load_pretrained(args.ckpt)
    test_encode_decode_shapes(vae)
    test_output_bounds(vae)
    test_frozen_params(vae)
    test_roundtrip_fidelity(vae, args.ckpt)
    test_action_helpers(vae)
    test_state_dict_roundtrip(vae, "/tmp/_dexlatent_test.pt")
    test_multi_hand_loading(args.ckpt)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {FAIL} TEST(S) FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
