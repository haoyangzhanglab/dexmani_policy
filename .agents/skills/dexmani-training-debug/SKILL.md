---
name: dexmani-training-debug
description: >
  Post-mortem NaN debug checkpoint inspection for DexMani_Policy.
  Covers the two-layer NaN defense (loss NaN vs gradient NaN), checkpoint
  weight inspection, and root-cause diagnosis. Use when: training produced
  NaN loss, asked to "debug training", "fix NaN", or "check why training died".
---

# DexMani Training Debug

Diagnose NaN training failures using DexMani_Policy's two-layer NaN defense.
Narrowly focused on NaN post-mortem — does not cover config validation
(use `dexmani-pr-check` for that) or general performance debugging.

## Before starting

Locate the experiment directory and NaN debug checkpoint:

```bash
EXP_DIR="experiments/<policy>/<task>/<timestamp>"
# Check for NaN debug checkpoints (keeps last 5)
ls "$EXP_DIR/checkpoints/nan_debug_epoch=*_step=*_*.pt" 2>/dev/null
# Check training logs
tail -n 200 "$EXP_DIR/metrics.jsonl"
```

Note whether the error was **Layer 1** (loss NaN, "Non-finite loss") or **Layer 2**
(gradient NaN, "Non-finite gradient in N parameter(s)").

## Workflow

### Step 1: Identify which NaN defense triggered

| Layer | Where | When | Error message | NaN source |
|-------|-------|------|--------------|------------|
| **Layer 1** | `trainer.py:271` | After `compute_loss()`, **before** `backward()` | `"Non-finite loss at epoch=X, step=Y"` | Data corruption or forward-pass instability |
| **Layer 2** | `trainer.py:140` | After `clip_grad_norm_()`, **before** `optimizer.step()` | `"Non-finite gradient in N parameter(s): [name1, name2, ...]"` | Backward-pass numerical instability in specific ops |

**Diagnostic implication**: Layer 1 → investigate data pipeline + forward pass.
Layer 2 → investigate the specific parameters listed in the error message.

Layer 1 saves a NaN debug checkpoint (atomic `.tmp → os.replace()`) and skips the
optimizer step (`optimizer.zero_grad(set_to_none=True)`). Layer 2 also zero-grads
and raises — both layers prevent corrupting optimizer state.

### Step 2: Inspect the NaN debug checkpoint

```python
import torch
from pathlib import Path

exp_dir = Path("experiments/<policy>/<task>/<timestamp>")
nan_ckpts = sorted(exp_dir.glob("checkpoints/nan_debug_epoch=*_step=*_*.pt"))
ckpt = torch.load(str(nan_ckpts[-1]), map_location="cpu")

# Basic info
print(f"epoch={ckpt['state']['epoch']}, step={ckpt['state']['global_step']}")
print(f"nan_loss={ckpt['state']['nan_loss']}")

# Scan for non-finite weights
any_nan = False
for k, v in ckpt['weights']['model'].items():
    if torch.isnan(v).any() or torch.isinf(v).any():
        print(f"NON-FINITE: {k}  nan={torch.isnan(v).sum().item()}  inf={torch.isinf(v).sum().item()}")
        any_nan = True
    if v.abs().max() > 100:
        print(f"LARGE: {k}  max={v.abs().max():.1f}")

if not any_nan:
    print("No non-finite weights found — NaN came from computation, not accumulated state.")

# For Layer 2: examine the params named in the error
# The error lists up to 5 parameter names with NaN/Inf gradients — inspect those:
# error_params = [...]  # from the RuntimeError message
# for name in error_params:
#     print(f"{name}: grad NaN/Inf")
```

### Step 3: Check data quality

For Layer 1 (loss NaN), verify the raw Zarr data:

```python
import numpy as np
import zarr

# Verify data integrity
data = zarr.open("robot_data/<task>.zarr", mode="r")
action = data["action"][:]
state = data["state"]["joint_state"][:]

print(f"action: shape={action.shape}  finite={np.isfinite(action).all()}")
print(f"state: shape={state.shape}  finite={np.isfinite(state).all()}")
if not np.isfinite(action).all():
    nan_idx = np.where(~np.isfinite(action))
    print(f"NaN in action at indices: {nan_idx}")
```

For Layer 1, also verify the normalizer wasn't corrupted:

```python
# Load the experiment's config and check normalizer_mode
import yaml
with open(f"{exp_dir}/config.yaml") as f:
    cfg = yaml.safe_load(f)
print(f"normalizer_mode={cfg.get('normalizer_mode', 'limits')}")
print(f"use_faas={cfg.get('use_faas', False)}")

# FAAS mode MUST use limits normalizer
if cfg.get('use_faas') and cfg.get('normalizer_mode', 'limits') != 'limits':
    print("ERROR: FAAS mode requires normalizer_mode='limits'")
```

### Step 4: Root-cause diagnosis

| Layer | Scenario | Most likely cause | Fix |
|-------|----------|------------------|-----|
| L1 | Step 0 | Corrupted Zarr or normalizer explosion | Verify Zarr `np.isfinite()`; ensure `normalizer_mode='limits'` |
| L1 | Mid-training | Diffusion predictions drifted outside normalizer range | Reduce LR; verify `prediction_type` (epsilon/sample/v_prediction) |
| L1 | FAAS mode only | Gaussian normalizer unstable on zero-padded FAAS dims | Switch to `normalizer_mode='limits'` (already enforced by `_validate_faas_config`; check if bypassed) |
| L1 | bfloat16 only | AMP overflow | Identify float32-sensitive ops (MoE softmax, FlowMatch exp); disable bfloat16 for that agent |
| L2 | Specific params | Custom op backward instability | Inspect the params named in the error; check for division-by-zero, log(0), or sqrt(negative) |
| L2 | MoE agent | load-balancing aux_loss produced NaN | MoE gate must use float32 (no bfloat16) and no compile — check config |
| L2 | FlowMatch | Time-derivative instability near t=0 | `t_sample_mode` configuration — try `beta` mode with higher `beta_s` |

### Step 5: Apply fixes and retry

After diagnosis, apply the fix and resume training:

```bash
# Training auto-resumes from latest.pt symlink
bash scripts/training/train.sh <policy> <task>

# Or if you want a fresh start with modified config:
bash scripts/training/train.sh <policy> <task> 'training.seed=43'  # new seed
```

## Contract

NaN debug checkpoint format (`simple.v1`):
```python
{
    "state": {"epoch": int, "global_step": int, "nan_loss": float},
    "weights": {
        "model": state_dict,      # from fix_state_dict(raw_model, is_current_ddp=False)
        "ema_model": state_dict,  # or None
        "optimizer": state_dict,
        "scheduler": state_dict,
    },
    "_format": "simple.v1",
    "_saved_at": timestamp,
}
```

- Atomic save: `.tmp → os.replace()` — no partial writes on crash
- Auto-cleanup: only last 5 retained (`nan_ckpts[:-5]` deleted)
- Location: `experiments/<policy>/<task>/<ts>/checkpoints/nan_debug_epoch=NNNN_step=NNNNNNNN_YYYYmmdd_HHMMSS.pt`

## Conventions (what is normal, not a bug)

- MoE without `bfloat16` / `compile` is **expected** — gate softmax needs float32
- `raw_loss / gradient_accumulation_steps` before backward is **correct** — not a missing division
- Normalizer fits on **all** replay buffer data (train + val) — by design, not a leak
- Milestone checkpoints at 20/40/60/80/100% — `latest.pt` is a symlink, not a copy

## Reporting back

- Which NaN layer triggered (Layer 1 or Layer 2)
- Any non-finite weights found in the debug checkpoint (list specific layers)
- Root cause diagnosis from the table above
- Recommended fix (command to apply)
- Whether training can resume or needs a fresh start
