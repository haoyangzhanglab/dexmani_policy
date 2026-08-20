---
name: dexmani-agent-integration
description: >
  Guide for adding a new agent variant to DexMani_Policy. Covers agent class,
  config YAML, CLAUDE.md registration, optional DDP overlay, and smoke test
  verification. Use when: adding a new policy type, creating a new agent class,
  or asked to "add a new agent", "create a new policy", "extend the agent zoo".
---

# DexMani Agent Integration

Add each agent variant as a self-contained class under `dexmani_policy/agents/core/` with a
corresponding Hydra config in `dexmani_policy/configs/`. The agent system is Hydra-driven with
no explicit registry — new agents are wired through `agent._target_` in the YAML config.
`dp3` is the simplest reference; `sat` is the most complex.

## Before starting

Confirm with the user, or state the assumption explicitly:

- Which **inheritance path**: `UNetDiffusionAgent` (DP3/MoE-style, UNet+Diffusion), `DiTXFlowMatchAgent`
  (ManiFlow-style, DiTX+FlowMatch+Consistency), or `BaseAgent` direct (SAT, R3D, DQRISE — full control over backbone + decoder).
- Which **modalities**: point cloud (`pc_dim`) or RGB (`rgb_backbone_name`).
- Which **action space**: joint (19D), action_ee (21D), or FAAS (39/41D).
- Whether **DDP** multi-GPU support is needed.

## Workflow

### 1. Create the agent core file

Path: `dexmani_policy/agents/core/<name>.py`

Choose one of four patterns (reference the existing agent nearest to your target):

| Pattern | Parent class | `obs_encoder` output | What you pass to parent | Examples |
|---------|-------------|---------------------|------------------------|---------|
| A: UNet+Diffusion | `UNetDiffusionAgent` | `(out_dim,)` flat vector | `obs_encoder.out_dim * n_obs_steps` → `context_dim` | `dp3.py`, `dp.py`, `moe.py` |
| B: DiTX+FlowMatch+Consistency | `DiTXFlowMatchAgent` | `(num_tokens, token_dim)` sequence | `num_obs_tokens`, `obs_token_dim` | (reserved for ManiFlow-like) |
| D: Direct BaseAgent | `BaseAgent` | Arbitrary | Build backbone + decoder yourself, pass to `super().__init__` | `sat.py`, `r3d.py`, `dqrise.py`, `multitask_dit.py` |

**Pattern-D minimum skeleton** (`dp3` is a better reference for patterns A-C):

```python
import torch
from torch import nn
from dexmani_policy.agents.core.base import BaseAgent

class MyObsEncoder(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.out_dim = ...  # MANDATORY: exposed for context_dim computation

    def forward(self, obs):
        # obs keys already preprocessed (normalized, time-flattened to B*T)
        # return (cond, aux_dict)
        return cond, {}

class MyAgent(BaseAgent):
    def __init__(self, horizon, n_obs_steps, n_action_steps, action_dim, ...):
        obs_encoder = MyObsEncoder(...)
        backbone = ...  # your nn.Module
        action_decoder = ...  # Diffusion / FlowMatch wrapping backbone
        super().__init__(obs_encoder, action_decoder, horizon, n_obs_steps, n_action_steps, action_dim)


def example():
    """Standalone smoke test — every agent must have this."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, A = 2, 2, 16, 19
    agent = MyAgent(horizon=H, n_obs_steps=T, n_action_steps=8, action_dim=A, ...).to(device)
    obs = {"joint_state": torch.randn(B * T, A, device=device)}
    action = torch.randn(B, H, A, device=device)
    from dexmani_policy.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    normalizer.fit({"action": action, "joint_state": obs["joint_state"].reshape(B, T, A)}, mode="limits")
    agent.load_normalizer_from_dataset(normalizer)
    batch = {"obs": {k: v.reshape(B, T, *v.shape[1:]) for k, v in obs.items()}, "action": action}
    loss, loss_dict = agent.compute_loss(batch)
    print(f"loss: {loss.item():.4f}  keys={list(loss_dict.keys())}")
    result = agent.predict_action({k: v.reshape(B, T, *v.shape[1:]) for k, v in obs.items()})
    print(f"pred_action: {result['pred_action'].shape}  control_action: {result['control_action'].shape}")
    print(f"=== {MyAgent.__name__} PASSED ===")

if __name__ == "__main__":
    example()
```

Must call `super().__init__(obs_encoder, action_decoder, horizon, n_obs_steps, n_action_steps, action_dim, ...)`.
Must include `example()` with `if __name__ == "__main__":` guard.

Optionally override:
- `compute_loss` — if action format differs (SAT's axis transpose)
- `predict_action` / `predict_action_from_cond` — if inference needs special logic
- `compile_backbone` — if CUDA graph incompatible (SAT uses `mode='default'`)
- `get_optim_param_groups` — for separate LR/WD on backbone vs obs_encoder
- `control_action_dim` (property) — for FAAS or auxiliary heads

### 2. Register in `__init__.py`

File: `dexmani_policy/agents/core/__init__.py`

Add: `from .<name> import <Name>Agent`

### 3. Create the config YAML

Path: `dexmani_policy/configs/<name>.yaml`

Copy `dp3.yaml` as a template. The required top-level fields (all 18 configs share these):

| Field | Typical value | Notes |
|-------|--------------|-------|
| `policy_name` | `"<name>"` | W&B group name |
| `task_name` | `pour` | Dataset task |
| `zarr_path` | `robot_data/pour.zarr` | `DATA_DIR` prefix handled by config resolver |
| `seed` | `0` | Training seed |
| `horizon` | `16` | **Invariant — never change** |
| `n_obs_steps` | `2` | **Invariant — never change** |
| `n_action_steps` | `8` | **Invariant — never change** |
| `action_key` | `action` | `action` (joint) or `action_ee` (end-effector) |
| `action_dim` | `${eval:'...'}` | See formula below |
| `dataloader` | `{batch_size, num_workers, ...}` | |
| `val_dataloader` | Same structure | |
| `dataset` | `{_target_, zarr_path, horizon, ...}` | Must include `_target_` for Hydra instantiation |
| `agent` | `{_target_, horizon, n_obs_steps, n_action_steps, action_dim, ...}` | Must include `_target_: dexmani_policy.agents.core.<name>.<Name>Agent` |
| `optimizer` | `{lr, weight_decay, betas, ...}` | AdamW `fused=True` |
| `ema` | `{_target_, ...}` | EMAModel config |
| `training` | `{seed, device, use_bfloat16, use_compile, use_ema, ...}` | |
| `workspace` | `{_target_, ...}` | TrainWorkspace |
| `env_runner` | `{_target_, task_name, ...}` | SimRunner |
| `eval` | `{denoise_steps, use_ema, select_best, ...}` | Copy from dp3.yaml exactly |
| `hydra` | `{job, run, sweep}` | Output dirs |

`action_dim` formula (paste into config):
```yaml
action_dim: ${eval:'21 if ${eq:${action_key},action_ee} else 19'}
```

Copy the `eval:` section **exactly** from `dp3.yaml` — all policies share the same eval structure.

### 4. Update CLAUDE.md

Four places to update:

1. **Agent 变体对比表** — add a row with columns: Agent name, Input, Encoder, Backbone, Decoder, Config, 独特点
2. **配置速查** — add `<name>.yaml` to the file list; add a column to the parameter quick-reference table
3. **训练命令** — add example: `bash scripts/training/train.sh <name> pour`
4. **DDP** (if applicable) — add to the DDP config list and batch-size table

### 5. Optionally create DDP overlay

Path: `dexmani_policy/configs/ddp/<name>.yaml`

Template:
```yaml
# @package _global_
defaults:
  - /<name>
  - _self_

policy_name: ddp/<name>

training:
  num_gpus: 4
  gpu_ids: null

dataloader:
  batch_size: <per_gpu_batch>
  num_workers: 4

val_dataloader:
  batch_size: <per_gpu_batch>
  num_workers: 4

hydra:
  job:
    override_dirname: ${policy_name}_${task_name}
  run:
    dir: experiments/${policy_name}/${task_name}/${now:%Y-%m-%d_%H-%M}_${training.seed}
  sweep:
    dir: experiments/${policy_name}/${task_name}/${now:%Y-%m-%d_%H-%M}_${training.seed}
    subdir: ${hydra.job.num}
```

**Known exceptions** (intentional, not errors): `dp3` (non-FAAS) and `moe_dp` have no DDP configs.

### 6. Verify with smoke test

```bash
conda activate policy
export DATA_DIR=/path/to/data
python dexmani_policy/smoke_test.py <name>
```

The 6 stages validate: (1) dataset+normalizer, (2) model+EMA, (3) optimizer+scheduler,
(4) forward+backward, (5) predict_action shape, (6) checkpoint roundtrip.
FAAS mode adds stage 5.0 (native-FAAS roundtrip). MoE adds stage 5.1 (enhanced gate).

## Conventions

- `action_dim` uses Hydra eval: `${eval:'21 if ${eq:${action_key},action_ee} else 19'}`
- Every agent includes `example()` with standalone smoke test
- `n_action_steps: 8` — never changes
- AdamW `fused=True` (all configs)
- `cond_predict_scale=True` for UNet backbones
- ViT backbones (DINO/CLIP/SigLIP): `bfloat16 + attn_implementation="sdpa"`
- `compile mode='reduce-overhead'` unless shuffle/动态索引 requires `mode='default'`
- For FAAS agents: use `dp3_faas.yaml` as the template, not `dp3.yaml`

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `action dim mismatch` in `compute_loss` | Config `action_dim` ≠ Agent init | Align config eval expression; for FAAS: `action_dim = tcp_dim + 32` |
| `_target_` not found | Class path or name wrong | Verify `agent._target_: dexmani_policy.agents.core.<name>.<Name>Agent` |
| `obs_encoder.out_dim` AttributeError | obs_encoder missing `out_dim` | Add `self.out_dim = ...` in obs encoder `__init__` |
| `context_dim` shape mismatch in UNet | `obs_encoder.out_dim * n_obs_steps` wrong | Verify encoder flattens time dims correctly |
| DDP config references wrong base | Typo in `defaults` | Match base config filename exactly (no `.yaml`) |
| Smoke test stage 1 fails with Zarr error | `DATA_DIR` not set | `export DATA_DIR=/path/to/data` |
| FAAS `state_dim` mismatch | Forgot to override `agent.state_dim` | FAAS needs `state_dim: ${eval:'7 + ${faas_hand_dim}'}` |

## Reporting back

Confirm: (a) smoke test passes, (b) DDP smoke if applicable, (c) CLAUDE.md table consistent
with config, (d) `__init__.py` has export, (e) anything still unverified.
