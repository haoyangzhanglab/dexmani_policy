# DexMani Policy → Real Deployment Implementation Plan

> **Audience**: Codex / repository maintainers  
> **Scope**: `dexmani_policy` only.  
> **Execution order**: complete this plan before starting semantic changes in `dexmani_real`.  
> **Baseline reviewed**: `main` at `a9de8b9b8c082edc7192b5a5bf7ffaf91a7f252a` (2026-08-31).  
> **Rule**: before every phase, re-read current `HEAD`, `AGENTS.md`, and the touched source files. If `main` has moved, re-evaluate the affected facts instead of blindly applying this snapshot.

---

## 1. Goal

Make an ordinary `dexmani_policy` experiment produce a **self-contained, deterministic, machine-verifiable deployment artifact** that `dexmani_real` can consume without understanding Policy training internals.

Target flow:

```text
resolved training config + Real Policy Zarr v5
        ↓
simple.v1 training checkpoint
        ↓
Policy-native deployment exporter
        ↓
dexmani.deployment.v2 checkpoint
+ schema-v2 sidecar
+ deployment_latest.pt selector
        ↓
strict/no-network restore in dexmani_real
```

The exporter is the only new cross-repo producer boundary. Do not create a registry/factory/plugin hierarchy.

---

## 2. Non-goals

Do **not** do the following in this plan:

- Do not modify `dexmani_real` runtime code.
- Do not invent `dexmani.deployment.v3`.
- Do not let Real read `simple.v1` training checkpoints directly.
- Do not make deployment depend on optimizer/scheduler/workspace/dataset objects.
- Do not add a global Policy builder registry; Hydra `_target_` remains the model construction mechanism.
- Do not change ActionFlow architecture, solver, NFE, tokenization, conditioning, or parameter topology unless a separate research task requires it.
- Do not change model quality/evaluation behavior to accommodate deployment.
- Do not claim RGB deployment support until the evaluation preprocessing contract is reproduced exactly.
- Do not run full training, DDP, long simulation evaluation, or any real-hardware command as part of Codex validation.

---

## 3. Cross-repo contract

### 3.1 Policy owns

- Agent/model construction semantics.
- `predict_action()` output semantics.
- `action_key`, `action_dim`, `control_action_dim`.
- `horizon`, `n_obs_steps`, `n_action_steps`.
- Normalizer state.
- EMA/model selection semantics.
- Diffusion/Flow solver and NFE.
- Model-specific constructor sanitization needed for self-contained restore.
- Production of `dexmani.deployment.v2` and its sidecar.

### 3.2 Real owns

- Artifact no-follow/identity/SHA/provenance verification.
- Causal observation construction.
- Observation timestamping and run generations.
- Action target timestamps and stale-action dropping.
- EE→IK.
- Workspace/collision/delta/joint-limit `SafetyGate`.
- Coupled arm/hand publication, ticketing, ACK, SDK IO.

### 3.3 `predict_action()` deployment contract

All deployable policies must return:

```python
{
    "pred_action": Tensor[B, horizon, model_action_dim],
    "control_action": Tensor[B, n_action_steps, control_action_dim],
    ... optional diagnostic/model-specific outputs ...
}
```

`tail` is optional. Real must not require it.

Canonical executable slice:

```python
start = n_obs_steps - 1
expected_control = pred_action[
    :,
    start:start + n_action_steps,
    :control_action_dim,
]
```

The exporter/verification path must establish that `control_action` matches this contract for supported policies.

---

## 4. Current reviewed issues

### P0 correctness

1. **ActionFlow state dimension is wrong for `action_ee`.**
   - Current config: `state_dim: ${action_dim}`.
   - Real `joint_state` is always arm7 + hand12 = **19**.
   - `action_ee` is 21-D.
   - `ActionFlowObsEncoder.forward()` already rejects a mismatching state dimension.

2. **`build_train_params()` omits `use_aux_ee`.**
   - Real deployment validation requires consistent model/data/inference auxiliary-EE semantics.

3. **`training/eval_utils.py` has a `raw_state` scope bug.**
   - `raw_state` is currently assigned only inside `if train_params is not None:`.
   - Older or metadata-light checkpoints can reach an unbound variable.

4. **Policy `AGENTS.md` contains a stale invariant.**
   - It says `joint_state` dimension equals action dimension.
   - Correct invariant: `joint_state=19`, while action is 19 (`action`) or 21 (`action_ee`).

5. **DQ-RISE checkpoint-selection commentary is stale.**
   - Current training flow saves milestone/interrupt checkpoints and does not provide online `val_loss`-driven top-k selection as described by the config comment.
   - Deployment `best` selection must follow the actual offline evaluation outputs.

6. **Dependency metadata is inconsistent.**
   - `pyproject.toml`: `hydra-core>=1.3`.
   - `requirements.txt`: `hydra-core==1.2.0`.
   - `pip install -e .` does not install all dependencies required by all strategies.

### Deployment producer gaps

7. No official producer currently writes `dexmani.deployment.v2`.
8. Exporter cannot rely on `ReplayBuffer.copy_from_path()` for the Real semantic contract because Zarr root attrs are not preserved there.
9. DQ-RISE fresh construction can depend on `codebook_path`, although codebook state is checkpoint-owned after restore.
10. R3D/Uni3D construction may load pretrained weights before checkpoint restore.
11. RGB policies have a training/eval preprocessing mismatch relative to current Real raw-RGB handoff.

---

## 5. Phase P0 — correctness before exporter

**Status**: TODO  
**Semantic goal**: fix existing Policy facts without introducing deployment artifacts.

### P0.1 ActionFlow state contract

Files:

```text
dexmani_policy/configs/action_flow.yaml
dexmani_policy/agents/core/action_flow.py
AGENTS.md
focused tests
```

Change:

```yaml
# before
state_dim: ${action_dim}

# after
state_dim: 19
```

Do not make `ActionFlowObsEncoder` silently slice/pad state. The strict forward check is useful and should remain.

Also fix examples in `action_flow.py` that equate state dimension to action dimension if they would be wrong for EE mode.

Acceptance:

```text
action_key=action     → action_dim=19, state_dim=19
action_key=action_ee  → action_dim=21, state_dim=19
```

Add a config/constructor test for both variants.

### P0.2 `train_params` metadata

File:

```text
dexmani_policy/common/checkpoint_io.py
```

Add:

```python
"use_aux_ee": bool(getattr(model, "use_aux_ee", False)),
```

Keep `simple.v1` top-level format unchanged.

For a newly saved checkpoint, `train_params` should include at least:

```text
n_obs_steps
n_action_steps
action_dim
horizon
action_key
tcp_dim
hand_dim
control_action_dim
use_aux_ee
num_training_steps
```

### P0.3 Fix evaluation checkpoint loading

File:

```text
dexmani_policy/training/eval_utils.py
```

Restructure to define weight selection independently of whether `train_params` exists:

```python
raw_state = checkpoint.model_state
if use_ema:
    if checkpoint.ema_model_state is None:
        # preserve existing simulation-eval warning fallback
        warn(...)
    else:
        raw_state = checkpoint.ema_model_state
```

`train_params` validation remains conditional.

Important distinction:

- Simulation evaluation may keep warning fallback to model weights when EMA is absent.
- Deployment export with `use_ema=true` must **fail closed** if EMA state is absent.

Tests:

- metadata present + EMA present;
- metadata absent + EMA present;
- EMA requested + EMA absent;
- strict state restore still succeeds for valid cases.

### P0.4 Correct DQ-RISE checkpoint-selection docs/config comments

Do not redesign training in this PR.

Document current truth:

```text
training saves milestone/interrupt checkpoints;
best deployment checkpoint is selected by offline evaluation output,
using best_ckpt.json / best.pt according to current repository utilities.
```

Deployment `--checkpoint best` must not rank milestone filenames with missing scores.

### P0.5 Fix repository guidance

Update `AGENTS.md`:

```text
joint_state is fixed 19-D (arm7 + hand12).
action is 19-D; action_ee is 21-D.
```

If `pip install -e .` is not sufficient for all policies, say so explicitly instead of implying a complete environment.

### P0 validation

Minimum:

```bash
python -m compileall -q dexmani_policy
python dexmani_policy/smoke_test.py action_flow
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py maniflow
python dexmani_policy/smoke_test.py sat
git diff --check
```

If CUDA/data/dependencies are unavailable, report the commands not run and the missing condition. Do not turn environment failures into code changes.

---

## 6. Phase P1 — minimal deployment exporter

**Status**: BLOCKED on P0  
**Semantic goal**: make Policy the official producer of the existing Real deployment contract.

### P1.1 Minimal package

Add only:

```text
dexmani_policy/deployment/
├── __init__.py
└── export.py
```

Do not introduce a registry/factory.

Suggested API:

```python
export_deployment_artifact(
    experiment_dir: Path,
    checkpoint_selector: str = "best",
    output_path: Path | None = None,
    verify: bool = True,
) -> ExportReceipt
```

CLI:

```bash
python -m dexmani_policy.deployment.export EXP --checkpoint best --verify
```

A console script can be added later if it improves actual use.

### P1.2 Checkpoint resolution

Use current repository utilities and actual evaluation outputs.

For `best`, preferred order:

```text
best_ckpt.json
→ best.pt
→ explicit repository-approved fallback
→ error
```

Do not guess a best milestone from filename order when no score exists.

### P1.3 Read the Real Zarr contract directly

Exporter must reopen `cfg.zarr_path`:

```python
root = zarr.open_group(str(cfg.zarr_path), mode="r")
attrs = dict(root.attrs)
```

Do not infer the Real data contract from the replay-buffer object.

Required first-phase checks should include the actual current Real Policy Zarr-v5 fields, including at least:

```text
schema_name == dexmani-real-policy-zarr
schema_version == 5
domain == real
deployment_equivalent == true
task/action semantics
control dt
observation reference/alignment
state alignment
sensor modalities
point-cloud semantics when requested
```

A simulation Zarr must be rejected by the real-deployment exporter.

### P1.4 Minimal resolved inference config

Do not embed the full training Hydra config.

Embed only the values Real/model restore actually needs, e.g.:

```yaml
task_name: ...
action_key: ...
action_dim: ...
horizon: ...
n_obs_steps: ...
n_action_steps: ...
use_aux_ee: ...
agent:
  _target_: dexmani_policy.agents....
  ... fully resolved agent subtree ...
eval:
  use_ema: ...
  denoise_steps: ...
```

No dataset/env_runner/workspace `_target_` may be instantiated by Real.

### P1.5 Deployment-safe constructor sanitization

Use narrow field-based sanitization, not policy-name registries.

#### DQ-RISE

Set in exported config:

```yaml
agent:
  codebook_path: null
```

Reason: current `CodebookManager` is checkpoint state. A deployment restore must not require the original training `.npz` after strict state restore.

Preserve/check:

```text
tcp_dim
hand_dim
codebook_num_groups
codebook_size
action_key
```

Current default DQ-RISE uses `action_ee`, so the produced artifact is an EE action artifact.

#### R3D/Uni3D

If the config contains:

```yaml
agent.pc_encoder_config.use_pretrained_weights
```

export it as `false` for deployment construction, then rely on strict checkpoint restore.

This must be proven topology-preserving by tests.

#### RGB backbones

Do not add ad-hoc constructor hacks in P1. Reject unsupported RGB artifacts until P4.

### P1.6 Deployment-v2 payload

The exporter must produce **exactly the existing Real contract**.

Do not rename fields or add v3-like optional semantics in this phase.

Requirements:

- `torch.load(..., weights_only=True)` compatible;
- exact top-level and state/weights key sets expected by Real;
- canonical state-dict keys;
- model or EMA selected according to artifact inference config;
- normalizer state included through the selected model state;
- producer provenance recorded;
- no optimizer/scheduler state.

### P1.7 Sidecar schema-v2

New exports should always write the current Real sidecar schema v2.

The sidecar writer must be verified against the actual Real parser/golden fixture. Do not reproduce the schema from memory.

`required_action_steps` remains the existing serialized compatibility field. Do not reinterpret it as `n_action_steps` in the schema. Real will separately fix execution semantics.

### P1.8 Atomic output

Export sequence:

```text
write temporary checkpoint
→ fsync
→ os.replace
→ SHA-256
→ write canonical sidecar atomically
→ write/update relative deployment_latest.pt symlink atomically
→ verify final files
```

Do not partially update selector/sidecar if checkpoint creation fails.

### P1 validation

Add focused tests under an appropriate `tests/deployment/` or existing test convention.

Must prove:

- exact v2 payload schema;
- exact sidecar-v2 parser compatibility;
- strict restore;
- finite normalizer/model output;
- `pred_action` and `control_action` shapes;
- unsupported sim/RGB fail before producing final artifact;
- interrupted export leaves no selected partial artifact.

---

## 7. Phase P2 — self-contained restore and parity

**Status**: BLOCKED on P1

### P2.1 No-network test

Patch/forbid network-backed loaders during deployment verification.

The exported artifact restore must not require:

- Hugging Face downloads;
- Google Drive/gdown;
- external pretrained model fetches;
- experiment-local Python package shadowing.

### P2.2 No-external-training-file test

For supported strategies:

1. export artifact;
2. remove/rename training-only initialization file in the test fixture;
3. instantiate deployment-safe agent;
4. strict restore;
5. run prediction.

At minimum cover DQ-RISE codebook and R3D pretrained initialization behavior.

### P2.3 Direct vs exported parity

For a deterministic synthetic observation and seed:

```text
direct experiment checkpoint restore
vs
exported deployment-v2 restore
```

Compare:

```text
pred_action
control_action
```

with an explicit tolerance appropriate to the actual dtype/solver. Do not hide mismatches with loose tolerances.

### P2.4 Supported matrix

First-phase support target:

| Policy | First-phase status | Notes |
|---|---|---|
| DP3 | supported | point cloud + joint state |
| ManiFlow | supported | preserve solver/NFE |
| SAT | supported | point cloud path |
| ActionFlow | supported after P0 | state_dim=19; preserve midpoint/NFE semantics |
| DQ-RISE | supported after self-contained codebook restore | current default is EE action |
| R3D | supported when deployment construction disables pretrained init | aux-EE needs full-output/control split |
| DP RGB | deferred | preprocessing mismatch |
| MoE-DP RGB/R3M | deferred | preprocessing + external init |
| MultiTask DiT | deferred | explicit task-text conditioning contract missing |

Do not describe deferred policies as “structurally supported”.

---

## 8. Phase P3 — cross-repo handoff to `dexmani_real`

**Status**: BLOCKED on P2  
**This is the gate before Real control-semantic changes begin.**

Deliver a handoff bundle containing:

```text
Policy commit SHA
Policy working-tree cleanliness/provenance result
exporter version/command
representative deployment-v2 fixture
fixture checkpoint SHA-256
schema-v2 sidecar
resolved supported-policy contract
parity-test result
no-network/no-external-file result
```

At minimum provide one deterministic fixture that Real can use for integration tests. DP3 is a reasonable first fixture; add ActionFlow/DQ-RISE/R3D fixtures as their deployment paths are qualified.

The handoff must explicitly state:

```text
prediction future steps != executable control steps
Real must execute only n_action_steps control_action
```

After this handoff is accepted, Codex moves to the Real repository plan:

```text
dexmani_real/docs/policy_deployment_refactor_plan.md
```

Do not continue changing both repos in parallel unless a newly discovered contract bug requires returning to Policy.

---

## 9. Phase P4 — deferred RGB/text deployment

**Status**: DEFERRED

### RGB prerequisite

Current evaluation preprocessing is not equivalent to Real passing raw RGB directly to the model.

A future artifact must encode deterministic evaluation preprocessing, e.g.:

```yaml
rgb_preprocess:
  input_shape: [H, W, 3]
  value_range: uint8_0_255
  color_order: rgb
  resize: [240, 240]
  eval_crop: [224, 224]
  crop_mode: center
  interpolation: bilinear
```

Deployment must reproduce **validation/evaluation**, not training random augmentation.

Also require side-effect-free model construction (`load_pretrained=false` or equivalent) and no network.

### Text prerequisite

MultiTask deployment needs an explicit static/dynamic task-text contract. Do not infer task text from experiment directory names.

---

## 10. Dependency/document cleanup

Do after the exporter is working in the actual `policy` environment.

- Resolve `hydra-core` version disagreement between `pyproject.toml` and `requirements.txt`.
- Document which dependencies are core vs strategy-specific.
- Do not perform broad dependency upgrades in the deployment PR.
- README should explicitly show:

```text
training checkpoint != deployment artifact
```

Recommended user flow:

```text
train
→ evaluate/select checkpoint
→ export deployment-v2
→ dexmani_real inspect/check
→ shadow
```

---

## 11. Codex execution rules

Before each PR:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

Then read repository guidance.

Each PR must:

- have one primary hypothesis/semantic change;
- preserve unrelated user changes;
- avoid drive-by formatting/renames;
- add focused regression tests;
- update authoritative docs/comments affected by the change;
- clearly report commands not run because of environment limitations.

Do not run:

- real hardware commands;
- long training;
- full DDP;
- long video evaluation;
unless explicitly requested by the user.

Stop rather than guess if:

- current `main` materially diverges from this plan;
- the Real schema/parser contradicts the assumed v2 contract;
- config and checkpoint metadata conflict;
- strict restore has missing/unexpected keys;
- deployment restore attempts a network call;
- `control_action` does not match the model's declared execution slice;
- an unsupported policy would require silent preprocessing or constructor behavior.

---

## 12. Recommended PR sequence

```text
Policy PR-1  correctness
    - ActionFlow state_dim=19
    - AGENTS correction
    - use_aux_ee metadata
    - eval raw_state fix
    - DQ-RISE checkpoint-selection docs

Policy PR-2  deployment-v2 exporter
    - checkpoint resolution
    - Zarr contract
    - minimal resolved cfg.agent
    - v2 checkpoint + sidecar + selector

Policy PR-3  self-contained restore/parity
    - DQ-RISE
    - R3D
    - no-network
    - direct/export parity

Policy PR-4  cross-repo fixtures/handoff
```

Do **not** start `dexmani_real` control-semantic PRs before Policy PR-4 handoff, unless the user explicitly changes the sequencing.

---

## 13. Definition of Done

Policy-side real deployment work is complete when:

- [ ] ActionFlow joint and EE configs both consume 19-D `joint_state`.
- [ ] New checkpoints record `use_aux_ee`.
- [ ] Evaluation checkpoint loading has no `raw_state` scope failure.
- [ ] `best` selection matches the actual repository evaluation workflow.
- [ ] One command exports an ordinary experiment to `dexmani.deployment.v2` + sidecar v2.
- [ ] Export rejects sim data and unsupported RGB/text policies explicitly.
- [ ] Supported point-cloud artifacts restore with strict state loading.
- [ ] Restore is independent of network and training-only initialization files.
- [ ] Direct and exported `control_action` parity is demonstrated.
- [ ] A fixture + SHA + Policy commit is handed to Real.
- [ ] README/AGENTS no longer contain the identified stale deployment/state-dimension claims.
