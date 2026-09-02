# DexMani Policy → Real Deployment Implementation Plan

> **Audience**: Codex / repository maintainers  
> **Scope**: `dexmani_policy` only.  
> **Execution order**: complete the Policy handoff in this document before starting semantic changes in `dexmani_real`.  
> **Reviewed baseline**: `main` at `a9de8b9b8c082edc7192b5a5bf7ffaf91a7f252a` (2026-08-31).  
> **Review status**: v2 — cross-checked against current Policy source and the current Real deployment-v2 consumer.  
> **Rule**: before every PR, re-read current `HEAD`, `AGENTS.md`, and touched source. If `main` moved, re-evaluate affected facts instead of applying this plan mechanically.

---

## 1. Goal

Make an ordinary Policy experiment produce a **self-contained, deterministic, machine-verifiable deployment artifact** that `dexmani_real` can consume without knowing training internals.

```text
resolved Policy config + Real Policy Zarr v5
        ↓
simple.v1 training checkpoint
        ↓
Policy-native deployment exporter
        ↓
dexmani.deployment.v2 checkpoint
+ schema-v2 sidecar
+ deployment_latest.pt selector
        ↓
strict / no-network restore in dexmani_real
```

The exporter is a narrow producer boundary. Do not create a registry/factory/plugin framework.

---

## 2. Non-goals

Do not:

- modify `dexmani_real` runtime in Policy PRs;
- invent `dexmani.deployment.v3`;
- make Real read `simple.v1` directly;
- export optimizer/scheduler/workspace/dataset/env-runner state;
- add a global Policy builder registry;
- change model architecture/solver/NFE/model quality for deployment convenience;
- claim RGB/text deployment support before their contracts exist;
- make `dexmani_policy` runtime-depend on `dexmani_real`;
- commit large real checkpoints or datasets as integration fixtures;
- run real-hardware commands, long training, DDP, or long simulation evaluation from Codex.

---

## 3. Cross-repo ownership

### Policy owns

- Hydra agent/model construction semantics;
- `predict_action()` output semantics;
- `action_key`, model `action_dim`, `control_action_dim`;
- `horizon`, `n_obs_steps`, `n_action_steps`;
- normalizer state;
- model/EMA selection semantics;
- Diffusion/Flow solver and NFE;
- deployment-safe constructor sanitization;
- deployment-v2 checkpoint and sidecar production.

### Real owns

- artifact no-follow/identity/SHA/TOCTOU/provenance checks;
- causal observations and run generations;
- action timestamps, plan deadline and stale dropping;
- EE→IK;
- workspace/collision/delta/joint-limit SafetyGate;
- arm/hand coupled publication, ticket, ACK and SDK IO.

Policy must not import Real in the exporter runtime. Cross-repo compatibility belongs in tests/handoff, not a production dependency.

---

## 4. `predict_action()` deployment contract

Every deployable policy must return:

```python
{
    "pred_action": Tensor[B, horizon, model_action_dim],
    "control_action": Tensor[B, n_action_steps, control_action_dim],
    ... optional model-specific diagnostics ...,
}
```

`tail` is optional and is not a deployment contract.

Canonical executable slice:

```python
start = n_obs_steps - 1
expected_control = pred_action[
    :,
    start:start + n_action_steps,
    :control_action_dim,
]
```

For supported policies, exporter verification must prove `control_action` matches this semantic slice for a deterministic test input/seed.

---

## 5. Current correctness issues

1. **ActionFlow EE state mismatch**
   - current `state_dim: ${action_dim}`;
   - `joint_state` is always arm7 + hand12 = 19;
   - `action_ee` is 21-D;
   - strict `ActionFlowObsEncoder.forward()` correctly rejects the mismatch.

2. **Checkpoint metadata missing `use_aux_ee`**
   - Real validates data/train/inference auxiliary-action semantics.

3. **`training/eval_utils.py` `raw_state` scope bug**
   - `raw_state` is currently assigned only under `train_params is not None`.

4. **Stale repository guidance**
   - `AGENTS.md` incorrectly says `joint_state` dimension equals action dimension.

5. **DQ-RISE checkpoint-selection commentary is stale**
   - current trainer saves milestone/interrupt checkpoints with `monitor={}`;
   - deployment-quality “best” comes from offline evaluation output, not online `val_loss` top-k.

6. **Dependency description is inconsistent**
   - `pyproject.toml`: `hydra-core>=1.3`;
   - `requirements.txt`: `hydra-core==1.2.0`;
   - editable install does not cover every strategy dependency.

7. **No official deployment-v2 producer exists.**

8. **Real Zarr root semantic attrs are not preserved by Policy ReplayBuffer copy logic.**

9. **DQ-RISE / R3D fresh construction can depend on training-time assets before restore.**

10. **RGB evaluation preprocessing differs from current Real raw-RGB handoff.**

---

# Phase P0 — Correctness before exporter

**Status**: TODO  
**Primary hypothesis**: current training/evaluation semantics can be corrected without introducing deployment behavior.

## P0.1 ActionFlow state contract

Change:

```yaml
# dexmani_policy/configs/action_flow.yaml
state_dim: 19
```

Do not slice/pad state in the encoder.

Update examples/comments that imply `state_dim == action_dim`.

Acceptance:

```text
action_key=action     → action_dim=19, state_dim=19
action_key=action_ee  → action_dim=21, state_dim=19
```

Tests must resolve both the single-GPU config and `ddp/action_flow` inheritance.

## P0.2 `train_params`

Add to `build_train_params()`:

```python
"use_aux_ee": bool(getattr(model, "use_aux_ee", False)),
```

Keep `simple.v1` top-level format unchanged.

Expected new metadata includes at least:

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

## P0.3 Fix evaluation checkpoint loading

Weight selection must be independent of metadata presence:

```python
raw_state = checkpoint.model_state
if use_ema:
    if checkpoint.ema_model_state is None:
        # simulation evaluation may preserve the existing warning fallback
        warn(...)
    else:
        raw_state = checkpoint.ema_model_state
```

Deployment export is stricter: `use_ema=true` with no EMA state is an error.

## P0.4 Correct checkpoint-selection documentation

Do not redesign training.

Document:

```text
training saves milestone/interrupt checkpoints;
offline evaluation produces best_ckpt.json / best.pt evidence.
```

Also remove/move unreferenced empirical claims from runtime configs (for example “verified +2.9pp across 7 tasks”) unless a stable experiment record is linked. Runtime configs should state behavior, not unsupported historical claims.

## P0.5 Repository guidance

Fix `AGENTS.md`:

```text
joint_state = 19-D (arm7 + hand12)
action = 19-D
action_ee = 21-D
```

If `pip install -e .` is not a complete all-strategy environment, say so explicitly.

## P0 validation

Minimum:

```bash
python -m compileall -q dexmani_policy
python dexmani_policy/smoke_test.py action_flow
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py maniflow
python dexmani_policy/smoke_test.py sat
git diff --check
```

Add focused config tests for ActionFlow joint/EE and DDP resolution. Report unavailable CUDA/data/dependencies; do not “fix” code to hide environment failures.

---

# Phase P1 — Minimal deployment-v2 exporter

**Status**: BLOCKED on P0  
**Primary hypothesis**: Policy can produce the existing Real contract without adding a cross-repo runtime dependency.

## P1.1 Minimal package

Add only:

```text
dexmani_policy/deployment/
├── __init__.py
└── export.py
```

Suggested API:

```python
export_deployment_artifact(
    experiment_dir: Path,
    checkpoint_selector: str = "best",
    output_path: Path | None = None,
    verify: bool = True,
    zarr_path: Path | None = None,
) -> ExportReceipt
```

CLI:

```bash
python -m dexmani_policy.deployment.export EXP --checkpoint best --verify
```

Do not add a registry/factory.

## P1.2 Strict checkpoint resolution

For deployment, selector meaning must be literal.

`--checkpoint best`:

```text
best_ckpt.json
→ best.pt
→ error
```

**Do not silently fall back from `best` to `latest`.** If the operator wants latest, they must request `--checkpoint latest` explicitly.

Explicit paths/milestone tags are allowed if existing repository utilities resolve them deterministically.

## P1.3 Source provenance gate

Current Real restore requires the installed Policy package to match the artifact producer commit and to be clean. Therefore the first-phase exporter must fail early unless it can establish:

```text
repository == haoyangzhanglab/dexmani_policy
HEAD is a 40-hex commit
working tree is clean
```

Record the current clean exporter/model-source `HEAD` as `producer.commit`.

Do not produce an artifact that current Real will necessarily reject later.

A future “dirty research source” workflow is a separate contract change; do not smuggle it into v2 export.

## P1.4 Resolve and read Real Policy Zarr v5

Exporter must reopen the original Zarr and read root attrs directly:

```python
root = zarr.open_group(str(resolved_zarr_path), mode="r")
attrs = dict(root.attrs)
```

Relative `cfg.zarr_path` must **not** depend on the caller's current working directory. Resolve it using the Policy repository/training path convention. If data has been relocated, allow an explicit `--zarr-path` override, but require exact task/schema/semantic agreement.

Reject simulation data.

Validate current Real v5 semantics, including:

```text
schema_name == dexmani-real-policy-zarr
schema_version == 5
domain == real
profile
task_name
dt
episode_start_policy
obs_alignment
observation_reference
state_alignment
max_observation_skew_s
action_semantics
arm/hand delta semantics
endpoint_delta_tolerance_rad
deployment_equivalent
point-cloud frame/config/sampling/transform semantics when used
```

Build deployment `sensor_modalities` from the **model/dataset modality subset**, not merely from every array present in an `rgb_pc` Zarr.

## P1.5 Minimal resolved inference config

Embed only model restore/inference values:

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

No dataset/env_runner/workspace target may enter the deployment config.

Current Real only consumes a positive integer `eval.denoise_steps`. First-phase exporter must reject unsupported inference overrides, including a non-null `eval.denoise_timesteps_list`, rather than silently changing inference semantics.

## P1.6 Deployment-safe constructor sanitization

Use narrow field-based changes, not a policy-name registry.

### DQ-RISE

Export:

```yaml
agent:
  codebook_path: null
```

First-phase support requires the selected checkpoint itself to contain the persistent runtime codebook buffers. If an old checkpoint only works because an external `.npz` is loaded before state restore, reject it with a clear migration/retrain message. Do not silently retrofit tensor state from an external file into deployment-v2.

Preserve/check:

```text
tcp_dim
hand_dim
codebook_num_groups
codebook_size
action_key
```

Current default DQ-RISE is `action_ee` and therefore produces an EE artifact.

### R3D / Uni3D

If present, export:

```yaml
agent:
  pc_encoder_config:
    use_pretrained_weights: false
```

Then rely on strict checkpoint restore. Tests must prove this is topology-preserving.

### RGB backbones

Reject in P1. Do not add DINO/R3M download hacks.

## P1.7 Exact deployment-v2 payload

The current Real reader requires exact top-level/state/weight key sets.

Important weight semantics:

```text
weights.model      → ALWAYS a non-empty canonical model state_dict
weights.ema_model  → canonical EMA state_dict or null
```

Do **not** replace `weights.model` with EMA when `eval.use_ema=true`. Real chooses the selected state at restore time. If `eval.use_ema=true`, `weights.ema_model` must exist or export fails.

State dict keys must be canonical: no `module.` and no `_orig_mod.`.

Metadata must be plain finite JSON-compatible values.

## P1.8 Producer and embedded deployment receipt

Current Real distinguishes checkpoint producer metadata from the smaller sidecar producer object.

Checkpoint `producer` must include the fields required by the Real receipt, including:

```text
repository
commit
metadata_provenance
retrofitted_train_params_fields
```

`retrofitted_train_params_fields` must always be a list, including `[]` for native metadata.

`deployment_contract` must use the exact current Real schema and carry the same retrofit list.

Rules:

- `metadata_provenance="native"` only when required training metadata already existed and matched;
- `metadata_provenance="retrofitted"` only for explicitly whitelisted metadata synthesis (for example old `use_aux_ee` metadata);
- never call tensor-weight mutation a metadata retrofit.

The schema-v2 **sidecar producer** remains the exact smaller object accepted by Real:

```text
repository
commit
metadata_provenance
```

Do not add extra sidecar producer keys without a consumer change.

## P1.9 Sidecar schema-v2

New exports always write current schema v2.

Generate against the actual current Real parser/golden test; do not rely on memory. This is a test-time compatibility check and must not introduce a Policy runtime dependency on Real.

Keep current compatibility field:

```text
required_action_steps = horizon - (n_obs_steps - 1)
```

It remains the serialized prediction-future/allocation length. It is **not** `n_action_steps`.

## P1.10 Atomic publication

Recommended order:

```text
validate all inputs first
→ write checkpoint staging file
→ fsync
→ atomic replace final checkpoint
→ compute SHA-256
→ write canonical sidecar staging file
→ fsync + atomic replace
→ atomically update relative deployment_latest.pt symlink
→ roundtrip verify selected final files
```

A failure may leave an unselected orphan checkpoint, but must never leave a selector pointing at an incomplete/incompatible artifact.

Do not overwrite an existing deployment artifact unless an explicit `--force` policy is implemented and itself remains atomic.

## P1 validation

Must prove:

- exact deployment-v2 payload schema;
- exact sidecar-v2 parser compatibility;
- exact checkpoint producer/deployment-contract receipt;
- strict state restore;
- normalizer completeness;
- `pred_action` / `control_action` shapes and finite values;
- unsupported sim/RGB/text/custom timestep-list cases fail before selector publication;
- interrupted export never selects a partial artifact.

---

# Phase P2 — Self-contained restore and parity

**Status**: BLOCKED on P1

## P2.1 No-network

During deployment verification, forbid/patch network-backed loaders. Restore must not need:

- Hugging Face download;
- gdown/Google Drive;
- external pretrained fetch;
- experiment-local Python package shadowing.

## P2.2 No training-only initialization files

For each supported strategy:

1. export;
2. remove/rename training-only initialization asset in the fixture environment;
3. instantiate from exported config;
4. strict restore;
5. predict.

At minimum cover DQ-RISE and R3D.

## P2.3 Direct/export parity

For one deterministic synthetic observation and seed, compare:

```text
direct experiment restore
vs
exported deployment-v2 restore
```

Compare both:

```text
pred_action
control_action
```

Use exact equality where deterministic dtype/implementation permits; otherwise use a narrow, justified tolerance. Never loosen tolerance merely to make the test pass.

## P2.4 First-phase support matrix

| Policy | Status | Qualification |
|---|---|---|
| DP3 | target supported | point cloud + joint state |
| ManiFlow | target supported | preserve solver/NFE |
| SAT | target supported | point-cloud path |
| ActionFlow | after P0 | state_dim=19; preserve midpoint/NFE |
| DQ-RISE | conditional | checkpoint must be self-contained; current default is EE action |
| R3D | conditional | pretrained init disabled; aux-EE full/control split |
| DP RGB | deferred | preprocessing mismatch |
| MultiTask DiT | deferred | explicit task-text contract missing |

Do not label deferred policies “structurally supported”.

---

# Phase P3 — Cross-repo handoff

**Status**: BLOCKED on P2  
**This is the gate before Real control-semantic changes.**

Deliver:

```text
Policy commit SHA
clean-source/provenance result
export command/version
representative deployment-v2 artifact or deterministic fixture generator
artifact SHA-256
schema-v2 sidecar
supported-policy matrix
strict-restore result
direct/export control_action parity result
no-network/no-external-file result
```

### Fixture rule

Do **not** commit a large real DP3/DQ/R3D checkpoint to Git.

Preferred options:

1. deterministic tiny/synthetic fixture generator committed as code;
2. externally stored/local integration artifact referenced by SHA-256;
3. CI-generated artifact.

A real representative checkpoint may be used locally for qualification but remains outside source control.

The handoff must state explicitly:

```text
prediction future steps != executable control steps
Real executes only n_action_steps control_action
```

After handoff acceptance, move to:

```text
dexmani_real/docs/policy_deployment_refactor_plan.md
```

Do not continue changing both repositories in parallel unless a newly discovered producer-contract bug requires returning to Policy.

---

# Phase P4 — Deferred RGB/text

## RGB prerequisites

Artifact must explicitly encode deterministic evaluation preprocessing, e.g. resize + center crop + interpolation + color/value convention. Deployment reproduces **evaluation**, not random training augmentation.

Also require no-network topology construction.

## Text prerequisite

MultiTask deployment requires explicit task-text conditioning semantics. Never infer task text from experiment directory names.

---

## 6. Dependency and long-lived documentation cleanup

Do after exporter works in the real `policy` environment.

- resolve Hydra version disagreement;
- document core vs strategy-specific dependencies;
- avoid broad Torch/Diffusers/Transformers upgrades in deployment PRs;
- README must explicitly state:

```text
training checkpoint != deployment artifact
```

User flow:

```text
train
→ offline evaluate/select checkpoint
→ export deployment-v2
→ Real inspect/check
→ shadow
```

---

## 7. Codex execution rules

Before each PR:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

Each PR:

- one primary semantic hypothesis;
- preserve unrelated user changes;
- no drive-by rename/reformat;
- focused regression tests;
- update authoritative docs/comments;
- report commands not run and why.

Stop instead of guessing when:

- current main materially diverges;
- Real parser/schema contradicts this plan;
- config/checkpoint/Zarr metadata conflict;
- strict restore has missing/unexpected keys;
- exporter cannot establish a clean producer commit;
- restore attempts network access;
- `control_action` differs from the declared slice;
- unsupported preprocessing/inference behavior would need a silent fallback.

---

## 8. Recommended Policy PR sequence

```text
Policy PR-1  correctness
    - ActionFlow state_dim=19
    - single/DDP joint+EE config tests
    - AGENTS correction
    - use_aux_ee metadata
    - eval raw_state fix
    - DQ-RISE checkpoint-selection/config-comment cleanup

Policy PR-2  deployment-v2 exporter
    - strict checkpoint selector
    - clean producer provenance
    - Zarr contract/path resolution
    - minimal resolved cfg.agent
    - exact checkpoint producer/receipt
    - v2 checkpoint + sidecar + selector

Policy PR-3  self-contained restore/parity
    - DQ-RISE
    - R3D
    - no-network
    - direct/export parity

Policy PR-4  cross-repo fixture/handoff
```

Do not start Real semantic PRs before Policy PR-4 handoff unless the user explicitly changes sequencing.

---

## 9. Definition of Done

Policy side is complete when:

- [ ] ActionFlow joint and EE configs both consume 19-D `joint_state`.
- [ ] DDP ActionFlow resolves the same invariant.
- [ ] New checkpoints record `use_aux_ee`.
- [ ] Evaluation load has no `raw_state` scope failure.
- [ ] `--checkpoint best` never silently selects latest.
- [ ] One command exports an ordinary experiment to exact deployment-v2 + schema-v2 sidecar.
- [ ] Exporter requires a clean, identifiable producer source under the current v2 contract.
- [ ] `weights.model` is always canonical and non-empty; EMA semantics match Real.
- [ ] Producer/deployment receipt retrofit metadata exactly matches Real expectations.
- [ ] Relative/relocated Zarr handling is deterministic.
- [ ] Unsupported sim/RGB/text/custom timestep-list cases fail explicitly.
- [ ] Supported point-cloud artifacts strict-restore without network/training-only initialization files.
- [ ] Direct/export `control_action` parity is demonstrated.
- [ ] Cross-repo fixture is reproducible without committing a large real checkpoint.
- [ ] SHA + Policy commit + qualification results are handed to Real.
