# DexMani Policy — Canonical Deployment Cleanup Guide

> Repository: `haoyangzhanglab/dexmani_policy`  
> Reviewed code baseline: `c212b5a9820f4e30eacbdd975748b1d4d7f5fc47`  
> Intended executor: Claude Code / Codex  
> Scope: personal PhD robot-learning research. Prefer the smallest change that makes training/sim/deployment semantics explicit and consistent. Do not refactor working boundaries merely to make them look smaller.

---

## 1. Goal

The canonical Policy inference path is already correct:

```text
raw observation
    ↓
Policy-owned preprocessing
    ↓
agent.predict_action(...)
    ↓
validate_prediction(...)
    ↓
snapshot.control_action
    ↓
finite float64 [N, D_control]
```

This task removes one obsolete generic concept — `temporal_ensemble_coeff` — from the active Policy pipeline while preserving everything else that already has clear research value.

Required outcomes:

1. Generic sim/deployment always executes canonical `control_action` directly.
2. `temporal_ensemble_coeff` disappears from new configs, eval APIs/results, checkpoint-selection writes, deployment artifacts, runtime state, and current docs.
3. Deployment artifact schema advances once to v4 because its persisted contract changes.
4. Existing expensive `best_ckpt.json` selection records remain usable **only when their old coefficient is null**; non-null historical records are rejected because they encode different evaluation semantics.
5. Current strict Git provenance, normalizer/preprocessing checks, `PolicySpec` fields, restore logic, and qualification logic remain intact except where they directly reference the removed field.

Do not add a replacement temporal mechanism.

---

## 2. Preserve these invariants

### Runtime

`LoadedPolicy.predict()` must remain equivalent to:

```python
result = restored.agent.predict_action(
    tensors,
    denoise_timesteps=restored.spec.denoise_steps,
)
snapshot = validate_prediction(result, restored.spec, batch_size=1)
return snapshot.control_action[0]  # copied to finite float64 NumPy [N,D]
```

It must not depend on:

```text
pred_action tail
previous chunks
overlap state
post-hoc smoothing
RTC history
```

### Sim runner

Until a future explicit experiment changes it:

```text
obs history
→ predict_action()
→ canonical control_action [N,D]
→ env.step() over N actions in order
→ next query
```

Do not add `steps_per_inference`, chunk merging, or overlapping replanning in this task.

### Restore / preprocessing

Preserve:

```text
EMA/raw checkpoint selection
denoise_steps
normalizer validation
RGB preprocessing reproduction
point-cloud/fingertip semantic validation
prediction parity / qualification
```

These protect real train/deploy consistency and are not production-only ceremony.

---

## 3. Do not slim `PolicySpec` in this task

Current `PolicySpec` exposes more metadata than Real presently consumes, including fields such as:

```text
action_dim
horizon
requires_hand
rgb_preprocessing
```

That is not currently a correctness problem and removing them would enlarge the cross-repository patch for little benefit.

Required change:

```text
remove PolicySpec.temporal_ensemble_coeff
```

Otherwise preserve the public `PolicySpec` shape and validation.

Real must still treat model-specific details as Policy-owned; that ownership rule does not require deleting harmless read-only metadata now.

---

## 4. Remove generic temporal semantics end-to-end

Current active references span at least:

```text
dexmani_policy/deployment/contract.py
dexmani_policy/deployment/export.py
dexmani_policy/deployment/runtime.py
dexmani_policy/deployment/qualify.py

dexmani_policy/env_runner/base_runner.py
dexmani_policy/env_runner/sim_runner.py
dexmani_policy/env_runner/multi_task_sim_runner.py

dexmani_policy/select_best_ckpt.py
dexmani_policy/eval_best_ckpt.py
dexmani_policy/record_demo.py
dexmani_policy/training/eval_utils.py

dexmani_policy/configs/*.yaml
README.md
docs/仿真评测机制.md
docs/项目架构.md
```

Search the tree before editing; do not rely on this list being exhaustive.

### 4.1 Env runners and configs

Remove `temporal_ensemble_coeff` from runner constructor signatures, `super(...)` forwarding, Hydra configs, and generic rejection guards.

Preserve exactly:

```python
result = agent.predict_action(...)
return result["control_action"]
```

Do not introduce another action postprocessor.

### 4.2 Eval / demo APIs and result metadata

Remove the unused coefficient argument/value propagation from:

```text
eval_best_ckpt.py
record_demo.py
related shell/config plumbing found by search
```

Do not write the field into new `result_details.json` or demo metadata when it has no runtime effect.

Historical result files are data; do not rewrite them.

---

## 5. `best_ckpt.json`: new v3, narrow v2 compatibility

Checkpoint selection can be expensive, so do not force valid historical null-coefficient selections to be rerun.

### New writes

`select_best_ckpt.py` should write:

```json
{
  "record_version": 3,
  "inference": {
    "use_ema": true,
    "denoise_steps": 10,
    "policy_seed_mode": "episode_seed"
  }
}
```

plus the existing checkpoint/selection/result fields.

### Reader

`read_best_ckpt_json()` should support only:

```text
v3:
    no temporal field

v2 legacy:
    temporal_ensemble_coeff must exist and be null
    normalize/ignore that obsolete null field
```

If a v2 record contains a non-null coefficient, reject it with an explicit message such as:

```text
This best_ckpt.json was selected under obsolete temporal-ensemble semantics.
Re-run checkpoint selection under the current canonical control_action semantics.
```

This is the only intentional compatibility shim in this task. Do not add a general record migration framework.

Update all callers to consume only:

```text
use_ema
denoise_steps
policy_seed_mode
```

from the normalized inference record.

---

## 6. Deployment artifact: clean v4

The deployment artifact is derived from an experiment/checkpoint and is cheap to regenerate compared with checkpoint selection. Prefer a clean artifact contract rather than carrying v3 compatibility indefinitely.

### Required

Advance:

```text
DEPLOYMENT_FORMAT        -> dexmani.deployment.v4
DEPLOYMENT_SCHEMA_VERSION -> 4
```

Remove `temporal_ensemble_coeff` from:

```text
DeploymentSpec
parse_deployment_contract()
export inference payload
runtime PolicySpec construction
runtime generic temporal guard
qualification field comparisons
```

The artifact still keeps all internal information required to restore the model, including where applicable:

```text
action_dim
horizon
denoise_steps
rgb_preprocessing
agent config
weights
normalizer-related state
```

Do not conflate:

```text
artifact restore metadata
```

with:

```text
Real-facing runtime semantics
```

### Legacy artifact policy

Do not teach the new runtime to load v3 deployment artifacts. Re-export old experiments to produce v4 artifacts.

Because the v2 `best_ckpt.json` null case remains readable, this re-export should not require repeating checkpoint selection.

---

## 7. Preserve Git provenance exactly unless a real bug is found

Do **not** relax the current exporter provenance rules as part of this cleanup.

Current checks for:

```text
repository root
valid 40-hex HEAD
clean working tree
expected origin
```

are low-cost research reproducibility guards: the recorded commit only identifies the exact exporting code when the tree is clean.

Do not add dirty-tree export, provenance flags, signing, registries, or remote abstraction in this task.

If the executor discovers an independent provenance bug, report it separately rather than bundling a redesign into this patch.

---

## 8. Preserve observation/data semantic checks

Do not simplify point-cloud/fingertip/RGB contracts just because this task removes an unrelated temporal field.

In particular preserve the checks that prevent:

```text
wrong coordinate frame
wrong units
wrong xyzrgb preprocessing
wrong point-cloud config identity
wrong fingertip geometry/order
wrong RGB preprocessing
```

No observation/data schema migration belongs in this task.

---

## 9. Expected files

Search first, then minimize the final patch. Expected production files are primarily:

```text
dexmani_policy/deployment/contract.py
dexmani_policy/deployment/export.py
dexmani_policy/deployment/runtime.py
dexmani_policy/deployment/qualify.py

dexmani_policy/env_runner/base_runner.py
dexmani_policy/env_runner/sim_runner.py
dexmani_policy/env_runner/multi_task_sim_runner.py

dexmani_policy/select_best_ckpt.py
dexmani_policy/eval_best_ckpt.py
dexmani_policy/record_demo.py
dexmani_policy/training/eval_utils.py

dexmani_policy/configs/*.yaml
README.md
docs/仿真评测机制.md
docs/项目架构.md
```

Do not modify agents, losses, samplers, datasets, normalizers, or training code unless a direct compile/call-site dependency is demonstrated.

---

## 10. Recommended implementation order

### Phase A — sim/eval metadata cleanup

1. Remove env-runner constructor/config field.
2. Remove eval/demo argument and result propagation.
3. Write `best_ckpt.json` v3.
4. Implement the narrow v2-null reader path.
5. Verify canonical sim chunk execution is unchanged.

### Phase B — deployment artifact v4

1. Remove coefficient from internal deployment spec and export payload.
2. Bump artifact v3 → v4.
3. Remove runtime/qualification temporal guards/fields.
4. Keep all unrelated `PolicySpec`, restore, preprocessing, and provenance behavior unchanged.

### Phase C — docs + lightweight regression

1. Update README and durable docs to implemented behavior.
2. Add focused offline checks.
3. Run final stale-reference search.

Do not combine a new inference algorithm with any phase.

---

## 11. Lightweight regression coverage

Neither this repository nor its `pyproject.toml` currently establishes pytest as a required test framework. Do not add pytest only for this task.

Prefer a small standard-library `unittest` module, for example:

```text
tests/test_deployment_contract.py
```

run with:

```bash
python -m unittest discover -s tests -p 'test_deployment_contract.py'
```

Alternatively, if the executor finds an existing repository convention better suited to lightweight contract checks, use it and document the command.

Required checks:

### A. Canonical runtime output

Use a tiny fake restored agent exposing both `pred_action` and `control_action`; assert exact equality between `LoadedPolicy.predict()` and validated `control_action[0]`.

### B. Best-checkpoint records

Verify:

```text
v3 record -> accepted
v2 + null coefficient -> accepted/normalized
v2 + non-null coefficient -> rejected
```

### C. Artifact v4

Verify a minimal v4 contract parses and the obsolete temporal field is neither required nor exposed.

### D. Runtime spec

Verify `PolicySpec` no longer has `temporal_ensemble_coeff`, while unrelated existing fields remain available.

No large checkpoint, dataset, simulator, or GPU is required for these checks.

---

## 12. Cross-repository handoff

After this Policy patch:

```text
dexmani_real
```

must remove its `PolicySpec.temporal_ensemble_coeff` compatibility check before using the new runtime.

Do not require Real to parse artifact v4 directly. Real continues to consume the public `dexmani_policy.deployment` API.

Recommended order:

```text
1. finish + offline-verify Policy cleanup
2. update Real public-contract validation
3. re-export one representative experiment to v4
4. run Policy check / Real shadow before physical execution
```

---

## 13. Explicit non-goals

Do not implement or redesign:

```text
Temporal Ensemble
ACT overlap aggregation
ChunkOverlapBlender replacement
RTC
previous-action conditioning
steps_per_inference
chunk overlap/merge
action smoothing
adaptive horizon
latency compensation
sampler / solver
Flow Matching objective/distillation
model architecture
training objective
normalizer semantics
observation/data schema
PolicySpec slimming beyond the removed field
Git provenance policy
artifact migration framework
```

---

## 14. Validation

Before editing:

```bash
git status --short
git rev-parse HEAD
rg -n "temporal_ensemble_coeff|ChunkOverlapBlender|temporal_ensembler" \
  dexmani_policy README.md docs scripts
```

After editing:

```bash
python -m compileall -q dexmani_policy
python -m unittest discover -s tests -p 'test_deployment_contract.py'

git diff --check
git diff --stat
rg -n "temporal_ensemble_coeff|ChunkOverlapBlender|temporal_ensembler" \
  dexmani_policy README.md docs scripts
```

The final search may contain the intentionally narrow v2 `best_ckpt.json` compatibility check and this guide. It must not show a current sim/runtime/deployment feature or config option.

Do not run unless separately authorized:

```text
training
DDP/NCCL
large checkpoint sweeps
full simulator benchmark
real robot code
```

---

## 15. Definition of Done

All must hold:

```text
new sim/config/eval path
→ no temporal ensemble option
→ canonical control_action only
```

```text
best_ckpt.json v3
→ no temporal field
v2 + null
→ still reusable
v2 + non-null
→ explicit rejection
```

```text
deployment artifact
→ v4
→ no temporal field
→ deterministic restore semantics unchanged
```

```text
LoadedPolicy.predict()
→ exact validated control_action
```

and:

- strict Git provenance unchanged;
- existing `PolicySpec` fields unchanged except removal of temporal coefficient;
- observation/preprocessing/normalizer contracts unchanged;
- compile and lightweight tests pass;
- durable docs match actual code;
- no new action aggregation/replanning mechanism is introduced.

---

## 16. Final Claude Code / Codex report

Report:

### Changed
- exact files;
- `best_ckpt.json` version behavior;
- deployment artifact version;
- removed temporal field/call paths.

### Preserved
Explicitly confirm:

```text
canonical control_action
sim sequential chunk execution
PolicySpec unrelated fields
normalizer/preprocessing semantics
strict provenance
```

### Verified
- compile command;
- lightweight regression command;
- `git diff --check`;
- stale-reference search.

### Not Verified
Explicitly list unrun items such as GPU checkpoint restore, full simulator evaluation, training, and real robot deployment.

### Remaining risk
Only concrete code/test risks. Do not propose temporal smoothing, RTC, or a new serving framework as generic follow-up work.
