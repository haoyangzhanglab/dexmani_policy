# DexMani Policy — Canonical Deployment Repair Guide

> Repository: `haoyangzhanglab/dexmani_policy`  
> Reviewed baseline: `9be70a70e0f6939ac6da777062e900e2b1b216a6`  
> Intended executor: Claude Code / Codex  
> Scope: personal PhD robot-learning research. Prefer correct experiment semantics, minimal patches, reproducibility, and explicit failure over general framework design.

---

## 1. Goal

The generic Policy action path is already structurally correct:

```text
agent.predict_action(...)
        ↓
validate_prediction(...)
        ↓
snapshot.control_action
        ↓
[N, D_control]
```

`ChunkOverlapBlender` has already been removed from the generic runtime/sim path, and `common/temporal_ensembler.py` no longer exists.

This task has only three goals:

1. Fix the remaining **exporter/runtime capability mismatch** for `temporal_ensemble_coeff`.
2. Remove active documentation that still describes the deleted blender as current behavior.
3. Add minimal offline regression coverage for the canonical deployment contract.

Stop after these are complete. Do not extend into model/training/sampler/data redesign.

---

## 2. Verified current facts

### Runtime — preserve

`dexmani_policy/deployment/runtime.py` currently:

- rejects unsupported generic temporal semantics;
- has no blender state;
- returns `snapshot.control_action` directly;
- validates exact `[n_action_steps, control_action_dim]` shape and finite values;
- resets only RNG / agent-owned episode state.

Do not reintroduce:

```text
ChunkOverlapBlender
EMA chunk smoothing
timestamp-aware overlap blending
generic ACT temporal ensemble
previous-chunk caches
```

### Sim runner — preserve

`dexmani_policy/env_runner/base_runner.py` currently executes:

```text
predict_action()
→ result["control_action"]
→ env.step over returned actions in order
```

and rejects non-null `temporal_ensemble_coeff`.

Do not rewrite sim execution semantics.

### Confirmed bug — exporter accepts an unsupported value

Runtime/sim now require:

```text
temporal_ensemble_coeff == null
```

but `dexmani_policy/deployment/export.py` still accepts a finite non-negative coefficient while resolving `_SelectedInferenceSettings`.

A historical experiment can therefore do:

```text
config / best_ckpt.json
    temporal_ensemble_coeff = 0.01
        ↓
export accepts it
        ↓
artifact is produced far enough to look deployable
        ↓
load_experiment() rejects it
```

This producer/consumer mismatch is the P0 bug for this repository.

---

## 3. Required generic temporal contract

After the fix, every generic layer must agree on exactly one supported value:

```yaml
temporal_ensemble_coeff: null
```

Required chain:

```text
resolved config / selection record
        ↓
export inference-setting resolution
        ↓
deployment artifact
        ↓
LoadedPolicy
        ↓
sim eval / Real deployment
```

At every layer, non-null means **unsupported**, not “valid but optional”.

Keep the serialized field in this task. Do not bump artifact schema and do not create a migration layer.

---

## 4. P0 fix — reject non-null during export

### Primary file

```text
dexmani_policy/deployment/export.py
```

Trace before editing:

```text
config / best_ckpt.json
→ selected inference settings
→ _SelectedInferenceSettings
→ DeploymentSpec / payload
→ verification / publication
```

### Required behavior

At the **single inference-setting owner boundary**:

```python
if temporal_ensemble_coeff is not None:
    raise InvalidExperimentError(...)
```

Use wording consistent with runtime, for example:

```text
Generic chunk deployment does not support temporal_ensemble_coeff.
Re-export/configure this experiment with temporal_ensemble_coeff=null.
```

Do not merely validate finite/non-negative. A finite float is no longer a supported generic deployment capability.

### Placement rule

Prefer one check at the selected-inference-setting resolution point:

```text
read config / best selection
→ resolve use_ema / denoise_steps / coefficient
→ reject non-null coefficient
→ continue export
```

Do not duplicate the same check across artifact writing and verification functions.

### Preserve unrelated behavior

Do not modify:

```text
use_ema weight selection
denoise_steps
checkpoint resolution
git provenance
Policy Zarr validation
observation field semantics
normalizer restore
prediction parity verification
artifact version
```

---

## 5. Runtime/sim invariants to lock, not modify

### Runtime

`LoadedPolicy.predict()` must remain equivalent to:

```python
result = restored.agent.predict_action(...)
snapshot = validate_prediction(...)
return snapshot.control_action.squeeze(0)...
```

It must not depend on:

```text
pred_action tail
previous predictions
overlap state
post-hoc smoothing
```

### Sim

One query means one executable chunk:

```text
observation history
→ predict_action()
→ control_action [N,D]
→ env.step × N
→ next query
```

### Config

Current generic configs should remain explicit:

```yaml
env_runner:
  temporal_ensemble_coeff: null
```

Do not remove the key as part of this repair.

---

## 6. Documentation repair

Update only factual stale sections.

### `README.md`

Remove statements equivalent to:

```text
non-null temporal_ensemble_coeff enables ChunkOverlapBlender at runtime
```

Replace with the current contract: generic Diffusion/Flow deployment and sim eval execute canonical `control_action` directly and require a null temporal coefficient.

### `docs/仿真评测机制.md`

Remove current-mechanism descriptions of:

```text
ACT temporal aggregation
ChunkOverlapBlender
overlap tail/head fusion
```

Describe actual execution:

```text
obs history
→ predict_action()
→ canonical control_action [N,D]
→ env.step × N
→ next observation / query
```

### `docs/项目架构.md`

Remove stale references to:

```text
common/temporal_ensembler.py
ChunkOverlapBlender
runtime episode-local blender state
```

Do not add speculative RTC/ACT extension architecture.

### Search gate

After edits, search the active tree for:

```text
ChunkOverlapBlender
temporal_ensembler.py
```

Historical records need not be rewritten, but active architecture/runtime docs must not describe the deleted mechanism as current.

---

## 7. Minimal focused regression coverage

This is a personal research repository. Do not restore the old broad suite.

Prefer one focused file:

```text
tests/test_deployment_runtime_contract.py
```

If the repository intentionally avoids `tests/`, use one equivalently small deterministic offline contract script. A repeatable regression proof is required.

### Test A — canonical runtime output

Use a tiny fake/restored agent that exposes both:

```text
pred_action    [B,H,D]
control_action [B,N,D_control]
```

Assert exact equality:

```text
LoadedPolicy.predict(obs)
== validated control_action[0]
```

Do not only check shape/dtype.

### Test B — runtime rejects non-null coefficient

```text
PolicySpec.temporal_ensemble_coeff = 0.01
→ runtime/load guard rejects
```

### Test C — exporter rejects non-null coefficient

Cover the unique inference-setting owner and, if easy, both sources:

```text
resolved config env_runner.temporal_ensemble_coeff = 0.01
best_ckpt.json inference.temporal_ensemble_coeff = 0.01
```

Expected:

```text
export fails before artifact publication
```

### Test D — null remains valid

```text
temporal_ensemble_coeff = None
→ inference settings resolve normally
```

No large real model is needed.

---

## 8. Cross-repository contract

`dexmani_real` independently rejects:

```text
PolicySpec.temporal_ensemble_coeff != None
```

Keep that Real guard.

After this repair, behavior must agree across:

```text
Policy config/selection
Policy export
Policy load
Policy sim runner
Real compatibility validation
```

Do not weaken Real to accept historical non-null artifacts.

---

## 9. Expected patch scope

Production edits should normally be limited to:

```text
dexmani_policy/deployment/export.py
README.md
docs/仿真评测机制.md
docs/项目架构.md
```

plus one focused regression file/script.

If `runtime.py` or `base_runner.py` appears to require substantive modification, stop and re-audit: those paths already implement the desired canonical behavior.

---

## 10. Explicit non-goals

Do not implement or redesign:

```text
ACT Temporal Ensemble
ChunkOverlapBlender replacement
EMA action smoothing
RTC / previous-action conditioning
adaptive n_action_steps
adaptive horizon
sampler/solver redesign
Flow Matching distillation
training objective
model architecture
normalizer semantics
Policy Zarr schema
deployment artifact schema/version
legacy artifact migration
```

---

## 11. Execution order

```text
1. git status --short; git rev-parse HEAD
2. verify HEAD and preserve unrelated user changes
3. trace selected inference settings in export.py
4. add one fail-fast at the owner boundary
5. add/run the focused regression test
6. verify runtime.py/base_runner.py semantics remain canonical
7. repair stale docs
8. search for stale blender references
9. run compile/test/diff checks
10. report verified vs unverified work
```

Do not change docs first and infer code behavior from them.

---

## 12. Offline validation

```bash
git status --short
git rev-parse HEAD

python -m compileall -q dexmani_policy
python -m pytest -q tests/test_deployment_runtime_contract.py

git diff --check
git diff --stat
git status --short
```

Adapt the pytest path only if the final focused test has a different name.

Do not run unless separately authorized:

```text
training
DDP/NCCL
large simulator rollouts
long checkpoint selection/evaluation
real robot code
```

---

## 13. Definition of Done

All must hold:

```text
non-null temporal_ensemble_coeff
→ rejected during export inference-setting resolution
→ no deployable artifact published
```

```text
LoadedPolicy.predict()
→ canonical snapshot.control_action only
```

```text
BaseRunner
→ canonical control_action only
```

Current generic configs remain null.

Active docs no longer describe `ChunkOverlapBlender` / `temporal_ensembler.py` as current functionality.

Focused offline regression covers:

```text
canonical output
runtime rejection
export rejection
null accepted
```

No temporal framework, compatibility adapter, or artifact migration is introduced.

---

## 14. Final report format

Claude Code / Codex should report only:

### Changed
- files changed;
- exact contract fixed;
- intentionally preserved runtime/sim behavior.

### Verified
- compileall result;
- focused test result;
- `git diff --check` result.

### Not Verified
Explicitly list unrun items, especially:

```text
GPU checkpoint inference
full simulator rollout
training
real robot
```

### Remaining risk
If none is found in scope:

```text
No remaining generic temporal-semantic producer/consumer mismatch was found in the reviewed path.
```

Do not propose or implement RTC/smoothing as a follow-up to this task.
