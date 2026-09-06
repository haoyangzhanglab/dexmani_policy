# DexMani Policy — Deployment Semantics Simplification Guide

> Repository: `haoyangzhanglab/dexmani_policy`  
> Reviewed baseline: `c212b5a9820f4e30eacbdd975748b1d4d7f5fc47`  
> Intended executor: Claude Code / Codex  
> Scope: personal PhD robot-learning research. Optimize for correct experiment semantics, a small public contract, fast iteration, and deterministic offline verification. Do **not** turn this repository into a backward-compatible deployment framework.

---

## 1. Goal

The canonical inference path is already correct and must remain the center of the design:

```text
raw Policy observation
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

This task simplifies the surrounding deployment contract. It does **not** change the model, training objective, sampler, action representation, or canonical `control_action` semantics.

Required outcomes:

1. Remove the obsolete generic `temporal_ensemble_coeff` concept from active configs, eval, selection, export, artifact, runtime, and docs.
2. Bump the deployment artifact to a clean v4 schema; do not add legacy migration logic.
3. Keep model/restore details inside Policy and expose a lean `PolicySpec` to Real.
4. Preserve exact canonical `control_action` execution in both sim and deployment runtime.
5. Relax Git provenance from an iteration-blocking gate to recorded metadata while retaining useful provenance.
6. Add only small deterministic offline tests that lock these semantics.

The target is a standard robot-learning Policy runtime, not a generic serving SDK.

---

## 2. Ownership boundary

### `dexmani_policy` owns

```text
model architecture
checkpoint restore
EMA weight selection
denoise / solver settings
normalizer
RGB/model preprocessing
training horizon and internal action layout
canonical control_action extraction
sim evaluation semantics
deployment artifact internals
```

### `dexmani_real` may depend only on

```text
load_experiment(...)
LoadedPolicy.spec
LoadedPolicy.predict(observation) -> [N, D_control]
LoadedPolicy.reset_episode()
LoadedPolicy.close()
```

Real must not parse artifact internals or know how Diffusion / Flow / RISE-style models produce the chunk.

---

## 3. Non-negotiable invariants

Do not change the following behavior while simplifying the contract.

### 3.1 Canonical runtime output

`LoadedPolicy.predict()` remains equivalent to:

```python
result = restored.agent.predict_action(
    tensors,
    denoise_timesteps=restored.spec.denoise_steps,
)
snapshot = validate_prediction(result, restored.spec, batch_size=1)
control_action = snapshot.control_action[0]
return finite_float64_numpy_copy(control_action)
```

It must not use:

```text
pred_action tail
previous action chunks
post-hoc smoothing
overlap blending
temporal ensemble state
RTC history
```

### 3.2 Canonical sim execution

Until a future experiment explicitly introduces `steps_per_inference`, one Policy query means one executable chunk:

```text
obs history
→ predict_action()
→ control_action [N,D]
→ env.step(action[0:N]) in order
→ next query
```

Do not introduce overlapping replanning in this task.

### 3.3 Observation preprocessing ownership

Raw observation validation is public; model preprocessing is private to Policy.

Real may know raw field shape/dtype/essential semantics. Real must not reproduce:

```text
ImageProcessor normalization
model resize/crop chain
normalizer internals
encoder-specific transforms
```

---

## 4. Verified current problems

The current tree still carries `temporal_ensemble_coeff` through many active layers even though generic runtime/sim reject non-null values. Search currently reaches at least:

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

configs / README / docs
```

This is dead configuration, not a supported research feature. Keeping a field whose only valid runtime value is `null` creates unnecessary producer/consumer checks and future ambiguity.

The exporter also currently requires a clean working tree and a specific origin URL. This is stronger than needed for a personal research workflow: provenance is useful, but local experimental changes should not be blocked solely because the tree is dirty.

---

## 5. Target public `PolicySpec`

After this task, the Real-facing `PolicySpec` should contain only information required to construct raw observations and execute the returned action chunk:

```python
@dataclass(frozen=True)
class PolicySpec:
    action_key: str                # "action" | "action_ee"
    control_action_dim: int        # 19 | 21
    n_obs_steps: int
    n_action_steps: int
    observation_fields: tuple[ObservationFieldSpec, ...]
    control_dt_s: float
```

### Keep `ObservationFieldSpec`

Each raw observation field still needs:

```text
name
shape
dtype
essential semantics
```

Essential semantics include information that can produce a real train/deploy mismatch, for example:

```text
coordinate frame
units
xyzrgb representation
point-cloud preprocessing/config identity
fingertip geometry identity
RGB raw value range / color order
```

Do not reduce the cross-repo contract to shape-only validation.

### Move out of public `PolicySpec`

The following may remain in the internal deployment/restore spec if needed, but should not be part of the Real-facing contract:

```text
action_dim
horizon
denoise_steps
requires_hand
rgb_preprocessing
model preprocessing internals
```

`action_key` + `control_action_dim` already describe the supported Real control boundary; both supported action spaces include the 12-DOF hand.

Do not rename `action` / `action_ee` in this task. Avoid pointless cross-repository churn.

---

## 6. Deployment artifact v4

### 6.1 Bump schema once

Update:

```text
DEPLOYMENT_FORMAT / DEPLOYMENT_SCHEMA_VERSION
```

to a v4 contract.

Do not add runtime support for both v3 and v4. Existing v3 artifacts can be re-exported when needed.

### 6.2 Remove temporal ensemble from artifact

Remove `temporal_ensemble_coeff` from:

```text
DeploymentSpec
parse_deployment_contract()
inference_config.eval
export payload construction
qualification / parity metadata
runtime PolicySpec construction
runtime generic temporal guard
```

After the change there must be no artifact field that means “generic temporal blending”.

### 6.3 Preserve internal information actually required for restore

The v4 artifact may still contain internal restore fields such as:

```text
action_dim
horizon
denoise_steps
rgb_preprocessing
agent config / weights / normalizer-related state
```

Do not remove data merely because Real does not need to see it.

This distinction is important:

```text
artifact/restore contract  !=  public Real-facing PolicySpec
```

---

## 7. Remove `temporal_ensemble_coeff` from the full active pipeline

This removal must be end-to-end rather than replacing a float with more `None` checks.

### 7.1 Env runners

Remove the argument/config field from:

```text
BaseRunner
SimRunner
MultiTaskSimRunner
Hydra env_runner configs
```

Delete the generic temporal rejection constant/guard.

Preserve:

```python
result = agent.predict_action(...)
return result["control_action"]
```

and sequential `env.step()` execution.

### 7.2 Best-checkpoint selection record

`best_ckpt.json` currently writes `record_version=2` with:

```json
"inference": {
  "use_ema": ...,
  "denoise_steps": ...,
  "temporal_ensemble_coeff": ...,
  "policy_seed_mode": "episode_seed"
}
```

Change the selection schema to a clean v3 record:

```json
"inference": {
  "use_ema": ...,
  "denoise_steps": ...,
  "policy_seed_mode": "episode_seed"
}
```

Update writer and reader together:

```text
select_best_ckpt.py
training/eval_utils.py
eval_best_ckpt.py
record_demo.py
export.py
```

Do not add a permanent v2 compatibility adapter. Old selection records can be recreated/updated when an old experiment needs deployment.

Do not rerun expensive selection automatically as part of tests.

### 7.3 Export inference settings

`_SelectedInferenceSettings` should become only:

```python
@dataclass(frozen=True)
class _SelectedInferenceSettings:
    use_ema: bool
    denoise_steps: int
```

Resolve these two values from either `best_ckpt.json` or resolved config. Remove all coefficient validation and all requirements that `env_runner.temporal_ensemble_coeff` exist.

### 7.4 Runtime / qualification

Delete:

```text
_GENERIC_TEMPORAL_ENSEMBLE_ERROR
_require_generic_chunk_semantics(...)
PolicySpec.temporal_ensemble_coeff
qualification comparisons for this field
```

Do not replace them with a differently named generic blending flag.

### 7.5 Active config and docs search gate

After code changes, search the active repository for:

```text
temporal_ensemble_coeff
ChunkOverlapBlender
temporal_ensembler
```

Expected result:

```text
no active code/config/current architecture documentation references
```

Historical experiment artifacts or Git history do not need rewriting.

---

## 8. Provenance: record it, do not use it as an iteration gate

Keep useful Git provenance but relax the current export policy.

### Required behavior

Exporter should still establish:

```text
repository top level
40-hex HEAD commit
working-tree dirty/clean state
origin URL when available
```

The v4 producer metadata should record, at minimum:

```yaml
producer:
  repository: haoyangzhanglab/dexmani_policy
  commit: <40-hex HEAD>
  dirty: true | false
  origin: <string or null>
```

### Required policy

```text
dirty tree
→ warning + dirty=true
→ export continues
```

```text
origin URL differs from canonical GitHub URL
→ record it
→ do not reject solely for that reason
```

It is acceptable to keep failure for cases where the exporter cannot establish that it is running inside the expected repository root or cannot obtain a valid HEAD commit; those indicate an ambiguous artifact producer rather than ordinary research iteration.

Do not implement signing, supply-chain verification, remote allowlists, or artifact registries.

---

## 9. Point-cloud / fingertip contracts

Do not weaken geometry contracts while simplifying deployment.

For point cloud, preserve validation that prevents train/Real mismatch. Prefer a compact representation such as:

```text
shape / dtype
frame = xarm_base
representation = xyzrgb
config_sha256
```

Existing detailed fields may remain if removing them would require an unrelated data-schema migration. Do not expand them further.

For fingertip points, preserve enough metadata to guarantee:

```text
frame
units
finger ordering
geometry identity
```

The objective is “minimal sufficient semantics”, not “fewest strings possible”.

---

## 10. Files expected to change

The executor must first search the tree and then confirm the final list. Expected production files include:

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

relevant Hydra configs
README.md
docs/仿真评测机制.md
docs/项目架构.md
```

Plus a small deployment contract test file.

Do not modify model/agent implementation unless a compile/test failure proves a direct dependency.

---

## 11. Implementation phases

Keep commits/reasoning separable even if the final change is one branch.

### Phase A — remove temporal semantics from sim/eval metadata

1. Remove runner args/config.
2. Remove best-checkpoint field and bump selection record v2 → v3.
3. Update readers/callers.
4. Verify sim still executes canonical `control_action` only.

### Phase B — deployment v4

1. Remove coefficient from artifact contract/export/restore metadata.
2. Bump deployment v3 → v4.
3. Remove runtime/qualification temporal guards.
4. Slim the public `PolicySpec` without deleting internal restore data.

### Phase C — provenance relaxation

1. Preserve valid commit/root checks.
2. Record dirty state and origin.
3. Convert dirty/noncanonical-origin failure to warning/metadata.

### Phase D — docs/tests

1. Add focused tests.
2. Update durable docs to actual code behavior.
3. Search for stale generic temporal references.

Do not mix a new inference algorithm into any phase.

---

## 12. Focused offline tests

Create a small deterministic test module, for example:

```text
tests/test_deployment_contract.py
```

Do not restore a large historical suite.

### Test A — canonical runtime output

Fake/restored agent returns both:

```text
pred_action
control_action
```

Assert exact array equality:

```text
LoadedPolicy.predict(obs) == validated control_action[0]
```

Shape-only testing is insufficient.

### Test B — public `PolicySpec` boundary

Assert the public spec exposes the required Real-facing fields and does not expose:

```text
temporal_ensemble_coeff
horizon
denoise_steps
rgb_preprocessing
```

Internal deployment spec may still contain restore fields.

### Test C — v4 contract round trip

Construct/minimally export a v4 payload and verify:

```text
parse
inspect_experiment / equivalent metadata path
restore metadata
```

without loading a large model where avoidable.

### Test D — best_ckpt v3

Verify:

```text
writer schema and reader schema agree
inference contains use_ema / denoise_steps / policy_seed_mode
no temporal coefficient
```

### Test E — dirty provenance does not block

Mock Git command results:

```text
valid repo root
valid commit
dirty status
noncanonical/local origin
```

Assert export provenance construction succeeds and records the state.

---

## 13. Cross-repository handoff to `dexmani_real`

After Policy v4 lands, Real should see only:

```text
PolicySpec.action_key
PolicySpec.control_action_dim
PolicySpec.n_obs_steps
PolicySpec.n_action_steps
PolicySpec.observation_fields
PolicySpec.control_dt_s
```

and:

```text
LoadedPolicy.predict(obs) -> finite float64 [N, D_control]
```

Do not add a v3/v4 compatibility layer to Real. Update Real after Policy v4 is available.

---

## 14. Explicit non-goals

Do not implement:

```text
Temporal Ensemble
ACT overlap aggregation
ChunkOverlapBlender replacement
RTC
previous-action conditioning
steps_per_inference < n_action_steps
receding-horizon merge logic
action smoothing
adaptive horizon
latency compensation
new sampler / solver
Flow Matching distillation
model architecture changes
training objective changes
data schema redesign
artifact migration framework
artifact signing / registry
```

`steps_per_inference` is intentionally deferred. If later experiments show that `n_action_steps * dt` is too open-loop, it must be introduced as one explicit sim+Real experiment, not hidden in runtime.

---

## 15. Validation commands

Before editing:

```bash
git status --short
git rev-parse HEAD
rg -n "temporal_ensemble_coeff|ChunkOverlapBlender|temporal_ensembler" .
```

After editing:

```bash
python -m compileall -q dexmani_policy
python -m pytest -q tests/test_deployment_contract.py

git diff --check
git diff --stat
rg -n "temporal_ensemble_coeff|ChunkOverlapBlender|temporal_ensembler" \
  dexmani_policy configs README.md docs
```

The final `rg` should produce no active current-behavior references. Adjust config root names to the actual tree discovered by the executor.

Do not run unless separately authorized:

```text
training
DDP/NCCL
large checkpoint sweeps
long simulator evaluations
real robot code
```

---

## 16. Definition of Done

All of the following must hold:

```text
generic Policy runtime
→ no temporal ensemble field/state/guard
→ exact canonical control_action
```

```text
sim runner
→ exact canonical control_action
→ no overlap/blending state
```

```text
best_ckpt selection
→ v3 metadata
→ no temporal ensemble field
```

```text
deployment artifact
→ v4
→ no temporal ensemble field
→ restore still deterministic
```

```text
public PolicySpec
→ only Real-required raw observation/action timing contract
```

```text
dirty working tree
→ provenance recorded
→ export not rejected solely for dirty state
```

and:

- compileall passes;
- focused tests pass;
- `git diff --check` passes;
- active docs match the implemented behavior;
- no generic blending/RTC/replanning feature was added.

---

## 17. Final Claude Code / Codex report

Return:

### Changed
- files changed;
- deployment schema version;
- best_ckpt record version;
- removed temporal semantics;
- public `PolicySpec` fields after the change.

### Preserved
Explicitly confirm:

```text
canonical control_action
sim sequential chunk execution
normalizer / preprocessing ownership
checkpoint restore semantics
```

### Verified
- compileall result;
- focused pytest result;
- `git diff --check` result;
- final stale-reference search result.

### Not Verified
Explicitly list unrun items, especially:

```text
large GPU checkpoint restore
full simulator benchmark
training
real robot deployment
```

### Remaining risk
Report only concrete remaining risks found from code/tests. Do not propose RTC, temporal smoothing, or a new deployment framework as generic follow-up work.
