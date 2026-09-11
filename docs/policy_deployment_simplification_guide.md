# DexMani Policy — Real Rollout Deployment API Upgrade Guide

> Repository: `haoyangzhanglab/dexmani_policy`  
> Intended executor: Claude Code / Codex  
> Baseline reviewed: `main@7d8018681bd1f38d4864b54ccf0b26500d9fd4de`  
> Cross-repository consumer: `haoyangzhanglab/dexmani_real`  
> Scope: make the existing Policy deployment runtime expose the complete canonical future action chunk required by a simple real-robot receding-horizon rollout. Keep the current model, training, checkpoint and artifact semantics otherwise unchanged.
>
> This document supersedes the previous temporal-ensemble cleanup plan in this file. The temporal-ensemble/schema cleanup has already landed. Do **not** reintroduce Temporal Ensemble, overlap blending, RTC, compatibility branches or artifact-version machinery.

---

Implementation status: the Policy runtime now exposes the API specified below.
`PolicySpec.chunk_size` and `DeploymentSpec.chunk_size` are derived properties;
`LoadedPolicy.predict_action_chunk(observation)` returns an independent finite
NumPy `float64 [chunk_size, control_action_dim]` copy of the validated canonical
future prediction. `predict()` retains its `n_action_steps` output, and the
artifact schema is unchanged. The baseline descriptions and implementation
checklist below record the requirements for this change. Real migration remains
a separate repository task.

## 1. Goal

`dexmani_policy` already computes everything needed by the desired real-robot rollout. The missing piece is only the public deployment surface.

Current model path:

```text
observation
    ↓
agent.predict_action(...)
    ↓
pred_action [B, horizon, action_dim]
    ↓
canonical control slice
    control_action [B, n_action_steps, control_action_dim]
```

Current `LoadedPolicy.predict()` exposes only the short canonical `control_action` slice to Real.

Target public deployment surface:

```text
observation
    ↓
LoadedPolicy.predict_action_chunk(...)
    ↓
canonical future control chunk
    [chunk_size, control_action_dim]
```

where:

```text
start = n_obs_steps - 1
chunk_size = horizon - start
           = horizon - n_obs_steps + 1
```

The first `n_action_steps` rows of this full future chunk must remain exactly equal to the existing `control_action`.

The target is a **small API exposure**, not a model or deployment-format redesign.

---

## 2. Current source facts that must guide the implementation

### 2.1 Agent already produces the full prediction

`dexmani_policy/agents/core/base.py` currently computes:

```python
pred = ...  # [B, horizon, action_dim]
start = self.n_obs_steps - 1
control_action = pred[:, start : start + self.n_action_steps]
tail = pred[:, start + self.n_action_steps :]
```

and returns:

```text
pred_action
control_action
tail
```

Therefore:

- do not change training;
- do not change the action decoder;
- do not add a second model call;
- do not reconstruct future actions from `control_action`;
- do not blend old/new chunks in Policy.

### 2.2 Deployment restore already validates the full prediction

`dexmani_policy/deployment/restore.py::validate_prediction()` already verifies:

```text
pred_action.shape == [B, horizon, action_dim]
control_action.shape == [B, n_action_steps, control_action_dim]
control_action == exact canonical slice of pred_action
all values finite
```

Reuse this validated `PredictionSnapshot`. Do not create a parallel validation path.

### 2.3 Artifact contract already contains enough information

The persisted deployment contract already contains:

```text
horizon
n_obs_steps
n_action_steps
action_key
action_dim
control_dt_s
...
```

No persisted `chunk_size` field is required.

`chunk_size` is a deterministic derived property:

```python
horizon - n_obs_steps + 1
```

Do **not** bump the artifact schema or modify `_format = "dexmani.deployment"` for this task.

### 2.4 Evaluation seed support already exists

`load_experiment(..., seed=...)` already accepts any non-negative integer and `LoadedPolicy.reset_episode()` resets Policy-owned stochastic state from that seed.

`LoadedPolicy.warmup()` already snapshots/restores Python, NumPy, Torch and initialized CUDA RNG states, so warmup must continue not to consume the rollout stochastic stream.

Do not add a training-seed argument to the deployment runtime. Training seed remains experiment/checkpoint identity; Real evaluation seed is the runtime stochastic seed.

---

## 3. Desired public contract

### 3.1 Add derived `chunk_size`

Add a read-only derived property to the public runtime `PolicySpec`:

```python
@property
def chunk_size(self) -> int:
    return self.horizon - self.n_obs_steps + 1
```

For symmetry, adding the same derived property to internal `DeploymentSpec` is recommended if it removes duplicated arithmetic in validation/tests.

Requirements:

```text
chunk_size > 0
n_action_steps <= chunk_size
```

The second invariant is already implied by the existing:

```text
n_obs_steps - 1 + n_action_steps <= horizon
```

Do not persist `chunk_size` separately.

### 3.2 Add `LoadedPolicy.predict_action_chunk()`

Recommended API:

```python
def predict_action_chunk(
    self,
    observation: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return the canonical finite float64 future control chunk [C,D]."""
```

Required implementation semantics:

```text
observation
→ existing _observation_tensors()
→ restored.agent.predict_action(...)
→ existing validate_prediction(...)
→ snapshot.pred_action
→ slice from start=n_obs_steps-1 to horizon
→ slice control dimensions only
→ ownership-copy NumPy float64 [chunk_size, control_action_dim]
→ finite/shape validation
```

Conceptually:

```python
start = self.spec.n_obs_steps - 1
chunk = snapshot.pred_action[
    0,
    start:,
    : self.spec.control_action_dim,
]
```

Return a standalone NumPy ownership copy, consistent with the current `predict()` boundary.

### 3.3 Keep `LoadedPolicy.predict()` during migration

Do not break existing simulation/deployment callers in the same patch.

Keep current behavior:

```text
predict(obs)
→ [n_action_steps, control_action_dim]
```

The exact required relationship is:

```python
predict(obs) == predict_action_chunk(obs)[:n_action_steps]
```

for the same model state and same stochastic draw.

Do **not** implement that relationship by calling the stochastic model twice. If both outputs are needed in one operation, derive them from the same validated model result.

It is acceptable for `predict()` and `predict_action_chunk()` to each perform one inference when called independently. Tests that compare them must reset the episode RNG or use a deterministic fake agent.

After `dexmani_real` migrates completely, a later cleanup may reconsider whether `predict()` remains public. That deletion is not required here.

---

## 4. Meaning of `n_action_steps`

Do not create a new Policy field such as:

```text
replan_steps
steps_per_inference
execution_horizon
async_horizon
```

For the current DexMani stack, preserve the existing `n_action_steps` meaning:

```text
number of canonical actions normally consumed before the next Policy query
```

This already matches current simulation execution and is sufficient for Real to choose its inference cadence.

The relationship is:

```text
full future chunk length = chunk_size
normal query interval     = n_action_steps
```

A useful future tail exists when:

```text
chunk_size > n_action_steps
```

If they are equal, the Policy is still valid; Real simply has no extra future tail with which to hide inference latency.

Do not force `n_action_steps < chunk_size` in Policy.

---

## 5. Do not add action aggregation semantics

This task must not introduce:

```text
Temporal Ensemble
ChunkOverlapBlender
weighted averaging
RTC
previous-chunk state
new/old chunk interpolation
adaptive horizon
queue thresholds
latency-dependent model behavior
```

The Policy side only returns a canonical prediction.

Real owns timestamp scheduling and may replace an old future plan with a newer valid future plan. Policy does not need to know about that scheduling decision.

If a future research model requires RTC, temporal consistency guidance or model-aware chunk stitching, implement it as a model/inference algorithm in `dexmani_policy` behind the same final action-chunk API. Do not put generic blending state into this deployment runtime now.

---

## 6. Keep simulation semantics unchanged

Do not modify simulation runners merely to mimic the Real scheduler.

Current simulation behavior can remain conceptually:

```text
obs
→ policy.predict_action()
→ control_action [n_action_steps,D]
→ env.step() over those actions
→ next Policy query
```

The new full-chunk API exists for the real deployment boundary, where a timestamped hardware executor can continue consuming the unexecuted future tail while the next inference is running.

Do not add Real timing or timestamp logic to `env_runner`.

---

## 7. Seed semantics: preserve and verify, do not redesign

The Policy repository already has two distinct concepts:

```text
training/checkpoint seed identity
runtime/evaluation seed
```

Simulation evaluation already exposes explicit `eval_seeds`. Preserve that design.

Deployment runtime requirements:

```text
load_experiment(seed >= 0) accepted
reset_episode() resets inference stochastic stream
warmup() does not consume the episode stream
```

Recommended lightweight regression checks:

1. negative seed is rejected;
2. two resets with the same seed reproduce a deterministic fake/stochastic test stream;
3. different seeds produce different stochastic streams where a synthetic stochastic fake is used;
4. warmup leaves the rollout RNG stream unchanged.

Do not encode an `eval_seed` field into the deployment artifact. It is a per-rollout runtime input owned by the caller.

---

## 8. EEF/tactile contract vs. model support

Current Real `main` can already construct:

```text
eef_pose [T,9]
tactile_force [T,5,120,3]
```

and the Policy exporter/data contract now admits these fields
(`_SUPPORTED_OBSERVATION_FIELDS` includes `eef_pose` and `tactile_force`).

Current model encoders, however, do **not** consume them: every encoder's
`consumed_observation_fields` is still `joint_state + point_cloud` (or
`joint_state + rgb`). Contract support is not model support — a strict restore
of an EEF/tactile artifact still fails the encoder-consumer check.

Do **not** opportunistically wire EEF/tactile into an encoder without a
hypothesis. EEF/tactile end-to-end model support requires a separate,
hypothesis-driven change across the Policy data/model/export path.

Similarly, do not alter:

```text
processed/Zarr schemas
normalizers
datasets
observation encoders
point-cloud semantics
fingertip semantics
RGB preprocessing
```

unless a direct compile dependency is discovered.

---

## 9. Preserve strict deployment provenance and current schema

Recent Policy cleanup intentionally converged on:

```text
one current deployment format
one current best_ckpt schema
strict metadata/provenance
no temporal compatibility dispatch
```

Do not undo that work.

Preserve:

```text
DEPLOYMENT_FORMAT = "dexmani.deployment"
strict top-level contract structure
strict restore dimensions
normalizer validation
RGB preprocessing validation
producer Git provenance
checkpoint selection/export semantics
```

No version branching or compatibility adapter should be added for `chunk_size`; it is derived.

---

## 10. Expected production patch

Primary files expected to change:

```text
dexmani_policy/deployment/runtime.py
```

Likely small supporting changes:

```text
dexmani_policy/deployment/contract.py      # derived property only, if useful
dexmani_policy/agents/core/base.py         # stale tail comment only, optional
tests/test_deployment_contract.py          # deterministic API regression tests
docs/policy_deployment_simplification_guide.md
```

Files that should normally **not** require functional changes:

```text
dexmani_policy/deployment/export.py
dexmani_policy/deployment/restore.py
training code
agent architectures
action decoders
normalizers
datasets
env runners
Hydra configs
checkpoint selection
```

If one of these must change, first demonstrate the direct dependency and keep the patch minimal.

---

## 11. Recommended implementation order

### P0 — verify current baseline

Before editing:

```bash
git status --short
git rev-parse HEAD
rg -n "pred_action|control_action|tail|n_action_steps|horizon|n_obs_steps" \
  dexmani_policy/deployment dexmani_policy/agents/core tests
```

Confirm the source facts in this guide still match current `main`.

### P1 — derived chunk contract

1. Add `PolicySpec.chunk_size`.
2. Optionally add `DeploymentSpec.chunk_size` to centralize the same arithmetic.
3. Add unit checks for the derived value and invariant.
4. Do not change serialized artifact fields.

### P2 — full future chunk API

1. Add `LoadedPolicy.predict_action_chunk()`.
2. Reuse `validate_prediction()`.
3. Slice `snapshot.pred_action` from `n_obs_steps - 1` through the end.
4. Slice only `control_action_dim` dimensions.
5. Return finite `float64 [chunk_size,D]` ownership copy.
6. Preserve `predict()` behavior.

### P3 — deterministic tests

Add CPU-only synthetic/fake-agent tests for:

```text
chunk shape
chunk dtype
finite values
exact canonical slice
first n_action_steps == control_action
control_action_dim trimming
chunk_size derivation
seed validation/reset semantics
warmup RNG preservation if inexpensive to isolate
```

Do not require robot hardware.

### P4 — cross-repo handoff

Only after the public API is green should `dexmani_real` switch its inference worker to `predict_action_chunk()`.

Final code should not contain a long-lived fallback such as:

```python
if hasattr(runtime, "predict_action_chunk"):
    ...
else:
    runtime.predict(...)
```

Coordinate repository revisions instead of maintaining a compatibility branch.

---

## 12. Required tests

At minimum, cover the following with a deterministic fake agent/restored object.

### A. Existing canonical control output remains unchanged

```text
predict(obs)
→ exact validated control_action
→ float64 [n_action_steps,D_control]
```

### B. Full chunk is the canonical future slice

Given synthetic:

```text
pred_action [1,horizon,action_dim]
```

assert:

```python
chunk == pred_action[
    0,
    n_obs_steps - 1 :,
    :control_action_dim,
]
```

### C. Prefix relation

Assert:

```python
chunk[:n_action_steps] == control_action[0]
```

### D. Auxiliary action dimensions never leak to Real

For a Policy where:

```text
action_dim > control_action_dim
```

verify the full deployment chunk contains exactly `control_action_dim` columns.

### E. No artifact schema change

Existing synthetic deployment contract/parser tests must pass without a new persisted `chunk_size` key.

---

## 13. Explicit non-goals

Do not implement in this task:

```text
Temporal Ensemble
RTC
chunk blending
old/new prediction aggregation
PolicyServer / remote inference
action queue
queue threshold
adaptive query cadence
new deployment schema version
persisted chunk_size
Real robot timing logic
hardware safety
rollout outcome handling
recording changes
EEF/tactile model support
training changes
sampler/solver changes
new action representation
```

---

## 14. Cross-repository ownership contract

After this task, the intended boundary is:

```text
dexmani_policy
    owns:
        model
        stochastic inference
        horizon
        n_obs_steps
        n_action_steps
        full canonical future action chunk

    exposes:
        PolicySpec.chunk_size          # derived
        predict_action_chunk(obs)
        reset_episode(seed-owned state)


dexmani_real
    owns:
        causal physical observation
        inference cadence = n_action_steps * control_dt
        target timestamps
        stale-action filtering
        future-plan replacement
        IK / safety
        robot execution
        recording / outcome
```

Neither repository should implement the other side's responsibilities.

---

## 15. Validation commands

Use the repository's existing environment and tests. At minimum:

```bash
python -m compileall -q dexmani_policy
python -m unittest discover -s tests -p 'test_deployment_contract.py'
git diff --check
git diff --stat
```

Also inspect for accidental reintroduction of removed mechanisms:

```bash
rg -n "temporal_ensemble|ChunkOverlapBlender|RTC|replan_steps|queue_threshold" \
  dexmani_policy/deployment dexmani_policy/agents/core
```

Descriptive comments/docs may mention non-goals, but no active generic mechanism should be introduced.

---

## 16. Definition of Done

All of the following must hold:

```text
Policy artifact schema
→ unchanged
```

```text
PolicySpec.chunk_size
→ derived from horizon and n_obs_steps
→ not persisted separately
```

```text
LoadedPolicy.predict()
→ unchanged canonical [n_action_steps,D] semantics
```

```text
LoadedPolicy.predict_action_chunk()
→ one inference
→ existing validate_prediction()
→ exact future slice of pred_action
→ finite float64 [chunk_size,D_control]
```

and:

- no model/training behavior changed;
- no temporal aggregation mechanism added;
- no compatibility branch added;
- arbitrary non-negative runtime seed support remains intact;
- warmup remains RNG-neutral for the episode stream;
- existing strict deployment/provenance checks remain intact;
- deterministic tests pass;
- `dexmani_real` can consume the new API without understanding Policy internals.

The final result should be a **narrower, more useful public deployment API**, not a broader deployment framework.
