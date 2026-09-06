# DexMani Policy — Canonical Deployment Cleanup Guide

> Repository: `haoyangzhanglab/dexmani_policy`  
> Intended executor: Claude Code / Codex  
> Scope: personal PhD robot-learning research. Keep exactly one current implementation and one current metadata contract. Do not keep compatibility layers or versioned schema branches.

---

## 1. Goal

The canonical inference path is already correct:

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

This task removes obsolete generic temporal-ensemble semantics and leaves the repository with one simple current deployment contract.

Required outcomes:

1. Generic sim/deployment executes canonical `control_action` directly.
2. `temporal_ensemble_coeff` disappears from active configs, eval, selection metadata, export, runtime, artifact contract, and current docs.
3. `best_ckpt.json` has one strict current shape with **no version field**.
4. deployment artifact has one strict current shape with **no v1/v2/v3 schema dispatch or compatibility reader**.
5. old metadata/artifacts are regenerated or manually updated outside the runtime when needed; production code does not carry migration logic.
6. existing checkpoint restore, preprocessing, normalizer, qualification, and provenance behavior remains unchanged unless directly coupled to the removed field.

The target is one readable research code path, not a compatibility framework.

---

## 2. Non-negotiable invariants

### Runtime

`LoadedPolicy.predict()` must remain equivalent to:

```python
result = restored.agent.predict_action(
    tensors,
    denoise_timesteps=restored.spec.denoise_steps,
)
snapshot = validate_prediction(result, restored.spec, batch_size=1)
return snapshot.control_action[0]  # finite float64 NumPy [N,D]
```

It must not use:

```text
pred_action tail
previous chunks
overlap state
action smoothing
temporal ensemble
RTC history
```

### Sim

Keep:

```text
obs history
→ predict_action()
→ canonical control_action [N,D]
→ env.step() over N actions in order
→ next query
```

Do not introduce `steps_per_inference`, overlap merging, or replanning in this task.

### Restore / preprocessing

Preserve:

```text
EMA/raw weight selection
denoise_steps
normalizer validation
RGB preprocessing
point-cloud/fingertip semantic validation
prediction parity / qualification
strict Git provenance
```

These are useful reproducibility/correctness checks and are unrelated to temporal cleanup.

---

## 3. No compatibility layer and no schema-version machinery

This repository is a personal research project. After this cleanup there should be exactly one supported shape for each current metadata object.

Do **not** implement or retain logic such as:

```python
if version == 1:
    ...
elif version == 2:
    ...
elif version == 3:
    ...
```

Do not add:

```text
legacy parser
migration adapter
compatibility shim
v2-null special case
artifact upgrader
schema registry
```

### Old files

If an old experiment has stale metadata:

```text
best_ckpt.json
```

update/regenerate that file once before using the experiment.

If an old experiment has a stale deployment artifact:

```text
checkpoints/deployment_latest.pt
```

re-export it using the current code.

The runtime should remain clean rather than permanently supporting historical formats.

---

## 4. `best_ckpt.json`: one current schema

Remove `record_version` entirely.

The current record should contain the existing checkpoint/result/selection fields plus:

```json
"inference": {
  "use_ema": true,
  "denoise_steps": 10,
  "policy_seed_mode": "episode_seed"
}
```

No:

```text
record_version
temporal_ensemble_coeff
legacy fields
```

### Writer

Update `select_best_ckpt.py` to write only the current schema.

### Reader

Update `read_best_ckpt_json()` to validate only the current required keys/types.

Do not inspect a version number and do not normalize historical records.

If the file does not match the current schema, fail with a concise message such as:

```text
best_ckpt.json does not match the current schema; regenerate/update it with the current code.
```

Do not teach the reader how to interpret old layouts.

### Existing experiments

Do not automatically rerun expensive checkpoint selection in tests or migration code.

For existing experiments that were already selected under canonical `temporal_ensemble_coeff=null` semantics, the developer may perform a one-time metadata edit that removes obsolete fields. That is repository maintenance, not runtime compatibility logic.

Experiments selected under non-null temporal ensemble semantics should be re-evaluated rather than silently reused.

---

## 5. Deployment artifact: one current unversioned contract

The artifact should also expose one current format only.

Prefer an unversioned marker such as:

```text
_format = "dexmani.deployment"
```

or an equivalent stable artifact-type marker.

Remove persisted version-dispatch fields such as:

```text
schema_version
v1/v2/v3 format suffixes
```

if they exist solely for compatibility dispatch.

The parser should validate the **current required structure**, not a historical version number.

Required current top-level semantics remain conceptually:

```text
artifact type marker
contract
weights
```

and contract contains the current restore/inference/data/provenance metadata required by Policy.

### Remove temporal field

Remove `temporal_ensemble_coeff` from:

```text
DeploymentSpec
contract parser
export payload
runtime PolicySpec construction
runtime generic temporal guard
qualification comparisons
```

### Preserve restore data

Do not remove fields that are actually needed to restore the trained model, for example:

```text
action_dim
horizon
denoise_steps
rgb_preprocessing
agent config
weights
normalizer-related state
```

### Old artifacts

Do not load them through compatibility code.

Simply:

```text
old deployment artifact
→ re-export experiment
→ current deployment artifact
```

---

## 6. Do not slim unrelated `PolicySpec` fields

The only mandatory public-spec removal in this task is:

```text
PolicySpec.temporal_ensemble_coeff
```

Keep existing unrelated fields such as:

```text
action_dim
horizon
requires_hand
rgb_preprocessing
```

unless a direct compile/call-site audit proves they are already dead and removing them is truly local.

Do not create unnecessary cross-repository churn under the label of cleanup.

---

## 7. Remove temporal semantics end-to-end

Search before editing. Current references are expected in areas including:

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
scripts/
```

### Env runners/configs

Remove:

```text
temporal_ensemble_coeff constructor args
forwarding
Hydra config keys
generic temporal rejection guards
```

Preserve canonical `control_action` execution.

### Eval/demo

Remove coefficient arguments and result metadata from:

```text
eval_best_ckpt.py
record_demo.py
related scripts/config plumbing
```

Do not rewrite historical result files.

### Selection/export/runtime

Remove the field from new checkpoint-selection metadata, export inference settings, artifact contract, runtime spec, and qualification comparisons.

Do not replace it with another generic blending flag.

---

## 8. Preserve strict provenance

Do not relax exporter provenance in this task.

Preserve existing checks for:

```text
repository root
valid HEAD commit
clean working tree
expected origin
```

For a research artifact, a clean tree makes the recorded commit meaningful.

Do not add dirty-tree compatibility metadata, signing, registries, or remote abstraction.

---

## 9. Preserve observation/data semantics

Do not simplify point-cloud/fingertip/RGB contracts while removing an unrelated temporal field.

Preserve checks that prevent:

```text
wrong coordinate frame
wrong units
wrong xyzrgb preprocessing
wrong point-cloud config identity
wrong fingertip geometry/order
wrong RGB preprocessing
```

No data-schema redesign belongs in this task.

---

## 10. Expected patch scope

Search first and keep the patch minimal. Expected production areas:

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
relevant scripts
```

Do not modify agents, losses, samplers, datasets, normalizers, or training objectives unless a direct dependency is demonstrated.

---

## 11. Recommended implementation order

### Phase A — remove temporal runtime/eval semantics

1. Remove env-runner config/constructor field.
2. Remove eval/demo argument/result propagation.
3. Remove coefficient from checkpoint-selection writer/reader.
4. Remove `record_version`; validate one current `best_ckpt.json` shape.
5. Verify sim execution is unchanged.

### Phase B — simplify deployment artifact contract

1. Remove coefficient from export/contract/runtime/qualification.
2. Remove artifact schema-version dispatch/version fields.
3. Keep one stable unversioned artifact marker + current required structure.
4. Preserve restore/preprocessing/provenance logic.

### Phase C — docs + lightweight regression

1. Update durable docs to the actual implementation.
2. Add small offline checks.
3. Search for obsolete temporal/version compatibility code.

---

## 12. Lightweight regression coverage

Do not create a large test framework.

A small standard-library `unittest` module is sufficient, for example:

```text
tests/test_deployment_contract.py
```

Required checks:

### A. Canonical runtime output

Fake agent exposes `pred_action` and `control_action`; assert:

```text
LoadedPolicy.predict(obs) == validated control_action[0]
```

### B. Current best_ckpt schema

Verify one valid current record is accepted and missing/obsolete fields are rejected through ordinary required-key validation.

Do **not** test v1/v2/v3 compatibility.

### C. Current deployment contract

Verify a minimal current artifact parses without any schema-version value.

Do **not** test old artifact versions.

### D. Runtime spec

Verify `PolicySpec` no longer exposes `temporal_ensemble_coeff` and canonical prediction semantics remain unchanged.

---

## 13. Cross-repository handoff

After Policy cleanup, Real should continue using only:

```text
dexmani_policy.deployment public API
```

Real must remove its `temporal_ensemble_coeff` compatibility guard.

Real does not need to know or validate artifact versions because artifact parsing remains Policy-owned.

Recommended sequence:

```text
1. finish Policy cleanup
2. compile/offline check
3. update/re-export one representative experiment metadata/artifact
4. update Real
5. Real shadow/check
6. only then physical rollout
```

---

## 14. Explicit non-goals

Do not implement:

```text
compatibility readers
record/artifact version dispatch
legacy migrations
Temporal Ensemble
ChunkOverlapBlender replacement
RTC
steps_per_inference
chunk merging
action smoothing
adaptive horizon
latency compensation
sampler/solver redesign
model architecture changes
training objective changes
normalizer/data schema changes
PolicySpec redesign beyond the removed field
Git provenance redesign
```

---

## 15. Validation

Before editing:

```bash
git status --short
git rev-parse HEAD
rg -n "temporal_ensemble_coeff|ChunkOverlapBlender|temporal_ensembler|record_version|schema_version" \
  dexmani_policy README.md docs scripts
```

After editing:

```bash
python -m compileall -q dexmani_policy
python -m unittest discover -s tests -p 'test_deployment_contract.py'

git diff --check
git diff --stat
rg -n "temporal_ensemble_coeff|ChunkOverlapBlender|temporal_ensembler|record_version|schema_version" \
  dexmani_policy README.md docs scripts
```

Expected final search:

- no active temporal mechanism/config;
- no best-checkpoint/deployment compatibility-version dispatch;
- this implementation guide may mention removed names descriptively.

Do not run training, DDP, large checkpoint sweeps, full simulator benchmarks, or real robot code unless separately authorized.

---

## 16. Definition of Done

All must hold:

```text
sim/runtime
→ canonical control_action only
→ no generic temporal option/state
```

```text
best_ckpt.json
→ exactly one current schema
→ no record_version
→ no compatibility parser
```

```text
deployment artifact
→ exactly one current contract
→ no schema-version dispatch
→ no temporal field
```

```text
old metadata/artifacts
→ regenerated or manually updated when needed
→ not supported by permanent runtime compatibility code
```

and:

- strict Git provenance unchanged;
- restore/preprocessing/normalizer semantics unchanged;
- unrelated `PolicySpec` fields unchanged;
- lightweight tests and compile checks pass;
- durable docs match current code;
- no new action aggregation/replanning mechanism exists.

---

## 17. Final Claude Code / Codex report

### Changed
Report exact files and the single current metadata/artifact shapes.

### Preserved
Confirm canonical `control_action`, sequential sim execution, restore/preprocessing, `PolicySpec` unrelated fields, and strict provenance.

### Verified
Report compile, lightweight tests, `git diff --check`, and final stale-reference search.

### Manual follow-up
List any old experiments whose `best_ckpt.json` or deployment artifact must be regenerated/updated once.

### Not Verified
List GPU/full-sim/training/real-robot work not executed.

Do not propose compatibility layers or temporal smoothing as follow-up work.
