# DexMani Policy — Canonical Deployment Cleanup Guide

> Repository: `haoyangzhanglab/dexmani_policy`  
> Intended executor: Claude Code / Codex  
> Scope: personal PhD robot-learning research. Keep exactly one current implementation and one current metadata contract. Do not keep compatibility layers or versioned schema branches.
>
> **Current repository constraint:** there is no trained Policy checkpoint available in this repository at this stage. This task must therefore be completed and verified without training, checkpoint selection, real checkpoint restore, or real deployment export.

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

1. Generic sim/deployment code has exactly one semantic target: canonical `control_action`.
2. `temporal_ensemble_coeff` disappears from active configs, eval, selection metadata, export, runtime, artifact contract, and current docs.
3. `best_ckpt.json` has one strict current shape with **no version field**.
4. deployment artifact has one strict current shape with **no v1/v2/v3 schema dispatch or compatibility reader**.
5. production code contains no legacy metadata/artifact migration logic.
6. existing checkpoint restore, preprocessing, normalizer, qualification, and provenance behavior remains unchanged unless directly coupled to the removed field.
7. all verification in this phase is source-level or synthetic/offline; a real trained checkpoint is not required for Definition of Done.

The target is one readable research code path, not a compatibility framework.

---

## 2. Hard execution constraints

Claude Code / Codex must **not** run or initiate any of the following during this task:

```text
policy training
resume training
fine-tuning
train.py
train_ddp.py
DDP / NCCL
checkpoint selection sweep
select_best_ckpt.py as a real evaluation job
eval_best_ckpt.py as a real simulator evaluation
long simulator rollout
real checkpoint restore
real deployment export
GPU inference that depends on trained weights
real robot code
```

Do not create or train a dummy neural policy merely to satisfy a validation step.

Do not fabricate a “representative trained experiment”.

Use only:

```text
source inspection
Hydra/config parsing where lightweight
synthetic metadata
synthetic deployment payloads
fake agents / fake restored objects
temporary files
deterministic CPU-only contract checks
compile/static checks
```

If a validation step genuinely requires trained weights, mark it **Deferred / Not Verified** rather than generating weights.

---

## 3. Non-negotiable invariants

### Runtime semantics

`LoadedPolicy.predict()` must remain logically equivalent to:

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

A fake agent/restored object may be used to test this code path; a trained checkpoint is not needed.

### Sim semantics

Keep the one intended execution rule:

```text
obs history
→ predict_action()
→ canonical control_action [N,D]
→ env.step() over N actions in order
→ next query
```

Do not introduce `steps_per_inference`, overlap merging, or replanning in this task.

Do not run a long simulator evaluation just to prove this. Inspect/test the runner logic directly.

### Restore / preprocessing

Preserve existing code and contracts for:

```text
EMA/raw weight selection
denoise_steps
normalizer validation
RGB preprocessing
point-cloud/fingertip semantic validation
prediction parity / qualification
strict Git provenance
```

These should be source/synthetic tested where possible. Real trained-weight restore is explicitly deferred.

---

## 4. No compatibility layer and no schema-version machinery

After this cleanup there should be exactly one supported shape for each current metadata object.

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
artifact upgrader
schema registry
```

The repository currently has no trained checkpoint that needs preservation during this task, so no runtime compatibility mechanism is justified.

Future stale experiment metadata/artifacts, if encountered later, should be updated/re-exported once with the then-current code rather than expanding the runtime parser.

---

## 5. `best_ckpt.json`: one current schema

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

Do **not** run checkpoint selection in this phase.

### Reader

Update `read_best_ckpt_json()` to validate only the current required keys/types.

Do not inspect a version number and do not normalize historical records.

If the file does not match the current schema, fail with a concise current-contract error.

### Offline verification

Use a temporary directory and hand-written synthetic JSON records to test:

```text
valid current record → accepted
missing required key → rejected
obsolete temporal field / obsolete version field → not part of the current expected schema
```

Do not create a checkpoint or invoke an environment.

---

## 6. Deployment artifact: one current unversioned contract

The artifact should expose one current format only.

Prefer a stable artifact-type marker such as:

```text
_format = "dexmani.deployment"
```

Remove persisted compatibility-dispatch fields such as:

```text
schema_version
v1/v2/v3 format suffixes
```

when they exist solely for historical version dispatch.

The parser validates the current required structure, not a history of structures.

Required current top-level semantics remain conceptually:

```text
artifact type marker
contract
weights
```

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

Do not remove fields actually needed for a future trained model restore, for example:

```text
action_dim
horizon
denoise_steps
rgb_preprocessing
agent config
weights
normalizer-related state
```

### Offline verification only

There is no trained checkpoint, therefore **do not call real export as an acceptance condition**.

Instead build a minimal synthetic artifact mapping in a test using small dummy `torch.Tensor` weights and the current contract structure. Verify:

```text
current artifact structure parses
required fields are enforced
temporal field is absent
schema-version dispatch is absent
```

Do not call `restore_deployment_agent()` on a real model in this phase unless it can be exercised with a deliberately tiny fake object without trained weights.

Real artifact export/restore is deferred until the first trained checkpoint exists.

---

## 7. Do not slim unrelated `PolicySpec` fields

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

unless a direct compile/call-site audit proves removal is truly local and necessary.

Do not create unnecessary cross-repository churn.

---

## 8. Remove temporal semantics end-to-end

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

Do not run those programs as long evaluations in this task.

### Selection/export/runtime

Remove the field from checkpoint-selection metadata, export inference settings, artifact contract, runtime spec, and qualification comparisons.

Do not replace it with another generic blending flag.

---

## 9. Preserve strict provenance

Do not relax exporter provenance in this task.

Preserve existing checks for:

```text
repository root
valid HEAD commit
clean working tree
expected origin
```

Do not run a real deployment export just to exercise those checks while no trained checkpoint exists.

Unit-test pure provenance helpers only if directly touched; otherwise leave them alone.

---

## 10. Preserve observation/data semantics

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

## 11. Expected patch scope

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

## 12. Recommended implementation order

### Phase A — remove temporal runtime/eval semantics

1. Remove env-runner config/constructor field.
2. Remove eval/demo argument/result propagation.
3. Remove coefficient from checkpoint-selection writer/reader.
4. Remove `record_version`; validate one current `best_ckpt.json` shape.
5. Inspect/synthetically test sim runner semantics without running a long evaluation.

### Phase B — simplify deployment artifact contract

1. Remove coefficient from export/contract/runtime/qualification.
2. Remove artifact schema-version dispatch/version fields.
3. Keep one stable unversioned artifact marker + current required structure.
4. Preserve restore/preprocessing/provenance logic.
5. Validate the parser with a synthetic payload; do not export a trained model.

### Phase C — docs + lightweight regression

1. Update durable docs to the actual implementation.
2. Add small CPU-only offline checks.
3. Search for obsolete temporal/version compatibility code.

---

## 13. Lightweight regression coverage

Do not create a large test framework and do not require real weights.

A small standard-library `unittest` module is sufficient, for example:

```text
tests/test_deployment_contract.py
```

Required checks:

### A. Canonical runtime output

Use a fake restored object / fake agent exposing `pred_action` and `control_action`; assert:

```text
LoadedPolicy.predict(obs) == validated control_action[0]
```

No trained checkpoint.

### B. Current best_ckpt schema

Use synthetic JSON in `tempfile.TemporaryDirectory()`.

Verify one valid current record is accepted and malformed/obsolete layouts are rejected by current required-key validation.

Do not test historical version compatibility.

### C. Current deployment contract

Construct a minimal synthetic artifact mapping with tiny dummy tensor weights.

Verify parsing without any schema-version value.

Do not export or restore a trained experiment.

### D. Runtime spec

Verify `PolicySpec` no longer exposes `temporal_ensemble_coeff` and unrelated fields remain intact.

### E. Config construction

Where inexpensive, compose representative Hydra configs and verify env-runner construction no longer expects the removed field.

Do not build datasets/models if that requires unavailable data or trained weights.

---

## 14. Cross-repository handoff for the current no-checkpoint phase

After Policy source cleanup, proceed to `dexmani_real` **without** requiring a real artifact.

Real should be updated against the public Policy Python contract using:

```text
synthetic/fake PolicySpec
synthetic/fake action chunks
```

Do not block Real cleanup on:

```text
trained checkpoint
deployment_latest.pt
real load_experiment()
GPU inference
```

Current sequence:

```text
1. finish Policy source cleanup
2. compile + synthetic contract checks
3. update Real source against current public contract
4. Real compile + synthetic rollout checks
5. stop this cleanup phase
```

A later integration phase begins only after a real trained checkpoint exists:

```text
trained checkpoint
→ current best/eval metadata
→ current deployment artifact export
→ Policy restore/inference validation
→ Real check/shadow
→ physical eval
```

That later phase is **not** part of this task.

---

## 15. Explicit non-goals

Do not implement or execute:

```text
compatibility readers
record/artifact version dispatch
legacy migrations
policy training / fine-tuning
DDP/NCCL
checkpoint generation
checkpoint selection sweep
real checkpoint restore
real deployment export
full simulator evaluation
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

## 16. Validation

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

If the chosen lightweight test path differs, run and report the actual deterministic CPU-only command.

Expected final search:

- no active temporal mechanism/config;
- no best-checkpoint/deployment compatibility-version dispatch;
- this implementation guide may mention removed names descriptively.

Do not invoke any training/evaluation/export command that needs real trained weights.

---

## 17. Definition of Done for this phase

All must hold **without a trained checkpoint**:

```text
sim/runtime source semantics
→ canonical control_action only
→ no generic temporal option/state
```

```text
best_ckpt.json code path
→ exactly one current schema
→ no record_version
→ no compatibility parser
→ synthetic reader tests pass
```

```text
deployment artifact code path
→ exactly one current contract
→ no schema-version dispatch
→ no temporal field
→ synthetic parser tests pass
```

and:

- strict Git provenance code remains unchanged;
- restore/preprocessing/normalizer semantics remain unchanged;
- unrelated `PolicySpec` fields remain unchanged;
- compile and CPU-only lightweight checks pass;
- durable docs match current code;
- no new action aggregation/replanning mechanism exists;
- **no strategy training, checkpoint creation, checkpoint selection, real artifact export, or real checkpoint restore was performed.**

---

## 18. Deferred integration validation

Explicitly defer until a trained checkpoint exists:

```text
real checkpoint restore
real deployment artifact export
artifact → restore round trip with actual agent weights
GPU inference
full simulator evaluation
Policy → Real integration using real artifact
real robot shadow/run/eval
```

These are not failures of the current cleanup and must not block completion.

---

## 19. Final Claude Code / Codex report

### Changed
Report exact files and the single current metadata/artifact shapes.

### Preserved
Confirm canonical `control_action`, sequential sim semantics, restore/preprocessing, unrelated `PolicySpec` fields, and strict provenance.

### Verified
Report only commands actually run: compile, synthetic/unit checks, diff check, stale-reference search.

### Explicitly Not Run
Must state that no policy training, checkpoint selection, real checkpoint restore, real deployment export, long simulator evaluation, or real robot work was executed.

### Deferred
List real-checkpoint integration validation as a future phase after a trained checkpoint exists.

Do not propose compatibility layers or temporal smoothing as follow-up work.
