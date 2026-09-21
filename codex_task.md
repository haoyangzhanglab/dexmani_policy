# Codex Task — DexMani Policy Correctness Repair and Repository Cleanup

Baseline reviewed: `main@bc0901d15e3f1d344bd472c50d8738fc8922fc71` (`0921 temp`, 2026-09-21).

This task is based on a repository-wide fact-check. Implement the confirmed fixes below in priority order. Keep the scope narrow: correctness first, then low-risk cleanup, then comments/docs. Do not revive intentionally removed systems.

---

## 0. Scope and hard constraints

### Goals

1. Eliminate confirmed semantic-drift and workflow bugs.
2. Make config validation fail fast on invalid values instead of silently coercing them.
3. Repair broken self-check/example paths that are currently misleading.
4. Remove confirmed dead code left by recent deployment refactors.
5. Remove small duplicated/dead training state where it does not require a persisted schema change.
6. Bring code comments, config comments, CLI help, and architecture docs back in sync with the current implementation.

### Explicit non-goals

Do **not**:

- restore or recreate a `tests/` directory;
- add, remove, pin, or change Python/Conda dependencies or library versions;
- modify `pyproject.toml` or `requirements.txt`;
- change the `simple.v3` training-checkpoint schema in this task;
- remove `TrainCheckpoint.monitor` or other persisted `simple.v3` fields;
- redesign EMA construction, DDP dataset loading, normalizer fitting, or other performance-only paths;
- refactor unrelated model architecture;
- reintroduce `qualify.py`, config/checkpoint reconciliation, fsync/rollback publication machinery, midpoint solver code, temporal ensemble, or ActionFlow;
- introduce a new compatibility layer for old checkpoint formats;
- weaken strict fail-fast behavior in deployment/evaluation.

### General implementation rules

- Prefer one source of truth over parallel validation logic.
- Preserve existing public CLI behavior unless this task explicitly changes it.
- Fail early with actionable `ValueError` / `RuntimeError` messages.
- Do not use `assert` for user/config validation introduced by this task.
- Do not silently fall back from checkpoint-owned model semantics to current config.
- Keep environment/task evaluation configuration separate from model-constructor configuration.
- Avoid broad renames or formatting-only churn.
- Re-read current files before editing; if HEAD moved after the baseline, adapt the patch to the current implementation rather than blindly applying line-based changes.

---

# Phase 1 — Correctness fixes

## 1. Make offline evaluation checkpoint-owned for model semantics

### Problem

Current evaluation builds the Agent from the experiment's current `config.yaml`:

`training/eval_utils.py::build_eval_components()`

then only compares the compact `resume_contract["agent"]` contract before loading the checkpoint.

The checkpoint already persists the full resolved constructor mapping at:

`checkpoint.resume_contract["agent_config"]`.

The compact agent contract intentionally does not include every behavior-affecting constructor option. For example `Diffusion.prediction_type` changes DDIM scheduler semantics, is not a model parameter, and is not part of `build_agent_contract()`. Editing `config.yaml.agent.prediction_type` after training can therefore change evaluation semantics while strict state-dict loading still succeeds.

### Required design

Make **checkpoint-saved `agent_config` the sole model-constructor source** for:

- `select_best_ckpt.py`
- `eval_best_ckpt.py`
- `record_demo.py`

Current experiment config may still own:

- experiment identity/path;
- `env_runner` and evaluation-environment configuration;
- evaluation seed/protocol settings;
- video settings;
- explicit inference ablations such as EMA/raw choice and denoise/NFE steps.

It must not override model-facing `agent.*` constructor semantics for an existing checkpoint.

### Recommended implementation

Refactor `training/eval_utils.py` so component construction is split into clear responsibilities.

A good shape is:

- build/checkpoint store + env runner from current evaluation config;
- resolve and load the selected training checkpoint;
- instantiate the Agent from `checkpoint.resume_contract["agent_config"]`;
- restore checkpoint-owned action and normalization contract before strict state load;
- load selected raw/EMA state strictly;
- validate the resulting Agent against the saved compact agent contract.

Do not require deployment export to run just to evaluate a checkpoint.

For DQ-RISE, the checkpoint is self-contained but the saved constructor may contain an external `codebook_path`. Offline checkpoint restore must not depend on that external file being present. Reuse the existing principle from deployment export: instantiate DQ-RISE without re-reading an external runtime codebook, then let strict checkpoint state restore the persistent `codebook_manager` buffers. Keep this logic narrow and explicit; do not duplicate the whole deployment export sanitizer.

Normalization semantics must also come from the checkpoint's saved normalization contract, not from current `config.yaml.normalization`.

### Override policy

Evaluation dotlist/CLI overrides may change evaluation/inference controls, but must not silently mutate checkpoint-owned Agent constructor config.

Preferred behavior:

- continue to accept documented `eval.*`, `env_runner.*`, video, seed, episode and NFE/EMA overrides;
- reject or ignore-with-explicit-error model-facing `agent.*` overrides for checkpoint evaluation;
- do not silently merge them into the model constructor.

Document the exact policy in CLI help and `docs/仿真评测机制.md`.

### Acceptance criteria

- Changing experiment `config.yaml.agent.prediction_type` after training cannot change the Agent restored for offline evaluation.
- An explicit NFE override still works without rewriting checkpoint architecture.
- EMA/raw selection still works exactly as before.
- DQ-RISE evaluation does not require the original external codebook file when the checkpoint contains the persistent codebook state.
- Existing env-runner/evaluation overrides remain usable.
- Strict state loading and saved contract validation remain fail-fast.

---

## 2. Fix remote workflow so it follows resolved dataset semantics

### Problem A — current MultiTask identity is rejected

The current MultiTask config uses:

`task_name: pick_bottle+open_box`

but `train_remote.sh`, `tail_log.sh`, and `sync_down.sh` validators reject `+`.

### Problem B — pre-flight guesses dataset path from task name

`train_remote.sh` checks:

`/data_ssd/ZHY/robot_data/${TASK}.zarr`

before launch.

That is not the source of truth. MultiTask uses multiple child Zarr paths, and ordinary Hydra overrides can also change `dataset.zarr_path`.

### Required design

Do not infer dataset paths from `task_name`.

Add a lightweight, config-only way for the remote launcher to resolve the actual dataset path(s) from the same Hydra config and overrides that will be used for training.

The pre-flight should:

1. resolve the requested config + overrides;
2. extract every actual Zarr path consumed by the dataset config:
   - single-task: `dataset.zarr_path`;
   - MultiTask: every child `dataset.datasets[*].zarr_path`;
3. map repository-relative paths to the configured remote persistent/project layout;
4. verify every required dataset exists;
5. report all missing datasets together.

Do not instantiate the dataset or read array data during this pre-flight.

### Naming/path validators

Update validators so the repository's own canonical composite task identity `pick_bottle+open_box` is valid where it is used as an experiment-directory component.

Keep path traversal protections:

- still reject absolute paths;
- still reject `..`;
- still reject shell metacharacters / uncontrolled whitespace;
- do not loosen validators beyond the characters actually required by repository-generated identities.

### Acceptance criteria

- `multitask_dit / pick_bottle+open_box` passes task/path validation.
- Remote pre-flight checks `pick_bottle.zarr` and `open_box.zarr`, not a fabricated `pick_bottle+open_box.zarr`.
- A CLI override of `dataset.zarr_path` changes the pre-flight target to the resolved path.
- A missing dataset aborts before launch with an error listing the exact missing path(s).
- `tail_log.sh` and `sync_down.sh` accept the canonical MultiTask experiment path without weakening traversal safety.
- Update `docs/SSH服务器训练部署.md` to describe resolved-dataset pre-flight instead of the old `${TASK}.zarr` assumption.

---

## 3. Unify train/eval window validation and validate dataset split ratios

### Window contract

Training currently does not require positive `horizon`, `n_obs_steps`, and `n_action_steps`, while evaluation already requires positive observation/action steps.

Create/reuse one shared validator for the common window contract:

- integer, not bool;
- `horizon >= 1`;
- `n_obs_steps >= 1`;
- `n_action_steps >= 1`;
- `n_obs_steps - 1 + n_action_steps <= horizon`.

Use it from both training and evaluation validation paths instead of keeping near-duplicate rules.

### `val_ratio`

Reject invalid split ratios instead of silently coercing them.

Required contract:

`0.0 <= val_ratio < 1.0`

Apply at the dataset/split boundary so callers outside Hydra validation also receive the same behavior.

Do not change the current documented behavior that normalization statistics are fitted on the complete replay buffer.

### Additional small split validation

Where low-risk and local, validate `max_train_episodes` as `None` or a positive integer before sampling. Do not silently turn negative values into odd NumPy sampling behavior.

### Acceptance criteria

- training config-only validation rejects zero/negative horizon/observation/action windows;
- eval uses the same shared rule;
- `val_ratio < 0` and `val_ratio >= 1` raise clearly;
- valid existing configs remain unchanged.

---

## 4. Repair DQ-RISE codebook example and remove the hard-coded 2-group restore assumption

### Broken example

`agents/core/dqrise.py::example()` manually writes an incomplete v3 NPZ and no longer satisfies `CodebookManager.load()`.

Replace manual schema construction with the canonical writer path. Prefer creating a synthetic `CodebookManager`, populating the minimum valid persistent state, and calling `CodebookManager.save()`, matching the principle already used by `smoke_test.py`.

Do not duplicate the v3 field list in the example.

### `num_groups=2` restore bug

`CodebookManager._load_from_state_dict()` currently forces:

`self.num_groups = 2`

and infers `codebook_size` by square root.

This contradicts the public `num_groups` / `codebook_size` API.

Best fix: derive and validate structure from persistent checkpoint state that actually identifies the group count. `layer_weights.numel()` is already persistent and should agree with `num_groups`; use it instead of a hard-coded constant. Validate that the number of poses is exactly `codebook_size ** num_groups`.

If structure cannot be inferred uniquely and safely, fail rather than guess.

Preserve the current default 2-group behavior.

### Acceptance criteria

- DQ-RISE module example creates a schema-valid temporary codebook through canonical APIs.
- Default 2-group checkpoints restore identically.
- A valid non-2-group synthetic state restores correct `num_groups` and `codebook_size`, or fails explicitly if the state is internally inconsistent.
- No legacy-format compatibility branch is added.

---

## 5. Make ffmpeg video encoding fail closed

In `env_runner/base_runner.py::_encode_video()`:

- check ffmpeg's exit status after `wait()`;
- a non-zero status must raise;
- capture enough stderr to produce an actionable error without dumping unbounded output;
- preserve the existing subprocess cleanup/timeout behavior;
- preserve the imageio fallback when ffmpeg is unavailable.

The caller may continue treating video encoding as best-effort, but it must not append a failed output path as if encoding succeeded.

---

## 6. Preserve immutable single-value evaluation results

Current single-NFE evaluation overwrites:

- `eval_dexsim/_result.txt`
- `eval_dexsim/result_details.json`

on subsequent runs.

Make evaluation artifacts immutable per invocation, consistent with sweep behavior.

Recommended design:

`eval_dexsim/<timestamp-or-run-id>/...`

for both single-value and sweep evaluation.

If a stable convenience pointer is useful, add a small selector/summary mechanism that does not destroy prior runs. Do not create symlink behavior unless it is already consistent with repository portability assumptions; a small JSON/text latest pointer is acceptable.

Update:

- `eval_best_ckpt.py`;
- `eval_pipeline.sh` final summary;
- `docs/仿真评测机制.md`.

The output must still make the exact checkpoint, EMA/raw choice, NFE, seed set and success metrics discoverable.

---

# Phase 2 — Low-risk code cleanup

## 7. Fix `clean_experiments.sh` completion off-by-one

The script's documented incomplete definition is “completed steps < total_train_steps”.

Change the actual condition accordingly.

Be careful about the repository's checkpoint naming semantics: a fully completed run reaches a checkpoint at `global_step == total_train_steps`.

Do not broaden deletion rules.

---

## 8. Fix resume progress-bar accounting

The Trainer initializes tqdm with `initial=global_step` but later updates by a fixed `log_interval_steps`, which over-counts when resuming from a non-log boundary.

Update progress by the actual delta between the current global step and the progress bar's current count, or update one step at each optimizer boundary.

Do not alter training-step semantics, logging cadence, scheduler stepping, or checkpoint timing.

---

## 9. Remove deployment parity dead code left by deleted `qualify.py`

Confirmed no external consumer remains for:

- `PredictionParityError`;
- `assert_prediction_parity()`;
- `_MAX_PARITY_TOLERANCE`;
- the tolerance helper(s) used only by that dead comparison path.

Remove only that dead parity-comparison subtree.

Do **not** remove:

- `PredictionSnapshot`;
- `prediction_snapshot()`;
- `validate_prediction()`;
- `verify_deployment_prediction()`;
- deterministic observation/restore helpers still used by export/runtime.

Update the `restore.py` module description so it no longer advertises a deleted direct/export parity subsystem.

---

## 10. Remove dead Trainer step alias without changing persisted checkpoint schema

`Trainer` stores both:

- `self.total_train_steps` — active;
- `self.num_training_steps` — assigned but not read.

Remove the dead Trainer constructor parameter/attribute and simplify callers.

However, do **not** change `simple.v3` persisted schema in this task.

The redundant `resume_contract.training.num_training_steps` alias and `TrainCheckpoint.monitor` are known schema debt. Leave them serialized for compatibility and add concise comments if needed to make that intentional compatibility status explicit.

Do not introduce a checkpoint-format bump.

---

## 11. Remove or repair misleading unused API parameters

### `SATBackbone.num_obs_tokens`

It is accepted but not consumed.

Choose one:

- remove it from the constructor and caller if it is not part of the actual SAT invariant; or
- use it for a real shape/token-count validation if the architecture depends on it.

Do not leave a no-op public parameter.

### `KNNGrouper.forward(use_fps=True)`

The implementation always performs FPS.

Choose one:

- remove `use_fps`; or
- implement the false branch if it is actually required by a current caller.

Prefer removal if no caller requests `False`.

Do not change point-cloud sampling semantics for existing configs.

---

## 12. Make module examples fail honestly

For the RGB module examples that currently catch `Exception`, print, and return success:

- CLIP;
- DINO;
- ResNet;
- SigLIP.

Either re-raise after diagnostic output or remove the broad catch.

Do not let a failed example exit with status 0.

Keep the examples lightweight; do not build a new test framework.

Also update the three dataset module examples that still use the stale `robot_data/sim/...zarr` path convention to the current `robot_data/<task>.zarr` convention, or require an explicit path argument instead of hard-coding a stale layout.

---

# Phase 3 — Comment and documentation synchronization

Treat current code/config as source of truth. Do not preserve prose that describes removed subsystems.

## 13. `docs/项目架构.md`

Remove/update stale descriptions of:

- `deployment/qualify.py`;
- direct/export parity as a current subsystem;
- checkpoint-directory fsync publication;
- midpoint ODE solver;
- Dataset-owned `get_normalizer` flow.

Replace normalization architecture with the current model:

`top-level normalization config -> build_normalizer()/streaming dataset statistics -> normalizer loaded into Agent -> normalization contract persisted in checkpoint`.

Describe deployment publication according to the current code:

`checkpoint-owned semantics -> build candidate -> safe reload -> strict restore -> synthetic prediction -> atomic selector swap`.

Do not document removed reconciliation or rollback systems.

---

## 14. `docs/仿真评测机制.md`

Update the evaluation architecture to state explicitly:

- Agent constructor semantics are checkpoint-owned;
- current experiment config owns env/evaluation protocol, not historical model architecture;
- EMA/NFE are explicit inference selections/ablations;
- same seed means deterministic seed selection and RNG reseeding, **not** a guarantee of bitwise-identical trajectories across GPU/driver/kernel environments;
- single-value evaluation now stores immutable per-run results.

Ensure wording matches the implementation after Phase 1.

---

## 15. `docs/SSH服务器训练部署.md`

Update remote pre-flight documentation:

- no longer claim dataset existence is checked by `<task>.zarr`;
- describe resolved dataset path(s);
- describe MultiTask child-dataset checking;
- document allowed composite experiment task identity;
- keep source/data/experiment ownership boundaries unchanged.

---

## 16. Fix stale code/config comments

Update these confirmed stale statements without changing runtime values:

### `record_demo.py`

Remove the claim that the best-selection record contains a “temporal-ensemble coefficient”.

Current best inference record contains EMA choice, denoise steps and policy seed mode.

### `agents/core/multi_task.py`

Replace the obsolete `FlowMatch` comment with the current `RectifiedFlow` naming.

### SAT backbone/core/config

Remove “same as ManiFlow” claims where SAT uses `pointnext_tokenizer` while ManiFlow uses `pointnet_dense`.

Replace obsolete “FlowMatch protocol” terminology with the actual RectifiedFlow/action-decoder interface wording.

### `common/normalizer.py`

Remove references to deleted config/checkpoint “reconciliation”.

### `configs/multitask_dit.yaml`

Fix comments that currently claim:

- R3M ResNet-18 when the active recipe is the regular `resnet` preset;
- 50% state-noise trigger when the active `prob` is 1.0.

### `configs/dqrise.yaml`

Fix top-level representation description for the active default:

- `action_key=action_ee`;
- `tcp_dim=9`;
- reduced diffusion output is `tcp_dim + 1 = 10` dimensions.

Do not describe its active encoder as “identical to DP3” when the active encoder selections differ.

### `configs/r3d.yaml`

Clarify the control-dimension comment so it is conditional on the actual action mode / `use_aux_ee` behavior rather than saying control is always joint-only.

### `rgb/base.py`

Fix `ViTEncoder` docstring ordering: current subclasses call `super().__init__()` before setting model-specific attributes.

### `state_mlp.py`

Do not claim an incomplete enumerated list is “all observation encoders”; include SAT or use generic wording.

### `R3DObsEncoder` and `SATObsEncoder`

Move intended class documentation to the actual class-docstring position before properties/methods.

---

## 17. Normalize best-checkpoint schema terminology

Current code/help repeatedly says “strict v2 `best_ckpt.json`”, but the JSON record has no explicit version field.

For this task, do **not** introduce a new persisted schema version.

Instead:

- call it the “strict/current best_ckpt.json schema”;
- remove misleading “v2” wording from CLI help, docstrings and docs.

Keep the current exact-key validation behavior.

---

# Phase 4 — Validation and acceptance

There is intentionally no `tests/` directory. Do not recreate it.

Use existing lightweight validation surfaces and temporary command-line checks.

## Required static validation

1. Python syntax:
   - run `python -m compileall dexmani_policy` (and the Python utility under `scripts/data/` if modified).
2. Shell syntax:
   - run `bash -n` on every modified shell script.
3. Search for stale removed-subsystem terms in active docs/comments:
   - `qualify.py`;
   - temporal-ensemble coefficient;
   - midpoint ODE where it is described as currently implemented;
   - Dataset `get_normalizer` current architecture;
   - checkpoint/config reconciliation as a current mechanism;
   - strict v2 `best_ckpt.json`.
   Remaining hits must be intentional historical/negative statements.

## Config validation

Run config-only smoke validation for all current base configs:

- dp
- dp3
- dqrise
- maniflow
- multitask_dit
- r3d
- sat

and current DDP overlays where the smoke entry supports them.

Verify invalid temporary overrides fail as intended:

- `n_obs_steps=0`;
- `n_action_steps=0`;
- invalid `val_ratio`;
- valid baseline configs remain accepted.

Do not commit temporary invalid configs.

## Targeted behavioral validation

Use temporary, non-committed checks where full simulator/data is unavailable.

At minimum verify:

### Checkpoint-owned evaluation constructor

Construct/save a small checkpoint through existing repository utilities, mutate the in-memory/current config's behavior-only Agent field (for example diffusion `prediction_type`), then verify the evaluation restore path still instantiates the checkpoint-saved value.

The regression must prove the specific bug is fixed; strict state loading alone is not sufficient evidence.

### DQ-RISE codebook

Exercise:

- canonical temporary v3 codebook creation;
- default 2-group state-dict restore;
- non-2-group metadata reconstruction or explicit inconsistency failure.

### Video encoder

Where ffmpeg is available, use a controlled failure invocation or temporary fake executable to confirm non-zero exit is surfaced. Do not depend on a real simulator.

### Remote config resolution

Use dry-run/config-resolution paths to demonstrate:

- single-task path extraction;
- MultiTask child path extraction;
- composite task name acceptance;
- path traversal rejection.

### Clean-experiments boundary

Use temporary directories/files or a dry-run fixture outside tracked source to prove:
- `max_step == total_steps` is complete;
- `max_step == total_steps - 1` is incomplete.

### Resume tqdm

Use a minimal local object/temporary invocation to confirm a non-log-boundary resume does not over-count.

---

# 5. Final self-review checklist

Before finishing, inspect the full diff and explicitly verify:

- [ ] No dependency/version/environment files changed.
- [ ] No `tests/` directory was created.
- [ ] No checkpoint format bump or `simple.v3` schema deletion occurred.
- [ ] Offline eval model semantics now come from checkpoint `agent_config`.
- [ ] Evaluation env/protocol overrides remain separate and functional.
- [ ] DQ-RISE checkpoint restore remains self-contained.
- [ ] MultiTask remote flow checks real child dataset paths.
- [ ] Path validation remains traversal-safe.
- [ ] Training/eval window validation is shared.
- [ ] Invalid `val_ratio` fails instead of coercing.
- [ ] Failed ffmpeg encoding cannot be reported as success.
- [ ] Single-value eval no longer destroys prior result artifacts.
- [ ] Deployment runtime helpers still used by export/runtime were not removed.
- [ ] Dead parity-only helpers are gone.
- [ ] Trainer step cleanup did not alter scheduler/training semantics.
- [ ] Comments/docs describe only currently implemented behavior.
- [ ] No unrelated formatting churn.

---

# 6. Suggested commit structure

Prefer small logical commits so regressions are bisectable.

1. `fix(eval): restore checkpoint-owned agent semantics`
2. `fix(remote): validate resolved dataset paths and multitask identity`
3. `fix(config): unify window and split validation`
4. `fix(dqrise): repair codebook restore and example`
5. `fix(eval): fail closed on video encoding and preserve result history`
6. `fix(tools): correct cleanup and resume progress accounting`
7. `refactor: remove confirmed dead parity and trainer state`
8. `docs: sync architecture comments and workflow documentation`

If implementation dependencies make a different grouping cleaner, keep each commit single-purpose.

---

# 7. Deferred items — do not implement in this task

These were reviewed but are deliberately deferred to avoid mixing correctness work with large performance/refactor changes:

- EMA Agent construction currently instantiates a second full Agent;
- DDP ranks independently load full Zarr data and fit normalizers;
- ReplayBuffer eager full-array loading/min-max logging cost;
- broad consolidation of 20+ module-level examples;
- R3M/ResNet common-base refactor;
- ViT example deduplication;
- orphan/future research modules such as token compressor / DiTXRMS / T5;
- removal of persisted `monitor` or duplicate resume-contract fields requiring a checkpoint schema decision;
- any dependency/environment/version cleanup;
- any restoration of the intentionally removed `tests/` tree.

Do not expand scope into these items unless a required correctness fix is impossible without a very small supporting change.

---

# 8. Definition of done

The task is complete when:

1. all Phase 1 correctness issues are fixed with explicit regression evidence;
2. Phase 2 cleanup is implemented without persisted-schema changes;
3. Phase 3 documentation/comments match the resulting code;
4. all required validation commands succeed or any unavailable external simulator/data validation is clearly reported as not runnable;
5. the final diff contains no unrelated dependency, test-tree, environment, model-architecture, or performance refactor changes;
6. the final report lists:
   - changed files;
   - behavior changes;
   - validation commands and results;
   - any intentionally deferred findings.
