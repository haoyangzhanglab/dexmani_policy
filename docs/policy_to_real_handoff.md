# Policy to Real deployment handoff

## 1. Scope

This document freezes the Policy PR-4 / Phase P3 handoff. It adds evidence and
handoff metadata only; it does not change exporter, restore, model, artifact, or
Real runtime semantics. The machine-readable receipt is
[`policy_to_real_handoff.json`](policy_to_real_handoff.json).

The representative checkpoint and Zarr remain outside Git. No checkpoint,
dataset, video, hardware identifier, hostname, username, or workstation path is
committed.

## 2. Frozen producer

| Field | Value |
|---|---|
| Repository | `haoyangzhanglab/dexmani_policy` |
| Commit | `fc6b7dfb45748f4187f2e82b5425721ed02b028e` |
| Working tree | clean |
| Origin | `https://github.com/haoyangzhanglab/dexmani_policy.git` |

This is the PR-3 merge commit on `main`, not the PR-4 documentation commit.
The representative artifact was generated from a detached clean worktree at
this exact commit.

Frozen provenance evidence:

```bash
git -C "$PRODUCER_ROOT" rev-parse HEAD
git -C "$PRODUCER_ROOT" status --short
git -C "$PRODUCER_ROOT" remote get-url origin
```

Result: the commit matched, status output was empty, and origin was canonical.

## 3. Frozen Real consumer

| Field | Value |
|---|---|
| Repository | `haoyangzhanglab/dexmani_real` |
| Commit | `f758f266f85fb6d73547e7965275eb95831347b3` |
| Branch | `main` |
| Working tree | clean |

The consumer source at this commit accepts checkpoint format
`dexmani.deployment.v2` and sidecar `schema_version=2`. Real remained read-only
throughout the handoff.

Real provenance evidence:

```bash
git -C "$REAL_ROOT" rev-parse HEAD
git -C "$REAL_ROOT" status --short
```

Result: the commit matched and status output was empty.

## 4. Artifact identity

Representative real checkpoint qualification:

| Field | Value |
|---|---|
| Policy / task | DP3 / `pick_place_toy` |
| Checkpoint selector | `100pct` |
| Source checkpoint | `epoch=1126-step=00080000-milestone=100pct.pt` |
| Source checkpoint SHA-256 | `0e5615cc3be4e5299791aae24c412df3667027b06ade6cad266be48e50150e84` |
| Deployment checkpoint | `epoch=1126-step=00080000-milestone=100pct-deployment-v2.pt` |
| Deployment checkpoint SHA-256 | `28ff79a6ca5d5b746bbde877ff96abbb88543539f4c73ef554348184f446effc` |
| Sidecar schema version | 2 |
| Sidecar SHA-256 | `721ba4de21977d5591ef24afb93aecb526af1a9fa60990e85b315cd659db1f7a` |
| Embedded contract SHA-256 | `eca6b7857428b8146323bc80d55c67d9b0298f34b04d9489ee093d8ab705338d` |
| Selected weights | `ema_model` (`use_ema=true`) |
| Action contract | `action`, model/control dimension 19 |

The experiment has no `best.pt` or `best_ckpt.json`. The evidence therefore
uses the explicit 100% milestone selector; it does not request `latest` and no
best-to-latest fallback is possible.

The committed Layer A generator is
`tests/deployment/real_restore_fixture.py`. Two independent generations produced
identical checkpoint and sidecar bytes:

```text
fixture checkpoint SHA-256 = fb8a6d8618398f51b48ddfcd1412d0079076d77765a3e2b2b10fcf9b46399156
fixture sidecar SHA-256    = 8e8a87ba4f453226241b687fae5113e134e0735d789d5e5c78441b7f5273daf0
embedded contract SHA-256 = 4dc9b68458de1ae24997a7540b4461c72f2c59b2ac2a2c9e4601730ab4d5858f
```

## 5. Export and qualification command

The formal run used the frozen producer worktree, the representative real DP3
experiment, its Real Policy Zarr v5, CPU inference, and exact tolerances:

```bash
PYTHONPATH="$PRODUCER_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n policy python -m dexmani_policy.deployment.qualify \
  "$DEXMANI_POLICY_HANDOFF_EXP" \
  --checkpoint 100pct \
  --zarr-path "$DEXMANI_POLICY_HANDOFF_ZARR" \
  --device cpu \
  --atol 0 \
  --rtol 0
```

The source experiment and checkpoint were copied into the handoff temporary
directory before this command, so the generated deployment artifact and selector
did not modify the user's experiment.

## 6. Strict restore result

PASS. The frozen producer strict-restored the exported EMA state and performed a
deterministic prediction. Fresh-process current Real checks also passed:

```text
resolve_policy_artifact
→ sidecar-v2 validation
→ load_deployment_checkpoint_stream(weights_only=True)
→ precheck_policy_package_provenance
→ Policy import
→ Hydra instantiate
→ strict load_state_dict
→ manifest validation
→ normalizer validation
→ isolated preflight prediction
```

The accepted package commit was the frozen producer commit, checkpoint SHA-256
verification was true, and `package_dirty=false`.

Both checks used import-clean Python processes and the current Real runtime
configuration; neither used `--synthetic-runtime`:

```bash
PYTHONPATH="$PRODUCER_ROOT:$REAL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n policy python \
  "$PRODUCER_ROOT/tests/deployment/real_restore_probe.py" \
  --experiment "$DEXMANI_POLICY_HANDOFF_EXP" \
  --mode direct

PYTHONPATH="$PRODUCER_ROOT:$REAL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
conda run -n policy python \
  "$PRODUCER_ROOT/tests/deployment/real_restore_probe.py" \
  --experiment "$DEXMANI_POLICY_HANDOFF_EXP" \
  --mode preflight
```

Direct result summary: producer commit accepted; manifest action/control
dimensions were 19; `n_obs_steps=2`; `n_action_steps=8`; normalizer dimensions
were action19, joint-state19, point-cloud6. Preflight result summary: checkpoint
SHA verified, producer commit accepted, package clean, isolated prediction
accepted, and legacy receipt `action_steps=15` observed.

## 7. Direct/export parity

PASS with exact equality:

| Measurement | Result |
|---|---:|
| `pred_action_max_abs_diff` | 0.0 |
| `control_action_max_abs_diff` | 0.0 |
| `canonical_slice_max_abs_diff` | 0.0 |
| `atol` | 0.0 |
| `rtol` | 0.0 |

Resolved parameters were `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`,
`action_key=action`, `action_dim=19`, and `control_action_dim=19`.

## 8. No-network and no-external-asset result

PASS. PR-3 deployment tests run restore under the existing fail-fast network
guard, covering socket, urllib, Torch Hub, Requests, Hugging Face Hub, and gdown
entry points when installed. The genuine DQ-RISE fixture deletes its training
codebook before strict restore; the genuine R3D fixture removes its pretrained
file and disables the pretrained loader before strict restore. Both preserve
prediction parity. The representative DP3 deployment config has no external
training-only constructor asset.

The fresh Real probes also verified that restore did not import `pyrealsense2`,
`xarm`, `dexmani_sim`, Real camera/robot/sensor workers, shared-memory owners, or
the deployment coordinator/lifecycle/worker modules. No hardware path ran.

Relevant regression commands:

```bash
conda run -n policy python -m unittest discover \
  -s tests/deployment -p 'test_network_guard.py' -v
conda run -n policy python -m unittest discover \
  -s tests/deployment -p 'test_self_contained_assets.py' -v
```

Both passed with no skips.

## 9. Supported-policy matrix

`dexmani_policy/deployment/qualification_matrix.py` is the code-owned evidence
source. This table is a handoff snapshot, not another registry.

| Status | Policies |
|---|---|
| Qualified | DP3 |
| Conditional | ActionFlow, DQ-RISE, R3D |
| Deferred | ManiFlow, SAT, DP RGB, MoE-DP RGB, MultiTask DiT |
| Rejected | none |

DP3 qualification applies to the evidenced real action-space checkpoint. Every
newly selected checkpoint must repeat per-artifact parity and Real restore.

## 10. Action semantics

The representative qualified DP3 artifact has:

```text
horizon = 16
n_obs_steps = 2
n_action_steps = 8

required_action_steps
= horizon - (n_obs_steps - 1)
= 15

prediction_future_steps = 15
executable_control_steps = 8
```

These numbers describe this representative artifact. They are not global
DexMani deployment protocol constants.

### Runtime temporal invariants

For every accepted deployment artifact, Real derives the temporal values from
the artifact and restored manifest:

```text
control_start
= n_obs_steps - 1

required_action_steps
= horizon - (n_obs_steps - 1)

prediction_future_steps
= required_action_steps

executable_control_steps
= n_action_steps
```

`horizon`, `n_obs_steps`, and `n_action_steps` are artifact-driven. They must be
read from the deployment artifact/restored manifest; they are not numeric
constants in `dexmani_real`.

`pred_action` is the full model output. `control_action` is the only default
executable Policy output. `tail` is not a Real execution contract.

```text
prediction future steps != executable control steps
Real must execute only control_action
```

For this representative artifact, frozen Real consumer
`f758f266f85fb6d73547e7965275eb95831347b3` reports `action_steps=15`. This is
an observed legacy result for this artifact and the current Real
allocation/executable conflation. It is not a protocol constant. PR-4 records it
and does not change it.

## 11. Known conditional and deferred policies

- ActionFlow remains conditional until genuine-agent parity can run with the
  required PyTorch3D point operators.
- DQ-RISE remains conditional on a complete seven-buffer runtime codebook and a
  matching policy/codebook hand normalizer in each selected checkpoint.
- R3D remains conditional on strict restore with
  `use_pretrained_weights=false`.
- ManiFlow and SAT have no PR-3 direct/export parity evidence.
- DP RGB and MoE-DP RGB remain deferred until deterministic deployment
  preprocessing and external-initialization contracts exist.
- MultiTask DiT remains deferred until task-text semantics are explicit.

No conditional or deferred policy is promoted by this handoff.

## 12. Gate decision for Real R1

```text
READY FOR REAL R1
```

After PR-4 review, merge, and handoff acceptance, Real R1 may validate the full
`pred_action`, validate the exact `control_action`, and execute only
`control_action`. Real R1 must replace the semantic use of
`required_action_steps` as the executable length with artifact-derived
`n_action_steps`. It must **not** implement this as a numeric 15-to-8 hard-coded
rewrite.

For another valid artifact:

```text
horizon = H
n_obs_steps = O
n_action_steps = A

required_action_steps = H - (O - 1)
executable_control_steps = A
```

No Real R1 implementation is part of this PR.

Regression evidence at handoff creation:

| Command | Result |
|---|---|
| `python -m compileall -q dexmani_policy tests` | PASS |
| `test_action_flow_config.py` | PASS, 1 test |
| `test_checkpoint_correctness.py` | PASS, 5 tests |
| deployment `test_*.py` | PASS, 66 tests; SKIP, 2 tests; FAIL, 0 |
| opt-in real-checkpoint full preflight rerun | PASS, 1 test |

The two default-suite skips were the unavailable PyTorch3D ActionFlow parity
case and the opt-in real-checkpoint integration without environment variables.
The latter was then rerun with the frozen producer, representative artifact, and
current Real consumer and passed. No CUDA, simulation, video, or hardware test
was run or claimed.
