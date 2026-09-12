---
name: dexmani-agent-integration
description: >
  Workflow for adding or materially changing a DexMani policy. Use for a new
  policy, new agent variant, architecture replacement, action-representation
  change, or policy-level integration work.
---

# DexMani Agent Integration

This skill defines the **process**, not the current policy catalog. Treat the resolved Hydra config and implementation as the source of truth.

## 1. Discover the real implementation

Start from the target config:

```text
dexmani_policy/configs/<config>.yaml
        ↓
agent._target_
        ↓
actual Agent class/module
```

Do not infer the module/class from the config name. Search existing configs and implementations for the closest semantic pattern only after resolving the target.

For an existing experiment, read its saved `config.yaml` first; the current repository config may have changed since training.

## 2. Define the policy contract

Before editing, state the intended design in five parts:

- **Observation Representation**
- **Policy Architecture**
- **Action Representation**
- **Training Objective**
- **Inference Algorithm**

Also identify the environment-facing `control_action` contract and any external artifact/dependency required by the policy.

This prevents accidental architecture drift from copy-pasting a superficially similar policy.

## 3. Trace interfaces end to end

Follow the actual call chain:

```text
dataset
→ Agent preprocessing / obs_encoder
→ backbone / action_decoder
→ compute_loss
→ predict_action
→ control_action
→ env runner
```

Check tensor semantics and shapes at interface boundaries. Reuse shared modules only when their semantics match; keep policy-specific behavior local.

Avoid speculative framework abstractions. One exceptional policy does not justify a new registry/plugin layer.

## 4. Implement and configure

Typical additions are:

- Agent / policy-local modules
- one Hydra config
- optional DDP overlay when DDP support is actually required
- focused regression coverage where useful

Hydra `_target_` is the wiring mechanism. Do not add a separate policy registry.

Do not blindly copy another config's optimizer, eval, horizon, action dimensions, or decoder settings. Resolve what the new policy actually requires and let repository-level validation enforce shared constraints.

## 5. Validate cheaply first

Run:

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

This is the first acceptance gate for a new/changed config.

Then, when data/GPU are available:

```bash
python dexmani_policy/smoke_test.py <config_name>
```

If shared code changed, search for real dependents and run additional targeted smoke tests. Do not run unrelated policies just to increase coverage.

## 6. Documentation rule

A normal policy addition or policy-local architecture/hyperparameter change should not require edits to:

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- project Skills

Update global files only when the change modifies a public CLI, repository-wide contract, common workflow, environment requirement, or other global interface.

`docs/` is frozen unless the user explicitly asks to change it.

## Reporting

Report:

- changed files
- the five-part policy contract
- config-only result
- full smoke result, or why it was not run
- additional dependent-policy checks for shared changes
- remaining unverified items

Never report PASS for a command that was not executed.
