---
name: dexmani-training-debug
description: >
  Evidence-driven workflow for diagnosing NaN/Inf or other numerical training
  failures in DexMani_Policy.
---

# DexMani Training Debug

Diagnose the current failure from logs, checkpoint state, config and the actual model path. Do not rely on fixed line numbers or policy-specific folklore.

## 1. Establish evidence

Collect:

- experiment directory and saved resolved `config.yaml`
- exact exception/error text
- recent `metrics.jsonl`
- any NaN debug checkpoint
- whether failure occurs at startup or after successful optimization steps

Use the experiment config, not the current repository config, to understand what actually ran.

## 2. Classify the failure

Determine whether the first non-finite signal is:

- input/data/normalization
- forward loss/output
- backward gradient
- optimizer/state after an update

Use the current trainer implementation and error message to identify which guard fired. Avoid assuming a cause from the policy family alone.

## 3. Trace the failing computation

Follow the active experiment's:

```text
config
→ agent._target_
→ obs_encoder
→ backbone/action_decoder
→ compute_loss
```

Check finite values and scales at the narrowest useful boundaries. For gradient failures, use the reported parameter/module names to localize the backward path.

Inspect checkpoint weights/state only when it helps distinguish accumulated corruption from a one-step computation failure.

## 4. Reproduce minimally

Prefer the smallest reproduction that still reaches the failing path:

- config-only validation for config errors
- targeted smoke for integration errors
- a short training reproduction for optimization/numerical errors

Do not launch a full training run as the first diagnostic step.

## 5. Fix the root cause

Make the smallest coherent correction to the actual failing computation or contract.

Do not apply generic recipes such as changing LR, precision, time sampling or scheduler merely because they sometimes help a similar model. Each such change must follow evidence from the failing run.

## 6. Validate and resume

Re-run the minimal reproduction, then the closest targeted smoke/check.

Training resume is explicit:

```bash
bash scripts/training/train.sh <config_name> \
  'task_name=<task>' \
  '+resume_from=/absolute/path/to/experiment_or_checkpoint'
```

Only resume when the saved resume contract remains compatible with the intended run; otherwise start a new experiment intentionally.

## Reporting

Report:

- first non-finite location/category
- evidence used
- root cause
- changed files
- reproduction/validation commands and results
- whether resume is compatible
- remaining unverified items
