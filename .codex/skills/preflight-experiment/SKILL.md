---
name: preflight-experiment
description: Sanity-check a DexMani training, resume, or evaluation run before spending GPU time.
---

# Preflight Experiment

Use this before an expensive training/evaluation run, or when a run command/config looks suspicious.

## Check

1. Resolve the intended config/overrides and confirm the actual `agent._target_`.
2. Check required observation keys/modalities, history/window/horizon, action dimensions, and obvious shape contracts.
3. Check normalization and the encoder → backbone → action-decoder interfaces touched by the config.
4. Check training vs inference settings that can silently diverge, especially checkpoint/EMA choice and `inference_steps`.
5. Check dataset/task paths and whether the requested modalities are actually available.
6. For resume/evaluation, check that the experiment config and checkpoint belong together and that the requested restore mode is valid.

## Validate

Run `python dexmani_policy/smoke_test.py --config-only <config_name>` when applicable.
Run a full smoke only when it materially tests the changed path and the required GPU/data environment is available.
Do not start the real training/evaluation job as part of preflight unless the user explicitly asks.

## Output

Return a short result with:

- `PASS`: verified and ready;
- `BLOCKER`: concrete issue that should be fixed first;
- `NOT VERIFIED`: checks blocked by missing GPU/data/weights/environment;
- the exact proposed run command.

