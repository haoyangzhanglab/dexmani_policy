---
name: dexmani-pr-check
description: >
  Read-only, diff-aware pre-PR review for DexMani_Policy. Use for review,
  audit, check-before-PR, or correctness validation of a change set.
---

# DexMani PR Check

Review the actual diff and validate only the affected contracts. Do not maintain or enforce a static policy inventory.

## 1. Determine blast radius

Inspect changed files first.

Classify changes as:

- policy-local config/Agent/component
- shared Agent/base/encoder/decoder component
- dataset/data contract
- training/resume/checkpoint
- evaluation/deployment
- scripts/public CLI
- documentation/agent infrastructure

For a policy config, follow its resolved `agent._target_`; do not infer implementation from filenames.

For shared code, search imports/config targets to identify real dependents instead of guessing from a hard-coded policy list.

## 2. Review correctness

For each affected path, check the relevant interfaces:

```text
config
→ construction
→ observation representation
→ policy/action representation
→ training objective
→ inference
→ environment-facing output
```

Look for concrete defects such as:

- unresolved Hydra interpolation or `_target_`
- inconsistent dataset/action/env contracts
- wrong tensor semantics across module boundaries
- objective/model-output mismatch
- inference shape or control-action mismatch
- missing optimizer coverage
- broken resume/eval/deployment state contract
- accidental cross-policy behavior changes
- unrelated refactor that enlarges risk without serving the task

Do not enforce universal architecture formulas that are not actually repository contracts.

## 3. Validate in cost order

For each changed/affected config:

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

For model/config integration changes, when data/GPU are available:

```bash
python dexmani_policy/smoke_test.py <config_name>
```

For shared modules, add targeted dependent configs based on actual usage.

Do not mark a command-based check PASS unless it was run. Environment-limited checks are **NOT VERIFIED**, not PASS.

## 4. Documentation review

Global docs should not duplicate policy-specific layer counts, hidden dims, LR/WD, NFE, batch size, parameter counts, policy lists, or experiment conclusions.

A normal policy addition should not require global documentation changes. Global documentation is expected only for public CLI/repository-contract/workflow changes.

`docs/` is frozen unless explicitly requested.

## Report format

Order findings by severity:

- **BLOCKER** — likely incorrect behavior/data/state contract
- **MAJOR** — significant regression or incomplete integration
- **MINOR** — real maintainability/clarity issue
- **NOT VERIFIED** — relevant validation blocked by environment

For each finding, give file/path, evidence, consequence, and smallest coherent fix.

End with:

- `ready for PR`, or
- `needs fixes`

Keep passing checks concise.
