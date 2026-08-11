---
name: dexmani-pr-check
description: >
  Read-only pre-PR audit. Validates config invariants (horizon/n_obs_steps/
  n_action_steps), dimension chains (action_dim/state_dim/tcp_dim), DDP
  coverage, and CLAUDE.md consistency. Works alongside smoke_test.py
  (build chain) by adding config- and documentation-level checks.
  Use when: preparing a PR, asked to "review my changes", "check before
  PR", or "audit the codebase".
---

# DexMani PR Check

Audit the codebase against DexMani_Policy standards and report pass/fail per item with
concrete fixes. This is a **read-only audit**: report the fixes, do not apply them unless
the user asks. Works alongside `smoke_test.py` (build-chain integrity) by adding config-level
and documentation-level verification.

Run commands from the project root unless noted.

## Checks

### 1. Core invariants

These four constants must hold across **all** config files (single-GPU + DDP).
Run from the project root:

```bash
# horizon must be 16 everywhere
grep -rn 'horizon:' dexmani_policy/configs/*.yaml dexmani_policy/configs/ddp/*.yaml | grep -v ': 16'

# n_obs_steps must be 2 everywhere
grep -rn 'n_obs_steps:' dexmani_policy/configs/*.yaml dexmani_policy/configs/ddp/*.yaml | grep -v ': 2'

# n_action_steps must be 8 everywhere
grep -rn 'n_action_steps:' dexmani_policy/configs/*.yaml dexmani_policy/configs/ddp/*.yaml | grep -v ': 8'

# use_aux_ee and use_faas cannot both be true (mutually exclusive)
grep -rn 'use_aux_ee.*true' dexmani_policy/configs/*.yaml
grep -rn 'use_faas.*true' dexmani_policy/configs/*.yaml
```

For the `use_faas` / `use_aux_ee` check: if a config has `use_aux_ee: true`, verify it does
NOT also have `use_faas: true`. `_validate_faas_config()` in `build_utils.py` catches this
at runtime, but checking at the config level catches it earlier.

### 2. Dimension chain consistency

For each config file, parse the YAML and verify:

| Rule | Non-FAAS | FAAS |
|------|----------|------|
| `action_dim = tcp_dim + hand_dim` | `tcp_dim + 12` | `tcp_dim + 32` |
| `state_dim` | `19` (7 arm + 12 hand) | `39` (7 arm + 32 FAAS) |
| `tcp_dim` | `7` (joint) or `9` (action_ee) | same |
| `use_aux_ee` → `action_dim` | `19 + 9 = 28` | N/A (mutually exclusive) |

Implementation (one-liner to extract key dims from a config):
```bash
python -c "
import yaml, sys
d = yaml.safe_load(open('$CONFIG'))
a = d.get('agent', d)
print(f'action_dim={d.get(\"action_dim\")} state_dim={a.get(\"state_dim\")} tcp_dim={d.get(\"tcp_dim\")} hand_dim={d.get(\"hand_dim\")} use_faas={d.get(\"use_faas\")} use_aux_ee={d.get(\"use_aux_ee\")} action_key={d.get(\"action_key\")}')
"
```

### 3. DDP coverage

List available configs and cross-reference:

```bash
echo "=== Single-GPU configs ==="
ls dexmani_policy/configs/*.yaml | grep -v '/ddp/' | sed 's|.*/||; s|\.yaml$||'
echo "=== DDP configs ==="
ls dexmani_policy/configs/ddp/*.yaml 2>/dev/null | sed 's|.*/ddp/||; s|\.yaml$||'
```

Known intentional gaps (not errors): `dp3` and `moe_dp` — CLAUDE.md documents these as
single-GPU-only.

### 4. CLAUDE.md table consistency

Three counts must match:
- Agent rows in the comparison table in CLAUDE.md
- Single-GPU config YAML files (`ls dexmani_policy/configs/*.yaml | grep -v ddp/ | wc -l`)
- Agent core Python files (`ls dexmani_policy/agents/core/*.py | grep -v '__init__\|base\.py' | wc -l`)

Also verify:
```bash
# Each config's _target_ points to a real class file
for cfg in dexmani_policy/configs/*.yaml; do
  target=$(grep '_target_:' "$cfg" | head -1 | sed 's/.*_target_: *//')
  module_path=$(echo "$target" | sed 's|\.|/|g; s|/agents/|dexmani_policy/agents/|')
  [ -f "${module_path%%.*}.py" ] || echo "MISSING: $target in $cfg"
done
```

### 5. Changed-file smoke test

```bash
# Identify configs changed in the current diff
git diff --name-only HEAD | grep 'dexmani_policy/configs/.*\.yaml' | while read cfg; do
  name=$(basename "$cfg" .yaml)
  echo "=== Smoke testing $name ==="
  python dexmani_policy/smoke_test.py "$name" || echo "FAILED: $name"
done
```

### 6. Known convention violations audit

These are design conventions documented in CLAUDE.md that look like bugs but are
intentional. Flag any change that "fixes" one of these as a regression:

- Normalizer fits on **all** replay buffer data (train + val) — not a leak, `limits` mode
  means val doesn't change min/max. Every codebase in this ecosystem does this.
- `tcp_dim` naming: means "arm control dim" (7 for joint, 9 for ee mode), not literally TCP.
- `MoEAgent.forward()` returns `dict` (with `aux_loss`); all other agents return `Tensor`.
  `BaseAgent.compute_loss()` handles both — do not "fix" by removing aux_loss.
- `DQRISEAgent` bypasses `UNetDiffusionAgent` — its `diffusion_action_dim = tcp_dim+1`
  (≠ `action_dim`) so it cannot reuse the standard UNet path. Do not refactor into UNetDiffusionAgent.
- MoE disables `bfloat16` and `compile` — gate softmax requires float32, CUDA Graphs have
  high memory overhead with MoE routing. Do not re-enable.

## Report format

- One line per check: ✅ pass / ❌ fail / ⚠️ not run, with a short reason.
- For each ❌: name the file and the exact change, pointing to an existing correct config
  (e.g. `dp3.yaml`) as reference.
- End with a verdict: **ready for PR** or **needs fixes** (list blocking items first).
- Then offer to apply the fixes.

Keep the report short — checks that pass need one line, not an explanation.
Never mark an item ✅ based on reading code alone when the check is a command you did not run.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `action_dim` doesn't match `tcp_dim + hand_dim` | `action_key` changed without updating `action_dim` | Use correct eval expression: `${eval:'21 if ${eq:${action_key},action_ee} else 19'}` |
| FAAS `state_dim` is 19 but should be 39 | `agent.state_dim` not overridden in FAAS config | Add `agent: {state_dim: ${eval:'7 + ${faas_hand_dim}'}}` |
| DDP config references nonexistent base | Typo in `defaults` | Match base config filename exactly (no `.yaml`) |
| CLAUDE.md counts don't match | Missed a CLAUDE.md update after adding/removing an agent | Sync the table, config list, and command examples |
