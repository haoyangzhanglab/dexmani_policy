---
name: research-iterate
description: Implement or modify a DexMani robot-policy research idea with the smallest coherent code/config change and a fast validation loop.
---

# Research Iterate

Use this when prototyping a new representation, encoder, backbone, decoder, conditioning mechanism, objective, or closely related Policy idea.

## Workflow

1. State the research hypothesis in one sentence and identify the main experimental variable.
2. Open the relevant Hydra config and follow `agent._target_` to the real implementation before editing.
3. Find the smallest end-to-end change surface. Reuse existing components when their semantics already match.
4. Implement the idea completely enough to run; keep experiment-specific logic local.
5. Run the cheapest relevant validation:
   - config/doc only → inspect diff / config-only smoke;
   - model/data path changed → targeted check or full smoke if the environment supports it.
6. Finish with:
   - what scientific variable changed;
   - files changed;
   - expected behavioral effect;
   - one exact command for the next experiment;
   - anything not verified.

## Keep It Vibe-Friendly

- Prefer a direct working implementation over a reusable framework.
- Do not add compatibility layers, registries, factories, or broad refactors unless the task truly needs them.
- Do not rewrite global docs for ordinary hyperparameter or architecture experiments.
- Do not launch long training or evaluation unless explicitly requested.
- If a non-critical design detail is underspecified, choose the simplest reasonable option and proceed.

