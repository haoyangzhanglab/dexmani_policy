"""Temporal ensembling for DexMani_Policy, adapted from ACT (Zhao et al. 2023).

Reference:
    Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation with
    Low-Cost Hardware" (arXiv:2304.13705), Section IV-D.

    LeRobot ``ACTTemporalEnsembler`` (huggingface/lerobot, Apache 2.0).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class ChunkOverlapBlender:
    """Blend overlapping action-chunk predictions with ACT exponential weights.

    In DexMani_Policy, the policy predicts ``horizon=16`` actions but only
    ``n_action_steps=8`` are executed per chunk.  The unexecuted tail of chunk
    *k* overlaps the to-be-executed head of chunk *k+1*, producing two
    independent predictions for each of ``overlap = 7`` real-world timesteps.

    This blender merges the two predictions at every overlapping position using
    ACT's exponential weighting.  For the two-prediction (single-overlap) case,
    the formula simplifies to::

        blended = (old * w0 + new * w1) / (w0 + w1)

    where ``w0 = exp(-coeff * 0) = 1.0`` and ``w1 = exp(-coeff * 1)``.
    """

    def __init__(self, temporal_ensemble_coeff: float = 0.01) -> None:
        # Pre-computed ACT weights for the exact 2-prediction case.
        self.w0: float = 1.0                                 # exp(-coeff * 0)
        self.w1: float = math.exp(-temporal_ensemble_coeff)  # exp(-coeff * 1)
        self.wsum: float = self.w0 + self.w1
        self.reset()

    def reset(self) -> None:
        """Clear stored state (call at episode boundaries)."""
        self._prev_tail: Tensor | None = None

    @torch.no_grad()
    def update(self, action_chunk: Tensor, n_action_steps: int) -> Tensor:
        """Blend the new chunk with the stored tail, return control actions.

        Args:
            action_chunk: ``(B, horizon, A)`` — full unnormalized prediction
                from ``BaseAgent.predict_action_from_cond``.
            n_action_steps: Number of action steps to execute per chunk (8).

        Returns:
            ``(B, n_action_steps, A)`` — blended actions to execute.
        """
        # Extract to-execute head: positions [1 : 1+n_action_steps].
        new_head = action_chunk[:, 1 : 1 + n_action_steps, :]

        if self._prev_tail is None:
            control = new_head
        else:
            # Overlap: prev_tail (from chunk k) vs new_head[0:overlap]
            # (chunk k+1).  The last head step has no counterpart in the tail.
            overlap = min(self._prev_tail.size(1), new_head.size(1) - 1)
            control = new_head.clone()
            control[:, :overlap] = (
                self._prev_tail[:, :overlap] * self.w0
                + new_head[:, :overlap] * self.w1
            ) / self.wsum

        # Save the unexecuted tail for the next chunk's blend.
        self._prev_tail = action_chunk[:, 1 + n_action_steps :, :]
        return control
