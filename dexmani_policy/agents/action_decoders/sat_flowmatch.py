"""SATFlowMatch — Flow Matching decoder with SAT-specific shuffle support.

Thin wrapper around ``FlowMatch`` that forwards the ``shuffle`` keyword
to the SAT backbone so that action tokens and their EJC identities are
permuted together during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dexmani_policy.agents.action_decoders.flowmatch import FlowMatch


class SATFlowMatch(FlowMatch):
    """Flow Matching decoder with structural token shuffle support.

    Identical to ``FlowMatch`` except:
    - ``compute_loss`` accepts and forwards ``shuffle`` to the backbone
    - ``predict_action`` always passes ``shuffle=False``
    - Inference uses Gaussian noise init (randn_like), per paper spec
    """

    def compute_loss(self, cond, actions, shuffle=True, **kwargs):
        """Flow matching loss with optional structural token shuffling.

        Args:
            cond: ``(B, N_obs, obs_token_dim)`` — observation tokens
            actions: ``(B, Da, T)`` — ground-truth actions (already
                     transposed to structural-centric format)
            shuffle: if True, the SAT backbone randomly permutes the
                     Da axis together with EJC identities

        Returns:
            ``(loss, loss_dict)``
        """
        B = actions.shape[0]

        # Gaussian noise init (paper spec: A⁰ ~ N(0, I))
        x0 = torch.randn_like(actions, device=actions.device)
        x1 = actions

        t = self.sampler.sample(B, self.t_sample_mode, device=actions.device)
        t = t.view(-1, 1, 1)

        xt = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0

        pred_v = self.model(
            x=xt,
            timestep=t.squeeze(),
            context=cond,
            shuffle=shuffle,
        )

        loss = F.mse_loss(pred_v, target_v)
        loss_dict = {
            "loss": loss,
            "loss_action": loss,
            "pred_v_magnitude": torch.sqrt(torch.mean(pred_v**2)),
        }
        return loss, loss_dict

    @torch.no_grad()
    def predict_action(self, cond, action_template, denoise_timesteps=None):
        """Euler ODE integration with Gaussian initial noise.

        No structural token shuffling during inference — joints are
        in their canonical order with EJC providing identity.
        """
        B = action_template.shape[0]
        device = action_template.device

        if denoise_timesteps is None:
            denoise_timesteps = self.num_inference_steps

        # Gaussian init (paper spec: A⁰ ~ N(0, I))
        x = torch.randn_like(action_template, device=device)
        dt = 1.0 / denoise_timesteps

        for i in range(denoise_timesteps):
            ti = torch.full((B,), i * dt, device=device, dtype=x.dtype)
            v = self.model(x=x, timestep=ti, context=cond, shuffle=False)
            x = x + v * dt

        assert torch.isfinite(x).all(), f"NaN/Inf in SATFlowMatch ODE output after {denoise_timesteps} steps"

        return x
