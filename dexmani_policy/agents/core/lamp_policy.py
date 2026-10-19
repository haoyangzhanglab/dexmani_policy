"""Single-step LAMP behavior cloning with a frozen history-conditioned prior."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.core.base_policy import BasePolicy
from dexmani_policy.agents.core.sat_policy import SATObservationEncoder
from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior, load_motion_prior
from dexmani_policy.utils.config import validate_window_contract


class LAMPRegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        layers = []
        widths = [input_dim, 512, 512, 256]
        for source, target in zip(widths, widths[1:]):
            layers.extend((nn.Linear(source, target), nn.SiLU()))
        self.model = nn.Sequential(*layers, nn.Linear(widths[-1], output_dim))

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.model(condition)


class LAMPPolicy(BasePolicy):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        prior_path: str,
        history_steps: int = 8,
        latent_dim: int = 2,
        prior_hidden_dim: int = 256,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        num_inference_steps: int = 1,
        hand_loss_weight: float = 1.0,
        offset_reg_weight: float = 1e-3,
        modality_dropout_probs: dict | None = None,
    ) -> None:
        validate_window_contract(horizon, n_obs_steps, n_action_steps)
        if (horizon, n_obs_steps, n_action_steps) != (1, 1, 1):
            raise ValueError("LAMP requires horizon=n_obs_steps=n_action_steps=1")
        if action_dim != 19 or state_dim != 19:
            raise ValueError("LAMP requires joint action/state: 7 arm + 12 hand")
        self.validate_inference_steps(num_inference_steps)
        for name, weight in (("hand_loss_weight", hand_loss_weight),
                             ("offset_reg_weight", offset_reg_weight)):
            if not 0 <= weight < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        encoder = SATObservationEncoder(
            pc_dim, state_dim, num_points, n_obs_steps, state_out_dim,
            pc_encoder_config, fps_random_config,
        )
        head = LAMPRegressionHead(encoder.obs_token_dim + 2 * latent_dim, 7 + latent_dim)
        super().__init__(encoder, head, horizon, n_obs_steps, n_action_steps,
                         action_dim, modality_dropout_probs)
        self.arm_dim, self.hand_dim = 7, 12
        self.history_steps, self.latent_dim = history_steps, latent_dim
        self.prior_path = prior_path
        self.hand_loss_weight, self.offset_reg_weight = hand_loss_weight, offset_reg_weight
        self.prior = HandMotionPrior(12, history_steps, latent_dim, prior_hidden_dim).freeze()
        self.prior_initialized = False
        self.register_load_state_dict_post_hook(self.validate_restored_prior)

    @staticmethod
    def validate_inference_steps(steps) -> None:
        if steps is not None and (type(steps) is not int or steps != 1):
            raise ValueError("LAMP direct regression requires inference_steps=None or 1")

    def initialize_training(self) -> None:
        self.prior_initialized = False
        if not self.prior_path:
            raise ValueError("Fresh LAMP training requires prior_path")
        prior, payload = load_motion_prior(self.prior_path)
        if prior.model_config != self.prior.model_config:
            raise ValueError("LAMP prior architecture differs from the policy configuration")
        self.prior.load_state_dict(prior.state_dict(), strict=True)
        from dexmani_policy.utils.hand_artifact import artifact_record

        self.action_space_artifact = artifact_record(
            "lamp", "lamp_prior.v2", payload["_file_sha256"], payload["metadata"], self.prior,
        )
        self.prior_initialized = True

    def validate_restored_prior(self, module, incompatible_keys) -> None:
        self.prior_initialized = False
        self.prior.validate_state()
        self.validate_prior_normalizer()
        self.prior_initialized = True

    def load_normalizer_from_dataset(self, normalizer) -> None:
        super().load_normalizer_from_dataset(normalizer)
        self.validate_prior_normalizer()

    def validate_prior_normalizer(self) -> None:
        if not bool(self.prior.ready) or "action" not in self.normalizer.params_dict:
            return
        params = self.normalizer["action"].params_dict
        for name in ("scale", "offset"):
            if not torch.allclose(params[name][-12:].detach().cpu(),
                                  getattr(self.prior, "hand_" + name).detach().cpu(),
                                  rtol=1e-5, atol=1e-6):
                raise ValueError("LAMP prior/policy hand affine mismatch; regenerate the prior")

    def train(self, mode: bool = True) -> LAMPPolicy:
        super().train(mode)
        self.prior.freeze()
        return self

    def encode_condition(self, obs: dict, action_context: dict | None):
        if not self.prior_initialized:
            raise RuntimeError("Initialize a LAMP prior or restore a complete policy checkpoint")
        context = {} if action_context is None else action_context
        if not isinstance(context, dict) or set(context) != {"hand_history"}:
            raise ValueError("LAMP requires only action_context.hand_history")
        history = context["hand_history"]
        if (not isinstance(history, torch.Tensor)
                or history.shape != (obs["joint_state"].shape[0], self.history_steps, 12)
                or not torch.isfinite(history).all()):
            raise ValueError("LAMP requires finite physical hand_history [B,K,12]")
        with torch.no_grad(), torch.autocast(device_type=history.device.type, enabled=False):
            mu, log_var = self.prior.encode_history(history.float())
            sigma = torch.exp(0.5 * log_var)
            if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
                raise ValueError("LAMP prior produced nonfinite statistics")
        global_token = self.encode_observation(obs)[:, 0, :]
        condition = torch.cat((global_token, mu.to(global_token.dtype),
                               sigma.to(global_token.dtype)), dim=-1)
        return condition, mu

    def decode_hand(self, offset: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=offset.device.type, enabled=False):
            return self.prior.decode(mu + offset.float())

    def compute_loss(self, batch: dict, **kwargs):
        self.validate_batch(batch)
        condition, mu = self.encode_condition(batch["obs"], batch.get("action_context"))
        native = self.action_decoder(condition).float()
        arm, offset = native[:, :7], native[:, 7:]
        target = self.normalizer["action"].normalize(batch["action"])[:, 0, :7]
        hand_target = self.prior.normalize_hand(batch["action"][:, 0, 7:].float())
        hand = self.decode_hand(offset, mu)
        arm_loss = F.mse_loss(arm, target)
        hand_loss = F.mse_loss(hand, hand_target)
        offset_l2 = offset.square().sum(-1).mean()
        loss = arm_loss + self.hand_loss_weight * hand_loss + self.offset_reg_weight * offset_l2
        if not torch.isfinite(loss):
            raise ValueError("LAMP produced a nonfinite loss")
        with torch.no_grad():
            prior_error = F.mse_loss(self.decode_hand(torch.zeros_like(offset), mu), hand_target)
            physical_mae = ((hand - hand_target) / self.prior.hand_scale).abs().mean()
        return loss, dict(loss=loss, loss_action=arm_loss + self.hand_loss_weight * hand_loss,
                          loss_arm=arm_loss, loss_decoded_hand=hand_loss,
                          loss_offset=self.offset_reg_weight * offset_l2,
                          offset_l2=offset_l2.detach(), offset_rms=offset.square().mean().sqrt().detach(),
                          prior_hand_mse=prior_error, policy_hand_mse=hand_loss.detach(),
                          physical_hand_mae=physical_mae)

    @torch.no_grad()
    def predict_action(self, obs_dict, inference_steps=None, *, action_context=None,
                       rtc_prefix=None, delay_steps=0, rtc_guidance_cap=0.0):
        self.validate_inference_steps(inference_steps)
        if rtc_prefix is not None or delay_steps or rtc_guidance_cap:
            raise NotImplementedError("LAMP supports synchronous execution only")
        self.validate_observations(obs_dict)
        condition, mu = self.encode_condition(obs_dict, action_context)
        return self.predict_action_from_cond(condition, inference_steps, mu=mu)

    @torch.no_grad()
    def predict_action_from_cond(self, cond, inference_steps=None, *, mu):
        self.validate_inference_steps(inference_steps)
        native = self.action_decoder(cond).float()
        params = self.normalizer["action"].params_dict
        arm = (native[:, :7] - params["offset"][:7]) / params["scale"][:7]
        hand = self.prior.unnormalize_hand(self.decode_hand(native[:, 7:], mu))
        prediction = torch.cat((arm, hand), dim=-1)[:, None]
        if not torch.isfinite(prediction).all():
            raise ValueError("LAMP produced a nonfinite action")
        return self.format_action_output(prediction)
