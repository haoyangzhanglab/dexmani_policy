"""LAMP history prior with chunk-level latent-offset flow matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.action_decoders.sampling.time_sampling import shift_time_to_noise
from dexmani_policy.agents.core.base_policy import BasePolicy
from dexmani_policy.agents.core.sat_policy import SATObservationEncoder
from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior, load_motion_prior
from dexmani_policy.utils.config import validate_window_contract


class LAMPFlowPolicy(BasePolicy):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        prior_path: str,
        latent_path: str,
        history_steps: int = 8,
        latent_dim: int = 2,
        prior_hidden_dim: int = 256,
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        n_layers: int = 8,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        num_inference_steps: int = 10,
        t_sample_mode_for_flow: str = "beta",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        num_flow_train_timesteps: int = 10,
        latent_loss_weight: float = 1.0,
        decoded_loss_weight: float = 1.0,
        modality_dropout_probs: dict | None = None,
    ) -> None:
        validate_window_contract(horizon, n_obs_steps, n_action_steps)
        if action_dim != 19 or state_dim != 19:
            raise ValueError("LAMP currently requires joint actions/state: 7 arm + 12 hand")
        for name, value in (
            ("latent_loss_weight", latent_loss_weight),
            ("decoded_loss_weight", decoded_loss_weight),
        ):
            if not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        obs_encoder = SATObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )
        backbone = DiTX(
            horizon=horizon,
            action_dim=7 + latent_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.obs_token_dim // n_obs_steps + 2 * latent_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        flow = RectifiedFlow(
            backbone,
            num_inference_steps=num_inference_steps,
            t_sample_mode=t_sample_mode_for_flow,
            beta_s=beta_s,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            num_flow_train_timesteps=num_flow_train_timesteps,
        )
        super().__init__(
            obs_encoder,
            flow,
            horizon,
            n_obs_steps,
            n_action_steps,
            action_dim,
            modality_dropout_probs,
        )
        self.arm_dim, self.hand_dim = 7, 12
        self.history_steps, self.latent_dim = history_steps, latent_dim
        self.model_action_dim = self.arm_dim + latent_dim
        self.prior_path, self.latent_path = prior_path, latent_path
        self.latent_loss_weight = latent_loss_weight
        self.decoded_loss_weight = decoded_loss_weight
        self.prior = HandMotionPrior(12, history_steps, latent_dim, prior_hidden_dim)
        self.prior.freeze()
        self.register_buffer("latent_ready", torch.tensor(False))
        # Cache readiness on the host; checkpoint validation refreshes it after loading.
        self.prior_initialized = False
        self.register_load_state_dict_post_hook(self.validate_restored_prior)

    def initialize_training(self) -> None:
        self.prior_initialized = False
        prior, payload = load_motion_prior(self.prior_path)
        cache = torch.load(self.latent_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(cache, dict)
            or cache.get("_format") != "lamp_latents.v1"
            or cache.get("prior_sha256") != payload["_file_sha256"]
        ):
            raise ValueError("LAMP latent labels were not exported with this prior")
        if prior.model_config != self.prior.model_config:
            raise ValueError("LAMP prior architecture differs from the policy configuration")
        self.prior.load_state_dict(prior.state_dict(), strict=True)
        scale = torch.as_tensor(cache["latent_scale"], dtype=torch.float32)
        if (
            scale.shape != (self.latent_dim,)
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("LAMP latent_scale must contain one finite positive value per latent")
        self.prior.latent_scale.copy_(scale)
        self.latent_ready.fill_(True)
        self.prior_initialized = True

    def validate_restored_prior(self, module: torch.nn.Module, incompatible_keys) -> None:
        self.prior_initialized = False
        self.prior.validate_state()
        if not self.latent_ready:
            raise ValueError("LAMP checkpoint must contain initialized latent statistics")
        self.prior_initialized = True

    def train(self, mode: bool = True) -> LAMPFlowPolicy:
        super().train(mode)
        self.prior.eval()
        return self

    def encode_condition(
        self, obs: dict, action_context: dict | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.prior_initialized:
            raise RuntimeError(
                "Initialize LAMP prior/labels or restore a complete policy checkpoint"
            )
        history = (action_context or {}).get("hand_history")
        batch_size = obs["joint_state"].shape[0]
        if (
            not isinstance(history, torch.Tensor)
            or history.shape != (batch_size, self.history_steps, self.hand_dim)
            or not torch.isfinite(history).all()
        ):
            raise ValueError("LAMP requires finite action_context.hand_history [B,K,12]")
        # Keep the frozen prior in FP32, matching offline latent construction.
        with torch.no_grad(), torch.autocast(device_type=history.device.type, enabled=False):
            mu, log_var = self.prior.encode_history(history.float())
        cond = self.encode_observation(obs)
        b, patches, _ = cond.shape
        cond = cond.reshape(b, patches, self.n_obs_steps, -1).transpose(1, 2)
        cond = cond.reshape(b, self.n_obs_steps * patches, -1)
        prior_features = torch.cat((mu, log_var), dim=-1).to(cond.dtype)
        cond = torch.cat((cond, prior_features[:, None].expand(-1, cond.shape[1], -1)), dim=-1)
        return cond, mu

    def decode_hand(self, offsets: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=offsets.device.type, enabled=False):
            z = mu[:, None] + offsets.float() * self.prior.latent_scale
            return self.prior.decode(z)

    def compute_loss(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        self.validate_batch(batch)
        cond, mu = self.encode_condition(batch["obs"], batch.get("action_context"))
        z = batch.get("hand_latent_target")
        expected_shape = (mu.shape[0], self.horizon, self.latent_dim)
        if not isinstance(z, torch.Tensor) or z.shape != expected_shape:
            raise ValueError("LAMP requires hand_latent_target [B,H,Z]")
        actions = self.normalizer["action"].normalize(batch["action"])
        offsets = (z.float() - mu[:, None]) / self.prior.latent_scale
        target = torch.cat((actions[..., : self.arm_dim], offsets), dim=-1)

        # Match RectifiedFlow's noise-to-data path with arm/hand losses and decoded supervision.
        flow = self.action_decoder
        noise = torch.randn_like(target)
        t = flow.time_sampler.sample(len(target), flow.t_sample_mode, device=target.device)
        t = shift_time_to_noise(t, flow.time_shift_alpha)
        weight = t[:, None, None]
        xt = (1 - weight) * noise + weight * target
        pred_v = flow.model(x=xt, timestep=t, context=cond).float()
        target_v = target - noise
        arm_loss = F.mse_loss(pred_v[..., : self.arm_dim], target_v[..., : self.arm_dim])
        latent_loss = F.mse_loss(pred_v[..., self.arm_dim :], target_v[..., self.arm_dim :])
        flow_loss = (
            self.arm_dim * arm_loss + self.hand_dim * self.latent_loss_weight * latent_loss
        ) / self.action_dim
        clean = xt + (1 - weight) * pred_v
        decoded = self.decode_hand(clean[..., self.arm_dim :], mu)
        hand_target = self.prior.normalize_hand(batch["action"][..., self.arm_dim :])
        decoded_loss = F.mse_loss(decoded, hand_target)
        loss = flow_loss + self.decoded_loss_weight * decoded_loss
        metrics = dict(
            loss=loss,
            loss_action=flow_loss,
            loss_arm=arm_loss,
            loss_latent=latent_loss,
            loss_decoded_hand=decoded_loss,
            t_mean=t.mean(),
        )
        return loss, metrics

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: dict,
        inference_steps: int | None = None,
        *,
        action_context: dict | None = None,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if rtc_prefix is not None or delay_steps or rtc_guidance_cap:
            raise NotImplementedError("LAMP currently supports synchronous chunk execution")
        self.validate_observations(obs_dict)
        cond, mu = self.encode_condition(obs_dict, action_context)
        return self.predict_action_from_cond(cond, inference_steps, mu=mu)

    @torch.no_grad()
    def predict_action_from_cond(
        self, cond: torch.Tensor, inference_steps: int | None = None, *, mu: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        template = torch.zeros(
            len(cond), self.horizon, self.model_action_dim, device=cond.device, dtype=torch.float32
        )
        native = self.action_decoder.predict_action(cond, template, inference_steps)
        affine = self.normalizer["action"].params_dict
        scale = affine["scale"][: self.arm_dim]
        offset = affine["offset"][: self.arm_dim]
        arm = (native[..., : self.arm_dim] - offset) / scale
        hand = self.prior.unnormalize_hand(self.decode_hand(native[..., self.arm_dim :], mu))
        pred = torch.cat((arm, hand), dim=-1)
        return self.format_action_output(pred)
