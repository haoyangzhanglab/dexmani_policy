"""Flow matching in a frozen temporal hand VAE (arXiv:2609.19666, Steps 1–2)."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.action_decoders.backbones.ditx import DiTX
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.action_decoders.sampling.time_sampling import shift_time_to_noise
from dexmani_policy.agents.hand_prior.hand_vae import HandVAE, load_hand_vae, validate_codec_metadata
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.hand_representation import validate_hand_chunk_compatibility


class LatentFlowObservationEncoder(nn.Module):
    """Frame-major Sparse3D global/patch tokens with state [B,T*(K+1),D]."""

    consumed_observation_fields = ("joint_state", "point_cloud")
    validate_normalization = staticmethod(Sparse3DEncoder.validate_normalization)

    def __init__(
        self,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ) -> None:
        super().__init__()
        config = dict(pc_encoder_config or {})
        config.setdefault("fps_random_config", fps_random_config)
        config.setdefault("num_patches", 96)
        self.num_points = positive_int(num_points, "num_points")
        self.n_obs_steps = positive_int(n_obs_steps, "n_obs_steps")
        self.pc_encoder = Sparse3DEncoder(input_channels=pc_dim, **config)
        self.state_mlp = StateEncoder(state_dim, state_out_dim)
        self.use_coord_only = pc_dim == 3
        self.fps_random_config = fps_random_config or {}
        self.out_dim = self.pc_encoder.out_dim + self.state_mlp.out_dim

    @property
    def effective_config(self):
        return {"pc_encoder_config": self.pc_encoder.effective_config}

    def forward(self, obs: dict) -> torch.Tensor:
        pc = preprocess_point_cloud(
            obs["point_cloud"], self.num_points, self.use_coord_only,
            self.fps_random_config, training=self.training,
        )
        patches = self.pc_encoder(pc)["tokens"]
        state = self.state_mlp(obs["joint_state"])
        pc_features = torch.cat((patches.mean(dim=1, keepdim=True), patches), dim=1)
        state = state.unsqueeze(1).expand(-1, pc_features.shape[1], -1)
        features = torch.cat((pc_features, state), dim=-1)
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)


class LatentFlowPolicy(BasePolicy):
    supported_execution_modes = ("sync",)
    requires_hand_pretraining = True
    future_action_window = True

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        codec_path: str,
        latent_dim: int = 9,
        codec_hidden_dim: int = 128,
        codec_n_layers: int = 3,
        codec_n_heads: int = 4,
        codec_mlp_ratio: float = 2.0,
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
        arm_loss_weight: float = 1.0,
        latent_loss_weight: float = 0.25,
        decoded_loss_weight: float = 0.75,
        modality_dropout_probs: dict | None = None,
        codec_metadata: dict | None = None,
    ) -> None:
        if horizon < 3 or n_obs_steps > horizon:
            raise ValueError("LatentFlow requires H>=3 and n_obs_steps<=H")
        if action_dim != 19 or state_dim != 19:
            raise ValueError("LatentFlow codec requires joint actions/state: 7 arm + 12 hand")
        for weight in (arm_loss_weight, latent_loss_weight, decoded_loss_weight):
            if not 0 <= weight < float("inf"):
                raise ValueError("LatentFlow codec loss weights must be finite and nonnegative")
        obs_encoder = LatentFlowObservationEncoder(
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
            obs_token_dim=obs_encoder.out_dim,
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
        self.latent_dim = latent_dim
        self.model_action_dim = self.arm_dim + latent_dim
        self.codec_path = codec_path
        self.codec_metadata = (
            OmegaConf.to_container(codec_metadata, resolve=True)
            if OmegaConf.is_config(codec_metadata) else codec_metadata
        )
        self.arm_loss_weight = arm_loss_weight
        self.latent_loss_weight = latent_loss_weight
        self.decoded_loss_weight = decoded_loss_weight
        self.codec = HandVAE(
            hand_dim=12,
            horizon=horizon,
            latent_dim=latent_dim,
            hidden_dim=codec_hidden_dim,
            n_layers=codec_n_layers,
            n_heads=codec_n_heads,
            mlp_ratio=codec_mlp_ratio,
        )
        self.codec.freeze()
        # Cache readiness on the host; checkpoint validation refreshes it after loading.
        self.codec_initialized = False
        self.register_load_state_dict_post_hook(self.validate_restored_codec)

    @property
    def action_context_shapes(self):
        return {"reference_state": [self.action_dim]}

    def initialize_training(self, data_description) -> None:
        self.codec_initialized = False
        if not self.codec_path:
            raise ValueError("Fresh LatentFlow training requires codec_path")
        codec, payload = load_hand_vae(self.codec_path)
        if codec.model_config != self.codec.model_config:
            raise ValueError("LatentFlow codec architecture differs from the policy configuration")
        self.codec.load_state_dict(codec.state_dict(), strict=True)
        self.codec_metadata = payload["metadata"]
        validate_hand_chunk_compatibility(
            self.codec_metadata["hand_data"], data_description["hand_data"]
        )
        self.codec_initialized = True

    def validate_restored_codec(self, module: torch.nn.Module, incompatible_keys) -> None:
        self.codec_initialized = False
        self.codec.validate_state()
        validate_codec_metadata(self.codec_metadata, self.codec.model_config)
        self.codec_initialized = True

    @property
    def effective_config(self):
        return {**super().effective_config, "codec_metadata": self.codec_metadata}

    def train(self, mode: bool = True) -> LatentFlowPolicy:
        super().train(mode)
        self.codec.eval()
        return self

    def encode_condition(self, obs: dict) -> torch.Tensor:
        if not self.codec_initialized:
            raise RuntimeError(
                "Initialize the hand codec or restore a complete policy checkpoint"
            )
        return self.encode_observation(obs)

    def resolve_reference_state(
        self, obs: dict, action_context: dict | None, *, required: bool = False
    ) -> torch.Tensor:
        reference = (action_context or {}).get("reference_state")
        if reference is None and not required:
            reference = obs["joint_state"][:, self.n_obs_steps - 1]
        if (
            not isinstance(reference, torch.Tensor)
            or reference.shape != (obs["joint_state"].shape[0], self.action_dim)
            or not torch.isfinite(reference).all()
        ):
            raise ValueError("LatentFlow codec requires raw action_context.reference_state [B,19]")
        return reference.float()

    def compute_loss(self, batch: dict, *, log_metrics: bool = True) -> tuple[torch.Tensor, dict]:
        # Dataset supplies the measured query state before observation augmentation.
        reference = self.resolve_reference_state(
            batch["obs"], batch.get("action_context"), required=True
        )
        condition = self.encode_condition(batch["obs"])
        actions = batch["action"].float()
        hand_reference = reference[:, self.arm_dim :]
        affine = self.normalizer["action"].params_dict
        arm = actions[..., : self.arm_dim] - reference[:, None, : self.arm_dim]
        arm = arm * affine["scale"][: self.arm_dim] + affine["offset"][: self.arm_dim]
        with torch.no_grad(), torch.autocast(device_type=actions.device.type, enabled=False):
            mu, _ = self.codec.encode(actions[..., self.arm_dim :], hand_reference)
        target = torch.cat((arm, mu), dim=-1)

        flow = self.action_decoder
        noise = torch.randn_like(target)
        t = flow.time_sampler.sample(len(target), flow.t_sample_mode, device=target.device)
        t = shift_time_to_noise(t, flow.time_shift_alpha)
        weight = t[:, None, None]
        xt = (1 - weight) * noise + weight * target
        pred_v = flow.model(x=xt, timestep=t, context=condition).float()
        target_v = target - noise
        arm_loss = F.mse_loss(pred_v[..., : self.arm_dim], target_v[..., : self.arm_dim])
        latent_loss = F.mse_loss(pred_v[..., self.arm_dim :], target_v[..., self.arm_dim :])
        flow_loss = self.arm_loss_weight * arm_loss + self.latent_loss_weight * latent_loss
        loss = flow_loss
        metrics = {}
        if self.decoded_loss_weight:
            # Our flow goes noise -> data, reversing the paper's time coordinate.
            clean = xt + (1 - weight) * pred_v
            with torch.autocast(device_type=actions.device.type, enabled=False):
                decoded = self.codec.decode(clean[..., self.arm_dim :].float(), hand_reference)
                hand_target = self.codec.normalize_actions(actions[..., self.arm_dim :], hand_reference)
                decoded_loss = F.mse_loss(decoded, hand_target)
            loss = loss + self.decoded_loss_weight * decoded_loss
            if log_metrics:
                physical_error = (decoded.detach() - hand_target) * self.codec.action_std
                metrics.update(
                    loss_decoded_hand=decoded_loss.detach(),
                    one_step_denoising_hand_mae=physical_error.abs().mean(),
                    one_step_denoising_hand_rmse=physical_error.square().mean().sqrt(),
                )
        if log_metrics:
            metrics.update(
                loss=loss.detach(), loss_action=flow_loss.detach(),
                loss_arm=arm_loss.detach(), loss_latent=latent_loss.detach(), t_mean=t.mean(),
            )
        return loss, metrics

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: dict,
        inference_steps: int | None = None,
        *,
        generator: torch.Generator | None = None,
        action_context: dict | None = None,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if rtc_prefix is not None or delay_steps or rtc_guidance_cap:
            raise NotImplementedError("LatentFlow codec supports synchronous chunk execution")
        reference = self.resolve_reference_state(obs_dict, action_context)
        condition = self.encode_condition(obs_dict)
        return self.predict_action_from_condition(
            condition, inference_steps, reference_state=reference, generator=generator
        )

    @torch.no_grad()
    def predict_action_from_condition(
        self,
        condition: torch.Tensor,
        inference_steps: int | None = None,
        *,
        reference_state: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        native = self.action_decoder.predict_action(
            condition, (len(condition), self.horizon, self.model_action_dim),
            inference_steps, dtype=torch.float32, generator=generator,
        )
        affine = self.normalizer["action"].params_dict
        scale = affine["scale"][: self.arm_dim]
        offset = affine["offset"][: self.arm_dim]
        arm = (native[..., : self.arm_dim] - offset) / scale + reference_state[:, None, : self.arm_dim]
        with torch.autocast(device_type=native.device.type, enabled=False):
            hand = self.codec.decode_actions(
                native[..., self.arm_dim :].float(), reference_state[:, self.arm_dim :].float()
            )
        prediction = torch.cat((arm, hand), dim=-1)
        # Bidirectional codec attention requires decoding the full horizon first.
        return self.format_action_output(prediction)
