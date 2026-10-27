"""Single-step LAMP behavior cloning with a frozen history-conditioned prior."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior, load_motion_prior
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder
from dexmani_policy.agents.obs_encoder.proprio import StateEncoder
from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.hand_representation import validate_pretraining_split


class LAMPObservationEncoder(nn.Module):
    """Pooled Sparse3D and state features in frame order [B,T*D]."""

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
        self.out_dim = n_obs_steps * (self.pc_encoder.out_dim + self.state_mlp.out_dim)

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
        features = torch.cat((patches.mean(dim=1), state), dim=-1)
        return features.unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class LAMPRegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        widths = [input_dim, 512, 512, 256]
        for source, target in zip(widths, widths[1:]):
            layers.extend((nn.Linear(source, target), nn.ReLU()))
        self.model = nn.Sequential(*layers, nn.Linear(widths[-1], output_dim))

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.model(condition)


class LAMPPolicy(BasePolicy):
    supported_execution_modes = ("sync",)
    requires_action_affine = False
    supported_action_keys = ("action",)

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
        modality_dropout_probs: dict | None = None,
        *,
        history_noise_std: float,
        hand_loss_weight: float,
        arm_state_noise_std: float,
        drift_reg_weight: float,
    ) -> None:
        if any(
            type(value) is not int or value != 1 for value in (horizon, n_obs_steps, n_action_steps)
        ):
            raise ValueError("LAMP requires horizon=n_obs_steps=n_action_steps=1")
        if action_dim != 19 or state_dim != 19:
            raise ValueError("LAMP requires joint action/state: 7 arm + 12 hand")
        for name, value in (
            ("hand_loss_weight", hand_loss_weight),
            ("arm_state_noise_std", arm_state_noise_std),
            ("drift_reg_weight", drift_reg_weight),
            ("history_noise_std", history_noise_std),
        ):
            if not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        encoder = LAMPObservationEncoder(
            pc_dim,
            state_dim,
            num_points,
            n_obs_steps,
            state_out_dim,
            pc_encoder_config,
            fps_random_config,
        )
        head = LAMPRegressionHead(encoder.out_dim + 128, 7 + latent_dim)
        super().__init__(
            encoder, head, horizon, n_obs_steps, n_action_steps, action_dim, modality_dropout_probs
        )
        self.arm_dim, self.hand_dim = 7, 12
        self.history_steps, self.latent_dim = history_steps, latent_dim
        self.prior_path = prior_path
        self.hand_loss_weight = hand_loss_weight
        self.arm_state_noise_std = arm_state_noise_std
        self.prior_feature_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.drift_reg_weight = drift_reg_weight
        self.history_noise_std = history_noise_std
        self.prior = HandMotionPrior(12, history_steps, latent_dim, prior_hidden_dim).freeze()
        self.prior_initialized = False
        self.register_load_state_dict_post_hook(self.validate_restored_prior)

    @property
    def action_context_shapes(self):
        return {"hand_history": [self.history_steps, self.hand_dim]}

    @classmethod
    def prepare_restore_config(cls, config) -> None:
        config.prior_path = None

    def initialize_training(self, data_description) -> None:
        self.prior_initialized = False
        if not self.prior_path:
            raise ValueError("Fresh LAMP training requires prior_path")
        payload = load_motion_prior(self.prior_path, self.prior)
        metadata = payload["metadata"]
        current, pretrained = data_description["hand_data"], metadata["hand_data"]
        if any(current[key] != pretrained[key] for key in ("action_key", "dt", "history_steps")):
            raise ValueError("LAMP prior/policy action history or time semantics mismatch")
        validate_pretraining_split(
            data_description["split_manifest"],
            metadata["hand_data"], metadata["split_manifest"],
        )
        self.prior_initialized = True

    def validate_restored_prior(self, module, incompatible_keys) -> None:
        self.prior_initialized = False
        self.prior.validate_state()
        self.prior_initialized = True

    def set_normalization_spec(self, spec, feature_dims) -> None:
        if spec.get("action") != "identity" or spec.get("joint_state") != "gaussian":
            raise ValueError("LAMP requires identity action and gaussian joint_state normalization")
        super().set_normalization_spec(spec, feature_dims)

    def preprocess(self, obs_dict: dict) -> dict:
        result = super().preprocess(obs_dict)
        if self.training and self.arm_state_noise_std > 0:
            state = result["joint_state"].clone()
            state[..., :7] += torch.randn_like(state[..., :7]) * self.arm_state_noise_std
            result["joint_state"] = state
        return result

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay) -> list[dict]:
        return [
            {
                "params": [
                    parameter for module in modules for parameter in module.parameters()
                    if parameter.requires_grad
                ],
                "lr": rate,
                "weight_decay": decay,
                "clip_group": clip_group,
            }
            for modules, rate, decay, clip_group in (
                ((self.action_decoder, self.prior_feature_mlp), lr, weight_decay, "head"),
                ((self.obs_encoder.state_mlp,), obs_lr, obs_weight_decay, "head"),
                ((self.obs_encoder.pc_encoder,), obs_lr, obs_weight_decay, "backbone"),
            )
        ]

    def train(self, mode: bool = True) -> LAMPPolicy:
        super().train(mode)
        self.prior.eval()
        return self

    def encode_condition(
        self, obs: dict, action_context: dict | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.prior_initialized:
            raise RuntimeError("Initialize a LAMP prior or restore a complete policy checkpoint")
        context = {} if action_context is None else action_context
        if not isinstance(context, dict) or set(context) != {"hand_history"}:
            raise ValueError("LAMP requires only action_context.hand_history")
        history = context["hand_history"]
        if (
            not isinstance(history, torch.Tensor)
            or history.shape != (obs["joint_state"].shape[0], self.history_steps, self.hand_dim)
            or not torch.isfinite(history).all()
        ):
            raise ValueError("LAMP requires finite physical hand_history [B,K,12]")
        with torch.no_grad(), torch.autocast(device_type=history.device.type, enabled=False):
            history = history.float()
            if self.training and self.history_noise_std > 0:
                history = history + torch.randn_like(history) * self.history_noise_std
            mu, log_var = self.prior.encode_history(history)
        global_token = self.encode_observation(obs)
        prior_features = self.prior_feature_mlp(torch.cat((mu, log_var), dim=-1))
        condition = torch.cat((global_token, prior_features.to(global_token.dtype)), dim=-1)
        return condition, mu

    def decode_hand(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return self.prior.decode(latent.float())

    def compute_loss(self, batch: dict, *, log_metrics: bool = True) -> tuple[torch.Tensor, dict]:
        condition, mu = self.encode_condition(batch["obs"], batch.get("action_context"))
        native = self.action_decoder(condition).float()
        arm, offset = native[:, : self.arm_dim], native[:, self.arm_dim :]
        target = batch["action"][:, 0, : self.arm_dim].float()
        hand_target = batch["action"][:, 0, self.arm_dim :].float()
        hand = self.decode_hand(mu + offset)
        arm_loss = F.mse_loss(arm, target)
        hand_loss = F.mse_loss(hand, hand_target)
        with torch.no_grad():
            hand_ref = self.decode_hand(mu)
        hand_drift_mse = F.mse_loss(hand, hand_ref)
        loss = arm_loss + self.hand_loss_weight * hand_loss + self.drift_reg_weight * hand_drift_mse
        if not log_metrics:
            return loss, {}

        with torch.no_grad():
            prior_error = F.mse_loss(hand_ref, hand_target)
            physical_mae = (hand - hand_target).abs().mean()
            metrics = dict(
                loss=loss.detach(),
                loss_action=arm_loss + self.hand_loss_weight * hand_loss,
                loss_arm=arm_loss.detach(),
                loss_decoded_hand=hand_loss.detach(),
                hand_drift_mse=hand_drift_mse.detach(),
                loss_drift=self.drift_reg_weight * hand_drift_mse,
                offset_rms=offset.square().mean().sqrt(),
                prior_hand_mse=prior_error,
                physical_hand_mae=physical_mae,
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
            raise NotImplementedError("LAMP supports synchronous execution only")
        condition, mu = self.encode_condition(obs_dict, action_context)
        return self.predict_action_from_condition(condition, inference_steps, mu=mu, generator=generator)

    @torch.no_grad()
    def predict_action_from_condition(
        self, condition: torch.Tensor, inference_steps: int | None = None, *, mu: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if inference_steps is not None and (type(inference_steps) is not int or inference_steps != 1):
            raise ValueError("LAMP direct regression requires inference_steps=None or 1")
        native = self.action_decoder(condition).float()
        arm = native[:, : self.arm_dim]
        hand = self.decode_hand(mu + native[:, self.arm_dim :])
        prediction = torch.cat((arm, hand), dim=-1)[:, None]
        return self.format_action_output(prediction)
