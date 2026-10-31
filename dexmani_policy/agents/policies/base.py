from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from dexmani_policy.common.config import POINT_CLOUD_FIELDS, positive_int, validate_action_window
from dexmani_policy.common.normalizer import LinearNormalizer


def parameter_groups(module: nn.Module, *, lr: float, weight_decay: float) -> list[dict]:
    """Decay linear, convolution and attention weights; group each trainable parameter once."""
    decay_types = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.MultiheadAttention,
    )
    groups: list[dict[str, Any]] = [
        {"params": [], "lr": lr, "weight_decay": weight_decay},
        {"params": [], "lr": lr, "weight_decay": 0.0},
    ]
    modules = dict(module.named_modules())
    parameters = ((name, param) for name, param in module.named_parameters() if param.requires_grad)
    for name, param in sorted(parameters):
        owner, _, field = name.rpartition(".")
        decay = field.endswith("weight") and isinstance(modules[owner], decay_types)
        groups[0 if decay else 1]["params"].append(param)
    return groups


def module_parameter_group(module: nn.Module, *, lr: float, weight_decay: float) -> dict:
    """Apply one decay rate to all trainable parameters, in module order."""
    return {
        "params": [p for p in module.parameters() if p.requires_grad],
        "lr": lr, "weight_decay": weight_decay,
    }


class BasePolicy(nn.Module):
    """Shared observation preprocessing, training and action-window interfaces."""

    supported_action_keys: tuple[str, ...] = ("action", "action_ee")

    def __init__(
        self,
        obs_encoder: nn.Module,
        action_decoder: nn.Module,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        modality_dropout_probs: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        validate_action_window(
            horizon, n_obs_steps, n_action_steps,
            action_start_index=n_obs_steps - 1,
        )
        positive_int(action_dim, "action_dim")
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.modality_dropout_probs = modality_dropout_probs or {}
        self.normalizer = LinearNormalizer()
        self.normalization_dims: dict[str, int] = {}

    @classmethod
    def prepare_restore_config(cls, config) -> None:
        """Adjust only external initialization options on a private constructor config."""

    @property
    def effective_config(self) -> dict:
        return dict(getattr(self.obs_encoder, "effective_config", {}))

    @property
    def action_start_index(self) -> int:
        return self.n_obs_steps - 1

    @property
    def future_steps(self) -> int:
        return self.horizon - self.action_start_index

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return self.obs_encoder.consumed_observation_fields

    supported_execution_modes: tuple[str, ...] = ("sync", "async")

    def configure_normalizer(self, feature_dims, feature_shapes) -> None:
        self.normalization_dims = dict(feature_dims)
        self.normalizer = LinearNormalizer(feature_shapes)

    def initialize_training(self, data_description) -> dict[str, torch.Tensor] | None:
        """Initialize fresh weights and optionally return inherited normalization state."""

    def load_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer = normalizer

    @property
    def requires_ema_for_loss(self) -> bool:
        return bool(getattr(self.action_decoder, "requires_ema_for_loss", False))

    def get_training_loss_kwargs(self, ema_agent: BasePolicy | None = None) -> dict:
        if not self.requires_ema_for_loss:
            return {}
        return {"ema_decoder": cast(BasePolicy, ema_agent).action_decoder}

    def preprocess(self, obs_dict: dict) -> dict:
        result = {}
        for key, value in obs_dict.items():
            if torch.is_tensor(value):
                # Runners may supply XYZRGB even when the policy consumes only XYZ.
                if key in POINT_CLOUD_FIELDS and self.normalization_dims.get(key) == 3:
                    value = value[..., :3]
                if key in self.normalizer.fields:
                    value = self.normalizer[key].normalize(value)
                probability = self.modality_dropout_probs.get(key, 0.0)
                if self.training and probability > 0:
                    mask = torch.rand(value.shape[0], device=value.device) > probability
                    value = value * mask.view(-1, *([1] * (value.ndim - 1)))
                value = value.flatten(0, 1)
            result[key] = value
        return result

    def encode_observation(self, obs_dict: dict) -> torch.Tensor:
        return self.obs_encoder(self.preprocess(obs_dict))

    def loss_groups(self) -> dict[str, tuple[int, int, float]] | None:
        """Optional named action slices for separate diffusion losses."""
        return None

    def forward(
        self, batch: dict, *, log_metrics: bool = True, **kwargs
    ) -> tuple[torch.Tensor, dict]:
        return self.compute_loss(batch, log_metrics=log_metrics, **kwargs)

    def compute_loss(
        self, batch: dict, *, log_metrics: bool = True, **kwargs
    ) -> tuple[torch.Tensor, dict]:
        condition = self.encode_observation(batch["obs"])
        normalized_actions = self.normalizer["action"].normalize(batch["action"])
        groups = self.loss_groups()
        if groups is not None:
            kwargs["dim_groups"] = groups
        return self.action_decoder.compute_loss(
            condition, normalized_actions, log_metrics=log_metrics, **kwargs
        )

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: dict,
        inference_steps: int | None = None,
        *,
        generator: torch.Generator | None = None,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        condition = self.encode_observation(obs_dict)
        if rtc_prefix is None:
            return self.predict_action_from_condition(condition, inference_steps=inference_steps, generator=generator)
        if "rtc" not in self.supported_execution_modes:
            raise NotImplementedError("This policy does not support RTC")
        return self.predict_action_from_condition(
            condition,
            inference_steps=inference_steps,
            generator=generator,
            rtc_prefix=rtc_prefix,
            delay_steps=delay_steps,
            rtc_guidance_cap=rtc_guidance_cap,
        )

    @property
    def control_action_dim(self) -> int:
        """Action dimensions sent to the environment, excluding auxiliary predictions."""
        return self.action_dim

    @torch.no_grad()
    def predict_action_from_condition(
        self,
        condition: torch.Tensor,
        inference_steps: int | None = None,
        *,
        generator: torch.Generator | None = None,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        action_shape = (condition.shape[0], self.horizon, self.action_dim)
        kwargs = {}
        if rtc_prefix is not None:
            if "rtc" not in self.supported_execution_modes:
                raise NotImplementedError("This policy does not support RTC")
            field = self.normalizer["action"]
            scale = field.scale[: self.control_action_dim]
            offset = field.offset[: self.control_action_dim]
            normalized = rtc_prefix.to(condition) * scale + offset
            kwargs = dict(
                rtc_prefix=normalized,
                delay_steps=delay_steps,
                history_offset=self.n_obs_steps - 1,
                rtc_guidance_cap=rtc_guidance_cap,
            )
        prediction = self.action_decoder.predict_action(
            condition, action_shape, inference_steps=inference_steps, generator=generator, **kwargs
        )
        prediction = self.normalizer["action"].unnormalize(prediction)
        return self.format_action_output(prediction)

    def format_action_output(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite physical policy prediction")
        return {"future_actions": prediction[:, self.action_start_index :, : self.control_action_dim]}

    @property
    def inference_metadata(self) -> dict:
        return {
            "action_start_index": self.action_start_index,
            "control_action_dim": self.control_action_dim,
            "supported_execution_modes": list(self.supported_execution_modes),
        }

    def compile_backbone(self, **compile_kwargs: Any) -> None:
        self.action_decoder.model = cast(nn.Module, torch.compile(self.action_decoder.model, **compile_kwargs))

    def optimizer_groups(
        self, lr: float, obs_lr: float, weight_decay: float, obs_weight_decay: float
    ) -> list[dict]:
        return parameter_groups(
            self.action_decoder.model, lr=lr, weight_decay=weight_decay
        ) + parameter_groups(self.obs_encoder, lr=obs_lr, weight_decay=obs_weight_decay)
