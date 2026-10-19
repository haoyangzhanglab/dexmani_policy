from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.core.normalizer import AffineNormalizer, validate_normalizer_state


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
    groups = [
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


class BasePolicy(nn.Module):
    """Shared observation preprocessing, training and action-window interfaces."""

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
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.modality_dropout_probs = modality_dropout_probs or {}
        self.normalizer = AffineNormalizer()
        self.normalization_dims = {}

    def set_normalization_spec(
        self, spec: dict[str, str], feature_dims: dict[str, int] | None = None
    ) -> None:
        from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder
        from dexmani_policy.utils.config import validate_action_clipping

        if (
            isinstance(getattr(self.obs_encoder, "pc_encoder", None), Sparse3DEncoder)
            and spec.get("point_cloud") != "identity"
        ):
            raise ValueError(
                "Sparse3DEncoder requires normalization.point_cloud: identity (metric XYZ)"
            )
        if isinstance(self.action_decoder, Diffusion):
            validate_action_clipping(spec, self.action_decoder.noise_scheduler.config.clip_sample)
        self.normalization_spec = dict(spec)
        self.normalization_dims = dict(feature_dims or {})
        if self.normalization_dims.get("action", self.action_dim) != self.action_dim:
            raise ValueError("Normalizer action width must match the full prediction layout")
        self.normalization_dims["action"] = self.action_dim

    def validate_normalizer(self) -> None:
        validate_normalizer_state(self.normalizer, self.normalization_spec, self.normalization_dims)

    def initialize_training(self) -> None:
        """Initialize fresh training weights; complete checkpoints skip this step."""

    def load_normalizer_from_dataset(self, normalizer: AffineNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    @property
    def requires_ema_for_loss(self) -> bool:
        """Whether the action decoder needs a local EMA copy to build targets."""
        return bool(getattr(self.action_decoder, "requires_ema_for_loss", False))

    def get_training_loss_kwargs(self, ema_agent: BasePolicy | None = None) -> dict:
        """Return decoder-specific training dependencies in a generic form."""
        if not self.requires_ema_for_loss:
            return {}
        if ema_agent is None:
            raise RuntimeError(
                f"{type(self.action_decoder).__name__} requires an EMA model for training"
            )
        return {"ema_decoder": ema_agent.action_decoder}

    def validate_batch(self, batch: dict) -> None:
        """Check action windows and matching observation batches before normalization."""
        action = batch["action"]
        if action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"action must be (B, {self.horizon}, {self.action_dim}), got {tuple(action.shape)}"
            )
        self.validate_observations(batch["obs"], expected_batch=action.shape[0])

    def validate_observations(self, obs_dict: dict, expected_batch: int | None = None) -> None:
        """Require (B, T, ...) tensors with T >= n_obs_steps and a common B."""
        for key, value in obs_dict.items():
            if not torch.is_tensor(value):
                continue
            if value.ndim < 2 or value.shape[1] < self.n_obs_steps:
                raise ValueError(
                    f"obs['{key}'] requires (B, T >= {self.n_obs_steps}, ...), "
                    f"got {tuple(value.shape)}"
                )
            if expected_batch is None:
                expected_batch = value.shape[0]
            elif value.shape[0] != expected_batch:
                raise ValueError(
                    f"obs['{key}'] batch size {value.shape[0]} differs from {expected_batch}"
                )

    def preprocess(self, obs_dict: dict) -> dict:
        result = {}
        for key, value in obs_dict.items():
            if torch.is_tensor(value):
                value = value[:, : self.n_obs_steps]
                # Runners may supply XYZRGB even when the policy consumes only XYZ.
                if key == "point_cloud" and self.normalization_dims.get(key) == 3:
                    value = value[..., :3]
                if key in self.normalizer.params_dict:
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

    def loss_groups(self) -> dict[str, tuple[int, int]] | None:
        """Optional named action slices for separate diffusion losses."""
        return None

    def forward(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        return self.compute_loss(batch, **kwargs)

    def compute_loss(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        self.validate_batch(batch)
        cond = self.encode_observation(batch["obs"])
        normed_actions = self.normalizer["action"].normalize(batch["action"])
        groups = self.loss_groups()
        if groups is not None:
            kwargs["dim_groups"] = groups
        return self.action_decoder.compute_loss(cond, normed_actions, **kwargs)

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: dict,
        inference_steps: int | None = None,
        *,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        self.validate_observations(obs_dict)
        cond = self.encode_observation(obs_dict)
        if rtc_prefix is None or rtc_guidance_cap == 0:
            return self.predict_action_from_cond(cond, inference_steps=inference_steps)
        if type(self).predict_action_from_cond is not BasePolicy.predict_action_from_cond:
            raise NotImplementedError("RTC requires the continuous BasePolicy DDIM path")
        return self.predict_action_from_cond(
            cond,
            inference_steps=inference_steps,
            rtc_prefix=rtc_prefix,
            delay_steps=delay_steps,
            rtc_guidance_cap=rtc_guidance_cap,
        )

    @property
    def control_action_dim(self) -> int:
        """Action dimensions sent to the environment, excluding auxiliary predictions."""
        return self.action_dim

    @torch.no_grad()
    def predict_action_from_cond(
        self,
        cond: torch.Tensor,
        inference_steps: int | None = None,
        *,
        rtc_prefix: torch.Tensor | None = None,
        delay_steps: int = 0,
        rtc_guidance_cap: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        template = torch.zeros(
            cond.shape[0],
            self.horizon,
            self.action_dim,
            device=cond.device,
            dtype=cond.dtype,
        )
        kwargs = {}
        if rtc_prefix is not None and rtc_guidance_cap != 0:
            if not isinstance(self.action_decoder, Diffusion):
                raise NotImplementedError("RTC requires the DDIM decoder")
            params = self.normalizer["action"].params_dict
            scale = params["scale"][: self.control_action_dim]
            offset = params["offset"][: self.control_action_dim]
            normalized = rtc_prefix.to(template) * scale + offset
            kwargs = dict(
                rtc_prefix=normalized,
                delay_steps=delay_steps,
                history_offset=self.n_obs_steps - 1,
                rtc_guidance_cap=rtc_guidance_cap,
            )
        pred = self.action_decoder.predict_action(
            cond, template, inference_steps=inference_steps, **kwargs
        )
        pred = self.normalizer["action"].unnormalize(pred)
        return self.format_action_output(pred)

    def format_action_output(self, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        start = self.n_obs_steps - 1
        stop = start + self.n_action_steps
        control_dim = self.control_action_dim
        return {
            "pred_action": prediction,
            "control_action": prediction[:, start:stop, :control_dim],
            "tail": prediction[:, stop:, :control_dim],
        }

    def compile_backbone(self, **compile_kwargs: Any) -> None:
        self.action_decoder.model = torch.compile(self.action_decoder.model, **compile_kwargs)

    def optimizer_groups(
        self, lr: float, obs_lr: float, weight_decay: float, obs_weight_decay: float
    ) -> list[dict]:
        return parameter_groups(
            self.action_decoder.model, lr=lr, weight_decay=weight_decay
        ) + parameter_groups(self.obs_encoder, lr=obs_lr, weight_decay=obs_weight_decay)

    def validate_optimizer_groups(self, groups: list[dict]) -> None:
        """Require every trainable policy parameter exactly once across all groups."""
        names = {id(param): name for name, param in self.named_parameters() if param.requires_grad}
        param_ids = [id(param) for group in groups for param in group["params"]]
        assigned = set(param_ids)
        if len(param_ids) != len(assigned):
            raise ValueError("Optimizer groups contain duplicate parameters")
        if assigned - names.keys():
            raise ValueError(
                "Optimizer groups must contain only this policy's trainable parameters"
            )
        missing = names.keys() - assigned
        if missing:
            raise ValueError(
                "Trainable parameters missing from optimizer: "
                f"{sorted(names[key] for key in missing)}"
            )

    def configure_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_lr: float | None = None,
        obs_weight_decay: float | None = None,
        betas: tuple[float, float] = (0.95, 0.999),
    ) -> torch.optim.AdamW:
        obs_lr = obs_lr if obs_lr is not None else lr
        obs_weight_decay = obs_weight_decay if obs_weight_decay is not None else weight_decay
        if obs_lr == 0:
            self.obs_encoder.requires_grad_(False)
        groups = self.optimizer_groups(lr, obs_lr, weight_decay, obs_weight_decay)
        groups = [group for group in groups if group["params"]]
        self.validate_optimizer_groups(groups)
        names = {id(p): n for n, p in self.named_parameters()}
        for group in groups:
            group["param_names"] = [names[id(p)] for p in group["params"]]
        return torch.optim.AdamW(
            groups,
            lr=lr,
            betas=betas,
            fused=torch.cuda.is_available(),
        )


class UNetDiffusionPolicy(BasePolicy):
    def __init__(
        self,
        obs_encoder: nn.Module,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        modality_dropout_probs: dict[str, float] | None = None,
        cond_predict_scale: bool = True,
        clip_sample: bool = True,
    ) -> None:
        backbone = ConditionalUNet1D(
            input_dim=action_dim,
            context_dim=obs_encoder.out_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        action_decoder = Diffusion(
            backbone,
            num_training_steps,
            num_inference_steps,
            prediction_type,
            clip_sample=clip_sample,
        )
        super().__init__(
            obs_encoder,
            action_decoder,
            horizon,
            n_obs_steps,
            n_action_steps,
            action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )
