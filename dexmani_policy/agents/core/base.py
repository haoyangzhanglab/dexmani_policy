from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.core.normalizer import AffineNormalizer


def parameter_groups(module: nn.Module, *, lr: float, weight_decay: float):
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


class BaseAgent(nn.Module):
    def __init__(
        self,
        obs_encoder: nn.Module,
        action_decoder: nn.Module,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        modality_dropout_probs: dict = None,
    ):
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

    def set_normalization_spec(self, spec, feature_dims=None):
        from dexmani_policy.utils.config import validate_action_clipping
        from dexmani_policy.agents.obs_encoder.pointcloud.sparse3d_encoder import Sparse3DEncoder

        if (isinstance(getattr(self.obs_encoder, "pc_encoder", None), Sparse3DEncoder)
                and spec.get("point_cloud") != "identity"):
            raise ValueError("Sparse3DEncoder requires normalization.point_cloud: identity (metric XYZ)")
        if isinstance(self.action_decoder, Diffusion):
            validate_action_clipping(spec, self.action_decoder.noise_scheduler.config.clip_sample)
        self.normalization_spec = dict(spec)
        self.normalization_dims = dict(feature_dims or {})
        if self.normalization_dims.get('action', self.action_dim) != self.action_dim:
            raise ValueError('Normalizer action width must match the full prediction layout')
        self.normalization_dims['action'] = self.action_dim

    def validate_normalizer(self):
        from dexmani_policy.agents.core.normalizer import validate_normalizer_state
        validate_normalizer_state(self.normalizer, self.normalization_spec, self.normalization_dims)

    def initialize_training(self) -> None:
        """Initialize fresh training weights; complete checkpoints skip this step."""

    def load_normalizer_from_dataset(self, normalizer: AffineNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @property
    def requires_ema_for_loss(self) -> bool:
        """Whether the action decoder needs a local EMA copy to build targets."""
        return bool(getattr(self.action_decoder, "requires_ema_for_loss", False))

    def get_training_loss_kwargs(self, ema_agent=None) -> dict:
        """Return decoder-specific training dependencies in a generic form."""
        if not self.requires_ema_for_loss:
            return {}
        if ema_agent is None:
            raise RuntimeError(
                f"{type(self.action_decoder).__name__} requires an EMA model for training"
            )
        return {"ema_decoder": ema_agent.action_decoder}

    def _validate_batch(self, batch: Dict) -> None:
        """Check action windows and matching observation batches before normalization."""
        action = batch.get("action")
        if action is not None and action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"action must be (B, {self.horizon}, {self.action_dim}), got {tuple(action.shape)}"
            )
        self._validate_obs_dict(
            batch.get("obs", {}), expected_batch=action.shape[0] if action is not None else None
        )

    def _validate_obs_dict(self, obs_dict: Dict, expected_batch: int | None = None) -> None:
        """Require (B, T, ...) tensors with T >= n_obs_steps and a common B."""
        for key, value in obs_dict.items():
            if not torch.is_tensor(value):
                continue
            if value.ndim < 2 or value.shape[1] < self.n_obs_steps:
                raise ValueError(
                    f"obs['{key}'] requires (B, T >= {self.n_obs_steps}, ...), got {tuple(value.shape)}"
                )
            if expected_batch is None:
                expected_batch = value.shape[0]
            elif value.shape[0] != expected_batch:
                raise ValueError(
                    f"obs['{key}'] batch size {value.shape[0]} differs from {expected_batch}"
                )

    def preprocess(self, obs_dict: Dict) -> Dict:
        # Sim runners can supply XYZRGB even when this run consumes XYZ only.
        # Select before normalization, just as Dataset and Real runtime do.
        if self.normalization_dims.get("point_cloud") == 3 and "point_cloud" in obs_dict:
            obs_dict = dict(obs_dict)
            obs_dict["point_cloud"] = obs_dict["point_cloud"][..., :3]
        obs = self.normalizer.normalize(obs_dict)
        result = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                p = self.modality_dropout_probs.get(k, 0.0)
                if self.training and p > 0:
                    mask = torch.rand(v.shape[0], device=v.device) > p
                    v = v * mask.view(-1, *([1] * (v.ndim - 1)))
                v = v[:, : self.n_obs_steps].flatten(0, 1)
            result[k] = v
        return result

    def encode_observation(self, obs_dict):
        return self.obs_encoder(self.preprocess(obs_dict))

    def loss_groups(self):
        """Optional named action slices for separate diffusion losses."""
        return None

    def forward(self, batch, **kwargs):
        return self.compute_loss(batch, **kwargs)

    def compute_loss(self, batch, **kwargs):
        self._validate_batch(batch)
        cond = self.encode_observation(batch["obs"])
        normed_actions = self.normalizer["action"].normalize(batch["action"])
        groups = self.loss_groups()
        if groups is not None:
            kwargs["dim_groups"] = groups
        return self.action_decoder.compute_loss(cond, normed_actions, **kwargs)

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict,
        inference_steps: int | None = None,
        *,
        rtc_prefix=None,
        delay_steps=0,
        rtc_guidance_cap=0.0,
    ) -> Dict:
        self._validate_obs_dict(obs_dict)
        cond = self.encode_observation(obs_dict)
        if rtc_prefix is None or rtc_guidance_cap == 0:
            return self.predict_action_from_cond(cond, inference_steps=inference_steps)
        if type(self).predict_action_from_cond is not BaseAgent.predict_action_from_cond:
            raise NotImplementedError("RTC requires the continuous BaseAgent DDIM path")
        return self.predict_action_from_cond(
            cond,
            inference_steps=inference_steps,
            rtc_prefix=rtc_prefix,
            delay_steps=delay_steps,
            rtc_guidance_cap=rtc_guidance_cap,
        )

    @property
    def control_action_dim(self):
        """Number of action dimensions passed to ``env.step``.

        Override in subclasses (e.g. R3DAgent) to return only the primary
        (joint) dims when auxiliary heads are present.
        """
        return self.action_dim

    @torch.no_grad()
    def predict_action_from_cond(
        self,
        cond,
        inference_steps: int | None = None,
        *,
        rtc_prefix=None,
        delay_steps=0,
        rtc_guidance_cap=0.0,
    ):
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
            # Saved affine statistics are sliced to the actual control subspace.
            params = self.normalizer["action"].params_dict
            c = self.control_action_dim
            scale, offset = params["scale"], params["offset"]
            scale = scale if scale.numel() == 1 else scale[:c]
            offset = offset if offset.numel() == 1 else offset[:c]
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

        start = self.n_obs_steps - 1
        control_action = pred[:, start : start + self.n_action_steps]
        # Remaining future action tail.
        tail = pred[:, start + self.n_action_steps :]
        if self.control_action_dim != self.action_dim:
            control_action = control_action[..., : self.control_action_dim]
            tail = tail[..., : self.control_action_dim]
        return {
            "pred_action": pred,
            "control_action": control_action,
            "tail": tail,
        }

    def compile_backbone(self, **compile_kwargs):
        self.action_decoder.model = torch.compile(self.action_decoder.model, **compile_kwargs)

    def optimizer_groups(self, lr, obs_lr, weight_decay, obs_weight_decay):
        return (
            parameter_groups(self.action_decoder.model, lr=lr, weight_decay=weight_decay)
            + parameter_groups(self.obs_encoder, lr=obs_lr, weight_decay=obs_weight_decay)
        )

    def validate_optimizer_groups(self, groups):
        """Require every trainable Agent parameter exactly once across all groups."""
        names = {id(param): name for name, param in self.named_parameters() if param.requires_grad}
        param_ids = [id(param) for group in groups for param in group["params"]]
        assigned = set(param_ids)
        if len(param_ids) != len(assigned):
            raise ValueError("Optimizer groups contain duplicate parameters")
        if assigned - names.keys():
            raise ValueError("Optimizer groups must contain only this Agent's trainable parameters")
        missing = names.keys() - assigned
        if missing:
            raise ValueError(f"Trainable parameters missing from optimizer: {sorted(names[key] for key in missing)}")

    def configure_optimizer(
        self,
        lr,
        weight_decay,
        obs_lr=None,
        obs_weight_decay=None,
        betas=(0.95, 0.999),
    ):
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


class UNetDiffusionAgent(BaseAgent):
    def __init__(
        self,
        obs_encoder: nn.Module,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_training_steps: int = 100,
        num_inference_steps: int = 10,
        prediction_type: str = "sample",
        modality_dropout_probs: dict = None,
        cond_predict_scale: bool = True,
        clip_sample: bool = True,
    ):
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
