from __future__ import annotations

import warnings
from typing import Any, Dict

import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.backbone.ditx import DiTXFlowMatch
from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.flowmatch import FlowMatchWithConsistency
from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
from dexmani_policy.common.normalizer import LinearNormalizer


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
        self.normalizer = LinearNormalizer()
        self._dropout_warned_keys = set()

    def load_normalizer_from_dataset(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------
    def _validate_batch(self, batch: Dict) -> None:
        """Validate *action* shape and *obs* batch consistency before ``compute_loss``.

        Catches config-data mismatches (wrong ``action_key``, wrong
        ``horizon``) and corrupted DataLoader outputs at the earliest
        possible point, before normalisation or encoder forward.
        """
        action = batch.get("action")
        obs = batch.get("obs", {})

        if action is not None:
            if action.ndim != 3:
                raise ValueError(
                    f"action must be 3D (B, horizon, action_dim), "
                    f"got {action.ndim}D shape {tuple(action.shape)}"
                )
            B, H, A = action.shape
            if H != self.horizon:
                raise ValueError(
                    f"action horizon mismatch: got {H}, expected "
                    f"{self.horizon}.  Shape: {tuple(action.shape)}"
                )
            if A != self.action_dim:
                raise ValueError(
                    f"action dim mismatch: got {A}, expected "
                    f"{self.action_dim}.  Shape: {tuple(action.shape)}.  "
                    f"Check action_key / use_faas consistency."
                )

        self._validate_obs_dict(obs, expected_batch=B if action is not None else None)

    def _validate_obs_dict(self, obs_dict: Dict, expected_batch: int | None = None) -> None:
        """Validate observation tensor shapes.

        Every observation tensor must be at least 2D ``(B, T, ...)`` with
        ``T >= n_obs_steps``, and all modalities must share the same batch
        size.
        """
        if not obs_dict:
            return

        batch_sizes: set[int] = set()
        for key, value in obs_dict.items():
            if not torch.is_tensor(value):
                continue
            if value.ndim < 2:
                raise ValueError(
                    f"obs['{key}']: expected >=2D (B, n_obs_steps, ...), "
                    f"got {value.ndim}D shape {tuple(value.shape)}"
                )
            Bk, Tk = value.shape[0], value.shape[1]
            batch_sizes.add(Bk)
            if Tk < self.n_obs_steps:
                raise ValueError(
                    f"obs['{key}']: time dim too small — got {Tk}, "
                    f"need >= {self.n_obs_steps}.  Shape: {tuple(value.shape)}"
                )

        if len(batch_sizes) > 1:
            shapes = {k: tuple(v.shape) for k, v in obs_dict.items() if torch.is_tensor(v)}
            raise ValueError(f"Observation batch-size mismatch across modalities: {shapes}")

        if expected_batch is not None and batch_sizes:
            obs_b = next(iter(batch_sizes))
            if obs_b != expected_batch:
                shapes = {k: tuple(v.shape) for k, v in obs_dict.items() if torch.is_tensor(v)}
                raise ValueError(
                    f"Batch size mismatch: obs batch={obs_b}, "
                    f"action batch={expected_batch}.  Obs shapes: {shapes}"
                )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def preprocess(self, obs_dict: Dict) -> Dict:
        obs = self.normalizer.normalize(obs_dict)
        # Clamp normalized point cloud coordinates to guard against float32
        # precision drift (±1e-7) that would trigger ValueError in
        # Uni3DPointcloudEncoder.PositionEmbeddingRandom (requires [-1, 1]).
        # Mirrors the defensive clamp from R3D-Policy.
        if "point_cloud" in obs:
            obs["point_cloud"] = torch.clamp(obs["point_cloud"], min=-1 - 1e-6, max=1 + 1e-6)
        result = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                p = self.modality_dropout_probs.get(k, 0.0)
                if self.training and p > 0 and k in self.normalizer.params_dict:
                    mask = torch.rand(v.shape[0], device=v.device) > p
                    v = v * mask.view(-1, *([1] * (v.ndim - 1)))
                v = v[:, : self.n_obs_steps].flatten(0, 1)
            result[k] = v
        if self.training:
            for k, p in self.modality_dropout_probs.items():
                if p > 0 and k not in self.normalizer.params_dict and k not in self._dropout_warned_keys:
                    warnings.warn(
                        f"modality_dropout for '{k}' (prob={p}) has no effect: "
                        f"'{k}' is not in normalizer.params_dict. "
                        f"Only fitted modalities support dropout.",
                        UserWarning,
                    )
                    self._dropout_warned_keys.add(k)
        return result

    def _build_cond(self, obs_dict):
        obs = self.preprocess(obs_dict)
        cond, aux = self.obs_encoder(obs)
        return cond, aux

    def _get_dim_groups(self):
        """Override in subclasses to return per-head dimension slices.

        Returns ``None`` (standard single-loss) or a dict ``{name: (start, end)}``
        mapping loss component names to dimension ranges.
        """
        return None

    def compute_loss(self, batch, **kwargs):
        self._validate_batch(batch)
        cond, aux = self._build_cond(batch["obs"])
        normed_actions = self.normalizer["action"].normalize(batch["action"])
        if not torch.isfinite(normed_actions).all():
            nan_count = (~torch.isfinite(normed_actions)).sum().item()
            raw = batch["action"]
            raise ValueError(
                f"NaN/Inf in normalized actions ({nan_count}/{normed_actions.numel()} "
                f"elements). Raw action stats: min={raw.min():.4f} max={raw.max():.4f} "
                f"mean={raw.mean():.4f}. Check normalizer, Zarr data, or data pipeline."
            )
        action_loss, loss_dict = self.action_decoder.compute_loss(
            cond, normed_actions, dim_groups=self._get_dim_groups(), **kwargs
        )
        return self._merge_aux_loss(action_loss, loss_dict, aux)

    def _merge_aux_loss(self, action_loss, loss_dict, aux):
        """Merge auxiliary encoder losses (e.g. MoE load-balance) into the total.

        Subclasses may override this to add domain-specific logging or
        warnings.  The default implementation silently returns
        ``(action_loss, loss_dict)`` when *aux* contains no ``'loss'`` key.
        """
        aux_loss = aux.get("loss")
        if aux_loss is None:
            return action_loss, loss_dict

        total = action_loss + aux_loss
        loss_dict["loss"] = total
        # Preserve per-group loss_action if the action decoder already set it
        # (e.g. via dim_groups); otherwise use the raw action loss.
        loss_dict["loss_action"] = loss_dict.get("loss_action", action_loss)
        for k, v in aux.items():
            if k == "loss":
                continue
            # Log scalar aux values; skip multi-element tensors
            # (e.g. router_probs / dispatch) that can't be logged.
            if not torch.is_tensor(v) or v.numel() <= 1:
                loss_dict[f"aux_{k}"] = v
        return total, loss_dict

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict, denoise_timesteps=None) -> Dict:
        self._validate_obs_dict(obs_dict)
        # FAAS: convert env-native joint_state (19D) → FAAS (39D) before the
        # normalizer (which was fitted on FAAS data) sees it.  Skipped when
        # called from compute_loss (training) because the dataset already
        # outputs FAAS-dim data.
        if getattr(self, "use_faas", False):
            obs_dict = self._convert_obs_to_faas(obs_dict)
        cond, _ = self._build_cond(obs_dict)
        return self.predict_action_from_cond(cond, denoise_timesteps)

    def _convert_obs_to_faas(self, obs_dict: Dict) -> Dict:
        """Convert env-native ``joint_state`` (19D) to FAAS (39D).

        **Idempotent**: only converts when the input is native-dim
        (19D = 7 arm + 12 hand).  If already FAAS (39D), returns unchanged.
        This guards against double-conversion when called from training
        paths that receive FAAS-dim data from the dataset.

        ``joint_state`` arm is **always 7D** arm joint angles, regardless of
        ``action_key``.  The action's ``tcp_dim`` is irrelevant here.
        """
        if "joint_state" not in obs_dict:
            return obs_dict
        js = obs_dict["joint_state"]
        native_hand_dim = getattr(self.faas_mapper, "NATIVE_HAND_DIM", 12)
        # Only convert if native-dim; skip if already FAAS
        if js.shape[-1] != 7 + native_hand_dim:  # 19
            return obs_dict
        arm_state = js[..., :7]  # STATE_ARM_DIM = 7 (fixed)
        hand_state = js[..., 7:]
        faas_hand = self.faas_mapper.native_to_faas(hand_state)
        return {**obs_dict, "joint_state": torch.cat([arm_state, faas_hand], dim=-1)}

    @property
    def control_action_dim(self):
        """Number of action dimensions passed to ``env.step`` (native space).

        In FAAS mode the model trains in 39/41D but outputs are converted
        back to native 19/21D by ``predict_action_from_cond``, so the
        control dimension is always the native joint count.

        Override in subclasses (e.g. R3DAgent) to return only the primary
        (joint) dims when auxiliary heads are present.
        """
        if getattr(self, "use_faas", False):
            return self.tcp_dim + getattr(self.faas_mapper, "NATIVE_HAND_DIM", 12)
        return self.action_dim

    @torch.no_grad()
    def predict_action_from_cond(self, cond, denoise_timesteps=None):
        template = torch.zeros(
            cond.shape[0],
            self.horizon,
            self.action_dim,
            device=cond.device,
            dtype=cond.dtype,
        )
        pred = self.action_decoder.predict_action(cond, template, denoise_timesteps)
        pred = self.normalizer["action"].unnormalize(pred)

        # FAAS: convert the entire prediction from FAAS back to native
        # joint space so that pred_action / control_action / tail are all
        # native-dim and can be consumed directly by temporal ensembling
        # (which uses pred_action) and env.step (which uses control_action).
        if getattr(self, "use_faas", False):
            pred = self.faas_mapper.inverse_transform_action(pred, self.tcp_dim)

        start = self.n_obs_steps - 1
        control_action = pred[:, start : start + self.n_action_steps]
        # Unexecuted tail for temporal ensembling (ACT, Zhao et al. 2023).
        tail = pred[:, start + self.n_action_steps :]
        if self.control_action_dim != self.action_dim:
            control_action = control_action[..., : self.control_action_dim]
            tail = tail[..., : self.control_action_dim]
        return {
            "pred_action": pred,
            "control_action": control_action,
            "tail": tail,
        }

    @torch.no_grad()
    def compute_action_mse(self, batch: Dict[str, Any]) -> float:
        obs = batch["obs"]
        gt_action = batch["action"]
        # FAAS: gt_action from dataset is FAAS-dim; inverse-transform to
        # native so it matches the native pred_action from predict_action.
        if getattr(self, "use_faas", False):
            gt_action = self.faas_mapper.inverse_transform_action(gt_action, self.tcp_dim)
        pred_action = self.predict_action(obs)["pred_action"]
        return torch.nn.functional.mse_loss(pred_action, gt_action).item()

    def compile_backbone(self, **compile_kwargs):
        self.action_decoder.model = torch.compile(self.action_decoder.model, **compile_kwargs)

    def get_optim_param_groups(self, lr, obs_lr, weight_decay, obs_wd):
        action_groups = self.action_decoder.model.get_optim_groups(weight_decay)
        for g in action_groups:
            g["lr"] = lr
        obs_groups = get_optim_group_with_no_decay(self.obs_encoder, weight_decay=obs_wd)
        for g in obs_groups:
            g["lr"] = obs_lr
        return action_groups + obs_groups

    def _check_params_in_optimizer(self, optimizer: torch.optim.Optimizer):
        """Verify all trainable parameters are covered by the optimizer."""
        model_param_ids = {id(p) for p in self.parameters() if p.requires_grad}
        optim_param_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optim_param_ids.add(id(p))

        missing_ids = model_param_ids - optim_param_ids
        if missing_ids:
            missing_params = [p for p in self.parameters() if id(p) in missing_ids]
            param_info = []
            for p in missing_params:
                name = next((n for n, pp in self.named_parameters() if pp is p), "?")
                param_info.append(f"  {name}: shape={tuple(p.shape)}, device={p.device}")
            warnings.warn(
                f"The following {len(missing_ids)} trainable parameter(s) are NOT "
                f"tracked by the optimizer:\n"
                + "\n".join(param_info)
                + "\nThis usually means get_optim_param_groups() is missing a module. "
                "These parameters will not be updated during training.",
                UserWarning,
            )

    def configure_optimizer(
        self,
        lr,
        weight_decay,
        obs_lr=None,
        obs_weight_decay=None,
        betas=(0.95, 0.999),
    ):
        obs_lr = obs_lr if obs_lr is not None else lr
        obs_wd = obs_weight_decay if obs_weight_decay is not None else weight_decay
        groups = self.get_optim_param_groups(lr, obs_lr, weight_decay, obs_wd)
        optimizer = torch.optim.AdamW(
            [g for g in groups if g["params"]],
            lr=lr,
            betas=betas,
            fused=torch.cuda.is_available(),
        )
        self._check_params_in_optimizer(optimizer)
        return optimizer


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
    ):
        backbone = ConditionalUnet1D(
            input_dim=action_dim,
            context_dim=obs_encoder.out_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        action_decoder = Diffusion(backbone, num_training_steps, num_inference_steps, prediction_type)
        super().__init__(
            obs_encoder,
            action_decoder,
            horizon,
            n_obs_steps,
            n_action_steps,
            action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )


class DiTXFlowMatchAgent(BaseAgent):
    def __init__(
        self,
        obs_encoder: nn.Module,
        num_obs_tokens: int,
        obs_token_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        timestep_embed_dim: int = 128,
        target_t_embed_dim: int = 128,
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        pre_norm_modality: bool = False,
        denoise_timesteps: int = 10,
        flow_batch_ratio: float = 0.75,
        t_sample_mode_for_flow: str = "beta",
        t_sample_mode_for_consistency: str = "discrete",
        dt_sample_mode_for_consistency: str = "uniform",
        target_t_sample_mode: str = "relative",
        modality_dropout_probs: dict = None,
    ):
        backbone = DiTXFlowMatch(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            num_obs_tokens=num_obs_tokens,
            obs_token_dim=obs_token_dim,
            timestep_embed_dim=timestep_embed_dim,
            target_t_embed_dim=target_t_embed_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            pre_norm_modality=pre_norm_modality,
        )
        action_decoder = FlowMatchWithConsistency(
            model=backbone,
            denoise_timesteps=denoise_timesteps,
            flow_batch_ratio=flow_batch_ratio,
            t_sample_mode_for_flow=t_sample_mode_for_flow,
            t_sample_mode_for_consistency=t_sample_mode_for_consistency,
            dt_sample_mode_for_consistency=dt_sample_mode_for_consistency,
            target_t_sample_mode=target_t_sample_mode,
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


