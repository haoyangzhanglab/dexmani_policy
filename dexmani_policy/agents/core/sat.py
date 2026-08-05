"""SATAgent — Structural Action Transformer agent for DexMani_Policy.

Implements the structural-centric action representation from the SAT paper
(CVPR 2026): actions are transposed from ``(B, T, Da)`` to ``(B, Da, T)``
so that each Transformer token represents one joint's full future trajectory.

The agent wraps:
- ``SATObsEncoder``: PointNeXT patch tokenizer + StateMLP (same as ManiFlow)
- ``SATBackbone``: structural-centric DiT with MultiModalAttention and EJC
- ``SATFlowMatch``: Flow Matching decoder with shuffle support
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.backbone.sat import SATBackbone
from dexmani_policy.agents.action_decoders.sat_flowmatch import SATFlowMatch
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_patch_tokenizer
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.common.normalizer import LinearNormalizer


class SATObsEncoder(nn.Module):
    """Observation encoder for SAT — paper §4.2 temporal fusion in feature dim.

    Encodes raw point clouds and joint state into a sequence of observation
    tokens consumed as the KV prefix by the SAT backbone.

    Unlike the default ManiFlow pattern (time concatenated along sequence dim),
    SAT fuses observation history along the *feature* dimension so that the
    token count stays at ``num_patches + 1`` regardless of ``n_obs_steps``.
    This keeps the obs:action token ratio balanced (§4.2).

    Output shape: ``(B, num_obs_tokens, obs_token_dim)`` where
    ``num_obs_tokens = num_patches + 1`` and
    ``obs_token_dim = n_obs_steps * (pc_out_dim + state_out_dim)``.
    """

    def __init__(
        self,
        encoder_type: str,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ):
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("fps_random_config", fps_random_config)

        self.pc_encoder = build_pc_patch_tokenizer(
            encoder_type, pc_dim, pc_encoder_config)
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        self.num_points = num_points
        self.use_coord_only = (pc_dim == 3)
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}

        patch_seq_len, pc_out_dim = self.pc_encoder.out_shape
        # Paper §4.2: time fused in feature dim → token count independent of T
        self.num_obs_tokens = patch_seq_len + 1
        self.obs_token_dim = n_obs_steps * (pc_out_dim + self.state_mlp.out_dim)

    def forward(self, obs: dict):
        """Encode observations with paper-style temporal feature fusion.

        Paper §4.2: each frame is encoded independently, then same-position
        tokens across time are concatenated along the feature dimension.
        This yields ``num_patches + 1`` tokens (not ``T*(num_patches + 1)``).

        Args:
            obs: dict with keys ``'point_cloud'`` (B*T, N, pc_dim) and
                 ``'joint_state'`` (B*T, state_dim)

        Returns:
            ``(cond, aux)`` where cond is ``(B, num_obs_tokens, obs_token_dim)``
        """
        pc = preprocess_point_cloud(
            obs['point_cloud'], self.num_points,
            self.use_coord_only, self.fps_random_config)

        pc_outputs = self.pc_encoder(pc, return_global_token=True)
        patch_token, _, global_token = (
            pc_outputs[0], pc_outputs[1], pc_outputs[2])
        # patch_token: (B*T, K, D_pc)
        # global_token: (B*T, D_pc)

        # [global_token, patch_0, patch_1, ...]
        pc_feat = torch.cat([global_token, patch_token], dim=1)  # (B*T, K+1, D_pc)

        # Broadcast state to every token
        state_feat = self.state_mlp(obs['joint_state'])  # (B*T, D_state)
        state_feat = state_feat.unsqueeze(1).expand(-1, pc_feat.size(1), -1)
        feat = torch.cat([pc_feat, state_feat], dim=-1)  # (B*T, K+1, D_pc+D_state)

        # Paper §4.2: fuse time in feature dim (not sequence dim)
        B = feat.shape[0] // self.n_obs_steps
        T = self.n_obs_steps
        D = feat.shape[-1]  # pc_out_dim + state_out_dim

        # (B*T, K+1, D) → (B, T, K+1, D) → (B, K+1, T*D)
        feat = feat.reshape(B, T, -1, D)
        feat = feat.transpose(1, 2).reshape(B, -1, T * D)

        return feat, {}


class SATAgent(BaseAgent):
    """Structural Action Transformer agent.

    Inherits ``BaseAgent`` directly (not ``DiTXFlowMatchAgent``) because
    SAT uses a fundamentally different action representation:
    ``(B, Da, T)`` instead of ``(B, T, Da)``.

    The agent handles axis transposition at the I/O boundary so that
    the backbone always operates on structural-centric tensors.
    """

    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        # Obs encoder
        encoder_type: str = "pointnext_tokenizer",
        pc_dim: int = 6,
        state_dim: int = 19,
        num_points: int = 1024,
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        # SAT backbone
        n_layers: int = 12,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        p_drop_attn: float = 0.1,
        # EJC
        ejc_num_embodiments: int = 1,
        ejc_num_functions: int | None = None,
        ejc_num_axes: int = 1,
        ejc_embodiment_dim: int = 8,
        ejc_function_dim: int = 32,
        ejc_axis_dim: int = 8,
        # Structural tokens
        shuffle_action_tokens: bool = True,
        # Flow matching
        denoise_timesteps: int = 10,
        t_sample_mode_for_flow: str = "beta",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        # BaseAgent
        modality_dropout_probs: dict | None = None,
    ):
        # 1. Observation encoder
        obs_encoder = SATObsEncoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )

        # 2. SAT backbone (structural-centric)
        backbone = SATBackbone(
            horizon=horizon,
            action_dim=action_dim,
            num_obs_tokens=obs_encoder.num_obs_tokens,
            obs_token_dim=obs_encoder.obs_token_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            p_drop_attn=p_drop_attn,
            ejc_num_embodiments=ejc_num_embodiments,
            ejc_num_functions=ejc_num_functions,
            ejc_num_axes=ejc_num_axes,
            ejc_embodiment_dim=ejc_embodiment_dim,
            ejc_function_dim=ejc_function_dim,
            ejc_axis_dim=ejc_axis_dim,
        )

        # 3. SATFlowMatch decoder (passes shuffle to backbone)
        action_decoder = SATFlowMatch(
            model=backbone,
            num_inference_steps=denoise_timesteps,
            t_sample_mode=t_sample_mode_for_flow,
            beta_s=beta_s,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )

        # 4. BaseAgent
        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )

        self.shuffle_action_tokens = shuffle_action_tokens

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch, **kwargs):
        """Compute Flow Matching loss with structural-centric actions.

        Actions are transposed from ``(B, T, Da)`` to ``(B, Da, T)``
        before being passed to the flow matching decoder.
        """
        self._validate_batch(batch)
        cond, aux = self._build_cond(batch['obs'])

        normed_actions = self.normalizer['action'].normalize(batch['action'])
        if not torch.isfinite(normed_actions).all():
            nan_count = (~torch.isfinite(normed_actions)).sum().item()
            raw = batch['action']
            raise ValueError(
                f"NaN/Inf in normalized actions ({nan_count}/{normed_actions.numel()} "
                f"elements). Raw action stats: min={raw.min():.4f} max={raw.max():.4f} "
                f"mean={raw.mean():.4f}."
            )

        # Axis transposition: (B, T, Da) → (B, Da, T)
        normed_actions = normed_actions.transpose(1, 2)

        action_loss, loss_dict = self.action_decoder.compute_loss(
            cond, normed_actions,
            shuffle=self.training and self.shuffle_action_tokens,
        )
        return self._merge_aux_loss(action_loss, loss_dict, aux)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, obs_dict, denoise_timesteps=None):
        """Predict action — delegates to ``predict_action_from_cond``."""
        self._validate_obs_dict(obs_dict)

        if getattr(self, 'use_faas', False):
            obs_dict = self._convert_obs_to_faas(obs_dict)

        cond, _ = self._build_cond(obs_dict)
        return self.predict_action_from_cond(cond, denoise_timesteps)

    @torch.no_grad()
    def predict_action_from_cond(self, cond, denoise_timesteps=None):
        """Inference from pre-built condition tensor.

        Uses structural-centric ``(B, Da, T)`` template matching the SAT
        backbone, then transposes back to ``(B, T, Da)`` for output.
        """
        # Template in structural-centric format: (B, Da, T) not (B, T, Da)
        template = torch.zeros(
            cond.shape[0], self.action_dim, self.horizon,
            device=cond.device, dtype=cond.dtype,
        )

        pred = self.action_decoder.predict_action(
            cond, template, denoise_timesteps)
        # pred is (B, Da, T)

        # Transpose back: (B, Da, T) → (B, T, Da)
        pred = pred.transpose(1, 2)

        pred = self.normalizer['action'].unnormalize(pred)

        # FAAS inverse transform
        if getattr(self, 'use_faas', False):
            pred = self.faas_mapper.inverse_transform_action(
                pred, self.tcp_dim)

        start = self.n_obs_steps - 1
        control_action = pred[:, start:start + self.n_action_steps]
        tail = pred[:, start + self.n_action_steps:]

        if self.control_action_dim != self.action_dim:
            control_action = control_action[..., :self.control_action_dim]
            tail = tail[..., :self.control_action_dim]

        return {
            'pred_action': pred,
            'control_action': control_action,
            'tail': tail,
        }

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile_backbone(self, **compile_kwargs):
        """Override to use ``mode='default'`` instead of ``'reduce-overhead'``.

        The shuffle path (``x[:, perm, :]``) produces data-dependent
        indexing that CUDA graphs cannot capture.  ``mode='reduce-overhead'``
        enables CUDA graphs and segfaults on the graph-capture attempt.
        ``mode='default'`` keeps inductor kernel fusion but skips CUDA
        graphs, which is safe for dynamic indexing.
        """
        import torch
        compile_kwargs['mode'] = 'default'  # force override (setdefault won't work)
        self.action_decoder.model = torch.compile(
            self.action_decoder.model, **compile_kwargs)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def get_optim_param_groups(self, lr, obs_lr, weight_decay, obs_wd):
        """Separate backbone and obs_encoder with different LR/WD."""
        action_groups = self.action_decoder.model.get_optim_groups(weight_decay)
        for g in action_groups:
            g['lr'] = lr

        from dexmani_policy.agents.optim_util import get_optim_group_with_no_decay
        obs_groups = get_optim_group_with_no_decay(
            self.obs_encoder, weight_decay=obs_wd)
        for g in obs_groups:
            g['lr'] = obs_lr

        return action_groups + obs_groups
