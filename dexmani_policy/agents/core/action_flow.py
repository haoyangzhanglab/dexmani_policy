from __future__ import annotations

import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.action_flow_flowmatch import SimpleRectifiedFlowDecoder
from dexmani_policy.agents.action_decoders.backbone.action_flow_dit import ActionFlowDiT
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.obs_encoder.pointcloud.geoformer import GeoFormer
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.pointnext_tokenizer import PointNextPatchTokenizer
from dexmani_policy.agents.position_encodings import SinusoidalPosEmb3D


class ActionFlowObsEncoder(nn.Module):
    """Two-frame 3D geometry memory for ActionFlow.

    PointNeXT extracts *local* patch geometry only; a joint GeoFormer over both
    observation frames does the patch-to-patch and frame-to-frame relational
    reasoning; the result is a static ``[B, 2*num_patches+1, memory_dim]``
    memory consumed by the ActionDiT through cross-attention.

    ``joint_state`` deliberately never enters the geometry tokens -- it is
    returned separately and reaches the ActionDiT only as global modulation.
    """

    def __init__(
        self,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        geo_hidden_dim: int = 576,
        geo_depth: int = 4,
        geo_num_heads: int = 12,
        geo_ffn_hidden_dim: int = 1536,
        geo_qk_norm: bool = True,
        geo_use_3d_rope: bool = True,
        geo_attn_drop: float = 0.0,
        geo_drop_path: float = 0.0,
        geo_min_wavelength: float = 0.02,
        geo_max_wavelength: float = 2.0,
        absolute_3d_pe_dim: int = 96,
        memory_dim: int = 768,
    ):
        super().__init__()
        cfg = dict(pc_encoder_config or {})
        self.pc_encoder = PointNextPatchTokenizer(
            input_channels=pc_dim,
            stem_channels=cfg.get("stem_channels", 128),
            token_channels=cfg.get("token_channels", 256),
            num_patches=cfg.get("num_patches", 192),
            patch_radii=tuple(cfg.get("patch_radii", (0.04, 0.08))),
            patch_neighbors=tuple(cfg.get("patch_neighbors", (16, 32))),
            fps_random_config=fps_random_config,
            use_patch_self_attn=cfg.get("use_patch_self_attn", False),
            include_global_token=False,
        )

        self.n_obs_steps = n_obs_steps
        self.state_dim = state_dim
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.fps_random_config = fps_random_config or {}

        token_channels = self.pc_encoder.out_dim
        num_patches = self.pc_encoder.out_shape[0]

        self.local_to_geo = nn.Linear(token_channels, geo_hidden_dim)
        self.frame_embedding = nn.Parameter(torch.randn(n_obs_steps, geo_hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, geo_hidden_dim) * 0.02)
        self.geoformer = GeoFormer(
            hidden_dim=geo_hidden_dim,
            depth=geo_depth,
            num_heads=geo_num_heads,
            ffn_hidden_dim=geo_ffn_hidden_dim,
            qk_norm=geo_qk_norm,
            use_3d_rope=geo_use_3d_rope,
            attn_drop=geo_attn_drop,
            drop_path_rate=geo_drop_path,
            min_wavelength=geo_min_wavelength,
            max_wavelength=geo_max_wavelength,
        )
        self.memory_proj = nn.Linear(geo_hidden_dim, memory_dim)
        self.abs_pe_embed = SinusoidalPosEmb3D(absolute_3d_pe_dim)
        self.abs_pe_proj = nn.Linear(absolute_3d_pe_dim, memory_dim)

        self.num_patches = num_patches
        self.num_memory_tokens = num_patches * n_obs_steps + 1
        self.memory_dim = memory_dim
        self.state_hist_dim = n_obs_steps * state_dim

    def forward(self, obs: dict) -> tuple[dict, dict]:
        """``obs`` tensors arrive flattened as ``[B*n_obs_steps, ...]`` (B-major).

        Returns ``({"memory": [B, 2G+1, memory_dim], "state": [B, T*state_dim]}, {})``.
        """
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.use_coord_only,
            self.fps_random_config,
            training=self.training,
        )
        patch_tokens, patch_center = self.pc_encoder(pc)

        T, G = self.n_obs_steps, self.num_patches
        B = patch_tokens.shape[0] // T

        # Local tokens -> geometry width, tagged with which frame they came from.
        tokens = self.local_to_geo(patch_tokens).reshape(B, T, G, -1)
        tokens = tokens + self.frame_embedding[None, :, None, :].to(dtype=tokens.dtype)
        tokens = tokens.reshape(B, T * G, -1)
        xyz = patch_center.reshape(B, T * G, 3)

        # Both frames enter the same GeoFormer so it can relate them; the CLS
        # token sits at index 0 with xyz=[0,0,0] (identity rotary phase).
        cls = self.cls_token.expand(B, -1, -1).to(dtype=tokens.dtype)
        tokens = torch.cat([cls, tokens], dim=1)
        xyz = torch.cat([xyz.new_zeros(B, 1, 3), xyz], dim=1)

        tokens = self.geoformer(tokens, xyz)

        # 3D RoPE inside GeoFormer carries *relative* geometry; this absolute
        # term carries where in the workspace each patch actually sits.
        memory = self.memory_proj(tokens) + self.abs_pe_proj(
            self.abs_pe_embed(xyz).to(dtype=tokens.dtype)
        )

        state = obs["joint_state"]
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"joint_state last dim {state.shape[-1]} != state_dim {self.state_dim}"
            )
        state = state.reshape(B, T * self.state_dim)

        return {"memory": memory, "state": state}, {}


class ActionFlowAgent(BaseAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        state_dim: int,
        pc_dim: int,
        num_points: int,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        geo_hidden_dim: int = 576,
        geo_depth: int = 4,
        geo_num_heads: int = 12,
        geo_ffn_hidden_dim: int = 1536,
        geo_qk_norm: bool = True,
        geo_use_3d_rope: bool = True,
        geo_attn_drop: float = 0.0,
        geo_drop_path: float = 0.0,
        geo_min_wavelength: float = 0.02,
        geo_max_wavelength: float = 2.0,
        absolute_3d_pe_dim: int = 96,
        hidden_dim: int = 768,
        context_dim: int | None = None,
        depth: int = 8,
        num_heads: int = 12,
        ffn_hidden_dim: int = 1536,
        timestep_embed_dim: int = 128,
        step_embed_dim: int = 64,
        state_embed_hidden_dim: int = 256,
        cond_bottleneck_dim: int = 384,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        use_step_conditioning: bool = False,
        denoise_steps: int = 2,
        noise_shift_alpha: float = 3.0,
        noise_shift_ratio: float = 0.75,
        solver: str = "euler",
        modality_dropout_probs: dict | None = None,
        **kwargs,
    ):
        # Resolved once so the memory width and the cross-attention context
        # width cannot drift apart when context_dim is left unset.
        ctx_dim = hidden_dim if context_dim is None else context_dim

        obs_encoder = ActionFlowObsEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
            geo_hidden_dim=geo_hidden_dim,
            geo_depth=geo_depth,
            geo_num_heads=geo_num_heads,
            geo_ffn_hidden_dim=geo_ffn_hidden_dim,
            geo_qk_norm=geo_qk_norm,
            geo_use_3d_rope=geo_use_3d_rope,
            geo_attn_drop=geo_attn_drop,
            geo_drop_path=geo_drop_path,
            geo_min_wavelength=geo_min_wavelength,
            geo_max_wavelength=geo_max_wavelength,
            absolute_3d_pe_dim=absolute_3d_pe_dim,
            memory_dim=ctx_dim,
        )
        backbone = ActionFlowDiT(
            horizon=horizon,
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            context_dim=ctx_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            timestep_embed_dim=timestep_embed_dim,
            step_embed_dim=step_embed_dim,
            state_embed_hidden_dim=state_embed_hidden_dim,
            cond_bottleneck_dim=cond_bottleneck_dim,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            use_step_conditioning=use_step_conditioning,
        )
        action_decoder = SimpleRectifiedFlowDecoder(
            model=backbone,
            denoise_steps=denoise_steps,
            noise_shift_alpha=noise_shift_alpha,
            noise_shift_ratio=noise_shift_ratio,
            solver=solver,
        )
        super().__init__(
            obs_encoder=obs_encoder,
            action_decoder=action_decoder,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            modality_dropout_probs=modality_dropout_probs,
        )
    @torch.no_grad()
    def predict_action_from_cond(self, cond: dict, denoise_timesteps=None) -> dict:
        """Local override: ActionFlow's ``cond`` is a dict, not a tensor.

        ``BaseAgent.predict_action_from_cond`` reads ``cond.shape``/``.device``/
        ``.dtype`` directly, so it cannot consume the ``{"memory", "state"}``
        contract. Behaviour is otherwise identical to the base implementation.
        """
        memory = cond["memory"]
        template = torch.zeros(
            memory.shape[0],
            self.horizon,
            self.action_dim,
            device=memory.device,
            dtype=memory.dtype,
        )
        pred = self.action_decoder.predict_action(cond, template, denoise_timesteps)
        pred = self.normalizer["action"].unnormalize(pred)

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

    def compile_backbone(self, **compile_kwargs):
        """Compile the ActionDiT backbone *and* the GeoFormer (ActionFlow-specific).

        ``BaseAgent.compile_backbone`` only compiles ``action_decoder.model``; the
        GeoFormer is a pure-torch transformer (no PyTorch3D FPS/Ball-Query/KNN) so
        it compiles cleanly. The PointNeXT tokenizer is deliberately left eager —
        its FPS/ball-query ops cause graph breaks.
        """
        super().compile_backbone(**compile_kwargs)
        self.obs_encoder.geoformer = torch.compile(
            self.obs_encoder.geoformer, **compile_kwargs
        )


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, A, state_dim, N = 2, 2, 16, 21, 19, 1024

    agent = ActionFlowAgent(
        horizon=H,
        n_obs_steps=T,
        n_action_steps=8,
        action_dim=A,
        state_dim=state_dim,
        pc_dim=6,
        num_points=N,
        pc_encoder_config={
            "num_patches": 192,
            "stem_channels": 128,
            "token_channels": 256,
            "patch_radii": [0.04, 0.08],
            "patch_neighbors": [16, 32],
            "use_patch_self_attn": False,
        },
        context_dim=384,
        denoise_steps=2,
        solver="midpoint",
    ).to(device)

    perception = sum(p.numel() for p in agent.obs_encoder.parameters())
    backbone = sum(p.numel() for p in agent.action_decoder.model.parameters())
    total = sum(p.numel() for p in agent.parameters())
    print(f"perception params: {perception:,}")
    print(f"backbone params:   {backbone:,}")
    print(f"total params:      {total:,}")

    obs = {
        "point_cloud": torch.randn(B * T, N, 6, device=device),
        "joint_state": torch.randn(B * T, state_dim, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    cond, _ = agent.obs_encoder(obs)
    print(f"memory shape: {tuple(cond['memory'].shape)}")
    print(f"state shape:  {tuple(cond['state'].shape)}")

    from dexmani_policy.common.normalizer import LinearNormalizer

    normalizer = LinearNormalizer()
    normalizer.fit(
        {"action": action, "joint_state": obs["joint_state"].reshape(B, T, state_dim)},
        mode="limits",
    )
    agent.load_normalizer_from_dataset(normalizer)

    batch = {
        "obs": {
            "point_cloud": obs["point_cloud"].reshape(B, T, N, 6),
            "joint_state": obs["joint_state"].reshape(B, T, state_dim),
        },
        "action": action,
    }
    loss, loss_dict = agent.compute_loss(batch)
    print(f"loss: {loss.item():.4f}  keys={list(loss_dict.keys())}")

    result = agent.predict_action({
        "point_cloud": obs["point_cloud"].reshape(B, T, N, 6),
        "joint_state": obs["joint_state"].reshape(B, T, state_dim),
    })
    print(f"pred_action: {tuple(result['pred_action'].shape)}")
    print(f"control_action: {tuple(result['control_action'].shape)}")
    print("=== PASSED ===")


if __name__ == "__main__":
    example()
