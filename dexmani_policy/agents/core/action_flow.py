from __future__ import annotations

import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.action_flow_flowmatch import SimpleRectifiedFlowDecoder
from dexmani_policy.agents.action_decoders.backbone.action_flow_dit import ActionFlowDiT
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_patch_tokenizer


class ActionFlowObsEncoder(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        n_obs_steps: int,
        hidden_dim: int = 512,
        context_dim: int | None = None,
        state_out_dim: int | None = None,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
    ):
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("fps_random_config", fps_random_config)
        self.pc_encoder = build_pc_patch_tokenizer(encoder_type, pc_dim, pc_encoder_config)
        self.n_obs_steps = n_obs_steps
        self.hidden_dim = hidden_dim
        self.context_dim = hidden_dim if context_dim is None else context_dim
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.fps_random_config = fps_random_config or {}

        token_channels = self.pc_encoder.out_dim
        num_patches = self.pc_encoder.out_shape[0]
        cdim = self.context_dim
        self._state_cond_geom = state_out_dim is not None

        if self._state_cond_geom:
            # State-conditioned geometry: state broadcast to each frame's
            # patch/global tokens, concat with point features, then project.
            sdim = state_out_dim
            self.patch_proj = nn.Linear(token_channels + sdim, cdim)
            self.global_proj = nn.Linear(token_channels + sdim, cdim)
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, sdim),
                nn.SiLU(),
                nn.Linear(sdim, sdim),
            )
            self.num_obs_tokens = (1 + num_patches) * n_obs_steps
        else:
            self.patch_proj = nn.Linear(token_channels, cdim)
            self.global_proj = nn.Linear(token_channels, cdim)
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, cdim),
                nn.SiLU(),
                nn.Linear(cdim, cdim),
            )
            self.state_embed = nn.Parameter(torch.randn(1, 1, cdim) * 0.02)
            self.num_obs_tokens = (1 + 1 + num_patches) * n_obs_steps

        self.global_embed = nn.Parameter(torch.randn(1, 1, cdim) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(1, 1, cdim) * 0.02)
        self.obs_time_embed = nn.Parameter(
            torch.randn(1, n_obs_steps, 1, cdim) * 0.02
        )
        self.obs_token_dim = cdim

    def forward(self, obs: dict):
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.use_coord_only,
            self.fps_random_config,
            training=self.training,
        )

        if getattr(self.pc_encoder, "supports_global_token", True):
            pc_out = self.pc_encoder(pc, return_global_token=True)
            patch_tokens, _, global_token = pc_out[0], pc_out[1], pc_out[2]
        else:
            patch_tokens = self.pc_encoder(pc)
            global_token = patch_tokens.max(dim=1).values
            global_token = global_token.unsqueeze(1)

        state_emb = self.state_encoder(obs["joint_state"])

        if self._state_cond_geom:
            # Broadcast state to each geometry token, concat, then project.
            patch_state = state_emb[:, None, :].expand(-1, patch_tokens.shape[1], -1)
            patch_tokens = self.patch_proj(
                torch.cat([patch_tokens, patch_state], dim=-1)
            ) + self.patch_embed
            global_token = self.global_proj(
                torch.cat([global_token, state_emb[:, None, :]], dim=-1)
            ) + self.global_embed
            tokens = torch.cat([global_token, patch_tokens], dim=1)
        else:
            patch_tokens = self.patch_proj(patch_tokens) + self.patch_embed
            global_token = self.global_proj(global_token) + self.global_embed
            state = state_emb.unsqueeze(1) + self.state_embed
            tokens = torch.cat([state, global_token, patch_tokens], dim=1)

        B = tokens.shape[0] // self.n_obs_steps
        tokens_per_frame = tokens.shape[1]
        tokens = tokens.reshape(B, self.n_obs_steps, tokens_per_frame, self.context_dim)
        tokens = tokens + self.obs_time_embed.to(dtype=tokens.dtype)
        return tokens.reshape(
            B, self.n_obs_steps * tokens_per_frame, self.context_dim
        ), {}


class ActionFlowAgent(BaseAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        encoder_type: str,
        pc_dim: int,
        state_dim: int,
        num_points: int,
        hidden_dim: int = 512,
        context_dim: int | None = None,
        state_out_dim: int | None = None,
        depth: int = 8,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        ffn_hidden_dim: int = 896,
        timestep_embed_dim: int = 128,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        denoise_steps: int = 2,
        noise_shift_alpha: float = 4.0,
        noise_shift_ratio: float = 0.75,
        solver: str = "euler",
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        modality_dropout_probs: dict | None = None,
        **kwargs,
    ):
        obs_encoder = ActionFlowObsEncoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
        )
        backbone = ActionFlowDiT(
            horizon=horizon,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            depth=depth,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            timestep_embed_dim=timestep_embed_dim,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
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


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, A, N = 2, 2, 16, 19, 256

    agent = ActionFlowAgent(
        horizon=H,
        n_obs_steps=T,
        n_action_steps=8,
        action_dim=A,
        encoder_type="pointnet_dense",
        pc_dim=3,
        state_dim=A,
        num_points=N,
        hidden_dim=128,
        depth=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_hidden_dim=256,
        denoise_steps=5,
        pc_encoder_config={"out_channels": 128, "num_points": N, "hidden_dims": (64, 128, 256)},
    ).to(device)

    obs = {
        "point_cloud": torch.randn(B * T, N, 3, device=device),
        "joint_state": torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    cond, _ = agent.obs_encoder(obs)
    print(f"cond (tokens): {cond.shape}  [B, T*K, obs_token_dim]")

    from dexmani_policy.common.normalizer import LinearNormalizer

    normalizer = LinearNormalizer()
    normalizer.fit({"action": action, "joint_state": obs["joint_state"].reshape(B, T, A)}, mode="limits")
    agent.load_normalizer_from_dataset(normalizer)

    batch = {
        "obs": {
            "point_cloud": obs["point_cloud"].reshape(B, T, N, 3),
            "joint_state": obs["joint_state"].reshape(B, T, A),
        },
        "action": action,
    }
    loss, loss_dict = agent.compute_loss(batch)
    print(f"loss: {loss.item():.4f}  keys={list(loss_dict.keys())}")

    result = agent.predict_action({
        "point_cloud": obs["point_cloud"].reshape(B, T, N, 3),
        "joint_state": obs["joint_state"].reshape(B, T, A),
    })
    print(f"pred_action: {result['pred_action'].shape}")
    print(f"control_action: {result['control_action'].shape}")
    print("=== PASSED ===")


if __name__ == "__main__":
    example()
