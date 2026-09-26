import torch
import torch.nn as nn

from dexmani_policy.agents.action_decoders.backbone.consistency_ditx import (
    ConsistencyDiTX,
)
from dexmani_policy.agents.action_decoders.consistency_flow import (
    ConsistencyFlowMatch,
)
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.registry import (
    build_pc_patch_tokenizer,
)
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.position_encodings import NeRFSinusoidalPosEmb3D


class ManiFlowObsEncoder(nn.Module):
    """Dense point features plus projected XYZ PE and broadcast state.

    Inputs are augmented and normalized upstream. Sampling here supplies the
    same points to PointNet and XYZ PE. Flattening preserves frame-major order:
    all points of frame 0, then all points of frame 1, and so on.
    """

    @property
    def consumed_observation_fields(self) -> tuple[str, ...]:
        return ("joint_state", "point_cloud")

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
        xyz_pe_num_frequencies: int = 8,
    ):
        super().__init__()
        if encoder_type != "pointnet_dense":
            raise ValueError("ManiFlow requires pointnet_dense for point-aligned XYZ PE")
        if n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be greater than 0")
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("num_points", num_points)
        self.pc_encoder = build_pc_patch_tokenizer(
            encoder_type, pc_dim, pc_encoder_config
        )
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}

        pc_out_dim = self.pc_encoder.out_dim
        self.xyz_pe = NeRFSinusoidalPosEmb3D(xyz_pe_num_frequencies)
        self.xyz_pe_proj = nn.Sequential(
            nn.Linear(self.xyz_pe.out_dim, pc_out_dim),
            nn.LayerNorm(pc_out_dim),
        )
        self.obs_token_dim = pc_out_dim + self.state_mlp.out_dim

    def forward(self, obs: dict):
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.use_coord_only,
            self.fps_random_config,
            training=self.training,
        )

        # Both branches see the same augmented, normalized, sampled points.
        pc_feat = self.pc_encoder(pc) + self.xyz_pe_proj(self.xyz_pe(pc[..., :3]))

        state_feat = self.state_mlp(obs["joint_state"])
        state_feat = state_feat.unsqueeze(1).expand(-1, pc_feat.size(1), -1)
        feat = torch.cat([pc_feat, state_feat], dim=-1)

        if feat.shape[0] % self.n_obs_steps != 0:
            raise ValueError("observation batch must be divisible by n_obs_steps")
        batch_size = feat.shape[0] // self.n_obs_steps
        return feat.reshape(batch_size, -1, self.obs_token_dim), {}


class ManiFlowAgent(BaseAgent):
    """ManiFlow policy using ConsistencyDiTX + ConsistencyFlowMatch."""

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
        state_out_dim: int = 64,
        pc_encoder_config: dict | None = None,
        fps_random_config: dict | None = None,
        xyz_pe_num_frequencies: int = 8,
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
        num_inference_steps: int = 10,
        denoise_timesteps: int = 10,
        flow_batch_ratio: float = 0.75,
        t_sample_mode_for_flow: str = "beta",
        t_sample_mode_for_consistency: str = "discrete",
        dt_sample_mode_for_consistency: str = "uniform",
        target_t_sample_mode: str = "relative",
        modality_dropout_probs: dict | None = None,
    ):
        obs_encoder = ManiFlowObsEncoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            state_out_dim=state_out_dim,
            pc_encoder_config=pc_encoder_config,
            fps_random_config=fps_random_config,
            xyz_pe_num_frequencies=xyz_pe_num_frequencies,
        )

        backbone = ConsistencyDiTX(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.obs_token_dim,
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
        action_decoder = ConsistencyFlowMatch(
            model=backbone,
            num_inference_steps=num_inference_steps,
            denoise_timesteps=denoise_timesteps,
            flow_batch_ratio=flow_batch_ratio,
            t_sample_mode_for_flow=t_sample_mode_for_flow,
            t_sample_mode_for_consistency=t_sample_mode_for_consistency,
            dt_sample_mode_for_consistency=dt_sample_mode_for_consistency,
            target_t_sample_mode=target_t_sample_mode,
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
