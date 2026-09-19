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


class ManiFlowObsEncoder(nn.Module):
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
    ):
        super().__init__()
        pc_encoder_config = dict(pc_encoder_config or {})
        pc_encoder_config.setdefault("fps_random_config", fps_random_config)
        self.pc_encoder = build_pc_patch_tokenizer(
            encoder_type, pc_dim, pc_encoder_config
        )
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        self.num_points = num_points
        self.use_coord_only = pc_dim == 3
        self.n_obs_steps = n_obs_steps
        self.fps_random_config = fps_random_config or {}

        token_seq_len, pc_out_dim = self.pc_encoder.out_shape
        if getattr(self.pc_encoder, "supports_global_token", True):
            self.num_obs_tokens = (token_seq_len + 1) * n_obs_steps
        else:
            self.num_obs_tokens = token_seq_len * n_obs_steps
        self.obs_token_dim = pc_out_dim + self.state_mlp.out_dim

    def forward(self, obs: dict):
        pc = preprocess_point_cloud(
            obs["point_cloud"],
            self.num_points,
            self.use_coord_only,
            self.fps_random_config,
            training=self.training,
        )

        if getattr(self.pc_encoder, "supports_global_token", True):
            pc_outputs = self.pc_encoder(pc, return_global_token=True)
            patch_token, _, global_token = pc_outputs[0], pc_outputs[1], pc_outputs[2]
            pc_feat = torch.cat([global_token, patch_token], dim=1)
        else:
            pc_feat = self.pc_encoder(pc)

        state_feat = self.state_mlp(obs["joint_state"])
        state_feat = state_feat.unsqueeze(1).expand(-1, pc_feat.size(1), -1)
        feat = torch.cat([pc_feat, state_feat], dim=-1)

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
        )

        backbone = ConsistencyDiTX(
            horizon=horizon,
            action_dim=action_dim,
            num_obs_tokens=obs_encoder.num_obs_tokens,
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


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, A, N = 2, 2, 16, 19, 256

    agent = ManiFlowAgent(
        horizon=H,
        n_obs_steps=T,
        n_action_steps=8,
        action_dim=A,
        encoder_type="pointnet_dense",
        pc_dim=3,
        state_dim=A,
        num_points=N,
        pc_encoder_config={
            "out_channels": 128,
            "num_points": N,
            "hidden_dims": (64, 128, 256),
        },
        n_layers=2,
        hidden_dim=128,
        n_head=4,
        mlp_ratio=2.0,
        p_drop_attn=0.0,
        timestep_embed_dim=64,
        target_t_embed_dim=64,
        num_inference_steps=5,
    ).to(device)

    obs = {
        "point_cloud": torch.randn(B * T, N, 3, device=device),
        "joint_state": torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    from dexmani_policy.common.normalizer import LinearNormalizer

    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": action,
            "joint_state": obs["joint_state"].reshape(B, T, A),
        },
        mode="limits",
    )
    agent.load_normalizer_from_dataset(normalizer)

    batch = {
        "obs": {
            "point_cloud": obs["point_cloud"].reshape(B, T, N, 3),
            "joint_state": obs["joint_state"].reshape(B, T, A),
        },
        "action": action,
    }

    import copy

    ema_agent = copy.deepcopy(agent)
    loss_kwargs = agent.get_training_loss_kwargs(ema_agent)
    loss, loss_dict = agent.compute_loss(batch, **loss_kwargs)
    print(f"loss: {loss.item():.4f}  keys={list(loss_dict.keys())}")

    result = agent.predict_action(batch["obs"])
    print(f"pred_action: {result['pred_action'].shape}")
    print(f"control_action: {result['control_action'].shape}")


if __name__ == "__main__":
    example()
