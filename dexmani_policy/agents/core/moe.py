import warnings

import torch
import torch.nn as nn
from torchvision.transforms import v2

from dexmani_policy.agents.core.base import UNetDiffusionAgent
from dexmani_policy.agents.obs_encoder.plugins.moe import MoE
from dexmani_policy.agents.obs_encoder.pointcloud.ops import preprocess_point_cloud
from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_global_encoder
from dexmani_policy.agents.obs_encoder.proprio.state_mlp import create_state_mlp
from dexmani_policy.agents.obs_encoder.rgb import R3M, ResNet
from dexmani_policy.agents.obs_encoder.rgb.registry import build_backbone


class MoEObsEncoder(nn.Module):
    def __init__(
        self,
        # PC params (optional — when rgb_backbone_name is not provided)
        encoder_type: str = None,
        pc_dim: int = None,
        pc_out_dim: int = None,
        num_points: int = None,
        fps_random_config: dict = None,
        # RGB params (optional — exclusive with PC params)
        rgb_backbone_name: str = None,
        rgb_backbone_config: dict = None,
        # Common params
        state_dim: int = None,
        n_obs_steps: int = 2,
        state_out_dim: int = 64,
        # MoE params
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = None,
        moe_hidden_ratio: float = None,
        moe_out_dim: int = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        aux_loss_weight: float = 1.0,
        use_boost: bool = False,
        boost_start_epoch: int = 0,
        boost_interval: int = 100,
        boost_experts_per_step: int = 4,
        boost_topk_per_step: int = 1,
        use_enhanced_gate: bool = False,
        gate_hidden_dim: int = None,
        gate_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.use_rgb = rgb_backbone_name is not None

        if self.use_rgb:
            # ── RGB backbone (mirrors DPObsEncoder) ──
            cfg = dict(rgb_backbone_config or {})
            self.crop_ratio = cfg.pop("crop_ratio", None)
            self.backbone, self.image_processor = build_backbone(rgb_backbone_name, config=cfg)
            backbone_out_dim = self.backbone.out_dim
        else:
            # ── PC backbone (existing logic) ──
            if encoder_type is None:
                raise ValueError("MoEObsEncoder: either rgb_backbone_name or encoder_type must be provided.")
            self.pc_encoder = build_pc_global_encoder(
                encoder_type,
                pc_dim,
                config={
                    "output_channels": pc_out_dim,
                    "fps_random_config": fps_random_config or {},
                },
            )
            self.num_points = num_points
            self.use_coord_only = pc_dim == 3
            self.fps_random_config = fps_random_config or {}
            backbone_out_dim = self.pc_encoder.out_dim

        # ── Common: state MLP + projection + MoE ──
        self.state_mlp = create_state_mlp(state_dim, state_out_dim)
        in_dim = backbone_out_dim + self.state_mlp.out_dim
        # Linear projection before MoE so the encoder output can be mapped
        # to a fixed embedding dimension (cond_obs_emb).
        proj_dim = moe_out_dim if moe_out_dim is not None else in_dim
        # Derive hidden_dim from ratio (like Transformer MLP ratio) when not
        # explicitly set.  Keeps configs DRY: change the backbone and
        # hidden_dim scales automatically.
        if moe_hidden_dim is None:
            if moe_hidden_ratio is not None:
                moe_hidden_dim = int(proj_dim * moe_hidden_ratio)
            else:
                moe_hidden_dim = 256  # fallback when moe_hidden_dim not configurable
        self.obs_proj = nn.Linear(in_dim, proj_dim)

        _activation = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(activation, nn.GELU)
        self.moe = MoE(
            dim=proj_dim,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=moe_hidden_dim,
            out_dim=moe_out_dim if moe_out_dim is not None else proj_dim,
            num_layers=moe_num_layers,
            lambda_load=lambda_load,
            beta_entropy=beta_entropy,
            aux_loss_weight=aux_loss_weight,
            use_boost=use_boost,
            boost_start_epoch=boost_start_epoch,
            boost_interval=boost_interval,
            boost_experts_per_step=boost_experts_per_step,
            boost_topk_per_step=boost_topk_per_step,
            use_enhanced_gate=use_enhanced_gate,
            gate_hidden_dim=gate_hidden_dim,
            gate_dropout=gate_dropout,
            activation=_activation,
        )
        self.n_obs_steps = n_obs_steps
        self.out_dim = self.moe.out_dim

    def encode_feat(self, obs: dict) -> torch.Tensor:
        if self.use_rgb:
            # ── RGB path (mirrors DPObsEncoder.forward) ──
            rgb = obs["rgb"]  # (B*T, 3, H, W) float32 [0,1]
            if self.training and self.crop_ratio is not None:
                h, w = rgb.shape[-2:]
                crop_size = int(min(h, w) * self.crop_ratio)
                rgb = v2.RandomCrop(size=crop_size)(rgb)
            rgb = self.image_processor.process_images(rgb)["image"]

            # channels_last: for CNN backbones (ResNet/R3M), convert to NHWC
            # layout to leverage cuDNN implicit NHWC convolution kernels.
            # ViT backbones (DINO/CLIP/SigLIP) use attention — skip.
            if isinstance(self.backbone, (ResNet, R3M)):
                rgb = rgb.to(memory_format=torch.channels_last)

            return torch.cat(
                [
                    self.backbone(rgb)["global_token"],
                    self.state_mlp(obs["joint_state"]),
                ],
                dim=-1,
            )
        else:
            # ── PC path (existing logic) ──
            pc = preprocess_point_cloud(
                obs["point_cloud"], self.num_points, self.use_coord_only, self.fps_random_config
            )
            return torch.cat(
                [
                    self.pc_encoder(pc)["global_token"],
                    self.state_mlp(obs["joint_state"]),
                ],
                dim=-1,
            )

    def forward(self, obs: dict, return_aux=True, override_idx=None):
        z = self.encode_feat(obs)
        z = self.obs_proj(z)
        if override_idx is not None:
            override_idx = override_idx.repeat_interleave(self.n_obs_steps)
        feat, aux = self.moe(z, return_aux=return_aux, override_idx=override_idx)
        B = feat.shape[0] // self.n_obs_steps
        return feat.reshape(B, -1), aux


class MoEAgent(UNetDiffusionAgent):
    def __init__(
        self,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        state_dim: int,
        # PC params (optional — used when rgb_backbone_name is None)
        encoder_type: str = None,
        pc_dim: int = None,
        pc_out_dim: int = None,
        num_points: int = None,
        fps_random_config: dict = None,
        # RGB params (optional — exclusive with PC params)
        rgb_backbone_name: str = None,
        rgb_backbone_config: dict = None,
        # Common
        state_out_dim: int = 64,
        # MoE params
        num_experts: int = 16,
        top_k: int = 2,
        moe_hidden_dim: int = None,
        moe_hidden_ratio: float = None,
        moe_out_dim: int = None,
        moe_num_layers: int = 2,
        lambda_load: float = 0.1,
        beta_entropy: float = 0.01,
        aux_loss_weight: float = 1.0,
        use_boost: bool = False,
        boost_start_epoch: int = 0,
        boost_interval: int = 100,
        boost_experts_per_step: int = 4,
        boost_topk_per_step: int = 1,
        use_enhanced_gate: bool = False,
        gate_hidden_dim: int = None,
        gate_dropout: float = 0.0,
        activation: str = "gelu",
        **kwargs,
    ):
        obs_encoder = MoEObsEncoder(
            encoder_type=encoder_type,
            pc_dim=pc_dim,
            pc_out_dim=pc_out_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            fps_random_config=fps_random_config,
            rgb_backbone_name=rgb_backbone_name,
            rgb_backbone_config=rgb_backbone_config,
            state_out_dim=state_out_dim,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
            moe_hidden_ratio=moe_hidden_ratio,
            moe_out_dim=moe_out_dim,
            moe_num_layers=moe_num_layers,
            lambda_load=lambda_load,
            beta_entropy=beta_entropy,
            aux_loss_weight=aux_loss_weight,
            use_boost=use_boost,
            boost_start_epoch=boost_start_epoch,
            boost_interval=boost_interval,
            boost_experts_per_step=boost_experts_per_step,
            boost_topk_per_step=boost_topk_per_step,
            use_enhanced_gate=use_enhanced_gate,
            gate_hidden_dim=gate_hidden_dim,
            gate_dropout=gate_dropout,
            activation=activation,
        )
        super().__init__(obs_encoder, horizon, n_obs_steps, n_action_steps, action_dim, **kwargs)

    def set_epoch(self, epoch: int):
        """Trainer hook – triggers boost expert/top_k schedule at epoch start."""
        if hasattr(self.obs_encoder, "moe"):
            self.obs_encoder.moe.update_expert_num(epoch)

    def _build_cond(self, obs_dict, override_idx=None):
        obs = self.preprocess(obs_dict)
        cond, aux = self.obs_encoder(obs, override_idx=override_idx)
        return cond, aux

    def compute_loss(self, batch, **kwargs):
        self._validate_batch(batch)
        cond, aux = self._build_cond(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        action_loss, loss_dict = self.action_decoder.compute_loss(cond, nactions, **kwargs)
        return self._merge_aux_loss(action_loss, loss_dict, aux)

    def _merge_aux_loss(self, action_loss, loss_dict, aux):
        # MoE aux loss is mandatory for correct training; warn if missing.
        if aux.get("loss") is None:
            warnings.warn(
                "MoEAgent.compute_loss: aux['loss'] is missing. "
                "Falling back to action_loss only. "
                "Check that obs_encoder was called with return_aux=True.",
                UserWarning,
            )
        return super()._merge_aux_loss(action_loss, loss_dict, aux)

    @torch.no_grad()
    def predict_action(self, obs_dict, denoise_timesteps=None, override_idx=None):
        self._validate_obs_dict(obs_dict)
        cond, _ = self._build_cond(obs_dict, override_idx=override_idx)
        return self.predict_action_from_cond(cond, denoise_timesteps)


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, A, N = 2, 2, 16, 19, 256

    # ── PC smoke test (backward compatible) ──
    print("=== MoEAgent PC smoke test ===")
    agent = MoEAgent(
        horizon=H,
        n_obs_steps=T,
        n_action_steps=8,
        action_dim=A,
        encoder_type="idp3",
        pc_dim=3,
        pc_out_dim=64,
        state_dim=A,
        num_points=N,
        num_experts=4,
        top_k=2,
        moe_hidden_dim=64,
        moe_num_layers=1,
        down_dims=[64, 128],
        diffusion_step_embed_dim=64,
        num_training_steps=10,
        num_inference_steps=3,
    ).to(device)

    obs = {
        "point_cloud": torch.randn(B * T, N, 3, device=device),
        "joint_state": torch.randn(B * T, A, device=device),
    }
    action = torch.randn(B, H, A, device=device)

    print(f"obs point_cloud: {obs['point_cloud'].shape}")
    print(f"obs joint_state: {obs['joint_state'].shape}")
    print(f"action:          {action.shape}")

    cond, aux = agent.obs_encoder(obs)
    print(f"cond:            {cond.shape}")
    print(f"aux loss:        {aux['loss'].item():.4f}")

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
    print(f"loss:            {loss.item():.4f}  keys={list(loss_dict.keys())}")

    result = agent.predict_action(
        {
            "point_cloud": obs["point_cloud"].reshape(B, T, N, 3),
            "joint_state": obs["joint_state"].reshape(B, T, A),
        }
    )
    print(f"pred_action:     {result['pred_action'].shape}")
    print(f"control_action:  {result['control_action'].shape}")

    override = torch.tensor([0, 1], dtype=torch.long, device=device)
    result_ov = agent.predict_action(
        {
            "point_cloud": obs["point_cloud"].reshape(B, T, N, 3),
            "joint_state": obs["joint_state"].reshape(B, T, A),
        },
        override_idx=override,
    )
    print(f"override pred:   {result_ov['pred_action'].shape}")
    print(f"override ctrl:   {result_ov['control_action'].shape}")

    print("\n--- Boost test ---")
    agent_boost = MoEAgent(
        horizon=H,
        n_obs_steps=T,
        n_action_steps=8,
        action_dim=A,
        encoder_type="idp3",
        pc_dim=3,
        pc_out_dim=64,
        state_dim=A,
        num_points=N,
        num_experts=8,
        top_k=2,
        moe_hidden_dim=64,
        moe_num_layers=1,
        use_boost=True,
        boost_start_epoch=0,
        boost_interval=50,
        boost_experts_per_step=4,
        boost_topk_per_step=1,
        down_dims=[64, 128],
        diffusion_step_embed_dim=64,
        num_training_steps=10,
        num_inference_steps=3,
    ).to(device)

    moe = agent_boost.obs_encoder.moe
    assert moe.use_boost
    print(f"base experts={moe.num_experts} top_k={moe.top_k}")

    agent_boost.set_epoch(0)
    print(f"epoch 0: active_experts={moe.current_num_experts} active_top_k={moe.current_top_k}")
    assert moe.current_num_experts == 4 and moe.current_top_k == 1, (
        f"Expected 4 experts, top_k=1 at epoch 0, got {moe.current_num_experts}, {moe.current_top_k}"
    )

    agent_boost.set_epoch(50)
    print(f"epoch 50: active_experts={moe.current_num_experts} active_top_k={moe.current_top_k}")
    assert moe.current_num_experts == 8 and moe.current_top_k == 2, (
        f"Expected 8 experts, top_k=2 at epoch 50, got {moe.current_num_experts}, {moe.current_top_k}"
    )

    agent_boost.set_epoch(200)
    print(f"epoch 200: active_experts={moe.current_num_experts} active_top_k={moe.current_top_k}")
    assert moe.current_num_experts == 8 and moe.current_top_k == 2, "Should cap at base values"

    print("\n--- Enhanced gate test ---")
    agent_gate = MoEAgent(
        horizon=H,
        n_obs_steps=T,
        n_action_steps=8,
        action_dim=A,
        encoder_type="idp3",
        pc_dim=3,
        pc_out_dim=64,
        state_dim=A,
        num_points=N,
        num_experts=4,
        top_k=2,
        moe_hidden_dim=64,
        moe_num_layers=1,
        use_enhanced_gate=True,
        gate_dropout=0.1,
        down_dims=[64, 128],
        diffusion_step_embed_dim=64,
        num_training_steps=10,
        num_inference_steps=3,
    ).to(device)
    gate_router = agent_gate.obs_encoder.moe.router
    assert isinstance(gate_router, nn.Sequential), (
        f"Enhanced gate should be nn.Sequential, got {type(gate_router)}"
    )
    print(f"gate type: {type(gate_router).__name__} (len={len(gate_router)})")

    feat, aux_gate = agent_gate.obs_encoder(obs)
    print(f"enhanced gate cond: {feat.shape}, aux loss: {aux_gate['loss'].item():.4f}")
    print("=== PC PASSED ===")

    # ── RGB smoke test ──
    print("\n=== MoEAgent RGB smoke test ===")
    agent_rgb = MoEAgent(
        horizon=H,
        n_obs_steps=T,
        n_action_steps=8,
        action_dim=A,
        rgb_backbone_name="resnet",
        rgb_backbone_config={"model_name": "resnet18", "tune_mode": "full", "norm_mode": "group_norm"},
        state_dim=A,
        num_experts=4,
        top_k=2,
        moe_hidden_dim=64,
        moe_num_layers=1,
        down_dims=[64, 128],
        diffusion_step_embed_dim=64,
        num_training_steps=10,
        num_inference_steps=3,
    ).to(device)

    obs_rgb = {
        "rgb": torch.rand(B * T, 3, 224, 224, device=device),
        "joint_state": torch.randn(B * T, A, device=device),
    }
    action_rgb = torch.randn(B, H, A, device=device)

    print(f"obs rgb:         {obs_rgb['rgb'].shape}")
    print(f"obs joint_state: {obs_rgb['joint_state'].shape}")
    print(f"action:          {action_rgb.shape}")

    cond_rgb, aux_rgb = agent_rgb.obs_encoder(obs_rgb)
    print(f"cond:            {cond_rgb.shape}")
    print(f"aux loss:        {aux_rgb['loss'].item():.4f}")

    normalizer_rgb = LinearNormalizer()
    normalizer_rgb.fit(
        {"action": action_rgb, "joint_state": obs_rgb["joint_state"].reshape(B, T, A)}, mode="limits"
    )
    agent_rgb.load_normalizer_from_dataset(normalizer_rgb)

    batch_rgb = {
        "obs": {
            "rgb": obs_rgb["rgb"].reshape(B, T, 3, 224, 224),
            "joint_state": obs_rgb["joint_state"].reshape(B, T, A),
        },
        "action": action_rgb,
    }
    loss_rgb, loss_dict_rgb = agent_rgb.compute_loss(batch_rgb)
    print(f"loss:            {loss_rgb.item():.4f}  keys={list(loss_dict_rgb.keys())}")

    result_rgb = agent_rgb.predict_action(
        {
            "rgb": obs_rgb["rgb"].reshape(B, T, 3, 224, 224),
            "joint_state": obs_rgb["joint_state"].reshape(B, T, A),
        }
    )
    print(f"pred_action:     {result_rgb['pred_action'].shape}")
    print(f"control_action:  {result_rgb['control_action'].shape}")
    print("=== RGB PASSED ===")


if __name__ == "__main__":
    example()
