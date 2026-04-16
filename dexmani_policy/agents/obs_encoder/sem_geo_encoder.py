import torch
import torch.nn as nn
from typing import Dict, List

# TODO: SemGeoEncoder is implemented but not yet wired to any agent.
# This encoder uses SpatialAligner for point cloud + image semantic-geometric fusion.
# When ready, create a SemGeoAgent or add encoder_type="semgeo" to ManiFlowAgent.

from dexmani_policy.agents.common.mlp import create_mlp
from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay


class SemGeoEncoder(nn.Module):
    """语义-几何融合编码器。

    接收已构建好的子模块，forward 仅负责编排：
        point_cloud → pc_encoder → patch tokens + centers + global token
        rgb + depth → image_processor + rgb_backbone → image patch tokens + 3D coords
        SpatialAligner 融合 point + image tokens
        最后拼接 joint_state

    输入 obs_dict keys:
        point_cloud      : (B*T, N, pc_dim)
        joint_state      : (B*T, state_dim)
        rgb              : (B*T, H, W, 3)  uint8
        depth            : (B*T, H, W) or (B*T, 1, H, W)
        camera_intrinsic : (B*T, 3, 3)
        camera_extrinsic : (B*T, 3, 4)  [可选]
    """

    STATE_OUT_DIM = 64

    def __init__(
        self,
        pc_encoder: nn.Module,
        rgb_backbone: nn.Module,
        image_processor,
        spatial_aligner: nn.Module,
        state_dim: int = 19,
        depth_scale: float = 1000.0,
        min_depth: float = 0.01,
        max_depth: float = 3.0,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.rgb_backbone = rgb_backbone
        self.image_processor = image_processor
        self.spatial_aligner = spatial_aligner
        self.state_mlp = create_mlp(state_dim, [64, self.STATE_OUT_DIM])

        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.out_shape = spatial_aligner.out_dim + self.STATE_OUT_DIM

    @property
    def passthrough_keys(self) -> set:
        return {"rgb", "depth", "camera_intrinsic", "camera_extrinsic"}

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        pc = obs_dict["point_cloud"]
        state = obs_dict["joint_state"]
        rgb = obs_dict["rgb"]
        depth = obs_dict["depth"]
        intrinsics = obs_dict["camera_intrinsic"]
        camera_to_world = obs_dict.get("camera_extrinsic", None)

        # 点云编码
        patch_token, patch_centers, global_scene_token = self.pc_encoder(pc)

        # RGB 编码 + 深度反投影
        processed = self.image_processor.process_rgbd(
            images=rgb,
            depths=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        vision_out = self.rgb_backbone(processed.image.to(pc.device))
        image_patch_tokens = vision_out["patch_tokens"]

        geo_out = self.rgb_backbone.backproject(
            depth=processed.depth.to(pc.device),
            intrinsics=processed.intrinsics.to(pc.device),
            camera_to_world=(
                processed.camera_to_world.to(pc.device)
                if processed.camera_to_world is not None
                else None
            ),
            depth_scale=self.depth_scale,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        image_patch_coords = geo_out["patch_coords"]
        image_patch_valid = geo_out["patch_valid_mask"]
        if image_patch_valid.ndim == 3 and image_patch_valid.shape[-1] == 1:
            image_patch_valid = image_patch_valid.squeeze(-1)
        if image_patch_valid.dtype != torch.bool:
            image_patch_valid = image_patch_valid > 0.5

        # SpatialAligner 融合
        fused_tokens, _ = self.spatial_aligner(
            point_token=patch_token,
            patch_center=patch_centers,
            image_patch_token=image_patch_tokens,
            image_patch_coord=image_patch_coords,
            image_patch_valid_mask=image_patch_valid,
        )

        # 重组 + state 拼接
        global_proj = self.spatial_aligner.point_proj(global_scene_token)
        all_tokens = torch.cat([global_proj, fused_tokens], dim=1)

        state_feat = self.state_mlp(state)
        state_feat = state_feat.unsqueeze(1).expand(-1, all_tokens.size(1), -1)
        return torch.cat([all_tokens, state_feat], dim=-1)

    def get_optim_groups(self, weight_decay: float) -> List[Dict]:
        return get_optim_group_with_no_decay(self, weight_decay)


def example():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    B, T, N = 2, 2, 1024
    H, W = 480, 640
    state_dim = 19

    obs = {
        "point_cloud": torch.randn(B * T, N, 3, device=device),
        "joint_state": torch.randn(B * T, state_dim, device=device),
        "rgb": torch.randint(0, 256, (B * T, H, W, 3), dtype=torch.uint8),
        "depth": torch.randint(1, 2000, (B * T, H, W), dtype=torch.int32),
        "camera_intrinsic": torch.tensor(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).unsqueeze(0).expand(B * T, -1, -1).clone(),
        "camera_extrinsic": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]],
            dtype=torch.float32,
        ).unsqueeze(0).expand(B * T, -1, -1).clone(),
    }

    from dexmani_policy.agents.obs_encoder.pointcloud.tokenizer import PointNextPatchTokenizer
    from dexmani_policy.agents.obs_encoder.plugins.spatial_aligner import SpatialAligner
    from dexmani_policy.agents.obs_encoder.rgb.registry import build_backbone

    pc_enc = PointNextPatchTokenizer(
        input_channels=3, stem_channels=64, token_channels=128, num_patches=96,
        patch_radii=(0.04, 0.08), patch_neighbors=(16, 32),
        global_radius=0.16, global_neighbors=32,
    )
    rgb_backbone, img_proc = build_backbone("dino")
    aligner = SpatialAligner(point_dim=128, image_dim=rgb_backbone.out_dim,
                             hidden_dim=256, out_dim=128)

    print("\n=== SemGeoEncoder (dino + tokenizer) ===")
    enc = SemGeoEncoder(
        pc_encoder=pc_enc,
        rgb_backbone=rgb_backbone,
        image_processor=img_proc,
        spatial_aligner=aligner,
        state_dim=state_dim,
    ).to(device)
    enc.eval()

    with torch.no_grad():
        feat = enc(obs)

    seq_len = 97  # 96 patches + 1 global
    print(f"  obs_seq_len = {seq_len}")
    print(f"  out_shape   = {enc.out_shape}")
    print(f"  feat.shape  = {tuple(feat.shape)}")
    print(f"  params: total={sum(p.numel() for p in enc.parameters()):,}")
    print(f"  passthrough_keys = {enc.passthrough_keys}")


if __name__ == "__main__":
    example()
