import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dexmani_policy.agents.common.mlp import create_mlp


class PCImageAligner(nn.Module):
    """Point cloud + image spatial alignment via kNN interpolation.

    For each point token, finds k nearest image patches in 3D space,
    aggregates their semantic features with inverse-distance weighting,
    then fuses with the original point token.

    Shapes:
        point_token          : (B, N_p, point_dim)
        patch_center         : (B, N_p, 3)
        image_patch_token    : (B, N_i, image_dim)
        image_patch_coord    : (B, N_i, 3)
        image_patch_valid_mask: (B, N_i) or None
        output               : (B, N_p, out_dim), aux dict
    """

    def __init__(
        self,
        point_dim: int,
        image_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        k: int = 3,
        eps: float = 1e-6,
        coord_scale: float = 1.0,
        fusion_mode: str = "residual_gated",
        chunk_size: int | None = None,
        return_aux: bool = True,
    ):
        super().__init__()
        self.out_dim = point_dim if out_dim is None else out_dim
        self.k = k
        self.eps = eps
        self.coord_scale = coord_scale
        self.fusion_mode = fusion_mode
        self.chunk_size = chunk_size
        self.return_aux = return_aux

        self.point_proj = (
            nn.Identity()
            if point_dim == self.out_dim
            else nn.Linear(point_dim, self.out_dim)
        )
        self.image_proj = (
            nn.Identity()
            if image_dim == self.out_dim
            else create_mlp(image_dim, [hidden_dim, self.out_dim])
        )

        if self.fusion_mode == "concat":
            self.fuse_mlp = create_mlp(self.out_dim * 2, [hidden_dim, self.out_dim])
        else:
            self.gate_mlp = create_mlp(self.out_dim * 2, [hidden_dim, self.out_dim])
            self.delta_mlp = create_mlp(self.out_dim * 2, [hidden_dim, self.out_dim])

    def pairwise_sqdist(self, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        q_sq = (q * q).sum(dim=-1, keepdim=True)
        s_sq = (s * s).sum(dim=-1).unsqueeze(1)
        return (q_sq + s_sq - 2.0 * torch.matmul(q, s.transpose(1, 2))).clamp_min_(0.0)

    def knn_search(
        self,
        query_xyz: torch.Tensor,
        support_xyz: torch.Tensor,
        support_mask: torch.Tensor,
    ):
        if self.coord_scale != 1.0:
            query_xyz = query_xyz / self.coord_scale
            support_xyz = support_xyz / self.coord_scale

        k = min(self.k, support_xyz.size(1))
        sqdist = self.pairwise_sqdist(query_xyz, support_xyz)
        sqdist = sqdist.masked_fill(~support_mask[:, None, :], float("inf"))
        neighbor_sqdist, neighbor_index = torch.topk(
            sqdist, k=k, dim=-1, largest=False, sorted=False
        )
        return neighbor_sqdist.sqrt_(), neighbor_index

    def forward(
        self,
        point_token: torch.Tensor,
        patch_center: torch.Tensor,
        image_patch_token: torch.Tensor,
        image_patch_coord: torch.Tensor,
        image_patch_valid_mask: torch.Tensor | None = None,
    ):
        # Flatten batch + seq tokens to (B, N, C)
        point_token, leading_shape = point_token.view(-1, *point_token.shape[-2:]), point_token.shape[:-2]
        patch_center = patch_center.view(-1, *patch_center.shape[-2:])
        image_patch_token = image_patch_token.view(-1, *image_patch_token.shape[-2:])
        image_patch_coord = image_patch_coord.view(-1, *image_patch_coord.shape[-2:])

        if image_patch_valid_mask is None:
            image_patch_valid_mask = torch.ones(
                image_patch_token.shape[:2], dtype=torch.bool, device=image_patch_token.device
            )
        else:
            image_patch_valid_mask = image_patch_valid_mask.view(-1, *image_patch_valid_mask.shape[-1:])
            if image_patch_valid_mask.ndim == 2:
                pass  # (B, N)
            elif image_patch_valid_mask.ndim == 3 and image_patch_valid_mask.shape[-1] == 1:
                image_patch_valid_mask = image_patch_valid_mask.squeeze(-1)

        point_feat = self.point_proj(point_token)
        image_feat = self.image_proj(image_patch_token)

        neighbor_distance, neighbor_index = self.knn_search(
            patch_center, image_patch_coord, image_patch_valid_mask
        )

        # gather: (B, N_p, k, C)
        batch_idx = torch.arange(neighbor_index.size(0), device=neighbor_index.device)[:, None, None]
        neighbor_semantic = image_feat[batch_idx, neighbor_index]
        neighbor_valid = image_patch_valid_mask[batch_idx, neighbor_index]

        weight = (1.0 / neighbor_distance.clamp_min(self.eps)) * neighbor_valid.to(neighbor_distance.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        aligned_semantic = (neighbor_semantic * weight.unsqueeze(-1)).sum(dim=2)
        has_neighbor = neighbor_valid.any(dim=-1)

        point_valid = torch.ones(point_token.shape[:2], dtype=torch.bool, device=point_token.device)
        fused = self._fuse(point_feat, aligned_semantic) * point_valid.unsqueeze(-1).to(point_feat.dtype)

        # Restore leading dimensions
        fused = fused.view(*leading_shape, *fused.shape[-2:])

        if not self.return_aux:
            return fused, {}

        aux = {
            "aligned_semantic": aligned_semantic.view(*leading_shape, *aligned_semantic.shape[-2:]),
            "neighbor_index": neighbor_index.view(*leading_shape, *neighbor_index.shape[-2:]),
            "neighbor_weight": weight.view(*leading_shape, *weight.shape[-2:]),
            "has_neighbor": has_neighbor.view(*leading_shape, *has_neighbor.shape[-1:]),
        }
        return fused, aux

    def _fuse(self, point_token: torch.Tensor, aligned_semantic: torch.Tensor) -> torch.Tensor:
        x = torch.cat([point_token, aligned_semantic], dim=-1)
        if self.fusion_mode == "concat":
            return self.fuse_mlp(x)
        gate = torch.sigmoid(self.gate_mlp(x))
        delta = self.delta_mlp(x)
        return point_token + gate * delta


class SemGeoEncoder(nn.Module):
    """Semantic-geometric fusion encoder.

    Wraps pc_encoder + rgb_backbone + image_processor + PCImageAligner + state_mlp.
    Receives raw observations and produces fused feature sequences.

    Input obs_dict keys:
        point_cloud      : (B*T, N, pc_dim)
        joint_state      : (B*T, state_dim)
        rgb              : (B*T, H, W, 3)  uint8
        depth            : (B*T, H, W) or (B*T, 1, H, W)
        camera_intrinsic : (B*T, 3, 3)
        camera_extrinsic : (B*T, 3, 4)  [optional]
    """

    STATE_OUT_DIM = 64

    def __init__(
        self,
        pc_encoder: nn.Module,
        rgb_backbone: nn.Module,
        image_processor,
        spatial_aligner: PCImageAligner,
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

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pc = obs_dict["point_cloud"]
        state = obs_dict["joint_state"]
        rgb = obs_dict["rgb"]
        depth = obs_dict["depth"]
        intrinsics = obs_dict["camera_intrinsic"]
        camera_to_world = obs_dict.get("camera_extrinsic", None)

        # Point cloud encoding
        patch_token, patch_centers = self.pc_encoder(pc)
        global_scene_token = self.pc_encoder.get_global_token(patch_token)

        # RGB encoding + depth backprojection
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

        # Spatial alignment fusion
        fused_tokens, _ = self.spatial_aligner(
            point_token=patch_token,
            patch_center=patch_centers,
            image_patch_token=image_patch_tokens,
            image_patch_coord=image_patch_coords,
            image_patch_valid_mask=image_patch_valid,
        )

        # Global token projection + state concatenation
        global_proj = self.spatial_aligner.point_proj(global_scene_token)
        all_tokens = torch.cat([global_proj, fused_tokens], dim=1)

        state_feat = self.state_mlp(state)
        state_feat = state_feat.unsqueeze(1).expand(-1, all_tokens.size(1), -1)
        return torch.cat([all_tokens, state_feat], dim=-1)

    def get_optim_groups(self, weight_decay: float):
        from dexmani_policy.agents.common.optim_util import get_optim_group_with_no_decay
        return get_optim_group_with_no_decay(self, weight_decay)


def example():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    batch_size = 256
    num_point_patches = 96
    image_h, image_w, patch_size = 224, 224, 14
    num_image_patches = (image_h // patch_size) * (image_w // patch_size)

    point_token = torch.randn(batch_size, num_point_patches, 128, device=device)
    patch_center = torch.randn(batch_size, num_point_patches, 3, device=device)
    image_patch_token = torch.randn(batch_size, num_image_patches, 128, device=device)
    image_patch_coord = torch.randn(batch_size, num_image_patches, 3, device=device)
    image_patch_valid_mask = torch.rand(batch_size, num_image_patches, device=device) > 0.1

    aligner = PCImageAligner(
        point_dim=128, image_dim=128, hidden_dim=256, out_dim=128, k=3
    ).to(device)
    fused_point_token, aux = aligner(
        point_token,
        patch_center,
        image_patch_token,
        image_patch_coord,
        image_patch_valid_mask=image_patch_valid_mask,
    )

    if device == "cuda:0":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fused_point_token, aux = aligner(
            point_token,
            patch_center,
            image_patch_token,
            image_patch_coord,
            image_patch_valid_mask=image_patch_valid_mask,
        )
    if device == "cuda:0":
        torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) * 1000 / 20

    print("=== PCImageAligner Example ===")
    print("device:", device)
    print("point_token:", tuple(point_token.shape))
    print("image_patch_token:", tuple(image_patch_token.shape))
    print("fused_point_token:", tuple(fused_point_token.shape))
    print("aligned_semantic:", tuple(aux["aligned_semantic"].shape))
    print(f"avg forward time: {avg_ms:.3f} ms")


if __name__ == "__main__":
    example()
