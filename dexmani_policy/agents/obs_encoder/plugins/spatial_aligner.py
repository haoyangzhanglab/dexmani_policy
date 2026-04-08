import time
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_ln: bool = False):
        super().__init__()
        layers = []
        if use_ln:
            layers.append(nn.LayerNorm(in_dim))
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def flatten_batch_tokens(x: torch.Tensor, trailing_ndim: int):
    leading_shape = tuple(x.shape[:-trailing_ndim])
    return x.reshape(-1, *x.shape[-trailing_ndim:]), leading_shape


def restore_batch_tokens(x: torch.Tensor, leading_shape, trailing_ndim: int):
    return x.reshape(*leading_shape, *x.shape[-trailing_ndim:])


class SpatialAligner(nn.Module):
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

        self.point_proj = nn.Identity() if point_dim == self.out_dim else nn.Linear(point_dim, self.out_dim)
        self.image_proj = nn.Identity() if image_dim == self.out_dim else MLP(image_dim, hidden_dim, self.out_dim, use_ln=True)

        if self.fusion_mode == "concat":
            self.fuse_mlp = MLP(self.out_dim * 2, hidden_dim, self.out_dim, use_ln=True)
        else:
            self.gate_mlp = MLP(self.out_dim * 2, hidden_dim, self.out_dim, use_ln=True)
            self.delta_mlp = MLP(self.out_dim * 2, hidden_dim, self.out_dim, use_ln=True)

    def check_input(
        self,
        point_token: torch.Tensor,
        patch_center: torch.Tensor,
        image_patch_token: torch.Tensor,
        image_patch_coord: torch.Tensor,
    ):
        assert point_token.shape[:-1] == patch_center.shape[:-1] and patch_center.shape[-1] == 3
        assert image_patch_token.shape[:-1] == image_patch_coord.shape[:-1] and image_patch_coord.shape[-1] == 3
        assert point_token.shape[:-2] == image_patch_token.shape[:-2]

    def masked_topk(self, sqdist: torch.Tensor, support_mask: torch.Tensor, k: int):
        sqdist = sqdist.masked_fill(~support_mask[:, None, :], float("inf"))
        return torch.topk(sqdist, k=k, dim=-1, largest=False, sorted=False)

    def gather_support_feat(self, support_feat: torch.Tensor, index: torch.Tensor):
        batch_index = torch.arange(index.size(0), device=index.device)[:, None, None]
        return support_feat[batch_index, index]

    def gather_support_mask(self, support_mask: torch.Tensor, index: torch.Tensor):
        batch_index = torch.arange(index.size(0), device=index.device)[:, None, None]
        return support_mask[batch_index, index]

    def compute_idw_weight(self, neighbor_distance: torch.Tensor, neighbor_valid_mask: torch.Tensor):
        weight = 1.0 / neighbor_distance.clamp_min(self.eps)
        weight = weight * neighbor_valid_mask.to(weight.dtype)
        return weight / weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def pairwise_sqdist(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor):
        query_sq = (query_xyz * query_xyz).sum(dim=-1, keepdim=True)
        support_sq = (support_xyz * support_xyz).sum(dim=-1).unsqueeze(1)
        sqdist = query_sq + support_sq - 2.0 * torch.matmul(query_xyz, support_xyz.transpose(1, 2))
        return sqdist.clamp_min_(0.0)

    def knn_search(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, support_mask: torch.Tensor):
        if self.coord_scale != 1.0:
            query_xyz = query_xyz / self.coord_scale
            support_xyz = support_xyz / self.coord_scale

        k = min(self.k, support_xyz.size(1))
        if self.chunk_size is None or query_xyz.size(1) <= self.chunk_size:
            neighbor_sqdist, neighbor_index = self.masked_topk(
                self.pairwise_sqdist(query_xyz, support_xyz), support_mask, k
            )
            return neighbor_sqdist.sqrt_(), neighbor_index

        dist_list, idx_list = [], []
        for start in range(0, query_xyz.size(1), self.chunk_size):
            query_chunk = query_xyz[:, start : start + self.chunk_size]
            neighbor_sqdist, neighbor_index = self.masked_topk(
                self.pairwise_sqdist(query_chunk, support_xyz), support_mask, k
            )
            dist_list.append(neighbor_sqdist.sqrt_())
            idx_list.append(neighbor_index)
        return torch.cat(dist_list, dim=1), torch.cat(idx_list, dim=1)

    def fuse_feature(self, point_token: torch.Tensor, aligned_semantic: torch.Tensor):
        fusion_input = torch.cat([point_token, aligned_semantic], dim=-1)
        if self.fusion_mode == "concat":
            return self.fuse_mlp(fusion_input)
        gate = torch.sigmoid(self.gate_mlp(fusion_input))
        delta = self.delta_mlp(fusion_input)
        return point_token + gate * delta

    def forward(
        self,
        point_token: torch.Tensor,
        patch_center: torch.Tensor,
        image_patch_token: torch.Tensor,
        image_patch_coord: torch.Tensor,
        point_valid_mask: torch.Tensor | None = None,
        image_patch_valid_mask: torch.Tensor | None = None,
    ):
        self.check_input(point_token, patch_center, image_patch_token, image_patch_coord)

        point_token, leading_shape = flatten_batch_tokens(point_token, 2)
        patch_center, _ = flatten_batch_tokens(patch_center, 2)
        image_patch_token, _ = flatten_batch_tokens(image_patch_token, 2)
        image_patch_coord, _ = flatten_batch_tokens(image_patch_coord, 2)

        if point_valid_mask is None:
            point_valid_mask = torch.ones(point_token.shape[:2], dtype=torch.bool, device=point_token.device)
        else:
            point_valid_mask, _ = flatten_batch_tokens(point_valid_mask, 1)

        if image_patch_valid_mask is None:
            image_patch_valid_mask = torch.ones(image_patch_token.shape[:2], dtype=torch.bool, device=image_patch_token.device)
        else:
            image_patch_valid_mask, _ = flatten_batch_tokens(image_patch_valid_mask, 1)

        point_feat = self.point_proj(point_token)
        image_feat = self.image_proj(image_patch_token)

        neighbor_distance, neighbor_index = self.knn_search(patch_center, image_patch_coord, image_patch_valid_mask)
        neighbor_semantic = self.gather_support_feat(image_feat, neighbor_index)
        neighbor_valid_mask = self.gather_support_mask(image_patch_valid_mask, neighbor_index)
        neighbor_weight = self.compute_idw_weight(neighbor_distance, neighbor_valid_mask)

        aligned_semantic = (neighbor_semantic * neighbor_weight.unsqueeze(-1)).sum(dim=2)
        has_neighbor = neighbor_valid_mask.any(dim=-1)

        point_valid = point_valid_mask.unsqueeze(-1).to(point_feat.dtype)
        aligned_semantic = aligned_semantic * point_valid
        fused_point_token = self.fuse_feature(point_feat, aligned_semantic) * point_valid

        fused_point_token = restore_batch_tokens(fused_point_token, leading_shape, 2)
        if not self.return_aux:
            return fused_point_token, {}

        aux = {
            "aligned_semantic": restore_batch_tokens(aligned_semantic, leading_shape, 2),
            "neighbor_index": restore_batch_tokens(neighbor_index, leading_shape, 2),
            "neighbor_weight": restore_batch_tokens(neighbor_weight, leading_shape, 2),
            "has_neighbor": restore_batch_tokens(has_neighbor, leading_shape, 1),
        }
        return fused_point_token, aux


def example():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    batch_size = 256
    num_point_patches = 96
    image_h, image_w, patch_size = 224, 224, 14
    num_image_patches = (image_h // patch_size) * (image_w // patch_size)

    # Faster path: let DINO output out_dim directly, e.g. DINO(..., out_dim=128)
    point_token = torch.randn(batch_size, num_point_patches, 128, device=device)
    patch_center = torch.randn(batch_size, num_point_patches, 3, device=device)
    image_patch_token = torch.randn(batch_size, num_image_patches, 128, device=device)
    image_patch_coord = torch.randn(batch_size, num_image_patches, 3, device=device)
    image_patch_valid_mask = (torch.rand(batch_size, num_image_patches, device=device) > 0.1)

    aligner = SpatialAligner(point_dim=128, image_dim=128, hidden_dim=256, out_dim=128, k=3).to(device)
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

    print("device:", device)
    print("point_token:", tuple(point_token.shape))
    print("image_patch_token:", tuple(image_patch_token.shape))
    print("fused_point_token:", tuple(fused_point_token.shape))
    print("aligned_semantic:", tuple(aux["aligned_semantic"].shape))
    print(f"avg forward time: {avg_ms:.3f} ms")


if __name__ == "__main__":
    example()