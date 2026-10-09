"""将图像 patch 特征按真实成员点对应汇聚到点云 patch。"""

import math

import torch
import torch.nn as nn

def resize_crop_transform(
    original_hw: tuple[int, int],
    resized_hw: tuple[int, int],
    crop_top_left: tuple[int, int] = (0, 0),
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """原图像素中心 -> resize 后 crop；对应 align_corners=False，无 padding。"""
    if min(*original_hw, *resized_hw) <= 0:
        raise ValueError("image sizes must be positive")
    scale_y = resized_hw[0] / original_hw[0]
    scale_x = resized_hw[1] / original_hw[1]
    top, left = crop_top_left
    return torch.tensor(
        [
            [scale_x, 0, (scale_x - 1) * 0.5 - left],
            [0, scale_y, (scale_y - 1) * 0.5 - top],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def project_points_to_images(
    point_xyz: torch.Tensor,
    depth_m: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    pixel_transform: torch.Tensor | None = None,
    depth_atol: float = 0.01,
    depth_rtol: float = 0.01,
    return_intermediate: bool = False,
) -> dict[str, torch.Tensor]:
    """米制固定坐标点 -> RGB 像素与可见性权重；几何计算不求梯度。

    point_xyz [B,N,3] 必须与分组点保持同一顺序，不能使用归一化/加噪 XYZ。
    depth_m [B,V,H,W] 为原始 RGB 对齐、去畸变后的光轴深度（米），不是射线距离。
    intrinsics [B,V,3,3]、world_to_camera [B,V,3,4] 或 [B,V,4,4]。
    pixel_transform [3,3] 或 [B,V,3,3] 将原图 UV 映射到实际编码器输入；
    遮挡检查在变换前的 depth_m 上进行。输出 point_uv [B,V,N,2]、
    view_weight [B,V,N]，后者在 [0,1] 内；裁剪后出界由池化函数屏蔽。
    return_intermediate 额外返回原图 UV、相机 z、采样深度、视野/深度有效掩码
    与未经过容差筛选的残差，用于核验标定；这些诊断字段不要传入 PointPatch。
    """
    if point_xyz.ndim != 3 or point_xyz.shape[-1] != 3 or depth_m.ndim != 4:
        raise ValueError("expected point_xyz [B,N,3] and depth_m [B,V,H,W]")
    if not depth_m.is_floating_point():
        raise TypeError("convert raw depth to floating-point meters before projection")
    batch_size, num_views, height, width = depth_m.shape
    if point_xyz.shape[0] != batch_size or min(batch_size, num_views, height, width) < 1:
        raise ValueError("point/depth batches must match and dimensions must be positive")
    if intrinsics.shape != (batch_size, num_views, 3, 3):
        raise ValueError("intrinsics must have shape [B,V,3,3]")
    if world_to_camera.shape not in (
        (batch_size, num_views, 3, 4), (batch_size, num_views, 4, 4)
    ):
        raise ValueError("world_to_camera must have shape [B,V,3,4] or [B,V,4,4]")
    if pixel_transform is not None and pixel_transform.shape not in (
        (3, 3), (batch_size, num_views, 3, 3)
    ):
        raise ValueError("pixel_transform must have shape [3,3] or [B,V,3,3]")
    tolerances_valid = all(math.isfinite(x) and x >= 0 for x in (depth_atol, depth_rtol))
    if not tolerances_valid or depth_atol + depth_rtol == 0:
        raise ValueError("depth tolerances must be finite, nonnegative and not both zero")

    with torch.autocast(device_type=point_xyz.device.type, enabled=False):
        extrinsic = world_to_camera.float()
        camera_xyz = (
            point_xyz.float()[:, None] @ extrinsic[..., :3, :3].transpose(-1, -2)
            + extrinsic[..., :3, 3].unsqueeze(-2)
        )
        projected = camera_xyz @ intrinsics.float().transpose(-1, -2)
        denominator = projected[..., 2:3]
        finite = torch.isfinite(projected).all(-1) & torch.isfinite(camera_xyz).all(-1)
        valid = finite & (camera_xyz[..., 2] > 1e-6) & (denominator[..., 0].abs() > 1e-6)
        uv = projected[..., :2] / torch.where(denominator.abs() > 1e-6, denominator, 1)
        uv = torch.nan_to_num(uv)
        u, v = uv.unbind(-1)
        # 像素中心为整数，图像的连续支持范围是 [-0.5, size-0.5)。
        valid &= (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
        in_image = valid
        col = (u.clamp(0, width - 1) + 0.5).floor().long()
        row = (v.clamp(0, height - 1) + 0.5).floor().long()
        observed_depth = depth_m.float().flatten(-2).gather(-1, row * width + col)
        depth_valid = in_image & torch.isfinite(observed_depth) & (observed_depth > 0)
        residual = (observed_depth - camera_xyz[..., 2]).abs()
        tolerance = depth_atol + depth_rtol * observed_depth
        valid = depth_valid & (residual <= tolerance)
        scaled_residual = residual / tolerance.clamp_min(1e-8)
        weight = torch.where(valid, torch.exp(-0.5 * scaled_residual.square()), 0)
        raw_uv = uv

        if pixel_transform is not None:
            homogeneous_uv = torch.cat((uv, torch.ones_like(uv[..., :1])), dim=-1)
            transformed = homogeneous_uv @ pixel_transform.float().transpose(-1, -2)
            divisor = transformed[..., 2:3]
            transform_valid = torch.isfinite(transformed).all(-1) & (divisor[..., 0].abs() > 1e-6)
            uv = transformed[..., :2] / torch.where(divisor.abs() > 1e-6, divisor, 1)
            weight = torch.where(transform_valid, weight, 0)
        outputs = {"point_uv": torch.nan_to_num(uv), "view_weight": weight}
        if return_intermediate:
            outputs.update(
                raw_uv=raw_uv,
                camera_depth=camera_xyz[..., 2],
                sampled_depth=observed_depth,
                in_image=in_image,
                depth_valid=depth_valid,
                depth_residual=residual,
            )
        return outputs


def _gather_members(value: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """仅收集标量对应信息；gather 对负索引报错，避免 -1 静默引用最后一点。"""
    batch_size, num_points = value.shape[:2]
    flat = value.reshape(batch_size, num_points, -1)
    index = neighbor_idx.reshape(batch_size, -1, 1).expand(-1, -1, flat.shape[-1])
    return flat.gather(1, index).reshape(*neighbor_idx.shape, *value.shape[2:])


@torch.no_grad()
def build_patch_image_weights(
    neighbor_idx: torch.Tensor,
    point_uv: torch.Tensor,
    view_weight: torch.Tensor,
    image_hw: tuple[int, int],
    patch_grid_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """返回 A [B,M,V*Hf*Wf]；先归一化视角，再等权汇聚可见成员点。

    neighbor_idx [B,M,G] 是无 padding 的真实点索引；point_uv [B,V,N,2]
    使用编码器输入图像的像素坐标，view_weight [B,V,N] 在 [0,1] 内。
    双线性插值采用 align_corners=False / border；无对应行严格为零。
    """
    if point_uv.ndim != 4 or point_uv.shape[-1] != 2 or view_weight.shape != point_uv.shape[:-1]:
        raise ValueError("expected point_uv [B,V,N,2] and view_weight [B,V,N]")
    batch_size, num_views, num_points, _ = point_uv.shape
    if neighbor_idx.ndim != 3 or neighbor_idx.shape[0] != batch_size or neighbor_idx.dtype != torch.long:
        raise ValueError("neighbor_idx must be int64 [B,M,G]")
    if min(*image_hw, *patch_grid_size, *neighbor_idx.shape, num_views, num_points) < 1:
        raise ValueError("image, grid, batch and grouping dimensions must be positive")

    height, width = image_hw
    grid_h, grid_w = patch_grid_size
    num_patches, group_size = neighbor_idx.shape[1:]
    with torch.autocast(device_type=point_uv.device.type, enabled=False):
        finite = torch.isfinite(point_uv).all(-1) & torch.isfinite(view_weight)
        uv = torch.nan_to_num(point_uv.float())
        u, v = uv.unbind(-1)
        valid = finite & (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
        confidence = torch.where(valid, view_weight.float().clamp(0, 1), 0)
        view_sum = confidence.sum(dim=1, keepdim=True)
        normalized = confidence / torch.where(view_sum > 0, view_sum, 1)
        point_valid = view_sum[:, 0] > 0

        x = ((u + 0.5) * (grid_w / width) - 0.5).clamp(0, grid_w - 1)
        y = ((v + 0.5) * (grid_h / height) - 0.5).clamp(0, grid_h - 1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp_max(grid_w - 1), (y0 + 1).clamp_max(grid_h - 1)
        dx, dy = x - x0, y - y0
        token_idx = torch.stack(
            (y0 * grid_w + x0, y0 * grid_w + x1, y1 * grid_w + x0, y1 * grid_w + x1), dim=-1
        )
        token_idx += torch.arange(num_views, device=uv.device)[None, :, None, None] * (grid_h * grid_w)
        bilinear = torch.stack(
            ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), dim=-1
        )
        point_weight = normalized.unsqueeze(-1) * bilinear

        member_valid = _gather_members(point_valid, neighbor_idx)
        count = member_valid.sum(-1)
        # 只 gather 小的标量权重和 token 索引，不复制高维图像特征。
        member_idx = _gather_members(token_idx.permute(0, 2, 1, 3), neighbor_idx)
        member_weight = _gather_members(point_weight.permute(0, 2, 1, 3), neighbor_idx)
        pool_weight = uv.new_zeros(batch_size, num_patches, num_views * grid_h * grid_w)
        pool_weight.scatter_add_(
            2,
            member_idx.reshape(batch_size, num_patches, -1),
            member_weight.reshape(batch_size, num_patches, -1) / count.clamp_min(1).unsqueeze(-1),
        )
        member_confidence = _gather_members(confidence.amax(dim=1), neighbor_idx)
        return {
            "pool_weight": pool_weight,
            "semantic_valid_mask": count > 0,
            "semantic_coverage": count.float() / group_size,
            "semantic_confidence": member_confidence.sum(-1) / count.clamp_min(1),
        }


class PointPatchSemanticFusion(nn.Module):
    """成员点语义池化 + 标量门控残差；输出与 local_token 同形状。"""

    def __init__(self, image_channels: int, token_channels: int = 192, gate_init: float = 0.1):
        super().__init__()
        if min(image_channels, token_channels) < 1 or not 0 < gate_init < 1:
            raise ValueError("channels must be positive; gate_init must be in (0,1)")
        self.image_channels = image_channels
        self.token_channels = token_channels
        self.image_norm = nn.LayerNorm(image_channels)
        self.proj = nn.Linear(image_channels, token_channels)
        self.geometry_norm = nn.LayerNorm(token_channels)
        self.semantic_norm = nn.LayerNorm(token_channels)
        self.gate = nn.Linear(token_channels * 2 + 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))

    def forward(
        self,
        local_token: torch.Tensor,
        neighbor_idx: torch.Tensor,
        image_tokens: torch.Tensor,
        point_uv: torch.Tensor,
        view_weight: torch.Tensor,
        image_hw: tuple[int, int],
        patch_grid_size: tuple[int, int],
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """image_tokens [B,V,Hf*Wf,C] 不含 CLS/register，所有特征须有限。

        图像编码每帧只运行一次；不在此处加载 backbone 或按 patch 编码图像。
        无效视角使用有限占位特征并将 view_weight 置零；几何对应不求梯度，
        特征池化、投影和门控保持梯度，支持冻结 backbone 或单独的微调实验。
        """
        if local_token.ndim != 3 or local_token.shape[-1] != self.token_channels:
            raise ValueError("local_token must have shape [B,M,token_channels]")
        if neighbor_idx.shape[:2] != local_token.shape[:2]:
            raise ValueError("local_token and neighbor_idx must share [B,M]")
        if image_tokens.ndim != 4 or image_tokens.shape[-1] != self.image_channels:
            raise ValueError("image_tokens must have shape [B,V,L,image_channels]")
        if image_tokens.shape[:2] != point_uv.shape[:2] or image_tokens.shape[0] != local_token.shape[0]:
            raise ValueError("image, projection and geometry batch/view dimensions must match")
        if image_tokens.shape[2] != math.prod(patch_grid_size):
            raise ValueError("image token count must equal Hf*Wf; remove prefix tokens first")

        geometry = build_patch_image_weights(neighbor_idx, point_uv, view_weight, image_hw, patch_grid_size)
        # 与 scene_compressor 一致，低精度特征以 FP32 累积。
        with torch.autocast(device_type=image_tokens.device.type, enabled=False):
            pool_dtype = torch.float64 if image_tokens.dtype == torch.float64 else torch.float32
            pooled = torch.bmm(
                geometry["pool_weight"].to(pool_dtype),
                image_tokens.flatten(1, 2).to(pool_dtype),
            )
        semantic = self.proj(self.image_norm(pooled.to(self.proj.weight.dtype)))
        quality = torch.stack((geometry["semantic_coverage"], geometry["semantic_confidence"]), dim=-1)
        gate_input = torch.cat((
            self.geometry_norm(local_token), self.semantic_norm(semantic), quality.to(local_token.dtype)
        ), dim=-1)
        gate = self.gate(gate_input).sigmoid()
        valid = geometry["semantic_valid_mask"].unsqueeze(-1)
        update = torch.where(valid, gate * semantic, 0).to(local_token.dtype)
        outputs = {"fused_token": local_token + update}
        if return_intermediate:
            outputs.update(geometry)
            outputs.update(semantic_token=pooled, semantic_gate=torch.where(valid, gate, 0))
        return outputs


def example() -> None:
    torch.manual_seed(0)
    xyz = torch.rand(2, 32, 3) - 0.5
    xyz[..., 2] = 1
    depth = torch.ones(2, 2, 64, 64)
    intrinsics = torch.tensor([[50., 0., 31.5], [0., 50., 31.5], [0., 0., 1.]]).expand(2, 2, 3, 3)
    extrinsic = torch.eye(4).expand(2, 2, 4, 4)
    correspondence = project_points_to_images(
        xyz, depth, intrinsics, extrinsic, resize_crop_transform((64, 64), (32, 32))
    )
    fusion = PointPatchSemanticFusion(image_channels=48, token_channels=24)
    output = fusion(
        torch.randn(2, 4, 24), torch.arange(32).reshape(1, 4, 8).expand(2, -1, -1),
        torch.randn(2, 2, 16, 48), **correspondence,
        image_hw=(32, 32), patch_grid_size=(4, 4), return_intermediate=True,
    )
    for name, value in output.items():
        print(f"{name}: {tuple(value.shape)}")


if __name__ == "__main__":
    example()
