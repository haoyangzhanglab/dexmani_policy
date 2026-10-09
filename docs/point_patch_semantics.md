# PointPatch 的图像语义融合

基于 `3bae525331e0f709d299c9e1b350054212b867eb`，研究变量是点云 patch 如何获取对应图像语义。
实现采用成员点投影、可见性约束、两级归一化池化与轻量残差门控；不增加 Top-K、跨模态 attention 或可学习采样偏移。

## 实现与接口

- `agents/obs_encoder/pointcloud/semantic_fusion.py`：投影、resize/crop 像素变换、对应矩阵和融合模块，仅依赖 PyTorch。标量 gather 拒绝负成员索引，不把 -1 当成最后一个点。
- `agents/obs_encoder/pointcloud/point_patch.py`：`semantic_channels` 可选开启融合；在 `local_token` 之后、RoPE blocks 之前注入，分组和 blocks 各执行一次。
- `scripts/utils/check_point_patch_semantics.py`：不下载视觉权重的 CPU 定向检查。
- `scripts/utils/validate_point_patch_semantics.py`：只读抽样实际 Zarr，比较外参方向、生成投影/patch 权重图、验证真实 backbone 和 PointPatch 的数值与梯度。

图像特征使用 DINO/SigLIP 已返回的 `patch_tokens`，不包含 CLS/register，不使用 `global_token`。
先在每个点内归一化有效视角权重，再对 patch 的有效成员点等权平均。双线性采样的标量权重通过 `scatter_add` 合并成 `A [B,M,V*L]`，然后 `S=A@F`。此过程没有高维 `[B,M,G,C]` 图像特征 gather，也不为每个 patch 单独运行 backbone。

融合采用 `local + valid * gate * Linear(LayerNorm(S))`。门由几何、语义、可见比例和深度置信度决定，初始值为 0.1；可见比例不直接衰减语义特征。无对应时完整残差置零，包括 Linear 的 bias。几何对应不求梯度，图像特征、融合参数和几何分支保留梯度。池化按现有 scene compressor 的方式使用 FP32 累积。

`PointPatchEncoder` 默认调用及旧参数键不变；启用语义会新增 `semantic_fusion.*` 参数。`return_intermediate=True` 增加 `fused_local_token`、`semantic_token`（保留原图像特征通道数的池化结果）、`semantic_gate`、`semantic_coverage`、`semantic_confidence`、`semantic_valid_mask` 和 `pool_weight`，用于诊断。

## 使用已有 DINO / SigLIP

以下是观测编码器内部的接入示例。模块在 `__init__` 中构造一次，不能在每次 `forward` 或 diffusion/flow 采样步骤中重新构造。

```python
import torch

from dexmani_policy.agents.obs_encoder.rgb.dino import DINO
from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
from dexmani_policy.agents.obs_encoder.rgb.utils import get_patch_grid_size
from dexmani_policy.agents.obs_encoder.pointcloud.point_patch import PointPatchEncoder
from dexmani_policy.agents.obs_encoder.pointcloud.semantic_fusion import (
    project_points_to_images,
    resize_crop_transform,
)

# 在 obs encoder 的 __init__ 中注册：
self.vision = DINO(tune_mode="freeze", out_dim=None)
self.image_processor = ImageProcessor.from_preset("dino")
self.point_encoder = PointPatchEncoder(
    input_channels=6, token_channels=192, num_patches=128, group_size=32,
    semantic_channels=self.vision.out_dim,
)

# 在 obs encoder 的 forward 中执行：
# raw_rgb: [B,V,3,H,W] uint8 / [0,1] float；没有未记录的空间裁剪。
# pointcloud: [B,N,6]，保持当前几何分支的输入语义。
# point_xyz_m: [B,N,3]，与 pointcloud 同序、增强前/归一化前的米制 XYZ。
# depth_m: [B,V,H,W]，原始 RGB 对齐的米制光轴深度。
# intrinsics: [B,V,3,3]；world_to_camera: [B,V,4,4] 或 [B,V,3,4]。
# 各 Tensor 应已在同一 device；时序 B,T 先合并为 batch，V 单独保留。
processed = self.image_processor.process_images(raw_rgb)
image = processed["image"]
spatial = processed["spatial"]

# 此处 out_dim=None、tune_mode=freeze，backbone 无梯度且没有可训练 proj。
# 若以后使用 LoRA 或可训练 vision.proj，必须移除这个 no_grad。
with torch.no_grad():
    image_tokens = self.vision(image)["patch_tokens"]

pixel_transform = resize_crop_transform(
    spatial["orig_hw"], spatial["resized_hw"], spatial["crop_top_left"],
    device=image.device,
)
correspondence = project_points_to_images(
    point_xyz=point_xyz_m,
    depth_m=depth_m,
    intrinsics=intrinsics,
    world_to_camera=world_to_camera,
    pixel_transform=pixel_transform,
    depth_atol=0.01, depth_rtol=0.01,
)
patch = self.point_encoder(
    pointcloud,
    image_tokens=image_tokens,
    **correspondence,
    image_hw=tuple(image.shape[-2:]),
    patch_grid_size=get_patch_grid_size(image.shape[-2:], self.vision.patch_size),
)
# patch["patch_token"]: [B,128,192]；patch_center 与输入 pointcloud 同坐标系。
```

使用 SigLIP 时，将 `DINO` 换为 `rgb.siglip.SigLIP`，preset 换为 `"siglip"`。其余代码仍从实际编码器读取 `out_dim` 和 `patch_size`；不要把 DINO 的网格大小用于 SigLIP。

若数据已经保存 UV，可以跳过 `project_points_to_images`，直接给 `point_uv [B,V,N,2]` 和 `view_weight [B,V,N]`。UV 必须先应用与实际 RGB 一致的变换；仅有 source view 时，其他视角的权重置零。像素中心为整数，插值采用 `align_corners=False` 和 border 规则，不支持带 padding/NaFlex 的非规则 token 布局。

## 数据与控制变量

1. `depth_m` 必须显式转换成米。仓库文档中的真机数据乘根属性 `depth_scale_m_per_unit`（约 0.00025），仿真示例乘 0.001；不能猜测单位。深度是 RGB 相机坐标下的 z，非沿射线的欧氏距离。
2. 投影使用去畸变、与深度配准的原始 RGB 内参和 **world-to-camera** 外参。camera-to-world 需要先求逆；不能依据字段名猜测仿真外参方向。
3. 当前 dataset 随机 RGB crop 不输出变换。使用该路径前须关闭未记录的空间增强，或记录完整变换并与 `pixel_transform` 相乘。`ImageProcessor` 的 metadata 只描述它自己执行的变换。
4. 点采样、重排、dropout 必须同步作用于 XYZ/UV/可见性索引。仅坐标噪声保留原始对应；被替换的无效点应将对应权重置零。`point_xyz_m`、K/T、深度和像素变换不进行统计归一化。
5. RGB 模态 dropout 时，同步将对应视角的 `view_weight` 置零；黑图经过 backbone 仍会产生非零特征。无效视角使用有限的占位特征，不能输入 NaN。
6. `depth_atol=0.01` 米、`depth_rtol=0.01` 是起点，应依据传感器残差调整。深度缺失视为未知且不提供语义，不把它当作可见表面。
7. 固定分组、RoPE、下游 action head 和 scene compressor 配置，先比较纯几何、中心投影、成员点池化。更高分辨率、更换 backbone、可学习聚合应分别消融。若点输入经过归一化，scene compressor 的米制中心可由 `patch_center_idx` 从原始 XYZ 中取回。

当前 PointPatch 本身尚未被策略 config/agent 调用。本次交付到 PointPatch 的完整语义输入/输出；现有训练命令不会自动切换到新结构。选择具体 policy 接入时还需明确 observation 数据契约、状态特征拼接及 action head 所需的通道数。真机 canonical 缺少的标定信息也不会由本模块自动恢复。

## 定向验证

在仓库根目录运行：

```bash
python -m dexmani_policy.agents.obs_encoder.pointcloud.semantic_fusion
python -m scripts.utils.check_point_patch_semantics
```

PyTorch 2.4.1+cpu 下已通过 10 项检查：grid_sample 数值/特征梯度等价、两级归一化、无有效对应的精确几何回退、点/视角重排不变性、外参和 resize/crop 投影、遮挡/深度空洞、CPU BF16 与反向传播、PointPatch 接线及 blocks 不重复执行、负成员索引拒绝，以及容差筛选前的投影残差诊断。

PointPatch 接线检查固定了 FPS/KNN 的输出，实际执行 PatchEncoder、融合和 RoPE blocks；未验证 PyTorch3D/CUDA 算子、真实 DINO/SigLIP 权重、真实数据、策略训练及成功率。也未声称 A@F 一定比 grid_sample 更快：目标 GPU 应测 batch=1 的图像编码、对应构建、融合和策略端到端 P50/P95 延迟。CPU 数值验证不代替 GPU 性能结论。

## 先验证 pick_apple_messy.zarr

验证脚本按 episode 抽样起点、中间、末尾帧，逐帧读取，不加载完整 RGB 数组、不修改 Zarr。输出放入新的 `outputs/point_patch_semantics_<timestamp>/`，或者由 `--output-dir` 指定一个尚不存在的目录。

先在本地核对 `.zattrs`、实际字段与数据生成代码中的单位、坐标系和 `extrinsic_cv` 方向。确认该仿真数据深度单位为 mm 后，运行：

```bash
python -m scripts.utils.check_point_patch_semantics
python -m scripts.utils.validate_point_patch_semantics \
  --dataset robot_data/pick_apple_messy.zarr \
  --depth-scale-m-per-unit 0.001 --num-frames 16 \
  --extrinsic-convention compare --device cpu
```

`report.json` 同时记录把存储外参解释为 world-to-camera 和 camera-to-world 的结果：正 z 比例、视野内比例、有效深度比例、可见比例、深度残差与点 RGB 回投影误差的 P50/P95。残差统计包含容差检查失败的点，避免用筛选后的少量点掩盖标定问题。`projection_*.png` 中绿色为可见点，红色为视野内但深度检查失败的点。几何-only 模式状态是 `GEOMETRY_DIAGNOSTICS`，不自动判定方向或宣称标定正确。

结合生成代码和投影图确认方向后，再执行真实预训练 DINO 验证。例如已确认是 world-to-camera 时：

```bash
python -m scripts.utils.validate_point_patch_semantics \
  --dataset robot_data/pick_apple_messy.zarr \
  --depth-scale-m-per-unit 0.001 --num-frames 16 \
  --extrinsic-convention world-to-camera --backbone dino --device cuda
```

如果确认存储的是 camera-to-world，显式改成 `--extrinsic-convention camera-to-world`。不能仅根据哪个候选的可见率更高，就把外参方向当成已验证。`--model-name` 支持实际模型 ID 或本地 Hugging Face 模型目录；不指定时分别使用 DINOv2-base / SigLIP-base。切换 `--backbone siglip` 可对同一批帧复核 SigLIP；没有 CUDA 时可以显式 `--device cpu`，结果不作为 GPU 延迟结论。

特征阶段使用真实 PyTorch3D 分组及预训练视觉 backbone，检查有限输出、A 的行归一化、至少存在有效语义、首帧全无效时的几何回退，以及融合投影的有限非零梯度。backbone 保持冻结，不做 optimizer step。`patch_weights_*.png` 展示几个 patch 对实际图像 token 的池化权重，青色叉号为 patch 中心的投影；这是对应权重图，不是语义分割真值。

成功时状态为 `PASS_FEATURE_NUMERICS`，仅表示接口、对应计算和梯度检查通过。PointPatch 是随机初始化的研究组件，不由此推断策略成功率或语义质量提升。图像编码、投影和 PointPatch 前向分别计时并输出 P50/P95；排除磁盘读取、图像预处理、CPU-GPU 传输、绘图和权重加载，不等于完整控制链路延迟。

缺少数据、标定、预训练权重或 PyTorch3D 时，脚本保存 `ERROR` 报告并退出，不用随机视觉特征或模拟分组回退。当前开发环境只验证了临时合成 Zarr 的读取、双方向诊断、JSON 报告和投影图，以及带明确测试替身的特征阶段接线；实际 `pick_apple_messy.zarr`、预训练视觉模型与 CUDA 结果须由本地执行确认。
