# Geometry-Aware Point Transformer（GAPT）

这是基于 `main@5975701948fbfb9c4575d05d897f18167c286e38` 实现的独立点云编码器。实现文件仍为 `dexmani_policy/agents/obs_encoder/pointcloud/geometry_patch.py`，类名改为 `GeometryAwarePointTransformer`。不保留旧类别名，不加载预训练权重，不提供多种 Stem 或双输出头。

## 固定的默认结构

```text
XYZ / XYZRGB（米制，共同坐标系）
 -> 一次 FPS256 + 原始点 kNN32
 -> 相对 XYZ、距离、8 频段 Fourier 特征
 -> 64/128 维共享成员 MLP + max/broadcast/refine
 -> 同一支持集上的近/宽软距离池化 + max + 二阶几何统计
 -> 每个 Patch 一次 384 维投影
 -> 一层近邻 Patch 几何 Attention（12 邻居、128 维关系空间）
 -> 六层 384 维、8-head 全局 3D RoPE Transformer
 -> LayerNorm
 -> 一套 [B,256,384] Tokens + XYZ + 有效性 Mask
```

默认 XYZRGB 参数量：**11,814,084**，小于 30M。最后没有 global pooling、CLS 压缩、local/context 双投影或额外的局部输出相加。局部与全局模块内部保留标准残差、Pre-LayerNorm 和 LayerScale。

## 几何与效率细节

成员特征只在 128 维空间执行 Uni3D 式 max/broadcast/refine；384 维宽投影放在成员聚合后。原始相对 XYZ 没有被 Fourier 特征替代，也不按每个 Patch 的半径独立缩放，避免抹去物理大小。Fourier 使用 8 个对数间隔波长（0.02–0.32 m）。

Gaussian 池化尺度为 0.04 m / 0.12 m。两种加权均值和 max **共用同一次 kNN 的相同成员**；宽权重不能扩大支持范围，不等于完整的 12 cm 邻域。很小的邻域里两种均值可能接近。协方差只提供六个二阶统计量，不做特征值分解，不把它称为法线、曲率或真实接触面。额外保留局部覆盖半径标量；点数增加、遮挡和采样改变都会影响这些统计。

近邻 Attention 的 QKV 在 gather 前投影。中心图只构建一次，self 按索引而不是零距离排除；重复中心不会导致除零，但重复观测不会提供新的几何信息。相对位置进入局部 logit bias 和 value。全局 Attention 使用 PyTorch SDPA，Q/K Norm 在 RoPE 之前执行；三轴相位在 FP32 计算并复用。RoPE 尺度 0.05 m、base=100。归一化、频率/尺度与协方差不是额外训练目标。

RoPE 使用同一观测内的共同原点以改善数值精度；原始米制中心原样返回。模型不保证旋转等变性。固定锚点下的排列等变性，不等于 first-point FPS 对任意原始点排列不变。无输入点跨帧配准或跟踪。

## 公共接口

```python
from dexmani_policy.agents.obs_encoder.pointcloud.geometry_patch import GeometryAwarePointTransformer

encoder = GeometryAwarePointTransformer(input_channels=6)
result = encoder(pointcloud)  # [B,N,6]
# result['tokens']: [B,256,384]
# result['xyz']:    [B,256,3]，米制
# result['mask']:   [B,256]，True 表示有效
```

输入必须为有限的实际观测点，每个样本 `N >= max(num_patches, group_size)`。不支持把 NaN、Inf 或补零 Padding 当作有效观测；当前固定观测契约下返回的 Mask 全为 True，它不是缺测检测器，也不表示中心唯一。RGB 建议使用一致的 [0,1] 标度。坐标不要逐帧、逐轴独立缩放为不同物理单位。

`return_geometry=True` 另外返回 `center_indices`、`neighbor_indices`、`patch_radius`，用于几何证据读取。这些只是索引与几何元数据，不是第二套特征。`patch_radius` 是原始支持集半径，不是经过全局 Attention 后完整的感受野。

编码器不接机器人状态、动作、图像 Tokens、语言或时间索引。需要这些模态时在编码器外融合。无独立图像 Backbone，因此原 `semantic_channels/image_tokens/point_uv/...` 参数不再属于此 API；独立 `PointImageFusion` 文件未删除或修改。

## 当前仓库衔接

`InteractionObsEncoder` 调用 GAPT 后，仅使用**一个共享的 384→192 投影**适配现有策略宽度。Scene 和 HandSceneRelation 都读取同一套投影后的 Tokens。近场和场景读取仍可使用不同 query、几何边与距离约束，但不再持有不同的 local/context 特征源。

`HandSceneRelationEncoder` 改为消费 `tokens/xyz/mask/neighbor_indices`，最近可见成员距离与无支持时 near_update=0 的保护保留。近场是受局部观测约束的读取，**其 value 已包含全场景上下文**，不能再描述为纯局部 Patch 特征。

Agent、ObsEncoder 和配置的默认 Patch 数同步为 256、深度为 6。策略输出仍是每帧 16 scene + 6 hand + 1 state，动作/触觉/normalization/训练配方未更改。`docs/interaction_representation.md` 描述旧版双特征 memory；本轮编码器及 memory 接口以本文和当前代码为准，其余交互算法没有重设计。

这是架构与状态字典的不兼容升级。旧 GeometryPatchEncoder/Compact/双投影版本 checkpoint 不能完整恢复到新模型，不提供 silent fallback 或用 strict=False 假装恢复。请使用新的运行目录；历史实验仍需匹配其保存的源码与配置。

## 验证

在 CPU PyTorch 2.10.0、按已读取源码构建的局部验证环境完成：

- 21 项测试通过，1 项原生 PyTorch3D 测试因依赖缺失跳过。
- 包括完整默认模型的前向、反向和一次优化器更新；XYZ/XYZRGB；严格 reload；固定锚点排列等变；平移与尺寸响应；重复点；CPU BF16 autocast；非法输入；共享 memory 和缺测几何读取契约。
- CPU 模块测试明确替换 FPS/kNN 为参考实现；reader 契约测试使用记录式 attention，只验证生产 reader 的接口与几何支持逻辑，不代表完整策略质量。
- Python 语法与 diff 检查通过。完整仓库无法直接克隆至验证容器，未运行完整回归、config-only/smoke、原生 PyTorch3D/CUDA、torch.compile、DDP、真实数据训练或机器人 rollout。

完整项目环境可运行以下检查，它们不是消融实验：

```bash
python -m pytest -q tests/test_geometry_aware_point_transformer.py
python dexmani_policy/smoke_test.py --config-only interaction_flow
# 有任务数据与完整依赖时运行 bounded smoke；不自动启动长训练或真机。
python dexmani_policy/smoke_test.py interaction_flow
```

架构合理性、参数量与模块测试不能保证任务成功率；本次不声称精度提升或 GPU 加速。

## 参考边界

- Sparse2Act：https://arxiv.org/abs/2606.12759 —— 稀疏点云 Patch 与空间关系建模的结构线索；不复现其动作对齐预训练。
- Uni3D：https://arxiv.org/abs/2310.06773 —— 局部 PointNet 式 Patch 特征及 Transformer；本实现不等于预训练 Uni3D。
- LitePT：https://arxiv.org/abs/2512.13689 —— 局部阶段与高层 Attention 的分工，以及三维 RoPE；这里没有复制其 SparseConv/多级序列化网络。
- Fourier Features：https://arxiv.org/abs/2606.12334 —— 多频率坐标输入的动机；具体波长、Gaussian 池化、协方差与单层局部 Attention 的组合是本实现的设计，不是该论文已验证的同一个模型。
