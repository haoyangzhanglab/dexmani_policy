# GeometryPatchEncoder：轻量局部编码与直接 Patch 关系建模

## 研究目标与基线

基于 `dcff244d34d4f451c3c9971efe19e0fb030fc92d` 的真实实现修改，而不是替换整个观测或动作框架。目标是在不加载预训练权重的情况下，降低局部几何编码的重复计算，保留每个 Patch 的空间锚点和跨 Patch 关系。

本仓库现有 `GeometryPatchEncoder` 已经是 FPS + kNN + Uni3D-derived PatchEncoder + 4 层连续坐标 3D RoPE Transformer，默认输出 `[B,128,192]`，并非全局池化编码器。此次保持 Patch 数、邻域数、Transformer 深度、RoPE 参数、动作模型、学习率和数据增强不变。

参考范围：

- Sparse2Act：稀疏局部 Patch、带三维位置的 Transformer。这里只借鉴结构，不使用其动作对齐预训练。
- Uni3D：邻域内 PointNet 编码、max/broadcast/refine、Patch 间 Transformer。本仓库版本是 R3D-derived LayerNorm 实现，不等于原 Uni3D 的 BN 实现。
- LitePT：原始点/局部阶段使用局部算子，在压缩后的 Patch 集合上做 Attention。这里借鉴计算分工，不声称复现其 SparseConv + Serialized Attention 网络。

原始资源：
https://arxiv.org/abs/2606.12759
https://github.com/baaivision/Uni3D
https://github.com/prs-eth/LitePT

## 最小结构改动

```
输入 XYZ[RGB]（米制，同一基座/世界坐标）
  -> 一次 FPS / kNN（原有 PyTorch3D ops，不改）
  -> PatchEncoder：uni3d 原版 或 compact
  -> local_token                    # 只描述原始 K 个成员
  -> 可选 PointImageFusion（保持邻域索引和 UV 对齐）
  -> 可选一个 LocalPatchMixer
  -> 原有 4 层 Global Patch Attention + 3D RoPE
  -> patch_token + patch_center
```

### CompactPatchEncoder

保留 Uni3D 的局部上下文反馈，但把邻域内部特征维度限制在 64：

```
relative XYZ[RGB] -> C→32→64
                  -> 邻域 max + broadcast concat（128 维）
                  -> 128→64→64
                  -> 邻域 max 与 mean 拼接（128 维）
                  -> 128→D→D（此时每个 Patch 只计算一次）
```

max/mean 均仅沿 K 个原始成员聚合，没有沿 M 个 Patch 做场景池化。宽通道投影移到局部池化后；mean 在 autocast 下以 FP32 累积。这里不新增距离/Fourier 输入，也不同时调整坐标归一化，让首轮对照更容易解释。

### 可选 LocalPatchMixer

只在现有 Patch Centers 上建立一次 kNN 图。每个 Token 先从 D 投影到至多 64 维，再 gather 邻居，使用特征差与相对 XYZ/距离编码形成低维消息，最后 max/mean + 输出投影 + 小增益残差。self edge 按索引排除，允许不同锚点具有相同坐标。

它不减少 Tokens，不再做 FPS，不使用动态特征图。默认关闭；开启是独立于 compact stem 的第二个实验变量。`local_mixer_scale` 是坐标除数而非半径筛选，所以 kNN 仍可能跨越不同物体。

### 为什么保持两种特征含义

`HandSceneRelationEncoder` 的 context 分支使用 `patch_token`，near 分支使用 `local_token`，还依赖 `neighbor_idx` 查询原始可见几何。因此不得把经过跨 Patch 混合的特征覆盖到 `local_token`。`PointImageFusion` 同样依赖相同成员索引；本次不增加只在几何分支生效的 radius mask，避免语义/近场分支仍使用被排除的成员。

`local_token`：局部原始成员几何；`patch_token`：保留锚点的场景上下文特征。3D RoPE 不替代显式 `patch_center`，也不保证整个机器人策略 SE(3) 等变。输入点顺序改变可能改变 first-point FPS；固定锚点的排列等变性与重新采样稳定性是不同测试。

## API 与配置

原有所有位置参数、返回键和 out_dim/out_shape 保持不变。新增的 GeometryPatchEncoder 关键字参数：

```python
GeometryPatchEncoder(
    input_channels=6,
    token_channels=192,
    num_patches=128,
    group_size=32,
    depth=4,
    patch_encoder_type="compact",   # 默认 uni3d，保留历史实现
    local_mixer_neighbors=0,        # 默认 0；单独测试 8
    local_mixer_scale=0.1,
)
```

InteractionFlowAgent -> InteractionObsEncoder -> GeometryPatchEncoder 已贯通对应配置：

```yaml
agent:
  patch_encoder_type: compact
  patch_local_neighbors: 0
  patch_local_scale: 0.1
```

`interaction_flow_compact.yaml` 继承 `interaction_flow.yaml`，仅选择 compact 及单独的日志标识。当前标准 flow matching、8 层 DiT-X、16 scene + 6 hand + 1 state 输出配方未改；这不是把 128 Patch Tokens 直接送入动作模型的另一次架构修改。

在完整仓库环境中：

```bash
python -m pytest -q tests/test_geometry_patch_encoder.py
python dexmani_policy/smoke_test.py --config-only interaction_flow_compact
# 有数据及运行依赖时再做 bounded smoke，不自动启动训练或真机。
python dexmani_policy/smoke_test.py interaction_flow_compact
# 正式运行由使用者决定：
python dexmani_policy/train.py --config-name=interaction_flow_compact
```

后续单独测试局部关系模块：

```bash
python dexmani_policy/train.py --config-name=interaction_flow_compact \
  agent.patch_local_neighbors=8 workspace.wandb_cfg.id=compact_local8_seed42
```

请为不同实验使用不同产物目录/日志 ID，不恢复旧实验来启动新结构。

## 兼容与边界

- 省略新增参数时仍使用原 `uni3d` stem、无 LocalMixer；旧参数名/形状和计算顺序保持，严格加载旧 checkpoint 可用。
- compact 或 LocalMixer 启用后属于新的结构配置，需从零训练；不能以 strict=False 冒充成功恢复旧模型。每个新变体自身严格保存/加载正常。
- 不修改共享 `uni3d.py`、`ops.py`、PointNext、PointImageFusion、动作解码器或 normalization。
- 不新增预训练、timm backbone、SpConv、FlashAttention 包依赖；继续依赖仓库已有的 PyTorch3D FPS/kNN。
- 保留现有输入约束：每个样本至少 max(M,K) 个有限观测点，不支持将 NaN/Inf 或无效 padding 当作真实观测。
- 图像语义开启时仍须提供全部五项语义输入，point_uv 保持输入点序列。纯几何实验不启用该分支。

## 本次验证

PASS：修改 Python 文件语法检查；28 项 CPU 模块测试；原始 geometry_patch.py 的 Git blob SHA 校验；使用原文件构造的旧模型 state_dict 严格加载到默认新模型，FP32 输出逐元素完全一致。

CPU 测试使用明确的参考 FPS/kNN 替代 native primitives；验证环境按已读取源码提取必要依赖，未取得完整运行环境。测试涵盖 XYZ/XYZRGB、原版/compact、local mixer 开关、梯度、局部成员特征含义、固定锚点排列等变性、相对位置平移性质、重复中心、CPU BF16 autocast、图像语义融合/无有效视图，以及各变体自身严格 reload。

SKIP / NOT VERIFIED：原生 PyTorch3D 测试（缺依赖）；完整仓库 config-only/smoke、CUDA、torch.compile、DDP、真实数据训练、机器人 rollout。未运行完整回归测试，不声称性能提升或 GPU 加速。

## 运算量审计（不是速度基准）

B=1、N=1024、M=128、K=32、D=192、4 层、4 heads、FFN ratio=4、无图像语义分支。统计 Linear MACs + QK/AV 理论 MACs；不含 FPS、kNN、cdist、gather、激活、norm、pool、内存流量。

| 变体 | 参数量 | Stem MACs | 已统计总 MACs |
|---|---:|---:|---:|
| uni3d / local=0 | 1,913,152 | 540,016,640 | 791,674,880 |
| compact / local=0 | 1,857,120 | 67,371,008 | 319,029,248 |
| compact / local=8 | 1,895,984 | 67,371,008 | 324,861,952 |

这是消除重复宽通道计算，而不是大幅削减 Transformer 参数。算术量降低不代表 GPU 时延按同样比例降低，也不证明成功率更高。

## 对照顺序

先固定原参数预算比较 uni3d/local0 与 compact/local0，再比较 compact/local8。必要时再独立测试 depth=2、mlp_ratio=2、RoPE 长度尺度或局部 Fourier 输入，不能把全部变化合并归因于 compact stem。当前新增配置不改原有较强点 dropout；若精细几何退化，应另做传感器增强消融。
