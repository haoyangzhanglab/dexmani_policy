# 2D 语义与 3D 几何的 patch-level 对应与融合

`PointImageFusion` 独立实现点云 patch 与图像语义 token 的几何对应和门控融合。当前默认 `interaction_flow` 未调用此模块；点云编码器、视觉 Backbone、数据预处理和下游策略均保持各自职责。以下记录接口、数值约束与定向验证证据。

## 1. 设计结论

保留原实现最重要的效率优势：**点只参与建立标量对应，高维语义只在 patch 层聚合**。
不创建 `[B,N,C_image]` 或 `[B,M,K,C_image]` 点/成员语义张量，也不增加第二套大 Transformer。

```text
点云 patch 的成员索引 + 每点 UV/视角权重 + 实际图像网格
                       ↓
     A[B,M,V*Hf*Wf]：双线性对应，累加重复 token 引用
                       ↓
     pool：S = A @ F_image
     content：仅在 A>0 的范围内修正权重，再聚合原始图像 token
                       ↓
         LayerNorm → Linear → 标量门控残差
                       ↓
                  fused_token[B,M,C_point]
```

3D patch 与图像 patch 是多对多关系，不强制数量相等；不只投影 patch 中心。
若 P 是点到 3D patch 的线性聚合、R 是图像到点的线性插值，则 `P(RF)=(PR)F`。
本实现直接构造 `A=PR`。重复引用累加而非取唯一 token 后均匀平均，因此不会改变成员采样所定义的权重。

### 固定几何池化：默认 `alignment="pool"`

默认保留原来的多视角归一化、有效成员均值，以及池化之后的 `image_norm/proj`。
保留原参数名与原有 positional forward 参数；新增选项均为 keyword-only。
默认有限 FP32 输入的数学语义保持不变，但不承诺改变浮点求和顺序后的逐位一致。
FP64 几何计算和 AMP 下 FP32 sigmoid 是本次明确的数值改进。

### 内容修正：显式启用 `alignment="content"`

几何 token 经 LayerNorm 和 Linear 得到 Q，图像 token 经无参数 LayerNorm 和 Linear 得到 K。
匹配维度默认 32；不为全部图像 token 构造完整维度的可学习 Value 投影。

\[
\alpha_{mj}=\operatorname{softmax}_{j:A_{mj}>0}
\left(\log A_{mj}+\lambda\frac{q_m^\top k_j}{\sqrt{d}}\right),
\qquad S_m=\sum_j\alpha_{mj}F_j.
\]

`content_scale` 即 λ，初始化为零。初始权重恢复几何池化（浮点容差内），后续只调整已有支持范围内的内容选择。
首步 Q/K 的梯度为零，但 λ 可以获得梯度；λ 离开零后 Q/K 开始学习。测试覆盖这一训练启动行为。
CI=384、CP=192、d=32 时，只增加 `(384+192)*32+1=18433` 个参数。

这里的“局部”是**权重支持范围局部**：当前使用稠密低维 QK 和 BMM，不是稀疏执行 kernel。
内容模式增加计算，不能把它称为纯加速重构，也尚无质量提升的训练证据。

## 2. 为什么选择这些机制

- [DeepFusion，CVPR 2022](https://research.google/blog/lidar-camera-deep-fusion-for-multi-modal-3d-detection/)：聚合后的 3D 特征对应多个图像位置，支持由几何特征查询其对应图像特征，而非强行一一匹配。
- [AutoAlignV2，ECCV 2022](https://arxiv.org/abs/2207.10316)：限制跨模态关联范围可以避免全局注意力的开销；本实现没有照搬其可变形采样，也不声称复现其结果。
- [DINOv2 官方接口说明](https://huggingface.co/docs/transformers/en/model_doc/dinov2)：使用实际 patch tokens 和网格，不使用 CLS/global embedding 做空间对应。
- [SigLIP2 官方接口说明](https://huggingface.co/docs/transformers/en/model_doc/siglip2)：NaFlex 的实际空间形状与有效 mask 不能忽略；不同网格不能仅凭 padding 后 token 数按正方形重排。

没有引入 NeRF、全局 Q-Former、Top-K 截断、第二套点采样或动作侧条件。
稀疏索引读取不一定快于小矩阵 BMM；高维局部 Value 的展开也可能比稠密标量权重更占内存。
因此先保留小 ViT 网格下明确、可复核的稠密计算路径，而不凭理论边数宣称 CUDA 加速。

## 3. 输入契约

| 输入 | 形状与语义 |
|---|---|
| `local_token` | `[B,M,C_point]`，有限的几何 patch 特征 |
| `neighbor_idx` | int64 `[B,M,K]`，当前点云中的有效索引 `[0,N)` |
| `image_tokens` | `[B,V,Hf*Wf,C_image]`，已去除非空间 prefix tokens，视角内按行排列 |
| `point_uv` | `[B,V,N,2]`，与当前成员索引相同的点顺序，对应处理后图像像素坐标 |
| `view_weight` | `[B,V,N]`，外部几何可见性/质量权重，不在模块内估计遮挡 |
| `image_hw` | 实际处理后 `(H,W)` |
| `patch_grid_size` | 实际 `(Hf,Wf)`，允许非正方形 |
| `image_token_valid` | 可选 bool `[B,V,Hf*Wf]`，True 为有效图像 token |
| `member_weight` | 可选浮点 `[B,M,K]`，复用点编码器已有局部聚合权重 |

每次调用要求各视角共享处理后图像/网格尺寸。可变网格的 NaFlex 输入应先按真实网格分组；mask 并不能修复错误的行宽。
不同 Backbone 的特征维度通过构造参数处理，不硬编码 DINO 的 14 像素或 SigLIP 的网格。
DINO 的 CLS/register 数和 SigLIP 的 prefix 约定由调用方按真实模型去除，不能统一 `tokens[:,1:]`。

像素到特征坐标采用 `x=(u+0.5)*Wf/W-0.5`、`y=(v+0.5)*Hf/H-0.5`，有效图像边界为 `[-0.5,W-0.5)` / `[-0.5,H-0.5)`。
在特征边界做 border clamping，等价于对应规则网格的 `align_corners=False` 双线性插值。
如实际 Backbone 有不同 padding/stride/裁剪约定，必须先在调用方校正布局；本模块不会推测它。

RGB 裁剪/缩放、标定、深度注册、遮挡检查、点的重排/采样应在建立对应之前完成。
UV 与点索引不同步是输入错误；增加注意力不能修复这种错误。

### Mask 与质量信息

非有限/越界 UV、非有限或非正 view weight 不提供支持；正 view weight 裁剪至 `[0,1]`。
图像 mask 在插值权重归一化前生效：某视角只有 1/4 双线性支持时，其有效视角置信度也乘 1/4，不能与完整支持视角同权。
非正/非有限 `member_weight` 被排除，其余按 patch 内相对权重处理；即使权重为零，索引也必须合法，**不接受 -1 sentinel**。

覆盖率是有效成员权重与总成员权重之比；`semantic_confidence` 是有效成员的最大视角置信度的加权均值。
二者都是支持诊断，不是校准后的可靠性概率。单视角的相对归一化不会自动按绝对置信度缩小语义向量，绝对质量信息通过门控输入保留。

完全无支持的 patch 输出严格等于原几何特征，即使投影 bias 和门控 bias 已被训练为非零。
未被任何 patch 引用的图像 token 在 LayerNorm/矩阵乘法前清零，因此 masked/padded NaN/Inf 不会通过 `0*NaN` 污染其他 patch。
**被引用的图像特征与几何 token 仍必须有限**；模块不把 Backbone 的 NaN 训练故障静默解释成可用观测。

## 4. 使用与显式复用

```python
from dexmani_policy.agents.obs_encoder.interaction.point_image_fusion import (
    PointImageFusion, build_patch_image_weights,
)

fusion = PointImageFusion(image_channels=384, token_channels=192)
out = fusion(
    local_token, neighbor_idx, image_tokens, point_uv, view_weight,
    image_hw=(224, 224), patch_grid_size=(16, 16),
    image_token_valid=token_mask,  # 不需要 mask 时可省略
)
fused_token = out["fused_token"]

# 多个融合操作使用同一份几何布局时，只构建一次。
geometry = build_patch_image_weights(
    neighbor_idx, point_uv, view_weight, (224, 224), (16, 16),
    image_token_valid=token_mask,
)
out = fusion.fuse_tokens(local_token, image_tokens, geometry)

# 单独的结构实验，不静默改变已有 pool checkpoint。
content_fusion = PointImageFusion(384, 192, alignment="content", attention_dim=32)
```

没有全局持久缓存。任何 UV、点/视角顺序、邻域、网格、mask 或权重改变，都必须重建；不能跨运动帧复用过期对应。
`fuse_tokens` 检查 shape/device 等结构，不对每次输入执行强制 GPU→CPU 的标量扫描；它只接受遵守上述契约的 builder 结果。

`return_intermediate=True` 保留原字段，并新增 `semantic_weight`：pool 模式等于几何 prior，content 模式为修正后的权重。
`pool_weight` 始终表示原几何 prior，`semantic_token` 仍表示投影前的池化图像特征。

参数与梯度边界：几何构图在 `no_grad` 中；图像 token、几何 token、融合参数及内容修正分支保留梯度。
LayerNorm 不是线性的，因此没有把原先的 `LN(A@F)` 改成 `A@LN(F)`。
矩阵池化、内容 logits/softmax 至少使用 FP32；任一几何输入为 FP64 时构图保持 FP64，FP64 图像保持 FP64 聚合。
低精度 gate logits 转 FP32 再 sigmoid，不承诺与旧 AMP 数值逐位相同。

## 5. 定向验证与性能记录（2026-10-16）

相关文件：[融合模块](../dexmani_policy/agents/obs_encoder/interaction/point_image_fusion.py)、[定向测试](../tests/test_point_image_fusion.py)、[微基准](../scripts/utils/benchmark_point_image_fusion.py)。
模块未接入现有策略配置；下列测试和微基准不能证明完整策略的训练或推理性能。

**PASS：37 tests；SKIP：1 CUDA test**。环境为 Python 3.13.5、PyTorch 2.10.0+cpu。
测试通过文件路径加载独立模块，不加载预训练权重或仿真/点云后端。

覆盖独立 `grid_sample → 逐点语义 → 成员均值` 参考的 FP32/FP64 前向及全部原参数梯度；原 pool state_dict keys；图像/成员 mask；重复边、视角与点顺序；非正方形/1×1 网格；空支持与 NaN padding；可学习权重不越出几何支持；零尺度初始化与训练启动；CPU BF16；Float64 gradcheck；`aot_eager` fullgraph 前向和反向。
`aot_eager` 仅验证图捕获/自动微分，不等于 CUDA Inductor 优化验证。

```bash
python -m pytest -q tests/test_point_image_fusion.py
python scripts/utils/benchmark_point_image_fusion.py --min-run-time .5
python scripts/utils/benchmark_point_image_fusion.py --min-run-time .5 --backward
# 在目标 GPU 环境另行执行：
python scripts/utils/benchmark_point_image_fusion.py --device cuda --bf16 --backward
```

CPU 单线程合成测量：B=8，V=1，N=1024，M=192，K=32，图像网格16×16，CI=384，CP=192，随机 token/UV，FP32。
下列是本次 median，不是旧版本加速比，也不是实际视觉 Backbone 或部署延迟。

| 路径 | forward (ms) | forward+backward (ms) |
|---|---:|---:|
| pool，重建对应 | 13.38 | 26.53 |
| pool，复用对应 | 9.93 | 23.90 |
| content，重建对应 | 24.69 | 46.52 |
| content，复用对应 | 19.85 | 42.82 |

构图单独约 2.08 ms（forward 测量），对应矩阵 1.5 MiB。随机 UV 下平均关联94.75个图像 token，不能视为真实场景的支持密度。
内容修正有可见成本，因此不默认开启；重用对应只在输入布局确实不变时成立。

**NOT VERIFIED**：仓库目标 Torch 2.4.1/CUDA、CUDA BF16、CUDA Inductor、完整仓库回归、真实 DINO/SigLIP 输出与数据、训练后的语义融合质量。
CUDA scatter/BMM 的确定性和性能需在目标设备测量；未实现和未宣称无损 Top-K、稀疏加速或普遍最优参数。
