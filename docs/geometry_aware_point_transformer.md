# Geometry-Aware Point Transformer（GAPT）

这是从 PR #14（`d4af23f`，基于 `main@5975701`）集成的独立点云编码器。实现文件仍为 `dexmani_policy/agents/obs_encoder/pointcloud/geometry_patch.py`，类名为 `GeometryAwarePointTransformer`。不保留旧类别名，不加载预训练权重，不提供多种 Stem 或双输出头。

## 固定的默认结构

```text
XYZ / XYZRGB（米制，共同坐标系）
 -> 一次 FPS192 + 原始点 kNN32
 -> 相对 XYZ、距离、8 频段 Fourier 特征
 -> 64/128 维共享成员 MLP + max/broadcast/refine
 -> 同一支持集上的近/宽软距离池化 + max + 二阶几何统计
 -> 每个 Patch 一次 384 维投影
 -> 一层近邻 Patch 几何 Attention（12 邻居、128 维关系空间）
 -> 六层 384 维、8-head 全局 3D RoPE Transformer
 -> LayerNorm
 -> 一套 [B,192,384] Tokens + XYZ + 有效性 Mask
```

默认 XYZRGB 参数量：**11,814,084**（约 11.81M），与 GAPT-256 完全一致；Patch 数不引入网络参数。最后没有 global pooling、CLS 压缩、local/context 双投影或额外的局部输出相加。局部与全局模块内部保留标准残差、Pre-LayerNorm 和 LayerScale。

## 几何与效率细节

成员特征只在 128 维空间执行 Uni3D 式 max/broadcast/refine；384 维宽投影放在成员聚合后。原始相对 XYZ 没有被 Fourier 特征替代，也不按每个 Patch 的半径独立缩放，避免抹去物理大小。Fourier 使用 8 个对数间隔波长（0.02–0.32 m）。

Gaussian 池化尺度为 0.04 m / 0.12 m。两种加权均值和 max **共用同一次 kNN 的相同成员**；宽权重不能扩大支持范围，不等于完整的 12 cm 邻域。很小的邻域里两种均值可能接近。协方差只提供六个二阶统计量，不做特征值分解，不把它称为法线、曲率或真实接触面。额外保留局部覆盖半径标量；点数增加、遮挡和采样改变都会影响这些统计。

协方差以成员均值中心化，除以成员数，六项按 `xx/xy/xz/yy/yz/zz` 排列；坐标先除以固定 `geometry_scale=0.10 m`，因此统计输入已无量纲。距离、相位、协方差与加权和采用 FP32。聚合结果在禁用 autocast 的范围内以 FP32 拼接，再在投影前统一转为成员特征 dtype，避免 PyTorch 2.4 Inductor 将拼接编译成 FP32 输入与 BF16 权重混用的 `addmm`。该修复不增加参数，修复前后 eager FP32/BF16 输出逐位一致。

近邻 Attention 的 QKV 在 gather 前投影。中心图只构建一次，self 按索引而不是零距离排除；重复中心不会导致除零，但重复观测不会提供新的几何信息。相对位置进入局部 logit bias 和 value。全局 Attention 使用 PyTorch SDPA，Q/K Norm 在 RoPE 之前执行；三轴相位在 FP32 计算并复用。RoPE 尺度 0.05 m、base=100。归一化、频率/尺度与协方差不是额外训练目标。

RoPE 使用同一观测内的共同原点以改善数值精度；原始米制中心原样返回。模型不保证旋转等变性。固定锚点下的排列等变性，不等于 first-point FPS 对任意原始点排列不变。无输入点跨帧配准或跟踪。

## 公共接口

```python
from dexmani_policy.agents.obs_encoder.pointcloud.geometry_patch import GeometryAwarePointTransformer

encoder = GeometryAwarePointTransformer(input_channels=6)
result = encoder(pointcloud)  # [B,N,6]
# result['tokens']: [B,192,384]
# result['xyz']:    [B,192,3]，米制
# result['mask']:   [B,192]，True 表示有效
```

输入必须为有限的实际观测点，每个样本 `N >= max(num_patches, group_size)`。不支持把 NaN、Inf 或补零 Padding 当作有效观测；当前固定观测契约下返回的 Mask 全为 True，它不是缺测检测器，也不表示中心唯一。RGB 建议使用一致的 [0,1] 标度。坐标不要逐帧、逐轴独立缩放为不同物理单位。

`return_geometry=True` 另外返回 `center_indices [B,192]`、`neighbor_indices [B,192,32]`、`patch_radius [B,192,1]`，用于几何证据读取。这些只是索引与几何元数据，不是第二套特征。`patch_radius` 是原始支持集半径，不是经过全局 Attention 后完整的感受野。

编码器直接消费 XYZ 或 XYZRGB，不接机器人状态、动作、图像 Tokens、语言或时间索引。其他模态在策略侧融合；点的 RGB 随邻域成员进入几何编码，不依赖独立图像 Backbone。

[PointImageFusion](../dexmani_policy/agents/obs_encoder/interaction/point_image_fusion.py) 是独立的图像语义对应模块；其输入契约、融合模式和测试见 [2D–3D patch 融合说明](point_image_fusion.md)。当前 GAPT 和交互策略未调用它，不应把该模块的结果当作策略训练或推理的结果。

## 训练随机稀疏增强

当前 `interaction_flow` 在 Dataset 完成原有 2 mm 坐标噪声和颜色增强后、转 Tensor 前，按帧独立无放回保留 768 个真实输入索引。配置为 `augmentation_cfg.pc.dropout: null`、`random_keep_points: 768`；旧复制式 PointDropout 仍供其他策略使用，`Aug` 原地且不改 shape 的契约保持不变。

训练 `[T,N,C] → [T,768,C]`，验证/推理保留 `[T,N,C]`，当前预期 `N=1024`、`C=3/6`。采样使用 Worker 已设种子的 NumPy RNG；XYZ/RGB 共用索引，其他非逐点模态与时间顺序不变，返回独立连续数组且保持 dtype。无放回只保证不新增重复输入索引，无法消除传感器已有的重复坐标。

配置必须为正整数（拒绝 bool），或省略/设为 null 关闭。配置校验要求 `random_keep_points >= num_patches` 且 `>= group_size`，Dataset 按实际存储 shape 检查 `N >= random_keep_points`；不足时直接报错，不复制、补零或有放回采样。Normalizer 继续读取原始训练源行，Zarr 不修改。

GAPT 与 HandSceneRelation 消费 Dataset 输出的**同一份**点云 Tensor，所有中心/邻域索引均相对这份输入；编码器内不再采样子集，无需维护原始 1024 点映射。当前策略不消费独立逐点 UV/语义/有效性字段。

本次只改 Patch 数与训练稀疏增强：Fourier 保留 8 个对数频段、2～32 cm 波长、原始 Relative XYZ/Distance 和 FP32 相位。现有数值测试覆盖其实现；未发现需要修复的错误，因此不同时改变 Fourier 这个研究变量。

## 当前仓库衔接

`InteractionObsEncoder` 调用 GAPT 后，仅使用**一个共享的 384→192 投影**适配现有策略宽度。Scene 和 HandSceneRelation 都读取同一套投影后的 Tokens。近场和场景读取仍可使用不同 query、几何边与距离约束，但不再持有不同的 local/context 特征源。

`HandSceneRelationEncoder` 改为消费 `tokens/xyz/mask/neighbor_indices`，最近可见成员距离与无支持时 near_update=0 的保护保留。近场是受局部观测约束的读取，**其 value 已包含全场景上下文**，不能再描述为纯局部 Patch 特征。

Agent、ObsEncoder 和配置的默认 Patch 数同步为 192、深度为 6。策略输出仍是每帧 16 scene + 6 hand + 1 state，动作、触觉、normalization 与优化配方未更改。[交互架构说明](interaction_representation.md) 已同步单一空间 memory 的调用链。

此前从旧编码器迁移至 GAPT 是架构与状态字典的不兼容升级。旧 GeometryPatchEncoder/Compact/双投影版本 checkpoint 不能完整恢复到新模型，不提供 silent fallback 或用 strict=False 假装恢复。请使用新的运行目录；历史实验仍需匹配其保存的源码与配置。

当前 GAPT-192 与 GAPT-256 的参数形状一致，但采样配方不同；新训练目录和 W&B 标识包含 `gapt192`，不得作为历史实验的静默续训。显式恢复仍以实验保存的配置为准。

## 编译边界

`InteractionFlowAgent.compile_backbone()` 沿用现有 `training.use_compile` 和 `compile_mode`，编译 DiT-X 以及 GAPT 的 `_global_forward` 方法。后者只包含六层全局 Transformer 与最终 LayerNorm，使用 `fullgraph=True`；输入检查、FPS/kNN、几何统计、Patch Embedding 和局部关系继续 eager 执行。这样原生采样与邻域索引反向不进入编译区域，不需要自定义算子或新的配置开关。

编译绑定方法不改变 `blocks.*`、`norm.*` 等参数名、参数对象或模块层级。raw/EMA 在各自实例上分别编译，现有 GAPT checkpoint 可严格加载；这次执行方式调整不构成新的模型版本。smoke 的 checkpoint 对照在还原 DiT-X 后，也还原该方法，再比较 eager 预测，保留原断言。

## GAPT-192 本次验证（2026-10-15）

基于 `main@645d729`，在 `policy` 环境、PyTorch 2.4.1+cu124、原生 PyTorch3D、RTX 4090 验证。以下使用当前 192 Patch 配置，未运行消融或长训练：

| 检查 | 实际结果 |
|---|---|
| `python -m pytest -q tests/test_geometry_aware_point_transformer.py` | **49 PASS / 0 SKIP**；768/1024 点、XYZ/XYZRGB、CPU/CUDA FP32/BF16、索引/半径/数值测试、AdamW 更新及编译输出/梯度对照 |
| `python -m pytest -q tests/test_data_semantics.py` | **55 PASS / 0 SKIP**；独立逐帧无放回采样、属性/dtype/连续性、种子复现、配置边界、训练/验证分离、多 Worker 堆叠、源数据与统计来源不变 |
| `python -m pytest -q tests` | **293 PASS / 0 SKIP**；包含 CUDA 专项；4 条提示来自 TF32 设置与已有 VQ PCA 排序诊断 |
| 完整策略合成集成 | **PASS**；768 点训练、1024 点推理，默认 BF16/compile、12 次小 batch 更新、raw/EMA 严格恢复；条件仍为 `[B,46,192]` |
| `smoke_test.py --config-only interaction_flow` 与 `ddp/interaction_flow` | **PASS**；DDP 继承配置，未重复覆盖参数 |
| Python 语法、文档链接、新运行目录与 `git diff --check` | **PASS**；输出目录与 W&B 标识均带 `gapt192` |
| `smoke_test.py interaction_flow --max-updates 4` | **NOT VERIFIED / 已尝试失败**；真实 Zarr 为 `[21293,1024,6]`，缺少 `fingertip_xyz`、`eef_pose`、`finger_tactile_force`，在 Dataset 校验时停止；未修改真实数据 |
| 多卡 DDP、真实数据训练/恢复、闭环效果 | **NOT VERIFIED**；本次不据合成测试推断任务效果 |

无放回采样不处理传感器原有重复坐标；当前训练路径未消费独立逐点 UV/语义字段。仿真 runner 与部署 runtime 的观测入口均不调用训练随机采样。CUDA 和多 Worker 测试在主机权限下执行；沙箱内多 Worker 检查曾中止，主机重跑通过。

## 历史 GAPT-256 验证记录

以下计数、延迟和显存均属于 **256 Patch 的历史配置**，不代表 GAPT-192 的测量结果。

2026-10-15 在本地 `policy` 环境（Python 3.10、PyTorch 2.4.1+cu124、原生 PyTorch3D、RTX 4090）完成最终集成检查：

| 检查 | 结果与范围 |
|---|---|
| Python 语法、实际 import、`git diff --check` | PASS |
| GAPT 定向测试 | **40 PASS**；含原生 CPU/CUDA，XYZ/XYZRGB，FP32/BF16，默认模型前反向与 AdamW 更新、全局阶段编译输出/梯度对照 |
| 几何与位置编码 | PASS；独立 Gaussian/中心化协方差/Fourier 数值对照、米制 RoPE 相位与 Q/K Norm 顺序、固定锚点下完整 forward 排列等变、重复坐标按索引排除 self、随机 FPS 索引对齐 |
| 策略接口 | PASS；真实 Scene 与两路 HandSceneRelation 共享同一 Tensor，原始成员距离/硬半径/无支持零更新，每帧 23 tokens |
| CUDA 合成策略集成 | PASS；默认模型、BF16、DiT-X 与 GAPT 全局阶段 compile、normalization 模式、optimizer 与 warmup；2 条合成观测执行 12 次更新后 GAPT 参数实际变化，raw/EMA 严格保存加载及预测一致 |
| `pytest -q tests` | **243 PASS / 9 SKIP**（CPU 可见环境，跳过 CUDA 专项；GAPT 的 CUDA 项另行通过） |
| 单卡与 DDP config-only | PASS；DDP 实际多卡运行 NOT VERIFIED |
| 真实数据完整 smoke | **NOT VERIFIED / 已尝试失败**；默认数据缺少 `fingertip_xyz`、`eef_pose`、`finger_tactile_force`，在建 Dataset 时拒绝运行；未替换、推测或改写旧数据 |

参考 FPS/kNN 仅用于部分 CPU 数学检查；原生算子检查独立执行。reader 测试直接调用真实模块，以 hook 观察 memory，不再用替身 attention。合成策略测试保持原学习率和 500-step warmup：零初始化动作输出与门控下，4 次更新不足以观察到 GAPT 首层 FP32 参数变化，故仅延长此小批量 fixture 的检查预算至 12 次。

初始集成曾额外尝试编译整个 GAPT：CUDA BF16 前反向可运行，但原生 PyTorch3D 产生 graph break，部分 backward 无法使用 CUDA Graph。该尝试的 FP32 compiled/eager 最大差异为 `2.38e-6`；BF16 最大差异为 `0.0382`、RMS 为 `0.00719`，在预设 `atol=rtol=0.025` 下有 1/98,304 元素超限。这个历史对比仍记为失败，阈值未放宽；当前正式路径采用上文的局部编译边界。

全局阶段编译验证：FP32/BF16 的完整 GAPT 输出、原始几何索引、输入及各阶段参数梯度对照通过；参数身份与 state_dict 键不变。三组 BF16 输入的最大输出差异约 `0.0044`，保留原 `atol=rtol=0.025`；梯度按阶段比较相对 L2，FP32/BF16 阈值分别为 `2e-4/0.025`。12 次合成策略更新及 raw/EMA 严格恢复通过。Profiler 在连续三次 GAPT 前反向中记录到 6 次 `cudaGraphLaunch`，编译区域实际发生 replay。

RTX 4090、PyTorch 2.4.1 上的 BF16 全局阶段编译前后短测（同一模型/输入布局，5 次预热，含同步的墙钟均值；GAPT 30 次，策略 20 次）：

| 范围 | GAPT 全部 eager | GAPT 全局阶段 compile | 峰值 allocated（前→后） |
|---|---:|---:|---:|
| 完整 GAPT 推理，`[1,1024,6]` | 2.64 ms | 2.08 ms | 84.2→81.3 MiB |
| 完整 GAPT 前反向，`[4,1024,6]` | 8.08 ms | 5.97 ms | 400.5→269.7 MiB |
| 完整策略前反向，B=2、T=2 | 22.88 ms | 17.66 ms | 898.2→738.5 MiB |

策略两组都编译 DiT-X，只改变 GAPT 全局阶段的执行方式；前反向耗时不含 optimizer/EMA 更新和数据加载。短测不是默认 batch=128 的吞吐承诺，也不证明任务效果。

初始集成时 GAPT 全部 eager 的单独测量（输入 `[1,1024,6]`，10 次预热、30 次前向，含同步的墙钟均值）：

| 精度 | 推理时延 | 推理峰值 allocated | 单次 AdamW 训练峰值 allocated |
|---|---:|---:|---:|
| FP32 | 2.83 ms | 79.4 MiB | 247.5 MiB |
| BF16 autocast | 2.28 ms | 113.2 MiB | 248.3 MiB |

以上包含模型与当前测试张量，BF16 推理还包含 autocast 权重缓存；不是整个策略或 batch=128 的显存估计。直接运行无目录的 `pytest -q` 会收集本地 `experiments/.../snapshot/smoke_test.py` 并发生同名模块冲突；正式回归范围使用 `tests/`，历史产物保持原样。

完整项目环境可运行以下检查，它们不是消融实验：

```bash
python -m pytest -q tests/test_geometry_aware_point_transformer.py
python -m pytest -q tests
python dexmani_policy/smoke_test.py --config-only interaction_flow
# 有任务数据与完整依赖时运行 bounded smoke；不自动启动长训练或真机。
python dexmani_policy/smoke_test.py interaction_flow
```

架构合理性、参数量与模块测试不能保证任务成功率；以上性能结论仅限所列环境与短测范围，不声称任务精度提升。

## 参考边界

- Sparse2Act：https://arxiv.org/abs/2606.12759 —— 稀疏点云 Patch 与空间关系建模的结构线索；不复现其动作对齐预训练。
- Uni3D：https://arxiv.org/abs/2310.06773 —— 局部 PointNet 式 Patch 特征及 Transformer；本实现不等于预训练 Uni3D。
- LitePT：https://arxiv.org/abs/2512.13689 —— 局部阶段与高层 Attention 的分工，以及三维 RoPE；这里没有复制其 SparseConv/多级序列化网络。
- Fourier Features：https://arxiv.org/abs/2606.12334 —— 多频率坐标输入的动机；具体波长、Gaussian 池化、协方差与单层局部 Attention 的组合是本实现的设计，不是该论文已验证的同一个模型。
