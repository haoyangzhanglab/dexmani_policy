# 轻量手—场景交互读取：整手布局与指尖表面

## 研究假设与边界

灵巧手需要两种互补的信息：腕部运动关心**当前手型如何相对目标和环境布局**，手指调整关心**局部表面相对各指尖如何分布**。共享场景编码不意味着六个锚点必须读取相同类型的信息。

本实现只验证这一感知假设：一层共享场景 Cross-Attention 读取已有 patch 特征；腕部 query 额外感知五指相对腕部的布局；五指共享一个小型相对点集编码器；两个线性投影分别融合腕部与指尖信息。没有新增 Encoder、时序预测、可变形采样、学习型门控、六点 Self-Attention 或动作专家。

这是启发式归纳偏置，不是已证明的最优架构，也不是完整的感知—动作分工策略。当前仓库没有调用该组件的默认策略配置，本次没有改变任何基线训练配方、动作表示或解码器。

## 文献中保留的最小思想

- [DexRepNet++](https://lqts.github.io/DexRepNet2/)：表面几何与手—物体空间关系互补。不引入占据网格和预训练分支。
- [FINGR](https://arxiv.org/abs/2609.33973)：共享点网络编码指尖相对几何。本实现读取局部观测点，不复现其魔方重建、触觉和未来监督。
- [GeoProp](https://alibaba-damo-academy.github.io/GeoProp/)：运动学锚点与感知位置显式对应。已有三维特征，不绕回二维投影。
- [MoPA](https://arxiv.org/abs/2609.12081)：感知职责应区分，但本次不修改动作访问路径，不能声称完成其感知—动作对齐。
- [ProxiDex](https://arxiv.org/abs/2609.16586)：显式 proximity 有交互意义。不把可见点距离等同于完整重建几何或真实接触，不增加动力学模块。

上述工作是动机来源；本实现不是任何论文的逐行复现，不继承论文的性能结论。

## 模型

```text
编码场景 patch XYZ + tokens ─────► 一层共享几何 Cross-Attention ───► context[6]
                                        ▲
hand_tokens[6] ──────────────────────────┤
五指相对腕部位移[15] ─► Linear ─► 仅调制 wrist query

原始候选表面点 ─► 六点 kNN ─► 最近位移/距离/proximity/有效性[6,7]
                         └─► 五指半径内点集 ─► shared 4→32→32 MLP
                                               └─► mean + max[5,64]

wrist:  Linear(context[0] + geometry[0])
finger: shared Linear(context[1:6] + local[5,64] + geometry[1:6])
```

这里图中的 `+` 表示特征拼接，不是逐元素相加。五指身份沿用调用方的 `HandSpatialEncoder`；不再新增重复身份表。

上下文中保留相对位移/距离的 logit bias 和 value 几何，以及逐头距离先验。每条边的 MLP 从 `6→D→D` 改为 `4→32→(D+H)`。固定零 null key/value 仅用于空 memory 的数值安全；其注意力权重不表示接触概率。无有效 memory 时上下文更新严格为零。

局部几何直接来自 `pointcloud[..., :3]`，不再为每个 patch 取一个最近成员后复用整个 patch 的语义特征。kNN 最多取 `surface_k` 个候选；只有距离不超过 `surface_radius` 的点进入指尖 MLP。没有近场点时局部特征为零，但有效远距离表面的距离描述和场景上下文仍保留。

## 输入与输出

入口：`dexmani_policy.agents.obs_encoder.interaction.hand_scene_attention.HandSceneAttention`。

输入中的 `B` 可以是已展平的 `B*T`，锚点顺序固定为 wrist/thumb/index/middle/ring/pinky。

| 输入 | 形状／含义 |
|---|---|
| `pointcloud` | `[B,N,C>=3]`，同坐标系米制 XYZ；允许 N=0 |
| `patches["tokens"]` | `[B,M,D]`，已投影到 `token_channels` 的编码特征 |
| `patches["xyz"]` | `[B,M,3]`，与 raw points/anchors 共用物理坐标系 |
| `patches["mask"]` | bool `[B,M]`，patch 有效性；允许 M=0 |
| `anchor_xyz` | 有限浮点 `[B,6,3]` |
| `hand_tokens` | 有限浮点 `[B,6,D]`，沿用现有 HandSpatialEncoder |
| `point_mask` | 可选 bool `[B,N]`，只约束原始表面点，不隐式更改 patch 语义 |
| `query_context` | 可选 `[B,D]`、`[B,1,D]` 或 `[B,6,D]` 的有限状态／任务条件 |

调用方仍负责投影 Sparse3DEncoder 的输出通道。输入不需要 `neighbor_indices` 或 `patch_radius`，因此也不必为本组件开启 `return_geometry=True`。

输出保留 `hand_query`、`relation_update[B,6,D]`、`interaction_center`。`wrist_relation` 和 `finger_relation` 是 update 的切片，不分配额外 Token。**调用方自行完成一次 `hand_query + relation_update` 残差**。

`geometry_descriptor[B,6,7]` 的顺序是：

```text
[nearest_dx/(s+d), nearest_dy/(s+d), nearest_dz/(s+d),
 d/(s+d), exp(-0.5*(d/r)^2), has_reference, near_has_support]
```

`s=metric_scale`，`r=surface_radius`。没有候选表面时描述子为零；`has_reference=False` 区分缺失与零距离观测。`min_observed_distance` 保留米制诊断量，缺失时为 `inf`，不能不加掩码地送入后续网络。

`return_intermediate=True` 额外返回上下文注意力、null mass、局部点集特征和有效邻域，便于排查，不将其当作标定过的物理概率。

### 最小使用

```python
import torch
from dexmani_policy.agents.obs_encoder.interaction.hand_scene_attention import HandSceneAttention
from dexmani_policy.agents.obs_encoder.proprio import HandSpatialEncoder

B, N, M, D = 2, 1024, 192, 192
points = torch.randn(B, N, 3) * 0.1
anchors = torch.randn(B, 6, 3) * 0.03
patches = {
    "tokens": torch.randn(B, M, D),  # 实际使用已有 Encoder + Linear 投影
    "xyz": torch.randn(B, M, 3) * 0.1,
    "mask": torch.ones(B, M, dtype=torch.bool),
}
hand = HandSpatialEncoder(token_dim=D)(anchors)
reader = HandSceneAttention(token_channels=D, surface_k=32, surface_radius=0.04)
out = reader(points, patches, anchors, hand)
condition = out["hand_query"] + out["relation_update"]  # [B,6,D]
condition.square().mean().backward()
```

先沿用已有的场景／状态条件，把这六个 Token 接到明确选定的策略实验中；不要在尚未对照验证前替换全部场景表示。

## 数据和迁移约束

- `point_mask` 只筛选局部几何，不是目标分割器。需要物体专属几何时，调用方提供目标支持点；手自身点应提前排除。全场景 patch 特征可能仍含手、障碍等上下文。
- NaN/Inf 表面点和无效 patch 会在归一化/投影之前排除；有限的全零点云不是“缺失”，应使用全 False 掩码或空张量。通用模态 dropout 不能靠乘零表达几何缺失。
- 坐标只做平移锚定，轴仍是输入坐标轴；不推断指腹法向，也不声称 SE(3) 等变。固定 query 特征时读取器具有共同平移一致性，但已有 HandSpatialEncoder 使用绝对位置，整个策略并非因此平移不变。
- 几何查询保留 observation-only/no_grad 语义，不支持通过 anchor 坐标梯度优化轨迹。网络参数、hand tokens 和编码场景特征仍能反传。
- 半径是有效局部支持范围，不是接触阈值；可见点距离不是 SDF，也不证明遮挡区域无碰撞。`0.04m` 和 `k=32` 只是初始配方。
- 保留主要返回接口，但删除了 `near_sigmas`、`radius_multiple`、`member_valid` 和旧 near-attention 诊断字段；`near_has_support` 现在是 `[B,6]` 而非按头标志。参数结构也改变，旧 checkpoint 应使用原源码，不提供静默兼容加载。

## 成本与验证

D=192、4 heads 时：旧结构参数 377,096，新结构 248,804，减少约 34.0%。旧数值按基准提交 `8deb01d665aa` 的构造器精确计数，新数值用 `sum(p.numel() for p in model.parameters())` 统计。这个比较不包含外部 Encoder/HandSpatialEncoder，也不是时延或成功率提升声明。

新模型只有一层六点对 M 个 patch 的注意力。几何查询为 `O(B*6*N)` 距离计算加 top-k；局部 MLP 最多处理 `B*5*k` 个点。不增加自定义 CUDA 或 PyTorch3D 依赖；大型点云的近邻搜索仍应按真实 profiling 决定是否优化。

从仓库根目录执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_hand_scene_attention.py' -v
```

定向测试覆盖正常/混合精度反向、空输入、掩码与非有限值、近/远/缺失区分、原始点独立于 patch 成员、腕部手型响应、局部形状响应、点顺序一致性和 query_context 广播。首次验证环境为 PyTorch 2.10.0 CPU；仓库的 Torch 2.4.1/CUDA、完整策略训练、仿真、真机和收益均未验证。

建议后续固定 Encoder、策略、数据和预算，分别对照：旧 HSA → 仅补真实局部表面 → 加腕部手型条件与职责融合。感知—动作对应和未来交互监督留作独立实验，不与本次读取器改动绑定。
