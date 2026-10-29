# 手部场景编码：先建立局部几何状态，再读取场景

HandSceneEncoder 将共享场景转换为六个与身体位置对齐的增量，供后续触觉融合和动作生成消费。六槽顺序为 wrist/thumb/index/middle/ring/pinky。当前组件仍未接入实际策略训练；本次验证覆盖接口、数值和梯度，不能据此宣称任务成功率提高。

旧实现同时使用 Gaussian 距离头、逐边几何 bias/value、局部 mean/max、七维距离描述、腕部形状查询及腕/指独立融合层。局部几何直到末端线性拼接才影响输出，不能改变这一层对场景的读取；多个机制的独立贡献也不清楚。

重构保留两种具有不同分辨率和职责的输入：原始点云描述锚点附近的观测表面，patch tokens 提供经过上游编码的场景上下文。两者按“局部状态 → 条件化读取”的顺序连接，所有身体槽共享相同算子。

## 论文依据与适用边界

以下为论文中的机制及本模块的设计推论，后者不是论文对本实现的验证。

| 工作 | 原论文提供的依据 | 本次采用与边界 |
| --- | --- | --- |
| [Act3D, CoRL 2023](https://proceedings.mlr.press/v229/gervet23a.html)，方法与附录 | 以空间查询读取 3D 特征场，用 3D rotary attention 表达查询与场景的相对位置 | 将实测身体锚点作为查询位置，采用 RoPE + 标准 attention。没有移植 ghost-point 搜索、动作检测或 coarse-to-fine 解码。 |
| [FINGR, arXiv v3](https://arxiv.org/pdf/2609.33973v3)，§3.2–3.3、附录 A | 在掌系轴下构造指尖相对坐标，共享点网络并做 mean pooling，再加到对应手指状态；未来预测是独立的训练目标 | 采用共享相对点编码和加性几何状态。论文使用 32 个重建魔方点，本项目输入含场景背景，所以保留半径约束 KNN。没有添加未来力、转角或关节预测头。 |
| [MoPA, arXiv v2](https://arxiv.org/html/2609.12081v2)，§III-C–D | 为不同执行子系统构造专属感知查询，动作阶段再协调 | 保留六个身体对齐的查询及输出。移动底盘/机械臂的分离结论不能直接证明手指需要六套参数，因此本模块共享编码器，不新增 MoT 或动作分支。 |
| [FingerViP, arXiv v2](https://arxiv.org/html/2604.21331v2)，§III-D | 指尖视觉与对应相机位姿、逐指电流对齐，五指投影共享参数，另有全局视觉 | 采用身体槽对齐和共享局部处理。当前没有指尖相机输入，局部点云不能恢复遮挡后未观测的表面，也不能宣称复现其感知增益。 |
| [VisForce, arXiv v1](https://arxiv.org/pdf/2609.25785v1)，§III-B–D | 将当前/目标 actuator-force 信号放到对应指尖位置，并通过目标条件交叉注意力连接 | 保持视觉几何与触觉的身体槽对应；本模块输出场景增量，触觉仍交给 GatedTactileFusion。未定义的目标力不能由几何距离替代，也不增加图像渲染管线。 |
| [PointNet, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html)，§4 | 共享逐点网络与对称聚合处理无序点集 | 保留共享 MLP 与置换不变聚合；mean 选择来自 FINGR，PointNet 原文使用 max，不能将其结果解释为 mean 更优。 |
| [Point Transformer V3, CVPR 2024](https://arxiv.org/html/2312.10035v2)，§3–4 | 昂贵的邻域和逐边位置机制会限制吞吐，简化结构有助于扩展 | 删除逐边高维几何 Value，使用标准 SDPA。这里只有六次锚点查询，保留精确 KNN，不引入面向大规模点云的序列化或稀疏卷积依赖。 |

这些论文支持的共同方向是把观测与其作用位置对齐。具体采用“两步共享编码”、局部隐藏宽度 32、所有六槽使用局部编码，属于本项目的设计选择，需要下游消融。

## 两步计算

记身体 token 为 H，锚点为 a，patch 中心与特征为 c、F，末端到世界的旋转为 R。所有输入位置保留世界系米制单位，行向量的局部方向为 (p-a) @ R。

第一步，在每个锚点的 KNN 候选中保留半径 r 内的观测点。surface_k=32、surface_radius=0.04；六槽共享 Linear(3,32) → GELU → Linear(32,32)，随后 masked mean，得到 32 维 L。无有效邻居时 L=0。七维最近点描述 g 保留原定义：

~~~text
g = [relative_eef / (s+d), d / (s+d), exp(-0.5*(d/r)^2),
     has_reference, near_has_support]
G = Linear_without_bias(concat(L, g))                 # [N,6,D]
~~~

前三维是缩放方向，后四维是距离比例、proximity 和两个观测有效性标志。没有参考点时 g=0，公开最近距离仍为 inf。存在远处参考点但没有近处支持时 L=0、g 可非零，保留接近物体之前的几何信号。

第二步，以 H+G+query_context 为内容查询，用锚点和 patch 中心的位置执行 3D RoPE 交叉注意力：

~~~text
Q = RoPE(Qproj(LN(H + G + query_context)), anchor_eef)
K, V = KVproj(LN(F))
K = RoPE(K, patch_eef)
C = OutputProj(SDPA(Q, K, V, valid_mask))
relation_update = G + C
~~~

Q/K 的位置共同计算 RoPE，不能分别减去各自点集均值，否则会丢失身体与场景之间的相对平移。复用 pointcloud/position.py 中的实现；该类从 Sparse3D 原样移出，原模块的导入名称与模型状态键保持可用。

场景读取没有固定 Gaussian 衰减或局部硬遮罩，各身体查询都可读取远处任务相关 patch。局部点几何提供细节，RoPE 决定相对位置如何影响内容匹配；不再为每条锚点–patch 边生成 D 维向量。

空 patch 样本才允许读取追加的零 K/V 槽。有有效 patch 时该槽被屏蔽。局部投影和上下文输出投影均无 bias，因此空观测的增量有直接的零值语义；仍保留全缺测时的计算图。

后续融合保持：

~~~text
X = H + relation_update = H + G + C
U = X + tactile_update
~~~

G 参与 Query 构造，并作为一次残差进入 X。hand_query 始终是原始 H；调用方不得提前加 G 或重复加场景增量。六槽间的动作协调继续由后续策略负责，不在这里新增身体 self-attention。

## 接口与行为变化

默认 D=192、4 头，每头宽度须为 6 的正整数倍。metric_scale=0.10 同时用于距离描述和 RoPE 的米制缩放，移除 context_sigmas。旧组件 checkpoint 的参数布局不兼容，不做静默迁移。

patch 宽度可以与身体宽度不同，通过现有 K/V 投影直接映射：

~~~python
hand_scene_encoder = HandSceneEncoder(
    token_channels=192,
    scene_channels=sparse_encoder.out_dim,  # 默认 Sparse3D 为 384
)
~~~

scene_channels=None 保持与 D 相同；身体 token 和 GatedTactileFusion 的宽度仍须一致。

| 项目 | 变化 |
| --- | --- |
| 公开输出 | hand_query、relation_update、腕/指切片、世界系锚点、七维描述和有效性字段均保留 |
| 残差精度 | relation_update=G+C 显式以 FP32 相加，线性层和 SDPA 遵循 dtype/autocast |
| 几何梯度 | 本模块的坐标构造、KNN 和距离描述不反传；学习层及输入 tokens 正常反传 |
| 腕部 | 使用与五指相同的局部几何算子；移除独立的 15 维整手形状查询 |
| 局部聚合 | 四维输入、mean/max 拼接改为 XYZ 三维输入及单一 mean |
| 诊断 | 新增 geometry_update [N,6,D]、surface_features [N,6,32]；finger_local 改为 [N,5,32] |
| 注意力诊断 | 仅在 return_intermediate=True 时额外计算，脱离梯度图；生产路径始终使用同一 SDPA |
| null mass | 空 patch 样本为 1，其余为 0；不再解释为可学习的拒绝强度 |

局部几何不变性只针对固定 token 内容、同时变换坐标与 EEF 旋转的情况。上游 token 包含绝对位置时，整条策略没有因此成为 SE(3) 不变或等变模型。缺测 mask、物理接触和低观测密度仍是不同概念。

## 验证与计算代价

测试环境为 Python 3.12、PyTorch 2.4.1+cpu。原实现的四个相关契约测试先全部通过；重构后检查：

- EEF 坐标、刚体变换一致性，以及只移动场景时注意力发生变化；
- 原始点和 patch 的置换一致性、三种 query_context 形状、诊断与普通路径相同；
- NaN/Inf 和 mask、空长度输入、单模态缺失、远距离参考点；
- 学习层与有效 token 的梯度、无效 patch 零梯度、几何支路无坐标梯度、全缺测计算图；
- 384 维 patch 到 192 维身体 token 的前向与反向；
- HandStateEncoder → HandSceneEncoder → GatedTactileFusion 的 FP32/BF16 串联，场景与触觉只加一次；
- 触觉 Force/Taxel/Patch 三类编码器的原有契约；
- 抽出的 RoPE 类与旧类 AST 相同，Sparse3D 原导入路径仍解析到同一类。

无需数据或 PyTorch3D 的相关组件检查：

~~~bash
python - <<'PY'
import torch
from dexmani_policy.test_tools import common_contracts as contracts
torch.set_num_threads(1)
for name in ("hand_state_encoder", "hand_scene_encoder",
             "gated_tactile_fusion", "hand_interaction_chain"):
    print(name, getattr(contracts, "validate_" + name)(torch.device("cpu")))
PY
~~~

同宽 D=Ds=192 时，参数量从 **248,804 降到 157,472，减少 36.7%**。移除 [N,6,M,D] 的逐边几何 Value；在 N=32、M=96、D=192 下，这个张量本身占 FP32 13.5 MiB，或 BF16 6.75 MiB。这只是移除的单个张量大小，不是实测峰值显存降幅。KNN 的 [N,6,P] 距离计算保留。

微基准使用相同合成输入、P=1024、M=96、D=192、K=32、CPU 单线程、FP32、关闭中间诊断；训练耗时包括清梯度、前向和反向，不包括优化器或上游场景编码器。两轮原始中位数和 IQR 见 [benchmark JSON](hand_scene_encoder_benchmark.json)。

| 模式 | 第一轮：旧 → 新，ms | 第二轮：旧 → 新，ms |
| --- | --- | --- |
| N=1，推理 | 2.80 → 2.06 | 2.78 → 2.77 |
| N=1，前向+反向 | 6.20 → 4.46 | 6.45 → 10.64 |
| N=32，推理 | 47.00 → 22.25 | 129.17 → 27.40 |
| N=32，前向+反向 | 93.19 → 49.78 | 270.67 → 71.01 |

当前共享运行环境波动很大，第二轮部分 IQR 与中位数同量级，且 N=1 的训练结果反转。因此本次只把参数与逐边张量减少视为确定收益，不报告稳定的墙钟加速倍数。CUDA、峰值显存、torch.compile 和真实策略训练均未验证。

~~~bash
python -m dexmani_policy.test_tools.benchmark_hand_scene_encoder \
  --baseline-ref ebdacf9 --device cpu --threads 1 --min-run-time 3
~~~

可在目标 GPU 用 --device cuda 及 --bf16 复测；SDPA 的实际 backend 由设备、dtype 和 mask 决定，不承诺始终使用 FlashAttention。

## 需要下游实验回答的问题

后续最小消融应使用相同数据划分、训练预算和随机种子，对照旧版、只替换 RoPE/SDPA、只简化局部编码、完整新版，以区分算子优化与表示变化。额外比较是否给腕部局部几何、是否在 Query 中加入 G，可以检验共享六槽与两步计算的必要性。

重点观察接近物体、精细接触、遮挡/稀疏观测三类阶段的成功率和失败类型，并在目标 GPU 上测量 N=1 与训练 batch 的端到端开销。移除 Gaussian 先验、max pooling 和腕部整手形状可能改变样本效率；更少参数本身不证明更好的策略。

FINGR 的未来监督需要同 episode 的未来标签及独立损失；VisForce 的目标力需要明确的任务条件。它们可以成为后续策略实验，但不应在当前观测模块中以未监督的额外 head 或几何伪标签代替。
