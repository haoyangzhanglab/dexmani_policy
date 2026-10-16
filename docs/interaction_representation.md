# Interaction Flow：逐指交互表征、实现与实验边界

实现核对日期：2026-10-16。以当前 [单卡配置](../dexmani_policy/configs/interaction_flow.yaml)、[DDP 配置](../dexmani_policy/configs/ddp/interaction_flow.yaml) 和实际调用链为准。历史实验读取各自保存配置与对应源码。

本次将身体位置路径重构为 [`HandSpatialEncoder`](hand_spatial_encoding.md)，移除腕部旋转与腕部坐标边；其余 GAPT、触觉融合、门控、手部协调和动作配方保持不变。完整的旧版研究线索、消融讨论及历史验证记录保留在 [2026-10-15 版本](https://github.com/haoyangzhanglab/dexmani_policy/blob/aefbddc0f561e1f57689dcc66705a2fe84f243b4/docs/interaction_representation.md)，不能作为新结构已验证的证据。

## 1. 研究问题与主张边界

灵巧操作需要对应三类信息：可见表面在哪里、哪根手指能作用于它、该手指当前感受到什么。场景点云提供可见几何，身体锚点提供作用位置，触觉提供局部物理交互测量。指尖靠近可见点不证明接触，接触读数也不能恢复遮挡表面。

研究问题是：**在有限的交互 token 内，何时、按什么身体粒度融合几何和触觉？** 当前假设是，在 patch 到逐指锚点的聚合阶段让当前触觉参与 query，同时保留直接触觉残差，可能更适合接触建立和调整。该假设尚需闭环 Q/L 对照，而不是由模块命名或注意力图证明。

工作标题仍可使用 **Touch-Conditioned Hand–Scene Relations for Dexterous Manipulation**。研究面向手—物交互，代码使用 `HandScene`，因为输入没有要求可靠物体分割，也不能保证已去除机器人自身点云。

空间编码器只构造位置、身份和逐指布局提示，不独立建模手—物关系，不推断接触，不确定完整手部姿态。世界轴位移残差是由 XYZ 得到的归纳偏置，不是新增观测。普通 Fourier、Linear、LayerNorm、门控或目录重构不作为独立算法创新，也不使用“首次身体锚定”“完整 SE(3) 等变”等主张。

## 2. 实际数据流

`BaseDataset -> BaseAgent.preprocess -> InteractionObsEncoder -> DiTX/RectifiedFlow`。Dataset 输出 `[B,T,...]`；preprocess 归一化并按历史截断、展平为 `[BT,...]`。直接调用观测编码器前必须遵循同一预处理语义。

```text
point_cloud -> GAPT -> shared geometry projection -> scene tokens
                                  |                    |
eef XYZ + fingertips -> HandSpatialEncoder -> gate <--- scene summary
                                  |            ^
joint_state -> StateMLP ----------+            |
tactile -> per-finger encoder -> evidence -----+
                                  |
       spatial + state (+ gated tactile only in Q)
                                  |
                    context + near geometry retrieval
                                  |
        spatial + state + gated tactile + gated geometry
                                  |
                  one hand coordination block
                                  |
              scene / wrist / five fingers / state
                                  |
                   DiTX + standard flow matching
```

默认每帧 `16 scene + 6 hand + 1 state = 23` 个192维 token；两帧 `[B,46,192]`。GAPT 内部384维，输出一次投影到192供 scene/context/near 共享。手部只编码一次，返回的空间 token 同时供门控和关系读取使用。

## 3. 世界系六点空间编码

`HandSpatialEncoder [BT,6,3] -> [BT,6,D]`，顺序为 wrist、thumb、index、middle、ring、pinky。观测层只拼接：

```python
hand_xyz = torch.cat((obs['eef_pose'][:, None, :3], obs['fingertip_xyz']), dim=1)
hand_tokens = self.hand_encoder(hand_xyz)
```

默认用原始数值缩放后的 XYZ、两频 Fourier、共享 `15 -> 128 -> 192` MLP、六个身份 embedding，另加无偏置、零初始化的逐点腕部位移残差，再做 LayerNorm：

$$
k_i=\operatorname{LN}\left(f_\theta(\gamma(p_i/0.25))+W_\delta(p_i-p_0)/0.10+e_i^{ID}\right).
$$

所有位置与位移均沿输入世界轴，不做 FK 或旋转。原始锚点仍以米交给关系模块，内部缩放不改变几何距离。`eef_pose` 的旋转分量不再被这一空间/关系路径消费，关节状态分支保留；不能宣称完全保留旧模型的姿态信息。公式、数值处理、参数量、消融与迁移见 [空间编码说明](hand_spatial_encoding.md)。

## 4. 场景与手—场景关系

GAPT 仍对每帧执行 FPS192/kNN32；成员几何 MLP、局部 attention 和六层384维3D RoPE Transformer 产生一套 token。训练默认随机 FPS 起点并打乱顺序，eval 确定性；采样不参与反传。其8频段2～32cm特征与手部的两频0.5/0.25m位置特征不同，不能混为一个分辨率参数。详见 [GAPT](geometry_aware_point_transformer.md)。

SceneContextPool 从 patch 中心确定性选锚点并分区，用区域均值初始化 query，然后读取全部 patch、执行 FFN/self-attention/LN。每帧重新分区，不保证时间槽跟踪同一物体。scene 分支不以手或触觉为输入，保留独立全局信息路径。

HandSceneRelationEncoder 不再持有 hand encoder，只接收原始 `anchor_xyz` 和已经编码的 `hand_tokens`。两路读取共享同一 patch memory：

| 属性 | context | near |
|---|---|---|
| 边 | 锚点到 patch 中心 | 锚点到 patch 内最近有效成员 |
| 每头 sigma（米） | 0.08 / 0.08 / 0.20 / infinity | 0.015 / 0.015 / 0.04 / 0.04 |
| 硬半径 | 无 | 3 sigma |
| 无局部支撑 | 仍可读取远处或 null | 全部头无支撑时 near_update 严格为零 |

全部边为世界轴向量 `scene_xyz - anchor_xyz`，不使用 `edge_frame` 或 `wrist_rotation`。边特征仍为 `[delta/0.10, distance/0.10, two Gaussian proximities]`，经 MLP 生成 attention bias 和 value。内容点积加边 bias、减高斯距离代价，加入 learned null 后做 FP32 softmax。返回 `context_update + near_update`，不在关系层再次加 query 残差。

near 支撑由原始 patch 成员而非中心距离决定，但不把整个 KNN patch 裁到接触半径内；读出的上下文 token 仍可能包含全局信息、自身机器人或背景。最近可见点不是接触标签，null mass 不是校准的接触概率或空域证明。移除边旋转不构成整个模型的等变性保证。

## 5. 触觉条件化读取与融合

记空间 token 为 $k_i$，关节状态为 $s$，同指触觉为 $e_i$，腕部 $e_0=0$：

$$
q_i^Q=k_i+s+g_i^t e_i,\qquad q_i^L=k_i+s,
$$
$$
u_i=\operatorname{ReadGeometry}(q_i),\qquad
z_i=\operatorname{LN}_{out}(\operatorname{HandAttention}(\operatorname{LN}_{in}(k_i+s+g_i^t e_i+g_i^v u_i))).
$$

**Q/L 只改变读取 query 是否包含触觉。** 两者保留相同直接残差、门控输入、归一化和手部协调。同权重同输入时读取前 gain 相同；独立训练后不保证数值相同。adaptive gate 下的 L 仍可让触觉影响 geometry gain，因此不是所有几何计算都与触觉无关的基线。

FingerModalityGate 读取空间+状态、同指触觉、scene均值、聚合触觉通道及有效位，通过零初始化末层和 `2*sigmoid` 输出独立 geometry/tactile gain，初始均为1。腕部固定1/0；缺测指触觉gain为0。gate 不接收 relation_update。BF16 下 logits/gain 保持FP32以保留小更新。gain是特征增益，不是置信度、接触概率或因果模态重要性。

`use_modality_gate=false` 跳过 gate MLP，几何gain为1，触觉gain由有效位决定。`tactile_fusion=none` 屏蔽全部触觉证据但保留布局，仍构造并调用触觉模块；不是删除输入契约或触觉计算的裁剪版本。

Force路径共享 `3->64->64->64` MLP；稠密taxel路径共享KineDex式1D CNN，每指输出64维。FingerEvidenceEncoder 投影触觉特征、保留聚合通道直连及有效读数标记，缺测严格置零；使用taxel时聚合读数仍被消费，不能删除。ReTouch-inspired可选编码器仍未接入本策略，不能作为完整ReTouch复现。

有效零读数不同于缺测。训练 tactile dropout 在样本/手指层采样，同一历史窗口共享；不用坐标归零模拟缺失。encoder mask不能替代数据过滤或训练统计拟合。真实聚合触觉通道未统一标定为世界系合力，不因几何使用世界系就旋转或改释触觉轴。

## 6. 数据契约与配置

| 字段 | 进入BaseAgent前形状 | normalization / 用途 |
|---|---|---|
| joint_state | `[B,T,19]` | limits / 状态 |
| point_cloud | `[B,T,N,6]` | identity / XYZRGB几何 |
| fingertip_xyz | `[B,T,5,3]` | identity / 五指锚点 |
| eef_pose | `[B,T,9]` | identity / 本路径只用XYZ |
| finger_tactile_force | `[B,T,5,3]` | gaussian / 触觉、证据、门控 |
| action | `[B,16,19]` | auto / 动作监督 |

不更改字段名、动作表示或数据导出。eef_pose必须来自当前测量/FK，不能来自未来action_ee。真实与仿真各自保证同一米制参考系，但二者数值坐标与触觉物理含义不能默认一致。详见 [数据与模态](data_modality.md)。

稠密触觉同时设置 `agent.tactile_input_key=taxel_tactile_force`，向dataset声明增加 `[B,T,5,120,3]` taxel字段并配置其gaussian归一化，保留聚合触觉字段。启用 `use_tactile_valid` 时提供bool `[B,T,5]` 并配置identity。真实数据使用已有Real runtime，不自动启动真机。

新增 `agent.hand_spatial` 配方：hidden_dim128、position_scale0.25、hand_scale0.10、num_frequencies2、use_wrist_offset true。`token_dim`仍由agent统一192。删除旧edge_frame配置，未知旧参数不静默忽略。

点云训练增强保持XYZ噪声0.002m、颜色扰动和每帧随机无放回保留768点；eval保留完整点云。XYZ/RGB共用索引，GAPT与关系模块共用采样后的Tensor。其他模态不随默认点云测量噪声移动；这不是共同刚体增强。状态噪声0.0002不变。

训练预算、优化器、NFE、EMA及动作模型均保持原配方：batch128、100000steps、AdamW1e-4、warmup500、BF16、8层DiTX。默认没有留出验证集。切换action_ee仍为21维动作，joint_state仍19维。

## 7. 动作与实验边界

标准flow matching：$x_t=(1-t)\epsilon+ta$，监督速度$a-\epsilon$；损失对batch/时间/动作维度取均值。默认时间采样为0.999乘Beta(1,1.5)。没有新增接触、轨迹或世界模型辅助监督。

DiTX读取scene/hand/state观测条件；观测在去噪循环外只编码一次，动作端每步仍重新投影条件，没有新增K/V缓存。默认10步Euler。输出完整16步，执行窗口为`pred[:,1:9]`，保留BaseAgent归一化和窗口语义；不是逐指独立动作头。

主研究对照仍为Q-L，保持数据、骨干、token、预算和重置条件一致。先关闭adaptive gate比较Q/L，再比较adaptive/unit增益；gate曲线不替代闭环结果。空间编码独立做A/B/C：原始XYZ、加入Fourier、加入逐指位移残差。不要同时改变触觉融合、边坐标、频率、姿态输入并把结果归因于单一机制。

诊断通过 `return_intermediate=True` 获得：hand_query、interaction_center、context/near更新、surface/context相对向量、米制距离、有效/支撑位、attention/null mass、geometry/tactile gain、指尖触觉证据和frame_tokens。正常训练不自动记录这些数组。先preprocess再调用encoder，避免重复归一化。

优先检查近场是否主要读取机器人自身、动作端是否绕过hand token、接触建立/释放是否需要时间信息。固定当前两帧条件；分割、额外时间模块、接触监督和预测头均留作独立后续研究。成功率与跨seed评测才支持收益主张。

## 8. 文件、迁移与验证

| 模块 | 当前职责 |
|---|---|
| `proprio/hand_spatial_encoder.py` | 世界位置、逐点身份、可选逐指布局残差 |
| `interaction/encoder.py` | 组装并仅编码一次六点；Q/L/N、门控、残差、frame-major组织 |
| `interaction/hand_scene_relation.py` | 只读外部hand token/米制锚点，context/near几何检索 |
| `interaction/finger_evidence.py`、`modality_gate.py` | 逐指触觉证据及读取前特征增益 |
| `pointcloud/geometry_patch.py`、`interaction/scene_context.py` | 共享空间memory与独立场景路径 |
| `core/interaction_flow.py` | 配置装配、几何identity契约、原动作配方 |

旧HandKinematicsEncoder、orientation/全手shape参数和edge_frame路径已删除；不是仅改类名的等价重构。旧checkpoint不能严格恢复成新模型，不做宽松自动迁移。历史实验使用对应源码；新模型从新run训练并保存完整effective_config。

本次定向验证见 [空间编码验证记录](hand_spatial_encoding.md#6-迁移与验证)：PyTorch2.10 CPU，**75 PASS / 22 SKIP**，包括实际观测融合链路；不是完整仓库或生产CUDA验证。历史文档中的293项CUDA/GAPT验证发生在本次重构之前，不能沿用于新模型。

在仓库的完整policy环境还需执行：

```bash
python dexmani_policy/smoke_test.py --config-only interaction_flow
python dexmani_policy/smoke_test.py --config-only ddp/interaction_flow
python -m pytest -q tests/test_hand_spatial_encoder.py tests/test_geometry_aware_point_transformer.py
# 准备有效数据/依赖后，才运行实际短程检查
python dexmani_policy/smoke_test.py interaction_flow --max-updates 4
```

config-only不构造实际Agent或读取数据；正式smoke不验证闭环收益。当前未验证完整Hydra smoke、原生CUDA/PyTorch3D、DDP、真实数据raw/EMA策略恢复或机器人性能。未启动长训练和真机运动。
