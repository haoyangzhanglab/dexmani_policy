# HandSpatialEncoder：面向手—场景交互的六点空间编码

实现日期：2026-10-16。源码：[`hand_spatial_encoder.py`](../dexmani_policy/agents/obs_encoder/proprio/hand_spatial_encoder.py)。

## 1. 职责与研究假设

空间编码器只回答“哪一个身体锚点、在哪里、相对腕部如何分布”。可见物体表面由 `HandSceneRelationEncoder` 读取，真实交互测量由独立触觉分支提供。坐标编码不推断接触、摩擦、载荷或物体身份。

输入是已经计算好的世界系米制位置，不执行 FK、旋转估计或坐标系变换：

```text
[BT,6,3] -> HandSpatialEncoder -> [BT,6,D]
order: wrist, thumb, index, middle, ring, pinky
```

绝对位置保留任务空间信息；逐点身份把位置与对应手指绑定；可选的腕部位移残差让每个指尖在进入场景检索之前，直接获得自己的手部布局提示。

**位移是输入 XYZ 的确定性函数，不增加传感信息。** 它是待检验的归纳偏置，而非完整手型、关节构型或接触状态。单个指尖的位置也不确定其传感器朝向。需要姿态信息的任务应另外评估姿态/关节状态路径，不能声称六点 XYZ 等价于旧姿态输入。

## 2. 编码公式

令 $p_0$ 为腕部，$p_1,\ldots,p_5$ 为五指位置。固定数值缩放为 $u_i=p_i/s_g$。对每轴应用 $L$ 个确定性频带：

$$
\gamma(u_i)=[u_i;\{\sin(\pi 2^k u_{i,a}),\cos(\pi 2^k u_{i,a})\}_{a\in\{x,y,z\},k=0}^{L-1}].
$$

实现复用 `NeRFSinusoidalPosEmb3D`，按照每轴先 sin 后 cos 的顺序展开；输入维度为 $3+6L$。保留原始坐标，不裁剪、逐帧中心化或依赖 batch 统计。

$$
\delta_i=(p_i-p_0)/s_h,\qquad
z_i=\operatorname{LN}\big(\operatorname{MLP}(\gamma(u_i))+W_\delta\delta_i+e_i^{ID}\big).
$$

`MLP` 为共享的 `Linear -> GELU -> Linear`。`W_delta` 为无偏置 `Linear(3,D)`，零初始化；因此腕部残差始终为零，初始函数与同权重的绝对位置基线相同，但线性权重仍能接收梯度。`use_wrist_offset=false` 时不构造这一层。

`p_i-p_0` 是世界轴下的位移向量，没有乘腕部旋转矩阵，也不将输入锚点改为腕部坐标。只有位移分支对共同平移不变；整体 token 包含绝对位置，不具备平移不变性或严格 SE(3) 等变性。

默认 $L=2,s_g=0.25\,m$，对应波长 $0.5\,m$ 和 $0.25\,m$。这是低开销的平滑位置基函数，不是毫米级接触分辨率承诺。近场尺度仍由关系模块的米制距离处理。$s_h=0.10\,m$ 仅缩放布局特征。

## 3. 实现与数值边界

- 默认 `D=192, hidden_dim=128`：绝对位置基线 28,352 参数；位移残差增加 576 参数；合计 **28,928**。不代表整套策略加速比例。
- 输入坐标应保留 FP32；Fourier 相位及位移使用 FP32。已量化为 BF16 的位置不能靠 `.float()` 恢复精度。
- MLP/Linear 允许 autocast；位移与身份残差相加使用 FP32，避免小更新在 BF16 加法中消失。标准 AMP 保留 FP32 权重，输出空间 token 为 FP32。
- 不加入 attention、FiLM、额外门控、时序模块或预测头。六个点各自编码；已有手部协调模块负责跨指信息交换。
- 输入形状、类型与有限值检查保留。几何无效应在数据入口修正，不用零点替代缺测；此模块不是可变长度点集编码器。
- 配置尺度影响函数语义；`state_dict` 之外必须保存并恢复配方。现有 checkpoint 的 `effective_config` 机制继续负责此事。

## 4. 在 InteractionFlow 中的集成

`InteractionObsEncoder` 持有且只调用一次 `self.hand_encoder`：

```python
hand_xyz = torch.cat((obs['eef_pose'][:, None, :3], obs['fingertip_xyz']), dim=1)
hand_tokens = self.hand_encoder(hand_xyz)
relation = self.relation(
    pointcloud, patches,
    anchor_xyz=hand_xyz,
    hand_tokens=hand_tokens,
    query_context=query_context,
)
```

数据格式仍为 `eef_pose [BT,9]` 与 `fingertip_xyz [BT,5,3]`，只在观测编码层拼接 XYZ；不重导出数据。`eef_pose` 来自当前测量而非未来动作。旋转分量不再被该空间/关系路径消费；关节状态路径保留。

原始 `hand_xyz` 用于米制距离，编码器内部缩放后的特征不回传给几何查询。关系向量统一为 `scene_xyz - anchor_xyz`，沿输入世界轴。删除旧 `wrist_rotation`、`edge_frame` 和关系模块内部的重复编码职责。

GAPT、scene memory、context/near 两路读取、触觉证据、模态门控、手部协调、动作骨干和训练超参数不变。Q/L 对照仍只改变读取 query 是否包含直接触觉；门控的输入不依赖读取结果。

## 5. 配置与受控消融

```yaml
agent:
  hand_spatial:
    hidden_dim: 128
    position_scale: 0.25
    hand_scale: 0.10
    num_frequencies: 2
    use_wrist_offset: true
```

无需额外工厂或编码器变体：

```bash
# A：原始 XYZ + identity
bash scripts/training/train.sh interaction_flow agent.hand_spatial.num_frequencies=0 agent.hand_spatial.use_wrist_offset=false
# B：加轻量 Fourier
bash scripts/training/train.sh interaction_flow agent.hand_spatial.num_frequencies=2 agent.hand_spatial.use_wrist_offset=false
# C：加逐指世界轴位移残差（默认）
bash scripts/training/train.sh interaction_flow agent.hand_spatial.num_frequencies=2 agent.hand_spatial.use_wrist_offset=true
```

这些命令是后续实验入口，本次没有启动长训练。固定数据、随机种子、坐标与锚点定义、触觉融合模式、其他网络和训练预算。C 对 B 检验布局偏置；B 对 A 检验 Fourier。禁用分支会少 576 个参数，不声称严格参数量匹配。

旧 `HandKinematicsEncoder` 同时使用腕部旋转、全手15D布局和6频编码，且关系边可用腕部轴。新模型相对旧模型改变了多个变量，不能把差异全归因于位移残差。原模型的历史实验应在原代码与保存配置下复现。

建议先报告成功率、位置变化下的泛化、接触后失败率和实际延迟。位置噪声、传感器有效位、接触区定义另做诊断；最近可见点不等于真实接触点，注意力或 null 权重不等于校准后的接触概率。

## 6. 迁移与验证

删除 `hand_kinematics.py`，新名称固定为 `HandSpatialEncoder`，不保留旧别名。旧 `edge_frame` 参数和旧 checkpoint 的模块布局不兼容；不要使用宽松加载静默续训。创建新实验；历史复现使用 [重构前提交](https://github.com/haoyangzhanglab/dexmani_policy/tree/aefbddc0f561e1f57689dcc66705a2fe84f243b4)。

本次在连接器读取的定向源码快照、PyTorch 2.10.0 CPU 环境执行：

```bash
python -m pytest -q tests/test_hand_spatial_encoder.py tests/test_geometry_aware_point_transformer.py --disable-warnings
```

结果：**75 PASS / 22 SKIP**。覆盖公式、梯度、零初始化学习、同位不同身份、逐帧顺序、FP32/BF16、严格权重往返、配置字段，以及实际观测编码器的单次编码、世界轴几何、Q/L 对照、缺测触觉和近场空支撑。CPU 几何集成使用测试参考 FPS/kNN；执行的 GAPT、关系 attention、门控及触觉编码计算保持原源码。

**NOT VERIFIED**：原生 PyTorch3D/CUDA、仓库 Torch 2.4.1 环境、完整 Hydra config-only/Agent 构造、DDP、真实数据训练/推理、完整 raw/EMA 策略恢复、闭环收益与硬件延迟。配置已做 YAML/AST 定向检查，但不冒充完整 Hydra smoke。75 个 PASS 不能解释为完整仓库测试通过。
