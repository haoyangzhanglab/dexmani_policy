# InteractionFlow：手部锚定的场景交互表征

本页记录当前的交互观测路径及其配置入口。模型实现与当次实验保存的配置优先于本文；[旧版研究讨论](https://github.com/haoyangzhanglab/dexmani_policy/blob/aefbddc0f561e1f57689dcc66705a2fe84f243b4/docs/interaction_representation.md)仅用于追溯历史机制，不适用于当前 checkpoint。

## 研究问题

灵巧操作需要区分可见表面的位置、手指相对表面的关系以及当前触觉证据。点云中的几何邻近不代表已经接触；注意力权重和门控增益也不是经过校准的接触概率。

当前可分别研究场景压缩与触觉进入几何读取的阶段：`tactile_fusion=query` 在手–场景检索前把同指触觉注入 Query；`late` 在检索后加入共同残差；`none` 屏蔽触觉。比较 GReST 时固定触觉模式和 token 预算；比较触觉模式时固定场景编码器、数据与训练预算。

## 实际调用路径

默认每帧输出 **12 scene + 6 hand + 1 state = 19 个 192 维 token**；两帧观测共 `[B,38,192]`，由 DiTX/RectifiedFlow 接收。scene 数由 `agent.num_scene_tokens` 控制，12 不是最优预算结论。

1. **场景几何**：`GeometryAwarePointTransformer`（GAPT）读取 XYZRGB，建立 192 个几何 Patch；共享 `geometry_projection` 将其投影到策略宽度，供 Scene 和 Hand 两条分支使用。
2. **场景压缩**：[GReST](grest_scene_tokenizer.md) 用几何约束竞争分组与一层区域关系编码输出 K 个 scene tokens；不消费触觉，不改变 hand reader 的完整 patch memory。
3. **身体锚点**：`eef_pose[..., :3]` 与有序的五指 `fingertip_xyz` 组成 `[BT,6,3]` 的世界/基座系坐标。`HandSpatialEncoder` 只编码位置和手指身份，输出 `[BT,6,D]`，不在此估计接触或执行 FK。具体见 [手部空间编码](hand_spatial_encoding.md)。
4. **场景读取**：`HandSceneRelationEncoder` 用独立的米制锚点 XYZ 计算场景中心与可见点成员的相对向量/距离，再通过 Context 和 Near 两条几何 Cross-Attention 检索；手部位置编码只计算一次。
5. **触觉与调制**：`FingerEvidenceEncoder` 产生逐指触觉证据，`FingerModalityGate` 根据检索前的共同观测计算增益。Query/Late 模式仅改变触觉进入几何检索的时机，直接触觉残差与最终协调顺序保持一致。
6. **协调输出**：六个 hand token 经一层手部 Self-Attention，与 scene/state token 合并输出；这里没有额外的接触监督或逐指独立动作头。

场景 XYZ、末端与指尖必须在同一米制坐标系。相对向量沿输入世界/基座轴表达；旧 `edge_frame=wrist`、`wrist_rotation` 路径已经移除。`eef_pose` 保持九维数据布局，但本空间读取分支只消费 XYZ。

## 文件与职责

| 文件 | 当前职责 |
| --- | --- |
| [`geometry_patch.py`](../dexmani_policy/agents/obs_encoder/pointcloud/geometry_patch.py) | GAPT、成员索引、共享几何 Patch |
| [`hand_spatial_encoder.py`](../dexmani_policy/agents/obs_encoder/proprio/hand_spatial_encoder.py) | 腕部与五指的简洁空间编码 |
| [`hand_scene_relation.py`](../dexmani_policy/agents/obs_encoder/interaction/hand_scene_relation.py) | 以身体位置为锚点的场景几何读取 |
| [`finger_evidence.py`](../dexmani_policy/agents/obs_encoder/interaction/finger_evidence.py) / [`modality_gate.py`](../dexmani_policy/agents/obs_encoder/interaction/modality_gate.py) | 触觉证据与逐指融合增益 |
| [`scene_context.py`](../dexmani_policy/agents/obs_encoder/interaction/scene_context.py) / [`encoder.py`](../dexmani_policy/agents/obs_encoder/interaction/encoder.py) | GReST 区域—关系场景编码与观测条件组装 |
| [`interaction_flow.py`](../dexmani_policy/agents/core/interaction_flow.py) | 组装观测编码器与 DiTX/Flow 解码器 |

独立的 [`PointImageFusion`](../dexmani_policy/agents/obs_encoder/interaction/point_image_fusion.py) 提供基于几何投影的图像语义 patch 融合，具体见 [2D–3D 融合说明](point_image_fusion.md)。**当前 GAPT/InteractionFlow 没有调用此模块**，其定向测试不能解释为已启用 2D 视觉语义的策略结果。

## 数据与配置

主配置为 [`configs/interaction_flow.yaml`](../dexmani_policy/configs/interaction_flow.yaml)，多卡入口为 `configs/ddp/interaction_flow.yaml`。以当前配置和真实调用方为准，不能把文档中的默认值视为历史实验保存的配置。

- 基础观测：`joint_state`、`point_cloud`、`eef_pose [BT,9]`、`fingertip_xyz [BT,5,3]` 和 `finger_tactile_force [BT,5,3]`；可选触觉阵列或 `tactile_valid` 以对应配置为准。
- `agent.num_scene_tokens` 是唯一 scene 预算入口；`agent.scene_tokenizer` 配置路由、几何尺度和关系层，诊断结果位于 `aux['scene']`。
- `agent.hand_spatial` 控制空间特征的 `hidden_dim`、`position_scale`、`hand_scale`、`num_frequencies` 与 `use_wrist_offset`，这些特征缩放**不改变**用于几何距离的真实 XYZ。
- 点云、腕部和指尖的 `normalization` 必须使用 `identity` 保留米制距离和同一参考系；Real 的 base 坐标值与 Sim 的 world 坐标值不需要数值相同，但不能在同一观测内混用。
- 图像语义融合还需要有效 RGB patch tokens、当前点的 UV/视角权重及对应点云分组；现有 `interaction_flow` 配方未声明这些输入。数据格式及标定边界见 [数据与模态说明](data_modality.md)。

## 运行检查

从仓库根目录、安装好依赖后运行：

```bash
python dexmani_policy/smoke_test.py --config-only interaction_flow
python dexmani_policy/smoke_test.py --config-only ddp/interaction_flow
python -m pytest -q tests/test_hand_spatial_encoder.py tests/test_geometry_aware_point_transformer.py
python -m pytest -q tests/test_grest_scene_tokenizer.py tests/test_grest_interaction_wiring.py
```

当依赖、数据和 GPU 齐备，且需要验证实际策略调用时，可运行：

```bash
python dexmani_policy/smoke_test.py interaction_flow --max-updates 4
```

`--config-only` 不读取真实数据或执行 GPU 算子，定向单测不代表完整策略评测；具体训练入口、checkpoint 和恢复边界以 [README](../README.md) 为准。

## 研究与验证边界

GReST 不兼容旧 SceneContextPool 权重、EMA 与优化器状态，必须新建实验；历史结果使用原源码与保存配置。新的手部空间编码同样不能严格恢复旧 `HandKinematicsEncoder` 权重。空间 Fourier、手指身份、相对位移和区域分组只是待消融的归纳偏置，不代表 SE(3) 等变、物体分割或接触判别器。没有基于此页声明真实数据闭环成功率或硬件时延。GReST 定向验证范围详见[专项说明](grest_scene_tokenizer.md#验证)。
