# Codex 任务书：Interaction Flow 视触觉交互表征重构

> 状态：待执行的实现规格；本文中的目标模块与性能收益尚未实现或验证。
>
> 编制日期：2026-10-15（Asia/Shanghai）。
>
> 代码依据：[haoyangzhanglab/dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，main 分支提交 [d615bb225658ed5e754344aaf2671c788502ab05](https://github.com/haoyangzhanglab/dexmani_policy/commit/d615bb225658ed5e754344aaf2671c788502ab05)，提交说明 “1014 temp”，提交时间 2026-10-14 04:13:55 UTC。
>
> 本任务只重构 interaction_flow 的观测表征。先遵守根目录 [AGENTS.md](AGENTS.md)，再执行本文。若执行时 HEAD 已变化，先检查本文涉及文件的差异，按实际接口调整，不回滚用户新改动。本任务书不是仓库级长期规则。

## 1. 目标与研究主线

灵巧操作需要联合利用手物空间关系与指尖接触反馈。我们以腕部和五指尖为手部锚点，从共享点云表征中读取场景上下文与手部几何特征，再通过触觉门控注意力融合逐指触觉，形成手物交互表征。交互表征、场景上下文和本体状态共同作为流匹配策略的条件。

实现收敛到以下边界：

1. **GeometryPatchEncoder** 只负责共享点云编码。
2. **GeometryReadout** 只接收几何，以一次 cross-attention 同时生成场景特征 S16 与手部几何 H6。
3. **TactileGatedInteraction** 接收 H6、其锚点元数据和逐指触觉，以 6 queries 读取 11 tokens，合并跨模态融合与指间信息交换。
4. **StateEncoder** 独立编码本体状态。
5. **InteractionObsEncoder** 只编排上述模块并打包条件；已有 DiT-X / RectifiedFlow 负责动作生成。

研究假设是：共享几何读取保留手物空间关系，读取后的触觉融合补充实际接触反馈，使几何、触觉和本体状态各自具有明确职责。相对几何进入 attention 的权重与 value；触觉门控控制触觉 token 对交互更新的贡献。不要将该假设写成已经证明的性能结论。

这里的输入仍是场景点云。最近可见 patch 点不一定属于目标物体，也不等同于真实接触点。当前任务不增加物体分割、接触监督或遮挡补全；论文中应准确说明可见几何的观测范围。

## 2. 当前实现与必须替换的路径

以下事实均对应上述提交的源码与配置，不能用旧架构图替代。

| 当前代码位置 | 已确认的行为 | 本次处理 |
| --- | --- | --- |
| [interaction/encoder.py](dexmani_policy/agents/obs_encoder/interaction/encoder.py) → InteractionObsEncoder | 分别调用 SceneContextPool 与 HandSceneRelationEncoder；state 加入几何 query 和手部残差；query 模式下触觉先进入几何读取 | 改为单向编排，几何读取前不接触 state/tactile |
| [interaction/scene_context.py](dexmani_policy/agents/obs_encoder/interaction/scene_context.py) → SceneContextPool | 空间分区池化初始化 16 queries，随后 cross-attention、FFN 和 scene self-attention | 复用分区与池化算法；取消独立的 scene attention 路径 |
| [interaction/hand_scene_relation.py](dexmani_policy/agents/obs_encoder/interaction/hand_scene_relation.py) → HandSceneRelationEncoder | 分别从 patch_token 和 local_token 做 context/near 两次读取；带 null token；默认将相对向量转到 wrist frame | 替换为一次共享读取；保留最近可见 patch 成员的几何定义 |
| [proprio/hand_kinematics.py](dexmani_policy/agents/obs_encoder/proprio/hand_kinematics.py) → HandKinematicsEncoder | 使用 XYZ、rot6d、五指相对腕部形状编码，并输出 wrist_rotation | 新路径仅使用六个 XYZ 与固定部位身份，不实例化此旋转编码器 |
| [interaction/finger_evidence.py](dexmani_policy/agents/obs_encoder/interaction/finger_evidence.py) | 同时投影帧触觉与原始聚合 contact_force，补 valid embedding，输出含零腕部槽位的 6 tokens | 由 Interaction 内部的五指投影替换；不再额外注入 raw-force 残差 |
| [interaction/modality_gate.py](dexmani_policy/agents/obs_encoder/interaction/modality_gate.py) → FingerModalityGate | 根据读取前 hand/state、scene summary、touch 和 force 生成独立 geometry/tactile gains | 改为读取后逐指 tactile logit gate；不再门控 geometry gain |
| [interaction/attention.py](dexmani_policy/agents/obs_encoder/interaction/attention.py) | 旧几何 attention 显式构造全维 pairwise edge_value；另有手部 self-attention | 用低维关系聚合替换；新 Interaction 不再叠加手部 self-attention |
| [pointcloud/geometry_patch.py](dexmani_policy/agents/obs_encoder/pointcloud/geometry_patch.py) | FPS + KNN + PatchEncoder + 4 层 3D RoPE patch attention；几何元数据目前依赖 return_intermediate | 保留编码算法，增加正常前向可用的最小几何输出 |
| [tactile/xhand_frame.py](dexmani_policy/agents/obs_encoder/tactile/xhand_frame.py) | 单帧 MLP / ArrayEncoder，输出每指 64 维，支持 bool 有效位 | 直接复用 |
| [core/interaction_flow.py](dexmani_policy/agents/core/interaction_flow.py)、[configs/interaction_flow.yaml](dexmani_policy/configs/interaction_flow.yaml) | 默认 SIM split pose 包含 eef_rot6d；8 层 DiT-X、标准 FM、10 步 Euler；每帧 23 tokens | 更新观测字段和 encoder 参数；保留动作配方及条件布局 |

当前的“场景读取一次 + 手部 context/near 各一次”总计是三条几何 cross-attention 路径。仅将 tactile_fusion 改为 late，或仅将 edge_frame 改为 base，都不能得到本任务架构：前者仍保留多次读取和额外残差，后者仍使用方向编码。

## 3. 固定约束与控制变量

### 3.1 表征约束

- 六锚点固定顺序：wrist、thumb、index、middle、ring、pinky。
- 点云与六锚点共用同一米制 base/world 坐标系。空间相对量 p-a 是同一帧、同一坐标系内的位移。
- 不新增旋转观测，不读取 eef_rot6d，不生成 wrist_rotation，不做腕部局部坐标变换。
- 不做五指局部原始点云裁剪/重复编码；只共享一份 patch memory。
- GeometryReadout 不接收触觉、本体状态或它们的摘要。
- Interaction 不接收原始点云、scene summary 或 joint_state，不重新查询 patch memory。
- 本体状态只进入独立 StateEncoder 和最终条件打包。
- 两帧使用共享权重分别编码；不引入触觉差分、历史 MLP、观测时序 attention、跨帧 feature pooling 或跨帧 BatchNorm 统计。
- 不新增接触预测、未来触觉、重建、动力学或其他辅助 loss。
- 保留传感器有效位，但有效位不作为额外 token 或“接触概率”。
- 不将 learned gate 称为已校准的置信度、可靠性或接触概率。

已有 DiT-X 的 frame position embedding 和动作对两帧条件的 attention 保留；它们属于动作生成器，不是新增观测时序编码。流时间 t 也不等于观测帧时间。

### 3.2 保持不变的配方

保持当前 action_key/action_dim、动作归一化与反归一化、horizon=16、n_obs_steps=2、n_action_steps=8、8 层 DiT-X、hidden_dim=768、8 heads、FM 时间采样、NFE=10、EMA、优化器、学习率、batch size、训练预算、数据划分及现有增强。

保持 BaseAgent 的动作执行窗口：预测 [B,16,Da]，执行从 n_obs_steps-1 开始的 8 步。action_ee 原有的旋转控制维度属于动作协议，不因移除旋转观测而被删除。

不启用 GeometryPatchEncoder 中现有的可选 PointImageFusion，不顺带引入 RGB backbone、预训练模型或新依赖。坐标 RoPE 的名称虽然含 rotary，它编码的是 XYZ 位置，不是 eef 姿态输入；本轮保留。

## 4. 输入输出契约

令 B 为样本数，T=2，BT=B*T，D=192，M=128，K=32，S=16，关系维度 r=16，几何读取与交互 attention 均为 4 heads。

| 模块 | 输入 | 输出 | 职责 |
| --- | --- | --- | --- |
| 观测适配，置于 InteractionObsEncoder | 已按原始布局归一化的逐帧观测 | point_cloud、A、touch、valid、state | 只做字段选择、reshape 和校验 |
| GeometryPatchEncoder | point_cloud [BT,N,3/6] | patches：token [BT,M,D]、center [BT,M,3]、member_xyz [BT,M,K,3]、member_valid [BT,M,K] | 一次共享编码 |
| GeometryReadout | patches、A [BT,6,3] | S [BT,16,D]、hand={token: H [BT,6,D], xyz: A [BT,6,3]} | 一次读取，保留锚点元数据 |
| XHandTactileFrameEncoder | touch [BT,1,5,3] 或 [BT,1,5,120,3]；valid [BT,1,5] | F [BT,5,64]，去掉长度为 1 的时间轴 | 每指、每帧触觉编码 |
| TactileGatedInteraction | hand、F、valid [BT,5] | Z [BT,6,D] | 一次 6×11 attention |
| StateEncoder | joint_state [BT,19] | s [BT,1,D] | 独立本体表征 |
| 条件打包，置于 InteractionObsEncoder | S、Z、s | cond [B,T*(16+6+1),D]=[B,46,192] | 按帧依次排列 [S16,Z6,s1] |
| 已有 RectifiedFlow / DiT-X | cond、noisy action、flow time | velocity；Euler 得到 action chunk | 保持现有动作生成行为 |

hand 使用一个简单字典即可；固定顺序已定义 part ID，无需引入大型数据容器或新的 registry。A 由 GeometryReadout 随 H 返回给 Interaction，不建立第二条触觉到几何或本体到锚点的反馈路径。

### 4.1 字段适配

- SIM：从 eef_pos [BT,3] 和 fingertip_points [BT,15] / [BT,5,3] 组成 A；删除默认 sensor_modalities 和 normalization 中的 eef_rot6d。
- Real：已有 canonical 数据只有 eef_pose [BT,9]，使用前三维作为腕部 XYZ；后六维不进入任何 encoder、query、gate 或位置变换。无需新增磁盘数组或修改原数据。
- 新配置使用 wrist_position_key: eef_pos 或 eef_pose，替换含姿态语义的 wrist_pose_key。
- 选择 contact_force 时，仅消费该字段；选择 tactile_force 时，仅消费该字段，不再强制读取额外 contact_force。
- SIM contact_force 的 [BT,15] 到 [BT,5,3] reshape 必须发生在现有 normalizer 之后，保留原有 15 通道统计。不要为了统一布局改变归一化粒度。
- 几何字段保持 identity；joint_state 和触觉沿用当前归一化配方。Real eef_pose 使用 identity，只有 XYZ 被模型消费。
- consumed_observation_fields、dataset.sensor_modalities、normalization 和 env_runner.sensor_modalities 必须一致。仓库的 validate_observation_fields 会拒绝遗漏和未消费字段。

## 5. GeometryReadout：一次共享几何读取

建议新增文件：dexmani_policy/agents/obs_encoder/interaction/geometry_readout.py。

### 5.1 正常输出几何元数据

修改 GeometryPatchEncoder 时保持默认接口兼容，增加明确的 return_geometry=False 关键字；新 encoder 使用 return_geometry=True。

直接复用 forward 中已计算的 neighbor_xyz，不再让下游携带整份原始点云通过 neighbor_idx 重新 gather。正常几何输出不依赖 return_intermediate；后者仅控制可视化/诊断信息。

当前 GeometryPatchEncoder 要求有限点且 N>=max(M,K)，因此常规 member_valid 全为 true。保留该输入要求。若接口接受部分无效成员，最近点计算必须使用 mask；不能把零坐标当作有效表面。不要借本任务扩展整个点云缺测/补全系统。

### 5.2 构造 22 queries

- Scene queries：迁移现有 _spatial_partition 与区域平均池化算法；由 16 个区域的 pooled patch features + seed XYZ 的 PE 初始化。
- Hand queries：共享的位置 MLP 编码 A，加 6 个可学习部位 identity embeddings。
- PE 可直接沿用 SceneContextPool 的两层 XYZ MLP 形式，作用于米制 XYZ；避免额外手形编码、姿态编码或新位置编码框架。
- Stack Q=[Q_scene;Q_hand]，形状 [BT,22,D]。
- K/V 共享 memory=LN(patch_token+PE(patch_center))，只投影一次。
- 只执行一次 22×M 的多头 cross-attention，然后共享 residual/FFN，拆分为 S 与 H。不再增加 scene self-attention 或手部第二次几何读取。

区域池化只是构造 scene query，不是第二个 learned attention。当前只有一种 patch 分辨率：称为“单次、多距离范围几何读取”，不要写成已经具备多分辨率点云层级的“多尺度编码”。

### 5.3 最近可见表面关系

迁移并保留 nearest_patch_evidence 的数学语义。每个锚点 a_i 对 patch j 的有效成员取：

~~~text
p*_ij = argmin_{p in valid_members(j)} ||p-a_i||
delta_ij = p*_ij-a_i
d_ij = ||delta_ij||
e_ij = MLP([delta_ij / metric_scale, d_ij / metric_scale])   # r=16
~~~

metric_scale=0.10 m。最近成员选择和距离计算用 FP32；离散索引选择可 no_grad，但关系 MLP 与投影必须能训练。

不要用 patch center 距离代替该默认表面距离，不对 XYZ 分别做标准化后再相减。距离仅代表可见几何接近程度，不等于接触、占据概率或无遮挡证明。

### 5.4 关系进入权重与 value，避免全维 pairwise value

仅对后 6 个 hand query 的几何关系生效；scene query 使用普通内容读取。

~~~text
L_hand[i,j,h] = dot(q_i^h,k_j^h)/sqrt(d_head)
                + w_h^T e_ij
                - d_ij^2/(2*sigma_h^2)

alpha = softmax(L)
u_hand_i^h = sum_j alpha_ijh * v_j^h
             + U_h(sum_j alpha_ijh * e_ij)
~~~

- 默认 sigma=[0.015,0.04,0.20,inf] m，分别覆盖近、中、远和无显式距离衰减的 head。
- YAML 中用 null 表示最后一个 global head，内部转换为无距离衰减；同时将规则和转换后的语义记录清楚。
- global head 仍保留可学习的关系 bias/value，不声称其注意力完全与距离无关。
- 不使用旧 near 分支的硬 radius cutoff，不增加 null token。生产输入至少有足够有效 patches 初始化 16 个区域；不足时明确报错，不伪造几何。
- 对无有效成员的 patch 屏蔽 key；对非法/非有限 memory 报错。不要让整行 -inf softmax 产生 NaN。
- 先聚合 r 维关系再用每头 U_h 映射至 d_head；U_h 使用无 bias 线性投影。
- 禁止构造 [BT,6,M,D] 或 [BT,6,M,heads,d_head] 的全维 pairwise edge_value。必要的关系张量是 [BT,6,M,r]。
- 计算 logits、距离偏置与 softmax 使用 FP32；其余投影按现有 AMP 配方。
- 训练默认不返回 attention maps；return_intermediate=True 才提供 surface_distance、surface_valid、scene/hand attention 等诊断。

这里的效率改进是减少独立读取与全维 pairwise 中间张量。GeometryPatchEncoder 本身仍可能占主要耗时；未经 profile 不承诺整条策略按相同比例加速。

## 6. TactileGatedInteraction：读取后的一次融合

建议新增文件：dexmani_policy/agents/obs_encoder/interaction/tactile_interaction.py。

### 6.1 逐帧触觉与身份

复用 XHandTactileFrameEncoder，保持 contact_force 的两层 MLP 和 tactile_force 的现有 ArrayEncoder，不使用 XHandTactileEncoder.history_projector。

投影 F [BT,5,64] 得到 E [BT,5,D]，加入固定五指顺序对应的 finger embeddings 和 tactile modality embedding。几何与触觉使用各自的 LayerNorm，避免将两类 token 的统计强行共享。

腕部没有触觉 token。不补第六个虚拟触觉 token，不另做 raw-force 残差、触觉残差支路或 valid embedding。合法零读数通过正常触觉 encoder 表示；缺测通过 key mask 表示。

### 6.2 一次 6×11 cross-attention

~~~text
Q = Wq(LN_q(H))                          # 6 queries
G = LN_geometry(H) + geometry_type
E = LN_tactile(Proj(F)) + finger_ID + tactile_type
K,V = Wk([G;E]), Wv([G;E])              # 6 geometry + 5 tactile

key_xyz = [A; A[:,1:]]
key_part_id = [0,1,2,3,4,5,1,2,3,4,5]

b_ijh = small_MLP([delta_xyz/scale, distance/scale])_h
         + learned_part_pair_bias[h, query_part_id, key_part_id]
~~~

空间关系 bias 使用 A 和 key_xyz；手指身份对应由固定排序确定，不通过原始传感器幅值推测。part-pair bias 可以用很小的 [heads,6,6] 参数表，不建立图网络或第二层协同模块。

### 6.3 逐指门控只进入触觉 key logits

~~~text
ell_f = GateMLP(concat(LN_gate(H_tip_f), F_f))
g_f = sigmoid(ell_f)

L_ijh = dot(Q_i^h,K_j^h)/sqrt(d_head) + b_ijh + log_rho_j

geometry key:           log_rho = 0
valid tactile key f:    log_rho = logsigmoid(ell_f)
invalid tactile key f:  log_rho = -inf

alpha = softmax(L)
Z0 = H + Wo(alpha @ V)
Z = Z0 + FFN(LN(Z0))
~~~

具体要求：

- 门控只读取同指 H_tip 与 F，不读取 s、S、原始力旁路或上一帧。
- GateMLP 是共享的小 MLP，建议 hidden_dim=64、输出每指一个标量；所有 query/head 共享该指的 key gate。
- 末层 weight=0、bias=0，初始 g=0.5；该初始化不是旧 2*sigmoid 的 unit gain。
- 使用 FP32 logsigmoid 和 softmax，避免先 sigmoid 再 log 的下溢；不能依赖 BF16 在零点附近保留微小门控差异。
- gate 只加在 logits，不再同时乘 E/V，避免双重抑制。
- 输出一次 attention residual + FFN 即止，不追加 hand self-attention。
- 6 个几何 keys 始终提供非空有效集合，因此触觉全部缺失时仍可正常前向。
- 每个固定 head 内，触觉 key gate 改变几何/触觉的混合比例，但不会改变两个几何 keys 的相对注意力比值。不要把本机制描述为“触觉重新定位物体几何”。

### 6.4 有效位及数据支持边界

- use_tactile_valid=false 时创建全 true 的 [BT,5] mask。
- true 时要求显式 bool mask；有效零读数不能据阈值改成 false。
- 在触觉编码前用 torch.where 清除无效值，编码后再次置零；投影/embedding 后仍屏蔽缺测 keys。不能用 NaN*0 作为清洗。
- tactile_dropout_prob 保留为编码阶段的 mask dropout，默认 0。新实现按 [BT,5] 采样，不借 dropout 将两帧绑定；不改变几何输入。
- tactile_fusion=none 时将 5 个 tactile keys 全部屏蔽，保持 6 queries、11 个槽位和最终 46-token 布局。

**数据层现有限制必须保留在实现说明中：** BaseDataset 按原始有限性筛选窗口；Real runtime 在进入 encoder 前拒绝非有限输入；normalizer 当前不会按 tactile_valid 排除有限占位值。encoder 层的 NaN-mask 测试不能被报告为端到端缺测数据训练已经支持。

本轮默认实验使用现有有限、无缺测污染的训练数据及编码后 mask dropout，不重写通用 Dataset/normalizer。若训练集实际存在 valid=false 的传感器样本，应在拟合触觉统计前拒绝该未支持配方，并报告需要 mask-aware fitting；不要静默将占位值纳入统计。该检查只在本策略且显式启用 tactile_valid 时触发，不能扩大成全仓库数据重构。Real 推理可使用有限占位值 + false mask，调用方仍须满足 runtime 的有限性契约。

## 7. 编排与策略集成

InteractionObsEncoder 的主要前向应保持如下直观结构；辅助字典仅用于诊断：

~~~python
# BaseAgent 已完成 normalization 和 [B,T,...] -> [BT,...]。
anchors, touch, valid = adapt_current_frame_inputs(obs)
patches = self.geometry(obs["point_cloud"], return_geometry=True)
scene, hand, geometry_aux = self.readout(
    patches, anchors, return_intermediate=return_intermediate
)
frames = self.tactile(touch[:, None], valid[:, None])[:, 0]
interaction, interaction_aux = self.interaction(
    hand, frames, valid, return_intermediate=return_intermediate
)
state = self.state(obs["joint_state"])[:, None]

frame_tokens = concat_with_existing_type_embeddings(scene, interaction, state)
cond = frame_tokens.reshape(B, T * (S + 6 + 1), D)
return cond, diagnostics_if_requested_else_empty
~~~

adapt/concat 可以是该类中的短方法或几行代码，不必再建立模块层级。实际返回类型按此契约实现，不把这里的伪代码 helper 当作仓库现有函数。

- 正常 forward 返回 (cond,{})；不通过 aux["loss"] 加额外目标。
- 保留现有三个 token_type embeddings；顺序为每帧 S、Z、s，再按时间从旧到新排列。
- 不拼出额外 F5、valid token 或 scene summary token；总数仍为 23/frame。
- 保留 metric_observation_fields 的 identity 约束，但删除对旋转轴的假设。
- modality_dropout_probs 继续禁止对几何与所选触觉字段直接乘零；触觉缺测使用专用 mask。
- 使用已有 BaseAgent、optimizer 参数收集与 EMA，不新增基类、工厂或跨目录依赖框架。

## 8. 文件级任务与提交顺序

| 阶段 | 文件/范围 | 工作与完成条件 |
| --- | --- | --- |
| P0：输入契约 | core/interaction_flow.py、interaction/encoder.py、configs/interaction_flow.yaml | 移除旋转依赖；字段声明与配置一致；Real 只切 eef_pose 的 XYZ；引入明确的 representation_version |
| P1：共享读取 | pointcloud/geometry_patch.py（小改）、新增 interaction/geometry_readout.py | 正常输出成员几何；迁移分区池化与最近成员算法；实现一次 22×128 读取和低维关系 value |
| P2：交互融合 | 新增 interaction/tactile_interaction.py、interaction/encoder.py | 复用逐帧触觉；实现 6×11 attention、逐指 logit gate 与 masking；删除 state/touch 到读取前的依赖 |
| P3：移除旧活动路径 | interaction 下旧 scene_context、hand_scene_relation、finger_evidence、modality_gate、attention 及 proprio/hand_kinematics 的相关引用 | 全仓库搜索实际调用者；迁移仍需的两个几何 helper；无调用后再删除被替代的文件/类。不要保留新旧两套可误选的活动路径 |
| P4：集成与文档 | configs/interaction_flow.yaml、configs/ddp/interaction_flow.yaml、docs/interaction_representation.md、docs/data_modality.md | 更新参数、SIM/Real 配方、有效位支持边界、接口与图；DDP 继承新配置，保持原 batch 配方 |
| P5：定向验证 | 新增 tests/test_interaction_representation.py、tests/test_interaction_flow_contract.py | 覆盖下一节的关键不变量及 Agent/normalizer/恢复接口；复用现有测试风格 |

上表简写路径均相对于 dexmani_policy/agents/obs_encoder，core/configs/docs/tests 使用其仓库实际目录。实施前对旧类做全仓库 rg；不要仅因本文建议删除就删除仍有其他调用者的共享实现。

仅在实际兼容要求出现时小改 dexmani_policy/training/build_utils.py，例如缺测训练配方的窄范围拒绝；不要为了新 encoder 改写通用 normalization、loader、runtime 或 checkpoint 框架。现有 strict restore 和字段校验应被复用。

当前 dexmani_policy/train_ddp.py 使用 find_unused_parameters=False。固定门控、关闭关系 value 或关闭触觉等消融，必须在构造时冻结/不实例化不会参与计算的参数，或保持明确可达的计算路径；不能留下未使用的可训练参数，再通过打开 find_unused_parameters 掩盖问题。有效位导致某批次梯度为零与结构性未使用参数应区分。此项属于消融配置验收，不要求本次自动启动多卡训练。

### 8.1 新配置建议

以下是加入/替换 agent 段的规格，不是可直接覆盖整个 YAML 的完整文件：

~~~yaml
agent:
  representation_version: 2
  wrist_position_key: eef_pos       # Real: eef_pose，只取 XYZ
  tactile_input_key: contact_force  # Real array: tactile_force
  tactile_fusion: interaction       # interaction | none
  tactile_gate_mode: learned        # learned | fixed
  tactile_gate_init: 0.5            # fixed 消融使用相同初始值
  tactile_dropout_prob: 0.0
  use_tactile_valid: false
  readout_heads: 4
  relation_dim: 16
  metric_scale: 0.10
  readout_sigmas: [0.015, 0.04, 0.20, null]
  use_surface_bias: true
  use_relative_value: true
~~~

D=192、M=128、K=32、patch_depth=4、S=16 等沿用现有键和值。校验 head 整除、正整数 relation_dim、sigma 个数、有限正尺度和 gate_init∈(0,1)。若 gate_init 允许不同值，初始化 bias=logit(gate_init)。

移除 edge_frame、wrist_pose_key、use_modality_gate 以及旧 query/late 配方；不要静默将旧键映射到新语义。Agent 构造参数与 YAML 必须同时更新。

默认 SIM 字段：joint_state、point_cloud、fingertip_points、eef_pos、contact_force。Real 使用 eef_pose 替换 eef_pos，按实际传感器二选一 contact_force / tactile_force；后者不再依赖前者。normalization 字段集合同步更新。

### 8.2 Checkpoint 与历史对照

- 新配置显式记录 representation_version=2，并在构造时验证。新 Agent 必须要求该版本字段，避免缺省值把旧配置误认成新架构。
- 旧配置/权重不保证在新 Agent 中运行。严格拒绝不匹配，不使用 strict=False、自动补参或自动转换掩盖差异。
- 旧实验在其保存代码版本和绑定配置下恢复；不要覆盖历史 config.yaml、checkpoint 或 normalization。
- 原始 baseline 使用本文固定的提交和该实验真实配置；模型重构后新训练。不要用新默认值反推旧实验配方。
- 本次不建设 legacy encoder 分支或跨版本兼容层。
- 虽然条件长度仍为 46，H/Z 的生成路径与权重布局已变；相同 shape 不代表语义兼容。

## 9. 验收：验证机制边界，不只检查 shape

新增测试必须检验可观察的不变量；小尺寸合成数据可用于单元测试，但不能冒充真实训练 smoke。不要重复堆积只镜像实现的测试。

| 检查 | 具体通过标准 |
| --- | --- |
| 单次读取 | hook 或明确的模块调用计数证明：GeometryReadout 只有一次 22-query cross-attention；没有独立 scene/context/near attention |
| 单次融合 | Interaction 的 attention 为 [BT,heads,6,11]，输出 Z6；不存在后置 hand self-attention |
| 旋转独立 | SIM 完全不提供 eef_rot6d 仍可前向；Real 固定 XYZ、只改变 eef_pose 后六个有限值，S/H/Z/cond 不变 |
| 状态解耦 | eval 确定性条件下，只改 joint_state，S/H/Z 不变，只有 s 变化 |
| 触觉顺序 | 只改一帧触觉，S/H 不变；检查 Interaction 对有效触觉有可用梯度，不要求随机初始化一定提升任务表现 |
| 帧独立 | 只扰动 t=0 的点云、锚点或触觉，t=1 的 frame_tokens 不变；测试前关闭随机增强/dropout，使用 eval FPS |
| 表面邻近 | 反例中一个 patch 的中心较远、成员较近，计算出的表面距离确实取成员；不偷偷回退到 center |
| 关系 value | 小张量数值对照：先聚合 r 维再线性投影，与等价的显式 pairwise value 公式在容差内一致；关系 MLP/U_h 有梯度 |
| 几何有效性 | 无效成员不参与最近点；无效 patch 不获 attention；不足有效 patch / 非法点云明确报错；不出现 NaN |
| 门控语义 | 固定 Q/K/V/b，降低单指 gate 降低该触觉 key 的概率；两个几何 key 的注意力比值不变 |
| 门控数值 | 极端 ell、FP32/BF16 场景前向与反向有限；使用 logsigmoid；FP32-only 环境不能冒称 BF16 CUDA 已验证 |
| 有效零与缺失 | 有效零保留可读 key；invalid key 概率严格为零；全部 invalid 等价于 tactile_fusion=none；encoder 边界无效 NaN 不污染输出 |
| 字段契约 | 配置字段与 consumed_observation_fields 精确对应；SIM/Real/array 路径均覆盖，array 路径无 contact_force 依赖 |
| 布局与梯度 | cond=[B,46,192]，按帧 [S16,Z6,s1]；所有新可训练参数进入 optimizer 一次，无断梯度/重复参数 |
| 策略接口 | 实际 compute_loss 与 predict_action 的接口保持；pred_action=[B,16,Da]，control_action=[B,8,Da] |
| 保存恢复 | 新配置、新权重、normalizer 与 EMA 按现有 strict 路径恢复；旧配方明确失败；不放宽 source recipe 校验 |

DiT-X 的 final layer 为零初始化，初次动作 loss 反向不一定立即给 encoder 非零梯度；门控末层零初始化也会使其上游在初始一步出现合法零梯度。表征梯度测试应直接对表示定义测试标量，或在真实受限更新后检查；不要把这一现有初始化性质误判为新模块断梯度，更不要为通过测试修改 decoder 初始化。

### 9.1 验证命令

完成实现后，先在仓库对应环境执行：

~~~bash
python -m pytest tests/test_interaction_representation.py tests/test_interaction_flow_contract.py -q
python dexmani_policy/smoke_test.py --config-only interaction_flow ddp/interaction_flow
~~~

注意当前 --config-only 只做配置校验、target 定位及 Agent import，不实际构造模型或运行 forward，不能替代新测试。

有原始 CUDA、依赖和数据时，再执行现有受限生产路径：

~~~bash
python dexmani_policy/smoke_test.py interaction_flow --max-updates 4
~~~

现有 smoke 不支持用单进程替代 multi-rank DDP 验证。缺少 CUDA、数据、PyTorch3D 或外部仿真依赖时标记 NOT VERIFIED；不将生产配置改成小模型/CPU 后宣称原配置通过。不启动完整训练、批量评测或真机运动。

## 10. 最小实验与性能判断

这些是后续实验设计，不是本任务要求自动启动的训练。

| 对照 | 唯一主要改变 | 回答的问题 |
| --- | --- | --- |
| 新架构完整配置 | 基准 | 新表示的总体表现 |
| tactile_fusion=none | 屏蔽触觉 keys，保留布局 | 触觉是否提供额外有效信息 |
| tactile_gate_mode=fixed | 每个有效指 g 固定为相同初始值 0.5 | 学习式门控是否优于同强度常数门控 |
| use_relative_value=false | 去掉关系 value，保留关系 logit 与表面偏置 | 显式关系内容是否有用 |
| use_surface_bias=false | 去掉解析距离惩罚，保留关系 logit/value | 表面邻近先验是否有用 |
| 原提交 baseline | 使用原代码、原实验配方 | 整体迁移是否值得；不是单机制归因 |

g=1 可以另作普通融合对照，但它与初始 g=0.5 的对照同时改变了触觉先验强度，不能单独证明“动态门控”收益。删除一个机制的消融只用于相应问题，不同时改 NFE、训练预算或动作模型。

性能报告分别测 GeometryPatchEncoder、GeometryReadout、Interaction、整个 obs encoder 和全策略推理；固定输入点数、batch、精度、设备、FPS 设置与 NFE，CUDA 同步并预热，报告参数量、延迟和峰值显存。分别记录模块与整体结果；只测 6×11 attention 的速度不能代表策略速度。

默认尺寸是依据当前代码与模块预算选择的实现起点，不是已经搜索得到的最优超参数。“最佳方案”在本文指当前约束下最清晰、可复用且可验证的修改路线，最终效果由实验决定。

## 11. 交付与完成定义

执行 Codex 应交付：

- 端到端接通的新 observation encoder 与明确的模块接口。
- 更新后的 interaction_flow / DDP 配置及 SIM、Real 字段说明。
- 旧活动路径清理结果；若有保留代码，说明真实调用者，不留下无人使用的新旧开关。
- 两个定向测试文件与实际运行记录，区分 PASS、FAIL、NOT VERIFIED。
- 更新 docs/interaction_representation.md：故事、计算流程、机制公式、输入输出、缺测支持边界和最小消融。
- 修改摘要：改了什么、为什么、哪些动作/训练行为保持不变、有哪些未验证项。
- 不自动修改 AGENTS.md/CLAUDE.md/项目 skills，不把单次架构写成长期规则；README 只在稳定入口确实变化时更新。

本任务书提交时仅完成了最新源码、配置、数据路径和恢复接口的静态核对，未实现新网络，未运行模型测试或训练，未声称性能收益。后续执行者不得将本文验收计划抄写为已经通过的结果。
