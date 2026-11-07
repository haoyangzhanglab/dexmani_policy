# Codex 工程任务书｜InterAct-Dex 结构化几何交互与触觉门控重构

> 项目：[`haoyangzhanglab/dexmani_policy`](https://github.com/haoyangzhanglab/dexmani_policy)  
> 工作对象：`interact-dex` 策略；**直接覆盖旧实现**，不兼容旧 checkpoint、旧实验产物。  
> 文档性质：可执行的代码重构规格与验收标准，不是已实施记录。  
> 优先级：**正确性 > 隔离性 > 简洁性 > 性能优化**。不得以删去必要校验换取代码短小。

## 0. 任务目标与不可变约束

将现有 InterAct-Dex 重构为一条明确、简洁的交互学习链路：

1. **HandGeometryEncoder**：由基坐标系腕部 XYZ+Rot6D、腕部坐标系下五指 XYZ 生成 6 个带独立身份嵌入的几何查询；**绝不接收 joint state**。
2. **Relative3DCrossAttention**：6 个查询（wrist + 5 fingertips）读取 Point Transformer 的 128 个场景 token；相对位置在 **robot base frame** 中计算；删除旧的重复局部 KNN/Surface MLP。
3. **TactileGatedFusion**：五指局部触觉编码 + 几何条件化的轻量全局触觉聚合 + 六个共享网络产生的独立标量 gate。腕部接收全局触觉残差，五指接收对应局部触觉残差。**不接收 joint state**。
4. **独立上下文**：复用已有 `StateEncoder(19, 192, (128,))` 产生一个 Joint State Token；用轻量 GlobalScenePool 产生一个 Global Scene Token；与 6 个 Interaction Token 并列。
5. **动作生成**：3 帧 ×（6 Interaction + 1 Joint + 1 Scene）= **24 个 192 维条件 token**，直接送现有 DiTX + Rectified Flow；删除 `HandHistoryEncoder`。
6. **未来接触力辅助任务**：从 3 帧、6 个 Interaction Token 预测 **8 个观测步后**五指 `5×3` 的**绝对接触力**；不使用动作前缀、不计算力增量、不使用力扩散或额外 Transformer；训练期辅助，推理期不调用。

固定默认配置：`n_obs_steps=3`、`horizon=16`、`n_action_steps=8`、`action_dim=19`、`num_points=1024`、`num_patches=128`、`token_dim=192`。`future_force_horizon=8`（默认等于 `n_action_steps`）。

**必须保留**公共策略名称 `policy_name: interact-dex`、Hydra 的 `InterActDex` 目标类、`Policy` 公共行为、19 维关节目标、`future_actions` 输出协议、RTC 支持、原有训练/评估/部署入口；可重命名 InterAct-Dex 内部私有类、文件、方法。**不得为了本任务改变 AdapTac、CordViP、DP3、ManiFlow、SAT、R3D、DQ-RISE、DexLatent 等策略的行为或公共接口。**

不得修改已有远端仓库内容，除非 Codex 在目标工作区中明确执行；不要自动 `git commit`、`git push` 或创建 PR。开始前查看 `git status`，尊重用户未提交的修改。若发现与本任务冲突的已有改动，先报告，不要覆盖。

---

## 1. 目标架构与数据契约

### 1.1 主路径

```text
Observation (B,3,...)
  ├─ point_cloud (XYZRGB, base frame) → PointTransformerEncoder → Scene Patches [BT,128,192]
  │                                                  ├─ GlobalScenePool → Scene Token [BT,1,192]
  │                                                  └─ Relative3DCrossAttention (K/V)
  ├─ eef_pose (base XYZ+rot6d), fingertip_xyz (base XYZ)
  │       → HandGeometryEncoder → 6 Body Queries [BT,6,192]
  │       → Relative3DCrossAttention (Q=6) → Geometry Interaction [BT,6,192]
  ├─ finger_tactile_force [BT,5,3] → zero-referenced tactile
  │       → TactileGatedFusion(Geometry, Force) → Interaction [BT,6,192]
  └─ joint_state (normalized, 19D) → existing StateEncoder → Joint Token [BT,1,192]

Per-frame concat: [6 Interaction, 1 Joint, 1 Scene] = [BT,8,192]
Reshape [B,3,8,192] → frame-major flatten [B,24,192]
  ├─ DiTX + RectifiedFlow → [B,16,19] normalized action positions
  │                         → Policy returns [B,14,19] future_actions → executor uses first 8
  └─ (training only) Interaction [B,3,6,192]
       → lightweight temporal pooling + shared FutureForceHead
       → [B,5,3] normalized absolute future force → masked SmoothL1
```

`BT = B * 3`（更一般地 `B * n_obs_steps`）。token 顺序不可变化：`[wrist, thumb, index, middle, ring, pinky, joint_state, global_scene]`。

### 1.2 唯一坐标与单位约定（重点）

| 张量 | 参考系 / 数值约定 |
|---|---|
| `point_cloud[..., :3]` | Robot base frame，单位米；保留 XYZRGB 的原始几何单位 |
| `eef_pose[..., :3]` | Base-frame wrist XYZ，米 |
| `eef_pose[..., 3:9]` | Base←Wrist 旋转矩阵的连续 6D 表示（按现有连续列排列） |
| `fingertip_xyz` | Base-frame five fingertip XYZ，米 |
| HandGeometryEncoder 的 fingertip MLP | Wrist-local XYZ，按固定几何尺度缩放 |
| Relative3DCrossAttention 位置 | Wrist + 5 fingertips + scene patch centers **均在 Base Frame** |
| `joint_state` | 现有 normalizer 输出，19 维 |
| `finger_tactile_force` → TactileGatedFusion | `N(F_raw) - N(0) = F_raw * scale`，零参考力 |
| `future_force` 的监督目标 | `N(F_raw_future)`，**完整 affine normalization** |

仓库旧注释使用 world frame；**不能将变量改名成 base 就视为坐标转换完成**。Codex 必须先核对 `dexmani_sim` 数据产生者或数据集元信息：若仿真 world 与 robot base 重合，写明并测试；若不重合，需在 InterAct-Dex 专属输入适配处应用同一个经过标定的 World→Base 变换，正确旋转 eef 姿态并平移/旋转点云和指尖，不改其他策略的公共读取路径。若缺少可靠标定，明确列为阻断项，不得猜测外参。

`finger_tactile_force` 的三维轴/力坐标系也必须在文档中明确：当前与未来使用同一传感器语义；不凭空将其解释为 Wrist-frame force，也不未经验证旋转未来力标签。物理零点校准与 normalizer 的 offset 是两回事。

---

## 2. HandGeometryEncoder（纯几何）

在 `agents/obs_encoder/proprio.py` 中将仅供 InterAct-Dex 使用的旧 `HandStateEncoder` 替换为 `HandGeometryEncoder`，**保留现有 `StateEncoder` 完全不变**。若纯 InterAct 的几何代码迁入独立模块可接受，但不得引起其他策略 import 变化。

接口建议：

```python
class HandGeometryEncoder(nn.Module):
    def forward(self, eef_pose, fingertip_xyz) -> dict:
        # input: [BT,9] and [BT,5,3]
        # return: {"tokens": [BT,6,192], "anchors_base": [BT,6,3]}
```

计算：

- 基座下腕部位置 `p_w = eef_pose[:, :3]`；用现有 `_rotation_6d_to_matrix` 恢复 `R_base_from_wrist`。
- 五指在 Wrist Frame 中位置，按**现有行向量约定**：
  `tip_wrist = (tip_base - wrist_base[:,None,:]) @ R_base_from_wrist`。
- Wrist 只经 `MLP(9→128→192)` 编码 `[p_w / wrist_scale, rot6d]`（`wrist_scale` 默认 0.25 m；仅缩放位置），五指共享 `MLP(3→128→192)` 编码 `tip_wrist / finger_scale`（默认 0.10 m）。
- **不读任何 joint state，不添加关节嵌入**。
- 使用 `body_identity = nn.Parameter([1,6,192])`，六个 slot 各自独立；`tokens = LayerNorm(cat([wrist, 5 tips]) + body_identity)`。
- `anchors_base = cat([wrist_xyz_base, fingertip_xyz_base], dim=1)`，与六个 token 严格对齐。
- 不引入 Fourier embedding、额外 kinematic Transformer 或额外姿态编码路径。输入校验和 6D 旋转数值稳定性使用原有逻辑。

**保留两种坐标的分工**：局部 Wrist Frame 用于指尖构型编码，Base Frame 用于跨身体锚点与场景的相对几何。不能因旋转得到局部指尖坐标，便将 Cross-Attention 的 anchors 也转换到 Wrist Frame。

---

## 3. Relative3DCrossAttention（六查询）

重写 InterAct-Dex 专属 `agents/obs_encoder/interaction/hand_interaction.py` 或将新类集中在一个 `interact_dex_modules.py`。建议优先使用已有 `hand_interaction.py`，移除旧的 `_local_surface` / `surface_mlp` / `geometry_projection` / `HandSceneFusion` / `HandHistoryEncoder`。

建议类：

```python
class Relative3DCrossAttention(nn.Module):
    def forward(self, body_tokens, scene_tokens, anchors_base, scene_xyz_base):
        # body_tokens: [BT,6,192]; scene_tokens: [BT,128,192]
        # anchors_base: [BT,6,3]; scene_xyz_base: [BT,128,3]
        # return: [BT,6,192]
```

设计：

- `d=192`、`heads=4`、`head_dim=48`；`Q` 从 6 个身体 token 来，`K/V` 从 128 个 Point Transformer token 来。
- **Base Frame 相对向量** `delta[BT,6,128,3] = scene_xyz[:,None] - anchors[:,:,None]`（以米为单位）。
- 轻量位置偏置 `bias = MLP(3→32→heads)(delta / 0.10)`，输出 `[BT,heads,6,128]`，每个 head 对三维方向和距离敏感。
- `score = QK^T / sqrt(48) + bias`，用 `torch.nn.functional.scaled_dot_product_attention` 实现；Q 和 K/V 归一化及线性映射用标准 Pre-LN。
- 输出 `LayerNorm(body_tokens + output_projection(context))`；无 Local Surface MLP，无重复 kNN，无 Cross-Attention 侧第二套 3D RoPE。Point Transformer 内原有 3D RoPE **继续保留**。
- 为非接触阶段的健壮性，可保留一个**零 Value 的 null attention slot**及每头一个可学习 null logit；仅一个 slot，不增加独立网络或距离阈值超参数。若实现 null slot，必须完整测试其 `K/V`、bias shape、dtype 和梯度；不支持则删去，不得写无效半实现。
- 相对位置偏置在 FP32 计算，并在 SDPA 入口转换到匹配的 attention dtype；避免混合精度溢出、无意义的全张量 `.float()` 链，以及在向量化算子中原地修改共享输入。

**不宣称 SE(3) 旋转等变。**此设计的相对几何在共同平移下不变，但在 base 坐标轴旋转后一般不保持不变；测试和论文表述必须一致。

---

## 4. TactileGatedFusion（简洁局部—全局融合，Wrist 也接受触觉）

**只接收 geometry `[BT,6,192]` 和 zero-referenced force `[BT,5,3]`；不接收 joint token、scene token、任何动作信息。** 不复用或修改共享 `XHandForceEncoder`；新建 InterAct-only 零参考触觉小网络，以保证物理零力不会产生伪造触觉残差。

采用低维潜表示 `d_t=64`，避免原讨论版本中过早扩展至 192 维引起的重复投影：

1. **Local tactile**：五指共享、所有 Linear `bias=False` 的 `3→64→64` MLP（中间 SiLU），得到 `t ∈ [BT,5,64]`。`t(0)=0`。
2. **Geometry-conditioned global tactile pooling**：只用五个**对应的 finger geometry tokens** 和五个 local tactile 特征计算 additive score：
   `score_i = w^T tanh(W_g LN(G_i) + W_t t_i)`，其中池化隐维建议 32；`α=softmax(score, finger_axis)`；`t_global = sum_i α_i t_i` `[BT,1,64]`。**不额外使用 finger identity**，因为六个几何 token 已有独立 `body_identity`。
3. **Aligned tactile**：`local_aligned = [0,t_1,...,t_5]`，`value_aligned=[t_global,t_1,...,t_5]`，`global_context=t_global` 广播到六个 slot。
4. **Shared per-slot scalar gate**：六个位置用**同一套** `MLP(192+64+64=320→64→1) + sigmoid`，输入 `[LN(G_i), local_aligned_i, global_context]`，产生 `[BT,6,1]`；无需独立 Wrist Gate / Finger Gate。
5. **Residual fusion**：`G_out = G + gate * Linear(64→192, bias=False)(value_aligned)`，不在残差相加**之后**强制做 LayerNorm，否则全零力时不能严格恢复输入几何。为稳定起步，gate 输出 bias 初始化为小负值（如 -1.5），残差投影权重**小但非零**（如 Xavier gain 0.1），不要把残差投影和门控所有可训练路径同时初始化为零。

Wrist 的 `value` 为五指全局触觉摘要，五指的 `value` 各自为局部触觉；全局触觉用于全部六个 gate 的条件。`forward(..., return_gates=False)` 正常只返回融合特征，诊断时可选择返回 `(fused, gates)`。

### 4.1 严格零力性质

现有 FieldNormalizer：`N(F_raw)=F_raw*scale + offset`。在 InterActDex 的**预处理边界**（不要改公共 `Policy._preprocess_obs`）先按原方法归一化，再对 tactile 唯一地做：

```python
pre = self._preprocess_obs(obs)  # 也 flatten B,T → BT
pre["finger_tactile_force"] = (
    pre["finger_tactile_force"]
    - self.normalizer["finger_tactile_force"].offset
)
```

从而模块输入是 `F_raw*scale`，物理零力严格为0。所有触觉 Value 路径 `bias=False`，且 `SiLU(0)=0`，因此当五指力均为零时，`t_local=t_global=residual=0`，**`fused == geometry`**（允许正常浮点容差）。门控和注意力分数可以非零，但不能凭空生成触觉 Value。不要对 tactile Value 加普通 LayerNorm 破坏零点/幅值语义。

注意：触觉归一化 offset 不是硬件传感器零偏标定；不要在预测标签中减去 offset。

---

## 5. Joint Token、Global Scene Token、条件序列

### 5.1 Joint Token：复用已有实现

**不要新增 JointStateEncoder。**直接导入现有 `dexmani_policy.agents.obs_encoder.proprio.StateEncoder`：

```python
self.joint_encoder = StateEncoder(state_dim=19, out_dim=192, hidden_dims=(128,))
joint_token = self.joint_encoder(obs["joint_state"]).unsqueeze(1)
```

`joint_state` 为由原 Policy normalizer 处理后的 `[BT,19]`，输出 `[BT,1,192]`。不向六个 Body Query 叠加 Joint 特征，不参与 TactileGatedFusion，不参与未来力预测头。不要为该 token 创建额外独立类型 MLP。Joint 和 Scene 两个全局 token 与六个 Interaction token **同级、并列输入 DiTX**。

### 5.2 Global Scene Token：简化 SceneTokenizer

移除旧的 12-query `SceneTokenizer` 路径，用 InterAct-only `GlobalScenePool`：

```text
scene_features [BT,128,192]
  → LayerNorm → Linear(192,1) → softmax over 128 patches
  → weighted sum [BT,1,192] → LayerNorm
```

这是整体场景上下文摘要，而不是额外的物体分割或第二个 Transformer。代码可放在 InterAct-only 文件。旧的 `pointcloud/scene_tokenizer.py` 若确认无任何其他有效引用或测试后，可以删除；**不要改动共享 `PointTransformerEncoder` 的实现**。

### 5.3 条件拼接

```python
frame_tokens = torch.cat([interaction_6, joint_1, scene_1], dim=1)  # [BT,8,192]
condition = frame_tokens.unflatten(0, (-1, n_obs_steps)).flatten(1, 2)
# default: [B,24,192]
```

保留 frame-major、slot 顺序；不要沿错误维度拼接。DiTX 原有每帧位置嵌入、跨 token 条件交叉注意力即可，不恢复 `HandHistoryEncoder`，不更改共享 `ditx.py`。背景：它通过动作 token 读取三帧上下文，并非独立的 observation temporal self-attention；不要在论文中误表述。

---

## 6. FutureForceHead：单时距、绝对力、训练专用

- 未来预测目标：`F_{q+h}`，默认 `h=8`；**不**传入示教动作前缀；**不**减去当前力；**不**使用动作生成网络隐藏状态。
- 唯一输入 `interaction[B,T,6,192]`，**不读取 joint token / global scene token**；辅助梯度需反馈到共享 HandGeometryEncoder、Point Transformer、Relative3DCrossAttention、TactileGatedFusion。
- 简洁时序汇聚：共享 `temporal_logits = Parameter([T])`，`softmax` 后对三个观测帧加权求和，得到 `[B,6,192]`。logits 初始全零（均匀平均），不用时序 Transformer、独立动作网络或泛化的可变长复杂接口。
- `finger = pooled[:,1:6]`、`body_context = pooled.mean(dim=1,keepdim=True)`；拼接 `[finger_i, body_context]` → 五指共享 `MLP(384→128→3)` → `[B,5,3]`。
- 头部最后一层使用**小非零权重**而非完全零初始化，以便辅助损失从第一步即能流向共享编码器；输出 bias 可以为零。
- Loss：`target=N(F_raw_{q+8})`，采用相同 `finger_tactile_force` **完整 affine normalizer**（同时使用 scale 和 offset），mask 后的 per-example SmoothL1，先均值 5×3，再有效样本均值：

```python
per_example = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean((-1,-2))
mask = batch["future_force_valid"].to(per_example.dtype)  # [B]
force_loss = (per_example * mask).sum() / mask.sum().clamp_min(1.0)
loss = action_loss + future_force_loss_weight * force_loss
```

- 如果无有效 future label，损失为有限零，不污染梯度。合法物理零力仍然有效。
- `future_force_loss_weight=0` 时**不创建辅助头、无需读取未来标签**，保持消融便利和训练效率。
- `future_force_head` 不参与 `_encode_obs`、`predict_action`、`_predict_from_condition`；训练期输出头可从模型 state_dict 保存，但推理永不调用。
- 辅助损失**不直接更新 DiTX、Joint State Encoder 和 Global Scene Pool**；动作损失仍更新所有相关动作条件分支。训练日志保留 `loss_future_force`、`loss_future_force_weighted`、`future_force_valid_fraction`，建议增加归一化空间 MAE。

### 6.1 时序索引必须一致

对 `n_obs_steps=3`，`action_start_index=2`，最新观测源行 `q=sample_source_rows(idx)[2]`。现有数据约定为 `O_q → a_q → O_{q+1}`，故连续执行 8 个动作后对应 `O_{q+8}`，标签取 `query+8`，不是 `query+7`，也不是动作窗口尾部。

现有 `Policy.future_steps=16-(3-1)=14`，predict_action 返回 `[B,14,19]` 的 future_actions；SimRunner 执行前8个。`h=8` 不超过可预测的未来动作范围。不要改变共用的动作窗口语义。

---

## 7. 数据集和配置修正（只动 InterAct 分支）

### 7.1 InterActDexDataset

只修改 `datasets/variants/tactile.py` 中 `InterActDexDataset`：

- 单一字段 `future_force_horizon: int`，替代旧 `future_force_horizons`。
- 复用未改动的 `read_future_force(store, query, (h,), field="finger_tactile_force")`；取 singleton `[0]` 并生成 `future_force` `[5,3]`、`future_force_valid` 标量 bool。
- episode 末尾若 `q+h` 不存在，读取末行作为有限填充值并置 mask=false，不跨 episode；当前零接触力不会被错误标为 invalid。
- **不要修改** `AdapTacDataset`、`DemonstrationDataset`、`SequenceSampler`、`read_future_force()` 的共享行为。

### 7.2 `training/common/config.py`

只修改 `if method == "interact-dex":` 的专属分支：

- 从单一 `cfg.future_force_horizon` 同步到 `cfg.agent.future_force_horizon`；配置值必须为正整数、默认8、且在策略支持范围内。
- 当 `future_force_loss_weight > 0`：数据集目标 `InterActDexDataset`，并注入 `dataset.future_force_horizon`。
- 当 `future_force_loss_weight == 0`：使用通用 `DemonstrationDataset`，不注入未来力参数；策略无未来力头。
- 原有物理字段 `point_cloud`、`eef_pose`、`fingertip_xyz` 的 `identity` 归一化检查必须保留；不要改变其他 policy_name 分支。

### 7.3 `configs/interact_dex.yaml`

基础：

```yaml
policy_name: interact-dex
horizon: 16
n_obs_steps: 3
n_action_steps: 8
future_force_horizon: ${n_action_steps}
normalization:
  point_cloud: identity
  eef_pose: identity
  fingertip_xyz: identity
  joint_state: limits
  finger_tactile_force: limits
dataset:
  sensor_modalities:
    - joint_state
    - point_cloud
    - eef_pose
    - fingertip_xyz
    - finger_tactile_force
agent:
  _target_: dexmani_policy.agents.policies.interact_dex.InterActDex
  future_force_loss_weight: 0.1
  # agent.future_force_horizon 由 assemble_policy_config 的本策略分支注入
  pc_dim: 6
  num_points: 1024
```

原配置中 `PointTransformerEncoder`（channels192、128 patches、K32、depth2、heads8）和 `DiTX`（layers8、hidden768、heads8、horizon16、10 inference steps）的训练超参先保持不变，保留 `fps_random_config`、augmentation、optimizer、EMA、eval 设置。**删除旧的 `tactile: force` defaults 继承**（否则 Hydra 可能继续向 agent 注入已删除的 `tactile_encoder`）；自行保留 `finger_tactile_force` 模态/归一化。删除所有旧的 `future_force_mode`、`future_force_time_weighting`、`future_force_horizons`、`scene_num_queries`、`tactile_encoder`、HandHistory 相关配置。全新 checkpoint 仅支持当前结构。

---

## 8. 推荐文件结构与修改边界

| 路径 | 操作 | 说明 |
|---|---|---|
| `agents/obs_encoder/proprio.py` | 限定范围修改 | 复用 `StateEncoder` 不动，删除/重命名旧 `HandStateEncoder` 为 `HandGeometryEncoder`；保留需要的 6D 旋转工具 |
| `agents/obs_encoder/interaction/hand_interaction.py` | 重写 | 只放 InterAct-only `Relative3DCrossAttention`、`TactileGatedFusion`、`GlobalScenePool`，可根据可维护性拆成专属小文件 |
| `agents/policies/interact_dex.py` | 直接覆盖 | 新 Encoder、24-token 组装、新 FutureForceHead、联合损失、optimizer groups、normalizer 预处理 |
| `datasets/variants/tactile.py` | 精确修改 | 只改 `InterActDexDataset` |
| `training/common/config.py` | 精确修改 | 只改 `policy_name=interact-dex` 分支，保持其它 config 路由 |
| `configs/interact_dex.yaml` | 重写旧配方 | 保留训练基础设施设置，换新架构配置 |
| `agents/obs_encoder/pointcloud/scene_tokenizer.py` | 可删除 | 只有确认无其他调用后才清理旧实现；不修改共享 Point Transformer |
| `test_tools/point_transformer_checks.py` | 精确修改 | 更新/迁出仅属 InterAct 的旧测试，其他检查不变 |
| `test_tools/interact_dex_checks.py` | 新增 | 独立纯单元、策略、数据边界及端到端 smoke 测试 |
| `docs/触觉与交互建模.md` 等 | 更新 | 明确新结构、坐标系、训练与推理差异、数据可用性 |

**禁止修改**共享 `StateEncoder`、`XHandForceEncoder`、`PointTransformerEncoder`、`RotaryPositionEmbedding3D`、`DiTX`、`RectifiedFlow`、`Policy`、`LinearNormalizer`、`SequenceSampler`、`SimRunner`、`AdapTacDataset` 以及其他策略实现。若发现确需公共层修改，先论证不可替代性并将改动缩至局部，优先在 InterAct-only 层解决。

---

## 9. 验收测试：必须可证伪，而不是只看 shape

建立独立 `python -m dexmani_policy.test_tools.interact_dex_checks --device cpu`（可增 GPU 参数）测试。若 PyTorch3D 无法在 CPU 环境运行，至少应能单独测试非点云模块；如依赖缺失则明确报告阻塞项，不能宣称已通过。

### A. 几何、坐标与注意力

- `HandGeometryEncoder`: 输入 `[BT,9]`,`[BT,5,3]` → `[BT,6,192]`，anchors `[BT,6,3]`，固定 slot 顺序；六个身份向量不同。
- Joint 信息与手部几何解耦：修改 `joint_state` 不改变 `HandGeometryEncoder` 输出；改变 finger XYZ 影响对应的 fingertip 几何编码。
- 构造非平凡旋转和平移，验证 `(tip_base-wrist_base) @ R_base_from_wrist` 与手工预期一致。
- 在 base 下同时平移 6 anchors 与全部 scene patch centers，相对位置偏置保持不变（**仅测 relative bias，不要求含绝对 wrist XYZ 的整个策略不变**）。
- 相对向量的方向反转通常改变 bias；不得把 3D 相对向量误退化为纯距离平方。
- 将 `scene_tokens` 与 `scene_xyz` 同步随机重排，Relative3DCrossAttention 输出一致（评估模式且考虑 FP 浮点容差）。
- 六个 Query 均参与，且该模块可反传到 Q/K/V 和 bias MLP；无 NaN/Inf。

### B. 触觉融合

- `[BT,6,192] + [BT,5,3] → [BT,6,192]`，诊断 gate `[BT,6,1]` 且在 `[0,1]` 内。
- 物理零力经 affine offset 校正后，`TactileGatedFusion(G,0) == G`，即使 normalizer offset 非零也成立。
- 对单根手指施加非零接触力，Wrist 输出必须能够响应，相关手指输出必须改变；**其他手指 gate 可以通过 global tactile 间接变化，不要错误断言跨手指完全独立**。
- 修改对应手指几何特征可影响 global tactile pooling 权重，验证 geometry-conditioned 而非单纯 tactile-only。
- 残差、pool、gate、force encoder 梯度均有效；前若干训练更新后仍无 NaN/Inf；全零力情况不给 Value 引入非零偏置。

### C. 时序、动作和未来力

- 默认条件 `[B,3,8,192] → [B,24,192]`，slot/frame 顺序精确。
- `StateEncoder(19,192,(128,))` 输出 `[BT,1,192]`；无新 `JointStateEncoder` 类。
- `GlobalScenePool` 输出 `[BT,1,192]`，权重跨128 patches 归一。
- `Policy.action_start_index=2`、`future_steps=14`；`predict_action()` 返回 `[B,14,19]`，执行端选择前8步。
- `FutureForceHead`: `[B,3,6,192]→[B,5,3]`；只读 Interaction，不接收示教动作/Joint/Scene。
- Dataset 明确验证 `query+8`（而非 +7），以及长/短 episode、末尾填充、全部 invalid、有效零力、不同 episode 隔离。
- 绝对力目标经过完整 affine 归一化，验证非零 offset 时预测损失与手工值一致；不发生 force delta 或 action-prefix 计算。
- `future_force_loss_weight>0` 时，辅助损失可反传到几何、cross-attn、tactile gate、head；不会直接更新 Joint State Encoder、Global Scene Pool、DiTX。单独对 `force_loss` 反传确认；**不要用混合了 action loss 的总梯度推断隔离性**。
- `future_force_loss_weight=0` 时正确走通普通 DemonstrationDataset，不创建 head，不读取 future force。
- 推理可将预测头临时替换为“调用即抛异常”的桩，验证完全不被调用。

### D. Hydra、checkpoint、其他策略回归

- `hydra` 配方可解析、实例化、保存有效配置；新策略 checkpoint 可以 `strict=True` 重建并恢复正常动作输出；不要求旧权重可加载。
- `optimizer_groups` 覆盖所有可训练参数，且无重复参数；训练时 auxiliary head 确实更新。
- 在现有基础设施可运行时执行：

```bash
python -m dexmani_policy.test_tools.interact_dex_checks --device cpu
python -m dexmani_policy.test_tools.point_transformer_checks --device cpu
python -m dexmani_policy.test_tools.core_checks --device cpu
```

可补充 CUDA/BF16 测试（环境支持时），并运行仓库其它既有策略检查。不要修改其他策略测试来“放过”失败；若依赖缺失，记录具体命令、错误和未验证范围。

---

## 10. 代码质量、性能与清理标准

- Python >=3.10，遵循仓库现有 PyTorch/Hydra/类型注解/模块风格。优先低维、可组合小模块；不创建冗余 abstraction / 注册机制 / 一次性通用框架。
- 前向传播尽可能一次 batch-flatten，再在 `B,T` 边界恢复；不重复运行 Point Transformer；无每手指 Python for-loop 处理注意力或 MLP。
- 保留 FP32 坐标和几何偏置计算，SDPA/mixed precision 的 dtype 与 device 正确；避免对包含状态的输入执行原地修改。
- 无硬编码到代码里的固定 batch size；`n_obs_steps` 可配置，默认3；对非预期输入维度作清晰错误检查。
- 默认 `tactile_input=zero-referenced normalized force`，直接在接口文档注明，不能同时重复减 offset。
- 训练用绝对力 `N(F_future)`；接触力物理坐标轴约定必须写入文档。
- 不额外生成 Action Prefix 模式、Delta 模式、Time Weighting、复杂 Future World Model、额外场景 Transformer 或历史 Transformer。
- 避免向 `DiTX` 再塞全局触觉第九个 token。全局触觉只参与六个身体 token 的内部融合。
- 研究用消融应通过 `future_force_loss_weight=0`、触觉输入置零、替换轻量融合等明确实验实现；不要为所有猜想建立大量长期维护的 config 开关。
- 清理被替换类、函数及其残余 import；公开的 `InterActDex` 和策略 YAML 路径保持可发现。

---

## 11. 建议 Codex 实施顺序

1. `git status`、确认 main/当前工作分支、列出旧 InterAct 专属符号及引用、核对 Base-frame / force 物理契约。
2. 重构 `HandGeometryEncoder`（六 identity，纯几何），同时独立保留 `StateEncoder`。
3. 替换六查询 `Relative3DCrossAttention`，删除冗余局部几何分支。
4. 实现 64 维 `TactileGatedFusion` + 1 token `GlobalScenePool`。
5. 重写 `InterActDexEncoder`，输出 3 帧 × 8 token；删除 HandHistoryEncoder。
6. 单时距 Dataset + FutureForceHead + 完整 affine 绝对力损失 + 梯度隔离 + λ=0 路径。
7. 修改只属于 InterAct 的 Hydra 配方、配置路由、可恢复 effective_config，更新文档和专属测试。
8. 运行新增测试和其他策略回归；输出变更摘要、测试结果、未验证项目、性能/参数量变化（**如未实测，不给出加速比例或性能提高结论**）。

---

## 12. 最终交付格式与完成定义

Codex 完成后应返回：

1. **实现摘要**：新增/删除/修改的类与主要模块依赖，解释保留 Joint / Scene 独立 token 的原因。
2. **变更文件表**：每个文件的实际改动和是否触及公共代码。
3. **输入与张量验证**：`[BT,6,192]`、`[BT,1,192]`、`[BT,24,192]`、`[B,5,3]`、`[B,14,19]` 等协议检查结果。
4. **测试记录**：每条命令的通过/失败/阻塞状态，含依赖与 GPU 环境限制。
5. **回归影响**：其他策略未改动及现有回归测试结果。
6. **待确认项**：World/Base 坐标关系、力传感器坐标轴定义、真实机器人输入可用性（若仓库证据不足）。

**只有满足以下条件才算完成**：新 InterAct-Dex 能从新配置实例化、训练损失前后向、读取合法未来力标签、预测14个未来动作并执行前8步、保存/恢复**新版本** checkpoint；不在推理期调用辅助头；并且没有引入其他策略的行为回归。

> 研究边界：未来力头是 **observation-conditioned future absolute contact force forecasting**，不是动作条件的显式接触动力学。相对三维几何编码不宣称旋转等变。仓库的场景点云如果没有目标实例分割，论文准确使用 **hand–scene interaction**，不要将其包装成带有物体级真值信息的 hand–object attention。