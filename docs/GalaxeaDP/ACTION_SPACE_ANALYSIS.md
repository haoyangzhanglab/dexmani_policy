# GalaxeaDP 动作空间与动作分块设计分析

## 一、动作空间设计

### 1.1 动作空间的组成

GalaxeaDP 的动作空间由 **末端执行器位姿（EE Pose）+ 夹爪（Gripper）** 构成，通过 `action_keys` 配置定义：

```yaml
# configs/data/r1pro/lerobot_eef.yaml
action_keys: 
  - left_ee_pose      # 左臂末端位姿
  - left_gripper      # 左臂夹爪
  - right_ee_pose     # 右臂末端位姿
  - right_gripper     # 右臂夹爪
```

对应的 `qpos_keys`（观测中的关节状态）与 `action_keys` 相同，意味着观测中的状态与动作空间一一对应：

```yaml
qpos_keys: ${data.train.action_keys}  # 继承 action_keys
```

### 1.2 动作维度拆解

| Action Key | 原始维度 | 转换后维度 | 说明 |
|------------|---------|-----------|------|
| `left_ee_pose` | 7 (position + quaternion) | `pose_dim` | 取决于 `rotation_type` |
| `left_gripper` | 1 | 1 | 夹爪开合度 |
| `right_ee_pose` | 7 | `pose_dim` | 同上 |
| `right_gripper` | 1 | 1 | 同上 |
| **总计** | **16** | **`2 * pose_dim + 2`** | 取决于旋转表示 |

### 1.3 旋转表示转换（Rotation Representation）

GalaxeaDP 通过 `SignalTransform` 中的 `PoseRotationTransformer` 支持三种旋转表示：

```python
# src/utils/rotation_conversions.py
class PoseRotationTransformer:
    def __init__(self, rotation_type):
        if rotation_type == "quaternion":
            self.pose_dim = 7      # 3 position + 4 quaternion
        elif rotation_type == "rotation_6d":
            self.pose_dim = 9      # 3 position + 6 rotation
        elif rotation_type == "rotation_9d":
            self.pose_dim = 12     # 3 position + 9 rotation
```

**转换流程**（`forward` 方向，训练时）：

```
原始 pose (position + quaternion) 
    │
    ├─► quaternion → rotation_matrix (3x3)
    ├─► rotation_matrix → rotation_6d (前2列) 或 rotation_9d (转置展平)
    └─► concat(position, rotation_6d/9d)
```

**反向流程**（`backward` 方向，推理时）：

```
网络输出 (position + rotation_6d/9d)
    │
    ├─► rotation_6d/9d → rotation_matrix (Gram-Schmidt / SVD)
    ├─► rotation_matrix → quaternion
    └─► concat(position, quaternion) → 原始格式
```

**三种表示的对比**：

| 表示 | 维度 | 优势 | 劣势 |
|------|------|------|------|
| `quaternion` | 4 | 紧凑、无奇异点 | 单位约束、双覆盖 (q 和 -q 等价) |
| `rotation_6d` | 6 | 连续无约束（Zhou et al. 2019） | 维度增加 50% |
| `rotation_9d` | 9 | 矩阵转置，完全无约束 | 维度翻倍，冗余最大 |

**维度计算示例**（默认 `rotation_type=quaternion`）：

```
action_dim = left_ee_pose(7) + left_gripper(1) + right_ee_pose(7) + right_gripper(1) = 16
```

如果使用 `rotation_6d`：

```
action_dim = left_ee_pose(9) + left_gripper(1) + right_ee_pose(9) + right_gripper(1) = 20
```

### 1.4 信号转换层（SignalTransform）的完整数据流

`SignalTransform` 是 GalaxeaDP 动作空间设计的**核心组件**，负责两件事：
1. **旋转表示转换**（quaternion ↔ 6D/9D）
2. **相对位姿控制**（relative control）

#### 训练时的 forward 流程

```python
# src/models/policy/signal_transform/signal_transform.py
def forward(self, batch):
    # === 处理 Action ===
    for category in ["action", "prev_action"]:
        if category in batch:
            actions = []
            for key in self.action_dims:
                cur_action = batch[category][key]
                if "ee_pose" in key:
                    # 1. 相对控制：动作相对于当前位姿
                    if self.use_relative_control:
                        obs_key = key.replace("action", "obs")
                        base_pose = batch["obs"][obs_key][:, -1:, :]  # 当前位姿
                        cur_action = self.relative_pose_transformer.forward(cur_action, base_pose)
                    # 2. 旋转表示转换
                    cur_action = self.pose_rotation_transformer.forward(cur_action)
                actions.append(cur_action)
            # 拼接所有 action：[ee_left, gripper_left, ee_right, gripper_right]
            batch[category] = torch.cat(actions, dim=-1)
    
    # === 处理 Qpos（观测状态）===
    qposes = []
    for key in self.qpos_keys:
        cur_qpos = batch["obs"][key]
        if "ee_pose" in key:
            if self.use_relative_control:
                # 1. 相对 episode 起点的位姿
                obs_key = f"episode_start_{key}"
                base_pose = batch["obs"][obs_key]
                if self.training:
                    base_pose = self.pose_rotation_transformer.add_noise(base_pose)  # 数据增强
                ee_pose_wrt_start = self.relative_pose_transformer.forward(cur_qpos, base_pose)
                ee_pose_wrt_start = self.pose_rotation_transformer.forward(ee_pose_wrt_start)
                qposes.append(ee_pose_wrt_start)

                # 2. 相对于当前位姿的位姿
                base_pose = cur_qpos[:, -1:, :]
                cur_qpos = self.relative_pose_transformer.forward(cur_qpos, base_pose)
            cur_qpos = self.pose_rotation_transformer.forward(cur_qpos)
        qposes.append(cur_qpos)
    # 拼接：[ee_left, ee_left_wrt_start, gripper_left, ee_right, ee_right_wrt_start, gripper_right]
    batch["obs"]["qpos"] = torch.cat(qposes, dim=-1)
```

#### 推理时的 backward 流程

```python
def backward(self, batch):
    actions = {}
    idx = 0
    for key, dim in self.action_dims.items():
        cur_action = batch["action"][:, :, idx: idx + dim]
        idx += dim
        if "ee_pose" in key:
            # 1. 旋转表示反转换
            cur_action = self.pose_rotation_transformer.backward(cur_action)
            # 2. 相对控制反转换
            if self.use_relative_control:
                obs_key = key.replace("action", "obs")
                base_pose = batch["obs"][obs_key][:, -1:, :]
                cur_action = self.relative_pose_transformer.backward(cur_action, base_pose)
            actions[key] = cur_action
        else:
            actions[key] = cur_action
    batch["action"] = actions  # 恢复为 dict 格式
```

#### 数据流全景图

```
原始数据 (LeRobot)                    SignalTransform.forward              网络输入
┌─────────────────────┐              ┌──────────────────────┐            ┌─────────────────────┐
│ action:             │              │                      │            │ action:             │
│   left_ee_pose: 7   │─────────────►│ relative + rotation  │───────────►│   拼接向量 (B, T, D)│
│   left_gripper: 1   │              │ transform            │            │   D = 2*pose_dim+2  │
│   right_ee_pose: 7  │              │                      │            │                     │
│   right_gripper: 1  │              │                      │            │ qpos: 拼接向量       │
│                     │              │                      │            │   (含 relative 增强) │
│ obs:                │              │                      │            │                     │
│   ee_pose_obs: 7    │─────────────►│ relative + rotation  │───────────►│                     │
│   gripper: 1        │              │ transform            │            │                     │
│   episode_start: 7  │              │ (+ noise in train)   │            │                     │
└─────────────────────┘              └──────────────────────┘            └─────────────────────┘

网络输出 (action_pred)               SignalTransform.backward             可执行动作
┌─────────────────────┐              ┌──────────────────────┐            ┌─────────────────────┐
│   拼接向量 (B, T, D) │─────────────►│ rotation inverse +   │───────────►│ left_ee_pose: 7     │
│   D = 2*pose_dim+2  │              │ relative inverse     │            │ left_gripper: 1     │
│                     │              │                      │            │ right_ee_pose: 7    │
│                     │              │                      │            │ right_gripper: 1    │
└─────────────────────┘              └──────────────────────┘            └─────────────────────┘
```

### 1.5 相对位姿控制（Relative Control）

这是 GalaxeaDP 动作空间设计中**最具特色的部分**。

#### 两种相对模式

| 模式 | 参考基准 | 用途 | 代码位置 |
|------|---------|------|---------|
| **Action 相对当前位姿** | `batch["obs"][obs_key][:, -1:, :]` | 动作表示相对于当前末端位姿的增量 | `forward()` 中 action 处理 |
| **Qpos 相对 episode 起点** | `batch["obs"]["episode_start_{key}"]` | 观测状态表示相对于 episode 起始位姿的偏移 | `forward()` 中 qpos 处理 |

#### 相对变换的数学原理

```python
# src/utils/relative_pose.py
def absolute_to_relative(pose_matrix, base_pose_matrix):
    """T_relative = T_base^{-1} @ T_absolute"""
    return torch.linalg.inv(base_pose_matrix) @ pose_matrix

def relative_to_absolute(pose_matrix, base_pose_matrix):
    """T_absolute = T_base @ T_relative"""
    return base_pose_matrix @ pose_matrix
```

**示例**：
```
绝对位姿：T_abs = [R | t; 0 | 1]  (4x4 齐次变换矩阵)
基准位姿：T_base = [R_b | t_b; 0 | 1]

相对位姿：T_rel = T_base^{-1} @ T_abs
         = [R_b^T | -R_b^T @ t_b; 0 | 1] @ [R | t; 0 | 1]
         = [R_b^T @ R | R_b^T @ (t - t_b); 0 | 1]
```

**为什么需要相对控制？**
- **位置不变性**：任务在不同起始位姿下执行时，相对动作模式保持一致
- **泛化能力**：模型学习的是"相对运动模式"而非"绝对坐标"
- **数据增强**：训练时对 episode 起点位姿加噪声（`add_noise`），进一步提升鲁棒性

#### 相对控制的完整流程

```
训练时：
  1. 采集数据：绝对位姿序列 [T_0, T_1, ..., T_n]
  2. 计算相对动作：ΔT_i = T_{i-1}^{-1} @ T_i  （相对于前一帧）
  3. 转换旋转表示：quaternion → 6D/9D
  4. 归一化 → 网络训练

  同时：
  5. 计算相对观测：ΔT_obs = T_start^{-1} @ T_current  （相对于起点）
  6. 对 T_start 加噪声 → 数据增强
  7. 转换旋转表示 → 拼接为 qpos

推理时：
  1. 网络输出相对动作 ΔT_pred
  2. 反转换旋转表示：6D/9D → quaternion
  3. 恢复绝对动作：T_abs = T_current @ ΔT_pred
  4. 发送给机器人执行
```

---

## 二、动作分块（Action Chunking）设计

### 2.1 动作分块的核心参数

```yaml
# configs/data/r1pro/lerobot_eef.yaml
chunk_size: 32              # 预测未来 32 步动作
vision_obs_size: 1          # 使用 1 帧视觉观测
qpos_obs_size: 1            # 使用 1 帧关节状态观测
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_size` | 32 | 网络一次性预测的动作序列长度 |
| `vision_obs_size` | 1 | 用于条件输入的视觉历史帧数 |
| `qpos_obs_size` | 1 | 用于条件输入的关节状态历史帧数 |
| `horizon` | 32 | Diffusion 模型的预测窗口（继承 `chunk_size`）|

### 2.2 动作分块的数据构建

在 `GalaxeaLerobotDataset` 中，通过 LeRobot 的 `delta_timestamps` 机制实现时序窗口采样：

```python
# src/data/galaxea_lerobot_dataset.py
fps = meta.fps  # 数据集帧率

# 观测：向后看 vision_obs_size 帧
delta_timestamps = {
    f"observation.images.{key}": [t / fps for t in range(0, -vision_obs_size, -1)]
    for key in self.image_keys
}
# 示例：vision_obs_size=1 → {"observation.images.head_rgb": [0.0]}

# 关节状态观测
for key in qpos_keys:
    delta_timestamps[f"observation.state.{key}"] = [t / fps for t in range(0, -qpos_obs_size, -1)]

# 动作：向前看 chunk_size 帧
for key in action_keys:
    delta_timestamps[f"action.{key}"] = [t / fps for t in range(0, chunk_size, 1)]
# 示例：chunk_size=32, fps=10 → [0.0, 0.1, 0.2, ..., 3.1] 秒
```

**数据样本的时序结构**：

```
当前时刻 t=0
│
├─► 观测 (obs):
│   ├─► head_rgb: [t=0]           (1 帧)
│   ├─► left_wrist_rgb: [t=0]     (1 帧)
│   ├─► right_wrist_rgb: [t=0]    (1 帧)
│   ├─► left_ee_pose: [t=0]       (1 帧)
│   └─► ...
│
└─► 动作 (action):
    ├─► left_ee_pose: [t=0, t=1, ..., t=31]    (32 帧)
    ├─► left_gripper: [t=0, t=1, ..., t=31]    (32 帧)
    ├─► right_ee_pose: [t=0, t=1, ..., t=31]   (32 帧)
    └─► right_gripper: [t=0, t=1, ..., t=31]   (32 帧)
```

### 2.3 推理时的动作分块执行

在 `DiffusionUnetImagePolicy.predict_action()` 中：

```python
# src/models/policy/diffusion_unet_image_policy.py
def predict_action(self, batch):
    batch = self.signal_transform.forward(batch)    # 旋转转换 + 相对控制
    nobs = self.normalizer.normalize(batch["obs"])  # 归一化
    
    # 1. 编码观测 → 全局条件向量
    nobs_features = self.obs_encoder(nobs)
    global_cond = nobs_features.reshape(batch_size, -1)
    
    # 2. 从噪声开始扩散采样
    trajectory = torch.randn(
        size=(batch_size, self.horizon, self.action_dim),  # (B, 32, 16)
        device=global_cond.device
    )
    nsample = self.conditional_sample(trajectory, global_cond=global_cond)
    
    # 3. 反归一化 + 截取有效动作
    action_pred = self.normalizer["action"].unnormalize(nsample[..., :self.action_dim])
    start = self.n_vision_obs_steps - 1  # = 0 (当 n_vision_obs_steps=1)
    batch["action"] = action_pred[:, start:]  # (B, 32, 16)
    
    # 4. 反转换：旋转表示 + 相对控制
    batch = self.signal_transform.backward(batch)
    # 恢复为 dict 格式并拼接
    batch["action"] = torch.cat([batch["action"][key] for key in batch["action"]], dim=-1)
    return batch["action"]
```

**关键行为**：

| 步骤 | 输入 Shape | 输出 Shape | 说明 |
|------|-----------|-----------|------|
| 扩散采样 | (B, 32, 16) | (B, 32, 16) | 从噪声去噪得到 32 步动作 |
| 截取 | `action_pred[:, start:]` | (B, 32, 16) | `start = n_vision_obs_steps - 1 = 0` |
| 反转换 | (B, 32, 16) | dict of (B, 32, dim) | 恢复为各 action key |

### 2.4 动作分块的执行策略

GalaxeaDP 采用 **完整 Chunk 输出** 策略，与 DexMani Policy 的 Action Chunking 机制不同：

```
GalaxeaDP 策略：
┌─────────────────────────────────────────────────┐
│  观测 (t=0)  ──► 网络 ──► 32步动作 [t=0...t=31]  │
│                                              │
│  观测 (t=1)  ──► 网络 ──► 32步动作 [t=1...t=32]  │  ← 每步重新推理
│                                              │
│  观测 (t=2)  ──► 网络 ──► 32步动作 [t=2...t=33]  │
└─────────────────────────────────────────────────┘

DexMani Policy 策略（对比）：
┌─────────────────────────────────────────────────┐
│  观测 (t-1, t)  ──► 网络 ──► 16步动作 [t-1...t+14]│
│                       │                          │
│                       ├─► 执行 [t+1...t+8] (8步)   │  ← 执行部分后重新推理
│                       │                          │
│                    重新推理...                     │
└─────────────────────────────────────────────────┘
```

**GalaxeaDP 的特点**：
- `n_vision_obs_steps=1`：只用当前帧观测
- 每次推理输出完整的 `chunk_size=32` 步动作
- 实际执行时**每步都重新推理**（而非执行多步再推理）
- 这确保了每一步都利用了最新的观测信息

### 2.5 训练时的扩散目标

```python
# compute_loss() 中的扩散过程
def compute_loss(self, batch):
    batch = self.signal_transform.forward(batch)
    nobs = self.normalizer.normalize(batch["obs"])
    nactions = self.normalizer["action"].normalize(batch["action"])  # (B, 32, action_dim)
    
    # 编码观测
    nobs_features = self.obs_encoder(nobs)
    global_cond = nobs_features.reshape(batch_size, -1)
    
    # 加噪
    trajectory = nactions  # 干净的 32 步动作序列
    noise = torch.randn(trajectory.shape)
    timesteps = torch.randint(0, num_train_timesteps, (bsz,))
    noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
    
    # 预测噪声
    pred = self.model(sample=noisy_trajectory, timestep=timesteps, global_cond=global_cond)
    
    # MSE 损失（根据 prediction_type 选择目标）
    if prediction_type == 'epsilon':
        target = noise        # 预测噪声
    elif prediction_type == 'sample':
        target = trajectory   # 预测干净样本
    elif prediction_type == 'v_prediction':
        target = v_t          # 预测速度
```

**训练与推理的对称性**：

```
训练: 干净动作 → 加噪 → 预测噪声/样本 → MSE 损失
推理: 纯噪声   → 去噪 → 预测干净动作   → 执行
```

---

## 三、设计亮点总结

### 3.1 动作空间设计的核心优势

| 设计 | 优势 |
|------|------|
| **模块化 action_keys** | 通过配置定义，支持不同机器人（左/右臂、夹爪/关节） |
| **旋转表示可切换** | quaternion/6D/9D 自由切换，平衡精度与效率 |
| **相对位姿控制** | 提升跨起始位姿的泛化能力，训练时加噪声增强鲁棒性 |
| **forward/backward 对称** | 训练时转换、推理时反转换，保证输出可直接执行 |
| **归一化独立** | action 和 qpos 分别归一化，避免量纲差异影响训练 |

### 3.2 动作分块设计的核心优势

| 设计 | 优势 |
|------|------|
| **delta_timestamps** | 利用 LeRobot 原生时序窗口，代码简洁 |
| **完整 Chunk 输出** | 每次推理输出 32 步，保证时序一致性 |
| **每步重新推理** | 利用最新观测，避免开环执行累积误差 |
| **horizon = chunk_size** | 训练/推理窗口一致，无分布偏移 |

### 3.3 与 DexMani Policy 的对比

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|---------------|
| **动作空间** | EE Pose + Gripper（末端执行器） | Joint State（关节空间） |
| **维度** | 16（quaternion 模式） | 19 |
| **旋转表示** | 支持 quaternion/6D/9D | 无（直接关节控制） |
| **相对控制** | 支持（相对当前位姿/起点） | 无 |
| **分块策略** | 完整输出 32 步，每步重新推理 | 输出 16 步，执行 8 步后重新推理 |
| **观测历史** | 1 帧 | 2 帧 |
| **分块截取** | 不截取（`start=0`） | 截取 `n_action_steps=8` 步 |

### 3.4 数据流全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GalaxeaDP 动作空间完整数据流                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LeRobot Dataset                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ __getitem__(idx)                                                │   │
│  │   action: {left_ee_pose: (32,7), left_gripper: (32,1), ...}     │   │
│  │   obs: {head_rgb: (1,3,H,W), left_ee_pose_obs: (1,7), ...}      │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                          │
│                             ▼                                          │
│  SignalTransform.forward                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. relative control: action ← action ⊖ current_pose             │   │
│  │ 2. rotation convert: quaternion → 6D/9D                         │   │
│  │ 3. concatenate: [ee_left(9), gripper_left(1), ee_right(9), ...]  │   │
│  │ 4. qpos: same process + episode_start_relative (+ noise)         │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                          │
│                             ▼                                          │
│  Normalization                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ action: min-max → [-1, 1]    qpos: min-max → [-1, 1]            │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                          │
│                             ▼                                          │
│  Diffusion Training                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ obs_encoder(obs) → global_cond (B, feature_dim)                  │   │
│  │ add_noise(action) → noisy_action (B, 32, action_dim)             │   │
│  │ UNet(noisy_action, global_cond) → pred                           │   │
│  │ MSE(pred, target)                                                 │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                          │
│                             ▼                                          │
│  Inference                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ randn(B, 32, action_dim) → DDIM/DDPM → clean_action (B, 32, D)  │   │
│  │ unnormalize(action)                                               │   │
│  │ SignalTransform.backward()                                        │   │
│  │   rotation inverse: 6D/9D → quaternion                           │   │
│  │   relative inverse: relative → absolute                           │   │
│  │ split: {left_ee_pose: (32,7), left_gripper: (32,1), ...}         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
