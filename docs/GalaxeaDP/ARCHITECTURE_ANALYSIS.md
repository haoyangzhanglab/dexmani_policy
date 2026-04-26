# GalaxeaDP 架构深度分析

## 项目概览

**GalaxeaDP** 是 Galaxea R1 系列机器人（R1、R1 Pro、R1 Lite）的**扩散策略（Diffusion Policy）**开源实现。项目基于 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[LeRobot](https://github.com/huggingface/lerobot) 和 [PyTorch Lightning](https://lightning.ai/) 构建，提供从数据生成、训练、开环评估到仿真闭环评估的完整端到端流水线。

---

## 一、模型架构

### 1.1 三层封装体系

模型代码分为三个层次的封装，职责清晰：

```
┌─────────────────────────────────────────────────────────┐
│ DiffusionPolicyBCModule (LightningModule)                │
│ 职责: 训练循环、优化器、调度器、日志、检查点                │
├─────────────────────────────────────────────────────────┤
│ DiffusionUnetImagePolicy (策略核心)                        │
│ 职责: 观测编码 → 条件构建 → 扩散去噪 → 动作输出              │
│   ├── ResNetImageEncoder    (多模态观测 → 全局条件向量)     │
│   ├── ConditionalUnet1D     (扩散去噪网络)                 │
│   ├── DDPMScheduler         (噪声调度器)                  │
│   └── SignalTransform       (姿态旋转 & 相对控制)          │
├─────────────────────────────────────────────────────────┤
│ SignalTransform / PoseRotationTransformer               │
│ 职责: 四元数 ↔ 6D/9D 旋转、绝对 ↔ 相对位姿变换              │
└─────────────────────────────────────────────────────────┘
```

### 1.2 观测编码器：ResNetImageEncoder

**输入**：多路图像 + 状态（时间维度 T=1）
**输出**：全局条件向量（512 维）

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  head_rgb        │    │ left_wrist_rgb   │    │ right_wrist_rgb  │
│  (B,1,3,240,320) │    │ (B,1,3,240,320)  │    │ (B,1,3,240,320)  │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         │  cat across batch     │                       │
         └───────────────┬───────┘                       │
                         │  (share_encoder=True 时)      │
                         ▼                               │
              ┌───────────────────────┐                  │
              │  ResNet18 (pretrained)│                  │
              │  + 3×Conv1d layers    │                  │
              │  512→128→64→32 ch     │                  │
              └───────────┬───────────┘                  │
                          │ flatten                      │
                          ▼                               │
              (B×3, 32×H/32×W/32) → reshape → concat     │
                                  │                      │
                                  ▼                      ▼
                          vision_feature (B, 512)
                                           │
                          ┌────────────────┼───────────────┐
                          │                                │
                          ▼                                ▼
              ┌───────────────────┐           ┌───────────────────┐
              │  State MLP        │           │  (其他 state keys) │
              │  input→64→128→256 │           │  ...              │
              └─────────┬─────────┘           └─────────┬─────────┘
                        │                               │
                        └───────────────┬───────────────┘
                                        ▼
                              concat all features
                                        │
                                        ▼
                          ┌─────────────────────────┐
                          │  Fusion MLP              │
                          │  input→2048→1024→1024→512│
                          └────────────┬────────────┘
                                       ▼
                              global_cond (B, 512)
```

**关键设计点：**

- **共享编码器**（`share_encoder=True`）：所有相机共享同一个 ResNet18 权重，在 batch 维度拼接图像后统一前传，再按相机数拆分并 concat。这比每路相机独立 encoder 节省显存和参数。
- **预训练权重**：使用 ImageNet 预训练的 ResNet18，作为迁移学习的起点。
- **额外卷积层**：ResNet18 输出的 512 通道特征经过 3 层额外卷积降维至 32 通道，减少后续 MLP 的输入维度。
- **状态编码**：关节状态通过独立 MLP 编码为 256 维，与视觉特征拼接后再通过 Fusion MLP。
- **不共享模式**（`share_encoder=False`）：每路相机有独立的 encoder 副本（`copy.deepcopy`），适用于相机视角差异大的场景。

### 1.3 扩散去噪网络：ConditionalUnet1D

**输入**：噪声动作轨迹 (B, T, action_dim) + 全局条件 (B, 512) + 时间步 (B,)
**输出**：预测噪声 (B, T, action_dim)

```
                   timestep ──→ SinusoidalPosEmb(128) ──→ MLP ──→ 128d
                                                                         │
                    global_cond (512d) ───────────────────────────────────┘
                                                 │
                                                 ▼
                                    global_feature (diffusion_step_embed + global_cond)

 input (B, horizon, action_dim) ──→ rearrange → (B, action_dim, horizon)
                                                      │
                     ┌────────────────────────────────┼────────────────────────────────┐
                     │                                │                                │
                     ▼                                ▼                                ▼
            ┌──────────────┐                ┌──────────────┐                ┌──────────────┐
            │  Down Block 0│                │  Down Block 1│                │  Down Block 2│
            │  512→512→1024│ ───downsample──│  1024→1024   │ ───downsample──│  → 2048      │
            │  ×2 ResBlock │                │  → 2048      │                │  → 2048      │
            └──────┬───────┘                └──────┬───────┘                └──────┬───────┘
                   │ skip                          │ skip                          │
                   ▼                               ▼                               ▼
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                              Mid Blocks                                     │
            │  2 × ConditionalResidualBlock (2048d, conditioned by global_feature)        │
            └────────────────────────────────────┬────────────────────────────────────────┘
                                                 │ upsample + skip concat
                                                 ▼
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                              Up Blocks                                      │
            │  Up Block 2: 2048+2048 → 1024 → 1024 ──→ upsample                          │
            │  Up Block 1: 1024+1024 → 512  → 512  ──→ upsample                          │
            │  Up Block 0: 512+512   → 512  → 512                                         │
            └────────────────────────────────────┬────────────────────────────────────────┘
                                                 ▼
                                    ┌────────────────────────┐
                                    │  Conv1d(512,512)       │
                                    │  Conv1d(512,action_dim)│
                                    └────────────┬───────────┘
                                                 ▼
                                    rearrange → (B, horizon, action_dim)
```

**条件注入方式 — FiLM（Feature-wise Linear Modulation）：**

全局条件向量通过一个小型 MLP 生成每个通道的 scale 和 bias：

```python
embed = MLP(global_feature)        # → [batch, out_channels * 2]
embed = embed.reshape(B, 2, C, 1)  # → scale 和 bias
scale, bias = embed[:, 0, ...], embed[:, 1, ...]
out = scale * out + bias           # 逐通道仿射变换
```

相比 Cross-Attention，FiLM 的计算开销更低，且不需要序列到序列的对齐，适合全局标量条件注入。

**残差块（ConditionalResidualBlock1D）：**

```
x (B, C_in, T)
    │
    ├──→ Conv1d(C_in→C_out, k=5) → GroupNorm(8) → Mish ──┐
    │                                                     │
    ├──→ FiLM conditioning (scale * out + bias)           │
    │                                                     │
    ├──→ Conv1d(C_out→C_out, k=5) → GroupNorm(8) → Mish ─┤
    │                                                     │
    └──→ Conv1d(C_in→C_out, 1×1) residual ───────────────→ + → out
```

**支持的降采样/上采样：**

- Downsample: `Conv1d(dim, dim, kernel=3, stride=2, padding=1)` — 在时间维度减半
- Upsample: `ConvTranspose1d(dim, dim, kernel=4, stride=2, padding=1)` — 在时间维度加倍

**其他条件类型（代码支持但默认未使用）：**

| 类型 | 描述 |
|------|------|
| `film` | FiLM 调制（默认） |
| `add` | 直接相加 |
| `cross_attention_add` | Cross-attention 后相加 |
| `cross_attention_film` | Cross-attention 后 FiLM |
| `mlp_film` | 多一层隐藏层的 MLP 生成 FiLM 参数 |

### 1.4 信号变换层：SignalTransform

位于策略最外层，负责两个任务：

#### 1.4.1 旋转表示转换（PoseRotationTransformer）

将原始的四元数位姿 (x, y, z, i, j, k, r → 7d) 转换为神经网络更友好的旋转表示：

| 模式 | 输出维度 | 说明 |
|------|----------|------|
| `quaternion` | 7d | 位置 3 + 四元数 4（默认） |
| `rotation_6d` | 9d | 位置 3 + 6D 连续旋转表示（取旋转矩阵前两列） |
| `rotation_9d` | 12d | 位置 3 + 9D 旋转表示（整个 3×3 旋转矩阵展平） |

**为什么需要 6D/9D 表示？** 四元数存在双覆盖问题（q 和 -q 表示同一旋转），且在归一化时可能产生不连续。6D 旋转表示（Zhou et al., CVPR 2019）提供了无奇点的连续旋转参数化。

#### 1.4.2 相对控制（RelativePoseTransformer）

在 `use_relative_control=True` 模式下：

- **动作相对化**：`T_relative = T_current_pose^{-1} @ T_action_target`
  - 网络预测的是相对于当前末端位姿的增量，而非绝对位姿
  - 有利于策略泛化到不同的起始位姿
- **观测相对化**：qpos 相对于 episode 起始位姿
  - `T_relative = T_episode_start^{-1} @ T_current_qpos`
  - 训练时对 base pose 添加噪声增强鲁棒性

**反向转换**：推理结束时，`SignalTransform.backward()` 将预测的相对位姿转回绝对位姿，供环境执行。

### 1.5 扩散调度器

使用 HuggingFace `diffusers` 库的 DDPMScheduler：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_train_timesteps` | 20 | 训练和推理的扩散步数 |
| `beta_start` | 0.0001 | 初始噪声方差 |
| `beta_end` | 0.02 | 最终噪声方差 |
| `beta_schedule` | `squaredcos_cap_v2` | 两端缓、中间陡的方差曲线 |
| `prediction_type` | `epsilon` | 预测噪声而非预测 x_0 |
| `clip_sample` | `true` | 采样时裁剪到 [-1, 1] |
| `variance_type` | `fixed_small` | 固定小方差 |

**训练时的前向扩散：**
```python
t = randint(0, 20)              # 随机采样时间步
noise = randn_like(action)       # 采样高斯噪声
noisy = scheduler.add_noise(action, noise, t)
pred = unet(noisy, t, global_cond)
loss = MSE(pred, noise)          # 预测噪声与真实噪声的 MSE
```

**推理时的反向扩散：**
```python
trajectory = randn(B, horizon, action_dim)  # 从纯噪声开始
for t in scheduler.timesteps:               # 20 → 18 → ... → 0
    pred_noise = unet(trajectory, t, global_cond)
    trajectory = scheduler.step(pred_noise, t, trajectory).prev_sample
```

### 1.6 优化器与调度器

**优化器配置：**
```yaml
optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.95]        # 比默认 [0.9, 0.999] 更低的 beta2，适应扩散策略的噪声梯度
  weight_decay: 0.0001
  pretrained_obs_encoder_lr_scale: 1.0  # 可设为 <1 降低预训练 encoder 的学习率
```

**参数分组策略：**
- `diffusion_model` 组：所有非预训练参数，使用默认 lr
- `pretrained_obs_encoder` 组：`obs_encoder.vision_encoders` 下的参数，使用 `lr * lr_scale`

**调度器：** OneCycleLR
```yaml
lr_scheduler:
  scheduler:
    type: OneCycleLR
    max_lr: ${model.optimizer.lr}   # 等于 optimizer.lr
    pct_start: 0.15                 # 前 15% 步骤升温
    anneal_strategy: cos            # 余弦退火
    div_factor: 100.0              # 初始 lr = max_lr / 100 = 1e-6
    final_div_factor: 1000.0       # 最终 lr = max_lr / 1000 = 1e-7
```

---

## 二、数据集设计

### 2.1 LeRobot 格式

数据集采用 HuggingFace LeRobot 标准格式，底层存储支持 zarr 数组和视频文件。

**核心字段结构（以 R1Pro 末端控制为例）：**

| 字段前缀 | 键名示例 | 形状 | 含义 |
|----------|----------|------|------|
| `observation.images.` | `head_rgb` | (360, 640, 3) | 头部 RGB 相机 |
| `observation.images.` | `left_wrist_rgb` | (480, 640, 3) | 左腕相机 |
| `observation.images.` | `right_wrist_rgb` | (480, 640, 3) | 右腕相机 |
| `observation.state.` | `left_ee_pose` | (7,) | 左末端位姿 (xyz + quat) |
| `observation.state.` | `right_ee_pose` | (7,) | 右末端位姿 |
| `observation.state.` | `left_gripper` | (1,) | 左夹爪开合度 |
| `observation.state.` | `right_gripper` | (1,) | 右夹爪开合度 |
| `observation.state.` | `left_arm` | (7,) | 左臂关节角 |
| `observation.state.` | `right_arm` | (7,) | 右臂关节角 |
| `observation.state.` | `chassis` | (3,) | 底盘状态 |
| `observation.state.` | `torso` | (4,) | 躯干状态 |
| `action.` | `left_ee_pose` | (7,) | 左末端目标位姿 |
| `action.` | `right_ee_pose` | (7,) | 右末端目标位姿 |
| `action.` | `left_gripper` | (1,) | 左夹爪目标 |
| `action.` | `right_gripper` | (1,) | 右夹爪目标 |
| — | `task` | string | 任务描述（来自 mcap） |

**速度字段：** `left_arm.velocities`、`right_arm.velocities`、`chassis.velocities`、`torso.velocities` 也存在数据集中，但默认配置中不作为观测输入。

### 2.2 时间窗口化机制（delta_timestamps）

LeRobot 的 `delta_timestamps` 参数实现了灵活的时间窗口采样：

```python
# 以 fps=30 为例
delta_timestamps = {
    # 视觉观测：取当前帧（负数表示过去）
    "observation.images.head_rgb":      [0 / 30],          # t=0
    "observation.images.left_wrist_rgb": [0 / 30],          # t=0
    "observation.images.right_wrist_rgb":[0 / 30],          # t=0

    # 状态观测：取当前状态
    "observation.state.left_ee_pose":   [0 / 30],          # t=0
    "observation.state.right_ee_pose":  [0 / 30],          # t=0
    "observation.state.left_gripper":   [0 / 30],          # t=0
    "observation.state.right_gripper":  [0 / 30],          # t=0

    # 动作：取未来 chunk_size 帧
    "action.left_ee_pose":      [0/30, 1/30, ..., 31/30],  # t=0 到 t=31
    "action.right_ee_pose":     [0/30, 1/30, ..., 31/30],
    "action.left_gripper":      [0/30, 1/30, ..., 31/30],
    "action.right_gripper":     [0/30, 1/30, ..., 31/30],
}
```

**设计意义：**
- `delta_timestamps` 由 LeRobotDataset 内部处理，`__getitem__(idx)` 自动返回对应时间窗口的数据
- 负值 `delta_timestamps` 实现观测历史（如 `[-1/30, 0]` 表示取过去 2 帧）
- 正值 `delta_timestamps` 实现动作未来（预测轨迹）
- 视频格式的数据不支持负索引（会自动截断到 0）

### 2.3 训练/验证分割

采用**周期性分割**策略：

```python
ratio = round((1 - val_set_proportion) / val_set_proportion)  # val=0.05 → ratio=19
# 训练集: episode 0, 2, 3, 4, ..., 18, 20, ... (每 19 个跳过 1 个)
# 验证集: episode 19, 38, 57, ... (每 19 个取 1 个)
```

优点：均匀分布，不会只取最后几个 episode（可能分布偏移）。

### 2.4 归一化统计

**计算方式：Min-Max 范围归一化**

```python
# get_norm_stats() 遍历全量数据
for batch in dataloader:
    batch = signal_transform.forward(batch)   # 先做旋转/相对变换
    qpos_min.append(qpos.amin(0))
    qpos_max.append(qpos.amax(0))
    action_min.append(action.amin(0))
    action_max.append(action.amax(0))

# 取全局 min/max
norm_stats = {
    "qpos":   {"min": qpos_min, "max": qpos_max},
    "action": {"min": action_min, "max": action_max},
}
```

**归一化到 [-1, 1]：**
```python
scale = (output_max - output_min) / (input_max - input_min)   # = 2 / range
offset = output_min - scale * input_min                       # = -1 - scale * min
normalized = raw * scale + offset
```

- 对接近常量的维度（range < 1e-4），不缩放，偏移设为均值
- 归一化参数保存在 `LinearNormalizer` 的 `ParameterDict` 中，随 checkpoint 持久化

### 2.5 缓存机制

```python
# use_cache=True 时
if idx in self.cache:
    return self.cache[idx]
# ... load from disk ...
self.cache[idx] = sample
```

- **适用场景**：LeRobotDataset 的随机访问较慢（尤其视频格式），缓存可显著加速
- **注意事项**：缓存占用内存 = 全量数据大小，建议配合 `num_workers=1` 使用
- **内存权衡**：小数据集可全缓存，大数据集建议关闭或使用 subsample

---

## 三、训练流程

### 3.1 完整训练管线

```
┌──────────────────────────────────────────────────────────────────────┐
│  Hydra Config Resolution                                             │
│  train.yaml + data/r1pro/lerobot_eef.yaml + model/unet_aug.yaml     │
│  + task/sim/R1ProBlocksStackEasy_eef.yaml                            │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  train.py: train(cfg)                                                │
│                                                                      │
│  1. datamodule = BaseDataModule(                                     │
│         train=GalaxeaLerobotDataset(is_training=True),               │
│         val=GalaxeaLerobotDataset(is_training=False)                 │
│     )                                                                │
│  2. model = DiffusionPolicyBCModule(policy=DiffusionUnetImagePolicy) │
│  3. callbacks = [ModelCheckpoint, RichProgressBar,                   │
│                   LRMonitor, ModelSummary]                           │
│  4. logger = WandBLogger                                             │
│  5. trainer = LightningTrainer(devices=[0,1,2,3], max_steps=20000)  │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  trainer.fit(model, datamodule)                                      │
│                                                                      │
│  Setup Phase:                                                        │
│  ├── datamodule.setup("fit") → 加载训练/验证数据集                    │
│  ├── model.setup("fit")                                             │
│  │   └── policy.set_normalizer(dataset.get_normalizer())             │
│  │       └── 遍历全量数据 → 计算 min/max → 创建 LinearNormalizer     │
│  │                                                                   │
│  Training Loop (max_steps=20000):                                    │
│  │   training_step(batch, batch_idx):                                │
│  │   ├── SignalTransform.forward(batch)                              │
│  │   │   ├── ee_pose → rotation conversion (quaternion/6d/9d)       │
│  │   │   ├── relative control: T_rel = T_cur^{-1} @ T_action        │
│  │   │   └── concat all action dims → batch["action"] (B, T, Da)    │
│  │   │       concat all qpos dims → batch["obs"]["qpos"]             │
│  │   │                                                               │
│  │   ├── normalizer.normalize(obs, action) → [-1, 1]                 │
│  │   │                                                               │
│  │   ├── Image Augmentation (train_transforms):                      │
│  │   │   Resize(252×336) → RandomCrop(240×320) → ColorJitter         │
│  │   │                                                               │
│  │   ├── obs / 255.0 → [0, 1] float                                  │
│  │   │                                                               │
│  │   ├── ResNetImageEncoder(nobs) → global_cond (B, 512)             │
│  │   │                                                               │
│  │   ├── Forward Diffusion:                                          │
│  │   │   t ~ Uniform(0, 20)                                          │
│  │   │   noise ~ N(0, I)                                             │
│  │   │   noisy = scheduler.add_noise(action, noise, t)               │
│  │   │                                                               │
│  │   ├── ConditionalUnet1D(noisy, t, global_cond) → pred             │
│  │   │                                                               │
│  │   ├── Loss: MSE(pred, noise)  # prediction_type=epsilon           │
│  │   │   ├── 总 loss: mean over all dims                             │
│  │   │   └── dim-wise loss: train_diffuse_loss/dim_00 ... dim_15     │
│  │   │                                                               │
│  │   └── log: train/_loss + dim-wise losses                          │
│  │                                                                   │
│  Checkpointing: every 5000 steps → step_XXXXX.ckpt                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 图像增广

**训练时增广链（train_transforms）：**

```python
for key in ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]:
    Resize(size=[252, 336])          # 先缩放到略大于目标
    RandomCrop(size=[240, 320])      # 随机裁剪到目标尺寸
    ColorJitter(                     # 颜色扰动
        brightness=0.3,              # ±30% 亮度
        contrast=0.4,                # ±40% 对比度
        saturation=0.5,              # ±50% 饱和度
        hue=0.3                      # ±30% 色调
    )
```

**评估时确定性处理（eval_transforms）：**

```python
    Resize(size=[252, 336])          # 同样先缩放
    CenterCrop(size=[240, 320])      # 中心裁剪（确定性）
```

### 3.3 关键训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_steps` | 20000 | 最大训练步数 |
| `devices` | `[0]` 或 `[0,1,2,3]` | GPU 数量 |
| `batch_size_train` | 64 | 训练 batch size |
| `batch_size_val` | 64 | 验证 batch size |
| `num_workers` | 16 | 训练数据 worker 数 |
| `gradient_clip_val` | 0.5 | 梯度裁剪 |
| `precision` | `32-true` | FP32 精度 |
| `sync_batchnorm` | `true` | DDP 下同步 BN |
| `log_every_n_steps` | 50 | 日志频率 |
| `seed` | 1000 | 随机种子 |

### 3.4 数据加载性能调优

**慢速数据加载的典型表现：** 每 epoch 开始前进度条卡顿 ~20 秒。

**解决方案（权衡内存换速度）：**
```yaml
# 在 data config 中设置
use_cache: True       # 全量缓存到内存
num_workers: 1        # 缓存模式下减少 worker 冲突
```

**数据加载流程：**
```
GalaxeaLerobotDataset.__getitem__(idx)
    ├── 确定 idx 属于哪个子数据集（支持多数据集合并）
    ├── LeRobotDataset[idx] → 自动根据 delta_timestamps 取时间窗口
    ├── 重组 sample 结构: {"obs": {...}, "action": {...}, "gt_action": {...}}
    ├── use_relative_control 时: 注入 episode_start_ee_poses
    ├── during_training 时: 图像从 [0,1] float 转为 [0,255] uint8
    └── use_cache 时: 写入缓存字典
```

---

## 四、评测方法

### 4.1 开环评估（Open-Loop Evaluation）

**文件：** `src/eval_open_loop.py`

**目的：** 验证模型对数据集动作的拟合程度，不依赖仿真环境。

**流程：**

```
1. 加载 checkpoint → 提取 state_dict → policy.cuda().eval()
2. 加载 val split 数据集 → 创建 DataLoader
3. 逐 batch 推理:
   ├── policy.predict_action(batch) → predicted_actions
   ├── batch["gt_action"] → ground_truth_actions
   └── 收集到列表
4. 按 episode 分组:
   ├── episode_data_index["from"] / ["to"] 确定每个 episode 的帧范围
   └── 每个 episode 一个输出目录
5. 可视化:
   └── plot_result(path, gt, pd) → 每个 action 维度一个 Plotly HTML
```

**关键细节：**
- 只取第一个时间步的 GT 动作做对比：`gt_actions[:, 0, :]`
- 预测动作是整个 chunk：`pd_actions` 形状为 `(N, chunk_size, action_dim)`
- 这反映了"策略从当前观测出发的预测能力"与"实际执行的第一步动作"的对比

**可视化效果：**
- 每个 action 维度（如 left_ee_pose_x, left_gripper 等）一个 HTML 文件
- 预测轨迹按时间步分组，不同组用不同颜色（5 色循环）
- 真实动作用红色曲线叠加
- 可通过 VS Code Live Preview 扩展在浏览器中查看

### 4.2 仿真闭环评估（Simulation Evaluation）

**文件：** `src/eval_sim.py`

**目的：** 在 GalaxeaManipSim 仿真环境中执行策略，评估闭环性能。

**流程：**

```
1. 加载 checkpoint → policy.cuda().eval()
2. env = gym.make(env_name, control_freq=30, max_episode_steps=600)
3. 循环 num_evaluations 次 (默认 100):
   ├── env.reset(seed=42)  # 固定种子保证可复现
   ├── action_seq = None, seq_idx = 0
   │
   ├── 循环直到 done:
   │   │
   │   ├── if action_seq is None or seq_idx >= len(action_seq):
   │   │   ├── 构建观测 batch (单帧, 无时间维度)
   │   │   │   ├── 图像: resize(224×224) → (1,1,3,224,224)
   │   │   │   └── 状态: (1, 1, dim)
   │   │   ├── policy.predict_action(batch) → (1, horizon, action_dim)
   │   │   └── action_seq = pred[:, :num_action_steps, :].squeeze()
   │   │       # 取前 16 步，丢弃后 16 步
   │   │
   │   ├── action = action_seq[seq_idx]
   │   ├── seq_idx += 1
   │   ├── numpy_obs, reward, terminated, truncated, info = env.step(action)
   │   └── terminated or truncated → done
   │
   ├── terminated → num_success++
   ├── 保存 rollout 视频 (mp4, libx264)
   ├── 保存动作曲线图 (8 维, png)
   └── 保存 info.json (包含奖励、终止原因等)
   │
4. 输出成功率: num_success / num_evaluations × 100%
```

**Action Chunking 机制：**

这是 Diffusion Policy 的核心推理策略：

```
时间线:  |---- inference ----|---- execute 16 steps ----|---- inference ----|
           产生 32 步动作         执行前 16 步              再次推理产生新动作
```

- **为什么只取一半？** 扩散策略的前几步预测通常比后几步更准确（因为条件信息更充分），取前半段可以减少累积误差
- **推理频率：** 每 16 步推理一次，30Hz 控制下 ≈ 每 0.53 秒推理一次
- **与开环评估的关系：** 开环评估看到的是整个 chunk 的预测质量，闭环评估只使用前半段

**两种控制模式的环境适配：**

| 模式 | 控制器类型 | 观测 state 键 | 动作维度 |
|------|-----------|--------------|----------|
| 末端控制 | `bimanual_relaxed_ik` | ee_pose + gripper | 左 7 + 左 1 + 右 7 + 右 1 = 16 |
| 关节控制 | `bimanual_joint_position` | arm_joints + gripper | 左 7 + 左 1 + 右 7 + 右 1 = 16 |

**仿真评估的可视化输出：**
- `rollout_N.mp4`：H.264 编码的仿真视频
- `rollout_N.png`：8 维动作轨迹曲线图（x, y, z, w, x, y, z, gripper）
- `info.json`：每个 episode 的奖励、终止原因等元数据

### 4.3 两种评估的对比

| 维度 | 开环评估 | 仿真评估 |
|------|----------|----------|
| **输入来源** | 数据集 val split | 仿真环境实时观测 |
| **策略调用** | 离线批量推理 | 在线循环推理 |
| **动作使用** | 全部 32 步用于对比 | 仅前 16 步用于执行 |
| **评估指标** | 轨迹拟合度（视觉对比） | 成功率（百分比） |
| **输出产物** | Plotly HTML 图表 | mp4 视频 + png + json |
| **运行速度** | 快（纯 GPU 推理） | 慢（含物理仿真） |
| **使用阶段** | 训练后快速验证 | 最终性能确认 |
| **仿真依赖** | 无需 GalaxeaManipSim | 需要 GalaxeaManipSim |

---

## 五、配置系统详解

### 5.1 Hydra 配置层级

```
configs/
├── train.yaml                      # 基础训练配置
│   ├── trainer: Lightning Trainer 参数
│   ├── callbacks: checkpoint, progress bar, lr monitor
│   ├── logger: WandB 配置
│   ├── hydra: 日志输出路径
│   └── defaults:                  # 默认不包含 data/model/task
│       ├── _self_
│       ├── data:                  # 空，由 task 指定
│       ├── model:                 # 空，由 task 指定
│       └── task:                  # 空，由 CLI 指定
├── data/
│   ├── r1pro/
│   │   ├── lerobot_eef.yaml       # R1Pro 末端控制数据集
│   │   └── lerobot_joints.yaml    # R1Pro 关节控制数据集
│   └── r1lite/
│       ├── lerobot_eef.yaml       # R1Lite 末端控制
│       └── lerobot_joints.yaml    # R1Lite 关节控制
├── model/
│   ├── unet_aug.yaml              # 扩散模型 + 图像增广
│   └── unet_aug_idpenc.yaml       # 同上（可能含独立策略编码）
└── task/
    └── sim/
        ├── R1ProBlocksStackEasy_eef.yaml    # R1Pro 堆方块（末端）
        ├── R1ProBlocksStackEasy_joints.yaml # R1Pro 堆方块（关节）
        ├── R1LiteBlocksStackEasy_eef.yaml   # R1Lite 堆方块（末端）
        └── R1LiteBlocksStackEasy_joints.yaml# R1Lite 堆方块（关节）
```

### 5.2 Task 配置的组合作用

以 `task/sim/R1ProBlocksStackEasy_eef.yaml` 为例：

```yaml
# @package _global_
defaults:
  - override /data: r1pro/lerobot_eef    # 覆盖 data 配置
  - override /model: unet_aug_idpenc     # 覆盖 model 配置

data:
  train:
    dataset_dirs:
      - /root/.cache/huggingface/lerobot/galaxea/R1ProBlocksStackEasy_joints/
```

这意味着：
1. 首先加载 `data/r1pro/lerobot_eef.yaml`（定义数据格式、action_keys 等）
2. 然后加载 `model/unet_aug_idpenc.yaml`（定义模型架构、优化器等）
3. 最后覆盖 `dataset_dirs` 指向具体数据集路径

### 5.3 常用 CLI 覆盖

```bash
# 切换 GPU
python src/train.py trainer.devices=[0]

# 减少训练步数（快速验证）
python src/train.py trainer.max_steps=1000

# 修改 batch size
python src/train.py data.train.batch_size_train=32

# 修改学习率
python src/train.py model.policy.optimizer.lr=5e-5

# 切换控制模式
python src/train.py task=sim/R1ProBlocksStackEasy_joints

# 修改扩散步数
python src/train.py model.policy.noise_scheduler.num_train_timesteps=50

# 启用缓存加速数据加载
python src/train.py data.train.use_cache=True data.train.num_workers=1
```

---

## 六、设计权衡与注意事项

### 6.1 优势

1. **完整流水线**：从数据生成到部署评估，端到端可复现
2. **模块化设计**：Hydra 配置系统使实验组合非常灵活
3. **轻量扩散**：20 步扩散相比传统 100 步大幅减少推理延迟
4. **预训练编码器**：ImageNet 预训练 ResNet18 加速收敛
5. **多模式支持**：末端控制和关节控制切换方便
6. **DDP 支持**：多 GPU 训练开箱即用

### 6.2 已知限制

1. **单帧观测**：`vision_obs_size=1, qpos_obs_size=1` 不使用历史帧，丢失速度信息
2. **无验证损失**：validation_step 为空（`return`），无验证损失监控
3. **检查点保存频率**：每 5000 步保存一次，间隔较大可能错过最佳点
4. **数据路径硬编码**：task yaml 中的 `dataset_dirs` 使用 `/root/.cache/...` 绝对路径
5. **相对控制未实现**：joint 模式下的 relative control 支持不完善
6. **R1 Lite 关节 6 DoF**：代码注释说明 R1 Lite 关节为 6 自由度，而 R1 Pro 为 7，但 `action_dims` 中写死为 6

### 6.3 扩展建议

- **多帧观测**：设置 `vision_obs_size=2` 或更大，让策略感知速度
- **验证监控**：实现 validation_step 计算验证集扩散损失
- **更频繁检查点**：设置 `every_n_train_steps: 1000` 保存更多中间状态
- **配置路径参数化**：通过环境变量或 CLI 参数覆盖 `dataset_dirs`
- **测试覆盖**：当前无单元测试，建议为核心组件（归一化、信号变换等）添加测试
- **评估自动化**：开环评估的 HTML 可视化可加入自动相似度评分（如 DTW 距离）