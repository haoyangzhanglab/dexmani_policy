# DexMani Policy 项目分析

## 项目概览

**DexMani Policy** 是一个面向灵巧手操作（Dexterous Manipulation）的机器人策略学习框架，支持多种主流扩散策略变体（DP3、DP、MoE-DP3、ManiFlow）。项目使用 Hydra 进行配置管理，基于 Zarr 格式存储数据集，提供完整的训练-验证-仿真评估闭环。

**运行环境**：conda env `policy`，从项目根目录运行（Hydra 以 CWD 解析 config 路径）。

**外部依赖**：`dexmani_sim`（仿真环境，独立安装），提供 `DATA_DIR` 和可复现的 eval 随机种子文件。

---

## 一、模型架构

### 1.1 整体数据流

```
obs_dict {point_cloud: (B,T,N,3), joint_state: (B,T,D)}
    │
    ├─► BaseAgent.preprocess()          # normalize + slice n_obs_steps + flatten B*T
    │       ├─► self.normalizer.normalize(obs_dict)
    │       ├─► slice [:, :n_obs_steps]
    │       └─► flatten(0, 1) → (B*T, ...)
    │
    ├─► ObsEncoder.forward()            # → cond: (B, out_dim) [film] 或 (B, T, out_dim) [cross_attn]
    │       ├─► PointCloud Encoder (PointNet / MultiStagePointNet / PointNext)
    │       └─► State MLP (64-dim)
    │       └─► [optional] MoE / TokenCompressor plugins
    │
    └─► ActionDecoder.compute_loss / predict_action
                ├─► Diffusion (DDIM)          → ConditionalUnet1D backbone
                └─► FlowMatch+Consistency     → DiTXFlowMatch backbone
                            └─► action chunk (B, horizon, action_dim)
```

### 1.2 关键维度约定

| 模式 | Encoder 输出 shape | 说明 |
|------|-------------------|------|
| `film` | `(B, out_dim * n_obs_steps)` | 展平向量，FiLM 条件注入时 context_dim = out_dim * n_obs_steps |
| `cross_attn` | `(B, n_obs_steps, out_dim)` | 序列形式，Cross-Attention 条件注入时 context_dim = out_dim |

**Action 输出**：`predict_action` 返回 `pred_action (B, horizon, action_dim)` 和 `control_action (B, n_action_steps, action_dim)`，后者从 `pred[:, n_obs_steps-1:]` 截取。

### 1.3 Agent 体系

所有 Agent 继承 `BaseAgent`（`agents/core/base.py`），通过 `hydra.utils.instantiate(cfg.agent)` 构建：

| Agent 类 | Config | 观测模态 | 解码器 | 特色 |
|----------|--------|---------|--------|------|
| `DP3Agent` | `dp3` | Point Cloud + Joint State | ConditionalUnet1D + DDIM | 经典 DP3 点云扩散策略 |
| `DPAgent` | `dp` | RGB + Joint State | ConditionalUnet1D + DDIM | RGB 视觉策略 |
| `MoEDP3Agent` | `moe_dp3` | Point Cloud + Joint State + MoE | ConditionalUnet1D + DDIM | 稀疏 MoE 增强表征 |
| `ManiFlowAgent` | `maniflow` | Point Cloud + Joint State | DiTXFlowMatch + Consistency | Rectified Flow + 一致性蒸馏 |

### 1.4 观测编码器（ObsEncoder）

#### 点云全局编码器（`pointcloud/registry.py: build_pc_global_encoder`）

| encoder_type | 实现类 | 说明 |
|-------------|--------|------|
| `dp3` | `PointNet` | 原版 DP3 PointNet，output_channels=256 |
| `idp3` | `MultiStagePointNet` | iDP3 多阶段 PointNet，output_channels=256 |
| `pointnext` | `PointNextEncoder` | 分层 SA 模块，多尺度半径和邻居数 |

**处理流程**（以 `DP3ObsEncoder` 为例）：
1. 输入点云 `point_cloud (B*T, N, 3)` → 若 N > num_points，使用 **最远点采样 (FPS)** 下采样
2. `pc_encoder(pc)` → `{'global_token': (B*T, pc_out_dim)}`
3. `state_mlp(joint_state)` → `(B*T, 64)`
4. `cat([pc_token, state_token], dim=-1)` → `(B*T, out_dim)`
5. 按 condition_type reshape：
   - `film` → `(B, out_dim * n_obs_steps)`
   - 其他 → `(B, n_obs_steps, out_dim)`

#### 点云 Patch Tokenizer（`build_pc_patch_tokenizer`）

用于 DiT-X 架构，输出 token 序列：

| tokenizer_type | 实现类 | 输出 token 数 |
|---------------|--------|-------------|
| `pointpn` | `PointPNTokenizer` | `(num_stages 汇总) * n_obs_steps` |
| `pointnext_tokenizer` | `PointNextPatchTokenizer` | `(num_patches + 1) * n_obs_steps` |

#### RGB 编码器（`backbone_2d/`）

| 类型 | 实现 | 输出维度 |
|------|------|---------|
| `resnet` | ResNet | 512-dim global_token |
| `clip` | CLIP Vision | 512-dim global_token |
| `dino` | DINOv2 | 512-dim global_token |
| `siglip` | SigLIP | 512-dim global_token |

#### Proprio（本体感觉）

`StateMLP`（`proprio/state_mlp.py`）：将关节状态映射为 64-dim 特征向量。

### 1.5 插件系统（Plugins）

#### MoE（Mixture of Experts，`plugins/moe.py`）

- **结构**：`num_experts` 个 ExpertMLP + 1 个 Router (Linear)
- **路由**：Softmax → Top-K 选择 → 加权混合
- **辅助损失**：
  - `load_balance_loss`：鼓励专家负载均衡（`lambda_load=0.1`）
  - `entropy_loss`：鼓励路由分布有足够熵（`beta_entropy=0.01`）
- **override_idx**：推理时可强制指定专家索引，实现确定性行为

```python
# MoE forward
probs = softmax(router(z))                    # (B, num_experts)
topk_prob, topk_idx = topk(probs, k=top_k)   # (B, K)
topk_prob = topk_prob / topk_prob.sum(-1)     # 重归一化
output = sum(topk_prob[i, k] * expert_k(z) for k in top_k)
```

#### TokenCompressor（`plugins/token_compressor.py`）

Cross-Attention + Self-Attention 降维，用于压缩 token 序列长度。

### 1.6 动作解码器（Action Decoders）

#### Diffusion（`diffusion.py`）

包装任意 backbone + `diffusers.DDIMScheduler`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_training_steps` | 100 | 训练时扩散步数 |
| `num_inference_steps` | 10 | 推理时去噪步数 |
| `prediction_type` | `sample` | `sample`（直接预测 x_0）或 `epsilon`（预测噪声） |
| `beta_schedule` | `squaredcos_cap_v2` | 余弦退火调度 |

**训练**：采样 t ∈ [0, num_training_steps)，添加噪声 → MSE(pred, target)
**推理**：DDIM 反向采样，从零均值高斯噪声逐步去噪

#### FlowMatchWithConsistency（`flowmatch.py`）

Rectified Flow + Consistency Distillation 混合训练：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `flow_batch_ratio` | 0.75 | Flow 分支占比 |
| `consistency_batch_ratio` | 0.25 | Consistency 分支占比 |
| `denoise_timesteps` | 10 | 推理时 Euler 积分步数 |

**Flow 分支**（75% batch）：
- 线性插值：`x_t = (1-t) * x_0 + t * x_1`，目标速度 `v = x_1 - x_0`
- t 采样模式：`beta`（偏向两端）

**Consistency 分支**（25% batch，需 EMA teacher）：
- 用 EMA 模型预测 `x_{t+dt}` 处的速度
- 从 `x_{t+dt}` 外推回 `x_1`，再计算从 `x_t` 到 `x_1` 的速度
- 核心思想：**用更远一步的趋势作为监督信号，指导模型学会更全局的趋势**
- t 采样模式：`discrete`（离散值），dt 采样：`uniform`

**推理**：Euler 积分 `x_{t+dt} = x_t + v_t * dt`

### 1.7 Backbone 网络

| Backbone | 架构 | 条件注入 | 适用场景 |
|----------|------|---------|---------|
| `ConditionalUnet1D` | 1D U-Net (256→512→1024) | FiLM / Cross-Attention | Diffusion |
| `DiTXFlowMatch` | DiT-X (12层, 768-dim, 8-head) | AdaLN-Zero + Cross-Attention | Flow Match |

### 1.8 信号转换（SignalTransform）

与 GalaxeaDP 不同，DexMani Policy **不包含**旋转表示转换（quaternion/6D/9D）和相对位姿控制模块。这是因为灵巧手操作通常直接使用关节空间控制（joint state + action dim=19），不需要末端执行器位姿的旋转表示。

---

## 二、数据集设计

### 2.1 Zarr 数据集

数据存储在 `robot_data/sim/<task_name>.zarr` 中，使用 Zarr 格式（支持内存映射和懒加载）。

**必需字段**：

| Key | Shape | 说明 |
|-----|-------|------|
| `point_cloud` | `(N_total, 3)` | 逐帧拼接的点云（N 为每帧点数） |
| `joint_state` | `(N_total, D)` | 关节状态（D=19 为默认 action_dim） |
| `action` | `(N_total, A)` | 动作（A=19 为默认 action_dim） |

**可选字段**：`rgb`、`depth`、`camera_intrinsic`、`camera_extrinsic`

### 2.2 ReplayBuffer（`datasets/common/replay_buffer.py`）

- 内存映射 Zarr，懒加载
- 维护 `episode_ends` 数组标记每个 episode 的结束索引
- 支持按 key 切片访问

### 2.3 SequenceSampler（`datasets/common/sampler.py`）

使用 **Numba-JIT** 编译的 `create_indices` 函数构建索引，支持：

1. **时序窗口采样**：按 `horizon` 长度切出连续帧序列
2. **训练/验证划分**：通过 `episode_mask` 控制哪些 episode 用于训练
3. **边界 Padding**：
   - `pad_before = n_obs_steps - 1`：序列开头补零（复制第一帧）
   - `pad_after = n_action_steps - 1`：序列末尾补零（复制最后一帧）

```python
# Padding 策略
if sample_start_idx > 0:
    data[:sample_start_idx] = sample[0]     # 向前复制首帧
if sample_end_idx < sequence_length:
    data[sample_end_idx:] = sample[-1]       # 向后复制末帧
```

### 2.4 训练/验证分割策略

```python
# get_val_mask: 随机选择 n_val 个 episode 作为验证集
n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
val_mask = random_choice(n_episodes, size=n_val, replace=False)
train_mask = ~val_mask
```

**验证集获取**：`get_validation_dataset()` 通过 `copy.copy(self)` 浅拷贝，替换 sampler 的 `episode_mask` 为 `val_mask`，并关闭 augmentation。

### 2.5 Dataset 类体系

| Dataset 类 | 传感器模态 | 用途 |
|-----------|-----------|------|
| `BaseDataset` | 可配置 sensor_modalities | 基础数据集 |
| `PCDataset` | `['point_cloud', 'joint_state']` | 点云数据集 |
| `RGBDataset` | `['rgb', 'joint_state']` | RGB 数据集 |
| `SemGeoDataset` | 语义+几何特征 | 高级数据集 |

### 2.6 归一化（LinearNormalizer）

**Min-Max 归一化**（`mode='limits'`）：
```
scale = (output_max - output_min) / (input_max - input_min)
offset = output_min - scale * input_min
x_normalized = x * scale + offset  # 映射到 [-1, 1]
```

- 对 `action` 和 `joint_state` 分别拟合
- `range_eps=1e-4`：方差太小的维度视为常量，不缩放
- 参数存储为 `nn.ParameterDict`（`requires_grad=False`），随 checkpoint 持久化

### 2.7 数据增强

- **点云增强**（`augmentation/pc_aug.py`）：随机旋转、缩放、平移等
- **RGB 增强**（`augmentation/rgb_aug.py`）：ColorJitter、RandomCrop 等
- 仅在训练集启用，验证集自动关闭（`val_set.augmentation_cfg = None`）

---

## 三、训练流程

### 3.1 完整训练流程

```
train.py (Hydra @hydra.main)
    │
    ├─► set_seed(cfg.training.seed)
    │
    ├─► build_train_components(cfg)
    │       ├─► dataset = hydra.utils.instantiate(cfg.dataset)
    │       ├─► normalizer = dataset.get_normalizer()     # 遍历 zarr 计算 min/max
    │       ├─► train_loader = DataLoader(dataset, **cfg.dataloader)
    │       ├─► val_loader = DataLoader(val_set, **cfg.val_dataloader)
    │       ├─► model = hydra.utils.instantiate(cfg.agent)
    │       ├─► model.load_normalizer_from_dataset(normalizer)
    │       ├─► [if use_ema] ema_model = copy.deepcopy(model)
    │       ├─► optimizer = model.configure_optimizer(**cfg.optimizer)
    │       └─► scheduler = get_scheduler(cosine, warmup=500, total_steps)
    │
    ├─► workspace = TrainWorkspace(output_dir, wandb_cfg, checkpoint_cfg)
    │
    ├─► [if eval_interval > 0] env_runner = SimRunner(task_name, n_obs_steps)
    │
    └─► Trainer(...).train(resume_tag="latest")
            │
            ├─► load_for_resume(model, ema_model, optimizer, scheduler)
            │
            └─► for epoch in range(num_epochs):
                    │
                    ├─► for batch in train_loader:
                    │       ├─► loss, log_dict = model.compute_loss(batch)
                    │       ├─► loss.backward()
                    │       ├─► clip_grad_norm_(max_norm=1.0)
                    │       ├─► optimizer.step()
                    │       ├─► scheduler.step()
                    │       ├─► optimizer.zero_grad()
                    │       └─► [if use_ema] ema_updater.step(model)
                    │
                    ├─► [sample] compute_action_mse_for_one_batch()
                    ├─► [validate] val_loss = validate(ema_model)
                    ├─► [evaluate] eval_metrics = env_runner.run(ema_model)
                    └─► [save] save_checkpoint() + save_topk()
```

### 3.2 训练超参数（dp3.yaml 默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `horizon` | 16 | 预测窗口长度 |
| `n_obs_steps` | 2 | 观测历史帧数 |
| `n_action_steps` | 8 | 每步执行的步数（Action Chunking） |
| `action_dim` | 19 | 动作空间维度 |
| `batch_size` | 256 | 训练 batch size |
| `lr` | 1e-4 | 学习率 |
| `obs_lr` | 1e-4 | 观测编码器学习率 |
| `weight_decay` | 1e-6 | 权重衰减 |
| `betas` | (0.95, 0.999) | AdamW beta |
| `num_epochs` | 1000 | 总训练轮数 |
| `lr_warmup_steps` | 500 | 学习率预热步数 |
| `lr_scheduler` | cosine | 余弦退火 |

### 3.3 周期性任务调度

| 任务 | 默认间隔 | 说明 |
|------|---------|------|
| `log_interval_steps` | 50 | 每 50 步记录一次训练指标 |
| `sample_interval_epochs` | 5 | 每 5 轮计算一次 action MSE |
| `val_interval_epochs` | 25 | 每 25 轮计算验证 loss |
| `eval_interval_epochs` | 250 | 每 250 轮执行仿真评估 |
| `checkpoint_interval_epochs` | = eval_interval (若启用) 或 val_interval | Checkpoint 保存间隔 |

### 3.4 EMA（指数移动平均）

```python
# EMAModel: update_after_step=0, power=0.75, min_value=0.0, max_value=0.9999
# decay = max_value * (1 - exp(-step / inv_gamma))^power
# ema_weight = decay * ema_weight + (1 - decay) * model_weight
```

- EMA 模型用于验证和仿真评估推理
- ManiFlow 的 consistency 分支使用 EMA 模型作为 teacher（`use_ema_teacher_for_consistency=true`）

### 3.5 优化器配置

```python
# BaseAgent.configure_optimizer()
# 1. action_decoder.model.get_optim_groups(weight_decay)  # 按层分组（含 weight decay 分离）
# 2. 将 obs_encoder 的 requires_grad 参数加入优化器（使用独立的 obs_lr/obs_wd）
# 3. torch.optim.AdamW(all_groups, lr=lr, betas=betas)
```

### 3.6 Workspace 管理

**TrainWorkspace**（`training/common/workspace.py`）：
- 管理输出目录：`experiments/<policy_name>/<task_name>/<date>_<time>_<seed>/`
- WandB（offline 模式）日志
- JSONL 追加式日志文件

**CheckpointIO**（`training/common/checkpoint_io.py`）：
- 原子保存（先写 `.tmp`，再 `os.rename`）
- `latest.pt` 软链接
- JSON manifest top-k 追踪（默认 `test_mean_score`，max 模式，top-3）
- 自动删除低分 checkpoint

### 3.7 数据加载流程

```
Zarr文件 (robot_data/sim/<task>.zarr)
    │
    ├─► ReplayBuffer.copy_from_path(zarr_path, keys=['point_cloud', 'joint_state', 'action'])
    │       └─► 内存映射，懒加载
    │
    ├─► get_val_mask(seed, val_ratio, n_episodes)
    │       └─► 随机选择 val_ratio 的 episode
    │
    ├─► SequenceSampler(replay_buffer, horizon, pad_before, pad_after, episode_mask)
    │       └─► Numba-JIT 索引构建
    │
    ├─► BaseDataset.__getitem__(idx)
    │       ├─► sampler.sample_sequence(idx)  → numpy dict
    │       ├─► sample_to_data()              → {'obs': {...}, 'action': ...}
    │       ├─► apply_augmentation(data)      → 仅训练集
    │       └─► dict_apply(torch.from_numpy)  → torch tensor
    │
    └─► DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=True, persistent_workers=True)
```

---

## 四、评测方法

### 4.1 开环评估（无专门脚本）

DexMani Policy 没有像 GalaxeaDP 那样的独立开环评估脚本。但训练过程中每 `sample_interval_epochs` 轮会计算 `train/action_mse_error`：

```python
# Trainer.compute_action_mse_for_one_batch()
pred_action = agent.predict_action(obs)["pred_action"]
mse = F.mse_loss(pred_action, gt_action)
```

### 4.2 仿真评估（`eval_sim.py`）

```
eval_sim.py --policy-name dp3 --task-name multi_grasp --exp-name 2026-04-01_11-18_233
    │
    ├─► 从 experiments/<policy>/<task>/<exp-name>/ 读取 config.yaml
    ├─► OmegaConf.merge(cfg, overrides)    # 支持命令行覆盖
    ├─► disable wandb (eval 不需要记录训练指标)
    │
    ├─► SimEvalBuilder.build_components()
    │       ├─► agent = hydra.utils.instantiate(cfg.agent)
    │       ├─► env_runner = hydra.utils.instantiate(cfg.env_runner)
    │       └─► workspace = hydra.utils.instantiate(cfg.workspace)
    │
    └─► SimEvaluator.run()
            ├─► 加载 checkpoint（ckpt_tag_or_path: best / latest / 具体路径）
            ├─► for episode_seed in seed_list[:eval_episodes]:
            │       ├─► env = make_env()    # 动态导入 dexmani_sim.envs
            │       ├─► reset env with seed
            │       ├─► 循环 rollout:
            │       │       ├─► 收集 n_obs_steps 帧观测
            │       │       ├─► agent.predict_action(obs) → action_seq
            │       │       └─► 执行 n_action_steps 步（Action Chunking）
            │       ├─► 记录 success / steps / video
            │       └─► cleanup env
            │
            ├─► 计算 success_rate, avg_steps
            ├─► 为每个 denoise_timesteps 生成 .mp4 视频
            ├─► 保存 eval_metrics.json
            └─► 保存 eval record
```

### 4.3 Action Chunking 机制

```python
# BaseAgent.predict_action()
# 1. 推理产生 pred_action: (B, horizon, action_dim)
# 2. 截取 control_action: (B, n_action_steps, action_dim)
#    从 pred[:, n_obs_steps-1:] 开始（跳过历史帧对应的动作）
# 3. 在环境中逐步执行 control_action 的每一步
# 4. 当 action_seq 执行完毕，重新推理
```

**示例**（n_obs_steps=2, horizon=16, n_action_steps=8）：
- 使用 t-1 和 t 两帧观测
- 推理产生 16 步动作
- 执行第 2~9 步（共 8 步）
- 重新推理

### 4.4 评估配置（dp3.yaml 默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eval_episodes` | 100 | 评估 episode 数 |
| `denoise_timesteps_list` | [10] | 测试不同的去噪步数 |
| `use_ema_for_eval` | true | 使用 EMA 模型评估 |
| `ckpt_tag_or_path` | best | 加载 best/top-k checkpoint |

### 4.5 SimRunner（`env_runner/sim_runner.py`）

动态导入仿真环境：
```python
# task_name = "pick_apple_messy" → PascalCase = "PickAppleMessy"
# 导入模块: dexmani_sim.envs.pick_apple_messy
# 实例化类: PickAppleMessy(render_mode="rgb_array")
```

随机种子从 `DATA_DIR/eval_seeds/<task_name>.txt` 读取，确保可复现。

---

## 五、配置系统详解

### 5.1 Hydra 配置结构

```
configs/
├── dp3.yaml         # DP3 点云扩散策略
├── dp.yaml          # RGB 扩散策略
├── moe_dp3.yaml     # MoE 增强点云扩散策略
└── maniflow.yaml    # Rectified Flow + Consistency 策略
```

每个配置包含：
- `policy_name` / `task_name` / `seed`
- `horizon` / `n_obs_steps` / `n_action_steps` / `action_dim`
- `dataloader` / `val_dataloader`
- `dataset`（含 zarr_path、sensor_modalities 等）
- `agent`（含 encoder、backbone、decoder 参数）
- `optimizer`（含 lr、weight_decay、betas）
- `ema`（含 decay 参数）
- `workspace`（含 wandb_cfg、checkpoint_cfg）
- `training`（含 loop 控制参数）
- `env_runner`
- `eval`
- `hydra`（输出目录配置）

### 5.2 常用 CLI 覆盖示例

```bash
# 基本训练
python dexmani_policy/train.py --config-name dp3 seed=233 task_name=multi_grasp

# 修改 batch size 和 epoch 数
python dexmani_policy/train.py --config-name dp3 \
  dataloader.batch_size=64 training.loop.num_epochs=2000

# 修改 encoder 类型
python dexmani_policy/train.py --config-name dp3 \
  agent.encoder_type=pointnext agent.num_points=2048

# 修改扩散步数
python dexmani_policy/train.py --config-name dp3 \
  agent.num_training_steps=200 agent.num_inference_steps=20

# ManiFlow 训练（启用 consistency 蒸馏）
python dexmani_policy/train.py --config-name maniflow \
  task_name=pick_apple_messy training.use_ema_teacher_for_consistency=true

# 评估（覆盖评估轮数）
python dexmani_policy/eval_sim.py --policy-name dp3 --task-name multi_grasp \
  --exp-name 2026-04-01_11-18_233 eval.sim.eval_episodes=200
```

---

## 六、设计权衡与注意事项

### 6.1 优势

1. **模块化设计**：Agent、ObsEncoder、ActionDecoder 完全解耦，通过 Hydra 配置组合
2. **多策略支持**：同一套训练框架支持 DP3、DP、MoE-DP3、ManiFlow 四种策略
3. **PointNet 系列编码器**：支持原版 PointNet（dp3）、多阶段 PointNet（idp3）、PointNext 分层编码器
4. **MoE 插件**：稀疏 Top-K 专家混合，增强表征能力，含 load balance 和 entropy 辅助损失
5. **Flow Match + Consistency**：ManiFlow 将 rectified flow 和一致性蒸馏结合，EMA teacher 提供更稳定的全局趋势信号
6. **完善的训练基础设施**：EMA、checkpoint top-k 追踪、断点续训、WandB offline、JSONL 日志

### 6.2 已知限制

1. **单帧观测**（`n_obs_steps=2`）：仅使用 2 帧历史观测，无法捕捉长期速度/加速度信息。灵巧操作可能需要更长的观测历史来理解动态变化趋势。
2. **无旋转表示转换**：与 GalaxeaDP 不同，本项目不包含旋转表示（quaternion/6D/9D）转换和相对位姿控制模块。这在灵巧手场景下是合理的（直接关节空间控制），但如果需要末端执行器控制则需要扩展。
3. **数据路径硬编码**：`zarr_path: robot_data/sim/${task_name}.zarr` 使用相对路径，部署时需要手动修改或设置软链接。
4. **单 GPU 训练**：`device: cuda:0` 硬编码为单卡，不支持 DDP 多卡训练。大模型（如 DiT-X 12层）可能需要多卡加速。
5. **Validation 不记录详细指标**：`Trainer.validate()` 仅返回 loss，没有像 GalaxeaDP 那样的 action MSE 或其他细粒度指标。
6. **评估脚本独立于 Hydra**：`eval_sim.py` 使用 `argparse` + 从实验目录读取 `config.yaml` 的方式，而不是直接通过 Hydra 启动，导致训练和评估的启动方式不一致。

### 6.3 扩展建议

1. **多卡训练**：添加 PyTorch DDP 支持，参考 GalaxeaDP 的 `trainer.devices=[0,1,2,3]` 覆盖方式
2. **多帧观测**：增加 `n_obs_steps=4` 或更多，捕获长期动态信息
3. **文本/语言条件**：项目已有 `obs_encoder/text/` 模块（T5、CLIP text），可扩展为语言条件的策略（Language-Conditioned Policy）
4. **RGB+Point Cloud 融合**：当前数据集要么用点云要么用 RGB，可扩展为双模态融合（如 `RGBPCDataset`）
5. **推理加速**：Diffusion 可引入 Consistency Models / DPM-Solver 加速，ManiFlow 已经实现了 Flow Match 的一致性蒸馏
6. **数据增强扩展**：当前点云增强较简单，可加入 CutMix、MixUp 等更复杂的增强策略
7. **开环评估脚本**：添加独立的开环评估脚本（类似 GalaxeaDP 的 `eval_open_loop.py`），生成预测 vs 真实的可视化对比
