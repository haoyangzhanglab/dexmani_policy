# GalaxeaDP 与 DexMani Policy 训练机制深度对比

## 一、总体训练架构概览

### GalaxeaDP：Lightning + Hydra 体系

```
train.py (Hydra @hydra.main)
    │
    ├─► hydra.utils.instantiate(datamodule)  → BaseDataModule(LightningDataModule)
    ├─► hydra.utils.instantiate(model)       → DiffusionPolicyBCModule(LightningModule)
    ├─► instantiate_callbacks()              → [ModelCheckpoint, RichProgressBar, LRMonitor]
    ├─► instantiate_loggers()                → WandBLogger
    ├─► hydra.utils.instantiate(trainer)     → Lightning.Trainer(devices=[0,1,2,3])
    └─► trainer.fit(model, datamodule)       → 自动 DDP + 训练循环
```

### DexMani Policy：自研 Trainer + Hydra 体系

```
train.py (Hydra @hydra.main)
    │
    ├─► hydra.utils.instantiate(dataset)     → PCDataset / RGBDataset
    ├─► dataset.get_normalizer()             # 遍历 zarr 计算 min/max
    ├─► DataLoader(dataset, **cfg.dataloader)
    ├─► hydra.utils.instantiate(cfg.agent)   → DP3Agent / DPAgent
    ├─► model.load_normalizer_from_dataset()
    ├─► [if use_ema] ema_model = copy.deepcopy(model)
    ├─► optimizer + scheduler
    └─► Trainer(...).train()                 → 手动训练循环 (无 Lightning)
```

**核心差异**：GalaxeaDP 依赖 PyTorch Lightning 框架，训练循环、分布式、日志全部委托给 Lightning；DexMani Policy 使用自研的 `Trainer` + `TrainWorkspace` 类，手动实现训练循环。

---

## 二、训练流程逐层剖析

### 2.1 GalaxeaDP 训练循环

```python
# DiffusionPolicyBCModule.training_step()
def training_step(self, batch, batch_idx):
    loss, loss_log = self.policy.compute_loss(batch)  # forward diffusion
    self.log_dict({"train/_loss": loss}, sync_dist=True)
    self.log_dict(loss_log, sync_dist=True)
    return loss

# DiffusionUnetImagePolicy.compute_loss()
def compute_loss(self, batch):
    batch = signal_transform.forward(batch)         # 旋转/相对变换
    nobs = self.normalizer.normalize(batch["obs"])   # 归一化
    nactions = self.normalizer["action"].normalize(batch["action"])

    # 图像增强 + 归一化到 [0,1]
    nobs = apply_train_transforms(nobs) / 255.0

    global_cond = self.obs_encoder(nobs)             # ResNet18 编码
    trajectory = nactions                            # 真实动作
    noise = torch.randn_like(trajectory)             # 采样噪声
    timesteps = torch.randint(0, 20, (bsz,))         # 随机时间步
    noisy_trajectory = scheduler.add_noise(trajectory, noise, timesteps)

    pred = unet(noisy_trajectory, timesteps, global_cond)

    # 目标：epsilon / sample / v_prediction
    target = noise  # (epsilon 模式)
    loss = F.mse_loss(pred, target, reduction='none')
    loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
    return loss, loss_log
```

### 2.2 DexMani Policy 训练循环

```python
# Trainer.train() — 手动循环
for epoch in range(num_epochs):
    for batch in train_loader:
        loss, log_dict = model.compute_loss(batch)
        loss.backward()
        clip_grad_norm_(max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if use_ema:
            ema_updater.step(model)  # EMA 更新

    # 周期性任务
    if epoch % sample_interval == 0:
        compute_action_mse()       # 计算预测 MSE
    if epoch % val_interval == 0:
        val_loss = validate()      # 验证集 loss
    if epoch % eval_interval == 0:
        eval_metrics = env_runner.run()  # 仿真评估
    if epoch % checkpoint_interval == 0:
        save_checkpoint()
```

**关键差异对比**：

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 循环控制 | Lightning 自动管理 | 手动 for 循环 |
| 梯度裁剪 | `trainer.gradient_clip_val=0.5` | 手动 `clip_grad_norm_(1.0)` |
| 日志 | `self.log_dict(sync_dist=True)` 自动聚合 | 手动写入 WandB + JSONL |
| 验证 | `validation_step` 返回空（**未实现**） | 定期 `validate()` 返回 loss |
| EMA | **无** | **有**（EMA 模型用于 eval） |
| 仿真评估 | 独立脚本 `eval_sim.py` | 集成到训练循环中（周期性调用） |
| Checkpoint | Lightning `ModelCheckpoint` 每 5000 步 | 原子保存 + top-k 追踪 |

---

## 三、多卡训练（DDP）深度分析

### 3.1 GalaxeaDP 的多卡训练机制

GalaxeaDP 的多卡训练**完全依赖 PyTorch Lightning 的 DDP 策略**：

```yaml
# configs/train.yaml
trainer:
  devices: [0]              # 单卡；改为 [0,1,2,3] 即 4 卡
  strategy: auto            # Lightning 自动选择 DDP
  num_nodes: 1
  sync_batchnorm: true      # DDP 下同步 BatchNorm
  accumulate_grad_batches: 1
  gradient_clip_val: 0.5
  precision: 32-true        # FP32（未启用混合精度）
```

**启动方式**：
```bash
bash scripts/train.sh trainer.devices=[0,1,2,3] task=sim/R1ProBlocksStackEasy_eef
```

#### Lightning DDP 工作原理

当 `devices > 1` 时，Lightning 自动：

1. **启动多进程**：为每个 GPU 创建一个进程，使用 `torch.multiprocessing.spawn`
2. **初始化进程组**：创建 `torch.distributed` 的 NCCL backend
3. **模型复制**：每个 GPU 持有一份完整的 `DiffusionPolicyBCModule` 副本
4. **数据分片**：`DataLoader` 自动使用 `DistributedSampler`，将 batch 按 GPU 数均分
5. **梯度同步**：每次 `backward()` 后，通过 `all_reduce` 操作同步梯度
6. **参数更新**：每个进程独立更新自己的参数（因为梯度已同步，更新结果一致）

#### GalaxeaDP 的多卡安全设计

```python
# train.py: line 103-105
if len(cfg.trainer.devices) > 1:
    log.info("Destroying process group")
    dist.destroy_process_group()
```

- **进程组清理**：训练结束后显式销毁进程组，防止残留
- **`sync_dist=True`**：在 `log_dict` 中启用跨进程日志聚合
- **`sync_batchnorm=True`**：确保 BN 的 running stats 跨卡同步
- **`RankedLogger(rank_zero_only=True)`**：确保日志只从 rank 0 进程打印

#### 实际的批大小计算

```bash
# devices=[0,1,2,3], batch_size_train=64
# 每张卡的实际 batch = 64（DataLoader 返回的）
# 全局有效 batch = 64 × 4 = 256
```

**注意**：Lightning 的 `DataLoader` 在 DDP 下会自动切分。如果 `batch_size_train=64`，每张卡拿到的实际是 `64 / num_gpus`。但具体取决于是否手动设置了 `DistributedSampler`。GalaxeaDP 使用 `BaseDataModule`，Lightning 会自动添加 `DistributedSampler`，此时 `batch_size` 是**每卡的局部 batch**，全局 batch = `batch_size × num_gpus`。

#### 多卡训练的数据流

```
GPU 0: batch[0:16]  ──→ compute_loss ──→ loss_0  ─┐
GPU 1: batch[16:32]  ──→ compute_loss ──→ loss_1   ├── all_reduce(grads)
GPU 2: batch[32:48]  ──→ compute_loss ──→ loss_2   │
GPU 3: batch[48:64]  ──→ compute_loss ──→ loss_3  ┘
                                                    ↓
                                              step(optimizer)
```

### 3.2 DexMani Policy 的多卡训练机制

**DexMani Policy 目前不支持多卡训练。**

从 `dp3.yaml` 配置文件和代码中可以确认：

```yaml
# dp3.yaml
device: cuda:0    # 硬编码为单卡
```

关键限制：

1. **无 DDP 策略**：自研 `Trainer` 类没有分布式训练逻辑
2. **无 DistributedSampler**：DataLoader 未配置 `DistributedSampler`
3. **设备硬编码**：模型直接 `.to('cuda:0')`，没有 `device_ids` 参数
4. **无 sync_batchnorm**：无 BatchNorm 同步机制
5. **EMA 与 DDP 冲突**：EMA 模型在多卡环境下需要同步状态，但当前实现未处理

#### DexMani Policy 的"伪多卡"限制分析

即使将 `device: cuda:0` 改为使用 `DataParallel`，也会遇到以下问题：

- **ConditionalUnet1D 的 GroupNorm**：n_groups=8 在多卡上可能导致 shape 不匹配
- **动态 batch 维度**：`flatten(0, 1)` 等操作在 `DataParallel` 下可能出问题
- **EMA 同步**：EMA 权重在多卡间需要 `all_reduce`，当前无此逻辑

---

## 四、多卡训练详细对比表

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| **多卡支持** | **原生支持**（Lightning DDP） | **不支持**（单卡硬编码） |
| **启动方式** | `trainer.devices=[0,1,2,3]` CLI 覆盖 | 无法启动 |
| **策略类型** | `strategy: auto` → DDP | 无 |
| **数据分片** | Lightning 自动 `DistributedSampler` | 无 |
| **梯度同步** | Lightning 自动 `all_reduce` | 无 |
| **BN 同步** | `sync_batchnorm: true` | 无 |
| **日志同步** | `sync_dist=True` 跨进程聚合 | 单进程，无需同步 |
| **进程清理** | 手动 `dist.destroy_process_group()` | 无 |
| **混合精度** | 注释掉了 `precision: 16`（未启用） | 无 |
| **梯度累积** | `accumulate_grad_batches: 1`（未启用） | 无 |
| **EMA** | 无 | 有（但不兼容多卡） |

---

## 五、优化器与学习率调度对比

### 5.1 GalaxeaDP

```yaml
optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.95]         # 低 beta2，适应扩散噪声梯度
  weight_decay: 0.0001
  pretrained_obs_encoder_lr_scale: 1.0

lr_scheduler:
  type: OneCycleLR
  max_lr: ${model.optimizer.lr}
  pct_start: 0.15            # 前 15% 升温
  anneal_strategy: cos       # 余弦退火
  div_factor: 100.0          # 初始 lr = 1e-6
  final_div_factor: 1000.0   # 最终 lr = 1e-7
```

**参数分组策略**（`diffusion_unet_image_policy.py:248-274`）：
```python
# Group 1: 扩散模型 + 其他参数 → 默认 lr
# Group 2: 预训练 ResNet18 视觉编码器 → lr × lr_scale
param_groups = [
    {"params": other_params, "lr": cfg.lr, ...},
    {"params": pretrained_obs_encoder_params, "lr": cfg.lr * cfg.lr_scale, ...},
]
```

**在多卡下的行为**：
- 每张卡的 optimizer 是独立的（因为梯度已通过 `all_reduce` 同步）
- 学习率调度器由 Lightning 管理，基于 `trainer.estimated_stepping_batches` 计算总步数
- **总步数 = global_batch_size × max_steps**，不受卡数影响

### 5.2 DexMani Policy

```yaml
optimizer:
  lr: 1e-4
  obs_lr: 1e-4              # 观测编码器学习率（未做差异化）
  weight_decay: 1e-6
  betas: [0.95, 0.999]

lr_scheduler: cosine
lr_warmup_steps: 500
```

**参数分组策略**（`BaseAgent.configure_optimizer()`）：
```python
# Group 1: action_decoder.model.get_optim_groups(weight_decay)  # 按层分组
# Group 2: obs_encoder.requires_grad params → 使用独立 obs_lr
# 但实际上 lr 和 obs_lr 默认相同（1e-4），没有差异化
```

**调度器**：
- 余弦退火 + 500 步 warmup
- 手动 `scheduler.step()` 调用

### 5.3 对比

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 优化器 | AdamW (β2=0.95) | AdamW (β2=0.999) |
| weight_decay | 1e-4 | 1e-6（弱得多） |
| 参数分组 | 扩散模型 vs 预训练编码器（不同 lr） | action_decoder vs obs_encoder（相同 lr） |
| 调度器 | OneCycleLR（升温 + 余弦退火） | 余弦退火 + 固定 warmup |
| 初始 lr | 1e-6（div_factor=100） | warmup 从 0 线性增长 |
| 最终 lr | 1e-7 | 接近 0 |

---

## 六、数据加载机制对比

### 6.1 GalaxeaDP

```
LeRobotDataset (HuggingFace 格式, zarr/视频存储)
    │
    ├─► delta_timestamps 时间窗口化
    │   ├─► obs: [0/fps] → 当前帧
    │   └─► action: [0/fps, 1/fps, ..., chunk_size/fps] → 未来动作
    │
    ├─► 周期性 train/val 分割
    │   (每 ratio 个 episode 跳过 1 个给 val)
    │
    ├─► use_cache: 内存缓存（适合小数据集）
    │
    └─► DataLoader(
            batch_size=64,
            num_workers=16,
            persistent_workers=True,
            pin_memory=True
        )
```

### 6.2 DexMani Policy

```
Zarr 文件 (robot_data/sim/<task>.zarr)
    │
    ├─► ReplayBuffer.copy_from_path()  # 内存映射
    │
    ├─► SequenceSampler (Numba-JIT 索引)
    │   ├─► 时序窗口采样 (horizon 长度)
    │   ├─► 边界 Padding (pad_before/pad_after)
    │   └─► episode_mask train/val 分割
    │
    ├─► 随机选择 n_val 个 episode 作为 val
    │
    └─► DataLoader(
            batch_size=256,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
```

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 存储格式 | LeRobot (zarr/视频) | Zarr |
| 时间窗口 | `delta_timestamps`（灵活时间戳） | `SequenceSampler`（固定 horizon） |
| 采样加速 | 内存缓存 (`use_cache`) | Numba-JIT 编译索引 |
| Padding | 视频格式不支持负索引 | `pad_before` 复制首帧，`pad_after` 复制末帧 |
| Batch Size | 64 | 256（大 4 倍） |
| Workers | 16 | 8 |

---

## 七、训练基础设施对比

### 7.1 EMA（指数移动平均）

| | GalaxeaDP | DexMani Policy |
|---|-----------|----------------|
| 是否使用 | **否** | **是** |
| 实现 | - | `EMAModel` (power=0.75, max_value=0.9999) |
| 用途 | - | 验证 + 仿真评估都使用 EMA 模型 |
| ManiFlow teacher | - | consistency 分支使用 EMA 作为 teacher |

### 7.2 Checkpoint 管理

| | GalaxeaDP | DexMani Policy |
|---|-----------|----------------|
| 机制 | Lightning `ModelCheckpoint` | 原子保存 + JSON manifest |
| 频率 | 每 5000 步 | 每 `checkpoint_interval_epochs` |
| Top-K | `save_top_k=-1`（全保存） | `save_top_k=3`（top-3） |
| 软链接 | `save_last: link` → `last.ckpt` | `latest.pt` 软链接 |
| 路径 | `${output_dir}/checkpoints/step_XXXXX.ckpt` | `experiments/<policy>/<task>/<date>/latest.pt` |

### 7.3 日志系统

| | GalaxeaDP | DexMani Policy |
|---|-----------|----------------|
| WandB | `WandBLogger`（Lightning 集成） | 手动 WandB API |
| 日志文件 | Hydra 自动 `train.log` | JSONL 追加式日志 |
| 训练指标 | `train/_loss`, `train/diffuse_loss/dim_XX` | `train/loss`, `train/action_mse_error` |
| 频率 | `log_every_n_steps: 50` | `log_interval_steps: 50` |

---

## 八、为什么 DexMani Policy 没有多卡训练？

### 8.1 架构限制

1. **自研 Trainer 无分布式逻辑**：DexMani 的训练循环是手动 for 循环，没有 `torch.distributed` 的集成
2. **EMA 同步缺失**：在多卡下，EMA 模型的更新需要在各卡间同步状态
3. **设备硬编码**：模型 `.to('cuda:0')` 和 DataLoader 无 `DistributedSampler`
4. **观测编码器插件化**：`ObsEncoder` 通过 registry 动态构建，不同 encoder（PointNet、PointNext）对分布式的支持需要单独验证

### 8.2 设计选择

DexMani Policy 面向灵巧手操作场景，模型规模相对较小（ConditionalUnet1D: 256→512→1024），单卡即可满足训练需求。而 GalaxeaDP 面向双臂机器人，需要处理多路高分辨率图像输入，显存需求更大，因此原生需要多卡支持。

### 8.3 为 DexMani 添加多卡训练的最佳方案

如果要为 DexMani Policy 添加多卡训练，有两种方案。选择取决于改动容忍度和长期规划。

#### 方案 A：迁移到 PyTorch Lightning

**改动量：大（重写 Trainer + Workspace + Checkpoint）**

```
需要重写的部分：
├── Trainer.train() 手动循环 → LightningModule.training_step()
├── TrainWorkspace → Lightning 的 default_root_dir + callbacks
├── CheckpointIO → Lightning ModelCheckpoint
├── WandB 手动调用 → WandBLogger
└── EMA → 需要自定义 Callback 实现
```

**好处**：
- DDP、日志、checkpoint 全自动，后续几乎零维护
- 直接获得 `trainer.devices=[0,1,2,3]` 的能力
- 和 GalaxeaDP 统一技术栈，两个项目可以共享基础设施

**代价**：
- DexMani 的周期性调度逻辑（`sample_interval` / `val_interval` / `eval_interval` 各自独立）在 Lightning 中没有直接对应物，需要自定义 Callback
- ManiFlow 的 EMA teacher consistency 分支需要特殊处理（Lightning 默认不支持训练中用 EMA 模型生成训练目标）
- 评估集成到训练循环这一点与 Lightning 的 `validation_step` 范式不同（仿真环境是外部依赖，不是数据）
- `workspace.py` 的输出目录管理、JSONL 日志、eval record 全部要重写

#### 方案 B：在现有 Trainer 上原地集成 DDP（推荐）

**改动量：中等（集中在 Trainer 和 DataLoader）**

核心改动只有三处：

**（1）Trainer 初始化 — 分布式初始化**
```python
def __init__(self, ...):
    self.world_size = int(os.environ.get('WORLD_SIZE', 1))
    self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    self.is_distributed = self.world_size > 1
    
    if self.is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.local_rank)
    self.device = torch.device(f'cuda:{self.local_rank}')
```

**（2）数据加载 — DistributedSampler + DDP 包装**
```python
# build_train_components()
if is_distributed:
    sampler = DistributedSampler(dataset, shuffle=True, seed=cfg.training.seed)
    train_loader = DataLoader(dataset, sampler=sampler, ...)
    model = DDP(model, device_ids=[local_rank],
                find_unused_parameters=False)
```

**（3）训练循环 — epoch 同步 + rank 0 仿真评估**
```python
for epoch in range(num_epochs):
    if is_distributed:
        train_loader.sampler.set_epoch(epoch)  # DDP 必需
    
    for batch in train_loader:
        loss, log_dict = model.compute_loss(batch)
        loss.backward()
        clip_grad_norm_(max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if use_ema:
            ema_updater.step(model)
    
    # 周期性任务：仿真评估只在 rank 0 执行
    if epoch % eval_interval == 0:
        if self.local_rank == 0:
            eval_metrics = env_runner.run(ema_model)
        if is_distributed:
            dist.barrier()  # 其他进程等待
```

**额外需要同步的部分**：

| 部分 | 处理方式 |
|------|----------|
| EMA 更新 | 模型权重已由 DDP 同步，EMA 在各卡上自动一致（不需要额外 `all_reduce`） |
| 日志写入 | 只在 rank 0 写 WandB 和 JSONL |
| Checkpoint 保存 | 只在 rank 0 保存（`if self.local_rank == 0`） |
| 仿真评估 | 只在 rank 0 执行，其他进程 `barrier` 等待 |
| normalizer 计算 | 需要 `all_reduce` min/max 统计量，或只在 rank 0 计算后 `broadcast` |

**改动文件清单**：

| 文件 | 改动 |
|------|------|
| `training/common/trainer.py` | 添加 DDP 初始化、`barrier`、rank 0 限定 |
| `training/common/dataset_builder.py` | 添加 `DistributedSampler` |
| `training/common/workspace.py` | 日志写入限定 rank 0 |
| `training/common/checkpoint_io.py` | 保存限定 rank 0 |
| `agents/base.py` | 确保 normalizer 跨卡同步 |

**启动方式**：
```bash
torchrun --nproc_per_node=4 dexmani_policy/train.py --config-name dp3
```

#### 推荐方案 B 的理由

1. **DexMani 的 Trainer 并不简单**——它嵌入了周期性调度、EMA、ManiFlow consistency 蒸馏等逻辑。这些不是 Lightning 的 standard hook 能覆盖的。
2. **改动集中且可控**——核心改动只在 DataLoader 的 sampler 和模型的 DDP 包装，约 100 行代码。
3. **保留现有评估集成**——DexMani 在训练中周期跑仿真评估是有意为之的设计，不需要为了迁就框架而拆出去。
4. **EMA 天然兼容**——DDP 同步模型权重后，各卡上的 EMA 更新结果一致，无需额外 `all_reduce`。

如果未来 DexMani 的功能需求越来越复杂、训练基础设施维护成本变高，再考虑迁移到 Lightning 不迟。

---

## 九、DDP 与在线仿真评估的关系

DDP 会为每张卡启动一个进程，所有进程执行相同的代码。如果直接在训练循环中加入仿真评估，会遇到冲突：

### 9.1 冲突点

**（1）多进程重复创建环境**
```python
# 如果在 validation_step 中直接做仿真评估
def validation_step(self, batch, batch_idx):
    env = gym.make("MySimEnv")  # 每张卡都会创建一个环境！
    # 4 卡 = 4 个仿真环境同时跑，资源浪费 + 结果重复
```

**（2）仿真环境通常不支持分布式**
- 物理引擎（MuJoCo、PyBullet、IsaacGym）可能不支持多进程同时初始化
- GPU 仿真时显存冲突（多个进程争抢 GPU）
- 环境变量/随机种子在多进程下不一致

**（3）结果聚合困难**
```
GPU 0: 20/25 = 80%    GPU 1: 21/25 = 84%
GPU 2: 19/25 = 76%    GPU 3: 22/25 = 88%
需要 barrier + all_reduce 才能拿到正确结果
```

### 9.2 两种项目的处理方式

**GalaxeaDP**：仿真评估完全独立于训练，通过 `eval_sim.py` 单独启动，只在单卡上跑，避免了 DDP 的所有问题。

**DexMani Policy**：自研 Trainer 是单进程，直接在训练循环中周期性调用 `env_runner.run()`，简单直接。但如果改成 DDP，同样会遇到上述问题。

### 9.3 DDP 下做在线评估的正确做法

```python
# 只在 rank 0 进程执行仿真评估
def on_validation_epoch_end(self):
    if self.trainer.global_rank != 0:
        return  # 非 rank 0 直接跳过
    success_rate = run_simulation_eval(self.policy)
    self.log("eval/success_rate", success_rate, sync_dist=False)
```

但仍有遗留问题：
- **显存冲突**：rank 0 跑仿真环境时，训练循环仍在执行，显存可能不够
- **阻塞训练**：仿真评估很慢，会拖慢所有卡（其他卡在 `barrier` 等待）
- **随机种子**：多进程下仿真环境的种子难以保证一致性

### 9.4 稳健方案

仿真评估放在训练循环外部：
1. **训练结束后单独跑**（GalaxeaDP 目前的做法）
2. **用独立的评估进程**，与训练进程并行但不干扰
3. **训练时定期保存 checkpoint**，由外部脚本监听并自动评估

**结论**：DDP 下可以在线仿真测评，但工程上不建议在训练循环内做。独立评估脚本是最稳健的选择。

## 十一、评测机制深度对比

### 11.1 评估体系架构

GalaxeaDP 提供**两级独立评估管线**，DexMani Policy 则将评估集成到训练循环中：

```
GalaxeaDP:
┌─────────────────────────────────────────────────────────────┐
│                    训练完成后                                │
│                          │                                  │
│              ┌───────────┴───────────┐                      │
│              ▼                       ▼                      │
│  eval_open_loop.py          eval_sim.py                     │
│  开环评估                   闭环仿真评估                     │
│  数据集 val split           Gymnasium 仿真环境               │
│  离线批量推理               在线循环推理                     │
│  轨迹拟合度可视化           成功率 + 视频                    │
│  无仿真依赖                 需要 GalaxeaManipSim             │
└─────────────────────────────────────────────────────────────┘

DexMani Policy:
┌─────────────────────────────────────────────────────────────┐
│                  训练循环内部                                │
│                          │                                  │
│        ┌─────────────────┼─────────────────┐                │
│        ▼                 ▼                 ▼                │
│   sample_interval   val_interval     eval_interval          │
│   action MSE         val loss        SimRunner              │
│   (单 batch)        (val set)       (仿真环境)              │
│                                                              │
│   + 独立 eval_sim.py (训练后手动运行)                        │
│   - 无独立开环评估脚本                                       │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 开环评估

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 独立脚本 | **有** (`eval_open_loop.py`) | **无** |
| 可视化 | Plotly HTML（交互式，5 色循环区分推理时刻） | 无 |
| 定量指标 | 无（仅视觉） | 训练中 `train/action_mse_error` |
| Episode 分组 | 按 `episode_data_index` 逐 episode 输出 | 无 |
| 数据源 | val split DataLoader | 训练 batch 抽样 |
| 使用阶段 | 训练后快速验证 | 训练中周期性监控 |

**GalaxeaDP 开环评估的关键设计**：
- GT 只取第一步 `[:, 0, :]`，预测保留完整 chunk `(N, horizon, action_dim)`
- Plotly 5 色循环：同一次推理产生的 chunk 用同一颜色，直观展示 chunk 边界处的连续性
- 按 episode 分组输出，每个 action 维度一个 HTML 文件

### 11.3 仿真评估

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 评估框架 | `gymnasium.make()` | `SimRunner` 动态导入 `dexmani_sim.envs` |
| 种子策略 | **固定 `seed=42`**（100 次评估完全相同） | 从文件读取种子列表（多样化） |
| Action Chunking | 取前 16/32 步 | 取 `n_action_steps` 步（跳过 `n_obs_steps-1`） |
| 图像预处理 | `cv2.resize(224×224)` | 数据集原生尺寸 |
| 多 denoise 步测试 | 无（固定 20 步） | `denoise_timesteps_list` 测试多步数 |
| EMA 模型评估 | **无** | 默认启用（`use_ema_for_eval=true`） |
| 评估集成 | **独立脚本，训练后手动运行** | **集成到训练循环（周期性自动触发）** |
| 视频编码 | H.264 (libx264, yuv420p) | 未指定 |
| 评估产物 | mp4 + png + info.json | mp4 + eval_metrics.json + eval record |

### 11.4 发现的关键问题

#### （1）图像预处理分布偏移（GalaxeaDP）

- **训练时**：`Resize(252×336) → RandomCrop/CenterCrop(240×320)`
- **仿真评估时**：`cv2.resize(224×224)`

策略在训练时看到的是 240×320 的裁剪图像，在评估时看到的是 224×224 的 resize 图像，**长宽比和视野都不同**。这可能导致策略在仿真评估中的表现低于预期。

#### （2）固定种子缺乏泛化性测试（GalaxeaDP）

GalaxeaDP 的 `env.reset(seed=42)` 使 100 次评估的初始条件完全相同。这保证了不同 checkpoint 之间的**可对比性**，但无法反映策略在多样化初始条件下的**泛化能力**。DexMani 从文件读取种子列表的做法更合理。

#### （3）评估集成方式的取舍

| 方式 | 优势 | 劣势 |
|------|------|------|
| 独立脚本（GalaxeaDP） | 不受 DDP 影响，可随时单独运行，不拖慢训练 | 训练过程中无法追踪性能变化，需手动触发 |
| 训练内集成（DexMani） | 自动监控策略性能变化，发现退化趋势 | 评估慢会拖慢训练，DDP 下需要 rank 0 限定 |

#### （4）EMA 评估缺失（GalaxeaDP）

GalaxeaDP 不使用 EMA 模型。DexMani 默认启用 EMA 评估（`use_ema_for_eval=true`），EMA 通常能提供更平滑、更稳定的预测，在扩散策略中尤其有效。

### 11.5 两种评估的互补关系

```
开环高拟合 + 闭环高成功率 = 模型健康 ✓
开环高拟合 + 闭环低成功率 = 数据集有问题 / 仿真-现实差距 / 预处理不一致
开环低拟合 + 闭环高成功率 = 偶然（可能任务太简单）
开环低拟合 + 闭环低成功率 = 训练失败 ✗
```

开环评估的价值在于在跑仿真之前快速发现训练问题；仿真评估是唯一能反映策略实际能力的指标。

## 十二、总结

| 特性 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 训练框架 | PyTorch Lightning | 自研 Trainer |
| **多卡训练** | **原生支持**（DDP，`trainer.devices=[0,1,2,3]`） | **不支持**（单卡硬编码） |
| 推荐 DDP 方案 | 开箱即用 | 原地集成 DDP（~100 行改动） |
| BN 同步 | `sync_batchnorm: true` | 无 |
| 日志同步 | `sync_dist=True` | 单进程 |
| EMA | 无 | 有（单卡，DDP 兼容） |
| 数据分片 | DistributedSampler 自动 | 无 |
| 优化器差异化 | 预训练编码器低 lr | 相同 lr |
| 调度器 | OneCycleLR | Cosine + warmup |
| Checkpoint | 全保存（每 5000 步） | Top-3 + latest |
| 开环评估 | 独立脚本 + Plotly 交互式 | 无 |
| 仿真评估 | 独立脚本（固定种子） | 训练内集成（多样化种子 + EMA） |
| 图像预处理一致性 | **不一致**（240×320 vs 224×224） | 一致 |

**核心结论**：GalaxeaDP 的多卡训练能力来自 Lightning 框架的内置 DDP 支持，只需修改 `trainer.devices` 即可水平扩展；DexMani Policy 的自研训练循环功能完善（EMA、周期性评估、checkpoint top-k），但在分布式训练方面存在空白。DexMani 最佳的多卡路径是在现有 Trainer 上原地集成 DDP（约 100 行改动），而非迁移到 Lightning，因为 DexMani 的周期性调度和 ManiFlow consistency 蒸馏逻辑不适合 Lightning 的标准 hook 范式。DDP 下的在线仿真评估需要限定在 rank 0 进程执行，但更稳健的做法是保持评估独立于训练。评估方面，GalaxeaDP 的开环评估体系完善但图像预处理存在分布偏移，DexMani 的训练内集成评估方式更利于实时监控但 DDP 兼容性差。
