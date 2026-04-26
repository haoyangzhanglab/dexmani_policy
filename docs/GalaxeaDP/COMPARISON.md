# DexMani Policy vs GalaxeaDP 深度对比与融合指南

> 核心问题：GalaxeaDP 的理念机制中，哪 5 条对提升 DexMani Policy 的代码质量和通用性帮助最大？

---

## 目录

- [一、Top 5 最具价值的 GalaxeaDP 机制](#一top-5-最具价值的-galaxeadp-机制)
  - [机制 1：PyTorch Lightning 训练框架](#机制-1pytorch-lightning-训练框架)
  - [机制 2：组合式 Hydra 配置系统](#机制-2组合式-hydra-配置系统)
  - [机制 3：SignalTransform 信号转换层](#机制-3signaltransform-信号转换层)
  - [机制 4：独立开环评估脚本](#机制-4独立开环评估脚本)
  - [机制 5：LightningDataModule 标准化数据接口](#机制-5lightningdatamodule-标准化数据接口)
- [二、全维度对比表](#二全维度对比表)
- [三、模型架构对比](#三模型架构对比)
- [四、数据集设计对比](#四数据集设计对比)
- [五、训练流程对比](#五训练流程对比)
- [六、评测方法对比](#六评测方法对比)
- [七、配置系统对比](#七配置系统对比)
- [八、设计优劣势对比](#八设计优劣势对比)
- [九、融合路线图](#九融合路线图)
- [十、总结](#十总结)

---

## 一、Top 5 最具价值的 GalaxeaDP 机制

### 机制 1：PyTorch Lightning 训练框架

**影响等级：最大**

**GalaxeaDP 的做法**（`src/train.py` + `src/models/dp_bc_module.py`）：

```python
# src/train.py — 训练入口
def train(cfg: DictConfig) -> tuple[dict, dict]:
    L.seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.data)    # LightningDataModule
    model = hydra.utils.instantiate(cfg.model)         # LightningModule
    callbacks = instantiate_callbacks(cfg.callbacks)   # Checkpoint / LR Monitor / Progress
    logger = instantiate_loggers(cfg.logger)           # WandB
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.5,    # 自动梯度裁剪
        sync_batchnorm=True,      # DDP 同步 BatchNorm
    )
    trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path"))
```

```python
# src/models/dp_bc_module.py — LightningModule
class DiffusionPolicyBCModule(LightningModule):
    def training_step(self, batch, batch_idx):
        loss, loss_log = self.model_step(batch)
        self.log_dict({"train/_loss": loss}, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(loss_log, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.policy.get_optimizer(self.hparams.optimizer)
        lr_scheduler = build_scheduler(...)
        return {"optimizer": optimizer, "lr_scheduler": {...}}
```

**DexMani Policy 的现状**（`training/trainer.py`）：

```python
# 手动管理的训练循环
class Trainer:
    def train_one_step(self, batch):
        loss, log_dict = self.model.compute_loss(batch)
        loss.backward()
        self.clip_grad_norm(max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        if self.ema_updater:
            self.ema_updater.step()
        return loss
```

**差距分析**：

| 维度 | GalaxeaDP (Lightning) | DexMani Policy (手动) |
|------|----------------------|----------------------|
| 多卡训练 | `trainer.devices=[0,1,2,3]` 一行开启 | 硬编码 `cuda:0`，不支持 |
| 梯度裁剪 | `gradient_clip_val=0.5` 配置化 | 手动 `clip_grad_norm_(1.0)` |
| Checkpoint | `ModelCheckpoint` 回调，自动按 step 命名 | 手动原子保存 + top-k manifest |
| 日志同步 | `self.log_dict(sync_dist=True)` 自动 DDP 同步 | 手动收集，单卡假设 |
| 断点续训 | `trainer.fit(..., ckpt_path="path")` | 手动 `load_for_resume()` |
| LR 监控 | `LearningRateMonitor` 回调自动记录 | 无 |
| 模型摘要 | `RichModelSummary` 自动打印参数统计 | 无 |
| 混合精度 | `precision: 16-mixed` 一行开启 | 不支持 |

**对 DexMani Policy 的价值**：迁移到 Lightning 后，**多卡训练**是最大收益（DiT-X 12层模型在单卡上显存压力大），其次是自动化的基础设施（checkpoint、LR 监控、DDP 同步）。同时，Lightning 的 `LightningModule` 接口将策略模型与训练循环解耦，代码更清晰。

---

### 机制 2：组合式 Hydra 配置系统

**影响等级：大**

**GalaxeaDP 的做法**（分层组合）：

```
configs/
├── train.yaml              # 基础配置：trainer + callbacks + logger + hydra
├── data/
│   └── r1pro/lerobot_eef.yaml   # 数据配置：dataset + shape_meta + batch_size
├── model/
│   └── unet_aug.yaml            # 模型配置：policy + optimizer + lr_scheduler
└── task/
    └── sim/R1ProBlocksStackEasy_eef.yaml   # 任务配置：组合 data + model

# 任务配置只需引用 data + model
defaults:
  - override /data: r1pro/lerobot_eef
  - override /model: unet_aug_idpenc
```

**DexMani Policy 的做法**（扁平单文件）：

```
configs/
├── dp3.yaml        # 完整定义所有：policy + task + agent + optimizer + training + eval
├── dp.yaml         # 重复大量相同字段
├── moe_dp3.yaml    # 重复大量相同字段
└── maniflow.yaml   # 重复大量相同字段
```

**差距分析**：

| 维度 | GalaxeaDP (组合式) | DexMani Policy (扁平式) |
|------|-------------------|----------------------|
| 配置复用 | data/model 独立定义，任务组合引用 | 每个策略文件完整复制公共字段 |
| 新增策略 | 新建 model YAML，引用已有 data | 复制整个 YAML，手动修改差异部分 |
| 新增任务 | 新建 task YAML，引用已有 data+model | 需要复制并修改整个 config |
| 配置覆盖 | `task=sim/xxx trainer.devices=[0]` 层次清晰 | `seed=233 task_name=multi_grasp` 扁平但无结构 |
| 维护成本 | 低（改一处，所有任务生效） | 高（需同步修改多个 YAML） |

**对 DexMani Policy 的价值**：组合式配置是**工程可维护性**的关键。当前 4 个 config 文件有大量重复字段（`horizon`、`n_obs_steps`、`batch_size` 等），改为组合式后，新增策略只需定义 `agent` 和 `decoder` 差异，其他复用已有配置。

---

### 机制 3：SignalTransform 信号转换层

**影响等级：大**

**GalaxeaDP 的做法**（`src/models/policy/signal_transform/signal_transform.py`）：

```python
class SignalTransform(nn.Module):
    def __init__(self, action_keys, qpos_keys, rotation_type, use_relative_control):
        self.pose_rotation_transformer = PoseRotationTransformer(rotation_type)  # quaternion/6d/9d
        self.relative_pose_transformer = RelativePoseTransformer()
        self.use_relative_control = use_relative_control

    def forward(self, batch):
        # 1. 旋转表示转换：欧拉角/四元数 → 6D/9D
        cur_action = self.pose_rotation_transformer.forward(cur_action)
        # 2. 相对控制：动作相对于当前位姿或 episode 起点
        if self.use_relative_control:
            base_pose = batch["obs"][obs_key][:, -1:, :]
            cur_action = self.relative_pose_transformer.forward(cur_action, base_pose)
        ...
```

**DexMani Policy 的现状**：**不存在此模块**。灵巧手直接关节空间控制（joint state + action dim=19），不涉及末端执行器位姿的旋转表示。

**对 DexMani Policy 的价值**：虽然当前灵巧手场景不需要旋转转换，但 SignalTransform 的设计理念——**数据预处理与模型解耦**——有重要借鉴意义：

1. **归一化/反归一化**当前散落在 `BaseAgent.preprocess()` 和 `predict_action()` 中，可提取为独立模块
2. **观测切片**（`n_obs_steps` 切片 + `flatten(0,1)`）硬编码在 `preprocess()` 中，可配置化
3. **未来扩展性**：如果 DexMani Policy 需要支持末端执行器控制（如灵巧手 + 机械臂协同），SignalTransform 的旋转转换和相对控制可直接复用

---

### 机制 4：独立开环评估脚本

**影响等级：中**

**GalaxeaDP 的做法**（`src/eval_open_loop.py`）：

```python
# 独立脚本，预测 vs 真实对比
# 1. 加载 checkpoint
# 2. 遍历验证集
# 3. 计算 action MSE、可视化对比
# 4. 生成 plots
```

**DexMani Policy 的现状**：仅在训练循环中计算 `train/action_mse_error`（`Trainer.compute_action_mse_for_one_batch()`），**无独立脚本**。

```python
# 仅在 Trainer.train_one_step() 后偶尔调用
def compute_action_mse_for_one_batch(self, agent, batch):
    with torch.no_grad():
        pred_action = agent.predict_action(obs)["pred_action"]
        mse = F.mse_loss(pred_action, gt_action)
```

**对 DexMani Policy 的价值**：
- 开环评估是**策略调试的核心工具**：在仿真评估前，先验证模型是否能从数据中复制专家行为
- 独立脚本支持**批量评估不同 checkpoint**、生成可视化对比图
- 当前仅在训练内计算 MSE，无法灵活调整评估参数（不同的 denoise_timesteps、不同数据集）

---

### 机制 5：LightningDataModule 标准化数据接口

**影响等级：中**

**GalaxeaDP 的做法**（`src/data/base_datamodule.py`）：

```python
class BaseDataModule(LightningDataModule):
    def __init__(self, train, val, test=None, **kwargs):
        self.save_hyperparameters(logger=False, ignore=["train", "val", "test"])
        self.data_train = train
        self.data_val = val
        self.data_test = test

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size_train,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,  # 加速 worker 初始化 3x
            pin_memory=self.hparams.pin_memory,
            collate_fn=_collate_fn,
        )
```

**DexMani Policy 的现状**（`training/trainer.py` + `datasets/base_dataset.py`）：

```python
# 在 build_train_components() 中手动构建
dataset = hydra.utils.instantiate(cfg.dataset)
normalizer = dataset.get_normalizer()
train_loader = DataLoader(dataset, **cfg.dataloader)
val_loader = DataLoader(val_set, **cfg.val_dataloader)
# DataLoader 配置散落在 YAML 中
```

**DexMani Policy 的 DataLoader 配置**（`dp3.yaml`）：

```yaml
dataloader:
  batch_size: 256
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  shuffle: true
  drop_last: true
val_dataloader:
  batch_size: 256
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  shuffle: false
  drop_last: false
```

**差距分析**：

| 维度 | GalaxeaDP (DataModule) | DexMani Policy (手动) |
|------|----------------------|----------------------|
| 生命周期管理 | `prepare_data` → `setup` → `train_dataloader` | 手动在 `build_train_components()` 中构建 |
| DDP 兼容 | `prepare_data` 只在 rank 0 执行 | 无 DDP 意识 |
| hyperparams 存储 | `save_hyperparameters()` 自动存入 ckpt | 手动管理 |
| 配置冗余 | train/val 共用 DataModule | `dataloader` 和 `val_dataloader` 大量重复 |

**对 DexMani Policy 的价值**：迁移到 DataModule 后，**配置冗余**可消除（train/val 共用参数只需定义一次），且在迁移到 Lightning 时可直接作为 `trainer.fit(model, datamodule)` 的参数。

---

## 二、全维度对比表

### 2.1 项目定位

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **定位** | 灵巧手操作策略学习框架 | 双臂机器人 Diffusion Policy 开源实现 |
| **目标机器人** | 灵巧手（关节空间，action_dim=19） | R1 系列双臂（EE pose + gripper，action_dim=16） |
| **仿真环境** | `dexmani_sim`（独立安装） | `GalaxeaManipSim`（独立安装） |
| **数据集格式** | 自定义 Zarr | LeRobot（HuggingFace Hub） |
| **传感器** | 点云 + 关节状态 / RGB + 关节状态 | 多相机 RGB + 关节状态 |

### 2.2 技术栈

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **训练框架** | 自定义 `Trainer` 类 | PyTorch Lightning |
| **配置系统** | Hydra（扁平单文件） | Hydra（组合式：data+model+task） |
| **扩散调度器** | DDIM（推理）/ DDPM / Rectified Flow | DDPM |
| **视觉编码器** | PointNet / PointNext / ResNet / CLIP / DINOv2 | ResNet18 |
| **扩散 Backbone** | ConditionalUnet1D / DiTXFlowMatch | ConditionalUnet1D |
| **多卡训练** | 不支持 | 支持（DDP） |
| **混合精度** | 不支持 | 支持（`precision: 16-mixed`） |

### 2.3 策略变体

| DexMani Policy | GalaxeaDP |
|---------------|-----------|
| 1. DP3Agent — 点云 + UNet + DDIM | 1. DiffusionUnetImagePolicy — 多相机 RGB + UNet + DDPM |
| 2. DPAgent — RGB + UNet + DDIM | |
| 3. MoEDP3Agent — 点云 + MoE + UNet + DDIM | |
| 4. ManiFlowAgent — 点云 + DiT-X + FlowMatch + Consistency | |

---

## 三、模型架构对比

### 3.1 数据流

```
DexMani Policy:
obs_dict {point_cloud: (B,T,N,3), joint_state: (B,T,D)}
    │
    ├─► BaseAgent.preprocess()    # normalize + slice n_obs_steps + flatten B*T
    │       ├─► self.normalizer.normalize(obs_dict)
    │       ├─► slice [:, :n_obs_steps]
    │       └─► flatten(0, 1)
    ├─► ObsEncoder                # → cond: (B, out_dim) [film] 或 (B, T, out_dim) [cross_attn]
    └─► ActionDecoder             # DDIM Diffusion 或 FlowMatch+Consistency

GalaxeaDP:
Observation (images + state)
    │
    ├─► SignalTransform           # 旋转表示转换 + 相对位姿控制
    ├─► Normalize                 # min-max 归一化
    ├─► ResNetImageEncoder        # 多相机编码 + State MLP → Global Condition
    └─► DiffusionUnetImagePolicy
                ├─► DDPMScheduler (noise injection)
                └─► ConditionalUnet1D (denoise)
```

### 3.2 观测编码器

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **视觉输入** | 点云 (N, 3) 或 RGB | 多相机 RGB（head + left_wrist + right_wrist） |
| **编码器** | PointNet / MultiStagePointNet / PointNext / ResNet / CLIP / DINOv2 / SigLIP | ResNet18（预训练） |
| **State 编码** | StateMLP → 64-dim | 直接拼接 + fusion MLP |
| **条件注入** | FiLM `(B, out_dim * n_obs_steps)` / Cross-Attn `(B, n_obs_steps, out_dim)` | 全局条件向量 |
| **时间窗口** | `n_obs_steps=2` | `vision_obs_size=1` |
| **下采样** | FPS（最远点采样）| 无 |

### 3.3 动作解码器

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **Diffusion** | DDIM（10 步推理，100 步训练） | DDPM（20 步） |
| **Beta 调度** | `squaredcos_cap_v2` | `squaredcos_cap_v2` |
| **预测类型** | `sample`（直接预测 x_0） | `epsilon`（预测噪声） |
| **Backbone** | UNet (256→512→1024) / DiT-X (768-dim, 12层) | UNet (512→1024→2048) |
| **替代方案** | FlowMatch + Consistency Distillation | 无 |
| **Action Chunking** | 截取 `n_action_steps` 从 `pred[:, n_obs_steps-1:]` | 完整输出 `chunk_size=32` |

---

## 四、数据集设计对比

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **格式** | Zarr（内存映射） | LeRobot（Parquet + 视频） |
| **路径** | `robot_data/sim/<task>.zarr`（相对） | HuggingFace Hub 路径 |
| **采样器** | `SequenceSampler`（Numba-JIT） | `delta_timestamps` |
| **Train/Val** | 随机选 `val_ratio` episode | `val_set_proportion=0.05` |
| **Padding** | 首帧向前复制 / 末帧向后复制 | LeRobot 内部处理 |
| **归一化** | `LinearNormalizer`（Min-Max → [-1, 1]） | Min-Max → [-1, 1] |
| **数据增强** | 点云增强 / RGB 增强 | RGB 增强（ColorJitter + RandomCrop） |

---

## 五、训练流程对比

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **入口** | `train.py`（`@hydra.main`） | `src/train.py`（Hydra + Lightning） |
| **训练循环** | 自定义 `Trainer` 类 | Lightning `Trainer` |
| **多卡** | 硬编码 `cuda:0` | `trainer.devices=[0,1,2,3]` |
| **Optimizer** | AdamW (lr=1e-4, wd=1e-6) | AdamW (lr=1e-4, wd=1e-4) |
| **LR 调度** | Cosine + Warmup (500步) | OneCycleLR (pct_start=0.15) |
| **Batch Size** | 256 / 128 | 64 |
| **Epoch 数** | 1000 | `max_steps=20000` |
| **EMA** | 支持（power=0.75） | 不支持 |
| **Gradient Clip** | 手动 `clip_grad_norm_(1.0)` | `gradient_clip_val=0.5` |
| **Checkpoint** | 原子保存 + top-k manifest | `ModelCheckpoint` 回调 |
| **断点续训** | 手动 `load_for_resume()` | `ckpt_path="path"` |

---

## 六、评测方法对比

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **开环评估** | 仅训练内 MSE，无独立脚本 | `eval_open_loop.py` 独立脚本 |
| **仿真评估** | `eval_sim.py`（argparse） | `eval_sim.py`（Hydra） |
| **启动方式一致性** | 训练（Hydra）≠ 评估（argparse） | 训练 = 评估（均 Hydra） |
| **模型选择** | EMA 模型 | 直接加载 checkpoint |
| **Checkpoint 加载** | `best` / `latest` / top-k manifest | 具体路径 `step_XXXXX.ckpt` |

---

## 七、配置系统对比

| 维度 | DexMani Policy | GalaxeaDP |
|------|---------------|-----------|
| **模式** | 扁平单文件 | 组合式（data + model + task） |
| **文件数** | 4 个（每策略一个） | 分层目录 |
| **复用性** | 低（大量重复字段） | 高（引用覆盖） |
| **新增策略** | 复制整个 YAML | 新建 model YAML |
| **输出目录** | `experiments/<policy>/<task>/<date>_<seed>/` | `out/<task>/<time>/` |

---

## 八、设计优劣势对比

### 8.1 DexMani Policy 独有优势（GalaxeaDP 不具备）

| 优势 | 说明 |
|------|------|
| **点云编码器** | PointNet/MultiStagePointNet/PointNext，支持 FPS 下采样 |
| **多策略变体** | DP3 / DP / MoE-DP3 / ManiFlow 四种 |
| **MoE 插件** | Top-K 稀疏路由 + load balance + entropy 辅助损失 |
| **Flow Match + Consistency** | Rectified Flow + EMA teacher 一致性蒸馏 |
| **条件注入灵活性** | FiLM 和 Cross-Attention 两种模式可选 |
| **EMA 模型** | 指数移动平均 + consistency teacher |
| **DDIM 推理加速** | 10 步推理（vs GalaxeaDP 的 20 步） |

### 8.2 GalaxeaDP 独有优势（DexMani Policy 不具备）

| 优势 | 说明 |
|------|------|
| **Lightning 训练** | 多卡 DDP、自动 checkpoint、gradient clipping、混合精度 |
| **组合式配置** | data + model + task 分层，复用性高 |
| **SignalTransform** | 旋转表示转换 + 相对位姿控制 |
| **独立开环评估** | `eval_open_loop.py` 预测 vs 真实可视化 |
| **LeRobot 集成** | HuggingFace 生态兼容 |
| **训练/评估一致性** | 均通过 Hydra 启动 |
| **生产级基础设施** | LR Monitor、Rich Progress Bar、Model Summary |

### 8.3 共同点

| 共同点 | 说明 |
|--------|------|
| Hydra 配置管理 | 都使用 Hydra |
| Diffusion Policy 核心 | ConditionalUnet1D + DDPM/DDIM |
| Min-Max 归一化 | 都映射到 [-1, 1] |
| 多相机/多模态编码 | 都支持多模态融合 |
| WandB 日志 | 都支持（offline 模式） |
| Beta 调度 | 都使用 `squaredcos_cap_v2` |

---

## 九、融合路线图

### 9.1 Phase 1：迁移到 Lightning（最高优先级）

**目标**：将 DexMani Policy 的自定义 `Trainer` 迁移到 PyTorch Lightning。

**步骤**：

```
1. 创建 LightningModule 包装器
   ├── dexmani_policy/lightning/
   │   ├── __init__.py
   │   ├── agent_module.py      # 类似 DiffusionPolicyBCModule
   │   └── data_module.py       # 类似 BaseDataModule
   │
2. 将 Trainer.train_one_step() 迁移到 training_step()
3. 将 build_train_components() 迁移到 DataModule.setup()
4. 用 ModelCheckpoint 替换手动 checkpoint 逻辑
5. 用 LearningRateMonitor 替换手动 LR 日志
```

**关键代码映射**：

```python
# 当前：Trainer.train_one_step()
# 迁移后：AgentModule.training_step()
class DexmaniAgentModule(L.LightningModule):
    def training_step(self, batch, batch_idx):
        loss, log_dict = self.agent.compute_loss(batch)
        self.log_dict({"train/loss": loss}, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.agent.configure_optimizer(**self.hparams.optimizer)
        return {"optimizer": optimizer, "lr_scheduler": ...}
```

**收益**：
- 多卡训练（`trainer.devices=[0,1]`）
- 自动 checkpoint、LR 监控
- 混合精度（`precision: 16-mixed`）
- 代码量减少 ~300 行（手动训练循环 + checkpoint + 日志）

### 9.2 Phase 2：重构配置系统为组合式

**目标**：将 4 个扁平 YAML 改为 data + model + task 分层。

**步骤**：

```
configs/
├── base.yaml                 # 公共配置：horizon, n_obs_steps, n_action_steps, training loop
├── data/
│   ├── pc.yaml               # 点云数据配置
│   │   sensor_modalities: [point_cloud, joint_state]
│   │   zarr_path: robot_data/sim/${task_name}.zarr
│   └── rgb.yaml              # RGB 数据配置
├── model/
│   ├── dp3.yaml              # DP3: encoder_type=idp3 + ConditionalUnet1D + DDIM
│   ├── dp.yaml               # DP: encoder_type=resnet + ConditionalUnet1D + DDIM
│   ├── moe_dp3.yaml          # MoE-DP3: dp3 + MoE plugin
│   └── maniflow.yaml         # ManiFlow: encoder_type=pointpn + DiTXFlowMatch + FlowMatch
└── task/
    ├── multi_grasp.yaml       # defaults: [/data/pc, /model/dp3]
    ├── pick_apple_messy.yaml  # defaults: [/data/pc, /model/maniflow]
    └── ...
```

**收益**：
- 消除 4 个文件中的 ~70% 重复字段
- 新增策略只需定义差异部分
- 新增任务只需引用已有 data + model

### 9.3 Phase 3：提取 SignalTransform 理念

**目标**：将预处理逻辑从 `BaseAgent.preprocess()` 中提取为独立模块。

**步骤**：

```python
# 新增：dexmani_policy/common/preprocess.py
class DexmaniPreprocess(nn.Module):
    """解耦观测预处理：归一化 + 切片 + 展平"""
    def __init__(self, n_obs_steps, normalizer=None):
        self.n_obs_steps = n_obs_steps
        self.normalizer = normalizer

    def forward(self, obs_dict):
        if self.normalizer:
            obs = self.normalizer.normalize(obs_dict)
        else:
            obs = obs_dict
        # 切片
        obs = {k: v[:, :self.n_obs_steps] for k, v in obs.items()}
        # 展平
        obs = {k: v.flatten(0, 1) for k, v in obs.items()}
        return obs
```

**收益**：
- `BaseAgent` 职责更清晰（仅管理组件生命周期）
- 预处理可独立测试、可配置化
- 为未来扩展（如相对关节控制、时序增强）预留接口

### 9.4 Phase 4：添加独立开环评估脚本

**目标**：创建 `eval_open_loop.py`，支持批量评估 + 可视化。

**步骤**：

```
1. 新建 dexmani_policy/eval_open_loop.py
2. 加载 checkpoint
3. 遍历验证集/测试集
4. 计算 action MSE、轨迹相似度
5. 生成预测 vs 真实对比图（matplotlib）
6. 支持评估不同 denoise_timesteps
```

**收益**：
- 策略调试效率提升（无需启动仿真即可验证）
- 支持批量评估多个 checkpoint
- 生成可视化报告

### 9.5 Phase 5：DataModule 标准化

**目标**：将 DataLoader 构建逻辑迁移到 LightningDataModule。

**步骤**：

```python
# dexmani_policy/lightning/data_module.py
class DexmaniDataModule(L.LightningDataModule):
    def __init__(self, dataset_cfg, dataloader_cfg):
        self.save_hyperparameters()
    
    def setup(self, stage):
        dataset = hydra.utils.instantiate(self.hparams.dataset_cfg)
        self.data_train = dataset
        self.data_val = dataset.get_validation_dataset()
    
    def train_dataloader(self):
        return DataLoader(self.data_train, **self.hparams.dataloader_cfg.train)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, **self.hparams.dataloader_cfg.val)
```

**收益**：
- 消除 `dataloader` / `val_dataloader` 配置冗余
- DDP 兼容（`prepare_data` 只在 rank 0 执行）
- 与 Lightning Trainer 无缝集成

---

## 十、总结

### 两个项目的核心差异

| | DexMani Policy | GalaxeaDP |
|---|---|---|
| **算法深度** | 更深：4种策略、MoE、Flow Match、Consistency | 更浅：1种策略、DDPM |
| **传感器** | 3D（点云） | 2D（RGB） |
| **工程成熟度** | 中等：自定义训练循环、扁平配置、单卡 | 较高：Lightning、组合配置、多卡 |
| **评估体系** | 较弱：无独立开环评估 | 较完善：开环 + 仿真 |
| **可扩展性** | 算法层面强（插件化），工程层面弱 | 工程层面强，算法层面单一 |

### DexMani Policy 应优先采纳的 5 条机制

| 优先级 | 机制 | 预期收益 | 实施难度 |
|--------|------|---------|---------|
| 1 | Lightning 训练框架 | 多卡、自动化基础设施 | 中 |
| 2 | 组合式配置 | 配置复用、维护性 | 低 |
| 3 | SignalTransform 理念 | 解耦预处理、扩展性 | 低 |
| 4 | 独立开环评估 | 策略调试效率 | 低 |
| 5 | DataModule 标准化 | 消除冗余、DDP 兼容 | 低 |

### 一句话总结

**DexMani Policy 在算法层面更丰富（点云、MoE、Flow Match），GalaxeaDP 在工程层面更成熟（Lightning、组合配置、标准化接口）。两者的最佳实践结合，可以构建一个既算法先进又工程稳健的通用策略学习框架。**
