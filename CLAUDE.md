# dexmani_policy

灵巧操作模仿学习策略框架。5 种策略 × 多模态观测 × 单卡/DDP 训练。

## 环境

```bash
conda activate policy
```

## 快速开始

```bash
# 单卡训练
bash scripts/train.sh dp3
bash scripts/train.sh maniflow

# 多任务训练
bash scripts/train_multi_task.sh multitask_dit

# DDP 多卡训练
bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[0,1,2,3]

# 仿真评估
bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233

# 启用数据增强：取消 config 文件末尾 augmentation_cfg 注释即可
```

## 策略矩阵

| Agent | 观测模态 | Encoder | Backbone | Decoder |
|-------|---------|---------|----------|---------|
| DP | RGB + joint_state | DINO/CLIP/ResNet/SigLIP + StateMLP | UNet1D (FiLM) | Diffusion (DDIM) |
| DP3 | 点云 + joint_state | PointNet/iDP3/PointNext + StateMLP | UNet1D (FiLM) | Diffusion (DDIM) |
| MoE | 点云 + joint_state | iDP3 + MoE Router + StateMLP | UNet1D (FiLM) | Diffusion (DDIM) |
| ManiFlow | 点云 + joint_state | PointPN/PointNext Tokenizer + StateMLP | DiT-X (AdaLN+CrossAttn) | FlowMatch + Consistency |
| MultiTask | RGB + joint_state + task_text | DINO/CLIP/ResNet/SigLIP + StateMLP + CLIPTextEncoder(frozen) + text_proj | DiT (AdaLN-Zero) | Diffusion (DDIM) |

## 目录结构

```
dexmani_policy/
├── configs/                     # Hydra 配置
│   ├── dp.yaml                  # RGB Diffusion
│   ├── dp3.yaml                 # 点云 Diffusion
│   ├── moe_dp3.yaml             # 点云 + MoE Diffusion
│   ├── maniflow.yaml            # 点云 FlowMatch (单卡)
│   ├── maniflow_ddp.yaml        # 点云 FlowMatch (DDP)
│   ├── multitask_dit.yaml       # 多任务 RGB + text + DiT
│   └── dataset/                 # 多任务数据集配置
├── agents/
│   ├── core/                    # 策略定义
│   │   ├── base.py              # BaseAgent / UNetDiffusionAgent / DiTXFlowMatchAgent
│   │   ├── dp.py                # DPAgent (RGB + Diffusion)
│   │   ├── dp3.py               # DP3Agent (点云 + Diffusion)
│   │   ├── moe.py               # MoEAgent (点云 + MoE + Diffusion)
│   │   ├── maniflow.py          # ManiFlowAgent (点云 + FlowMatch)
│   │   └── multi_task.py        # MultiTaskAgent (RGB + text + DiT)
│   ├── action_decoders/
│   │   ├── diffusion.py         # DDIM (100 train / 10 inference steps)
│   │   ├── flowmatch.py         # FlowMatch + Consistency Distillation
│   │   ├── backbone/
│   │   │   ├── unet1d.py        # ConditionalUnet1D (FiLM / CrossAttn+FiLM)
│   │   │   ├── dit.py           # DiT_Diffusion (AdaLN-Zero, MultiTask 使用)
│   │   │   └── ditx.py          # DiTXFlowMatch (AdaLN+CrossAttn, ManiFlow 使用)
│   │   └── common/sample.py     # t 采样策略 (beta/uniform/lognorm/discrete/…)
│   ├── obs_encoder/
│   │   ├── pointcloud/          # PointNet / iDP3 / PointNext / PointPN / PointNext Tokenizer
│   │   ├── rgb/                 # ResNet / DINO / CLIP / SigLIP + ImageProcessor
│   │   ├── proprio/state_mlp.py # StateMLP (joint_state → embedding)
│   │   ├── text/                # CLIPTextEncoder (MultiTask 使用) / T5TextEncoder (预留)
│   │   └── plugins/             # MoE Router / TokenCompressor (未使用)
│   └── common/                  # optim_util (get_optim_group_with_no_decay) / param_counter
├── datasets/
│   ├── base_dataset.py          # Zarr replay buffer → BaseDataset
│   ├── pc_dataset.py            # 点云 + proprio
│   ├── rgb_dataset.py           # RGB + proprio
│   ├── rgb_pc_dataset.py        # RGB + 深度 + 点云 + proprio
│   ├── multi_task_dataset.py    # 多任务混合采样 (balanced/weighted/proportional)
│   ├── common/
│   │   ├── replay_buffer.py     # Zarr episode buffer
│   │   └── sampler.py           # SequenceSampler (horizon 窗口) + train/val mask
│   └── augmentation/
│       ├── base.py              # Aug 基类 (enabled + prob)
│       ├── pc_color.py          # PCColorJitter (brightness/contrast/saturation/hue)
│       ├── pc_spatial.py        # PCSpatialAug (rot_z/trans_xy/scale)
│       ├── pc_dropout.py        # PCDropout (随机丢点)
│       ├── rgb.py               # RGBAug (ColorJitter)
│       └── state.py             # StateNoiseAug (高斯噪声)
├── training/
│   ├── trainer.py               # 单卡训练循环 (梯度累积 + EMA + val/eval + on_epoch_start)
│   ├── ddp_trainer.py           # DDP 封装 (normalizer 同步 + barrier)
│   ├── sim_evaluator.py         # 离线评估调度 (扫描 denoise_timesteps + 视频录制)
│   └── common/
│       ├── workspace.py         # TrainWorkspace (wandb + checkpoint topk 管理)
│       ├── ema_model.py         # EMA (power=0.75, max_value=0.9999)
│       ├── checkpoint_io.py     # TrainCheckpoint 序列化
│       └── lr_scheduler.py      # Cosine warmup scheduler
├── env_runner/
│   ├── base_runner.py           # BaseRunner (obs deque + stack + rollout 循环)
│   ├── sim_runner.py            # SimRunner (动态加载 dexmani_sim 环境)
│   └── multi_task_sim_runner.py # MultiTaskSimRunner + TaskTextSimRunner
├── common/
│   ├── normalizer.py            # LinearNormalizer (limits/gaussian/identity)
│   └── pytorch_util.py          # dict_apply / optimizer_to / fix_state_dict
├── train.py                     # 单卡训练入口
├── train_multi_task.py          # 多任务训练入口
├── train_ddp.py                 # DDP 训练入口
└── eval_sim.py                  # 仿真评估入口
```

## 核心数据流

**训练**:
```
batch['obs']: (B, T, …) → normalize → flatten(B*T) → obs_encoder → cond
batch['action']: (B, horizon, A) → normalize
action_decoder.compute_loss(cond, nactions) → loss
```

**推理**:
```
obs_dict: (1, T, …) → preprocess → obs_encoder → cond
randn(B, horizon, A) → predict_action(cond, denoise_steps) → unnormalize
→ {pred_action, control_action[:, s:s+n_action_steps]}
```

## 关键机制

### Agent 体系

- **BaseAgent** (`agents/core/base.py`): 所有策略的基类。`preprocess()` 将 `(B,T,…)` 展平为 `(B*T,…)`；`_build_cond()` 统一构建 condition（含 cond_dropout）；`configure_optimizer()` 支持 encoder/decoder 独立 lr 和 weight_decay
- **UNetDiffusionAgent**: obs_encoder + ConditionalUnet1D + Diffusion。子类：DPAgent / DP3Agent / MoEAgent
- **DiTXFlowMatchAgent**: obs_encoder + DiTXFlowMatch + FlowMatchWithConsistency。子类：ManiFlowAgent
- **MultiTaskAgent**: 直接继承 BaseAgent。CLIPTextEncoder(frozen) + text_proj(trainable) 编码 task_text，与 obs_cond 在特征维度拼接后送入 DiT_Diffusion。支持 register_buffer 预计算文本嵌入缓存

### 观测编码

- **点云**: PointNet(全局maxpool) / iDP3(多阶段MLP) / PointNext(层级聚合) → `(B*T, out_dim)`；PointPN/PointNext Tokenizer → `(B*T, num_tokens, token_dim)`
- **RGB**: ResNet/DINO/CLIP/SigLIP + ImageProcessor(resize→centercrop→normalize) → `(B*T, out_dim)`
- **Proprio**: StateMLP (Linear→ReLU→Linear) → `(B*T, 64)`
- **Text**: CLIPTextEncoder (frozen, MultiTask 使用) → text_proj (trainable Linear) → `(B, n_emb)`

### 条件注入

| Backbone | 注入方式 | cond 形状 |
|----------|---------|----------|
| UNet1D (film) | FiLM: scale/bias 调制每个分辨率级别的特征 | `(B, n_obs_steps * token_dim)` |
| UNet1D (cross_attention_film) | CrossAttn + FiLM | `(B, n_obs_steps, token_dim)` |
| DiT (AdaLN-Zero) | AdaLN-Zero: scale/shift/gate 调制每层 | `(B, obs_cond_dim + n_emb)` (MultiTask 拼接后) |
| DiT-X (AdaLN+CrossAttn) | AdaLN 调制 + CrossAttn 注入序列 token | `(B, num_tokens, token_dim)` |

### 动作解码

- **Diffusion** (DP/DP3/MoE/MultiTask): DDIM scheduler, 100 train steps, 10 inference steps, prediction_type=sample
- **FlowMatchWithConsistency** (ManiFlow): batch 按 75:25 切分为 flow 和 consistency 两部分。Flow 学习速度场 `v_t = x_1 - x_0`；Consistency 用 EMA 模型预测目标速度。B<2 时返回零损失保留梯度图

### 训练循环

- **Trainer**: 梯度累积 (`grad_accum_steps`) + EMA 更新 + epoch 结束 `flush_gradient_accumulation()`
- **on_epoch_start()**: 通过 `hasattr` 自动支持 `dataset.set_epoch()` 和 `model.set_epoch()`
- **DDPTrainer**: DDP 只包装训练模型，EMA 不包装。normalizer 从 rank 0 broadcast。checkpoint 统一用 `raw_model.state_dict()` 保存（无 `module.` 前缀）
- **验证**: 始终用训练模型计算 loss；ManiFlow 的 EMA 仅作为 consistency teacher 通过 kwargs 传入
- **Checkpoint**: topk by `test_mean_score`，latest symlink，`fix_state_dict()` 兼容单卡↔多卡

### 数据集

- **Zarr replay buffer**: episode 级存储，`episode_ends` 索引边界
- **SequenceSampler**: horizon 窗口滑动采样，`pad_before/after` 支持下标越界填充（最早/最晚帧重复）
- **Train/Val 划分**: episode 级随机 mask，确保无数据泄露
- **MultiTaskDataset**: 3 种采样策略 (balanced/weighted/proportional)；训练集每 epoch 预计算 `epoch_indices`；验证集 `deterministic=True` 固定顺序遍历；注入 `task_text` 到 obs

### 数据增强

默认关闭。启用方式：取消 config 末尾 `augmentation_cfg` 注释。

| 增强器 | 适用模态 | 参数 |
|--------|---------|------|
| PCColorJitter | 点云颜色 | brightness, contrast, saturation, hue, prob |
| PCSpatialAug | 点云坐标 | rot_z, trans_xy, scale, prob |
| PCDropout | 点云 | dropout_ratio, prob |
| RGBAug | RGB | brightness, contrast, saturation, hue, prob |
| StateNoiseAug | joint_state | noise_std, prob |

所有增强器继承 `Aug` 基类，支持 `enabled` 开关和 `prob` 概率控制。点云增强器通过 `PC_AUG_CLASSES` 注册，按 color→spatial→dropout 顺序应用。

### 评估

- **SimRunner**: `importlib` 动态加载 `dexmani_sim` 环境。seed 来源优先级：`eval_seeds` 参数 > `eval_seeds/{task}.txt` > `range(100)` fallback
- **MultiTaskSimRunner**: 通过 `TaskTextSimRunner` 逐 task 注入 `task_text` 后分别评估，汇总 `success_rate` 取平均
- **SimEvaluator**: 扫描多种 `denoise_timesteps`，保存视频 + metrics

## 默认超参

```
horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19
num_epochs=1000, lr=1e-4, cosine scheduler, warmup=500 steps
EMA: power=0.75, max_value=0.9999
Diffusion: 100 train / 10 inference (DDIM), prediction_type=sample
FlowMatch: 10 denoise steps, t~Beta(1.0, 1.5), flow:consistency=0.75:0.25
cond_dropout_prob=0.0
grad_accum_steps=1, grad_clip_norm=1.0
```

## 已知死代码

不影响功能，保留作为实验性代码：

- `agents/action_decoders/backbone/ditx.py` — `DiTXDiffusion` 类（134 行）
- `agents/obs_encoder/plugins/token_compressor.py` — 四个类（345 行）
- `agents/action_decoders/common/sample.py` — `logit_normal_density` 函数

## 依赖

- **dexmani_sim**: 仿真环境（外部包，importlib 动态加载）
- **核心库**: torch, hydra-core, diffusers, einops, timm, wandb, zarr

## 待开发功能

来自 RoboTwin 对比审查（2026-05-11），确有价值但暂不实现。

### 高优先级

| 功能 | 说明 | 阻塞条件 |
|------|------|---------|
| `max_train_steps` / `max_val_steps` | 限制每 epoch 步数上限，debug 快速验证 | 无 |
| `debug` 训练模式 | `training.debug=True` 自动压缩 epoch/步数/eval 间隔 | 无 |
| Expert-check 评估筛选 | eval 前跑 expert demo 确认场景可解，排除不可解 seed | 需 `dexmani_sim` 支持 `play_once()`/`check_success()` |
| global_cond dropout | 对 condition 做随机 dropout，增强不完整观测鲁棒性 | 已有 `cond_dropout_prob` 参数但仅在 `_apply_cond_dropout` 中做全零 dropout |

### 中优先级

| 功能 | 说明 | 触发条件 |
|------|------|---------|
| `_sample_to_data` 键名映射 | Dataset 中解耦存储 key 与消费 key | 需要加载外部数据时 |
| Modality dropout augmentation | 对 point_cloud / joint_state 独立随机 dropout | 多模态策略需要时 |
| `shape_meta` 多臂 action shape | 支持 2D action shape（如双臂 2×7） | 需要双臂任务时 |

### 不建议引入

以下已有更好方案或不适用：

- **freeze_encoder**: 已有 `obs_lr=0` 等价方案，更灵活
- **LowdimMaskGenerator**: film 模式下完全 no-op，不适用
- **确定性 BatchSampler**: 非标准 PyTorch 模式
- **点云归一化**: 可能破坏几何信息
- **DDPM scheduler**: DDIM 10 步推理更高效
- **单体 Workspace 架构**: Trainer/Workspace 分离更好
- **Per-policy eval function**: `predict_action()` 黑盒接口已足够
