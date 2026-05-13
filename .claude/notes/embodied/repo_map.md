# Embodied Repo Research Note - dexmani_policy

**生成时间**: 2026-05-08  
**仓库**: dexmani_policy - 灵巧操作模仿学习策略框架  
**上次更新**: 2026-05-13 — 基于完整代码审计同步更新

---

## TL;DR

**dexmani_policy** 是一个灵巧操作模仿学习策略框架，支持 5 种策略（DP/DP3/MoE/ManiFlow/MultiTask）× 多模态观测（RGB/点云/proprio）× 单卡/DDP 训练。核心设计：Hydra 配置驱动的模块化架构，统一的 `BaseAgent` 接口，episode 级 train/val 划分的 Zarr replay buffer，EMA 模型用于 consistency distillation 和评估，topk checkpoint 管理。代码量约 70 个 Python 文件，核心训练/策略/数据模块约 2365 行。

**关键特性**:
- 配置驱动：所有组件通过 `_target_` 实例化，切换策略只需换 config
- 统一接口：`BaseAgent.compute_loss()` / `predict_action()` / `configure_optimizer()`
- 双目标训练：ManiFlow 支持 flow matching + consistency distillation（75:25 比例）
- 多任务支持：CLIP text encoder (frozen) + text_proj (trainable) 特征拼接 + 任务感知采样 + shared/per_task normalizer
- 分布式训练：DDP 封装 + normalizer 同步 + 无 `module.` 前缀 checkpoint

---

## Execution Skeleton

```
config (Hydra YAML) 
  → dataset (Zarr replay buffer → PCDataset/RGBDataset/RGBPCDataset/MultiTaskDataset)
  → agent (obs_encoder → action_decoder)
  → trainer (Trainer/DDPTrainer)
  → evaluator (SimRunner/MultiTaskSimRunner)
```

**训练数据流**:
```
batch['obs']: (B, T, obs_dim)
  → normalize
  → flatten: (B*T, obs_dim)
  → obs_encoder: (B*T, obs_dim) → (B*T, token_dim)
  → reshape + concat: (B, n_obs_steps*token_dim) [film] 或 (B, n_obs_steps, token_dim) [cross_attention]
  → cond

batch['action']: (B, horizon, action_dim)
  → normalize: (B, horizon, action_dim)
  → action_decoder.compute_loss(cond, nactions) → loss
```

**推理数据流**:
```
obs_dict: {pc/rgb/joint_state: (1, T, ...)}
  → preprocess (normalize + flatten)
  → obs_encoder → cond: (1, cond_dim)
  → randn template: (1, horizon, action_dim)
  → action_decoder.predict_action(cond, template, denoise_steps=10)
  → pred: (1, horizon, action_dim)
  → unnormalize
  → control_action = pred[:, n_obs_steps-1:n_obs_steps-1+n_action_steps]
```

---

## Repository Map

### 核心模块层级

```
dexmani_policy/
├── agents/                          # 策略定义
│   ├── core/                        # Agent 实现
│   │   ├── base.py                  # BaseAgent / UNetDiffusionAgent / DiTXFlowMatchAgent
│   │   ├── dp.py                    # RGB + proprio → Diffusion
│   │   ├── dp3.py                   # 点云 + proprio → Diffusion
│   │   ├── moe.py                   # 点云 + MoE → Diffusion
│   │   ├── maniflow.py              # 点云 + proprio → FlowMatch
│   │   └── multi_task.py            # RGB + state + CLIP text → DiT backbone (特征拼接)
│   ├── action_decoders/             # 动作生成模型
│   │   ├── diffusion.py             # DDIM scheduler (100 train / 10 inference steps)
│   │   ├── flowmatch.py             # Flow matching + Consistency distillation
│   │   ├── backbone/
│   │   │   ├── unet1d.py            # ConditionalUnet1D (FiLM / CrossAttention)
│   │   │   ├── ditx.py              # DiT-X Transformer (AdaLN + CrossAttention)
│   │   │   └── dit_blocks.py        # DiT building blocks
│   │   └── common/sample.py         # t 采样策略 (beta/uniform/lognorm/cosmap/discrete)
│   ├── obs_encoder/                 # 观测编码器
│   │   ├── pointcloud/              # PointNet / iDP3 / PointNext / PointPN Tokenizer
│   │   ├── rgb/                     # ResNet / DINO / CLIP / SigLIP + ImageProcessor
│   │   ├── proprio/                 # StateMLP (joint_state → hidden)
│   │   ├── text/                    # CLIPTextEncoder (MultiTaskAgent 使用) / T5TextEncoder (预留)
│   │   └── plugins/                 # MoE / TokenCompressor
│   └── common/                      # optim_util / param_counter
├── datasets/                        # 数据加载
│   ├── base_dataset.py              # BaseDataset (Zarr replay buffer 封装)
│   ├── pc_dataset.py                # PCDataset (点云 + proprio)
│   ├── rgb_dataset.py               # RGBDataset (RGB + proprio)
│   ├── rgbpc_dataset.py             # RGBPCDataset (RGB + 点云 + proprio)
│   ├── multi_task_dataset.py        # MultiTaskDataset (多任务混合)
│   ├── common/
│   │   ├── replay_buffer.py         # Zarr episode replay buffer
│   │   └── sampler.py               # SequenceSampler (horizon 窗口采样)
│   └── augmentation/                # PCColorJitter/PCSpatialAug/PCDropout/RGBAug/StateNoiseAug
├── training/                        # 训练循环
│   ├── trainer.py                   # 单卡训练 (梯度累积 + EMA + 验证/评估 + on_epoch_start)
│   ├── ddp_trainer.py               # DDP 训练封装 (normalizer 同步)
│   ├── sim_evaluator.py             # 仿真评估 + 视频录制
│   └── common/
│       ├── workspace.py             # Checkpoint/logging 管理 (topk + wandb)
│       ├── ema_model.py             # EMA 更新 (power=0.75, max_value=0.9999)
│       ├── checkpoint_io.py         # Checkpoint 序列化
│       └── lr_scheduler.py          # Cosine scheduler with warmup
├── env_runner/                      # 仿真接口
│   ├── base_runner.py               # BaseRunner 抽象类
│   ├── sim_runner.py                # SimRunner (单任务)
│   └── multi_task_sim_runner.py     # MultiTaskSimRunner (多任务)
├── common/                          # 工具函数
│   ├── normalizer.py                # LinearNormalizer (limits/gaussian 模式)
│   └── pytorch_util.py              # dict_apply / optimizer_to
├── configs/                         # Hydra YAML 配置
│   ├── dp.yaml                      # RGB + Diffusion
│   ├── dp3.yaml                     # 点云 + Diffusion
│   ├── moe_dp3.yaml                 # 点云 + MoE + Diffusion
│   ├── maniflow.yaml                # 点云 + FlowMatch
│   ├── maniflow_ddp.yaml            # 点云 + FlowMatch + DDP
│   ├── multitask_dit.yaml           # 多任务 + RGB + CLIP text + DiT Diffusion
│   └── dataset/                     # 数据集配置
│       └── multitask_rgb.yaml       # 多任务 RGB 数据集
├── train.py                         # 单卡训练入口
├── train_multi_task.py              # 多任务训练入口
├── train_ddp.py                     # DDP 训练入口
└── eval_sim.py                      # 仿真评估入口
```

---

## Main Entrypoints

| Purpose | File | Key Functions | Notes |
|---------|------|---------------|-------|
| 单卡训练 | `train.py` | `build_train_components()`, `Trainer.train()` | 默认入口，支持 DP/DP3/MoE/ManiFlow |
| 多任务训练 | `train_multi_task.py` | `build_train_components()`, `Trainer.train()` | 复用 Trainer，通过 `on_epoch_start()` 自动调用 `set_epoch` |
| DDP 训练 | `train_ddp.py` | `mp.spawn(ddp_worker)`, `DDPTrainer.train()` | 需配置 `training.num_gpus > 1` |
| 仿真评估 | `eval_sim.py` | `SimEvaluator.run()` | 从 `experiments/{policy}/{task}/{exp_name}` 加载 checkpoint |
| Shell 脚本 | `scripts/*.sh` | - | Bash 包装器，简化命令行调用 |

**训练命令示例**:
```bash
# 单卡训练
python dexmani_policy/train.py --config-name=dp3

# 多任务训练
python dexmani_policy/train_multi_task.py --config-name=multitask_dit

# DDP 训练
python dexmani_policy/train_ddp.py --config-name=maniflow_ddp training.gpu_ids=[0,1,2,3]

# 仿真评估
python dexmani_policy/eval_sim.py --policy-name dp3 --task-name pick_apple_messy --exp-name 2026-04-01_11-18_233
```

---

## Embodied Module Breakdown

### 1. Agent 架构

**BaseAgent** (`agents/core/base.py`):
- **接口**: `compute_loss(batch, **kwargs)`, `predict_action(obs_dict)`, `configure_optimizer()`
- **关键设计**: `preprocess()` 将 `(B, T, ...)` 展平为 `(B*T, ...)`，推理时返回 `control_action[:, n_obs_steps-1:n_obs_steps-1+n_action_steps]`
- **cond_dropout**: `_build_cond()` → `_apply_cond_dropout()`，训练时以 `cond_dropout_prob` 概率将 condition 置零
- **compute_loss**: kwargs 中的 `ema_model` 自动转换为 `ema_model.ema_backbone` 后传递给 action_decoder

**UNetDiffusionAgent** (`agents/core/base.py`):
- **组件**: `obs_encoder` + `ConditionalUnet1D` + `Diffusion`
- **条件注入**: `film` (全局向量) 或 `cross_attention_film` (序列 token)
- **子类**: `DPAgent` (RGB), `DP3Agent` (点云), `MoEAgent` (点云+MoE)

**DiTXFlowMatchAgent** (`agents/core/base.py`):
- **组件**: `obs_encoder` + `DiTXFlowMatch` + `FlowMatchWithConsistency`
- **条件注入**: AdaLN (全局) + CrossAttention (序列)
- **子类**: `ManiFlowAgent` (点云)

**MultiTaskAgent** (`agents/core/multi_task.py`):
- **组件**: `CLIPTextEncoder` (frozen) + `text_proj` (trainable Linear) + `DPObsEncoder` (RGB+state) + `DiT_Diffusion` backbone
- **条件注入**: `obs_cond` 与 `text_emb` 在特征维度拼接 → `full_cond_dim = obs_cond_dim + n_emb`
- **文本缓存**: `register_buffer("task_emb_table")` 预计算所有 task_text 的 CLIP embedding，训练时 O(1) 查表
- **关键**: 直接继承 `BaseAgent`，override `_build_cond()` / `predict_action()` / `configure_optimizer()`

### 2. 观测编码器

**点云编码器** (`agents/obs_encoder/pointcloud/`):
- **PointNet**: 全局 max pooling，输出 `(B, out_dim)`
- **iDP3**: PointNet + 3 层 MLP，输出 `(B, out_dim)`
- **PointNext**: 层级点云处理，输出 `(B, out_dim)`
- **PointPN Tokenizer**: 点云 tokenizer，输出 `(B, num_tokens, token_dim)`

**RGB 编码器** (`agents/obs_encoder/rgb/`):
- **ResNet**: 预训练 ResNet18/34/50，冻结/微调可选
- **DINO**: 预训练 ViT，输出 CLS token 或 patch tokens
- **CLIP**: 预训练 ViT，输出 CLS token
- **SigLIP**: 预训练 ViT，输出 CLS token

**Proprio 编码器** (`agents/obs_encoder/proprio/state_mlp.py`):
- **StateMLP**: 2 层 MLP，`joint_state → hidden`

### 3. 动作解码器

**Diffusion** (`agents/action_decoders/diffusion.py:90`):
- **Scheduler**: DDIM，`num_train_timesteps=100`, `num_inference_steps=10`
- **Prediction type**: `'sample'` (直接预测 x0) 或 `'epsilon'` (预测噪声)
- **训练**: `noisy_action = scheduler.add_noise(actions, noise, timestep)` → `model(noisy_action, timestep, cond)` → MSE loss
- **推理**: DDIM 采样，10 步去噪

**FlowMatchWithConsistency** (`agents/action_decoders/flowmatch.py:120+`):
- **Flow matching**: `xt = (1-t)*x0 + t*x1`, 预测 `vt = x1 - x0`
- **Consistency distillation**: 学生预测 `vt`，教师（EMA）预测 `v_next`，目标 `vt_target = (pred_x1 - xt) / (1-t)`
- **t 采样**: flow 用 `beta(1, 1.5)`，consistency 用 `discrete`
- **Batch 分配**: `flow_batch_ratio=0.75`, `consistency_batch_ratio=0.25`

**Backbone**:
- **ConditionalUnet1D**: 1D UNet，支持 FiLM 和 CrossAttention 条件注入
- **DiTXFlowMatch**: DiT-X Transformer，AdaLN + CrossAttention

### 4. 数据集

**BaseDataset** (`datasets/base_dataset.py:100+`):
- **Zarr replay buffer**: episode 级存储，`{obs/action/episode_ends}`
- **SequenceSampler**: `horizon` 窗口采样，`pad_before=n_obs_steps-1`, `pad_after=n_action_steps-1`
- **Train/Val 划分**: episode 级随机划分，`get_val_mask(seed, val_ratio, n_episodes)`
- **Normalizer**: `get_normalizer()` 从 replay buffer 拟合 `LinearNormalizer`

**MultiTaskDataset** (`datasets/multi_task_dataset.py:174`):
- **采样策略**: `balanced` (均匀) / `proportional` (按数据量) / `weighted` (自定义权重)
- **训练集**: 有放回随机采样，`hash(seed, epoch, idx)` 确定 task 和 local_idx
- **验证集**: 无放回固定顺序遍历
- **Normalizer**: `shared` (合并所有任务) / `per_task` (独立拟合)
- **关键**: `set_epoch(epoch)` 改变采样种子

### 5. 训练循环

**Trainer** (`training/trainer.py`):
- **梯度累积**: `scaled_loss = raw_loss / grad_accum_steps`，`flush_gradient_accumulation()` 处理 epoch 结束时的未完成累积
- **EMA 更新**: `ema_updater.step(model)` 在 optimizer.step() 后调用
- **验证/评估**: 每 `val_interval_epochs` 验证，每 `eval_interval_epochs` 评估
- **on_epoch_start**: 通过 `hasattr` 自动支持 `set_epoch`（多任务）等 epoch 级钩子
- **validate**: 始终使用训练模型；EMA 仅作为 consistency teacher 通过 kwargs 传入
- **Checkpoint**: topk by `test_mean_score`，latest symlink 支持断点续训

**DDPTrainer** (`training/ddp_trainer.py`):
- **DDP 包装**: `ddp_model = DDP(model)`, `Trainer(model=ddp_model)`
- **Normalizer 同步**: `dist.broadcast(normalizer.state_dict())` 从 rank 0 同步
- **Checkpoint**: 使用 `self.raw_model.state_dict()` 保存（无 `module.` 前缀）

### 6. 评估

**SimRunner** (`env_runner/sim_runner.py`):
- **依赖**: 外部 `dexmani_sim` 包（通过 `importlib` 动态加载）
- **评估指标**: `success_rate`, `avg_steps`
- **视频录制**: 保存为 MP4

**MultiTaskSimRunner** (`env_runner/multi_task_sim_runner.py`):
- **设计**: 为每个 task 创建 `_TaskAwareSimRunner`
- **评估**: 逐 task 评估后汇总 `success_rate`

---

## Reproducibility Checklist

### 高风险 ⚠️

- [ ] **外部依赖 dexmani_sim**: 记录版本号和环境配置
  - 位置: `env_runner/sim_runner.py` 中 `from dexmani_sim.envs import make_env`
  - 缓解: 提供 dexmani_sim 版本号、环境配置、成功标准定义

- [ ] **Episode 级 train/val 划分**: 记录 `n_episodes` 和 `val_ratio`
  - 位置: `datasets/common/sampler.py` 中 `get_val_mask(seed, val_ratio, n_episodes)`
  - 缓解: 保存 train/val episode 索引到 checkpoint

- [ ] **Normalizer 状态**: 确保 checkpoint 包含 normalizer state
  - 位置: `common/normalizer.py` 中 `_fit()` 计算归一化参数
  - 缓解: 已实现（checkpoint 包含 normalizer state）

- [ ] **多任务采样随机性**: 使用 `np.random.default_rng(seed)` 替代 Python `hash()`
  - 位置: `datasets/multi_task_dataset.py` 中 `item_seed = hash((self.seed, self._epoch, idx))`
  - 缓解: 替换为 numpy RNG

### 中风险 ⚡

- [ ] **EMA 初始化**: 优先使用 `copy.deepcopy(model)`，记录初始化方式
  - 位置: `train.py` 中 `try: ema_model = copy.deepcopy(model) except: ...`

- [ ] **点云采样**: 固定点云数量或记录采样算法版本
  - 位置: `agents/core/dp3.py` 中 `farthest_point_sample(pc, self.num_points)`

- [ ] **Checkpoint 格式**: 保存 RNG 状态
  - 位置: `training/common/checkpoint_io.py` 中 `TrainCheckpoint` dataclass
  - 缓解: 添加 `torch.get_rng_state()` 和 `np.random.get_state()`

### 低风险 ✓

- [ ] **数据增强**: 记录增强参数（如 `color_std=0.05`）
- [ ] **Hydra 配置**: 保存完整的 resolved config
- [ ] **Wandb 日志**: 使用 online 模式或定期同步

---

## Ablation Surface

### 1. 观测编码器 (Encoder Swap)

**点云编码器**:
- **配置**: `agent.encoder_type` in `['dp3', 'idp3', 'pointnext']`
- **文件**: `agents/obs_encoder/pointcloud/registry.py`
- **影响**: 表征能力、参数量、训练速度

**RGB 编码器**:
- **配置**: `agent.rgb_backbone` in `['resnet', 'clip', 'dino', 'siglip']`
- **文件**: `agents/obs_encoder/rgb/registry.py`
- **影响**: 预训练权重、冻结/微调策略

### 2. 骨干网络 (Backbone Swap)

**UNet1D vs DiT-X**:
- **配置**: 切换 `dp3.yaml` ↔ `maniflow.yaml`
- **影响**: 条件注入方式（FiLM vs AdaLN+CrossAttention）、参数量

**条件注入方式**:
- **配置**: `agent.condition_type` in `['film', 'cross_attention_film']`
- **影响**: 观测序列建模能力

### 3. 动作解码器 (Decoder Swap)

**Diffusion vs FlowMatch**:
- **配置**: 切换 `dp3.yaml` ↔ `maniflow.yaml`
- **影响**: 采样速度、训练稳定性

**Prediction type**:
- **配置**: `agent.prediction_type` in `['sample', 'epsilon']`
- **影响**: 训练目标、收敛速度

### 4. t 采样策略

**Flow matching**:
- **配置**: `agent.t_sample_mode_for_flow` in `['beta', 'uniform', 'lognorm', 'cosmap']`
- **文件**: `agents/action_decoders/common/sample.py`
- **影响**: 训练稳定性、不同时间步的学习权重

**Consistency distillation**:
- **配置**: `agent.t_sample_mode_for_consistency` in `['discrete', 'uniform']`
- **影响**: consistency 目标的时间步分布

### 5. MoE 配置

**专家数量**: `agent.num_experts` in `[4, 8, 16, 32]`
**Top-k**: `agent.top_k` in `[1, 2, 4]`
**辅助损失权重**: `agent.lambda_load`, `agent.beta_entropy`

### 6. 多任务配置

**文本嵌入维度**: `agent.n_emb` in `[256, 512, 768]`
**采样策略**: `dataset.sampling_strategy` in `['balanced', 'proportional', 'weighted']`
**Normalizer 模式**: `dataset.normalizer_mode` in `['shared', 'per_task']`
**RGB backbone**: `agent.rgb_backbone_name` in `['resnet', 'clip', 'dino', 'siglip']`

### 7. 训练超参

**学习率**: `optimizer.lr`, `optimizer.obs_lr` (支持 encoder/decoder 独立学习率)
**EMA 参数**: `ema.power`, `ema.max_value`
**梯度累积**: `training.loop.gradient_accumulate_every`

---

## Open Questions

### 架构设计

1. **MultiTaskAgent text_proj 初始化**: `text_proj` (Linear) 使用默认 Xavier 初始化，是否需要零初始化以支持从预训练模型热启动？
   - 位置: `agents/core/multi_task.py:90`

2. **MoE 辅助损失权重**: `lambda_load=0.1`, `beta_entropy=0.01` 是否对所有任务都适用？
   - 位置: `agents/obs_encoder/plugins/moe.py`

3. **Consistency distillation dt 采样**: 学生和教师使用相同的 `dt`，是否应该独立采样？
   - 位置: `agents/action_decoders/flowmatch.py:78`

### 数据流

4. **观测窗口对齐**: `control_action` 从 `pred[:, n_obs_steps-1:]` 开始，是否符合所有环境的时序假设？
   - 位置: `agents/core/base.py:61-64`

5. **点云坐标系**: `use_coord_only = (pc_dim == 3)` 假设 3 维点云只包含坐标，是否支持法向量？
   - 位置: `agents/core/dp3.py:27, 33`

### 训练机制

6. **梯度累积 flush**: epoch 结束时累积步数不是整数倍，`flush_gradient_accumulation()` 会 scale 梯度。是否影响稳定性？
   - 位置: `training/trainer.py:92-101`

7. **验证集 teacher**: 验证时使用训练模型 + EMA teacher（ManiFlow），loss 中的 consistency 项有梯度吗？
   - 位置: `training/trainer.py:119-138`

### 评估

8. **MultiTaskSimRunner 视频合并**: 合并所有任务的视频但没有标记 task_name，如何区分？
   - 位置: `env_runner/multi_task_sim_runner.py:71`

9. **Success rate 定义**: 不同任务的成功标准可能不同，如何确保公平比较？
   - 位置: `env_runner/sim_runner.py` 中 `env.get_success()`

### 配置

10. **Hydra 输出目录**: 同一分钟内多次运行会覆盖，是否需要添加随机后缀？
    - 位置: `configs/dp3.yaml:148`

---

## Tensor Shapes Reference

### 训练时
- `batch['obs']['pc']`: `(B, T, N, 3)` → flatten → `(B*T, N, 3)`
- `batch['obs']['joint_state']`: `(B, T, state_dim)` → flatten → `(B*T, state_dim)`
- `batch['action']`: `(B, horizon, action_dim)`
- `cond` (film): `(B, n_obs_steps * (pc_out_dim + state_out_dim))`
- `cond` (cross_attention): `(B, n_obs_steps, token_dim)`

### 推理时
- `obs_dict['pc']`: `(1, T, N, 3)`
- `obs_dict['joint_state']`: `(1, T, state_dim)`
- `cond`: `(1, cond_dim)`
- `pred`: `(1, horizon, action_dim)`
- `control_action`: `pred[:, n_obs_steps-1:n_obs_steps-1+n_action_steps]`

---

## 关键配置依赖

- **条件注入方式**: `agent.condition_type` 决定 `obs_encoder` 输出形状
  - `film` / `mlp_film`: `(B, n_obs_steps * token_dim)` 全局向量
  - `cross_attention_film`: `(B, n_obs_steps, token_dim)` 序列 token

- **MultiTask 条件维度**: `full_cond_dim = obs_cond_dim + n_emb`
  - `obs_cond_dim = obs_encoder.out_dim * n_obs_steps` (film 模式)
  - `n_emb` 由 `agent.n_emb` 配置

- **Normalizer 模式**: `dataset.normalizer_mode` 影响动作空间归一化
  - `shared`: 合并所有任务的 joint_state/action 拟合
  - `per_task`: 每个任务独立拟合

---

## 已知限制

1. **外部依赖**: 评估依赖外部 `dexmani_sim` 包，不在本仓库
2. **多任务采样**: 训练集使用 hash-based 随机采样，可能在不同 Python 版本间不一致
3. **视频标记**: MultiTaskSimRunner 合并视频时以 `{task_name}/{video_key}` 命名，已可区分 task
4. **Hydra 输出**: 同一分钟内多次运行会覆盖输出目录
5. **MultiTask + DDP**: 组合尚未实现
6. **死代码**: `DiTXDiffusion` (ditx.py)、`token_compressor.py`、`logit_normal_density` 未使用

---

**最后更新**: 2026-05-13  
**审查状态**: 基于完整代码审计同步更新
