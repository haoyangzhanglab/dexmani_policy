# Embodied Repo Research Note

## TL;DR

dexmani_policy 是灵巧手操作模仿学习框架，Hydra 配置驱动，支持 5 种策略架构（DP/DP3/ManiFlow/MoE/MultiTask）、3 种训练模式（单卡/DDP/多任务）、Diffusion/FlowMatch 动作解码、Zarr 数据加载、dexmani_sim 仿真评测。核心不变约束：horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19。数据流：Zarr → ReplayBuffer → SequenceSampler（numba 滑动窗口）→ Dataset（增强）→ Agent（normalize → encoder → decoder）→ Trainer（loss → EMA → checkpoint）→ SimEvaluator（env_runner → success_rate）。Normalizer 在 train+val 全量数据上拟合，存储在 agent.state_dict() 中随 checkpoint 持久化。

## Execution Skeleton

### train.py (单卡训练)
```
main() [@hydra.main, L282]
  → validate_config(cfg) [L105-231: 窗口关系、Zarr 路径、ManiFlow/MoE 约束、增强一致性]
  → set_seed(cfg.training.seed) [L285: 全局 seed，模型初始化前]
  → build_train_components(cfg) [L233-278]
      → build_dataset_and_normalizer(cfg)
          → hydra.utils.instantiate(cfg.dataset) → BaseDataset 变体
          → dataset.get_normalizer() [BaseDataset.L107-114: LinearNormalizer.fit(train+val 全量, mode='limits')]
      → DataLoader(dataset, worker_init_fn, **cfg.dataloader)
      → build_model_and_ema(cfg, device, normalizer)
          → hydra.utils.instantiate(cfg.agent) → Agent 变体
          → model.load_normalizer_from_dataset(normalizer)
          → EMAModel(model) if cfg.training.use_ema
      → build_optimizer_and_scheduler(cfg, model, batches_per_epoch)
          → model.configure_optimizer(**cfg.optimizer)
          → get_scheduler(optimizer, cfg.training.lr_scheduler, warmup, total_steps)
      → hydra.utils.instantiate(cfg.workspace) → TrainWorkspace
      → hydra.utils.instantiate(cfg.env_runner) → SimRunner
  → Trainer(components, train_loop_cfg)
  → trainer.train(resume_tag="latest")
      → workspace.load_for_resume(model, ema, opt, sched, "latest")
      → for epoch in range(start_epoch, num_epochs):
          → train_one_step(batch) → model.compute_loss → backward → clip_grad → opt.step → EMA.step
          → finish_epoch(epoch) → validate/evaluate/save_checkpoint/save_topk/save_latest
```

### train_ddp.py (多卡 DDP)
```
main() [@hydra.main, L141]
  → validate_config(cfg)
  → 设置 MASTER_ADDR/MASTER_PORT [自动分配空闲端口]
  → mp.spawn(ddp_worker, args=(num_gpus, cfg, gpu_ids), nprocs=num_gpus)

ddp_worker(rank, world_size, cfg, gpu_ids)
  → setup_ddp(rank, world_size) [dist.init_process_group(backend='nccl')]
  → set_seed(cfg.training.seed) [模型初始化前，所有 rank 相同 seed]
  → build_dataset_and_normalizer(cfg, rank=rank, world_size=world_size)
  → DistributedSampler(dataset, num_replicas, rank, shuffle, seed)
  → build_model_and_ema(cfg, device, normalizer)
  → set_seed(cfg.training.seed + rank) [模型初始化后，各 rank 不同 seed 增加数据增强多样性]
  → DDPTrainer(rank, world_size, ...) [继承 Trainer，rank 0 独享 logging/checkpoint/eval]
  → ddp_trainer.train(resume_tag="latest")
```

### eval_sim.py (独立离线评测)
```
main() [argparse: --policy-name, --task-name, --exp-name, overrides]
  → run_eval(exp_dir, overrides)
      → OmegaConf.load(exp_dir / "config.yaml")
      → set_seed(cfg.eval.seed) [固定值 0]
      → agent = hydra.utils.instantiate(cfg.agent)
      → SimEvaluator(device, agent, env_runner, checkpoint_store, eval_root_dir)
      → evaluator.run(eval_episodes, denoise_timesteps_list, ckpt_tag_or_path, use_ema_for_eval)
          → _load_for_inference(ckpt_tag_or_path, use_ema)
          → for denoise_timesteps in denoise_timesteps_list:
              → env_runner.run(agent, denoise_timesteps, eval_episodes)
                  → for seed in eval_seeds:
                      → eval_one_episode(agent, env, seed, denoise_timesteps)
                          → env.reset(seed) → while not truncated:
                              → agent.predict_action(obs_batch, denoise_timesteps)
                              → for i in range(8): env.step(action_chunk[i])
              → recorder.save_case_result(result, denoise_timesteps) [video mp4 + metrics.json]
```

### smoke_test.py (冒烟测试)
```
main() [L91-105]
  → for config_name in sys.argv[1:]:
      → smoke_test(config_name) [L36-88]
          → load_config(config_name) [L22-33: Hydra compose API]
          → build_dataset_and_normalizer(cfg)
          → build_model_and_ema(cfg, device, normalizer)
          → build_optimizer_and_scheduler(cfg, model, len(train_loader))
          → batch = next(iter(train_loader))
          → model.compute_loss(batch, ema_backbone=...) [L69: 与 trainer.py:108 逻辑一致]
          → raw_loss.backward()
          → assert torch.isfinite(raw_loss)
          → model.predict_action(obs_sample)
          → assert control_action.shape == (1, n_action_steps, action_dim)
```

## Repository Map

```
dexmani_policy/
├── train.py                        # 单卡训练入口，Hydra main，validate_config + build_train_components + Trainer.train
├── train_ddp.py                    # DDP 多卡入口，mp.spawn(ddp_worker)，复用单卡构建函数
├── train_multi_task.py             # 多任务训练入口（复用 train.py 逻辑 + MultiTaskDataset）
├── eval_sim.py                     # 独立评测 CLI，Hydra-free，CheckpointStore 直接加载
├── smoke_test.py                   # 冒烟测试，Hydra compose API，复用 train.py 构建函数
├── configs/                        # Hydra YAML 配置
│   ├── dp.yaml                     # RGB + Diffusion
│   ├── dp3.yaml                    # PC + Diffusion
│   ├── maniflow.yaml               # PC + FlowMatch
│   ├── maniflow_ddp.yaml           # maniflow DDP 版本
│   ├── moe_dp3.yaml                # PC + MoE + Diffusion
│   ├── multitask_dit.yaml          # RGB + Text + DiT
│   └── dataset/multitask_rgb.yaml  # MultiTask dataset preset
├── agents/
│   ├── core/
│   │   ├── __init__.py             # 导出 5 种 Agent
│   │   ├── base.py                 # BaseAgent (preprocess, compute_loss, predict_action, configure_optimizer)
│   │   │                           # UNetDiffusionAgent (DP/DP3/MoE 基类)
│   │   │                           # DiTXFlowMatchAgent (ManiFlow 基类)
│   │   ├── dp.py                   # DPAgent: DPObsEncoder (RGB + state) + UNetDiffusionAgent
│   │   ├── dp3.py                  # DP3Agent: DP3ObsEncoder (PC + state) + UNetDiffusionAgent
│   │   ├── maniflow.py             # ManiFlowAgent: ManiFlowObsEncoder (PC + state) + DiTXFlowMatchAgent
│   │   ├── moe.py                  # MoEAgent: MoEObsEncoder (PC + state + MoE gating) + UNetDiffusionAgent
│   │   └── multi_task.py           # MultiTaskAgent: DPObsEncoder + CLIPTextEncoder + DiT_Diffusion
│   ├── action_decoders/
│   │   ├── diffusion.py            # Diffusion: DDIMScheduler, compute_loss (noise + timestep → MSE), predict_action (DDIM 迭代)
│   │   ├── flowmatch.py            # FlowMatchWithConsistency: batch 拆分 flow/consistency, Euler ODE 采样
│   │   ├── backbone/
│   │   │   ├── unet1d.py           # ConditionalUnet1D: FiLM conditioning, down/mid/up blocks
│   │   │   ├── dit.py              # DiT_Diffusion: self-attention + AdaLN, for MultiTask
│   │   │   └── ditx.py             # DiTXFlowMatch: cross-attention, for ManiFlow
│   │   └── common/sample.py        # TimeSampler: beta/discrete/uniform 时间采样
│   ├── obs_encoder/
│   │   ├── pointcloud/
│   │   │   ├── registry.py         # build_pc_global_encoder: idp3/pointnext/dp3/pointpn
│   │   │   ├── pointnet.py         # PointNet: MLP + max pooling
│   │   │   ├── pointnext.py        # PointNeXT: InvResMLP blocks
│   │   │   ├── pointnext_tokenizer.py  # PointNeXT tokenizer: patch embedding
│   │   │   └── point_pn.py         # PointPN: LGA (Local Geometry Aggregation) blocks
│   │   ├── rgb/
│   │   │   ├── registry.py         # build_rgb_encoder: resnet/clip/dino/siglip
│   │   │   ├── resnet.py           # ResNet18/34: torchvision pretrained
│   │   │   ├── clip.py             # CLIP ViT: openai/clip-vit-base-patch16
│   │   │   ├── dino.py             # DINOv2: facebook/dinov2-base
│   │   │   └── siglip.py           # SigLIP: google/siglip-base-patch16-224
│   │   ├── text/
│   │   │   ├── clip.py             # CLIPTextEncoder: CLIP text encoder, 冻结
│   │   │   └── t5.py               # T5Encoder: google/t5-base (未使用)
│   │   ├── proprio/
│   │   │   └── state_mlp.py        # StateMLP: 2-layer MLP for joint_state
│   │   └── plugins/
│   │       ├── moe.py              # MoEGating: top-k expert routing + aux loss (load_balancing + entropy)
│   │       └── token_compressor.py # TokenCompressor: cross-attention token reduction (未使用)
│   └── common/
│       ├── param_counter.py        # print_param_count: 统计可训练/冻结参数
│       ├── optim_util.py           # get_optim_groups: weight decay 分组
│       └── module_attr_mixin.py    # ModuleAttrMixin: 动态属性访问
├── datasets/
│   ├── base_dataset.py             # BaseDataset: Zarr → ReplayBuffer → SequenceSampler → __getitem__ (增强 + torch)
│   ├── pc_dataset.py               # PCDataset: BaseDataset + PC 增强
│   ├── rgb_dataset.py              # RGBDataset: BaseDataset + RGB 增强
│   ├── rgb_pc_dataset.py           # RGBPCDataset: BaseDataset + RGB + PC 增强
│   ├── multi_task_dataset.py       # MultiTaskDataset: 多任务采样 (proportional/balanced/weighted), 注入 task_text/task_name
│   ├── common/
│   │   ├── replay_buffer.py        # ReplayBuffer: Zarr → numpy 全量加载
│   │   └── sampler.py              # SequenceSampler: numba 滑动窗口, pad_before/pad_after
│   └── augmentation/
│       ├── base.py                 # BaseAugmentation: prob 控制执行概率
│       ├── pc_spatial.py           # PCSpatialAugmentation: rot_z, trans_xy, scale
│       ├── pc_color.py             # PCColorAugmentation: brightness, contrast, saturation, hue
│       ├── pc_dropout.py           # PCDropoutAugmentation: 随机丢弃点
│       ├── rgb.py                  # RGBColorAugmentation: torchvision ColorJitter
│       └── state.py                # StateNoiseAugmentation: Gaussian noise
├── training/
│   ├── trainer.py                  # Trainer: train_one_step, validate, evaluate, finish_epoch
│   ├── ddp_trainer.py              # DDPTrainer: 继承 Trainer, 覆盖 train_one_step (DDP wrapper), finish_epoch (rank 0 独享)
│   ├── sim_evaluator.py            # SimEvaluator: _load_for_inference, run (env_runner.run → video + metrics)
│   └── common/
│       ├── workspace.py            # TrainWorkspace: output_dir, wandb, checkpoint_store, topk_tracker, save_hydra_config
│       ├── checkpoint_io.py        # CheckpointStore (save/load/resolve_path), TopKCheckpointTracker (update/best_path)
│       ├── ema_model.py            # EMAModel: exponential moving average, power schedule
│       ├── lr_scheduler.py         # get_scheduler: cosine/linear/constant with warmup
│       └── logging.py              # to_log_scalars: 递归提取 scalar 值
├── env_runner/
│   ├── base_runner.py              # BaseRunner: eval_one_episode, run (循环 eval_seeds, 异常处理, 统计指标)
│   ├── sim_runner.py               # SimRunner: make_env (动态导入 dexmani_sim.envs.<task_name>), get_seed_list
│   └── multi_task_sim_runner.py    # MultiTaskSimRunner: 多任务评测, task_configs
└── common/
    ├── normalizer.py               # LinearNormalizer: fit (limits/gaussian), normalize/unnormalize, DictOfTensorMixin (nn.Module)
    ├── pytorch_util.py             # set_seed, fix_state_dict (DDP module. 前缀转换), worker_init_fn, dict_apply
    └── resolver.py                 # register_resolvers: ${eval:'...'} 数学插值
```


## Main Entrypoints

| 入口 | 模式 | 关键差异 |
|---|---|---|
| `train.py` | 单卡 | `@hydra.main`，`build_train_components()` 完整装配 |
| `train_ddp.py` | 多卡 DDP | `mp.spawn(ddp_worker, nprocs=N)`，两阶段 seed，DistributedSampler，rank 0 独享 logging/checkpoint/eval |
| `train_multi_task.py` | 多任务 | 复用 `build_train_components()`，MultiTaskDataset + MultiTaskAgent，task_texts 预计算 CLIP embedding cache |
| `eval_sim.py` | 独立评测 | Hydra-free CLI，CheckpointStore 直接加载，`eval.seed=0` 固定，输出 video mp4 + metrics.json |
| `smoke_test.py` | 冒烟测试 | Hydra compose API，复用 `train.py` 构建函数，forward + backward + predict_action，assert 形状和有限性 |

## Embodied Module Breakdown

### Observation Space
- **joint_state**: (19,) 绝对关节角，XArm7 (7 DoF) + XHand (12 DoF)
- **point_cloud**: (1024, 3/6) 点云，3 维 XYZ 或 6 维 XYZRGB
- **rgb**: (3, 224, 224) 单视角 RGB 图像
- **task_text**: 字符串，自然语言任务描述（MultiTask 专用）

### Action Space
- **action**: (19,) 绝对关节角，与 joint_state 同维度
- **归一化**: LinearNormalizer, mode='limits', 缩放到 [-1,1]
- **执行**: 预测 16 步轨迹，实际执行 `pred[:, 1:9]` 共 8 步

### Dataset Format (Zarr)
```
<task_name>.zarr/
├── joint_state: (N, 19) float32
├── action: (N, 19) float32
├── point_cloud: (N, 1024, 3/6) float32
├── rgb: (N, 3, 224, 224) uint8 (可选)
└── episode_ends: (M,) int64  # 累积索引
```

### Policy Architecture

**DP**: RGB → DINO/CLIP → FiLM condition → ConditionalUnet1D → Diffusion  
**DP3**: PC → iDP3/PointNeXT → FiLM condition → ConditionalUnet1D → Diffusion  
**ManiFlow**: PC → PointPN tokenizer → cross-attention condition → DiTXFlowMatch → FlowMatch + Consistency  
**MoE**: PC → iDP3 → MoE gating → FiLM condition → ConditionalUnet1D → Diffusion + aux_loss  
**MultiTask**: RGB + Text → DINO + CLIP Text → full condition → DiT_Diffusion → Diffusion

### Loss Functions

**Diffusion (DP/DP3/MoE/MultiTask)**:
```python
# diffusion.py L32-58
noise = torch.randn_like(actions)
timestep = torch.randint(0, num_train_timesteps, (B,))
noisy_action = noise_scheduler.add_noise(actions, noise, timestep)
pred = backbone(x=noisy_action, timestep=timestep, context=cond)

if prediction_type == 'epsilon':
    target = noise
elif prediction_type == 'sample':
    target = actions

loss = F.mse_loss(pred, target, reduction='none').mean()
```

**FlowMatch (ManiFlow)**:
```python
# flowmatch.py L115-184
flow_batchsize = max(1, min(B-1, int(B * flow_batch_ratio)))
consistency_batchsize = B - flow_batchsize

# Flow loss (target_t=0)
t_flow = sample_t(flow_batchsize, mode='beta')
xt_flow = (1-t_flow) * noise + t_flow * actions
vt_flow_target = actions - noise
pred_vt_flow = backbone(xt_flow, t_flow, target_t=0, cond)
loss_flow = F.mse_loss(pred_vt_flow, vt_flow_target).mean()

# Consistency loss (target_t=dt>0)
t_ct = sample_t(consistency_batchsize, mode='discrete')
dt = sample_dt(consistency_batchsize, mode='uniform')
xt_ct = (1-t_ct) * noise + t_ct * actions
xt_next = (1-t_ct-dt) * noise + (t_ct+dt) * actions
with torch.no_grad():
    pred_x1_from_ema = ema_backbone(xt_next, t_ct+dt, target_t=0, cond)
vt_ct_target = (pred_x1_from_ema - xt_ct) / (1 - t_ct)
pred_vt_ct = backbone(xt_ct, t_ct, target_t=dt, cond)
loss_consistency = F.mse_loss(pred_vt_ct, vt_ct_target).mean()

loss = loss_flow + loss_consistency
```

**MoE aux_loss**:
```python
# moe.py
aux_loss = load_balancing_loss + entropy_loss
total_loss = action_loss + aux_loss
```

### Evaluation Protocol

- **训练期评测**: 每 `eval_interval_epochs` 触发，用 EMA 模型，结果以 `eval/` 前缀写入 wandb/jsonl，用于 TopK 排序
- **独立离线评测**: `eval_sim.py`，参数由 `eval.offline` 配置段控制，输出 video mp4 + metrics.json
- **指标**: `success_rate`（done 比例）、`avg_steps`（成功 episode 的平均步数）
- **TopK 排序**: monitor `test_mean_score` = `eval/success_rate`，mode `max`，默认 topk=3


## Reproducibility Checklist

### Seed Points
- **训练期**: `set_seed(cfg.training.seed)` 在模型初始化前（train.py:285）
- **DDP 两阶段**: 模型初始化前 `set_seed(seed)` 保证 rank 权重一致；模型构建后 `set_seed(seed + rank)` 增加各 rank 数据增强多样性（train_ddp.py:51, 94）
- **评测期**: `set_seed(cfg.eval.seed)` 固定值 0，不随 `training.seed` 变化（eval_sim.py:41）
- **DataLoader**: `worker_init_fn` 设置每个 worker 的 seed 为 `base_seed + worker_id`（pytorch_util.py:48-52）

### Normalizer
- **拟合**: `LinearNormalizer.fit(train+val 全量, mode='limits')`，缩放到 [-1,1]（base_dataset.py:107-114）
- **range_eps**: 变化范围 < 2% 的维度 zero-center 但不缩放（`ignore_dim` 逻辑），防止微小抖动被放大（normalizer.py:68-72）
- **持久化**: 存储在 `agent.state_dict()` 中（通过 `nn.ModuleDict`），checkpoint 保存/加载时随 agent 一起持久化（base.py:34-35）
- **DDP 同步**: normalizer 通过 `dist.broadcast` 从 rank 0 同步（train_ddp.py 中 `build_dataset_and_normalizer` 逻辑）

### Checkpoint Format
- **格式**: `simple.v1`，`{state: {...}, weights: {...}}`（checkpoint_io.py:34-58）
- **latest.pt**: 符号链接，指向最新 checkpoint（workspace.py:196-233）
- **TopK**: monitor `test_mean_score`，mode `max`，默认 topk=3（workspace.py:173-195）
- **DDP 兼容**: `fix_state_dict()` 自动处理单卡 ↔ DDP 的 `module.` 前缀转换（pytorch_util.py:55-72）

### DDP Synchronization
- **backend**: nccl（train_ddp.py:30-36）
- **DistributedSampler**: 分片，`dataloader.batch_size` 是每卡值（总 batch = × N）（train_ddp.py:57-63）
- **rank 0 独享**: logging/checkpoint/eval（ddp_trainer.py:15-82）
- **梯度同步**: DDP wrapper 自动在 backward 后同步梯度（ddp_trainer.py:38-48）

### Eval Determinism
- **eval.seed**: 固定值 0，不随 `training.seed` 变化（eval_sim.py:41）
- **eval_seeds**: `get_seed_list(eval_episodes, seed)` 生成确定性种子列表（sim_runner.py:20-21）
- **env.reset(seed)**: 每个 episode 使用固定 seed（base_runner.py:110-137）
- **denoise_timesteps**: 推理步数固定，DDIM/Euler ODE 采样确定性（diffusion.py:79-84, flowmatch.py:224-231）

## Ablation Surface

### Config-Level Ablations (无需修改代码)

**Encoder 变体**:
- `agent.obs_encoder.pc_encoder.name`: `idp3` / `pointnext` / `dp3` / `pointpn`
- `agent.obs_encoder.rgb_encoder.name`: `resnet18` / `clip` / `dino` / `siglip`
- `agent.obs_encoder.text_encoder.model_name`: `openai/clip-vit-base-patch16` / `google/t5-base`

**Backbone 变体**:
- `agent.action_decoder.backbone.down_dims`: `[256, 512, 1024]` → 调整 UNet 深度
- `agent.action_decoder.backbone.num_blocks`: DiT/DiTX 的 transformer block 数量
- `agent.action_decoder.backbone.num_heads`: attention head 数量

**Decoder 超参**:
- `agent.action_decoder.num_train_timesteps`: Diffusion 训练步数（默认 100）
- `agent.action_decoder.denoise_timesteps`: 推理步数（默认 10）
- `agent.action_decoder.prediction_type`: `epsilon` / `sample`
- `agent.action_decoder.flow_batch_ratio`: FlowMatch 中 flow/consistency 批次比例（默认 0.5）
- `agent.action_decoder.use_ema_teacher_for_consistency`: FlowMatch 是否用 EMA 作为 consistency teacher

**Modality Dropout**:
- `agent.modality_dropout_probs`: `{joint_state: 0.1}` → per-modality 独立 dropout 概率

**Augmentation**:
- `dataset.augmentation.pc_spatial.prob`: PC 空间增强概率
- `dataset.augmentation.pc_color.prob`: PC 颜色增强概率
- `dataset.augmentation.rgb.prob`: RGB 颜色增强概率
- `dataset.augmentation.state.prob`: state 噪声增强概率

**Optimizer & Scheduler**:
- `optimizer.lr`: 学习率（默认 1e-4）
- `optimizer.obs_lr`: obs encoder 学习率（默认继承 `lr`）
- `optimizer.weight_decay`: 权重衰减（默认 1e-6）
- `training.lr_scheduler.name`: `cosine` / `linear` / `constant`
- `training.lr_scheduler.warmup_steps`: warmup 步数

**Training Loop**:
- `training.loop.num_epochs`: 训练轮数
- `training.loop.grad_accum_steps`: 梯度累积步数
- `training.loop.max_grad_norm`: 梯度裁剪阈值（默认 1.0）
- `training.use_ema`: 是否使用 EMA
- `training.ema_decay`: EMA 衰减率（默认 0.995）

### Code-Level Ablations (需要修改代码)

**Encoder 架构修改**:
- `agents/obs_encoder/pointcloud/`: 修改 PointNet/iDP3/PointNeXT/PointPN 架构
- `agents/obs_encoder/rgb/`: 修改 ResNet/CLIP/DINO/SigLIP 架构
- `agents/obs_encoder/plugins/moe.py`: 修改 MoE gating 逻辑（top-k, routing 策略）

**Backbone 架构修改**:
- `agents/action_decoders/backbone/unet1d.py`: 修改 UNet 架构（FiLM conditioning, down/mid/up blocks）
- `agents/action_decoders/backbone/dit.py`: 修改 DiT 架构（self-attention, AdaLN）
- `agents/action_decoders/backbone/ditx.py`: 修改 DiTX 架构（cross-attention）

**Decoder 逻辑修改**:
- `agents/action_decoders/diffusion.py`: 修改 Diffusion 采样策略（DDIM → DDPM/DPM-Solver）
- `agents/action_decoders/flowmatch.py`: 修改 FlowMatch 路径（直线 → 曲线）、consistency training 逻辑
- `agents/action_decoders/common/sample.py`: 修改时间采样策略（beta/discrete/uniform）

**Loss 函数修改**:
- `agents/core/base.py`: 修改 `compute_loss` 逻辑（加权、多任务 loss）
- `agents/obs_encoder/plugins/moe.py`: 修改 MoE aux_loss（load_balancing + entropy 权重）

**Normalizer 修改**:
- `common/normalizer.py`: 修改归一化策略（limits → gaussian, range_eps 阈值）

**Augmentation 修改**:
- `datasets/augmentation/`: 修改增强逻辑（旋转角度、平移范围、颜色抖动强度）


## Open Questions

### 1. FlowMatch Consistency Training 的 target_t 语义
- **位置**: `agents/action_decoders/flowmatch.py:115-184`
- **问题**: `target_t` 作为 mode indicator（`target_t=0` 为 flow mode，`target_t>0` 为 consistency mode），但 backbone 如何利用 `target_t` 信息？DiTXFlowMatch 中 `target_t` 是否作为额外输入？
- **影响**: 如果 backbone 未使用 `target_t`，consistency training 可能无法区分两种模式，导致训练不稳定。
- **验证**: 检查 `agents/action_decoders/backbone/ditx.py` 中 `forward` 签名是否包含 `target_t` 参数。

### 2. Modality Dropout 的执行时机
- **位置**: `agents/core/base.py:37-48`
- **问题**: Modality dropout 在 truncate 之前执行，同一 batch 样本的两个时序步共享 dropout 状态。这是否符合预期？如果希望每个时序步独立 dropout，需要在 truncate 后执行。
- **影响**: 当前实现下，如果 `joint_state` 被 dropout，则 `obs[:, 0]` 和 `obs[:, 1]` 都会被置零。这可能不符合时序建模的直觉。
- **验证**: 检查 `preprocess` 中 modality dropout 的执行顺序，确认是否需要调整。

### 3. MultiTask 固定索引生成的确定性
- **位置**: `datasets/multi_task_dataset.py`
- **问题**: 固定索引生成支持 `proportional`/`balanced`/`weighted` 三种策略，MD5 hash 保证确定性。但 hash 输入是什么？如果包含 `training.seed`，则不同 seed 下固定索引不同，可能影响多任务训练的可复现性。
- **影响**: 如果固定索引依赖 `training.seed`，则相同 checkpoint 在不同 seed 下评测结果可能不同。
- **验证**: 检查 `MultiTaskDataset` 中固定索引生成逻辑，确认 hash 输入是否包含 seed。

### 4. DDP Normalizer 同步的实现
- **位置**: `train_ddp.py:54`（`build_dataset_and_normalizer(cfg, rank=rank, world_size=world_size)`）
- **问题**: CLAUDE.md 提到 "normalizer 通过 `dist.broadcast` 从 rank 0 同步"，但 `build_dataset_and_normalizer` 函数签名中未见 `rank`/`world_size` 参数（train.py:23-39）。DDP 版本是否有单独的实现？
- **影响**: 如果 normalizer 未正确同步，各 rank 的归一化参数不一致，导致训练不稳定。
- **验证**: 检查 `train_ddp.py` 中 `build_dataset_and_normalizer` 的实际实现，确认 normalizer 同步逻辑。

### 5. Eval 期间 `eval_one_episode` 返回最后一帧 `done`
- **位置**: `env_runner/base_runner.py:110-137`
- **问题**: CLAUDE.md 提到 "`eval_one_episode` 返回最后一帧 `done`（而非首次 done）"，但 `task_done_step` 已通过 `prev_done` guard 正确捕获首次成功步数。这是否意味着返回的 `done` 值本身无关紧要，只有 `task_done_step` 有意义？
- **影响**: 如果 `done` 值被用于其他逻辑（如 success_rate 计算），可能导致指标不准确。
- **验证**: 检查 `base_runner.py` 中 `eval_one_episode` 的返回值如何被使用，确认 `done` 值的语义。

### 6. FlowMatch 各阶段模型使用的一致性
- **位置**: `training/trainer.py`, `agents/action_decoders/flowmatch.py`
- **问题**: CLAUDE.md 提到训练/验证阶段传入 `self.ema_model.action_decoder.model` 作为 EMA backbone，但采样/Eval 阶段直接使用 EMA 模型。这两种使用方式是否一致？
- **影响**: 如果训练时 consistency loss 使用的 EMA backbone 与推理时的模型不一致，可能导致训练-推理 gap。
- **验证**: 检查 `FlowMatchWithConsistency.compute_loss` 和 `predict_action` 中 EMA 模型的使用方式，确认一致性。

### 7. 梯度累积边界未完成累积的处理
- **位置**: `training/trainer.py:94-103`
- **问题**: epoch 边界未完成累积的梯度会 scale 后 flush。scale 因子是什么？是否为 `remaining_steps / grad_accum_steps`？
- **影响**: 如果 scale 不正确，epoch 边界的梯度更新可能过大或过小，影响训练稳定性。
- **验证**: 检查 `flush_gradient_accumulation` 中的 scale 逻辑，确认计算方式。

### 8. Loss NaN 检测后的恢复策略
- **位置**: `training/trainer.py:111-112`
- **问题**: `torch.isfinite(loss)` 检查失败时立即 `RuntimeError`，训练中断。是否有更优雅的恢复策略（如跳过该 batch、回滚到上一个 checkpoint）？
- **影响**: 单个 batch 的 NaN 可能导致整个训练失败，浪费计算资源。
- **验证**: 讨论是否需要添加 NaN 恢复策略，或在配置中提供选项。

