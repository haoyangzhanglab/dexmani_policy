# dexmani_policy

灵巧操作模仿学习策略框架。5 种策略 × 多模态观测 × 单卡/DDP 训练。

## 快速开始

```bash
# 单卡训练
bash scripts/train.sh dp3
bash scripts/train.sh maniflow

# 多任务训练
bash scripts/train_multi_task.sh multi_task_dp3

# DDP 多卡训练（仅支持单任务）
bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[0,1,2,3]

# 仿真评估
bash scripts/eval_sim.sh dp3 pick_apple_messy 2026-04-01_11-18_233
```

## 策略架构

| Agent | 观测 | Encoder | Backbone | Decoder |
|-------|------|---------|----------|---------|
| DP | RGB + proprio | DINO/CLIP/ResNet/SigLIP + StateMLP | UNet1D | DDIM |
| DP3 | 点云 + proprio | PointNet/iDP3 + StateMLP | UNet1D | DDIM |
| MoE | 点云 + proprio | iDP3 + MoE Router + StateMLP | UNet1D | DDIM |
| ManiFlow | 点云 + proprio | PointPN/PointNext + StateMLP | DiT-X | Flow + Consistency |
| MultiTask | 同 base_agent | base_encoder + TaskEmbedding + FiLM | 同 base | 同 base |

## 目录结构

```
dexmani_policy/
├── configs/                     # Hydra 配置（dp/dp3/moe_dp3/maniflow/multi_task_dp3）
├── agents/
│   ├── core/                    # 策略定义（base → dp/dp3/moe/maniflow/multi_task）
│   ├── action_decoders/         # Diffusion / FlowMatch + Consistency
│   ├── obs_encoder/             # 点云/RGB/proprio/text 编码器 + MoE 插件
│   └── common/                  # 优化器工具 / 参数统计
├── datasets/                    # Zarr replay buffer → Dataset
├── training/                    # Trainer / MultiTaskTrainer / DDPTrainer
├── env_runner/                  # SimRunner / MultiTaskSimRunner
├── common/                      # LinearNormalizer / pytorch_util
├── train.py                     # 单卡训练入口
├── train_multi_task.py          # 多任务训练入口
├── train_ddp.py                 # DDP 训练入口
└── eval_sim.py                  # 评估入口
```

## 核心数据流

**训练**:
```
batch['obs'] → normalize → flatten(B*T) → obs_encoder → cond
batch['action'] → normalize → action_decoder.compute_loss(cond, nactions) → loss
```

**推理**:
```
obs_dict → preprocess → obs_encoder → cond
randn → action_decoder.predict_action(cond, denoise_steps) → unnormalize → action
```

## 关键机制

- **Hydra 配置驱动**: 所有组件通过 `_target_` 实例化，切换策略只需换 config
- **EMA 模型**: 训练时维护 EMA，用于 consistency teacher 和评估
- **ManiFlow**: flow_batch_ratio 做 flow matching，consistency_batch_ratio 做 consistency distillation
- **MoE**: 返回 action_loss + load_balance_loss + entropy_loss
- **MultiTask**: TaskEmbedding → FiLM(scale, shift) 调制 cond，MultiTaskTrainer 每 epoch 调用 set_epoch() 改变任务采样
- **条件注入**: UNet1D 支持 film / cross_attention_film，DiT-X 通过 AdaLN + Cross-Attention
- **Normalizer**: joint_state/action 归一化到 [-1,1]，点云/RGB 不归一化
- **数据采样**: Zarr episode buffer，SequenceSampler 按 horizon 窗口采样，episode 级 train/val 划分
- **Checkpoint**: topk by success_rate，latest symlink 支持断点续训

## 默认超参

```python
horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19
num_epochs=1000, lr=1e-4, cosine scheduler, warmup=500 steps
EMA: power=0.75, max_value=0.9999
Diffusion: 100 train steps / 10 inference steps, DDIM
FlowMatch: 10 denoise steps, t~Beta(1,1.5), flow:consistency=0.75:0.25
```

## 依赖

- **dexmani_sim**: 仿真环境（外部包，通过 importlib 动态加载）
- **核心库**: torch, hydra-core, diffusers, einops, timm, wandb, zarr
