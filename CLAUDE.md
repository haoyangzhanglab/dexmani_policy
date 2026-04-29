# dexmani_policy

灵巧操作模仿学习策略框架。4 种策略 × 多模态观测 × 单卡/DDP 训练。

## 常用命令

```bash
# 单卡训练
python dexmani_policy/train.py --config-name=dp3
python dexmani_policy/train.py --config-name=maniflow

# DDP 多卡训练
python dexmani_policy/train_ddp.py --config-name=maniflow_ddp
bash scripts/train_ddp.sh dp3 pick_apple_messy 4

# 仿真评估
python dexmani_policy/eval_sim.py --policy-name dp3 --task-name pick_apple_messy --exp-name 2026-04-01_11-18_233

# 单文件 smoke test
python -m dexmani_policy.agents.core.dp3
python -m dexmani_policy.agents.core.maniflow
```

## 策略一览

| Agent | 观测 | Encoder | Backbone | Decoder | Config |
|-------|------|---------|----------|---------|--------|
| `DPAgent` | RGB + proprio | DINO/CLIP/ResNet/SigLIP + StateMLP | ConditionalUnet1D | DDIM Diffusion | `dp.yaml` |
| `DP3Agent` | 点云 + proprio | PointNet/iDP3 + StateMLP | ConditionalUnet1D | DDIM Diffusion | `dp3.yaml` |
| `MoEAgent` | 点云 + proprio | iDP3 + StateMLP + MoE Router | ConditionalUnet1D | DDIM Diffusion | `moe_dp3.yaml` |
| `ManiFlowAgent` | 点云 + proprio | PointPN/PointNext Tokenizer + StateMLP | DiT-X Transformer | Flow + Consistency | `maniflow.yaml` |

## 目录结构

```
dexmani_policy/
├── configs/                     # Hydra YAML（dp3/dp/maniflow/moe_dp3/maniflow_ddp）
├── agents/
│   ├── core/                    # Agent 定义（base → dp/dp3/moe/maniflow）
│   ├── action_decoders/         # Diffusion(DDIM) / FlowMatchWithConsistency
│   │   ├── backbone/            # ConditionalUnet1D / DiTXFlowMatch / DiTXDiffusion
│   │   └── common/sample.py     # t 采样策略（beta/uniform/lognorm/cosmap/discrete）
│   ├── obs_encoder/
│   │   ├── pointcloud/          # PointNet / iDP3 / PointNext / PointPN Tokenizer
│   │   ├── rgb/                 # ResNet / DINO / CLIP / SigLIP + ImageProcessor
│   │   ├── proprio/             # StateMLP
│   │   └── plugins/             # MoE / TokenCompressor
│   └── common/                  # optim_util / param_counter
├── datasets/                    # Zarr replay buffer → PC/RGB/RGBPC Dataset
├── training/
│   ├── trainer.py               # 单卡训练循环
│   ├── ddp_trainer.py           # DDP 训练封装
│   ├── sim_evaluator.py         # 评估 + 视频录制
│   └── common/                  # workspace / checkpoint / EMA / lr_scheduler / logging
├── env_runner/                  # BaseRunner / SimRunner（依赖 dexmani_sim）
├── common/                      # LinearNormalizer / pytorch_util
├── train.py                     # 单卡入口
├── train_ddp.py                 # DDP 入口
└── eval_sim.py                  # 评估入口
```

## 核心数据流

```
训练:
  batch['obs'] → normalize → flatten(B*T) → obs_encoder → cond
  batch['action'] → normalize → action_decoder.compute_loss(cond, nactions) → loss

推理:
  obs_dict → preprocess → obs_encoder → cond
  randn template → action_decoder.predict_action(cond, template, denoise_steps)
  → unnormalize → pred[:, s:s+n_action_steps] = control_action
```

## 关键设计点

- **Hydra 配置驱动**: 所有组件通过 `_target_` 实例化，切换策略只需换 config
- **EMA**: 训练时维护 EMA 模型，用于 consistency teacher 和评估
- **ManiFlow 双目标**: `flow_batch_ratio` 部分做 flow matching，`consistency_batch_ratio` 部分做 consistency distillation（EMA teacher 提供目标）
- **MoE 辅助损失**: `MoEAgent.compute_loss` 返回 action_loss + load_balance_loss + entropy_loss
- **条件注入**: UNet1D 支持 `film`（全局向量）和 `cross_attention_film`（序列 token）；DiT-X 通过 AdaLN + Cross-Attention
- **Normalizer**: `LinearNormalizer` 对 joint_state/action 做 limits 归一化到 [-1,1]；点云/RGB 不归一化
- **数据**: Zarr episode replay buffer，`SequenceSampler` 按 horizon 窗口采样，episode 级 train/val 划分
- **Checkpoint**: topk by success_rate，latest symlink 支持断点续训

## 默认超参

- `horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19`
- `num_epochs=1000, lr=1e-4, cosine scheduler, warmup=500 steps`
- `EMA: power=0.75, max_value=0.9999`
- `Diffusion: 100 train steps / 10 inference steps, DDIM, prediction_type=sample`
- `FlowMatch: denoise_timesteps=10, t~Beta(1,1.5), flow:consistency=0.75:0.25`

## 外部依赖

- **dexmani_sim**: 仿真环境包（不在本仓库），SimRunner 通过 `importlib` 动态加载
- **Python 包**: torch, hydra-core, omegaconf, diffusers, einops, timm, wandb, zarr, torchvision, transformers, termcolor, imageio
