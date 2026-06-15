# CLAUDE.md — dexmani_policy

灵巧手操作模仿学习框架。Hydra 配置驱动，Zarr replay buffer，Diffusion/FlowMatch 动作解码，`dexmani_sim` 仿真评测。

> 本文件为 Claude 工作速查索引。详细设计见 `docs/`，审查约定见 `docs/08-审查约定.md`。

## 环境与常用命令

- Conda env: `policy`；仿真评测依赖 `dexmani_sim`

```bash
# 单卡训练
bash scripts/train.sh dp3                # dp / dp3 / maniflow / moe_dp3 / r3d
bash scripts/train.sh dp3 'training.loop.num_epochs=10'  # Hydra override

# 多卡 DDP
bash scripts/train_ddp.sh maniflow_ddp    # dp_ddp / maniflow_ddp / multitask_dit_ddp / r3d_ddp

# 多任务训练
bash scripts/train.sh multitask_dit

# 冒烟测试（构建链完整性）
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py dp3 maniflow moe_dp3 r3d

# 仿真评测
bash scripts/eval_sim.sh dp3 pick_apple_messy <exp_dir>
```

## 架构

```
Hydra config (configs/*.yaml) → Dataset (Zarr → ReplayBuffer → SequenceSampler)
  → Agent (obs_encoder + action_decoder) → Trainer (loss → backward → EMA → checkpoint)
  → SimEvaluator (env_runner.run → success_rate)
```

核心不变约束：`horizon=16  n_obs_steps=2  n_action_steps=8  action_dim=19`

### 入口点

| 入口 | 模式 | 关键差异 |
|---|---|---|
| `train.py` | 单卡 | `@hydra.main`，`build_train_components()` 完整装配 |
| `train_ddp.py` | 多卡 DDP | `mp.spawn(ddp_worker, nprocs=N)`，复用单卡构建函数 |
| `eval_sim.py` | 独立评测 | Hydra-free CLI，`CheckpointStore` 直接加载 checkpoint；`hasattr(cfg, 'eval')` 前置保护兼容历史 config |

### Agent 变体

| Agent | 输入 | Encoder | Backbone | Decoder | 配置 |
|---|---|---|---|---|---|
| `DPAgent` | RGB+state | DINO/CLIP/SigLIP+StateMLP | `ConditionalUnet1D` (FiLM) | `Diffusion` | `dp.yaml` |
| `DP3Agent` | PC+state | iDP3/PointNeXT+StateMLP | `ConditionalUnet1D` (FiLM) | `Diffusion` | `dp3.yaml` |
| `ManiFlowAgent` | PC+state | PointNeXT+StateMLP | `DiTXFlowMatch` (cross-attn) | `FlowMatchWithConsistency` | `maniflow.yaml` |
| `MoEAgent` | PC+state | DP3 encoder+MoE gating | `ConditionalUnet1D` | `Diffusion` | `moe_dp3.yaml` |
| `MultiTaskAgent` | RGB+state+text | DINO/CLIP/SigLIP+CLIP Text+StateMLP | `DiT_Diffusion` (self-attn+AdaLN) | `Diffusion` | `multitask_dit.yaml` |
| `R3DAgent` | PC+state | Uni3D (ViT-tiny)+StateMLP | `OneWayTransformer` (cross-attn) | `Diffusion` | `r3d.yaml` |

> **视觉 backbone 加载约定**: DINO/CLIP/SigLIP 均以 `torch_dtype=torch.bfloat16` 加载（参数显存减半）；CLIP/SigLIP 额外启用 `attn_implementation="flash_attention_2"`（DINOv2 默认 SDPA 无需变更）。LoRA 参数自动对齐 backbone dtype。

## 关键数据流

### 数据加载

```
Zarr (robot_data/sim/<task>.zarr) → ReplayBuffer.copy_from_path()  # 全量 numpy, float32
  → SequenceSampler (numba, pad_before=1, pad_after=7)             # 滑动窗口
    → 短 episode (<8帧) 自动 warn + skip（不 crash）
  → BaseDataset.__getitem__() (增强 + numpy→torch) → DataLoader → batch (B,16,*)
```

### 训练前向 (Agent.compute_loss)

```
obs (B,16,*) → normalizer.normalize() → modality_dropout → truncate[:,:2] → flatten → encoder → cond
action (B,16,19) → normalizer['action'].normalize()  # → [-1,1]

cond + action → action_decoder.compute_loss()
  [Diffusion]:  noise→denoise→MSE(pred, target)
  [FlowMatch]:  拆分 flow/consistency → 速度 MSE(v_pred, x1-x0) + consistency teacher
  [MoE]:        + aux_loss (load_balancing + entropy)

loss.backward() → optimizer.step → scheduler.step → EMA
```

### 推理 (Agent.predict_action)

```
obs_dict (无 T 维度) → normalize+truncate+flatten+encoder → cond
  → action_decoder.predict_action(cond, template=zeros(B,16,19))
    [Diffusion]:  DDIM 迭代去噪 (默认 10 步)
    [FlowMatch]:  Euler ODE 积分 (默认 10 步, target_t=dt)
  → unnormalize → control_action = pred[:, 1:9, :]  # (B,8,19)
```

### 评测与检查点

```
SimEvaluator: _load_for_inference(ckpt_tag, use_ema=True)
  → env_runner.run(agent) → {success_rate, avg_steps, videos}

Checkpoint: finish_epoch() → TrainCheckpoint → .tmp → os.replace() → epoch=XXXX-step=YYYY.pt
  → save_latest() (symlink) + save_topk() (monitor test_mean_score, topk=3)
Resume: load_for_resume("latest") → 恢复 model/ema/opt/sched + normalizer
DDP: fix_state_dict() 自动处理 module. 前缀 → checkpoint 始终以 unwrapped 格式保存
```

### 数据流形状

```
Zarr:   joint_state (N,19)  action (N,19)  point_cloud (N,1024,3|6)
Sample: obs (16,*)          action (16,19)
Batch:  obs (B,16,*)        action (B,16,19)
Preprocessed: obs (B×2,*)   # truncate → flatten batch+time
Cond:   (B, out_dim×2)      # UNet/DP/DP3/MoE; ManiFlow: (B, N_tokens, token_dim)
```

### 文件组织

```
dexmani_policy/
  train.py, train_ddp.py, eval_sim.py, smoke_test.py
  configs/               # Hydra YAML（6 策略 + dataset preset）
  agents/core/           # BaseAgent → DP/DP3/ManiFlow/MoE/MultiTask
  agents/action_decoders/ # Diffusion, FlowMatchWithConsistency, backbone/(unet1d, dit, ditx)
  agents/obs_encoder/    # pointcloud, rgb, text, plugins/(moe, token_compressor)
  datasets/              # BaseDataset → PC/RGB/RGBPC/MultiTask; common/(ReplayBuffer, Sampler)
  training/              # Trainer, DDPTrainer, SimEvaluator, workspace, checkpoint_io, ema
  env_runner/            # BaseRunner, SimRunner, MultiTaskSimRunner
  common/                # LinearNormalizer, pytorch_util, resolver
```

---

## 关键约定

### 配置与数据
- Hydra + OmegaConf，`${eval:'...'}` 插值在 `common/resolver.py` 注册
- CLI override 任意字段；配置校验基于字段存在性判断，不依赖 `_target_` 字符串匹配
- `eval.seed: 0` 固定，保证同一 checkpoint 多次评测可复现
- Normalizer: `mode='limits'`，train+val 全量拟合 → [-1,1]；`range_eps=1e-4`（与官方 diffusion_policy 一致），低方差维度 zero-center 不缩放（scale=1.0, offset=-mean）
- 数据增强默认禁用，通过 `prob` 控制执行概率；`pc_dim` 必须与 Zarr 点云通道数一致
- 数据增强（`augmentation_cfg`）与 modality dropout（`modality_dropout_probs`）职责不同：增强生成合理的观测变体（加噪、旋转、颜色抖动），在 Dataset 层 normalize 前执行；modality dropout 是模型正则化，故意将整个模态置零来防止过拟合，在 Agent 层 normalize 后执行
- SequenceSampler：短于 8 帧的 episode 自动 warn + skip（不 crash）；边界 padding 复制首尾帧
- 配置全链路可追踪，所有 YAML key 均可追溯到代码消费点，**无 dead config**

### 训练
- `modality_dropout_probs`: **模型正则化**（非数据增强）。per-modality 独立 dropout，仅对已归一化模态生效（`k in normalizer.params_dict`），truncate 前执行，同一样本两时序步共享 dropout 状态。语义为"该模态完全不可用"，与数据增强的"生成合理变体"不同
- FlowMatch `target_t` 语义：`target_t=0` → 预测瞬时速度 `v = x1-x0`（rectified flow 直线路径的解析导数）；`target_t>0` → 预测向 x1 的速度。直线路径下两者目标一致
- FlowMatch consistency training：Teacher EMA 估计 `pred_x1_ct` → 推导 target velocity → 学生 MSE 匹配。t_next=1.0 时 target 退化为精确的 `x1-x0`（约 45% 样本）。Teacher 在 `no_grad()` + `eval()` 下运行
- `use_ema_teacher_for_consistency=true` 仅 ManiFlow 需要：推理时 `target_t=dt>0` 依赖 consistency 训练泛化

| 阶段 | Student（预测） | Teacher（target） | Loss |
|---|---|---|---|
| 训练 | `self.model.action_decoder.model` | `self.ema_model.action_decoder.model` | flow + consistency |
| 验证 | `self.model`（固定，非 EMA） | `self.ema_model` | flow + consistency |
| 推理 | EMA（如果 `use_ema`） | N/A | N/A |

- NaN 检测（三层防护）: ① loss NaN → `zero_grad` + raise；② grad NaN → `zero_grad` + raise；③ DDP `dist.all_reduce(nan_flag, MAX)` in `backward()` 前，防集群死锁
- MoE aux loss 全程生效；`get_optim_param_groups()` 中 obs encoder 使用 `get_optim_group_with_no_decay` 按模块类型分拆 decay/no-decay，bias/Norm 不受 weight decay

### DDP
- `mp.spawn(ddp_worker, nprocs=N)`，nccl backend，`DistributedSampler` 分片
- `dataloader.batch_size` 是每卡值；rank 0 独享 logging/checkpoint/eval
- 两阶段 seed: 模型初始化前统一 seed，构建后 `seed+rank` 差异化增强；normalizer 通过 `dist.broadcast` 同步
- `find_unused_parameters=False`（DDP 性能优化）；MoE 全专家计算（`torch.stack` 所有 expert）确保所有参数获得梯度
- 变 world_size resume 时 LR schedule 失真（total_steps 重算但 step counter 不变），模型/EMA/optimizer 状态不受影响

### MultiTask
- `MultiTaskDataset` 注入 `obs['task_text']` 和 `obs['task_name']`；epoch 通过 `multiprocessing.Value` 共享内存同步到 persistent workers
- CLIP text encoder 显式 `requires_grad_(False)` 冻结，仅 `text_proj` 可训练；text cache 预计算全任务 embedding
- `dataset.task_texts` 与 `agent.task_texts` 通过 `${dataset.task_texts}` 引用保持一致
- 固定索引生成支持 `proportional`/`balanced`/`weighted` 三种策略，MD5 hash 保证确定性

### 评测
- 评测时 agent fresh 构造，`load_state_dict` 从 checkpoint 恢复 normalizer（不重新拟合）
- `eval_sim.py` 通过 `hasattr(cfg, 'eval')` 前置检查兼容历史 config（`cfg.eval` 不存在时安全降级）
- `per_task/{name}/success_rate` 原始为小数 (0-1)，`evaluate()` 存储时 ×100 转百分比

### 已知硬编码（不可从配置修改，审查时注意）
- DDIM `beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2'` — `diffusion.py:19-28`
- StateMLP `hidden_channels=[64]` — `state_mlp.py` 默认值，所有 agent 统一
- FlowMatch consistency weight = 1.0 — `flowmatch.py:198`
- `torch.optim.AdamW` — `base.py:137`
- UNet `use_{down,mid,up}_condition=True` — `base.py:178-181`
- DINO/CLIP/SigLIP vision backbone 以 bfloat16 加载，CLIP/SigLIP 启用 Flash Attention 2 — `dino.py:52`, `clip.py:33-35`, `siglip.py:30-32`

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/01-项目概览.md` | 环境搭建、最小命令、架构速览 |
| `docs/02-配置系统.md` | Hydra 配置层级、插值、校验、策略对比 |
| `docs/03-数据集与增强.md` | Zarr 加载、ReplayBuffer、SequenceSampler、增强、Normalizer |
| `docs/04-模型架构.md` | 5 种 Agent、Encoder/Backbone/Decoder 详解 |
| `docs/05-训练机制.md` | Trainer 循环、EMA、checkpoint、LR schedule |
| `docs/06-仿真评测.md` | SimEvaluator、env_runner、dexmani_sim 环境 |
| `docs/07-多卡训练.md` | mp.spawn、DistributedSampler、rank 隔离、checkpoint 兼容 |
| `docs/08-审查约定.md` | 已知设计模式与架构决策（9 章 34 条），防止审查误报 |
