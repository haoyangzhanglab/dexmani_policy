# CLAUDE.md — dexmani_policy

灵巧手操作模仿学习框架。Hydra 配置驱动，Zarr replay buffer，Diffusion/FlowMatch 动作解码，`dexmani_sim` 仿真评测。

> 本文件为 Claude 工作速查索引。详细设计见 `docs/`。

## 环境与常用命令

- Conda env: `policy`；仿真评测依赖 `dexmani_sim`

```bash
# 单卡训练
bash scripts/train.sh dp3                # dp / dp3 / maniflow / moe_dp3 / r3d / dqrise
bash scripts/train.sh dp3 'training.loop.num_epochs=10'  # Hydra override

# 多卡 DDP
bash scripts/train_ddp.sh maniflow_ddp    # dp_ddp / maniflow_ddp / multitask_dit_ddp / r3d_ddp / ddp/dqrise

# 多任务训练
bash scripts/train.sh multitask_dit

# DQ-RISE: VQ-VAE 预训练（Stage 1）
bash scripts/train_vq_hand.sh pour '--num_epochs 1500'

# 冒烟测试（构建链完整性）
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py dp3 maniflow moe_dp3 r3d dqrise

# 仿真评测
bash scripts/eval_sim.sh dp3 pick_apple_messy <exp_dir>
```

## 架构

```
Hydra config (configs/*.yaml) → Dataset (Zarr → ReplayBuffer → SequenceSampler)
  → Agent (obs_encoder + action_decoder) → Trainer (loss → backward → EMA → checkpoint)
  → SimEvaluator (env_runner.run → success_rate)
```

核心不变约束：`horizon=16  n_obs_steps=2  n_action_steps=8`

动作空间有两种模式（由 `action_key` 控制）：
- `action`（joint 空间）：**19 维** = 7 个臂关节角 + 12 个手部关节
- `action_ee`（end-effector 空间）：**21 维** = 9 维末端位姿 (pos3+rot6d6) + 12 个手部关节

### 入口点

| 入口 | 模式 | 关键差异 |
|---|---|---|
| `train.py` | 单卡 | `@hydra.main`，`build_train_components()` 完整装配 |
| `train_ddp.py` | 多卡 DDP | `mp.spawn(ddp_worker, nprocs=N)`，复用单卡构建函数 |
| `eval_sim.py` | 独立评测 | Hydra-free CLI，`CheckpointStore` 直接加载 checkpoint；`hasattr(cfg, 'eval')` 前置保护兼容历史 config |
| `dexmani_policy/scripts/train_vq_hand.py` | VQ-VAE 预训练 | DQ-RISE Stage 1，从 `dqrise.yaml` 的 `vq_vae` 段读取配置 |

### Agent 变体

| Agent | 输入 | Encoder | Backbone | Decoder | 配置 |
|---|---|---|---|---|---|
| `DPAgent` | RGB+state | DINO/CLIP/SigLIP+StateMLP | `ConditionalUnet1D` (FiLM) | `Diffusion` | `dp.yaml` |
| `DP3Agent` | PC+state | iDP3/PointNeXT+StateMLP | `ConditionalUnet1D` (FiLM) | `Diffusion` | `dp3.yaml` |
| `ManiFlowAgent` | PC+state | PointNeXT+StateMLP | `DiTXFlowMatch` (cross-attn) | `FlowMatchWithConsistency` | `maniflow.yaml` |
| `MoEAgent` | PC+state | DP3 encoder+MoE gating | `ConditionalUnet1D` | `Diffusion` | `moe_dp3.yaml` |
| `MultiTaskAgent` | RGB+state+text | DINO/CLIP/SigLIP+CLIP Text+StateMLP | `DiT_Diffusion` (self-attn+AdaLN) | `Diffusion` | `multitask_dit.yaml` |
| `R3DAgent` | PC+state | Uni3D (ViT-tiny)+StateMLP | `OneWayTransformer` (cross-attn) | `Diffusion` | `r3d.yaml` |
| `DQRISEAgent` | PC+state | iDP3/PointNeXT+StateMLP | `ConditionalUnet1D` (FiLM) | `Diffusion` (reduced dim) | `dqrise.yaml` |

> DQRISEAgent 动作空间：扩散头输出 `tcp_dim+1` 维（臂控制 + 1 个连续 VQ 索引），VQ 索引经 PCA 排序的 CodebookManager 查表还原为手部关节角。详见 `docs/DQ-RISE-完整分析.md`。

> **视觉 backbone 加载约定**: DINO/CLIP/SigLIP 均以 `torch_dtype=torch.bfloat16` 加载（参数显存减半）；统一启用 `attn_implementation="sdpa"`（PyTorch 内置 Flash Attention dispatch，无需 `flash-attn` pip 包，且性能相当或更优 — CLIP +7%、SigLIP 持平；DINOv2 不支持 `flash_attention_2` 故同样走 SDPA）。LoRA 参数自动对齐 backbone dtype。

## 关键数据流

### 数据加载

```
Zarr (robot_data/<task>.zarr) → ReplayBuffer.copy_from_path()  # 全量 numpy, float32
  → SequenceSampler (numba, pad_before=1, pad_after=7)           # 滑动窗口
    → 短 episode (<8帧) 自动 warn + skip（不 crash）
  → BaseDataset.__getitem__() (增强 + numpy→torch) → DataLoader → batch (B,16,*)
```

### 训练前向 (Agent.compute_loss)

```
obs (B,16,*) → normalizer.normalize() → modality_dropout → truncate[:,:2] → flatten → encoder → cond
action (B,16,A) → normalizer['action'].normalize()  # A=19(joint) or 21(ee), → [-1,1]

cond + action → action_decoder.compute_loss()
  [Diffusion]:        noise→denoise→MSE(pred, target)
  [DQ-RISE]:          同上，但 action 为 [arm_ctrl + continuous_vq_idx]（tcp_dim+1 维）
  [FlowMatch]:        拆分 flow/consistency → 速度 MSE(v_pred, x1-x0) + consistency teacher
  [MoE]:              + aux_loss (load_balancing + entropy)

loss.backward() → optimizer.step → scheduler.step → EMA
```

### 推理 (Agent.predict_action)

```
obs_dict (无 T 维度) → normalize+truncate+flatten+encoder → cond
  → action_decoder.predict_action(cond, template=zeros(B,16,A))
    [Diffusion/DQ-RISE]: DDIM 迭代去噪 (默认 10-20 步)
    [FlowMatch]:         Euler ODE 积分 (默认 10 步, target_t=dt)
  → unnormalize → control_action = pred[:, 1:n_action_steps+1, :]  # (B, n_action_steps, A)
```

### DQ-RISE 特有数据流

```
训练：
  action (B,16,A) → normalize → split: arm_ctrl (B,16,tcp_dim) | hand (B,16,hand_dim)
    → hand → CodebookManager.hand_pose_to_continuous_index() → vq_idx (B,16,1)
    → joint_action = cat([arm_ctrl, vq_idx]) → diffusion.compute_loss()

推理：
  DDIM sample → cat([arm_ctrl_pred, vq_idx_pred])
    → CodebookManager.continuous_index_to_hand_pose(vq_idx_pred) → hand_pred
    → cat([arm_ctrl_pred, hand_pred]) → unnormalize → full action (B, n_action_steps, A)
```

### 评测与检查点

```
SimEvaluator: _load_for_inference(ckpt_tag, use_ema=True)
  → env_runner.run(agent) → {success_rate, avg_steps, videos}

Checkpoint: finish_epoch() → TrainCheckpoint → .tmp → os.replace()
  → epoch=XXXX-step=YYYY-score=ZZ.ZZZZ.pt
  → save_latest() (symlink) + save_topk() (monitor test_mean_score, topk=3)
Resume: load_for_resume("latest") → 恢复 model/ema/opt/sched + normalizer
DDP: fix_state_dict() 自动处理 module. 前缀 → checkpoint 始终以 unwrapped 格式保存
```

### 数据流形状

```
Zarr:       joint_state (N,19)           action (N,19|21)            point_cloud (N,1024,3|6)
Sample:     obs (16,*)                   action (16,A)               A=19(joint)|21(ee)
Batch:      obs (B,16,*)                 action (B,16,A)
Preprocessed: obs (B×2,*)                # truncate → flatten batch+time
Cond:       (B, out_dim×2)               # UNet/DP/DP3/MoE/DQRISE; ManiFlow: (B, N_tokens, token_dim)
```

### 文件组织

```
dexmani_policy/
  train.py, train_ddp.py, eval_sim.py, smoke_test.py
  configs/               # Hydra YAML（7 策略 + dataset preset）
  configs/ddp/           # DDP 多卡 overlay（dp, dqrise, maniflow, multitask_dit, r3d）
  agents/core/           # BaseAgent → DP/DP3/ManiFlow/MoE/MultiTask/R3D/DQRISE
  agents/vq_hand/        # VQ-VAE 手部量化模块（DQ-RISE 专用：vqvae, codebook_manager, residual_vq, vector_quantize）
  agents/action_decoders/ # Diffusion, FlowMatchWithConsistency, backbone/(unet1d, dit, ditx)
  agents/obs_encoder/    # pointcloud, rgb, text, plugins/(moe, token_compressor)
  datasets/              # BaseDataset → PC/RGB/RGBPC/MultiTask; common/(ReplayBuffer, Sampler)
  scripts/               # Python 工具脚本（extract_codebook, train_vq_hand, analyze_*, measure_vq_usage）
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
- Normalizer: `mode='limits'`，**全量 replay buffer 拟合**（含验证集，非 bug，是跨项目统一惯例）→ [-1,1]；`range_eps=1e-4`（与官方 diffusion_policy 一致），低方差维度 zero-center 不缩放（scale=1.0, offset=-mean，与官方 R3D-Policy 的 offset=-input_min 有微小偏差但数值差异 < 5e-5）。全量拟合的理由：① `limits` 模式下验证集几乎不改变 min/max；② 保证推理时对新采集数据的覆盖；③ 与 ManiFlow_Policy、SAT、RoboTwin、DexJoco 等兄弟项目一致
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

### DQ-RISE
- **三阶段管道**: VQ-VAE 预训练（`train_vq_hand.py`）→ 码本提取+PCA 排序（`extract_codebook.py`）→ 联合扩散训练（`train.py dqrise`）
- VQ-VAE 冻结：DQRISEAgent 训练时 `CodebookManager` 的 `sorted_hand_poses` 为普通 tensor，不参与梯度。`find_unused_parameters=False` 兼容
- `vq_idx_used`（码本利用率）是决定性下游成功率预测器：<8 判死（~0%），≥12 健康（~60%）。详见 `docs/DQ-RISE-完整分析.md` 第 4 节
- CodebookManager 封装 PCA 重排序算法（硬编码 0.5 group weights，匹配官方 `eval_vqvae.py:96`），支持 `.npz` 格式持久化

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
- DINO/CLIP/SigLIP vision backbone 以 bfloat16 加载，统一启用 SDPA（PyTorch 内置 Flash Attention dispatch）— `agents/obs_encoder/rgb/dino.py:51`, `clip.py:33`, `siglip.py:30`。`"sdpa"` 相比 `"flash_attention_2"` 无需 `flash-attn` pip 包，且性能相当或更优（CLIP +7%、SigLIP 持平），DINOv2 不支持 `flash_attention_2`

### 已知设计模式（审查时易被误报为问题，勿修复）
- **Normalizer 全量拟合**: `get_normalizer()` 使用全部 replay buffer（含验证集），不按 `train_mask` 过滤。这是生态系统统一惯例（ManiFlow_Policy / R3D-Policy / SAT / RoboTwin / DexJoco 均如此），非数据泄露 bug。`limits` 模式下验证集不影响 min/max 边界，且全量统计量利于推理泛化。
- **checkpoint 频率 = eval 频率**: `checkpoint_interval_epochs = eval_interval_epochs`（当 env eval 启用时）。设计意图：只在评估成功的 epoch 保存 checkpoint，避免保存无评估的中间状态。
- **FlowMatch `target_t` 训练/推理偏移**: 无 EMA teacher 时训练用 `target_t=0`，推理用 `target_t=dt>0`。ManiFlow 通过 `use_ema_teacher_for_consistency=true` 已缓解（consistency 路径提供 `target_t>0` 训练信号）。
- **DDP 覆盖 5 种策略**: `dp`/`maniflow`/`multitask_dit`/`r3d`/`dqrise`。`dp3` 和 `moe_dp3` 当前仅单卡训练，无需 DDP 配置。
- **`tcp_dim` 命名**: 在 `action`（joint 空间）模式下取值为 7，实际含义是臂关节角数（非 TCP 位姿）；仅在 `action_ee` 模式下才表示末端执行器控制维数（9）。该命名是历史遗留，审查时勿据此推断语义。

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/DQ-RISE-完整分析.md` | DQ-RISE 完整分析：论文精读 + 官方代码走读 + 本项目实现差异 + 实验踩坑与优化方向 |
| `docs/12-FlowMatch变种集成方案.md` | FlowMatch 变种集成方案 |
