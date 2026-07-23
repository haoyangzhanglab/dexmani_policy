# CLAUDE.md — dexmani_policy

灵巧手操作模仿学习框架。Hydra 配置驱动，Zarr replay buffer，Diffusion/FlowMatch 动作解码，`dexmani_sim` 仿真评测。

> 本文件为 Claude 工作速查索引。详细设计见 `docs/`。

## 环境与常用命令

- Conda env: `policy`；仿真评测依赖 `dexmani_sim`

```bash
# 单卡训练
bash scripts/train.sh dp3                # dp / dp3 / maniflow / moe_dp / r3d / dqrise
bash scripts/train.sh dp3 'training.loop.num_epochs=10'  # Hydra override

# 多卡 DDP
bash scripts/train_ddp.sh maniflow_ddp    # dp_ddp / maniflow_ddp / multitask_dit_ddp / r3d_ddp / ddp/dqrise

# 多任务训练
bash scripts/train.sh multitask_dit

# DQ-RISE: VQ-VAE 预训练（Stage 1）
bash scripts/train_vq_hand.sh pour '--num_epochs 1500'

# 冒烟测试（构建链完整性）
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py dp3 maniflow moe_dp r3d dqrise

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
| `MoEAgent` | RGB+state (also supports PC) | R3M-resnet18+StateMLP+MoE gating (shipped); DP3 encoder+MoE gating (PC path) | `ConditionalUnet1D` | `Diffusion` | `moe_dp.yaml` |
| `MultiTaskAgent` | RGB+state+text | DINO/CLIP/SigLIP+CLIP Text+StateMLP | `DiT_Diffusion` (self-attn+AdaLN) | `Diffusion` / `FlowMatch` | `multitask_dit.yaml` |
| `R3DAgent` | PC+state | Uni3D (ViT-tiny)+StateMLP | `OneWayTransformer` (cross-attn) | `Diffusion` | `r3d.yaml` |
| `DQRISEAgent` | PC+state | iDP3/PointNeXT+StateMLP | `ConditionalUnet1D` (FiLM) | `Diffusion` (reduced dim) | `dqrise.yaml` |

> DQRISEAgent 动作空间：扩散头输出 `tcp_dim+1` 维（臂控制 + 1 个连续 VQ 索引），VQ 索引经 PCA 排序的 CodebookManager 查表还原为手部关节角。详见 `docs/DQ-RISE-知识体系.md`。

> **视觉 backbone 加载约定**: DINO/CLIP/SigLIP 均以 `torch_dtype=torch.bfloat16` 加载（参数显存减半）；统一启用 `attn_implementation="sdpa"`（PyTorch 内置 Flash Attention dispatch，无需 `flash-attn` pip 包，且性能相当或更优 — CLIP +7%、SigLIP 持平；DINOv2 不支持 `flash_attention_2` 故同样走 SDPA）。LoRA 参数自动对齐 backbone dtype。

> **Agent 类层级**: `BaseAgent` → `UNetDiffusionAgent` (DP/DP3/MoE/DQRISE — 共享 UNet+Diffusion 构建) | `DiTXFlowMatchAgent` (ManiFlow — DiTX+FlowMatchWithConsistency)；`MultiTaskAgent` 和 `R3DAgent` 直接继承 `BaseAgent`。

> **Action Decoder 类型体系**: `Diffusion` (3 种 prediction_type: `epsilon` noise-prediction / `sample` x0-prediction / `v_prediction` velocity; 支持 `dim_groups` 按维度组独立 MSE); `FlowMatch` (纯速度预测, MultiTaskAgent 使用); `FlowMatchWithConsistency` (速度+consistency 双 loss, ManiFlow 使用); `TimeSampler` (7 种采样模式: uniform/lognorm/mode/cosmap/beta/discrete/discrete_pow, 两个 FlowMatch 变体共用)。

## 关键数据流

### 数据加载

```
Zarr (robot_data/<task>.zarr) → ReplayBuffer.copy_from_path()  # 全量 numpy, float32
  → SequenceSampler (numba, pad_before=1, pad_after=7)           # 滑动窗口
    → 短 episode (<8帧) 自动 warn + skip（不 crash）
  → BaseDataset.__getitem__() (增强 + numpy→torch) → DataLoader → batch (B,16,*)
```

- `use_aux_ee`: 将辅助 end-effector action (`action_ee[..., :9]`) 拼接到主 action 后，action_dim 从 19 增至 28
- `obs_horizon` 截断: `BaseDataset.sample_to_data()` 中应用，与 `n_obs_steps` 交互
- `RGBPCDataset`: 5 模态 (rgb, depth, point_cloud, camera_intrinsic, camera_extrinsic)，相机矩阵使用 identity normalizer
- `downsample_mask` / `max_train_episodes`: 限制训练 episode 数；验证集通过 `copy.copy` 浅拷贝构造

### 训练前向 (Agent.compute_loss)

```
obs (B,16,*) → normalizer.normalize() → modality_dropout → truncate[:,:2] → flatten → encoder → cond
action (B,16,A) → normalizer['action'].normalize()  # A=19(joint) or 21(ee), → [-1,1]

cond + action → action_decoder.compute_loss()
  [Diffusion]:        prediction_type: epsilon (noise) / sample (x0) / v_prediction (velocity); 支持 dim_groups 按维度组独立 MSE loss (R3D 用于关节/EE 分量分离)
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
TopKCheckpointTracker: 三级分数解析 (内存缓存 → 文件名正则 → 文件读取)，自动淘汰，stale index 检测
CheckpointStore.resolve_path("best"): 按文件名 score 排序或自定义 best_fn 选择最优 checkpoint
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
  agents/action_decoders/ # Diffusion, FlowMatch, FlowMatchWithConsistency, sample (TimeSampler), backbone/(unet1d, dit, ditx)
  agents/obs_encoder/    # pointcloud (iDP3/PointNeXT/Uni3D/R3DObsEncoder), rgb (DINO/CLIP/SigLIP/R3M/ResNet), text, plugins/(moe, token_compressor)
  agents/position_encodings.py  # SinusoidalPosEmb, TimestepMLP, SinusoidalPosEmb3D, RelativePositionalEncoding3D
  agents/optim_util.py   # OptimGroupMixin: per-module get_optim_groups() (UNet1D, DiTX 等 backbone 实现)
  datasets/              # BaseDataset → PC/RGB/RGBPC/MultiTask; common/(ReplayBuffer, Sampler)
  scripts/               # Python 工具脚本（extract_codebook, train_vq_hand, measure_vq_usage）
  training/              # Trainer, DDPTrainer, SimEvaluator, workspace, checkpoint_io, ema, logging, lr_scheduler
  env_runner/            # BaseRunner → SimRunner → TaskTextSimRunner; MultiTaskSimRunner (独立 orchestrator)
  common/                # LinearNormalizer, pytorch_util, config, checkpoint_io
```

---

## 关键约定

### 配置与数据
- Hydra + OmegaConf，`${eval:'...'}` 和 `${eq:...}` 插值在 `common/config.py` 的 `register_resolvers()` 注册
- CLI override 任意字段；配置校验基于字段存在性判断，不依赖 `_target_` 字符串匹配
- `eval.seed: 0` 固定，保证同一 checkpoint 多次评测可复现
- `normalize_action_key()`: 将废弃的 `action_mode` (eef_hand/joint) 转为 `action_key` (action_ee/action)，发 `FutureWarning`。历史 checkpoint 加载必需
- `validate_action_key_consistency()`: 校验 `action_key` vs `env_runner.env_kwargs.control_mode` 一致性，防止 CLI override 导致的静默配置错误
- Normalizer: `mode='limits'`，**全量 replay buffer 拟合**（含验证集，非 bug，是跨项目统一惯例）→ [-1,1]；`range_eps=1e-4`（与官方 diffusion_policy 一致），低方差维度 zero-center 不缩放（scale=1.0, offset=-mean，与官方 R3D-Policy 的 offset=-input_min 有微小偏差但数值差异 < 5e-5）。全量拟合的理由：① `limits` 模式下验证集几乎不改变 min/max；② 保证推理时对新采集数据的覆盖；③ 与 ManiFlow_Policy、SAT、RoboTwin、DexJoco 等兄弟项目一致
- `build_mixed_action_normalizer()` (action_ee 模式): rot6d (dim 3:ee_dim) 故意不做归一化 (scale=1, offset=0)，旋转表示不应归一化
- `LinearNormalizer.__getitem__`: 惰性 view 构造；`SingleFieldLinearNormalizer` 工厂方法 (`create_fit`/`create_manual`/`create_identity`)；`DictOfTensorMixin` 自定义 `_load_from_state_dict` (绕过 PyTorch strict matching) + unfitted-device RuntimeError 守卫
- 数据增强默认禁用，通过 `prob` 控制执行概率；`pc_dim` 必须与 Zarr 点云通道数一致
- 增强器分类：**点云** — `PointColorJitter` (HSV 空间，光照变化) / `PointColorNoiseAug` (per-channel 独立噪声，传感器噪声) / `PointDropout` (稀疏性) / `PointCoordNoiseAug` (几何扰动)；**状态** — `StateNoiseAug`；**RGB** — `ImageAug` (torchvision 组合 + fused RNG)。设计约定：`__slots__` 内存优化，`_augment` 原地修改，`apply_augmentation` 惰性拷贝，`AUGMENTOR_REGISTRY` 声明式映射
- 数据增强（`augmentation_cfg`）与 modality dropout（`modality_dropout_probs`）职责不同：增强生成合理的观测变体（加噪、旋转、颜色抖动），在 Dataset 层 normalize 前执行；modality dropout 是模型正则化，故意将整个模态置零来防止过拟合，在 Agent 层 normalize 后执行
- `rgb_keep_uint8`: 快速路径 — RGB uint8 保持到 GPU 再 cast，DataLoader→GPU 传输量减少 4x
- SequenceSampler：短于 8 帧的 episode 自动 warn + skip（不 crash）；边界 padding 复制首尾帧
- 配置全链路可追踪，所有 YAML key 均可追溯到代码消费点，**无 dead config**

### 训练
- **梯度累积**: `gradient_accumulation_steps` (默认 1)，loss 除以 steps 后 `backward()`；DDP 中微 batch 间 `model.no_sync()`；scheduler 总步数通过 ceiling division 计算
- **bfloat16 AMP**: `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` 包裹 `train_one_step()`；`torch.set_float32_matmul_precision('high')`
- **torch.compile**: `use_compile` 开关，`compile_models()` 在 device move 后调用，DDP 兼容 (compiled model 需先 unwrap 取 `_orig_mod`)
- **obs_lr**: `cfg.optimizer.obs_lr` 设置 obs encoder 独立学习率，`obs_lr=0` 即冻结 encoder
- **日志**: 双系统 `JsonlLogger` (过滤 video key) + `WandbLogger` (numpy video → `wandb.Video` 自动转换)，`atexit` 注册 cleanup，`WANDB_SILENT=true` 在 import 时设置
- **NaN 调试 checkpoint**: NaN loss 时 `_save_nan_debug()` 写入完整快照 (model+EMA+opt+sched) 到 `checkpoints/nan_debug_epoch=...step=..._timestamp.pt` (原子写入)，保留最近 5 个
- **OptimGroupMixin**: `UNet1D`/`DiTX` 等 backbone 实现 `get_optim_groups()`，Agent 层按模块聚合参数组 (decay/no-decay 分离)
- **LR Scheduler**: 支持 PyTorch `OneCycleLR` + diffusers 类型；`cosine_annealing` 已硬废弃 → `NotImplementedError` 引导使用 `cosine`
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
- **MoE plugin 高级特性**: `use_boost` (递进式 expert 激活，boost_start_epoch/interval/experts_per_step 控制)；`use_enhanced_gate` (多层 gating + dropout)；`ExpertMLP` 类；`override_idx` 推理时路由控制
- **Epoch hook**: `on_epoch_start()` 在 trainer 每 epoch 前调用 (EMA 模型切换等)；`max_train_steps`/`max_val_steps` 限制单 epoch 步数
- **Workspace 初始化**: Hydra config 在 workspace init 时自动保存；`latest.pt` 通过 `.tmp.pt` → `os.replace()` 原子 symlink

### DDP
- `mp.spawn(ddp_worker, nprocs=N)`，nccl backend，`DistributedSampler` 分片
- `dataloader.batch_size` 是每卡值；rank 0 独享 logging/checkpoint/eval
- 两阶段 seed: 模型初始化前统一 seed，构建后 `seed+rank` 差异化增强；normalizer 通过 `dist.broadcast` 同步
- `find_unused_parameters=False`（DDP 性能优化）；MoE 全专家计算（`torch.stack` 所有 expert）确保所有参数获得梯度
- 变 world_size resume 时 LR schedule 失真（total_steps 重算但 step counter 不变），模型/EMA/optimizer 状态不受影响
- 梯度累积 `model.no_sync()` 微 batch 间保持梯度未同步，仅在累积边界 `optimizer.step()` 前 reduce
- `finish_epoch` 失败时 broadcast 错误到所有 rank 避免死锁；resume 时可选 `resume_state` 跳过以重置 state
- `num_training_steps` 与 dataloader epoch 步数不匹配时 warning

### DQ-RISE
- **三阶段管道**: VQ-VAE 预训练（`train_vq_hand.py`）→ 码本提取+PCA 排序（`extract_codebook.py`）→ 联合扩散训练（`train.py dqrise`）
- VQ-VAE 冻结：DQRISEAgent 训练时 `CodebookManager` 的 `sorted_hand_poses` 为普通 tensor，不参与梯度。`find_unused_parameters=False` 兼容
- `vq_idx_used`（码本利用率）是决定性下游成功率预测器：<8 判死（~0%），≥12 健康（~60%）。详见 `docs/DQ-RISE-知识体系.md`
- CodebookManager 封装 PCA 重排序算法（使用学到的 `layer_weights` 而非硬编码 0.5），**nn.Module** 设计使 checkpoint 自包含，支持 `.npz` 格式持久化

### MultiTask
- `MultiTaskDataset` 注入 `obs['task_text']` 和 `obs['task_name']`；epoch 通过 `mp.Manager().Value` 共享内存同步到 persistent workers（兼容 fork 和 spawn 启动方式）
- CLIP text encoder 显式 `requires_grad_(False)` 冻结，仅 `text_proj` 可训练；text cache 预计算全任务 embedding
- `dataset.task_texts` 与 `agent.task_texts` 通过 `${dataset.task_texts}` 引用保持一致
- 固定索引生成支持 `proportional`/`balanced`/`weighted` 三种策略，MD5 hash 保证确定性
- `normalizer_mode='per_task'` 在标准训练中 `NotImplementedError`，仅多任务评测可用；`build_normalizer` 辅助函数统一构造流程

### 评测
- 评测时 agent fresh 构造，`load_state_dict` 从 checkpoint 恢复 normalizer（不重新拟合）
- `eval_sim.py` 通过 `hasattr(cfg, 'eval')` 前置检查兼容历史 config（`cfg.eval` 不存在时安全降级）
- `per_task/{name}/success_rate` 原始为小数 (0-1)，`evaluate()` 存储时 ×100 转百分比
- **Checkpoint 参数校验**: `SimEvaluator._load_for_inference()` 校验 `n_obs_steps`/`n_action_steps`/`action_dim`/`horizon`/`action_key` 与 agent 实例匹配，不匹配抛 `ValueError`
- **多 denoise step 扫描**: `SimEvaluator.run()` 接受 `denoise_timesteps_list`，每个值独立评测，生成 per-timestep 子目录 + 汇总结果
- **Env 协议**: `run_one_episode` 返回 `(success_condition, success, action_cnt)`，hold delay 后采集初始观测；`BaseRunner` 使用预分配环形缓冲区 (`_obs_buffer`/`_obs_cursor`/`_obs_count`) 管理观测历史
- **SimRunner**: `_expand_env_kwargs` high-level 随机化开关映射，`name_to_pascal_case` 命名约定
- **TaskTextSimRunner**: 继承 `SimRunner`，`get_action_chunk` 中注入 `task_text` 到 `obs_batch`；`MultiTaskSimRunner` 是独立 orchestrator (非 BaseRunner 子类)，持有 `dict[str, TaskTextSimRunner]`

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
- **DDP 覆盖 5 种策略**: `dp`/`maniflow`/`multitask_dit`/`r3d`/`dqrise`。`dp3` 和 `moe_dp` 当前仅单卡训练，无需 DDP 配置。
- **`tcp_dim` 命名**: 在 `action`（joint 空间）模式下取值为 7，实际含义是臂关节角数（非 TCP 位姿）；仅在 `action_ee` 模式下才表示末端执行器控制维数（9）。该命名是历史遗留，审查时勿据此推断语义。
- **MoE dual-backbone 架构**: `MoEAgent` 通过 `hasattr(config, 'pc_encoder')` 自动切换 RGB/PC 路径，非配置显式字段控制。shipped config (`moe_dp.yaml`) 使用 RGB (R3M-resnet18)，但 PC 路径同样可用。
- **Obs encoder forward 签名不一致**: `MoEAgent` 的 forward 返回 `dict` (含 aux_loss)，其他 agent (DP/DP3/ManiFlow/R3D) 返回 `Tensor`。`BaseAgent.compute_loss()` 统一处理 `isinstance(cond, dict)` 提取 aux，屏蔽差异。
- **EMAModel BatchNorm 特殊处理**: BatchNorm affine 参数直接复制 (不 EMA 平均)；frozen 参数也直接复制。与标准 EMA 平滑逻辑不同。
- **`R3DObsEncoder` 拼接模式**: patch_tokens + state_emb + pc_pe 沿 feature 维拼接 (非 `torch.cat`)，与标准 encoder 不同。

## 文档索引

| 文档 | 内容 |
|---|---|
| `docs/DQ-RISE-知识体系.md` | **主文档**：DQ-RISE 知识体系 — 论文精读 + 官方代码走读 + 本项目实现差异 + 交叉验证 |
