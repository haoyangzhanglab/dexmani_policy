# CLAUDE.md — dexmani_policy

灵巧手操作模仿学习框架。Hydra 配置驱动，Zarr replay buffer，Diffusion/FlowMatch 动作解码，`dexmani_sim` 仿真评测。

> **速查**: [训练](#训练命令) · [Agent对比](#agent-变体) · [SAT](#sat-structural-action-transformer--论文对齐状态) · [配置速查](#配置速查) · [硬编码与约定](#已知硬编码与设计约定) · [文件地图](#文件组织)

## 环境

```bash
conda activate policy
export DATA_DIR=/path/to/data          # 需设，否则 dataset 路径报错
```

## 训练命令

```bash
# === 单卡 ===
bash scripts/train.sh dp3                  # dp / dp3 / maniflow / moe_dp / r3d / dqrise / multitask_dit / dp3_faas / sat
bash scripts/train.sh dp3 'training.loop.total_train_steps=500'

# === 多卡 DDP ===
bash scripts/train_ddp.sh ddp/maniflow     # ddp/dp / ddp/maniflow / ddp/multitask_dit / ddp/r3d / ddp/dqrise / ddp/dp3_faas
```

> 实际存在的单卡配置 (9): `dp, dp3, dp3_faas, dqrise, maniflow, moe_dp, multitask_dit, r3d, sat`
> 实际存在的 DDP 配置 (6): `ddp/dp, ddp/dp3_faas, ddp/dqrise, ddp/maniflow, ddp/multitask_dit, ddp/r3d`
> 所有策略默认 `total_train_steps: 80000`

## 评测命令

```bash
bash scripts/eval_sim.sh dp3 pick_apple_messy <exp_dir>
# <exp_dir> = experiments/<policy>/<task>/<timestamp>
```

## 冒烟测试

```bash
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py dp3 maniflow moe_dp r3d dqrise sat
```

## VQ-VAE 预训练（DQ-RISE Stage 1）

```bash
bash scripts/train_vq_hand.sh pour '--num_epochs 1500'
```

---

## 架构概览

```
Hydra config (configs/*.yaml)
  → Dataset (Zarr → ReplayBuffer → SequenceSampler → __getitem__)
    → Agent (obs_encoder → backbone → action_decoder)
      → Trainer (loss → backward → grad_accum → EMA → checkpoint)
        → SimEvaluator (env_runner.run → success_rate)
```

### 核心不变量（碰了就炸）

| 常量 | 值 | 位置 |
|------|-----|------|
| `horizon` | **16** | 所有 config |
| `n_obs_steps` | **2** | 所有 config |
| `n_action_steps` | **8** | 所有 config |
| DDIM scheduler | `beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2'` | `diffusion.py:39-41` |
| StateMLP hidden | `[64]` | `state_mlp.py` |
| 优化器 | `AdamW(fused=True)` | `base.py:238` |
| ViT backbone dtype | `bfloat16 + attn_implementation="sdpa"` | dino/clip/siglip |

### 动作空间

| `action_key` | 含义 | action dim | arm 部分 | 手部 |
|-------------|------|-----------|----------|------|
| `action` | 关节空间 | **19** | 7 (臂关节角) | 12 (XHand) |
| `action_ee` | 末端执行器 | **21** | 9 (pos3+rot6d) | 12 (XHand) |
| `action` + FAAS | 关节+FAAS | **39** | 7 | 32 (FAAS) |
| `action_ee` + FAAS | EE+FAAS | **41** | 9 | 32 (FAAS) |

`joint_state` 始终 19D（或 FAAS 下 39D），与 `action_key` 无关。

### 入口点

| 入口 | 模式 | 关键点 |
|------|------|--------|
| `train.py` | 单卡 | `@hydra.main`，`build_train_components()` |
| `train_ddp.py` | 多卡 DDP | `mp.spawn(ddp_worker, nprocs=N)`，compile 在 DDP 包装前 |
| `eval_sim.py` | 独立评测 | Hydra-free CLI，从实验目录加载 `config.yaml`；`hasattr(cfg, 'eval')` 兼容历史 config |
| `smoke_test.py` | 构建验证 | Hydra `compose` API，6 阶段 + FAAS roundtrip + MoE 子检查 |
| `scripts/train_vq_hand.py` | VQ-VAE 预训练 | DQ-RISE Stage 1 |

---

## Agent 变体

### 类层级

```
BaseAgent
  ├── UNetDiffusionAgent        ← DP, DP3, MoE (共享 UNet+Diffusion 构建)
  ├── DiTXFlowMatchAgent        ← ManiFlow (DiTX+FlowMatchWithConsistency)
  ├── SATAgent                  ← 直接继承 BaseAgent (SATBackbone+SATFlowMatch)
  ├── MultiTaskAgent            ← 直接继承 BaseAgent
  ├── R3DAgent                  ← 直接继承 BaseAgent (OneWayTransformer)
  └── DQRISEAgent               ← 直接继承 BaseAgent (自建 UNet+Diffusion，action_dim 缩减)
```

### 完整对比

| Agent | 输入 | Encoder | Backbone | Decoder | 配置 | 独特点 |
|-------|------|---------|----------|---------|------|--------|
| **DP** | RGB+state | DINO/CLIP/SigLIP+StateMLP | UNet1D(FiLM) | Diffusion(DDIM 10步) | `dp.yaml` | CNN用channels_last |
| **DP3** | PC+state | iDP3/PointNeXT+StateMLP | UNet1D(FiLM) | Diffusion(DDIM 10步) | `dp3.yaml` | FPS下采样, pc_dim=3 |
| **ManiFlow** | PC+state | PointNeXT(patch)+StateMLP | DiTX(cross-attn) | FlowMatch+Consistency | `maniflow.yaml` | Token化条件, EMA教师, weight_decay=1e-3 |
| **MoE** | RGB+state | R3M+StateMLP+MoE门控 | UNet1D(FiLM) | Diffusion(DDPM 100步) | `moe_dp.yaml` | 16专家top-2, 无bfloat16/compile |
| **MultiTask** | RGB+state+text | DPObsEncoder+CLIP Text | DiT(self-attn+AdaLN) | Diffusion/FlowMatch | `multitask_dit.yaml` | 预缓存text, 均衡采样 |
| **R3D** | PC+state | Uni3D(ViT+Fourier PE)+StateMLP | OneWayTransformer(cross-attn) | Diffusion(DDIM 10步) | `r3d.yaml` | 级联mask, 分组loss |
| **DQRISE** | PC+state | iDP3+StateMLP | UNet1D(tcp+1维) | Diffusion(epsilon, 20步) | `dqrise.yaml` | VQ码本16种手势, 3x学习率 |
| **DP3(FAAS)** | PC+state | 同DP3 | 同DP3 | 同DP3 | `dp3_faas.yaml` | 32D FAAS空间, 零agent代码变更 |
| **SAT** | PC+state | PointNeXT(patch)+StateMLP | SATBackbone(MultiModalAttn+EJC) | FlowMatch(Gaussian init, Euler) | `sat.yaml` | 结构中心(B,Da,T), EJC 3-field sum, per-sample shuffle, 论文对齐 |

### 各 Agent 关键差异

- **DP/DINO/CLIP/SigLIP** → 以 `bfloat16` + SDPA 加载，支持 LoRA
- **MoE** → 唯一禁用 `bfloat16` 和 `compile` 的配置（MoE gate softmax 需 float32，CUDA Graphs 内存开销）
- **R3D** → `dim_groups` 分组 loss（关节+EE分量独立MSE），cascading self-attn mask（关节token不能attend EE token）
- **DQRISE** → UNet 输入只有 `tcp_dim+1` 维（10D），非标准 `action_dim`
- **ManiFlow** → 唯一用 `betas=[0.9,0.95]` + `weight_decay=1e-3` 的配置（Transformer/FlowMatching 标准）
- **MultiTask** → CLIP text encoder 冻结，仅 `text_proj` 可训练；`MultiTaskSimRunner` 编排 per-task 评测
- **SAT** → 结构中心动作表示 `(B, Da, T)` 代替 `(B, T, Da)`；EJC (Embodied Joint Codebook) 3-field sum 提供 per-joint identity；MultiModalAttention 单次拼接注意力 (obs-as-KV-prefix)；per-sample shuffle 训练时随机排列关节 token；Obs 时间融合在特征维（token 数不随 `n_obs_steps` 增长）；`compile mode='default'`（CUDA graph 不兼容 shuffle 动态索引）；8 层 DiT-B 规模，hidden_dim=768
- **DP/DP3/ManiFlow/MoE/MultiTask/R3D** → 全部通过 config + `inject_faas_into_agent()` 兼容 FAAS，零 agent 代码变更

### SAT (Structural Action Transformer) — 论文对齐状态

SAT (CVPR 2026) 是本项目最新的 agent 变体，已通过完整论文对齐审计（2026-08-04）。当前实现与论文在全部 10 个关键设计点上一致：

| 论文设计 | 实现位置 | 对齐 |
|----------|---------|:---:|
| 结构中心表示 `(B, Da, T)` | `sat.py:246` (train), `sat.py:287` (infer) | ✅ |
| EJC 3-field SUM `E_emb+E_func+E_axis` | `sat.py:90` (`EmbodiedJointCodebook.forward`) | ✅ |
| Action token = traj + EJC (ADD) | `sat.py:453` (`x = x + ejc`) | ✅ |
| Per-sample shuffle + unshuffle | `sat.py:440-476` | ✅ |
| Flow Matching (Gaussian init + Euler) | `sat_flowmatch.py:41,82-85` | ✅ |
| Obs 特征维时间融合 | `sat.py:108-110` | ✅ |
| MultiModalAttention (obs-as-KV-prefix) | `sat.py:146-181` | ✅ |
| AdaLN 调制 (block + final layer) | `sat.py:250-252,539-543` | ✅ |
| AdaLN-Zero obs pre-norm | `sat.py:497-520` (`_AdaLNZeroObs`) | ✅ |
| 8 层 DiT-B (hidden_dim=768) | `sat.yaml:94-95` | ✅ |

**关键文件**:
- `agents/core/sat.py` — `SATObsEncoder` + `SATAgent` (直接继承 BaseAgent)
- `agents/action_decoders/backbone/sat.py` — `EmbodiedJointCodebook`, `MultiModalAttention`, `SATBlock`, `SATBackbone`, `_FinalLayer`, `_AdaLNZeroObs`
- `agents/action_decoders/sat_flowmatch.py` — `SATFlowMatch` (继承 `FlowMatch`，转发 shuffle)
- `configs/sat.yaml` — 训练配置

**已知微差异**（不影响论文忠实度）:
- `x_embedder`: 使用 2 层 MLP (`Linear→Mish→Linear`) 代替单 Linear — 论文简化描述，2 层 ≥ 单层
- Obs token 仅经过 attention 更新（不经 MLP）— 合理 prefix 设计，论文未要求 identical processing
- `compile mode='default'`: per-sample shuffle 的动态索引不兼容 CUDA graph (`reduce-overhead`)

**pour 任务已知结果**: 71.0% (100ep) — +12.0pp vs 旧 SAT (59.0%), +5.0pp vs DiTX-Control (66.0%)

### Action Decoder 类型


| Decoder | 预测目标 | 推理 | 关键参数 |
|---------|---------|------|---------|
| `Diffusion` | `epsilon`(noise) / `sample`(x0) / `v_prediction`(velocity) | DDIM 迭代 | `prediction_type`, `num_inference_steps` |
| `FlowMatch` | 瞬时速度 `v=x1-x0` | Euler ODE | 仅 MultiTask 用 |
| `FlowMatchWithConsistency` | 速度 + consistency(EMA教师) | Euler ODE | `flow_batch_ratio`, 4种时采样模式 |
| `SATFlowMatch` | 瞬时速度 `v=x1-x0` (Gaussian init) | Euler ODE | 继承 `FlowMatch`，转发 `shuffle` 到 backbone |

---

## 数据管线

```
Zarr (robot_data/<task>.zarr)
  → ReplayBuffer.copy_from_path()           # 全量 numpy float32
  → SequenceSampler (numba)                 # 滑动窗口，pad_before=1 pad_after=7
  → BaseDataset.__getitem__():
    1. sampler.sample_sequence(idx)         # numpy dict
    2. sample_to_data()                     # 重组 obs，拼接 aux EE
    3. apply_augmentation()                 # 原位 numpy（首次触发时拷贝）
    4. _preprocess_rgb_cpu()                # RGB: resize/crop/增强 → torch tensor
    5. dict_apply(ensure_tensor)            # numpy → torch
    6. _apply_faas_mapping()                # FAAS 模式: native → FAAS
  → DataLoader → batch (B, 16, *)
```

### Dataset 类

- `BaseDataset` — 全部核心逻辑。`PCDataset` 加 point_cloud 模态；`RGBPCDataset` 加相机矩阵
- 验证集通过 `copy.copy` 浅拷贝（共享 replay buffer）+ `augmentation_cfg=None`
- 短 episode (<8 帧) warn + skip（不 crash）

### 数据增强（5+1 种）

| 增强器 | 模态 | 效果 |
|--------|------|------|
| `PointColorJitter` | PC(C>=6) | HSV 色彩抖动 |
| `PointColorNoiseAug` | PC(C>=6) | 逐通道高斯噪声 |
| `PointCoordNoiseAug` | PC | XYZ 几何扰动 |
| `PointDropout` | PC | 随机丢点（补采样不补零） |
| `StateNoiseAug` | joint_state | 关节状态噪声 |
| `ImageAug` | RGB(tensor) | torchvision 链（ColorJitter+Grayscale+Noise+Blur） |

**关键实现细节**：所有增强器 `_augment` 原地修改。`apply_augmentation()` 首次触发时做一次 `.copy()`，同模态后续增强器重用该拷贝——省最多 3 次冗余拷贝。

### Normalizer

- `mode='limits'` → [-1,1]；`range_eps=1e-4` 时低方差维度零中心不缩放（`scale=1.0, offset=-mean`）
- **全量 replay buffer 拟合**（含验证集）——生态惯例，非 bug。`limits` 模式下验证集不改变 min/max
- `action_ee` 模式：rot6d (dim 3:9) identity normalizer（旋转不应归一化）
- FAAS 模式：normalizer 拟合前先转换数据到 FAAS 空间；补零维度 identity mapping

### Modality Dropout vs 数据增强

| | 数据增强 | Modality Dropout |
|---|---|---|
| **何时** | Dataset `__getitem__`（numpy层，normalize前） | Agent `compute_loss`（normalize后，truncate前） |
| **目的** | 生成合理传感器变体 | 强制模型在模态不可用时鲁棒 |
| **效果** | 加噪/抖动/丢点 | 整个模态置零 |
| **配置** | `augmentation_cfg` | `modality_dropout_probs`（当前全为0.0） |

---

## 训练

### Train Loop

```python
global_step = resume_step
while global_step < config.total_train_steps:
    model.train()
    on_epoch_start(epoch)          # MoE boost, dataset epoch sync
    optimizer.zero_grad()
    for micro_step, batch in enumerate(train_loader):
        with model.no_sync() if DDP+非边界:
            train_one_step(batch, is_accumulation_boundary)
        if boundary:
            optimizer.step(); scheduler.step(); zero_grad(); EMA; log
            global_step += 1
            check_milestone(epoch, global_step)  # 20/40/60/80/100%
        if global_step >= config.total_train_steps: break
    model.eval()
    epoch += 1
```

### train_one_step

1. batch → GPU (`non_blocking=True`)
2. `model.compute_loss()` under `torch.amp.autocast(dtype=bfloat16)`（如启用）
3. NaN 检测 #1：`raw_loss` 有限性检查；DDP `dist.all_reduce(nan_flag, MAX)`
4. `(raw_loss / gradient_accumulation_steps).backward()`
5. 仅在累积边界调用 `apply_gradient_step()`

### apply_gradient_step

1. `clip_grad_norm_(max_grad_norm)`
2. NaN 检测 #2：扫描所有 `.grad` 是否有限；`zero_grad` 后 raise
3. `optimizer.step()` → `scheduler.step()` → `zero_grad(set_to_none=True)`
4. EMA 更新

### NaN 两层防护

| 层 | 位置 | 检测内容 | 响应 |
|----|------|---------|------|
| 1 | `train_one_step` | `raw_loss` NaN | DDP 广播 → 保存 NaN debug checkpoint → raise |
| 2 | `apply_gradient_step` | 梯度 NaN | `zero_grad` → raise（含参数名） |

> NaN debug checkpoint: 原子 `.tmp→os.replace()`，最多保留 5 个

### Shape 验证（`compute_loss` / `predict_action` 入口）

`BaseAgent._validate_batch()` + `_validate_obs_dict()` 在以下入口点执行（MoE/DQRISE 覆写中独立调用）：

| 检查项 | 错误条件 | 信息 |
|--------|---------|------|
| action ndim | ≠ 3D | `action must be 3D (B, horizon, action_dim)` |
| action horizon | dim 1 ≠ `self.horizon` | `horizon mismatch: got X, expected Y` |
| action dim | dim 2 ≠ `self.action_dim` | `dim mismatch ... Check action_key / use_faas` |
| obs 最低 ndim | < 2D | `expected >=2D (B, n_obs_steps, ...)` |
| obs 时间维度 | dim 1 < `n_obs_steps` | `time dim too small — got X, need >= Y` |
| obs 模态间 batch size | 不一致 | `batch-size mismatch across modalities: {shapes}` |
| obs vs action batch | 不同 | `obs batch=X, action batch=Y` |

### 梯度累积

- `raw_loss / gradient_accumulation_steps` 保证全批次梯度等价
- DDP: 非边界微批次用 `model.no_sync()`，仅边界 all-reduce
- Scheduler 总步数通过 ceiling division 计算

### Checkpoint 系统

- `TrainCheckpoint` dataclass：epoch, global_step, model/ema/opt/sched state, monitor, train_params
- **里程碑保存**: 在 20%/40%/60%/80%/100% 进度各保存一个 checkpoint
- 文件名: `epoch=XXXX-step=YYYY-milestone=XXpct.pt`（无 score）
- 保存: 原子 `.tmp→os.replace()`
- `train_params`：n_obs_steps, n_action_steps, action_dim, horizon, action_key, tcp_dim, use_faas, hand_dim, control_action_dim, num_training_steps
- Latest: 原子 symlink `.tmp.pt→os.replace()`，指向最新里程碑 checkpoint
- Resume: 从 `latest.pt` symlink 加载，`_init_milestone_state()` 通过 `global_step` 推导已通过的里程碑

### DDP 要点

- `mp.spawn(ddp_worker, nprocs=N)`，NCCL backend
- 两阶段 seed: 模型初始化前统一seed，构建后 `seed+rank` 差异化增强
- Checkpoint 加载顺序: **先于 compile，先于 DDP 包装**（避免重编译 + 前缀问题）
- `find_unused_parameters=False`, `static_graph=True`
- Normalizer 从 rank 0 broadcast
- `OmegaConf.resolve(cfg)` 在 `mp.spawn` 前（子进程无 Hydra 运行时）
- 仅覆盖 6 种策略: dp/dp3_faas/dqrise/maniflow/multitask_dit/r3d（dp3/moe_dp/sat 仅单卡）
- **DDP 超时**: `dist.init_process_group(timeout=30min)` — 覆盖所有集体操作（all_reduce/broadcast/barrier），防止死 rank 导致整个集群无限挂起

### EMA

- 逆 gamma 衰减: `decay = 1 - (1 + step/inv_gamma)^(-power)`
- BatchNorm affine 参数直接复制（不平均）；frozen 参数跳过
- `update_after_step` 延迟启动

---

## 评测

### SimEvaluator

- `_load_for_inference(ckpt_tag)`: 加载 checkpoint → 恢复 normalizer → 校验 `n_obs_steps`/`n_action_steps`/`action_dim`/`horizon`/`action_key`
- FAAS 一致性: 拒绝 FAAS checkpoint 与非 FAAS config 混用（反之亦然）
- `run()`: 遍历 `denoise_timesteps_list`，每步数独立子目录 + 汇总
- 多任务检测: `self.env_runner.is_multi_task`（属性检测）

### Env Runner 层级

```
BaseRunner (抽象: 环形观测缓冲, run_one_episode, run)
  └── SimRunner (具体: dexmani_sim, 域随机化开关, seed列表)
        └── TaskTextSimRunner (注入 task_text)

MultiTaskSimRunner (独立编排器, 持有 dict[str, TaskTextSimRunner])
```

- 环形观测缓冲: 预分配，模索引，冷启动补第一帧
- `run_one_episode`: reset → 采集初始obs → agent推理 → 执行n_action_steps → 检查success
- `clear_cache_freq=25`: 定期重建 env 防 GPU 内存累积
- `info["success_condition"]` vs `info["success"]`：前者是原始信号(无保持延迟)，用于 `avg_steps`；后者含保持判断

### ACT 时序融合

- `temporal_ensemble_coeff=0.01`（全部配置已启用，之前是注释掉的）
- 公式: `blended = (old*1.0 + new*exp(-coeff)) / (1.0+exp(-coeff))`
- 标注 "+2.9pp avg across 7 tasks"

---

## 配置速查

### 文件列表

```
configs/
  dp.yaml  dp3.yaml  maniflow.yaml  moe_dp.yaml  multitask_dit.yaml  r3d.yaml  dqrise.yaml  dp3_faas.yaml  sat.yaml
configs/ddp/
  dp.yaml  maniflow.yaml  multitask_dit.yaml  r3d.yaml  dqrise.yaml  dp3_faas.yaml
```

`dp3_faas.yaml` 通过 Hydra `defaults: [- /dp3, - _self_]` 继承 dp3 全部超参，仅覆盖维度字段。

### 关键参数速查

| 参数 | dp | dp3 | maniflow | moe_dp | multitask_dit | r3d | dqrise | dp3_faas | sat |
|------|----|-----|----------|--------|---------------|-----|--------|----------|-----|
| action_dim | 19/21 | 19/21 | 19/21 | 19/21 | 19/21 | 19/28 | 21 | **39/41** | 19/21 |
| state_dim | 19 | 19 | 19 | 19 | 19 | 19 | 19 | **39** | 19 |
| backbone dims | [256,512,1024] | 同 | 12L×768d | [256,512,1024] | 8L×512d | 4L×256d | [256,512] | [256,512,1024] | **8L×768d** |
| diff train/infer | 100/10 | 100/10 | -/10 | 100/**100** | 100/10 | 100/10 | 100/**20** | 100/10 | -/10 |
| prediction_type | sample | sample | velocity | sample | sample | sample | **epsilon** | sample | velocity (FM) |
| lr | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | **3e-4** | 1e-4 | 1e-4 |
| weight_decay | 1e-6 | 1e-6 | **1e-3** | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| betas | [.95,.999] | [.95,.999] | **[.9,.95]** | [.95,.999] | [.95,.999] | [.95,.999] | [.95,.999] | [.95,.999] | [.95,.999] |
| warmup | 500 | 500 | 500 | 500 | 500 | 500 | **2000** | 500 | 500 |
| total_train_steps | 80000 | 80000 | 80000 | 80000 | 80000 | 80000 | 80000 | 80000 | 80000 |
| bfloat16 | ✓ | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ |
| compile | ✓ | ✓ | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ (mode=default) |
| val_ratio | 0.10 | 0.05 | 0.05 | 0.10 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |

### DDP 批次大小（每卡 × 4）

| DDP | batch×4=总 | 单卡 |
|-----|-----------|------|
| ddp/dp | 48×4=**192** | 64 |
| ddp/dqrise | 32×4=128 | 128 |
| ddp/maniflow | 32×4=128 | 128 |
| ddp/multitask_dit | 16×4=64 | 64 |
| ddp/r3d | 16×4=64 | 48 |
| ddp/dp3_faas | 32×4=128 | 128 |

---

## FAAS 集成

### 是什么

FAAS (Function-Actuator-Aligned Space) 来自 UniDex (CVPR 2026)，将 XHand 12 个原生关节映射到 **32 维功能对齐空间**（12 活跃 + 20 零填充），按关节功能组织。模型在 FAAS 空间训练/去噪，I/O 边界自动转换。

### 使用

```bash
bash scripts/train.sh dp3_faas                                    # joint mode (39D)
bash scripts/train.sh dp3_faas 'action_key=action_ee'             # EE mode (41D)
bash scripts/train_ddp.sh ddp/dp3_faas                            # DDP 多卡
```

### 维度

```
                  Native          FAAS
                  ──────          ────
action (joint):   19 =  7+12     39 =  7+32
action (ee):      21 =  9+12     41 =  9+32
joint_state:      19 =  7+12     39 =  7+32  (arm 始终 7D!)
```

### 转换边界（仅 I/O）

| 层 | 位置 | 操作 |
|----|------|------|
| Dataset | `base_dataset.py._apply_faas_mapping()` | 增强后、torch转换后：native→FAAS |
| 推理输入 | `base.py.predict_action()` | `_convert_obs_to_faas()`（幂等：仅19D→39D时转换） |
| 推理输出 | `base.py.predict_action_from_cond()` | unnormalize后：FAAS→native |
| 训练指标 | `base.py.compute_action_mse()` | GT逆转换到native比较 |

### 关键模块

| 模块 | 文件 | 职责 |
|------|------|------|
| `FAASHandMapper` | `common/faas_mapper.py` | nn.Module，scatter/gather + scale/offset，buffer自包含于checkpoint |
| `inject_faas_into_agent` | `training/build_utils.py` | train/eval/smoke-test 共享后注入 |
| `_validate_faas_config` | `training/build_utils.py` | 7项校验（互斥、维度、normalizer mode等） |

### 活跃 FAAS 索引

`(1,2,3,6,7,8,12,13,17,18,22,23)` — 仅 `index_bend` (native[3]→FAAS[6]) 有 `scale=-1.0`；其余 `scale=1.0, offset=0.0`。

### 兼容性

- **直接兼容**: DP, DP3, ManiFlow, MoE, MultiTask, R3D, SAT（仅 config 差异）
- **不兼容**: DQRISE（VQ-VAE codebook 12D→32D 需重跑三阶段管道）
- **互斥**: `use_aux_ee` 和 `use_faas` 不能同时启用

---

## DQ-RISE

### 三阶段管道

| 阶段 | 脚本 | 内容 |
|------|------|------|
| 1 | `scripts/train_vq_hand.py` | VQ-VAE 预训练：EncoderMLP→ResidualVQ(2组×4码字=16种手势)→DecoderMLP |
| 2 | `scripts/extract_codebook.py` | 码本提取+PCA排序（使连续VQ索引平滑插值） |
| 3 | `train.py dqrise` | 联合扩散训练：UNet输入从21D压缩到tcp_dim+1(10D)，epsilon预测 |

### CodebookManager (`agents/vq_hand/codebook_manager.py`)

- `nn.Module` — checkpoint 自包含，无需外部文件
- `hand_pose_to_continuous_index()`: L2最近原型 → 连续标量
- `continuous_index_to_hand_pose()`: 半上取整+查表 → 手势
- 支持 `.npz` 持久化 (v1/v2/v3 格式)
- 自动校验 normalizer 一致性 (`torch.testing.assert_close`)

### 关键发现

- `vq_idx_used`（码本利用率）是决定性下游成功率预测器：<8→~0%，≥12→~60%
- 60% baseline 在 `pour` 任务上已实现
- 更高 LR (3e-4)，更长 warmup (2000)，更浅 UNet ([256,512])

---

## 文件组织

```
dexmani_policy/
  train.py                  # 单卡入口
  train_ddp.py              # 多卡 DDP 入口
  eval_sim.py               # 独立评测入口（Hydra-free）
  smoke_test.py             # 构建链冒烟测试
  configs/                  # 9 个 Hydra YAML
  configs/ddp/              # 6 个 DDP overlay
  agents/
    core/                   # BaseAgent, UNetDiffusionAgent, DiTXFlowMatchAgent, 8 agent variants
    action_decoders/        # Diffusion, FlowMatch, FlowMatchWithConsistency, SATFlowMatch, TimeSampler
      backbone/             # UNet1D (ConditionalUnet1D), DiT, DiTX (DiTXFlowMatch), OneWayTransformer, SATBackbone
    obs_encoder/            # pointcloud/, rgb/, text/, state_mlp.py, plugins/(moe, token_compressor)
    vq_hand/                # DQ-RISE 专用: VQVAEHand, CodebookManager, ResidualVQ, VectorQuantize
    optim_util.py           # OptimGroupMixin, get_optim_group_with_no_decay
    position_encodings.py   # SinusoidalPosEmb, TimestepMLP, RelativePositionalEncoding3D
  datasets/                 # BaseDataset, PCDataset, RGBPCDataset, common/(ReplayBuffer, Sampler)
  training/                 # Trainer, build_utils, SimEvaluator, workspace, ema, logging, lr_scheduler, eval_utils
  env_runner/               # BaseRunner, SimRunner, MultiTaskSimRunner
  common/                   # LinearNormalizer, faas_mapper, checkpoint_io, pytorch_util, config
  scripts/                  # extract_codebook.py, train_vq_hand.py, measure_vq_usage.py
```

---

## 已知硬编码与设计约定

### 硬编码常量（不可从 config 修改）

| 项 | 值 | 位置 |
|----|-----|------|
| DDIM scheduler 参数 | `beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2'` | `diffusion.py:39-41` |
| StateMLP hidden | `[64]` | `state_mlp.py` |
| FlowMatch consistency 权重 | `1.0` (implicit: `loss = loss_flow + loss_consistency`) | `flowmatch.py:234` |
| 优化器类型 | `torch.optim.AdamW` + `fused=True` | `base.py:238` |
| DINO/CLIP/SigLIP 加载 | `torch_dtype=bfloat16`, `attn_implementation="sdpa"` | dino.py, clip.py, siglip.py |
| UNet conditioning | `cond_predict_scale=True` | `unet1d.py` |
| FAAS 映射索引 | `(1,2,3,6,7,8,12,13,17,18,22,23)`, 仅 index_bend scale=-1.0 | `faas_mapper.py` |
| horizon/n_obs_steps/n_action_steps | 16/2/8 | 所有 config |

### 设计约定（审查时易误报为 bug，勿修）

- **Normalizer 全量拟合**: `get_normalizer()` 使用全部 replay buffer（含验证集），不按 `train_mask` 过滤。生态统一惯例（ManiFlow_Policy/R3D-Policy/SAT/RoboTwin 均如此）。`limits` 模式下验证集不影响 min/max
- **里程碑 checkpoint**: 按 20%/40%/60%/80%/100% 进度保存，共 5 个；`latest.pt` symlink 指向最新里程碑
- **FlowMatch `target_t` 训练/推理偏移**: 训练用 `target_t=0`，推理用 `target_t=dt>0`。ManiFlow 通过 EMA teacher consistency 路径缓解
- **`tcp_dim` 命名**: 在 `action`(joint)模式下=7(臂关节角)，在 `action_ee` 模式下=9(TCP位姿)。历史命名遗留，勿据此推断语义
- **DDP 覆盖不全**: 仅 6 种策略有 DDP 配置。`dp3`(非FAAS)和`moe_dp` 仅单卡
- **MoE dual-backbone**: 通过 `hasattr(config, 'pc_encoder')` 自动切换 RGB/PC 路径，非显式 `backbone_type` 字段
- **MoE forward 返回 dict**: `MoEAgent.forward()` 返回 `dict`(含 aux_loss)，其他 agent 返回 `Tensor`。`BaseAgent.compute_loss()` 统一处理
- **R3DObsEncoder 拼接**: patch_tokens + state_emb + pc_pe 沿 feature 维拼接（非 `torch.cat`）
- **EMAModel BatchNorm**: affine 参数直接复制（不 EMA 平均）；frozen 参数也直接复制
- **DQRISEAgent 独立 UNet**: 直接继承 `BaseAgent`，自建 UNet+Diffusion（因 `diffusion_action_dim = tcp_dim+1 ≠ action_dim`）
### 已实现但未启用的功能

| 功能 | 位置 | 状态 |
|------|------|------|
| Modality Dropout | `base.py.preprocess()` | 全部配置中 `modality_dropout_probs=0.0` |
| TokenCompressor | `obs_encoder/plugins/token_compressor.py` | 未接入任何 agent config |
| T5TextEncoder | `obs_encoder/text/t5.py` | 预留代码，未使用 |

---

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/DQ-RISE-知识体系.md` | DQ-RISE 论文精读 + 官方代码走读 + 本项目实现差异 + 12 个官方代码 bug |
| `docs/UniDex-知识体系.md` | UniDex CVPR 2026 完整分析 — VLA+FAAS+多手，架构 + 与 DexMani 对比 |
| `docs/FAAS-集成方案.md` | FAAS 集成设计文档 — 架构设计、维度分析、agent 兼容矩阵 |
| `docs/FAAS-迁移-最佳方案.md` | FAAS 迁移 v5 实施方案 — 3 轨对比、实施步骤、风险缓解 |
| `docs/UniDex-可借鉴设计.md` | 8 项 UniDex 可借鉴设计，3 维评分，分优先级执行路线图 |
