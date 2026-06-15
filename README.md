# dexmani_policy

灵巧手操作模仿学习框架。Hydra 配置驱动，Zarr replay buffer，Diffusion/FlowMatch 动作解码，支持单任务/多任务训练与 `dexmani_sim` 仿真评测。

> 详细设计见 `docs/`（01 → 08）。工作速查见 `CLAUDE.md`。审查约定见 `docs/08-审查约定.md`。

---

## 核心约束

```
horizon=16   n_obs_steps=2   n_action_steps=8   action_dim=19
```

所有策略接收 2 步历史观测，预测 16 步动作轨迹（future），实际执行前 8 步后重新推理（receding horizon control）。动作空间 = XArm7（7-DOF 关节角）+ XHand（12-DOF 关节位置）= 19 维绝对关节控制。

**这些值不可随意修改**：SequenceSampler 的 `pad_before=1, pad_after=7` 与之耦合；`n_obs_steps-1 + n_action_steps ≤ horizon` 是 control_action 切片 `pred[:, 1:9, :]` 的硬约束。

---

## 策略矩阵

| Agent | 感知 | 编码器 | 骨干网络 | 解码器 | 配置 |
|---|---|---|---|---|---|
| **DP** | RGB + joint_state | DINO/CLIP + StateMLP | ConditionalUnet1D (FiLM) | DDIM Diffusion | `dp.yaml` |
| **DP3** | PointCloud(1024,3) + joint_state | PointNeXT + StateMLP | ConditionalUnet1D (FiLM) | DDIM Diffusion | `dp3.yaml` |
| **ManiFlow** | PointCloud(1024,3) + joint_state | PointNeXT + StateMLP | DiTXFlowMatch (cross-attn) | Euler ODE + Consistency | `maniflow.yaml` |
| **MoE DP3** | 同 DP3 | DP3 encoder + 16-expert MoE router | ConditionalUnet1D | DDIM Diffusion | `moe_dp3.yaml` |
| **MultiTask** | RGB + joint_state + language | DINO + CLIP Text + StateMLP | DiT_Diffusion (AdaLN-Zero) | DDIM Diffusion | `multitask_dit.yaml` |

关键差异：
- **DP vs DP3**：RGB 图像 vs 点云。DP3 对遮挡和视角变化更鲁棒，适合灵巧操作中手-物交互场景。
- **DP3 vs ManiFlow**：Diffusion (DDIM 迭代去噪) vs FlowMatch (rectified flow 直线路径 + consistency training)。FlowMatch 理论推理步数更少，consistency training 提供自校正。
- **DP3 vs MoE DP3**：MoE 在 encoder 中引入 16 个 expert 的稀疏路由，增加模型容量但不显著增加推理 FLOPs（top-k=2 选取）。
- **MultiTask**：通过 CLIP text embedding 为不同任务提供语言条件，共享视觉和动作 backbone。

---

## 环境搭建

- **Conda env**: `policy`
- **Python**: 3.10+ / **CUDA**: 11.8+ / **PyTorch**: 2.x
- **仿真评测依赖**: `dexmani_sim`（需单独安装）

```bash
conda activate policy
pip install -e .
cd ~/Desktop/dexmani_sim && pip install -e .
```

### 数据

```
robot_data/sim/
├── pick_apple_messy.zarr    ├── place_milk_box.zarr
├── pour.zarr                ├── open_box.zarr
├── multi_grasp.zarr         ├── pick_bottle.zarr
└── stack_cups.zarr
```

Zarr 格式，包含 `joint_state` (N,19)、`action` (N,19)、`point_cloud` (N,1024,3)、`rgb` (N,H,W,3)、`episode_ends` (n_ep,)。

---

## 快速开始

### 训练

```bash
bash scripts/train.sh dp3                                  # 单卡
bash scripts/train.sh dp3 'training.loop.num_epochs=100'   # Hydra override
bash scripts/train_ddp.sh maniflow_ddp                     # 多卡 DDP
bash scripts/train_ddp.sh maniflow_ddp 'training.num_gpus=2'
bash scripts/train_multi_task.sh multitask_dit             # 多任务
```

### 冒烟测试

```bash
python dexmani_policy/smoke_test.py dp3                    # 单 config
python dexmani_policy/smoke_test.py dp3 maniflow moe_dp3   # 批量
```

### 仿真评测

```bash
bash scripts/eval_sim.sh dp3 pick_apple_messy <exp_dir>
# 指定 checkpoint 和 denoise 步数
bash scripts/eval_sim.sh dp3 pick_apple_messy <exp_dir> \
    'eval.offline.ckpt_tag_or_path=best' \
    'eval.offline.denoise_timesteps_list=[5,10,20]'
```

### 日志

```bash
bash scripts/wandb_sync.sh <exp_dir>   # 同步 Wandb 离线日志
```

---

## 数据流

```
Zarr (N,*) → ReplayBuffer → SequenceSampler(pad_before=1, pad_after=7)
  → Dataset.__getitem__() (增强 + torch) → DataLoader → batch (B, 16, *)
  → Agent.compute_loss():
      obs → normalize(joint_state) → modality_dropout → truncate[:,:2]
        → flatten(B×2,*) → encoder → cond
      action → normalize → decoder(cond, action) → loss
  → loss.backward() → grad_accum → clip_grad → opt.step → EMA → ckpt
```

推理：
```
obs → normalize → truncate → encoder → cond
  → decoder.predict_action(cond, template=noise)
    [Diffusion]: DDIM 10步 → [FlowMatch]: Euler ODE 10步
  → unnormalize → control_action = pred[:, 1:9, :]
```

| 阶段 | 形状 | 说明 |
|---|---|---|
| Zarr | `joint_state (N,19)`, `action (N,19)`, `point_cloud (N,1024,3)` | 原始存储 |
| Sample | `obs (16,*)`, `action (16,19)` | 滑动窗口 |
| Batch | `obs (B,16,*)`, `action (B,16,19)` | DataLoader 批次 |
| Preprocessed | `obs (B×2,*)` | truncate + flatten batch+time |
| Cond (UNet) | `(B, out_dim×2)` | DP/DP3/MoE |
| Cond (DiT) | `(B, N_tokens, token_dim)` | ManiFlow/MultiTask |
| Output | `pred (B,16,19)` → `control_action (B,8,19)` | 全部策略一致 |

---

## 实验目录

```
experiments/
└── <policy_name>/<task_name>/<timestamp>_<seed>/
    ├── config.yaml          # Hydra 配置快照
    ├── checkpoints/
    │   ├── latest.pt → epoch=XXXX-step=YYYY-score=Z.pt
    │   ├── epoch=XXXX-step=YYYY-score=Z.pt
    │   └── topk_manifest.json
    ├── logs.jsonl           # 结构化日志
    ├── eval/                # 评测视频 + metrics.json
    └── wandb/               # Wandb 离线日志
```

---

## 关键约定

| 约定 | 详情 |
|---|---|
| **Normalizer** | `mode='limits'`，全量数据拟合 → [-1,1]；`range_eps=1e-4`，极低方差维度仅零中心化不缩放；仅拟合 `joint_state` + `action`，点云/RGB 直通 |
| **增强** | 默认禁用；`pc_dim` 须与 Zarr 点云通道数一致；`PCSpatialAug` 对 XYZ+法向施加同步旋转 |
| **FlowMatch** | EMA teacher 提供 consistency target；推理 `target_t=dt>0` 依赖 consistency 训练泛化；`flow_batch_ratio=0.75` |
| **MoE** | 16 experts, top-k=2；aux loss (load_balancing + entropy) 全程生效；全专家计算确保 DDP 兼容 |
| **MultiTask** | CLIP text encoder `requires_grad_(False)` 冻结，仅 `text_proj` 可训练；text cache 预计算 |
| **DDP** | `DistributedSampler` 分片；`batch_size` 为每卡值；`find_unused_parameters=False`；normalizer 通过 `dist.broadcast` 同步 |
| **Checkpoint** | unwrapped 保存（无 `module.` 前缀）；`fix_state_dict()` 自动处理加载；TopK=3，monitor `test_mean_score` |
| **NaN 防护** | 三层：loss NaN → zero_grad + raise；grad NaN → zero_grad + raise；DDP `all_reduce(nan_flag)` 防死锁 |
| **评测** | `eval.seed: 0` 固定可复现；评测时使用 checkpoint normalizer（不从数据集重新拟合） |
| **序列采样** | 短于 8 帧的 episode 自动 warn + skip；边界 padding 复制首尾帧 |

---

## 常见问题

**Q: 如何新增任务？**
1. 准备 Zarr → `robot_data/sim/<task>.zarr`
2. 修改配置 `task_name` 和 `zarr_path`
3. 若 `dexmani_sim` 有对应环境，设置 `env_runner.task_name`

**Q: 如何启用数据增强？**
取消配置中 `augmentation_cfg` block 注释。RGB 增强需 `sensor_modalities` 包含 `rgb`；PC 颜色增强需 `pc_dim >= 6`。

**Q: 单卡 checkpoint 能用于 DDP 续训吗？**
能。Checkpoint 始终以 unwrapped 格式保存，`fix_state_dict()` 自动处理 `module.` 前缀。

**Q: 训练中断后如何续训？**
直接重新运行相同命令，自动 `load_for_resume("latest")`。

**Q: 如何选择评测 checkpoint？**
- `best` → TopK 中 score 最高
- `latest` → `latest.pt` 符号链接
- 也可传具体文件名

**Q: 纯 flow 模式（无 consistency）能用吗？**
可以，`training.use_ema_teacher_for_consistency: false`。但推理时 `target_t=dt>0` 落在训练分布外（训练时 target_t 恒为 0），`train.py` 的 `validate_config()` 会发出 warning。

**Q: 如何修改观测/动作步数？**
修改 `horizon`、`n_obs_steps`、`n_action_steps`，且须满足 `n_obs_steps - 1 + n_action_steps ≤ horizon`。`pad_before`/`pad_after` 需同步调整。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `README.md` | 项目概览、快速开始、策略矩阵、FAQ |
| `CLAUDE.md` | Claude 工作速查（调用链、关键约定、代码模式） |
| `docs/01-项目概览.md` | 环境搭建、最小命令、架构速览 |
| `docs/02-配置系统.md` | Hydra 配置层级、插值、校验、策略对比 |
| `docs/03-数据集与增强.md` | Zarr 加载、ReplayBuffer、SequenceSampler、增强、Normalizer |
| `docs/04-模型架构.md` | 5 种 Agent：Encoder/Backbone/Decoder 详解 |
| `docs/05-训练机制.md` | Trainer 循环、EMA、checkpoint、LR schedule |
| `docs/06-仿真评测.md` | SimEvaluator、env_runner、dexmani_sim 环境 |
| `docs/07-多卡训练.md` | mp.spawn、DistributedSampler、rank 隔离、checkpoint 兼容 |
| `docs/08-审查约定.md` | 已知设计模式与架构决策，防止审查误报 |
