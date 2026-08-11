# DexMani_Policy

灵巧手操作模仿学习框架 —— Hydra 配置驱动 · Zarr Replay Buffer · Diffusion/FlowMatch 动作解码 · `dexmani_sim` 仿真评测

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green)](https://developer.nvidia.com/cuda-toolkit)

---

## 快速开始

### 环境搭建

```bash
conda activate policy                                         # Conda 环境
pip install -e .                                              # 安装 dexmani_policy
cd ~/Desktop/dexmani_sim && pip install -e .                  # 仿真环境（评测依赖）
```

> **数据路径**: 训练前需设 `export DATA_DIR=/path/to/robot_data`，或确保 `robot_data/` 在项目根目录下。

### 训练

```bash
# 单卡 —— 11 种策略可选
bash scripts/training/train.sh dp3 pour                      # DP3 训练 pour 任务
bash scripts/training/train.sh maniflow pour                  # ManiFlow
bash scripts/training/train.sh sat pour                       # SAT
bash scripts/training/train.sh standard_flowmatch pour        # StandardFlowMatch
bash scripts/training/train.sh opfa pour                      # OPFA

# 多卡 DDP —— 9 种策略可选
bash scripts/training/train_ddp.sh ddp/maniflow pour          # ManiFlow 4 卡

# Hydra 参数覆盖
bash scripts/training/train.sh dp3 pour 'training.seed=42'
bash scripts/training/train.sh dp3 pour 'training.loop.total_train_steps=500'
```

### 评测

```bash
# 一键管道：select best ckpt → eval → demo
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_dir>

# 或分步执行
bash scripts/eval/select_best_ckpt.sh dp3 pour <exp_dir>     # 阶段 1: 选出最优 ckpt
bash scripts/eval/eval_best_ckpt.sh dp3 pour <exp_dir>       # 阶段 2: 最终评测
bash scripts/eval/record_demo.sh dp3 pour <exp_dir>           # 录制 demo 视频
```

### 冒烟测试

```bash
python dexmani_policy/smoke_test.py dp3                       # 单策略
python dexmani_policy/smoke_test.py dp3 maniflow standard_flowmatch sat opfa  # 批量
```

---

## 策略矩阵

9 种 Agent，覆盖 RGB/点云/语言多模态，Diffusion/FlowMatch 双解码范式。另有 2 种实验性策略（OPFA、StandardFlowMatch）。

| Agent | 感知模态 | 编码器 | 骨干网络 | 解码器 | 配置 |
|:------|:---------|:-------|:---------|:-------|:-----|
| **DP** | RGB + Joint | DINO/CLIP/SigLIP + StateMLP | UNet1D (FiLM) | Diffusion DDIM | `dp.yaml` |
| **DP3** | PC(1024,3) + Joint | PointNeXT + StateMLP | UNet1D (FiLM) | Diffusion DDIM | `dp3.yaml` |
| **ManiFlow** | PC(1024,3) + Joint | PointNeXT(patch) + StateMLP | DiTX (cross-attn) | FlowMatch + Consistency | `maniflow.yaml` |
| **StandardFlowMatch** | PC(1024,6) + Joint | PointNetDense(per-pt) + StateMLP | DiTDiffusion (cross-attn) | FlowMatch (纯) | `standard_flowmatch.yaml` |
| **MoE** | RGB + Joint | R3M + MoE(16×top-2) + StateMLP | UNet1D (FiLM) | Diffusion DDPM | `moe_dp.yaml` |
| **MultiTask** | RGB + Joint + Text | DINO + CLIP Text + StateMLP | DiT (AdaLN-Zero) | Diffusion / FlowMatch | `multitask_dit.yaml` |
| **R3D** | PC(1024,3) + Joint | Uni3D(ViT+Fourier) + StateMLP | OneWayTransformer | Diffusion DDIM | `r3d.yaml` |
| **DQ-RISE** | PC(1024,3) + Joint | iDP3 + StateMLP + Codebook | UNet1D (缩减) | Diffusion ε-pred | `dqrise.yaml` |
| **DP3 FAAS** | PC(1024,3) + Joint | 同 DP3 | 同 DP3 | 同 DP3 | `dp3_faas.yaml` |
| **SAT** | PC(1024,3) + Joint | PointNeXT(patch) + StateMLP | SATBackbone (EJC+MMA) | FlowMatch Euler | `sat.yaml` |
| **OPFA** | PC(1024,3) + HandLatent | PointNet + GaLR Encoder | UNet1D (512,1024,2048) | Diffusion DDIM | `opfa.yaml` |

### 关键差异速览

| 对比维度 | 说明 |
|:---------|:-----|
| **DP vs DP3** | RGB 图像 vs 点云。DP3 对遮挡和视角变化更鲁棒 |
| **DP3 vs ManiFlow** | Diffusion (DDIM) vs FlowMatch (直线路径 + consistency) |
| **ManiFlow vs StandardFlowMatch** | DiTXFlowMatch(双时间步+consistency) vs DiTDiffusion(单时间步+纯flow) |
| **DP3 vs MoE** | MoE 在 encoder 中引入 16-expert 稀疏路由 (top-2)，增容量不增推理 FLOPs |
| **DP3 vs SAT** | SAT 使用结构中心动作表示 (B,Da,T) + EJC 关节编码，CVPR 2026 |
| **DP3 vs R3D** | R3D 使用级联 self-attn mask + 分组 loss |
| **DP3 vs DQ-RISE** | DQ-RISE 通过 VQ-VAE 将手势离散化为 16 种码本 |
| **Native vs FAAS** | FAAS 将 12D 手势映射到 32D 功能对齐空间，零 agent 代码变更 |

> 详细对比见 [`CLAUDE.md`](CLAUDE.md) — Agent 变体章节。

---

## 核心约束

以下常量**不可随意修改**，SequenceSampler、control_action 切片、环境接口均与之耦合：

| 常量 | 值 | 说明 |
|:-----|:---|:-----|
| `horizon` | **16** | 预测总帧数 |
| `n_obs_steps` | **2** | 历史观测帧数 |
| `n_action_steps` | **8** | 每步执行的动作帧数 |
| `action_dim` | **19** | 动作维度 (7-DOF 臂 + 12-DOF 手) |
| `pad_before` / `pad_after` | **1** / **7** | 序列采样边界填充 |

关系式：`n_obs_steps - 1 + n_action_steps ≤ horizon` → `1 + 8 ≤ 16 ✓`

---

## 数据流

```
Zarr (N,*) ──→ ReplayBuffer ──→ SequenceSampler(pad_before=1, pad_after=7)
  └── Dataset.__getitem__() ──→ DataLoader ──→ batch (B, 16, *)

Agent.compute_loss():                          Agent.predict_action():
  obs │→ normalize │→ truncate[:,:2]            obs │→ normalize │→ truncate
      │→ flatten(B×2,*) │→ encoder │→ cond          │→ encoder │→ cond
  act │→ normalize │→ decoder(cond, act)             │→ decoder.predict_action(cond)
      │→ loss (MSE)                                  │→ unnormalize
                                                     │→ control_action = pred[:, 1:9]
```

| 阶段 | 形状 |
|:-----|:-----|
| Zarr 存储 | `action (N,19)` `joint_state (N,19)` `point_cloud (N,1024,3)` |
| Sequence Sample | `obs (*,16)` `action (16,19)` |
| Batch | `obs (B,16,*)` `action (B,16,19)` |
| Preprocessed | `obs (B×2,*)` → flatten batch+time |
| Model Output | `pred (B,16,action_dim)` → `control_action (B,8,action_dim)` |

---

## 实验目录

```
experiments/
└── <policy>/<task>/<timestamp>_<seed>/
    ├── config.yaml              # Hydra 配置快照
    ├── checkpoints/
    │   ├── latest.pt            # → 最新里程碑 symlink
    │   ├── best.pt              # → 最优 ckpt symlink
    │   ├── best_ckpt.json       # select_best_ckpt 输出
    │   └── epoch=*-step=*-milestone=*pct.pt
    ├── logs.jsonl               # 结构化训练日志
    ├── eval_dexsim/             # 评测产出
    │   ├── _result.txt
    │   ├── result_details.json
    │   └── <YYYYmmdd_HHMMSS>/   # 视频（默认录制）
    └── wandb/                   # Wandb 离线日志
```

---

## 文档导航

| 文档 | 内容 | 适合 |
|:-----|:-----|:-----|
| [`CLAUDE.md`](CLAUDE.md) | AI 工作速查 —— 完整 Agent 对比、训练/评测细节、配置速查、硬编码约定 | 日常开发 |
| [`docs/项目架构.md`](docs/项目架构.md) | 架构全景 —— 完整目录树、模块依赖图、类层级、数据流、设计模式 | 深入理解 |
| [`docs/评测机制.md`](docs/评测机制.md) | 评测全链路 —— CLI→Checkpoint→Agent→EnvRunner→SuccessRate 完整代码走读 | 评测开发 |
| [`docs/SSH服务器训练部署.md`](docs/SSH服务器训练部署.md) | 远程训练部署 —— SSH 配置、三向同步、GPU 多租户、tmux 管理 | 服务器运维 |

---

## 常见问题

**Q: 如何新增任务？**
1. 准备 Zarr 数据 → `robot_data/<task>.zarr`
2. 修改配置中的 `task_name` 和 `zarr_path`
3. 若 `dexmani_sim` 有对应环境，设置 `env_runner.task_name`

**Q: 如何启用数据增强？**
取消配置中 `augmentation_cfg` 的注释。RGB 增强需 `sensor_modalities` 含 `rgb`；PC 颜色增强需 `pc_dim >= 6`。

**Q: 单卡 checkpoint 能用于 DDP 续训吗？**
能。Checkpoint 始终以 unwrapped 格式保存，`fix_state_dict()` 自动处理 `module.` 前缀。

**Q: 训练中断后如何续训？**
直接重新运行相同命令，自动从 `latest.pt` 续训。

**Q: 如何选择评测 checkpoint？**
- `best` → `best_ckpt.json` 或 `best.pt` symlink（推荐）
- `latest` → `latest.pt` symlink
- `20pct`..`100pct` → 里程碑 checkpoint
- 直接路径 → 指定 `.pt` 文件

**Q: 纯 FlowMatch 模式（无 consistency）能用吗？**
可以。使用 `StandardFlowMatch` 策略（`standard_flowmatch.yaml`），基于 DiTDiffusion backbone（单时间步），无需 EMA 教师或 consistency 训练。
旧的 ManiFlow 配置设 `use_ema_teacher_for_consistency: false` 也可禁用 consistency，但推理时 `target_t=dt>0` 落在训练分布外（DiTXFlowMatch 需要 `target_t` 参数），有分布偏移风险。

**Q: 如何修改观测/动作步数？**
修改 `horizon`、`n_obs_steps`、`n_action_steps`，须满足 `n_obs_steps - 1 + n_action_steps ≤ horizon`。`pad_before`/`pad_after` 需同步调整。
