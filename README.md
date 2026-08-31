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

> **数据路径**: 训练数据从 `robot_data/<task>.zarr` 读取（`zarr_path` 相对仓库根目录）；确保 `robot_data/` 在项目根目录下即可，无需设环境变量。

### 训练

```bash
# 单卡 —— 9 种策略可选（task 用 task_name= 覆盖）
bash scripts/training/train.sh dp3 'task_name=pour'           # DP3 训练 pour 任务
bash scripts/training/train.sh action_flow 'task_name=pour'   # ActionFlow
bash scripts/training/train.sh maniflow 'task_name=pour'      # ManiFlow
bash scripts/training/train.sh sat 'task_name=pour'           # SAT

# 多卡 DDP —— 7 种策略可选
bash scripts/training/train_ddp.sh ddp/maniflow 'task_name=pour'  # ManiFlow 4 卡

# Hydra 参数覆盖
bash scripts/training/train.sh dp3 'task_name=pour' 'training.seed=42'
bash scripts/training/train.sh dp3 'task_name=pour' 'training.loop.total_train_steps=500'
```

> 单卡策略: `action_flow dp dp3 dqrise maniflow moe_dp multitask_dit r3d sat`
> DDP 策略: `ddp/action_flow ddp/dp ddp/dqrise ddp/maniflow ddp/multitask_dit ddp/r3d ddp/sat`

### 评测

```bash
# 一键管道：select best ckpt → eval → demo
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_name>

# 或分步执行
bash scripts/eval/select_best_ckpt.sh dp3 pour <exp_name>     # 阶段 1: 选出最优 ckpt
bash scripts/eval/eval_best_ckpt.sh dp3 pour <exp_name>       # 阶段 2: 最终评测
bash scripts/eval/record_demo.sh dp3 pour <exp_name>           # 录制 demo 视频

# ActionFlow：同 checkpoint、同 seeds 的 Euler/Midpoint × NFE 配对评测
bash scripts/eval/eval_action_flow_solvers.sh action_flow pour <exp_name> --episodes 25
```

`<exp_name>` = `experiments/<policy>/<task>/` 下的时间戳/名称目录（非完整路径）。

ActionFlow 的 `denoise_steps` 就是 NFE。Euler 支持任意正整数 NFE（包括 1 和 10）；
Midpoint 只支持偶数 NFE。`eval_action_flow_solvers.sh` 的固定首轮组合为
Euler-1、Midpoint-2、Midpoint-4、Midpoint-8、Midpoint-10；需要评测其他 NFE 时可通过
`eval_best_ckpt.sh --denoise-steps N` 单独评测。

### 冒烟测试

```bash
python dexmani_policy/smoke_test.py dp3                       # 单策略
python dexmani_policy/smoke_test.py dp3 maniflow sat          # 批量
```

---

## 策略矩阵

9 种策略配置，覆盖 RGB/点云/语言多模态，Diffusion/FlowMatch 双解码范式。

| Agent | 感知模态 | 编码器 | 骨干网络 | 解码器 | 配置 |
|:------|:---------|:-------|:---------|:-------|:-----|
| **DP** | RGB + Joint | DINO/CLIP/SigLIP + StateMLP | UNet1D (FiLM) | Diffusion DDIM | `dp.yaml` |
| **DP3** | PC(1024,6) + Joint | PointNet + StateMLP | UNet1D (FiLM) | Diffusion DDIM | `dp3.yaml` |
| **ManiFlow** | PC(1024,6) + Joint | PointNetDense + StateMLP | DiTX (cross-attn) | FlowMatch + Consistency | `maniflow.yaml` |
| **ActionFlow** | PC(1024,6) + Joint | PointNeXT 192-patch local tokenizer → GeoFormer 4L×576(3D RoPE) → 385×384 memory | ActionFlowDiT 8L×768 (12Q/12KV full CA, SwiGLU-1536, shared AdaRMS, KV cache) | SimpleRectifiedFlow | `action_flow.yaml` |
| **MoE** | RGB + Joint | R3M + MoE(16×top-2) + StateMLP | UNet1D (FiLM) | Diffusion DDIM (100步) | `moe_dp.yaml` |
| **MultiTask** | RGB + Joint + Text | ResNet(resnet18) + CLIP Text + StateMLP | DiT (AdaLN) | Diffusion / FlowMatch | `multitask_dit.yaml` |
| **R3D** | PC(1024,6) + Joint | Uni3D(ViT+Fourier) + StateMLP | OneWayTransformer | Diffusion DDIM | `r3d.yaml` |
| **DQ-RISE** | PC(1024,6) + Joint | iDP3 + StateMLP + Codebook | UNet1D (缩减) | Diffusion ε-pred | `dqrise.yaml` |
| **SAT** | PC(1024,6) + Joint | PointNeXT(patch) + StateMLP | SATBackbone (EJC+MMA) | FlowMatch Euler | `sat.yaml` |

### 关键差异速览

| 对比维度 | 说明 |
|:---------|:-----|
| **DP vs DP3** | RGB 图像 vs 点云。DP3 对遮挡和视角变化更鲁棒 |
| **DP3 vs ManiFlow** | Diffusion (DDIM) vs FlowMatch (直线路径 + consistency) |
| **ManiFlow vs ActionFlow** | FlowMatch+Consistency(EMA教师) vs SimpleRectifiedFlow(NoiseShift α=3+mix, KV cache, 2步推理) |
| **DP3 vs MoE** | MoE 在 encoder 中引入 16-expert 稀疏路由 (top-2)，增容量不增推理 FLOPs |
| **DP3 vs SAT** | SAT 使用结构中心动作表示 (B,Da,T) + EJC 关节编码，CVPR 2026 |
| **DP3 vs R3D** | R3D 使用级联 self-attn mask + 分组 loss |
| **DP3 vs DQ-RISE** | DQ-RISE 通过 VQ-VAE 将手势离散化为 16 种手势 |

### Agent 继承模式

添加新策略时，选择合适的父类：

| 模式 | 父类 | 何时用 | 最近参考 |
|------|------|--------|---------|
| **A: UNet+Diffusion** | `UNetDiffusionAgent` | Flat encoding → UNet1D(FiLM) → Diffusion | `dp3.py` |
| **B: DiTX+FlowMatch+Consistency** | `DiTXFlowMatchAgent` | Token seq → DiTX(cross-attn) → FlowMatch+consistency | `maniflow.py` |
| **C: Fully custom** | `BaseAgent` | 完全自定义 backbone + decoder | `sat.py`, `r3d.py` |

> 详细集成步骤 → `dexmani-agent-integration` skill (`.agents/skills/`)

### 动作空间

| `action_key` | arm | hand | total |
|-------------|-----|------|-------|-----------|
| `action` (joint) | 7 (关节角) | 12 (XHand) | **19** |
| `action_ee` (ee) | 9 (pos3+rot6d) | 12 (XHand) | **21** |

`joint_state` dim ≡ action dim。

### Action Decoder 类型

| Decoder | 预测目标 | 推理 | 使用策略 |
|---------|---------|------|---------|
| `Diffusion` | ε / x0 / v | DDIM 迭代 | DP, DP3, MoE, R3D, DQRISE |
| `FlowMatch` / `SATFlowMatch` | v=x1-x0 | Euler ODE | MultiTask, SAT |
| `FlowMatchWithConsistency` | v + consistency(EMA教师) | Euler ODE | ManiFlow |
| `SimpleRectifiedFlow` | v=x1-x0 (NoiseShift α=3 + 75/25 uniform mixture) | Euler（任意正 NFE）/ Explicit Midpoint（偶数 NFE，KV cache） | ActionFlow |

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

其他硬编码常量（不可从 config 修改）：

| 项 | 值 | 位置 |
|----|-----|------|
| 优化器 | `AdamW(fused=torch.cuda.is_available())` | `base.py:297` |
| DDIM scheduler | `beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2'` | `diffusion.py:43` |
| StateMLP hidden | `[64]` | `state_mlp.py` |
| ViT backbone dtype | `bfloat16 + attn_implementation="sdpa"` | dino/clip/siglip |
| UNet conditioning | `cond_predict_scale=True` | `unet1d.py` |

---

## 配置参考

### 关键参数 (跨策略差异)

| 参数 | action_flow | dp3 | maniflow | moe_dp | multitask_dit | r3d | dqrise | sat |
|------|-------------|-----|----------|--------|---------------|-----|---------|-----|
| action_dim | 19/21 | 19/21 | 19/21 | 19/21 | 19/21 | 19/21/28 | 19/21 | 19/21 |
| backbone | ActionFlowDiT 8L×768 | UNet[256,512,1024] | DiTX 12L×768 | UNet[256,512,1024] | DiT 8L×512 | OneWay 4L | UNet[256,512] | SAT 8L×768 |
| train/infer steps | -/2 NFE | 100/10 | -/4 | 100/100 | 100/10 | 100/10 | 100/20 | -/10 |
| prediction_type | velocity | sample | velocity | sample | sample | sample | epsilon | velocity |
| lr / wd | 1e-4 / **1e-3** | 1e-4 / 1e-6 | 1e-4 / **1e-3** | 1e-4 / 1e-6 | 1e-4 / 1e-6 | 1e-4 / 1e-6 | **3e-4** / 1e-6 | 1e-4 / 1e-6 |
| betas | **[.9,.95]** | [.95,.999] | **[.9,.95]** | [.95,.999] | [.95,.999] | [.95,.999] | [.95,.999] | [.95,.999] |
| bfloat16 / compile | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **✗ / ✗** | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓(default) |
| val_ratio | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

> dp = dp3 参数; multitask_dit = 8L×512, lr=1e-4
> 全部 `total_train_steps: 100000`, `warmup: 500` (dqrise: 2000)

### 新建 Config

复制 `dp3.yaml` 作为模板。必须字段：

`policy_name, task_name, zarr_path, seed, horizon(16), n_obs_steps(2), n_action_steps(8), action_key, action_dim, dataloader, val_dataloader, dataset, agent, optimizer, ema, training, workspace, env_runner, eval, hydra`

`action_dim` 公式：
```yaml
action_dim: ${eval:'21 if ${eq:${action_key},action_ee} else 19'}
```

`agent._target_` 指向 `dexmani_policy.agents.core.<name>.<Name>Agent`（无显式注册表，Hydra 直接导入）。

### Eval 配置

```yaml
eval:
  denoise_steps: 10  # 默认值；action_flow=2 / maniflow=4 / dqrise=20 由各自 config 覆盖
  use_ema: true      # 所有策略共享
  select_best: {initial_episodes: 25, batch_size: 5, max_episodes: 100}
  offline: {episodes: 100}
  demo: {episodes: 5, viewer_resolution: [1920, 1080]}
```

参数优先级: CLI > 子节覆盖 > eval 共享层 > 旧字段兼容 > hardcoded fallback

### DDP 批次大小（4 卡，grad-accum=1）

| DDP 策略 | 每卡 batch | 4 卡总 batch | 单卡 batch |
|---------|-----------|-------------|-----------|
| ddp/action_flow | 32 | 128 | 64 |
| ddp/dp | 48 | **192** | 64 |
| ddp/dqrise | 32 | 128 | 128 |
| ddp/maniflow | 32 | 128 | 128 |
| ddp/multitask_dit | 16 | 64 | 64 |
| ddp/r3d | 32 | 128 | 128 |
| ddp/sat | 32 | 128 | 128 |

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

## 训练机制

### NaN 两层防护

| 层 | 何时 | 检测 | 响应 |
|----|------|------|------|
| 1 | backward **前** | `raw_loss` NaN | DDP 广播 → 保存 NaN debug checkpoint (最近5个) → raise |
| 2 | optimizer.step **前** | 梯度 NaN | zero_grad → raise (含参数名) |

诊断用 `dexmani-training-debug` skill。

### 关键机制

- **梯度累积**: `raw_loss / gradient_accumulation_steps` → backward; DDP 非边界 `model.no_sync()`, 仅边界 all-reduce
- **Checkpoint**: 20/40/60/80/100% 里程碑各一个; `latest.pt` symlink 指向最新; Resume 自动从 latest 恢复
- **EMA**: 逆 gamma 衰减; BatchNorm affine 直接复制 (不平均)
- **DDP**: `mp.spawn`, NCCL, `find_unused_parameters=False`; ckpt 加载在 compile + DDP 包装**之前**; timeout=30min; `dp3` 和 `moe_dp` 有意仅单卡
- **Shape 验证**: `BaseAgent._validate_batch()` 在 `compute_loss`/`predict_action` 入口校验 action ndim/horizon/dim + obs 时间维/模态batch一致性

---

## DQ-RISE

三阶段管道：VQ-VAE 预训练 → 码本提取+PCA排序 → 联合扩散训练。

| 阶段 | 脚本 | 内容 |
|------|------|------|
| 1 | `dexmani_policy/tools/train_vq_hand.py` | VQ-VAE 预训练：EncoderMLP→ResidualVQ(2组×4码字=16种手势)→DecoderMLP |
| 2 | `dexmani_policy/tools/extract_codebook.py` | 码本提取+PCA排序（使连续VQ索引平滑插值） |
| 3 | `train.py dqrise` | 联合扩散训练：UNet输入从21D压缩到tcp_dim+1(10D)，epsilon预测 |

关键发现：`vq_idx_used`（码本利用率）是决定性下游成功率预测器——<8→~0%，≥12→~60%。

---

## 实验目录

```
experiments/
└── <policy>/<task>/<timestamp>_<seed>/
    ├── config.yaml              # Hydra 配置快照
    ├── best_ckpt.json           # select_best_ckpt 输出（实验根目录）
    ├── checkpoints/
    │   ├── latest.pt            # → 最新里程碑 symlink
    │   ├── best.pt              # → 最优 ckpt symlink（仅 --link-best 时生成）
    │   └── epoch=*-step=*-milestone=*pct.pt
    ├── logs.jsonl               # 结构化训练日志
    ├── eval_dexsim/             # 评测产出
    │   ├── _result.txt
    │   ├── result_details.json
    │   └── <YYYYmmdd_HHMMSS>/   # 视频（默认录制）
    └── wandb/                   # Wandb 离线日志
```

---

## 设计约定

以下设计在代码审查时容易被误判为 bug，但它们是有意为之：

- **Normalizer 全量拟合**: 用全部 replay buffer (含验证集)。生态惯例 (ManiFlow/R3D/SAT/RoboTwin 均如此)。`limits` 模式下 val 不影响 min/max
- **`tcp_dim` 命名**: joint模式=7(臂关节), ee模式=9(TCP位姿) — 历史命名，勿据此推断语义
- **MoE forward 返回 `dict`** (含 `aux_loss`): `BaseAgent.compute_loss()` 统一处理 dict/Tensor
- **MoE 无 bfloat16/compile**: gate softmax 需 float32; CUDA Graphs + MoE routing 内存开销大
- **DQRISE 直接继承 `BaseAgent`**: `diffusion_action_dim = tcp_dim+1` ≠ `action_dim`，无法复用 UNetDiffusionAgent
- **R3DObsEncoder 拼接**: patch_tokens + state_emb + pc_pe 沿 feature 维 (非 `torch.cat`)
- **EMAModel BatchNorm**: affine 参数直接复制，不 EMA 平均
- **FlowMatchWithConsistency `target_t`**: flow 分支训练=0，consistency 分支训练=dt1(>0)，推理=dt(>0)
- **Milestone checkpoint**: 仅 20/40/60/80/100% 五个; `latest.pt` 是 symlink

### 未启用功能 (不要意外激活)

| 功能 | 位置 | 状态 |
|------|------|------|
| Modality Dropout | `base.py` | 全配置 `modality_dropout_probs=0.0` |
| TokenCompressor | `obs_encoder/plugins/` | 未接入任何 config |
| T5TextEncoder | `obs_encoder/text/t5.py` | 预留代码 |

---

## 文档导航

| 文档 | 内容 | 适合 |
|:-----|:-----|:-----|
| [`CLAUDE.md`](CLAUDE.md) | AI 工作速查 —— 命令、不变量、文件地图、Agent Skills | AI 编码助手 |
| [`docs/项目架构.md`](docs/项目架构.md) | 架构全景 —— 完整目录树、模块依赖图、类层级、数据流、设计模式 | 深入理解 |
| [`docs/仿真评测机制.md`](docs/仿真评测机制.md) | 评测全链路 —— CLI→Checkpoint→Agent→EnvRunner→SuccessRate 完整代码走读 | 评测开发 |
| [`docs/SSH服务器训练部署.md`](docs/SSH服务器训练部署.md) | 远程训练部署 —— SSH 配置、三向同步、GPU 多租户、tmux 管理 | 服务器运维 |
| [`docs/DP3-R3D-ManiFlow测试结果0813.md`](docs/DP3-R3D-ManiFlow测试结果0813.md) | 策略对比评测 —— DP3 vs R3D vs ManiFlow 五项任务成功率 + 里程碑分析 | 策略选型 |
| [`docs/ActionFlow-架构与实验结果.md`](docs/ActionFlow-架构与实验结果.md) | ActionFlow 唯一权威文档 —— 历史架构沿革/当前架构/实验记录/结论方法论 | ActionFlow 开发 |

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

**Q: 如何修改观测/动作步数？**
修改 `horizon`、`n_obs_steps`、`n_action_steps`，须满足 `n_obs_steps - 1 + n_action_steps ≤ horizon`。`pad_before`/`pad_after` 需同步调整。
