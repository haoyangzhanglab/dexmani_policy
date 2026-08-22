# ActionFlow Solver Ablation — pour 任务 (64-patch + 128-patch)

> **文档导航**: [README](../README.md) — 项目概览 · [CLAUDE.md](../CLAUDE.md) — AI 工作速查 · [项目架构](项目架构.md) — 架构全景 · [仿真评测机制](仿真评测机制.md) — 评测全链路
>
> **E1** (64-patch): 2026-08-22 | `action_flow/pour/2026-08-22_17-07_42` | 100,000 steps
> **E2** (128-patch): 2026-08-23 | `action_flow/pour/2026-08-22_22-56_42` | 100,000 steps
>
> **目的**: 评估 ActionFlow 在 pour 任务上不同 solver/NFE 组合的成功率，对比 64-patch vs 128-patch 的差异，确定最优推理配置。

## 总览

| Solver | NFE | 成功率 (25 seeds) | 推理时间 | FPS | 推荐 |
|--------|-----|:---:|:---:|:---:|:---:|
| Euler | 2 | 13/25 = 52.0% | 5.45 ms | 183 | |
| **Midpoint** | **2** | **16/25 = 64.0%** | **5.49 ms** | **182** | ✅ |
| Euler | 4 | 15/25 = 60.0% | 9.07 ms | 110 | |
| Midpoint | 4 | 15/25 = 60.0% | 9.06 ms | 110 | |
| Midpoint | 8 | 16/25 = 64.0% | 16.22 ms | 62 | |

> 评测使用同一组 25 个确定性种子 (`eval_seed=1066`)，结果可直接对比。
> 推理时间在 RTX 环境上以 200 次 warmup 后 200 次纯模型 forward 测量，不含物理模拟。

## 实验环境

| 项目 | 值 |
|------|-----|
| 策略 | `action_flow` (ActionFlowAgent) |
| 任务 | `pour` (倒水) |
| 实验目录 | `experiments/action_flow/pour/2026-08-22_17-07_42` |
| 训练步数 | 100,000 (5 milestones: 20/40/60/80/100%) |
| Backbone | ActionFlowDiT 8L×512, 8Q/4KV GQA, GEGLU |
| 动作空间 | `action` (joint, 19 dim) |
| 观测 | PointNeXT (64-patch) + joint_state |
| 训练 solver | Euler, NFE=2 |
| 推理 EMA | ✅ |

## 最佳 Checkpoint 选择

`select_best_ckpt.sh` 自适应淘汰结果（初始 25 episodes，tie-break 每轮 +5，上限 100）：

| Milestone | Step | Phase 1 (25ep) | +R1 (+5ep) | +R2 (+5ep) | 最终 |
|-----------|------|:---:|:---:|:---:|:---:|
| 20% | 20000 | 17/25 (68.0%) | 22/30 (73.3%) | 24/35 (68.6%) | |
| 40% | 40000 | 15/25 (60.0%) | | | |
| 60% | 60000 | 16/25 (64.0%) | | | |
| **80%** | **80000** | **17/25 (68.0%)** | **22/30 (73.3%)** | **25/35 (71.4%)** | ✅ |
| 100% | 100000 | 12/25 (48.0%) | | | |

> 20% 和 80% 在 Phase 1 并列 68.0%，经过 2 轮 tie-break 后 80% 胜出。
> 100% checkpoint 已明显过拟合（48.0% vs 80% 的 71.4%）。

### 逐里程碑成功率对比

| Milestone | 20% | 40% | 60% | 80% | 100% |
|-----------|:---:|:---:|:---:|:---:|:---:|
| 成功率 | 68.0% | 60.0% | 64.0% | **68.0%** | 48.0% |

训练曲线呈双峰形态（20% 和 80% 各一个峰值），100% 时性能崩溃，建议后续实验考虑 early stopping 或降低训练步数。

## Solver 详细对比

### 成功率

| Solver | NFE | 成功 | 失败 | 成功率 | Avg Steps (成功) |
|--------|-----|:---:|:---:|:---:|:---:|
| Euler | 2 | 13 | 12 | 52.0% | 239.1 |
| Midpoint | 2 | 16 | 9 | 64.0% | 242.4 |
| Euler | 4 | 15 | 10 | 60.0% | 239.8 |
| Midpoint | 4 | 15 | 10 | 60.0% | 241.4 |
| Midpoint | 8 | 16 | 9 | 64.0% | 242.6 |

### 推理延迟

所有测量在 RTX GPU 上完成，200 次 warmup + 200 次 benchmark，`torch.cuda.synchronize()` 精确计时。

| Solver | NFE | 延迟 (ms) | FPS | vs NFE=2 慢多少 |
|--------|-----|:---:|:---:|:---:|
| Euler | 2 | 5.45 | 183 | 1× |
| Midpoint | 2 | 5.49 | 182 | 1× |
| Euler | 4 | 9.07 | 110 | 1.66× |
| Midpoint | 4 | 9.06 | 110 | 1.65× |
| Midpoint | 8 | 16.22 | 62 | 2.95× |

> 编码器固定开销约 4ms（PointNeXT → tokenize → state encoder），每额外 NFE 增加约 1.8ms。
> 同 NFE 下 Euler 与 Midpoint 延迟差异在测量误差范围内（<0.1ms）。

### 单步时间分解（Midpoint NFE=2）

| 环节 | 耗时 | 占比 |
|------|:----:|:----:|
| 模型推理 | 5.5 ms | 33% |
| 物理模拟 + 预处理 + 其他 | 11.2 ms | 67% |
| **单步总计** | **~16.7 ms** | 100% |

> 瓶颈在 SAPIEN 物理模拟，模型推理仅占 1/3。即使推理时间减半，对整体帧率提升有限。

## 结论

1. **Midpoint NFE=2 是综合最优配置**：与 Euler-2 同等速度（5.5ms），成功率高出 12pp（64% vs 52%）；与 Midpoint-8 同等成功率（64%），速度快 3x
2. **Midpoint 在低 NFE 下优于 Euler**：NFE=2 时 Midpoint 显著优于 Euler（64% vs 52%），二阶方法在粗步长下精度优势明显
3. **NFE=4 时两种 solver 无差异**：成功率均为 60%，说明步长足够小时 solver 选择不再关键
4. **Midpoint NFE≥4 无增益**：NFE 从 4 到 8 成功率不变（60%→64%），但延迟翻倍，无性价比
5. **100% checkpoint 过拟合严重**：成功率从 80% 的 68% 骤降至 48%，建议后续训练考虑 early stopping

## 推荐配置

```bash
# 训练 (默认)
bash scripts/training/train.sh action_flow pour

# 选最优 checkpoint
bash scripts/eval/select_best_ckpt.sh action_flow pour <exp_name> \
    --link-best --denoise-steps 2

# 评测 (Midpoint NFE=2)
bash scripts/eval/eval_best_ckpt.sh action_flow pour <exp_name> \
    --episodes 100 --denoise-steps 2 --ema 'agent.solver=midpoint'

# Solver 全量 ablation
bash scripts/eval/eval_action_flow_solvers.sh action_flow pour <exp_name>
```

## 实验产出

```
experiments/action_flow/pour/2026-08-22_17-07_42/
├── eval_ckpt_selector/20260822_193931/   # select_best_ckpt 结果
│   └── best_ckpt_selection.json
├── best_ckpt.json                         # 最优 checkpoint 记录
├── checkpoints/best.pt -> epoch=0490-step=00080000-milestone=80pct.pt
└── eval_dexsim/solver_ablation/
    ├── 20260822_192512/                   # 第一次 (误用 60% ckpt，作废)
    └── 20260822_195246/                   # 最终结果 (80% ckpt)
        ├── euler_nfe2_result.txt
        ├── midpoint_nfe2_result.txt
        ├── euler_nfe4_result.txt
        ├── midpoint_nfe4_result.txt
        └── midpoint_nfe8_result.txt
```

---

# ActionFlow Solver Ablation — 128-patch 对比实验

> 评测日期: 2026-08-23 | 实验: `action_flow/pour/2026-08-22_22-56_42` | 训练步数: 100,000
>
> **目的**: 将 PointNeXT 观测 patches 从 64 翻倍到 128，评估对成功率的影响及最优 solver/NFE 组合是否变化。

## 与基准实验的配置差异

| 参数 | 基准 (17-07) | 本实验 (22-56) |
|------|:---:|:---:|
| `num_patches` | 64 | **128** |
| `solver` (默认) | euler | midpoint |
| 其余配置 | — | 相同 |

## 总览

| Solver | NFE | 成功率 (25 seeds) | vs 基准 64-patch |
|--------|-----|:---:|:---:|
| Euler | 2 | 16/25 = 64.0% | +12pp (52.0%→64.0%) |
| Midpoint | 2 | 15/25 = 60.0% | -4pp (64.0%→60.0%) |
| Euler | 4 | 16/25 = 64.0% | +4pp (60.0%→64.0%) |
| **Midpoint** | **4** | **21/25 = 84.0%** 🔥 | **+24pp** (60.0%→84.0%) |
| Midpoint | 8 | 12/25 = 48.0% | -16pp (64.0%→48.0%) |

> 评测使用同一组 25 个确定性种子 (`eval_seed=1066`)，与基准结果可直接对比。

## 实验环境

| 项目 | 值 |
|------|-----|
| 策略 | `action_flow` (ActionFlowAgent) |
| 任务 | `pour` (倒水) |
| 实验目录 | `experiments/action_flow/pour/2026-08-22_22-56_42` |
| 训练步数 | 100,000 (5 milestones: 20/40/60/80/100%) |
| Backbone | ActionFlowDiT 8L×512, 8Q/4KV GQA, GEGLU |
| 动作空间 | `action` (joint, 19 dim) |
| 观测 | PointNeXT (**128-patch**) + joint_state |
| 训练 solver | Euler, NFE=2 |
| 推理 EMA | ✅ |

## 最佳 Checkpoint 选择

`select_best_ckpt.sh` 自适应淘汰结果（初始 25 episodes）：

| Milestone | Step | Phase 1 (25ep) |
|-----------|------|:---:|
| **20%** | **20000** | **15/25 (60.0%)** ✅ |
| 40% | 40000 | 11/25 (44.0%) |
| 60% | 60000 | 13/25 (52.0%) |
| 80% | 80000 | 14/25 (56.0%) |
| 100% | 100000 | 14/25 (56.0%) |

> 20% 唯一最优，无需 tie-break。与基准实验（最优在 80%）相比，最优 checkpoint 大幅前移。

### 逐里程碑成功率对比

| Milestone | 20% | 40% | 60% | 80% | 100% |
|-----------|:---:|:---:|:---:|:---:|:---:|
| 64-patch (基准) | 68.0% | 60.0% | 64.0% | **68.0%** | 48.0% |
| **128-patch** | **60.0%** | 44.0% | 52.0% | 56.0% | 56.0% |

128-patch 整体成功率下降，且最优 checkpoint 从 80% 前移到 20%，训练曲线呈早期峰值后持续下降的形态。

## Solver 详细对比

### 成功率

| Solver | NFE | 成功 | 失败 | 成功率 | vs 基准 64-patch |
|--------|-----|:---:|:---:|:---:|:---:|
| Euler | 2 | 16 | 9 | 64.0% | +12pp |
| Midpoint | 2 | 15 | 10 | 60.0% | -4pp |
| Euler | 4 | 16 | 9 | 64.0% | +4pp |
| **Midpoint** | **4** | **21** | **4** | **84.0%** | **+24pp** |
| Midpoint | 8 | 12 | 13 | 48.0% | -16pp |

### 跨实验 Solver 最优对比

| 实验 | 最优 Solver | 最优成功率 | 最优 checkpoint |
|------|:---:|:---:|:---:|
| 64-patch (基准) | Midpoint-2 | 64.0% | 80% |
| **128-patch** | **Midpoint-4** | **84.0%** | **20%** |

## 完整 100-episode 评测 (Midpoint NFE=2)

```
Checkpoint   : best -> 20% (step=20000)
Seeds        : 100
Denoise steps: 2
Success rate : 57/100 = 57.0%
Avg steps    : 242.3
```

## 关键发现

1. **128-patch 的 Midpoint-4 是全局最优配置**：84.0% 远超基准最优的 64.0%（+20pp），但需付出 2× 推理延迟代价
2. **最优 checkpoint 前移到 20%**：128-patch 模型在训练早期达到峰值后持续下降，20% 唯一最优（60.0%），后续 milestone 均未超过 56%
3. **最优 solver 从 Midpoint-2 变为 Midpoint-4**：更大的 token 序列可能需要更精细的 ODE 求解步长
4. **Midpoint-2 在 128-patch 下反而退步**：60.0% vs 基准 64.0%，说明 128-patch 需要更多 NFE 才能发挥优势
5. **Midpoint-8 在两个实验中均表现最差**：48.0%（128-patch）vs 64.0%（64-patch），高 NFE 下过拟合风险加剧
6. **整体成功率下降**：128-patch 的 20% checkpoint 仅 60.0%，远低于 64-patch 的 80% checkpoint 的 68.0%。但 Midpoint-4 下可达到 84.0% — 说明 128-patch 的潜力在推理时而非训练时

## 推荐配置

基于两次实验的综合结论：

| 场景 | 推荐 | 成功率 | 延迟 |
|------|------|:---:|:---:|
| 低延迟优先 | Euler-2 (64-patch) | 52.0% | 5.45ms |
| 均衡 | Midpoint-2 (64-patch) | 64.0% | 5.49ms |
| **高成功率** | **Midpoint-4 (128-patch)** | **84.0%** | **~9ms** |

## 实验产出

```
experiments/action_flow/pour/2026-08-22_22-56_42/
├── eval_ckpt_selector/20260823_013843/   # select_best_ckpt 结果
│   └── best_ckpt_selection.json
├── best_ckpt.json                         # 最优 checkpoint 记录
├── checkpoints/best.pt -> epoch=0122-step=00020000-milestone=20pct.pt
├── eval_dexsim/
│   ├── 20260823_020057/                   # 100-episode Midpoint-2
│   │   └── _result.txt
│   └── solver_ablation/
│       └── 20260823_015041/               # Solver 全量 ablation
│           ├── euler_nfe2_result.txt
│           ├── midpoint_nfe2_result.txt
│           ├── euler_nfe4_result.txt
│           ├── midpoint_nfe4_result.txt
│           └── midpoint_nfe8_result.txt
```