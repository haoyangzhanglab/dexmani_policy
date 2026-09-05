# 策略对比: DP3 vs R3D vs ManiFlow

> 评测日期: 2026-08-08 ~ 2026-08-13 | 评测标准: 100-seed 最终成功率
>
> **注意**: 此文档中的 ManiFlow 指 `ManiFlowAgent`（DiTXFlowMatch + FlowMatchWithConsistency）。
## 总览

| 任务 | DP3 | R3D | ManiFlow | 最佳 |
|------|:---:|:---:|:---:|:---:|
| multi_grasp | 48.0% | **89.0%** | 85.0% | R3D |
| pour | 45.0% | **83.0%** | 66.0% | R3D |
| place_milk_box | 45.0% | **56.0%** | 43.0% | R3D |
| pick_apple_messy | 9.0% | 69.0% | **72.0%** | ManiFlow |
| peg_insertion | 8.0% | 7.0% | **14.0%** | ManiFlow |

## 实验详情

### DP3

| 任务 | 最终 SR | Best Ckpt | Select SR | 实验目录 |
|------|:---:|:---:|:---:|------|
| multi_grasp | 48.0% | 40% | 60.0%@25ep | `dp3/multi_grasp/2026-08-08_14-59_0` |
| pour | 45.0% | 80% | 54.3%@35ep | `dp3/pour/2026-08-08_15-02_0` |
| place_milk_box | 45.0% | 20% | 56.0%@25ep | `dp3/place_milk_box/2026-08-08_14-59_0` |
| pick_apple_messy | 9.0% | 60% | 12.3%@65ep | `dp3/pick_apple_messy/2026-08-08_20-25_0` |
| peg_insertion | 8.0% | 60% | 7.3%@55ep | `dp3/peg_insertion/2026-08-08_15-02_0` |

### R3D

| 任务 | 最终 SR | Best Ckpt | Select SR | 实验目录 |
|------|:---:|:---:|:---:|------|
| multi_grasp | 89.0% | 60% | 90.0%@60ep | `r3d/multi_grasp/2026-08-08_02-29_0` |
| pour | 83.0% | 40% | 90.0%@30ep | `r3d/pour/2026-08-06_20-58_0` |
| place_milk_box | 56.0% | 20% | 56.0%@25ep | `r3d/place_milk_box/2026-08-08_02-29_0` |
| pick_apple_messy | 69.0% | 40% | 72.0%@25ep | `r3d/pick_apple_messy/2026-08-07_14-16_0` |
| peg_insertion | 7.0% | 80% | 8.0%@25ep | `r3d/peg_insertion/2026-08-07_14-17_0` |

### ManiFlow

| 任务 | 最终 SR | Best Ckpt | Select SR | 实验目录 |
|------|:---:|:---:|:---:|------|
| multi_grasp | 85.0% | 40% | 86.7%@30ep | `maniflow/multi_grasp/2026-08-12_14-14_42` |
| pour | 66.0% | 20% | 76.0%@25ep | `maniflow/pour/2026-08-12_01-36_42` |
| place_milk_box | 43.0% | 100% | 36.0%@25ep | `maniflow/place_milk_box/2026-08-13_00-46_42` |
| pick_apple_messy | 72.0% | 60% | 84.0%@25ep | `maniflow/pick_apple_messy/2026-08-12_01-17_42` |
| peg_insertion | 14.0% | 100% | 28.0%@25ep | `maniflow/peg_insertion/2026-08-13_15-28_42` |

### 逐里程碑对比 (Select 阶段 SR)

**multi_grasp**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 56.0 | 86.7 | 83.3 |
| 40% | **60.0** | 88.3 | **86.7** |
| 60% | 40.0 | **90.0** | 83.3 |
| 80% | 40.0 | 85.7 | 84.0 |
| 100% | 36.0 | 88.0 | 84.0 |

**pour**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 52.0 | 80.0 | **76.0** |
| 40% | **56.0** | **90.0** | 64.0 |
| 60% | 51.4 | 88.0 | 64.0 |
| 80% | 54.3 | 80.0 | 56.0 |
| 100% | 52.0 | 86.7 | 56.0 |

**place_milk_box**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | **56.0** | **56.0** | 16.0 |
| 40% | 44.0 | 44.0 | 24.0 |
| 60% | 44.0 | 32.0 | 24.0 |
| 80% | 24.0 | 36.0 | 24.0 |
| 100% | 36.0 | 36.0 | **36.0** |

**pick_apple_messy**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 7.3 | 60.0 | 48.0 |
| 40% | 8.6 | **72.0** | 80.0 |
| 60% | 9.2 | 68.0 | **84.0** |
| 80% | 10.8 | 68.0 | 76.0 |
| 100% | 8.6 | 60.0 | 80.0 |

**peg_insertion**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 4.0 | 4.0 | 8.0 |
| 40% | 3.3 | 4.0 | 16.0 |
| 60% | **7.3** | 4.0 | 24.0 |
| 80% | 4.0 | **8.0** | 20.0 |
| 100% | 5.5 | 4.0 | **28.0** |

## 关键发现

1. **R3D 整体领先 (3/5 任务最佳)**: multi_grasp 89%、pour 83%、place_milk_box 56%，在需要空间推理与多阶段操作的任务上保持优势。

2. **ManiFlow 在 pick_apple_messy 与 peg_insertion 领先**: pick_apple_messy 72%（超 R3D 69%、DP3 9%）、peg_insertion 14%（略高于 DP3 8%、R3D 7%）。

3. **peg_insertion 全员极低 (7-14%)**: 4mm 间隙 + 10 阶段精插入管道的固有难度，详见[下文分析](#peg_insertion-成功率低原因分析)。ManiFlow 略高。

4. **ManiFlow 各任务达峰阶段不一**: pour 在 20% 即达 Select 峰值 76% 后持续回落至 56%，存在过拟合/高方差迹象；multi_grasp/pick_apple_messy 在 40-60% 达峰；place_milk_box/peg_insertion 在 100% 达峰。

5. **place_milk_box 三策略上限接近**: R3D 56% > DP3 45% > ManiFlow 43%。

## peg_insertion 成功率低原因分析

> 详见 2026-08-09 会话中的完整分析。此处仅列关键结论。

- **4mm 单边间隙** 是其他任务精度的 10-20 倍
- 100% 失败为超时 (320 步耗尽)，非掉落/碰撞——模型接近孔但无法完成精插入
- Diffusion/Flow Matching 的 action chunking (n_action_steps=8) + 去噪方差在亚厘米精度上失效
- 1024 pt 点云分辨率不足以区分 4mm 级偏差
