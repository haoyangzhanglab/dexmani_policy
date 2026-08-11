# 策略对比: DP3 vs R3D vs ManiFlow

> 评测日期: 2026-08-08 ~ 2026-08-09 | 评测标准: 100-seed 最终成功率
>
> **注意**: 此文档中的 ManiFlow 指 `ManiFlowAgent`（DiTXFlowMatch + FlowMatchWithConsistency）。
> 后续新增的 `StandardFlowMatch`（DiTDiffusion + 纯 FlowMatch，无 Consistency）是独立变体，未包含在此对比中。

## 总览

| 任务 | DP3 | R3D | ManiFlow | 最佳 |
|------|:---:|:---:|:---:|:---:|
| multi_grasp | 48.0% | 89.0% | **93.0%** | **ManiFlow** |
| pour | 45.0% | **83.0%** | 62.0% | R3D |
| place_milk_box | 45.0% | **56.0%** | **56.0%** | R3D / ManiFlow 平 |
| pick_apple_messy | 9.0% | **69.0%** | 21.0% | R3D |
| peg_insertion | 8.0% | 7.0% | **13.0%** | ManiFlow |

## ManiFlow 配置速查

> ⚠️ **后续可能变更**: RGB 输入、点云分辨率。当前配置以本节为准。

```yaml
# 数据模态 (当前: 仅点云，无 RGB)
sensor_modalities: ["joint_state", "point_cloud"]

# 点云
encoder_type: pointnet_dense         # 逐点 MLP，无 pooling
pc_dim: 3                            # xyz-only，无颜色通道
num_points: 256                      # 每帧 per-point token 数
fps_random:                          # FPS 下采样 + 随机增强
  use_random: true
  use_random_start: true
  use_shuffle_output: true

# PC Encoder (PointNetDense)
pc_encoder_config:
  out_channels: 128                  # 每点输出 128-d
  num_points: 256
  hidden_dims: [64, 128, 256]

# DiTX Backbone (Transformer)
n_layers: 12
hidden_dim: 768
n_head: 8
mlp_ratio: 4.0
p_drop_attn: 0.1                    # attention dropout
qkv_bias: false
qk_norm: false
pre_norm_modality: false

# Flow Matching (Rectified Flow + Consistency)
denoise_timesteps: 10
flow_batch_ratio: 0.75               # 75% flow loss, 25% consistency
t_sample_mode_for_flow: beta
t_sample_mode_for_consistency: discrete
dt_sample_mode_for_consistency: uniform
target_t_sample_mode: relative

# 训练
lr: 1.0e-4
weight_decay: 1.0e-3                 # DiT backbone: Transformer 标配高 wd
betas: [0.9, 0.95]                   # 低 β1，Flow Matching 惯例
obs_lr: 1.0e-4
obs_weight_decay: 1.0e-6             # PC encoder: 轻量正则化
batch_size: 128                      # 单卡
bfloat16: true
compile: true
ema:
  use_ema_teacher_for_consistency: true  # 推理用 target_t=dt>0
  inv_gamma: 1.0, power: 0.75

# 推理
eval.denoise_steps: 4                # 4 步 Euler ODE 推理

# 数据增强
pc.coord_noise: {noise_std: 0.002, prob: 1.0}
state.noise: {noise_std: 0.0002, prob: 1.0}

# 训练集: 80 episodes (max_train_episodes: 80)
# eval seed: 1024 (与训练 seed 42 隔离)
```

### 与其他策略的关键差异

| 维度 | DP3 | R3D | ManiFlow |
|------|-----|-----|----------|
| **点云分辨率** | 1024 pts | 1024 pts | **256 pts** |
| **PC 通道** | xyz+rgb (6) | xyz (3) | xyz (3) |
| **Encoder** | iDP3/PointNeXT | Uni3D (ViT+Fourier PE) | **PointNetDense** (per-point MLP) |
| **Backbone** | UNet1D (FiLM) | OneWayTransformer (cross-attn) | **DiTX** (cross-attn, 12L×768d) |
| **Decoder** | Diffusion (DDIM 10步) | Diffusion (DDIM 10步) | **Flow Matching** (Euler 4步) |
| **推理步数** | 10 | 10 | **4** |
| **weight_decay** | 1e-6 | 1e-6 | **1e-3** |
| **betas** | [.95,.999] | [.95,.999] | **[.9,.95]** |

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
| multi_grasp | 93.0% | 80% | 100.0%@25ep | `maniflow/multi_grasp/2026-08-08_22-21_0` |
| pour | 62.0% | 100% | 66.7%@30ep | `maniflow/pour/2026-08-08_20-30_0` |
| place_milk_box | 56.0% | 20% | 60.0%@25ep | `maniflow/place_milk_box/2026-08-08_22-43_0` |
| pick_apple_messy | 21.0% | 40% | 32.0%@25ep | `maniflow/pick_apple_messy/2026-08-08_22-21_0` |
| peg_insertion | 13.0% | 60% | 16.0%@25ep | `maniflow/peg_insertion/2026-08-08_22-43_0` |

### 逐里程碑对比 (Select 阶段 SR)

**multi_grasp**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 56.0 | 86.7 | 88.0 |
| 40% | **60.0** | 88.3 | 96.0 |
| 60% | 40.0 | **90.0** | 92.0 |
| 80% | 40.0 | 85.7 | **100.0** |
| 100% | 36.0 | 88.0 | — |

**pour**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 52.0 | 80.0 | **60.0** |
| 40% | **56.0** | **90.0** | 44.0 |
| 60% | 51.4 | 88.0 | 48.0 |
| 80% | 54.3 | 80.0 | 44.0 |
| 100% | 52.0 | 86.7 | **60.0** |

**place_milk_box**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | **56.0** | **56.0** | **60.0** |
| 40% | 44.0 | 44.0 | 52.0 |
| 60% | 44.0 | 32.0 | 56.0 |
| 80% | 24.0 | 36.0 | 56.0 |
| 100% | 36.0 | 36.0 | 56.0 |

**pick_apple_messy**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 7.3 | 60.0 | 24.0 |
| 40% | 8.6 | **72.0** | **32.0** |
| 60% | 9.2 | 68.0 | 20.0 |
| 80% | 10.8 | 68.0 | 20.0 |
| 100% | 8.6 | 60.0 | 28.0 |

**peg_insertion**

| Ckpt | DP3 | R3D | ManiFlow |
|------|:---:|:---:|:---:|
| 20% | 4.0 | 4.0 | 8.0 |
| 40% | 3.3 | 4.0 | 8.0 |
| 60% | **7.3** | 4.0 | **16.0** |
| 80% | 4.0 | **8.0** | 12.0 |
| 100% | 5.5 | 4.0 | 8.0 |

## 关键发现

1. **ManiFlow 在 multi_grasp (93%) 上最强**: 256pt 低分辨率点云 + Flow Matching 4 步推理在此简单抓取任务上表现最佳，且 select 阶段 100%。80% checkpoint 最佳说明未过拟合。

2. **R3D 在需要空间推理的任务上整体领先**: pour (+38pp vs ManiFlow)、pick_apple_messy (+48pp vs ManiFlow)。级联 self-attention mask + 分组 loss 对多阶段空间操作任务有明显优势。

3. **place_milk_box 三种策略天地上限接近 (~56%)**: 此任务对三种架构的难度相近，ManiFlow 在 20% checkpoint 即收敛到上限，后续不再提升。

4. **peg_insertion 全员极低 (7-13%)**: 4mm 间隙 + 10 阶段精插入管道的固有难度，详见[上文分析](#peg_insertion-成功率低原因分析)。ManiFlow 的 13% 略高可能归因于 4 步推理的更低方差。

5. **ManiFlow 收敛快**: place_milk_box 20% 即达峰值，multi_grasp 也是 20%=88%→80%=100%。Flow Matching + Consistency 训练可能加速了早期收敛。

## peg_insertion 成功率低原因分析

> 详见 2026-08-09 会话中的完整分析。此处仅列关键结论。

- **4mm 单边间隙** 是其他任务精度的 10-20 倍
- 100% 失败为超时 (320 步耗尽)，非掉落/碰撞——模型接近孔但无法完成精插入
- Diffusion/Flow Matching 的 action chunking (n_action_steps=8) + 去噪方差在亚厘米精度上失效
- 1024/256 pt 点云分辨率不足以区分 4mm 级偏差
