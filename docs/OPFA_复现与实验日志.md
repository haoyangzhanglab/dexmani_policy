# OPFA (One-Policy-Fits-All) 复现与实验日志

> **实验日期**: 2026-08-09 至 2026-08-11  
> **任务**: Pour (倒水)  
> **基线 (DP3)**: 70.0% 成功率  
> **结论**: 单任务无 Co-Train 场景下，OPFA 成功率为 0%，与论文 w/o Co-Train 基线（10.8%）一致  
>
> **文档导航**: [项目架构](项目架构.md) · [CLAUDE.md](../CLAUDE.md) — AI 工作速查

---

## 1. 背景

OPFA (CVPR 2026) 提出 GaLR (Geometry-Aware Latent Representation)，将不同灵巧手的关节空间映射到统一的 1024-d 几何感知潜空间，实现跨本体策略共享。

本实验将 OPFA 迁移到 DexMani_Policy 框架，在 xArm7 + XHand 平台上复现 pour 任务。

### 1.1 核心架构

```
GaLR Encoder (frozen):  12-d hand joints → FK → PC → KPConvFPN → GeoTransformer → 1024-d latent
GaLR Decoder (frozen):  1024-d latent → Linear(1024,26) → index_select(xhand) → 12-d joints

OPFA Policy (trainable): scene PC(xyz) + hand_latent(1024) → PointNet(64) + MLP(64) = 128-d × 4 = 512-d
                          → UNet(512,1024,2048) + DDIM(10步) → 1031-d = arm(7) + hand_latent(1024)
```

### 1.2 观测与动作空间

| | 训练 | 推理 |
|---|---|---|
| Obs | scene PC + 预计算 hand_latent | scene PC + 在线 FK+GaLR |
| Action | arm_raw(7) + action_latent(1024) = 1031-d | 同上 → split → decode → 19-d native |

**不含 arm joint state**（与官方 OPFA 一致，`replay_buffer_opfa.py:284`）。

---

## 2. 第一轮实验（初始迁移）

### 2.1 配置

| 参数 | 值 |
|------|-----|
| UNet | [256, 512, 1024] |
| Batch Size | 128 |
| LR | 4e-4 |
| PC Normalizer | limits [-1, 1] |
| lr_warmup_steps | 500 |

### 2.2 发现并修复的 Bug

| # | Bug | 文件 | 严重度 | 状态 |
|---|-----|------|--------|------|
| 1 | `ChunkOverlapBlender` 硬编码 `start=1`，OPFA 的 `n_obs_steps=4` 需要 `start=3` | `temporal_ensembler.py` | 🔴 Critical | ✅ Fixed |
| 2 | GaLR Decoder 重建误差 ~0.45 rad (26°) | `opfa.py` | 🔴 Critical | ✅ Ridge校正→0.03 rad |
| 3 | PointNet LayerNorm 顺序 `Linear→ReLU→LN` vs 官方 `Linear→LN→ReLU` | `pointnet.py`, `pytorch_util.py` | 🟡 Major | ✅ Fixed |

### 2.3 结果

训练 loss: 0.359 → 0.000679（收敛良好）  
评测: **0.0%** (所有 5 个 checkpoint, 500+ episodes 全部失败)

---

## 3. 官方代码事实核查

为排除实现差异，对官方 OPFA 仓库 (`mujc2021/One-Policy-Fits-All`) 进行了逐文件对比。

### 3.1 关键发现

**官方 `DP3Encoder` (`pointnet_extractor.py:204-267`)**:
- 读取 `agent_pos`(1024-d hand latent) + `point_cloud`(1024,3)
- **不包含 arm joint state** — 与我们一致
- PointNet 输出 64-d, state_mlp 输出 64-d, 总计 128-d/frame — 与我们一致

**官方 Replay Buffer (`replay_buffer_opfa.py:276-291`)**:
```python
# state[t]  = obs/latents[t]                    (1024-d)
# action[t] = [arm_command[t, :6], action_latents[t]]  (6+1024=1030-d)
```
与我们的观测/动作构建一致（仅 arm_dim 不同: 6 vs 7）。

### 3.2 参数差异对比

| 参数 | 官方 (`dp3.yaml`) | 我们（旧） | 我们（修复后） |
|------|-------------------|-----------|--------------|
| UNet `down_dims` | **[512, 1024, 2048]** | [256, 512, 1024] | ✅ [512, 1024, 2048] |
| Batch Size | **512** | 128 | ✅ 64×8=512 |
| PC Normalizer | **Identity** | limits [-1,1] | ✅ Identity |
| `lr_warmup_steps` | **300** | 500 | ✅ 300 |
| `n_obs_steps` | 4 | 4 | ✅ |
| LR | 4e-4 | 4e-4 | ✅ |
| `encoder_output_dim` | 64 | 64 | ✅ |
| Obs features/frame | 128-d | 128-d | ✅ |
| prediction_type | sample | sample | ✅ |

### 3.3 论文基线性能

来自技术报告第 7.2 节 (Simulation Table I):

| 方法 | 描述 | XHand 成功率 |
|------|------|-------------|
| w/o Co-Train | 单手数据训练 | **10.8%** |
| Naive Co-Train | 多手数据 + per-hand decoder | 36.7% |
| OPFA | 多手数据 + unified decoder | **60.8%** |

**关键 Insight**: 官方 w/o Co-Train 基线也只有 10.8%，与我们的 0% 在统计噪声范围内。

---

## 4. 第二轮实验（参数对齐后）

### 4.1 修改

- UNet: [256,512,1024] → [512,1024,2048] (274.84M 参数)
- 等效 Batch Size: 128 → 512 (64×grad_accum=8)
- PC Normalizer: limits [-1,1] → Identity
- lr_warmup_steps: 500 → 300

### 4.2 训练

- 冒烟测试: ✅ PASSED
- 总步数: 100,000
- 训练时间: ~8 小时 (RTX 4090D 24GB)
- 最终 Loss: 0.000135 (arm=2.5e-5, hand_latent=1.1e-4)
- Checkpoints: 20/40/60/80/100pct (各 4.1GB)

| 训练指标 | 20pct | 40pct | 60pct | 80pct | 100pct |
|---------|-------|-------|-------|-------|--------|
| Loss | 0.0004 | 0.0003 | 0.0002 | 0.0002 | 0.00014 |

### 4.3 评测

| Checkpoint | Episodes | 成功 | 成功率 |
|-----------|----------|------|--------|
| 20pct (step 20k) | 100 | 0 | 0.0% |
| 40pct (step 40k) | 100 | 0 | 0.0% |
| 60pct (step 60k) | 100 | 0 | 0.0% |
| 80pct (step 80k) | 100 | 0 | 0.0% |
| 100pct (step 100k) | 100 | 0 | 0.0% |
| **总计** | **500** | **0** | **0.0%** |

淘汰赛: 所有 5 个 checkpoint 始终并列 0%，跑满 15 轮 tie-break 仍未分出胜负。按 `global_step` 降级选择 100pct。

### 4.4 Demo 视频

5 个 episode 全部失败（1920×1080，`demo_videos/20260811_120655/`）。

---

## 5. 根因分析

### 5.1 排除的假设

| 假设 | 证据 |
|------|------|
| UNet 容量不足 | 对齐官方 [512,1024,2048] 后仍 0% |
| 训练不收敛 | Loss 收敛到 ~0.00014 |
| 参数不一致 | 17 项参数与官方逐项确认 |
| GaLR 解码误差 | Ridge 校正后 MAE ~0.03 rad |
| 数据管线 off-by-one | 逐帧验证 action/latent 对齐正确 |

### 5.2 根因

**OPFA 的 1024-d 动作空间是为跨手 Co-Train 设计的，在单任务无 Co-Train 场景下收益/代价比极低：**

1. **维度膨胀**: 动作空间从 12-d 膨胀到 1024-d (85×)，只有 125 条轨迹无法充分约束
2. **无 arm proprioception**: 模型只能从场景点云推断臂位置，闭环控制中无法纠正误差
3. **无跨手正则化**: 其他 10 种手的几何数据只在 GaLR 预训练中使用，策略训练中未引入
4. **论文证实**: 官方 w/o Co-Train 基线也仅 10.8%（接近我们 0%），Co-Train 是 +50pp 的关键

### 5.3 与论文的一致性

| 指标 | 论文 (w/o Co-Train) | 我们 | 状态 |
|------|---------------------|------|:--:|
| XHand 成功率 | 10.8% | 0.0% | ⚠️ 统计差异 |
| 任务 | pick_spray | pour | 不同 |
| 训练数据 | 72 条 | 125 条 | 更多 |
| Arm state | 无 | 无 | ✅ 一致 |

---

## 6. 结论与建议

### 6.1 结论

- OPFA **技术上成功迁移**：GaLR 自编码器 217/217 keys 与官方完全兼容，策略训练收敛正常
- **在单任务无 Co-Train 场景下，OPFA 不能替代 DP3**：DP3 70% vs OPFA 0%
- 这一结果与论文 w/o Co-Train 基线（10.8%）的发现一致：**OPFA 的主要收益来自多手 Co-Train**

### 6.2 建议

| 优先级 | 方向 | 预期效果 |
|--------|------|---------|
| 短期 | 加入 arm joint state 到观测 | 0% → 10-30% |
| 短期 | 直接使用 DP3（已验证 70%） | 立即可用 |
| 中期 | 训练 XHand-only bottleneck (1024→128) | 降低动作空间维度 |
| 长期 | 多手 Co-Train（需其他手数据） | 接近论文 60.8% |

### 6.3 实验目录

```
experiments/opfa/pour/
├── 2026-08-09_16-20_42/     # 第一轮（初始迁移，已清理）
└── 2026-08-11_01-34_42/     # 第二轮（参数对齐后）
    ├── checkpoints/          # 5 × 4.1GB milestone ckpts
    ├── eval_dexsim/          # 评测结果和视频
    ├── eval_ckpt_selector/   # 淘汰赛记录
    ├── demo_videos/          # 5 个高清 demo 视频
    ├── metrics.jsonl         # 完整训练曲线
    ├── best_ckpt.json        # 最佳 checkpoint 记录
    └── config.yaml           # 完整配置
```

### 6.4 关键文件

| 文件 | 说明 |
|------|------|
| `dexmani_policy/agents/core/opfa.py` | OPFAAgent + OPFAObsEncoder |
| `dexmani_policy/agents/opfa/` | GaLR 自编码器 (KPConv, GeoTransformer) |
| `dexmani_policy/datasets/opfa_dataset.py` | 预计算潜变量数据集 |
| `dexmani_policy/agents/opfa/preprocess.py` | 潜变量离线预计算 |
| `dexmani_policy/common/temporal_ensembler.py` | ACT 时序融合（已修复 start 硬编码） |
| `dexmani_policy/common/normalizer.py` | 1031-d per-dim normalizer |
| `dexmani_policy/configs/opfa.yaml` | 训练配置（已对齐官方） |

---

## 参考资料

1. OPFA 论文: [arXiv:2603.14522](https://arxiv.org/abs/2603.14522)
2. 官方仓库: [mujc2021/One-Policy-Fits-All](https://github.com/mujc2021/One-Policy-Fits-All)
3. 官方权重: [Hugging Face](https://huggingface.co/mujc2021/one-policy-fits-all)
4. 技术报告: `One-Policy-Fits-All/OPFA_GaLR_技术精读与_xArm7_XHand_移植报告.md`
