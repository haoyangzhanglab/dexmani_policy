# ActionFlow 架构与实验（权威文档）

> ActionFlow 是 DexMani_Policy 的一个 policy。本文件是 ActionFlow 的**唯一**权威文档，合并了历史架构沿革、当前最新架构、实验记录、结论与方法论。原「分阶段方案」「最新重构计划」「文档地图」「archive/」已并入本文档后删除。
> 更新日期：2026-08-31。工程约束 / 不变量 / 命令见 [CLAUDE.md](../CLAUDE.md)，评测机制见 [仿真评测机制.md](仿真评测机制.md)。

---

## 1. 当前架构（canonical）

### 1.1 数据流

```
PointNeXT (局部 patch tokenizer，192 patch，无全局 token)
  → GeoFormer (两帧 3D 几何关系建模，4L×576，3D RoPE)
    → static geometry memory [B, 385, 384]
      → ActionFlowDiT (8L×768，SA→CA→FFN，cross-attn 消费 memory)
        → SimpleRectifiedFlow (NoiseShift α=3，NFE=2 midpoint)
```

`joint_state` 只作 global modulation 进 ActionDiT，绝不进入几何 token。

### 1.2 组件与维度

| 组件 | 维度/结构 | 来源 |
|------|----------|------|
| 输入 | point_cloud [B·2, 1024, 6] + joint_state [B·2, 19] | — |
| **PointNeXT** | num_patches=192, stem=128, token=256, patch_radii=[0.04,0.08], patch_neighbors=[16,32]，无全局 token、无内部 patch self-attn | pointnext_tokenizer.py |
| **GeoFormer** | hidden=576, depth=4, heads=12 (head_dim=48), ffn=1536, qk_norm, 3D RoPE [0.02,2.0], attn_drop=0, drop_path=0, 末尾 norm_out=RMSNorm(eps=1e-5) | geoformer.py |
| **memory** | num_memory_tokens = 192×2+1 = **385**；memory_proj 576→384；abs_pe 96→384 | action_flow.py |
| **ActionFlowDiT** | hidden=768, **context=384**, depth=8, heads=12, ffn=1536, qk_norm, attn_drop=0；每 block SA→CA→FFN；12Q/12KV 全 CA（无 GQA）；SwiGLU；shared_modulation 384→9×768；**KV cache**（普通 python 属性） | action_flow_dit.py |
| **conditioning** | state(38) + timestep(128) → compact 384（**use_step_conditioning=false**，step_embedder 冻结） | — |
| **SimpleRectifiedFlow** | denoise_steps=2(NFE), solver=midpoint, noise_shift_alpha=3.0, ratio=0.75 | — |

### 1.3 参数量

| 档位 | context_dim | ffn_hidden_dim | perception | backbone | **total** |
|------|-------------|----------------|-----------|----------|-----------|
| legacy baseline（已弃） | 768 | 2048 | 17,109,984 | 80,051,603 | 97,161,587 (97.2M) |
| **当前 config（PR-11 后）** | 384 | 1536 | 16,851,168 | 58,806,675 | **75,657,843 (75.7M)** |

### 1.4 当前 config = 最优配方

实验已证伪全部正则方向，当前 config 回退到唯一正收益组合（原始维度见 §1.2）：

| 项 | 当前值 | 结论 |
|----|--------|------|
| xyz 归一化 | 逐轴 min-max（legacy） | metric xyz（PR-08）负收益，已删代码 |
| weight_decay / obs_weight_decay | 1e-3 / 1e-6 | PR-10a 正则负收益，回退 baseline |
| lr_scheduler | cosine | cosine_min_lr 负收益，回退 |
| attn_drop / geo_attn_drop / geo_drop_path / pc.dropout | 0 / 0 / 0 / prob 0 | PR-10b dropout 负收益，关闭 |
| use_step_conditioning | false | step_size 恒 0，gate 掉无意义偏置 |
| context_dim / ffn_hidden_dim | **384 / 1536** | PR-11 bandwidth 正收益（0.65），保留 |
| 3D RoPE 波长 | [0.02, 2.0]（归一化单位） | 随 metric 回退 |

---

## 2. 设计要点（为什么这样设计）

| 设计 | 要点 |
|------|------|
| **GeoFormer** | 两帧 patch 的 3D 几何关系推理；3D RoPE 相位只依赖 xyz（patch 位置），不依赖序列索引；pre-norm 后仍有 `norm_out`（RMSNorm）约束 residual stream 输出，否则 backward Jacobian 随权重尺度增长远快于 forward |
| **3D RoPE** | 波长 [0.02,2.0]（归一化单位）——base=10000 正弦频太粗，无法分辨 0.04-0.08 patch 半径，RoPE 会失效 |
| **KV cache** | static geometry memory 的 K/V 投影跨 NFE 迭代复用；是**普通 python 属性**（不能 register_buffer，否则进 state_dict，eval strict=True 加载会崩） |
| **零初始化** | shared_modulation / final_modulation / action_out 零初始化 → 初始输出精确为 0 |
| **state 不进几何** | joint_state 只作 global modulation，绝不 broadcast 到 geometry token |
| **`_rms_norm` eps=1e-5** | 同时决定近零行 backward 增益上限（1/√eps），1e-6→1000×，1e-5→316×，bf16 下是余量差别 |

> **实现独立性**：ActionFlow 三文件（`action_flow_flowmatch.py` / `action_flow_dit.py` / `geoformer.py`）完全独立，不共享 `time_sampler.py` / `flowmatch.py` / `ditx.py`；`cond` 是 dict。完整约定见 [CLAUDE.md](../CLAUDE.md)「设计约定」。

---

## 3. 实验记录

### 3.1 统一口径

- **任务**：pour（125 demos，`max_train_episodes=80`）。注：config 默认 `task_name=pick_apple_messy`（历史残留），pour 实验均通过 CLI `task_name=pour` 覆盖。
- **seed**：training.seed=42，eval_seed=1066
- **推理**：denoise_steps=2（NFE=2），solver=midpoint
- **评测**：`select_best_ckpt`（milestone 25ep 自适应淘汰）→ `eval_best_ckpt`（100ep 终值）
- **SR 口径**：success/total，Reset Failed 的 seed 记 failure；读 `result_details.json` 核 `denoise_steps`/`n_total`

### 3.2 基线

| 口径 | SR |
|------|-----|
| best ckpt（40%），**100ep NFE=2** | **0.68**（68/100） |
| milestone（25ep） | 20%=0.40, 40%=0.68, 60%=0.64, 80%=0.60, 100%=0.48 |

> ⚠️ 早期引用的「0.72」是 NFE=10/25ep 的 solver ablation 残留，**不是** NFE=2/100ep 基线。

### 3.3 各 PR 消融（100ep / NFE=2）

| 配置 | SR | vs 基线（§3.2 = 0.68） |
|------|-----|----------------------|
| PR-08 metric xyz（patch [0.010,0.020] 最优） | 0.58 | −10pt |
| PR-10a 全强度（cosine_min_lr + wd=1e-2 + obs_wd=1e-3） | 0.59 | −9pt |
| PR-10b dropout 全开 | 0.59 | −9pt |
| **PR-11 context 384/ffn 1536（baseline 正则）** | **0.65** | −3pt（~4pt 噪声内） |

### 3.4 PR-10a OFAT 单因子归因（Round-2，40k steps）

| label | 单变量 | SR |
|-------|--------|-----|
| pr11_ctx384_ffn1536 | context 384/ffn 1536（baseline 正则） | 0.65 |
| pr10a_wd_only | wd=1e-2 | 0.63 |
| pr10a_obswd_only | obs_wd=1e-3 | 0.60 |
| pr10a_sched_only | cosine_min_lr | 0.59 |

### 3.5 metric sweep（PR-08，40k）

use_metric_xyz=true：patch_radii 非单调 U 型（两端 ~0.57 高，中间 [0.015,0.030] 最差 0.42），RoPE 波长基本不敏感（spread 3pt）→ metric 切换未证明收益。

### 3.6 solver / NFE（0822）

midpoint NFE=2 ≈ NFE=10（G2≈0），NFE=2 已高效。

### 3.7 跨模型对照（pour，100-seed）

R3D 83%（历史最强）/ ManiFlow 66% / DP3 45%。

---

## 4. 结论与方法论

### 4.1 已得结论

1. **PR-11 compact 配方（75.7M）可行**：0.65，距基线 0.68 仅 −3pt（~4pt 噪声内）→ 砍 22% 参数几乎不损 SR。
2. **正则方向全部证伪**：wd / obs_wd / cosine_min_lr（PR-10a）与 dropout（PR-10b）都伤峰值、不修尾巴，单因子归因下 cosine_min_lr 最毒（0.59）。
3. **metric xyz（PR-08）负收益**：patch_radii 线性换算失效（U 型）、RoPE 不敏感 → 米制切换无收益，已删代码。
4. **40k→100k 过拟合无法用正则修复**（退化 0.68→0.48）。剩余可行方向：**扩数据**（80→125 demos）或**感知预训练**（vs-R3D #1 结构性天花板，估 5-7pt，ActionFlow 感知栈 17M 在 80 demo 从零训是硬伤）。

### 4.2 方法论

1. **NFE 口径必须对齐**：基线 `_result.txt` 会被 solver ablation（NFE=10/25ep）覆盖，读结果前先看 `result_details.json` 的 `denoise_steps`/`n_total`，别默认 100ep。
2. **25ep 不可信**：milestone selector 有 ~17pt 选择噪声，所有结论以 100ep 为准。
3. **run-to-run 方差 ~4pt**：<4pt 差异属噪声（源自 fps_random 随机点采样）。
4. **25ep vs 100ep 种子子集不同**：selector 与 eval 用不同 seed 子集，导致系统性偏差。

---

## 5. 附录：历史架构沿革

ActionFlow 经历了三代架构（均已废弃，其数字不可与当前并读）：

### 5.1 v0 — PointNeXT + DiT 8L×512（0822，已废弃）

- Backbone：`ActionFlowDiT` 8L×512，8Q/4KV **GQA**，GEGLU。
- 观测：PointNeXT（`pointnext.py` 的 `PointNextEncoder`，SetAbstraction 全局 token）64/128-patch + joint_state。
- **无 GeoFormer、无 3D RoPE、无 static geometry memory**。

### 5.2 vNext — DiT-X 升级尝试（0823-0824，已废弃）

- 基线 B0（commit main@bd6ca60）：4 SA + 4 CA 交错 block，512D context，32.44M，8Q/4KV GQA，GEGLU，zero-gated residual。
- 消融 B1-B4（Shared AdaRMS / 非对称 CA 等）+ F0-F2（flow 变体）。
- 任务 pick_apple_messy。**被 GeoFormer 架构取代**；其中 `Shared AdaRMS`、`非对称 CA`、`75-25 mixture sampler` 构想被当前架构吸收。

### 5.3 v2 — local tokenizer + GeoFormer + ActionFlowDiT（当前）

重构蓝图（原「最新重构计划」）落地：PointNeXT 收缩为**纯 local tokenizer**（关掉 global token 与内部 patch self-attn），几何关系推理交给 GeoFormer（两帧 joint 3D RoPE），产出 static memory，ActionFlowDiT 8L×768 消费 memory。PR-11 把 context/ffn 从 768/2048 压到 384/1536。
