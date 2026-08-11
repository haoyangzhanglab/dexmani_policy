# CLAUDE.md — DexMani_Policy

灵巧手操作模仿学习。Hydra 配置驱动，Zarr replay buffer，Diffusion/FlowMatch 动作解码，`dexmani_sim` 仿真评测。

> **Commands** · **Invariants** · **Agents** · **Config** · **Conventions** · **File Map** · **Skills** · **[Docs](docs/)**

## 环境

```bash
conda activate policy
export DATA_DIR=/path/to/data          # 必须，否则 dataset 路径报错
```

## 命令速查

### 训练

```bash
# 单卡: train.sh <policy> <task> [Hydra覆盖...]
bash scripts/training/train.sh dp3 pour
bash scripts/training/train.sh dp3 pour 'training.seed=42'

# 多卡 DDP: train_ddp.sh ddp/<policy> <task>
bash scripts/training/train_ddp.sh ddp/maniflow pour
```

> 单卡 (10): `dp dp3 dp3_faas dqrise maniflow moe_dp multitask_dit r3d sat standard_flowmatch`
> DDP (8): `ddp/dp ddp/dp3_faas ddp/dqrise ddp/maniflow ddp/multitask_dit ddp/r3d ddp/sat ddp/standard_flowmatch`
> 全部 `total_train_steps: 100000`

### 评测

```bash
# 一键管道: select best ckpt → eval all seeds
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_dir>
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_dir> --no-videos

# 分步
bash scripts/eval/select_best_ckpt.sh dp3 pour <exp_dir>
bash scripts/eval/eval_best_ckpt.sh dp3 pour <exp_dir> --ckpt-tag 40pct --episodes 50
```

### Demo 录制

```bash
bash scripts/eval/record_demo.sh dp3 pour <exp_dir>
bash scripts/eval/record_demo.sh sat pour <exp_dir> --ckpt-tag 100pct --seeds 5 12 33
bash scripts/eval/record_demo.sh maniflow pour <exp_dir> --no-ema --resolution 3840 2160
```

> 需 X11/Wayland。视频 → `experiments/<policy>/<task>/<exp>/demo_videos/<timestamp>/`。
> `--seeds` 覆盖 `--episodes`。

### 验证

```bash
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py dp3 maniflow sat
```

---

## 核心不变量 (碰了就炸)

| 常量 | 值 | 位置 |
|------|-----|------|
| `horizon` / `n_obs_steps` / `n_action_steps` | **16 / 2 / 8** | 所有 config |
| 优化器 | `AdamW(fused=True)` | `base.py:238` |
| DDIM scheduler | `beta_start=0.0001, beta_end=0.02, schedule='squaredcos_cap_v2'` | `diffusion.py:39` |
| StateMLP hidden | `[64]` | `state_mlp.py` |
| ViT backbone dtype | `bfloat16 + attn_implementation="sdpa"` | dino/clip/siglip |
| UNet conditioning | `cond_predict_scale=True` | `unet1d.py` |
| FlowMatch consistency 权重 | `1.0` (implicit) | `flowmatch.py:234` |
| FAAS 活跃索引 | `(1,2,3,6,7,8,12,13,17,18,22,23)`, 仅 index_bend scale=-1.0 | `faas_mapper.py` |

---

## 架构速览

### 数据流 (5 秒心智模型)

```
Config (Hydra YAML) → Dataset (Zarr→ReplayBuffer→Sampler→__getitem__)
  → Agent (obs_encoder → backbone → action_decoder)
    → Trainer (loss → backward → grad_accum → EMA → checkpoint)
      → eval_best_ckpt.py (env_runner.run → success_rate)
```

### Agent 类层级

```
BaseAgent
  ├── UNetDiffusionAgent        ← DP, DP3, MoE (UNet1D + Diffusion)
  ├── DiTXFlowMatchAgent        ← ManiFlow (DiTX + FlowMatchWithConsistency)
  ├── StandardFlowMatchAgent    ← Standard FlowMatch (DiT + StandardFlowMatch, 无 consistency)
  ├── SATAgent                  ← SAT (SATBackbone + SATFlowMatch)
  ├── MultiTaskAgent            ← MultiTask (DiT + Diffusion/FlowMatch)
  ├── R3DAgent                  ← R3D (OneWayTransformer + Diffusion)
  └── DQRISEAgent               ← DQ-RISE (自定义 UNet + Diffusion, action_dim 缩减)
```

### Agent 继承模式 (添加新 Agent 时选)

| 模式 | 父类 | 何时用 | 最近参考 |
|------|------|--------|---------|
| **A: UNet+Diffusion** | `UNetDiffusionAgent` | Flat encoding → UNet1D(FiLM) → Diffusion | `dp3.py` |
| **B: DiTX+FlowMatch+Consistency** | `DiTXFlowMatchAgent` | Token seq → DiTX(cross-attn) → FlowMatch+consistency | `maniflow.py` |
| **C: DiT+FlowMatch** | `StandardFlowMatchAgent` | Token seq → DiT(self-attn) → FlowMatch (no consistency) | — |
| **D: Fully custom** | `BaseAgent` | 完全自定义 backbone + decoder | `sat.py`, `r3d.py` |

### Agent 对比 (找最接近的参考实现)

| Agent | 输入 | Encoder | Backbone | Decoder | 独特点 |
|-------|------|---------|----------|---------|--------|
| **DP3** | PC+state | PointNeXT+StateMLP | UNet1D(FiLM) | Diffusion(DDIM 10步) | **最简参考**, pc_dim=3 |
| **DP** | RGB+state | DINO/CLIP/SigLIP+StateMLP | UNet1D(FiLM) | Diffusion(DDIM 10步) | RGB, channels_last |
| **ManiFlow** | PC+state | PointNeXT+StateMLP | DiTX(cross-attn) | FlowMatch+Consistency | Token条件, EMA教师, wd=1e-3 |
| **StandardFlowMatch** | PC+state | PointNeXT+StateMLP | DiT(self-attn) | StandardFlowMatch | 纯 flow matching, 无 consistency |
| **MoE** | RGB+state | R3M+StateMLP+MoE | UNet1D(FiLM) | Diffusion(DDPM 100步) | 16专家top-2, **无bfloat16/compile** |
| **SAT** | PC+state | PointNeXT+StateMLP | SATBackbone(EJC+MMAttn) | SATFlowMatch | (B,Da,T), shuffle, compile=default |
| **R3D** | PC+state | Uni3D+StateMLP | OneWayTransformer | Diffusion(DDIM 10步) | 级联mask, 分组loss |
| **DQRISE** | PC+state | iDP3+StateMLP | UNet1D(tcp+1维) | Diffusion(epsilon,20步) | VQ码本, lr=3e-4, warmup=2000 |
| **MultiTask** | RGB+state+text | DPObsEncoder+CLIP | DiT(AdaLN) | Diffusion/FlowMatch | 多任务, 预缓存text |
| **DP3(FAAS)** | PC+state | 同DP3 | 同DP3 | 同DP3 | 32D FAAS, **零 agent 代码变更** |

### 动作空间

| `action_key` | arm | hand | total | FAAS total |
|-------------|-----|------|-------|-----------|
| `action` (joint) | 7 (关节角) | 12 (XHand) | **19** | 39 (7+32) |
| `action_ee` (ee) | 9 (pos3+rot6d) | 12 (XHand) | **21** | 41 (9+32) |

`joint_state` dim ≡ action dim。FAAS 通过 `inject_faas_into_agent()` 对 DP/DP3/ManiFlow/MoE/MultiTask/R3D/SAT 零代码变更兼容。`use_aux_ee` 与 `use_faas` 互斥。

### Action Decoder

| Decoder | 预测目标 | 推理 | 谁用 |
|---------|---------|------|------|
| `Diffusion` | ε / x0 / v | DDIM 迭代 | DP, DP3, MoE, R3D, DQRISE |
| `FlowMatch` / `SATFlowMatch` | v=x1-x0 | Euler ODE | MultiTask, SAT |
| `FlowMatchWithConsistency` | v + consistency(EMA教师) | Euler ODE | ManiFlow |
| `StandardFlowMatch` | v (target_t=0) | Euler ODE | StandardFlowMatch |

---

## 配置速查

### 关键参数 (跨策略差异)

| 参数 | dp3 | maniflow | standard_flowmatch | moe_dp | r3d | dqrise | sat |
|------|-----|----------|-------------------|--------|-----|--------|-----|
| action_dim | 19/21 | 19/21 | 19/21 | 19/21 | 19/28 | 21 | 19/21 |
| backbone | UNet[256,512,1024] | DiTX 12L×768 | DiT 12L×768 | UNet[256,512,1024] | 4L×256 | UNet[256,512] | SAT 8L×768 |
| diff train/infer | 100/10 | -/4 | -/10 | 100/100 | 100/10 | 100/20 | -/10 |
| prediction_type | sample | velocity | velocity | sample | sample | epsilon | velocity |
| lr / wd | 1e-4 / 1e-6 | 1e-4 / **1e-3** | 1e-4 / **1e-3** | 1e-4 / 1e-6 | 1e-4 / 1e-6 | **3e-4** / 1e-6 | 1e-4 / 1e-6 |
| betas | [.95,.999] | **[.9,.95]** | **[.9,.95]** | [.95,.999] | [.95,.999] | [.95,.999] | [.95,.999] |
| bfloat16 / compile | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **✗ / ✗** | ✓ / ✓ | ✓ / ✓ | ✓ / ✓(default) |
| val_ratio | 0.05 | 0.05 | 0.05 | 0.10 | 0.05 | 0.05 | 0.05 |

> dp = dp3 参数; dp3_faas = dp3 参数 + FAAS 维度 (action_dim=39/41, state_dim=39); multitask_dit = 8L×512, lr=1e-4, val=0.05
> 全部 `total_train_steps: 100000`, `warmup: 500` (dqrise: 2000)

`action_dim` 公式: `${eval:'21 if ${eq:${action_key},action_ee} else 19'}` (FAAS: 39/41)
`agent._target_`: `dexmani_policy.agents.core.<name>.<Name>Agent` (Hydra 直接导入，无显式注册表)
Eval 所有策略共享 `denoise_steps=10, use_ema=true`；参数优先级 CLI > 子节 > eval 共享层。

> Config 模板字段清单、Eval YAML 结构 → [README](README.md#配置参考)

---

## 训练内幕

**NaN 两层防护**: L1 (backward前, loss NaN → 保存 debug ckpt → raise) / L2 (optimizer.step前, 梯度 NaN → zero_grad → raise)。诊断用 `dexmani-training-debug` skill。

**Checkpoint**: 20/40/60/80/100% 里程碑; `latest.pt` symlink; 自动 resume。**DDP**: ckpt 加载在 compile + DDP 包装**前**; timeout=30min; `dp3`/`moe_dp` 仅单卡。**Shape 验证**: `_validate_batch()` 在 `compute_loss`/`predict_action` 入口。

> 详细机制 → [README](README.md#训练机制)

---

## 设计约定 (不是 Bug，不要修)

- **Normalizer 全量拟合**: 用全部 replay buffer (含验证集)。生态惯例 (ManiFlow/R3D/SAT/RoboTwin 均如此)。`limits` 模式下 val 不影响 min/max
- **`tcp_dim` 命名**: joint模式=7(臂关节), ee模式=9(TCP位姿) — 历史命名，勿据此推断语义
- **MoE forward 返回 `dict`** (含 `aux_loss`): `BaseAgent.compute_loss()` 统一处理 dict/Tensor
- **MoE 无 bfloat16/compile**: gate softmax 需 float32; CUDA Graphs + MoE routing 内存开销大。**不要重新启用**
- **DQRISE 直接继承 `BaseAgent`**: `diffusion_action_dim = tcp_dim+1` ≠ `action_dim`，无法复用 UNetDiffusionAgent。**不要重构**
- **R3DObsEncoder 拼接**: patch_tokens + state_emb + pc_pe 沿 feature 维 (非 `torch.cat`)
- **EMAModel BatchNorm**: affine 参数直接复制，不 EMA 平均
- **FlowMatchWithConsistency `target_t`**: 训练=0, 推理=dt>0。StandardFlowMatch 无此机制
- **DDP 覆盖不全**: `dp3`(非FAAS) 和 `moe_dp` 有意仅单卡
- **Milestone checkpoint**: 仅 20/40/60/80/100% 五个; `latest.pt` 是 symlink

### 未启用功能 (不要意外激活)

| 功能 | 位置 | 状态 |
|------|------|------|
| Modality Dropout | `base.py` | 全配置 `modality_dropout_probs=0.0` |
| TokenCompressor | `obs_encoder/plugins/` | 未接入任何 config |
| T5TextEncoder | `obs_encoder/text/t5.py` | 预留代码 |

---

## 文件地图

```
dexmani_policy/
  train.py                    # 单卡入口 (@hydra.main)
  train_ddp.py                # DDP 入口 (mp.spawn)
  eval_best_ckpt.py           # 离线评测 (Hydra-free CLI)
  smoke_test.py               # 构建验证 (6 阶段)
  configs/                    # 10 YAML + 8 DDP overlay
  agents/
    core/                     # BaseAgent + 9 variants
    action_decoders/          # Diffusion, FlowMatch variants
      backbone/               # UNet1D, DiT, DiTX, OneWayTransformer, SATBackbone
    obs_encoder/              # pointcloud/, rgb/, text/, state_mlp.py, plugins/
    vq_hand/                  # DQ-RISE: VQVAEHand, CodebookManager, ResidualVQ
  datasets/                   # BaseDataset, PCDataset, ReplayBuffer, SequenceSampler
  training/                   # Trainer, build_utils, ema, logging, lr_scheduler, eval_utils
  env_runner/                 # BaseRunner, SimRunner, MultiTaskSimRunner
  common/                     # LinearNormalizer, FAASMapper, checkpoint_io, pytorch_util
  tools/                      # train_vq_hand.py, extract_codebook.py
```

### 找最近参考 (Vibe Coding)

| 你要做什么 | 从哪开始 |
|-----------|---------|
| 加 UNet+Diffusion agent | `dp3.py` + `dp3.yaml` |
| 加 Transformer+FlowMatch agent | `maniflow.py` + `maniflow.yaml` |
| 加完全自定义 backbone agent | `sat.py` + `sat.yaml` |
| 加 FAAS 变体 | `dp3_faas.yaml` (继承 dp3，仅覆盖维度) |
| 加 DDP overlay | `configs/ddp/maniflow.yaml` |
| 改数据增强 | `datasets/base_dataset.py` → `apply_augmentation()` |
| 改评测逻辑 | `env_runner/sim_runner.py` |
| 改训练循环 | `training/trainer.py` |
| 加新观测模态 | `agents/obs_encoder/` → 对应子目录 |
| 改 normalizer | `common/normalizer.py` → `LinearNormalizer` |

---

## Agent Skills

通过 `Skill` 工具调用。源文件在 `.agents/skills/`。

| Skill | 类型 | 何时用 |
|-------|------|--------|
| `dexmani-agent-integration` | Build | 添加新 Agent 变体 (3-6 文件 checklist) |
| `dexmani-pr-check` | Audit | Pre-PR 审计 (config 不变量/维度链/DDP覆盖/CLAUDE.md一致性) |
| `dexmani-training-debug` | Diagnostic | 训练 NaN 事后诊断 (两层防护 triage + 根因表) |

**Skill vs CLAUDE.md**: 过程性工作流 (多步骤、需 checklist) → Skill；声明性知识 (命令、不变量、约定) → 本文。

---

## 外部文档

| 文档 | 内容 |
|------|------|
| `README.md` | 项目概览、快速开始、策略矩阵、动作空间、配置参考、训练机制、设计约定、FAQ |
| `docs/项目架构.md` | 完整目录树、模块依赖图、类层级、数据流全景 |
| `docs/仿真评测机制.md` | 评测全链路 — CLI → EnvRunner → Agent 推理 → Decoder 去噪 |
| `docs/SSH服务器训练部署.md` | 远程训练部署 + SSH 常识附录 |
| `docs/DP3-R3D-ManiFlow测试结果0808.md` | DP3 vs R3D vs ManiFlow 五项任务对比评测 |
