# CLAUDE.md — DexMani_Policy

灵巧手操作模仿学习。Hydra 配置驱动，Zarr replay buffer，Diffusion/FlowMatch 动作解码，`dexmani_sim` 仿真评测。

> **Commands** · **Invariants** · **Agents** · **Config** · **Conventions** · **File Map** · **Skills** · **[Docs](docs/)**

## 环境

```bash
conda activate policy
# 训练数据从 robot_data/<task>.zarr 读取（zarr_path 相对仓库根目录，运行时 chdir 到根目录）。
# 评测种子由已安装的 dexmani_sim 包提供（其内部 DATA_DIR 为包内常量，非环境变量）。
```

## 命令速查

### 训练

```bash
# 单卡: train.sh <config_name> [Hydra覆盖...]（task 用 task_name= 覆盖，无独立位置参数）
bash scripts/training/train.sh dp3 'task_name=pour'
bash scripts/training/train.sh action_flow 'task_name=pour'
bash scripts/training/train.sh dp3 'task_name=pour' 'training.seed=42'

# 多卡 DDP: train_ddp.sh ddp/<config_name> [Hydra覆盖...]
bash scripts/training/train_ddp.sh ddp/maniflow 'task_name=pour'
```

> 单卡 (8): `action_flow dp dp3 dqrise maniflow multitask_dit r3d sat`
> DDP (7): `ddp/action_flow ddp/dp ddp/dqrise ddp/maniflow ddp/multitask_dit ddp/r3d ddp/sat`
> 全部 `total_train_steps: 100000`

### 评测

```bash
# 一键管道: select best ckpt → eval all seeds
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_name>
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_name> --no-videos

# 分步
bash scripts/eval/select_best_ckpt.sh dp3 pour <exp_name>
bash scripts/eval/eval_best_ckpt.sh dp3 pour <exp_name> --ckpt-tag 40pct --episodes 50

# ActionFlow solver paired ablation（Euler1/Mid2/Mid4/Mid8/Mid10）
bash scripts/eval/eval_action_flow_solvers.sh action_flow pour <exp_name> --episodes 25
```

ActionFlow 的 `denoise_steps` 即 NFE：Euler 支持任意正整数（包括 1 和 10），Midpoint 只支持偶数。
`<exp_name>` = `experiments/<policy>/<task>/` 下的时间戳/名称目录（非完整路径）。
需要单独评测其他 NFE 时，使用 `eval_best_ckpt.sh --denoise-steps N`。

NFE 判据（重构计划 §19）：`G2 = SR10 - SR2 ≤ 3~5%`，`R2 = (SR2-SR1)/(SR10-SR1) ≥ 0.75`。
仅当 `SR10 - SR2 > 5%` 才启动 FlexRF。

### Demo 录制

```bash
bash scripts/eval/record_demo.sh dp3 pour <exp_name>
bash scripts/eval/record_demo.sh sat pour <exp_name> --ckpt-tag 100pct --seeds 5 12 33
bash scripts/eval/record_demo.sh maniflow pour <exp_name> --no-ema --resolution 3840 2160
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
| 优化器 | `AdamW(fused=torch.cuda.is_available())` | `base.py:297` |
| DDIM scheduler | `beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2'` | `diffusion.py:43` |
| StateMLP hidden | `[64]` | `state_mlp.py` |
| ViT backbone dtype | `bfloat16 + attn_implementation="sdpa"` | dino/clip/siglip |
| UNet conditioning | `cond_predict_scale=True` | `unet1d.py` |
| FlowMatch consistency 权重 | `1.0` (implicit) | `flowmatch.py:236` |

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
  ├── UNetDiffusionAgent        ← DP, DP3 (UNet1D + Diffusion)
  ├── DiTXFlowMatchAgent        ← ManiFlow (DiTX + FlowMatchWithConsistency)
  ├── SATAgent                  ← SAT (SATBackbone + SATFlowMatch)
  ├── MultiTaskAgent            ← MultiTask (DiT + Diffusion/FlowMatch)
  ├── R3DAgent                  ← R3D (OneWayTransformer + Diffusion)
  ├── DQRISEAgent               ← DQ-RISE (自定义 UNet + Diffusion, action_dim 缩减)
  └── ActionFlowAgent           ← ActionFlow (GeoFormer 感知 + 独立 DiT backbone + SimpleRectifiedFlow, KV cache)
```

### Agent 继承模式 (添加新 Agent 时选)

| 模式 | 父类 | 何时用 | 最近参考 |
|------|------|--------|---------|
| **A: UNet+Diffusion** | `UNetDiffusionAgent` | Flat encoding → UNet1D(FiLM) → Diffusion | `dp3.py` |
| **B: DiTX+FlowMatch+Consistency** | `DiTXFlowMatchAgent` | Token seq → DiTX(cross-attn) → FlowMatch+consistency | `maniflow.py` |
| **C: Fully custom** | `BaseAgent` | 完全自定义 backbone + decoder | `sat.py`, `r3d.py`, `action_flow.py` |

### Agent 对比 (找最接近的参考实现)

| Agent | 输入 | Encoder | Backbone | Decoder | 独特点 |
|-------|------|---------|----------|---------|--------|
| **DP3** | PC+state | PointNet+StateMLP | UNet1D(FiLM) | Diffusion(DDIM 10步) | **最简参考**, pc_dim=6 |
| **DP** | RGB+state | DINO/CLIP/SigLIP+StateMLP | UNet1D(FiLM) | Diffusion(DDIM 10步) | RGB, channels_last |
| **ManiFlow** | PC+state | PointNetDense+StateMLP | DiTX(cross-attn) | FlowMatch+Consistency | Token条件, EMA教师, wd=1e-3 |
| **ActionFlow** | PC+state | PointNeXT 192-patch local tokenizer → GeoFormer 4L×576(3D RoPE) → 385×384 memory | ActionFlowDiT 8L×768(Shared AdaRMS) | SimpleRectifiedFlow | cond 是 **dict** {memory,state}; 12Q/12KV full CA, SwiGLU-1536, compact 384 conditioner, CA KV cache; ~75.7M |
| **SAT** | PC+state | PointNeXT+StateMLP | SATBackbone(EJC+MMAttn) | SATFlowMatch | (B,Da,T), shuffle, compile=default |
| **R3D** | PC+state | Uni3D+StateMLP | OneWayTransformer | Diffusion(DDIM 10步) | 级联mask, 分组loss |
| **DQRISE** | PC+state | iDP3+StateMLP | UNet1D(tcp+1维) | Diffusion(epsilon,20步) | VQ码本, lr=3e-4, warmup=2000 |
| **MultiTask** | RGB+state+text | DPObsEncoder+CLIP | DiT(AdaLN) | Diffusion/FlowMatch | 多任务, 预缓存text |

### 动作空间

| `action_key` | arm | hand | total |
|-------------|-----|------|-------|-----------|
| `action` (joint) | 7 (关节角) | 12 (XHand) | **19** |
| `action_ee` (ee) | 9 (pos3+rot6d) | 12 (XHand) | **21** |

`joint_state` 固定为 19 维（7 臂 + 12 手），与 `action_ee` 的 21 维动作空间不同。

### Action Decoder

| Decoder | 预测目标 | 推理 | 谁用 |
|---------|---------|------|------|
| `Diffusion` | ε / x0 / v | DDIM 迭代 | DP, DP3, R3D, DQRISE |
| `FlowMatch` / `SATFlowMatch` | v=x1-x0 | Euler ODE | MultiTask, SAT |
| `FlowMatchWithConsistency` | v + consistency(EMA教师) | Euler ODE | ManiFlow |
| `SimpleRectifiedFlow` | v=x1-x0 (NoiseShift α=3 + 75/25 uniform mixture) | Euler（任意正 NFE）/ Explicit Midpoint（偶数 NFE，KV cache） | ActionFlow |

---

## 配置速查

### 关键参数 (跨策略差异)

| 参数 | action_flow | dp3 | maniflow | multitask_dit | r3d | dqrise | sat |
|------|-------------|-----|----------|---------------|-----|---------|-----|
| action_dim | 19/21 | 19/21 | 19/21 | 19/21 | 19/21/28 | 19/21 | 19/21 |
| backbone | ActionFlowDiT 8L×768 | UNet[256,512,1024] | DiTX 12L×768 | DiT 8L×512 | OneWay 4L | UNet[256,512] | SAT 8L×768 |
| train/infer steps | -/2 NFE | 100/10 | -/4 | 100/10 | 100/10 | 100/20 | -/10 |
| prediction_type | velocity | sample | velocity | sample | sample | epsilon | velocity |
| lr / wd | 1e-4 / **1e-3** | 1e-4 / 1e-6 | 1e-4 / **1e-3** | 1e-4 / 1e-6 | 1e-4 / 1e-6 | **3e-4** / 1e-6 | 1e-4 / 1e-6 |
| betas | **[.9,.95]** | [.95,.999] | **[.9,.95]** | [.95,.999] | [.95,.999] | [.95,.999] | [.95,.999] |
| bfloat16 / compile | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓(default) |
| val_ratio | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

> dp = dp3 参数; multitask_dit = 8L×512, lr=1e-4
> 全部 `total_train_steps: 100000`, `warmup: 500` (dqrise: 2000)
> action_flow 当前 config = 最优配方（归一化 xyz + baseline 优化器 wd=1e-3/obs_wd=1e-6 + cosine + context 384/ffn 1536 + 无 dropout + step-gate off）。「科学 PR 全开」（metric xyz / wd=1e-2 / cosine_min_lr / dropout）已证伪负收益。见 [docs/ActionFlow-架构与实验结果.md](docs/ActionFlow-架构与实验结果.md)

`action_dim` 公式: `${eval:'21 if ${eq:${action_key},action_ee} else 19'}`
`agent._target_`: `dexmani_policy.agents.core.<name>.<Name>Agent` (Hydra 直接导入，无显式注册表)
Eval 各策略 denoise_steps 不同 (action_flow=2, maniflow=4, dqrise=20, 其余=10)；use_ema=true 共享。ActionFlow 中 denoise_steps 即 NFE；参数优先级 CLI > 子节 > eval 共享层。

> Config 模板字段清单、Eval YAML 结构 → [README](README.md#配置参考)

---

## 训练内幕

**NaN 两层防护**: L1 (backward前, loss NaN → 保存 debug ckpt → raise) / L2 (optimizer.step前, 梯度 NaN → zero_grad → raise)。诊断用 `dexmani-training-debug` skill。

**Checkpoint**: 20/40/60/80/100% 里程碑; `latest.pt` symlink; 自动 resume。**DDP**: ckpt 加载在 compile + DDP 包装**前**; timeout=30min; `dp3` 仅单卡。**Shape 验证**: `_validate_batch()` 在 `compute_loss`/`predict_action` 入口。

> 详细机制 → [README](README.md#训练机制)

---

## 设计约定 (不是 Bug，不要修)

- **Normalizer 全量拟合**: 用全部 replay buffer (含验证集)。生态惯例 (ManiFlow/R3D/SAT/RoboTwin 均如此)。`limits` 模式下 val 不影响 min/max
- **`tcp_dim` 命名**: joint模式=7(臂关节), ee模式=9(TCP位姿) — 历史命名，勿据此推断语义
- **DQRISE 直接继承 `BaseAgent`**: `diffusion_action_dim = tcp_dim+1` ≠ `action_dim`，无法复用 UNetDiffusionAgent。**不要重构**
- **R3DObsEncoder 拼接**: patch_tokens + state_emb + pc_pe 沿 feature 维 (非 `torch.cat`)
- **EMAModel BatchNorm**: affine 参数直接复制，不 EMA 平均
- **FlowMatchWithConsistency `target_t`**: flow 分支训练=0，consistency 分支训练=dt1(>0)，推理=dt(>0)
- **ActionFlow 独立实现**: `action_flow_flowmatch.py` / `action_flow_dit.py` / `geoformer.py` 完全独立，不共享 `time_sampler.py` / `flowmatch.py` / `ditx.py`。**不要合并或交叉引用**
- **ActionFlow `cond` 是 dict** `{"memory": [B,385,384], "state": [B,38]}` (其余策略是 Tensor)。`BaseAgent.predict_action_from_cond` 会直接取 `cond.shape/.device/.dtype`，所以 `ActionFlowAgent` **局部 override** 该方法。**不要为此改 BaseAgent**
- **ActionFlow state 不进入几何**: joint_state 只作为 global modulation 进 ActionDiT，绝不 broadcast 到 geometry token
- **GeoFormer 末尾有 `norm_out` (RMSNorm)**: pre-norm 只约束每个 sublayer 的**输入**，不约束 residual stream 的**输出**。缺它时 backward Jacobian 随权重尺度增长远快于 forward (实测 ~wscale¹¹ vs ~wscale³)；加上后输出与梯度对权重尺度**不变**。ActionDiT 末尾的 `modulate_rms` 已起同样作用。**不要删**
- **`_rms_norm` eps = 1e-5 (非 1e-6)**: eps 同时决定近零行的 backward 增益上限 (1/√eps)，1e-6→1000×，1e-5→316×。bf16 下这是余量差别
- **ActionFlow KV cache 是普通 python 属性**: 不能 `register_buffer`，否则进 `state_dict()`，eval 时 `strict=True` 加载会因训练/评测 batch 不同而失败 (smoke test 的 save→load 同一模型，**测不出来**)
- **ActionFlow batch 64 × grad-accum 2**: 模型在 24GB 卡上 batch 128 放不下 (实测 bf16 峰值 21.7 GiB / fp32 OOM)。等效 batch 仍是 128，`total_train_steps` 只计 optimizer step，配方不变
- **PointNeXT `include_global_token`**: 默认 True 保持 sat 行为（maniflow 已改用 pointnet_dense）；ActionFlow 传 False，**在构造期**就不建全局分支 (DDP `find_unused_parameters=False` 下建而不用会直接崩)
- **DDP 覆盖不全**: `dp3` 有意仅单卡
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
  select_best_ckpt.py         # 里程碑自适应淘汰 → best_ckpt.json
  eval_best_ckpt.py           # 离线评测 (Hydra-free CLI)
  record_demo.py              # 离线 demo 视频录制
  smoke_test.py               # 构建验证 (6 阶段)
  configs/                    # 9 YAML + 7 DDP overlays
  agents/
    core/                     # BaseAgent + 9 variants
    action_decoders/          # Diffusion, FlowMatch variants
      backbone/               # UNet1D, DiT, DiTX, OneWayTransformer, SATBackbone
    obs_encoder/              # pointcloud/, rgb/, text/, state_mlp.py, plugins/
    vq_hand/                  # DQ-RISE: VQVAEHand, CodebookManager, ResidualVQ
  datasets/                   # BaseDataset, PCDataset, ReplayBuffer, SequenceSampler
  training/                   # Trainer, build_utils, ema, logging, lr_scheduler, eval_utils
  env_runner/                 # BaseRunner, SimRunner, MultiTaskSimRunner
  common/                     # LinearNormalizer, checkpoint_io, pytorch_util
  tools/                      # train_vq_hand.py, extract_codebook.py
```

### 找最近参考 (Vibe Coding)

| 你要做什么 | 从哪开始 |
|-----------|---------|
| 加 UNet+Diffusion agent | `dp3.py` + `dp3.yaml` |
| 加 Transformer+FlowMatch agent | `maniflow.py` + `maniflow.yaml` |
| 加完全自定义 backbone agent | `sat.py` + `sat.yaml` 或 `action_flow.py` + `action_flow.yaml` |
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
| `docs/DP3-R3D-ManiFlow测试结果0813.md` | DP3 vs R3D vs ManiFlow 五项任务对比评测 |
| `docs/ActionFlow-架构与实验结果.md` | ActionFlow 唯一权威文档（历史架构沿革 + 当前架构 + 实验记录 + 结论方法论） |
