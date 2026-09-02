# ActionFlow Stabilization 与 NFE 校准执行指南

> **用途**：指导 Claude Code / Codex 在不大幅修改模型架构、不扩展成自动化大规模对比实验的前提下，完成 ActionFlow 的小型 correctness/stability 修复、一次完整训练，以及受控的 NFE 最佳步数校准。
>
> **Review 基线**：`main@aa4a0a39dd5a69e3a4ad85ea8190d6889610d175`（2026-09-01）。若执行时 HEAD 已变化，先重新核对本文列出的相关文件，再实施修改；不要机械套用旧结论。

---

## 1. 目标与边界

本轮不是 architecture research，而是 **ActionFlow Stabilization v1**：

1. 修复确定性的接口/配置错误和 fail-fast 缺口；
2. 加强 regression tests 与 smoke checks；
3. **保持当前模型函数、训练分布、优化器、数据增强、Flow objective 不变**；
4. 先做一次短 numerical preflight，再做 **一次** `pour` 完整 100k 训练；
5. 完整训练后，仅对固定 `40pct` checkpoint 做受控 NFE 校准；
6. 不运行 checkpoint tournament、solver sweep、multi-seed training 或 architecture ablation。

> 范围：本轮仅用单卡 `train.sh action_flow` 训练与评测；`ddp/action_flow` 只由 config contract test（`tests/test_action_flow_config.py`）覆盖其配置组合，不纳入本轮训练/评测。

### 当前 canonical recipe 必须保持

```text
horizon=16
n_obs_steps=2
n_action_steps=8
state_dim=19
action_dim=19 (action) / 21 (action_ee)
PointNeXT: 192 patches
GeoFormer: 4L x 576
memory: [B,385,384]
ActionDiT: 8L x 768, context=384, FFN=1536
Rectified Flow: NoiseShift alpha=3, shifted_ratio=0.75
solver=midpoint
NFE(default)=2
batch=64, grad_accum=2, effective batch=128
AdamW lr=1e-4, wd=1e-3, obs_lr=1e-4, obs_wd=1e-6
cosine + warmup 500
bf16=true, compile_mode=default, EMA=true
max_train_episodes=80
total_train_steps=100000
```

---

## 2. 本轮允许修改的内容

### 2.1 P0 — 明确 ActionFlow 仅支持两帧 observation

**问题**：`ActionFlowObsEncoder` 按 `n_obs_steps * state_dim` 组织 state history，但 `ActionFlowDiT.state_mlp` 与 forward shape contract 固定为 `2 * state_dim`。当前项目本身也把 `n_obs_steps=2` 作为核心不变量。

**要求**：本轮不要把 backbone 泛化到任意 T；在 `ActionFlowAgent.__init__` fail-fast：

```python
if n_obs_steps != 2:
    raise ValueError(...)
```

同时建议在 Agent 初始化时检查：

```python
if n_obs_steps - 1 + n_action_steps > horizon:
    raise ValueError(...)
```

**不得**把 `state_mlp` 改成动态 `n_obs_steps * state_dim` 并宣称支持 T!=2；这属于新的 temporal architecture feature。

涉及文件：

- `dexmani_policy/agents/core/action_flow.py`
- `tests/test_action_flow_config.py`
- 新增/扩展 ActionFlow contract test

---

### 2.2 P0 — Rectified Flow / NFE 参数 fail-fast

涉及：`dexmani_policy/agents/action_decoders/action_flow_flowmatch.py`

初始化及 override 都必须拒绝非法值：

```text
noise_shift_alpha > 0
0 <= noise_shift_ratio <= 1
denoise_steps / override NFE 必须是真正的正整数
solver in {euler, midpoint}
midpoint => NFE 必须为偶数
```

不要使用 `int(2.7) -> 2` 这种静默截断；非整数直接报错。

> 现状：`solver ∈ {euler,midpoint}` 已在 init 校验（`action_flow_flowmatch.py:52`），midpoint 偶数 NFE 已在 sampling 时校验（`:122`）。本轮需补齐的是 `noise_shift_alpha > 0`、`0 <= noise_shift_ratio <= 1`、以及 NFE 正整数校验（`_resolve_nfe` 目前 `int(nfe)` 会静默截断 `2.7 -> 2`，`:65`），并把 midpoint 偶数校验提前到 init/override。

目标是避免 100k 训练完成后才在 evaluation 阶段发现非法 solver/NFE 配置。

---

### 2.3 P1 — KV cache exception-safe cleanup

当前 `predict_action` 已用 `try/finally` 包裹 sampling 且 `finally` 中调用 `clear_kv_cache()`（`action_flow_flowmatch.py:169-175`），正常路径的异常安全已满足。唯一缺口是 `setup_kv_cache(memory)` 位于 try 块外（`:168`）；把它移入 try 内即可（可选、非必需——`setup_kv_cache` 是单条原子赋值 `action_flow_dit.py:177`，不存在 partial cache 状态）：

```python
try:
    model.setup_kv_cache(memory)
    ... sampling ...
finally:
    model.clear_kv_cache()
```

**正常路径输出必须完全不变。**

---

### 2.4 P1 — 正式 ActionFlow regression tests

建议新增：

`tests/test_action_flow_contract.py`

至少覆盖：

1. `n_obs_steps=2` 可构建，`n_obs_steps!=2` fail-fast；
2. `action`: action_dim=19 / state_dim=19 / state history=38；
3. `action_ee`: action_dim=21 / state_dim=19 / state history仍为38；
4. Flow 参数验证：midpoint NFE 2/4 pass，3/0/2.5 fail，非法 ratio/alpha fail；
5. KV cache forward parity；
6. sampling 异常后所有 `_cached_k/_cached_v` 被清空。

已有模块 `__main__` 中的 CUDA/bf16/compile 自测不要全部复制进 CPU pytest；正式 tests 只保留稳定、快速的 contract regression。

---

### 2.5 P1 — 更新 smoke test 的 ActionFlow 参数预算

当前 PR-11 后约：

```text
perception ~= 16.85M
ActionDiT   ~= 58.81M
total       ~= 75.66M
```

`dexmani_policy/smoke_test.py` 中旧的 95M–101M 参数注释已过期（该处仅 print 参数量、无 assert gate，见 `smoke_test.py:126`）；本轮把注释更新为 PR-11 数值，并**新增**宽松 gate（当前不存在，是新增而非「更新预算」）。

建议更新并加入宽松 regression gate，例如：

```text
16M < perception < 18M
58M < backbone   < 60M
74M < total      < 78M
```

用于及时发现 context 回退到 768、误启用冗余分支等回归。

---

### 2.6 P2 — 修正文档中的 coordinate/RoPE 语义

`geoformer.py` 当前部分注释仍使用 “Metric-wavelength” / “physical distance” 表述，但 ActionFlow canonical 路径实际消费 dataset-normalized xyz，config 已按 normalized `[-1,1]` units 描述 wavelength。

本轮只修注释/docstring，统一成 **normalized-workspace / coordinate-wavelength 3D RoPE**。

**不要**再次启用 raw metric xyz，也不要改 radius / wavelength。

涉及文件：

- `dexmani_policy/agents/obs_encoder/pointcloud/geoformer.py`（位于 obs_encoder/pointcloud，而非 action_decoders）

---

## 3. 本轮明确禁止的修改

为了保持与历史 `seed=42` baseline 的可比性，本轮不得顺手执行：

- 不关闭/修改 FPS `use_shuffle_output`；即使 patch permutation 数学上冗余，改变 RNG consumption 会改变后续 Flow noise/timestep 的随机轨迹；
- 不改 random FPS start；
- 不改 point-cloud normalization / metric xyz / isotropic normalization；
- 不改 patch count/radius/neighbors；
- 不加入 state token / FK keypoint / EE auxiliary；
- 不加入 multi-scale memory；
- 不换 Uni3D / Concerto / Utonia / pretrained encoder；
- 不启用 `use_step_conditioning`；
- 不改 NoiseShift sampler；
- 不改 dropout / drop-path / weight decay / LR scheduler；
- 不扩大/缩小 ActionDiT；
- 不新增 consistency/shortcut objective；
- 不改 EMA、temporal ensemble 或 action representation；
- 不执行 80→125 demos 数据量实验；
- 不运行第二/第三 training seed。

若执行者认为某项额外修改“顺手且合理”，**也不要合入本轮**；记录成后续 issue/notes 即可。

---

## 4. 实施顺序

### Phase A — Code stabilization

按以下顺序实施：

1. `n_obs_steps=2` + horizon/control slice fail-fast；
2. Flow/NFE 参数 validation；
3. KV cache exception-safe cleanup；
4. ActionFlow formal regression tests；
5. smoke test 参数预算修正；
6. GeoFormer/RoPE 文档语义修正。

不要同时重构公共 BaseAgent / 通用 flowmatch 代码；ActionFlow 保持独立实现。

---

## 5. 静态与 smoke 验证

代码修改完成后，至少执行：

```bash
python -m pytest tests/test_action_flow_config.py tests/test_action_flow_contract.py -q
python dexmani_policy/smoke_test.py action_flow
```

如本机环境缺 CUDA / 数据 / `dexmani_sim`，明确记录未执行项和环境原因；不要因为环境缺失而改写核心逻辑。

### 必须确认

- 所有 trainable params 都能收到 gradient；
- action/action_ee state contract 正确；
- strict checkpoint roundtrip 正确；
- KV cache parity 正确；
- 当前参数量落在 PR-11 预算内。

---

## 6. 600-step numerical preflight

这是数值/工程 preflight，不是性能实验，不做仿真评测。

```bash
bash scripts/training/train.sh action_flow \
  'task_name=pour' \
  'training.seed=42' \
  'dataset.max_train_episodes=80' \
  'training.loop.total_train_steps=600'
```

选择 600 step 的原因：跨过 warmup=500，并留出数值余量。历史上约 step 326 曾在 `reduce-overhead`(CUDA graphs) 编译模式下出现 bf16 非有限梯度；当前 config 已用 `compile_mode: default` 规避（默认模式数值稳定），该故障点不应再复现。

注意：600-step run 的 cosine schedule 被压缩，**不得把其 loss 或性能当作 100k 的缩短实验**。

Preflight 只检查：

```text
loss finite
grad finite
grad_norm / clip_ratio 合理
compile 无异常                  （stdout 观察；日志不记录）
checkpoint 保存正常
无重复 recompile / 明显资源泄漏   （stdout + nvidia-smi 观察；无日志/checkpoint 记录）
```

不运行 simulation、video、NFE sweep。

> 注：训练日志只记录 `grad_norm` / `clip_ratio`（`trainer.py`），「compile 无异常」「资源泄漏」只能从终端 stdout 与 `nvidia-smi` 目测，无自动化门禁。

---

## 7. 一次完整 100k 训练

preflight 通过后，只跑一次 canonical full training：

```bash
bash scripts/training/train.sh action_flow \
  'task_name=pour' \
  'training.seed=42' \
  'dataset.max_train_episodes=80'
```

不要添加其它 override。

Trainer 本身只做训练和 20/40/60/80/100% milestone checkpoint 保存，不应插入 online simulation evaluation。

完整训练阶段禁止：

```text
select_best_ckpt
eval_pipeline
eval_action_flow_solvers
solver sweep
architecture sweep
multi-seed training
video demo
```

---

## 8. NFE 最佳步数校准（唯一允许的受控 sweep）

NFE 校准只发生在 **完整 100k 训练之后**。

### 8.1 固定条件

只使用：

```text
checkpoint = 40pct
weights    = EMA
solver     = midpoint
task       = pour
same eval seed subset
video      = off（sweep 路径例外：仍会写视频，见 §8.2 注 1）
```

不得同时搜索 checkpoint 或 solver。

### 8.2 Screening：25 个相同 seeds

仅搜索：

```text
NFE = [2, 4, 8, 10]
```

允许使用现有 `eval_best_ckpt.py` 的 multi-value sweep，因为它会 **checkpoint 只加载一次，并对所有 NFE 使用同一批 seeds**。这是本轮唯一允许的自动化比较。

示意命令：

```bash
python dexmani_policy/eval_best_ckpt.py \
  --policy-name=action_flow \
  --task-name=pour \
  --exp-name=<EXP_NAME> \
  --ckpt-tag=40pct \
  --episodes=25 \
  --no-videos \
  'eval.denoise_timesteps_list=[2,4,8,10]'
```

不要使用 `eval_action_flow_solvers.sh`，因为它包含 Euler-1 等额外 solver candidate，本轮只校准 midpoint NFE。

> 注 1（video）：`--no-videos` 只对 single-value 评测生效；sweep 路径（`denoise_timesteps_list` 长度 > 1）会无条件写 timestamped 视频目录（`eval_best_ckpt.py:485-488`），因此上述 sweep 仍会生成 ~25 seeds × 4 NFE 的视频。若要避免，改用 `--denoise-steps` 逐值跑 single-value（每次单独加载 checkpoint），或接受额外视频输出。

> 注 2（NFE 判据）：本轮 midpoint-only 校准不含 Euler-1，因此无法计算 CLAUDE.md NFE 判据中的效率腿 `R2 = (SR2−SR1)/(SR10−SR1)`（SR1 需 Euler-1）；本指南用「SR plateau 最小 NFE」规则替代效率判定。

### 8.3 Screening 判定规则

不要只看 raw SR，要读取 `per_seed_details` 做 paired seed comparison。

设 `S_N` 为 25 seeds 中 NFE=N 的成功数。

Sweep 结果落于 `exp_dir/eval_dexsim/<timestamp>/denoise_timesteps<N>/result_details.json`（每 NFE 一份，`per_seed_details` 字段为 `{seed, success, steps, total_steps}`）；`eval_summary.json` 只有聚合 SR/avg_steps/n_total，无 per-seed 数据。paired net wins 定义：`#{seed: 候选 NFE 成功且 NFE2 失败} − #{seed: NFE2 成功且候选 NFE 失败}`。

1. 若 `max(S_4,S_8,S_10) - S_2 <= 1`，且 paired comparison 没有明显单向改善：
   - `N* = 2`。
2. 若更高 NFE 有明显改善：
   - 取**进入性能平台的最小 NFE**，而不是最大 NFE；
   - 实用规则：从 `[2,4,8,10]` 中选择最小的、且成功数距离 screening 最大值不超过 1 的 NFE；
   - 如果该候选相对 NFE2 的 paired net wins 不足 2 个 seed，则视为信号不稳，保留 `N*=2`。

示例：

```text
NFE2 : 15/25
NFE4 : 20/25
NFE8 : 20/25
NFE10: 19/25
=> candidate N*=4
```

核心原则：

> **选择达到 SR plateau 的最小 NFE，而不是 argmax SR 的最大 NFE。**

---

## 9. NFE 最终 100-seed 确认

### 若 screening 得到 `N*=2`

只运行一次：

```text
40pct / NFE2 / 100 seeds / EMA / midpoint
```

无需再测其它 NFE。

### 若 `N* != 2`

只最终确认两组：

```text
NFE=2
NFE=N*
```

各 100 个相同 seeds；不要把 2/4/8/10 全部扩大到 100 episodes。

可以分别调用 `eval_best_ckpt.py`，或仅对 `[2,N*]` 使用同一 multi-value sweep：

```bash
python dexmani_policy/eval_best_ckpt.py \
  --policy-name=action_flow --task-name=pour --exp-name=<EXP_NAME> \
  --ckpt-tag=40pct --episodes=100 --no-videos \
  'eval.denoise_timesteps_list=[2,N*]'
```

（`--episodes=100` 是 100-seed 确认的关键覆盖项；sweep 仍会写视频，见 §8.2 注 1。）

### 最终 default NFE 判定

只有当：

```text
SR(N*) - SR(2) >= 5 percentage points   (== 0.05 absolute，对 0-1 成功率等价于 5%)
```

且 paired seeds 明显单向改善时，才把 default quality operating point 改成 `N*`。

否则保留：

```text
default NFE = 2
```

如果 `N*` 明显更好，记录两种 operating points：

```text
NFE2  = efficiency operating point
NFE*  = quality operating point
```

不要删除 NFE2 结果。

---

## 10. Tail diagnostic

NFE 校准结束后，只做一个便宜的后期退化检查：

```text
checkpoint = 100pct
episodes   = 25
NFE        = 最终选定的 default/quality NFE
EMA        = true
video      = off
```

该 25-episode 结果只用于判断 `40k -> 100k` late degradation 是否仍然明显，**不作为正式模型比较结论**。

不要评测 20/60/80pct，也不要生成 `best_ckpt.json`。

---

## 11. 明确禁止的评测扩展

除上述 NFE calibration 外，执行者不得自行运行：

```text
scripts/eval/eval_pipeline.sh
scripts/eval/select_best_ckpt.sh
scripts/eval/eval_action_flow_solvers.sh
所有 milestone × NFE 笛卡尔积
Euler/Heun/其它 solver 搜索
training seed sweep
architecture/config grid search
视频 demo 批量录制
```

若结果异常，先停止并 review；不要自动扩大搜索空间。

---

## 12. 验收标准

### 12.1 Engineering PASS

必须满足：

- regression tests 通过；
- `smoke_test.py action_flow` 通过；
- 600-step preflight 无 NaN/Inf；
- 100k training 无 non-finite loss/gradient；
- 5 个 milestones 正常保存；
- strict EMA checkpoint restore 正常；
- `40pct` NFE screening/最终评测完成；
- `result_details.json` 中核对 `ckpt_tag`/`ckpt_path`、`n_total`、`denoise_steps` 与 per-seed details。

### 12.2 40pct 性能 gate

历史 anchor：`68/100`（NFE2）。

单 train-seed 下采用宽松 gate（此处 `SR` 指**最终选定的 default NFE** 在 40pct 的 100-seed SR——`N*=2` 时为 NFE2，否则为 N*）：

```text
Green : SR >= 0.65
Yellow: 0.60 <= SR < 0.65
Red   : SR < 0.60
```

`SR > 0.70` 只能记为 positive signal，不得凭单 train seed 宣称 architecture improvement。

Yellow（0.60–0.65）不自动停止也不自动通过：记录为边缘结果，人工 review 后决定是否进入后续阶段（§14 的 stop 条件只覆盖 Red `< 0.60`）。

---

## 13. 执行结束后的输出

Claude Code / Codex 最终必须给出简短报告，至少包含：

```text
1. 修改文件与每项修改的原因
2. 明确声明未改变的 canonical recipe
3. pytest / smoke / 600-step preflight 状态
4. 100k experiment path
5. 40pct NFE screening: 2/4/8/10 的 success count + paired seed摘要
6. 最终 NFE candidate N* 与选择理由
7. 100-seed NFE2 vs NFE*（如需要）
8. 100pct/25ep tail diagnostic
9. 是否通过 engineering/performance gate
10. 后续研究建议（只列建议，不自动执行）
```

---

## 14. Stop conditions

出现以下任一情况立即停止，不要自动扩大实验：

- static/smoke test 失败；
- 600-step preflight 出现 NaN/Inf 或异常 recompile；
- full training 出现非有限 gradient/loss；
- checkpoint strict restore 失败；
- `40pct` 100-seed SR < 0.60；
- NFE evaluation 的 seed/checkpoint/EMA 条件无法保证一致。

此时先定位 correctness/工程问题，再决定是否继续。

---

## 15. 本轮完成后才允许讨论的下一阶段

只有 Stabilization v1 + NFE calibration 完成后，才进入新的 controlled research，例如：

1. pretrained 3D perception（Uni3D random vs pretrained）；
2. 80→125 demos / data diversity；
3. proprio/state token 或 FK keypoints；
4. fine/global multi-scale memory；
5. isotropic/granularity-aware geometry normalization；
6. 当且仅当 NFE2 明显低于高 NFE 时，再研究 Shortcut / Consistency Flow。

**这些均不属于本文执行范围。**
