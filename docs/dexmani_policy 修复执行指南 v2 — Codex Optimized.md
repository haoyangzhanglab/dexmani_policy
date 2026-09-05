# dexmani_policy 修复执行指南 v2

> Repository: `haoyangzhanglab/dexmani_policy`
>
> Review baseline: `main @ 9e9597f4e3a06702f23cc88084b821217053a580`
>
> Executor: Codex
> Repository role: PhD / personal Robot Learning research codebase

---

# 1. Objective

本轮目标不是重构 `dexmani_policy`，而是修复会实际影响以下四类结果的问题：

```text
Training Correctness
        ↓
Simulation Evaluation Correctness
        ↓
Checkpoint Selection Reliability
        ↓
Deployment Inference Consistency
```

所有修改遵循：

```text
先证明问题
→ 写最小 regression test
→ 最小修改
→ targeted tests
→ stage gate
```

不得为了代码风格、通用性或未来可能需求扩大 patch。

---

# 2. Scope

允许修改：

```text
dexmani_policy/
tests/
必要的 dexmani_policy configs
必要的 docs/comments
```

禁止修改：

```text
dexmani_sim/
dexmani_real/
其他 repository
```

即使发现跨仓库问题，也只记录到最终报告的 `Deferred`。

---

# 3. 本轮明确不修改

以下内容不是本轮 bug fix：

```text
normalizer full_dataset → train_only
XYZ isotropic normalization
dense point token ordering
SAT temporal fusion
RGB double-crop architecture choice
Flow / consistency loss weighting
tactile sim-real contract
dataset fingerprint framework

EMA deepcopy optimization
frozen-param EMA optimization
training metric aggregation
exact mid-epoch resume
pyproject / requirements
exporter clean-tree / fixed-origin
```

其中 normalizer 保持当前 community-compatible：

```text
full replay-buffer normalization
```

不要改变现有 checkpoint scaling。

---

# 4. 不得“误修”的已有正确机制

保持不变：

```text
DDP non-boundary no_sync()

SequenceSampler:
episode start = repeated first observation

Action execution:
start = n_obs_steps - 1

SAT:
shuffle / unshuffle
temporal feature fusion

ManiFlow:
student perception conditioning
EMA teacher action-backbone design

ChunkOverlapBlender:
older prediction weight > newer prediction weight
```

---

# 5. Execution Structure

本轮压缩成：

```text
Preflight
   ↓
Phase 1 — Training Correctness
   ↓
Phase 2 — Runner / Evaluation Correctness
   ↓
Phase 3 — Selection & Evaluation Contract
   ↓
Phase 4 — Deployment Consistency
   ↓
Phase 5 — Local Low-risk Fixes
```

Phase 1–4 是核心修复。

Phase 5 不允许阻塞或扩大 Phase 1–4。

---

# 6. Preflight

首先执行：

```bash
git rev-parse HEAD
git status --short

python -m unittest discover -s tests -p "test_*.py" -v
```

记录 baseline：

```text
HEAD
working tree status
passed
failed
skipped
```

如果存在与本计划无关的 pre-existing failure：

```text
记录
不顺手修
继续本计划
```

不要 reset 用户已有修改。

---

# 7. Phase 1 — Training Correctness

只处理：

```text
1. DDP forward lifecycle
2. gradient accumulation tail
3. empty train loader fail-fast
```

目标文件：

```text
dexmani_policy/agents/core/base.py
dexmani_policy/training/trainer.py
dexmani_policy/training/build_utils.py
dexmani_policy/train_ddp.py       # only if validator rename requires
tests/test_training_regressions.py
```

---

## 7.1 DDP Forward

### Current

Trainer：

```python
self.model.compute_loss(batch, **loss_kwargs)
```

这绕过：

```text
DDP.__call__
→ DDP.forward
→ module.forward
```

### Fix

在 `BaseAgent`：

```python
def forward(self, batch, **kwargs):
    return self.compute_loss(batch, **kwargs)
```

Trainer：

```python
raw_loss, log_dict = self.model(batch, **loss_kwargs)
```

### Do not

禁止：

```python
self.model.module.compute_loss(...)
```

禁止 DDP / non-DDP 两套 loss path。

---

## 7.2 Gradient Accumulation

### Important correction

不要采用：

```text
只有 tail 最后一个 microbatch
使用 tail divisor
```

这是错误的。

假设：

```text
K = 4
L = 10
```

最后 group：

```text
microbatch 8
microbatch 9
```

两者都必须除以：

```text
2
```

而不是：

```text
8 → /4
9 → /2
```

---

## 7.3 Recommended accumulation plan

在每个 epoch 开始时：

```python
num_batches = len(self.train_loader)
```

对于 microbatch `i`：

```python
K = self.gradient_accumulation_steps

group_start = (i // K) * K
group_size = min(K, num_batches - group_start)
group_pos = i - group_start

is_boundary = (group_pos + 1) == group_size
```

每个该 group 的 microbatch：

```python
loss_divisor = group_size
```

然后：

```python
(raw_loss / loss_divisor).backward()
```

---

## 7.4 Trainer API

推荐：

```python
def train_one_step(
    self,
    batch,
    *,
    is_accumulation_boundary=True,
    loss_divisor=1,
):
```

内部只负责：

```text
forward
loss finite check
backward / loss_divisor
optional optimizer step
```

不要让 `train_one_step()` 自己推断 dataloader 位置。

---

## 7.5 DDP synchronization

继续：

```text
not boundary
→ self.model.no_sync()

boundary
→ normal DDP synchronization
```

最后 partial group 的最后一个 microbatch：

```text
必须正常 sync
```

---

## 7.6 Empty loader

当前：

```text
len(train_loader) == 0
```

会造成：

```text
while global_step < total_steps
→ empty epoch
→ epoch++
→ global_step unchanged
→ infinite loop
```

因此必须 fail-fast：

```python
if batches_per_epoch <= 0:
    raise ValueError(...)
```

---

## 7.7 Replace stale divisibility warning

当前：

```text
validate_grad_accum_divisibility()
```

建立在：

```text
non-divisible tail 会被丢弃
```

这个旧事实上。

修复 tail 后不要继续保留该 warning。

推荐改为：

```python
validate_gradient_accumulation(
    batches_per_epoch,
    gradient_accumulation_steps,
)
```

只验证：

```text
batches_per_epoch > 0
gradient_accumulation_steps >= 1
```

不要要求 divisibility。

同时删除 Trainer 中类似：

```python
max(1, int(...))
```

的 silent correction。

非法 `gradient_accumulation_steps=0` 应直接报错。

---

## 7.8 Phase 1 tests

至少测试：

```text
agent.forward delegates to compute_loss

Trainer uses model(...)
not model.compute_loss(...)

K=4, L=10
→ optimizer steps = 3

K=4, L=3
→ optimizer steps = 1

K=1
→ unchanged behavior

L=0
→ fail immediately
```

Gradient equivalence test只使用：

```text
equal-sized microbatches
```

不要声称在最后一个 physical DataLoader batch 更小时等价于 sample-weighted full-batch loss。

当前 repository 的语义是：

```text
mean over accumulated microbatch losses
```

本轮不要改变成新的 sample-weighting protocol。

---

## Phase 1 Gate

运行：

```bash
python -m unittest tests.test_training_regressions -v
python -m unittest discover -s tests -p "test_*.py" -v
```

通过后进入 Phase 2。

---

# 8. Phase 2 — Runner / Evaluation Correctness

一次性处理所有 runner execution boundary 问题：

```text
RGB preprocessing
auxiliary action dimensions
MultiTask control validation
MultiTask fatal error propagation
```

目标文件：

```text
dexmani_policy/env_runner/base_runner.py
dexmani_policy/env_runner/sim_runner.py
dexmani_policy/env_runner/multi_task_sim_runner.py
dexmani_policy/common/config.py

dexmani_policy/configs/dp.yaml
dexmani_policy/configs/multitask_dit.yaml

tests/test_eval_regressions.py
```

---

# 9. RGB Simulation Preprocessing

## Current contract

RGB model consumes：

```text
[B, T, 3, H, W]
float32
[0, 1]
```

而 simulator observation 为：

```text
[T, H, W, 3]
uint8
[0, 255]
```

仓库已经存在：

```python
preprocess_validation_rgb(...)
```

deployment 也已经复用。

因此不得重新实现第二套 RGB pipeline。

---

## 9.1 Runner fields

只增加：

```python
rgb_preprocess_size: tuple[int, int] | None = None
rgb_random_crop_size: tuple[int, int] | None = None
```

不要增加新的 RGB preprocess class。

---

## 9.2 Location

在：

```python
BaseRunner.get_obs_batch()
```

执行：

```text
get_stacked_obs()
        ↓
raw RGB history
        ↓
preprocess_validation_rgb(...)
        ↓
torch/device conversion
        ↓
batch dimension
```

ring buffer 继续存 raw observation。

---

## 9.3 Current RGB dtype

当前 DP / MultiTask RGB contract 要求 float32 `[0,1]`。

因此：

```python
preprocess_validation_rgb(
    ...,
    keep_uint8=False,
)
```

即可。

本轮不要把 DataLoader 的 uint8 transfer optimization 扩展到 online runner。

---

## 9.4 Config

`dp.yaml`：

```yaml
env_runner:
  rgb_preprocess_size: ${dataset.rgb_preprocess_size}
  rgb_random_crop_size: ${dataset.rgb_random_crop_size}
```

MultiTask 使用当前 child dataset 对应值。

不要额外实现新的：

```text
all child preprocessing validator
```

当前 MultiTask 本来就必须向同一个模型提供可 batch 的 RGB tensor。

---

# 10. R3D Auxiliary Action + Blender

当前 temporal blender 使用：

```python
result["pred_action"]
```

R3D aux-EE 时：

```text
action_dim = 28
control_action_dim = 19
```

因此改为：

```python
full_pred = result["pred_action"][..., : agent.control_action_dim]
```

再进入：

```python
self._blender.update(...)
```

---

## Design decision

不要新增：

```text
pred_control_action
control_prediction
execution_action API
```

现有 canonical contract 已经定义：

```text
pred_action[..., :control_action_dim]
```

在两个 execution boundary 显式 slice 更简单。

---

# 11. MultiTask control validator

扩展已有：

```python
validate_action_key_consistency(cfg)
```

不要建立第二套 validator。

如果：

```text
env_runner.task_configs
```

存在，则逐 task 检查：

```text
action
→ joint

action_ee
→ ee
```

错误信息包含：

```text
task_name / task index
expected
actual
```

single-task path 保持现状。

---

# 12. MultiTask fatal errors

在：

```python
MultiTaskSimRunner.run()
```

catch order 必须：

```python
except KeyboardInterrupt:
    raise

except EvalEpisodeError:
    raise

except (...):
    ...
```

fatal model / CUDA / contract failure：

```text
不得继续跑下一 task
```

普通 aggregated task failure 机制不需要重构。

---

## Phase 2 tests

覆盖：

```text
raw uint8 RGB
→ runner preprocess
→ same tensor as preprocess_validation_rgb()

non-RGB obs unchanged

28D prediction + 19D control + blender
→ output [..., 19]

single-task action/control validation

MultiTask:
all valid → pass
one task invalid → fail

EvalEpisodeError in task 1
→ task 2 never runs
```

---

# 13. Phase 3 — Selection & Evaluation Contract

这一阶段把原方案的：

```text
EMA fail-closed
seed isolation
inference signature
checkpoint path portability
```

合并为一个机制：

# `best_ckpt.json = Selection Record`

目标文件：

```text
dexmani_policy/select_best_ckpt.py
dexmani_policy/eval_best_ckpt.py
dexmani_policy/training/eval_utils.py
tests/test_eval_regressions.py
```

---

# 14. EMA must fail closed

`load_ckpt_for_inference()`：

```text
use_ema = true
EMA missing
```

必须：

```python
raise RuntimeError(...)
```

禁止：

```text
warning
→ raw weights fallback
```

Raw checkpoint 必须显式：

```text
use_ema=false
```

---

# 15. Selection Record v2

保持 `best_ckpt.json` 为简单 JSON。

不要创建新的 dataclass/schema framework。

推荐：

```json
{
  "record_version": 2,

  "ckpt_relpath": "checkpoints/...",

  "pct": 80,
  "global_step": 80000,
  "success_rate": 0.84,
  "avg_steps": 73.2,
  "n_episodes": 25,

  "inference": {
    "use_ema": true,
    "denoise_steps": 10,
    "temporal_ensemble_coeff": 0.01,
    "policy_seed_mode": "episode_seed"
  },

  "selection": {
    "shuffle_seed": 1066,
    "seeds": [ ... ],
    "initial_episodes": 25,
    "tie_break_used": false
  }
}
```

---

## 15.1 Store actual resolved values

必须保存：

```text
CLI overrides
>
OmegaConf overrides
>
config defaults
```

之后真正使用的值。

例如 selector：

```text
--no-ema
--denoise-steps 4
```

记录必须是：

```json
"use_ema": false,
"denoise_steps": 4
```

---

## 15.2 Selection seeds

保存：

```text
真正执行过的 unique environment seeds
```

如果没有 tie-break：

```text
phase1 seeds
```

如果发生 tie-break：

```text
phase1 + tie seeds
```

不要保存“本来可能使用但实际没有执行”的 seeds。

MultiTask 中 seed 仍表示 shared environment seed，不重复按 task 写入。

---

# 16. Best checkpoint path resolution

`best_ckpt.json` 必须提供严格 v2 的：

```text
ckpt_relpath
```

relative path 以当前 `experiment_dir` 为基准；不读取 legacy `ckpt_path`。

---

## Important

如果：

```text
best_ckpt.json exists
```

但其中 checkpoint 无法解析：

```text
直接 error
```

不要继续 silent fallback：

```text
latest.pt
```

`best_ckpt.json` 缺失、字段无效或目标 checkpoint 不存在时均直接报错；用户可显式选择
`latest`、milestone 或路径，不提供 `best.pt` / `latest.pt` 回退。

---

# 17. Final held-out evaluation

修改：

```python
_select_eval_seeds(...)
```

支持：

```python
excluded_seeds
```

逻辑：

```python
all_seeds = ...
shuffle(all_seeds, eval_seed)
eligible = [
    seed for seed in all_seeds
    if seed not in excluded_seeds
]
eval_seeds = eligible[:episodes]
```

保证：

\[
S_{selection}\cap S_{final}=\emptyset
\]

---

## 17.1 Insufficient remaining seeds

考虑个人研究仓库，不要因为默认：

```text
episodes=100
```

而让只有 100 个总 seed 的实验全部报错。

如果：

```text
requested = 100
held-out remaining = 75
```

使用：

```text
75
```

但必须明确输出：

```text
Requested 100 episodes, only 75 disjoint held-out seeds remain;
evaluating all 75.
```

结果 JSON 记录真实：

```text
n_total = 75
```

禁止重复 selection seeds 补足 100。

---

# 18. Historical records

旧 `best_ckpt.json` 不受支持。重新运行 selector 生成严格 v2 record，避免把非 held-out
结果误标为最终评测。

---

# 19. Best inference defaults

当：

```text
ckpt = best
```

且存在 v2 Selection Record 时：

默认 final eval 应复用：

```text
EMA/raw
denoise_steps
temporal_ensemble_coeff
```

---

## Precedence

建议：

```text
explicit CLI
>
explicit dotlist override
>
best_ckpt.json inference
>
config.yaml
```

实现顺序：

```text
load config
↓
apply best-record defaults
↓
merge dotlist overrides
↓
resolve explicit CLI values
```

这样默认：

```text
eval best
```

就是 selection 时的同一个 policy。

但研究者仍然可以显式执行：

```text
NFE sweep
raw vs EMA comparison
```

---

# 20. Evaluation result metadata

`result_details.json` 至少增加：

```json
{
  "evaluation_seeds": [...],
  "selection_seeds_excluded": [...],
  "heldout_from_selection": true,
  "use_ema": true,
  "denoise_steps": 10
}
```

避免以后只剩 `_result.txt` 而无法解释结果。

---

# 21. Do not redesign selector

保持：

```text
Phase 1 initial seeds
+
exact-tie optional batch
```

本轮不要实现：

```text
sequential elimination
confidence intervals
bootstrap selection
adaptive stopping
```

但把：

```text
adaptive evaluation
```

等过强描述改成：

```text
fixed two-stage evaluation with optional tie-break
```

`max_episodes` 的 help/comment 也应与真实实现一致。

---

## Phase 3 tests

至少覆盖：

```text
EMA requested + missing
→ error

new record:
actual EMA/NFE persisted

actual selection seeds persisted

tie/no-tie seed record correct

best path prefers relpath

record exists but checkpoint missing
→ no silent latest fallback

final best eval:
selection/test intersection = empty

insufficient held-out seeds:
cap + warning
never overlap

legacy record:
works
heldout_from_selection=false
```

---

# 22. Phase 4 — Deployment Consistency

目标：

```text
selected sim policy
=
deployment runtime policy
```

至少在以下 Policy-owned inference semantics 上：

```text
selected weights
NFE
temporal chunk blending
control dimensions
```

目标文件：

```text
dexmani_policy/deployment/export.py
dexmani_policy/deployment/qualify.py
dexmani_policy/deployment/contract.py
dexmani_policy/deployment/runtime.py

tests/test_deployment_contract.py
tests/test_deployment_export.py
tests/test_deployment_runtime.py
```

---

# 23. Shared inference resolver

这是 Phase 4 最重要的设计约束。

不要让：

```text
export.py
```

和：

```text
qualify.py
```

分别重新解析 inference settings。

建立一个小的 private helper，例如：

```python
_resolve_selected_inference_settings(
    experiment_dir,
    checkpoint_selector,
    cfg_plain,
)
```

---

## Behavior

如果：

```text
checkpoint_selector == "best"
```

且 v2 `best_ckpt.json` 存在：

使用：

```text
best_ckpt.json.inference
```

如果：

```text
non-best checkpoint
```

使用：

```text
config.yaml
```

旧 best record 没 inference：

```text
fallback config
+
explicit warning
```

---

# 24. Export / Qualification must share it

以下两处必须调用同一个 helper：

```text
export_deployment_artifact()
restore_direct_policy()
```

否则：

```text
direct branch
```

与：

```text
export branch
```

可能选择不同：

```text
EMA
NFE
```

导致 parity test 自己产生 mismatch。

---

# 25. Temporal blending in DeploymentSpec

当前 artifact：

```text
dexmani.deployment.v3
```

不需要升级到 v4。

在：

```text
inference_config.eval
```

增加必填字段：

```json
"temporal_ensemble_coeff": 0.01
```

或者：

```json
"temporal_ensemble_coeff": null
```

---

## Current v3 contract

`DeploymentSpec` 增加：

```python
temporal_ensemble_coeff: float | None
```

artifact 必须显式写入 float 或 `null`；缺字段直接拒绝，不提供 migration 或回退。

---

## Validation

若非 `None`：

```text
finite
>= 0
```

即可。

---

# 26. Deployment weights

Export 当前 artifact 最终只保存：

```text
selected state_dict
```

因此继续保持：

```text
use_ema
```

作为 export-time selection information。

Artifact runtime 不需要再携带：

```text
use_ema=true/false
```

因为：

```text
weights
```

已经是被选中的权重。

不要增加重复状态。

---

# 27. LoadedPolicy Temporal Blender

`LoadedPolicy.__init__()`：

```python
if self.spec.temporal_ensemble_coeff is not None:
    self._blender = ChunkOverlapBlender(
        temporal_ensemble_coeff=...,
        n_obs_steps=self.spec.n_obs_steps,
    )
else:
    self._blender = None
```

复用已有：

```text
common/temporal_ensembler.py
```

禁止复制 weighting implementation。

---

# 28. LoadedPolicy.predict()

仍然先：

```python
result = agent.predict_action(...)
snapshot = validate_prediction(...)
```

如果没有 blender：

```text
保持当前 snapshot.control_action
```

如果有：

```python
full_control_pred = snapshot.pred_action[
    ...,
    : self.spec.control_action_dim
]

control_action = self._blender.update(
    full_control_pred,
    n_action_steps=self.spec.n_action_steps,
)
```

这里使用已经 validation + CPU snapshot 的 prediction：

优点：

```text
不信任未验证 tensor
不增加 GPU→CPU transfer
blending computation 极小
aux dimensions 自动排除
```

---

# 29. Episode reset

`reset_episode()`：

```python
if self._blender is not None:
    self._blender.reset()
```

必须保证：

```text
episode N tail
```

不会进入：

```text
episode N+1
```

---

# 30. Warmup state preservation

新增 blender 后：

```text
warmup()
```

不能污染 episode-local blending state。

不要只：

```text
warmup 后 blender.reset()
```

因为用户有可能在已有 runtime state 后调用 warmup。

更稳且仍然简单：

```text
temporarily swap in a fresh blender
→ warmup predictions
→ restore original blender object
```

即：

```python
original_blender = self._blender
self._blender = self._new_blender()

try:
    ...
finally:
    self._blender = original_blender
```

继续保持现有 RNG save/restore。

不要为此实现 blender state serialization。

---

# 31. Qualification

`qualify_policy_parity()` 继续验证：

```text
direct model prediction
vs
exported model prediction
```

不要把 stateful multi-step blender 塞进现有 direct/export model parity。

只需：

```text
DeploymentSpec comparison
```

增加：

```text
temporal_ensemble_coeff
```

真正的 blender execution parity 放在：

```text
test_deployment_runtime.py
```

完成。

---

## Phase 4 tests

测试：

```text
v3 artifact without coeff
→ reject

new artifact
→ coeff parsed exactly

best selector:
export & direct qualification
use same EMA/NFE

non-best selector:
still uses config settings

coeff=None
→ LoadedPolicy behavior unchanged

blender:
first prediction correct

two consecutive predictions:
matches ChunkOverlapBlender reference

action_dim=28
control_dim=19
→ runtime output 19D

reset_episode()
→ clears overlap history

warmup()
→ preserves pre-existing blender state
→ preserves RNG behavior
```

---

# 32. Phase 5 — Local Low-risk Fixes

只有 Phase 1–4 全部通过后执行。

不要扩大 scope。

目标文件大致：

```text
agents/action_decoders/flowmatch.py
datasets/augmentation.py
common/normalizer.py
datasets/replay_buffer.py
datasets/multi_task_dataset.py

tests/test_algorithm_regressions.py
```

---

# 33. ManiFlow absolute target

只修 absolute consistency branch。

应满足：

```text
teacher:
state = t_next
target_t = clamp(t_next + dt)

student:
state = t
target_t = t_next
```

relative mode：

```text
完全不改
```

禁止修改：

```text
teacher architecture
EMA conditioning
ODE solver
flow target
relative mode
```

---

# 34. PointDropout

当前：

```python
max(1, ...)
```

导致：

```text
dropout_ratio=0
```

仍 drop 一个 point。

修：

```python
if self.dropout_ratio == 0:
    return
```

constructor 只验证：

```text
0 <= dropout_ratio <= 1
```

不要顺手重构所有 augmentation parameter validation。

---

# 35. LinearNormalizer stale field view

`LinearNormalizer.fit()` 完成：

```python
self._field_views.clear()
```

`__setitem__()`：

```python
self._field_views.pop(key, None)
```

仅修 cache invalidation。

不要修改 normalization formula。

---

# 36. FlowMatch validation

只增加最有价值的 constructor fail-fast：

```text
denoise_timesteps > 0

0 < flow_batch_ratio < 1

target_t_sample_mode
∈ {"relative", "absolute"}
```

不要在 FlowMatch 再复制一套：

```text
TimeSampler supported mode registry
```

因为 `TimeSampler.sample()` 已经会对未知 mode 明确报错。

避免 duplicate source of truth。

同时把 `flow_batch_ratio` 注释明确为：

```text
sample allocation ratio
```

不是：

```text
loss weight
```

---

# 37. ReplayBuffer root attrs

加载 Zarr 时保留：

```python
attrs = dict(group.attrs)
```

例如 root：

```python
{
    "meta": meta,
    "data": data,
    "attrs": attrs,
}
```

并增加：

```python
@property
def attrs(self):
    return self.root.get("attrs", {})
```

不要：

```text
validate attrs
reinterpret attrs
modify source Zarr
```

---

# 38. MultiTaskDataset input validation

当前已有：

```text
per_task training fail-fast
```

因此不要再实现新的 normalizer-mode mechanism。

只修真正的 constructor invariants。

把：

```python
assert ...
```

改成明确：

```python
ValueError
```

并检查：

```text
len(datasets) == len(task_names)

task_texts is None
or len(task_texts) == num_tasks

sampling_strategy valid

normalizer_mode valid
```

weighted 模式：

```text
task_weights exists
length correct
all finite
all >= 0
sum > 0
```

不要修改 sampling algorithm。

---

# 39. Phase 5 tests

覆盖：

```text
ManiFlow absolute:
student target = t_next
teacher target = t_next+dt

relative mode unchanged

PointDropout ratio=0
→ exact no-op

LinearNormalizer:
view → fit → new view
view → setitem → new view

FlowMatch invalid core args
→ fail in constructor

ReplayBuffer attrs preserved

MultiTask task_text length mismatch
→ ValueError

weighted zero/negative/nonfinite weights
→ ValueError
```

---

# 40. Test Strategy

保持现有 unittest 风格。

只建议新增：

```text
tests/test_training_regressions.py
tests/test_eval_regressions.py
tests/test_algorithm_regressions.py
```

deployment 继续扩展已有：

```text
test_deployment_contract.py
test_deployment_export.py
test_deployment_runtime.py
```

不要建立新的 test framework。

---

# 41. Unit Test Constraints

优先：

```text
tiny fake model
tiny tensor
temporary directory
temporary Zarr
mock runner
CPU
```

禁止 unit tests：

```text
下载 HuggingFace model
启动 dexmani_sim rollout
要求真实机器人
要求多 GPU
训练完整 policy
```

---

# 42. Optional manual DDP smoke

如果机器有 ≥2 GPUs：

Phase 1 后可以执行一个：

```text
2-rank
1–2 optimizer step
tiny experiment
```

smoke test。

如果没有：

最终报告：

```text
multi-GPU DDP runtime smoke: NOT VERIFIED
```

不要把 unit test 当成真实 NCCL DDP 运行验证。

---

# 43. Full-suite cadence

为了提高 Codex 执行效率：

```text
Preflight:
full suite

After Phase 1:
targeted + full

After Phase 2:
targeted

After Phase 3:
targeted

After Phase 4:
deployment targeted + full

After Phase 5:
targeted + final full
```

不要每改一个函数就重跑整个 suite。

---

# 44. Recommended logical commits

建议按：

```text
fix(training): restore ddp forward and accumulation semantics

fix(eval): align runner preprocessing and execution contracts

fix(eval): persist checkpoint selection protocol

fix(deploy): reproduce selected inference semantics

fix(core): tighten local correctness guards
```

如果 Codex 不负责 commit：

仍按这个逻辑组织 diff。

---

# 45. Expected core file-touch order

尽量避免一个文件在多个 phase 反复重写。

推荐：

```text
Phase 1
base.py
trainer.py
build_utils.py

Phase 2
base_runner.py
sim_runner.py
multi_task_sim_runner.py
config.py
RGB configs

Phase 3
eval_utils.py
select_best_ckpt.py
eval_best_ckpt.py

Phase 4
export.py
qualify.py
contract.py
runtime.py

Phase 5
isolated utility/model files
```

---

# 46. Static Audit

完成后搜索：

```text
self.model.compute_loss(
self.model.module.compute_loss(
```

Training forward 不应再绕过 DDP。

Validation 对明确的 raw/unwrapped agent 调：

```text
agent.compute_loss()
```

可以保留。

---

搜索：

```text
EMA weights requested
```

确认：

```text
EMA requested + missing
```

不会 fallback raw。

---

搜索：

```text
result["pred_action"]
```

检查所有 execution path：

```text
temporal blender
```

之前必须裁到：

```text
control_action_dim
```

---

检查：

```text
best_ckpt.json
```

确保：

```text
selection seeds
inference settings
relative checkpoint path
```

都存在于新 record。

---

# 47. Final Acceptance Matrix

Codex 最终逐项回答。

| Contract | Required result |
|---|---|
| DDP training goes through `forward()` | PASS |
| Non-divisible grad-accum tail is stepped | PASS |
| Empty DataLoader fails immediately | PASS |
| RGB sim input equals validation preprocessing | PASS |
| Aux action dims never reach execution | PASS |
| MultiTask fatal episode error aborts immediately | PASS |
| EMA requested but absent fails closed | PASS |
| Selection seeds are persisted | PASS |
| Final best eval excludes selection seeds | PASS |
| Best inference settings are persisted | PASS |
| Default final best eval reuses selected inference | PASS |
| Export and direct qualification resolve same inference | PASS |
| Temporal blending survives into deployment runtime | PASS |
| Episode reset clears temporal state | PASS |
| Warmup does not corrupt temporal state | PASS |
| ManiFlow absolute target semantics fixed | PASS |
| Full regression suite passes | PASS |

不能使用：

```text
probably
seems
should
```

只能：

```text
PASS
FAIL
NOT VERIFIED
```

---

# 48. Deferred by Design

最终报告必须明确未处理：

```text
normalizer train-only
XYZ isotropic normalization
point permutation sensitivity
SAT architecture ablations
RGB crop ablation
Flow/consistency objective weighting
policy stochastic-seed research protocol

EMA construction performance
obs_lr=0 true-freeze optimization
training metric aggregation

dataset fingerprint

dexmani_sim issues
dexmani_real issues
tactile integration
```

这些不是本轮修复遗漏，而是主动控制 scope。

---

# 49. Codex Final Report

最终输出：

```markdown
# Repair Summary

## Phase 1 — Training
Status:
Changed:
Tests:
Unverified:

## Phase 2 — Runner/Eval
Status:
Changed:
Tests:
Unverified:

## Phase 3 — Selection Contract
Status:
Changed:
Tests:
Legacy behavior:

## Phase 4 — Deployment
Status:
Changed:
Tests:
Unverified:

## Phase 5 — Local Fixes
Status:
Changed:
Tests:

## Full Test Suite
Command:
Result:

## Changed Files
...

## Deferred
...

## Remaining Risks
...
```

不要只输出：

```text
Done
```

---

# 50. Definition of Done

本轮核心 Definition of Done：

\[
\boxed{
Training\ Semantics
=
Intended\ Optimization\ Semantics
}
\]

\[
\boxed{
Selection\ Policy
=
Final\ Evaluation\ Policy
}
\]

以及：

\[
\boxed{
Simulation\ Execution\ Semantics
=
Deployment\ Execution\ Semantics
}
\]

对于当前 repository，达到以上三点后，应停止继续工程化，回到模型与实验研究。

任何额外修改如果不能明确改善：

```text
correctness
evaluation validity
inference consistency
```

则不要加入本轮 patch。
