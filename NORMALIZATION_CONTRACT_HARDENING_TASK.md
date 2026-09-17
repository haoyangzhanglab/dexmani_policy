# DexMani Policy Normalization Contract 修补任务书

> 基线：`main@69b79596234ce88ad63db07b9239435c916322de`
>
> 类型：correctness / contract hardening patch
>
> 目标：在**不改动 Dataset/Agent/Encoder 主干算法、不改变现有 7 个默认 Policy 数值 recipe、不改变 `simple.v3` checkpoint 顶层格式**的前提下，修复 normalization semantic contract 在 config、checkpoint、eval、deployment 之间尚未完全闭环的问题。

---

## 1. 背景与验收结论

`69b7959` 已完成 normalization ownership 的主体重构：

```text
Dataset-owned normalization
        ↓
Config-owned feature-level normalization contract
        +
LinearNormalizer fitted state
        +
Encoder-specific preprocessing
```

以下主干已经确认正确，本任务**不得重新设计或推翻**：

- full replay-buffer statistics；
- `BaseDataset.iter_normalization_data()`；
- `action / action_ee / use_aux_ee` effective-action semantics；
- `LinearNormalizer.fit_field / fit_field_chunks` 数学语义；
- SAT `point_cloud: identity`；
- R3D-only XYZ clamp；
- modality dropout 与 normalizer 解耦；
- model / EMA 持有 fitted normalizer state；
- `simple.v3` checkpoint 顶层结构；
- simulation eval 对完整 agent resume contract 的 strict comparison。

本次 fact-check 确认的问题集中在**契约验证层**，不是 normalization 数学或模型架构问题。

---

## 2. 已确认问题

### P1-1 Deployment export 未证明 config normalization 与 checkpoint normalization 一致

训练 checkpoint 已保存：

```text
resume_contract.agent.normalization = {
    version: 1,
    fields: {...}
}
```

但 deployment export 当前 `_reconcile_train_params()` 没有比较 normalization contract，随后 `_build_inference_config()` 又从当前 `config.yaml` 重新生成 normalization metadata。

因此存在语义错配风险：

```text
training checkpoint:
    point_cloud: limits
    scale/offset = limits stats

config.yaml 被修改为:
    point_cloud: gaussian

export:
    artifact contract 写 gaussian
    artifact weights 仍是 limits scale/offset
```

由于 `limits` 与 `gaussian` 都使用同样的 `scale/offset` key hierarchy，仅检查 state_dict 结构无法识别此错配。

**要求：deployment export/qualify 必须 strict compare saved checkpoint normalization contract 与当前 resolved config normalization contract。**

---

### P1-2 `action: identity` 当前可通过 config validation，但运行时必然失败

当前 generic grammar 允许 `identity` 用于 `action`，builder 会跳过该 field，不创建：

```text
normalizer.params_dict.action
```

但 BaseAgent training/inference 固定调用：

```python
self.normalizer["action"].normalize(...)
self.normalizer["action"].unnormalize(...)
```

因此：

```text
action: identity
→ config PASS
→ builder PASS
→ runtime KeyError
```

**要求：第一版明确禁止 `normalization.action: identity`。**

合法 action modes：

```text
auto | limits | gaussian
```

不要为了支持 action identity 修改 BaseAgent。

---

### P1-3 RGB 非 identity 当前可通过 training config，但 generic affine tensor semantics 错误

RGB statistics 来自 replay buffer HWC：

```text
[..., H, W, 3]
```

而 Dataset runtime spatial preprocessing 后传给 Agent 的 RGB 是 CHW：

```text
[..., 3, H, W]
```

`LinearNormalizer` generic affine path 把最后维当 feature dim，因此 `rgb: limits/gaussian` 对 CHW runtime tensor 不具有正确 channel semantics。

同时 deployment export 已明确禁止 RGB normalizer，造成 training/deployment contract 不一致。

**要求：当前 RGB modality 必须使用 `identity` generic normalization。**

RGB preprocessing 保持：

```text
Dataset resize/crop/augmentation
→ generic normalization identity
→ ImageProcessor / pretrained backbone normalization
```

不要为此新增 image-aware generic normalizer。

---

### P1/P2-4 `joint_state: identity` training 合法，但 deployment export 硬编码要求 joint_state params

Training/normalizer architecture 本身支持：

```yaml
joint_state: identity
```

因为 observation passthrough 后可直接进入 StateMLP。

但 deployment checkpoint validator 当前硬编码：

```text
joint_state / action 必须存在 normalizer params
```

这与 feature-level semantic spec 冲突。

**要求：deployment normalizer state validation 改为 spec-driven。**

规则：

```text
mode == identity
    → 必须没有 params

mode in {limits, gaussian, auto}
    → 必须有完整 scale/offset
```

其中 `auto` 只会出现在 action。

---

### P2-5 PointNext encoder 与 point-cloud normalization 缺少组合约束

DP3 / DQ-RISE 支持 `encoder_type=pointnext`；ManiFlow underlying patch-tokenizer builder 支持 `pointnext_tokenizer`。

PointNext 内部使用 metric-sensitive：

```text
FPS
ball query radius
relative_xyz / radius
```

如果仅通过 CLI 把 encoder 改为 PointNext，但继续使用：

```yaml
point_cloud: limits
```

per-axis min-max 会改变 Euclidean metric，而固定 radius 仍保持原值。

当前 SAT 已正确使用：

```yaml
point_cloud: identity
```

**要求：对当前 metric PointNext family 增加 explicit fail-fast compatibility rule。**

```text
encoder_type == pointnext
encoder_type == pointnext_tokenizer
    → normalization.point_cloud 必须为 identity
```

不要 silent auto-rewrite normalization；实验 contract 必须由 config 显式声明。

---

### P2-6 Deployment normalization contract parser 不够严格

当前 restore-side extraction 主要检查：

```text
normalization is dict
version == 1
fields is non-empty dict
```

没有统一验证：

- mode 是否属于合法集合；
- `auto` 是否只用于 action；
- action 是否非法 identity；
- RGB 是否 identity；
- normalization fields 是否和实际 model numeric observation fields 精确一致；
- 是否存在额外 identity metadata field。

例如理论上：

```yaml
normalization:
  joint_state: banana
  action: auto
```

仍可能被 state validator 当成普通“需要 params”的模式。

**要求：training / checkpoint / deployment 共用同一套 semantic spec validator。**

---

## 3. 修补原则

本任务必须遵守：

1. **只修 contract validation，不改 normalization 数学。**
2. **不修改 Dataset normalization data source。**
3. **不修改 BaseAgent action normalize/unnormalize 流程。**
4. **不修改 SAT/R3D encoder 主体。**
5. **不增加新 Processor framework。**
6. **不改变已有 7 个默认 base config 的 normalization mapping。**
7. **不改变 `simple.v3` 顶层 checkpoint schema。**
8. **不做旧 checkpoint migration。**
9. **所有错误尽量 startup/export fail-fast，而不是等到 forward/runtime。**
10. **不把 `fit_field_chunks` memory optimization 混入本 correctness patch。**

---

## 4. 目标架构：一个 semantic validator + 一个 versioned contract parser

### 4.1 Shared semantic spec validator

在现有 normalization/checkpoint 模块中增加或整理一个共享纯函数，例如：

```python
def validate_normalization_spec(
    spec,
    *,
    observation_fields=None,
) -> dict[str, str]:
    ...
```

返回 canonical plain mapping。

第一版 grammar：

```text
identity | limits | gaussian | auto
```

必须验证：

1. spec 是 plain/canonical mapping；
2. key/mode 是 string；
3. mode ∈ allowed modes；
4. `auto` 只允许 `action`；
5. `joint_state` 必须显式存在；
6. `action` 必须显式存在；
7. `action != identity`；
8. `rgb` 若存在，必须 `identity`；
9. `task_text/task_name` 等 non-numeric fields 禁止进入 spec；
10. 若提供 `observation_fields`，则执行 exact field coverage：

```text
set(spec.keys()) == set(numeric_observation_fields) | {"action"}
```

即同时拒绝：

```text
missing numeric observation field
extra/nonexistent normalization field
```

不要只检查 missing。

### 4.2 Versioned normalization contract parser

与现有：

```python
make_normalization_contract(spec)
```

配套增加严格 parser，例如：

```python
def parse_normalization_contract(
    value,
    *,
    observation_fields=None,
) -> dict[str, str]:
    ...
```

必须验证：

```text
exact keys == {version, fields}
version == NORMALIZATION_CONTRACT_VERSION
fields 通过 validate_normalization_spec()
```

返回 canonical plain `fields` mapping。

最终形成：

```text
validated spec
    ↓
make_normalization_contract
    ↓
checkpoint / deployment artifact
    ↓
parse_normalization_contract
    ↓
validated spec
```

避免 training / deployment 各自实现一套 mode grammar。

---

## 5. Training config validation 修复

### 5.1 `resolve_normalization_spec`

应复用 shared semantic validator，不再独立维护一套 mode grammar。

推荐：

```python
raw = ...  # OmegaConf -> plain mapping
return validate_normalization_spec(raw)
```

### 5.2 observation exact coverage

现有 `_validate_normalization_config(cfg)` 从：

```text
missing = dataset_modalities - spec.keys
```

升级为 exact coverage。

单任务：

```text
numeric observation fields = dataset.sensor_modalities - nonnumeric fields
```

MultiTask：

- 聚合所有 child `sensor_modalities`；
- child numeric field set 必须与 shared policy normalization observation field set 一致；
- 发现 task 间 observation feature set 不一致时 fail-fast，而不是构造一个 union 后静默继续。

然后验证：

```text
normalization keys == numeric observation fields + action
```

### 5.3 architecture-dependent compatibility

增加小型 helper，例如：

```python
def _validate_encoder_normalization_contract(cfg, spec):
    ...
```

当前至少：

```text
agent.encoder_type in {pointnext, pointnext_tokenizer}
    → point_cloud must be identity
```

SAT 原有 agent-target special case可被这一通用规则覆盖；不要保留两套重复规则。

如果某模型的 PointNext config 不通过 `agent.encoder_type` 暴露，必须基于当前真实 config/source 找到最小稳定判据，不要猜测 class name。

---

## 6. Deployment checkpoint/config reconciliation

### 6.1 Saved checkpoint contract 是 fitted state 的语义来源

新增专用 helper，例如：

```python
def _reconcile_normalization_contract(
    checkpoint,
    cfg_plain,
    observation_fields,
):
    ...
```

步骤：

```text
1. 读取 checkpoint.resume_contract.agent.normalization
2. parse_normalization_contract(saved, observation_fields=...)
3. validate/resolve cfg_plain.normalization
4. 构造 current versioned contract
5. saved != current → InvalidCheckpointError
6. 返回 saved canonical contract/spec
```

禁止：

```text
checkpoint weights 来自 A semantics
artifact metadata 单独从 config B semantics 重新生成
```

### 6.2 Artifact contract 使用 saved canonical contract

`_build_inference_config()` 不应该自行从未经 checkpoint reconciliation 的 config 重建最终 artifact normalization contract。

可选两种实现，优先选择更小的：

A. 传入已经 reconcile 的 `normalization_contract`；

```python
_build_inference_config(..., normalization_contract=...)
```

B. 由 caller 构建 inference 后显式设置。

要求只有一个 canonical source，避免重复生成。

### 6.3 qualify direct path 必须走相同 reconciliation

`deployment/qualify.py::restore_direct_policy()` 当前也会读 config 并 attach normalization spec。

必须复用 exporter 的 normalization reconciliation，确保：

```text
checkpoint contract != config contract
```

时 direct branch 也在 prediction 前失败。

否则 direct/export parity 可能在相同错误 metadata 下同时通过。

---

## 7. Deployment state validation 改为 spec-driven

现有 export-side `_validate_normalizer_state(...)` 增加 normalization spec 参数：

```python
_validate_normalizer_state(
    state_dict,
    observation_fields,
    action_dim,
    normalization_spec,
)
```

逻辑：

```python
expected_param_fields = {
    key for key, mode in normalization_spec.items()
    if mode != "identity"
}
```

随后要求：

### identity field

```text
must NOT have scale/offset
```

### non-identity field

```text
must have exactly scale + offset
finite
scale non-zero
correct feature dimension
```

维度：

```text
action → action_dim
observation → observation contract feature dim
```

这应自然支持：

```text
joint_state: identity
rgb: identity
point_cloud: identity
```

而不是再 hard-code：

```text
joint_state/action always require params
rgb always special-cased separately
```

对于 RGB，semantic validator 已经规定必须 identity，因此 state validator 只需要按 generic identity rule 工作。

Restore-side `validate_normalizer_state(agent.normalizer, spec)` 继续保留，作为 strict state load 后的 module-level integrity check。

---

## 8. 不应修改的现有行为

本 patch 后 7 个默认 config 必须保持：

```text
DP:
  joint_state: limits
  action: auto
  rgb: identity

DP3:
  joint_state: limits
  action: auto
  point_cloud: limits

DQ-RISE:
  joint_state: limits
  action: auto
  point_cloud: limits

ManiFlow:
  joint_state: limits
  action: auto
  point_cloud: limits

R3D:
  joint_state: limits
  action: auto
  point_cloud: limits

SAT:
  joint_state: limits
  action: auto
  point_cloud: identity

MultiTask DiT:
  joint_state: limits
  action: auto
  rgb: identity
```

因此默认实验的 fitted scale/offset 不应发生变化。

---

## 9. Targeted tests

必须补充/更新测试，至少覆盖以下情况。

### 9.1 Spec grammar

```text
action: identity → reject
action: auto → pass
action: limits → pass
action: gaussian → pass

auto on non-action → reject

rgb: identity → pass
rgb: limits → reject
rgb: gaussian → reject

unknown mode → reject
nonnumeric field → reject
extra normalization field → reject
missing numeric observation field → reject
```

### 9.2 Joint-state identity

```text
joint_state: identity
→ spec valid
→ builder does not register joint_state params
→ BaseAgent observation passthrough works
→ deployment state validation accepts missing joint_state params
```

### 9.3 PointNext compatibility

至少：

```text
DP3 pointnext + point_cloud: limits → reject
DP3 pointnext + point_cloud: identity → pass

DQ-RISE pointnext + point_cloud: limits → reject

SAT default → pass
```

若 ManiFlow 支持 config override 到 `pointnext_tokenizer`：

```text
ManiFlow pointnext_tokenizer + limits → reject
ManiFlow pointnext_tokenizer + identity → pass
```

### 9.4 Saved-vs-current checkpoint contract

关键 regression test：

```text
checkpoint:
  point_cloud: limits
config:
  point_cloud: gaussian

state_dict params keys/shape 完全合法
→ deployment reconciliation 必须 reject
```

也测试：

```text
saved == current → pass
version mismatch → reject
missing saved normalization contract → reject
```

不做旧 checkpoint migration。

### 9.5 Deployment contract parser

至少：

```text
version 99 → reject
missing version → reject
extra top-level contract key → reject
mode banana → reject
auto on point_cloud → reject
action identity → reject
rgb limits → reject
extra nonexistent identity field → reject
missing observation field → reject
```

### 9.6 Existing regressions

原有以下测试必须继续通过：

```text
fit_field / fit_field_chunks parity
action/action_ee/use_aux_ee semantics
SAT identity behavior
R3D XYZ-only clamp
modality dropout
checkpoint roundtrip
synthetic deployment contract roundtrip
```

---

## 10. Validation Ladder

完成后依次执行：

```bash
python -m compileall -q dexmani_policy
pytest tests/ -q
```

对所有 base config：

```bash
python dexmani_policy/smoke_test.py --config-only dp
python dexmani_policy/smoke_test.py --config-only dp3
python dexmani_policy/smoke_test.py --config-only dqrise
python dexmani_policy/smoke_test.py --config-only maniflow
python dexmani_policy/smoke_test.py --config-only r3d
python dexmani_policy/smoke_test.py --config-only sat
python dexmani_policy/smoke_test.py --config-only multitask_dit
```

显式做 invalid-config startup tests，至少覆盖：

```text
action: identity
rgb: limits
DP3 pointnext + point_cloud limits
DQ-RISE pointnext + point_cloud limits
```

最后：

```bash
git diff --check
git status --short
git diff --stat
git diff
```

并搜索：

```bash
git grep -n "normalization" dexmani_policy/training dexmani_policy/common dexmani_policy/deployment
git grep -n "joint_state.*normalizer\|normalizer.*joint_state" dexmani_policy/deployment
git grep -n "rgb.*normalizer\|normalizer.*rgb" dexmani_policy/deployment
git grep -n "pointnext" dexmani_policy/configs dexmani_policy/training
```

---

## 11. 明确不在本任务范围

以下全部不做：

- `fit_field_chunks()` peak-memory / Welford 性能优化；
- train-only normalization statistics；
- action identity runtime support；
- RGB generic limits/gaussian support；
- per-task MultiTask normalizer；
- 新 tactile encoder；
- PointACT/Any3D implementation；
- SAT/R3D architecture 修改；
- full training / long benchmark；
- 旧 checkpoint migration。

如发现与上述无关的问题，只记录，不顺手重构。

---

## 12. Definition of Done

本任务完成必须满足：

1. normalization spec grammar 在 training/deployment 使用同一 shared validator；
2. versioned normalization contract 有严格 parser；
3. `action: identity` startup fail-fast；
4. RGB 非 identity startup fail-fast；
5. `joint_state: identity` training + deployment contract 均合法；
6. PointNext family + non-identity point cloud startup fail-fast；
7. deployment export/qualify strict compare checkpoint saved contract 与 resolved config；
8. artifact normalization metadata 来源与 selected weights 的 checkpoint contract 一致；
9. deployment state validation 完全 spec-driven，不再硬编码 joint_state/RGB presence；
10. existing 7 base config numerical normalization mapping 不变；
11. all targeted tests pass；
12. config-only smoke for all 7 base configs pass；
13. `git diff --check` clean；
14. 无 silent fallback、无 migration、无无关重构。

---

## 13. 最终设计不变量

修补后应满足：

```text
Resolved Config Semantic Contract
                ==
Checkpoint Saved Semantic Contract
                ==
Eval Agent Semantic Contract
                ==
Deployment Artifact Semantic Contract
```

同时：

```text
Semantic contract
    决定 field 应如何解释

model/EMA state_dict
    保存实际 fitted scale/offset
```

二者缺一不可，也不得互相替代。