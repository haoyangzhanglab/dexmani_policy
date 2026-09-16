# DexMani Policy 多模态归一化重构任务书

> 基线：`main@d9e097bf27cab007ec59196d3169e6b41a1e3cb6`
>
> 目标：在不引入独立 Processor artifact、不改变 `simple.v3` checkpoint 顶层格式、尽量保持旧实验兼容的前提下，统一 RGB、point cloud、proprioception、未来 tactile 等模态的 statistical normalization contract，并修复当前 SAT/R3D 点云归一化职责混杂问题。

---

## 1. 审查结论

方案审查通过，建议实施。

最终方案不是新增 `PointCloudAdapter`、`MultiModalNormalizer` 或完整 LeRobot-style `ProcessorPipeline`，而是：

```text
Feature-level Normalization Spec
        +
现有 LinearNormalizer（增强为 per-field fit）
        +
Encoder-specific deterministic preprocessing
```

核心原则：

1. **Dataset 负责 canonical data、stochastic augmentation 和 normalization statistics 的训练集数据来源。**
2. **顶层 Policy config 决定每个 feature 的 statistical normalization mode。**
3. **现有 `BaseAgent.normalizer` 继续拥有全部 fitted normalization state，并随 model/EMA checkpoint 保存。**
4. **resize、ImageNet normalization、FPS、KNN、voxel、GridSample、CenterShift、centroid shift 等结构/几何操作不属于 generic normalizer，继续由对应 Encoder/Processor 负责。**
5. **Inference 以 checkpoint 中的 normalizer state 为唯一统计量来源，不重新依赖 training dataset。**
6. **旧 resolved config 若没有 `normalization:` 字段，必须走 legacy normalizer 路径，以保证旧 checkpoint/resume/eval 尽可能保持兼容。**

审查后额外确认的必要修正：

- `MultiTaskDataset` 当前自行 eager 构建 shared/per-task normalizer，必须纳入统一架构，否则会出现两套 normalization 逻辑。
- `BaseAgent.preprocess()` 当前对所有 point cloud 全局 clamp，这是 R3D/Uni3D 特有约束泄漏，必须局部化。
- `modality_dropout` 当前错误依赖 `key in normalizer.params_dict`；identity modality（RGB、SAT point cloud、未来 tactile RGB）无法 dropout，必须解耦。
- 新 normalization semantic contract 必须贯穿 training / eval / deployment；实际 `scale/offset` 仍只保存在 `model.state_dict()` 中。

---

## 2. 当前问题

### 2.1 Dataset 决定了 model-specific normalization

当前：

```text
BaseDataset.get_normalizer()
PCDataset.get_normalizer()
MultiTaskDataset.get_normalizer()
```

Dataset 不仅提供数据，还决定 `limits` 等策略。

尤其 `PCDataset.get_normalizer()` 无条件把 `point_cloud` 与 `joint_state/action` 一起按 `last_n_dims=1, mode="limits"` 拟合，因此所有使用 `PCDataset` 的 Policy 被迫共享 per-channel dataset min-max。

这对 DP3/ManiFlow/R3D baseline 尚可，但对 SAT/PointNeXT 的 fixed-radius metric geometry 不合理。

### 2.2 Point cloud XYZ/RGB 被统一当作普通 6D feature

当前 `XYZRGB` 六个 channel 均独立 min-max，导致：

```text
x/y/z scale_x != scale_y != scale_z
```

Euclidean geometry 被 anisotropic scaling 改变。

对 SAT 当前：

```yaml
patch_radii: [0.05, 0.10]
```

固定半径是在 normalized space 中执行，失去统一 metric semantics。

### 2.3 R3D clamp 泄漏到所有 Policy

当前 `BaseAgent.preprocess()`：

```python
obs = self.normalizer.normalize(obs_dict)
if "point_cloud" in obs:
    obs["point_cloud"] = torch.clamp(...)
```

该 clamp 的真实需求来自 Uni3D `PositionEmbeddingRandom` 的 `[-1, 1]` coordinate contract，却影响 DP3 / ManiFlow / SAT / DQ-RISE 等全部 point-cloud policy，并且错误地连 RGB channels 一起 clamp。

### 2.4 Normalizer statistics 不是 train-only

当前 Base/PC/MultiTask normalizer 直接读取完整 ReplayBuffer array，没有应用实际 `train_mask`。

因此：

- `val_ratio > 0` 时 validation episode 进入 statistics；
- `max_train_episodes < total episodes` 时未参与训练的 episode 也进入 statistics。

### 2.5 Identity modality 与 modality dropout 耦合错误

当前 modality dropout 只有当：

```python
key in self.normalizer.params_dict
```

才生效。

但新架构中 `identity` 的最佳实现是**不注册 normalizer params**，因此 RGB、SAT point cloud、未来 tactile RGB 等 identity modality 必须仍可独立 dropout。

---

## 3. 最终架构与职责边界

### 3.1 数据流

```text
Dataset
  │
  ├─ load canonical data
  ├─ stochastic augmentation
  └─ deterministic dataset-level spatial preprocessing（仅数据定义需要时）
  │
  ▼
raw observation
  │
  ▼
BaseAgent.normalizer
  │  statistical normalization only
  │
  ├─ joint_state : limits / gaussian / identity
  ├─ point_cloud : limits / identity
  ├─ tactile     : gaussian / limits / identity
  └─ rgb         : normally identity
  │
  ▼
ObsEncoder
  │
  ├─ RGB backbone processor / ImageNet normalization
  ├─ PointNeXT FPS / ball query / local geometry
  ├─ R3D XYZ safety clamp / Uni3D
  ├─ future PointACT voxel / centroid transform
  ├─ future Any3D CenterShift / GridSample / NormalizeColor
  └─ future tactile-specific preprocessing
  │
  ▼
Policy
  │
  ▼
normalized action
  │
  ▼
normalizer["action"].unnormalize()
  │
  ▼
robot / env
```

### 3.2 不变量

以下操作属于 generic normalization：

```text
identity
limits
Gaussian
future fixed affine / quantile（只有明确需求时再加）
```

以下操作禁止放入 generic normalization：

```text
resize / crop
ImageNet mean/std（pretrained vision backbone contract）
FPS / KNN / ball query
voxelization / GridSample
CenterShift / centroid subtraction
workspace crop
normal estimation
RGB-D lifting
camera fusion
coordinate-frame coupled transforms
```

---

## 4. Config Contract

所有**新实验 config**增加顶层：

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: limits
```

第一版只支持：

```text
identity
limits
gaussian
auto
```

约束：

- `auto` 只允许用于 `action`；
- `identity` 表示 generic normalizer 不做任何操作，也不注册 params；
- `limits/gaussian` 必须从 train-only data 拟合；
- `normalization` 只允许 numeric feature key，不包含 `task_text/task_name`。

### 4.1 当前 Policy mapping

#### DP

```yaml
normalization:
  joint_state: limits
  action: auto
  rgb: identity
```

#### DP3

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: limits
```

#### DQ-RISE（当前默认 iDP3）

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: limits
```

#### ManiFlow（当前 PointNetDense）

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: limits
```

#### R3D

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: limits
```

#### SAT

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: identity
```

#### MultiTask DiT（当前 RGB）

```yaml
normalization:
  joint_state: limits
  action: auto
  rgb: identity
```

### 4.2 未来扩展示例

#### Force/Torque / numeric tactile

```yaml
normalization:
  joint_state: limits
  action: auto
  tactile_force: gaussian
```

#### GelSight / DIGIT image tactile

```yaml
normalization:
  joint_state: limits
  action: auto
  tactile_rgb: identity
```

#### PointACT / Any3D-VLA / metric 3D backbone

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: identity
```

其 voxel / CenterShift / centroid / NormalizeColor 等继续属于 model-specific preprocessing。

---

## 5. Legacy Compatibility

这是本任务的硬约束。

### 5.1 旧 resolved config

如果：

```python
"normalization" not in cfg
```

则：

```text
继续调用现有 dataset.get_normalizer()
不使用新 feature-spec builder
model.normalization_spec = None
```

目的：

- 旧实验目录保存的 `config.yaml` 无需人工修改；
- DP/DP3/ManiFlow/R3D/DQ-RISE 的旧 checkpoint 尽可能保持 strict-load；
- 旧 SAT checkpoint 仍按旧 `limits` 语义恢复，不能被新 SAT identity semantics 静默改变。

### 5.2 新 config

只要存在显式 `normalization:`：

```text
走新 normalization builder
使用 train-only statistics
启用 versioned normalization semantic contract
```

### 5.3 Legacy Dataset API

本任务**不要求立即删除**：

```python
BaseDataset.get_normalizer()
PCDataset.get_normalizer()
MultiTaskDataset.get_normalizer()
```

这些方法保留为 legacy compatibility path；所有新 config / 标准训练路径不再依赖其策略决策。

后续单独 cleanup 时再删除，避免本任务同时承担 API 迁移风险。

---

## 6. Dataset Normalization Data API

### 6.1 BaseDataset

新增统一接口：

```python
def iter_normalization_data(self, key: str):
    ...
```

语义：

- 只遍历 `self.train_mask == True` 的 episode；
- 每次 yield 一个 episode slice，避免构造大规模 train-only concatenate；
- `joint_state` / generic replay-buffer numeric field 直接 yield 对应 slice；
- `action` 必须返回与训练 sample 中**相同的 effective action representation**。

### 6.2 Action semantics

`key == "action"` 时必须保持现有语义：

- `action_key == "action"`：joint action；
- `action_key == "action_ee"`：EE action；
- `use_aux_ee == true`：`joint action + action_ee[..., :9]` concat。

不能直接机械返回 `replay_buffer[action_key]` 而忽略 auxiliary target。

### 6.3 MultiTaskDataset

新增：

```python
def iter_normalization_data(self, key):
    for dataset in self.datasets:
        yield from dataset.iter_normalization_data(key)
```

并满足：

- `normalizer_mode="shared"`：标准 training builder 从所有 child dataset 的 train-only chunks 构建 shared stats；
- `normalizer_mode="per_task"`：保持当前标准训练入口 `NotImplementedError` 行为，本任务不扩展 per-task runtime；
- 将当前 MultiTaskDataset eager normalizer construction 改为 **lazy legacy construction**，避免新路径实例化 dataset 时仍无意义地计算旧 normalizer；
- legacy `get_normalizer()` 第一次调用时再计算并 cache。

需要校验跨 task 同名 feature 的最后一维 shape 一致，否则 fail-fast。

---

## 7. LinearNormalizer 修改

### 7.1 保留现有类与 state_dict hierarchy

继续使用：

```python
LinearNormalizer
SingleFieldLinearNormalizer
normalizer.params_dict.<field>.scale
normalizer.params_dict.<field>.offset
```

禁止本任务重写为新的 module tree，避免破坏旧 checkpoint key path。

### 7.2 新增 per-field fitting

新增：

```python
LinearNormalizer.fit_field(
    key,
    data,
    *,
    last_n_dims=1,
    mode="limits",
    ...,
)
```

用于小规模连续 array。

新增：

```python
LinearNormalizer.fit_field_chunks(
    key,
    chunks,
    *,
    last_n_dims=1,
    mode="limits",
    ...,
)
```

用于 point cloud / tactile / multi-task 等大 field。

### 7.3 Streaming statistics

`fit_field_chunks` 必须计算：

```text
count
min
max
mean
std
```

要求：

- 不把全部 chunks concatenate 到一个大 array；
- 使用 numerically stable merge/Welford 方式；
- 最终 `scale/offset` 语义与现有 `fit_params()` 相同；
- `limits` 的 near-constant dimension 行为保持当前实现：zero-center without noise amplification；
- `gaussian` 保持当前 z-score semantics；
- 统计结果与一次性 fit 在合理 floating tolerance 内一致。

### 7.4 Identity

`identity` 不创建：

```text
params_dict[field]
```

直接依赖现有 passthrough：

```python
if key not in self.params_dict:
    result[key] = value
```

因此 identity path 必须保持：

```text
0 additional normalization kernel
0 scale/offset state
0 dtype conversion
```

尤其不能破坏 RGB uint8 fast path。

---

## 8. Normalizer Builder

在 `training/build_utils.py` 或一个小型共享 helper 中增加：

```python
resolve_normalization_spec(cfg)
build_normalizer(dataset, spec, action_key)
attach_normalization_spec(model, cfg)
```

### 8.1 build_dataset_and_normalizer

目标逻辑：

```python
dataset = hydra.utils.instantiate(cfg.dataset)

if "normalization" not in cfg:
    # legacy experiment
    normalizer = dataset.get_normalizer()
    return dataset, normalizer

spec = resolve_normalization_spec(cfg)
normalizer = build_normalizer(dataset, spec, cfg.action_key)
return dataset, normalizer
```

### 8.2 build_normalizer

伪代码：

```python
normalizer = LinearNormalizer()

for key, mode in spec.items():
    if mode == "identity":
        continue

    if key == "action" and mode == "auto":
        # preserve current action/action_ee semantics
        ...
        continue

    normalizer.fit_field_chunks(
        key,
        dataset.iter_normalization_data(key),
        last_n_dims=1,
        mode=mode,
    )
```

`auto` 不允许用于 observation field。

### 8.3 `action=auto`

必须保持当前行为：

- ordinary action → `limits`；
- `action_ee` → existing `build_mixed_action_normalizer()`；
- auxiliary effective action → 按当前 combined action semantics 拟合。

Action dim 很小，可以 concatenate train-only chunks；大 observation field 才走 streaming fit。

---

## 9. BaseAgent 修改

### 9.1 保留 `self.normalizer`

不改 checkpoint ownership。

现有：

```python
self.normalizer = LinearNormalizer()
```

继续保留。

`load_normalizer_from_dataset()` 可暂时保留原名，避免扩大调用面；它接收的实际是 builder 构造的 normalizer state。

### 9.2 删除全局 point-cloud clamp

删除：

```python
if "point_cloud" in obs:
    obs["point_cloud"] = torch.clamp(...)
```

BaseAgent 不应知道 Uni3D 的 coordinate bound。

### 9.3 修 modality dropout

当前：

```python
if self.training and p > 0 and k in self.normalizer.params_dict:
```

改成 normalization-independent：

```python
if self.training and p > 0:
```

删除“not in normalizer.params_dict therefore dropout has no effect”相关 warning。

Normalization mode 与 modality dropout 是独立机制。

---

## 10. R3D / Uni3D 修改

将 Uni3D-specific coordinate clamp 移至 R3D observation encoder boundary。

推荐：

```python
pc = obs["point_cloud"]

if pc.dtype != torch.float32:
    pc = pc.float()

pc = pc.clone()
pc[..., :3] = pc[..., :3].clamp(
    min=-1 - 1e-6,
    max=1 + 1e-6,
)
```

然后再送入：

```python
self.pc_encoder(pc, ...)
```

要求：

- 只 clamp XYZ；
- 不 clamp RGB；
- 其他 point-cloud policy 不受影响；
- Uni3D `PositionEmbeddingRandom` 的 `[-1,1]` contract 保持满足。

---

## 11. SAT 行为修复

新 SAT config：

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: identity
```

因此新训练：

```text
raw metric XYZ
→ coordinate augmentation（σ=0.002，物理单位）
→ generic normalizer passthrough
→ FPS / PointNeXT
→ radius 0.05 / 0.10 保持 metric semantics
```

注意：

- 旧 SAT experiment config 没有 `normalization:`，必须继续走 legacy `limits` path；
- 旧 SAT checkpoint 不应被解释成新 identity semantics；
- 新 SAT 是 semantic behavior change，应重新训练，不做旧权重迁移。

---

## 12. Checkpoint Contract

### 12.1 顶层格式保持不变

禁止修改：

```text
simple.v3
├── state
└── weights
    ├── model
    ├── ema_model
    ├── optimizer
    └── scheduler
```

Normalization fitted state 继续位于：

```text
weights.model.normalizer.params_dict.*
weights.ema_model.normalizer.params_dict.*
```

不新增：

```text
normalizer.pt
stats.json
processor.json
pointcloud_stats.npz
```

### 12.2 Semantic contract

对**新 config**，`build_agent_contract(model)` 增加：

```python
"normalization": {
    "version": 1,
    "fields": {
        "joint_state": "limits",
        "action": "auto",
        "point_cloud": "identity",
    },
}
```

要求：

- 只保存 mode/semantic version；
- 不保存 min/max/mean/std 数值；
- fitted statistics 仍只在 model state 中。

### 12.3 Legacy contract

旧 config：

```text
model.normalization_spec = None
```

`build_agent_contract()` 必须完全省略 `normalization` 字段，使旧 checkpoint 的 resume contract schema 不发生变化。

### 12.4 Agent spec attachment

训练和评测必须使用同一个 helper：

```python
attach_normalization_spec(model, cfg)
```

EMA model 同样 attach。

必须在 `build_resume_contract()` / inference contract validation **之前**完成。

---

## 13. Inference / Eval

### 13.1 Eval

当前 eval 流程：

```text
instantiate agent
→ validate resume_contract.agent
→ load_state_dict(strict=True)
```

新流程：

```text
instantiate agent
→ attach normalization spec from resolved config
→ validate resume_contract.agent
→ load_state_dict(strict=True)
→ validate fitted normalizer keys
```

### 13.2 Required-key validation

对显式 new spec：

```text
mode == identity  → 不要求 params_dict key
mode != identity  → checkpoint load 后必须存在 params_dict key
```

因此 inference validation 不能只检查 `action`，而应检查：

```python
required_keys = {
    key for key, mode in spec.items()
    if mode != "identity"
}
```

`auto` action 视为 required。

Legacy spec 为 `None` 时保持现有兼容检查逻辑。

### 13.3 Dataset independence

评测/推理不得为了重新计算 normalization statistics 而打开 training Zarr。

Checkpoint model state 是 fitted statistics 的唯一权威来源。

---

## 14. Deployment

当前 deployment 已经支持包括：

```text
joint_state
point_cloud
rgb
contact_force
fingertip_points
eef_pose
tactile_force
```

因此 normalization semantic contract 必须同步进入 deployment artifact。

要求：

1. exporter 从 resolved experiment config 读取顶层 `normalization`；
2. 新 artifact 的 inference/deployment contract 显式携带 versioned normalization spec；
3. restore / qualify instantiate Agent 后，先 attach normalization spec，再做 Agent/deployment contract 校验；
4. actual fitted scale/offset 仍只从 artifact model weights `state_dict` 恢复；
5. deployment 不读取 training dataset stats；
6. legacy artifact/config 没有 normalization spec 时保持当前行为。

不要把 normalization mapping 塞入各 Agent constructor，仅作为 runtime semantic metadata attach 到 model。

---

## 15. DDP / EMA

### 15.1 DDP

现有 DDP 会广播 `model.normalizer.state_dict()`；保持该机制。

新 architecture 不新增独立 processor state，因此无需新增 broadcast channel。

Identity feature 无 state，无需广播。

### 15.2 EMA

Normalizer 参数当前是 non-trainable state；EMA updater 对 `requires_grad=False` parameter 直接 copy。

保持现状。

要求 model 与 EMA 初始化时使用同一个 fitted normalizer state 和同一个 normalization spec。

---

## 16. Validation / Config Checks

`validate_config(cfg)` 增加 normalization 校验，仅在显式 `cfg.normalization` 存在时执行。

至少检查：

- mapping 非空；
- mode ∈ `{identity, limits, gaussian, auto}`；
- `auto` 只能给 `action`；
- `action` 必须存在；
- `joint_state` 若被 Policy 消费则必须显式存在；
- numeric sensor modality 建议显式声明 mode；
- `task_text/task_name` 等非 numeric field 不允许出现在 normalization spec；
- config 与 encoder-specific contract 明显冲突时 fail-fast（当前至少 SAT/R3D 可做 targeted check）。

不要在 generic validator 中硬编码未来所有 Policy recipe；只检查公共 contract 和明确的非法组合。

---

## 17. 文件级修改清单

### 必改

#### `dexmani_policy/common/normalizer.py`

- 增加 `fit_field()`；
- 增加 `fit_field_chunks()`；
- 保持现有 `params_dict` state hierarchy；
- identity 继续使用 absent-key passthrough。

#### `dexmani_policy/datasets/base_dataset.py`

- 增加 `iter_normalization_data(key)`；
- 使用 `train_mask`；
- action 返回 effective action representation；
- legacy `get_normalizer()` 保留。

#### `dexmani_policy/datasets/pc_dataset.py`

- legacy `get_normalizer()` 保留，仅供无 `normalization:` 的旧 config；
- 新标准路径不再调用它；
- 可增加明确 legacy 注释，禁止新代码依赖。

#### `dexmani_policy/datasets/multi_task_dataset.py`

- 增加 shared `iter_normalization_data()`；
- old normalizer eager construction → lazy legacy construction；
- 标准 `per_task` 行为继续保持 unsupported；
- shared new path 使用 child train-only streams。

#### `dexmani_policy/training/build_utils.py`

- `resolve_normalization_spec()`；
- `build_normalizer()`；
- legacy fallback；
- `attach_normalization_spec()`；
- model/EMA 同步 attach。

#### `dexmani_policy/agents/core/base.py`

- 删除 global point-cloud clamp；
- modality dropout 与 normalizer params 解耦；
- 保留 normalizer ownership。

#### `dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py`

- 加 R3D-only XYZ defensive clamp。

#### `dexmani_policy/common/checkpoint_io.py`

- 新 config 的 Agent contract 增加 normalization version + fields；
- legacy agent contract 不增加字段。

#### `dexmani_policy/training/eval_utils.py`

- eval Agent attach normalization spec；
- required normalizer key validation 从仅 action 扩展为 explicit-spec required keys。

#### `dexmani_policy/training/resume.py`

- 确保 model 已 attach spec 后再构造 resume contract；
- 不改变 `simple.v3` payload schema。

#### `dexmani_policy/deployment/export.py`

- deployment inference contract 传播 normalization semantic spec。

#### `dexmani_policy/deployment/restore.py`

- restore Agent attach normalization spec；
- fitted statistics 仍来自 model state。

#### `dexmani_policy/deployment/qualify.py`

- direct-policy qualification 与 artifact restore 使用相同 normalization semantics。

#### `dexmani_policy/configs/*.yaml`

新 active configs 增加显式 `normalization:`：

```text
dp
dp3
dqrise
maniflow
r3d
sat
multitask_dit
```

DDP overlays 若只继承主 config，则不重复配置。

### 不修改

- `simple.v3` checkpoint 顶层 schema；
- `docs/`（本仓库 contract 指定为冻结背景文档，本任务不需要修改）；
- action decoder / Diffusion / FlowMatch 算法；
- RGB encoder 本身的 pretrained processor contract；
- PointNeXT radius 等 architecture hyperparameter。

---

## 18. 测试与验收

### P0 — Unit / CPU

#### A. Per-field fit

验证：

```text
fit_field(array)
≈
fit_field_chunks(split(array))
```

覆盖：

```text
limits
gaussian
near-constant dims
float32 input
```

#### B. Train-only statistics

构造多个 synthetic episodes：

- train episode 正常范围；
- validation / excluded episode 注入极端 outlier；

要求新 normalizer 的 min/max/mean/std 不受 excluded episode 影响。

#### C. Identity zero-cost semantics

要求：

- `rgb` uint8 normalize 后 dtype/value/object semantics 不被 generic normalizer改变；
- identity point cloud pairwise XYZ distance 完全保持；
- identity field 不进入 `params_dict`。

#### D. Action auto

覆盖：

```text
action
action_ee
use_aux_ee
```

确认 action dim、rot6d identity segment 与当前实现一致。

#### E. MultiTask shared stats

验证只合并各 child dataset 的 train-only stream；validation/excluded episode 不进入 shared stats。

### P1 — Config-only smoke

执行：

```bash
conda run -n policy python dexmani_policy/smoke_test.py --config-only dp
conda run -n policy python dexmani_policy/smoke_test.py --config-only dp3
conda run -n policy python dexmani_policy/smoke_test.py --config-only dqrise
conda run -n policy python dexmani_policy/smoke_test.py --config-only maniflow
conda run -n policy python dexmani_policy/smoke_test.py --config-only r3d
conda run -n policy python dexmani_policy/smoke_test.py --config-only sat
conda run -n policy python dexmani_policy/smoke_test.py --config-only multitask_dit
```

### P2 — Full representative smoke

至少：

```bash
conda run -n policy python dexmani_policy/smoke_test.py dp
conda run -n policy python dexmani_policy/smoke_test.py dp3
conda run -n policy python dexmani_policy/smoke_test.py r3d
conda run -n policy python dexmani_policy/smoke_test.py sat
conda run -n policy python dexmani_policy/smoke_test.py multitask_dit
```

环境缺少 GPU/数据/预训练权重时必须标记 **NOT VERIFIED**，不能修改核心逻辑绕过。

### P3 — Checkpoint roundtrip

新 config：

```text
train/model state
→ save simple.v3
→ fresh Agent
→ attach same normalization spec
→ strict=True load
→ normalizer required-key validation
→ prediction equivalence
```

EMA 同样验证。

### P4 — Legacy compatibility

至少使用已有 experiment/resolved config 或构造等价 fixture 验证：

```text
old config without normalization
→ legacy dataset.get_normalizer path
→ old agent contract shape unchanged
→ old DP/DP3/R3D checkpoint strict-load
```

若真实旧 checkpoint 不在开发环境，标记 runtime load 为 **NOT VERIFIED**，但必须有 structural/unit coverage。

### P5 — SAT geometry

新 SAT 要求：

- `point_cloud` 不存在于 `normalizer.params_dict`；
- PointNeXT 接收 metric XYZ；
- generic BaseAgent 不 clamp；
- fixed-radius neighborhood 的输入 scale 未被 per-axis min-max 扭曲。

### P6 — R3D boundary

要求：

- R3D XYZ 在进入 Uni3D 前满足 PE bound；
- clamp 不修改 RGB；
- DP3/ManiFlow/SAT 等其他 Policy 不执行该 clamp。

### P7 — Modality dropout

对 identity modality 配置非零 dropout，验证训练态确实生效；eval 态不生效。

### P8 — Deployment

至少验证：

```text
resolved config normalization spec
→ export artifact
→ restore artifact
→ attach semantic spec
→ strict model load
→ required normalizer keys valid
→ inference 不访问 training stats
```

已有 deployment observation fields（含 tactile_force）不能因本次重构退化。

---

## 19. 性能要求

1. `identity` path 必须是零统计拟合、零 normalizer tensor op。
2. RGB uint8 fast path不得因 generic normalizer 被强制转 float。
3. point-cloud/tactile large field statistics 不允许先 concatenate 全部 train data。
4. normalizer fit 只发生在 training build，不进入 per-batch hot path。
5. inference 不执行任何 statistics fitting。
6. 不新增额外 checkpoint sidecar I/O。

---

## 20. Non-Goals

本任务不做：

- 实现 PointACT；
- 实现 Any3D-VLA；
- 实现新的 tactile encoder；
- 重写 RGB encoder processor；
- 引入 LeRobot ProcessorPipeline；
- 引入新的 PointCloudAdapter layer；
- 支持 MultiTask `normalizer_mode=per_task` 标准训练；
- 新增 quantile/fixed-affine normalization；
- 修改 replay-buffer storage 格式；
- 修改 checkpoint `simple.v3` 顶层格式；
- 迁移旧 SAT 权重到新 identity geometry。

这些能力应在本任务完成后的稳定 normalization contract 上独立扩展。

---

## 21. 推荐实施顺序

### Phase 1 — Normalizer primitive

1. `fit_field()`；
2. `fit_field_chunks()`；
3. unit tests。

### Phase 2 — Dataset stats source

1. BaseDataset train-only iterator；
2. action semantics；
3. MultiTask shared iterator + lazy legacy normalizer；
4. leakage tests。

### Phase 3 — Builder / Config

1. top-level normalization spec；
2. resolver/validator；
3. legacy fallback；
4. active configs 更新。

### Phase 4 — Runtime semantics

1. BaseAgent global clamp 删除；
2. R3D clamp 局部化；
3. modality dropout 解耦；
4. SAT identity 生效。

### Phase 5 — Checkpoint / Eval / Deployment

1. normalization semantic contract；
2. train/eval spec attachment；
3. required-key validation；
4. deployment propagation；
5. legacy compatibility。

### Phase 6 — Smoke / Regression

按 Validation Ladder 完成 CPU/unit → config smoke → representative full smoke → checkpoint/deployment roundtrip。

---

## 22. Definition of Done

满足以下全部条件才算完成：

- [ ] 新 config 的 normalization policy 由顶层 `normalization:` 唯一定义；
- [ ] Dataset 不再为新路径决定 normalization mode；
- [ ] statistics 严格来自 actual train episodes；
- [ ] MultiTask shared stats 同样 train-only；
- [ ] `identity` 不产生 normalizer state/计算；
- [ ] DP/RGB uint8 path 不退化；
- [ ] DP3/ManiFlow/R3D 新训练仍使用 point-cloud limits；
- [ ] SAT 新训练使用 metric point cloud identity；
- [ ] BaseAgent 不再包含 Uni3D-specific clamp；
- [ ] R3D 仅 clamp XYZ；
- [ ] modality dropout 与 normalization params 解耦；
- [ ] action/action_ee/use_aux_ee semantics 保持正确；
- [ ] `simple.v3` checkpoint 顶层 schema 不变；
- [ ] fitted stats 仍随 model/EMA state 保存；
- [ ] new normalization semantic contract 可在 resume/eval/deployment 前校验；
- [ ] old config 无 normalization 时走 legacy path，旧 agent contract 不变；
- [ ] inference 不依赖 training dataset statistics；
- [ ] unit/config smoke/checkpoint roundtrip 按可用环境完成，未验证 GPU 项明确标记 NOT VERIFIED。

---

## 23. 最终设计原则

今后增加任何 observation modality，先回答三个问题：

1. **Canonical representation 是什么？**
   - 例如 XYZ 单位 meter、RGB uint8/[0,1]、force/torque 单位和 axis order。
2. **是否需要 statistical normalization？**
   - `identity / limits / gaussian / auto(action only)`。
3. **Encoder 是否还有 structural / geometry preprocessing？**
   - 例如 ImageNet normalize、voxel、CenterShift、FPS、KNN、tactile image processor。

只要严格保持这三层分离，后续扩展 RGB、多摄像头、numeric tactile、GelSight、PointACT、Any3D-VLA、Concerto、PointTransformer 等策略时，都不需要再次修改 Dataset → Normalizer → Checkpoint 的主干架构。
