# DexMani Policy 多模态归一化重构任务书

> 基线代码：`d9e097bf27cab007ec59196d3169e6b41a1e3cb6`
>
> 目标：在**不兼容旧 config / 旧 checkpoint** 的前提下，以最小改动建立统一、明确、高效的 feature-level normalization contract；保持 `simple.v3` checkpoint 顶层格式不变，并修复 SAT/R3D 当前点云归一化职责混杂问题，同时为 RGB、tactile、PointACT、Any3D-VLA 等后续模态/策略扩展提供稳定主干。

---

## 1. 最终结论

本任务采用：

```text
Feature-level Normalization Spec
        +
现有 LinearNormalizer（增强为 per-field fit）
        +
Encoder-specific deterministic preprocessing
```

明确**不引入**：

```text
MultiModalNormalizer
PointCloudAdapter
独立 Processor checkpoint/artifact
完整 LeRobot-style ProcessorPipeline
legacy normalization fallback
旧 checkpoint migration
```

### 1.1 核心原则

1. **Dataset 只负责 canonical data、stochastic augmentation，以及提供 normalization statistics 的数据来源。**
2. **顶层 Policy config 明确决定每个 feature 的 statistical normalization mode。**
3. **默认 normalization statistics 使用完整 replay buffer / 完整 dataset。**
   - 与 Diffusion Policy / DP3 family 官方实现和社区常见做法一致；
   - `val_ratio`、`train_mask`、`max_train_episodes` 不改变默认统计范围；
   - 本任务不增加 `scope: train`。
4. **`BaseAgent.normalizer` 继续拥有 fitted `scale/offset`，并随 model/EMA checkpoint 保存。**
5. **结构、空间、几何、预训练 backbone 输入处理不属于 generic normalization。**
6. **推理只依赖 config + checkpoint；不得重新访问训练数据集计算 statistics。**
7. **所有当前 Policy config 必须显式声明 `normalization:`；缺失即 fail-fast。**
8. **旧 config / checkpoint 不要求兼容，不保留双轨逻辑。**

### 1.2 本次有意改变的模型行为

当前 Policy 中，只有 SAT 的 point-cloud numerical contract 有意变化：

```text
SAT: point_cloud limits -> identity
```

其余当前策略保持原有数值 recipe：

```text
DP                RGB identity
DP3               point_cloud limits
DQ-RISE / iDP3    point_cloud limits
ManiFlow          point_cloud limits
R3D               point_cloud limits
MultiTask DiT     RGB identity
```

---

## 2. 当前问题

### 2.1 Dataset 决定了 model-specific normalization

当前存在：

```text
BaseDataset.get_normalizer()
PCDataset.get_normalizer()
MultiTaskDataset.get_normalizer()
```

特别是 `PCDataset.get_normalizer()` 无条件把 `point_cloud` 与 `joint_state/action` 一起以：

```python
last_n_dims=1
mode="limits"
```

拟合，导致所有 `PCDataset` Policy 被迫共享同一 point-cloud normalization contract。

这对 DP3 / ManiFlow / R3D baseline 尚可，但不适用于 SAT / PointNeXT 的 metric fixed-radius geometry。

### 2.2 SAT 的 metric geometry 被 per-axis min-max 改变

当前 XYZRGB 六个 channel 独立 min-max，因此 XYZ 实际为：

```text
scale_x != scale_y != scale_z
```

这会改变 Euclidean geometry。

SAT 当前 PointNeXT tokenizer 使用：

```yaml
patch_radii: [0.05, 0.10]
```

若先做 dataset-wise per-axis limits，固定半径失去稳定 metric semantics。

因此新 SAT 必须：

```text
point_cloud normalization = identity
```

### 2.3 R3D clamp 泄漏到所有 point-cloud Policy

当前 `BaseAgent.preprocess()` 在 normalizer 后对所有 `point_cloud` 做全 tensor clamp。

该限制实际上来自 Uni3D `PositionEmbeddingRandom` 对 XYZ `[-1, 1]` 的要求，却影响 DP3 / ManiFlow / SAT / DQ-RISE，且连 RGB channel 一起 clamp。

必须局部化到 R3D observation encoder，并仅处理 XYZ。

### 2.4 Identity modality 与 modality dropout 错误耦合

当前 modality dropout 依赖：

```python
key in self.normalizer.params_dict
```

但新设计中：

```text
identity = 不注册 params_dict[field]
```

因此 RGB、SAT point cloud、未来 tactile RGB 等 identity modality 仍必须正常支持 dropout。

### 2.5 MultiTaskDataset 具有第二套 normalizer ownership

当前 `MultiTaskDataset` 在 Dataset 内部 eager 构建 shared/per-task normalizer。

这会与新的统一 builder 形成双轨逻辑，必须删除 Dataset-side normalizer ownership。

---

## 3. 最终职责边界

### 3.1 数据流

```text
Dataset
  │
  ├─ load canonical data
  ├─ stochastic augmentation
  └─ dataset-defined representation preprocessing（仅数据表示需要时）
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
  ├─ tactile     : limits / gaussian / identity
  └─ rgb         : normally identity
  │
  ▼
ObsEncoder
  │
  ├─ RGB backbone processor / ImageNet normalization
  ├─ PointNeXT FPS / fixed-radius local geometry
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

### 3.2 Generic normalization 第一版只允许

```text
identity
limits
gaussian
auto
```

定义：

- `identity`：generic normalizer 原样通过，不注册任何参数；
- `limits`：沿用当前 per-channel min-max affine transform；
- `gaussian`：沿用当前 z-score affine transform；
- `auto`：只允许用于 `action`，根据 action representation 选择现有 action normalizer 逻辑。

### 3.3 禁止进入 generic normalization 的操作

以下全部属于 Encoder / modality-specific preprocess：

```text
resize / crop
ImageNet mean/std / pretrained image processor
FPS / KNN / ball query
voxelization / GridSample
CenterShift / centroid subtraction
workspace crop
normal estimation
RGB-D lifting
camera fusion
coordinate-frame coupled transform
```

未来新增策略不得把这些操作伪装成 normalization mode。

---

## 4. Config Contract

所有当前和未来 Policy config **必须**存在顶层：

```yaml
normalization:
  joint_state: limits
  action: auto
```

根据实际 observation 再增加 feature。

### 4.1 校验规则

`validate_config(cfg)` 必须验证：

1. `normalization` 必须存在且为 mapping；
2. mode 只能是：
   ```text
   identity | limits | gaussian | auto
   ```
3. `auto` 只能用于 `action`；
4. `joint_state` 和 `action` 必须显式存在；
5. 每个 `dataset.sensor_modalities` 中的数值 observation feature 必须在 normalization spec 中显式声明；
6. `task_text/task_name` 等非数值字段不能放入 normalization；
7. `identity` 是显式 contract，不能用“省略 key”代替 config declaration；
8. DDP overlay 不重复定义 normalization，继承 base config。

### 4.2 当前 Policy mapping

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

#### DQ-RISE / iDP3

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: limits
```

#### ManiFlow

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

#### MultiTask DiT

```yaml
normalization:
  joint_state: limits
  action: auto
  rgb: identity
```

### 4.3 未来扩展示例

Numeric tactile / force-torque：

```yaml
normalization:
  joint_state: limits
  action: auto
  tactile_force: gaussian
```

GelSight / DIGIT：

```yaml
normalization:
  joint_state: limits
  action: auto
  tactile_rgb: identity
```

PointACT / Any3D-VLA / metric 3D backbone：

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: identity
```

其 voxel / centroid / CenterShift / GridSample / NormalizeColor 由各自 encoder/preprocessor 处理。

---

## 5. Normalization Statistics Scope

### 5.1 默认使用 full dataset

所有 `limits/gaussian` statistics 默认来自完整 replay buffer：

```text
all dataset episodes
    -> min / max / mean / std
```

以下字段**不改变**默认 normalization statistics 范围：

```text
train_mask
val_mask
val_ratio
max_train_episodes
```

理由：

1. 与 Diffusion Policy / DP3 family 官方实现和社区常见实践一致；
2. 保持 DP3 / ManiFlow / R3D 等 baseline 重构前后 numerical recipe 不变；
3. 避免一次架构清理同时引入新的 data-efficiency protocol。

### 5.2 本任务不实现 train-only scope

不增加：

```yaml
normalization:
  scope: train
```

未来如研究严格 data-efficiency 或 held-out distribution，再独立增加 `full | train` scope 并作为实验变量验证。

---

## 6. Dataset Normalization Data API

### 6.1 删除 Dataset-side normalization policy

本任务完成后删除：

```text
BaseDataset.get_normalizer()
PCDataset.get_normalizer()
MultiTaskDataset.get_normalizer()
BaseDataset._get_normalizer_data()
MultiTaskDataset._compute_shared_normalizer()
MultiTaskDataset.normalizer / normalizers ownership
```

Dataset 不再构建或保存 normalizer。

### 6.2 BaseDataset 新增统一数据接口

新增：

```python
def iter_normalization_data(self, key: str):
    ...
```

第一版语义：

- 单任务 dataset 通常只 yield 一个完整 replay-buffer array；
- 不按 episode 切分；
- 不应用 `train_mask`；
- generic numeric observation field 直接 yield `replay_buffer[key]`；
- `action` yield 与训练 target 完全一致的 effective action representation。

推荐同时抽取：

```python
def _get_effective_action_data(self):
    ...
```

### 6.3 Effective action semantics

必须保持当前行为：

```text
action_key == action
    -> joint action

action_key == action_ee
    -> EE action

use_aux_ee == true
    -> joint action + action_ee[..., :9]
```

`auto` action normalization 规则：

- `action_key == "action_ee"`：继续使用 `build_mixed_action_normalizer()`；
- 其他 action representation：对 effective action 使用 `limits`；
- `use_aux_ee=true` 的 concat action 继续按现有 limits 语义处理。

### 6.4 MultiTaskDataset

`MultiTaskDataset` 不再拥有：

```text
normalizer_mode
normalizer
normalizers
```

当前标准 MultiTask 只支持 **shared normalization**。

新增：

```python
def iter_normalization_data(self, key):
    for dataset in self.datasets:
        yield from dataset.iter_normalization_data(key)
```

要求：

- builder 对所有 child dataset 的完整 field 数据构建 shared statistics；
- 同名 feature 的最后一维 shape 必须一致，否则 fail-fast；
- 从 `multitask_dit.yaml` 删除 `normalizer_mode`；
- 本任务不实现 per-task normalization，未来确有需求时再设计 runtime routing。

---

## 7. LinearNormalizer 修改

### 7.1 保留现有类和 state_dict hierarchy

继续使用：

```python
LinearNormalizer
SingleFieldLinearNormalizer
normalizer.params_dict.<field>.scale
normalizer.params_dict.<field>.offset
```

不创建新的 module tree，以保持代码简单并让 checkpoint 中 fitted statistics 仍归 Agent 所有。

### 7.2 新增 per-field fit

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

直接复用现有 `fit_params()`。

### 7.3 新增 chunk fit

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

只用于：

```text
MultiTask shared normalization
未来多 dataset point cloud / tactile 聚合
```

要求：

- 不先 concatenate 巨大 observation arrays；
- streaming 合并 `count/min/max/mean/variance`；
- 使用稳定的 Welford/parallel variance merge；
- `limits` near-constant behavior 与当前 `fit_params()` 一致；
- `gaussian` semantics 与当前实现一致；
- 与 concatenate + existing fit 在合理 tolerance 内一致。

单任务 dataset 不做逐 episode streaming，直接用完整 array。

### 7.4 Identity = 零开销 passthrough

对于：

```yaml
rgb: identity
point_cloud: identity
```

不创建：

```text
params_dict[field]
```

利用现有：

```python
if key not in self.params_dict:
    result[key] = value
```

因此 identity 必须保持：

```text
0 normalization kernel
0 scale/offset state
0 dtype conversion
```

尤其不能破坏 RGB uint8 fast path。

---

## 8. 统一 Normalizer Builder

在 `training/build_utils.py` 或一个轻量共享 helper 中增加：

```python
resolve_normalization_spec(cfg)
build_normalizer(dataset, spec, action_key)
attach_normalization_spec(model, cfg)
```

### 8.1 build_dataset_and_normalizer

改为唯一标准路径：

```python
dataset = hydra.utils.instantiate(cfg.dataset)
spec = resolve_normalization_spec(cfg.normalization)
normalizer = build_normalizer(
    dataset=dataset,
    spec=spec,
    action_key=cfg.action_key,
)
return dataset, normalizer
```

不存在 fallback。

### 8.2 build_normalizer 规则

伪代码：

```python
normalizer = LinearNormalizer()

for key, mode in spec.items():
    if mode == "identity":
        continue

    if key == "action" and mode == "auto":
        action = collect_small_feature(
            dataset.iter_normalization_data("action")
        )
        if action_key == "action_ee":
            normalizer["action"] = build_mixed_action_normalizer(action)
        else:
            normalizer.fit_field("action", action, mode="limits")
        continue

    chunks = dataset.iter_normalization_data(key)
    normalizer.fit_field_chunks(key, chunks, mode=mode)
```

实现可对单 chunk 做 fast path：直接 `fit_field()`，避免额外 streaming overhead。

### 8.3 normalization spec 绑定到 model/EMA

`build_model_and_ema()` 在构造 model 后统一：

```python
model.normalization_spec = normalized_plain_spec
```

EMA model 同样设置相同 spec。

该 spec 是 semantic metadata，不是 trainable state。

---

## 9. BaseAgent 修改

### 9.1 删除 global point-cloud clamp

删除：

```python
if "point_cloud" in obs:
    obs["point_cloud"] = torch.clamp(...)
```

`BaseAgent` 不再知道 Uni3D 的输入范围要求。

### 9.2 modality dropout 与 normalizer 解耦

改为：

```python
if self.training and p > 0:
    mask = ...
    v = v * ...
```

不能依赖：

```python
key in self.normalizer.params_dict
```

删除对应“没有 normalizer params 因而 dropout 无效”的 warning。

### 9.3 Normalization state validation

增加轻量 helper，例如：

```python
validate_normalizer_state(normalizer, normalization_spec)
```

规则：

- `identity` field：允许/要求没有 params entry；
- `limits/gaussian` field：必须存在 params entry；
- `action:auto`：必须最终存在 `action` params；
- scale/offset 必须 finite；
- loaded checkpoint 与 spec 不一致时 fail-fast。

Training build、eval restore、deployment restore 复用同一 validator。

---

## 10. R3D-specific 修复

R3D 继续：

```yaml
point_cloud: limits
```

但 clamp 从 `BaseAgent` 移到 `R3DObsEncoder` 输入边界。

推荐：

```python
pc = obs["point_cloud"]
if pc.dtype != torch.float32:
    pc = pc.float()

pc = pc.clone()
pc[..., :3].clamp_(
    min=-1 - 1e-6,
    max=1 + 1e-6,
)

patch_tokens, pc_pe = self.pc_encoder(...)
```

要求：

- 只 clamp XYZ；
- RGB 不 clamp；
- clamp 发生在 Uni3D 的 FPS/KNN/PE 之前；
- 其他 point-cloud Policy 不受该逻辑影响。

---

## 11. SAT-specific 修复

SAT config 改为：

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: identity
```

结果：

```text
raw metric XYZ
  -> generic normalizer identity
  -> PointNeXT FPS
  -> patch radius 0.05 / 0.10
```

必须验证：

- `normalizer.params_dict` 不包含 `point_cloud`；
- point cloud 经过 BaseAgent generic preprocessing 后数值不变；
- XYZ pairwise distance 不变；
- PointNeXT fixed-radius neighborhood 在 metric coordinate 上执行。

SAT 按新 contract 重新训练；不考虑旧 SAT checkpoint。

---

## 12. RGB / Tactile / Future 3D Contract

### 12.1 RGB

```yaml
rgb: identity
```

Generic normalizer 不处理 RGB。

现有路径保持：

```text
Dataset CPU resize/crop/augmentation
-> uint8 or [0,1]
-> vision encoder processor
-> backbone-specific normalization
```

不得因为统一 normalization 而破坏现有 RGB uint8 fast path。

### 12.2 Numeric tactile

例如：

```yaml
tactile_force: gaussian
```

由 generic normalizer 拟合 full-dataset mean/std。

### 12.3 Image tactile

例如 GelSight：

```yaml
tactile_rgb: identity
```

resize/crop/backbone normalization 留给 tactile encoder。

### 12.4 PointACT

```yaml
point_cloud: identity
```

其：

```text
voxel
centroid shift
RGB fixed transform
与 absolute state/action 的坐标同步变换
```

属于 PointACT-specific preprocessing，不进入 generic normalizer。

### 12.5 Any3D-VLA / Concerto

```yaml
point_cloud: identity
```

其：

```text
CenterShift
GridSample
NormalizeColor
normal / grid_coord construction
```

属于 Any3D/Concerto encoder contract。

---

## 13. Checkpoint Contract

### 13.1 `simple.v3` 顶层格式不变

继续：

```text
checkpoint
├─ state
│  ├─ epoch
│  ├─ global_step
│  ├─ resume_contract
│  └─ ...
└─ weights
   ├─ model
   ├─ ema_model
   ├─ optimizer
   └─ scheduler
```

不增加：

```text
normalizer.pt
stats.json
processor.json
pointcloud_stats.npz
```

### 13.2 Fitted statistics 仍属于 model state

例如：

```text
normalizer.params_dict.joint_state.scale
normalizer.params_dict.joint_state.offset
normalizer.params_dict.action.scale
normalizer.params_dict.action.offset
normalizer.params_dict.point_cloud.scale
normalizer.params_dict.point_cloud.offset
```

`identity` field 无 state entry。

### 13.3 Semantic contract 加入 agent contract

`build_agent_contract(model)` 增加：

```python
"normalization": {
    "version": 1,
    "fields": model.normalization_spec,
}
```

contract 保存**类型和语义**，不保存 min/max/mean/std 数值。

实际 statistics 只存在 `model.state_dict()`。

### 13.4 不做旧 checkpoint migration

本任务不支持：

```text
旧 config
旧 checkpoint
旧 resume contract
```

新代码只保证新 normalization contract 产生的实验内部 train/resume/eval/deployment 一致。

---

## 14. Training / DDP / EMA

### 14.1 Training

训练构建顺序：

```text
resolve config
-> validate normalization spec
-> build dataset
-> build normalizer from full dataset
-> instantiate Agent
-> load normalizer state into Agent
-> attach normalization_spec
-> build EMA copy
-> optimizer / scheduler
```

### 14.2 DDP

保持现有 normalizer broadcast 机制，但必须验证：

- rank 0 构建的 normalizer state 完整广播；
- identity field 不要求不存在的 params；
- 各 rank normalization_spec 完全一致；
- DP / DP3 / SAT 至少各做一个 targeted DDP/static contract test（有环境时）。

### 14.3 EMA

EMA model：

```text
normalizer state = main model 初始化时完整复制
normalization_spec = main model 相同 semantic metadata
```

Normalizer params 非 trainable，不做 EMA average；继续保持固定 state。

---

## 15. Eval / Inference

### 15.1 Eval build

`build_eval_components()` 在实例化 agent 后必须从 resolved config 设置：

```python
agent.normalization_spec = resolve_normalization_spec(cfg.normalization)
```

然后在 checkpoint load 前校验 agent contract。

### 15.2 Inference restore

流程：

```text
resolved experiment config
-> instantiate Agent
-> attach normalization_spec
-> load checkpoint
-> validate contract
-> strict load model / EMA state
-> validate normalizer state
-> raw env obs
-> Agent.normalizer.normalize(obs)
-> ObsEncoder preprocess
-> Policy
-> normalizer["action"].unnormalize(pred)
```

**Inference 不读取 training dataset。**

---

## 16. Deployment Artifact

当前 deployment inference config 主要携带 agent config，而 `normalization` 为顶层 experiment config，因此必须显式传播。

Deployment inference contract 增加：

```yaml
normalization:
  version: 1
  fields:
    joint_state: limits
    action: auto
    point_cloud: identity
```

要求：

1. export 从 resolved experiment config 读取 normalization spec；
2. artifact contract 保存 normalization semantic contract；
3. restore 实例化 Agent 后将 spec 设置到 `agent.normalization_spec`；
4. fitted `scale/offset` 仍只从 artifact model/EMA weights 恢复；
5. restore 后调用统一 `validate_normalizer_state()`；
6. deployment runtime 不依赖 training Zarr 计算 normalization statistics。

---

## 17. MultiTask 规则

当前 MultiTask 只支持 shared normalizer。

`multitask_dit.yaml` 删除：

```yaml
normalizer_mode: shared
```

统一由顶层：

```yaml
normalization:
  joint_state: limits
  action: auto
  rgb: identity
```

控制。

Shared statistics 来源：

```text
child dataset 1 full replay buffer
+ child dataset 2 full replay buffer
+ ...
-> shared statistics
```

本任务不实现 per-task normalization routing。

---

## 18. 文件级修改清单

### 必改

#### `dexmani_policy/common/normalizer.py`

- 新增 `fit_field()`；
- 新增 `fit_field_chunks()`；
- 保持现有 `params_dict` state_dict hierarchy；
- 保持 identity passthrough；
- 增加 normalization-state validation helper（也可放 build/checkpoint util，但只保留一份实现）。

#### `dexmani_policy/datasets/base_dataset.py`

- 删除 Dataset-side `get_normalizer()`；
- 删除 `_get_normalizer_data()`；
- 新增 `_get_effective_action_data()`；
- 新增 `iter_normalization_data(key)`。

#### `dexmani_policy/datasets/pc_dataset.py`

- 删除 `get_normalizer()`；
- 保留数据加载/augmentation职责。

#### `dexmani_policy/datasets/multi_task_dataset.py`

- 删除 `normalizer_mode`；
- 删除 normalizer/normalizers ownership；
- 删除 `_compute_shared_normalizer()` / `get_normalizer()`；
- 新增 `iter_normalization_data(key)`。

#### `dexmani_policy/training/build_utils.py`

- 新增/复用 normalization spec resolver；
- 新增统一 `build_normalizer()`；
- `build_dataset_and_normalizer()` 只走新路径；
- build model/EMA 时绑定相同 normalization spec；
- config validation 强制 `normalization:`。

#### `dexmani_policy/agents/core/base.py`

- 删除 global point-cloud clamp；
- modality dropout 与 normalizer params 解耦；
- 删除 obsolete warning。

#### `dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py`

- 增加 R3D-only XYZ clamp。

#### `dexmani_policy/common/checkpoint_io.py`

- `build_agent_contract()` 增加 versioned normalization contract。

#### `dexmani_policy/training/eval_utils.py`

- eval agent 实例化后绑定 normalization spec；
- load 后验证 normalizer state。

#### `dexmani_policy/deployment/export.py`

- deployment inference contract 显式携带 normalization spec。

#### `dexmani_policy/deployment/restore.py`

- restore agent 后绑定 normalization spec；
- strict weight load 后验证 normalizer state。

#### 当前所有 base config

- `dp.yaml`
- `dp3.yaml`
- `dqrise.yaml`
- `maniflow.yaml`
- `r3d.yaml`
- `sat.yaml`
- `multitask_dit.yaml`

全部加入显式 `normalization:`。

### 需要同步检查

- `train.py`
- `train_ddp.py`
- `training/resume.py`
- `smoke_test.py`
- deployment qualify/export/restore tests
- 所有直接调用 `dataset.get_normalizer()` 或依赖 `normalizer_mode` 的代码

---

## 19. Validation Matrix

### 19.1 Config-only

所有当前 config：

```bash
python dexmani_policy/smoke_test.py --config-only dp
python dexmani_policy/smoke_test.py --config-only dp3
python dexmani_policy/smoke_test.py --config-only dqrise
python dexmani_policy/smoke_test.py --config-only maniflow
python dexmani_policy/smoke_test.py --config-only r3d
python dexmani_policy/smoke_test.py --config-only sat
python dexmani_policy/smoke_test.py --config-only multitask_dit
```

要求：

- normalization spec resolve 正确；
- 缺字段/非法 mode fail-fast；
- Hydra targets 正常。

### 19.2 Full-dataset numerical parity

对 DP3 / DQ-RISE / ManiFlow / R3D：

```text
新 limits scale/offset
≈
重构前完整 replay-buffer limits scale/offset
```

允许合理浮点误差。

### 19.3 Identity correctness

DP RGB：

```text
dtype unchanged
value unchanged
uint8 fast path preserved
```

SAT point cloud：

```text
value unchanged
XYZ pairwise distance unchanged
normalizer.params_dict has no point_cloud entry
```

### 19.4 R3D clamp

验证：

```text
only XYZ clamped
RGB unchanged by clamp
Uni3D PE bounds valid
other PC policies not clamped
```

### 19.5 Modality dropout

构造 identity point cloud / RGB 且 `p=1.0`：

```text
training -> dropped
inference -> not dropped
```

与 normalizer params presence 无关。

### 19.6 MultiTask

验证 shared stats 等价于：

```text
concat(all child full datasets) -> existing fit
```

并验证 shape mismatch fail-fast。

### 19.7 Checkpoint roundtrip

新训练产生的 checkpoint：

```text
save
-> fresh Agent
-> attach same normalization spec
-> strict load
-> validate normalizer state
```

验证 model 与 EMA。

### 19.8 Resume contract mismatch

人工改：

```text
point_cloud: limits
```

为：

```text
point_cloud: identity
```

加载 checkpoint 必须在 rollout 前 fail-fast。

### 19.9 Inference without Dataset

只提供：

```text
resolved config + checkpoint
```

必须完成：

```text
Agent construction
normalizer restore
observation normalization
predict_action
Action unnormalize
```

不得访问 training dataset statistics。

### 19.10 Deployment

验证：

```text
export
-> artifact contract carries normalization spec
-> restore
-> strict weight load
-> normalizer validation
-> prediction verification
```

---

## 20. Definition of Done

全部满足才完成：

- [ ] 所有当前 config 显式包含 `normalization:`。
- [ ] config 缺少 normalization 时 fail-fast，不存在 fallback。
- [ ] Dataset 不再决定 normalization mode，也不持有 normalizer。
- [ ] MultiTask 不再有 `normalizer_mode` 或 per-task normalizer ownership。
- [ ] `LinearNormalizer` 支持 per-field fit 与 chunk fit。
- [ ] 默认 statistics 来自完整 replay buffer / full dataset。
- [ ] DP / DP3 / DQ-RISE / ManiFlow / R3D baseline normalization recipe 保持一致。
- [ ] SAT point cloud 改为 identity，并重新训练。
- [ ] BaseAgent 不再全局 clamp point cloud。
- [ ] R3D 仅在自身 encoder boundary clamp XYZ。
- [ ] modality dropout 支持 identity modality。
- [ ] RGB uint8 fast path 不受 generic normalizer 影响。
- [ ] checkpoint `simple.v3` 顶层格式不变。
- [ ] fitted stats 只保存在 model/EMA state，不新增 sidecar artifact。
- [ ] normalization semantic contract 进入 train/eval/deployment contract。
- [ ] inference/deployment 不依赖 training dataset 重算 statistics。
- [ ] 不包含旧 config/checkpoint compatibility、migration 或 legacy 分支。
- [ ] config-only smoke 全部通过。
- [ ] 有环境时 representative full smoke 覆盖 DP、DP3、R3D、SAT、MultiTask。

---

## 21. 非目标

本任务不做：

```text
旧 checkpoint migration
legacy config compatibility
train-only normalization scope
PointACT 实现
Any3D-VLA 实现
tactile encoder 实现
quantile normalization
fixed-affine generic mode
ReplayBuffer 重写
完整 LeRobot ProcessorPipeline
checkpoint simple.v3 schema 升级
额外 sidecar normalization artifact
```

这些均应在有明确研究/工程需求时单独实施。

---

## 22. 最终工程规则

后续新增任意模态或 3D policy，只回答三个问题：

1. **Dataset 的 canonical representation 是什么？**
2. **该 feature 需要何种 statistical normalization？**
   - `identity / limits / gaussian / action:auto`
3. **Encoder 还需要什么结构/几何/backbone-specific preprocessing？**

只要保持：

```text
Dataset data contract
        ≠
Statistical normalization contract
        ≠
Encoder preprocessing contract
```

RGB、多摄像头、numeric tactile、GelSight、PointACT、Any3D-VLA、Concerto、PointTransformer 等后续扩展都不应再次修改 Dataset -> Normalizer -> Checkpoint 的主干架构。
