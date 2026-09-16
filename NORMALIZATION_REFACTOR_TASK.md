# DexMani Policy 多模态归一化重构任务书

> 基线代码：`d9e097bf27cab007ec59196d3169e6b41a1e3cb6`
>
> 目标：在**不引入独立 Processor artifact、不改变 `simple.v3` checkpoint 顶层格式、尽量保持已有实验行为与旧 checkpoint 兼容**的前提下，统一 RGB、point cloud、proprioception、未来 tactile 等模态的 statistical normalization contract，并修复 SAT/R3D 当前点云归一化职责混杂问题。

---

## 1. 最终审查结论

方案审查通过，按本任务书实施。

最终架构采用：

```text
Feature-level Normalization Spec
        +
现有 LinearNormalizer（增强为 per-field fit）
        +
Encoder-specific deterministic preprocessing
```

不新增完整 `MultiModalNormalizer`、`PointCloudAdapter` 或 LeRobot-style 独立 `ProcessorPipeline`。

### 1.1 核心原则

1. **Dataset 提供 canonical data、stochastic augmentation 和 normalization statistics 的数据来源。**
2. **Policy config 决定每个 feature 使用何种 statistical normalization。**
3. **默认 normalization statistics 使用完整 replay buffer / 完整 dataset。**
   - 保持 Diffusion Policy / DP3 等官方实现和社区常见做法；
   - 不因 `val_ratio` 或 `max_train_episodes` 改变统计范围；
   - 本任务不引入 `scope: train`。
4. **`BaseAgent.normalizer` 继续拥有 fitted `scale/offset`，并随 model/EMA checkpoint 保存。**
5. **结构、空间和几何预处理不属于 generic normalization。**
   - resize / crop / ImageNet normalization；
   - FPS / KNN / ball query；
   - voxel / GridSample；
   - CenterShift / centroid shift；
   - workspace crop / RGB-D lifting / normal estimation；
   均由具体 encoder / modality processor 负责。
6. **Inference 以 checkpoint 中的 normalizer state 为统计量唯一来源，不重新访问训练数据集。**
7. **旧 resolved config 没有 `normalization:` 时走 legacy path，保证旧实验不会被新语义静默改变。**

### 1.2 本次真正改变的模型行为

当前 Policy 中，唯一有意改变 point-cloud 数值语义的是：

```text
SAT: point_cloud limits -> identity
```

DP3 / DQ-RISE(iDP3) / ManiFlow / R3D 保持现有 `limits` 语义；DP/RGB 保持 generic normalizer 不处理 RGB。

---

## 2. 当前问题

### 2.1 Dataset 决定了 model-specific normalization

当前存在：

```text
BaseDataset.get_normalizer()
PCDataset.get_normalizer()
MultiTaskDataset.get_normalizer()
```

`PCDataset.get_normalizer()` 无条件把 `point_cloud` 与 `joint_state/action` 一起按 `mode="limits", last_n_dims=1` 拟合。

结果是所有使用 `PCDataset` 的策略共享同一 point-cloud normalization，即使其 encoder contract 不同。

### 2.2 SAT 的 metric geometry 被 per-axis min-max 改变

当前 `XYZRGB` 六个 channel 分别 min-max。XYZ 对应：

```text
scale_x != scale_y != scale_z
```

所以 Euclidean geometry 被 anisotropic scaling 改变。

SAT/PointNeXT 当前使用固定：

```yaml
patch_radii: [0.05, 0.10]
```

若在此之前做 dataset-wise per-axis limits，固定半径不再具有稳定 metric semantics。

SAT 官方实现采用 point-cloud identity 路径，因此新 SAT 应切换为 `identity`。

### 2.3 R3D clamp 泄漏到所有 point-cloud policy

当前 `BaseAgent.preprocess()` 在 normalizer 之后对所有 `point_cloud` 做全 tensor clamp。

这个要求实际来自 Uni3D `PositionEmbeddingRandom` 对 XYZ `[-1, 1]` 的 contract，却影响 DP3 / ManiFlow / SAT / DQ-RISE，并连 RGB channel 一起 clamp。

必须局部化到 R3D/Uni3D encoder boundary，并仅 clamp XYZ。

### 2.4 Identity modality 与 modality dropout 错误耦合

当前 modality dropout 依赖：

```python
key in self.normalizer.params_dict
```

新架构中 `identity` 最佳实现是不注册 normalizer params，因此 RGB、SAT point cloud、未来 tactile RGB 等 identity modality 也必须正常支持 dropout。

### 2.5 MultiTaskDataset 存在独立 normalizer 构建路径

当前 `MultiTaskDataset` 在构造阶段自行 eager 计算 shared/per-task normalizer。

如果只修改 Base/PC dataset，会形成两套 normalization 构建逻辑。本任务必须同时统一 MultiTask shared-normalizer 路径。

---

## 3. 归一化与预处理职责边界

### 3.1 最终数据流

```text
Dataset
  │
  ├─ load canonical data
  ├─ stochastic augmentation
  └─ dataset-defined deterministic preprocessing（仅数据表示需要时）
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

### 3.2 Generic normalization 第一版只支持

```text
identity
limits
gaussian
auto
```

- `identity`：generic normalizer 完全不处理该 field；
- `limits`：当前 per-channel min-max affine normalization；
- `gaussian`：当前 z-score affine normalization；
- `auto`：只允许用于 `action`，保留 `action` / `action_ee` 的现有特殊规则。

### 3.3 明确禁止放入 generic normalization 的操作

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
coordinate-frame coupled transforms
```

未来新增策略时必须继续遵守这一边界。

---

## 4. Config Contract

所有**新实验 config**增加顶层：

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: limits
```

约束：

- `auto` 只允许用于 `action`；
- `identity` 不注册任何 `params_dict[field]`；
- `limits/gaussian` 默认从完整 dataset / replay buffer 拟合；
- normalization key 只允许数值 feature，不包含 `task_text/task_name` 等非数值字段。

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

DDP overlay 不重复定义 normalization，默认继承对应 base config。

### 4.2 未来扩展示例

Numeric tactile / force-torque：

```yaml
normalization:
  joint_state: limits
  action: auto
  tactile_force: gaussian
```

GelSight / DIGIT image tactile：

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

其 voxel / centroid / CenterShift / GridSample / NormalizeColor 属于各自 encoder contract。

---

## 5. Normalization Statistics Scope

### 5.1 默认行为：full dataset

新 normalization builder 默认使用完整 replay buffer：

```text
all dataset episodes
    -> min / max / mean / std
```

`train_mask`、`val_mask` 和 `max_train_episodes` **不参与默认 normalization statistics**。

理由：

1. 保持 Diffusion Policy / DP3 family 官方和社区常见行为；
2. 保持本次重构前后的数值 recipe 尽可能一致；
3. 避免一次架构重构同时改变 baseline 的 normalization protocol；
4. `max_train_episodes` 在当前代码中主要控制训练采样量，不把剩余 demo 视作严格 unseen test set。

### 5.2 本任务不新增 train-only scope

本任务不增加：

```yaml
normalization:
  scope: train
```

未来若开展严格 data-efficiency / held-out-distribution 实验，再作为独立研究 protocol 增加 `full | train` scope，并做单变量 ablation。

---

## 6. Dataset Normalization Data API

### 6.1 BaseDataset

新增统一接口：

```python
def iter_normalization_data(self, key: str):
    ...
```

第一版语义：

- 单任务 dataset 通常只 yield **一个完整 replay-buffer array**；
- 不按 episode 遍历，不应用 `train_mask`；
- `joint_state` / generic numeric observation field 直接返回完整 array；
- `action` 必须返回与训练 sample 相同的 effective action representation。

示意：

```python
def iter_normalization_data(self, key):
    if key == "action":
        yield self._get_effective_action_data()
        return

    if key in self.replay_buffer:
        yield self.replay_buffer[key]
        return

    raise KeyError(...)
```

### 6.2 Effective action semantics

统一抽出一个明确 helper，例如：

```python
def _get_effective_action_data(self):
    ...
```

必须保持当前训练语义：

- `action_key == "action"`：joint action；
- `action_key == "action_ee"`：EE action；
- `use_aux_ee == true`：`joint action + action_ee[..., :9]` concat。

不能只机械读取 `replay_buffer[action_key]` 而遗漏 auxiliary target。

### 6.3 MultiTaskDataset

新增：

```python
def iter_normalization_data(self, key):
    for dataset in self.datasets:
        yield from dataset.iter_normalization_data(key)
```

语义：

- `normalizer_mode="shared"`：builder 对所有 child dataset 的完整数据做 shared statistics；
- `normalizer_mode="per_task"`：保持当前标准训练入口 `NotImplementedError`，本任务不扩展 per-task runtime；
- child datasets 同名 field 的最后一维 shape 必须一致，否则 fail-fast。

### 6.4 Legacy normalizer 构建改为 lazy

当前 `MultiTaskDataset.__init__` eager 构建 normalizer。修改为：

- 新 config 路径不在 Dataset constructor 内计算 normalizer；
- legacy `get_normalizer()` 第一次被调用时再构建并 cache；
- 保留 legacy API，避免旧 experiment config 失效。

---

## 7. LinearNormalizer 修改

### 7.1 保留现有类和 state_dict key hierarchy

继续使用：

```python
LinearNormalizer
SingleFieldLinearNormalizer
normalizer.params_dict.<field>.scale
normalizer.params_dict.<field>.offset
```

本任务禁止迁移到新的 module tree，以保持已有 checkpoint key 路径。

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

该方法复用现有 `fit_params()`，只拟合一个 feature。

### 7.3 新增 chunk fitting，仅用于多 dataset / 大 field 聚合

新增轻量：

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

用途：

- MultiTask shared normalization；
- 未来大规模 tactile / point-cloud 多 dataset 聚合；
- 避免先 `np.concatenate` 超大 observation arrays。

实现要求：

- streaming 合并 `count/min/max/mean/variance`；
- 使用稳定的 parallel/Welford merge；
- 最终 `scale/offset` 与现有 `fit_params()` 语义一致；
- `limits` near-constant dimension 行为保持当前实现；
- `gaussian` 保持当前 z-score semantics；
- 与一次性 concatenate + fit 在合理 tolerance 内一致。

对单任务 dataset，不应为了形式统一而逐 episode streaming；直接一个完整 array 即可。

### 7.4 Identity 为零开销 passthrough

`identity` 不创建：

```text
params_dict[field]
```

直接依赖现有：

```python
if key not in self.params_dict:
    result[key] = value
```

必须保证：

```text
0 additional normalization kernel
0 scale/offset state
0 forced dtype conversion
```

尤其不能破坏 RGB uint8 fast path。

---

## 8. Normalizer Builder

在 `training/build_utils.py` 或一个小型共享 helper 中实现：

```python
resolve_normalization_spec(cfg)
build_normalizer(dataset, spec, action_key)
attach_normalization_spec(model, cfg)
```

### 8.1 build_dataset_and_normalizer

目标：

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
def build_normalizer(dataset, spec, action_key):
    normalizer = LinearNormalizer()

    for key, mode in spec.items():
        if mode == "identity":
            continue

        chunks = dataset.iter_normalization_data(key)

        if key == "action" and mode == "auto":
            action = collect_small_field(chunks)
            if action_key == "action_ee":
                normalizer["action"] = build_mixed_action_normalizer(action)
            else:
                normalizer.fit_field("action", action, mode="limits")
            continue

        normalizer.fit_field_chunks(key, chunks, mode=mode)

    return normalizer
```

`action` 维度小，可以 concatenate；point cloud/tactile 等大 field 使用 chunk fit。

### 8.3 Config validation

增加 fail-fast：

- mode 仅允许 `identity/limits/gaussian/auto`；
- `auto` 仅允许 `action`；
- `action` 必须存在且最终拥有 normalizer params；
- normalization spec 中未知 numeric field 应在 dataset build 时明确报错；
- MultiTask shared field shape 不一致时报错。

---

## 9. BaseAgent 修改

### 9.1 删除 generic point-cloud clamp

删除 `BaseAgent.preprocess()` 中：

```python
if "point_cloud" in obs:
    obs["point_cloud"] = torch.clamp(...)
```

BaseAgent 不应该知道 Uni3D 的坐标范围要求。

### 9.2 Modality dropout 与 normalization 解耦

由：

```python
if self.training and p > 0 and k in self.normalizer.params_dict:
```

改为：

```python
if self.training and p > 0:
```

并删除“没有 normalizer params 所以 dropout 无效”的 warning。

Normalization ownership 和 modality dropout 是两个独立机制。

### 9.3 normalization_spec

BaseAgent 增加简单 metadata attribute：

```python
self.normalization_spec = None
```

由 training/eval/deployment builder attach；不作为 constructor 参数，避免修改全部 Agent signature。

---

## 10. R3D / Uni3D 修改

R3D 保持：

```yaml
point_cloud: limits
```

在 `R3DObsEncoder` 或 `Uni3DPointcloudEncoder` 的明确 input boundary 对 XYZ 做 defensive clamp：

```python
xyz = pc[..., :3].clamp(-1 - 1e-6, 1 + 1e-6)
```

要求：

- 只 clamp XYZ；
- RGB 不 clamp；
- 不影响其他 point-cloud policy；
- 保持 `PositionEmbeddingRandom` 的 fail-fast range check。

---

## 11. SAT 修改

SAT 新 config：

```yaml
normalization:
  joint_state: limits
  action: auto
  point_cloud: identity
```

目标数据流：

```text
canonical metric XYZRGB
    -> generic normalizer identity
    -> PointNeXT FPS
    -> radius 0.05 / 0.10 neighborhoods
```

新 SAT 必须重新训练。

旧 SAT experiment 使用旧 resolved config（无 `normalization:`）时继续走 legacy `PCDataset.get_normalizer()`，保持旧 limits semantics，避免旧 checkpoint 被新语义误加载。

---

## 12. Legacy Compatibility

### 12.1 旧 resolved config

若：

```python
"normalization" not in cfg
```

则：

```text
继续调用 dataset.get_normalizer()
model.normalization_spec = None
不向 agent contract 添加 normalization 字段
```

目标：

- 旧实验目录无需修改 `config.yaml`；
- 旧 DP/DP3/ManiFlow/R3D/DQ-RISE/SAT checkpoint 仍按旧语义恢复；
- `load_state_dict(strict=True)` 保持可用。

### 12.2 新 config

显式存在 `normalization:` 时：

```text
走新 builder
使用 full-dataset statistics
启用 versioned normalization semantic contract
```

### 12.3 Legacy Dataset API

本任务保留：

```python
BaseDataset.get_normalizer()
PCDataset.get_normalizer()
MultiTaskDataset.get_normalizer()
```

仅用于 legacy path。

本任务不做删除 API 的 cleanup，以降低迁移风险。

---

## 13. Checkpoint / EMA / Resume Contract

### 13.1 `simple.v3` 顶层格式不变

继续：

```text
state
weights
  ├─ model
  ├─ ema_model
  ├─ optimizer
  └─ scheduler
```

不新增：

```text
normalizer.pt
stats.json
processor.json
```

### 13.2 Normalizer state 继续属于 Agent

`scale/offset` 继续存在：

```text
model.normalizer.params_dict.<field>.scale
model.normalizer.params_dict.<field>.offset
```

EMA model 同样携带。

### 13.3 normalization semantic contract

对**新 config**，`build_agent_contract(model)` 增加：

```python
"normalization": {
    "version": 1,
    "fields": model.normalization_spec,
}
```

例如 SAT：

```json
{
  "version": 1,
  "fields": {
    "joint_state": "limits",
    "action": "auto",
    "point_cloud": "identity"
  }
}
```

注意：

- contract 只保存 mode / version；
- 不保存 min/max/mean/std/scale/offset；
- 数值 state 只来自 model checkpoint。

旧 model `normalization_spec=None` 时不增加该 contract key，以兼容旧 checkpoint。

---

## 14. Eval / Inference / Deployment

### 14.1 Evaluation

`build_eval_components()` 在 instantiate agent 后：

```python
attach_normalization_spec(agent, cfg)
```

加载顺序：

```text
instantiate agent
-> attach semantic spec
-> validate checkpoint agent contract
-> load_state_dict(strict=True)
-> normalizer scale/offset restored
-> inference
```

评测不得重新构建 dataset normalizer。

### 14.2 Training Resume

新实验 resume contract 已包含 normalization semantics，因此 mode 改变必须在 load weights 前 fail-fast。

旧 experiment 因 config 无 normalization 字段，继续生成旧式 contract。

### 14.3 Deployment artifact

Deployment 需要同时保证两件事：

1. model weights 中包含实际 normalizer state；
2. deployment inference semantic contract 中包含 normalization version/mapping（新实验）。

Exporter 构建 inference config / deployment contract 时应从 resolved experiment config 或 checkpoint agent contract 携带该 mapping。

Restore：

```text
instantiate agent
-> attach normalization semantic spec
-> validate contract
-> strict-load selected model/EMA state
-> raw real observation uses checkpoint normalizer
```

部署运行时绝不能访问训练 Zarr 来重算 statistics。

---

## 15. RGB / Tactile / Future 3D Extension Rules

### 15.1 RGB

通常：

```yaml
rgb: identity
```

Generic normalizer 必须保持 uint8/float 输入原样。

`/255`、resize/crop、ImageNet mean/std、DINO/SigLIP/Qwen-VL processor 等属于 RGB encoder contract。

### 15.2 Numeric tactile / force-torque

优先根据模型设计使用：

```yaml
tactile_force: gaussian
```

或 `limits`。

### 15.3 Tactile image

GelSight / DIGIT 等：

```yaml
tactile_rgb: identity
```

后续图像处理属于 tactile encoder。

### 15.4 PointACT

```yaml
point_cloud: identity
```

其 voxel、centroid shift、RGB fixed mapping，以及与 absolute EEF state/action 的 coordinate-frame 同步由 PointACT-specific processor/encoder 处理。

### 15.5 Any3D-VLA / Concerto

```yaml
point_cloud: identity
```

CenterShift / GridSample / NormalizeColor / normals / camera-patch association 属于 Any3D/Concerto input pipeline。

---

## 16. 文件级修改清单

### 必改

1. `dexmani_policy/common/normalizer.py`
   - `fit_field()`；
   - `fit_field_chunks()`；
   - 保持原 state_dict hierarchy。

2. `dexmani_policy/datasets/base_dataset.py`
   - `_get_effective_action_data()`；
   - `iter_normalization_data(key)`，默认 full replay buffer；
   - 保留 legacy `get_normalizer()`。

3. `dexmani_policy/datasets/pc_dataset.py`
   - 保留 legacy `get_normalizer()`；
   - 新 config 标准路径不再依赖该方法。

4. `dexmani_policy/datasets/multi_task_dataset.py`
   - `iter_normalization_data(key)`；
   - legacy normalizer 改 lazy construction/cache；
   - 新 shared path 由统一 builder 构建。

5. `dexmani_policy/training/build_utils.py`
   - normalization spec resolve / validation；
   - new normalizer builder；
   - attach spec to model/EMA。

6. `dexmani_policy/agents/core/base.py`
   - 删除 generic PC clamp；
   - modality dropout 与 normalizer params 解耦；
   - `normalization_spec` metadata。

7. `dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py` 或 `uni3d.py`
   - R3D-only XYZ clamp。

8. `dexmani_policy/common/checkpoint_io.py`
   - 新实验 agent contract 增加 normalization version/mapping。

9. `dexmani_policy/training/eval_utils.py`
   - eval instantiate 后 attach normalization spec。

10. deployment export / qualify / restore 相关文件
   - 新实验携带并验证 normalization semantic contract；
   - 不改变实际 normalizer state ownership。

11. 当前所有 base Policy YAML
   - 增加顶层 `normalization:`；
   - SAT point cloud 改 `identity`。

### 不做

- 不改 `simple.v3` checkpoint root schema；
- 不新增 standalone processor artifact；
- 不引入 train-only statistics；
- 不重写 ReplayBuffer；
- 不实现 PointACT / Any3D 本体；
- 不新增 quantile/fixed-affine，除非后续有明确需求；
- 不删除 legacy `get_normalizer()` API。

---

## 17. Validation Plan

### 17.1 Config-only

对全部 current policy：

```bash
python dexmani_policy/smoke_test.py --config-only dp
python dexmani_policy/smoke_test.py --config-only dp3
python dexmani_policy/smoke_test.py --config-only dqrise
python dexmani_policy/smoke_test.py --config-only maniflow
python dexmani_policy/smoke_test.py --config-only r3d
python dexmani_policy/smoke_test.py --config-only sat
python dexmani_policy/smoke_test.py --config-only multitask_dit
```

### 17.2 Numerical parity

对 DP3 / DQ-RISE / ManiFlow / R3D：

- 新 builder 的 full-dataset `limits` scale/offset 与旧 `dataset.get_normalizer()` 一致；
- tolerance 内数值等价；
- action/action_ee/use_aux_ee 行为一致。

### 17.3 Identity zero-cost

验证：

- DP RGB identity 后 dtype/range 不被 generic normalizer 改变；
- SAT `point_cloud` 不存在 `normalizer.params_dict.point_cloud.*`；
- SAT normalization 前后 XYZ pairwise distances 一致。

### 17.4 R3D clamp isolation

验证：

- only R3D/Uni3D path clamp XYZ；
- RGB 不 clamp；
- SAT/DP3/ManiFlow/DQ-RISE 不执行该 clamp；
- `PositionEmbeddingRandom` contract 仍满足。

### 17.5 MultiTask

验证：

- shared stats 来自所有 child datasets 的完整 normalization data；
- 与当前 full-array shared normalizer 在 state/action 上数值一致；
- constructor 不为新 config eager 构建 legacy normalizer。

### 17.6 Checkpoint round-trip

对 representative DP、DP3、SAT、R3D：

```text
build -> fit normalizer -> state_dict
-> fresh agent -> strict load
-> normalization output equal
-> prediction path runnable
```

EMA 同样验证。

### 17.7 Legacy compatibility

使用至少一个旧 experiment config/checkpoint 验证：

- 无 `normalization:` 时走 legacy path；
- agent contract 不出现新增 normalization key；
- `strict=True` 正常加载；
- 旧 SAT 仍保持旧 limits semantics。

### 17.8 Deployment

至少完成静态/round-trip 验证：

- deployment artifact 包含新 normalization semantic contract；
- restore 不访问 training Zarr 计算 stats；
- state_dict 恢复后 required `action` normalizer 存在；
- identity observation field 不要求 params。

---

## 18. Definition of Done

全部满足后任务完成：

1. 新 config 的 normalization mode 由顶层 `normalization:` 唯一声明。
2. 默认 statistics 明确使用 full dataset / full replay buffer。
3. DP3 / DQ-RISE / ManiFlow / R3D full-dataset limits 与旧实现数值等价。
4. SAT point cloud 改为 identity，并重新训练新实验。
5. BaseAgent 不再含通用 point-cloud clamp。
6. R3D clamp 仅作用 XYZ 且局部化。
7. Identity modality 仍可 modality dropout。
8. MultiTask shared normalization 走统一 builder，不形成双轨逻辑。
9. `simple.v3` 顶层 checkpoint schema 不变。
10. Normalizer statistics 继续由 model/EMA state_dict 保存。
11. 新实验 resume/eval/deployment 对 normalization semantic mismatch fail-fast。
12. 旧 resolved config + checkpoint 仍可走 legacy path。
13. RGB uint8 fast path不受影响。
14. 未来 tactile / PointACT / Any3D-VLA 不需要再次修改 Dataset -> Normalizer -> Checkpoint 主干。

---

## 19. 实施顺序

建议严格按以下顺序实施，降低回归定位成本：

```text
P0  normalizer per-field API + tests
 ↓
P1  BaseDataset / MultiTask normalization-data API
 ↓
P2  unified builder + config validation + current YAML
 ↓
P3  BaseAgent dropout 解耦 + 删除 global clamp
 ↓
P4  R3D XYZ clamp localize + SAT identity
 ↓
P5  resume/eval/deployment normalization contract
 ↓
P6  legacy checkpoint + full smoke regression
```

每一步均应先完成 targeted unit/static test，再进入下一步；不在本任务中顺带进行无关架构重构。

---

## 20. 最终工程约束

后续任何新模态/新策略都按三个问题归类：

1. **Canonical data 是什么？** —— Dataset contract。
2. **需要什么 statistical normalization？** —— `normalization:` + `LinearNormalizer`。
3. **还需要什么结构/几何/预训练 backbone preprocessing？** —— Encoder contract。

只要保持这三层分离，RGB、多摄像头、numeric tactile、GelSight、PointACT、Any3D-VLA、Concerto、PointTransformer 等后续扩展都不应再次修改 Dataset -> Normalizer -> Checkpoint 的主干架构。
