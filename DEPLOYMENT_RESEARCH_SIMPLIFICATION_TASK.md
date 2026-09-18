# DexMani Policy Research-Grade Deployment Simplification 任务书

> 基线：`main@2576d0bd2322f3c4081ebcdb3c691e689d84803e`
>
> 类型：deployment correctness + research-code simplification
>
> 仓库定位：**PhD 个人研究代码，不是生产服务/多用户发布平台**
>
> 目标：在不牺牲真机实验语义正确性与论文复现性的前提下，删除生产式复杂度，把 deployment 收敛成一条窄、清楚、低维护成本的链路。

---

## 0. 先读：本任务的设计结论

本任务不是继续“加固成生产系统”，而是做两件事：

1. 修掉真正影响科研正确性的 silent semantic drift；
2. 删除对单人研究工作流价值很低的 publication / qualification / public API 复杂度。

最终希望得到：

```text
training dataset
    ↓
training checkpoint
    ├─ model/normalization semantics
    └─ deployment_data_semantics   # 训练时实际数据语义
    ↓
export
    ├─ selected checkpoint owns trained semantics
    ├─ selected/override Zarr 只能作为物理位置，semantic 必须 exact-match
    ├─ strict restore
    └─ one deterministic synthetic prediction
    ↓
immutable *-deployment.pt
    ↓
deployment_latest.pt   # atomic selector only
    ↓
dexmani_real
```

### 必须保留

- selected checkpoint owns trained model semantics；
- raw/EMA 只处理选中的一套；
- `torch.load(..., weights_only=True)`；
- strict state-dict restore；
- normalization validation；
- observation consumer validation；
- deterministic synthetic prediction before publish；
- immutable deployment artifact filename；
- `deployment_latest.pt` selector；
- runtime inspect 后 pin resolved artifact filename 再 load 的现有行为；
- `--zarr-path`，但它只能是 **semantic-equivalent relocation**。

### 明确删除/简化

- publication fsync/rollback/double-fault transaction machinery；
- `qualify.py` developer/release parity subsystem；
- 无 repo consumer 的 `LoadedPolicy.predict_action_chunk()`；
- 与上项仅配套的 `PolicySpec.chunk_size` / `DeploymentSpec.chunk_size`；
- 已完成的旧任务文档 `DEPLOYMENT_CORRECTNESS_REFACTOR_TASK.md`；
- 对上述已删除机制的 transaction/parity tests 与文档描述。

### 本任务明确不做

- 不引入 hash / signature / flock / content-addressed artifact；
- 不引入 schema registry / migration framework；
- 不给 deployment data semantics 增加版本号；
- 不做旧 checkpoint retrofit；
- 不把所有 weight grammar 塞进 `contract.py`；
- 不给每种 non-RGB modality 再写一套 artifact parser；
- 不做 decoder-specific NFE registry；
- 不做生产级 power-loss durability；
- 不修改 `dexmani_real` 的同步推理/动作调度逻辑。

---

# 1. 已 fact-check 的根因

当前 checkpoint 保存：

```text
resume_contract.agent
resume_contract.agent_config
resume_contract.dataset
resume_contract.agent.normalization
```

但 `resume_contract.dataset` 只是 **dataset constructor config**，不是训练时实际 Zarr 的物理/预处理语义。

当前 export 会重新打开一个 Zarr，并从 export-time Zarr 读取：

```text
dt
obs/state alignment
point-cloud processing_config_json
table plane
point_cloud_policy_id / sampling / transform
fingertip_config_json
EEF semantics
tactile semantics
raw RGB shape
...
```

因此可能出现：

```text
train with semantics A
    ↓
weights

export with semantics B
    ↓
artifact says B

dexmani_real runs B
```

此时 artifact 与 Real runtime 可以完全 self-consistent，但与训练权重的历史语义不一致。

最危险的确认实例：

- training 16 Hz，export Zarr 10 Hz，最终 artifact `control_dt_s` 变成 10 Hz；
- training point-cloud preprocessing A，export/live Real preprocessing B；
- `action_ee` 的 frame/components 当前没有被 training checkpoint 冻结。

这才是本任务必须修掉的 correctness gap。

---

# 2. 设计原则：一个极小的 training data semantic snapshot

不要新建一套重量级 `TrainingDataContract` framework。

在 training checkpoint 的 `resume_contract` 中只增加一个 plain dict：

```python
resume_contract["deployment_data_semantics"] = {...}
```

名字可以微调，但必须表达“**训练时实际 Real Policy Zarr 的 deployment-relevant semantics**”。

## 2.1 数据来源

必须来自**训练实际使用的 Zarr**，不能从当前 config 重新推导。

推荐最小实现：

1. `BaseDataset` 保存实际解析后的数据路径，例如：
   ```python
   self.zarr_path = str(Path(zarr_path).expanduser().resolve())
   ```
2. `build_resume_contract(...)` 从 `train_loader.dataset.zarr_path` 读取 Real Policy Zarr；
3. 使用共享 semantic extractor 生成 plain dict；
4. 写入 `resume_contract.deployment_data_semantics`。

不要从 experiment 当前 `config.yaml` 再定位数据。

对于不支持 Real deployment 的 dataset（例如 dynamic/multi-task/sim-only）：

```python
deployment_data_semantics = None
```

即可。现有 deployment export 本来就不支持这些路径，不要为此扩大任务。

## 2.2 不做旧 checkpoint migration

新代码下，Real deployment export 要求：

```text
resume_contract.deployment_data_semantics
```

存在且合法。

旧 checkpoint 缺失该字段时：

- `CheckpointStore.load()` 仍可读取；
- offline analysis 仍可使用；
- 新版 Real deployment export 明确 fail；
- 不尝试从当前 Zarr retrofit；
- 不增加 legacy branch。

错误信息应清楚说明：

> checkpoint predates deployment_data_semantics and cannot be safely exported for Real deployment.

保持 `simple.v3` 根 schema 不变；这里只扩展 `resume_contract` 内容。

---

# 3. 新增一个小型共享 semantic extractor

把当前 `deployment/export.py` 里 Real Zarr semantic parsing 的主体移到一个共享、无 deployment runtime 副作用的小模块。

推荐位置：

```text
dexmani_policy/datasets/real_policy_contract.py
```

不要放到 `deployment/contract.py`：前者描述 **training/Real Zarr data semantics**，后者描述 **persisted deployment artifact metadata**，两者职责不同。

推荐 public/internal helper：

```python
def build_real_policy_data_semantics(
    zarr_path: str | Path,
    *,
    task_name: str,
    observation_fields: Sequence[str],
    agent_config: Mapping[str, Any],
    action_key: str,
) -> dict[str, Any]:
    ...
```

如果命名更短也可以，但只保留一套逻辑。

## 3.1 semantic snapshot 必须覆盖什么

保留真正可能改变模型输入/动作物理语义的字段。

### Core

```text
schema_name
domain
task_name
dt
obs_alignment
observation_alignment
state_alignment
contact_force_source
action_semantics
```

`schema_version` 是 informational metadata，不要作为 semantic equality gate；可以继续保留在最终 artifact provenance/data metadata，但不要因为 version integer 改变而判 semantic drift。

### Action EE

Real producer 已有：

```text
action_ee_frame
action_ee_components
```

必须纳入 snapshot。

至少要求：

```text
action_ee_frame == xarm_base
action_ee_components == eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)
```

这不需要新增 ActionSpec/dataclass/schema version。

### Observation fields

对 checkpoint 实际使用的每个 observation field，保存：

```text
raw shape (excluding T)
dtype
semantic metadata
```

继续沿用当前 export 已经验证的内容：

- `joint_state`
  - shape 19
  - float32
  - joint_position / rad / xarm7_xhand12
- `point_cloud`
  - point count / feature dim
  - frame / units / RGB order
  - policy id
  - sampling
  - transform
  - `processing_config_json`
  - `point_cloud_table_plane_abcd_json`
- `rgb`
  - raw HWC uint8 shape
  - raw RGB semantics
  - camera extrinsic semantics
- `contact_force`
- `fingertip_points`
  - `fingertip_config_json`
- `eef_pose`
  - derivation / algorithm id
- `tactile_force`
  - finger/sensor/point order
  - axes
  - unit
  - verification flags

不要新发明语义；直接复用当前 exporter 已经 fact-check 的 Real producer contract。

## 3.2 时间维长度

共享 helper 在读取 Zarr 时顺手统一验证：

```text
joint_state/action/action_ee
以及所有实际 observation fields
```

第一维 `T` 一致且 > 0。

这是一个低成本补全，做在共享 Zarr parser 内，不要另建 validator 层。

---

# 4. Training：checkpoint 保存 semantic snapshot

修改：

```text
dexmani_policy/training/resume.py
```

`build_resume_contract(...)` 增加：

```python
"deployment_data_semantics": ...
```

要求：

1. 只读实际 instantiated training dataset；
2. 不从 current experiment config 重建 Zarr 路径；
3. 输出必须是 JSON/plain-Python-safe metadata；
4. Real Policy Zarr 才生成 dict；
5. unsupported dataset 返回 `None`；
6. DDP 各 rank 读取结果必须相同；
7. 不引入 hash。

建议增加一个很小的 helper，例如：

```python
def _deployment_data_semantics(train_dataset, model) -> dict[str, Any] | None:
    ...
```

不要把 Real deployment 逻辑扩散进 trainer。

---

# 5. Export：Zarr 只负责“证明 relocation 等价”

保留：

```bash
--zarr-path PATH
```

这是研究环境下有价值的跨机器/磁盘 relocation 能力。

但它的语义必须真正变成：

> locate the same semantic dataset elsewhere.

## 5.1 新逻辑

export 读取：

```text
expected = checkpoint.resume_contract.deployment_data_semantics
```

然后选择物理路径：

```text
override --zarr-path
    or
checkpoint.resume_contract.dataset.zarr_path
```

从该物理 Zarr 用**同一个共享 helper**生成：

```text
actual
```

必须：

```python
actual == expected
```

否则直接：

```text
InvalidZarrError:
selected Zarr does not match the training data semantics saved in the checkpoint
```

错误中最好指出首批差异 key，避免调试困难，但不需要通用 diff framework。

## 5.2 Artifact ownership

最终 artifact 的 model-facing data semantics 必须来自：

```text
checkpoint saved semantics
```

而不是把 override Zarr 当新的 source of truth。

允许从实际 Zarr带入纯 informational metadata（例如 schema version），但所有会改变输入/动作物理意义的字段必须以 checkpoint snapshot 为准，并且 actual 已证明等价。

因此修正当前注释/文档：

旧：

> checkpoint owns dataset/preprocessing because resume_contract.dataset is saved.

新：

> checkpoint owns dataset constructor config **and the actual training data semantic snapshot**.

---

# 6. 不新增复杂 ActionContract

不要新增：

```text
ActionSpec dataclass
action schema version
component tree
unit object model
```

只把现有 Real producer 的：

```text
action_semantics
action_ee_frame
action_ee_components
```

纳入 `deployment_data_semantics` 和最终 artifact `data_contract`。

`action_key/action_dim/control_action_dim/use_aux_ee` 继续由现有 checkpoint agent contract 管理。

这是本项目当前 action space 下足够且最简单的方案。

---

# 7. Publication：保留 selector/pinning，删除生产式事务复杂度

上一版“改成单一 deployment.pt”方案 **不要实施**。

原因已 fact-check：

- 当前 immutable `*-deployment.pt` 文件允许真机 session pin exact artifact；
- `inspect_experiment()` 会 resolve selector 到真实 filename；
- `dexmani_real` 可以随后按该 filename load；
- 这个能力很便宜且有科研价值。

因此保留：

```text
checkpoints/<checkpoint>-deployment.pt
checkpoints/deployment_latest.pt -> <checkpoint>-deployment.pt
```

但删除生产级 durability/rollback 复杂度。

## 7.1 推荐 publish 流程

```text
build payload
→ validate metadata
→ write temp artifact
→ os.replace(temp, final immutable artifact)
→ safe reload final artifact
→ strict restore
→ deterministic synthetic prediction
→ atomically replace deployment_latest.pt symlink
→ return
```

最后一步完成后**不要再执行任何可能抛异常的 verification/report/fsync**。

## 7.2 删除

从 `deployment/export.py` 删除或大幅收缩：

```text
_fsync_directory
_capture_selector
_rollback_selector
_verify_published_selector
_selector_points_at
post-swap rollback logic
double-fault handling
```

如果当前 helper 只服务上述逻辑，也一起删除。

## 7.3 selector 更新

只需：

1. 在 selector 同目录创建临时 symlink；
2. `os.replace(temp_symlink, deployment_latest.pt)`；
3. 返回。

POSIX atomic replace 足够满足单用户研究 workflow。

不做：

- directory fsync；
- power-loss durability proof；
- rollback transaction。

## 7.4 verification failure

如果 candidate 已写入 final immutable filename，但 strict restore/prediction 失败：

- 删除该 candidate；
- 不修改 selector；
- cleanup 失败时显式报错即可；
- 不需要 fsync。

---

# 8. 删除 qualify.py，保留“有价值的 parity”作为 targeted tests

当前：

```text
dexmani_policy/deployment/qualify.py
```

没有正常 researcher workflow consumer；它主要验证：

```text
direct checkpoint restore
≈
sanitized + serialized deployment restore
```

对于个人研究仓库，不值得长期维护整套：

```text
DirectRestoredPolicy
ParityReport
restore_direct_policy
direct_prediction_snapshot
qualify_policy_parity
CLI/tolerance/report/publication integration
```

因此删除：

```text
dexmani_policy/deployment/qualify.py
```

并删除 README / AGENTS / tests 中对应入口描述。

### 但保留关键测试价值

在 `tests/test_deployment_integrity.py` 中保留/改写少量 targeted parity tests，直接使用 test helper，不建立 production module：

至少覆盖：

1. representative point-cloud policy：selected checkpoint direct prediction 与 exported+restored prediction 相同；
2. DQ-RISE：`codebook_path -> None` 后 persistent codebook state 仍保证 parity；
3. pretrained point-cloud encoder：export 禁用 constructor-time pretrained loading 后 strict restore 仍一致。

如果现有 fixture 无法低成本覆盖第 2/3 项，保留现有针对 sanitizer/state 的 targeted assertions，不为了 parity 建重型测试 infra。

---

# 9. 删除无 consumer 的 future-chunk public API

已 fact-check：

- `dexmani_real` 使用 `policy.predict(...)`；
- repo 内没有 `LoadedPolicy.predict_action_chunk()` consumer；
- Real 的 `policy_trace.chunk_size` 是另一套 trace 字段，不依赖 Policy 的 `chunk_size` property。

因此删除：

```text
LoadedPolicy.predict_action_chunk()
PolicySpec.chunk_size
DeploymentSpec.chunk_size
```

同步删除对应 tests / docs。

Public runtime API 收敛为：

```python
policy = load_experiment(...)
actions = policy.predict(observation)
```

返回：

```text
[n_action_steps, control_action_dim]
```

不要添加替代 API。

---

# 10. contract.py 的职责收缩，而不是继续扩张

修改文档/注释，将：

> contract.py is the single grammar for everything

收敛为：

> contract.py is the shared grammar for persisted deployment **metadata**.

继续由它管理：

- action/window metadata；
- observation field shape/dtype container grammar；
- RGB preprocessing；
- normalization contract；
- agent nested `_target_` allowlist。

不要把下面内容塞进去：

- state-dict tensor/key grammar；
- Real point-cloud/tactile/fingertip physics semantics；
- filesystem publication transaction；
- NFE decoder-specific limits。

weight correctness 继续在 restore/load strict validation 阶段验证即可。

这意味着：

```text
inspect_experiment
```

保证 metadata 可读/合法；

```text
load_experiment
```

保证 model weights 可 strict restore + run。

对个人研究代码这是合理边界，不需要追求“任意 malformed weight 必须 inspect 阶段失败”。

---

# 11. 明确不修的低价值项

以下 finding 已审查，但在本仓库定位下不值得增加代码：

### 11.1 decoder-specific inference_steps validation

保持：

```python
inference_steps >= 1
```

即可。

非法 scheduler/NFE 组合由 warmup/inference fail-fast；Real 在 hardware workers 启动前 warmup。

### 11.2 observation field ordering

继续保留：

```python
consumed_observation_fields == artifact field tuple
```

内置 encoder 都使用固定 canonical tuple；无需改 set semantics。

### 11.3 weight grammar at inspect boundary

不修。

strict load 已是正确的 fail boundary。

### 11.4 non-RGB modality artifact-specific parser

不新增。

Zarr physics semantics 由共享 Real Policy data semantic extractor 验证；artifact parser 只保存/freeze metadata。

### 11.5 source_commit

保留当前 best-effort `producer.source_commit`。

本任务不扩展 code hash / dependency hash / git-tree hash。

如果实现过程中添加 `source_dirty: bool` 只需很少代码，可以做；但它不是 acceptance requirement，不要因此扩大 patch。

---

# 12. 测试清理与新增

目标不是保留当前所有 transaction/release-tool tests，而是让 suite 聚焦科研 correctness。

## 12.1 必须新增/保留

### Checkpoint semantics

1. Real training checkpoint 保存非空 `deployment_data_semantics`；
2. snapshot 来自实际 training Zarr，而不是 later current config；
3. `dt` 被保存；
4. point-cloud `processing_config_json` / policy id / transform 被保存；
5. `action_ee_frame/components` 被保存；
6. observation raw shape/dtype 被保存。

### Export relocation

7. 默认 checkpoint Zarr semantics 与 snapshot 相同 -> export PASS；
8. `--zarr-path` 指向不同路径但 semantic 完全相同 -> PASS；
9. override `dt` 改变 -> FAIL；
10. override point-cloud processing config 改变 -> FAIL；
11. override table plane / point-cloud transform 改变 -> FAIL；
12. override action_ee frame/components 改变 -> FAIL；
13. override raw observation shape/dtype 改变 -> FAIL；
14. task_name 不一致 -> FAIL。

### Artifact/model

15. selected raw/EMA 逻辑保留；
16. artifact safe reload；
17. strict restore；
18. synthetic prediction finite + expected shape；
19. canonical control slice；
20. DQ-RISE persistent codebook self-contained；
21. RGB deterministic validation preprocessing；
22. `LoadedPolicy.predict()` NumPy contract。

### Publication

23. successful export updates `deployment_latest.pt`；
24. verification failure leaves old selector untouched and removes failed candidate；
25. selector points to immutable artifact filename。

## 12.2 删除

删除只服务 production transaction 的 tests：

- directory fsync failure；
- post-swap fsync rollback；
- rollback itself fails；
- durable cleanup fsync；
- double-fault transaction matrix。

删除只服务 `qualify.py` CLI/report/publication 的 tests。

删除 `predict_action_chunk/chunk_size` tests。

---

# 13. 文件级修改建议

预计主要修改：

```text
dexmani_policy/datasets/base_dataset.py
dexmani_policy/datasets/real_policy_contract.py   # new
dexmani_policy/training/resume.py
dexmani_policy/deployment/export.py
dexmani_policy/deployment/contract.py             # comments/scope, minimal code only
dexmani_policy/deployment/runtime.py
dexmani_policy/deployment/restore.py              # only if chunk_size cleanup requires
tests/conftest.py
tests/test_deployment_integrity.py
README.md
AGENTS.md
```

删除：

```text
dexmani_policy/deployment/qualify.py
DEPLOYMENT_CORRECTNESS_REFACTOR_TASK.md
```

默认不要修改：

```text
docs/
dexmani_real/
```

若发现 `dexmani_real` 因删除 Policy public API 发生实际 import/consumer break，先以代码搜索证明，再做最小修改；不要主动重构 Real。

---

# 14. AGENTS.md 最终 Deployment Boundary 应收敛成什么

实现完成后，更新 `AGENTS.md`，核心只保留以下稳定规则：

```text
Checkpoint owns:
    agent / agent_config / normalization
    dataset config
    deployment_data_semantics captured from the actual training Zarr

config.yaml owns:
    experiment identity
    inference recipe

--zarr-path:
    physical relocation only;
    exported Zarr semantics must exactly match checkpoint deployment_data_semantics

Artifact:
    selected raw or EMA weights
    checkpoint-owned observation/action/data semantics
    default denoise_steps

Runtime:
    explicit inference_steps override

Export:
    build -> save -> safe reload -> strict restore -> synthetic predict
    -> atomic deployment_latest.pt selector swap
```

并明确：

- no legacy deployment retrofit；
- no hashes/signatures/locks；
- no fsync/rollback transaction；
- no qualify release subsystem；
- `contract.py` owns metadata grammar, not every possible weight/runtime invariant。

---

# 15. README 更新

只更新 researcher-facing workflow。

保留：

```text
export -> run_policy
```

说明：

- checkpoint 已保存实际 training data semantics；
- `--zarr-path` 只能用于 semantic-equivalent relocation；
- export 成功前一定 strict restore + synthetic prediction；
- `deployment_latest.pt` 是最近一次成功 export 的 atomic selector；
- runtime `--inference-steps` 仍用于 NFE ablation。

删除 `qualify.py` 相关说明。

---

# 16. 验收标准

完成后必须满足：

### Correctness

- training checkpoint 能证明真实训练 Zarr 的关键时空/预处理/action semantics；
- export-time Zarr 无法静默改变 `dt`；
- export-time Zarr 无法静默改变 point-cloud preprocessing；
- export-time Zarr 无法静默改变 action_ee physical semantics；
- current experiment config agent/dataset/normalization drift 仍不能改变旧 checkpoint model semantics；
- successful artifact 仍必须 safe reload + strict restore + synthetic predict。

### Simplicity

- 不新增 deployment schema framework；
- 不新增 migration layer；
- 不新增 hash；
- publication 无 fsync/rollback/double-fault machinery；
- `qualify.py` 删除；
- runtime 只有一个主 action API：`predict()`；
- `export.py` 明显比当前更短、更容易顺序阅读。

### Cross-repo behavior

- `deployment_latest.pt` selector 继续存在；
- `inspect_experiment().checkpoint_name` 继续返回 resolved immutable artifact filename；
- `dexmani_real` 现有 inspect -> pinned load workflow 不被破坏；
- 不修改 Real synchronous scheduling。

---

# 17. 推荐实施顺序

严格按以下顺序做，避免边改边补 workaround：

1. 新建共享 Real Policy semantic extractor；
2. training checkpoint 写入 `deployment_data_semantics`；
3. export 改成 expected(checkpoint) vs actual(selected Zarr) exact semantic comparison；
4. 用 checkpoint snapshot 作为 artifact semantic source；
5. 补 relocation/drift tests；
6. 简化 selector publication，删除 fsync/rollback helpers 与 tests；
7. 删除 `qualify.py`，把必要 parity 变成 targeted tests；
8. 删除 `predict_action_chunk/chunk_size`；
9. 更新 README / AGENTS；
10. 删除旧 `DEPLOYMENT_CORRECTNESS_REFACTOR_TASK.md`；
11. 全量运行 deployment integrity tests；
12. 最后做一次 diff-aware review，确认没有重新引入 config↔checkpoint reconciliation。

---

# 18. 验证命令

至少执行：

```bash
conda run -n policy python -m pytest tests/test_deployment_integrity.py -q
```

若共享 dataset 模块改动影响正常训练构造，再执行代表性：

```bash
conda run -n policy python dexmani_policy/smoke_test.py --config-only dp3
conda run -n policy python dexmani_policy/smoke_test.py --config-only sat
conda run -n policy python dexmani_policy/smoke_test.py --config-only dqrise
```

如果本地有对应小型测试数据且成本可接受，再跑一个代表性 full smoke。

GPU / RealSense / XHand / 真机未实际执行时，报告中明确标注 **NOT VERIFIED**。

---

# 19. Claude Code 实施约束

实施时：

- 先读 `AGENTS.md`；
- 先搜索所有将删除 symbol 的真实 consumer；
- 不修改与本任务无关的 policy architecture；
- 不格式化/重写无关文件；
- 不把当前 config 重新变成 deployment model semantics source；
- 不为了兼容旧 checkpoint 加 retrofit 分支；
- 不新增抽象层来“为未来扩展”；
- 每增加一个 helper，都应能替代现有重复代码；
- 优先删除代码，而不是在旧 transaction/reconciliation 上继续 patch。

最终汇报必须包含：

1. modified/deleted files；
2. semantic ownership 最终状态；
3. tests actually run；
4. NOT VERIFIED 项；
5. 是否发现任务书假设与当前代码不一致；若有，说明采用了什么更小的修正。
