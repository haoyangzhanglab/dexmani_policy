# DexMani Policy Deployment Correctness Refactor 任务书

> 基线：`main@b4811ba62f18ea1b5ae5f738fd4e70a3524a4127`
>
> 类型：deployment correctness / research workflow simplification
>
> 目标：面向论文真机实验，将 deployment 收敛为一条简单、确定、可复现的链路：
>
> ```text
> training checkpoint
>     -> deployment artifact
>     -> dexmani_real run_policy
> ```
>
> 核心原则：**Checkpoint owns trained semantics; Artifact owns deployment contract; Runtime owns explicit inference-time overrides.**

---

## 1. 背景与已确认问题

当前 deployment 已具备较强的 observation/action contract、safe `torch.load(weights_only=True)`、strict state-dict restore、synthetic prediction verification 和 selector rollback；本任务不推翻这些机制。

已 fact-check 的主要问题集中在 source-of-truth 和 artifact lifecycle：

1. training checkpoint 已保存 `resume_contract.agent_config`、`resume_contract.dataset`、`resume_contract.agent.normalization`，但 export 仍从当前 `config.yaml` 重新读取/推导 agent、dataset 和 normalization 语义；这允许训练完成后 config drift 改变旧 checkpoint 的 deployment 构造语义。
2. 当前 `_reconcile_train_params()` 只比较少量 shape/action 字段，无法覆盖 `n_head` 等可能改变 forward semantics、但不改变 state-dict key/shape 的 constructor 参数。
3. `export_deployment_artifact(..., verify=False)` / CLI `--no-verify` 可以绕过 strict restore + synthetic prediction 后仍 publish `deployment_latest.pt`。
4. artifact 已 rename 到最终路径后若 verification 失败，candidate 文件会残留；同名 retry 随后触发 `FileExistsError`。
5. `_validate_agent_targets()` 目前位于 exporter；restore 从 artifact 读取 `_target_` 后直接 `hydra.utils.instantiate()`，没有在统一 contract boundary 上验证。
6. `retrofitted_train_params_fields` / `metadata_provenance` 当前没有真实 retrofit path，属于无效复杂度。
7. raw/EMA 当前先同时 canonicalize/部分重复 validate，再选择实际 deployment weights；可缩短为先选择、后只处理一套 state。
8. normalization 已有 shared semantic validator，但 exporter 仍保留一套 raw-state duplicate validation，可在 public export 永远 strict restore 后删除。

本任务的方向不是增加更多 checkpoint/config reconciliation，而是**删除双 source-of-truth**。

---

## 2. 最终 ownership 模型

### 2.1 Checkpoint-owned trained semantics

以下信息必须只来自 selected training checkpoint：

```text
resume_contract.agent
resume_contract.agent_config
resume_contract.dataset
resume_contract.agent.normalization
```

具体职责：

- `resume_contract.agent`
  - `n_obs_steps`
  - `n_action_steps`
  - `action_dim`
  - `horizon`
  - `action_key`
  - `tcp_dim`
  - `hand_dim`
  - `control_action_dim`
  - `use_aux_ee`
  - versioned normalization contract
- `resume_contract.agent_config`
  - deployment agent constructor source
- `resume_contract.dataset`
  - sensor modalities
  - RGB preprocessing source
  - default Zarr path

禁止再从当前 `config.yaml` 重新构造这些语义。

### 2.2 Current experiment config 的保留职责

当前 `config.yaml` 只保留以下 deployment-facing 职责：

1. experiment identity，尤其 `policy_name` / `task_name`；
2. non-`best` checkpoint 的当前 inference recipe（当前行为）：
   - `eval.use_ema`
   - `eval.denoise_steps`
3. `best` checkpoint 继续使用 `best_ckpt.json["inference"]`。

`config.yaml.agent`、`config.yaml.dataset`、`config.yaml.normalization` 不再决定 deployment model/data semantics。

### 2.3 Runtime-owned explicit override

`dexmani_real --inference-steps N` 保持现状：

- artifact 保存 default `denoise_steps`；
- runtime 可显式 override NFE；
- NFE 是论文 ablation 参数，不属于 immutable architecture semantics。

---

## 3. 实现任务 A：新增 checkpoint deployment source parser

在 `dexmani_policy/deployment/export.py` 内增加一个小型 private structure/helper，例如：

```python
@dataclass(frozen=True)
class _CheckpointDeploymentSource:
    agent: dict[str, Any]
    agent_config: dict[str, Any]
    dataset: dict[str, Any]
    normalization_contract: dict[str, Any]
    observation_fields: tuple[str, ...]
```

以及：

```python
def _parse_checkpoint_deployment_source(
    checkpoint: TrainCheckpoint,
) -> _CheckpointDeploymentSource:
    ...
```

要求：

1. `resume_contract` / `agent` / `agent_config` / `dataset` 必须是 plain mappings；缺失直接 `InvalidCheckpointError`。
2. 对 `resume_contract.agent` 做**内部一致性 parsing**，而不是与当前 config 对比：
   - action key 合法；
   - positive `horizon/n_obs_steps/n_action_steps/action_dim/control_action_dim`；
   - `n_obs_steps - 1 + n_action_steps <= horizon`；
   - `action_key/action_dim/control_action_dim/use_aux_ee/tcp_dim/hand_dim` 自洽；
   - normalization contract 使用现有 strict `parse_normalization_contract()`。
3. observation modalities 从 checkpoint-saved dataset/agent 解析，不从 current config 解析。
4. normalization exact field coverage 必须对 checkpoint-derived numeric observation fields 生效。
5. 返回 canonical plain metadata；不要引入新 checkpoint schema。

注意：**删除 reconciliation，不等于删除 validation。** 新 parser 验证 checkpoint 自己是否合法，但不再要求 checkpoint 与当前 `config.yaml.agent/dataset/normalization` 一致。

---

## 4. 实现任务 B：删除 config/checkpoint 双重 reconciliation

完成任务 A 后，删除/替换以下旧路径：

```text
_expected_train_params(cfg_plain)
_reconcile_train_params(checkpoint, cfg_plain)
_validate_resolved_config_contract(cfg_plain, train)
_reconcile_normalization_contract(checkpoint, cfg_plain, ...)
```

同时删除已经没有真实语义的：

```text
retrofitted
retrofitted_train_params_fields
metadata_provenance
ExportReceipt.metadata_provenance
```

要求：

- 不保留“以防未来 migration”的 dead compatibility branch；
- 本仓库当前策略是不做旧 checkpoint migration；
- `simple.v3` 保持不变。

---

## 5. 实现任务 C：selected weights first

当前 export 同时 canonicalize raw/EMA，再决定最终使用哪套 state。改为：

```python
selected_weights = "ema_model" if selected.use_ema else "model"
selected_raw = (
    checkpoint.ema_model_state
    if selected.use_ema
    else checkpoint.model_state
)
if selected_raw is None:
    raise InvalidCheckpointError(...)

selected_state = _canonicalize_state_dict(
    selected_raw,
    f"weights.{selected_weights}",
)
```

随后所有 deployment-only state validation / DQ-RISE sanitization 只针对 `selected_state`。

要求：

- artifact 仍只保存 selected state；
- `producer.selected_weights` 保留；
- `producer.source_checkpoint` 保留；
- 删除 raw/EMA 双 sanitize 分支。

---

## 6. 实现任务 D：checkpoint-owned constructor + dataset/preprocess

`_sanitize_agent_config()` 的输入改成 checkpoint-saved `agent_config`。

以下函数/逻辑改为消费 checkpoint source，而不是 `cfg_plain.agent/dataset`：

```text
_dataset_modalities(...)
_rgb_preprocessing(...)
_build_observation_contract(...)
DQ-RISE codebook deployment sanitization
point-cloud pretrained deployment sanitization
```

### Zarr path

默认 Zarr path 从 checkpoint-saved dataset config 获取。

仍保留显式：

```bash
--zarr-path PATH
```

作为研究者主动的数据位置 override。

但无论是否 override：

```text
Zarr attrs.task_name == experiment config task_name
```

必须继续成立。显式 `--zarr-path` 只改变 physical dataset location，不得改变 task identity。

---

## 7. 实现任务 E：一个 shared deployment contract parser

将 agent target grammar 移入 `dexmani_policy/deployment/contract.py`，例如：

```python
def validate_agent_targets(value: Any, path: str = "agent") -> None:
    ...
```

规则保持现有 allowlist：所有 nested `_target_` 必须位于允许的 `dexmani_policy.agents.*` namespace。

同时让 `parse_deployment_contract()` 完成 artifact metadata 的统一 strict validation，包括：

```text
action/window contract
observation fields
RGB preprocessing
normalization contract
agent constructor target allowlist
```

目标：

```text
export payload validation
inspect_experiment
load_experiment
restore_deployment_agent
```

都依赖同一 parser，不允许 inspect PASS、restore 才因 metadata grammar FAIL。

restore 不应再自己维护另一份 normalization/target grammar。

如果为了消费 canonical normalization，需要给 internal `DeploymentSpec` 增加 normalization spec 字段，可以增加；不要把新的复杂 public API 暴露到 Real。

---

## 8. 实现任务 F：public export 永远 strict verify

研究者 public API/CLI 不再允许 publish 未验证 artifact。

删除：

```python
verify: bool
```

以及 CLI：

```text
--verify
--no-verify
```

public `export_deployment_artifact()` 固定执行：

```text
build payload
-> validate payload
-> write candidate
-> safe reload
-> strict restore
-> deterministic synthetic predict
-> publish deployment_latest.pt
```

任何能成功 publish 的 artifact 必须已经完成 strict restore + prediction。

`qualify.py` 可以继续使用 private candidate builder 跑 direct/export parity；不要为了复用而重新暴露 `publish=False/verify=False` 的 public boolean mode matrix。

推荐内部结构：

```text
_export_candidate(...)
export_deployment_artifact(...)  # researcher-facing, always verify + publish
qualify_policy_parity(...)       # developer/release tool
```

---

## 9. 实现任务 G：失败必须可直接重试

当前 verification failure 发生在最终 `.pt` 已创建后，但 candidate 不会被删除。

修复目标：

```text
任何 selector publish 前失败
    -> 删除本次创建的 candidate artifact
    -> fsync checkpoint directory
    -> deployment_latest.pt 保持旧值
    -> 同一命令可直接 retry
```

candidate cleanup 不得 silent `except OSError: pass`。

如果 cleanup 本身失败，应抛出明确的 `ArtifactPublicationError`，说明 artifact state 可能需要人工检查。

`qualify_policy_parity()` 同样要求：

```text
build candidate
-> restore/parity
-> 完整构造 ParityReport
-> publish selector  # 函数中最后一个外部状态修改
-> return report
```

publish 之后不要再做可能抛异常的 report 计算。

### 本任务不要求

本 patch 不引入 multi-process file lock、content hash 或 complex no-clobber transaction framework。论文实验按单用户/单 exporter workflow 设计；保留现有“已有同名 artifact 时拒绝覆盖”的基本行为即可。

---

## 10. 实现任务 H：删除 duplicate normalizer export validator

前提：任务 E/F 完成，所有 public artifact publish 前必经 shared strict restore。

届时 exporter-side raw state duplicate validation：

```text
_validate_normalizer_state(...)
_normalizer_parameter_names(...)
_normalizer_feature_dim(...)
```

如果其 invariants 已被：

```text
parse_normalization_contract()
restore.validate_deployment_normalizer()
common.validate_normalizer_state()
```

完整覆盖，则删除 exporter duplicate 实现。

DQ-RISE 对 codebook/action-normalizer 的专用一致性检查不是 generic duplicate validator，继续保留。

---

## 11. Producer metadata

至少保留：

```yaml
producer:
  source_checkpoint: <training checkpoint filename>
  selected_weights: model | ema_model
```

可增加一个低成本 research provenance 字段：

```yaml
source_commit: <git HEAD sha>
```

要求：

- 仅用于论文实验追溯；
- 不作为 runtime compatibility gate；
- 无法获取 git revision 时不得阻塞 export（可省略该字段）。

不要增加 source tree hash、signature 或 dependency lock。

---

## 12. Qualify 的最终定位

`qualify_policy_parity()` 保留，但定位为 developer/release regression tool，不是日常研究者入口。

Direct branch 改为：

```text
checkpoint saved agent_config
-> direct constructor
-> selected raw/EMA state
```

Exported branch：

```text
same checkpoint saved agent_config
-> deployment sanitization
-> artifact restore
```

因此 parity 真正验证：

```text
constructor sanitization + serialization + deployment restore
```

而不是“direct/export 两边共同读取当前 config 后得到同一个错误结果”。

日常论文实验仍应保持：

```text
export -> run_policy
```

而不是：

```text
export -> qualify -> run_policy
```

---

## 13. 测试要求：小而强

不要恢复旧 340 行 deployment normalization test 原样；旧测试中包含已删除 reconciliation 设计假设。

新增精简的 deployment integrity tests，至少覆盖：

1. **checkpoint owns constructor**
   - checkpoint `n_head=8`；
   - 当前 config 改成 `n_head=12`；
   - export 必须仍使用 checkpoint 的 `8`，不得读取 current config agent semantics。
2. **checkpoint owns dataset/preprocessing**
   - 当前 config dataset preprocessing drift 不影响 artifact；
   - checkpoint-saved dataset semantics 被使用。
3. **normalization direct round-trip**
   - artifact normalization 来自 checkpoint contract；
   - malformed/field mismatch 在 shared parser 阶段 fail。
4. **target validation at inspect boundary**
   - malicious/unsupported nested `_target_` 在 `inspect_experiment()` 就 fail；
   - 不等待 Hydra instantiate。
5. **selected state only**
   - raw selected 时 EMA 缺失不应影响；
   - EMA selected 且 EMA 缺失必须 fail；
   - `producer.selected_weights` 与实际 state source 一致。
6. **DQ-RISE selected state**
   - selected state 缺完整 persistent codebook 必须 fail；
   - deployment constructor `codebook_path=None` 后 parity 保持。
7. **export always verifies**
   - CLI/API 不存在 public no-verify publish bypass。
8. **verification failure cleanup**
   - candidate 不存在；
   - selector 保持原值；
   - 同名 retry 可成功。
9. **cleanup failure is explicit**
   - 不得 silent swallow。
10. **qualify publication order**
   - parity/report failure不修改 selector；
   - publish 是最后一个外部 side effect。
11. **runtime compatibility regression**
   - existing `LoadedPolicy.predict()` contract 保持 `[n_action_steps, control_action_dim]`。

同时恢复/保留必要的 shared normalization grammar unit tests；不要为了行数恢复与新设计无关的历史 tests。

---

## 14. 明确暂缓 / 非目标

以下内容本轮**不要实现**：

### 14.1 RGB pretrained constructor 完全 self-contained

DINO/CLIP/SigLIP/R3M 当前 constructor 可能依赖 HuggingFace/R3M cache 或网络。此问题已确认，但暂缓为独立 `deployment portability` task。

本轮要求 public export 永远 strict restore，因此缺 cache/网络时必须在 **export verification 阶段 fail-fast**，不能等到真机现场才暴露。

不要在本任务中引入：

```text
HF config serialization
from_config deployment constructors
R3M packaged pretrained architecture assets
```

### 14.2 Python implementation / code-version immutability

Checkpoint-owned config 不能冻结 Python class implementation。

本轮只允许 best-effort `producer.source_commit` provenance；不要做：

```text
commit mismatch -> reject
source file hash
container/package embedding
runtime ABI version framework
```

### 14.3 Artifact cryptographic integrity

不要增加：

```text
SHA256 pinning
signatures
content-addressed filenames
```

checkpoint directory 按 trusted local artifact store 处理。

### 14.4 Multi-process publication locking

不要增加 `flock`、distributed lock 等。当前论文实验假设单一 exporter。

### 14.5 Public API cleanup

本轮不删除：

```text
LoadedPolicy.predict_action_chunk()
PolicySpec.chunk_size
DeploymentSpec.chunk_size
```

即使当前 repo 内无 consumer，也不要把 public API break 混入 correctness patch。

同理，不因为本任务升级到 `simple.v4`。

### 14.6 dexmani_real

除非测试暴露真实 incompatibility，本任务不要修改 `dexmani_real`。

当前研究者 workflow 应保持：

```bash
python -m dexmani_policy.deployment.export EXPERIMENT --checkpoint CHECKPOINT

python examples/run_policy.py policy/task/experiment \
  --num-episodes N \
  --inference-steps K
```

---

## 15. 验收标准

完成后必须满足：

```text
Checkpoint owns:
    architecture/config
    action/window semantics
    normalization semantics
    dataset/preprocessing semantics

Artifact owns:
    selected weights
    immutable observation/action contract
    default inference steps

Runtime owns:
    explicit --inference-steps override

Real owns:
    hardware timing / safety / command publication
```

并满足：

1. 修改 experiment `config.yaml.agent` 后，旧 checkpoint deployment behavior 不再随之改变。
2. 修改 experiment `config.yaml.dataset/normalization` 后，旧 checkpoint artifact semantics 不再随之改变。
3. experiment task identity 与 Zarr task identity 仍严格匹配。
4. 任何 public export 成功都意味着 artifact 已 safe reload + strict restore + synthetic prediction 成功。
5. export/qualify 在 publish 前失败不会留下阻塞 retry 的 candidate。
6. selector 在 qualification/verification 失败时保持原值。
7. EMA/raw 选择明确且 artifact 内只保存 selected state。
8. `dexmani_real` 日常真机 CLI 不增加新的必填参数。
9. NFE ablation 继续只需 `--inference-steps N`，无需重新 export。
10. 不新增 checkpoint schema、不引入 migration framework、不扩大到 production security platform。

---

## 16. 实施原则

实现时遵循：

```text
删除双 source-of-truth
> 新增 reconciliation

一个 canonical parser
> export/restore 各写一套 grammar

fail-fast at export/inspect
> 真机现场才发现问题

研究者 API 简单
> 暴露大量安全/兼容开关
```

优先删代码、复用现有 shared validator、保持 `simple.v3` 和 `dexmani_real` 接口稳定。
