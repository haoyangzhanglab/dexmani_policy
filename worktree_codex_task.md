# DexMani Policy 目录重构任务书（Final）

> 文件：`worktree_codex_task.md`  
> 实现基线：`dexmani_policy/main@66c6b826de2b44849c7a131bf51b25b21b077abe`（之后仅新增本任务书文档提交）  
> 性质：纯架构/目录重构。除修正依赖方向、移动代码和删除确认无用代码外，不改变训练、评测、推理、checkpoint、Real 部署语义。  
> 核心约束：目录数量克制；`dexmani_policy/tools/` 与 `dexmani_policy/common/` 最终消失；根目录现有 Python 用户入口保持路径不变。

## 1. 最终目标

本任务只采用下列结构方案，不再探索替代架构。

### 1.1 三个核心归位

此前讨论的三个共享模块只按以下方式归位：

1. checkpoint → `dexmani_policy/training/checkpoint.py`
2. inference → `dexmani_policy/agents/loader.py`
3. evaluation → `dexmani_policy/evaluation/protocol.py`

不要创建 package-root 的 `checkpoint.py`、`inference.py`、`evaluation.py` 或 `experiment.py`。

### 1.2 删除模糊目录

完成后必须删除：

```text
dexmani_policy/common/
dexmani_policy/tools/
```

新增且只新增两个一级 package domain：

```text
dexmani_policy/evaluation/
dexmani_policy/utils/
```

不要新增 `core/`、`contracts/`、`artifacts/`、`cli/`、`commands/`、`processors/` 等一级目录。

### 1.3 根入口不移动

以下文件路径保持不变：

```text
dexmani_policy/train.py
dexmani_policy/train_ddp.py
dexmani_policy/select_best_ckpt.py
dexmani_policy/eval_best_ckpt.py
dexmani_policy/record_demo.py
dexmani_policy/smoke_test.py
```

它们只允许更新 import / 显式路径，不得迁入 training/evaluation。

### 1.4 外部 API 不变

`dexmani_real` 当前通过：

```python
from dexmani_policy.deployment import ...
```

访问 Policy 仓库。本任务不得要求 `dexmani_real` 修改 import。

以下公开名字必须继续从 `dexmani_policy.deployment` 导入：

- `LoadedPolicy`
- `PolicyInfo`
- `inspect_policy`
- `list_experiments`
- `load_experiment_config`
- `load_policy`
- `resolve_checkpoint`
- `resolve_experiment`

---

## 2. 工作前必须执行

1. 读取根目录 `AGENTS.md`，严格遵守 Source of Truth、Validation Ladder、Deployment Boundary。
2. 执行：
   ```bash
   git status --short
   git rev-parse HEAD
   ```
3. 本任务实现基线为 `66c6b826`。当前 main 之后已有本任务书文档提交；若还有其他代码提交，必须先阅读相关 diff，再按当前代码调整迁移，禁止覆盖用户后续修改。
4. 先搜索真实 consumers，再修改；不要根据文件名猜用途。
5. 当前 code/config 是事实源。README/AGENTS 是稳定接口说明；`docs/` 默认冻结，仅在本任务移动后会导致现有可执行命令/路径明确失效时做最小修正。
6. 不修改或删除与本任务无关的用户改动。
7. 不启动完整训练、DDP、长时间仿真评测或真机运动。

---

## 3. 最终目录目标

```text
dexmani_policy/
├── __init__.py
├── train.py
├── train_ddp.py
├── select_best_ckpt.py
├── eval_best_ckpt.py
├── record_demo.py
├── smoke_test.py
│
├── agents/
│   ├── loader.py
│   ├── normalization.py
│   ├── core/
│   ├── obs_encoder/
│   ├── action_decoders/
│   ├── vq_hand/
│   ├── optim_util.py
│   └── position_encodings.py
│
├── configs/
│   ├── *.yaml
│   └── ddp/
│
├── datasets/
│   ├── base_dataset.py
│   ├── rgb_dataset.py
│   ├── pc_dataset.py
│   ├── multi_task_dataset.py
│   ├── replay_buffer.py
│   ├── sampler.py
│   ├── resumable_sampler.py
│   ├── augmentation.py
│   └── preprocessing.py
│
├── training/
│   ├── checkpoint.py
│   ├── build_utils.py
│   ├── trainer.py
│   ├── resume.py
│   ├── workspace.py
│   ├── ema_model.py
│   ├── lr_scheduler.py
│   └── logging.py
│
├── evaluation/
│   ├── __init__.py
│   └── protocol.py
│
├── env_runner/
│   ├── base_runner.py
│   ├── sim_runner.py
│   └── multi_task_sim_runner.py
│
├── deployment/
│   ├── __init__.py
│   └── runtime.py
│
└── utils/
    ├── __init__.py
    ├── config.py
    ├── path.py
    ├── random.py
    ├── tensor.py
    └── validation.py

scripts/
├── training/
│   ├── train.sh
│   ├── train_ddp.sh
│   ├── train_vq_hand.sh
│   ├── train_vq_hand.py
│   ├── extract_vq_codebook.py
│   └── measure_vq_usage.py
├── eval/
├── remote/
│   ├── ...
│   └── resolve_remote_datasets.py
└── utils/
```

说明：
- 不改 `agents → policies`。
- 不改 `env_runner → rollout`。
- 不引入 `src/` layout。
- 不把 `training/build_utils.py` 改名为 `build.py`。
- 不创建 committed `tests/`。

---

## 4. Core migration A：checkpoint → training/checkpoint.py

来源：

- `dexmani_policy/common/checkpoint_io.py`
- `dexmani_policy/common/pytorch_util.py::fix_state_dict`

### 4.1 training/checkpoint.py 只拥有 checkpoint artifact

最终包含：

- `TRAIN_CHECKPOINT_FORMAT`
- `TrainCheckpoint`
- `CheckpointStore`
- `fix_state_dict`

保持现有：
- `TRAIN_CHECKPOINT_FORMAT == "simple.v3"`
- payload root keys；
- state/weights keys；
- `torch.load(..., map_location="cpu", weights_only=False)` 行为；
- atomic tmp save + replace；
- `CheckpointStore.resolve_path()` 的 latest / absolute experiment dir / explicit checkpoint 行为；
- save 后 CUDA cache 行为（本任务不做行为清理）。

### 4.2 strict resume semantic 归 training/resume.py

以下不是 checkpoint 文件 I/O，而是 exact training resume contract，应迁入 `training/resume.py`：

- `NORMALIZATION_CONTRACT_VERSION`
- `make_normalization_contract`
- `build_agent_contract`
- `validate_resume_contract`
- `validate_ema_resume_state`

`training/resume.py` 继续拥有：
- DataLoader/resumable sampler 构建；
- resume contract；
- optimizer/scheduler/model/EMA/RNG state restoration；
- GPU id validation。

### 4.3 依赖要求

`training/checkpoint.py` 必须是低层 artifact module：
- 不 import trainer/workspace/resume/deployment/evaluation；
- 可以被 training lifecycle 与 `agents.loader` 消费；
- `agents.loader` 对 `training.checkpoint` 的 import 应放在 restore 函数内部，保持 loader 模块本身轻量。

---

## 5. Core migration B：common/inference → agents/loader.py

最终 `agents/loader.py` 的职责定义为：

> 轻量解析 saved experiment/checkpoint，并在显式 restore 时构造完整 Agent。

它包含且只包含以下现有能力：

- `load_experiment_config`
- `read_best_ckpt_json`
- `resolve_checkpoint`
- `restore_policy_agent`

这是本任务的最终 owner，不再创建 `experiment.py`。

### 5.1 轻量 import 是硬约束

`agents/loader.py` 在普通 import 时不得加载：
- torch；
- training checkpoint module；
- heavyweight Agent/backbone；
- CUDA。

`restore_policy_agent()` 内部继续使用 lazy import：
- hydra/OmegaConf（可局部导入）；
- `training.checkpoint`；
- normalization restore/validation；
- 其他只在实际 restore 时需要的组件。

目的：`dexmani_real` parent-side inspection 只读 config/metadata 时不初始化 Policy/CUDA。

### 5.2 restore_policy_agent 行为不变

必须保持：
- saved resolved config 构造 Agent；
- `validate_window_contract`；
- `action_key` 校验；
- raw / EMA 明确选择；
- 请求 EMA 缺失时明确失败，不 fallback raw；
- strict `load_state_dict`；
- normalization spec 恢复与 fitted state validation；
- `agent.to(device)`；
- `agent.eval()`；
- full checkpoint inference 不读取仅训练初始化所需的 Uni3D/VQ 外部资产。

### 5.3 common/inference.py 其他 symbol 归位

不要把下列内容塞进 loader：

- `rgb_preprocessing_kwargs`
  → `datasets/preprocessing.py`

- `resolve_inference_steps`
  → `agents/action_decoders/utils.py`

- `positive_int`
  → `utils/validation.py`

这样 `agents/loader.py` 不会变成新的 common。

---

## 6. Core migration C：training/eval_utils → evaluation/protocol.py

创建：

```text
dexmani_policy/evaluation/__init__.py
dexmani_policy/evaluation/protocol.py
```

将当前 `training/eval_utils.py` 的评测共享机制迁入 `evaluation/protocol.py`，包括：

- `resolve_eval_seed`
- `validate_eval_config`
- `parse_eval_overrides`
- `validate_inference_steps`
- `add_inference_steps_argument`
- `build_eval_runner`
- `iter_leaf_env_runners`
- `load_ckpt_for_inference`
- `collect_episode_details`
- `compute_eval_stats`
- `_get_eval_param`
- `MilestoneCheckpoint`
- `discover_milestone_checkpoints`
- `resolve_checkpoint_path`

不要为了“公共 API 漂亮”重命名 `_get_eval_param`；本任务优先最小 diff、行为不变。

消费者保持原路径：
- `select_best_ckpt.py`
- `eval_best_ckpt.py`
- `record_demo.py`

三者只改 import 与本次迁移直接相关的路径，不改 CLI。

`evaluation/protocol.py` 允许依赖：
- `agents.loader`
- `env_runner`
- `utils.config`
- `utils.validation`
- Hydra / OmegaConf

禁止依赖：
- `training.trainer`
- `training.workspace`
- `training.build_utils`
- `deployment.runtime`

---

## 7. common/normalizer → agents/normalization.py

整体迁入：

```text
dexmani_policy/agents/normalization.py
```

保留现有：
- `DictOfTensorMixin`
- `LinearNormalizer`
- `SingleFieldLinearNormalizer`
- `fit_params`
- `normalize_tensor`
- `build_mixed_action_normalizer`
- `validate_normalization_spec`
- `validate_normalizer_state`
- `ALLOWED_NORMALIZATION_MODES`
- `NON_NUMERIC_OBSERVATION_FIELDS`

同时把当前 `training/build_utils.py::resolve_normalization_spec(cfg)` 迁到 `agents/normalization.py`。

原因：
- normalization grammar 与 fitted normalizer 都是 Agent numerical semantics；
- `training/build_utils` 和 `agents/loader` 均可直接依赖 `agents.normalization.resolve_normalization_spec`；
- 避免错误的 `utils → agents` 依赖。

行为必须保持：
- normalizer state_dict keys；
- limits / gaussian / identity / auto grammar；
- `action:auto`；
- mixed EE action normalization；
- streaming Welford fit；
- strict fitted-state validation。

---

## 8. common/config → utils/config + datasets/sampler

### 8.1 utils/config.py

只迁入真正跨生命周期的 config helper：

- `register_resolvers`
- `validate_window_contract`
- `validate_action_key_consistency`

不得 import：
- agents；
- datasets；
- training；
- evaluation；
- deployment。

不要把 `resolve_normalization_spec` 放进 utils/config；它已归 `agents/normalization.py`。

### 8.2 datasets/sampler.py

将 split/sampling 规则迁入：

- `validate_val_ratio`
- `validate_max_train_episodes`
- `validate_dataset_splits`

`BaseDataset` 与 `training/build_utils.validate_config` 从 `datasets.sampler` 引用这些规则。

这样避免 `utils → datasets` 的反向依赖，也不新增 `datasets/validation.py`。

---

## 9. common/pytorch_util → 精确归位

### 9.1 utils/random.py

迁入：
- `set_seed`
- `get_rng_state`
- `set_rng_state`
- `worker_init_fn`

注意：`worker_init_fn` 当前实际被 `training/resume.py` 的 DataLoader 使用，禁止删除。

### 9.2 utils/tensor.py

迁入：
- `dict_apply`
- `ensure_tensor`

注意：`ensure_tensor` 当前实际被 `BaseDataset.sample_to_data` 使用，禁止删除。

### 9.3 utils/path.py

迁入并保持现有行为：
- `set_project_root`

本任务是目录重构，不顺手改变 cwd 语义。根入口当前依赖它将 cwd 切到仓库根；先原样迁移，未来若要删除全局 `chdir`，另立任务评估。

### 9.4 utils/validation.py

迁入：
- `positive_int`

它目前被 deployment、evaluation、env_runner、多个 action decoder 使用，属于真正跨 domain 的纯 validation helper。

### 9.5 domain-local

迁移：
- `optimizer_to` → `training/resume.py`
- `to_log_scalars` → `training/logging.py`
- `compile_models` → `training/build_utils.py`
- `fix_state_dict` → `training/checkpoint.py`
- `create_mlp` → `agents/obs_encoder/proprio/state_mlp.py`
- `count_params` / `print_param_count` → `training/logging.py`
- `format_success_rate` → `env_runner/base_runner.py`，`multi_task_sim_runner.py` 从 base_runner 复用。

禁止创建：
- `utils/pytorch.py`
- `utils/misc.py`

---

## 10. Action decoder inference helper

创建已有 domain 内的小文件：

```text
dexmani_policy/agents/action_decoders/utils.py
```

迁入：
- `resolve_inference_steps`

它使用 `utils.validation.positive_int`。

更新：
- `diffusion.py`
- `rectified_flow.py`
- `consistency_flow.py`

保持：
- training grid 参数语义不变；
- runtime `inference_steps` 仅覆盖 NFE；
- 不恢复任何旧 `denoise_steps` alias。

---

## 11. datasets/preprocessing.py：保持 parent-side lightweight

迁移：

- `datasets/base_dataset.py::preprocess_validation_rgb`
- `common/inference.py::rgb_preprocessing_kwargs`

到：

```text
dexmani_policy/datasets/preprocessing.py
```

消费者：
- `BaseDataset`
- `env_runner`
- `deployment/runtime.py`

### 硬约束

`datasets/preprocessing.py` 在模块 import 时不得 top-level import torch / torchvision。

使用：
- `from __future__ import annotations`
- `TYPE_CHECKING`
- 或函数内部 lazy import。

原因：`deployment/runtime.py` 的 config/metadata inspection 必须继续保持轻量，不能因为读取 RGB preprocessing recipe 就加载 torch/CUDA 生态。

保持 deterministic RGB 行为：
- raw uint8 HWC；
- resize；
- center crop；
- keep_uint8；
- float 路径范围；
- Agent-owned ImageProcessor 不变。

---

## 12. 修复最新 main 的 training → deployment 依赖

当前最新代码：

```text
training/build_utils.py
    → deployment/runtime.py::capture_real_runtime
```

必须消除。

将 `capture_real_runtime(dataset, cfg)` 原样迁到：

```text
training/build_utils.py::_capture_real_runtime(dataset, cfg)
```

它是 training-time saved-config metadata capture，仅有 training consumer。

目标数据流：

```text
Dataset / ReplayBuffer
      ↓
training._capture_real_runtime
      ↓
resolved config.yaml.real_runtime
      ↓
deployment/runtime.py consume
```

禁止重新引入：
- `datasets/real_policy_contract.py`
- duplicated Real semantic dictionary
- deployment export artifact/ABI
- legacy cache/experiment compatibility。

保持最新 `f82b3e18 / 66c6b826` 的方向：
- Dataset/ReplayBuffer generic；
- Real producer 负责 Raw/canonical 完整正确性；
- Policy 只捕获最小数值 metadata；
- `real_runtime` 仍只有当前需要的 `dt`、optional `pointcloud`、optional `fingertip_link_names`。

---

## 13. tools/ 完全删除，workflow 进入已有 scripts

当前 `dexmani_policy/tools/` 没有 Python library consumer。迁移：

```text
dexmani_policy/tools/train_vq_hand.py
→ scripts/training/train_vq_hand.py

dexmani_policy/tools/extract_codebook.py
→ scripts/training/extract_vq_codebook.py

dexmani_policy/tools/measure_vq_usage.py
→ scripts/training/measure_vq_usage.py

dexmani_policy/tools/resolve_remote_datasets.py
→ scripts/remote/resolve_remote_datasets.py
```

同步：
- `scripts/training/train_vq_hand.sh`
- `scripts/remote/train_remote.sh`
- README 中真实命令/路径；
- `dexmani_policy/configs/dqrise.yaml` 中指向旧 tools 文件的注释；
- 仓库内所有 `dexmani_policy.tools` / `tools/*.py` 的有效引用。

### 13.1 不保留 sys.path hack

这些 scripts 在受管 Conda + `pip install -e .` 环境运行。移除原 tools 脚本中的 `sys.path.insert` hack。

### 13.2 resolve_remote_datasets 的 config_dir 必须因移动而修正

移动到 `scripts/remote/resolve_remote_datasets.py` 后，原来的：

```python
Path(__file__).resolve().parents[1] / "configs"
```

不再正确。

必须显式解析为 repo 中的：

```text
<repo-root>/dexmani_policy/configs
```

并保持：
- config name validation；
- Hydra compose；
- overrides；
- task identity validation；
- multi-task recursive dataset path resolution；
- `--check`；
- JSON 输出；
- relative dataset path 以正式 repo-root launcher 语义解析。

### 13.3 docs 最小例外

`docs/` 默认冻结。但 `docs/SSH服务器训练部署.md` 当前若包含已被删除的 `python -m dexmani_policy.tools.resolve_remote_datasets` 可执行命令，则必须只做该命令路径的最小更新，否则文档会直接不可执行。

不要顺带重写其他 docs 架构内容。

---

## 14. deployment public façade 与轻量性

`deployment/__init__.py` 可直接：
- 从 `agents.loader` re-export `load_experiment_config` / `resolve_checkpoint`；
- 从 `deployment.runtime` re-export `PolicyInfo` / `LoadedPolicy` / `inspect_policy` / `load_policy` / `resolve_experiment` / `list_experiments`。

也可以由 runtime 中 import 后再 re-export，但外部路径必须不变。

### 14.1 Import lightweight invariant

仅执行：

```python
import dexmani_policy.deployment
```

不得因为本次重构而主动加载 torch 或构造模型。

实际 `load_policy()` / `LoadedPolicy.predict()` 时才允许 lazy import torch。

---

## 15. 目标依赖方向

允许：

```text
agents              → utils
datasets            → utils
training            → agents / datasets / utils
env_runner          → datasets / utils
evaluation          → agents.loader / env_runner / utils
deployment          → agents.loader / datasets.preprocessing / utils

training.resume     → training.checkpoint
training.workspace  → training.checkpoint
training.trainer    → training.checkpoint
agents.loader       -lazy→ training.checkpoint
```

禁止：

```text
agents.loader       → training.build_utils
training.build_utils→ deployment.runtime
evaluation.protocol → training.trainer/workspace/build_utils
utils               → agents/datasets/training/evaluation/deployment
datasets            → deployment
common/tools        → anywhere
```

---

## 16. 行为不变量

### 16.1 Checkpoint
- format 仍为 `simple.v3`；
- payload schema 不变；
- raw/EMA/optimizer/scheduler/RNG/training position 不变；
- strict load 不变；
- strict resume contract 不变；
- latest/explicit/experiment-dir resume resolution 不变。

### 16.2 Agent / Normalization
- 所有 Hydra `agent._target_` 不变；
- Agent constructor/config recipe 不变；
- normalizer state_dict keys 不变；
- normalization grammar与数值计算不变；
- full checkpoint inference 的外部初始化 bypass 行为不变。

### 16.3 Evaluation
- `best_ckpt.json` schema 不变；
- selection seed protocol 不变；
- held-out exclusion 不变；
- ranking/tie-break 不变；
- result output schema/path 不变；
- CLI 参数不变；
- 不恢复旧 inference aliases。

### 16.4 Real
- saved resolved config 仍是 inference/deployment 模型语义事实源；
- Dataset/ReplayBuffer generic；
- `real_runtime` 内容不变；
- deployment 不重新打开 training Zarr；
- `dexmani_policy.deployment` public API 不变。

### 16.5 Entry points
以下路径保持不变：
- `dexmani_policy/train.py`
- `dexmani_policy/train_ddp.py`
- `dexmani_policy/select_best_ckpt.py`
- `dexmani_policy/eval_best_ckpt.py`
- `dexmani_policy/record_demo.py`
- `dexmani_policy/smoke_test.py`

`set_project_root` 本任务只移动、不删除，因此 cwd 行为保持。

---

## 17. 推荐实施顺序

按以下顺序或等价的低风险顺序执行；不要让仓库长时间停留在一半新 import、一半旧 import 的状态。

### Phase A — utils 基础
1. 创建 `utils/{config,path,random,tensor,validation}.py`。
2. 迁移对应 common helpers。
3. 更新 consumers。
4. 验证 `ensure_tensor`、`worker_init_fn`、`set_project_root` 仍有正确 consumer。

### Phase B — normalization
5. 创建 `agents/normalization.py`。
6. 移动 common normalizer + `resolve_normalization_spec`。
7. 更新 BaseAgent、各 Agent、training、VQ scripts、loader consumers。

### Phase C — checkpoint/resume
8. 创建 `training/checkpoint.py`。
9. strict resume contract helpers 移入 `training/resume.py`。
10. 更新 trainer/workspace/train_ddp/smoke imports。
11. 做 checkpoint + resume targeted regression。

### Phase D — loader
12. 创建 `agents/loader.py`。
13. 移动 experiment config / best selector / checkpoint resolution / Agent restore。
14. 保持 module import lightweight。
15. 更新 deployment/evaluation/root entry/smoke consumers。

### Phase E — preprocessing
16. 创建轻量 `datasets/preprocessing.py`。
17. 移动 RGB recipe 与 deterministic preprocessing。
18. 更新 BaseDataset/env_runner/deployment。

### Phase F — evaluation
19. 创建 `evaluation/protocol.py` 与空的 `evaluation/__init__.py`。
20. 迁移 `training/eval_utils.py`。
21. 三个 root eval/demo 入口只改 import。
22. 删除旧 eval_utils。

### Phase G — latest-main dependency correction
23. `capture_real_runtime` 移入 `training/build_utils.py` 作为 private helper。
24. 移除 training → deployment import。
25. 验证 simulation config 不新增 real_runtime、canonical Real config 捕获内容不变。

### Phase H — tools
26. 四个 tools workflow 移入已有 scripts。
27. 修正移动后的 config-dir/path 解析。
28. 修改 shell wrapper、README、dqrise config comments 和必要的 SSH docs 命令。
29. 删除 `dexmani_policy/tools/`。

### Phase I — cleanup
30. 全仓搜索旧 `common/tools` imports 和路径。
31. 删除 `dexmani_policy/common/`。
32. 更新 README 稳定仓库结构。
33. 更新 AGENTS Repository Entry Points：加入 evaluation 与 utils；删除任何已失效结构描述。
34. 不修改与本任务无关的 docs。

---

## 18. 验证要求

只报告真正执行过的检查为 PASS。受环境限制的检查写明 `NOT VERIFIED`。

### 18.1 Syntax / import

至少执行：

```bash
python -m compileall dexmani_policy
python -m py_compile   scripts/training/train_vq_hand.py   scripts/training/extract_vq_codebook.py   scripts/training/measure_vq_usage.py   scripts/remote/resolve_remote_datasets.py
```

### 18.2 Stale reference gate

完成后以下搜索必须无有效结果（文档中的历史叙述不算，但所有当前命令/import 必须清零）：

```bash
grep -R "dexmani_policy.common" -n dexmani_policy scripts README.md AGENTS.md
grep -R "dexmani_policy.tools" -n dexmani_policy scripts README.md AGENTS.md
grep -R "training.eval_utils" -n dexmani_policy scripts
grep -R "from dexmani_policy.deployment" -n dexmani_policy/training
grep -R "training.build_utils" -n dexmani_policy/agents/loader.py
```

确认目录：
```bash
test ! -d dexmani_policy/common
test ! -d dexmani_policy/tools
```

### 18.3 Config-only smoke

至少运行：

```bash
conda run --no-capture-output -n policy   python dexmani_policy/smoke_test.py --config-only   dp dp3 r3d maniflow sat dqrise multitask_dit
```

### 18.4 Temporary targeted regression（不要提交 tests/）

至少覆盖：

1. `TrainCheckpoint` save/load schema roundtrip。
2. `fix_state_dict`：
   - DDP `module.`
   - torch.compile `_orig_mod.`
   - 二者组合。
3. normalizer state_dict roundtrip / spec validation。
4. `read_best_ckpt_json`：
   - valid relative checkpoint；
   - absolute/path traversal rejection。
5. `resolve_checkpoint`：
   - best；
   - latest；
   - explicit file；
   - outside checkpoint dir rejection。
6. evaluation milestone discovery/path resolution。
7. RGB deterministic preprocessing shape/dtype，并确认 importing `datasets.preprocessing` 不需要 top-level torch。
8. remote dataset resolver 以至少一个 config 做“不带 --check”的解析，确认移动后仍指向 `dexmani_policy/configs`。
9. deployment public imports。
10. lightweight import：在一个干净 Python process 中 import `dexmani_policy.deployment`，确认本次重构没有主动导入 torch。

### 18.5 Full smoke

若当前环境具备所需 CUDA/数据/权重，至少执行现有 full smoke 的代表性：
- RGB：dp
- 3D：dp3 或 r3d
- Flow：maniflow
- VQ：dqrise

若不具备，标记 `NOT VERIFIED`，不得为了通过验证修改算法或降低校验。

### 18.6 dexmani_real boundary

至少验证：

```python
from dexmani_policy.deployment import (
    LoadedPolicy,
    PolicyInfo,
    inspect_policy,
    list_experiments,
    load_experiment_config,
    load_policy,
    resolve_checkpoint,
    resolve_experiment,
)
```

若本机同时有 dexmani_real，可运行不连接硬件的 import/config preflight。不得启动任何真机 motion。

---

## 19. 完成条件

全部满足才完成：

1. `dexmani_policy/common/` 不存在。
2. `dexmani_policy/tools/` 不存在。
3. `training/checkpoint.py` 是 checkpoint artifact owner。
4. strict resume contract 位于 `training/resume.py`。
5. `agents/loader.py` 是 saved experiment/checkpoint → Agent restore owner，且 import lightweight。
6. `evaluation/protocol.py` 是三个 offline evaluation/demo root entry 的共享 protocol owner。
7. root Python entry 路径不变。
8. `training/build_utils` 不 import deployment。
9. `agents/loader` 不 import `training.build_utils`。
10. utils 不 import任何上层 domain。
11. Dataset/ReplayBuffer 继续 generic。
12. deployment public API 不变。
13. checkpoint/normalizer/best_ckpt/real_runtime schema 不变。
14. `ensure_tensor` / `worker_init_fn` / `set_project_root` 未被误删，行为保持。
15. tools workflows 已迁到已有 scripts，shell/remote 上下游路径全部同步。
16. README/AGENTS 的当前结构说明与代码一致。
17. 不新增 legacy compatibility。
18. 不新增无必要一级目录。
19. 所有实际验证结果在最终报告中逐项列出；未执行项明确 `NOT VERIFIED`。

---

## 20. 明确禁止

不要：

- 移动六个根目录 Python entry。
- 新建 package-root checkpoint/inference/evaluation/experiment 模块。
- 新建 core/contracts/artifacts/cli/commands/processors 等一级目录。
- 重命名 agents、env_runner。
- 引入 `src/` layout。
- 引入 LeRobot processor/factory/plugin 体系。
- 创建 committed `tests/`。
- 修改 Policy architecture / action representation / objective / scheduler / NFE recipe。
- 修改 Hydra `_target_`。
- 修改 checkpoint format/schema。
- 修改 best_ckpt schema。
- 修改 real_runtime schema。
- 添加旧 checkpoint/旧 CLI/旧 Real cache compatibility。
- 删除当前仍在使用的 `ensure_tensor`、`worker_init_fn` 或 `set_project_root`。
- 将 VQ workflow 放进 utils。
- 创建 utils/pytorch.py、utils/misc.py。
- 为单 consumer 小函数制造新的跨-domain abstraction。
- 顺手清理与本任务无关的代码风格、算法或超参数。
- commit/push，除非调用者显式要求。

---

## 21. 最终执行原则

**领域代码归领域；训练 checkpoint 归 training；saved policy 恢复归 agents；评测 protocol 归 evaluation；真正跨 domain 且无业务语义的 helper 才归 utils；research workflow 归 repo scripts。**

本任务追求：
- 最小目录数；
- 最短 import 链；
- 明确 ownership；
- 零行为漂移；
- 上下游一次性同步；
- 便于后续研究快速迭代。

完成实现后，最终回复必须简洁列出：
1. 实际改动摘要；
2. 最终关键路径；
3. 验证命令及 PASS / NOT VERIFIED；
4. 是否存在剩余风险或未验证项；
5. `git status --short` 摘要。

不要只说“完成”，必须给出可审计的验证结果。
