# DexMani Policy 目录重构任务书

> 文件：worktree_codex_task.md  
> 基线：2026-09-28 审查时 dexmani_policy/main = 66c6b826de2b44849c7a131bf51b25b21b077abe  
> 参考：real-stanford/diffusion_policy、huggingface/lerobot  
> 性质：纯架构/目录重构。除为消除重复或修复依赖方向所必需的最小调整外，不改变训练、评测、推理、checkpoint、Real 部署语义。

## 1. 任务目标

对 dexmani_policy 的目录和共享代码进行一次最小而完整的收敛，目标是：

1. 删除 dexmani_policy/tools/，不再保留 tools 这一模糊目录。
2. 删除 dexmani_policy/common/，避免 common 继续充当跨 domain 的兜底层。
3. 新增且只新增两个一级 domain：evaluation/ 与 utils/。
4. 只把此前讨论的三块共享机制移动进一级 domain：
   - checkpoint.py → training/checkpoint.py
   - inference.py → agents/loader.py
   - evaluation.py → evaluation/protocol.py
5. 根目录现有用户入口全部保持路径不动：
   - dexmani_policy/train.py
   - dexmani_policy/train_ddp.py
   - dexmani_policy/select_best_ckpt.py
   - dexmani_policy/eval_best_ckpt.py
   - dexmani_policy/record_demo.py
   - dexmani_policy/smoke_test.py
6. 保持 dexmani_policy.deployment 作为 dexmani_real 的稳定公开 façade；dexmani_real 不应因本次内部重构而需要同步修改。
7. 不引入新的 factory、registry、processor、contract、artifact、core、cli、commands 等抽象层。
8. 优先减少依赖和代码，而不是为了 DRY 抽象 3～5 行的小 helper。

本任务的判定标准不是“目录更漂亮”，而是：
- 模块所有权清晰；
- 依赖方向单向；
- 行为不变；
- import 链更短；
- common/tools 完全消失；
- 不增加无必要的一级目录。

---

## 2. Source of Truth 与工作前检查

执行前必须先：

1. 读取仓库根目录 AGENTS.md，并严格遵守其中的 Source of Truth、Validation Ladder、Deployment Boundary。
2. 检查当前 worktree：
   - git status --short
   - git rev-parse HEAD
3. 本任务书基于 66c6b826 设计；若 HEAD 已更新，先阅读从 66c6b826 到当前 HEAD 的相关 diff，再执行迁移。不得覆盖用户后续修改。
4. 当前代码/config/checkpoint 行为优先于 docs/ 中可能滞后的架构描述。
5. docs/ 默认不修改，除非本任务导致其中的公共路径说明必须同步；README.md 与 AGENTS.md 在公共入口或稳定仓库结构变化时必须更新。

---

## 3. 最终目录目标

完成后，dexmani_policy 的主要结构应为：

~~~text
dexmani_policy/
├── __init__.py
├── smoke_test.py
├── train.py
├── train_ddp.py
├── select_best_ckpt.py
├── eval_best_ckpt.py
├── record_demo.py
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
    ├── random.py
    └── tensor.py

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
~~~

完成后必须不存在：

~~~text
dexmani_policy/common/
dexmani_policy/tools/
~~~

注意：
- 不移动根目录 train/eval/demo/smoke Python 入口。
- 不把 training/build_utils.py 改名为 build.py；本轮避免无功能收益的 rename。
- 不创建 experiment.py、inference.py、evaluation.py、checkpoint.py 等新的 package-root 模块。
- 不创建 scripts/vq_hand/ 新目录；复用现有 scripts/training/ 与 scripts/remote/。

---

## 4. 三个核心迁移

### 4.1 checkpoint → training/checkpoint.py

来源：
- dexmani_policy/common/checkpoint_io.py
- dexmani_policy/common/pytorch_util.py 中与 state-dict checkpoint 兼容相关的 fix_state_dict()

最终 training/checkpoint.py 只负责“训练 checkpoint 文件/tensor artifact”，包含：
- TRAIN_CHECKPOINT_FORMAT
- NORMALIZATION_CONTRACT_VERSION（若仍由 checkpoint schema 直接需要）
- TrainCheckpoint
- CheckpointStore
- fix_state_dict

不要把严格续训语义继续放在 checkpoint 模块。

以下内容移动到 training/resume.py：
- make_normalization_contract（若只服务 resume contract，则一并下沉）
- build_agent_contract
- validate_resume_contract
- validate_ema_resume_state
- 任何只服务 exact resume 的 contract 构造/比较逻辑

设计要求：
- training/checkpoint.py 必须保持基础设施性质。
- training/checkpoint.py 不得 import trainer.py、workspace.py、resume.py、deployment 或 evaluation。
- agents/loader.py 可以读取 training/checkpoint.py，因为完整 Agent 恢复消费的是训练 checkpoint artifact；但 training/checkpoint.py 本身不得反向依赖 Agent loader。
- checkpoint format 字符串、payload key、strict schema、保存原子性、latest path 行为全部保持不变。

### 4.2 inference → agents/loader.py

来源：
- dexmani_policy/common/inference.py 中真正属于“saved config/checkpoint → Agent”的恢复逻辑。

最终 agents/loader.py 只负责：
- restore_policy_agent(...)

职责：
- 从 saved config instantiate Agent；
- 读取 training/checkpoint.py；
- raw / EMA 选择；
- strict load_state_dict；
- 恢复 normalization_spec；
- validate normalizer state；
- move to device；
- agent.eval()。

必须移除当前 common.inference → training.build_utils.resolve_normalization_spec 的反向依赖。

解决方式：
- 将 resolve_normalization_spec(cfg) 的 config-only 解析/校验逻辑移动到 utils/config.py；
- training/build_utils.py 和 agents/loader.py 共同调用 utils/config.py；
- agents/loader.py 不得 import training/build_utils.py。

common/inference.py 中其他内容按真实 owner 下沉：
- load_experiment_config / read_best_ckpt_json / resolve_checkpoint：放入 agents/loader.py 仅在它们确实与 Agent restore 强绑定时保留；优先避免新建顶层 experiment.py。若 deployment/runtime.py 与 evaluation/protocol.py 均需要，则可以作为 agents/loader.py 中的轻量、Torch-free 函数，但重 import 必须继续保持 lazy，确保 parent-side inspection 不加载模型/CUDA。
- rgb_preprocessing_kwargs：移动到 datasets/preprocessing.py。
- resolve_inference_steps：移动到 agents/action_decoders/utils.py，或在 decoder domain 内用一个小 helper 实现；不得继续放全局 common。
- positive_int：不要为了它单独制造 utils/validation.py。优先在调用域局部校验；若 utils/config.py 已经自然需要同类纯 validation，可放其中，但禁止形成杂项 validation 垃圾桶。

特别要求：
- agents/loader.py 的模块级 import 必须轻量。torch/hydra/大模型依赖继续按需要 lazy import，保持 dexmani_real parent 只做 config/metadata inspection 时不会初始化 CUDA 或加载大型 backbone。

### 4.3 evaluation → evaluation/protocol.py

来源：
- dexmani_policy/training/eval_utils.py

training/eval_utils.py 当前没有训练生命周期 consumer，因此必须退出 training。

创建：
- dexmani_policy/evaluation/__init__.py
- dexmani_policy/evaluation/protocol.py

protocol.py 承担三个现有根入口共享的评测机制：
- resolve_eval_seed
- validate_eval_config
- parse_eval_overrides
- validate_inference_steps
- add_inference_steps_argument
- build_eval_runner
- iter_leaf_env_runners
- load_ckpt_for_inference
- collect_episode_details
- compute_eval_stats
- _get_eval_param（如保留，优先改成非下划线公共 helper，除非它只在 protocol 内使用）
- MilestoneCheckpoint
- discover_milestone_checkpoints
- resolve_checkpoint_path

消费者保持在 package 根目录：
- select_best_ckpt.py
- eval_best_ckpt.py
- record_demo.py

三者只改 import，不移动路径、不改 CLI。

evaluation/protocol.py 可以依赖：
- agents.loader
- env_runner
- utils.config
- OmegaConf / Hydra

evaluation/protocol.py 不得依赖：
- training.trainer
- training.workspace
- training.build_utils
- deployment.runtime

---

## 5. common/ 的其余迁移

### 5.1 common/normalizer.py → agents/normalization.py

整体迁移到：
- dexmani_policy/agents/normalization.py

保持：
- LinearNormalizer
- SingleFieldLinearNormalizer
- DictOfTensorMixin
- fit_params / normalize_tensor
- build_mixed_action_normalizer
- validate_normalization_spec
- validate_normalizer_state
- ALLOWED_NORMALIZATION_MODES
- NON_NUMERIC_OBSERVATION_FIELDS

原因：
- normalizer 是 BaseAgent 状态的一部分；
- fitted params 随 Agent state_dict 保存；
- 它定义 Policy observation/action numerical coordinate system；
- 它不是 generic utils。

行为不变量：
- state_dict key 不变；
- limits/gaussian/auto/identity 语义不变；
- mixed EE action normalization 不变；
- streaming fit 数值语义不变；
- normalizer validation 不变。

### 5.2 common/config.py → utils/config.py + datasets/sampler.py

utils/config.py 接收真正跨 train/eval/smoke/loader 的配置基础设施：
- register_resolvers
- validate_window_contract
- validate_action_key_consistency
- resolve_normalization_spec（从 training/build_utils.py 迁入）

不要把 utils/config.py 做成新的 common.py：
- 不放 dataset construction；
- 不放 Agent construction；
- 不放 evaluation protocol；
- 不放 deployment 逻辑；
- 不放 checkpoint I/O。

以下 split 规则直接放到 datasets/sampler.py：
- validate_val_ratio
- validate_max_train_episodes
- validate_dataset_splits 可保留在 utils/config.py 作为递归配置校验入口，内部调用 datasets.sampler 的两个原子校验；或者将递归逻辑局部放到 validate_config。优先选择依赖最少、调用最直观的实现，不新增 datasets/validation.py。

training/build_utils.py 中当前 validate_config 及其私有 config consistency helpers可以继续留在 build_utils.py，本任务不强制再拆一个 config-validation subsystem。关键要求只有：
- agents/loader.py 不得再 import training/build_utils.py；
- common/config.py 最终删除。

### 5.3 common/pytorch_util.py → 少量 utils + domain local

迁移：
- set_seed / get_rng_state / set_rng_state → utils/random.py
- dict_apply → utils/tensor.py
- optimizer_to → training/resume.py
- to_log_scalars → training/logging.py
- compile_models → training/build_utils.py
- fix_state_dict → training/checkpoint.py
- create_mlp → agents/obs_encoder/proprio/state_mlp.py
- count_params / print_param_count → training/logging.py
- format_success_rate → env_runner/base_runner.py 或 env_runner 内最小共享位置

删除：
- ensure_tensor（最新 main 无有效 consumer时删除）
- worker_init_fn（最新 main 无有效 consumer时删除）
- set_project_root（见第 9 节）

规则：
- 不创建 utils/pytorch.py。
- 不创建 utils/misc.py。
- 单 consumer helper 优先局部化。
- 少量重复的 3～5 行 validation 不值得引入跨 domain dependency。

---

## 6. tools/ 必须完全删除

当前 dexmani_policy/tools/ 没有 Python library consumer，应作为 leaf workflow 移出 package。

迁移：
- dexmani_policy/tools/train_vq_hand.py
  → scripts/training/train_vq_hand.py

- dexmani_policy/tools/extract_codebook.py
  → scripts/training/extract_vq_codebook.py

- dexmani_policy/tools/measure_vq_usage.py
  → scripts/training/measure_vq_usage.py

- dexmani_policy/tools/resolve_remote_datasets.py
  → scripts/remote/resolve_remote_datasets.py

同步修改：
- scripts/training/train_vq_hand.sh
- scripts/remote/train_remote.sh
- README.md 中对应命令/路径
- docs/SSH服务器训练部署.md 仅在该路径仍是当前操作说明且必须同步时修改；遵循 AGENTS.md 对 docs 的冻结约束。

脚本原则：
- scripts 下 Python 文件是 executable workflow，不是 library API。
- 不把 VQ-specific workflow 塞入 utils。
- 删除原先为 package import 服务的 sys.path.insert hack；正常环境依赖仓库既有的 pip install -e . / 受管 Conda 环境。
- shell wrapper 必须继续从 repo root 执行，参数传递、exit code、日志语义不变。

---

## 7. 最新 main 特有问题：消除 training → deployment 反向依赖

当前最新 main 中：
- training/build_utils.py 调用 deployment/runtime.py::capture_real_runtime()

这是本次必须修复的结构性问题。

将 capture_real_runtime(dataset, cfg) 移到：
- training/build_utils.py::_capture_real_runtime(dataset, cfg)

原因：
- 它只在 training-time dataset construction 后执行；
- 它从已加载 ReplayBuffer root attrs 捕获 real_runtime metadata；
- deployment runtime 只消费保存后的 config，不调用 capture；
- 最正确的上下游关系是通过 config.yaml 传递，而不是 training import deployment。

目标：
~~~text
training
  └─ capture real_runtime
        ↓
saved config.yaml
        ↓
deployment
  └─ consume real_runtime
~~~

不得重新引入：
- datasets/real_policy_contract.py
- duplicated Real semantic dictionary
- 额外 deployment artifact / ABI / compatibility layer

必须保持最新 f82b3e18 / 66c6b826 的方向：
- Dataset/ReplayBuffer 通用；
- Real Raw/canonical 负责生产侧完整数据正确性；
- Policy 只捕获最小数值 runtime metadata；
- 旧实验不新增兼容路径。

---

## 8. datasets/preprocessing.py

从现有位置集中 deterministic evaluation/deployment RGB preprocessing：

移动：
- datasets/base_dataset.py::preprocess_validation_rgb
- common/inference.py::rgb_preprocessing_kwargs

到：
- dexmani_policy/datasets/preprocessing.py

消费者：
- BaseDataset
- env_runner
- deployment/runtime.py

保持：
- 输入仍为 raw uint8 HWC；
- resize / center crop / keep_uint8 语义不变；
- RGB model-owned ImageProcessor 不改变；
- deployment 不再为了一个图像 transform import 整个 base_dataset 模块。

---

## 9. 根目录入口保持不动；删除全局 cwd hack

以下路径必须保持：
- dexmani_policy/train.py
- dexmani_policy/train_ddp.py
- dexmani_policy/select_best_ckpt.py
- dexmani_policy/eval_best_ckpt.py
- dexmani_policy/record_demo.py
- dexmani_policy/smoke_test.py

不创建 cli/，不迁入 training/evaluation。

删除 common/pytorch_util.py::set_project_root。

理由：
- scripts/training、scripts/eval 已经显式 cd 到 repo root；
- README 已声明正式命令从 repo root 执行；
- Python library 不应隐式 os.chdir 改变全局进程状态。

需要路径时使用显式 Path(__file__) 推导，或者沿用现有 experiment path 解析，不改变 cwd。

删除 set_project_root 后必须逐一检查：
- train.py
- train_ddp.py
- select_best_ckpt.py
- eval_best_ckpt.py
- record_demo.py
- smoke_test.py

确保 Hydra config_path、experiments 路径、结果输出路径在正式 shell 入口下仍与重构前完全一致。

---

## 10. deployment public API 必须保持

dexmani_real 当前只通过 dexmani_policy.deployment public façade 访问 policy repo。

以下公开名字必须继续可用：
- LoadedPolicy
- PolicyInfo
- inspect_policy
- list_experiments
- load_experiment_config
- load_policy
- resolve_checkpoint
- resolve_experiment

允许 deployment/__init__.py 改为从 agents.loader 或其他内部新位置 re-export，但外部 import 路径不得变化：

~~~python
from dexmani_policy.deployment import ...
~~~

仍必须工作。

不得要求 dexmani_real 因本次内部整理修改 import。

---

## 11. 依赖方向约束

完成后目标依赖为：

~~~text
utils
  ↑
  ├──────── agents
  ├──────── datasets
  ├──────── training
  ├──────── evaluation
  ├──────── env_runner
  └──────── deployment

training/checkpoint
        ↑
        ├── training/resume
        ├── training/trainer/workspace
        └── agents/loader

agents/loader
        ↑
        ├── evaluation/protocol
        └── deployment/runtime

datasets/preprocessing
        ↑
        ├── datasets/base_dataset
        ├── env_runner
        └── deployment/runtime

evaluation/protocol
        ↑
        ├── select_best_ckpt.py
        ├── eval_best_ckpt.py
        └── record_demo.py
~~~

禁止出现：
- agents/loader → training/build_utils
- training/build_utils → deployment/runtime
- evaluation/protocol → training/trainer/workspace/build_utils
- utils → agents/datasets/training/evaluation/deployment
- datasets → deployment
- common 或 tools 的任何 import

training/checkpoint.py 是特殊的低层 training artifact 模块：
- 可以被 agents/loader 消费；
- 但它自身必须保持叶子化，不 import training lifecycle 高层模块。

---

## 12. 行为不变量

本任务禁止改变以下行为：

### Checkpoint
- TRAIN_CHECKPOINT_FORMAT 保持 simple.v3。
- payload root/state/weights schema 不变。
- raw / EMA state key 不变。
- optimizer/scheduler/RNG/position state 不变。
- atomic temp-save + replace 行为不变。
- latest.pt 与 explicit path resolution 语义不变。
- strict resume contract 行为不变。

### Agent / Normalization
- Hydra agent._target_ 全部不变。
- Agent constructor 参数不变。
- normalizer state_dict key 不变。
- normalization modes 与 action:auto 行为不变。
- full-checkpoint inference 仍跳过仅训练初始化依赖的现有逻辑。
- raw / EMA 缺失处理不变。

### Evaluation
- best_ckpt.json schema 不变。
- selection seed protocol 不变。
- held-out seed exclusion 不变。
- checkpoint ranking / tie-break 逻辑不变。
- inference_steps 只表示 inference NFE；不恢复任何 legacy alias。
- result_details / selection summary 输出路径和 schema 不变。

### Real
- saved config 仍是模型构造/输入语义事实源。
- real_runtime 仍只保存当前最小字段：
  - dt
  - optional pointcloud
  - optional fingertip_link_names
- Dataset/ReplayBuffer 继续 domain-agnostic。
- Real producer 继续负责 canonical 数据正确性。
- deployment 不重新打开训练 Zarr。
- dexmani_policy.deployment public API 不变。

### CLI
- 根目录 Python 入口路径不变。
- scripts/training、scripts/eval、scripts/remote 的用户命令行为不变。
- CLI 参数名不变。
- 不重新引入 denoise_steps 等旧 alias。
- exit code / fail-fast 语义不变。

---

## 13. 推荐实施顺序

必须按以下顺序或等价的低风险顺序实施，避免中间状态产生循环 import：

### Phase A — 低层 helper
1. 建立 utils/random.py、utils/tensor.py、utils/config.py。
2. 迁移纯 helper 并更新 consumers。
3. 删除已确认无 consumer 的 ensure_tensor / worker_init_fn。
4. 暂不删除 common 文件，先让新旧路径完成切换。

### Phase B — Agent 数值语义
5. common/normalizer.py → agents/normalization.py。
6. 更新 BaseAgent、各 Agent、training、VQ workflow imports。
7. 保证 normalizer state_dict roundtrip 不变。

### Phase C — Checkpoint
8. 建 training/checkpoint.py。
9. 将 resume-only contract 迁入 training/resume.py。
10. 更新 trainer/workspace/train_ddp/smoke imports。
11. targeted 验证 checkpoint save/load + strict resume。

### Phase D — Loader
12. 建 agents/loader.py。
13. 将 resolve_normalization_spec 移出 training/build_utils 到 utils/config.py。
14. 更新 deployment/evaluation/smoke 的 Agent restore 调用。
15. 确认 agents/loader 不 import training/build_utils。

### Phase E — Dataset preprocessing
16. 建 datasets/preprocessing.py。
17. 移动 RGB deterministic preprocessing helper。
18. 更新 BaseDataset、EnvRunner、Deployment imports。

### Phase F — Evaluation
19. 建 evaluation/protocol.py。
20. training/eval_utils.py 内容迁移进去。
21. 根目录三个评测入口只改 import，文件路径保持不动。
22. 删除 training/eval_utils.py。

### Phase G — Real metadata dependency
23. capture_real_runtime 移到 training/build_utils.py 私有 helper。
24. 删除 training → deployment import。
25. 验证 saved config 中 real_runtime 内容与重构前一致。

### Phase H — Tools
26. tools 下四个 executable workflow 移到已有 scripts/training 与 scripts/remote。
27. 修改 shell wrapper。
28. 删除 dexmani_policy/tools/。

### Phase I — Cleanup
29. 删除 set_project_root 并改为显式路径。
30. 确认所有 common consumer 已切换。
31. 删除 dexmani_policy/common/。
32. 更新 README.md / AGENTS.md 的稳定仓库结构与入口说明。
33. 不主动改 docs/，除非当前公共命令路径已实际失效。

---

## 14. Validation Ladder

遵循 AGENTS.md，且至少执行：

### 14.1 静态检查
- python -m compileall dexmani_policy
- 检查所有新脚本的语法。
- 搜索 stale imports。

必须为 0：
~~~bash
grep -R "dexmani_policy.common" -n dexmani_policy scripts
grep -R "dexmani_policy.tools" -n dexmani_policy scripts
grep -R "from dexmani_policy.deployment" -n dexmani_policy/training
grep -R "training.build_utils" -n dexmani_policy/agents/loader.py
~~~

确认目录不存在：
~~~bash
test ! -d dexmani_policy/common
test ! -d dexmani_policy/tools
~~~

### 14.2 Config-only smoke
至少覆盖当前代表性配置：
- dp
- dp3
- r3d
- maniflow
- sat
- dqrise
- multitask_dit

执行当前仓库规定的：
~~~bash
conda run --no-capture-output -n policy   python dexmani_policy/smoke_test.py --config-only <config_name>
~~~

### 14.3 Targeted CPU regression
必须有最小、临时、不提交 tests/ 的针对性检查：
- TrainCheckpoint save/load schema roundtrip；
- fix_state_dict 对 DDP / _orig_mod prefix 的行为；
- normalizer state_dict roundtrip；
- load_experiment_config / best checkpoint path security checks；
- evaluation milestone discovery/path resolution；
- RGB preprocessing shape/dtype；
- resolve_dataset_paths remote preflight helper；
- deployment package public imports。

### 14.4 Full smoke
若环境具备 GPU/数据/权重，至少选代表性：
- 一个 RGB policy（dp）
- 一个 3D policy（dp3 或 r3d）
- 一个 Flow policy（maniflow）
- 一个 VQ policy（dqrise）

执行现有 full smoke。

若环境不具备，明确标记 NOT VERIFIED，不得为了让 smoke 通过修改算法逻辑。

### 14.5 External boundary
至少执行纯 import/inspection 级验证：
~~~python
from dexmani_policy.deployment import (
    PolicyInfo,
    inspect_policy,
    load_experiment_config,
    load_policy,
    resolve_checkpoint,
    resolve_experiment,
)
~~~

如果本机同时有 dexmani_real，可执行其不连接硬件的 policy preflight/import path；不得启动真机运动。

---

## 15. 完成条件

全部满足才算完成：

1. dexmani_policy/common/ 不存在。
2. dexmani_policy/tools/ 不存在。
3. training/checkpoint.py 是唯一 checkpoint artifact owner。
4. agents/loader.py 是共享 Agent restore owner。
5. evaluation/protocol.py 是三个 offline evaluation 根入口的共享 protocol owner。
6. 根目录 train/eval/demo/smoke 文件路径保持不变。
7. training 不 import deployment。
8. agents/loader 不 import training/build_utils。
9. Dataset/ReplayBuffer 保持通用，不重新引入 Real semantic contract。
10. dexmani_policy.deployment public API 不变。
11. checkpoint / normalizer / best_ckpt / real_runtime schema 不变。
12. 没有新增 legacy compatibility。
13. 没有新增无必要目录。
14. README/AGENTS 与当前公共结构一致。
15. 所有实际执行的验证结果被记录；未执行的高成本验证标记 NOT VERIFIED。

---

## 16. 明确禁止事项

本任务不要：

- 移动 train.py、train_ddp.py、select_best_ckpt.py、eval_best_ckpt.py、record_demo.py、smoke_test.py。
- 新建 package-root checkpoint.py、inference.py、evaluation.py、experiment.py。
- 新建 core/、contracts/、artifacts/、cli/、commands/、processors/。
- 将 agents 改名 policies。
- 将 env_runner 改名 rollout。
- 引入 src/ layout。
- 引入 LeRobot 风格 processor/factory/plugin 系统。
- 创建 committed tests/ 目录。
- 修改 Policy architecture、action representation、loss、scheduler、NFE recipe。
- 修改 Hydra _target_。
- 修改 checkpoint format/schema。
- 为旧实验或旧 CLI 增加兼容分支。
- 把 VQ workflow 塞进 utils。
- 创建 utils/pytorch.py 或 utils/misc.py。
- 为单 consumer 或 3～5 行 helper 过度抽象。
- 在没有必要的情况下修改 docs/。

---

## 17. 设计原则摘要

本任务最终采用的架构原则：

**领域代码归领域；训练产物归 training；Agent 恢复归 agents；评测协议归 evaluation；真正无领域语义的 helper 才归 utils；可执行研究 workflow 归 repo scripts。**

借鉴 Diffusion Policy：
- 保持研究代码入口直接；
- 不做平台级抽象；
- normalization 属于 model/policy semantic，而不是杂项 helper。

借鉴 LeRobot：
- domain ownership 清晰；
- random/tensor 等真正通用 helper 放 utils；
- executable workflow 与 library mechanism 分离。

但不复制 LeRobot 的平台复杂度。

最终目标不是“最抽象”，而是：
**最少目录、最短依赖、最清晰 owner、零行为漂移、可持续研究迭代。**
