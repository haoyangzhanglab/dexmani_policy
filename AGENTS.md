# AGENTS.md — DexMani_Policy

本文件是 **Codex 与 Claude 共用的项目工程规范**。Codex 直接读取本文件，Claude 通过根目录 `CLAUDE.md` 的原生导入加载同一份内容。共享规则只在此维护；操作命令见 [README.md](README.md)。工具专属的模型、权限和子代理设置留在各自配置文件。

**当前 Policy 事实来自 config/code，本文只维护稳定的工程规则和工作方法。**

## Source of Truth

按任务类型使用以下事实来源：

1. **已有实验**：优先读取实验目录保存的 resolved `config.yaml` 和 checkpoint contract。离线评测的模型构造、action/window/normalization 语义以 selected checkpoint 的 `resume_contract` 为准；当前 config 提供环境与评测 protocol。Deployment 的模型/数据语义也属于 selected checkpoint，见 “Deployment Boundary”。
2. **当前 Policy**：读取 `dexmani_policy/configs/<config>.yaml`，再沿 `agent._target_` 进入实际 Python 实现。
3. **运行时接口**：以 `validate_config`、Agent/runtime 校验、训练/评测入口代码为准。
4. `README.md` 提供操作说明，`docs/` 提供背景；二者均不作为当前 Policy 架构、超参数或 tensor shape 的最终依据。`CLAUDE.md` 仅负责加载本规范。

不要假设 config 名、Python 文件名和 Agent 类名相同。Hydra `_target_` 是实现入口；仓库没有需要同步维护的 Policy registry。

## Working Method

- 先读实际调用链，再修改；优先修根因，改动保持小而完整，不顺带重构无关模块。
- 修改共享基类/组件前，通过 config/import 搜索真实依赖者和影响范围，再选择定向回归。
- 对特殊 conditioning、scheduler、cache 等看起来奇怪的实现，先检查局部代码、验证场景和调用者；不要依赖历史文档断言它一定是 intentional design。

### Policy 工作路径

处理 Policy 新增、修改或 review 时，从 resolved config 开始追踪：

```text
config
→ agent._target_
→ Agent.__init__
→ obs_encoder
→ backbone / action_decoder
→ compute_loss
→ predict_action
→ environment-facing control_action
```

明确区分并核对：

- Observation Representation
- Policy Architecture
- Action Representation
- Training Objective
- Inference Algorithm

不要从其他 Policy 机械复制实现。优先复用语义真正一致的共享组件；Policy 特有逻辑保持局部化。

### 文档维护

- 正常的 Policy architecture / hyperparameter 修改，或新增一个不改变公共接口的 Policy，**不需要同步修改 README、AGENTS、CLAUDE 或 Skills**。只有公共 CLI、仓库级接口、通用工作流或环境契约变化时才修改对应全局文件。
- 不要把 layer 数、hidden dim、LR/WD、NFE、batch size、参数量、当前 Policy 列表或实验结论加入全局 AI 文档。易变的研究 recipe 留在 config、实现、实验快照或局部验证附近。
- `docs/` 视为冻结背景文档；除非用户明确要求，不要修改。

## Repository Entry Points

- `dexmani_policy/configs/`：Hydra Policy config；`ddp/` 为可选 DDP overlay。
- `dexmani_policy/agents/`：Agent、observation encoder、backbone、action decoder。
- `dexmani_policy/datasets/`：数据和 sampler。
- `dexmani_policy/training/`：build、trainer、EMA、resume、workspace。
- `dexmani_policy/env_runner/`：simulation runner。
- `dexmani_policy/deployment/`：Real deployment artifact/runtime。
- `dexmani_policy/train.py` / `train_ddp.py`：训练入口。
- `dexmani_policy/smoke_test.py`：配置与集成验证。
- `scripts/training/`、`scripts/eval/`、`scripts/deployment/`、`scripts/remote/`：操作入口。

## Environment and Safety

- 使用 Python 3.10+ 和 Conda 环境 `policy`；非交互 shell 优先使用 `conda run --no-capture-output -n policy <command>`。
- `pip install -e .` 只覆盖项目声明的核心依赖；完整 Policy 依赖由受管环境提供。仿真评测需要安装 `dexmani_sim`。
- Shell 启动器通过 `conda run` 选择环境；不要在启用 `set -u` 的 shell 中直接执行 Conda 激活钩子。
- 训练数据路径来自 config，通常为仓库根目录下 `robot_data/<task>.zarr`。
- 完整 smoke、训练和大多数评测依赖 CUDA/GPU。环境缺少 GPU、数据、权重、显示服务或 `dexmani_sim` 时，报告限制，不要为了让检查通过而改核心逻辑。
- 不修改或提交 `robot_data/`、`experiments/`、checkpoint、视频、W&B 日志、预训练权重等生成物；不覆盖与当前任务无关的用户改动。
- 除非用户明确要求，不自动启动完整训练、DDP、长时间评测或视频录制。
- 续训必须显式指定 `resume_from` 并满足 strict resume contract；重复训练命令不会自动续训。

## Validation Ladder

按改动范围选择验证，成本从低到高：

1. Python 改动：语法/导入检查。
2. Config 改动或新增 Policy：
   `conda run --no-capture-output -n policy python dexmani_policy/smoke_test.py --config-only <config_name>`
3. Agent/encoder/backbone/decoder/config 改动：
   `conda run --no-capture-output -n policy python dexmani_policy/smoke_test.py <config_name>`
4. 共享模块改动：搜索实际依赖者，对代表性受影响 config 增加 targeted smoke。
5. 训练/评测行为改动：使用最小可证明场景；只有用户明确要求时再扩大到完整运行。

`--config-only` 负责 Hydra resolve、公共 config validation、target module 检查和 `agent._target_` import；它不实例化 env runner。full smoke 才验证 dataset/normalizer → model/EMA → optimizer/scheduler → forward/backward → inference → checkpoint roundtrip。语法和 config-only 检查不能代替 checkpoint strict restore 或实际 inference 验证。

仓库有意不保留 `tests/`；使用现有 smoke 入口与临时定向回归验证，不为检查恢复测试目录。纯文档改动检查链接、命令与源码的一致性及 diff，不要求运行模型 smoke。

只有实际执行的检查才能报告 PASS。环境限制或未执行的相关验证明确标为 **NOT VERIFIED**，尤其是 GPU/数据相关验证。

## Deployment Boundary

Deployment 的 trained semantics 属于 **selected checkpoint**，不属于当前 experiment `config.yaml`：

```text
Checkpoint owns:   resume_contract.agent / agent_config / dataset / agent.normalization
                   + deployment_data_semantics
                   （architecture、action/window、normalization、dataset constructor
                     config，以及从训练实际使用的 Zarr 捕获的数据语义快照）
config.yaml owns:  experiment identity（policy_name / task_name）
                   + inference recipe（eval.use_ema / eval.denoise_steps；
                     best selector 用 best_ckpt.json["inference"]）
Artifact owns:     selected weights（raw 或 EMA，只存选中的一套）、
                   checkpoint-owned observation/action/data semantics、
                   default denoise_steps
Runtime owns:      显式 --inference-steps override（NFE ablation）

Export:            build -> save -> safe reload -> strict restore
                   -> synthetic predict -> atomic deployment_latest.pt selector swap
```

规则：

- 不要为 deployment 从当前 config 重新推导 agent/dataset/normalization 语义，也不要恢复 config↔checkpoint 的 reconciliation。修改 `config.yaml.agent/dataset/normalization` 不得改变旧 checkpoint 的 deployment 行为。
- 删除 reconciliation **不等于**删除 validation：checkpoint 自身的 action/window/normalization contract 仍必须 strict parse、fail-fast，不允许 blind trust。
- Real Policy Zarr 数据语义的唯一 extractor 是 `datasets/real_policy_contract.py`，由 training（写 checkpoint 快照）与 deployment export（解析 selected/override Zarr）共用；不要维护第二套同义解析逻辑。
- `--zarr-path` 只允许 **semantic-equivalent relocation**：export-time Zarr 的语义必须与 checkpoint `deployment_data_semantics` strict equality（dt、point-cloud 预处理/table plane、action EE frame/components、raw shape/dtype、task identity 等任何漂移都 fail）。Artifact 数据语义的 source of truth 永远是 checkpoint snapshot；`schema_version` 仅为 informational，可取自实际 Zarr。
- 旧 checkpoint 缺少 `deployment_data_semantics` 时：`CheckpointStore.load()` 与 offline analysis 照常，Real deployment export 明确 fail；不做 retrofit / migration / legacy 分支。
- Public export 永远 verify（safe reload -> strict restore -> deterministic synthetic prediction）后才 atomic swap `deployment_latest.pt` selector，不引入 `verify` / `publish` 公开开关，不允许跳过验证；selector swap 成功后不要再执行可能失败的操作。verify 阶段任何失败必须删除本次 candidate、保持旧 selector，使同名命令可直接 retry；candidate cleanup 失败必须显式报错，不得 silent swallow。不使用 fsync / rollback / double-fault transaction machinery。
- `deployment/contract.py` 是 persisted deployment **metadata** 的 shared grammar（action/window、observation fields、RGB preprocessing、normalization、nested `_target_` allowlist），export / inspect / restore 不得各写一套。它不负责 state-dict grammar、Real physics 语义、publication 或 decoder-specific NFE 限制；`inspect_experiment()` 保证 metadata 可读/合法，weights 由 `load_experiment` 的 strict restore 验证。
- 保持 `simple.v3`，不要为 deployment 引入 hashing、signature、flock 或 code-version gate；没有 qualify/release subsystem，研究者日常路径是 `export -> run_policy`。

## Project Skills

按任务读取对应流程；如果当前工具没有自动发现该 skill，直接打开下面链接的 `SKILL.md` 并遵循其流程，无需复制到另一工具的目录。

| 任务 | 共享流程 |
| --- | --- |
| 新增 Policy、架构替换或 action representation 等重大变更 | [dexmani-agent-integration](.agents/skills/dexmani-agent-integration/SKILL.md) |
| Review、审计或 PR 前 correctness 检查 | [dexmani-pr-check](.agents/skills/dexmani-pr-check/SKILL.md) |
| 训练 NaN/Inf 等数值失败诊断 | [dexmani-training-debug](.agents/skills/dexmani-training-debug/SKILL.md) |

Skill 只定义过程，不拥有当前 Policy 事实。当前实现始终回到 resolved config 和源码确认。
