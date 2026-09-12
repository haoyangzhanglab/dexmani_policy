# AGENTS.md — DexMani_Policy

本文件是本仓库的项目级 coding contract。目标是让 Policy 高频迭代保持低维护成本：**当前 Policy 事实来自 config/code，本文只维护稳定的工程规则和工作方法。**

## Source of Truth

按任务类型使用以下事实来源：

1. **已有实验**：优先读取实验目录保存的 resolved `config.yaml` 和 checkpoint contract。
2. **当前 Policy**：读取 `dexmani_policy/configs/<config>.yaml`，再沿 `agent._target_` 进入实际 Python 实现。
3. **运行时接口**：以 `validate_config`、Agent/runtime 校验、训练/评测入口代码为准。
4. `README.md`、`CLAUDE.md` 和 `docs/` 用于导航与背景，不应作为当前 Policy 架构、超参数或 tensor shape 的最终依据。

不要假设 config 名、Python 文件名和 Agent 类名相同。Hydra `_target_` 是实现入口；仓库没有需要同步维护的 Policy registry。

## Policy-specific Work

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

正常的 Policy architecture / hyperparameter 修改，或新增一个不改变公共接口的 Policy，**不需要同步修改 README、AGENTS、CLAUDE 或 Skills**。只有公共 CLI、仓库级接口、通用工作流或环境契约变化时才修改对应全局文件。

不要把 layer 数、hidden dim、LR/WD、NFE、batch size、参数量、当前 Policy 列表或实验结论加入全局 AI 文档。

## Repository Entry Points

- `dexmani_policy/configs/`：Hydra Policy config；`ddp/` 为可选 DDP overlay。
- `dexmani_policy/agents/`：Agent、observation encoder、backbone、action decoder。
- `dexmani_policy/datasets/`：数据和 sampler。
- `dexmani_policy/training/`：build、trainer、EMA、resume、workspace。
- `dexmani_policy/env_runner/`：simulation runner。
- `dexmani_policy/deployment/`：Real deployment artifact/runtime。
- `dexmani_policy/train.py` / `train_ddp.py`：训练入口。
- `dexmani_policy/smoke_test.py`：配置与集成验证。
- `scripts/training/`、`scripts/eval/`、`scripts/remote/`：操作入口。

`docs/` 当前视为冻结背景文档；除非用户明确要求，不要修改。

## Environment and Safety

- 使用 Python 3.10+ 和 Conda 环境 `policy`。
- 非交互 shell 优先使用 `conda run -n policy <command>`。
- `pip install -e .` 只覆盖项目声明的核心依赖；完整 Policy 依赖由受管环境提供。
- 训练数据路径来自 config，通常为仓库根目录下 `robot_data/<task>.zarr`。
- 仿真评测需要安装 `dexmani_sim`。
- 完整 smoke、训练和大多数评测依赖 CUDA/GPU。环境缺少 GPU、数据、权重、显示服务或 `dexmani_sim` 时，报告限制，不要为了让检查通过而改核心逻辑。
- 不修改或提交 `robot_data/`、`experiments/`、checkpoint、视频、W&B 日志、预训练权重等生成物。
- 不覆盖与当前任务无关的用户改动。
- 除非用户明确要求，不自动启动完整训练、DDP、长时间评测或视频录制。

## Engineering Rules

- 先读实际调用链，再修改；优先修根因。
- 改动保持小而完整，不顺带重构无关模块。
- 修改共享基类/组件前，先通过 config/import 搜索真实依赖者和 blast radius。
- 不用全局 prompt 固化当前研究 recipe。易变的架构选择和实验结论应留在 config、实现、局部注释或测试附近。
- 对“看起来奇怪”的实现，先检查局部代码、测试和调用者；不要依赖历史文档断言它一定是 intentional design。

## Validation Ladder

按成本从低到高验证：

1. Python 语法/导入检查。
2. Config 改动或新增 Policy：
   `python dexmani_policy/smoke_test.py --config-only <config_name>`
3. Agent/encoder/backbone/decoder/config 改动：
   `python dexmani_policy/smoke_test.py <config_name>`
4. 共享模块改动：搜索实际依赖者，并对代表性受影响 config 增加 targeted smoke。
5. 训练/评测行为改动：使用最小可证明场景；只有用户明确要求时再扩大到完整运行。

`--config-only` 负责 Hydra resolve、公共 config validation、target module 检查和 `agent._target_` import；它不实例化 env runner。full smoke 才验证 dataset → model → optimizer → forward/backward → inference → checkpoint roundtrip。

未执行的 GPU/数据相关验证必须明确标为 **NOT VERIFIED**，不能仅凭代码阅读报告 PASS。

## Project Skills

- `dexmani-agent-integration`：新增或重构 Policy。
- `dexmani-pr-check`：PR 前的 diff-aware correctness review。
- `dexmani-training-debug`：训练 NaN/Inf 等数值失败诊断。

Skill 只定义过程，不拥有当前 Policy 事实。当前实现始终回到 resolved config 和源码确认。
