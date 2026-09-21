# CLAUDE.md — DexMani_Policy

本文件是精简且独立可用的 AI 工作速查。若 `AGENTS.md` 可用，以其作为主要项目 contract；本文件保持必要的入口与规则，避免维护 Policy-specific 快照。

## Source of Truth

- 已有实验：读取实验目录保存的 resolved `config.yaml` 和 checkpoint contract。离线评测的模型构造、action/window/normalization 属于 selected checkpoint；当前 config 提供环境与评测 protocol。
- 当前 Policy：读取 Hydra config，并沿 `agent._target_` 进入实际实现。
- 当前架构、超参数、tensor shape、训练目标和推理算法：以 config + Python 代码为准。
- README / CLAUDE / docs 只做导航和背景，不维护当前 Policy 列表或参数矩阵。

不要假设 config 名与 Python module/class 同名；只信 `_target_`。

## 命令速查

```bash
# 发现当前单卡 / DDP config
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml

# 单卡 / DDP 训练
bash scripts/training/train.sh <config_name> 'task_name=<task>'
bash scripts/training/train_ddp.sh ddp/<config_name> 'task_name=<task>'

# VQ 手部预训练（后续参数使用 --option value）
bash scripts/training/train_vq_hand.sh <task_name>

# 快速 config 检查（无数据/GPU forward）
python dexmani_policy/smoke_test.py --config-only <config_name>

# 完整集成 smoke
python dexmani_policy/smoke_test.py <config_name>

# 评测
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>

# 显式续训
bash scripts/training/train.sh <config_name> \
  'task_name=<task>' \
  '+resume_from=/absolute/path/to/experiment_or_checkpoint'

# Deployment export（研究者日常路径；run_policy 位于 dexmani_real）
python -m dexmani_policy.deployment.export <experiment_dir> --checkpoint best

# Python 语法检查
conda run --no-capture-output -n policy python -m compileall dexmani_policy
```

本地 `scripts/training/` 启动器通过 `conda run --no-capture-output -n policy` 和 `python -u` 自动选择环境并实时输出日志，无需提前激活；要求 `conda` 在 `PATH` 中可用。不要在启用 `set -u` 的 shell 中直接执行 Conda 激活钩子，以免触发未定义变量错误。

单卡和 DDP 启动器的 `-h` / `--help` 无需 Conda，返回退出码 0；VQ 的帮助参数转发给 Python，需要 `policy` 环境。单卡 / DDP 命令追加 `--cfg job --resolve` 可仅预览解析配置，不启动训练，也不替代 config validation。

重复执行同一训练命令不会自动 resume；resume 必须显式指定。

## Policy 工作路径

```text
resolved config
→ agent._target_
→ Agent construction
→ obs_encoder
→ backbone / action_decoder
→ compute_loss
→ predict_action
→ control_action
```

修改前明确：

- Observation Representation
- Policy Architecture
- Action Representation
- Training Objective
- Inference Algorithm

Policy-local 改动保持局部；共享组件改动先搜索真实依赖者。正常新增或修改 Policy 不要求同步更新 README/AGENTS/CLAUDE。

## 验证

按以下顺序逐步升级：

```text
syntax/import
→ smoke_test.py --config-only <config>
→ smoke_test.py <config>
→ shared-code dependent configs
→ minimal train/eval reproduction when needed
```

不要自动启动完整训练、DDP、长评测或视频。环境限制导致的未执行项明确报告为 NOT VERIFIED。

仓库有意不保留 `tests/`；使用现有 smoke 入口与临时定向回归验证，不为检查恢复测试目录。语法和 config-only 检查不能代替 checkpoint strict restore 或实际 inference 验证。

## Deployment

Deployment 的 model/data semantics 属于 **selected checkpoint**（`resume_contract.agent` / `agent_config` / `dataset` / `agent.normalization`），不属于当前 `config.yaml`。`config.yaml` 只为 deployment 提供 experiment identity 与 inference recipe。完整规则见 `AGENTS.md` 的 “Deployment Boundary”。

改 deployment 时不要：从当前 config 重新推导 agent/dataset/normalization 语义、恢复 config↔checkpoint reconciliation、在 `contract.py` 之外另写一套 metadata grammar、重新引入 `verify` / `publish` 公开开关。`deployment/contract.py` 是唯一 artifact metadata parser。

## Repository Map

- `dexmani_policy/configs/`：Hydra config
- `dexmani_policy/agents/`：Policy implementation
- `dexmani_policy/datasets/`：data pipeline
- `dexmani_policy/training/`：training/resume/workspace
- `dexmani_policy/env_runner/`：simulation
- `dexmani_policy/deployment/`：Real deployment
- `scripts/`：training/eval/remote entry points

`docs/` 作为背景材料使用；除非用户明确要求，不修改。

## NaN 两层防护

Trainer 当前在 loss-side 和 gradient-side 都有 non-finite protection。诊断具体失败时读取当前 `trainer.py` 和错误日志，不依赖本文件保存固定行号或经验性修复结论；可使用 `dexmani-training-debug` skill。

## 已知硬编码与设计约定

实现中的 fixed scheduler、特殊 conditioning、cache 或其他 policy-local 约束，以其源码、测试和调用者为准。本文件不复制具体数值。修改这类行为时应追踪真实依赖并做 targeted regression。

## Skills

- `dexmani-agent-integration`：新增 / 重构 Policy。
- `dexmani-pr-check`：diff-aware PR review。
- `dexmani-training-debug`：NaN/Inf 训练诊断。

Skill 负责流程；Policy 事实始终由 resolved config 和源码提供。
