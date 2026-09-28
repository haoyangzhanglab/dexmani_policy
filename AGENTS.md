# AGENTS.md — DexMani_Policy

这是一个**个人机器人学习研究仓库**。AI coding 的首要目标是帮助快速、可靠地验证研究想法，而不是追求企业级抽象、兼容层或文档完备度。

## Priorities

按以下顺序做判断：

1. 研究问题与实验变量是否清楚。
2. 当前 config / code / data 的真实行为是否正确。
3. 实现是否足够简单，能快速修改和比较。
4. 结果是否可复现、可解释。
5. 工程美化只做到当前研究所需程度。

避免为了“更通用”“更优雅”主动引入 framework、registry、factory、兼容层或大范围重构。

## Source of Truth

- **当前 Policy**：`dexmani_policy/configs/*.yaml` + `agent._target_` 指向的实际源码。
- **已有实验**：实验目录中保存的 resolved `config.yaml` + checkpoint。
- **运行行为**：实际训练、评测、loader、runtime 校验代码。
- `README.md` 和 `docs/` 用于导航与背景，不替代 config/code。

不要假设 config 名、文件名和 Agent 类名一一对应。

## Working Style

- 先打开相关 config 和真实调用链，再改代码；不要对没读过的实现做推断。
- 优先最小、完整的 diff。修当前问题，不顺带整理邻近代码。
- 能复用现有组件就复用；只有研究语义确实不同才新建组件。
- 研究特有机制尽量局部化，不把一次实验抽象成全仓基础设施。
- 保持数据、observation、action、normalization、objective、inference 的语义边界清楚。
- 遇到粗略研究想法时，补齐最简单合理的实现细节并推进；不要因为非关键选择阻塞工作。
- 不为单次超参数、模型宽度、NFE、batch size 或实验结论修改全局文档。

## Policy Changes

处理一个 Policy idea 时优先沿这条路径：

```text
config
→ agent._target_
→ Agent
→ observation encoder
→ backbone / action decoder
→ compute_loss
→ predict_action
```

明确当前改动属于哪一层：

- observation / representation
- policy architecture
- action representation
- training objective
- inference algorithm

一次实验尽量只改变少数层，便于解释结果。

## Validation

验证遵循“便宜优先”：

1. 文档改动：检查 diff、路径和命令是否真实。
2. Python 改动：语法 / import / targeted check。
3. Config 或 Policy 改动：
   `python dexmani_policy/smoke_test.py --config-only <config_name>`
4. 改到 model/data/forward 链路时，再考虑：
   `python dexmani_policy/smoke_test.py <config_name>`
5. 完整训练、DDP、长时间评测或视频录制，只在用户明确要求或验证确实需要时启动。

只把真正执行过的检查报告为 PASS；受 GPU、数据、权重、仿真环境限制的项目写明 `NOT VERIFIED`。

## Research Artifacts and Safety

- 不删除或覆盖与当前任务无关的 `robot_data/`、`experiments/`、checkpoint、视频、W&B 日志、预训练权重。
- 不为了通过检查修改算法语义或降低验证标准。
- 不自动启动真机运动。Real robot motion 必须由用户明确要求。
- 已有实验恢复时尊重保存的 resolved config 与 checkpoint；不要凭当前默认 config 猜历史模型。

## Stable Entry Points

- `dexmani_policy/configs/` — Policy configs
- `dexmani_policy/agents/` — Agent / encoders / backbones / decoders
- `dexmani_policy/datasets/` — datasets and preprocessing
- `dexmani_policy/training/` — training / checkpoint / resume
- `dexmani_policy/evaluation/` — offline evaluation
- `dexmani_policy/env_runner/` — simulation
- `dexmani_policy/deployment/` — saved-policy / Real runtime
- `dexmani_policy/train.py`, `train_ddp.py`, `smoke_test.py` — root entry points
- `scripts/training/`, `scripts/eval/`, `scripts/remote/` — user workflows

## Skills

当任务匹配时优先使用仓库内的短 workflow skill：

- `research-iterate`：研究想法 → 最小实现 → targeted validation → 可运行实验命令。
- `preflight-experiment`：训练/评测前检查 config、模态、shape、normalization、checkpoint 与 inference 设置。

这些 skill 是工作流提示，不是新的工程层；不要为了 skill 再新增 supporting framework。

