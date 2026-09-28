# DexMani_Policy

面向灵巧手操作研究的个人策略仓库。这里的目标不是建设一个通用、长期兼容的机器人学习平台，而是让新的研究想法能够以尽量小的改动快速落地、验证和比较。

当前主要关注：3D / 点云视觉运动策略、机械臂与灵巧手协同、本体感觉与触觉、多模态表示，以及 diffusion / flow matching 等动作生成方法。

> **事实来源**：当前 Policy 以 Hydra config 和实际源码为准；已有实验以保存的 resolved `config.yaml` 与 checkpoint 为准。README 只维护稳定入口和研究工作流，不记录易过期的模型细节与超参数。

## 研究定位

这个仓库优先服务于研究迭代：

- **问题驱动**：先明确研究假设和主要变量，再改模型。
- **小步修改**：尽量保持 baseline 不变，只改真正需要验证的机制。
- **实现直接**：研究特有逻辑优先局部实现，不为了潜在复用提前搭框架。
- **结果可比较**：config、实验目录和 checkpoint 应足以还原一次实验的实际语义。

## 快速开始

项目要求 Python 3.10+，默认使用 Conda 环境 `policy`：

```bash
conda activate policy
pip install -e .
```

仿真评测需要另外安装 `dexmani_sim`。训练数据路径由 config 决定，常见位置为 `robot_data/<task>.zarr`。

查看当前可用配置：

```bash
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml
```

不要根据 config 文件名猜模型实现；真正的入口是 config 中的 `agent._target_`。

## 推荐研究流程

一次典型迭代可以保持得很短：

```text
研究假设
  ↓
选择 baseline config
  ↓
沿 agent._target_ 找到真实实现
  ↓
完成最小可运行修改
  ↓
config-only / smoke 验证
  ↓
训练 → 评测 → 对比
```

具体的模型宽度、学习率、NFE、batch size、实验结论等应留在 config、实验目录和日志中，而不是写进全局文档。

## 训练

单卡：

```bash
bash scripts/training/train.sh <config_name> 'task_name=<task>'
```

DDP：

```bash
bash scripts/training/train_ddp.sh ddp/<config_name> 'task_name=<task>'
```

Hydra override 直接追加在命令后，例如：

```bash
bash scripts/training/train.sh <config_name> \
  'task_name=<task>' \
  'training.seed=42'
```

需要续训时显式指定 `resume_from`；重复运行同一训练命令不会自动续训。

## 验证

修改 config 或 Policy 后，优先执行轻量检查：

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

当改动涉及 dataset、model 构造、forward/backward 或推理链路时，再运行完整 smoke：

```bash
python dexmani_policy/smoke_test.py <config_name>
```

完整训练、DDP 和长时间评测不是普通代码改动后的默认验证步骤。

## 仿真评测

常用一键入口：

```bash
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>
```

也可以分步执行：

```bash
bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/eval_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name>
```

评测已有实验时，模型结构、输入模态、action/window/normalization 与预处理从该实验保存的 resolved config 恢复；权重和 fitted normalizer 从 checkpoint 恢复。

## 其他工作流

- VQ / codebook 等辅助训练流程：`scripts/training/`
- 远端训练、数据同步与日志管理：`scripts/remote/`
- 仓库级辅助脚本：`scripts/utils/`
- Real policy inspection / inference：`dexmani_policy/deployment/`

真机运动不属于普通开发验证；只有在明确需要时才进入 Real 流程。

## 项目结构

```text
dexmani_policy/
  configs/       Hydra 配置
  agents/        Agent、观测编码器、backbone、action decoder
  datasets/      Dataset、ReplayBuffer、预处理
  training/      训练、checkpoint、resume、EMA
  evaluation/    离线评测共享逻辑
  env_runner/    仿真环境执行
  deployment/    已保存策略检查与 Real runtime
  utils/         少量跨模块通用工具

scripts/
  training/      训练与辅助研究流程
  eval/          仿真评测入口
  remote/        远端训练与同步
  utils/         仓库辅助脚本
```

## 文档与 AI 编码

- [`AGENTS.md`](AGENTS.md)：Codex / Claude 共用的仓库级 AI 研究协作约定。
- [`CLAUDE.md`](CLAUDE.md)：Claude Code 的轻量入口，不重复维护规则。
- [`docs/项目架构.md`](docs/项目架构.md)：项目背景与架构说明。
- [`docs/仿真评测机制.md`](docs/仿真评测机制.md)：仿真评测机制说明。
- [`docs/SSH服务器训练部署.md`](docs/SSH服务器训练部署.md)：远端训练与同步说明。

`docs/` 用于背景和机制说明；当文档与当前 config/code 不一致时，以当前实现为准。

仓库还提供两个短 workflow skill：

- `research-iterate`：研究想法 → 最小实现 → 定向验证 → 下一条实验命令。
- `preflight-experiment`：在消耗 GPU 时间前检查 config、数据、模型、checkpoint 和推理设置是否对齐。

Codex skill 位于 `.codex/skills/`，Claude Code skill 位于 `.claude/skills/`。

