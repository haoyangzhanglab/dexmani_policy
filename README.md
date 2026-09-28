# DexMani_Policy

面向灵巧手操作研究的个人策略仓库。目标是让新的机器人学习想法能够以尽量小的改动快速落地、验证和比较，而不是建设一个通用、长期兼容的软件平台。

当前研究内容会持续变化，可能涉及 3D / 点云、多模态感知、灵巧操作、生成式策略等方向。具体模型结构、输入输出和训练方式始终以**当前 config 与源码**为准，不在 README 中固化。

> **事实来源**：当前实现以 config 和源码为准；已有实验以其保存的 resolved `config.yaml` 与 checkpoint 为准。README 只维护稳定入口、常用命令和研究工作流。

## 研究定位

这个仓库优先服务于快速研究迭代：

- **问题驱动**：先明确研究假设和主要变量，再改代码。
- **控制变量**：尽量保持 baseline 其他部分不变，只修改真正需要验证的机制。
- **直接实现**：研究特有逻辑优先放在最自然的位置，不为潜在复用提前搭框架。
- **结果可追溯**：config、实验目录、日志和 checkpoint 应足以还原一次实验。

## 快速开始

项目要求 Python 3.10+，默认使用 Conda 环境 `policy`：

```bash
conda activate policy
pip install -e .
```

仿真评测需要另外安装 `dexmani_sim`。训练数据路径由当前 config 决定，常见位置为 `robot_data/<task>.zarr`。

查看当前可用配置：

```bash
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml
```

配置名只用于定位实验入口，不代表内部模型一定采用某种固定结构；理解实现时应继续进入实际源码。

## 推荐研究流程

一次典型迭代保持为一个短闭环：

```text
研究问题 / 假设
      ↓
选择当前 baseline
      ↓
确认真实实现与执行路径
      ↓
定义主要实验变量
      ↓
完成最小可运行修改
      ↓
最低成本验证
      ↓
训练 / 评测 / 对比
```

模型层数、宽度、学习率、NFE、batch size、具体模块组合和实验结论等易变化信息，应留在 config、源码、实验目录与日志中，不写成仓库级长期约定。

## 训练

单卡：

```bash
bash scripts/training/train.sh <config_name> 'task_name=<task>'
```

DDP：

```bash
bash scripts/training/train_ddp.sh ddp/<config_name> 'task_name=<task>'
```

Hydra override 可直接追加，例如：

```bash
bash scripts/training/train.sh <config_name> \
  'task_name=<task>' \
  'training.seed=42'
```

需要续训时显式指定 `resume_from`；重复运行同一训练命令不会自动续训。

## 验证

修改 config 或研究实现后，优先执行轻量检查：

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

当改动进入实际数据、模型构造、训练或推理链路时，再运行完整 smoke：

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

评测已有实验时，使用该实验保存的 resolved config 与 checkpoint 恢复其真实语义，不用当前默认配置推测历史实验。

## 其他工作流

- 辅助训练与研究脚本：`scripts/training/`
- 远端训练、数据同步与日志管理：`scripts/remote/`
- 仓库级辅助脚本：`scripts/utils/`
- Real policy inspection / inference：`dexmani_policy/deployment/`

真机运动不属于普通开发验证；只有在明确需要时才进入 Real 流程。

## 项目结构

```text
dexmani_policy/
  configs/       当前实验与运行配置
  agents/        策略与研究实现
  datasets/      数据读取、采样与预处理
  training/      训练、状态保存与恢复
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

## 文档分工

三份根目录文档只承担三个稳定职责：

- **README.md：怎么使用这个研究仓库。** 面向人，维护定位、入口、常用命令和项目地图。
- **AGENTS.md：AI 应该怎样和我一起做研究。** 维护研究协作原则、改动方式与验证方法，不描述某个具体模型。
- **CLAUDE.md：Claude 去哪里读取这些规则。** 只作为加载入口，不维护第二份规范。

较长的背景和机制说明放在 `docs/`：

- [`docs/项目架构.md`](docs/项目架构.md)
- [`docs/仿真评测机制.md`](docs/仿真评测机制.md)
- [`docs/SSH服务器训练部署.md`](docs/SSH服务器训练部署.md)

当背景文档与当前 config/code 不一致时，以当前实现为准。

## AI Skills

仓库提供两个**模型无关**的研究工作流 skill：

- `research-iterate`：研究假设 → 识别真实改动面 → 最小实现 → 验证 → 下一步实验。
- `preflight-experiment`：昂贵实验前检查当前配置、数据、依赖、产物与运行条件是否一致。

Skill 只规定研究工作方法，不写死当前模型的模块划分、输入模态、动作表示或训练算法；模型变化时无需随之重写。

Codex skill 位于 `.codex/skills/`，Claude Code skill 位于 `.claude/skills/`。

