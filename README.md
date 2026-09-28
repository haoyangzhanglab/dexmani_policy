# DexMani_Policy

个人研究用的灵巧手操作策略仓库。目标是用尽量短的代码路径快速验证 robot learning 想法，而不是把仓库建设成通用框架或长期兼容的软件产品。

当前主要关注 3D / point-cloud visuomotor policy、灵巧手与机械臂协同、proprioception / tactile、多模态表示，以及 diffusion / flow-matching 类动作生成。

> **事实来源**：当前 Policy 以 Hydra config 和源码为准；已有实验以保存的 resolved `config.yaml` 与 checkpoint 为准。README 只保留稳定入口。

## Quick Start

```bash
conda activate policy
pip install -e .
```

仿真评测需要另外安装 `dexmani_sim`。训练数据路径由 config 决定，常见位置为 `robot_data/<task>.zarr`。

查看可用配置：

```bash
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml
```

## Research Loop

这个仓库默认采用“小步研究迭代”：

1. 从一个明确假设开始，只改变少量变量。
2. 从 config 的 `agent._target_` 追到真实实现，不根据文件名猜模型。
3. 优先复用已有 encoder / backbone / decoder；研究特有逻辑保持局部。
4. 先做最便宜的 sanity check，再决定是否跑 GPU smoke、训练或评测。
5. 实验结果与具体 recipe 留在 experiment/config/log 中，不写成全局工程规则。

## Train

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

需要续训时显式提供 `resume_from`；重复启动同一命令不会自动续训。

## Validate

修改 config 或 Policy 后，优先跑轻量检查：

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

需要验证实际 model / dataset / forward-backward 链路时再跑：

```bash
python dexmani_policy/smoke_test.py <config_name>
```

完整训练与长时间评测默认不作为代码改动后的自动验证步骤。

## Evaluate

常用仿真评测入口：

```bash
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>
```

也可以单独执行：

```bash
bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/eval_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name>
```

已有实验的模型结构、输入模态、action/window/normalization 与预处理从保存的 resolved config 恢复；权重与 fitted normalizer 从 checkpoint 恢复。

## Other Workflows

- VQ / codebook：`scripts/training/`
- 远端训练与数据同步：`scripts/remote/`，见 [SSH 服务器训练部署](docs/SSH服务器训练部署.md)
- 仿真评测背景：见 [仿真评测机制](docs/仿真评测机制.md)
- Real inference/deployment：`dexmani_policy/deployment/`；真机动作必须在明确授权下执行

## Repository Map

```text
dexmani_policy/
  configs/       Hydra configs
  agents/        Agent / observation encoder / backbone / action decoder
  datasets/      dataset / replay buffer / preprocessing
  training/      training, checkpoint, resume, EMA
  evaluation/    shared offline evaluation logic
  env_runner/    simulation runners
  deployment/    saved-policy inspection / Real runtime
  utils/         small cross-domain helpers

scripts/
  training/
  eval/
  remote/
```

## AI Coding

`AGENTS.md` 是仓库级 AI coding 约定，`CLAUDE.md` 作为 Claude Code 入口加载它。

仓库还提供两个短 workflow skill：

- `research-iterate`：把一个研究想法以最小 diff 落地并闭环验证。
- `preflight-experiment`：昂贵训练或评测前检查 config / data / model / checkpoint 是否对齐。

Codex 版本位于 `.codex/skills/`，Claude Code 版本位于 `.claude/skills/`。

