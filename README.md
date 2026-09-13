# DexMani_Policy

灵巧手操作 imitation learning / robot policy 研发仓库。训练由 Hydra 配置驱动，数据使用 Zarr，评测通过 `dexmani_sim` 完成。

Policy 的**当前实现事实以 Hydra config 和源码为准**。README 只维护稳定的使用入口，不维护 Policy 架构、超参数或策略数量快照。

## 快速开始

### 环境

```bash
conda activate policy
pip install -e .
```

`pip install -e .` 安装项目声明的核心依赖；完整研究环境仍以受管 Conda 环境 `policy` 为准。仿真评测需要另外以 editable 模式安装 `dexmani_sim`：

```bash
pip install -e /path/to/dexmani_sim
```

训练数据默认由 config 中的 `zarr_path` 指向仓库根目录下的 `robot_data/<task>.zarr`。

### 发现可用配置

单卡 Policy config：

```bash
ls dexmani_policy/configs/*.yaml
```

DDP overlay：

```bash
ls dexmani_policy/configs/ddp/*.yaml
```

不要从 README/CLAUDE 的静态列表推断当前 Policy 集合；配置目录是当前入口。

## 训练

```bash
# 单卡
bash scripts/training/train.sh <config_name> 'task_name=<task>'

# DDP（存在对应 overlay 时）
bash scripts/training/train_ddp.sh ddp/<config_name> 'task_name=<task>'

# Hydra override
bash scripts/training/train.sh <config_name> \
  'task_name=<task>' \
  'training.seed=42'
```

`<config_name>` 直接对应 Hydra config。Policy 的 Agent 实现由 config 中的 `agent._target_` 决定；文件名和类名不要求与 config 名同名。

### 续训

训练不会因为“重复执行同一命令”而自动续训。需要显式传入：

```bash
bash scripts/training/train.sh <config_name> \
  'task_name=<task>' \
  '+resume_from=/absolute/path/to/experiment_or_checkpoint'
```

Resume 使用严格 contract 校验训练状态。Checkpoint 权重能够被其他流程读取，不代表不同 world size、batch/loader 或训练配置之间可以直接 strict resume。

## 配置与验证

修改或新增 Policy 后，优先执行轻量 config check：

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

它会完成 Hydra compose/resolve、公共 config 校验、`_target_` module 存在性检查以及 `agent._target_` 实际 import；不会构建数据集、仿真环境或运行 GPU forward。

需要验证完整训练构建链时：

```bash
python dexmani_policy/smoke_test.py <config_name>
```

完整 smoke test 覆盖 dataset/normalizer、model/EMA、optimizer/scheduler、forward/backward、inference 和 checkpoint roundtrip。共享模块改动应再选择实际依赖该模块的其他 config 做回归。

## Policy 开发入口

处理某个 Policy 时，从 config 动态追踪，而不是依赖文档中的架构快照：

```text
resolved Hydra config
→ agent._target_
→ Agent construction
→ obs_encoder
→ backbone / action_decoder
→ compute_loss
→ predict_action
→ control_action
```

重点分别确认：

- Observation Representation
- Policy Architecture
- Action Representation
- Training Objective
- Inference Algorithm

具体 layer 数、hidden dim、encoder、optimizer、NFE、tensor shape 等都应从当前 config 与实现读取。

## 评测

```bash
# 一键 select → held-out eval → demo
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>

# 分步
bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/eval_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name>
```

`<exp_name>` 是 `experiments/<policy>/<task>/` 下的实验目录名。评测具体参数以实验保存的 config、CLI override 和对应评测代码为准。

## Deployment

Real deployment 入口位于 `dexmani_policy/deployment/`。Deployment artifact、observation contract 和 runtime restore 的具体语义以实现和 `docs/项目架构.md` 为准；README 不复制其内部 schema。

## 仓库结构

```text
dexmani_policy/
  configs/           # Hydra Policy configs + DDP overlays
  agents/            # Agent / encoder / backbone / decoder
  datasets/          # Dataset, replay buffer, sampler
  training/          # Build, trainer, EMA, resume, workspace
  env_runner/        # Simulation runners
  common/            # Shared config/checkpoint/normalizer utilities
  deployment/        # Real deployment artifact/runtime
  train.py           # Single-GPU entry
  train_ddp.py       # DDP entry
  smoke_test.py      # Config + integration validation

scripts/
  training/
  eval/
  remote/
```

## 文档与 AI 编码入口

- `AGENTS.md`：项目级 coding contract，Codex 的主要入口。
- `CLAUDE.md`：精简、独立可用的 AI 工作速查。
- `.agents/skills/`：新增 Policy、PR review、训练数值问题的过程性 workflow。
- `docs/项目架构.md`：架构背景。
- `docs/仿真评测机制.md`：仿真评测链路。
- `docs/SSH服务器训练部署.md`：服务器训练与同步。

全局文档不维护 Policy-specific 超参数、当前 Policy 数量或实验结论；这些信息属于 config、代码、实验快照和局部测试。
