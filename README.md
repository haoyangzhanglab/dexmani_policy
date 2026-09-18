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

本地训练脚本要求 `conda` 命令在 `PATH` 中可用，并已配置好 `policy` 环境。脚本通过 `conda run --no-capture-output -n policy` 启动 `python -u`，无需提前手动激活环境，日志实时输出。命令参数保持不变，脚本会自动切换到仓库根目录。

Conda 激活钩子由 `conda run` 执行，避免在训练脚本的 `set -u` 下读取未定义变量而退出（例如 GCC 激活脚本中的 `SYS_SYSROOT: 未绑定的变量`）。

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

查看启动器帮助：

```bash
bash scripts/training/train.sh --help
bash scripts/training/train_ddp.sh --help
```

这两个帮助命令无需调用 Conda，成功返回退出码 0；缺少 config 参数时显示用法并返回非零退出码。需要仅预览 Hydra 解析后的配置时，可在单卡或 DDP 训练命令末尾追加 `--cfg job --resolve`；这不会开始训练，也不替代下文的 config validation。

### VQ 手部预训练

```bash
bash scripts/training/train_vq_hand.sh <task_name>
bash scripts/training/train_vq_hand.sh --help
```

该脚本同样自动使用 `policy` 环境。首个位置参数为任务名，后续参数使用 Python CLI 的 `--option value` 格式；`-h` 和 `--help` 均转发给 Python，查看帮助需要该环境及模块依赖可用。任务和路径也可通过 `TASK_NAME`、`ZARR_PATH`、`OUTPUT_DIR` 环境变量指定；显式任务位置参数优先于 `TASK_NAME`。

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

```bash
# 研究者日常路径：export -> run_policy（run_policy 位于 dexmani_real）
python -m dexmani_policy.deployment.export <experiment_dir> --checkpoint best
```

Export 使用 **selected checkpoint 自己保存的** agent/dataset/normalization 语义，`config.yaml` 只提供 experiment identity 与 inference recipe，因此训练后修改 config 不会改变旧 checkpoint 的 deployment 行为。成功 export 意味着已通过 safe reload + strict restore + deterministic synthetic prediction；没有跳过验证的开关。

运行时 `--inference-steps N` 是显式 override（NFE ablation），不需要重新 export。`qualify.py` 是 developer/release regression 工具，不属于日常流程。

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
