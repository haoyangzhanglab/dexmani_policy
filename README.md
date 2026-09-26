# DexMani_Policy

灵巧手操作 imitation learning / robot policy 研发仓库。训练由 Hydra 配置驱动，数据使用 Zarr，评测通过 `dexmani_sim` 完成。

Policy 的**当前实现事实以 Hydra config 和源码为准**。README 只维护稳定的使用入口，不维护 Policy 架构、超参数或策略数量快照。

## 快速开始

### 环境

使用 Python 3.10+ 和已配置好依赖的 Conda 环境 `policy`：

```bash
conda activate policy
pip install -e .
```

`pip install -e .` 安装项目声明的核心依赖；完整研究环境仍以受管 Conda 环境 `policy` 为准。仿真评测需要另外以 editable 模式安装 `dexmani_sim`：

```bash
pip install -e /path/to/dexmani_sim
```

训练数据默认由 config 中的 `zarr_path` 指向仓库根目录下的 `robot_data/<task>.zarr`。

以下命令默认从仓库根目录执行。直接运行 Python 前先激活 `policy`；非交互 shell 可使用 `conda run --no-capture-output -n policy <command>`。完整 smoke、训练和大多数评测需要 CUDA/GPU 及对应数据、权重；仿真评测还需要 `dexmani_sim` 和可用的显示环境。

### 发现可用配置

单卡 Policy config：

```bash
ls dexmani_policy/configs/*.yaml
```

DDP overlay：

```bash
ls dexmani_policy/configs/ddp/*.yaml
```

可用 Policy 以配置目录为准。

## 训练

本地训练脚本要求 `conda` 在 `PATH` 中可用，并已配置好 `policy` 环境。脚本自动切换到仓库根目录，通过 `conda run --no-capture-output -n policy` 启动 `python -u`，无需手动激活环境，日志实时输出。

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

这两个帮助命令无需调用 Conda。需要仅预览 Hydra 解析后的配置时，可在单卡或 DDP 训练命令末尾追加 `--cfg job --resolve`；这不会开始训练，也不替代下文的 config validation。

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

完整 smoke test 覆盖 dataset/normalizer、model/EMA、optimizer/scheduler、forward/backward、inference 和 checkpoint roundtrip。Policy 开发流程及按改动范围选择验证的要求见 [AGENTS.md](AGENTS.md)。

PointNet 与 ManiFlow 的定向回归可独立运行，无需外部训练数据、预训练权重或仿真环境：

```bash
python -m dexmani_policy.agents.obs_encoder.pointcloud.smoke_test
python -m dexmani_policy.agents.core.maniflow_smoke_test
python -m dexmani_policy.agents.core.inference_smoke_test
python -m dexmani_policy.training.eval_smoke_test
python -m dexmani_policy.deployment.inference_smoke_test
```

ManiFlow 回归使用合成数据，覆盖点排列不变性、训练时间网格与推理步数独立性、增强语义、EMA、strict resume/inference restore，以及 BF16/compile。CUDA 专项在 CUDA 不可用时跳过；跳过不代表 GPU 验证通过。

## 评测

```bash
# 一键 select → held-out eval → demo
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>

# 分步
bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/eval_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name>
```

流水线最后会录制 demo。追加 `--no-videos` 只关闭 held-out eval 阶段的视频，最后的 demo 仍会录制；只需评测时使用分步入口。

`<exp_name>` 是 `experiments/<policy>/<task>/` 下的实验目录名。历史模型的构造参数及 action/window/normalization 语义来自所选 checkpoint；实验 config 提供环境与评测 protocol。评测拒绝 `agent.*` override，EMA/raw、NFE、episodes、seed 与视频等评测控制仍可按入口参数覆盖。

最终评测每次写入独立的 `eval_dexsim/<run-id>/`，CLI 会打印准确路径；checkpoint、推理设置、seeds 和指标保存在该目录的 `result_details.json`。背景说明见 [仿真评测机制](docs/仿真评测机制.md)，当前参数与行为以入口帮助和源码为准。

推理步数统一通过 `--inference-steps N` 或 `eval.inference_steps` 设置；sweep 使用 `eval.inference_steps_list`。旧 CLI `--denoise-steps` 和旧 eval 字段仅在输入边界兼容，新旧同层配置冲突会报错，两个 CLI flag 不能同时使用。Python Agent/decoder/runner 接口只接受 `inference_steps`。ManiFlow 的 `denoise_timesteps` 与 RectifiedFlow 的 `num_flow_train_timesteps` 只定义训练网格，`num_inference_steps` 定义 decoder 默认 NFE。

## Deployment

新 deployment artifact 使用 `dexmani.deployment.v3`，不读取旧版本；训练 checkpoint 保持 `simple.v3` 和严格 resume 校验，不提供旧构造配置迁移。

Real deployment 入口位于 `dexmani_policy/deployment/`。Deployment artifact、observation contract 和 runtime restore 的具体语义以实现为准；README 不复制其内部 schema。

```bash
# 研究者日常路径：export -> run_policy（run_policy 位于 dexmani_real）
bash scripts/deployment/export.sh <experiment_dir>  # 默认 latest，使用 Conda policy 环境
bash scripts/deployment/export.sh <experiment_dir> --checkpoint 80pct
bash scripts/deployment/export.sh <experiment_dir> --output deployment-v3.pt

# 原 Python CLI 保持默认 best；best 必须有有效的 best_ckpt.json，不自动回退
python -m dexmani_policy.deployment.export <experiment_dir> --checkpoint best
```

脚本可从任意工作目录调用（仓库外请使用脚本的绝对路径）。相对实验目录相对于调用目录解析；相对 checkpoint 文件路径和 `--output` 相对于实验的 `checkpoints/` 解析。产物只能写入该目录，已有文件拒绝覆盖；再次导出可指定新的 `--output`。使用 `bash scripts/deployment/export.sh --help` 查看参数说明，成功时输出 exporter 的 JSON 回执。

- Export 使用 **selected checkpoint 保存的**模型、归一化与训练数据语义；`config.yaml` 只提供实验身份和推理设置。缺少训练数据语义快照的旧 checkpoint 无法 export，load/离线分析不受影响。
- Export 不再打开训练 Zarr。Checkpoint 必须保存有序 joint_names 和所需的 PointCloudConfig；缺失时明确拒绝，不猜测旧格式。
- Export 通过结构检查和 weights-only reload 后原子更新 `deployment_latest.pt`；运行时严格恢复模型、normalizer 并 warmup 后才就绪。它指向本次不可覆盖的 artifact（默认 `<checkpoint>-deployment.pt`）；真机 session 可以固定使用解析后的文件名。
- 运行时 `--inference-steps N` 是显式 override（NFE ablation），不需要重新 export。
- Real 使用当前相机、桌面和手安装标定；重新标定不使旧 policy 失效。点云算法配置和去桌面开关来自 artifact，RGB resize/crop/normalization 由 Policy 执行。
- 离线跨仓库回归：`python -m dexmani_policy.deployment.smoke_test`（需安装 dexmani_real）。

开发和修改 deployment 时的完整约束见 [AGENTS.md 的 Deployment Boundary](AGENTS.md#deployment-boundary)。

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
  deployment/
  remote/
```

## 文档与 AI 编码入口

- [AGENTS.md](AGENTS.md)：Codex 与 Claude 共用的完整工程规范，包含事实来源、开发流程、验证要求和 deployment 边界。
- [CLAUDE.md](CLAUDE.md)：Claude 的加载入口，通过 `@AGENTS.md` 导入同一份规范。
- [项目 Skills](AGENTS.md#project-skills)：新增 Policy、review 和训练数值问题的共享流程；工具未自动发现时可按链接读取。

交替使用 Codex 与 Claude 时，共享工程规则只需修改 `AGENTS.md`；操作命令维护在 README，`CLAUDE.md` 不维护规则副本。模型、权限和子代理等工具专属设置留在各自配置文件。

背景文档：[项目架构](docs/项目架构.md)、[仿真评测机制](docs/仿真评测机制.md)、[SSH 服务器训练部署](docs/SSH服务器训练部署.md)。这些文档可能滞后，当前实现与实验语义以 config、源码和 checkpoint 为准。
