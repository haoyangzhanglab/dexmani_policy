# DexMani_Policy

面向论文实验与方法验证的个人灵巧手策略研究仓库，覆盖数据、训练、checkpoint selection、仿真评测与策略推理。新实验从 `dexmani_policy/configs/` 进入；历史实验使用保存配置、checkpoint 和对应源码。方法名表示本地研究适配，行为以配置和调用链为准。

| 需要了解 | 文档 |
| --- | --- |
| 数据流、策略接口、恢复与部署 | [项目架构](docs/项目架构.md) |
| 选择检查、运行 smoke 或短训练 | [验证指南](docs/验证指南.md) |
| 准备 DQ-RISE、LAMP、Temporal 手部表示 | [手部动作表示](docs/手部动作表示.md) |
| 选点、固定测试 seeds、解释评测结果 | [仿真评测机制](docs/仿真评测机制.md) |
| 远端启动、同步、监控与停止 | [SSH 服务器训练部署](docs/SSH服务器训练部署.md) |
| 修改代码与 AI 协作 | [AGENTS.md](AGENTS.md)、[仓库导航](#仓库导航) |

以下命令从仓库根目录执行，`<...>` 替换为实际值。Python 命令使用 Conda `policy` 环境；本地训练和评测脚本自动通过该环境启动，远端环境另行配置。

## 安装与配置

要求 Python 3.10+。准备 `policy` 环境并安装与设备匹配的 Torch / torchvision，再安装依赖；版本见 [requirements.txt](requirements.txt) 和 [pyproject.toml](pyproject.toml)。

```bash
conda activate policy
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

部分点云算子另需匹配 Torch/CUDA 的 PyTorch3D，仿真评测需 `dexmani_sim`。Uni3D 权重可用 `bash scripts/utils/download_pretrained.sh` 下载至 `data/pretrained/uni3d/`；其他权重以配置和 encoder 加载逻辑为准。

配置位于 `dexmani_policy/configs/*.yaml`，DDP overlay 位于 `dexmani_policy/configs/ddp/`。`agent._target_` 决定策略，`policy_name` 决定实验目录；对应关系见 [策略接口](docs/项目架构.md#策略接口)。

## 数据准备

数据通常位于 `robot_data/<task>.zarr`，实际路径由 resolved Dataset 配置决定。解析路径并检查目录存在：

```bash
python -m dexmani_policy.common.config --config-name dp3 'task_name=<task>' --check
```

该命令只检查路径，不读取数据内容；字段、窗口、split manifest 和预处理见 [数据接口](docs/项目架构.md#数据接口)。重新生成数据需使用新路径。通用 affine normalizer 仅按最终选中的 train episodes 流式拟合，排除 validation 和子集 exclusions；统计总体与手部 codec 的区别见 [归一化](docs/项目架构.md#归一化)。

### 手部先验准备

使用 `dqrise`、`lamp` 或 `temporal_codec_flow` 开始新训练前，先按 [手部动作表示](docs/手部动作表示.md) 准备各自的 Codebook、prior 或 codec，并配置产物路径。产物必须匹配训练数据、动作坐标和划分；DQ 还需匹配手部 action affine，LAMP raw prior 不依赖观测 normalizer。

## 验证

按当前任务选择检查，下面的 smoke 使用配置中的数据路径和任务设置：

```bash
python dexmani_policy/smoke_test.py --config-only dp3
python dexmani_policy/smoke_test.py --data-only dp3 --max-batches 2
python dexmani_policy/smoke_test.py dp3 --max-updates 4
```

| 模式 | 覆盖范围与前提 |
| --- | --- |
| config-only | 检查配置与 target 导入，不加载数据或执行 GPU 算子 |
| data-only | 在 CPU 上检查实际数据批次，不构建模型；需配置对应的数据 |
| 完整 smoke | 检查训练更新、预测及 raw/EMA 保存恢复；需 CUDA、数据和所需权重 |

Smoke 接受配置名或独立 YAML 路径，不接受 Hydra 字段覆盖；data-only 的 batch 上限不限制数据扫描与统计。Python 语法检查、各模式的覆盖边界和可传 overrides 的短训练入口见 [验证指南](docs/验证指南.md)。

## 训练与续训

```bash
bash scripts/training/train.sh dp3 'task_name=<task>' 'training.seed=42'
bash scripts/training/train_ddp.sh ddp/sat 'task_name=<task>' 'training.seed=42'
bash scripts/training/train.sh dp3 '+resume_from=experiments/dp3/<task>/<run>'
```

训练入口转发 Hydra overrides。单卡要求 `training.num_gpus=1` 或不设置；DDP 需核对可见 GPU、`training.num_gpus` 和每卡 batch。训练预算按 optimizer update 计数，`best` 由离线选点产生。

新 run 默认写入 `experiments/<policy_name>/<task_name>/<时间戳>_seed<seed>/`，已有训练产物的目录拒绝重用。标量保存在 `metrics.jsonl`；W&B 默认 `offline`，可用 `workspace.wandb_cfg=null` 关闭，详见 [训练日志](docs/项目架构.md#训练日志与可选远程服务)。

`resume_from` 接受实验目录或 checkpoint，恢复到新 run；相对路径按项目根目录解析。配方来自 checkpoint 的 `effective_config`，模型与训练状态从同一 checkpoint 恢复；任务、数据语义、模型、world size 和总训练预算必须一致。允许覆盖项见 [完整恢复边界](docs/项目架构.md#完整恢复边界)。

## 仿真评测

```bash
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <run> --seed 1024 --episodes 70
bash scripts/eval/record_demo.sh <policy_name> <task_name> <run>
```

流水线先固定 test/selection 计划，再选点并复用记录进行不录视频的 held-out eval。Seed 池需同时满足 test 与 selection 的容量，预算以保存配置和显式覆盖为准；分步入口与参数规则见 [仿真评测机制](docs/仿真评测机制.md)。Demo 录像需要 X11 或 XWayland。

比较实验需同时记录数据划分、有效 batch、训练预算、选点和最终评测协议。

## 远端与其他工作流

```bash
bash scripts/remote/train_remote.sh --dry-run <config> <task>
bash scripts/remote/train_remote.sh --gpus 0 <config> <task> 'training.seed=42'
bash scripts/remote/tail_log.sh <policy> <task>
bash scripts/remote/sync_down.sh <policy>/<task>/<run>
```

远端运行使用独立源码副本，共享数据、权重与实验；dry-run 仍访问 SSH。详见 [SSH 服务器训练部署](docs/SSH服务器训练部署.md)。`scripts/utils/clean_experiments.sh` 只读盘点实验，`scripts/utils/wandb_sync.sh` 上传 offline runs。

策略加载与真机调用接口见 [部署边界](docs/项目架构.md#部署边界)，真机执行由 `dexmani_real` 管理。

## 仓库导航

| 路径 | 职责 |
| --- | --- |
| `dexmani_policy/configs/` | 实验配置与 DDP overlay |
| `dexmani_policy/agents/` | Policy、观测编码、动作解码与手部表示 |
| `dexmani_policy/datasets/` | 读取、窗口、划分、增强与多任务组合 |
| `dexmani_policy/training/` | 训练构建、循环、预训练与恢复 |
| `dexmani_policy/evaluation/` | 评测请求、checkpoint 选点与评测流水线 |
| `dexmani_policy/env_runner/` | 环境交互、rollout、seed 与指标工具、录像 |
| `dexmani_policy/deployment/` | 已保存策略的检查与推理接口 |
| `dexmani_policy/common/` | 共享配置、checkpoint、归一化、RGB 预处理与运行工具 |
| `dexmani_policy/smoke_test.py` | 分层 smoke，覆盖范围见 [验证指南](docs/验证指南.md) |
| `scripts/` | 运行入口 |
| `robot_data/`、`data/`、`experiments/` | 数据、资源与实验产物 |
| [AGENTS.md](AGENTS.md)、[CLAUDE.md](CLAUDE.md) | 研究协作规则及 Claude Code 加载入口 |
