# DexMani_Policy

个人灵巧手策略研究仓库，覆盖数据读取、训练、checkpoint selection、仿真评测与策略推理。新实验从 `dexmani_policy/configs/` 进入；历史实验使用保存的配置、checkpoint 和源码版本。方法名表示本地研究适配，行为以配置和调用链为准。

| 文档 | 内容 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 数据流、策略接口、归一化、恢复与部署 |
| [手部动作表示](docs/手部动作表示.md) | DQ-RISE、LAMP、Temporal 的准备命令与机制 |
| [验证指南](docs/验证指南.md) | 配置检查、smoke 与短训练 |
| [仿真评测机制](docs/仿真评测机制.md) | 选点、固定测试 seeds、参数与结果 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远端启动、同步、监控与停止 |

以下命令从仓库根目录执行，`<...>` 替换为实际值；本地 Shell 入口使用 Conda `policy` 环境。

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

该检查不加载 Zarr。字段、窗口、split manifest 和预处理见 [数据接口](docs/项目架构.md#数据接口)。通用 affine normalizer 使用完整 replay buffer，包含验证 episode 和 `max_train_episodes` 外的行；手部 codec 专用统计另有定义。重新生成数据需使用新路径；数据身份按 episode 结构与实际划分核对，不计算内容哈希。

## 训练与续训

```bash
bash scripts/training/train.sh dp3 'task_name=<task>' 'training.seed=42'
bash scripts/training/train_ddp.sh ddp/sat 'task_name=<task>' 'training.seed=42'
bash scripts/training/train.sh dp3 '+resume_from=experiments/dp3/<task>/<run>'
```

训练入口转发 Hydra overrides。单卡要求 `training.num_gpus=1` 或不设置；DDP 需核对可见 GPU、`training.num_gpus` 和每卡 batch。训练按 optimizer update 推进，best 由离线选点产生。

新 run 默认写入 `experiments/<policy_name>/<task_name>/<时间戳>_seed<seed>/`，已有训练产物的目录拒绝重用。`metrics.jsonl` 保存标量日志；每次启动（含续训）尝试创建新的 W&B run，默认 `offline`，`workspace.wandb_cfg=null` 可关闭 W&B。服务异常时保留本地记录，详见[训练日志与可选远程服务](docs/项目架构.md#训练日志与可选远程服务)。

`resume_from` 接受实验目录或 checkpoint，支持 `~`，相对路径按项目根目录解析，恢复写入新 run。配方和状态来自 checkpoint 的 `effective_config`；任务、数据语义、模型、world size 和训练预算必须保持一致。允许覆盖项见 [完整恢复边界](docs/项目架构.md#完整恢复边界)。

### 手部先验准备

`sat` 与 `dqrise` 固定使用 19D Joint Action，并各自实例化相同的 Sparse3D Global 观测编码器。新 SAT 不兼容旧 SAT checkpoint。

`dqrise` 默认 `codebook_path=null`，需先训练 VQ 并导出匹配当前 Joint Action 数据、normalizer 和划分的码本，再显式指定路径；`lamp` 和 `temporal_codec_flow` 需先准备 prior/codec。各阶段复用数据与划分设置，命令见 [手部动作表示](docs/手部动作表示.md)。LAMP/Temporal 仅支持 sync；完整策略 checkpoint 内嵌表示，推理无需原预训练产物。

## 验证

```bash
python dexmani_policy/smoke_test.py --config-only dp3
python dexmani_policy/smoke_test.py --data-only dp3 --max-batches 2
python dexmani_policy/smoke_test.py dp3 --max-updates 4
```

config-only 检查配置；data-only 在 CPU 上检查生产数据路径，不构建模型；完整 smoke 检查训练更新、预测及 raw/EMA 保存恢复，需要 CUDA、数据和所需权重。各模式的覆盖边界、批量检查和保留短跑产物的命令见 [验证指南](docs/验证指南.md)。

## 仿真评测

```bash
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <run> --seed 1024 --episodes 70
bash scripts/eval/record_demo.sh <policy_name> <task_name> <run>
```

流水线固定 test/selection 计划后选点，再复用本次记录进行不录视频的 held-out eval。当前配置默认每任务 test 70 个、selection 预留 30 个，需至少 100 个不同有效 seeds；实际预算以实验保存配置和显式覆盖为准。分步入口、受限 dotlist、记录字段及参数优先级见 [仿真评测机制](docs/仿真评测机制.md)。实际 demo 录像需要 X11 或 XWayland。

## 远端与其他工作流

```bash
bash scripts/remote/train_remote.sh --dry-run <config> <task>
bash scripts/remote/train_remote.sh --gpus 0 <config> <task> 'training.seed=42'
bash scripts/remote/tail_log.sh <policy> <task>
bash scripts/remote/sync_down.sh <policy>/<task>/<run>
```

远端运行使用独立源码副本，共享数据、权重与实验；dry-run 仍访问 SSH。详见 [SSH 服务器训练部署](docs/SSH服务器训练部署.md)。`scripts/utils/clean_experiments.sh` 只读盘点实验，`scripts/utils/wandb_sync.sh` 上传 offline runs。

策略加载与真机调用接口见 [部署边界](docs/项目架构.md#部署边界)，真机执行由 `dexmani_real` 管理。比较实验需同时记录数据、有效 batch、训练预算、选点和最终评测协议。

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
| `scripts/` | 运行入口 |
| `robot_data/`、`data/`、`experiments/` | 数据、资源与实验产物 |
| [AGENTS.md](AGENTS.md)、[CLAUDE.md](CLAUDE.md) | 研究协作规则及 Claude Code 加载入口 |
