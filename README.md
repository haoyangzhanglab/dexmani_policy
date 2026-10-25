# DexMani_Policy

灵巧手策略学习研究仓库，覆盖数据读取、训练、checkpoint 选点、仿真评测与策略推理。实验配置位于 `dexmani_policy/configs/`，DDP 配置位于其 `ddp/` 子目录；历史实验使用保存配置、checkpoint 和对应源码。

以下命令从仓库根目录执行，`<...>` 替换为实际值。Python 使用 Conda `policy` 环境，本地训练和评测脚本自动调用该环境。

## 安装与配置

要求 Python 3.10+。先准备 `policy` 环境和匹配设备的 Torch / torchvision，再安装依赖：

```bash
conda activate policy
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

依赖版本见 [requirements.txt](requirements.txt) 和 [pyproject.toml](pyproject.toml)。部分点云算子另需匹配 Torch/CUDA 的 PyTorch3D，仿真评测需 `dexmani_sim`。启用 Uni3D 预训练时，先运行 `bash scripts/utils/download_pretrained.sh` 准备本地权重。

## 数据准备

数据通常位于 `robot_data/<task>.zarr`，实际路径由 Dataset 配置决定。检查路径：

```bash
python -m dexmani_policy.common.config --config-name dp3 'task_name=<task>' --check
```

该命令只检查目录，不读取数据。数据字段、划分和归一化见 [数据接口](docs/项目架构.md#数据接口)。使用 `dqrise`、`lamp` 或 `latent_flow` 前，按 [手部动作表示](docs/手部动作表示.md) 准备 Codebook、prior 或 codec。

## 验证

```bash
# 配置与导入检查
python dexmani_policy/smoke_test.py --config-only dp3
# CPU 数据检查，需实际数据
python dexmani_policy/smoke_test.py --data-only dp3 --max-batches 2
# 训练更新、预测及 raw/EMA 推理恢复，需 CUDA、数据和所需权重
python dexmani_policy/smoke_test.py dp3 --max-updates 4
```

Smoke 接受配置名或独立 YAML 路径，不接受 Hydra 字段覆盖；data-only 的 batch 上限不限制数据扫描与统计。

## 训练与续训

```bash
bash scripts/training/train.sh dp3 'task_name=<task>' 'training.seed=42'
bash scripts/training/train_ddp.sh ddp/sat 'task_name=<task>' 'training.seed=42'
bash scripts/training/train.sh dp3 '+resume_from=experiments/dp3/<task>/<run>'
```

入口支持 Hydra overrides；DDP 需核对 GPU 数与每卡 batch。产物位于 `experiments/<policy_name>/<task_name>/<run>/`，`ddp/sat` 的方法名为 `sat`。`resume_from` 接受实验目录或 checkpoint，恢复到新 run；约束见 [完整恢复边界](docs/项目架构.md#完整恢复边界)。

## 仿真评测

```bash
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <run> eval.seed=1024 eval.episodes=70
bash scripts/eval/evaluate.sh <policy_name> <task_name> <run> --checkpoint best eval.video.enabled=true
```

流水线执行选点和冻结的 held-out test；普通 `--checkpoint best` 评测用于研究，不作为正式测试证据。评测协议、seeds 和录像设置见 [仿真评测机制](docs/仿真评测机制.md)。真机执行由 `dexmani_real` 管理，接口见 [部署边界](docs/项目架构.md#部署边界)。

## 仓库导航

| 入口 | 内容 |
| --- | --- |
| `dexmani_policy/configs/` | 实验配置与 DDP overlay |
| `scripts/` | 训练、评测、远端与工具脚本 |
| [项目架构](docs/项目架构.md) | 数据流、策略接口、训练恢复与部署 |
| [手部动作表示](docs/手部动作表示.md) | DQ-RISE、LAMP、LatentFlow/HandVAE 的准备与实现 |
| [仿真评测机制](docs/仿真评测机制.md) | 选点、固定 seeds、评测记录与录像 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远端启动、同步、监控与停止 |
| [AGENTS.md](AGENTS.md)、[CLAUDE.md](CLAUDE.md) | AI 协作规则与加载入口 |
