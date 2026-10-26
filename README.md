# DexMani Policy

灵巧手操作策略学习研究仓库，支持多种策略架构的训练、Checkpoint 恢复、仿真评测与推理部署。

项目采用统一的数据、策略和实验接口，支持单卡与分布式训练，并提供标准化的 Checkpoint 选点与评测流程。

## 环境准备

Python 3.10+，使用 Conda `policy` 环境，预先安装匹配设备的 PyTorch / torchvision。

```bash
conda activate policy
python -m pip install -r requirements.txt
python -m pip install -e .
```

Sparse3D（SAT、DQ-RISE、LAMP、LatentFlow）的 patch FPS 和 R3D 需要与已装 Torch/CUDA 匹配的 PyTorch3D 原生扩展；通用依赖不自动安装它。本次 CPU 环境验证为 Torch 2.4.1+cu124 / torchvision 0.19.1 / PyTorch3D 0.7.8。请在目标环境构建匹配的 PyTorch3D，先执行 `python -c 'import pytorch3d.ops'`，再运行对应方法检查。DP3 输入点数无需下采样时不需要该后端；触发 FPS 的其他路径同样需要它。

## 快速开始

以下命令均从仓库根目录执行。训练数据通常位于 `robot_data/<task>.zarr`，实际路径由配置决定。

### 1. 检查配置与数据

实验配置位于 `dexmani_policy/configs/`，其中 `ddp/` 存放多卡配置。

```bash
# 配置及导入检查（无需数据或 GPU）
python dexmani_policy/smoke_test.py --config-only dp3

# 检查指定任务的数据路径
python -m dexmani_policy.common.config \
  --config-name dp3 'task_name=<task>' --check

# 检查默认配置的数据内容
python dexmani_policy/smoke_test.py --data-only dp3
```

### 2. 训练

```bash
# 单卡训练
bash scripts/training/train.sh dp3 'task_name=<task>'

# 多卡 DDP 训练
bash scripts/training/train_ddp.sh ddp/sat 'task_name=<task>'

# 从已有实验恢复训练
bash scripts/training/train.sh dp3 \
  '+resume_from=experiments/dp3/<task>/<run>'
```

入口支持 Hydra 配置覆盖。续训使用保存的实验配置，并创建新的运行目录；首次 update 前归档恢复点之前的 milestone，缺失或冲突时失败。完整训练 `.pt` 仅接受可信项目产物。

### 3. 仿真评测

仿真评测依赖 `dexmani_sim`。

```bash
# Checkpoint 选点与冻结测试
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <run>

# 指定 Checkpoint 的普通研究评测
bash scripts/eval/evaluate.sh <policy_name> <task_name> <run> \
  --checkpoint best
```

正式测试使用冻结的选点与评测协议；普通研究评测不作为正式 held-out 测试证据。

## 项目结构

```text
dexmani_policy/
├── dexmani_policy/
│   ├── agents/        # 策略与模型
│   ├── datasets/      # 数据读取与处理
│   ├── training/      # 训练流程
│   ├── evaluation/    # 仿真评测与选点
│   ├── deployment/    # 推理部署
│   └── configs/       # 实验配置
├── scripts/           # 训练、评测及辅助脚本
├── docs/              # 技术文档
└── README.md
```

## 实验管理

实验产物统一保存在：

```text
experiments/<policy_name>/<task_name>/<run>/
```

每次实验保存配置、Checkpoint 和相关运行记录。评测结果与选点记录保存在对应实验目录中。

历史实验应使用对应的保存配置、Checkpoint 和源码复现。具体算法实现、参数与运行约束以代码、配置及技术文档为准。

## 技术文档

- [项目架构](docs/项目架构.md)：整体设计、数据接口、训练恢复与部署
- [手部动作表示](docs/手部动作表示.md)：手部动作建模及预训练
- [仿真评测机制](docs/仿真评测机制.md)：Checkpoint 选点、评测协议与结果记录
- [SSH 服务器训练部署](docs/SSH服务器训练部署.md)：远程训练、同步与管理
- [本轮修复与验证结果](docs/CODEX_REFACTOR_RESULTS.md)：R01–R20 的实际处理、CPU 验证和外部阻塞
