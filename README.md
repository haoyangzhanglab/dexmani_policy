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

Sparse3D（SAT、DQ-RISE、LAMP、LatentFlow）的 patch FPS 和 R3D 需要与已装 Torch/CUDA 匹配的 PyTorch3D 原生扩展；通用依赖不自动安装它。构造这些点云模型前可用 `python -c 'import pytorch3d.ops'` 检查后端。纯配置检查及 VQ/LAMP 手部预训练数据准备不需要 PyTorch3D；DP3 输入点数无需下采样时也不需要。实际 FPS/kNN 使用点缺少后端会明确报错，不自动替换算法。

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

入口支持 Hydra 配置覆盖。续训使用 Checkpoint 保存的训练配方，并创建新的运行目录；首次 update 前归档恢复点及之前实际存在的有效 milestone。缺历史 milestone 不阻止恢复，已有候选损坏、冲突或配方不一致仍报错；不复制未来候选。附加 `source_evidence` 不参与恢复配方比较，数据划分、normalizer、模型和优化参数仍须一致。完整训练 `.pt` 仅接受可信项目产物。

R3D 默认 B=128 在 24 GiB GPU 上可能 OOM。单卡可使用现有梯度累积配置：

```bash
bash scripts/training/train.sh r3d 'task_name=<task>' \
  dataloader.batch_size=4 training.loop.gradient_accumulation_steps=32
```

该配方的 nominal batch 为 4×32=128；实际 microbatch、累积及 epoch 尾组会记录在训练输出中。默认 YAML 不变，尾组和随机采样过程不保证与 B=128、accumulation=1 逐位一致。模型结构、动作布局和 loss 不变；恢复这个实验时继续使用保存的 4/32 配方。

### 3. 仿真评测

仿真评测依赖 `dexmani_sim`。

```bash
# Checkpoint 选点与冻结测试
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <run>

# 指定 Checkpoint 的普通研究评测
bash scripts/eval/evaluate.sh <policy_name> <task_name> <run> \
  --checkpoint best
```

正式选点在仿真前要求训练预算对应的完整 milestone 集合（通常为 20/40/60/80/100%）；缺失时列出 step，不能以部分候选代替。普通研究评测仍可指定单个 Checkpoint；`best` 在加载前检查实际权重 SHA-256，但不重跑排名。正式测试使用冻结的选点与评测协议，普通研究评测不作为正式 held-out 测试证据。

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
- [本轮修复与验证结果](docs/CODEX_REFACTOR_RESULTS.md)：定向修复、验证结果与未验证范围
