# DexMani Policy

灵巧手操作策略学习研究仓库，支持多种策略架构的训练、Checkpoint 恢复、仿真评测与推理部署。

项目采用统一的数据、策略和实验接口，支持单卡训练，并提供标准化的 Checkpoint 选点与评测流程。

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

实验配置位于 `dexmani_policy/configs/`；八个 Policy 仅共享一层相同的 workspace/hydra 默认设置。

```bash
# 配置接口及导入检查（无需数据或 GPU）
python dexmani_policy/smoke_test.py --config-only dp3

# 检查指定任务的数据路径
python -m dexmani_policy.training.common.config \
  --config-name dp3 'task_name=<task>' --check

# 检查默认配置的数据内容
python dexmani_policy/smoke_test.py --data-only dp3
```

`--config-only` 检查配置解析、组件之间的必要一致性和 target 导入；模型自身参数及 normalization 兼容性在实际构造或绑定时检查，数据配方在 Dataset 构建时检查。它不保证完整训练配置可运行。

### 2. 训练

```bash
# 单卡训练
bash scripts/training/train.sh dp3 'task_name=<task>'

# 从已有实验恢复训练
bash scripts/training/train.sh dp3 \
  '+resume_from=experiments/dp3/<task>/<run>'
```

入口支持 Hydra 配置覆盖。续训使用 Checkpoint 保存的训练配方，并创建新的运行目录；首次 update 前归档恢复点及之前实际存在的有效 milestone。缺历史 milestone 不阻止恢复，已有候选损坏、冲突或配方不一致仍报错；不复制未来候选。实际划分内嵌于 `dataset.split_manifest`；恢复复用保存的 split 和 normalizer，禁止重新拟合。完整训练 `.pt` 仅接受可信项目产物。

数据装配对所选 train/val episodes 的消费字段和列做一次分块有限值检查，坏值报出路径、字段、episode、row 和通道后停止。普通归一化使用训练 episodes 的全部真实帧，每帧一次；LatentFlow 的相对动作按窗口及 query reference 统计。详见 [数据接口](docs/项目架构.md#数据接口)。

恢复采用 epoch 级语义：中途 checkpoint 从当前 epoch 开头重读，权重及 global_step 不回退；完成 epoch 后的 checkpoint 从下一 epoch 开始。不保存 batch 游标，不保证逐位重放；训练预算、scheduler、EMA 和 milestone 仍按实际 optimizer update 计数。

训练使用 `training.device` 指定的单个设备，默认 GPU；小规模验证可设为 `cpu`，同时关闭 `training.use_compile` 和 `training.use_bfloat16`。Checkpoint 保存该设备的单进程 RNG，续训保持 CPU/CUDA 设备类型一致。

R3D 默认 B=128 在 24 GiB GPU 上可能 OOM。单卡可使用现有梯度累积配置：

```bash
bash scripts/training/train.sh r3d 'task_name=<task>' \
  dataloader.batch_size=4 training.loop.gradient_accumulation_steps=32
```

该配方的 nominal batch 为 4×32=128；实际 microbatch、累积及 epoch 尾组会记录在训练输出中。默认 YAML 不变，尾组和随机采样过程不保证与 B=128、accumulation=1 逐位一致。模型结构、动作布局和 loss 不变；恢复这个实验时继续使用保存的 4/32 配方。

DQ/LAMP 使用相同的显式 split 默认参数，也可复用已有 manifest。DQ 的 VQ 预训练自动保存 `codebook.npz`，随后直接训练 Policy；完整命令见 [手部动作表示](docs/手部动作表示.md)。

### 3. 仿真评测

仿真评测依赖 `dexmani_sim`。

```bash
# Checkpoint 选点与冻结测试
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <run>

# 指定 Checkpoint 的普通研究评测
bash scripts/eval/evaluate.sh <policy_name> <task_name> <run> \
  --checkpoint best
```

正式选点在仿真前要求训练预算对应的完整 milestone 集合（通常为 20/40/60/80/100%）；缺失时列出 step，不能以部分候选代替。普通研究评测仍可指定单个 Checkpoint；`best` 直接读取选中路径和推理配置，加载后核对 checkpoint step，不重跑排名。正式测试使用冻结的选点与评测协议，普通研究评测不作为正式 held-out 测试证据。

公开推理入口直接解析保存的实验：

```python
from dexmani_policy.deployment import inspect_policy, load_policy

info = inspect_policy(experiment, checkpoint="best")  # 无 Torch 的需求预览
policy = load_policy(info, device="cuda:0", seed=0)
```

inspect 支持 best/latest/百分比 tag/显式路径；weights 和 inference_steps 可显式覆盖。load 使用已解析结果，完整模型无需原 prior、codec、codebook 或训练数据。Real 执行长度仍由 execution 配置或模型 n_action_steps 决定。

## 项目结构

```text
dexmani_policy/
├── dexmani_policy/
│   ├── agents/        # 策略与模型
│   ├── datasets/
│   │   ├── common/       # 通用 Dataset、Zarr 读取、窗口、划分与增强
│   │   └── hand_prior/   # 手部动作窗口、历史与专用元数据
│   ├── training/
│   │   ├── trainer.py    # 主策略训练循环
│   │   ├── common/       # 配置、数据构建、优化、日志与续训
│   │   └── pretrain/
│   │       └── hand_prior/  # 手部先验预训练、码本计算与导出
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

实验复现使用保存的配置、Checkpoint 和对应源码。`workspace` 和 `workspace.wandb_cfg` 使用普通参数映射（不含 `_target_`），设置 `workspace.wandb_cfg=null` 仅保留本地 JSONL 日志。具体算法实现、参数与运行约束以代码、配置及技术文档为准。

## 技术文档

- [项目架构](docs/项目架构.md)：整体设计、数据接口、训练恢复与部署
- [手部动作表示](docs/手部动作表示.md)：手部动作建模及预训练
- [仿真评测机制](docs/仿真评测机制.md)：Checkpoint 选点、评测协议与结果记录
- [SSH 服务器训练部署](docs/SSH服务器训练部署.md)：远程训练、同步与管理
