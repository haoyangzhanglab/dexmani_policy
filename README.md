# DexMani Policy

灵巧手操作策略学习研究仓库，提供单卡训练、Checkpoint 恢复、仿真评测与推理部署接口。

## 环境准备

Python 3.10+，使用 Conda `policy` 环境，先安装匹配设备的 PyTorch / torchvision。

```bash
conda activate policy
python -m pip install -r requirements.txt
python -m pip install -e .
```

Sparse3D、R3D 及需要 FPS/kNN 的点云路径还需安装与 Torch/CUDA 匹配的 PyTorch3D 原生扩展。仿真评测依赖 `dexmani_sim`。

## 快速开始

以下命令从仓库根目录执行，将 `<task>`、`<run>` 替换为实际名称。配置位于 `dexmani_policy/configs/`，训练数据默认位于 `robot_data/<task>.zarr`。

```bash
# 检查配置接口与导入，无需数据或 GPU
python dexmani_policy/smoke_test.py --config-only dp3

# 检查任务的数据路径
python -m dexmani_policy.training.common.config \
  --config-name dp3 'task_name=<task>' --check

# 训练，支持 Hydra 配置覆盖
bash scripts/training/train.sh dp3 'task_name=<task>'

# 从已有实验恢复到新的运行目录
bash scripts/training/train.sh dp3 \
  '+resume_from=experiments/dp3/<task>/<run>'

# Checkpoint 选点与独立测试
bash scripts/eval/eval_pipeline.sh dp3 '<task>' '<run>'
```

配置检查不代表完整训练可运行。续训复用 Checkpoint 中的训练配方、数据划分和 normalizer，根据已完成的 optimizer updates 从下一个 batch 继续；不保证随机增强或编译计算逐位重放。

主策略默认最多使用 100 个训练 episodes，在 20/40/60/80/100% optimizer updates 保存 Checkpoint，以 25 回合选点、100 回合独立测试，选点与测试 seed 分离。正式选点要求训练预算对应的完整 Checkpoint 集合。

DQ/LAMP 的手部预训练见 [手部动作表示](docs/手部动作表示.md)；AFRO/CordViP 需先导出 `encoder.pt`，流程见 [观测预训练](docs/观测预训练.md)。模块预训练使用各自预算。

## 实验产物

配置、Checkpoint、日志与评测结果保存在：

```text
experiments/<policy_name>/<task_name>/<run>/
```

设置 `workspace.wandb_cfg=null` 可仅保留本地 JSONL 日志。复现实验需保留配置、Checkpoint 和对应源码。

## 文档

- [项目架构](docs/项目架构.md)：目录职责、数据接口、训练恢复与部署
- [仿真数据集格式](docs/仿真数据集格式.md)：Zarr 字段与数据约定
- [手部动作表示](docs/手部动作表示.md)：DQ/LAMP 等手部表示与预训练
- [观测预训练](docs/观测预训练.md)：AFRO/CordViP 数据准备、预训练与 encoder 加载
- [仿真评测机制](docs/仿真评测机制.md)：选点、独立测试与单 Checkpoint 评测
- [SSH 服务器训练部署](docs/SSH服务器训练部署.md)：远程训练、同步与管理
