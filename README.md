# DexMani Policy

灵巧手操作策略学习研究仓库，围绕示教数据、观测表征和动作生成，提供统一的训练与仿真评测流程。

## 整体流程

示教轨迹被组织为观测历史与动作窗口。策略编码图像、点云、机器人状态或触觉，通过 Diffusion、Flow 或 ODE 模型生成动作序列。执行端执行配置的动作步数，再根据新观测重新预测，形成闭环；ODE 要求完整执行训练时的 chunk 长度。

研究主要围绕三个部分展开：

- **观测表征**：从全局场景特征、局部空间特征到手部交互历史。
- **动作表示**：直接生成关节目标，或借助手姿码本、连续手部潜表示生成动作。
- **学习目标**：在动作学习之外，引入观测预训练、一致性学习或未来触觉监督。

本仓库负责策略学习与推理；`dexmani_sim` 提供仿真环境，`dexmani_real` 负责真实观测构建和硬件执行。模块分工见[项目架构](docs/项目架构.md)。

## 策略方向

| 方法 | 核心思路 |
| --- | --- |
| DP / DP3 | 分别以 RGB、点云结合关节状态，学习扩散策略 |
| ODE | 点云历史作为 condition，最新实测状态进入 FiLM 向量场，从上一条已下发指令积分绝对关节目标 |
| R3D | 使用预训练三维表征和空间条件生成动作 |
| ManiFlow | 结合 Flow Matching 与一致性学习 |
| SAT / DQ-RISE / DexLatent | 共享空间观测编码，研究关节轨迹、手姿码本与连续手部潜表示 |
| AFRO / CordViP | 先学习场景动力学或接触与协调表征，再迁移到策略 |
| AdapTac / InterAct-Dex | 融合触觉与场景，研究未来力引导和手部交互建模 |

各方法均接入本项目的数据与机器人设置，具体结构和监督方式有所不同。

## 安装

在仓库根目录操作。启动脚本使用名为 `policy` 的 Conda 环境；已有环境可直接激活。

```bash
conda create -n policy python=3.10
conda activate policy
# CUDA 12.4 构建；其他设备使用与本机匹配的 Torch 2.4.1 / torchvision 0.19.1。
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

点云 FPS / KNN 路径还需要针对当前 Torch/CUDA 编译的 PyTorch3D 0.7.8；它是独立的编译依赖，`pip install -e .` 不会自动安装。离线接触标签额外使用 Open3D，可安装 `python -m pip install -e '.[preprocess]'`。

仿真评测需要安装 `dexmani_sim` 并准备其任务资源与评测 seed 池；若它位于相邻目录，可执行 `python -m pip install -e ../dexmani_sim`，然后重新运行 `pip check`。硬件执行由 `dexmani_real` 提供。

## 数据与初始化权重

示教数据默认位于 `robot_data/<task_name>.zarr`，字段与时间对应关系见[数据集格式](docs/仿真数据集格式.md)。运行其他任务时修改 `task_name`；自定义位置用 `zarr_path=/path/to/data.zarr`。

DP3、ODE、ManiFlow、AdapTac 和 InterAct-Dex 可直接从示教训练。其他方法按需准备：

| 方法 | 新训练需要的初始化 |
| --- | --- |
| DP | `bash scripts/utils/download_pretrained.sh dp`：DINOv2 |
| SAT / DexLatent / R3D | 将脚本参数换成 `sat` / `dexlatent` / `r3d`；分别准备 DINOv3、DINOv3 + 手部先验、Uni3D |
| DQ-RISE | `download_pretrained.sh dqrise` 准备 DINOv3；另外提供 `initialization.codebook`，见[手部动作表示](docs/手部动作表示.md) |
| AFRO | 提供 `initialization.encoder`；从头训练时显式设置 `agent.freeze_visual=false` |
| CordViP | 使用带手物元数据的 `.cordvip.zarr`；预训练编码器可用 `initialization.encoder` 加载 |

AFRO/CordViP 的预训练与接触数据生成命令见[观测预训练](docs/观测预训练.md)。外部权重与可选主干见 [data/README](data/README.md)。完整 checkpoint 已包含部署所需权重和归一化参数，续训、推理不再读取初始化文件。

## 训练与续训

```bash
RUN="$PWD/experiments/train/dp3/pick_apple_messy/run_01"
bash scripts/training/train.sh dp3 task_name=pick_apple_messy hydra.run.dir="$RUN"
# 中断后在同一输出目录恢复；总预算继续使用 checkpoint 中的配方。
python -m dexmani_policy.training.common.resume --run "$RUN"
```

新实验使用新目录。省略 `hydra.run.dir` 时自动生成带时间标识的目录。`config.yaml` 记录有效配方，`metrics.jsonl` 记录指标，`checkpoints/` 保存完整训练状态和 `latest.pt`。默认 W&B 为离线模式，可用 `workspace.wandb_cfg=null` 关闭。

续训恢复 raw/EMA、优化器、学习率调度器、步数和随机数状态；本地日志回退到 checkpoint 步数后继续追加。`--max-updates 100` 只限制本次续训执行量，不修改总预算。多进程数据增强的 worker 状态不在 checkpoint 中，因此不承诺跨中断逐位复现。

常用配置的含义：

| 配置 | 含义 |
| --- | --- |
| `horizon` / `n_obs_steps` | 训练窗口长度 / 输入观测历史长度；可预测未来长度为 `horizon - n_obs_steps + 1` |
| `n_action_steps` | 每次预测后实际执行的前缀长度；ODE 要求完整执行其训练 chunk |
| `eval.inference_steps` | 推理迭代次数；ODE 中是每个控制周期的积分子步数 |
| `training.loop.total_train_steps` | 优化器更新总次数，不是 epoch 或 microbatch 数 |
| `training.loop.gradient_accumulation_steps` | 每次更新累计的 microbatch 数 |
| `dataset.val_episodes` / `dataset.max_train_episodes` | 固定留出回合数 / 训练回合上限；同数据、seed 下调整上限不会改变验证集 |

主策略训练只留出验证回合，当前不计算验证 loss；AFRO/CordViP 预训练会按各自配方验证。公共运行默认值在 `configs/common.yaml`，方法差异保留在各自配方中。

## 评估与选模

```bash
bash scripts/eval/evaluate.sh --checkpoint "$RUN/checkpoints/latest.pt" --episodes 100
bash scripts/eval/eval_pipeline.sh \
  --checkpoints "$RUN/checkpoints/step_00020000.pt" "$RUN/checkpoints/step_00040000.pt" \
  --selection-episodes 25 --test-episodes 100
```

将候选替换为实际存在的 checkpoint。默认每个候选使用自己的 raw/EMA、推理步数和执行长度；只有显式传 `--weights`、`--inference-steps` 或 `--action-steps` 才统一对应项。胜者在独立 test 回合继续使用选模时的实际设置，结果文件记录这些设置。跨方法比较的条件见[仿真评测机制](docs/仿真评测机制.md)。

## 部署与检查

下面以 DP3 为例，`observation` 是外部执行端提供的观测历史字典，字段和长度由 `runtime.info` 描述：

```python
from contextlib import closing
from dexmani_policy.deployment import load_policy

with closing(load_policy("/path/to/run")) as runtime:
    print(runtime.info.to_dict())
    runtime.warmup(observation)
    chunk = runtime.predict(observation)
    targets = chunk[:runtime.info.n_action_steps]
```

输出为物理单位下的控制目标，调度和发送由执行端负责。ODE 还需承接上一条实际已下发目标，完整使用方式见 [ODE 策略](docs/ODE策略.md)。

```bash
python -m dexmani_policy.test_tools.smoke_test --device cpu --max-updates 2 --resume-updates 2
```

上述检查使用临时数据；真实任务成功率需要独立仿真或实机评测。

## 阅读导航

| 文档 | 内容 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 数据流、核心模块与策略组成 |
| [仿真数据集格式](docs/仿真数据集格式.md) | 回合组织、主要模态与观测动作关系 |
| [手部动作表示](docs/手部动作表示.md) | SAT、DQ-RISE 与 DexLatent 的表示和解码 |
| [观测预训练](docs/观测预训练.md) | AFRO 与 CordViP 的学习目标及迁移 |
| [触觉与交互建模](docs/触觉与交互建模.md) | 当前触觉融合与未来力预测的作用 |
| [仿真评测机制](docs/仿真评测机制.md) | 闭环执行、候选选择与独立测试 |
| [ODE 策略](docs/ODE策略.md) | ODE 时序与执行承接 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远程同步、训练、续训与结果取回 |
