# DexMani Policy

灵巧手操作策略学习研究仓库，围绕示教数据、观测表征和动作生成，提供统一的训练与仿真评测流程。

## 整体流程

示教轨迹被组织为观测历史与动作窗口。策略编码图像、点云、机器人状态或触觉，通过 Diffusion、Flow 或 ODE 模型生成动作序列。执行端执行配置的动作步数，再根据新观测重新预测，形成闭环；ODE 要求完整执行训练时的 chunk 长度。

研究主要围绕三个部分展开：

- **观测表征**：从全局场景特征、局部空间特征到手部交互历史。
- **动作表示**：直接生成关节目标，或借助手姿码本、连续手部潜表示生成动作。
- **学习目标**：在动作学习之外，引入观测预训练、一致性学习或未来触觉监督。

当前链路为 `dexmani_sim` 数据生成 → 本仓库策略训练 → `dexmani_sim` 闭环评测，不包含 Real 数据训练或 Sim-to-Real。模块分工见[项目架构](docs/项目架构.md)。

## 策略方向

| 方法 | 核心思路 |
| --- | --- |
| DP / DP3 | 分别以 RGB、点云结合关节状态，学习扩散策略 |
| ODE | 点云历史作为 condition，最新实测状态进入 FiLM 向量场，从上一条已下发指令积分绝对关节目标 |
| R3D | 使用预训练三维表征和空间条件生成动作 |
| ManiFlow | 结合 Flow Matching 与一致性学习 |
| SAT / DQ-RISE / DexLatent | 共享空间观测编码，研究关节轨迹、手姿码本与连续手部潜表示 |
| AFRO / CordViP | 可通过预训练学习场景动力学或接触与协调表征，再迁移到策略 |
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

仿真评测需要安装 `dexmani_sim` 并准备其任务资源与评测 seed 池；若它位于相邻目录，可执行 `python -m pip install -e ../dexmani_sim`，然后重新运行 `pip check`。

## 数据与初始化权重

示教数据默认位于 `robot_data/<task_name>.zarr`，字段与时间对应关系见[数据集格式](docs/仿真数据集格式.md)。运行其他任务时修改 `task_name`；自定义位置用 `zarr_path=/path/to/data.zarr`。

DP、DP3、ODE、ManiFlow、CordViP、AdapTac 和 InterAct-Dex 可直接从示教训练。各方法按需准备：

| 方法 | 新训练需要的初始化 |
| --- | --- |
| DP | 默认随机初始化 ResNet18＋GroupNorm，联合训练，无需外部权重；可选 DINOv2 使用 `bash scripts/utils/download_pretrained.sh dinov2`，配置见 `dp.yaml` 末尾完整注释 |
| SAT / DexLatent / R3D | 将脚本参数换成 `sat` / `dexlatent` / `r3d`；分别准备 DINOv3、DINOv3 + 手部先验、Uni3D |
| DQ-RISE | `download_pretrained.sh dqrise` 准备 DINOv3；另外提供 `initialization.codebook`，见[手部动作表示](docs/手部动作表示.md) |
| AFRO | 正式示例显式提供预训练 `initialization.encoder=.../encoder.pt`；默认 null 用于随机冻结特征实验，也可设置 `agent.freeze_visual=false` |
| CordViP | 使用包含手点云、物体点云、joint_state、action 的 `.cordvip.zarr`，可直接训练；预训练编码器可用 `initialization.encoder` 加载 |

AFRO/CordViP 的预训练与接触数据生成命令见[观测预训练](docs/观测预训练.md)。外部权重与可选主干见 [data/README](data/README.md)。完整 checkpoint 已包含推理所需权重和归一化参数，续训、推理不再读取初始化文件。

## 训练与续训

```bash
RUN="$PWD/experiments/train/dp3/pick_apple_messy/run_01"
bash scripts/training/train.sh dp3 task_name=pick_apple_messy hydra.run.dir="$RUN"
# 中断后在同一输出目录恢复；默认沿用 checkpoint 总预算，可用 --total-steps 延长。
python -m dexmani_policy.training.common.resume --run "$RUN"
```

新实验使用新目录。省略 `hydra.run.dir` 时自动生成带时间标识的目录。`config.yaml` 记录初始配置快照，续训时保留原文件；checkpoint 的 `effective_config` 是恢复训练的权威配方，`metrics.jsonl` 只追加指标，恢复不解析或裁剪旧内容（允许坏行、残行和重复 step），`checkpoints/` 保存完整训练状态和 `latest.pt`。默认 W&B 为离线模式，可用 `workspace.wandb_cfg=null` 关闭。

第一次 `Ctrl-C` 或 `SIGTERM` 请求在更新边界保存退出。停止请求若晚于本次更新的结束判断到达，可能再完成一次更新后保存退出；第二次信号强制退出，不保证保存。正常到达总预算或 `max_updates` 时同样先保存，与周期保存重合时只保存一次。

Resume 默认使用 checkpoint 保存的 Zarr 路径；可直接通过 `--zarr-path /new/data.zarr` 替换。检查数据任务名与训练配置一致、采集动作空间为 joint，以及必要的模型输入、动作维度和控制周期（`rtol=1e-6`、`atol=1e-9`），不比较数据内容或 episode 布局。允许同任务、同 shape 且控制周期一致的 joint 采集数据继续训练，数据适用性由研究者负责。顶层 `effective_config.control_dt_s` 和归一化参数沿用 checkpoint。

续训恢复 raw/EMA、优化器、学习率调度器、步数和 normalizer。`--max-updates 100` 只限制本次续训执行量，不修改总预算。恢复使用 seed + global_step 开始新的随机过程和数据遍历，不恢复旧 batch、RNG 或 worker 状态，不承诺逐位复现。

延长预算先更新调度器的总预算，再按累计步数恢复 LR；已结束的实验也可继续：

```bash
python -m dexmani_policy.training.common.resume --run "$RUN" --total-steps 150000 --max-updates 100
# 更改 LR、loss 系数、batch size 或冻结策略，使用新目录初始化实验。
bash scripts/training/train.sh dp3 initialization.policy="$RUN/checkpoints/latest.pt" \
  optimizer.lr=5e-5 hydra.run.dir="$PWD/experiments/train/dp3/initialized_run"
```

`initialization.policy` 使用来源 raw 权重和配套 normalizer，补齐构造需要的 architecture、num_codes 等信息；不读取原始预训练文件，不重新拟合归一化。当前配方必须保持同结构；optimizer、scheduler 和 EMA 新建，step 从 0 开始。它与 resume 互斥，不支持 LoRA/非 LoRA 结构转换或部分权重加载。

normalization 可以包含未使用字段，实际使用字段仍必须提供正确模式；这不使 encoder 自动适配缺失模态。梯度累积对本组 microbatch 等权平均，接受尾批较小 microbatch 的样本权重略高。训练采用标准流式 DataLoader，不缓存整轮 batch。

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

默认不录像；快速实验使用 `--episodes 5`，需要录像时显式加 `--video`。pipeline 的 `--video` 仅用于胜者 test。

将候选替换为实际存在的 checkpoint。默认每个候选使用自己的 raw/EMA、推理步数和执行长度；只有显式传 `--weights`、`--inference-steps` 或 `--action-steps` 才统一对应项。胜者在独立 test 回合继续使用选模时的实际设置，结果文件记录这些设置。跨方法比较的条件见[仿真评测机制](docs/仿真评测机制.md)。

## 推理与检查

下面以 DP3 为例，`observation` 是外部执行端提供的观测历史字典，字段和长度由 `runtime.info` 描述：

```python
from contextlib import closing
from dataclasses import asdict
from dexmani_policy.deployment import load_policy

with closing(load_policy("/path/to/run")) as runtime:
    print(asdict(runtime.info))
    runtime.warmup(observation)
    chunk = runtime.predict(observation)
    targets = chunk[:runtime.info.n_action_steps]
```

输出为物理单位下的控制目标，调度和发送由执行端负责。ODE 还需承接上一条实际已下发目标，执行约定见[手部动作表示中的 ODE 说明](docs/手部动作表示.md#ode从上一条指令积分生成轨迹)。

```bash
python -m dexmani_policy.test_tools.smoke_test --device cpu --max-updates 2 --resume-updates 2
```

上述检查使用合成数据；任务成功率需要独立的仿真闭环评测。

## 阅读导航

| 文档 | 内容 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 数据流、核心模块与策略组成 |
| [仿真数据集格式](docs/仿真数据集格式.md) | 回合组织、主要模态与观测动作关系 |
| [手部动作表示](docs/手部动作表示.md) | SAT、DQ-RISE、DexLatent 与 ODE 的动作表示及生成 |
| [观测预训练](docs/观测预训练.md) | AFRO 与 CordViP 的学习目标及迁移 |
| [触觉与交互建模](docs/触觉与交互建模.md) | 当前触觉融合与未来力预测的作用 |
| [仿真评测机制](docs/仿真评测机制.md) | 闭环执行、候选选择与独立测试 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远程同步、训练、续训与结果取回 |
