# DexMani Policy

灵巧手操作策略学习研究仓库，围绕示教数据、观测表征和动作生成，提供统一的训练与仿真评测流程。

## 快速使用

在仓库根目录执行以下命令。脚本默认使用名为 `policy` 的 Conda 环境：

```bash
conda create -n policy python=3.10
conda activate policy
```

先安装与本机 CUDA 匹配的 PyTorch 2.4.1 和 torchvision 0.19.1，再安装其余依赖与本项目：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

点云 FPS/KNN 另需安装与 Torch/CUDA 匹配的 PyTorch3D，已验证版本为 0.7.8。仿真评测需安装 `dexmani_sim`；各策略需要的初始化权重见[预训练资源](data/README.md)。

准备好 `robot_data/pick_apple_messy.zarr` 后，以 DP3 为例启动训练；新训练使用独立输出目录，续训复用同一目录：

```bash
RUN=experiments/train/dp3/pick_apple_messy/run01
bash scripts/training/train.sh dp3 workspace.output_dir="$RUN" hydra.run.dir="$RUN"
python -m dexmani_policy.training.resume --run "$RUN"
bash scripts/eval/evaluate.sh --checkpoint "$RUN/checkpoints/latest.pt" --episodes 10
```

使用配方配置的数据和初始化权重，快速检查训练与续训：

```bash
python -B -m dexmani_policy.test_tools.smoke_test dp3
```

默认训练 4 步、续训 2 步，`--device` 可覆盖配方设备；不指定 `--output-dir` 时自动清理产物。`--config-dir` 可加载外部目录中的配置，相对数据、权重和输出路径均相对于启动目录。

`smoke_test.py` 检查更新步数和 checkpoint 发布。`test_regression.py` 包含数据采样、训练、状态恢复及推理评测的回归测试，运行 `python -B -m pytest -q dexmani_policy/test_tools`。

## 整体流程

**`dexmani_sim` 示教 / `dexmani_real` 显式字段导出 → 本仓库训练策略 → 仿真评测 / 真机部署**

训练将示教轨迹组织为观测历史与动作窗口，策略编码图像、点云、机器人状态或触觉，学习生成动作序列。执行端按策略约定执行动作，再根据新观测重新预测，形成闭环。

研究主要围绕三个部分展开：

- **观测表征**：全局场景特征、局部空间特征与视觉触觉交互。
- **动作表示**：直接生成关节目标，或通过手姿码本、连续手部潜表示还原动作。
- **学习目标**：动作模仿学习，以及观测预训练、一致性学习和未来触觉监督。

仿真数据的每个 episode 独立划分；Real 数据按原始采集分组划分，连续段仍独立采样。同一采集的段不会跨 train/val；`val_episodes` 和 `max_train_episodes` 均按采集组计数，日志同时显示组数与段数。核心模块及其数据流见[项目架构](docs/项目架构.md)。

Real 数据要求完整的逐段 `source_episodes` 和 `source_row_ranges`。普通训练、续训和部署允许处理参数及合法模板变化；接触预训练仅在 `model.contact_loss_weight > 0` 时要求 1024 个固定物体点槽及共享 `geometry_sha256`，详见[观测预训练](docs/观测预训练.md)。

## 策略方向

| 方法 | 核心思路 |
| --- | --- |
| DP / DP3 | 分别以 RGB、点云结合关节状态，学习扩散策略 |
| ODE | 以点云历史和实测状态为条件，从上一条实际下发目标积分生成关节轨迹 |
| R3D | 使用预训练三维表征和空间条件生成动作 |
| ManiFlow | 结合 Flow Matching 与一致性学习 |
| SAT / DQ-RISE / DexLatent | 共享空间观测编码，分别研究关节轨迹、手姿码本与连续手部潜表示 |
| AFRO / CordViP | 可通过预训练学习场景动力学或接触与协调表征，再迁移到策略 |
| AdapTac | 融合触觉与场景，通过未来力预测引导动作生成 |

各方法均接入本项目的数据与机器人设置，具体结构和监督方式有所不同。

## 阅读导航

| 文档 | 内容 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 数据流、核心模块与训练推理流程 |
| [仿真数据集格式](docs/仿真数据集格式.md) | 回合组织、主要模态与观测动作关系 |
| [手部动作表示](docs/手部动作表示.md) | SAT、DQ-RISE、DexLatent 与 ODE 的动作表示及生成 |
| [观测预训练](docs/观测预训练.md) | AFRO 与 CordViP 的学习目标及迁移 |
| [触觉与交互建模](docs/触觉与交互建模.md) | 触觉表征、未来力引导与几何融合组件 |
| [仿真评测机制](docs/仿真评测机制.md) | 闭环执行、候选选择与独立测试 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远程同步、训练、续训与结果取回 |
| [预训练资源](data/README.md) | 外部视觉权重与手部先验 |
