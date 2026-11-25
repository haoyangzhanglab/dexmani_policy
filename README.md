# DexMani Policy

从仿真或真机示教训练灵巧手操作策略，在闭环仿真中比较观测、动作表示和学习目标。

## 快速使用

### 安装

在仓库根目录创建脚本使用的 `policy` 环境。

```bash
conda create -n policy python=3.10
conda activate policy
```

先安装与本机 CUDA 匹配的 PyTorch 2.4.1 和 torchvision 0.19.1。点云策略还需要匹配的 PyTorch3D。

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### dexmani_sim

以 DP3 训练 `multi_grasp` 为例。安装 `dexmani_sim`，将示教放在 `robot_data/dexmani_sim/multi_grasp.zarr`，并准备该任务的评测种子池。 [仿真数据准备](https://github.com/haoyangzhanglab/dexmani_sim) · [Zarr 格式](docs/仿真数据集格式.md) · [评测种子](docs/仿真评测机制.md#运行评测)

```bash
RUN=experiments/train/dp3/multi_grasp/run01
bash scripts/training/train.sh dp3 task_name=multi_grasp hydra.run.dir="$RUN"
bash scripts/eval/evaluate.sh --checkpoint "$RUN/checkpoints/latest.pt" --episodes 100
```

新实验使用新的 `RUN` 目录。要沿用原数据和训练进度，在 `policy` 环境运行 `python -m dexmani_policy.training.resume --run "$RUN"`。

### DexJoCo

将导出的示教放在 `robot_data/dexjoco/<task>.zarr`。在 `dexmani_policy/configs/` 下新建配方，`defaults` 依次引用 `dp` 或 `dp3`、`eval/dexjoco` 和 `_self_`。训练时将配方名和 `task_name=<task>` 传给 `train.sh`。评测使用与数据导出相同的 DexJoCo 提交，并设置 `MUJOCO_GL=egl`。 [DexJoCo 示教导出与评测](https://github.com/haoyangzhanglab/dexjoco/blob/main/docs/dexmani_policy.md)

## 策略配置

| 配方名 | 方法 |
| --- | --- |
| `dp` / `dp3` | 分别基于 RGB 和点云的扩散策略 |
| `ode` | 从上一条实际应用的关节目标积分生成轨迹 |
| `r3d` | 用预训练三维表征生成动作 |
| `maniflow` | 结合 Flow Matching 与一致性学习 |
| `interaction_flow` | 融合五指触觉与场景几何，用 Rectified Flow 生成关节轨迹。 [InteractionFlow 模型架构](docs/InteractionFlow模型架构.md) |
| `sat` | 用 RGB、点云和关节状态观测生成关节轨迹 |
| `dqrise` / `dexlatent` | 用 RGB、点云和关节状态观测生成手姿码本索引或连续手部潜变量 |
| `afro` / `cordvip` | 分别从场景变化和手物交互学习观测表征 |
| `adaptac` | 用未来接触力引导动作生成 |

## 文档

- 了解项目和数据　[项目架构](docs/项目架构.md) · [示教数据格式](docs/仿真数据集格式.md)
- 理解策略　[InteractionFlow 模型架构](docs/InteractionFlow模型架构.md) · [触觉与交互建模](docs/触觉与交互建模.md) · [手部动作表示](docs/手部动作表示.md)
- 运行实验　[预训练权重](data/README.md) · [观测预训练](docs/观测预训练.md) · [仿真评测与选模](docs/仿真评测机制.md) · [SSH 远程训练](docs/SSH服务器训练部署.md)
