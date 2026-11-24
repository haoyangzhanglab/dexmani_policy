# DexMani Policy

从仿真或真机示教训练灵巧手操作策略，在闭环仿真中比较观测、动作表示和学习目标。

## 快速使用

### 安装

在仓库根目录创建 Conda 环境。下文的 Shell 脚本使用 `policy` 环境。

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

安装 `dexmani_sim`，将 `multi_grasp` 的示教 Zarr 放在 `robot_data/dexmani_sim/multi_grasp.zarr`。评测从 `dexmani_sim.DATA_DIR/eval_seeds/multi_grasp.txt` 读取该任务的种子池。 [仿真数据准备](https://github.com/haoyangzhanglab/dexmani_sim) · [Zarr 格式](docs/仿真数据集格式.md)

```bash
RUN=experiments/train/dp3/multi_grasp/run01
bash scripts/training/train.sh dp3 task_name=multi_grasp hydra.run.dir="$RUN"
bash scripts/eval/evaluate.sh --checkpoint "$RUN/checkpoints/latest.pt" --episodes 100
```

每次新训练使用新的 `RUN` 目录。续训时保留原目录和数据，在 `policy` 环境运行 `python -m dexmani_policy.training.resume --run "$RUN"`。

### DexJoCo

将导出的示教放在 `robot_data/dexjoco/<task>.zarr`。在 `dexmani_policy/configs/` 下创建配方，`defaults` 依次引用 `dp` 或 `dp3`、`eval/dexjoco` 和 `_self_`。

训练时将配方名和 `task_name=<task>` 传给 `train.sh`。评测使用与导出相同的 DexJoCo 提交，并在运行 `evaluate.sh` 时设置 `MUJOCO_GL=egl`。 [DexJoCo 示教导出与评测](https://github.com/haoyangzhanglab/dexjoco/blob/main/docs/dexmani_policy.md)

## 策略配置

| 配方名 | 方法 |
| --- | --- |
| `dp` / `dp3` | 分别基于 RGB 和点云的扩散策略 |
| `ode` | 从上一条实际应用的关节目标积分生成轨迹 |
| `r3d` | 用预训练三维表征生成动作 |
| `maniflow` | 结合 Flow Matching 与一致性学习 |
| `interflow` | 用手物交互和关节状态作为 Flow Matching 的条件 |
| `sat` / `dqrise` / `dexlatent` | 分别使用关节轨迹、手姿码本和连续手部潜表示 |
| `afro` / `cordvip` | 分别从场景变化和手物交互学习观测表征 |
| `adaptac` | 用未来接触力引导动作生成 |

### InterFlow

InterFlow（Interaction-Conditioned Flow Policy for Dexterous Manipulation）把两帧观测各自的 6 个手物交互 token 和 1 个关节状态 token 拼成 14 个条件 token。Rectified Flow 用速度 MSE 训练 DiTX，推理默认通过 8 次 Euler 更新生成长度为 16 的关节目标轨迹。策略返回其中 15 个未来动作，评测按执行窗口取前 8 个。

将 `DATA` 设为包含关节目标、关节状态、RGB、点云、相机内外参、末端姿态、指尖坐标和逐指触觉力的示教 Zarr 路径。将 `DINO` 设为包含 `model.safetensors` 的 DINOv3 Small 权重目录。运行 `bash scripts/utils/download_pretrained.sh dinov3` 可将权重下载到 `data/pretrained/dinov3-small`。

```bash
DATA=/path/to/pick_apple_messy.zarr
DINO=/path/to/dinov3-small
RUN=experiments/train/interflow/pick_apple_messy/run01
bash scripts/training/train.sh interflow task_name=pick_apple_messy \
    zarr_path="$DATA" initialization.rgb="$DINO" hydra.run.dir="$RUN"
bash scripts/eval/evaluate.sh --checkpoint "$RUN/checkpoints/latest.pt" --episodes 100
```

## 文档

- [项目架构](docs/项目架构.md) · [示教数据格式](docs/仿真数据集格式.md)
- [手部动作表示](docs/手部动作表示.md) · [观测预训练](docs/观测预训练.md) · [触觉与交互建模](docs/触觉与交互建模.md)
- [仿真评测与选模](docs/仿真评测机制.md) · [SSH 远程训练](docs/SSH服务器训练部署.md) · [预训练资源](data/README.md)
