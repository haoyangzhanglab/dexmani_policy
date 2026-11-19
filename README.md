# DexMani Policy

面向灵巧手操作的策略学习研究仓库。从仿真或真机示教学习动作，用闭环评测比较不同的观测、动作表示和学习目标。

## 快速使用

在仓库根目录创建并启用 Conda `policy` 环境。

```bash
conda create -n policy python=3.10
conda activate policy
```

在该环境安装与本机 CUDA 匹配的 PyTorch 2.4.1 和 torchvision 0.19.1，再安装项目。

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

点云策略需要与 PyTorch 和 CUDA 匹配的 PyTorch3D。下例评测需要安装 `dexmani_sim`。

用 `dexmani_sim` 生成 `pick_apple_messy` 的示教和评测种子，将转换后的 Zarr 放在 `robot_data/pick_apple_messy.zarr`。 [仿真数据准备](https://github.com/haoyangzhanglab/dexmani_sim) · [示教数据格式](docs/仿真数据集格式.md)

```bash
RUN=experiments/train/dp3/pick_apple_messy/run01
bash scripts/training/train.sh dp3 hydra.run.dir="$RUN"
bash scripts/eval/evaluate.sh --checkpoint "$RUN/checkpoints/latest.pt" --episodes 100
```

新实验使用新目录。训练中断后，沿用原目录和示教数据，运行 `python -m dexmani_policy.training.resume --run "$RUN"`。

### DexJoCo

在 DexJoCo 导出示教后，将整个 Zarr 目录移到本仓库的 `robot_data/dexjoco/<task>.zarr`。实验配方组合 `dp` 或 `dp3` 与共用的 `dexjoco` 配置。评测环境安装导出时使用的 DexJoCo 提交，并在评测命令前设置 `MUJOCO_GL=egl`。 [DexJoCo 导出、训练和评测说明](https://github.com/haoyangzhanglab/dexjoco/blob/main/docs/dexmani_policy.md)

## 策略方向

| 方法 | 研究内容 |
| --- | --- |
| DP / DP3 | 分别用 RGB 或点云学习扩散策略 |
| ODE | 从上一条有效关节目标积分生成轨迹 |
| R3D | 用预训练三维表征生成动作 |
| ManiFlow | 结合 Flow Matching 与一致性学习 |
| SAT / DQ-RISE / DexLatent | 比较关节轨迹、手姿码本和连续手部潜表示 |
| AFRO / CordViP | 从场景变化或手物交互学习观测表征 |
| AdapTac | 用未来接触力预测引导动作生成 |

## 文档

- [项目架构](docs/项目架构.md) · [示教数据格式](docs/仿真数据集格式.md)
- [手部动作表示](docs/手部动作表示.md) · [观测预训练](docs/观测预训练.md) · [触觉与交互建模](docs/触觉与交互建模.md)
- [仿真评测与选模](docs/仿真评测机制.md) · [SSH 远程训练](docs/SSH服务器训练部署.md) · [预训练资源](data/README.md)
