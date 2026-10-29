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

## AdapTac 触觉实验

只需一个 `tactile=...` override 切换来源和观测编码。四种配方均保留 `joint_state`、`point_cloud`，三种 encoder 都输出五个有序 64D token；future target 和 current guide 始终使用原始 aggregate 通道，固定 15D。

| preset | 来源 | 必需触觉输入 | encoder | 默认尺度 |
| --- | --- | --- | --- | --- |
| `sim_force`（默认） | sim | `finger_tactile_force [5,3]` | XHandForceEncoder | aggregate RMS |
| `real_force` | real | `finger_tactile_force [5,3]` | XHandForceEncoder | aggregate RMS |
| `real_taxel` | real | aggregate + `taxel_tactile_force [5,120,3]` | XHandTaxelEncoder | 两字段独立 RMS |
| `real_patch` | real | aggregate + dense | XHandPatchEncoder | 两字段独立 RMS |

```bash
bash scripts/training/train.sh adaptac tactile=sim_force 'zarr_path=<sim_zarr>'
bash scripts/training/train.sh adaptac tactile=real_force 'zarr_path=<real_zarr>'
bash scripts/training/train.sh adaptac tactile=real_taxel 'zarr_path=<real_zarr>'
bash scripts/training/train.sh adaptac tactile=real_patch 'zarr_path=<real_zarr>'
```

任务名称不同于默认值时，追加 `'task_name=<task>'`。每个 preset 用于独立新实验；对照可以覆盖 `normalization.finger_tactile_force=max_abs`、`normalization.taxel_tactile_force=max_abs`（dense 模式）及 `agent.force_loss_weight=...`。默认 loss weight 为 1.0，要求有限、非负。

RMS 使用训练划分的全部真实原始标量拟合一个字段级尺度 `S=sqrt(sum(x²)/N)`，然后固定执行 `x/S`，不去均值、不裁剪；非空全零字段使用 identity affine。aggregate 与 dense 独立统计，完整 FP32 affine 随 checkpoint 保存。非退化字段应满足 `mean(z²)≈1`，不要求均值为零。该变换保留零点、方向和相对幅度，不能消除异常峰值或跨源物理差异。

```bash
# 四种配方的配置和合成链路检查，无需实际 Zarr；CPU 也检查 BF16 autocast
for tactile in sim_force real_force real_taxel real_patch; do
  python dexmani_policy/smoke_test.py --config-only --override "tactile=$tactile" adaptac
  python dexmani_policy/smoke_test.py --contracts-only --device cpu --override "tactile=$tactile" adaptac
done

# 实际数据短训、续训及保存恢复；按硬件资源设置 batch/worker
python dexmani_policy/smoke_test.py --max-updates 4 --resume-updates 2 \
  --override tactile=real_taxel --override 'zarr_path=<real_zarr>' \
  --override 'task_name=<task>' adaptac

# 正式续训由 checkpoint 拥有配方，不再次传 tactile 或 normalization override
bash scripts/training/train.sh adaptac '+resume_from=experiments/adaptac/<task>/<run>'
```

契约检查使用缩小的 eager 网络、32 点点云和合成 episode，保留所选来源、encoder、normalization、horizon 和 point-cloud channels；梯度、guide 与恢复检查共用实际 Dataset 拟合的模型。短训命令使用实际配置，不再自动重复合成契约检查；`--compile` 仅用于短训模式。

离线恢复可用 `inspect_policy(experiment, checkpoint="latest", weights="raw")` 与 `load_policy(info, device="cpu")`，二者从 `dexmani_policy.deployment` 导入。推理不需要原训练数据；续训仍需原数据及保存的数据划分。恢复严格匹配配方和 state_dict，不自动迁移旧产物：旧 `force_scale` 数学配方，以及保存了冗余 `patch_mask` 权重的中间版本，需使用生成产物的源码恢复。

数据边界必须区分来源：sim 是世界系 N，来自接触 impulse/physics_dt，再按控制周期子步进行有符号向量平均，目前没有 dense producer。real aggregate/dense 分别来自 SDK `calc_force`/`raw_force`，不是代码中的求和关系；单位为 `sdk_native_unknown_physical_scale`，轴为 `sdk_sensor_axes_not_world_frame`，canonical 尚未完整复制这些 raw 元信息。`fingertip_xyz` 只有位置，不能用于完成轴变换。五指顺序为 thumb/index/middle/ring/pinky，dense 保留 SDK 索引顺序。

future 标签从 query+1 的下一观测行开始，真实后继有效，重复末帧补齐后缀不计力损失或误差。零净力、力相消、reset 零值和合法 SDK hold 不代表无效；real 两通道有独立时间戳及有界 previous-valid hold，后继行不能解释为精确瞬时动作后真值。沿用现有数据时钟与 freshness 准入，不插值或补零修复。

`real_taxel` 是基础 dense 选项；`real_patch` 使用未经 SDK 验证的 `RIGHT_T30_PATCH_ROWS` / `RIGHT_T16_PATCH_ROWS` 重建区域图，是带区域假设的实验选项。Patch 在 normalized dense 上计算 `w=||z||/(||z||+activity_scale)`，默认 activity_scale=1.0，须能表示为有限正 FP32 值；soft activity 不代表接触概率或标定面积。其 dense normalization 只允许 RMS/MaxAbs。零输入的描述为零，含 embedding/bias 的最终 token 可以非零。区域图仅保存在构造配方的 `region_ids` 中，运行时重建 mask。

sim rollout 拒绝 real-source AdapTac，Real 执行入口拒绝 sim-source AdapTac；通用离线 PolicyRuntime 可以恢复两种来源。跨 sim/real 共享 checkpoint、混合统计、自动单位/坐标转换均不支持。合成 smoke、RMS 性质、patch 描述和 loss weight=1 不代表已证明任务成功率提高或配方最优。

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
