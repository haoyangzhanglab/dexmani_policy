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
python -m dexmani_policy.test_tools.smoke_test --config-only dp3

# 公共 action/EMA/FPS/Dataset 契约，无需训练数据
python -m dexmani_policy.test_tools.smoke_test --contracts-only --device cpu dp3

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

RGB 编码器独立自检使用 `python -m dexmani_policy.test_tools.rgb_encoder_check resnet`，也可选择 `dino` 或 `siglip`（需要对应模型资源）。

配置检查不代表完整训练可运行。续训复用 Checkpoint 中的训练配方、数据划分和 normalizer，根据已完成的 optimizer updates 从下一个 batch 继续；不保证随机增强或编译计算逐位重放。

action 归一化由策略和动作表示固定：连续 joint 使用 `auto`（等价 `limits`），EEF 只使用分段 `auto`，rot6d 保持 identity；DQ、LAMP、LatentFlow/codec 和 R3D 辅助目标保留各自规则。`action=rms/gaussian/max_abs` 等非既定配方会在配置或恢复边界报错，关闭 clipping 也不能绕过。新训练先拟合训练划分的 affine；使用 AFRO/CordViP 预训练包时，包内观测统计覆盖对应字段，其余保留下游值。续训、评测和部署复用完整策略 checkpoint 保存的最终统计。

FPS 使用 PyTorch3D 原生随机起点。评测关闭随机起点、选点噪声和输出 shuffle；训练随机数消耗及距离并列时的选点可能与旧实现不同，checkpoint 结构兼容不代表随机轨迹逐位续接。

主策略默认最多使用 100 个训练 episodes，在 20/40/60/80/100% optimizer updates 保存 Checkpoint，以 25 回合选点、100 回合独立测试，选点与测试 seed 分离。正式选点要求训练预算对应的完整 Checkpoint 集合。

DQ/LAMP 的手部预训练见 [手部动作表示](docs/手部动作表示.md)；AFRO/CordViP 需先导出 `encoder.pt`，流程见 [观测预训练](docs/观测预训练.md)。模块预训练使用各自预算。

## AdapTac 触觉实验

只需一个 `tactile=...` override 切换来源和观测编码。基础配置声明 `tactile_source: ???`，preset 显式填入 `sim` 或 `real`；agent/dataset 引用同一顶层值。触觉输入不允许来源缺失、null 或非法值；`fingertip_xyz` 单独存在不要求来源。四种配方均保留 `joint_state`、`point_cloud`，三种 encoder 都输出五个有序 64D token；future target 和 current guide 始终使用原始 aggregate 通道，固定 15D。

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
  python -m dexmani_policy.test_tools.smoke_test --config-only --override "tactile=$tactile" adaptac
  python -m dexmani_policy.test_tools.smoke_test --contracts-only --device cpu --override "tactile=$tactile" adaptac
done

# 实际数据短训、续训及保存恢复；按硬件资源设置 batch/worker
python -m dexmani_policy.test_tools.smoke_test --max-updates 4 --resume-updates 2 \
  --override tactile=real_taxel --override 'zarr_path=<real_zarr>' \
  --override 'task_name=<task>' adaptac

# 正式续训由 checkpoint 拥有配方，不再次传 tactile 或 normalization override
bash scripts/training/train.sh adaptac '+resume_from=experiments/adaptac/<task>/<run>'
```

测试诊断集中在 `dexmani_policy/test_tools/`：`smoke_test.py` 提供 CLI，`smoke_checks.py` 提供配置、数据、短训与恢复公共流程，`common_contracts.py` 集中基础回归，`policy_contracts.py` 集中 AFRO/CordViP/SAT 专属检查，`adaptac_contracts.py` 集中触觉检查及合成数据。契约模块依赖公共检查，公共检查不反向导入契约模块或 CLI。`--contracts-only` 对任意策略运行公共 action 配方、EMA 数值/恢复、原生 FPS 和 Dataset 回归，并按实际策略类追加检查：AFRO/CordViP 验证统计迁移及预训练文件缺失后的短训恢复，SAT 验证公共短训恢复路径，AdapTac 验证触觉契约；修改 policy_name 不会跳过。检查使用临时合成数据，保留所选归一化和输入布局，缩小网络与执行预算。公共 FPS 检查缺少原生后端时报告 not_run，不宣称通过；SAT/AdapTac 模型检查仍需该后端。

AdapTac 契约检查使用缩小的 eager 网络、32 点点云和合成 episode，保留所选来源、encoder、normalization、horizon 和 point-cloud channels；梯度、guide 与恢复检查共用实际 Dataset 拟合的模型。短训命令使用实际配置，不再自动重复合成契约检查；`--compile` 仅用于短训模式。

离线恢复可用 `inspect_policy(experiment, checkpoint="latest", weights="raw")` 与 `load_policy(info, device="cpu")`，二者从 `dexmani_policy.deployment` 导入。推理不需要原训练数据；续训仍需原数据及保存的数据划分。恢复直接使用 checkpoint 内置配置和严格 state_dict 加载，保留参数精度；不重拟合 affine 或迁移产物。评测与部署选择的文件须位于当前实验的 checkpoints 目录；实验 config.yaml 与 checkpoint 按配套、未手工篡改的内部产物使用。

数据边界必须区分来源：sim 是世界系 N，来自接触 impulse/physics_dt，再按控制周期子步进行有符号向量平均，目前没有 dense producer。real aggregate/dense 分别来自 SDK `calc_force`/`raw_force`，不是代码中的求和关系；单位为 `sdk_native_unknown_physical_scale`，轴为 `sdk_sensor_axes_not_world_frame`，canonical 尚未完整复制这些 raw 元信息。`fingertip_xyz` 只有位置，不能用于完成轴变换。五指顺序为 thumb/index/middle/ring/pinky，dense 保留 SDK 索引顺序。

future 标签从 query+1 的下一观测行开始，真实后继有效，重复末帧补齐后缀不计力损失或误差。零净力、力相消、reset 零值和合法 SDK hold 不代表无效；real 两通道有独立时间戳及有界 previous-valid hold，后继行不能解释为精确瞬时动作后真值。沿用现有数据时钟与 freshness 准入，不插值或补零修复。

`real_taxel` 是基础 dense 选项；`real_patch` 使用未经 SDK 验证的 `RIGHT_T30_PATCH_ROWS` / `RIGHT_T16_PATCH_ROWS` 重建区域图，是带区域假设的实验选项。Patch 在 normalized dense 上计算 `w=||z||/(||z||+activity_scale)`，默认 activity_scale=1.0，须能表示为有限正 FP32 值；soft activity 不代表接触概率或标定面积。其 dense normalization 只允许 RMS/MaxAbs。零输入的描述为零，含 embedding/bias 的最终 token 可以非零。区域图仅保存在构造配方的 `region_ids` 中，运行时重建 mask。

仿真公开入口 `SimRunner.run()` 在建环境前拒绝 real-source 触觉策略，内部 `_run_episode()` 由它调用；下划线仅表示内部 API。Real 执行入口在设备启动前只准入 real-source 触觉策略，按保存的输入字段和来源判断，不依赖策略显示名称；通用离线 PolicyRuntime 可以恢复两种来源。跨 sim/real 共享 checkpoint、混合统计、自动单位/坐标转换均不支持。合成 smoke、RMS 性质、patch 描述和 loss weight=1 不代表已证明任务成功率提高或配方最优。

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
