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

数据准备与完整检查使用独立工具，普通 Dataset 构造只做必要结构检查，不再完整扫描浮点字段：

```bash
python -m dexmani_policy.datasets.tools.check_data --zarr-path INPUT
python -m dexmani_policy.datasets.tools.make_split --zarr-path INPUT --output split.json \
  --seed 42 --val-ratio 0.1 --max-train-episodes 100
python -m dexmani_policy.datasets.tools.contact_map --zarr-path INPUT --output OUTPUT
```

接触标签生成需要 Open3D；另外两个工具及常规数据读取不加载它。目录、数据接口和检查责任见[项目架构](docs/项目架构.md#数据接口)。

## 快速开始

以下命令从仓库根目录执行，将 `<task>`、`<run>` 替换为实际名称。配置位于 `dexmani_policy/configs/`，训练数据默认位于 `robot_data/<task>.zarr`。

```bash
# 检查配置接口与导入，无需数据或 GPU
python -m dexmani_policy.test_tools.smoke_test --config-only dp3

# 检查实际数据的训练与验证 batch
python -m dexmani_policy.test_tools.smoke_test --data-only --max-batches 2 dp3

# 检查任务的数据路径
python -m dexmani_policy.training.common.config \
  --config-name dp3 'task_name=<task>' --check

# 训练，支持 Hydra 配置覆盖
bash scripts/training/train.sh dp3 'task_name=<task>'

# 从已有实验恢复到新的运行目录
bash scripts/training/train.sh dp3 \
  '+resume_from=experiments/dp3/<task>/<run>'

# Checkpoint 选点与独立测试
bash scripts/eval/eval_pipeline.sh 'experiments/dp3/<task>/<run>'
```

配置检查不代表完整训练可运行。续训复用 Checkpoint 中的训练配方、数据划分和 normalizer，根据已完成的 optimizer updates 从下一个 batch 继续；不保证随机增强或编译计算逐位重放。

action 归一化由策略和动作表示固定：连续 joint 使用 `auto`（等价 `limits`），EEF 只使用分段 `auto`，rot6d 保持 identity；DQ、LAMP、LatentFlow/codec 和 R3D 辅助目标保留各自规则。`action=rms/gaussian/max_abs` 等非既定配方会在配置或恢复边界报错，关闭 clipping 也不能绕过。新训练先确定 affine 来源；AFRO/CordViP 预训练包内的观测统计直接继承，仅对剩余字段拟合训练划分的统计。续训、评测和部署复用完整策略 checkpoint 保存的最终统计。

FPS 使用 PyTorch3D 原生随机起点。评测关闭随机起点、选点噪声和输出 shuffle；训练随机数消耗及距离并列时的选点可能与旧实现不同，checkpoint 结构兼容不代表随机轨迹逐位续接。

主策略默认最多使用 100 个训练 episodes，在 20/40/60/80/100% optimizer updates 保存 Checkpoint，以 25 回合选点、100 回合独立测试，选点与测试 seed 分离。正式选点要求训练预算对应的完整 Checkpoint 集合。

DQ/LAMP 的手部预训练见 [手部动作表示](docs/手部动作表示.md)；AFRO/CordViP 需先导出 `encoder.pt`，流程见 [观测预训练](docs/观测预训练.md)。模块预训练使用各自预算。

## InterAct-Dex 逐指触觉监督

`interact_dex` 使用六个身体槽的手—场景—当前触觉表示及历史编码，通过 DiTX 生成 19D joint actions。支持 `tactile=sim_force` 和 `tactile=real_force`，分别使用各来源训练数据的 aggregate 触觉统计。

```bash
bash scripts/training/train.sh interact_dex tactile=sim_force 'zarr_path=<sim_zarr>'
python -m dexmani_policy.test_tools.smoke_test --config-only --override tactile=real_force interact_dex
```

训练时，小型逐指读出头与动作输出 MLP 复用同一次 DiTX 前向的归一化隐状态，仅监督 `n_action_steps` 执行窗口内的后继触觉。标签保留在 batch 顶层，复用当前触觉统计计算 FP32 masked Smooth L1；不改变原采样窗口，也不作为动作条件。新增参数复用动作侧优化器设置、整策略 EMA 和严格 checkpoint 恢复。

标准 YAML 的 `agent.interaction_loss_weight=0.1` 是未经闭环调优的初始值，Python 默认 `0.0` 保留无辅助头的动作路径。部署不执行读出头，保持原积分步数、KV 缓存和完整 `future_actions` 输出。时序及边界契约见[项目架构](docs/项目架构.md#interact-dex-逐指触觉监督)，代码检查和短训通过不代表闭环性能提升。

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
# 四种触觉配方的配置检查，无需实际 Zarr
for tactile in sim_force real_force real_taxel real_patch; do
  python -m dexmani_policy.test_tools.smoke_test --config-only --override "tactile=$tactile" adaptac
done

# 实际数据短训、续训及保存恢复；按硬件资源设置 batch/worker
python -m dexmani_policy.test_tools.smoke_test --max-updates 4 --resume-updates 2 \
  --override tactile=real_taxel --override 'zarr_path=<real_zarr>' \
  --override 'task_name=<task>' adaptac

# 正式续训由 checkpoint 拥有配方，不再次传 tactile 或 normalization override
bash scripts/training/train.sh adaptac '+resume_from=experiments/adaptac/<task>/<run>'
```

`test_tools/smoke_test.py` 提供配置、数据 batch、短训和 checkpoint 续训检查。数据模式检查训练与验证 split 的核心形状及有限值，打印各字段的 shape/dtype，不额外拟合 normalizer。训练模式复用 `PolicyTrainer`，检查参数更新、checkpoint 保存加载和续训步数。

默认短训 4 次更新、续训 2 次；`--resume-updates 0` 关闭续训，总预算耗尽时明确打印未执行。`--output-dir` 保留新目录中的训练产物，否则使用临时目录。编译和 BF16 沿用训练配置，可通过 `--override training.use_compile=false` 关闭编译；其他配方同样使用重复的 `--override` 覆盖。

离线恢复可用 `inspect_policy(experiment, checkpoint="latest", weights="raw")` 与 `load_policy(info, device="cpu")`，二者从 `dexmani_policy.deployment` 导入。调用 `policy.warmup(observation, samples=3)` 显式提供观测样本，然后调用 `policy.predict(observation)`；LAMP 同时传入符合 `info.action_context_shapes` 的 `action_context`。原始历史可不同长，调用方须保证最后 N 帧对齐。推理不读取训练数据；续训仍需原数据及保存的数据划分。恢复直接使用 checkpoint 内置配置和严格 state_dict 加载，保留参数精度；不重拟合 affine 或迁移产物。inspect 与评测的相对 checkpoint 文件路径统一以实验 checkpoints 目录解析，选择的文件须位于该目录；实验 config.yaml 与 checkpoint 按配套、未手工篡改的内部产物使用。

训练将 Canonical 实际使用的数值配方保存到配置和 checkpoint，由公开 `PolicyInfo` 提供给 Real。非 Canonical 的训练、inspect 和纯策略推理不要求真机配方；设备入口负责拒绝缺失事实。字段来源和动作上下文约定见[部署边界](docs/项目架构.md#离线推理与部署边界)。

数据边界必须区分来源：sim 是世界系 N，来自接触 impulse/physics_dt，再按控制周期子步进行有符号向量平均，目前没有 dense producer。real aggregate/dense 分别来自 SDK `calc_force`/`raw_force`，不是代码中的求和关系；单位为 `sdk_native_unknown_physical_scale`，轴为 `sdk_sensor_axes_not_world_frame`，canonical 尚未完整复制这些 raw 元信息。`fingertip_xyz` 只有位置，不能用于完成轴变换。五指顺序为 thumb/index/middle/ring/pinky，dense 保留 SDK 索引顺序。

future 标签从 query+1 的下一观测行开始，真实后继有效，重复末帧补齐后缀不计力损失或误差。零净力、力相消、reset 零值和合法 SDK hold 不代表无效；real 两通道有独立时间戳及有界 previous-valid hold，后继行不能解释为精确瞬时动作后真值。沿用现有数据时钟与 freshness 准入，不插值或补零修复。

`real_taxel` 是基础 dense 选项；`real_patch` 使用未经 SDK 验证的 `RIGHT_T30_PATCH_ROWS` / `RIGHT_T16_PATCH_ROWS` 重建区域图，是带区域假设的实验选项。Patch 在 normalized dense 上计算 `w=||z||/(||z||+activity_scale)`，默认 activity_scale=1.0，须能表示为有限正 FP32 值；soft activity 不代表接触概率或标定面积。其 dense normalization 只允许 RMS/MaxAbs。零输入的描述为零，含 embedding/bias 的最终 token 可以非零。区域图仅保存在构造配方的 `region_ids` 中，运行时重建 mask。

仿真公开入口 `SimRunner.run()` 在建环境前拒绝 real-source 触觉策略，内部 `_run_episode()` 由它调用；下划线仅表示内部 API。通用离线 PolicyRuntime 可以恢复两种来源；来源与数值语义保存在模型和数据配置中，设备侧适配由独立的 real 项目负责。跨 sim/real 共享 checkpoint、混合统计、自动单位/坐标转换均不支持。smoke 检查、RMS 性质、patch 描述和 loss weight=1 不代表已证明任务成功率提高或配方最优。

## 实验产物

配置、Checkpoint、日志与评测结果保存在：

```text
experiments/<policy_name>/<task_name>/<run>/
```

设置 `workspace.wandb_cfg=null` 可仅保留本地 JSONL 日志。复现实验需保留配置、Checkpoint 和对应源码，并在实验目录的人工记录中关联数据身份与各阶段的版本，例如：

```text
source_id: <实际 Canonical source_id>
导出 Real commit SHA: <导出时的完整 SHA>
训练 Policy commit SHA: <训练时的完整 SHA>
手模板路径: assets/pointcloud_template/hand/hand_<num_points>.npz
```

Git 已跟踪的模板可用对应提交和路径恢复；不同处理阶段使用不同版本时分别记录仓库、SHA 和模板路径。未提交的代码或自定义模板应随实验手动归档，或先提交再记录。缺少历史记录时如实注明未知，不猜测补写。修改加工算法或模板后须重新导出数据。

Canonical 数据的 checkpoint resume 要求数据与保存的 split 都有非空 `source_id`。原样搬迁保留 ID，可继续 resume；重新导出的数据获得新 ID，不能复用旧 normalizer。旧数据仍可从头训练、指定 split 或读取评价；需要可靠 resume 时请显式重新导出并开始新实验。

## 文档

- [项目架构](docs/项目架构.md)：目录职责、数据接口、训练恢复与部署
- [仿真数据集格式](docs/仿真数据集格式.md)：Zarr 字段与数据约定
- [手部动作表示](docs/手部动作表示.md)：DQ/LAMP 等手部表示与预训练
- [观测预训练](docs/观测预训练.md)：AFRO/CordViP 数据准备、预训练与 encoder 加载
- [仿真评测机制](docs/仿真评测机制.md)：选点、独立测试与单 Checkpoint 评测
- [SSH 服务器训练部署](docs/SSH服务器训练部署.md)：远程训练、同步与管理
