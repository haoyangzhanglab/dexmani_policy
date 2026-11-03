# DexMani Policy

灵巧手操作策略学习研究仓库，提供单卡训练、Checkpoint 恢复、仿真评测与推理部署接口。

## 环境准备

Python 3.10+，使用 Conda `policy` 环境，先安装匹配设备的 PyTorch / torchvision。

```bash
conda activate policy
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Sparse3D、R3D 及需要 FPS/kNN 的点云路径还需安装与 Torch/CUDA 匹配的 PyTorch3D 原生扩展。仿真评测依赖 `dexmani_sim`。

`pyproject.toml` 声明直接依赖和必要 API 范围；`requirements.txt` 固定已验证环境版本。更新依赖时只在环境文件维护精确版本，不顺带升级。RGB 组件使用的 transformers 及其相关依赖继续保留。

CordViP 接触标签使用独立工具生成：

```bash
python -m dexmani_policy.datasets.preprocess.contact_map --zarr-path INPUT --output OUTPUT
```

接触标签生成需要 Open3D，常规数据读取不加载它。Episode 划分由训练入口按 seed 和数量参数直接完成。目录、数据接口和检查责任见[项目架构](docs/项目架构.md#数据接口)。

## 快速开始

以下命令从仓库根目录执行，将 `<task>`、`<run>` 替换为实际名称。项目约定使用固定 `dexmani_policy/configs/` 下的名称或相对子路径（如 `dp3`、`pretrain/obs/afro`，可带 `.yaml`）；配置入口不提供安全隔离保证。续训使用下述显式 resume 入口。训练数据仅支持本地 Zarr 2 目录，默认位于 `robot_data/<task>.zarr`；不支持 ZIP、远程 store 或后端插件。

```bash
# 用实际数据完成短训、保存和原目录恢复；按硬件设置 batch/worker
python -m dexmani_policy.test_tools.smoke_test dp3 --max-updates 4 --resume-updates 2 \
  --override 'task_name=<task>' --override dataloader.num_workers=0

# 训练，支持 Hydra 配置覆盖
bash scripts/training/train.sh dp3 'task_name=<task>'

# 从同一实验目录的 latest.pt 恢复完整状态
python -m dexmani_policy.training.common.resume \
  --run experiments/dp3/<task>/<run> --device cuda:0 --num-workers 4

# 单个 Checkpoint 评测
python -m dexmani_policy.evaluate \
  --checkpoint <run>/checkpoints/latest.pt --weights ema --inference-steps 10 \
  --episodes 100 --seed 1024

# 显式候选选点与独立测试，保存胜者及实际推理设置
python -m dexmani_policy.eval_pipeline \
  --checkpoints <run>/checkpoints/<a>.pt <run>/checkpoints/<b>.pt \
  --weights ema --inference-steps 10 --selection-episodes 25 --test-episodes 100 \
  --seed 1024 --output-dir <run>/eval_ckpt_selector/<new-evaluation>
```

续训直接使用 Checkpoint 中的有效配方与完整训练状态，以保存的 dataset seed、训练上限及验证数量重建划分，根据已完成的 optimizer updates 从下一个 batch 继续。只有 fresh 装配配方、访问初始化来源和拟合统计；恢复不重写保存的部署事实或训练总预算，`--max-updates` 只限制本次调用。恢复要求同数据、同配方、同训练布局，不保证 worker 增强或编译计算逐位重放。日志直接追加，允许重复或较小 step，历史损坏行保留；训练进度只取 checkpoint。

fresh 与 resume 均使用 FP32 master parameters 和同设备 FP32 EMA；`training.use_bfloat16=true` 仅启用 BF16 autocast，不维护旧低精度 master parameters 的原精度训练轨迹。普通配方 `eval.use_ema` 默认跟随 `training.use_ema`，显式请求缺失 EMA 仍失败；ManiFlow 的 loss 继续要求 EMA。正式部署、仿真评测及 warmup 严格加载后使用 FP32，并合并 RGB LoRA；预测入口显式关闭 autocast，训练 checkpoint 保留适配器。

`dataset.sensor_modalities` 是唯一配方输入列表，实际字段宽度保存为 `normalization_dims`；不另声明模型消费字段或自动过滤额外模态。错误自定义配方可在消费处直接失败。Policy 不再支持整模态 dropout；Dataset 的点云、颜色和状态增强保持。loss 有限性在 accumulation group 结束、optimizer 更新前检查；有裁剪时保留非有限梯度检查，无裁剪路径不逐参数扫描。每次训练调用的首个 update、日志周期和已知最后一个 update 都记录指标。

归一化默认模式写在 YAML，可通过 `normalization.*` 覆盖；默认 joint action 使用训练集 `limits`，EEF 的 position/hand 使用 `limits`、rot6d 保持 identity。几何输入、DQ/DexLatent 手部 affine、R3D 辅助目标和触觉字段保留各自数学约束。新统计仅来自本阶段 train episodes 的真实源帧，不包含 val、unused、padding 或随机增强。AFRO/CordViP encoder 所属输入的 affine 与 identity 均直接继承，只有新字段重新拟合；有效配置记录实际使用的模式。续训、评测和部署直接复用完整策略 checkpoint 保存的最终统计。

FPS 使用 PyTorch3D 原生随机起点，训练采样消耗训练 RNG；评测关闭随机起点、选点噪声和输出 shuffle。

主策略默认 `dataset.seed=42`、`max_train_episodes=null`、`val_episodes=0`，使用全部 episodes，不建立离线验证流程。相较原先最多 100 条，这是训练配方变化；论文预算需显式设置 `dataset.max_train_episodes=100`。留出 V 条后，训练数为 `N−V` 或 `min(M,N−V)`，同一随机排列依次选择 train、val、unused；150/100/20 得到 100/20/30，80/100/10 得到 70/10/0。主策略 val 仅留出，未运行离线验证；resume 保留 checkpoint 保存的预算。训练在 20/40/60/80/100% optimizer updates 保存 Checkpoint，以 25 回合选点、100 回合独立测试，选点与测试 seed 分离。

RGB 在 CPU 预处理后统一为连续 CHW uint8，模型内转浮点；颜色增强和标定几何处理次序保留。checkpoint 按当前保存的配置和权重直接读取，不迁移预处理配方。非默认无界动作实验应显式配对，例如 `normalization.action=gaussian agent.clip_sample=false`；默认 limits 和 DDIM clipping 保持。

独立预训练默认 `dataset.seed=42`、`max_train_episodes=null`、`val_episodes=0`，数据与预算独立于主策略。DQ-RISE 默认从 VQ last 导出三键码表，显式 val 时按 `lookup_mse` 选 best；运行时仅依赖有序姿态表与 hand affine，fresh 自动保存实际 `num_codes`。详见 [手部动作表示](docs/手部动作表示.md)。AFRO/CordViP 使用包含输入归一化的轻量 `encoder.pt`，独立导出命令及继承规则见 [观测预训练](docs/观测预训练.md)。

## SAT / DQ-RISE / DexLatent dense observation

三种策略共用两帧 XYZ、RGB、标定和 joint state 编码器，保留 `[B,194,256]` 空间/状态 token，并统一标准 uniform flow matching、10 步 Euler 和 `Nobs=2,H=16,C=8`。SAT 采用“SAT 动作表示 + 6 层 768D DiTX 的本地适配”：`SATActionIO` 参照[官方动作表示](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/diffusion/sat.py)拼接关节轨迹与身份，形成 19 个 token，经公共 DiTX 后读出 `[B,16,19]` 动作速度；Flow 始终在归一化动作坐标中训练与积分。DQ 使用普通时间 token 输入输出，预测 7D arm 和连续码本索引后 lookup。三者共享 6 层、768D DiTX 层型与主干配置，但表示维度、token 数、输入输出头和量化机制不同，不是“只交换 token 轴”的消融，也不能直接横比 Flow MSE。默认 YAML 使用 `agent.fusion_stage=pre_global`、`agent.semantic_pool_sigma=0.04`，DINOv2-small 默认冻结。

DexLatent 使用同一观测及普通时间 token DiTX，内部预测 7D arm + 32D hand latent，对外保持 19D 绝对关节动作。官方 XHand E/D 冻结并使用 FP32 确定性 mean，保留官方 URDF 尺度；内部 latent limits 只拟合训练真实动作行。原始权重只在 fresh 初始化读取，完整策略内嵌先验和统计。训练、恢复、离线推理命令及预训练来源差异见 [手部动作表示](docs/手部动作表示.md#dexlatent-连续手部表示)。

```bash
bash scripts/utils/download_pretrained.sh dexlatent  # 仅准备 DINOv2-small 和官方 DexLatent 权重。
python -m dexmani_policy.test_tools.observation_checks --device cpu
python -m dexmani_policy.test_tools.hand_prior_checks --device cpu
bash scripts/training/train.sh sat 'task_name=<task>'
bash scripts/training/train.sh dqrise 'task_name=<task>' 'initialization.codebook=<codebook.npz>'
bash scripts/training/train.sh dexlatent 'task_name=<task>'
# B0：晚融合、均匀权重；B4：在新默认基础上单独启用既有 LoRA。
bash scripts/training/train.sh sat 'task_name=<task>' \
  agent.fusion_stage=post_global agent.semantic_pool_sigma=null
bash scripts/training/train.sh sat 'task_name=<task>' agent.rgb_backbone_config.tune_mode=lora
```

DP、SAT、DQ-RISE、DexLatent 可通过 `agent.rgb_backbone_name` 选择 `dino`、`dinov3` 或 `siglip`。默认仍为 DINOv2-small；DINOv2/SigLIP 保留 Transformers 实现，DINOv3-S 使用 timm 1.0.22，共用冻结、全量微调和 LoRA 接口。切换 DINOv3-S：

```bash
bash scripts/utils/download_pretrained.sh dinov3
bash scripts/training/train.sh sat 'task_name=<task>' \
  agent.rgb_backbone_name=dinov3 \
  initialization.rgb=data/pretrained/dinov3-small
```

DINOv3-S 默认输入 224×224，输出 14×14 patch 网格，语义对齐排除 CLS 和 4 个 register tokens；沿用当前双线性图像预处理。默认 DINOv2-small 从 `data/pretrained/dinov2-small` 初始化，DINOv3-S 使用 `data/pretrained/dinov3-small`；`initialization.rgb` 仍支持 HF repo ID。完整 checkpoint 保存架构和权重，续训与部署不访问初始来源。

可选 SigLIP 使用 Google Base patch16 / 224：

```bash
bash scripts/utils/download_pretrained.sh siglip
bash scripts/training/train.sh sat 'task_name=<task>' \
  agent.rgb_backbone_name=siglip \
  initialization.rgb=data/pretrained/siglip-base-patch16-224
```

下载参数 `sat`、`dqrise`、`dp` 仅准备默认 DINOv2-small，`r3d` 仅准备 Uni3D，`dinov3` 仅准备可选 DINOv3-S，`siglip` 仅准备可选 SigLIP Base；无参数或 `all` 准备 Uni3D、DINOv2-small、DexLatent。DQ-RISE 码本仍需通过原有预训练流程获得。小型检查使用随机视觉模型，不代表闭环性能。数据流、B0–B4 配置覆盖与恢复约定见 [观测编码](docs/项目架构.md#观测编码)；初始权重的本地路径见 [数据目录说明](data/README.md)。

## InterAct-Dex 动作耦合交互状态预测

`interact_dex` 使用手—场景—当前触觉表示及历史编码，在 `CoupledDiTX` 中让 wrist＋五指的六个交互 token 与动作共享计算，生成 19D joint actions。使用 `tactile=force`，通过 `tactile_source=sim`（默认）或 `tactile_source=real` 选择来源，分别使用各来源训练数据的 aggregate 触觉统计。

```bash
bash scripts/training/train.sh interact_dex tactile=force 'zarr_path=<sim_zarr>'
python -m dexmani_policy.test_tools.smoke_test interact_dex \
  --override tactile=force --override tactile_source=real --override 'zarr_path=<real_zarr>'
```

InterAct-Dex 包含以下四部分：

1. **动作耦合的力残差监督**：标量 `interaction_horizon=4` 定义预测边界 K，与实际执行长度 C 独立；K 不得超过可用未来动作数，默认窗口下上限为13。16个动作与6个交互 token 组成22-token 动态序列；每层固定 mask 使预测边界内的四个动作 `3:7` 与交互状态只能互读，全部 token 仍可读取静态观测。共享主干的五指特征预测单个 `+4` 力残差；标签只进入 loss，合法零值有效。
2. **arm/hand 分组动作损失**：7D arm 和 12D hand 各自求均值，权重各 0.5，避免维数决定两组相对权重。
3. **同指触觉融合**：每指通过共享门控 MLP 融合自身触觉，腕槽读取五指触觉均值；HandHistoryEncoder 编码空间关系和因果历史，动态交互 token 在主干中继续交换信息。
4. **场景上下文引导交互**：先得到十二个场景 token，其均值进入 HandSceneFusion 查询；四帧观测形成 `[B,72,192]` 静态条件，复用请求内观测 K/V 缓存。

力目标使用原始 batch：sim 先作世界系力差，再使用 query 时刻的末端旋转；real 直接作 SDK 通道差值。残差乘已有触觉 normalizer 的 scale，不另拟合统计。共享 `768→128→3` 力 head 的末层零初始化。

标准 YAML 的 `agent.future_force_loss_weight=0.1` 是未经闭环调优的初始值，Python 默认 `0.0` 不创建辅助头。权重为0时使用普通 Dataset，无需未来标签，但六个交互 token 仍参与动作生成。训练只编码一次观测并执行一次 Flow forward；推理跳过力 head，每次 velocity evaluation 重建交互状态，仅积分 FP32 动作。保持768维、8层、8 heads、10步 Euler，以及 `future_actions[B,13,19]` 输出。

完整 checkpoint 保存配置、optimizer、EMA 和归一化统计，按当前模型结构严格加载。时序及边界契约见[项目架构](docs/项目架构.md#interact-dex-动作耦合交互状态预测)。

interaction 组件按 `correspondence.py / future_tactile.py / hand_interaction.py` 组织。InterAct-Dex 固定使用场景聚合与因果历史编码；AdapTac 的力引导融合使用 cross-attention＋FFN。

## AdapTac 触觉实验

通过 `tactile=force`、`tactile=taxel` 或 `tactile=patch` 选择观测编码。`force` 默认使用 sim，真实 aggregate 数据追加 `tactile_source=real`；`taxel`、`patch` 默认使用 real，目前仅支持真实 dense 数据。preset 声明默认 `tactile_source`，命令行可覆盖；顶层值只传给解释触觉语义的 agent，Dataset 不接收该参数。模型触觉输入不允许来源缺失、null 或非法值；`fingertip_xyz` 单独存在不要求来源。三种配方均保留 `joint_state`、`point_cloud`，三种 encoder 都输出五个有序 64D token；future target 和 current guide 始终使用原始 aggregate 通道，固定 15D。

| preset | 来源 | 必需触觉输入 | encoder | 默认尺度 |
| --- | --- | --- | --- | --- |
| `force`（默认） | sim（默认）或 real | `finger_tactile_force [5,3]` | XHandForceEncoder | aggregate RMS |
| `taxel` | real | aggregate + `taxel_tactile_force [5,120,3]` | XHandTaxelEncoder | 两字段独立 RMS |
| `patch` | real | aggregate + dense | XHandPatchEncoder | 两字段独立 RMS |

```bash
bash scripts/training/train.sh adaptac tactile=force 'zarr_path=<sim_zarr>'
bash scripts/training/train.sh adaptac tactile=force tactile_source=real 'zarr_path=<real_zarr>'
bash scripts/training/train.sh adaptac tactile=taxel 'zarr_path=<real_zarr>'
bash scripts/training/train.sh adaptac tactile=patch 'zarr_path=<real_zarr>'
```

任务名称不同于默认值时，追加 `'task_name=<task>'`。每个 preset 用于独立新实验；对照可以覆盖 `normalization.finger_tactile_force=max_abs`、`normalization.taxel_tactile_force=max_abs`（dense 模式）及 `agent.force_loss_weight=...`。默认 loss weight 为 1.0，要求有限、非负。设为 0 仅关闭显式 force MSE；未来真值 force 的加噪输入及 action loss 仍可训练 force predictor，不是无未来标签基线。

AdapTac 默认关闭 compile；启用 `training.use_compile=true` 时仅编译 `action_decoder.model`，force predictor 始终使用 eager。此前在 PyTorch 2.4.1 / RTX 4090 的缩小模型上，编译 force predictor 并启用 BF16、accumulation=2 可出现 loss 有限但上游观测梯度非有限。本项目停止支持该特殊编译路径；这不表示底层问题已修复，也不承诺性能提升。

RMS 使用训练划分的全部真实原始标量拟合一个字段级尺度 `S=sqrt(sum(x²)/N)`，然后固定执行 `x/S`，不去均值、不裁剪；非空全零字段使用 identity affine。aggregate 与 dense 独立统计，完整 FP32 affine 随 checkpoint 保存。非退化字段应满足 `mean(z²)≈1`，不要求均值为零。该变换保留零点、方向和相对幅度，不能消除异常峰值或跨源物理差异。

```bash
# 实际数据短训、续训及保存恢复；按硬件资源设置 batch/worker
python -m dexmani_policy.test_tools.smoke_test --max-updates 4 --resume-updates 2 \
  --override tactile=taxel --override 'zarr_path=<real_zarr>' \
  --override 'task_name=<task>' adaptac

# 正式续训由 checkpoint 拥有配方，不再次传 tactile 或 normalization override
python -m dexmani_policy.training.common.resume --run experiments/adaptac/<task>/<run>
```

`test_tools/smoke_test.py` 每次接收一个配置，调用正式训练与原目录恢复路径。多配置检查使用 shell 循环；配置组合直接调用 `load_policy_config`，不在 smoke 中建立第二套配置或扫描流程。

检查按职责独立运行，支持 `--device cpu` 或 `--device cuda`；点云检查使用实际 PyTorch3D 后端，数据窗口检查固定 CPU。数值回归与合成短训无需外部数据集或预训练权重，不能代替正式数据训练和闭环验证：

```bash
python -m dexmani_policy.test_tools.decoder_checks --device cpu
python -m dexmani_policy.test_tools.observation_checks --device cpu
python -m dexmani_policy.test_tools.hand_prior_checks --device cpu
python -m dexmani_policy.test_tools.training_checks --device cpu
python -m dexmani_policy.test_tools.dataset_checks
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
python -m dexmani_policy.test_tools.remote_checks
```

默认短训 4 次更新、续训 2 次，两段执行上限独立于研究总预算，首次短训后须仍有可恢复的更新。`--output-dir` 指定保留产物的目录，空目录可用，已有训练产物由正式 fresh 入口拒绝覆盖；未指定时使用临时目录。编译和 BF16 沿用训练配置，可通过 `--override training.use_compile=false` 关闭编译；其他配方同样使用重复的 `--override` 覆盖。

离线恢复可用 `inspect_policy(experiment, checkpoint="latest", weights="raw")` 与 `load_policy(info, device="cpu")`，二者从 `dexmani_policy.deployment` 导入。调用 `policy.warmup(observation, samples=3)` 显式提供观测样本，然后调用 `policy.predict(observation)`。原始历史可不同长，调用方须保证最后 N 帧对齐。推理不读取训练数据；续训仍需未改写、未重排的原数据及保存的划分参数。恢复使用保存配置、affine 与 strict load，不读取外部 config.yaml。inspect 默认 latest，也接受明确文件路径，相对文件名以实验 checkpoints 目录解析；不支持 best 或百分比 selector。Runtime 返回完整未来动作窗口，调用方设置实际执行长度，详见[执行长度](docs/仿真评测机制.md#执行长度)。

fresh 按数据事实和配方输入保存控制周期、点云配方、depth 尺度及指尖名称，resume 原样保留，由公开 `PolicyInfo` 提供给 Real。inspect 和纯策略推理允许缺少硬件事实；设备入口按需检查。字段来源和动作上下文约定见[部署边界](docs/项目架构.md#离线推理与部署边界)。

数据边界必须区分来源：sim 是世界系 N，来自接触 impulse/physics_dt，再按控制周期子步进行有符号向量平均，目前没有 dense producer。real aggregate/dense 分别来自 SDK `calc_force`/`raw_force`，不是代码中的求和关系；单位为 `sdk_native_unknown_physical_scale`，轴为 `sdk_sensor_axes_not_world_frame`，canonical 尚未完整复制这些 raw 元信息。`fingertip_xyz` 只有位置，不能用于完成轴变换。五指顺序为 thumb/index/middle/ring/pinky，dense 保留 SDK 索引顺序。

future 标签从 query+1 的下一观测行开始，真实后继有效，重复末帧补齐后缀不计力损失或误差。零净力、力相消、reset 零值和合法 SDK hold 不代表无效；real 两通道有独立时间戳及有界 previous-valid hold，后继行不能解释为精确瞬时动作后真值。沿用现有数据时钟与 freshness 准入，不插值或补零修复。

`taxel` 是基础 dense 选项；`patch` 使用未经 SDK 验证的 `RIGHT_T30_PATCH_ROWS` / `RIGHT_T16_PATCH_ROWS` 重建区域图，是带区域假设的实验选项。Patch 在 normalized dense 上计算 `w=||z||/(||z||+activity_scale)`，默认 activity_scale=1.0，须能表示为有限正 FP32 值；soft activity 不代表接触概率或标定面积。其 dense normalization 只允许 RMS/MaxAbs。零输入的描述为零，含 embedding/bias 的最终 token 可以非零。区域图仅保存在构造配方的 `region_ids` 中，运行时重建 mask。

仿真公开入口 `SimRunner.run()` 在建环境前拒绝 real-source 触觉策略，内部 `_run_episode()` 由它调用；下划线仅表示内部 API。通用离线 PolicyRuntime 可以恢复两种来源；来源与数值语义保存在模型配置中，设备侧适配由独立的 real 项目负责。跨 sim/real 共享 checkpoint、混合统计、自动单位/坐标转换均不支持。smoke 检查、RMS 性质、patch 描述和 loss weight=1 不代表已证明任务成功率提高或配方最优。

## 实验产物

配置、Checkpoint、日志与评测结果默认保存在：

```text
experiments/<policy_name>/<task_name>/<run>/
```

设置 `workspace.wandb_cfg=null` 可仅保留本地 JSONL 日志。复现实验保留配置、Checkpoint 和对应源码；数据加工版本、模板等笔记按实验需要记录，不要求 source_id/commit/template 手工清单作为使用条件。

选模完成独立测试后保存 `selection_result.json`，不自动发布 best 指针。结果路径相对结果目录，`--output-dir` 可指定任意空目录。使用胜者时显式指定 checkpoint、weights、NFE 与执行长度；Real 的执行长度在其运行配置设置。历史 `best_ckpt.json` 保留但不再读取，详见[仿真评测机制](docs/仿真评测机制.md)。

每个 run 使用单写者，fresh 在昂贵初始化前检查输出冲突。保存采用同目录临时文件加 `os.replace`，不提供并发排他、强制 fsync 或多文件事务。远程下载用一次普通 rsync，同步已停止/完成的 run，全部成功后再使用。时间采样接口仅保留 uniform、beta、discrete；旧额外分布实验使用原源码版本。

训练划分只依赖 episode 顺序、数量、dataset seed 和数量参数，不读取或比较 `source_id`、episode/trial 身份，也不自动按 trial 分组。续训由研究者保证使用同一份未改写、未重排的数据；原样搬迁可更新路径。重新导出或改变数据顺序应开启新实验，避免复用失配的模型状态和 normalizer。若要避免同一长轨迹的片段跨集合，应在该实验的数据准备阶段处理。

## 文档

- [项目架构](docs/项目架构.md)：目录职责、数据接口、训练恢复与部署
- [仿真数据集格式](docs/仿真数据集格式.md)：Zarr 字段与数据约定
- [手部动作表示](docs/手部动作表示.md)：DQ-RISE 等手部表示与预训练
- [观测预训练](docs/观测预训练.md)：AFRO/CordViP 数据准备、预训练与 encoder 加载
- [仿真评测机制](docs/仿真评测机制.md)：选点、独立测试与单 Checkpoint 评测
- [SSH 服务器训练部署](docs/SSH服务器训练部署.md)：远程训练、同步与管理
