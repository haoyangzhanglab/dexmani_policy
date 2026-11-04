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
  --run experiments/train/dp3/<task>/<run> --device cuda:0 --num-workers 4

# 单个 Checkpoint 评测
python -m dexmani_policy.evaluate \
  --checkpoint <run>/checkpoints/latest.pt --weights ema --inference-steps 10 \
  --episodes 100 --seed 1024

# 显式候选选点与独立测试，保存胜者及实际推理设置
python -m dexmani_policy.eval_pipeline \
  --checkpoints <run>/checkpoints/<a>.pt <run>/checkpoints/<b>.pt \
  --weights ema --inference-steps 10 --selection-episodes 25 --test-episodes 100 \
  --seed 1024 --output-dir <run>/eval/<new-evaluation>
```

续训直接使用 Checkpoint 中的有效配方与完整训练状态，以保存的 dataset seed、训练上限及验证数量重建划分，根据已完成的 optimizer updates 从下一个 batch 继续。只有 fresh 装配配方、访问初始化来源和拟合统计；恢复不重写保存的部署事实或训练总预算，`--max-updates` 只限制本次调用。恢复要求同数据、同配方、同训练布局，不保证 worker 增强或编译计算逐位重放。日志追加前丢弃末尾未以换行结束的残段，完整历史行保持原样，允许重复或较小 step；训练进度只取 checkpoint。

fresh 与 resume 均使用 FP32 master parameters 和同设备 FP32 EMA；`training.use_bfloat16=true` 仅启用 BF16 autocast，不维护旧低精度 master parameters 的原精度训练轨迹。普通配方 `eval.use_ema` 默认跟随 `training.use_ema`，显式请求缺失 EMA 仍失败；ManiFlow 的 loss 继续要求 EMA。正式部署、仿真评测及 warmup 严格加载后使用 FP32，并合并 RGB LoRA；预测入口显式关闭 autocast，训练 checkpoint 保留适配器。

`dataset.sensor_modalities` 是唯一配方输入列表，实际字段宽度保存为 `normalization_dims`；不另声明模型消费字段或自动过滤额外模态。错误自定义配方可在消费处直接失败。Policy 不再支持整模态 dropout；Dataset 的点云、颜色和状态增强保持。loss 有限性在 accumulation group 结束、optimizer 更新前检查；有裁剪时保留非有限梯度检查，无裁剪路径不逐参数扫描。每次训练调用的首个 update、日志周期和已知最后一个 update 都记录指标。日志 LR 为本次更新实际使用的值，零起点 epoch 对应产生该 loss 的训练轮次，step 为已完成的更新数。

归一化默认模式写在 YAML，可通过 `normalization.*` 覆盖；默认 joint action 使用训练集 `limits`，EEF 的 position/hand 使用 `limits`、rot6d 保持 identity。几何输入、DQ/DexLatent 手部 affine、R3D 辅助目标和触觉字段保留各自数学约束。新统计仅来自本阶段 train episodes 的真实源帧，不包含 val、unused、padding 或随机增强。AFRO/CordViP encoder 所属输入的 affine 与 identity 均直接继承，只有新字段重新拟合；有效配置记录实际使用的模式。续训、评测和部署直接复用完整策略 checkpoint 保存的最终统计。

FPS 使用 PyTorch3D 原生随机起点，训练采样消耗训练 RNG；评测关闭随机起点、选点噪声和输出 shuffle。

主策略默认 `dataset.seed=42`、`max_train_episodes=null`、`val_episodes=0`，使用全部 episodes，不建立离线验证流程。相较原先最多 100 条，这是训练配方变化；论文预算需显式设置 `dataset.max_train_episodes=100`。留出 V 条后，训练数为 `N−V` 或 `min(M,N−V)`，同一随机排列依次选择 train、val、unused；150/100/20 得到 100/20/30，80/100/10 得到 70/10/0。主策略 val 仅留出，未运行离线验证；resume 保留 checkpoint 保存的预算。训练在 20/40/60/80/100% optimizer updates 保存 Checkpoint，以 25 回合选点、100 回合独立测试，选点与测试 seed 分离。

RGB 在 CPU 预处理后统一为连续 CHW uint8，模型内转浮点；颜色增强和标定几何处理次序保留。checkpoint 按当前保存的配置和权重直接读取，不迁移预处理配方。非默认无界动作实验应显式配对，例如 `normalization.action=gaussian agent.clip_sample=false`；默认 limits 和 DDIM clipping 保持。

独立预训练默认 `dataset.seed=42`、`max_train_episodes=null`、`val_episodes=0`，数据与预算独立于主策略。DQ-RISE 默认从 VQ last 导出三键码表，显式 val 时按 `lookup_mse` 选 best；运行时仅依赖有序姿态表与 hand affine，fresh 自动保存实际 `num_codes`。详见 [手部动作表示](docs/手部动作表示.md)。AFRO/CordViP 使用包含输入归一化的轻量 `encoder.pt`，独立导出命令及继承规则见 [观测预训练](docs/观测预训练.md)。

## SAT / DQ-RISE / DexLatent dense observation

三种策略共用两帧 XYZ、RGB、标定和 joint state 编码器，保留 `[B,194,256]` 空间/状态 token，并统一为 uniform + LBS 高噪声偏置训练采样（`time_shift_alpha=3.0`）、5 步均匀 Euler 和 `Nobs=2,H=16,C=8`。SAT 采用“SAT 动作表示 + 6 层 768D DiTX 的本地适配”：`SATActionIO` 参照[官方动作表示](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/diffusion/sat.py)拼接关节轨迹与身份，形成 19 个 token，经公共 DiTX 后读出 `[B,16,19]` 动作速度；Flow 始终在归一化动作坐标中训练与积分。DQ 使用普通时间 token 输入输出，预测 7D arm 和连续码本索引后 lookup。三者共享 6 层、768D DiTX 层型与主干配置，但表示维度、token 数、输入输出头和量化机制不同，不是“只交换 token 轴”的消融，也不能直接横比 Flow MSE。默认 YAML 使用 `agent.fusion_stage=pre_global`、`agent.semantic_pool_sigma=0.04`，DINOv2-small 默认冻结。

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

## InterAct-Dex 身体表征与未来力监督

InterAct-Dex 保留 Sparse3D、HandStateEncoder、原始点级局部几何及空间/因果历史编码，共享一次 Sparse3D，返回 `InterActFeatures(interaction[B,T,6,192], scene[B,T,12,192])`；scene 由12个 FPS 空间 query 经一次共同原点 3D RoPE cross-attention 读取全部 patches。Policy 按帧拼接两类 token，默认 condition `[B,72,192]`。一次几何 cross-attention 中，只有五指前三个 head 可读取零 K/V null；五指通过共享 `Linear(67,384)` 残差 FiLM 融合已归一化的 aggregate force，腕部由后续空间注意力接收全手信息。

普通 DiTX 保持768D、8层、8 heads和10步 Euler，输出 `future_actions[B,13,19]`。共享 `272→128→3` future head 从同一次观测编码的最新五指表征及有序动作前缀预测 h={1,2,4} 的力增量。前缀用原始 action 减当前 joint_state 后只乘 action scale；力目标只对原始力差乘 force scale。两种 mask 分离，不旋转力、不重复归一化。默认 lambda=0.1 尚未经闭环调优；lambda=0 不构造头、不读取未来标签，未来误差为 N.A.。在线推理不执行未来头，eval 模式仍支持离线受力评测。

本结构从 fresh training 开始，使用整策略 EMA；恢复要求保存的实际配置和 strict state_dict。模型、数据契约见[项目架构](docs/项目架构.md#handscenefusion-触觉融合与历史)；实际参数、检查命令、真实数据4+2步 smoke 及后续预算见[实施记录](docs/interact_dex_implementation_task.md#12-最终交付与实施记录)。有限验证不代表速度、成功率或泛化提升。

## AdapTac 触觉实验

默认采用多帧扩展配方 `N/H/C=2/16/8`：共享视觉、触觉和状态编码器逐帧处理，再按从旧到新的顺序拼接历史特征。batch=128、梯度累积1、100000次 optimizer updates及原学习率、EMA、网络宽度和损失权重保持不变；动作与力 diffusion 的默认推理步数均为20，与 horizon=16 无关。

以最后观测时刻 t 为准，无 padding 时观测为 `[t-1,t]`，训练动作为 `[t-1,...,t+14]`；策略返回 `[t,...,t+14]` 共15步，默认执行前8步。未来力标签为 `[t+1,...,t+15]`，超出 episode 的目标用末行填充并由 mask 排除，无有效后继力的窗口过滤。guide 按 `[观测力历史,预测未来力]` 拼接，force 为255维、dense 为30600维；动作损失继续通过预测力反传到 force predictor。

新配方使用新的训练 run。aggregate 可显式使用 `n_obs_steps=1 horizon=20 n_action_steps=8`；完整 checkpoint 按保存配置严格加载，配置窗口必须与权重形状一致。多帧结构与合成检查不代表速度或任务成功率提升。

通过 `tactile=force`、`tactile=taxel` 或 `tactile=patch` 选择观测编码。`force` 接收仿真或真实 aggregate 原始通道；`taxel`、`patch` 目前仅有真实 dense 数据生产端。无需指定来源参数，SimRunner 根据请求的观测字段，在创建环境前明确拒绝 dense 输入。三种配方均保留 `joint_state`、`point_cloud`，三种 encoder 都输出五个有序 64D token；每次运行仅选择一种触觉字段，观测、future target 和 current guide 均来自该字段。force 的力预测宽度为 15，taxel/patch 为 1800；数据文件可以包含两种字段，但运行只读取和统计选中的一种。

| preset | 来源 | 必需触觉输入 | encoder | 默认尺度 |
| --- | --- | --- | --- | --- |
| `force`（默认） | sim（默认）或 real | `finger_tactile_force [5,3]` | XHandForceEncoder | aggregate 三轴共享 Min-Max (`limits`) |
| `taxel` | real | `taxel_tactile_force [5,120,3]` | XHandTaxelEncoder | dense 三轴共享 Min-Max (`limits`) |
| `patch` | real | `taxel_tactile_force [5,120,3]` | XHandPatchEncoder | dense 三轴共享 Min-Max (`limits`) |

```bash
bash scripts/training/train.sh adaptac tactile=force 'zarr_path=<sim_zarr>'
bash scripts/training/train.sh adaptac tactile=force 'zarr_path=<real_zarr>'
bash scripts/training/train.sh adaptac tactile=taxel 'zarr_path=<real_zarr>'
bash scripts/training/train.sh adaptac tactile=patch 'zarr_path=<real_zarr>'
```

任务名称不同于默认值时，追加 `'task_name=<task>'`。每个 preset 用于独立新实验；对照可将各触觉字段的 `normalization.*` 显式设为 `rms` 或 `max_abs`，并覆盖 `agent.force_loss_weight=...`。默认 loss weight 为 1.0，要求有限、非负。设为 0 仅关闭显式 force MSE；未来真值 force 的加噪输入及 action loss 仍可训练 force predictor，不是无未来标签基线。

AdapTac 默认关闭 compile；启用 `training.use_compile=true` 时仅编译 `action_decoder.model`，force predictor 始终使用 eager。

触觉新训练默认使用 `limits`（Min-Max），无需额外参数。aggregate 沿时间和手指归约，dense 再归约 taxel，每次仅对选中字段保留 XYZ 三轴极值。每轴 `scale=2/(max-min)`、`offset=-1-scale*min`；`span<1e-4` 使用 `scale=1、offset=-min`。三个参数广播为原有 `(5,3)` / `(5,120,3)` buffers。只统计 train 真实源帧，合法零值保留；不含验证集、padding、窗口重复和增强。

非退化拟合数据映射至 `[-1,1]`，新数据可能越界；affine 可逆且不裁剪。AdapTac 当前与未来绝对力共用完整 affine（含 offset），力扩散不裁剪。InterAct-Dex 当前输入和统计都使用原始力，未来增量直接作差后只乘 scale，offset 相消，且不受 `[-1,1]` 限制。仿真原始力为世界系；真实传感器保持原生 SDK 通道，不能宣称为世界系。本仓库不自动对齐不同来源，也不提供跨来源混训功能。字段、形状、手指顺序和存储格式保持原样；`eef_pose` 及其他几何、动作变换继续保留。

逐轴尺度改变力损失的数值权重，当前 loss 类型、mask 和权重不自动调整，效果须实际训练验证。续训、EMA、评测和部署复用 checkpoint 的保存配置与 affine，不在线拟合或迁移。加载检查必需触觉 affine 完整性、shape、参数有限性和正 scale；limits 允许非零 offset，RMS/MaxAbs 要求 offset=0。恢复直接保留保存的实际参数，允许跨指或跨 taxel 不同；默认拟合仍产生共享参数。

参考 [KineDex normalizer](https://github.com/DinoMini00/KineDex_code/blob/9246f77d7b969b181b0f1b600601b9a9c8083011/diffusion_policy/model/common/normalizer.py) 的三轴拟合方式与 [FTP-1 normalization](https://github.com/michaelyuancb/ftp1-policy/blob/89fa681d6c014cce28300946b7526db808e0b1c1/src/openpi/normalization.py) 的输入语义统计组织；不引入分位数、在线统计或额外裁剪。

显式可选的 RMS/MaxAbs 分别从 train 原始标量拟合字段级 `S=sqrt(sum(x²)/N)` 或 `S=max(abs(x))`，固定执行 `x/S`，offset 为零，不裁剪；非空全零字段使用 identity affine。每次仅统计选中字段。这些已有可选模式同样使用原始训练观测，不做触觉旋转。

```bash
# 合成数据：三轴统计、坐标、Patch、raw/EMA 保存及连续训练/续训对照
python -m dexmani_policy.test_tools.tactile_normalization_checks
python -m dexmani_policy.test_tools.tactile_normalization_checks --device cuda:0

# 实际数据短训、续训及保存恢复；按硬件资源设置 batch/worker
python -m dexmani_policy.test_tools.smoke_test --max-updates 4 --resume-updates 2 \
  --override tactile=taxel --override 'zarr_path=<real_zarr>' \
  --override 'task_name=<task>' adaptac

# 正式续训由 checkpoint 拥有配方，不再次传 tactile 或 normalization override
python -m dexmani_policy.training.common.resume --run experiments/train/adaptac/<task>/<run>
```

`test_tools/smoke_test.py` 每次接收一个配置，调用正式训练与原目录恢复路径。多配置检查使用 shell 循环；配置组合直接调用 `load_policy_config`，不在 smoke 中建立第二套配置或扫描流程。

检查按职责独立运行，支持 `--device cpu` 或 `--device cuda`；点云检查使用实际 PyTorch3D 后端，数据窗口检查固定 CPU。数值回归与合成短训无需外部数据集或预训练权重，不能代替正式数据训练和闭环验证：

```bash
python -m dexmani_policy.test_tools.decoder_checks --device cpu
python -m dexmani_policy.test_tools.observation_checks --device cpu
python -m dexmani_policy.test_tools.hand_prior_checks --device cpu
python -m dexmani_policy.test_tools.training_checks --device cpu
python -m pytest dexmani_policy/test_tools/config_checks.py  # pytest 来自 pip install -e ".[test]"
python -m dexmani_policy.test_tools.tactile_normalization_checks --device cpu
python -m dexmani_policy.test_tools.dataset_checks
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
# 可选：安装 [preprocess] 后检查接触标签生成
OMP_NUM_THREADS=1 python -m dexmani_policy.test_tools.contact_map_checks
python -m dexmani_policy.test_tools.remote_checks
```

默认短训 4 次更新、续训 2 次，两段执行上限独立于研究总预算，首次短训后须仍有可恢复的更新。`--output-dir` 指定保留产物的目录，空目录可用，已有训练产物由正式 fresh 入口拒绝覆盖；未指定时使用临时目录。编译和 BF16 沿用训练配置，可通过 `--override training.use_compile=false` 关闭编译；其他配方同样使用重复的 `--override` 覆盖。

离线恢复可用 `inspect_policy(experiment, checkpoint="latest", weights="raw")` 与 `load_policy(info, device="cpu")`，二者从 `dexmani_policy.deployment` 导入。调用 `policy.warmup(observation, samples=3)` 显式提供观测样本，然后调用 `policy.predict(observation)`。原始历史可不同长，调用方须保证最后 N 帧对齐。推理不读取训练数据；续训仍需未改写、未重排的原数据及保存的划分参数。恢复使用保存配置、affine 与 strict load，不读取外部 config.yaml。experiment 使用相对仓库根目录或绝对路径指定完整 run；inspect 默认 latest，也接受明确 checkpoint 文件路径，相对文件名以该 run 的 checkpoints 目录解析。Runtime 返回完整未来动作窗口，调用方设置实际执行长度，详见[执行长度](docs/仿真评测机制.md#执行长度)。

fresh 按数据事实和配方输入保存控制周期、点云配方、depth 尺度及指尖名称，resume 原样保留，由公开 `PolicyInfo` 提供给 Real。inspect 和纯策略推理允许缺少硬件事实；设备入口按需检查。字段来源和动作上下文约定见[部署边界](docs/项目架构.md#离线推理与部署边界)。

数据边界必须区分来源：sim 是世界系 N，来自接触 impulse/physics_dt，再按控制周期子步进行有符号向量平均，目前没有 dense producer。real aggregate/dense 分别来自 SDK `calc_force`/`raw_force`，不是代码中的求和关系；单位为 `sdk_native_unknown_physical_scale`，轴为 `sdk_sensor_axes_not_world_frame`，canonical 尚未完整复制这些 raw 元信息。`fingertip_xyz` 只有位置，不能用于完成轴变换。五指顺序为 thumb/index/middle/ring/pinky，dense 保留 SDK 索引顺序。

future 标签从 query+1 的下一观测行开始，真实后继有效，重复末帧补齐后缀不计力损失或误差。零净力、力相消、reset 零值和合法 SDK hold 不代表无效；real 两通道有独立时间戳及有界 previous-valid hold，后继行不能解释为精确瞬时动作后真值。沿用现有数据时钟与 freshness 准入，不插值或补零修复。

合成检查覆盖 InterAct-Dex 与 AdapTac 三种 preset 的 limits、rms、max_abs 短训、raw/EMA 加载及续训。GPU 对照固定 CUDA/cuDNN 算法，要求模型与 EMA 权重在测试容差内一致、affine 完全一致；生产训练不承诺逐位确定性。本次单源改动的 CPU 与 RTX 4090 GPU 小模型检查已通过；真实触觉任务效果仍未验证。

力 MSE 按有效时间数 × 力宽度取均值。Patch 编码器使用 `raw*scale`，其历史 guide 与未来目标使用同一 dense 字段的完整 `raw*scale+offset`。

新 dense-only 结构与旧“dense 观测＋aggregate 未来力”checkpoint 不兼容，必须新建训练 run；不迁移或放宽严格加载。aggregate 的参数形状和数值行为保持不变。按默认宽度，dense 力预测与 guide 投影相对 15D 分支增加 16,452,345 个参数（FP32 参数约 62.76 MiB，不含梯度、优化器和 EMA）。

RTX 4090（24 GiB，PyTorch 2.4.1+cu124）合成资源短测使用正式 AdapTac 网络宽度、batch 128、1024 点、N/H/C=2/16/8、FP32、EMA、compile=false，动作和力推理均为 20 步。每种 preset 独立进程执行一次预热、三次更新及有限推理，均通过；关闭数据增强、num_workers=0，计时不含 DataLoader 取数，包含训练输入搬运、backward、optimizer 和 EMA。表中为 PyTorch peak allocated（含模型、EMA 和优化器），不含桌面和其他进程占用：

| preset | 首次更新峰值 GiB | 后续更新峰值 GiB | 后续更新均值 ms | B=1 推理均值 ms |
|---|---:|---:|---:|---:|
| force | 6.25 | 5.74 | 109.4 | 47.6 |
| taxel | 20.38 | 6.15 | 111.3 | 47.4 |
| patch | 6.46 | 6.12 | 113.0 | 48.1 |

上表使用 cuDNN benchmark=true；taxel 首次更新峰值较高，下面的独立 A/B 测量检验关闭 benchmark 的影响。这些合成短测不能视为真实数据吞吐、长训稳定性或方法性能比较，也不代表更小显存设备能够运行正式 batch。

2026-11-04 在 `fea0bb5` 配方上补做 taxel 完整更新 A/B：RTX 4090（23.51 GiB 可见显存）、driver 580.173.02、Python 3.10.20、PyTorch 2.4.1+cu124、CUDA 12.4、cuDNN 9.1.0，CPU threads=1。保持上述正式网络、batch 128、1024 点、N/H/C=2/16/8、FP32、EMA、compile=false、seed=42；合成输入、关闭增强、num_workers=0。各进程按相同 seed 和顺序初始化模型、生成输入。仅切换 benchmark；实际 deterministic=false、cuDNN TF32=true、matmul TF32=false、matmul precision=highest、benchmark_limit=10 均相同。GPU 同时承载约 1.7 GiB 桌面进程，未调整时钟。

顺序为 True/False 交替三轮，每行一个独立进程。首次更新前无训练预热；随后至少预热 30 次，直到相邻两个 10 次窗口的耗时中位数相差不超过 3%（六次均在 30 次满足），再测 50 次。更新包含 pinned CPU 输入搬运、forward、backward、梯度裁剪、AdamW、scheduler 和 EMA；生成输入、拟合 normalization、输出检查不计时。每次更新前后 CUDA 同步，首次和稳态分别重置 peak 统计，保留模型、EMA 和优化器，不调用 empty_cache。显存列为 **peak allocated / peak reserved**，含常驻内存；IQR 为稳态逐步耗时的四分位距。

| 轮次 | benchmark | 首次 ms | 首次 GiB | 稳态中位数 ms | 稳态 IQR ms | 稳态 GiB |
|---|---|---:|---:|---:|---:|---:|
| 1 | True | 1053.50 | 20.378 / 20.629 | 111.870 | 1.996 | 6.155 / 7.262 |
| 1 | False | 336.93 | 5.744 / 7.105 | 113.235 | 2.015 | 6.157 / 7.129 |
| 2 | True | 1025.94 | 20.378 / 20.629 | 111.187 | 2.386 | 6.155 / 7.262 |
| 2 | False | 337.36 | 5.744 / 7.105 | 112.223 | 1.734 | 6.157 / 7.129 |
| 3 | True | 1021.93 | 20.378 / 20.629 | 111.037 | 2.038 | 6.155 / 7.262 |
| 3 | False | 338.93 | 5.744 / 7.105 | 112.695 | 2.107 | 6.157 / 7.129 |

已确认：关闭 benchmark 的首次 allocated 峰值三轮均降低 **71.81%**；按三个进程中位数再取中位数，稳态耗时增加 **1.36%**，逐轮增加 0.93–1.49%。全部 loss、梯度及测量结束后的 raw/EMA B=1 输出有限（动作/力推理仍各 20 步）。另以独立诊断进程在 Conv1d 前后及权重梯度处读取累计峰值：True 时 taxel 两层 Conv1d 前向把峰值从 4.848 提升至 9.932、15.045 GiB，第二层反向进一步到 20.378 GiB，发生在 optimizer.step 之前；False 时该高峰消失。结合唯一变量 A/B，证据指向 taxel CNN 首次 cuDNN 算法搜索的临时工作区；未逐一解析候选算法。诊断带 hooks 的计时不纳入上表。

这满足“首次峰值至少降低 20%、稳态耗时增加不超过 5%”的工程判据，因此该环境下推荐 taxel 显式使用 `training.cudnn_benchmark=false`。共享默认仍为 True，未改变 force、patch 或其他策略默认行为；未验证其他 GPU/软件版本、真实数据吞吐或长训效果。`set_seed()` 会设置 benchmark=True，主训练在其后应用此字段。续训默认沿用 checkpoint 设置，字段缺省为 True；可用 `--no-cudnn-benchmark` / `--cudnn-benchmark` 显式覆盖同一设置，随后保存到新的 checkpoint。预训练和评测不使用这个主训练配置字段。

复现（在 policy 环境、仓库根目录运行；JSON 保存逐步耗时、loss、梯度范数、峰值、环境和实际后端值）：

```bash
for run in 1 2 3; do
  for benchmark in true false; do
    OMP_NUM_THREADS=1 python -m dexmani_policy.test_tools.taxel_benchmark \
      --benchmark "$benchmark" --output "/tmp/dexmani-taxel-ab/$benchmark-$run.json" || exit
  done
done

python -m dexmani_policy.train --config-name adaptac tactile=taxel training.cudnn_benchmark=false
# 已有 taxel run 也可在恢复时关闭 benchmark，无需改写源 checkpoint。
python -m dexmani_policy.training.common.resume --run <run> --no-cudnn-benchmark
```

`taxel` 是基础 dense 选项；`patch` 使用未经 SDK 验证的 `RIGHT_T30_PATCH_ROWS` / `RIGHT_T16_PATCH_ROWS` 重建区域图，是带区域假设的实验选项。Patch 使用零参考输入 `u=N(x)-N(0)=x*scale`，全部活动度、区域加权力、峰值和强度描述均基于 u，活动度为 `w=||u||/(||u||+activity_scale)`，默认 activity_scale=1.0，须能表示为有限正 FP32 值；soft activity 不代表接触概率或标定面积。其 dense normalization 默认 limits，也支持 RMS/MaxAbs；零参考是描述构造步骤，其结果不保证位于 `[-1,1]`。原始零输入的描述为零，含 embedding/bias 的最终 token 可以非零。区域图仅保存在构造配方的 `region_ids` 中，运行时重建区域索引与大小。

仿真入口在创建环境前拒绝 dense tactile。aggregate 字段不代表已验证 sim/real 来源兼容；当前不自动校验或转换触觉单位和坐标系。该检查位于公开入口 `SimRunner.run()`，内部 `_run_episode()` 由它调用；下划线仅表示内部 API。通用离线 PolicyRuntime 按 checkpoint 中的模型配置与 affine 恢复模型，不据此保证跨 sim/real 兼容；触觉来源及数值语义未被完整保存或校验，设备侧适配由独立的 real 项目负责。smoke 检查、RMS 性质、patch 描述和 loss weight=1 不代表已证明任务成功率提高或配方最优。

## 实验产物

默认布局如下，所有训练输出通过 `workspace.output_dir` 消费；保留现有 Hydra、预训练 CLI 和显式目录覆盖方式。

| 内容 | 默认路径 |
| --- | --- |
| 外部权重 | `data/pretrained/<model>/` |
| 示教／加工数据 | `robot_data/<dataset>.zarr`／`robot_data/<dataset>.cordvip.zarr` |
| AFRO／CordViP／VQ 预训练 | `experiments/pretrain/<afro或cordvip或vqvae>/<task>/<run>/` |
| 主策略 | `experiments/train/<policy>/<task>/<run>/` |
| 单 run 评测 | `<run>/eval/<evaluation>/` |

run 使用固定的 `run_id`，例如 `2026-11-03_14-30-00_seed42`；同次运行重复解析不更新时间，Hydra 批量运行保留 job 编号。分组使用 `task_name`，跨任务预训练可指定集合名称并显式提供数据路径。消融名称通过输出目录表达。

每个 run 根目录保留 `config.yaml`、`metrics.jsonl` 和启用时的 `wandb/`。观测预训练滚动保存 `checkpoints/latest.pt`，在线 `encoder.pt` 留在根目录；VQ 保存 `checkpoints/epoch_0100.pt`、`checkpoints/last.pt` 和有验证集时的 `checkpoints/best.pt`，根目录保留 `codebook.npz`。主训练只保存 `checkpoints/step_00020000.pt` 这样的 update 文件名，`checkpoints/latest.pt` 是指向最近保存文件的相对链接。

下游直接指定 `initialization.encoder=experiments/pretrain/afro/<task>/<run>/encoder.pt` 或 `initialization.codebook=experiments/pretrain/vqvae/<task>/<run>/codebook.npz`；正式实验优先使用已停止或完成的预训练 run。跨 run 比较显式指定 `--output-dir experiments/eval/<comparison>/`。

设置 `workspace.wandb_cfg=null` 可仅保留本地 JSONL 日志。复现实验保留配置、Checkpoint 和对应源码；数据加工版本、模板等笔记按实验需要记录，不要求 source_id/commit/template 手工清单作为使用条件。

选模完成独立测试后保存 `selection_result.json`，不自动发布 best 指针。`selection_result.json` 内的路径及单次评测 `result_details.json` 的 `checkpoint` 均相对各自结果目录；完整 run 搬迁保留内部引用，外部 checkpoint 仍需保持相对位置。`--output-dir` 可指定任意空目录。使用胜者时显式指定 checkpoint、weights、NFE 与执行长度；Real 的执行长度在其运行配置设置。详见[仿真评测机制](docs/仿真评测机制.md)。

每个 run 使用单写者，fresh 在昂贵初始化前检查输出冲突。保存采用同目录临时文件加 `os.replace`，不提供并发排他、强制 fsync 或多文件事务。远程下载用一次普通 rsync，同步已停止/完成的 run，全部成功后再使用。时间采样接口仅保留 uniform、beta、discrete；旧额外分布实验使用原源码版本。

训练划分只依赖 episode 顺序、数量、dataset seed 和数量参数，不读取或比较 `source_id`、episode/trial 身份，也不自动按 trial 分组。续训由研究者保证使用同一份未改写、未重排的数据；原样搬迁可更新路径。重新导出或改变数据顺序应开启新实验，避免复用失配的模型状态和 normalizer。若要避免同一长轨迹的片段跨集合，应在该实验的数据准备阶段处理。

## 文档

- [项目架构](docs/项目架构.md)：目录职责、数据接口、训练恢复与部署
- [仿真数据集格式](docs/仿真数据集格式.md)：Zarr 字段与数据约定
- [手部动作表示](docs/手部动作表示.md)：DQ-RISE 等手部表示与预训练
- [观测预训练](docs/观测预训练.md)：AFRO/CordViP 数据准备、预训练与 encoder 加载
- [仿真评测机制](docs/仿真评测机制.md)：选点、独立测试与单 Checkpoint 评测
- [SSH 服务器训练部署](docs/SSH服务器训练部署.md)：远程训练、同步与管理
