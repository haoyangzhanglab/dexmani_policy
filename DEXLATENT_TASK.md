# DexLatent 动作表示复现实施任务书

## 1. 任务目标

在 `haoyangzhanglab/dexmani_policy` 中接入 DexLatent 手部连续动作表示，用于与现有 SAT、DQ-RISE 比较。直接完成可训练、可恢复、可推理的最小实现；本任务书不是要求重新设计方案。

采用官方冻结 XHand 编码器/解码器，复用现有 `SpatialObservationEncoder + DiTX + RectifiedFlow`。机械臂保持 7D 绝对关节目标，手部使用 32D 连续 latent，最终仍向环境输出 19D 绝对关节目标。

实施遵守根目录 [AGENTS.md](AGENTS.md)。优先正确、直观、简洁、高效；不引入额外架构、校验器、测试组件或无关重构。保留工作区已有改动；如果当前源码已有相关实现，核对后补齐，避免重复创建。

本任务研究动作表示方案的整体效果。单 XHand 结果不能表述为跨手泛化验证，也不能称为完整 XL-VLA 复现。

## 2. 固定技术决策

| 项目 | 要求 |
| --- | --- |
| 外部数据与控制 | 保留 `action_key=action`、radians、19D joint commands |
| 机械臂 | 7D joint，沿用现有训练集 limits affine |
| 手部 | 官方 XHand encoder 的确定性 32D mean，E/D 完全冻结 |
| Flow 动作 | `[B,H,39] = arm7 + latent32` |
| 动作 token | 普通 DiTX 的 H 个时间 token，不使用 SATActionIO |
| 对外输出 | 解码并逆归一化后的 `[B,H,19]`，复用现有动作窗口接口 |
| 观测 | 保持 SAT/DQ 的两帧 XYZ、RGB、标定、19D joint_state |
| 训练损失 | 当前标准 RectifiedFlow 的全元素平均 MSE |
| 额外目标 | 不添加 KL、随机重参数采样、cycle、对齐、几何或辅助损失 |
| 控制链路 | 保持 joint 控制；不引入抓取中心、EEF action、FK 或 IK |

主比较不修改现有 SAT/DQ 损失，也不采用固定 arm/hand 分组权重。DQ-RISE 通过降低手部表示维度改善联合动作学习的平衡，这是其动作空间设计的一部分。保留表示维数、量化和 token 组织带来的影响；不同方法的 Flow MSE 不直接横向排名。

使用官方四手几何先验与使用本地示范训练 DQ 码本具有不同的预训练来源，实验说明中如实注明。共同策略配方不意味着全部预训练数据、参数量或计算量相同。

## 3. 阅读顺序与参考源码

先沿当前调用链阅读以下文件，确认实际接口后实现：

1. [SATPolicy](dexmani_policy/agents/policies/sat.py)、[DQRISEPolicy](dexmani_policy/agents/policies/dqrise.py)、[BasePolicy](dexmani_policy/agents/policies/base.py)。
2. [SAT/DQ 配方](dexmani_policy/configs/)、[动作表示说明](docs/手部动作表示.md)。
3. [DiTX](dexmani_policy/agents/action_decoders/backbones/ditx.py)、[RectifiedFlow](dexmani_policy/agents/action_decoders/flow.py)。
4. [normalizer](dexmani_policy/common/normalizer.py)、[Dataset](dexmani_policy/datasets/demonstration.py)、[训练数据组装](dexmani_policy/training/common/data.py)。
5. [配置组装](dexmani_policy/training/common/config.py)、[Trainer](dexmani_policy/training/trainer.py)、[checkpoint 恢复](dexmani_policy/common/checkpoint.py)、[Runtime](dexmani_policy/deployment/runtime.py)。
6. [权重目录约定](data/README.md)、[已有下载脚本](scripts/utils/download_pretrained.sh)、[远程数据同步](scripts/remote/sync_data.sh)。

设计核对基线为 `dexmani_policy@8011f0357e19b60a67a246abc417b29ed664629a`。以当前工作树接口为实施依据，不为匹配该基线回退或覆盖后续改动。

官方依据：

- [XL-VLA 论文](https://xl-vla.github.io/xl-vla.pdf)及 [HTML §3](https://arxiv.org/html/2603.10158v1#S3)：手部先验与下游策略分开，策略训练冻结 E/D。
- [DexLatent/model.py](https://github.com/EmptyBlueBox/DexLatent/blob/502750df34b9fa6a1b56a66481586887ec722f9c/HandLatent/model.py)：MLP、确定性 mean 和归一化入口。
- [DexLatent/train.py](https://github.com/EmptyBlueBox/DexLatent/blob/502750df34b9fa6a1b56a66481586887ec722f9c/HandLatent/train.py)：四手预训练入口。
- [官方 XHand URDF](https://github.com/EmptyBlueBox/DexLatent/blob/502750df34b9fa6a1b56a66481586887ec722f9c/Assets/xarm7_xhand/xarm7_xhand_right_hand.urdf)：独立关节顺序与尺度。
- [仿真 XHand 组合机器人 URDF](https://github.com/haoyangzhanglab/dexmani_sim/blob/6ccaed388aecfc0d78579fdaf5db2c012312c8e9/dexmani_sim/assets/robots/xhand/xarm7_xhand_right.urdf)及[独立手 URDF](https://github.com/haoyangzhanglab/dexmani_sim/blob/6ccaed388aecfc0d78579fdaf5db2c012312c8e9/dexmani_sim/assets/robots/xhand/xhand_right.urdf)：当前仿真关节位置上下限参考，不在本任务修改仿真仓库。
- [DQ-RISE v2](https://arxiv.org/html/2509.17450v2)：单步手姿量化、连续松弛与联合动作预测。

官方论文描述与公开代码不完全一致时，本任务的先验网络和数值路径以上述固定代码、已发布权重为准，不推测未公开的 VLA 实现。

## 4. 预训练权重准备

- [x] 确认或准备官方原始 checkpoint。
- [x] 将下载方式加入已有 `scripts/utils/download_pretrained.sh`；若已存在则复用。
- [x] 更新 `data/README.md` 中的来源、路径、用途和远程同步说明。

使用 `bash scripts/utils/download_pretrained.sh dexlatent`，仅准备 DINOv2-small 缓存和官方 DexLatent 原始权重。

固定来源：

```text
https://raw.githubusercontent.com/EmptyBlueBox/DexLatent/502750df34b9fa6a1b56a66481586887ec722f9c/Checkpoints/20260311_225425/checkpoint_epoch_1000.pt
```

仓库内目标位置：

```text
data/pretrained/dexlatent/checkpoint_epoch_1000.pt
```

原始文件约 2.16 MB，含四手联合预训练参数。不要假设先前聊天环境中的文件已经同步到当前 Codex 环境。文件缺失时从固定来源下载；保留原始文件，不另建 XHand-only 导出文件、转换工具或模型包格式。

复用已有下载入口及标准库；沿用已有文件复用行为，下载未完成的临时文件不得当作完整权重使用。不新增下载器框架、重试机制或内容校验系统。不得通过调整 `.gitignore` 将权重二进制强行纳入源码提交。

拟定配置：

```yaml
initialization:
  rgb: facebook/dinov2-small
  dexlatent: data/pretrained/dexlatent/checkpoint_epoch_1000.pt
```

权重只在 fresh 初始化读取。模型构造、续训和部署恢复不读取此来源路径，也不自动下载。`data/` 已被源码同步排除，远程沿用 `sync_data.sh` 与 `--data-root`；无需另建服务器路径约定。

## 5. 手部先验模块

- [x] 新增 `dexmani_policy/agents/hand_prior/dexlatent.py`，仅包含必要的 PyTorch 网络和选定权重加载逻辑。

网络要求：

- 输入 normalized hand12；隐藏宽度 `(64,128,64)`，各隐藏层使用官方的 `Linear + LayerNorm + ReLU`。
- 编码输出为 32D mean；解码输入 32D、隐藏宽度相同、12D 输出接 `Tanh`。
- 逐帧支持 `[B,H,12] → [B,H,32]`，不添加时间编码器。32D 对 XHand 是维度扩展，不宣称维度压缩。
- 从 checkpoint 的 `autoencoders` 中提取 `xarm7_xhand_right.` 前缀并去除前缀，保持层结构和参数名可对应。可保留官方 logvar 层以直接加载原状态，但编码路径只计算 mean。
- 使用已有 PyTorch 严格加载机制，不使用宽泛的 `strict=False` 掩盖不匹配；不增加独立参数检查器。
- 构造时就令 E/D 参数不可训练，确保完整策略恢复后也冻结。训练目标生成与推理解码均不反传至先验。
- E/D 保持 FP32，显式关闭外层 autocast 对其前向的影响；不要把先验整体转成 BF16。
- 官方 forward 使用确定性 mean，发布配方 `lambda_kl=0`。不计算无用的后验采样或 KL。
- 不导入 `CrossEmbodimentTrainer`，不引入 Pink、Pinocchio、FK、URDF 解析或可视化依赖。复制官方模型代码时保留其 MIT 版权许可说明。

## 6. 坐标与统计

- [x] 复用 Trainer 已有的 `hand_affine` 注入方式。
- [x] 用训练真实行一次性拟合内部 latent affine，并随完整策略保存。

| 字段 | affine 来源 | 保存位置 |
| --- | --- | --- |
| 观测 joint_state | 当前训练真实帧 limits，与 SAT/DQ 相同 | `normalizer["joint_state"]` |
| 动作 arm7 | 当前训练真实帧 limits，与 SAT/DQ 相同 | `normalizer["action"]` 前 7D |
| 动作 hand12 | 固定官方 XHand URDF limits | `normalizer["action"]` 后 12D |
| hand_latent32 | 当前训练真实动作经冻结编码器后的逐维 limits | `normalizer["hand_latent"]` |

物理手部顺序为 thumb bend/rota1/rota2、index bend/joint1/joint2、middle1/2、ring1/2、pinky1/2。官方 affine 常量为：

```python
lower = torch.tensor([0, -0.698, 0, -0.174, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
upper = torch.tensor([
    1.832, 1.745, 1.745, 0.174, 1.919, 1.919,
    1.919, 1.919, 1.919, 1.919, 1.919, 1.919,
], dtype=torch.float32)
scale = 2 / (upper - lower)
offset = -1 - lower * scale
```

唯一的 action normalizer 保存该手部 affine，不在 prior 中再持有副本。normalized hand 输入按官方入口限制到 `[-1,1]`；不能把本地示范 min/max 归一化的 hand 直接送入官方 encoder。

`dexmani_sim` 的组合机器人和独立手 URDF 已将 `right_hand_thumb_rota_joint1` 和 `right_hand_thumb_rota_joint2` 上限统一为 1.745 rad。当前 `dexmani_sim`、`dexmani_real` 与 DexLatent 官方 XHand URDF 的 12 个主动关节位置上下限一致；此处不代表完整 URDF、坐标系、effort 或实际运行限制全部一致，`dexmani_real` 的运行配置仍可能进一步收紧下限。

编码和解码继续使用现有官方固定 hand affine；物理关节目标沿用公共执行端处理，无需增加 DexLatent 专用限位转换、比例缩放或裁剪逻辑。

Fresh 时先加载先验并安装最终 action affine，再以 `dataset.iter_normalization_data("action")` 分块生成 hand latent，调用现有 `LinearNormalizer.fit_field("hand_latent", ..., mode="limits")`。编码与训练使用同一条 mean/归一化路径。

只统计当前 train split 的真实动作行，不消费 validation、不遍历有 padding 的训练窗口，也不读取图像来计算 latent 统计。不改写 Zarr、不缓存整份 latent 数据集、不保存独立统计文件。

`hand_latent` 是 Policy 内部字段，不加入 YAML 的数据 `normalization` 或 Dataset modalities。沿用现有 `normalizer_shapes` 保存/恢复内部字段形状。官方几何先验无需绑定本地任务 split；不要照搬 DQ 码本的预训练 split 一致性要求。

latent limits 是为本地 Flow 训练明确采用的数值适配，不宣称是未公开的 XL-VLA 下游归一化细节。

## 7. Policy 与动作链路

- [x] 新增 `dexmani_policy/agents/policies/dexlatent.py`。
- [x] 参考 DQRISEPolicy 组装普通 DiTX，保持公共 BasePolicy 接口。
- [x] 实现 39D Flow 训练目标和 19D 物理动作输出。

固定 `action_dim=19`、`control_action_dim=19`、`state_dim=19`。仅动作 backbone/Flow 内部使用 `flow_action_dim=39`。不要将顶层 action_dim 改为 39，也不要让 Dataset 输出 latent。

核心数据流如下，示意代码中的 E/D、归一化和目标构造均位于 FP32/no_grad 路径：

```python
# N 是 action normalizer，Nz 是内部 hand_latent normalizer。
q = N.normalize(batch["action"])
z = E(q[..., 7:].clamp(-1, 1))
flow_target = torch.cat([q[..., :7], Nz.normalize(z)], dim=-1)
# 用现有 RectifiedFlow.compute_loss(condition, flow_target)，不传 dim_groups。

# Flow 以 FP32 状态完成整个 Euler 积分后解码一次。
y = flow_prediction  # [B,H,39]
hand = D(Nz.unnormalize(y[..., 7:]))
prediction = N.unnormalize(torch.cat([y[..., :7], hand], dim=-1))
result = format_action_output(prediction)
```

训练标签来自 `batch["action"]` 的目标命令，不能用实测 `obs.joint_state` 替代。观测编码保持现有 19D state 输入，不编码为 latent，也不增加额外条件输入。

训练时观测和 DiTX 前向沿用当前 BF16 配方，续训保留训练参数存储及 optimizer/EMA 状态。正式部署、仿真评测及 warmup 在 `restore_policy()` 严格加载后整体转为 FP32，预测入口关闭 autocast。Flow 推理复用既有 `dtype=torch.float32` 参数；显式传递现有 generator。E 每个训练 batch 调用一次，D 每个完整预测调用一次；Euler 中不 decode、re-encode、投影或裁剪 latent。

沿用现有请求内观测编码和条件 K/V 复用，不新增跨请求缓存。完整 16 步先解码再调用 `format_action_output()`，返回 `[1:]` 的 15 步，执行其中前 8 步。保持 sync/async 能力，不扩展 RTC。

## 8. 配置、初始化与恢复

- [x] 新增 `dexmani_policy/configs/dexlatent.yaml`。
- [x] 在 `training/common/config.py` 加入现有 dense observation 的归一化与 DINO architecture fresh 分支。
- [x] 在 `training/trainer.py` 加入 DINO 初始化、官方先验加载、hand affine 注入和一次性 latent affine 拟合。

共同配方以当前 SAT/DQ YAML 为直接参考：

| 项目 | 既定配置 |
| --- | --- |
| 观测输出 | 两帧 XYZ/RGB/标定/state，`[B,194,256]` |
| 视觉 | 冻结 DINOv2-small，`fusion_stage=pre_global`，`semantic_pool_sigma=0.04` |
| DiTX | 6 层、768D、8 heads；其余主干参数与 SAT/DQ 一致 |
| 窗口 | `n_obs_steps=2, horizon=16, n_action_steps=8` |
| Flow | uniform 时间采样、time_shift_alpha=1、高斯起点、10 步 Euler |
| 策略训练 | batch 128、100000 updates、LR 1e-4；其余 optimizer/EMA/compile/调度配方一致 |
| 数据时序 | 保留数据实际 dt，不强制重采样为论文的 20 Hz 或改为 64 帧 |

同批比较使用相同实际 split、示范预算、训练 seeds、评测场景、选点预算和推理步数。

Fresh 初始化顺序：

1. 只构造网络，先验参数固定为不可训练。
2. 加载 DINO 和官方手部先验。
3. 拟合现有观测/action 统计，再覆盖 action 的手部 affine。
4. 拟合内部 latent affine。
5. 按现有流程创建 EMA、优化器及保存实际配置。

完整策略 checkpoint 内嵌所需 E/D 参数、action affine、latent affine 和模型构造配置。续训/评测/部署严格恢复这些内容，无需原始先验文件或训练数据来重建网络；续训读取训练数据仍遵守原有数据约定。不要在构造函数中读取路径、拟合统计或下载权重。

## 9. 修改范围与交付

预计核心改动为三个新增文件和两处已有接入点。文件路径均相对仓库根目录：

| 文件 | 职责 |
| --- | --- |
| `dexmani_policy/agents/hand_prior/dexlatent.py` | 官方手部网络、固定 affine、权重加载 |
| `dexmani_policy/agents/policies/dexlatent.py` | Policy、latent 统计、训练和输出路径 |
| `dexmani_policy/configs/dexlatent.yaml` | 共同策略配方与先验来源 |
| `dexmani_policy/training/common/config.py` | 现有配置分支接入 |
| `dexmani_policy/training/trainer.py` | fresh 初始化接入 |
| `scripts/utils/download_pretrained.sh`、`data/README.md` | 官方权重准备和存放约定 |
| `docs/手部动作表示.md`、必要的 README 段落 | 实际使用方式与比较边界 |

复用现有 BasePolicy、Dataset、DiTX、RectifiedFlow、normalizer、checkpoint、Runtime 和评测入口。不要为三种策略抽出新的通用 codec 框架、工厂、注册表或共享策略基类；不要修改 SAT/DQ 表示、损失和超参数，也不要增加通用模式开关。

不新增测试文件、检查脚本、验证框架、数据全量扫描、hash 准入规则、自动重试或静默回退。允许使用现有检查入口或临时命令做与改动相称的基本确认，不将检查发展成新的项目组件。缺失权重不得回退为随机先验。

交付要求：

- [x] fresh 配置可通过现有训练入口组装，权重已放在指定位置，示例命令与实际实现一致。
- [x] 19D 外部接口、39D 内部 Flow、FP32 冻结 E/D、train-only latent affine 均按上述契约接通。
- [x] 保存、续训与推理复用完整策略状态，先验不进入优化器更新。
- [x] 权重下载入口及现有文档同步更新，不提交预训练二进制。
- [x] 完成必要的就地确认和最终 diff 检查；报告真实执行结果与未验证项。

若现有环境具备真实数据和依赖，优先复用已有短训/恢复入口做最小确认；否则完成可行的源码、导入或张量路径检查并说明限制。不要为了检查引入 mock 后端、修改无关环境或搭建新测试系统。无需启动 100000 步正式训练或批量闭环评测，不得把未执行的训练/评测称为通过。

完成后给出改动文件、权重实际位置、可复制的训练/恢复/推理命令、实际运行过的确认及结果。根据真实完成情况更新本任务书复选框，不通过降低任务要求来标记完成。


## 10. 实际完成记录（2026-11-01）

已完成上述接入。新增 hand prior、Policy、YAML 三个文件；修改配置组装、Trainer fresh 分支、hand_prior 模块说明、已有下载脚本、`data/README.md`、`docs/手部动作表示.md`、`docs/项目架构.md`、根 README 及本任务书。SAT/DQ 实现和配方、公共 Flow、Dataset、checkpoint 与 Runtime 未修改；未新增测试文件、检查脚本或项目依赖，未提交或推送远程。

官方原始权重已通过更新后的下载脚本保存至：

```text
/home/zhanghaoyang/Desktop/dexmani_policy/data/pretrained/dexlatent/checkpoint_epoch_1000.pt
```

实际文件为 2,155,219 bytes，原始四手内容保留；仅在 fresh 中提取 XHand 分支并严格加载。`git check-ignore` 确认该路径仍被排除。

实际执行的基本确认：

- `policy` 环境中配置组装、DINO 架构解析及完整模型构造通过；外部 action/control/state 为 19D，内部 Flow 为 39D，普通时间 token，支持 sync/async。
- 官方 checkpoint 严格加载通过；FP32 mean 与使用官方参数逐层执行 Linear/LayerNorm/ReLU 的结果相等；外层 BF16 下 E/D 输出仍为 FP32，重复编码一致且无梯度。
- 真实 `robot_data/pick_apple_messy.zarr` 默认划分为 100 train / 15 validation episodes；CPU 原生 PyTorch3D FPS 可用。使用完整 6 层 768D 主干、BF16 配方、batch=1、workers=0、compile=false，已有 smoke 入口完成 fresh 1 update、保存、resume 至第 2 update。
- 临时命令确认真实观测 condition 为 `[1,194,256]`、Flow 为 `[1,16,39]`、Runtime 输出为 `(15,19)`；训练 loss 与全元素 Flow MSE 一致。每次 loss 只编码手部一次，完整预测只解码一次；BF16 外层下 Euler 状态保持 FP32，重置 generator 后预测一致。
- raw/EMA 的先验参数在训练后与官方权重逐张量相等，先验不在 optimizer 参数组中；action 手部 affine 为官方尺度，内部 `hand_latent` 形状保存为 `[32]`，未进入 Dataset normalization 配置。
- 临时移走原始权重后，raw/EMA 都完成严格恢复和真实观测推理；随后在原始权重缺失、RGB 初始化来源不可用时续训至第 3 update（loss=1.83636），所有先验参数和 affine 保持不变。检查结束后原始权重已恢复原路径。
- 下载脚本 shell 语法与最终 `git diff --check` 通过。验证全部使用现有入口或临时命令，无 mock 后端。

本次 smoke 产物保留于 `/tmp/dexlatent-smoke-20261101`，最终 `checkpoints/latest.pt` 为第 3 update。复现基础短训命令如下（输出目录须不存在）：

```bash
conda activate policy
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m dexmani_policy.test_tools.smoke_test dexlatent --device cpu \
  --max-updates 1 --resume-updates 1 --output-dir /tmp/dexlatent-smoke-new \
  --override dataloader.batch_size=1 --override dataloader.num_workers=0 \
  --override training.use_compile=false
```

正式训练、恢复、离线真实观测推理及共用 split 命令见 [动作表示文档](docs/手部动作表示.md#dexlatent-连续手部表示)。此前 CUDA 不可访问是 sandbox 限制；经用户授权在 sandbox 外完成以下 GPU 确认。

### Sandbox 外 CUDA 补充确认

- 环境：RTX 4090（24 GiB），driver 580.173.02，PyTorch 2.4.1+cu124，CUDA 12.4；`torch.cuda.is_available()` 为 True。
- 同一真实数据及默认训练划分，保留 batch 128、BF16、EMA、完整主干和 `torch.compile`，只将 workers 改为 0、日志间隔改为 1。已有 smoke 入口完成 2 步 fresh、保存、恢复至第 3 步；loss 分别为 1.46498、1.47228、1.47497，未出现非有限值或 OOM。编译输出 symbolic_shapes 的 unknown range 警告，运行正常完成。
- GPU checkpoint 严格恢复通过，保存的参数名不含 `_orig_mod.`；raw/EMA 先验均与官方权重逐张量相等，未进入 optimizer，所有 affine 保持 FP32 且 raw/EMA 一致。
- 临时移走原始先验文件后，raw/EMA 都在 CUDA 上完成真实观测推理；检查后权重已恢复。外层 BF16 下 E/D 仍为 FP32，mean 重复编码一致；每次预测编码一次观测、完成 10 次 FP32 Euler 更新、解码一次，输出 `(15,19)` 且全部有限。重置 generator 后输出相等；默认 Runtime 无外层 autocast 的推理也通过。
- GPU 产物：`/tmp/dexlatent-cuda-smoke-20261101/checkpoints/latest.pt`，global_step=3。

可复制命令（在可访问 GPU 的环境运行，输出目录须不存在）：

```bash
conda activate policy
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m dexmani_policy.test_tools.smoke_test dexlatent --device cuda:0 \
  --max-updates 2 --resume-updates 1 --output-dir /tmp/dexlatent-cuda-smoke-new \
  --override dataloader.num_workers=0 --override training.loop.log_interval_steps=1
```

未验证正式长训练、默认 8-worker 数据加载、sync/async 闭环执行或任务成功率；短训仅确认数值及生命周期链路，不代表性能评估。


### 全链路审查与清理

- 先验改用已有 `agents.modules.build_mlp`，删除重复 MLP 构造函数，保持官方层序、参数名和严格加载。logvar 层继续用于原始官方 state 的直接严格加载，不参与编码计算。
- 删除 Policy 中重复的 autocast/no_grad 包装：FieldNormalizer 保证 affine 运算为 FP32，先验在 E/D 边界关闭 autocast，内部 `_encode_actions` 切断标签梯度。固定维度改为类常量供构造和推理共用，补齐相关方法返回类型。
- 对齐模块说明、项目架构中的类路径/观测/动作/统计/执行能力、三种策略比较及训练入口。训练示例显式传入对应数据的 task_name；删除 DQ 会返回连续/离散索引的过时描述。
- 审查了 fresh、EMA、optimizer、strict checkpoint 恢复、Dataset 真实行统计、Runtime、脚本及现有检查入口，未引入新的兼容、回退或校验机制。
- 清理后重新完成 RTX 4090、batch 128、BF16、compile 的 fresh 2 步及 resume 1 步；三步 loss 与前次记录一致。新产物为 `/tmp/dexlatent-review-smoke-20261101/checkpoints/latest.pt`。参数键、先验和统计与清理前产物一致。
- CUDA/BF16 临时命令确认 raw/EMA 的目标与显式 FP32 参考逐值相等，标准全元素 MSE 一致，标签和先验无梯度；原始文件缺失时恢复、10 步 FP32 Euler、单次解码和 generator 重放均通过。
- 四个核心/接入文件的定向 mypy 检查通过（`--follow-imports=skip --ignore-missing-imports`）；全包 compileall、下载脚本 `bash -n`、最终 diff 检查通过。离线 wheel 构建通过，并核对两个新模块及 YAML 均完整入包。未新增测试文件或检查脚本，未执行长训练或闭环评测。
