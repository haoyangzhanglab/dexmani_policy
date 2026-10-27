# Codex 任务书：AFRO 与 CordViP 原生复现

状态：待实施的设计与验收契约，非已实现的功能说明。核对日期：2026-10-28。文中的新增命令须在实施后验证。

目标：完成两条可独立使用的流程：**观测预训练 → encoder 导出 → 19 维关节策略训练 → checkpoint 恢复 → 仿真评测**。保留论文核心机制，新增算法尽量直接移植官方代码，复用本项目数据、策略、训练和评测组件。

先读根目录 [AGENTS.md](../../AGENTS.md)、[项目架构](../项目架构.md)、[仿真数据集格式](../仿真数据集格式.md)、[仿真评测机制](../仿真评测机制.md)。执行提示见 [afro_cordvip_prompt.md](afro_cordvip_prompt.md)。

## 1. 版本、范围与证据

| 来源 | 本任务核对版本 | 用途 |
| --- | --- | --- |
| [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy/tree/874090b5c4ea9e3d5b50e7e317772d0185dd06d0) | `874090b5c4ea9e3d5b50e7e317772d0185dd06d0`，核对时 main | 原生接口与数据/运行语义 |
| [AFRO 论文](https://arxiv.org/html/2512.00074v4) | arXiv `2512.00074v4` | §4.1–4.4、§5.3 |
| [AFRO 官方代码](https://github.com/KolaKivy/AFRO/tree/283024a49e2c14bc030debf15383a4e7ee73836d) | `283024a49e2c14bc030debf15383a4e7ee73836d` | PointNet 分支的实际算法与配置 |
| [CordViP 论文](https://arxiv.org/html/2502.08449v2) | arXiv `2502.08449v2` | §III、附录 B、表 XI |
| [CordViP 官方代码](https://github.com/xuanxuanzzzii/cordvip/tree/ad6d441abe7b0480cf79b0311db266965c1313b6) | `ad6d441abe7b0480cf79b0311db266965c1313b6` | PointNet、contact map、预训练与迁移 |

实施时先检查当前 HEAD 对上述本项目版本的相关变化；不回退仓库、不覆盖已有工作。官方参考代码固定到表中 SHA，不混用版本。允许 clone 到仓库外 `/tmp`，或仓库内不纳入 Git 的 `tmp/`；不整体提交，不作为运行依赖。

本任务是**官方方法在本项目数据和机器人上的复现/迁移**，不声称复现论文成功率表格。机器人、数据、频率、任务、预算和评测协议均有差异。首版限定仿真、XYZ、PointNet、`action=arm7+hand12`；不增加触觉、RGB、语言、替代 encoder、真机感知系统或新的动作表示。

### 1.1 最新接口必须采用

- 普通 normalizer 使用所选训练 episodes 的**全部真实帧，每帧一次**；不使用旧方案的“窗口消费的唯一 source rows”。验证/排除 episodes 不拟合，padding 不重复加权。
- 实际 split 内嵌在 config/checkpoint 的 `dataset.split_manifest`。续训复用保存的 split 与 affine，不重划分、不重拟合。可用现有 split CLI 生成一个共享 manifest，不为每次运行复制一套文件。
- 新策略实现 `BasePolicy.prepare_restore_config()` 与 `initialize_training()`。最新恢复已使用类级 hook，不恢复旧版按名称遍历初始化路径的方案。
- 下游 YAML 复用一层 `configs/common.yaml`；独立预训练配置不伪造 policy/action，也不依赖 Hydra 运行时目录。
- 当前 `Diffusion` 的 loss group 元素为 `(start,end,weight)`；本任务下游仍使用完整动作 MSE，CordViP 三个独立 MSE 仅属于预训练。
- 仿真沿用 `eval.seeds`、milestone 选点和冻结 held-out 测试。当前仓库已无旧版 `tests/test_data_paths.py`、`test_evaluation.py`，不要列为必跑的现有测试。

## 2. 复现决策：显式保留论文与代码差异

默认保留官方运行代码的网络与损失；具体运行参数选择有来源的配置。以下差异须进入最终使用文档和实验记录。

| 项目 | 论文 / 其他官方材料 | 官方代码事实 | 本任务决策 |
| --- | --- | --- | --- |
| AFRO 帧间隔 | §4.4：k=4 | loss 硬编码 interval=2 | 默认 `frame_stride=2`；4 为显式消融，原始窗口仍为 33 |
| AFRO latent action | 论文 16；YAML la_dim=32 | IDM/DiT 实际均为 16，YAML 此项未消费 | 使用 16，不移植失效配置 |
| AFRO teacher | 论文描述 EMA target | 两个 encoder 分别随机初始化，deepcopy 被注释；optimizer 前更新 teacher | 保留初始化及训练更新相位，见 §5 |
| AFRO epoch / betas | 论文预训练 300、策略 100；§5.3 betas=(0.9,0.999) | YAML 301/101、betas=(0.95,0.999) | 预训练 301、betas 按代码；原生下游预算单独标注 |
| CordViP state embedding | 附录 B 写 16 | PointROEncoder 默认且实际为 64 | 两支均为 64 |
| CordViP 协调目标 | §III-C 描述 action sequences | 回归最后观测时刻的归一化 joint_state | 保留代码目标，不发明未来动作窗口 |
| CordViP diffusion target | §III-D Eq.7 描述预测噪声 | 发布 YAML 为 sample | 默认 x0；epsilon 只作为显式变体 |
| CordViP 执行长度 | 表 XI、README 实例为 H/O/A=12/4/6 | 下游 YAML 默认 A=8，README 参数覆盖为 6 | 默认 **12/4/6**；A=8 显式覆盖并同步 padding |
| CordViP 预训练权重 | workspace 维护 EMA | checkpoint_save 导出 raw obs_encoder，验证也调用 raw | 导出 raw；可省去不影响 loss/导出的预训练 EMA 副本并记录 |
| 原始 normalizer | 官方 dataset 对整个 replay buffer 拟合 | 不限 train split | 使用本项目 train-only 规则，不复制验证泄漏 |
| CordViP 动作标签 | 转换脚本读 file_num+2 的实测状态，边界判断却为 +1 | 不等于本项目实际命令 | 使用原生 action[t]，不移植转换逻辑 |

不建设 official/paper/native 多层 profile 框架；默认配置、少量显式覆盖及差异表足够。CordViP“未来动作协调预测”没有唯一可确定的发布实现，不纳入本任务。

## 3. 目录与复用边界

下表相对仓库根目录；补齐必要 `__init__.py`。可按实际代码量微调，保持职责不变。

| 文件 / 目录 | 职责 |
| --- | --- |
| `dexmani_policy/agents/policies/afro.py` | AFROPolicy + 本方法 observation assembly，独立继承 BasePolicy |
| `dexmani_policy/agents/policies/cordvip.py` | CordViPPolicy + 本方法 observation assembly，独立继承 BasePolicy |
| `dexmani_policy/agents/obs_encoder/interaction/cordvip_encoder.py` | 官方 PointROEncoder 的活跃 PointNet 路径，预训练/policy 共用 |
| `dexmani_policy/agents/obs_encoder/interaction/cordvip_transformer.py` | 直接移植官方 Transformer 及必要类 |
| `dexmani_policy/training/pretrain/obs/afro.py` | 预训练模型、IDM、VICReg、teacher、小训练循环、CLI |
| `dexmani_policy/training/pretrain/obs/afro_dit.py` | 官方 DiTFeaturePredictor 及必要类，仅供预训练 |
| `dexmani_policy/training/pretrain/obs/cordvip.py` | 官方 contact/coordination heads、loss、小训练循环、CLI |
| `dexmani_policy/training/pretrain/obs/contact_map.py` | 官方几何标签的离线计算，适配原生 Zarr |
| `dexmani_policy/datasets/obs/afro_dataset.py` | 只读点云序列，不依赖 action/state |
| `dexmani_policy/datasets/obs/cordvip_dataset.py` | 只读双点云、joint_state 和 contact 标签 |
| `dexmani_policy/configs/pretrain/obs/{afro,cordvip}.yaml` | 独立预训练配置 |
| `dexmani_policy/configs/{afro,cordvip}.yaml` | 原生策略配置，沿用训练/评测脚本 |
| `docs/观测预训练.md` | 数据要求、四步用法、来源/差异、恢复及内存调整 |

复用原则：

1. 复用 ReplayBuffer、SequenceSampler、split_episodes、validate_fields、LinearNormalizer.fit_field。新增两个小 dataset，不把 RobotDataset 扩成 action_key=None 的通用多模式框架，不伪造 action 来调用 PolicyTrainer。
2. AFRO 优先复用 `PointNet(3,64)` 与 `StateEncoder(19,64,(64,))`。CordViP 的两个 PointNet 输出 1024、权重独立；state projection 是单个 Linear，不用两层 StateEncoder。
3. 官方 PointNet 使用 `torch.max(x,dim=1)[0]`，现有组件是 amax。正向相同不保证 ties 时输入梯度相同。增加一个保留旧默认的最小 pooling 选项，使新方法使用官方 max；先做同权重输出/梯度对齐，不改其他方法默认行为。
4. 复用 ConditionalUNet1D、Diffusion、BasePolicy、PolicyTrainer、checkpoint、日志和 SimRunner。先核对 U-Net 活跃 FiLM 分支，不复制两份 U-Net。AFRO feature DiT 不得替换为现有 action DiT。
5. 两个 Policy 不互相导入，也不继承 DP3Policy；预训练不导入策略装配文件，推理不导入预训练 loss/Open3D。
6. 不整体搬运官方 workspace、驱动、FoundationPose、PointNet++/PointNext/DGCNN 或旧数据转换。可删除未触达分支及调试 import，保留活跃运算、初始化和参数布局。
7. 移植文件注明仓库、SHA、原路径、必要改动，保留适用版权/许可及上游归属；未发布许可不自行标 MIT。只在出现真实重复后提取小函数，不预建 trainer 基类、registry、factory 或多后端系统。

## 4. 数据契约与最小基础设施改动

### 4.1 原生信号

| 字段 | 使用方式 |
| --- | --- |
| joint_state[T,19] | 实测 rad；arm=:7，hand=7:19 |
| action[T,19] | 实际执行的绝对关节目标 rad，直接作为下游标签 |
| point_cloud[T,1024,6] | 相机可见场景，AFRO 取 XYZ |
| hand_pointcloud[T,1024,6] | 手部模型完整表面，CordViP 取 XYZ |
| object_pointcloud[T,1024,6] | 所有任务物体/部件共享 1024 点，CordViP 取 XYZ，不是每物体 1024 点 |
| meta/episode_ends | 累计排他结束位置，采样的 episode 边界 |

XYZ 是世界坐标、米；RGB 不参与首版。时序为 `obs[t]_before_action[t]`。H/O/A 分别是预测跨度、观测步数、执行步数；query 在观测窗口 O-1，不做额外 +1/+2 平移。`future_actions=prediction[:,O-1:,:19]`，runner 执行前 A 步；不能提前把返回值裁成 A。

保留数据真实 dt。目前仿真约 16 Hz，CordViP 原设置 5 Hz，同步数不等于同物理时长。首版不隐藏重采样/插值，记录 frame_stride*dt 和 A*dt。

### 4.2 采样、split、normalizer

- AFRO：原始 sequence_length=33、pad_before=0、pad_after=7；窗口内取 0,2,...,32，共 17 帧/16 transitions。不能把 sampler horizon 改为 17。IO 优化须保持相同 source rows/有效窗口集合；重复边界帧参与原损失，不擅加 mask。
- CordViP 预训练：原始跨度 12、padding=3/5，只消费前 4 帧；用 SequenceSampler 的 key_first_k 减少读取，不把跨度改为 4。contact 标签由同一个 sample_source_rows 对齐，取前 4 帧。
- 下游：AFRO H/O/A=16/2/8、padding=1/7；CordViP=12/4/6、padding=3/5。下游 pad_after=A-1 使用配置表达式；覆盖 A=8 后变为 7。
- 同源预训练/下游共享 train/val episodes；训练 seed 与划分 seed 分离。独立 play corpus 可有自己的 split，但不能与下游验证/held-out 数据重叠；没有跨源身份信息时明确无法自动证明无重叠。
- 两阶段默认划分建议 seed=42、val_ratio=0.1、max_train_episodes=null，这是本项目实验选择。数据不足时显式覆盖，不静默取消 holdout。已知 trial 沿用现有分组规则。
- normalizer 按 §1.1 使用全部真实训练帧，XYZ 按通道拟合，contact 保持原始 [0,1]。下游 fresh training 使用下游 train 统计，与官方迁移入口重设 normalizer 一致；预训练 affine 留作来源记录。续训/推理只恢复保存值。
- 默认关闭噪声、随机点 dropout、重采样和模态 dropout；现有 DP3 YAML 增强不是这两种方法的官方默认值。

### 4.3 多点云字段支持

当前只对 point_cloud 特判 pc_dim。用一个明确集合覆盖已存在的三种点云字段，不只在 encoder 末尾切片：

| 位置 | 必须处理 |
| --- | --- |
| datasets/common/robot_dataset.py | shape、consumed_fields、XYZ 截取和统计；仅检查消费列 |
| training/common/data.py | 三种点云 normalization_dims 使用实际 pc_dim=3 |
| common/config.py | normalization_dims_from_config 核对所选点云宽度 |
| agents/policies/base.py | 三种点云在 normalize 前取 XYZ，允许 runner 提供 XYZRGB |
| smoke_test.py | 三种点云的消费 shape，保留原入口 |

不扩展双点云随机增强。预训练可复用配置加载/affine，但不能调用强制 joint_state/action 的完整 policy config/normalization validator。

## 5. AFRO：必须保持的计算

来源：官方 [afro.py](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/policy/afro.py)、[DiT1D.py](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/model/transformer/DiT1D.py)、[pointnet_extractor.py](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/model/vision/pointnet_extractor.py)。

### 5.1 网络与目标

1. Online/teacher PointNet 各输出 D=64，独立随机初始化；teacher 不参与 autograd；使用同一归一化点云。
2. delta=f_online(t+k)-f_online(t)，IDM 为 64→256→128→16，中间 GELU；反向用相同 IDM 处理 -delta，不能直接取 -IDM(delta)。
3. FDM 输入加噪 teacher 目标，条件为 online 当前特征、latent action、diffusion timestep。反向交换时间并独立采样 noise/timestep。每个 transition 单次前向预测 clean feature，不跑多步 DDIM 训练循环。
4. 直接移植 DiTFeaturePredictor：hidden=256、depth=4、heads=4、单 feature token、learned position、官方 timestep embedding、AdaLN-Zero、输出 Linear。三路条件投影 concat 后进 MLP，不能改为相加。
5. 保留 Linear Xavier、AdaLN modulation 最后一层置零、position 初始化等。官方 pred_norm 未进入 forward，不启用；param_type 等未触达辅助逻辑不能引入 v-prediction 目标。

### 5.2 VICReg 与 teacher

统计轴 N=B*(M-1)、D=64，每个方向独立计算：

```text
invariance = mean((prediction - stopgrad(target))**2)
centered = feature - mean(feature, dim=0)
std = sqrt(var(centered, dim=0, unbiased=False) + 1e-4)
variance = 0.5 * (mean(relu(1-std_prediction)) + mean(relu(1-std_target)))
cov_matrix = centered.T @ centered / (N-1)
covariance = offdiag_square_sum(prediction)/D + offdiag_square_sum(target)/D
direction_loss = 25*invariance + 25*variance + covariance
loss = forward_loss + reverse_loss
```

保留官方 N<=1 的 covariance 行为，不省略 target 的 variance/covariance 标量，不改变 0.5 和 reduction。var/cov 默认没有额外 warmup。

Momentum：m(e)=0.996+0.004*clamp(e,0,300)/300，e 从 0 开始。将 teacher 状态改变移到显式训练循环：**loss → backward → 用尚未 step 的 online 更新 teacher → optimizer.step → scheduler.step**，默认 accumulation=1；与官方 optimizer 前更新同相位，同时 validation 不改变 teacher。恢复真实 epoch，不重置为 0。

AFRO 预训练默认 batch=128。VICReg 方差/协方差依赖每次 forward 的样本集合，小 microbatch+梯度累积并不等价于原 batch。首版只支持 accumulation=1；降低 batch 作为显式实验变化，不加跨 microbatch 特征缓存框架。

### 5.3 迁移

只导出/加载 online visual encoder，不把 teacher、IDM、FDM 带入策略。下游每帧 PointNet64 与 StateEncoder64 concat，再展平时间，O=2 时 condition=256。

默认只冻结 visual，state MLP/action decoder 训练；不能设 optimizer.obs_lr=0，它冻结整个 obs encoder。使用明确的 agent.freeze_visual，在构造时按保存配置设置以保持恢复时 optimizer groups 一致。Scratch 对照同时设置 agent.pretrained_path=null、agent.freeze_visual=false；fresh training 不得无权重却默认冻结随机 encoder。

## 6. CordViP：必须保持的计算

以下官方路径前缀记为 `C = train_policy/policy/CordViP/CordViP/CordViP/`：[PointROEncoder](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/model/vision/encoder.py)、[Transformer](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/model/vision/transformer.py)、[pretrain policy](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/policy/CordViP_pretrain.py)。

### 6.1 Encoder 与张量轴

- 两个独立 PointNet：逐帧 `[B,O,1024,3] → [B,O,1024]`。两个 Linear：hand12→64、arm7→64。原仓库 hand16 在前/arm6 在后；适配时明确原生 arm7 在前/hand12 在后，不能机械截断 22 维权重。
- 四个独立 Transformer：hand cloud↔object cloud、hand state↔arm state。每个保留一层 encoder、一层 decoder、4 heads、FFN=1024、外部 residual、无时间位置编码。state Transformer 的 FFN 也是 1024，不按 embedding dim 猜测缩小。
- Attention 轴是全局特征的 **O 个观测时刻**，不是 1024 个点。官方 Transformer(src,tgt) 内部执行 model(tgt,src,None,None)，方向、self-attention/cross-attention 一起保留。
- 官方自定义 LayerNorm 用 x.std(-1) 的样本标准差和 std+eps；不同于 PyTorch LayerNorm 的总体方差与 sqrt(var+eps)。直接移植，不换成通用 Transformer/HandSceneEncoder。
- BasePolicy.preprocess 会 flatten B/O；policy assembly 进入共享 CordViP encoder 前恢复 `[B,O,...]`，明确 O 并检查可整除；预训练直接传序列。不能让 attention 跨 batch。
- 下游按每个时间步 concat `[hand_pc,object_pc,hand_state,arm_state]`，再 flatten 时间，condition=`4*(1024+1024+64+64)=8704`。这与协调头的 modality-first flatten 顺序不同。

### 6.2 三个预训练头

直接移植 C/policy/CordViP_pretrain.py 与 C/model/vision/create_fc.py 的活跃 MLP：

| 头 | 输入 | 网络 | 监督 |
| --- | --- | --- | --- |
| contact | 每帧 concat(hand_pc,object_pc)，2048 | 2048→2048→1024→1024→1024，隐藏层 ReLU，最后 Linear | [B,4,1024] contact |
| arm | flatten 全部 hand_pc，再 flatten object_pc，再 flatten fused hand_state，concat 后 8448 | 8448→512→256→64→7 | normalized joint_state[:,3,:7] |
| hand | 同上，但第三项为 fused arm_state | 8448→512→256→64→12 | normalized joint_state[:,3,7:19] |

contact head 没有末层 sigmoid。损失为 `mean_MSE(contact)+mean_MSE(arm)+mean_MSE(hand)`，权重默认均为 1；手/臂分别平均后相加，不合为一个 19 维 MSE。可保留三个明确 loss weight 做论文消融，不加 loss 注册系统。

预训练不需要 action：原代码 normalize joint_action 只为取 B，改从 obs 取 B。contact 只是 target，不能进入 encoder/policy observation fields。下游加载完整 raw obs encoder 并默认全部 finetune，不保留三个头。

### 6.3 离线 contact map

直接移植官方 [contact_map.py](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/script/contact_map.py) 的数学计算，改写 IO 和明显 CLI 错误：

```text
object_normals = Open3D estimate_normals(KNN=30)
u_ij = (hand_j-object_i) / (norm(hand_j-object_i)+1e-8)
D_i = min_j norm(hand_j-object_i) * exp(1*(1-abs(dot(u_ij,normal_i))))
C_i = 1 - 2*(sigmoid(10*D_i)-0.5)
```

采用运行调用中的 gamma=1、theta=10，而非 helper 默认 0.2/5。用**未归一化的世界坐标米制 XYZ**；在完整 object cloud 上估计 normals，保持官方语义；分物体单独估计 normals 是另一个变体。min 遍历全部 hand points，不能先找欧氏最近点再加权。

Open3D 只在离线工具延迟导入。可按 object 点分块做 NumPy 向量化，先与原逐点函数对齐；不分配全数据的 [T,Nobj,Nhand,3]。不在 DataLoader/训练循环重复估计 normals 或计算距离。原脚本缺 argparse import、调用参数不匹配，不能照搬这些入口错误。

输出独立 Zarr sidecar：仅含 data/contact_map[T,1024]、meta/episode_ends 和生成元信息；不修改原数据、不复制全部点云。复用 ReplayBuffer 的 worker-local 句柄。元信息至少绑定来源身份（存在时的 source/episode/trial）、行数、episode 边界、几何/点序配方、gamma/theta/KNN、upstream SHA。默认拒绝覆盖，完整写完后发布输出，失败文件不可作为有效标签。

contact 向量按固定点槽位监督，而 PointNet 是 permutation-invariant：**即使同时重排点云和标签，随机改变目标点顺序也会改变监督任务**。禁止随机 shuffle/FPS/dropout 破坏 object point 顺序，保持模板槽位与标签逐行一致。读取 observation_geometry_json 核对 task roles、parts、点数/模板布局，不能只检查 shape 一样。

首个闭环优先固定模板/物体实例；多模型、缩放或部件点数变化时，说明点槽位语义是否变化，作为泛化条件记录。不要为消除该局限偷偷换成 point-wise decoder。仿真完整几何是输入优势，不把相对相机点云 baseline 的全部提升归因于预训练。

## 7. 初始化、导出与恢复

### 7.1 下游生命周期

1. 构造只创建确定结构，不访问磁盘、不猜权重路径。
2. initialize_training(data_description) 仅在 fresh training 加载 agent.pretrained_path；严格核对方法、结构、XYZ/feature dim，CordViP 还核对 19 维布局及点云配方。缺文件/key 不匹配显式报错，不打印后继续。
3. 然后现有流程加载下游 normalizer、创建 EMA/optimizer。CordViP EMA 必须复制已加载预训练权重的模型；不复制官方 workspace 先建 EMA、后只为 raw 加载 encoder 的顺序。
4. prepare_restore_config 只在私有构造配置中清空 pretrained_path，保留架构/freeze 语义；完整 checkpoint strict restore。将 agent.pretrained_path 加入 training/common/resume.py 的 ARTIFACT_LOCATION_FIELDS，使清空/移动初始化路径不改变恢复配方。
5. Scratch 是明确选择：AFRO 同时解除 freeze；CordViP null 路径表示 scratch 时日志须明确。缺失文件不能当作 null；不做 auto-search 或任意官方大 checkpoint 兼容层。

配置声明 pretrained_path:null，默认训练示例明确传 encoder.pt。AFRO freeze_visual:true 的 fresh 缺权重检查放在 initialize_training，不能阻止完整 checkpoint 构造/恢复。

### 7.2 预训练产物

| 产物 | 保存内容 | 消费者 |
| --- | --- | --- |
| encoder.pt | CPU raw encoder state、方法/结构/输入配方、关节布局（若有）、upstream SHA、pretrain affine、实际 split、步数和来源 | 下游 fresh initialization：只加载 encoder weights，metadata 用于核对 |
| resume.pt | 完整预训练模型（AFRO 含 teacher/IDM/FDM，CordViP 含 heads）、optimizer、scheduler、normalizer、epoch/global_step、RNG、实际配置/split | 对应预训练入口恢复 |

复用原子写入、RNG/配置保存工具；不强塞 policy checkpoint 的 action metadata，不建通用 artifact 层。encoder.pt 是支持 weights_only=True 的 tensor/dict payload。保存真实 affine，不假设当前 input_stats 自动进入 state_dict。

下游完整 checkpoint 包含全部 encoder 权重和 normalizer。删除参考 clone、预训练目录、contact sidecar 和训练数据后，仍可对提供的 obs 恢复/推理；训练续跑仍需训练数据。requires_grad 不属于 state_dict，freeze 必须从保存构造配置恢复。

预训练入口支持 --resume-from，采用明确的 epoch 边界恢复语义。中断/短跑保存须说明是否从本 epoch 开头重读，不承诺逐 batch 位级复放；AFRO momentum 使用恢复 epoch。下游保持既有 resume 语义。

首版仿真走 restore_policy_agent→SimRunner。当前 deployment.inspect_policy/load_policy 对 hand cloud 依赖 real_runtime，warmup 也缺 object cloud 分支；不要宣称已经支持 CordViP 真机部署。真实物体模型/位姿跟踪及完整公开部署 API 属于后续任务。

## 8. 默认配置与性能约束

### 8.1 算法默认值

| 参数 | AFRO | CordViP |
| --- | --- | --- |
| 观测 | scene XYZ + joint19；预训练仅 scene | hand/object XYZ + joint19 |
| 每种点云 | 原生 1024；不从未消费的 npoints 字段推断论文输入 | 1024 |
| 下游 H/O/A | 16/2/8 | 12/4/6 |
| visual / state dim | 64 / 64 | 每支 1024 / 每支 64 |
| U-Net widths | [512,1024,2048] | [512,1024,2048] |
| diffusion timestep embedding | 128 | 64 |
| kernel / groups / FiLM | 5 / 8 / scale+bias | 相同 |
| diffusion | squaredcos_cap_v2；100 train timesteps；sample target；10 DDIM steps；clip_sample=true | 相同 |
| 下游 encoder | visual frozen，state trainable | 全部 finetune |
| 下游 EMA | false；eval.use_ema=false | true；eval.use_ema=true |
| AdamW | lr=1e-4，betas=(0.95,0.999)，eps=1e-8，wd=1e-6 | 相同 |
| batch（源配置起点） | 128 | 256 |
| LR warmup（optimizer updates） | pretrain=5，policy=0 | pretrain/policy=500 |

原代码 AdamW 对可训练参数统一 weight decay，本项目 BasePolicy 默认排除 bias/norm。新 Policy 可用短 optimizer_groups override 返回 decoder/obs 两组可训练参数，仍复用 build_optimizer，保留官方 decay 语义；不修改其他方法。默认不加原代码没有的 gradient clipping，现有 trainer 用 max_grad_norm=0 关闭。

预算分开声明：AFRO 预训练默认 301 epochs，按实际 loader 长度计算 scheduler 总 updates；CordViP 预训练及两种下游默认使用本项目 **100000 optimizer updates** 作为原生实验起点。后者是工程预算，不是论文值。保存 loader 长度、样本数、更新数/epoch。复查上游预算时按对应 YAML/启动命令的 epochs 展开 scheduler；CordViP 的 YAML 30000 epochs、README 预训练示例 10000、加载第 2000 epoch 的权重不能当成同一配方。

### 8.2 先保持语义，再优化

- 基准核对使用 FP32/eager、固定 seed/noise/timestep；通过后才显式开 BF16/compile。VICReg 统计、teacher EMA、contact 几何保持 FP32 或更高精度。
- 不默认缩小官方 feature dim、U-Net widths、token 数/目标窗口。CordViP 可降低 microbatch 并按现有规则累积 MSE 梯度，记录实际 batch/尾组；AFRO 预训练不具有该等价性。
- 已有 1024 点时不再调 FPS/PyTorch3D；保持点序，减少无效 kernel/依赖。
- 按 Zarr chunk 读取消费字段；Cord 预训练不读 action，AFRO 不读 state/action；contact 离线生成一次。
- 指标到日志周期再转 CPU，不在每个 loss term 中 item() 同步；仅记录有用分项和必要耗时/显存。
- --max-updates 2 / 下游 +max_updates=2 只是执行上限，不能把总预算改成 2，否则 scheduler/EMA 语义改变。
- 更新 pyproject.toml package-data，包含 configs/pretrain/obs/*.yaml；不能只在源码目录运行。Open3D 为标签工具可选依赖，普通训练/推理 import 不要求它。

## 9. 用户入口：实施后必须直接可用

两个预训练入口统一支持 --config、--output-dir、--resume-from、--max-updates 和配置 overrides。复用 load_policy_config 的组合能力，不复用完整 policy 校验。默认输出 `experiments/pretrain/obs/<method>/<task>/<run>/`，保存 resolved config、日志和 §7.2 两种权重；不隐式覆盖旧运行。

下列是目标用法，实施后用实际数据验证，写入 docs/观测预训练.md。新 YAML 均显式声明 dataset.split_manifest:null；使用共享 manifest 时直接覆盖，不用 +。以下 TASK_NAME 仅是示例任务，不能假设执行环境已有其数据。

```bash
conda activate policy
TASK_NAME=pick_apple_messy

# 可选：同源两阶段只生成一次共享划分，后续 config/checkpoint 内嵌内容。
python -m dexmani_policy.datasets.common.split \
  --zarr-path "robot_data/${TASK_NAME}.zarr" \
  --output "tmp/${TASK_NAME}_split.json" --seed 42 --val-ratio 0.1

# AFRO：仅观测预训练，不要求数据中存在 action/joint_state。
python -m dexmani_policy.training.pretrain.obs.afro \
  --config pretrain/obs/afro \
  --output-dir "experiments/pretrain/obs/afro/${TASK_NAME}/run01" \
  task_name="$TASK_NAME" dataset.split_manifest="tmp/${TASK_NAME}_split.json"

# CordViP：离线标签→预训练。
python -m dexmani_policy.training.pretrain.obs.contact_map \
  --zarr-path "robot_data/${TASK_NAME}.zarr" \
  --output "robot_data/${TASK_NAME}.contact.zarr"
python -m dexmani_policy.training.pretrain.obs.cordvip \
  --config pretrain/obs/cordvip \
  --output-dir "experiments/pretrain/obs/cordvip/${TASK_NAME}/run01" \
  task_name="$TASK_NAME" \
  dataset.contact_map_path="robot_data/${TASK_NAME}.contact.zarr" \
  dataset.split_manifest="tmp/${TASK_NAME}_split.json"

# 下游复用原脚本，只换配置和初始化权重。
bash scripts/training/train.sh afro task_name="$TASK_NAME" \
  agent.pretrained_path="experiments/pretrain/obs/afro/${TASK_NAME}/run01/encoder.pt" \
  dataset.split_manifest="tmp/${TASK_NAME}_split.json"
bash scripts/training/train.sh cordvip task_name="$TASK_NAME" \
  agent.pretrained_path="experiments/pretrain/obs/cordvip/${TASK_NAME}/run01/encoder.pt" \
  dataset.split_manifest="tmp/${TASK_NAME}_split.json"

# 完整预算训练后选点，并执行冻结的 held-out 测试。
bash scripts/eval/eval_pipeline.sh afro "$TASK_NAME" RUN_NAME
bash scripts/eval/eval_pipeline.sh cordvip "$TASK_NAME" RUN_NAME
```

RUN_NAME 替换为实际下游运行目录名。短跑使用独立 smoke 输出目录并追加 --max-updates 2 / +max_updates=2；短跑 checkpoint 走普通 evaluate，不能伪造完整 milestone 集合调用正式 pipeline。最终使用文档还须给出两种预训练 --resume-from 和下游 +resume_from 的实际完整命令。

Scratch 对照保持模型、输入、数据与预算相同，只改变初始化；AFRO 同时 freeze_visual=false。不要仅与默认 RGB/更窄 U-Net 的 DP3 比较。至少保留 AFRO scratch vs pretrained-frozen、CordViP scratch vs pretrained-finetuned；若研究冻结策略的独立作用，另列同冻结/finetune 条件的消融。相机点云与完整模型几何的输入差异独立报告。

## 10. 实施顺序与验收

按阶段持续推进，不因完成一个阶段自动结束。无需启动完整昂贵训练证明接口打通。

### P0：版本与数据接口

- 记录当前本项目/上游 SHA，阅读实际调用链，只处理相关版本漂移。
- 完成最小多点云支持与两个 observation-only dataset，核对 split、全真实训练帧统计、padding/source rows/XYZ 列。
- 配置能组合、模块能 import、安装产物包含嵌套 YAML；未安装 Open3D、官方仓库不在 PYTHONPATH 时，策略仍能 import。

### P1：AFRO 完整链路

- 直接移植并核对 DiT、IDM、VICReg、teacher，导出 online visual。
- 加入 AFROPolicy，检查仅冻结 visual、state 可训练、condition/action window 正确。
- 完成有界更新、保存/恢复、离线动作推理；有 simulator 时跑最小仿真回合。

### P2：CordViP 完整链路

- contact 与官方函数对齐，写 sidecar，核对模板点序/source rows。
- 移植 encoder/Transformer/heads，先在原始 22 维结构上做参考 parity，再核对原生 arm7+hand12。22 维只用于参考验证，不建设通用机器人框架。
- 打通预训练、raw encoder 导出、finetune、EMA 初始化、恢复和仿真。

### P3：必要验证与收敛

| 验收组 | 防止的错误 | 最低证据 |
| --- | --- | --- |
| 官方数值对齐 | 网络、concat、LayerNorm、loss reduction、teacher 相位 | 同权重/输入、固定 noise/time；比较输出、分项 loss、输入/参数梯度、一次 teacher 更新；记录 dtype、容差和最大误差，不只看 shape |
| 数据与几何 | 跨 episode、off-by-one、伪 action 依赖、RGB 混入、验证泄漏、contact 错位 | 小 Zarr fixture 含边界 padding、验证极值、不同 point slots；AFRO 源仅 point_cloud、Cord 源无 action 也能训练 |
| 生命周期 | 冻错模块、没加载预训练、EMA 未同步、恢复时重拟合 | 有界更新、参数分组、AFRO visual 不变/训练支路有梯度、Cord raw/EMA 初始一致、恢复 scheduler/epoch/split/affine |
| 推理与评测 | 依赖标签/外部初始化文件、动作顺序/长度错、选点污染 | 移除 reference/pretrain 文件后 strict restore；future_actions[B,H-O+1,19] 为有限物理 rad；固定 RNG 推理可比；可用时 SimRunner 回合 |

AdaLN-Zero 会使部分可训练参数首次 update 的梯度为零，不要求每个 trainable tensor 第一轮都变化；检查预期梯度路径和短段更新。contact 验证包含“欧氏最近点与加权距离最小点不同”的例子，防止错误近邻优化。PointNet ties 和 CordViP batch/time 混淆须有针对性例子。

优先扩展现有 smoke；只为上述真实风险增加少量测试/参考核对脚本，不建大测试框架、不恢复无关已删测试。正式实验沿用当前 milestone、selection/test seeds、统计并保留 matched scratch；短跑 loss 下降不等于论文性能复现。

交付：两种可训练/恢复/推理 Policy、两个预训练入口、contact 工具、四份配置、简明使用文档，以及**实际执行命令、结果、数值误差、未验证项**。检查最终 diff，仅提交相关文件；不提交参考 clone、数据、标签、checkpoint、训练输出。

缺少 Torch/CUDA、真实数据、Open3D、dexmani_sim 时，继续不依赖它们的工作；列出具体缺项、影响的验收和可执行补验命令。不得把静态检查、synthetic smoke 或未执行测试写成真实训练/闭环通过。

## 11. 补充源码索引

路径对应 §1 固定版本；本项目行相对于 dexmani_policy/ 包目录，AFRO/CordViP 行相对于各自仓库根目录（C 前缀见 §6）。按实际符号/调用链核对，不只抄配置名。

| 仓库 | 路径 | 核对内容 |
| --- | --- | --- |
| AFRO | AFRO_Pretrain.py | epoch、5 update warmup、teacher/optimizer 相位、online 权重来源 |
| AFRO | afro_workspace/config/afro.yaml、afro_policy.yaml、选中的 config/task/*.yaml | 覆盖关系、H/O/A、padding、batch/epochs/EMA |
| AFRO | afro_workspace/policy/afro_policy.py | visual 权重筛选、freeze、visual/state concat |
| AFRO | afro_workspace/model/diffusion/conditional_unet1d.py | 活跃 FiLM 与本地 U-Net 同权重对应 |
| CordViP | C/model/vision/create_fc.py | ReLU/Linear，无末层 sigmoid |
| CordViP | C/policy/CordViP_train.py | 每时间步拼接、flatten、diffusion target、action slice |
| CordViP | C/config/CordViP_pretrain.yaml、CordViP_train.yaml、task/all_tasks_160.yaml | task override；默认 task 名不一定对应发布文件 |
| CordViP | train_policy/policy/CordViP/CordViP/pretrain.py:checkpoint_save | raw obs encoder 导出，不是 EMA |
| CordViP | train_policy/policy/CordViP/CordViP/load_pretrain.py | 下游 normalizer、EMA 构造顺序差异 |
| CordViP | train_policy/policy/CordViP/scripts/{pretrain_policy,load_pretrain_policy}.sh、README | epochs、A=6 等启动覆盖 |
| 本项目 | agents/policies/base.py、training/trainer.py | preprocess、initialize/restore、freeze、EMA、optimizer、future_actions |
| 本项目 | datasets/common/{robot_dataset,sampler,replay_buffer,split,validation}.py | 消费字段、窗口、IO、split、统计 |
| 本项目 | common/checkpoint.py、training/common/{config,data,resume,optimization}.py | 恢复 hook、normalization dims、白名单 |
| 本项目 | evaluation/{config,rollout,selection}.py、env_runner/sim_runner.py | restore_policy_agent 仿真链路及选点/测试隔离 |

本任务书已完成论文与固定版本代码的静态核对，未实施模型移植、模型数值对齐或训练。执行本任务的 Codex 须据实填写验收结果。
