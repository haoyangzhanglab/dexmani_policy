# Codex 任务书：基于 Diffusion Policy 精简 dexmani_policy 数据层

> 状态：最终设计与实施要求，尚未实施代码重构。  
> 目标仓库：`haoyangzhanglab/dexmani_policy`。  
> 审查日期：2026-10-27。  
> 已审查代码基线：`4cd6275ecf65e8a6f281aac2b70352eb3609da9e`。  
> 本任务书只规定数据层及其真实调用方的必要修改，不授权扩大为模型算法、评测协议或其他仓库的重写。

## 0. 执行目标与优先级

请直接实施本任务书，而不只提交分析或新的方案。先阅读仓库 `AGENTS.md`、本任务书和实际调用链，保护工作区已有修改；制定简短实施计划后，完成代码、相关配置、必要文档和针对性验证。

核心目标：让 `dexmani_policy` 的普通数据路径采用官方 Diffusion Policy 的常规做法，基础模块干净、直接、易读、易适配。删除收益不足的公共机制，把算法确实需要的差异留在少量具体 Dataset 或函数中。

优先级依次为：

1. 观测、动作、时间和数值语义正确。
2. 训练、手部预训练、checkpoint 恢复和部署调用链完整。
3. 减少公共机制、重复状态和配置来源。
4. 保留已有的低成本 I/O 优化，针对实际瓶颈优化。
5. 接口与命名直观；不为形式统一扩大修改范围。

本任务的成功标准不是类更少、文件更少或行数达到某个阈值，而是：新增普通 diffusion/flow policy 通常不需要修改 ReplayBuffer 和 SequenceSampler；新增真正不同的训练样本只需要少量局部代码。

若当前代码相对上述基线已经变化，先核对差异并按实际调用关系调整实施位置；不得强制回退仓库。任务书中的“当前行为”绑定已审查基线，目标要求优先于过时的路径或行号。影响目标或算法定义的实质冲突需要明确报告，普通实现选择自行完成。

## 1. 最终审查后的设计调整

本节覆盖此前讨论中容易被机械执行的建议。以下决定为最终要求。

| 讨论中的建议 | 最终决定 | 理由 |
| --- | --- | --- |
| 全面引入 DP 的 `shape_meta` | 本轮不强制引入，默认保留现有明确的字段配置 | 不能在 `sensor_modalities`、模型维度之外再增加一份事实来源 |
| 将所有命名改为与 DP 一致 | 优先保留有实际消费者的配置键；不做纯命名迁移 | `obs_horizon` 与 policy 的 `n_obs_steps` 语义一致即可，避免增加兼容别名 |
| 删除 finite-window filtering 后完全信任数据 | 训练装配边界进行一次按需、分块、遇错即停的数值检查 | 只改 policy 时，不能假定所有历史 sim/real Zarr 已离线检查 |
| normalizer 仍按窗口覆盖的唯一源帧统计 | 普通字段改为选定 train episodes 的全部真实帧 | 删除窗口与统计总体的耦合；这是明确的统计协议变更 |
| 直接换成官方 Normalizer 实现 | 优先保留现有数学、affine 参数形状与 state_dict 键，仅统一职责和接口 | 避免破坏已保存 checkpoint、手部 codebook 和部署 |
| RGB 按官方示例统一映射到 `[-1,1]` | 保留当前 `rgb: identity`、uint8 路径和 ImageProcessor 契约 | 现有视觉骨干内部已有 `/255` 与 mean/std，不能重复归一化 |
| 删除 Dataset 中全部 metadata | 从公共 Dataset 迁出；保留实际被训练恢复、手部产物和部署消费的最小信息 | 删除 producer 而不更新 consumer 会造成完整链路失效 |
| 为七个参考项目预建适配层 | 本次不新增这些方法的实现 | 参考项目用于约束设计，不构成七个新功能需求 |
| 保证恢复训练逐 batch 与不中断训练相同 | 保持当前 epoch 起点恢复语义 | 当前没有 minibatch cursor/prefetch 状态，不应顺便发明精确重放机制 |
| 同时调 chunk、缓存和 DataLoader | 核心重构先完成；只有实测瓶颈才做额外优化 | 不把未经测量的配置改变称为性能提升 |

## 2. 官方参考与证据边界

### 2.1 固定版本

下列引用是本次设计使用的代码快照，不声称它们始终是未来最新版本。

| 项目 | 已审查提交 | 本任务借鉴内容 |
| --- | --- | --- |
| [Diffusion Policy][dp_repo] | `5ba07ac6661db573af695b419a7947ecb704690f` | Zarr episodes、ReplayBuffer、SequenceSampler、Dataset 与 normalizer 接口、动作时间对齐 |
| [ManiFlow Policy][maniflow_repo] | `ef2f116f1f90163ed36e657b8c5503740bb468af` | 普通状态/点云/动作序列可复用同一种数据骨架 |
| [RDT-1B][rdt_repo] | `cd79363a1387e8f81c7724d070ef7e45fd23150f` | 机器人表示、图像历史、语言等差异应在方法适配处处理 |
| [CordViP][cordvip_repo] | `ad6d441abe7b0480cf79b0311db266965c1313b6` | 几何与接触等重处理离线完成，训练 Dataset 保持简短 |
| [AFRO][afro_repo] | `283024a49e2c14bc030debf15383a4e7ee73836d` | action-free 的长时间视觉预训练需要自己的明确样本定义 |
| [MoPA：用户指定的 XPolicyLab 路径][mopa_repo] | `c43a74653bec480ebe82c84770664e8035bc56c9` | 同一动作序列可供多个子系统预测头使用；维度划分不应污染 sampler |
| [TransDex][transdex_repo] | `a4151879701bfb85a45447fa6d99fea07dd8c7dc` | 预训练输入/监督目标分开，空间变换成对一致，重几何处理与训练读取分工 |

配套真机仓库仅作为输入/部署契约参考：`haoyangzhanglab/dexmani_real@533b3ea5f36d81734ecfa56327866c1bcdaa96f7`。本任务默认不修改它。

### 2.2 参考实现不能直接照搬的部分

- DP 的官方接口名称是 `BaseLowdimDataset` / `BaseImageDataset`，并非必须新建一个自定义统一 BaseDataset。直接实现所需方法即可。[接口][dp_base]
- DP 的 `RealPushTImageDataset.get_normalizer()` 对整个 replay buffer 拟合；本任务采用 train-only 统计，是主动选择的简单修正，不能写成官方所有 Dataset 原本如此。[官方统计][dp_dataset]
- 官方 `key_first_k` 有未读区占位实现；借鉴其少读观测的语义，不照抄整型 NaN 占位或调试器异常分支。[官方 sampler][dp_sampler]
- ManiFlow 的某些公开 Dataset 全量读入数组，并读取未实际返回的字段；不把这些实现细节当成效率标准。[ManiFlow Dataset][maniflow_dataset]
- RDT 的多机器人向量空间、语言/图像适配、预训练数据生产缓存不是本仓库普通单机器人数据层的默认需求。[RDT Dataset][rdt_dataset]
- CordViP 的公开转换脚本存在以未来 measured joints 作为目标的局部语义，不可带入本仓库的 commanded action 定义。[CordViP 转换][cordvip_convert]
- AFRO 活跃预训练配置使用普通 AdroitDataset；不能因仓库里存在 only-pointcloud 文件就称其为当前入口。模型预训练消费点云时间特征，不能据此把其内部 latent action 当作 XHand 动作。[AFRO 配置][afro_task]、[目标][afro_loss]
- MoPA 的动作子系统划分位于模型，不能据此创建两套 episode/sampler；其归一化与训练划分也不自动替代本仓库协议。[MoPA Dataset][mopa_dataset]、[模型划分][mopa_model]
- TransDex 的变量名 `rel_actions` 不足以证明是相对命令；其未来 measured joints、在线 FK、颜色语义标签和重叠窗口随机划分不应直接复制。[TransDex 数据读取][transdex_dataset]
- 本仓库的 R3D 与参考 RDT-1B 是不同方法，不得因名称相近改变 R3D 行为。

## 3. 当前代码路径与必须覆盖的消费者

当前普通训练主链为：配置解析 → `build_dataset_and_normalizer` → Dataset/ReplayBuffer/Sampler → Policy → normalizer 绑定 → effective config → logger/checkpoint → resume 或 deployment。

先以 `rg` 查阅实际引用，再修改。下面是最低查阅范围，不要求逐文件无差别通读。

| 范围 | 相关文件 | 必须核对的问题 |
| --- | --- | --- |
| 数据基础 | `dexmani_policy/datasets/common/{replay_buffer,sampler,robot_dataset,split,metadata,augmentation}.py` | 读取、过滤、窗口、增强、split 与 metadata 的耦合 |
| 数值与输入 | `dexmani_policy/common/{normalizer,config,image_util,hand_representation}.py` | feature axes、字段映射、RGB、动作表示 |
| 数据装配 | `dexmani_policy/training/common/{data,config,logging,resume}.py` | `data_recipe` 的生产、保存、比较和恢复 |
| 主训练 | `dexmani_policy/training/trainer.py` | 模型初始化、normalizer、checkpoint、epoch 恢复语义 |
| 入口与产物 | `dexmani_policy/train.py`、`dexmani_policy/common/checkpoint.py` | 装配返回值、strict state restore、权重精度与初始化路径 |
| 具体 Dataset | `dexmani_policy/datasets/hand_prior/{hand_chunk_dataset,lamp_dataset,latent_flow_dataset,metadata}.py` | 未来窗口、严格过去历史、reference 与相对统计 |
| 手部预训练 | `dexmani_policy/training/pretrain/hand_prior/{hand_vae,vqvae,lamp,codebook,export_codebook}.py` | 对 sampler source rows、split、normalizer 和产物 metadata 的直接依赖 |
| 手部产物加载 | `dexmani_policy/agents/hand_prior/{hand_vae,vqvae,motion_prior,codebook}.py` | 结构/数据语义兼容、split 防泄漏、affine 一致性 |
| 具体策略 | `dexmani_policy/agents/policies/{base,dqrise,lamp,latent_flow,r3d}.py` | action 起点、auxiliary target、artifact 初始化与恢复 |
| 图像处理 | `dexmani_policy/agents/obs_encoder/rgb/image_processor.py` | float/uint8 范围、mean/std、空间变换 |
| 部署 | `dexmani_policy/deployment/{metadata,runtime}.py` | `real_runtime`、`policy_metadata`、`normalization_dims`、`inference_recipe` |
| 评测配置 | `dexmani_policy/evaluation/config.py` | 保存的 sensors/RGB 预处理和 action_start_index；保持外部键时可能只需核对 |
| 配置/检查 | `dexmani_policy/configs/*.yaml`、`dexmani_policy/smoke_test.py` | 公开参数及 shape/dtype 断言同步 |
| 测试/文档 | `tests/test_data_paths.py`、`tests/test_evaluation.py`、`README.md`、相关 `docs/` | 旧协议断言更新，新的行为和限制说明 |

特别容易漏掉的真实消费者：

1. `ExperimentLogger.save_config()` 当前遍历 `data_recipe` 写出 split 文件；必须一起更新。
2. `deployment/runtime.py::inference_recipe()` 比较预处理、normalization dims、policy metadata；不能只改 `deployment/metadata.py`。
3. VQ/LAMP 预训练当前直接使用 `sampler.action_source_rows`；删除公共覆盖集合后必须替换这些读取。
4. DQ-RISE 的 codebook 与 policy 必须保持 hand affine 的数值兼容检查。
5. `smoke_test.py` 根据 Dataset 属性推导 RGB/点云输出、验证集和 `action_context`，必须同步。

上述事实来自 [当前数据装配][current_data]、[日志][current_logging]、[恢复][current_resume]、[部署][current_runtime]、[VQ 预训练][current_vq] 和 [码本][current_codebook]。

## 4. 修改范围与明确不做的事情

### 4.1 必须完成

- 恢复 DP 式的纯窗口 sampler。
- 从公共路径删除动态坏窗口过滤与窗口覆盖源帧统计。
- 用一次数据装配边界检查代替静默过滤。
- 普通 normalizer 改为 train episodes 全部真实帧统计。
- Dataset 提供清楚的 `get_normalizer()` / `get_validation_dataset()` 与 `obs/action` 接口。
- 减少 Dataset 对训练产物、来源证明和部署的职责。
- 更新所有实际消费者、配置与必要文档。
- 保持已有手部算法、RGB 预处理、checkpoint 和部署数值契约。
- 用小型真实 Zarr fixture 完成有意义的验证。

### 4.2 本次不做

- 不新增 RDT、AFRO、MoPA、CordViP 或 TransDex 策略、预训练流程或数据下载器。
- 不替换现有 Zarr v2，不升级到 Zarr v3、LeRobot、RLDS 或新存储格式。
- 不改 `dexmani_real` / `dexmani_sim` 的录制器、同步、控制、数据导出和 chunk 布局。
- 不引入 Dataset/Backend/Transform/Role 的 registry、manager、工厂层或配置语言。
- 不新增缓存证书、validity sidecar、自动内存探测、多级 LRU 或 learned-feature cache。
- 不改变网络、loss、求解器、动作表示、augmentation 分布、实验默认超参数或评测选点协议。
- 不把本次扩展为主 trainer 新增 validation-loss 调度的功能。Dataset 的验证视图和已有预训练验证必须正确；不能把“存在 val split”描述成主循环已计算 val loss。
- 不通过重写测试为旧的复杂机制继续保留存在理由。

## 5. 数据契约与职责边界

### 5.1 存储

保留普通 DP 形态：`data/<field>` 的首维是总时间 T，`meta/episode_ends` 是 E 个 episode 的累计结束位置。

| 字段 | 示例形状 | 类型/含义 |
| --- | --- | --- |
| `joint_state` | `[T,Dq]` | 实测状态；当前任务常为 19 维 |
| `action` | `[T,Da]` | 明确定义的监督动作；当前 joint command 常为 19 维 |
| `action_ee` | `[T,21]` | 当前末端目标表示，具体分量语义保留 |
| `rgb` | `[T,H,W,3]` | `uint8`，HWC 存储 |
| `point_cloud` | `[T,N,C]` | `float32`，消费 XYZ 或 XYZRGB 由现有配置明确 |
| 其他观测 | `[T,...]` | 只要求本次实验真正使用的字段 |
| `episode_ends` | `[E]` | 严格递增整数，每个边界结束一个 episode |

基础层不硬编码 xArm7/XHand 的 19 维、手部 12 维或某个 task 名称。机器人语义存在于具体任务/动作适配处。

保留廉价结构检查：字段存在、首维一致、episode 边界合法、实际消费 shape/dtype 合法。不要求一个普通 Zarr 必须携带 `dexmani.real.canonical` 标签才能训练。

### 5.2 动作、时间与物理语义

- 基础 Dataset 不将 `action[t]` 替换为 next measured state，不引入 +1/+2 的隐藏偏移。
- 当前真机 action 是最终绝对目标的提交语义，不表示下一帧已经到达；技术失败 Raw 的准入由生产端处理。
- `action_ee` 来自目标 arm FK，`eef_pose` 来自实测 arm FK；不可互换。
- 真机触觉的 SDK 原始单位/传感器轴不等于仿真的 N/world；Normalizer 不负责猜测或修复物理定义。
- 本仓库检查数组可用，不在 Dataset 中执行传感同步、时间重采样、FK、点云重建、物体跟踪或接触估计。
- 不删除坏帧后将剩余行压紧，不在训练时自动切分或修补 episode。

真实物理语义参考 [Raw schema][real_schema] 与 [Canonical processing][real_processing]。本任务不需要把它们复制成一套新的通用 metadata 协议。

## 6. T1：保留并收紧 ReplayBuffer

当前只读 ReplayBuffer 很小，优先原地保留，不为“使用官方”搬入整套写入、视频转换或机器人依赖。[当前实现][current_buffer]

要求：

- 只打开/暴露本次使用的 keys。
- 以只读方式访问 Zarr；保留 worker 内句柄重开能力。
- 读取区间和分块遍历清楚；不整库解压到 RAM。
- 不理解 policy 名称、normalization population、手部历史、real/sim 身份。
- 数据切片被原地增强时，不得污染原始数组或后续样本。
- 不为没有实测需求的 RAM 后端增加新配置；若本次确需小型算法数组缓存，保持局部且只读。

DP 的 `create_from_path()` 支持磁盘 Zarr，而 `copy_from_path()` 的默认 NumPy 分支会整数组读取；不要混淆两者。[官方 ReplayBuffer][dp_buffer]

## 7. T2：Sampler 回到纯 DP 窗口

### 7.1 必需能力

只保留连续窗口、episode mask、左右边界复制、keys 选择和 observation prefix 读取。采用 DP 熟悉的 `sequence_length`、`pad_before`、`pad_after`、`episode_mask`、`keys`、`key_first_k` 语义。

不要求逐字复制上游实现。可以保留当前高效的 Numba 索引生成和小型读取实现，只要行为等价且没有算法特例。

删除：

- `filter_valid()` 与由其产生的 `validity_summary`。
- 为归一化维护的 `observation_source_rows` / `action_source_rows` 全局覆盖集合。
- 供所有算法共用的 `role_lengths` / `role_offsets` 协议。
- sampler 对 action representation、future-only、hand prior 的认识。

若具体算法确实需要从采样索引得到少量源行号，可以保留一个纯索引 helper，或在专用 Dataset 中局部计算；不要为了删除函数名而复制索引算法，更不能重新建立全局窗口覆盖机制。

### 7.2 标准 DP 时间例子

当 H=16、To=2、Ta=8，最后观测时刻为 t：

| 项目 | 时间 |
| --- | --- |
| obs | `o[t-1], o[t]` |
| action target | `a[t-1] ... a[t+14]`，共 16 步 |
| policy 执行片段 | 从 prediction 下标 `To-1=1` 开始取 8 步，即 `a[t] ... a[t+7]` |

这是 [官方 policy][dp_policy] 的时间关系。`n_action_steps` 是执行侧概念，不能让 sampler 根据它偷偷移动动作起点。

本次保持现有 policy 对外返回结构和执行语义；不为了与官方返回键同名而重写 deployment。

### 7.3 `key_first_k` 的边界正确性

必须先得到有效 observation 前缀，再做任何增强、颜色统计、归一化或浮点扩大。不能先处理含未读占位尾部的 H 帧，再裁 To 帧。

对合法 `1 <= K <= H` 的窗口，prefix 读取必须等价于“完整 episode 内窗口 + 边界复制 + 取前 K 帧”。短 episode、左右 padding 都应满足。

- 可直接构造所需前缀，或使用安全初始化。
- 不依赖 `np.nan` 在 uint8 中作为哨兵。
- 不保留上游异常分支中的交互调试器。
- train 与 val 都使用一致的 keys/prefix 优化。
- 普通 action 仍返回完整 H 步；特殊 future slicing 由具体 Dataset 处理。

空验证集应能被正常跳过。可保留当前 `None` 约定，或统一为空 Dataset，但必须同步所有消费者；不为返回形式单独增加兼容层。训练集没有任何合法窗口时应明确报错。

## 8. T3：用一次边界检查替代动态坏窗口过滤

### 8.1 为什么本轮保留检查

本轮只改 policy，历史 Zarr 未必经过同一导出检查。直接删 finite filtering 会让坏值进入损失。因此在现有数据装配/准备入口集中进行一次流式检查，检查完成后训练数据按不可变源使用。

“一次”是指一次数据装配：train 和 val 共享检查结果，`__getitem__`、DataLoader worker、后续 epoch 和获取验证视图不重复扫描。不是为跨进程、跨训练运行建立全局验证缓存。

使用简单函数即可，不建立 Validator 类体系。公共 Dataset 不保存每行 valid mask，不再依照启用模态或 horizon 静默丢窗口。

### 8.2 检查范围

扫描 `train_mask | val_mask` 对应 episodes 的实际消费浮点字段与列，包括这些 episodes 中没有被某个窗口覆盖的真实帧。excluded episodes 和未选模态不参与。

| 消费路径 | 数值检查范围 |
| --- | --- |
| 普通策略 | 所选 obs 与实际 action 表示 |
| HandChunk / 仅手部样本 | 实际使用的 `action[7:19]`、`joint_state[7:19]` |
| XYZ-only 点云 | 只检查消费的 XYZ 通道 |
| R3D auxiliary EE | 主 action 与 `action_ee[:9]`，不扩大到未消费的重复 hand 列 |
| LAMP | 普通输入/目标，以及历史实际需要的 hand commands 和初始 measured hand state |
| LatentFlow | observation/reference 与 future command 来源；相对目标定义不变 |

数值检查分块进行，不构造 `[num_windows,H]` 大矩阵，不整份物化 RGB/点云。列选择不代表底层压缩 chunk 一定按列减少物理 I/O，不作此性能承诺。

发现 NaN/Inf 时立即报错，至少包含数据路径、字段、episode、全局 row 和相关通道。不得插值、置零、跳窗、删行或自动更换数据源。

### 8.3 行为变更必须明确

旧版本可能通过丢掉坏窗口继续训练；新版本会拒绝选中消费范围内的坏数据。这是有意变化。

对干净数据，窗口集合、padding 和对齐必须保持等价。对含坏值数据，不得宣称行为完全不变。

以后若生产端已有可靠的一次性数据准备流程，可以单独讨论取消重复检查；本任务不新增 `skip_validation` 开关、证书、mtime/hash 缓存或跨仓库改造来达成这一点。

## 9. T4：简化 RobotDataset 和配置

### 9.1 基础接口

保留 `__len__`、`__getitem__`，提供清楚的 `get_validation_dataset()` 与 `get_normalizer(...)`。

普通样本只有：

```python
{
    "obs": {
        # 启用的观测；时间长度 To，形状和 dtype 由已有输入契约明确
    },
    "action": ...,  # [H, Da]
}
```

具体算法需要的 `action_context` 留在具体 Dataset；普通 Dataset 不预建所有扩展 section。

正常 `__getitem__` 顺序：采样 → 裁观测前缀/选择字段 → 轻量转换与增强 → tensor。重处理不进入此路径。

`get_normalizer()` 不需要把整个训练装配器搬入类中：可调用现有简单统计函数，但普通训练与预训练不能各维护一套不同的拟合算法。

### 9.2 配置的单一来源

本轮默认保留 `sensor_modalities`、`action_key`、`pc_dim` 等实际使用配置，并保持 `obs_horizon` 与 policy 的 `n_obs_steps` 一致性检查。不要为了形式像 DP 而重写所有模型配置。

只有在引入 `shape_meta` 能同步删除重复字段、且不扩大模型和 checkpoint 改造时才采用；如采用，必须明确存储 shape、Dataset 输出 shape 和 encoder 消费 shape 的区别，且只有一份权威声明。不能将它作为未完成核心任务的理由。

图像处理优先复用 `common/image_util.py`，增强复用现有实现或少量普通函数。可以保留明确的构造参数；不为了缩短签名引入隐式 `**kwargs`、通用 transform DAG 或 registry。

R3D auxiliary target 的拼接和专属 normalizer 处理迁入实际方法适配处；可用一个短 Dataset 子类或局部函数，不建立辅助任务框架。

### 9.3 Dataset 不再生成实验治理协议

从公共 Dataset 迁出 `describe_training_data()`、`data_recipe`、real runtime 识别、hand provenance 组装等职责。

保持小型可读训练摘要即可：数据路径、train/val episode 数、窗口数和所选字段。不是必须删除所有日志，而是不再让这些日志反过来决定 sampler 和 normalizer。

## 10. T5：划分保持简单、可恢复、无泄漏

默认按 DP real loader 做法：先确定独立 val mask，再在其补集中下采样 train。

```python
val_mask = get_val_mask(n_episodes, val_ratio, seed)
train_mask = downsample_mask(~val_mask, max_train_episodes, seed)
```

具体函数签名以实现为准。不要在 train 下采样后重新使用 `~train_mask` 作为 val。

必须保存实际使用的划分。仅保存 seed/val_ratio 不足以在数据追加或排序变化后恢复原集合。

最小可恢复信息为：实际 train/val episode 索引、对应 `episode_ends`；若源已有稳定 `episode_ids`，一并保留并核对。与配置一起保存一份即可，允许简单 split JSON，不需要强制路径绑定和 source identity 框架。

- 未使用的训练 episodes 可以自然保留为未选集合，不要求额外保存一个完整 exclusions 协议。
- 如源已有 trial/group 关系，不能静默退化为随机逐片段划分。已有分组 split 能力可留在独立 split 函数/准备入口，不要求所有 Dataset 具备该元数据。
- VQ/LAMP 与对应下游复用实际 episode 划分；不能因移除通用 `requires_split_manifest` 而失去防泄漏约束。HandVAE/LatentFlow 保留使用独立 codec 语料的能力；仅在同源且共享 episode 时核对 holdout membership 一致，允许不同子集，不要求整份 manifest 相等。
- 移动相同数据路径不应单凭绝对路径不同被误判为新数据；也不能将相同长度误称为内容完全相同的证明。默认按不可变数据版本使用，保留实际已有的身份信息即可。
- 验证视图共享只读源和训练 normalizer，关闭随机增强，保留确定性预处理。

## 11. T6：明确归一化协议与数学不变量

### 11.1 普通字段的新统计总体

**新训练的普通 obs/action：对选定 train episodes 的全部真实帧拟合，每帧一次。**

包括短到无法为当前 H 生成完整窗口的 train episode；这是所选统计规则的自然结果。val 和 excluded 不参与；padding 和窗口重叠不增加权重。

普通统计不依赖 H、To、padding 或“是否被窗口覆盖”。它仍然依赖实际 train split、字段、通道和动作表示，这些依赖是正确的。

这是从 `valid_train_windows_unique_source_rows` 到 train-episode population 的行为变更，必须在交付和文档中明说。不能要求新旧统计处处数值相等。

### 11.2 必须保持的数学

- `normalize(x) = x * scale + offset`；`unnormalize` 使用对应逆变换。
- 当前 limits 的近常量维度、阈值与映射规则保持；不顺手改变到另一种 epsilon/中心定义。
- limits 不新增 clipping；val 或推理值允许超出训练范围。
- Gaussian 保留样本标准差的 `n-1` 口径，不悄悄换成 NumPy 默认 `ddof=0`。
- 保留当前分块 FP64 累计、FP32 affine 存储等已正确实现；大字段分块统计即可。
- `identity` 跳过统计拟合，但不表示可以跳过必要的数据消费检查。
- `action_ee` 的 auto 模式保留 rot6d 子空间的 identity 处理；R3D auxiliary 的既有数学也不得借重构改变。
- 保存的 `fields.<name>.scale/offset` 等 state_dict 键和实际 tensor shape 优先保持，不能为使用某个上游类而破坏已有产物。

依据：[当前 normalizer][current_normalizer]、[当前动作统计处理][current_data]、[官方 normalizer][dp_normalizer]。

### 11.3 Feature axes 必须明确

| 数据 | 独立 feature | 统计合并轴 |
| --- | --- | --- |
| `[T,D]` 状态/动作 | D 个坐标 | T |
| `[T,N,C]` 点云 | C 个通道 | T、N |
| `[T,5,3]` 手指向量 | 15 个独立坐标，恢复为 5×3 affine | T |
| RGB | 不拟合 learned affine | 不适用 |

移除 normalizer 内的 `FINGER_VECTOR_FIELDS` 名称分支时，用一个明确的 feature shape / `last_n_dims` 参数替代即可；该参数由已有任务/字段适配处提供。不能把所有字段统一按最后一维统计，也不能给每个点序号拟合独立 affine。

对当前 identity-only 的大触觉字段等，不为假想归一化需求扩展新的机制。

### 11.4 算法目标例外

LatentFlow 的 action 统计来自每个合法训练窗口中的 `future_command - reference_state`。相同 command 在不同 reference 下是不同目标，必须保留窗口/reference 分布；不能按绝对 action 的源行去重或套普通 train-frame 统计。

DQ/VQ 的单帧绝对 hand action 路径应与下游使用同一选定 train population 和 limits affine 数学；保留实际数值一致性检查。LAMP 当前 action 为 identity、joint_state 为 gaussian，prior 使用物理 hand commands；不得为统一这两类方法给 LAMP 新增 hand affine 拟合。

## 12. T7：RGB、增强和样本独立性

当前 DP 配置采用 `rgb: identity`、`rgb_keep_uint8: true`；DINO/ResNet/SigLIP processor 接收 uint8 `[0,255]` 或 float `[0,1]`，再做骨干 mean/std。[配置][current_dp_config]、[ImageProcessor][current_image_processor]

必须保持：

- 盘上 RGB 为 uint8 HWC；输出布局依据已有确定性预处理契约，不能假定所有路径已经转 CHW。
- 不为接近 DP 示例而给 RGB 新增 `[-1,1]` 数据 normalizer。
- 不得对已经处于 `[0,1]` 的 float 再除以 255；backbone mean/std 仅由 ImageProcessor 执行。保留增强阶段 `uint8 → float [0,1] → uint8` 的临时往返转换；回到 uint8 后，processor 再转浮点是正确转换，不属于重复归一化。
- 不强制将本来保留 uint8 的 batch 提前扩大成 float32。
- 同一 observation window 的历史帧共享 crop 参数，不能改成逐帧独立随机裁剪。
- 保持 resize/crop 尺寸、视场、插值、antialias、浮点/整型插值路线及舍入方式。
- 训练增加明确随机项；val/deployment 关闭随机项但保留一致的确定性空间处理。

特别注意：当前只要配置颜色增强，空间插值就在 float 上进行，即使某次随机判断没有真正应用颜色变换；验证的 `float_spatial_before_uint8` 与之对应。不能简单将 `val.transform=None` 理解为删除全部 preprocessing，也不能把 `resize 240 → crop 224` 改为直接 `resize 224`。[现有图像工具][current_image_util]

Dataset 中现有状态/点云增强继续在物理量域执行，随后模型入口归一化。Policy 内已有的算法专属增强仍保留原位置，例如 LAMP 的 arm state noise，不因整理 Dataset 而统一搬移。不要在本次重构中同时改变增强顺序、随机点采样顺序或随机数消耗分布。

任意一次样本增强不得污染源数据。保留 Zarr 独立切片语义；若使用 RAM view，增强前复制。用针对性测试证明，不能仅凭多 worker 能运行就视为正确。

## 13. T8：保留现有具体算法的最小适配

### 13.1 普通策略

DP、DP3、ManiFlow、SAT 以及不启用特殊目标的 R3D，共用普通 `RobotDataset`。不为每种 backbone 建一个 Dataset。

### 13.2 R3D auxiliary EE

在方法适配处构造主 joint target 与额外 9 维 EE target。保留模型预测宽度、控制输出宽度、normalizer 与部署截取的现有关系。数据检查和统计仅消费真正使用的列。

### 13.3 HandChunk / HandVAE

保留 hand12 切片、完整动作 chunk、reference state、无跨 episode 的样本定义。未消费 arm 列的坏值不能让纯手部任务失败。

结构参数、动作语义、dt、joint order、horizon、latent layout 等必要兼容信息留在手部产物/加载器，不回流到公共 sampler。HandVAE codec 可使用独立语料；同源共享 episode 保持 holdout 一致性，不能改成强制同一完整 split。

### 13.4 LatentFlow

- 观测历史结束于 query 时刻 t。
- 未来动作从 t 开始，共 H 步；基础 sampler 可采更长共同序列后由该 Dataset 切片。
- 历史可按原约定补齐；未来动作不做尾部 padding。
- reference 是未被观测噪声污染的 query measured state。
- `action_context.reference_state`、relative target、专属 action normalizer 保持。
- 不在普通 Dataset 中增加 future mode、任意 role offset 或 normalization-population 开关。
- 保留独立 codec 语料及同源不同子集的兼容规则；时间/动作 contract 仍须相容，同源共享 episode 的验证归属必须一致。

### 13.5 LAMP

- 保留当前 H=To=Ta=1 等方法约束，不顺便推广多步版本。
- 保留 `action: identity`、`joint_state: gaussian` 的方法约束；prior 的输入/目标保持物理 hand commands，不新增为与 VQ 统一而设的 hand normalizer。
- history 只包含严格过去的 hand commands。
- episode 起点历史不足，使用该 episode 初始 measured hand state bootstrap，不能用当前 action 填充。
- 去掉动态 `history_valid` 过滤后，由边界检查保证所需数据可用，历史语义仍然保留。
- 小型 hand command 缓存可保留，禁止扩张为通用缓存系统。

### 13.6 DQ-RISE / VQ / 码本导出

- 修改 VQ/LAMP 准备代码中对 `sampler.action_source_rows` 的依赖，按明确 train/val episodes 读取真实 hand action 行。
- 只消费手部 target/history 的预训练，应直接复用读取和 split 函数取得所需字段/列，避免为了获取划分或 metadata 而构造并扫描完整视觉 policy Dataset。
- 保留 hand affine 与 policy action hand 切片的数值一致性。
- 同步 codebook export、artifact metadata、selection split/metric 和加载检查。
- 预训练 train episodes 不能包含下游 held-out val episodes；已有实际来源与 split 信息用于这一检查，不把它削减为只比较 hand_dim。
- 旧 codebook 与新统计不兼容时明确报告；不得删掉 affine 检查、自动重标定码本或悄悄重新 fit 使其通过。

相关实现：[Hand Dataset][current_hand_chunk]、[LatentFlow Dataset][current_latent]、[LAMP Dataset][current_lamp]、[VQ 准备][current_vq]、[码本构造][current_codebook]、[码本加载][current_codebook_load]。

## 14. T9：metadata、checkpoint 与部署必须联动

### 14.1 责任迁移，不删除数值契约

训练入口保存普通有效配置、实际 split、normalizer 与模型训练状态。真实部署所需参数由训练产物装配/部署边界处理，不由公共 Dataset 根据 `format` 分支生成。

优先保留已有外部消费者实际使用的键与结构，例如 `policy_metadata`、`real_runtime`、`normalization_dims`。移动生产位置即可；不要为追求字段名称新颖强制所有历史实验迁移。

`real_runtime` 中实际影响数值或部署的 dt、点云/手点云处理、深度尺度、指尖顺序和 RGB 确定性预处理不能丢失。需要真机参数的部署入口仍应核对必需信息，不能用当前机器默认值替代训练保存值。

手部 artifact 的结构、语义、split 与 affine 兼容检查留在专用模块。不要将整棵旧 `data_recipe` 搬到一个新服务里，也不要将其有效内容一并删除。

### 14.2 兼容的新协议恢复

新协议 checkpoint 的完整恢复必须：

- 使用保存的实际 split 和输入配置。
- 恢复保存的 normalizer，绝不重新 fit。
- 保持 model/EMA/optimizer/scheduler/RNG/step/epoch 的现有恢复行为。
- 保持本仓库已有的 epoch 起点恢复语义，不新增 minibatch cursor、worker/prefetch 状态保存或逐位重放承诺。
- 保留对影响训练数值的配置差异检查；移除 recipe 不等于允许任意配置继续训练。

### 14.3 旧实验处理

新鲜训练采用新的统计总体；旧 checkpoint 的 affine 不能被重算或替换。

本次不强制实现跨协议的旧 full resume 自动迁移。若旧实验的窗口过滤、统计协议或配置结构无法在新路径下证明兼容，应尽早给出明确错误，说明使用对应旧代码继续恢复，或另开新实验。不能静默降级成 warm start、重新训练或放宽比较。

旧 checkpoint 的 inference 尽量通过保持参数键/shape、输入预处理和既有输出 metadata 来维持。如果确有必须改变的配置格式，只允许边界处一个明确、可测试的无损配置转换；无法无损时说明限制。禁止引入遍布 Dataset/Sampler 的 legacy 模式。

完整 policy checkpoint 已内嵌 prior/codec/codebook 参数时，恢复推理不应重新依赖原预训练文件路径或训练 Zarr。保留现有 `prepare_restore_config` 等恢复行为；在不下载外部权重的条件下验证支持范围内的 raw/EMA round-trip。

如需新增一个简短协议标记用于区分恢复兼容性，必须说明其具体消费者；不得演变成 schema version manager。优先利用已有保存内容识别不兼容。

### 14.4 必须更新的串联位置

逐一核对：训练入口、`build_dataset_and_normalizer`、`assemble_effective_config`、`ExperimentLogger.save_config`、`validate_source_recipe/resolve_input_recipe`、`archive_resume_milestones`、checkpoint strict restore、手部预训练准备与 export、`inspect_policy`、`inference_recipe`、`restore_policy_agent`、evaluation config、smoke test。核对不等于每个文件都必须修改。

Milestone 归档也比较各 checkpoint 的 effective config。若实施任何一次性配置转换，必须一致处理所需历史 milestone，或明确拒绝该跨协议恢复；不能只转换 latest checkpoint 而留下归档冲突。

任何被删除公共字段的引用都必须清理或迁移；不要留到运行时出现 `KeyError` / `AttributeError` 才发现。

## 15. 配置和用户使用体验

保持现有启动入口与已公开脚本。普通实验不应新增必填 manifest/source_id/format/registry 配置，也不应要求用户为不同普通策略重新导出同一数据。

推荐操作体验：

1. 配置已有 `zarr_path`、观测字段、动作表示和 horizon。
2. 默认由 seed/ratio 产生 episode split；严格对比实验可传入共享 split。
3. 构建时一次检查所选数据，清楚报告不合格位置。
4. 新训练拟合 normalizer；恢复使用已保存参数。
5. 进入普通 DataLoader / policy 训练。

有明确外部消费者的 RGB、pc_dim、hand history 等配置可以保留。不要为“最少参数”改成隐藏默认行为，也不要新增若干含义重叠的配置副本。

## 16. 性能要求与测量边界

本次必须保留或实现：

- 所选字段读取；未选字段不扫描、不载入。
- observation prefix 读取，动作读取完整目标范围。
- 大数组分块检查/统计；无窗口规模的 dense validity 矩阵。
- train/val 共享源，不重复全量 admission。
- 保留现有 pin memory、persistent workers、prefetch、non-blocking device copy 等已有设置，不把它们称为本次新增。
- 不因接口重构持久保留不必要的完整 action/state 临时数组。
- 完整恢复不重新 fit normalizer。

一次 finite 检查仍可能读取所选点云；不能声称新设计启动不扫描大模态。消除的是窗口过滤、覆盖集合和无关读取，实际速度须测量。

先通过读取次数/范围和内存生命周期确认上述性质。有实际数据和环境时，再记录构建时间、samples/s、DataLoader 等待与峰值 RSS。比较时固定数据、split、H/To、batch、worker、增强和硬件，并区分冷/热缓存。

不为缺少真实数据编造性能结果，不设未经证实的“至少加速 X 倍”门槛。Chunk 重新打包、整库 RAM、GPU augmentation、episode locality batching 属于后续有证据再做的优化。

## 17. 验证计划：覆盖真实风险，避免镜像实现测试

### 17.1 测试环境与 fixture

优先扩展已有 `tests/test_data_paths.py`。使用真实 numpy、Zarr v2、torch、OmegaConf，在临时目录建立小型数据；不下载视觉模型、不需要 GPU、机器人或真实数据集。

现有 fixture 在公共 setUp 中注入 NaN，需要拆成 clean fixture 与独立坏数据 case。删除的是旧过滤/覆盖统计的预期，不是删除时间、归一化或恢复正确性测试。

可以监控 I/O、fit 调用和局部外部模型初始化；不要用假 Zarr/假 torch 替代核心依赖来宣称测试通过。

### 17.2 最小验收矩阵

| 编号 | 验证内容 | 必须证明 |
| --- | --- | --- |
| V1 | DP 窗口参考等价 | 多 episode 唯一数值标记；普通/左 pad/右 pad/短 episode；To=1 与 To=H；与独立完整读取+clip 参考一致，不跨 episode |
| V2 | Prefix I/O | 不读取未选字段；每个观测源片段读取长度不超过所需前缀；action 取得完整目标；uint8 不依赖 NaN 占位 |
| V3 | 坏值拒绝范围 | selected train/val 消费列 NaN/Inf 明确报错；excluded、未选模态、未消费列中的坏值不误伤 |
| V4 | Split | train/val 无交集；减小 max_train 不改变 val；val_ratio=0 正常；保存实际索引可重建 |
| V5 | 新统计总体 | train 全行每行一次，包括无法产出完整窗口的短 episode；val/excluded 极端值不影响；H/To/padding 改变不影响普通统计 |
| V6 | 数学与 feature axes | 常量/近常量、Gaussian n−1、无 clipping、round-trip、PC 按通道、五指按 15 个坐标、action_ee 旋转 identity |
| V7 | 手部目标 | HandChunk 切片；LatentFlow query/future/reference 与相对统计；LAMP 严格过去历史和 measured bootstrap |
| V8 | 预训练联动 | VQ/LAMP 读取明确的 episode 行；DQ hand affine 一致；LAMP 保留物理动作/identity；独立 HandVAE codec 语料可用，同源共享 episode 的 holdout 不泄漏；不兼容 artifact 清楚失败 |
| V9 | RGB 一致 | 匹配确定性条件后 val/deployment 预处理一致；等价 uint8/float 输入到 processor 一致；不对 [0,1] float 重复 /255，不额外套 [-1,1]；保留必要量化往返，同窗口共享 crop |
| V10 | 生命周期 | 增强不污染源；验证视图不重新 admission；真实 DataLoader workers=0 和一次小规模 worker 检查能正确读取 |
| V11 | 保存与恢复 | affine 保存恢复一致；新协议 resume 不 fit；模型/EMA/训练状态按现有语义恢复；不兼容旧 full resume 明确拒绝 |
| V12 | 完整调用链 | 新 effective config 能保存；完整 policy 的支持范围内推理恢复不依赖原 Zarr/预训练文件；raw/EMA round-trip 和 milestone 配置校验正确；config-only/smoke 没有残留旧字段依赖 |

验收可合并成少量高信息量测试，不要求为表格每格新建一个测试文件。

具体建议：

- V1 的参考计算不要调用待测 sampler 的内部 source-row helper，否则只是实现的镜像。
- V3 至少覆盖 hand-only 未用 arm、XYZ-only 未用 RGB、aux EE 未用 hand 列。
- V5 用手算可得统计；不要只比较两个调用同一 fitter 的函数。
- V6 对浮点分块累计使用合理、说明过的容差；不强求不同精度路径逐位相等。
- V7 将 reference_state 与观测增强分开验证，避免相对目标跟着噪声漂移。
- V11 不要求连续训练与 epoch 中断恢复逐 minibatch 完全一致。
- V11/V12 对本次声明支持的旧 checkpoint/config fixture，验证实际存在的 raw/EMA state 能 strict load，affine 不变，过程不读取训练 Zarr 或原始 prior/codebook/codec。具备依赖时比较固定输入/随机种子的输出；缺依赖时明确未验证，不能用 config-only 代替。
- 多进程测试受环境限制时明确记录，不能将跳过说成通过。

### 17.3 建议命令

在已安装仓库及其声明依赖的环境中，根据最终修改后的入口执行。以下是基线中存在的命令形式；若接口调整，文档与命令同步更新。

```bash
python -m compileall -q dexmani_policy
python -m unittest discover -s tests -p 'test_data_paths.py' -v
python -m unittest discover -s tests -p 'test_evaluation.py' -v
python dexmani_policy/smoke_test.py --config-only dp
python dexmani_policy/smoke_test.py --config-only dp3
python dexmani_policy/smoke_test.py --config-only maniflow
python dexmani_policy/smoke_test.py --config-only r3d
python dexmani_policy/smoke_test.py --config-only sat
python dexmani_policy/smoke_test.py --config-only dqrise
python dexmani_policy/smoke_test.py --config-only lamp
python dexmani_policy/smoke_test.py --config-only latent_flow
git diff --check
```

`--config-only` 不代表完整训练已验证。数据/模型 smoke 在具备实际依赖、数据和产物时才运行；缺少环境不应引发自动下载大模型、改依赖版本或缩小正式模型来假装复现。可用真实小模型 fixture 验证保存/恢复机制，但不改变正式 YAML 默认值，也不称其为完整算法复现。用独立的小型真实数据 fixture 验证本任务的数据逻辑，并明确未验证项。

测试范围按实际改动收敛；已有相关门槛通过后，不为假想问题反复扩大测试。

## 18. 实施顺序

### 阶段 A：确认依赖与冻结不变量

- 查看 git 状态、当前提交和 `AGENTS.md`。
- 沿主训练、手部预训练、resume 和 deployment 查实际引用。
- 列出要删除/迁移字段的 consumer，确定本任务的两个行为变化：坏值 fail-fast、普通 train-episode 统计。
- 先准备干净 fixture 和时间/统计参考，避免重构后靠旧实现自证。

### 阶段 B：核心模块精简

- 保留只读 ReplayBuffer。
- 将 sampler 收敛到 DP 窗口/prefix 语义。
- 移除动态过滤和覆盖集合。
- 加入集中、按消费范围的一次边界检查。
- 实现普通 get_normalizer 和验证视图，保留 RGB 与增强契约。

### 阶段 C：完成所有方法与产物联动

- 更新 HandChunk/LatentFlow/LAMP、R3D aux 与 VQ/codebook 数据准备。
- 将 metadata 生产迁到真正拥有其语义的位置。
- 同步 config、logger、resume、deployment、smoke 和必要脚本。
- 删除无消费者的旧 helper、字段和配置，保留实际数值兼容检查。

### 阶段 D：验证与文档

- 运行针对性测试和可执行检查。
- 核对未选字段 I/O、prefix、一次 admission、缓存独立性和 resume 不 fit。
- 更新 README/相关架构与手部文档，说明新统计和旧实验兼容范围。
- 检查 diff 无无关格式化、依赖升级、算法改变或新框架。

可以在本地分成便于审查的逻辑提交，但不要以阶段完成为由留下主训练/部署断链。若存在真实环境阻塞，完成不受影响的实现和验证，明确阻塞，不伪造成功。

## 19. 完成定义与交付格式

只有以下条件成立，才称本任务完成：

- 公共 sampler 不再管理坏数据、统计总体或方法模式。
- 普通 Dataset 的职责可以用采样、转换、验证视图、normalizer 接口简洁解释。
- 没有全局 valid-window source-row 覆盖机制的残留消费者。
- 普通新统计与算法专属 relative 统计分清且验证。
- 历史/动作时间、RGB、normalizer affine、hand artifact 和部署契约未被无意改变。
- 实际 split 可保存/恢复，预训练与下游无已知交叉泄漏。
- 删除 data_recipe 后 logger、resume、deployment、smoke 均已联动。
- 必要检查通过；未具备条件的实测被明确标记。
- 新增抽象、配置和状态均有具体消费者，无为假想需求搭建的平台。

最终交付用中文说明：

1. 改了哪些文件、为什么，核心复杂度减少在哪里。
2. 哪些是结构重构，哪些是有意的行为变更。
3. 旧 checkpoint / hand artifact 的可用范围与不兼容处理。
4. 实际执行的测试、命令、结果；未执行项及原因。
5. 有测量才报告性能数据，并说明环境与比较条件。
6. 剩余真实限制；不要把“可以以后做的优化”写成已完成工作。

## 20. 配套 Codex Prompt

以下文本可直接复制到已连接 `haoyangzhanglab/dexmani_policy` 的 Codex 任务中：

```text
请在 haoyangzhanglab/dexmani_policy 中直接实施数据层重构，执行仓库文件 docs/CODEX_DATASET_REFACTOR_TASK.md 的最终要求。先完整阅读该任务书、AGENTS.md 和相关实际调用链，再制定简短计划并持续完成实现、必要测试和文档；不要停在方案或仅给代码建议。

目标是以官方 Diffusion Policy 的数据结构、窗口语义和 Dataset 接口为基准，让基础数据模块简洁、正确、高效、易适配。保持只读 Zarr、episode 级 split、纯 SequenceSampler、清楚的 obs/action 和 get_normalizer/get_validation_dataset。不要强制引入 shape_meta、整套配置重命名或新的 registry/manager/缓存框架。

本任务批准的两项行为变化必须明确实现并记录：① 取消动态坏窗口过滤，在数据装配边界对所选 train/val episodes 的实际消费字段/列做一次流式检查，遇错定位并停止；② 普通字段改为训练 episodes 全部真实帧统计，不再追踪窗口覆盖去重行。LatentFlow 的相对目标/统计、LAMP 的严格过去历史与实测 bootstrap、DQ/VQ 的 affine 和 split 一致性等算法语义必须保留。

保留现有 RGB uint8/identity/ImageProcessor 数值契约、确定性空间预处理和 augmentation 顺序；保持 normalizer 数学、feature axes、state_dict 参数键及形状。不要照搬官方图像 [-1,1] normalizer，也不要把 command 改成 future measured state。

完整更新训练装配、effective config、logger、resume、手部预训练与 codebook、deployment/runtime、smoke test 和相关配置。不能只删 Dataset 字段而留下 consumer。兼容 checkpoint 恢复必须使用保存的 normalizer；不兼容旧 full resume 明确拒绝，不自动 refit 或降级 warm start。保持当前 epoch 起点恢复语义。

用任务书的独立小型真实 Zarr fixture 和针对性验收覆盖窗口/padding、prefix I/O、坏值范围、split、新统计、RGB、手部目标、样本独立性和恢复。不要下载大模型或依赖假模块来声称验证通过；如环境受限，完成可执行部分并准确报告未验证项。

修改前检查 git 状态并保护已有工作。任务书审查基线是 4cd6275ecf65e8a6f281aac2b70352eb3609da9e；若当前代码已变化，核实并适配，不强制回退。本任务限于 dexmani_policy 及其必要调用方，不新增七个参考方法，不改 dexmani_real/dexmani_sim，不改变网络、loss、评测协议或实验默认超参数。

完成后给出文件级改动、行为变更、旧产物兼容范围、实际测试结果和剩余限制。性能仅在实测后报告。除真实阻塞或目标冲突外，按任务书自主完成，不反复请求确认。
```

## 21. 源码参考链接

引用绑定上述审查快照，用来解释设计依据。后续实施应同时核对当前代码，不把上游细节机械当成新需求。

[dp_repo]: https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f
[dp_base]: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/base_dataset.py
[dp_buffer]: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/replay_buffer.py
[dp_sampler]: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/sampler.py
[dp_dataset]: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/real_pusht_image_dataset.py
[dp_policy]: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py#L265-L272
[dp_normalizer]: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/normalizer.py
[maniflow_repo]: https://github.com/geyan21/ManiFlow_Policy/tree/ef2f116f1f90163ed36e657b8c5503740bb468af
[maniflow_dataset]: https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/dataset/adroit_dataset.py
[rdt_repo]: https://github.com/thu-ml/RoboticsDiffusionTransformer/tree/cd79363a1387e8f81c7724d070ef7e45fd23150f
[rdt_dataset]: https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/cd79363a1387e8f81c7724d070ef7e45fd23150f/data/hdf5_vla_dataset.py
[cordvip_repo]: https://github.com/xuanxuanzzzii/cordvip/tree/ad6d441abe7b0480cf79b0311db266965c1313b6
[cordvip_convert]: https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/script/pkl2zarr_cordvip.py
[afro_repo]: https://github.com/KolaKivy/AFRO/tree/283024a49e2c14bc030debf15383a4e7ee73836d
[afro_task]: https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/config/task/adroit_pen.yaml
[afro_loss]: https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/policy/afro.py#L188-L265
[mopa_repo]: https://github.com/XPolicyLab/XPolicyLab/tree/c43a74653bec480ebe82c84770664e8035bc56c9/policy/MoPA
[mopa_dataset]: https://github.com/XPolicyLab/XPolicyLab/blob/c43a74653bec480ebe82c84770664e8035bc56c9/policy/MoPA/mopa/data/dataset.py
[mopa_model]: https://github.com/XPolicyLab/XPolicyLab/blob/c43a74653bec480ebe82c84770664e8035bc56c9/policy/MoPA/mopa/models/action_model.py#L299-L312
[transdex_repo]: https://github.com/LFGfg/TransDex/tree/a4151879701bfb85a45447fa6d99fea07dd8c7dc
[transdex_dataset]: https://github.com/LFGfg/TransDex/blob/a4151879701bfb85a45447fa6d99fea07dd8c7dc/policy_ws/src/VTFusion/src/VTFusion_dataset.py
[current_buffer]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/datasets/common/replay_buffer.py
[current_data]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/training/common/data.py
[current_logging]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/training/common/logging.py
[current_resume]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/training/common/resume.py
[current_runtime]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/deployment/runtime.py
[current_vq]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/training/pretrain/hand_prior/vqvae.py
[current_codebook]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/training/pretrain/hand_prior/codebook.py
[current_codebook_load]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/agents/hand_prior/codebook.py
[current_normalizer]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/common/normalizer.py
[current_dp_config]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/configs/dp.yaml
[current_image_processor]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/agents/obs_encoder/rgb/image_processor.py
[current_image_util]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/common/image_util.py
[current_hand_chunk]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/datasets/hand_prior/hand_chunk_dataset.py
[current_latent]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/datasets/hand_prior/latent_flow_dataset.py
[current_lamp]: https://github.com/haoyangzhanglab/dexmani_policy/blob/4cd6275ecf65e8a6f281aac2b70352eb3609da9e/dexmani_policy/datasets/hand_prior/lamp_dataset.py
[real_schema]: https://github.com/haoyangzhanglab/dexmani_real/blob/533b3ea5f36d81734ecfa56327866c1bcdaa96f7/dexmani_real/data/raw/schema.py
[real_processing]: https://github.com/haoyangzhanglab/dexmani_real/blob/533b3ea5f36d81734ecfa56327866c1bcdaa96f7/dexmani_real/data/canonical/processing.py
