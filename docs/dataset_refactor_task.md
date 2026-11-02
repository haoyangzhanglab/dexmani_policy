# 数据集模块精简重构任务书

> 历史实施记录：后续数据集收敛状态见[当前任务书](dataset_simplification_task.md)。本文状态和验证结果只对应原实施阶段；与本轮冲突的数量默认、上限语义等以当前任务书及实际代码为准。

> 状态：已实施。已完成 CPU 与 CUDA 合成数据验收；未执行真实数据长训。实际验证与产物说明见第 12.5 节。
> 后续：恢复日志、normalizer 形状、部署事实及评测发布行为以[第二阶段任务书](research_simplification_task.md)为准；本文验证记录仅对应上一阶段。
>
> 仓库：haoyangzhanglab/dexmani_policy  
> 编写日期：2026-11-02  
> 研究与代码核查基线：ca1f5dc649a2e0bc5dc3b94856d6a9546b533965  
> 定位：个人机器人策略研究代码；优先正确、直接、易读易改，避免平台化机制。
>
> 执行前阅读根目录 [AGENTS.md](../AGENTS.md)，并核对当前分支与相关调用方。本文确定行为与范围，局部实现可按当前代码作更简单的调整；不要为了复刻本文的文字结构新增代码层次。

## 1. 目标与最终决策

完成一次最小但完整的数据管线收敛，让主策略、观测预训练、手部码本预训练使用一致的数据选择语义，同时保留各方法真正不同的样本和训练目标。

本轮采用以下最终方案；前期讨论中的其他备选方案不再实现：

| 事项 | 最终决策 |
| --- | --- |
| 核心结构 | 保留 EpisodeStore → SequenceSampler → Dataset；继续使用原生 PyTorch DataLoader |
| 划分方式 | 一个仅依赖 episode 数量、seed 和数量参数的纯函数 |
| 参数 | 删除 val_ratio 和 split_manifest；使用 val_episodes |
| 选取顺序 | 有训练上限时先选 train，再从剩余 episodes 选 val；其余 unused |
| 主策略默认 | seed=42、max_train_episodes=100、val_episodes=0；不构造 val dataset/loader |
| 独立预训练默认 | seed=42、max_train_episodes=null、val_episodes=0；可显式启用 val |
| 新统计量 | 仅从本阶段选中的 train episodes 拟合；不加入范围切换选项 |
| 继承统计量 | 预训练模块携带的输入归一化直接沿用，包括 identity；新字段才拟合 |
| Dataset 的内容 | 保存读取与采样所需状态，返回物理含义明确的张量；不持有 normalizer 或训练器 |
| 预训练连接 | AFRO/CordViP 传 encoder 与其输入归一化；DQ-RISE 传有序码表与手部 affine |
| 预训练选模 | 默认固定预算、最终产物；显式启用 VQ val 时可保留基于 lookup_mse 的 best |
| 校验 | 必要边界检查一次；信任内部结果；错误直接暴露 |
| 数据身份 | 删除训练路径中的来源身份协议、manifest 恢复和自动 trial 分组 |
| 兼容 | 同步当前仓库的调用方和文档；不新增旧参数别名、格式迁移或自动兼容层 |

**完成标准是行为清楚、接口缩小、重复机制消失，并通过相关验证。不要以减少行数、文件数或类数作为独立目标。**

## 2. 范围

### 2.1 必须完成

- [x] 统一 episode 划分语义及默认值，更新主策略与所有独立预训练入口。
- [x] 删除 manifest、数据身份比较、自动 trial 分组以及附属配置、CLI、测试和文档。
- [x] 明确 normalizer 的数据范围、所有权和预训练继承行为。
- [x] 简化 AFRO/CordViP 的可选验证与 DQ-RISE 码本预训练、导出、加载。
- [x] 移除妨碍正常实验的重复或流程性检查；保持必要的数学、时序和几何约束。
- [x] 同步运行脚本、说明文档、相关检查工具，并完成第 12 节的验收。

### 2.2 本轮不扩展

不重写模型架构、损失、量化算法、EMA、AMP、梯度累积、学习率调度、rollout 评测、RTC 或真机控制。为接口变化适配这些模块的调用点即可。

不新增 DataModule、注册表、通用字段 schema、通用预训练框架、数据缓存管理器、任务调度器、审计记录系统、自动重试、自动跳过坏数据、多格式加载器或“strict/relaxed”模式。

不更改原始 Zarr，不删除已有实验目录、checkpoint 或预训练产物。不要求新接口自动恢复所有历史产物；在文档中明确需要重新导出的情况。

不以“统一”为由更改所有策略的 horizon、padding、动作时序、归一化默认算法或辅助监督定义。

## 3. 当前实现与重构边界

基线已经具备按需读取、进程内 Zarr 句柄、分块统计、短观测读取、轻量验证视图、独立预训练产物和完整策略 checkpoint。保留这些能力。

| 位置 | 当前问题或职责 | 本轮处理 |
| --- | --- | --- |
| datasets/split.py | 比例划分、完整 manifest、身份恢复、trial 分组、导出 CLI 混在一起 | 收敛为数量划分纯函数 |
| datasets/episode_store.py | 只读 Zarr、边界、进程句柄、按字段/列读取、分块迭代 | 保持主体，不增加全量加载或额外缓存 |
| datasets/sampler.py | episode 内窗口、边界重复、字段读取长度 | 保持语义与实现主体 |
| datasets/demonstration.py | 主策略样本、增强、训练/验证 mask | 删除协议性参数和元数据门槛，保留实际读取与样本职责 |
| datasets/pretrain/ | AFRO/CordViP 独立样本和验证视图 | 共用简单划分，不强行改成主策略 Dataset |
| training/common/data.py | 数据构造、维度信息、normalizer 装配、DataLoader | 保留直接函数，缩小数据描述，明确继承优先级 |
| training/common/config.py | Python 注入并禁止覆盖部分归一化配置，保存 manifest | 把普通默认值放回 YAML，去掉 manifest 传播 |
| training/pretrain/obs/common.py | 训练、可选验证、恢复、encoder 导出 | 默认不验证；保留现有轻量恢复与导出 |
| training/pretrain/hand_prior/vqvae.py | 无 val 时仍遍历 train 评估并选 best | 无 val 直接 last；有 val 才执行验证选模 |
| training/pretrain/hand_prior/codebook.py | 导出依赖 split 与选模 metadata | 任意完整 VQ checkpoint 可导出，不要求 best 身份 |
| agents/hand_prior/codebook.py | 运行时依赖预训练分组结构和 metadata | 仅使用有序姿态表、总码数和 hand affine |
| agents/policies/dqrise.py | 预训练与主策略完整 manifest 相等才允许训练 | 删除该比较，保留手部坐标和码表使用语义 |
| test_tools/、scripts/、docs/ | 仍描述旧参数、共享清单和强制绑定 | 与实现一起更新 |

表中路径默认相对 dexmani_policy/，scripts/ 和 docs/ 位于仓库根目录。

## 4. Episode 划分：唯一明确的数量语义

### 4.1 函数接口

在 datasets/split.py 保留一个直接函数，建议接口：

~~~python
def split_episodes(
    n_episodes: int,
    *,
    seed: int = 42,
    max_train_episodes: int | None = None,
    val_episodes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    ...
~~~

返回两个长度为 n_episodes 的 bool mask：train_mask、val_mask。不返回 manifest、Split 对象或包装 dataclass，不接收 Store、metadata、文件路径或 OmegaConf 对象。

数量参数只在这一消费边界检查：n_episodes 为正整数，max_train_episodes 为 None 或正整数，val_episodes 为非负整数。不要在每个调用方重复检查相同条件。

### 4.2 算法

设总数为 N、训练上限为 M、验证数量为 V：

1. 使用局部 RNG：np.random.default_rng(seed).permutation(N)。不消费训练或增强的全局 RNG。
2. M 非空时，T=min(M,N)。M 为空时，T=N-V。
3. 若 T≤0 或 T+V>N，直接报一条清楚的数量错误。
4. 排列的前 T 个索引给 train，随后 V 个索引给 val，其余 unused。
5. 直接由切片构造 mask。生成算法已经保证互斥，不再逐层校验同一内部结果。

**M 有值时不得为了容纳 val 而把 T 改成 min(M,N−V)。不得缩小 V、重复采样补数或把 unused 并入 val。**

同一数据顺序、N、seed、M、V 得到同一划分。数据 seed 与 training.seed 保持独立。不新增“跨预算固定 val”“对数据重排保持身份不变”“跨任务共享划分”等协议。

### 4.3 必须满足的例子

| N | M | V | train | val | unused / 结果 |
| --- | --- | --- | --- | --- | --- |
| 150 | 100 | 20 | 100 | 20 | 30 |
| 150 | 100 | 0 | 100 | 0 | 50 |
| 150 | null | 20 | 130 | 20 | 0 |
| 150 | null | 0 | 150 | 0 | 0 |
| 80 | 100 | 0 | 80 | 0 | 0，不重复补齐 |
| 110 | 100 | 20 | — | — | 报错：100 个 train 后只剩 10 个 |
| 100 | 100 | 20 | — | — | 报错，不暗中变成 80/20 |
| 1 | null | 0 | 1 | 0 | 0 |
| 1 | null | 1 | — | — | 报错：没有 train |
| 任意合法 N | 0 或负数 | 任意 | — | — | 参数错误 |
| 任意合法 N | 合法 M | 负数 | — | — | 参数错误 |

### 4.4 与现有 Dataset 的连接

为减少改动，保留当前 Dataset 的 train_mask、val_mask 两个小数组，以及 ObservationPretrainDataset 已有的轻量验证视图。不要为了让 Dataset 只剩一个 mask 而重写所有构造入口。

- 主策略 Dataset 调用纯函数一次，仅构造 train sampler。
- AFRO/CordViP 调用同一个函数一次；验证视图共享只读 Store，使用明确的 val_mask 建 sampler，不重新抽样。
- VQ 的数据准备函数调用同一个函数，直接按 mask 读取手部真实帧。
- 未选中的 episodes 通过 ~(train_mask | val_mask) 得到，仅在需要统计数量时计算。
- 不使用 ~train_mask 代表 val。
- val_episodes=0 时，不创建验证 sampler、Dataset、DataLoader，也不做空验证循环。
- 主策略可以显式保留 val episodes，但只报告这些数据没有参与本次训练，不为此新增验证流程。
- 独立预训练显式请求 val，但这些 episodes 无法产生任何验证样本时，在验证构造处给出清楚错误；不自动换数据或回退 train。
- 个别过短 episode 没有合法窗口时，保留既有窗口枚举行为；不要自动挑另一条补齐训练数量。整个 train 没有样本时再明确报错。

### 4.5 删除清单

移除 val_ratio、split_manifest、validate_episode_metadata、_restore_split、比例 get_val_mask、自动 trial 分组、贪心预算和 split 导出 CLI 及全部当前调用。

source_id、episode_ids、trial_ids 可以继续存在于用户数据中，但训练划分不读取或校验它们，也不修改原数据。若某实验需要避免同一长轨迹的片段跨集合，在该实验的数据准备阶段处理，不恢复通用分组框架。

不新增独立 split.json、dataset hash 或身份注册。配置保存 seed/M/V，启动日志打印 N/train/val/unused/windows 即可。无需强制把完整 episode 索引再次写入产物并作为加载条件。

## 5. 数据读取与样本：保留现有薄结构

### 5.1 EpisodeStore

只负责：

- 只读打开 Zarr，读取 episode_ends 及实际需要的数组信息；
- 在打开边界检查 episode_ends 非空、递增、有效，以及所选数组时间轴一致；
- 按 key、连续行段、必要列读取；
- 每个进程使用自己的句柄，保持已有 pickle/spawn/fork 行为；
- 迭代所选 episodes 的连续 chunks，避免一次物化完整视觉数据。

保持 RGB uint8 等原始存储类型，浮点观测按当前约定转为 FP32。不要为了初始化验证而读取所有数组内容；不增加全数据 finite 扫描。

必需 key 缺失时优先使用底层清晰异常，不用宽泛 except 包装为成功或跳过。原有加载手部数组时顺带检查 finite 可以保留，因为数组本来就要读取，不额外再扫一遍数据集。

### 5.2 SequenceSampler

继续负责 episode 内的索引、边界重复和字段读取长度：

- 索引不得跨 episode；
- action 使用方法当前的预测窗口，obs 只读取 obs_horizon；
- 先确定行和列，再读取数据，不先取完整 H 帧 RGB/点云再截取；
- 保留当前源行映射接口，让辅助目标与 stride 读取复用；
- 不新增 sampler 注册表、全局 padding mask 或窗口缓存。

本轮不统一改写各方法的 pad_after，也不缩短 CordViP 的逻辑 horizon 来伪装 I/O 优化。这些会改变窗口数量、末尾查询位置或监督分布。读取长度优化与改变训练样本不是同一件事。

### 5.3 Dataset 和方法差异

| 路径 | 必须保持的语义 |
| --- | --- |
| DemonstrationDataset | 返回 obs 与 action；数据增强和必要 RGB 预处理留在当前路径，affine 归一化留给模型 |
| R3DDataset | 辅助 action_ee 前 9 维与 joint action 按同一源行拼接；统计与训练布局一致 |
| AdapTacDataset | future force 从当前查询的下一时刻开始；末尾无下一帧的查询按现有规则处理 |
| InterActDexDataset | 使用实际 interaction_horizon 的未来目标；有效性按未裁剪的目标行计算 |
| AFROPretrainDataset | 只依赖 point_cloud 和 episode 边界；先应用 frame_stride 再读取，不要求 action/joint_state |
| CordViPPretrainDataset | hand/object XYZ、joint_state、contact_map 使用一致源行；contact_map 不归一化，也不混入 obs |
| VQ 数据准备 | 直接读取 joint19 的 action[:,7:19] 真实帧；不构造视觉 Dataset 或滑动窗口 |

未来目标越界可使用末帧作为填充值，但必须保持 valid=false，不能把 padding 伪装为真实未来监督。普通 BC 的既有 edge-padding 行为不在本轮改变。

数据约定继续保持 obs[t] 在 action[t] 执行前。action 的 commanded target 与 measured joint_state 不互相替代；action_ee 的含义不因重构改成测量状态。

增强不得写回共享存储。涉及标定投影的 RGB、点云和相机参数必须一致；不通过删除检查来允许不更新内参的 resize 或破坏对应关系的独立 XYZ 噪声。

## 6. 归一化：数据来源唯一，所有权明确

### 6.1 拟合总体

新的数据统计只使用本阶段 train_mask 对应的真实源帧，并应用与模型一致的通道选择、字段布局和确定性表示转换：

- 不包含 val 或 unused；
- 不包含 padding 重复，不按重叠窗口重复加权；
- 不包含随机增强、随机点抽样后的统计；
- point_cloud 的 XYZ/XYZRGB 统计遵循实际 pc_dim，不能混进未消费的列；
- R3D 的组合动作保持现有排列；
- identity、固定物理范围、继承 affine 不再扫描数据拟合。

这里的“源帧统计”适用于当前表示。若将来新增相对当前帧的动作表示，统计应对应其实际变换后的表示；本轮不为此建立通用转换图或实现尚不存在的相对动作。

Dataset 通过现有 iter_normalization_data 提供数据，模型持有唯一 normalizer，训练/验证/推理共用。同一新训练只拟合一次，resume 从 checkpoint 恢复，不重新拟合。

### 6.2 保持数值和字段含义

保留 limits、gaussian、rms、max_abs、identity 的已有数学含义。常量和全零字段应产生有限可逆的 affine，不因为“关节没动”而拒绝合法数据。

保留：

- action_ee 中 rot6d 的现有 identity 处理；
- 几何投影所需 XYZ、相机参数等物理坐标含义；
- 触觉 rms/max_abs 的全字段标量尺度与零 offset，不改成每个 taxel 独立标准化；
- FP32 模型 affine 和实际字段的多维 shape；
- DexLatent 的既有 URDF hand affine 及其内部 latent 统计路径。

R3D joint19+ee9 的既有归一化不顺手改动。不要以统一规范名义改变模型表示。

limits/max_abs 的输入原本为 FP32 时，可以保留 FP32 做极值规约，避免整块转 FP64；均值、方差、平方和累积保留所需高精度。只做局部修改并验证数值，不引入另一套统计实现或缓存。

### 6.3 预训练输入归一化的继承，包括 identity

模块对输入字段的归属明确写在已有方法初始化处：

| 产物 | 随模块继承的输入字段 | 下游新拟合 |
| --- | --- | --- |
| AFRO encoder | point_cloud | joint_state、action 等新训练部分 |
| CordViP encoder | hand_pointcloud、object_pointcloud、joint_state | action 等新训练部分 |
| DQ codebook | action 的手部 12 维 affine | 手臂部分；其他观测按各自配置 |
| DexLatent prior | 当前实现已有的手部物理范围/affine | 保持其余当前逻辑 |

AFRO/CordViP 继续使用小型 encoder.pt，保存 encoder_state 和 normalizer_state，无需完整 workspace、预训练数据路径或来源清单。

**identity 也必须继承。** 对这两个已知模块，产物固定覆盖表中的归属字段：normalizer_state 中某归属字段没有 scale/offset 表示该字段在预训练中为 identity。下游应保留 identity，不能把它当作“尚未初始化”后按下游 defaults 拟合。已有非 identity 字段直接加载实际 affine。

这是一条本项目 encoder 产物的明确约定，不用于猜测任意外部 checkpoint。部分缺失的 scale/offset 是损坏，不是 identity。用当前方法已知的字段集合处理即可，不新增资产类型注册表或自动推断框架。

导入的输入归一化优先于该字段的下游新训练默认值；保存的有效配置应与实际使用一致。用户无需手工保证两个 YAML 的归一化模式字面相同。冻结和微调都沿用输入归一化。

DQ 可以继续先拟合小型 action affine 再覆盖手部子段的直接实现；不要为省掉十几个数值通道的统计而新增复杂的部分字段拟合框架。最终手部 affine 必须来自码表，避免改变最近邻度量。

## 7. 配置与校验：消除流程限制，保留实际错误边界

### 7.1 YAML 是普通实验选项的来源

将 assemble_policy_config 中普通归一化默认项移到相应 YAML/配置组合，去掉“只要声明了固定字段就报错”的 overlap 禁令。保持本轮所有默认模式的实际数值行为。

- 可实验的参数正常通过 Hydra 覆盖，不要求绕开 Python 隐藏默认值。
- 不增设 normalizer_scope、strict_split_match、legacy_mode、allow_extra_fields 等开关。
- 不把少量方法相关配置复制成新的大型映射表或第二套配置系统。
- 几何、动作布局等确实影响算法正确性的约束在消费者处表达一次，不因为开放配置而允许错误的投影或表示。
- 明确派生值继续在一个位置产生，不保留 YAML 和 Python 两份会相互覆盖的事实源。

Dataset sensor_modalities 仍决定读取字段。模型观测检查改为“所需字段必须存在”；额外合法观测不再自动触发错误。内部 _preprocess_obs 对额外字段也应正常工作或直接忽略，不为了放宽入口在更深处又报同类错误。

归一化仅处理实际参与该数据路径的字段；缺失必需配置仍应清楚失败。允许额外观测不等于默默接受拼错的字段、未知归一化模式或必需输入缺失，不为容错新建字段猜测机制。

### 7.2 单一负责位置

| 检查 | 保留位置与要求 |
| --- | --- |
| episode 边界与数组时间轴 | EpisodeStore 打开时一次 |
| 划分参数和数量是否可满足 | split_episodes 一次 |
| 窗口长度、padding 的合法范围 | sampler 构造处 |
| 方法专用输入和辅助目标布局 | 对应 Dataset 或模型绑定处的一处 |
| 归一化结果 finite、scale 非零 | 拟合结束；外部预训练产物导入边界 |
| encoder 权重结构 | load_state_dict 的严格加载，不逐层另写重复扫描 |
| 码表 shape、hand affine | 码表导入边界一次；后续使用可信 buffer |
| 模型真实需要的动作/输入维度 | 最近的模型构造或绑定处，不在配置、Dataset、Trainer 三处重复 |

不删除实际防止 silent wrong 的检查来追求少行数。也不把所有类型、路径、字段一致性再包装成 validate_everything。

### 7.3 收缩 exporter 元数据契约

训练不应要求完整复述数据生成过程。梳理 validate_tactile_source 的消费者，保留实际计算与语义需要的 source、dt、单位/对齐信息；把“必须由某 exporter 生成”“format 必须恰好有/没有某值”“physics_dt×frame_skip 的生成一致性”从通用训练准入移除。

如果这些属性用于真实计算或部署，就继续保存并在对应消费者检查。不要臆造默认 dt、单位或来源来掩盖缺失。数据准备与部署已有的检查不为本次训练模块精简而全仓删除。

不添加广泛 except 后继续训练、不自动排除坏 episode、不用空对象或默认统计量掩盖失败。已有资源清理、原子 checkpoint 写入保留。

## 8. 预训练流程与产物

### 8.1 AFRO / CordViP

- 默认 val_episodes=0；保留已有 train-only 轻量入口。
- 显式 V>0 时建立一次验证视图，使用模型已有 affine，不重新 fit。
- 验证频率等配置只在启用验证时消费和检查，关闭验证不要求满足闲置验证配置。
- 验证关闭随机训练增强；沿用已有 eval/no_grad 与模型模式恢复。AFRO 验证不得更新 teacher。
- 验证指标作为预训练诊断，不新增 best/Top-K 框架。
- 固定预算完成后导出最终在线 encoder；不默认导出 AFRO EMA teacher。
- 保留 encoder.pt 与 resume.pt 的分工。前者供下游，后者保存继续预训练所需状态；不为省一个文件丢掉恢复能力。
- 保留已有冻结/微调选项，不另建初始化模式枚举。AFRO 的随机 encoder 却被要求冻结仍是实际配置错误，可保留一处提示。
- 当前 AFRO 仅点云读取、stride 后读取等优化已经正确，不重新搭建。

### 8.2 VQ 训练

默认只训练：

~~~text
选 train episodes → 读取真实 hand commands → 拟合 hand affine
→ VQ 训练 → last checkpoint → codebook.npz
~~~

V=0 时：

- 不加载 val hand 数组，不创建 selection_loader；
- 不每个 epoch 重新遍历 train 计算 lookup_mse；
- 不维护 train-best、selection_split=train_no_holdout 等回退状态；
- 正常记录训练本来产生的 reconstruction/commitment/total 等指标；
- 保留周期 checkpoint 与 last，结束时自动从 last 导出。

V>0 时：

- 验证数据使用训练 hand affine，允许超出 [-1,1]，不为统计好看而裁剪标签；
- 可沿用每 epoch lookup_mse 选 best：先用固定模型构造最终 prototype，再计算最近原型误差；
- 按实际样本数加权，保留尾批；eval/no_grad，不更新量化器 EMA；
- 结束时从该验证 best 导出；无验证样本或非有限指标明确失败，不回退 train；
- 不引入通用 metric selector、early stopping、Top-K 或第二套评估模型。

PCA 排序、量化组合、learned layer weights、FP32 近邻距离和连续索引算法保持当前含义。它们是算法，不属于应删除的工程机制。

### 8.3 导出与运行时码表的最小接口

export_codebook(checkpoint_path, output_path, ...) 接受明确指定的完整 VQ checkpoint。导出只需要模型结构/权重及 hand affine，不要求 checkpoint 包含 split_metadata、selection_metric 或“best”标记。last、周期或用户指定 checkpoint 都可导出。

codebook.npz 的运行必需内容缩减为：

| 键 | 含义 |
| --- | --- |
| sorted_hand_poses | [K,12] FP32、已按当前 PCA 规则排序的 normalized prototypes |
| hand_normalizer_scale | [12]，解释 prototype 坐标所需 affine |
| hand_normalizer_offset | [12]，解释 prototype 坐标所需 affine |

格式固定为 normalized poses，不增加 pose_space 选择器或自动模式识别。hand_dim 与 K 可由数组 shape 得到；num_groups、每组 codebook_size 和 VQ model_config 留在 VQ checkpoint/训练记录，不成为主策略运行条件。来源路径和选模指标可记在现有训练日志，不强制注入运行时 metadata_json。

HandStateCodebook 使用 hand_dim 与 num_codes，不继续用 num_groups/codebook_size 构造运行时协议。fresh 训练读取一次码表取得 K；把实际 num_codes 保存进主策略有效构造配置，之后从完整主策略 checkpoint 恢复表和 affine，不需要外部 NPZ 再存在。保持直接的数据构造顺序，不新增延迟加载器或模型工厂。

保留编码、解码的当前数学关系：连续索引按 K−1 缩放，预测先裁剪再按既有 half-up 取整查表。若保留 K=1 的既有支持，使用当前单码退化行为即可。不要把官方代码里的 TCP_DIM+HAND_DIM 分母当成通用码表长度公式。

normalized 空间的 PCA、近邻和 lookup_mse 不改为物理空间计算：逐关节 affine 会改变距离权重，换空间不是无行为变化的清理。

### 8.4 删除跨阶段绑定

删除 DQRISE.initialize_training 中完整 split_manifest 相等判断以及为此传播的 metadata。模型只关心产物能否正确解释和使用。

预训练可以有更多 episodes、独立数据、独立 seed 与独立 val；同源实验也可自行选择相同数据参数。实验公平性和使用额外数据的说明留在配方与报告，不编码成通用模型拒绝条件。

如果某实验的 val 用于评估整个训练流程，预训练也应避开这批 held-out 数据；这是该实验的划分选择，不恢复全局 manifest 校验系统。

## 9. 日志、checkpoint 与恢复

保留现有完整策略 checkpoint 的单文件设计、normalizer buffers、优化器/调度器/RNG 状态、原子替换和索引层恢复。

删除 effective_config 中的 dataset.split_manifest 及相关恢复比较。同一未改变的 Zarr、保存的数量参数与 seed 足以重建 mask；模型归一化直接从 checkpoint 加载。

只需记录：

- 输入数据路径、实际配置和 seed；
- total/train/val/unused episode 数；
- train windows，启用验证时的 val windows；
- 新统计量来自 train、哪些字段继承自预训练；
- 产物路径和必要训练指标。

不增加数据内容 hash、不可变来源注册、完整索引身份协议或数据搬迁准入程序。更改数据内容或顺序属于换实验，不承诺通过元数据证明它没有变化。

本轮不重构 resume CLI、日志尾部恢复、signal handler 或 optimizer 状态机制。若移除 manifest 需要调整它们的字段读取，局部修正即可。不要把“可复现”扩大为对 worker 随机增强逐位重放的新承诺。

主策略与 AFRO/CordViP 已有完整恢复路径，需要保持。VQ checkpoint 保证模型结构、权重与 affine 足以重新导出；本轮不为 VQ 新建一套完整训练恢复框架。

## 10. 配置、脚本与文档迁移

### 10.1 新默认配置

主策略公共配置：

~~~yaml
dataset:
  seed: 42
  max_train_episodes: 100
  val_episodes: 0
~~~

AFRO、CordViP、VQ 各自独立预训练配置：

~~~yaml
dataset:
  seed: 42
  max_train_episodes: null
  val_episodes: 0
~~~

显式预训练 holdout 示例：

~~~text
dataset.max_train_episodes=100 dataset.val_episodes=20
~~~

该例要求至少 120 条数据；150 条时保留 30 条 unused。主策略和预训练参数独立，不能把主策略 100 条上限暗中套用到全部预训练，也不能强制主策略复用预训练验证集合。

### 10.2 需要同步的文件范围

以下是导航，不要求为凑列表修改无关文件：

- dexmani_policy/datasets/split.py、demonstration.py、pretrain/__init__.py、pretrain/afro.py、pretrain/cordvip.py。
- dexmani_policy/datasets/episode_store.py、sampler.py、variants/r3d.py、variants/tactile.py：仅在接口或局部验证确有需要时调整。
- dexmani_policy/training/common/data.py、dexmani_policy/training/common/config.py、dexmani_policy/training/trainer.py，以及 dexmani_policy/common/config.py、dexmani_policy/common/normalizer.py。
- dexmani_policy/training/pretrain/obs/common.py，以及手部 vqvae.py、codebook.py。
- dexmani_policy/agents/hand_prior/codebook.py、agents/policies/dqrise.py。
- configs/common.yaml、各策略 normalization 配方、configs/pretrain/ 下全部相关 YAML。
- scripts/pretrain/ 下脚本，以及仓库内其他旧参数的实际调用方。
- docs/项目架构.md、docs/观测预训练.md、docs/手部动作表示.md、README.md、data/README.md 中受影响的说明。
- test_tools/pretrain_checks.py、agent_checks.py 及受影响的 smoke 示例。

更新文档中的：比例划分、共享 manifest 示例、trial 自动分组、默认 val、同源强绑定、VQ train-best、codebook payload、runtime groups 参数、归一化不可覆盖和 identity 继承约定。

不要只改默认 YAML 而遗漏 Python 构造参数或命令脚本。不保留 val_ratio 自动换算、旧参数别名和多版本码表 loader。旧产物不删除；需要重新导出时给出真实可执行的新命令。

任务书本身及历史研究记录可保留旧术语用于解释，不能据此要求全仓字符串零命中。

## 11. 建议实施顺序

1. **划分与默认值。** 先实现纯函数及数量例子，再串通主策略、两个观测预训练、VQ；删除旧协议与调用。阶段内即可跑一个合成 Zarr 样本检查。
2. **归一化与配置。** 确认 train-only、字段通道和 inherited identity；迁移普通配置默认值，减少重复检查。
3. **预训练分支。** 默认无 val；VQ 删除 train-best 回退；有 val 路径保持清楚的实际评估。
4. **产物和主策略连接。** 简化 codebook 导出/加载和 runtime num_codes；移除 manifest 比较；验证完整策略 checkpoint 自包含。
5. **局部 I/O 与数值清理。** 保留短观测读取，必要时消除无意义 FP64 极值转换或重复拷贝；不扩大到训练器性能重构。
6. **文档、验证与收尾。** 同步示例和既有检查，检查 diff，报告实际结果及限制。

每一步以完成行为为目标，不新建“阶段框架”、迁移任务管理器或要求用户逐步确认。遇到不影响目标的局部实现选择，采用最简单正确的方案继续。

## 12. 验证与验收

这些是行为验收点，不要求为每一行建立独立测试文件。优先复用 test_tools 现有检查；必要时集中补一个小型数据检查模块，不新增测试框架、CI 矩阵或大规模随机测试系统。

### 12.1 低成本、必须覆盖的检查

| 检查 | 关键断言 |
| --- | --- |
| 划分数量 | 第 4.3 节例子；互斥、无重复、unused 不落入 val；同 seed 重复结果一致 |
| 训练 seed 独立 | 改 training.seed 不改变由 dataset.seed 决定的 episode 集合 |
| 统计范围 | train/val/unused 放置明显不同数值；只有 train 影响新 affine |
| 窗口不污染统计 | 改 horizon/padding 不改变当前绝对字段的源帧统计 |
| 基本数值 | 常量/全零、多维触觉 shape、identity 无统计读取；已有 modes 的结果与参考计算相符 |
| 采样边界 | 小型不同长度 episodes、左右 padding、obs/action 不同长度；源行不跨 episode |
| 方法样本 | AFRO stride、CordViP contact 对齐、R3D 拼接、未来 force 与 valid mask 保持 |
| 预训练归一化 | 不同预训练/下游数据；继承 affine 与 identity 都不重新拟合；只拟合新字段 |
| 验证分支 | V=0 不构造/遍历验证对象；V>0 使用独立 val，尾批加权正确 |
| VQ 默认导出 | 无 val 只用 last，无 train selection pass；从不带选模 metadata 的周期 checkpoint 也能导出 |
| 码表衔接 | 不同训练集合不阻止加载；K、hand 维度、encode/decode 和 affine 正确 |
| 恢复 | 主策略与 AFRO/CordViP 的新格式 fresh→save→resume 使用保存的 normalizer；完整策略恢复不访问原 encoder.pt/codebook.npz |

可以在临时目录生成极小 Zarr 和码表。不要下载真实大数据或依赖网络预训练权重来验证纯数据行为。不要修改用户原数据验证异常情况。

### 12.2 复用已有工具

查看当前工具真实接口后运行相关检查，例如：

~~~bash
python -m compileall -q dexmani_policy
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
~~~

pretrain_checks 含接触数据准备检查，实际环境可能需要已有 preprocess/Open3D 依赖；缺少时明确报告，仍完成可运行的数据/归一化/码表检查，不伪称整套通过。

agent_checks.py 已有 check_sampler、check_codebook、check_codebook_pipeline 等函数。修改原本测试“split 不一致应拒绝”的断言，改为“数据不同但产物兼容应成功”；保留真实 shape、数值和推理往返检查。按修改范围调用相关函数，不为单项检查强行拉起所有 RGB、GPU 或模型下载路径。

如果环境具备训练依赖与数据，再用已有 smoke_test 对代表性策略做少量 fresh/resume updates。无需跑完整训练或 rollout 来证明一次数据模块重构完成。不得把静态核查当成训练验证。

### 12.3 性能验收

必须保持：不全量物化视觉 replay buffer、不额外扫描所有字段、不读取不需要的 H 帧视觉数据、不在 V=0 时做每轮全训练集码表评估。

若声称加速，用同一小数据、batch、worker 条件做简短前后对比，并说明测量范围；没有 profiling 就只报告消除了什么读取/计算，不给推测的加速倍数。不新增基准平台。

### 12.4 最终完成清单

- [x] 当前配置、构造参数、脚本与文档使用同一套 val_episodes 语义。
- [x] train / val / unused 三者准确，主策略默认用满实际可用的最多 100 条训练数据。
- [x] manifest、自动 trial 分组和跨阶段集合比较退出运行路径。
- [x] 数据统计 train-only；继承参数与 identity 正确；resume 不重新拟合。
- [x] 默认预训练不依赖 val；VQ 无 train-best 回退；任意完整 checkpoint 可导出。
- [x] 运行时码表只依赖实际表和 affine，不依赖 VQ 分组或选模身份。
- [x] 原有时序、辅助目标、接触标签、几何与动作含义未被“统一化”改变。
- [x] 必要检查集中，新增抽象/配置/回退没有超过已证明的需求。
- [x] 相关测试和短训结果如实记录；缺少依赖或未验证项明确列出。
- [x] 最终 diff 无无关模型重构、数据改写或用户改动丢失。

完成后用中文报告：改了什么、删了什么、保留了什么、运行了哪些检查及结果、是否有旧产物需要重新导出。不承诺未经实验验证的成功率或训练速度收益。

### 12.5 本次实际验证（2026-11-02）

使用现有 `policy` 环境在 CPU 上运行，下列检查均通过：

```bash
conda activate policy
python -m compileall -q dexmani_policy
python -m dexmani_policy.test_tools.dataset_checks
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
python - <<'PY_CHECKS'
import torch
from dexmani_policy.test_tools.agent_checks import (
    check_sampler, check_vq, check_codebook, check_vq_training, check_codebook_pipeline,
)
torch.set_num_threads(2)
for check in (check_sampler, check_vq, check_codebook, check_vq_training, check_codebook_pipeline):
    check(torch.device("cpu"))
PY_CHECKS
git diff --check
```

`dataset_checks` 覆盖数量例子、数据 seed 独立、train/val/unused 统计隔离、窗口和增强不污染统计、已有 affine 数学与多维触觉、identity 零读取、R3D 和未来触觉源行。AFRO/CordViP 主策略及 DQ-RISE 均实际运行 2 次 fresh updates 与 1 次 resume update；继承字段不重新拟合，删除外部 encoder/NPZ 后仍恢复。DQ fresh 单次读取码表并保存实际 K。

`pretrain_checks` 覆盖接触预处理的串行/spawn 一致性、失败清理、AFRO stride、CordViP 接触对齐、两方法 train-only、验证尾批加权、teacher 不更新、完整预训练恢复及导出。VQ 两分支各短训 2 epochs，验证无 val 时没有 selection loader/pass、默认 last、有 val 才选 best，以及不带 epoch/选模信息的完整 checkpoint 导出。码表检查包含 K=1/3/256/512、非 identity affine、FP32 距离、half-up decode、DQ loss/backward/推理和删除外部 NPZ 后的 raw/EMA 恢复。

最初 sandbox 内的 CUDA 检测返回不可用；经用户授权，在 sandbox 外确认 RTX 4090 可用（Torch 2.4.1+cu124），并补跑以下 CUDA 检查，全部退出 0：

```bash
python -m dexmani_policy.test_tools.dataset_checks --device cuda:0
python -m dexmani_policy.test_tools.pretrain_checks --device cuda:0
python - <<'PY_CUDA_CHECKS'
import torch
from dexmani_policy.test_tools.agent_checks import (
    check_sampler, check_vq, check_codebook, check_vq_training, check_codebook_pipeline,
)
torch.set_num_threads(2)
for check in (check_sampler, check_vq, check_codebook, check_vq_training, check_codebook_pipeline):
    check(torch.device("cuda:0"))
PY_CUDA_CHECKS
```

CUDA 检查实际覆盖 AFRO/CordViP 预训练的 train-only、验证、严格恢复与导出，以及三种主策略各 2 次 fresh updates 和 1 次 resume update。VQ 有/无 val 两分支各在 CUDA 上短训 2 epochs；码表、BF16 loss/backward、推理及删除 NPZ 后的 raw/EMA 恢复均通过。采样、统计和离线接触预处理仍在 CPU 执行。为保证设备选择真实生效，`dataset_checks` 增加 `--device`，`check_vq_training(device)` 消除原有硬编码 CPU。

未执行真实数据完整训练、rollout 或性能 profiling；以上结果不代表这些验证已通过。Store/Sampler 读取主体保留，V=0 的额外训练集选模遍历已删除，不报告推测加速倍数。

当前 encoder 独立导出与恢复命令见 [观测预训练](观测预训练.md#续训与评测)；完整 VQ checkpoint 的三数组码表导出命令见 [手部动作表示](手部动作表示.md#dq-rise-码本)。

### 12.6 后续系统审查与收敛

沿 Dataset、训练装配、产物导入、恢复、部署和仿真调用链完成复核，并直接清理以下问题：

- `fit_normalizer` 依赖已加载 affine 和最终模式，删除重复的继承字段参数；已有 affine 优先于 identity 默认值，缺失的必需配置在训练装配处明确失败。
- Dataset 声明但模型未消费的字段不要求归一化配置、不读取统计，也不进入模型的 `normalization_dims`。数据描述用 `field_dims` 表示声明字段维度；部署与仿真从保存的模型维度选择实际输入，避免要求未消费的传感器。
- DQ 初始化直接安装已验证码表，不再传入/返回同一 affine 或复制内部码表 buffers。`num_codes` 为必需构造参数，由 fresh 导入时取得并保存；VQ 删除不可达空 val 帧检查，验证 loader 统一命名为 `val_loader`。
- AFRO 在现有 Dataset 形状边界拒绝空点云，避免延迟到 PointNet 内部报错。保留真实坐标、有限性、动作布局、原子保存及资源清理约束。
- 操作文档仅描述当前接口，移除旧产物迁移和兼容流程说明。当前数据运行路径没有 SHA-256、来源身份或格式兼容协议。

当前版本的 `dataset_checks`、`pretrain_checks` 及 sampler/VQ/codebook 相关检查在 CPU 与 RTX 4090 上均通过。新增回归覆盖未声明归一化的额外观测、预训练输入模式由产物补齐、必需 action 配置缺失、公开部署 Runtime 仅凭消费字段完成预测，以及删除外部产物后的主策略恢复；AFRO 空点云负例单独验证通过。

`compileall` 与 `git diff --check` 通过。21 个相关 Python 文件通过局部 mypy 检查，使用 `--check-untyped-defs --follow-imports=skip --ignore-missing-imports`，不表示第三方依赖已完整类型检查。通过 `pip wheel --no-build-isolation --no-deps --no-index` 离线构建，并确认 wheel 包含当前检查模块、部署模块和预训练 YAML。

本轮未执行真实数据长训、闭环 rollout 或性能 profiling，不报告未经测量的性能收益。

## 13. 研究依据与采用范围

下面链接固定到本次核查版本。参考项目用于说明设计取舍，不意味着它们每个实现都应复制，也不据此断言整个项目所有 Dataset 都采用同一统计方式。

| 项目 | 核查位置与事实 | 对本任务的意义 |
| --- | --- | --- |
| Diffusion Policy | [PushT Dataset](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/pusht_image_dataset.py) 在完整 replay buffer 上拟合 | 全量统计有实际先例，train-only 是本项目明确选择，不是行业唯一规范 |
| DP3 | [Adroit Dataset](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/dataset/adroit_dataset.py) 也直接取完整 buffer | 保留简单 Store/Sampler 思路，不照搬统计范围 |
| ACT | [utils.py](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/utils.py) 对指定全部 episodes 计算统计，供 train/val 共用 | 统计数据范围与 loader 采样范围必须分开看 |
| robomimic | [训练入口](https://github.com/ARISE-Initiative/robomimic/blob/d309eaecc18acf4152a830a895a6984b8ac71b05/robomimic/scripts/train.py) 从 trainset 取得统计量 | 支持按训练集合拟合并共享到评估 |
| Octo | [dataset.py](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/dataset.py) 默认统计路径使用经过其过滤的 split=all，也可传入统计 | 不把大型多数据集基础设施移植到个人研究项目 |
| OpenPI | [compute_norm_stats.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/compute_norm_stats.py) 对配置选择且经过输入变换的数据计算统计 | 统计应匹配实际表示；无需复制独立资产/缓存机制 |
| LeRobot | [Dataset 工厂](https://github.com/huggingface/lerobot/blob/17aea13fc09111ba2c0f86453ca5a5fe5e779ba3/src/lerobot/datasets/factory.py) 支持 episode 选择并使用 dataset metadata stats | 不能仅见 episode 参数就认定统计已针对该子集重拟合 |
| DexPort | [train.py](https://github.com/amazon-far/dexport/blob/6143c51704e57a423f8bd73500f762e0ebb32daf/dexport/minbc/train.py) 将训练集统计传给 test；[Dataset](https://github.com/amazon-far/dexport/blob/6143c51704e57a423f8bd73500f762e0ebb32daf/dexport/minbc/data/dataset.py) 对相对动作在窗口表示上统计 | 采用 train-only 的直接数据流；不引入其强制 test、动作 MSE 选模和跳过坏轨迹机制 |
| AFRO | [预训练入口](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/AFRO_Pretrain.py) 创建 val/Top-K 但循环未用于选模，保存 pretrain_latest；[下游 Policy](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/policy/afro_policy.py) 提取在线视觉 encoder 并冻结 | 默认预训练无需验证选模；下游不加载 teacher/IDM/FDM |
| DQ-RISE | [train_vqvae.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/train_vqvae.py) 无验证循环；[eval_vqvae.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_vqvae.py) 解码组合、PCA 排序并保存表；[train_dqrise.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/train_dqrise.py) 直接加载表 | 默认 last、离线导出固定码表、删除跨阶段数据身份要求 |

### 不照搬的官方实现细节

- AFRO 的 [Adroit Dataset](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/dataset/adroit_dataset.py) 用完整 buffer 拟合，并用 ~train_mask 构造 val；限制 train 数量后，unused 会被纳入 val。这里必须遵循第 4 节。
- AFRO 官方下游只提取视觉权重，再从下游数据构造 normalizer。本项目保留随 encoder 导出的输入归一化，以支持不同预训练语料。
- DQ-RISE 实际入口由 [EXAMPLE.md](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/assets/docs/EXAMPLE.md) 指向根目录 train_vqvae.py；不要依据子目录遗留模板设计本项目流程。
- DQ-RISE 的 [PretrainDataset](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/dataset/pretrain.py) 使用固定设备量程，而本项目 XHand 使用弧度和 train-only affine。保持本地输入语义，不复制 RoHand 的 6 维或 65536 量程。
- 不复制官方 checkpoint 的宽松加载、未使用 loader/manager、每 batch 分布式 barrier 或依赖命名拼接的产物定位规则。

## 14. 执行原则

这份任务书已经确定主要取舍。执行者应读取实际代码、完成实现、更新调用方、运行相关检查，再给出结果；不重复进行一轮开放式方案评选，也不只提交新的计划文档。

如果发现当前代码与基线不同，按最终行为目标作局部适配，尊重已有用户改动。只有影响目标且无法从仓库证据解决的歧义才需要询问；普通命名、局部函数拆合、测试组织由执行者判断。

遇到本文要求与真实数学/时序正确性冲突，先说明具体冲突和证据，再采用最小正确修正。不得通过恢复 manifest、扩大校验体系或加入新模式来回避问题。

