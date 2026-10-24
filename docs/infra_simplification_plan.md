# DexMani 基础设施简化方案

> 状态（2026-10-25）：A–E 已实施并完成可用的离线验证；F 未发现足以支持性能改动的测量证据，保持数值和加载默认。逐批改动、协议变化与未验证项见 [实施状态](infra_simplification_status.md)。下文“现状”仍指审阅基线，不是当前源码说明。
> 整理日期：2026-10-24。范围：数据集、训练、仿真评测、策略推理与真机部署边界。
> 执行入口：[Codex 执行 prompt](infra_refactor_codex_prompt.md)。现行行为以源码及[项目架构](项目架构.md)、[仿真评测机制](仿真评测机制.md)为准。

## 1. 目标、版本与适用边界

目标是让同一语义只有一个负责位置，减少配置来源、重复结果结构、方法分支和状态所有者，同时保持研究正确性、运行效率和可追溯性。保留现有 Dataset/Sampler、策略 Trainer、Policy、PolicyInfo 和 EvaluationRequest；优先缩小职责，不新建通用框架。

审阅基线：

| 仓库 | 已核对提交 |
| --- | --- |
| dexmani_policy | `cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74` |
| dexmani_real | `f2ec75621ede85d24fce8b189b348405926f642f` |
| R3DP | `4b2703e272b15a9190c6842a82bb0c340f3393fa` |
| PEAfowl | `ed95a3704e8060213a74f673f7d86b3d4a1a8fa4` |
| LingBot-VLA-v2 | `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` |

实施前重新检查实际 HEAD、工作区、各仓库 AGENTS.md 与调用者；新代码已经解决的问题不重复改造。下文“现状”绑定上述版本，“目标”是工程决策，不代表参考项目原样采用，也不代表已有性能收益。

范围内包含必要的 dexmani_real 配套修改；机器人驱动、标定、遥操作和 Raw 格式不因本方案重写。模型架构、损失、先验、动作表示、训练窗口及超参数不是通用基础设施清理对象。

### 1.1 定稿时消除的歧义

| 前期讨论中容易误读的说法 | 本方案的最终约定 |
| --- | --- |
| 由 Runtime 统一裁剪动作 | Policy 完成时间前缀与辅助维度裁剪；Runtime 只做数值适配 |
| 所有 normalizer 都对源行去重 | 普通绝对量按源行去重；相对动作按窗口与参考状态统计 |
| 把 n_action_steps 全部移到执行端 | 先固定其对训练 padding 的有效影响，再分离执行长度 C |
| 增加 EvaluationPlan | 沿用 EvaluationRequest，计划只是已解析参数和明确 seed 列表 |
| 完整恢复等于逐位重放 | 保留完整训练状态；worker 增强、预取及非确定性 kernel 的保证另行说明 |
| 统一同步、异步动作时间轴 | 保留各模式真实的槽位与接管语义，只统一数值输出 |
| 真机默认只保存结果 | 标准真机评测结果必存、Raw 可选；本轮不静默改变现有 Raw 默认值 |
| 删除所有旧接口同时保证所有历史产物在新代码可用 | 本次配套源码与当前产物须一致；历史实验可使用对应旧提交，不维护通用兼容层 |

### 1.2 不引入的机制

不迁移训练存储格式，不引入通用 registry、插件式调度、统一大 Runner、默认 RPC、自动重试恢复、自动调参、第二种推理权重格式或持久缓存管理系统。不新增哈希计算、指纹或自定义兼容性摘要；记录已有 Git commit 和明确数据身份即可。


### 1.3 实施时的优化调整

- 复用原有 Policy、EvaluationRequest 和 split manifest，不增加 registry、缓存池或第二份输出格式。
- n_action_steps 仍被方法构造与默认执行长度实际使用，因此保留；去掉它与训练 padding 的插值联动，固定原有效值。
- 数据划分 seed 与训练 seed 分离，但 MultiTaskDataset 的运行采样 seed 继续跟随 training.seed；两者不是同一随机性。
- 输出目录属于运行选项，交给 evaluate_policy，不混入冻结研究请求；一个请求只执行一个 NFE，需要扫描时由外层多次调用。
- 当前配方明确使用 selection=25/test=70；固定预算与成功步数均值修正均为协议/统计变化，不声称和旧选点完全等价。
- 正式测试仅允许 device/output 改变；包括录像在内的其他请求条件在选择时固定。普通 best 默认继承选择的推理参数，但仍标为普通评测。
- PolicyInfo 读取实际构建产生的元信息；Real 入口解析 best，Runtime 不遍历选点证据、不持有调度模式。Raw 默认 true 保持，标准轻量结果必存。
- 有真实 trial 身份才分组；索引 manifest 只有匹配的稳定 source_id 才允许路径搬迁。历史数据不补造身份、不重写。

## 2. 参考项目的选择性借鉴

| 项目及证据 | 实际机制 | DexMani 的取舍 |
| --- | --- | --- |
| [R3DP Dataset][r3dp-data]、[Workspace][r3dp-workspace] | Zarr、时序窗口、normalizer 与训练连接；EpisodeBatchSampler 按 episode 内固定间隔组批 | 保留窗口采样与训练循环分离；专用时序组批只属于相应方法 |
| [PEAfowl 部署][pea-deploy]、[数据构建][pea-config]、[训练入口][pea-train] | 训练与部署复用 build_transforms；部署用验证变换；训练依赖外部 SimpleTrainer | 复用确定性变换，继续使用现有 Trainer，不迁入另一套训练框架 |
| [LingBot 数据说明][ling-data]、[评测说明][ling-eval] | 显式字段映射、多数据集运行时组合；常驻模型，推理与仿真环境分离 | 借鉴字段契约、无合并副本的数据组合和模型常驻；仅有实际依赖隔离需求时使用进程通信 |

参考实现并非所有方面都应照搬：R3DP 的 normalizer 读取整个 replay buffer；PEAfowl 部署固定截取 32 步。DexMani 的训练集统计和方法专属 horizon 应保留。LingBot 公开评测说明区分 FP32 与 BF16 的结果，因此不能把训练精度自动当成评测精度。

## 3. 唯一责任与依赖方向

| 机制 | 负责位置 | 其他模块的行为 |
| --- | --- | --- |
| 物理字段、单位、关节顺序、坐标系、采样周期 | 数据配方与实际数据元信息 | 消费保存事实，不从模型名猜测 |
| episode 划分及训练子集 | 固定 split manifest | 正式实验复用，续训使用原划分 |
| 窗口、padding、目标对齐、统计总体 | Dataset/Sampler 与方法数据适配 | Trainer 接收 batch，不重构学习问题 |
| 确定性 RGB 等预处理 | 现有共享数值函数 | 数据验证、仿真、真机复用 |
| 归一化、方法 context、解码、完整未来动作 | Policy | Runtime 和 Runner 不二次解释 |
| optimizer、scheduler、EMA、更新、checkpoint | 现有训练构建组件与 Trainer | 入口负责启动 |
| 完全解析的评测请求 | 评测入口与 EvaluationRequest | rollout 只执行，不读取多层配置 |
| episode 事实与统计 | rollout 产生记录，统计函数统一聚合 | 选点、终端、文件共享相同结果 |
| 模型加载、设备、预热与数值适配 | PolicyRuntime | 不拥有硬件时钟和动作计划 |
| 轻量部署输入输出检查 | PolicyInfo | 不加载 Torch 即可读取保存元信息 |
| 观测历史、执行历史、调度、设备调用 | 各自的仿真/真机执行端 | 使用同一 Policy 数值语义 |

Dataset 和 Policy 不依赖训练入口；部署 Runtime 不依赖评测执行；真机硬件层不进入模型包。数值函数可以共享，仿真与真机的执行循环分别维护。

## 4. 数据集：保留 Zarr，固定实验总体

### 4.1 数据链路

保留 Raw → Canonical Zarr → 固定 split → 窗口采样 → batch。Raw 是证据，Canonical 是可重建派生数据；训练只读取需要的字段。保持真机“控制命令/提交尝试目标”与“实测状态”的区别，不能用下一帧关节反馈替代动作目标。

保留惰性读取、进程独立句柄、有界 chunk 缓存、不跨 episode 取样、有限值筛选和训练集统计。确定性几何处理的离线与在线实现须具有相同数学语义；不为每种策略导出另一份数据格式。

### 4.2 split 与随机性

现状：DP3 等配方的 `dataset.seed` 插值到 `training.seed`，轨迹数超上限时改变训练 seed 也会改变训练子集。[证据][dp3-config]

目标：

- 正式方法比较使用已有格式的固定 manifest；划分 seed 与训练 seed 分离。
- manifest 包含实际 train/val episode 身份；同一原始 trial 的切片在有对应身份信息时成组划分。
- seed、val_ratio、max_train_episodes 属于划分生成；显式 manifest 的训练消费遵循现有互斥检查。
- 实验准备可在启动阶段完成，无需强制用户多执行一道命令；一组比较必须引用同一个已固定 manifest。
- 恢复使用保存的实际划分，不按新默认 seed 重新抽样。
- DQ/LAMP 的先验与策略继续共享划分；LatentFlow 独立同构 codec 语料及同源 holdout 检查保留。
- 数据重导出使用新路径，保留来源身份和边界；不改写旧数据或补造稳定 source_id。

这属于新实验协议。旧实验继续按其保存配置、实际 episode 集合和对应源码解释。

### 4.3 统计总体

| 类型 | 保留的总体 |
| --- | --- |
| 普通物理观测与绝对动作 | 有效训练窗口实际消费的源数据行，按字段去重 |
| LatentFlow 相对动作 | 每个训练窗口的命令减该窗口参考状态，不能只按命令源行去重 |
| 多任务 | 显式约定覆盖的任务集合；采样概率与统计权重不是同一个机制 |

[LatentFlowDataset][latent-data] 已按窗口/reference 统计；[MultiTaskDataset][multi-data] 当前统计遍历所有 child，包括零采样权重任务。等价重构保留这两种现状。排除零权重任务或改变统计权重须作为独立数据协议变更，不能借整理代码悄悄实施。

模型损失使用的增强、几何坐标和 reference_state 关系保留。LatentFlow 的参考状态来自增强前的原始窗口语义，不能由公共转换层顺手替换。

### 4.4 准备与热路径

数据/配置读入时检查完整性，准备有效窗口与统计；取样热路径只读取、构造目标、应用训练增强。不重复扫描同一数据，不另建自动缓存失效系统。DDP 现有父进程一次准备数据与统计的机制保留。

## 5. 时间与动作契约

定义 N 为观测历史长度，H 为模型预测 token 数，s 为解码后历史前缀长度，P=H-s 为完整有效未来动作数，C 为本次消费长度。

| 当前方法例子 | s | P |
| --- | --- | --- |
| 普通策略的现有窗口 | N-1 | H-N+1 |
| LatentFlow | 0 | H |
| LAMP 的当前单步配置 | 0 | 1 |

这些值由具体 Policy 和数据窗口决定；不能据此硬编码所有未来方法。普通策略与 LatentFlow 的训练目标起点不同，继续保留。

### 5.1 执行长度不再隐式决定训练 padding

DP3 当前 `n_action_steps=8`，`dataset.pad_after=n_action_steps-1`。[证据][dp3-config]

迁移时先保存等价有效值，例如该配方继续 `pad_after=7`；执行端的 `action_steps=C` 单独决定重查询频率。已有 eval.action_steps 覆盖本来就不改变训练 padding，这一行为必须保留。不得把 pad_after 改成 P-1 或 H-1，也不能将所有方法改成同一种窗口。

阶段内同步所有构造、配置和消费者；保留参数只为完成当前迁移，不新增永久兼容别名。参数最终去留以调用者盘点为准。

### 5.2 Policy 的唯一控制输出

目标输出为 `future_actions: Tensor[B,P,D_control]`：

- 已完成反归一化与方法专属码本/latent 解码。
- 已裁掉训练历史前缀和辅助监督维度。
- 索引 i 对应最新观测查询槽位 t 的命令 a[t+i]；第 0 项是 a[t]，不自行平移到 t+1。
- 动作是控制目标，不是预计下一帧实测到位状态。
- 单位、坐标和关节顺序来自保存配方。

先完整完成方法所需的 H 步解码，再取得有效未来段；执行 C 不能提前截断 LatentFlow 等方法的整段解码。

[BasePolicy][policy-base] 统一格式化；[仿真 Runner][base-runner] 和 [PolicyRuntime][policy-runtime] 删除对 pred_action 的二次起点/辅助维度裁剪。仿真通常消费前 C 步；真机按调度语义消费有效部分。

仓库内及 dexmani_real 的消费者、smoke、测试、示例须同时迁移。完整原始预测若有真实算法/诊断消费者，留在明确内部接口；控制接口不长期同时维护 pred_action、control_action、future_actions 三套同义出口。

等价阶段保持现有边界 dtype（包括 Real CPU NumPy float64）与预处理结果，不借统一接口改变数值精度。

## 6. 训练与 checkpoint

### 6.1 保留一个策略训练主干

保留当前 Trainer 和两个薄启动入口；本方案不要求迁移 torchrun、Accelerate、FSDP 或强制合并单卡/DDP 启动。手部先验预训练保留真实阶段差异，不强塞进策略循环。

按 optimizer update 定义 scheduler、EMA、日志和保存进度。保留梯度累积尾组语义、DDP no_sync/同步边界、非有限值失败、EMA teacher、参数分组与恢复检查。

### 6.2 builder 收回方法分支

[builder][training-builder] 当前检查 backbone、pc_encoder、task_emb_table 和特定 LatentFlow metadata。目标是：

- Policy/组件提供实际生效的结构配置，公共 builder 不遍历内部结构。
- 复用现有 initialize_training、validate_training_data 和组件 effective_config 等接口。
- 数据描述、动作语义、跨阶段兼容性纯函数移出 training.common.hand_representation；训练批次与 KL schedule 等训练逻辑留在训练侧。
- 方法专属检查留在方法模块，不新建 registry、能力字典或每事项一个 hook。
- 配置合并一次，构建完成后保存实际有效值；保留默认构造参数被固化的能力。

### 6.3 保存与恢复

保留模型/normalizer、raw/EMA、optimizer、scheduler、EMA 更新计数、RNG、epoch/update/采样游标和有效配置的完整保存。继续原子写入，latest 只是最近完整保存点的便捷入口。

构造结构与外部预训练初始化分开；完整 checkpoint 恢复不能依赖已经删除的外部 codebook/prior/codec 初始化文件。保持 strict 加载和必要的配置/数据/优化器参数顺序检查。

历史实验以保存配置和对应源码为准，不以当前 YAML 默认值补齐。现有 config.yaml 的轻量检查视图与 checkpoint 内配置须来自同一次构建；读取时在边界核对，不发展为两套可独立编辑的事实。

[get_rng_state][rng-code] 不包含 worker 增强和预取状态；完整恢复不承诺多 worker 随机增强逐位复现。要求严格增强重放时再引入由 seed、epoch、逻辑抽样位置确定的局部 RNG；本轮不增加 DataLoader 状态序列化系统。

## 7. 仿真评测：执行、统计、选点分离

### 7.1 一次解析为明确请求

沿用 EvaluationRequest；最终请求包括具体 checkpoint、use_ema、单个 inference_steps、action_steps、实际推理数值设置、逐任务环境/输入条件与有序 seeds、输出位置。任务文本按保存 child 配方解析，不能用新的默认文本替代。

两种入口来源互斥：

1. 普通评测/调试：checkpoint 与显式评测配方，启动合并一次。
2. 正式测试：完整选择记录中的冻结协议；只允许设备、输出位置等不改研究条件的运行选项。

正式测试冻结权重、NFE、C、精度、环境设置、任务/语言条件与 seeds。改动研究参数时产生新的普通评测，不能继续标为沿用原协议的正式测试。不能因修改了设备而宣称逐位相同，实际硬件与软件仍写入记录。

删除 inference_steps_list 与标量协调、多层 eval/offline/demo 回退、重复参数别名；使用一种规范配置路径，不为同一字段同时增加专用 CLI 和 dotlist 优先级。best/latest/百分比 tag 可以在入口保留便利解析，随后冻结到具体 checkpoint；执行层不理解别名或实验目录三元组。

### 7.2 rollout 不知道 selection

现状：[evaluate_checkpoint][eval-rollout] 在直接评测无选择记录的 checkpoint 时仍会预留 selection seeds，并读取选点预算。

目标：

- rollout 只执行收到的任务与 seed 列表，不生成或修改 seeds，不读 selection 配置。
- 正式实验开始前固定 test 与 selection 清单并检查互斥；同组候选使用相同清单。
- test 清单不随训练 seed、候选数或是否发生平局而变化。增加 selection 预算不能改动已有 test 清单。
- 缺少/重复/越界 seed 与池容量问题在执行前报错；正式测试缺少来源清单不静默回退到 range(100)。
- 单独普通评测只需要本次 seeds，不被无关 selection 容量阻塞；没有有效选点证据时不伪标 heldout_from_selection。
- 环境 RNG 与策略采样 RNG 的重置规则保存并在每个 episode 执行；预热不消耗正式 episode 的随机轨迹。

### 7.3 固定预算选点

删除“一次初评＋一次平局追加”及相关 batch_size/max_episodes；所有候选每任务使用相同 K 个 selection seeds。25/70 等数量仅是实验配方起点，不能成为框架常量，也不承诺统计上最优。

选点执行复用普通 evaluate；排序部分是纯结果函数，核对预先声明的候选、共同评测条件与完整结果。默认沿用现有排名：成功率、成功 episode 的有效首次达标步数均值、较高 global_step；全部失败时步数不可用，仍按声明的确定性规则排序。预算变化单独标明新协议。

若 NFE、C、精度或权重类型参与选择，它们必须成为明确候选条件，在 selection 集比较，测试前冻结；不根据 test 再挑设置。固定最终 checkpoint 的实验可以省选点，但基础设施不强制改变现有实验习惯。

### 7.4 一个平面 episode 结果结构

所有任务统一产生 task_name、seed、success、steps（首次 success_condition，未达标为 None）、total_steps 等事实。最终 success 保持环境的 hold/grace 语义；命令历史仍读取实际应用的控制目标。

单个 checkpoint/请求的结果保存：

- 完全解析的 request、来源与代码/环境信息。
- 预期 task/seed 清单、已完成的平面 episodes。
- running/completed/failed 状态及明确错误。
- 统一统计函数计算的派生 statistics。

小规模评测按 episode 原子更新一个结果 JSON，先保存请求；异常保留已经完成的记录，不能发布部分结果为完整成功率。技术错误不伪造 success=False，不跳过或替换 seeds；恢复/重试若以后有实际需求，仍必须针对原任务 seed，不能自动改变评测总体。

单任务与多任务都只调用同一个统计函数，终端与结果文件使用同一值：

- micro：全部正常 episode 的总成功率。
- macro：各任务成功率等权平均。
- 平均成功步数：成功且有有效首次达标步数的 episode 级浮点均值。
- 取整只用于显示；不能进入选点排序。
- Wilson 区间不解释为训练 seed 方差或宏平均不确定性；保留当前适用范围说明。

[MultiTaskSimRunner][multi-runner] 当前对任务步数均值再取平均，[最终评测][eval-rollout] 则按成功 episode 汇总。统一上述定义是可见统计修正，应标注，不能宣称所有旧显示数字逐位不变。

去掉嵌套 per_task 执行返回、collect_episode_details 展平和多处聚合。多任务是普通任务循环；同一 checkpoint 的模型常驻。外层 NFE 扫描可复用已加载模型，但每次请求仍只含一个 NFE，各 episode 正确 reset，不新建模型缓存池。

### 7.5 Runner 与入口

目标保留一个具体 SimRunner 承担环境生命周期和单 episode 交互；在核对所有调用者后合并只有一个生产子类的 BaseRunner，移除多任务专用包装。dexmani_sim 延迟到环境创建时导入，使数值测试不依赖仿真安装。

record_demo 归为同一评测的录像/显示选项；pipeline 只是薄顺序调用，不重新解析协议。视频/显示依赖检查保留；录像错误与任务技术错误按已有输出语义区分，不把失败视频登记为成功产物。

selection_result 记录候选结果引用、冻结协议与选中结果；best_ckpt 仅作可选入口指针，不能成为另一份独立事实。读取选择证据时检查完整性、候选与协议一致及 seed 隔离，部署 Runtime 不重新遍历整条证据链。

现有环境定期重建先保留，验证完整 reset 与资源稳定后才删除。清理输出别名/旧文件前查明脚本消费者；当前格式一次性同步迁移，不维护并行输出格式。

## 8. 共享推理与 PolicyInfo

### 8.1 数值接口

仿真和真机共享模型加载、确定性预处理、Policy 输出契约及必要转换，不强制仿真经过 NumPy-only 部署入口。仿真已经有 Tensor 时保持 Tensor；真机 NumPy 输入转换一次；仅在执行端需要时转 CPU 数组。

PolicyInfo 保持无 Torch 检查能力。action_start_index/future_steps、control_action_dim、context 需求、支持的执行方式等少量事实由实际 Policy/数据构建自动写入已有有效配置，禁止手工覆盖；元信息只保存当前消费者确实需要的内容。已有字段直接复用，不额外建立通用 schema/PolicySpec 系统。

加载时一次核对 checkpoint、保存元信息与实际 Policy。此后内部使用确定契约；现场输入和模型输出的有限值、维度及动态执行边界继续检查。

### 8.2 方法 context

- LAMP hand_history 来自执行约束后的已提交/应用手部目标，初始历史按原实测手姿规则补齐。提交不等于 ACK 或实测到位。
- LatentFlow reference_state 保留原始查询参考语义；显式输入与省略后的默认来源按方法约定处理。
- Policy 声明和解释字段；执行端掌握真实执行事实；Runtime 只转换数值载体。
- 不为消除两处分支而构建 context registry，也不以模型预测队列冒充执行历史。
- 当前 MultiTask 真机 task/child 配方尚未接通，不能通过公共重构假装支持。

### 8.3 模式与采样分离

Runtime 不保存 sync/async 执行模式，不调用 decoder 内部 scheduler 来配置真机调度。

- Real 入口读取 Policy 的 supported_execution_modes 并校验适用性。
- LAMP、LatentFlow 当前仅 sync，限制保留；不是删除模式配置后自动开放 async。
- 普通 predict 接收观测与必要 context，返回完整未来动作。
- RTC 前缀/延迟/指导参数只走支持它的方法；采样器检查其数学约束。
- RTC 的局部 autograd 与 FP32 schedule 处理保留，不能套全局无梯度或统一低精度假设。
- 模型缓存/RNG reset 与执行历史 reset 各归其所有者，episode 边界同时正确触发。

### 8.4 部署与评测解耦

best/selection 只在启动入口解析为明确 checkpoint 与推理设置；加载后的 PolicyInfo/Runtime 不读取评测阶段、候选排名或目录命名。保留现有 checkpoint 恢复能力，不另导出一套默认权重格式。加载时不依赖训练 Dataset、optimizer 构建或外部初始化资产。

## 9. 真机部署：保持明确的资源和状态所有者

### 9.1 生命周期

保留 [InferenceWorker][real-worker]、[schedule][real-schedule]、[Episode][real-episode] 与 [Session][real-session] 的实际边界：

| 所有者 | 状态与责任 |
| --- | --- |
| Session | 设备/模型资源、启动、关闭和 session 记录 |
| Episode | 观测历史、真实命令历史、当前计划、query/run 身份、时间锚点 |
| InferenceWorker | 唯一模型线程、在途任务、完成结果、串行 reset/close |
| 现有 RuntimeState/robot | HOME、运动授权、停止与急停事实 |
| schedule 纯函数 | 根据显式状态与时钟做决策，不拥有第二份状态 |

最多一个当前计划和一个在途查询，不积压旧观测，不新增消息总线、自动恢复或通用异步调度器。Future 完成不等于动作有效；run/query 身份、决策年龄与实际提交时授权仍需检查。已终止 episode 的迟到结果只被回收和记录，不能用于下一轮。

### 9.2 时间语义

C 是名义执行/重规划长度，异步可能消费旧计划的仍有效尾部；记录实际消费，不把名义 C 当作每次必然执行步数。

d=j-k 仅适用于查询与执行共享连续控制槽位的异步路径。同步等待、首次 bootstrap 与 RTC 接管沿用现有语义，不统一按墙钟延迟/控制周期裁掉前缀。P >= d+C 是覆盖条件之一，不替代实际过期、剩余长度与接管检查。

保持 max_decision_age、max_wait、episode 时长及设备停止边界的各自含义；不能合成一个 timeout。暂停逻辑槽位不能掩盖墙钟观测过期。硬件 IK、限位、速度和停止措施留在 Real。

授权与新鲜度是时变量，必须在提交前检查；静态元信息才适合启动时检查一次。线程中的 CUDA/原生调用不能因 Future 超时被假定已经取消，关闭状态须如实记录。

### 9.3 观测构造与缓存

保留最新 Real 按观测时间戳复用当前历史窗口内 FK 特征的有界缓存，并在 reset 清理。[证据][real-observation] 这是同帧确定性计算复用，不是跨时刻近似视觉特征复用。

保留训练表示与现场物理设置的边界：数值表示及控制 dt 读取 artifact；当前有效设备、URDF/安装与标定按 Real 规则处理，不把历史物理标定硬编码成长期准入条件。

模型加载与合成预热当前发生在设备连接之前，先保留。不要为了少写 warmup 输入而改变硬件连接顺序；预热后的 RNG/缓存状态须回到正式 episode 的正确起点。

### 9.4 结果与 Raw

目标标准真机评测始终保存轻量配置、episode 结果、终止原因和必要延迟信息；Raw 录制可选，保留两种组合“结果”与“结果+Raw”。移除正式评测的 record_results=False 组合前盘点实际调用者，不额外增加 formal/debug 模式来替代旧开关。

本轮保留既有 Raw 默认值；如特定性能实验只保存结果，在实验配方显式关闭 Raw，单独记录运行条件变化。Raw 会扩展 RGB-D/触觉等传感器与资源需求，控制需求和录制需求的区分保留。

正常结束不自动表示任务成功；保留原有人工/任务结果判定。录制、控制和清理失败的归属沿用 Raw 规则，不能删除失败证据、改写提交目标或因为结果必存就吞掉 I/O 错误。无物理运动授权不连接硬件或执行真实 rollout。

## 10. 减少配置、检查与性能开销的边界

### 10.1 可以集中

- 配置、数据与 checkpoint 输入边界的静态契约检查。
- 保存配置的解析和模型结构恢复。
- 一次输出裁剪、一次必要设备转换、一次统计。
- 输入读取采用已有选字段与有界缓存。
- 结果与文档的同一机制保留一个维护位置。

### 10.2 不可按“重复”删除

- 每次外部观测/输出中新出现的 NaN、维度或类型错误。
- DDP 集体失败与累积边界。
- 提交时授权、停止、新鲜度、deadline 和执行约束。
- 产物原子发布、不可覆盖和必要资源收尾。
- 方法必需的 EMA、历史、先验、辅助目标和相对动作统计。

### 10.3 性能实验次序

先使用已有 data_wait_time、step_time、samples_per_sec 和推理耗时定位；host data wait 与异步 GPU 计算可重叠，不能相减得到精确 GPU compute time。

按证据尝试：减少重复转换/构建 → 数据加载参数 → 推理精度/compile → selection 集上的 NFE/C 对照 → 确有瓶颈时再考虑加载或几何编码优化。保持当前数值默认，候选精度不自动替换正式配方。

分别报告冷启动/编译、预热后模型推理、包含预处理与回传的完整推理、真机 source age 与接管/提交延迟。少量 warmup 样本不是可靠 p95 估计；有足够样本并固定输入/条件后再报告分位数。吞吐提升不能代替物理动作等价或闭环质量验证。

## 11. 实施批次、文件落点与完成标准

按依赖顺序实施，每批独立说明行为是否改变；不把多个研究假设塞进一次提交。

| 批次 | 修改范围与主要落点 | 完成标准 |
| --- | --- | --- |
| A：保持行为的边界收敛 | hand_representation 纯语义迁出；builder/Policy 专属检查；有效配置来源；common/checkpoint | batch、统计总体、损失和训练状态不变；外部先验删除后完整 checkpoint 仍可恢复 |
| B：统一数值推理 | agents/policies、common 预处理、deployment、env_runner、dexmani_real 配套调用 | 同一输入/权重/RNG 下 future_actions 与旧完整未来段等价；N/P/C、辅助维度、context 和 dtype 正确 |
| C：评测结构与协议 | evaluation、env_runner、evaluate/select_checkpoint/record_demo/pipeline 及 scripts/docs | 平面记录、单一统计、普通评测与选点解耦；固定预算与冻结测试作为独立协议提交 |
| D：新实验数据配方 | configs、episode_split 入口及相关说明 | 固定 manifest，训练 seed 不改变集合；原 n_action_steps 对 padding 的有效影响显式保存 |
| E：真机职责与记录 | dexmani_real deployment/config、policy metadata/runtime | 模式/状态所有权清楚；迟到/过期/reset/撤权离线测试通过；标准结果必存，Raw 默认不暗改 |
| F：经测量的性能改进 | 已确认瓶颈的局部位置 | 正确性检查通过，报告实际延迟/吞吐与硬件条件；无设备则不声称收益 |

B 与 E 中跨仓库接口变化必须成对提交和验证，不能先发布会破坏另一仓库的半个迁移。无 dexmani_real 可写工作区时继续完成独立任务，将接口破坏性变更留在未发布分支并明确缺口。

C 内区分“执行结构等价”和“协议/统计变化”：固定预算、正式测试覆盖限制、平均步数口径修正必须单独说明。D 的新 split 与默认数据 seed 不改变历史实验。E 的结果记录选项收敛属于运行行为变化。

已讨论的 HandSceneEncoder、SceneAggregator、FeatureAlignment、TactileGuidedRefinement 仅为清理候选：重新核对生产配置、动态导入、测试、脚本与外部实验引用后才决定；删除不属于本方案必要前置步骤，不因类名看似未用就移除。

## 12. 验证矩阵

优先复用[现有验证指南](验证指南.md)与 tests；只新增覆盖本次具体回归风险的定向检查，不以大面积镜像测试替代语义验证。

| 场景 | 必须验证的行为 |
| --- | --- |
| 数据/窗口 | episode 边界、padding、普通/LatentFlow 起点；C 覆盖不改变训练窗口；无验证泄漏 |
| 统计 | 绝对量去重、relative 按窗口/reference、多任务零权重现状；真实数据检查与合成检查分开 |
| Policy 输出 | 普通策略 N=2/H=16 得到 P=15；LatentFlow 完整 H；LAMP 单步；含辅助输出不泄漏；训练目标不变 |
| 两端数值 | 同一物理观测、context、权重和 RNG 的预处理与未来动作一致；不人为统一当前 dtype |
| checkpoint/resume | raw/EMA、normalizer、optimizer、scheduler、EMA 计数、游标/RNG；外部初始化资产不再可用；worker 保证不夸大 |
| 评测覆盖 | 任务/seed 精确覆盖，重复/缺失/额外/异常均拒绝完成；结果落盘后遇错保留前缀 |
| 评测统计 | 不同成功次数的多任务案例；终端与文件同源；全失败、空成功步数、micro/macro 与排序口径 |
| 选点/测试 | 同候选固定预算、平局确定排序、完整候选结果、test/selection 互斥；冻结请求禁止研究参数覆盖 |
| 推理生命周期 | 不支持模式在执行前拒绝；预热/reset 不污染正式 episode；模块导入无硬件连接 |
| Real 可控时钟 | 迟到、超时、过期、bootstrap、当前计划尾部耗尽、reset 后旧结果、撤权、关闭期间未完成任务 |
| Raw/结果 | 两种记录组合、结果 I/O 失败、提交失败与清理失败归属、实际执行历史正确 |
| 效率 | 冷/热启动及稳态延迟、吞吐/内存、完整数值链耗时；改精度/NFE/C 后仍需闭环质量证据 |

命令执行前核对最新入口；现行可用的基础检查示例：

```bash
git diff --check
python -m compileall -q dexmani_policy
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat r3d maniflow multitask_dit lamp latent_flow
python dexmani_policy/smoke_test.py --config-only ddp/dp ddp/dqrise ddp/maniflow ddp/multitask_dit ddp/r3d ddp/sat
```

data-only 需要实际数据；完整 smoke 需要 CUDA、匹配数据/权重。Smoke 不接受 Hydra dotlist。定向 unittest 按修改选取，不能把 config-only、CPU 或合成验证说成 DDP/GPU/闭环通过。

dexmani_real 遵守其 AGENTS.md，默认只运行无设备副作用的检查；不使用 Ruff/pytest，不连接 xArm/XHand/RealSense/VR/触觉设备，不执行 HOME/运动/实机诊断。缺依赖时标记 NOT VERIFIED，不擅自改正式配置或升级环境掩盖问题。

## 13. 交付与维护性验收

- 新增普通策略通常只改策略、必要的数据适配与方法配置，公共 Trainer、选点和普通 Runtime 不新增策略类名分支。
- 一个物理动作窗口只在 Policy 解释一次；执行端只解释消费时机与有效性。
- 一个参数只有一个规范输入位置，启动后不继续回退查找。
- 一种 episode 事实结构、一个统计实现；selection 复用普通评测。
- 静态检查集中，时变执行检查保留；必要日志、失败证据与恢复能力不减损。
- 同步更新调用方、配置 target、README、专题文档和必要示例，不把待实施设计写成已实现功能。
- 历史产物不改写；接口/配置路径变更前核对当前 checkpoint，历史实验可保留对应源码，避免无限兼容层。
- 实施交付分别列出完成项、行为/协议变化、实际 PASS 命令、NOT VERIFIED 缺口与后续实验。没有 GPU 或真机验证时，不声称提速、成功率提升或物理执行安全已经证明。

本方案的确定性收益目标是减少语义重复、配置组合与跨模块改动面；性能与闭环收益仍需要实验。

## 14. 固定版本源码索引

以下链接用于复核本文现状依据；实施时还需检查实际工作区版本。

- R3DP：[Dataset][r3dp-data]、[Workspace][r3dp-workspace]。
- PEAfowl：[部署][pea-deploy]、[数据构建][pea-config]、[训练入口][pea-train]。
- LingBot-VLA-v2：[数据][ling-data]、[评测][ling-eval]。
- DexMani Policy：[DP3 配置][dp3-config]、[LatentFlow 数据][latent-data]、[多任务数据][multi-data]、[Policy][policy-base]、[Runner][base-runner]、[Runtime][policy-runtime]、[训练构建][training-builder]、[RNG][rng-code]、[评测][eval-rollout]、[多任务 Runner][multi-runner]。
- DexMani Real：[推理线程][real-worker]、[调度][real-schedule]、[episode][real-episode]、[session][real-session]、[观测构造][real-observation]。

[r3dp-data]: https://github.com/dazazh/R3DP/blob/4b2703e272b15a9190c6842a82bb0c340f3393fa/policy/R3DP/diffusion_policy/dataset/robot_image_dataset.py
[r3dp-workspace]: https://github.com/dazazh/R3DP/blob/4b2703e272b15a9190c6842a82bb0c340f3393fa/policy/R3DP/diffusion_policy/workspace/robotworkspace.py
[pea-deploy]: https://github.com/Qingyu-Fan/PEAfowl/blob/ed95a3704e8060213a74f673f7d86b3d4a1a8fa4/peafowl_policy/deploy_policy.py
[pea-config]: https://github.com/Qingyu-Fan/PEAfowl/blob/ed95a3704e8060213a74f673f7d86b3d4a1a8fa4/config_peafowl_robotwin.py
[pea-train]: https://github.com/Qingyu-Fan/PEAfowl/blob/ed95a3704e8060213a74f673f7d86b3d4a1a8fa4/train.py
[ling-data]: https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/lingbotvla/data/vla_data/README.md
[ling-eval]: https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/experiment/robotwin/README.md
[dp3-config]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/configs/dp3.yaml
[latent-data]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/datasets/hand_prior/latent_flow_dataset.py
[multi-data]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/datasets/multitask_dataset.py
[policy-base]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/agents/policies/base.py
[base-runner]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/env_runner/base_runner.py
[policy-runtime]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/deployment/runtime.py
[training-builder]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/training/builder.py
[rng-code]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/common/runtime.py
[eval-rollout]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/evaluation/rollout.py
[multi-runner]: https://github.com/haoyangzhanglab/dexmani_policy/blob/cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74/dexmani_policy/env_runner/multi_task_sim_runner.py
[real-worker]: https://github.com/haoyangzhanglab/dexmani_real/blob/f2ec75621ede85d24fce8b189b348405926f642f/dexmani_real/deployment/inference.py
[real-schedule]: https://github.com/haoyangzhanglab/dexmani_real/blob/f2ec75621ede85d24fce8b189b348405926f642f/dexmani_real/deployment/schedule.py
[real-episode]: https://github.com/haoyangzhanglab/dexmani_real/blob/f2ec75621ede85d24fce8b189b348405926f642f/dexmani_real/deployment/episode.py
[real-session]: https://github.com/haoyangzhanglab/dexmani_real/blob/f2ec75621ede85d24fce8b189b348405926f642f/dexmani_real/deployment/session.py
[real-observation]: https://github.com/haoyangzhanglab/dexmani_real/blob/f2ec75621ede85d24fce8b189b348405926f642f/dexmani_real/deployment/observation.py
