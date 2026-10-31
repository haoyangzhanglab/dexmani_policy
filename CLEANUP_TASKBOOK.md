# DexMani Policy 轻量化清理任务书

> 目标：让个人研究中的改模型、换数据、短训、续训、评测和部署更直接。优先消除重复的事实、状态和控制流程，保留影响研究结果的数学与数据语义。
>
> 状态：M0–M5、M6a 与 M7 已实施；M6b 公开迁移待跨仓联调，CUDA/真实仿真未通过环境验证。各批实际结果见进度记录。
>
> 核对基线：[`58e29d91ddbd056054b5b5537e43226e6dc5009d`](https://github.com/haoyangzhanglab/dexmani_policy/tree/58e29d91ddbd056054b5b5537e43226e6dc5009d)，核对日期 2026-10-31。制定本文时 `main` 仍指向该提交。已审阅仓库源码、配置、调用关系和文档；未运行真实训练、仿真或性能基准，未审阅外部 Real 仓库及私人实验脚本。

源码简写路径相对 `dexmani_policy/`；`scripts/`、`docs/` 和根目录文件按仓库根目录定位。保护文件清单使用完整仓库路径。

## 1. 已作出的设计决定

本轮采用**保留研究能力、重做支撑流程、收窄无实际生产者的内部协议**的方案。清理力度主要落在配置、normalizer 所有权、checkpoint 生命周期、评测状态和层层转发上。不同算法仍保留各自直观的实现。

完成后的日常路径应当是：

1. 选择一个方法配置，覆盖少量研究参数，开始训练。
2. 新训练在一个组装位置确定数据划分、模型结构、初始化和归一化；Dataset 负责取样和增强。
3. 中断后从原实验目录的 `latest.pt` 恢复完整状态；不重新拼训练配方，不复制历史 checkpoint。
4. 给出 checkpoint 和评测参数即可运行；选模就是遍历显式候选，使用相同预算，再独立测试胜者。
5. 推理从 checkpoint 得到结构和 affine，不依赖原训练数据、预训练权重路径或另一个可独立修改的配置文件。

### 1.1 硬边界

以下要求优先于任何删减建议，包括通过共享依赖进行的间接删减：

- **保留文本组件**：`agents/obs_encoder/text.py`、`CLIPTextEncoder` 及其实际所需依赖、测试和文档。没有接入当前策略不是删除理由。
- **保留未接入的普通 DiT**：`agents/action_decoders/backbones/dit.py`，以及它使用的 `SelfAttention`、`AdaLNOutput` 等 helper 和测试。
- **不改写 `RectifiedFlow`**：不改变类实现、构造能力、损失、采样、精度、输出或 cache 行为；不删除 `TimeSampler` 的任何现有模式、`sample_dyadic`、`shift_time_to_noise`、`euler_time_grid`；不删 DiTX/Output 承接的 `return_features` 和 cache 接口。
- 为缩小本轮边界，默认不修改 `dexmani_policy/agents/action_decoders/flow.py`、`dexmani_policy/agents/action_decoders/sampling/time_sampling.py`、`dexmani_policy/agents/action_decoders/backbones/dit.py`、`dexmani_policy/agents/action_decoders/backbones/ditx.py` 和 `dexmani_policy/agents/action_decoders/backbones/modules.py`。本轮的配置收敛、EMA 包装调整不得改变这些实现收到的实际参数或能力。
- 保留现有策略、AFRO/CordViP/VQ 预训练的研究目标；保留 LoRA、现有 RGB backbone 替代、SAT/DQ 的融合位置与语义池化消融、AdapTac 的触觉配方、RTC、BF16、有效的 compile 和 EMA teacher。
- 不删除真实实验数据、已有实验目录、权重或评测结果，不批量清理历史产物。产物生命周期调整只影响之后的写入；原目录续训对日志尾部的处理按 M4 执行并保留原日志。

### 1.2 本轮不建立的东西

不新增配置 schema 平台、registry、Initializer 类族、通用 Trainer/PretrainTrainer、回调框架、artifact manager、provenance/evidence 平台、通用迁移系统、自动修复器或静默 fallback。不要把删除的代码换个目录重新实现。

继续使用已有 Python/PyTorch、Hydra/OmegaConf 和目录组织。保留短而有实际复用的 helper。没有必要为了“更薄”移除 Hydra、合并所有文件、内联所有工具或重命名整个项目。

## 2. 目标职责与唯一来源

“唯一来源”指一个事实只有一个可编辑或运行时权威来源。允许输出供人阅读的快照，禁止再次从快照和运行状态各读一份再做一致性握手。

| 事实/职责 | 唯一来源与目标实现 | 应退出的重复机制 |
| --- | --- | --- |
| 任务、数据、动作表示、触觉来源 | 顶层研究配置；组装时显式传给实际消费者 | agent/dataset/env 三处独立设置及等值校验 |
| `horizon`、观测/执行窗口、未来力 offsets | 各保留一个可编辑定义 | 多层插值副本和“必须一致”校验 |
| 动作维度、padding、control mode、必要 Dataset 类型 | 短 Python 组装逻辑，按实际方法和数据语义派生 | YAML 中的 `eval`/`eq` 表达式、可编辑的派生字段 |
| 方法结构与真实研究开关 | 方法配置和直接构造代码 | 已被实现锁死的伪参数、重复默认配置 |
| 数据划分 | 一份实际 train/val 索引及必要数据身份信息 | Dataset 的 train/val 相互回指；多处重新生成划分 |
| normalizer affine | Policy 持有运行状态；新训练在组装处拟合或继承一次 | Dataset 缓存、恢复后回填 Dataset、config→Policy→Encoder 重复解释 |
| 初始权重来源 | fresh 训练入口；作为实验记录保存一次 | 恢复模型构造还携带下载/加载开关与临时来源路径 |
| 恢复结构、参数、affine | checkpoint 内的实际构造配置和状态 | 恢复依赖外部 `config.yaml`、重新下载初始模型、重新拟合统计 |
| 训练进度 | `global_step` 与真实 optimizer/scheduler/EMA 状态 | 可以可靠派生的主训练 epoch/EMA 镜像计数 |
| 评测结果 | 每次评测的唯一输出目录；一次选模记录候选、胜者和实际参数 | protocol/request/result/best 的多份状态链 |
| 部署元信息 | 同一 checkpoint；模型可直接推导的值从模型取得 | 另一个独立可编辑的配置来源和大量镜像字段 |

### 2.1 配置保留原则

- 继续暴露有研究意义的量：方法、动作表示、数据/划分、训练预算、batch、学习率、必要优化设置、模型宽深、采样步数、损失权重、真实增强和消融参数。
- 固定数学规则写在代码里。例如 EEF 的 rot6d 不应成为任意 affine 菜单；不同方法的动作表示不能靠统一维度消掉。
- 固定值只能在一个地方定义。收敛共享默认时使用**当前实际配方**：例如 EMA `power=0.75`，不能删 YAML 后意外回落到构造默认 `2/3`。
- 不以“所有当前 YAML 一样”判定研究选项无用。小而有意义的分支可以保留在构造器，删除重复配置即可。
- checkpoint 中保存恢复所需的完整实际构造参数；`config.yaml` 可以作为一次导出的阅读快照，恢复不再读它作为第二权威来源。
- `joint=19`、`EEF=21`、R3D 辅助预测布局 `28`、点云 XYZ/XYZRGB `3/6` 是不同事实，不统一成一个常数。

### 2.2 生命周期

| 路径 | 必须发生 | 不再发生 |
| --- | --- | --- |
| fresh | 确定 split → 构造结构 → 加载初始来源 → 继承/拟合 train affine → 建 EMA/optimizer → 训练 | Dataset/Policy/Encoder 重复解释同一 normalizer 配方 |
| resume | 读 checkpoint → 校对实际数据/split → 构造结构 → strict 恢复 raw/EMA → 建并恢复 optimizer/scheduler → 恢复 RNG/进度 | fresh 初始化、重新拟合、复制历史 milestones、通用 Hydra override 合并 |
| inference | 读 checkpoint → 无网络构造 → strict 恢复选定权重及 affine → 输入转换 → 推理 | 打开训练数据、寻找初始权重、重新协商归一化规则 |
| 观测预训练导出 | 训练中保存恢复状态；结束或显式调用时导出 encoder | 每次保存恢复点都完整复制并重写 encoder 包 |
| VQ 分析 | 训练保留选模所需标量；完整诊断/绘图由显式命令执行 | 训练结束必须跑大规模 CPU 诊断和绘图 |

## 3. 实施顺序与状态

优先级表示价值，批次表示依赖顺序。先做一批短而明确的删除，再解决结构根因。不要把所有文件放进一个不可审阅的大提交，也不要为每个删除的函数单独开一轮流程。

| 状态 | 批次 | 优先级 | 交付结果 | 前置条件 |
| --- | --- | --- | --- | --- |
| [x] | M0 基线与必要对照 | 必需 | 确认当前实现、保护范围和少量行为对照 | 无 |
| [x] | M1 确定冗余清理 | P2/P3 | 删除无效恢复 hook、闲置属性和采样/脚本转发 | M0 |
| [x] | M2 配置、数据与归一化 | P1 | 组装一次，Policy 持有 affine，Dataset 无回指/缓存 | M1 |
| [x] | M3 初始化与 DQ 状态 | P1/P2 | fresh/restore 分开；码本只持运行必需状态 | M2 |
| [x] | M4 续训与预训练生命周期 | P1/P2 | 同目录续训、完整恢复、按需导出/诊断 | M2、M3 |
| [x] | M5 内部观测接口与封装 | P2 | 明确 Tensor 布局、收窄内部协议、展平固定壳层 | M2；相关路径需 M3 |
| [~] | M6 评测与部署 | P1/P2 | 简单评测函数、显式选模、同源部署配置 | M3、M4；公开接口切换需实际消费者 |
| [x] | M7 工具与文档收尾 | P2/P3 | 远程脚本、smoke、依赖职责和日常文档收敛 | 对应前序批次 |

实施起点 `c1ce1c177ffb677d2e8ec0d878d3b5062eabb3a4`，分支 `cleanup/research-workflow`，起始工作区干净。M0：InterAct 10 项、agent_checks CPU 8 组、10 方法配置解析、相关 shell 语法基线通过；配方与保护文件 hash 保存在 `/tmp/dexmani-cleanup-baseline`。数据与 policy conda 环境存在；CUDA 不可用，未验证 CUDA/仿真/硬件。Real 源码存在且有用户改动，公开迁移另列待验证。

每批允许按完整职责拆成少量提交。影响同一格式或接口的 producer/consumer 必须在同一可用版本里一起更新。每批完成后只在本表或该批小节添加简短结果和未验证项，不新增报告体系。

特别注意：M2/M3 若改变 checkpoint 构造配置或状态布局，`restore_policy`、`inspect_policy`、eval 等读取方必须同批跟进内部读取，保持当时的公开接口和字段值可用；不能等到 M6 才修复已破坏的消费者。M6 负责后续的流程和公开协议收敛。

## 4. 各批任务与验收

### M0：建立足够的对照，不重做全仓审计

**任务**

- 阅读 `AGENTS.md`，检查分支和工作区。记录实施起点 SHA；尊重已有改动。
- 如果代码晚于本文基线，仅检查与当前批次相关的差异。本文中的“无消费者”结论要用当前调用方、配置、文档、测试复核，不能按旧行号机械删除。
- 保存少量可复用对照：解析后的关键配方、固定权重/固定输入下的输出与关键梯度、一次受控保存恢复。优先使用已有测试和临时对照脚本。
- 记录保护文件的基线，后续用 diff 检查；不新建“契约检查框架”。

**验收**

能够区分环境缺依赖、原有失败和此次修改造成的失败。没有 GPU、数据或 Real 源码不妨碍独立的结构清理，但相应验证不能标为通过。

### M1：确定冗余先删干净

对应 R08、R18、R19、R20，以及不涉及格式/API 的 R14 子项。

**任务**

1. 删除 R3D、AFRO、CordViP 中仅清理 fresh 初始化字段的 `prepare_restore_config` override；恢复路径本来就不调用它们的 `initialize_training`。DQ 只先删无效的 `codebook_path=None` 清理。**DP/SAT/DQ 的 RGB 恢复处理此时保留**，直到 M3 建立无网络构造。
2. `scripts/pretrain/vqvae.sh` 与其他预训练脚本一致，只透传 `key=value`；移除 TASK 位置参数、环境变量和 override 的多来源转换。
3. `SequenceSampler` 的 source-row 生成按单样本直接实现，删除唯一调用方加一维再去一维的批量包装；`episode_mask` 改为实际必需参数。保留 Numba、窗口预分配、字段选择和 `field_horizons`。
4. 删除六处无读取的 `out_shape`：State/Proprio、Spatial、Sparse3D、SceneAggregator、HandScene、GatedTactileFusion；同步相关文档。`out_dim` 有实际消费者，保留。
5. 无消费者的诊断返回键可以连同调用方直接删；涉及算法中间量、公共动作 API 或 protected decoder 的项目留到对应批次。

**验收**

恢复分支不再调用被删 hook；采样窗口/source rows 与原行为一致；脚本参数可直接到达现有入口；源码没有残留引用。小型属性/转发删除只做相称检查，不为它们新增镜像测试。

**主要文件**：`agents/policies/{r3d,afro,cordvip,dqrise}.py`、`datasets/sampler.py`、上述 encoder、`scripts/pretrain/vqvae.sh`。

M1 完成：无效恢复 hook、六处 out_shape、sampler 批包装和 VQ 参数翻译已删除，调用方及相关文档同步。728 组采样配置/3112 个窗口与旧实现一致；InterAct 10 项、脚本语法及带空格/引号 argv 对照、diff 检查通过。

### M2：从根上消除重复配置和 normalizer 协商

对应 R01、R02、R03，C19 的默认配置收敛，以及 C16 的重复调用收敛。

**M2a — 配置组装**

- 在现有 `common/config.py` / `training/common/config.py` 附近收敛少量普通函数，明确输入/输出；不再增加 Builder/Schema 类。
- 将派生维度、窗口 padding、控制模式和必要 Dataset 选择移出 YAML 表达式。研究参数只从其指定来源传下去，删除失去意义的跨副本等值检查。
- 保留一次窗口合法性、方法支持的动作表示、观测字段、sim/real 来源和真实单位/几何约束检查。这些错误可能被广播或切片静默吞掉，不能全交给自然报错。
- 移除局部已锁死的伪参数：Uni3D 仅支持的 `pc_in_channels=6`、ResNet 仅支持的 `global_token_type=avg`。不要扩大为删除所有输入维度和 backbone 选项。
- 共同 EMA 配方、cosine scheduler 默认收敛到公共位置；保留方法差异。记录清理前后的解析结果，避免靠构造默认猜测。

**M2b — Dataset 去状态**

- Dataset 持有存储、采样索引和明确的 `training` 标志，负责读取、构造样本、训练增强。
- 删除 `training_dataset` 回指和 `_normalizer` 缓存；验证 Dataset 独立持验证索引，关闭增强，不回访 train Dataset。
- 归一化拟合直接使用存储对 **train 索引**的分块迭代。可保留一个短的数据迭代函数，不要求把正常读取逻辑全部搬进 Trainer。
- 主策略、AFRO/CordViP 预训练、R3D 的辅助目标拼接一起更新。R3D 的拼接数学保持原样。
- split 生成和应用只做一次；保留 trial 隔离及数据适用性判断。当前不存在内容 hash 平台，不新建一个。

**M2c — normalizer 组装**

- 新训练一次性确定：继承预训练包中的哪些 affine、哪些字段需要根据 train split 拟合、哪些字段保持 identity。
- 最终 affine 只装入 Policy。恢复时从 checkpoint 严格加载，不再回填 Dataset，不再执行配方确认链。
- 固定数学规则直接实现：EEF 分段规则/rot6d identity、metric 点云/几何、DQ 手部坐标、R3D 辅助目标；真正有研究意义的触觉 RMS/MaxAbs 选择保留。
- 在导入或拟合边界验证 affine 的形状、目标 dtype 有限性和可逆性一次。保留 FP64 统计、合法全零字段的 identity 语义和 VQ 使用的 NumPy 输入适配。
- 恢复构造所需字段形状放在实际构造信息中；不再靠多轮配方检查重建同一结构。

**验收**

- 随仓库配置解析后的动作窗口、维度、EMA/LR 曲线、loss 权重、增强参数与基线一致。
- train 统计不含 val；验证关闭增强；预训练继承字段不重新拟合；fresh、resume、inference 使用相同 affine。
- joint、EEF、R3D 和触觉至少覆盖受影响的数学边界。对 protected decoder 传入的参数和归一化结果不变。
- 没有用另一个“normalizer service”替代 Dataset 缓存，也没有给每个模块补一个空校验接口。

**主要文件**：`common/config.py`、`common/normalizer.py`、`configs/`、`training/common/{config,data}.py`、`training/trainer.py`、`datasets/{demonstration,split}.py`、`datasets/pretrain/`、`datasets/variants/r3d.py`、`agents/policies/base.py`。

M2 完成：配置派生改为普通函数，共享 EMA/cosine 默认收敛；Dataset 回指/缓存与多轮配方校验退出，Policy 独持 affine，checkpoint 保存实际字段形状。10 方法关键配方与基线一致；joint/EEF/R3D/触觉/AFRO/CordViP 六场景统计逐值对照、train-only/验证无增强/继承不重拟合/错误 affine 检查通过；InterAct 10 项及 agent_checks CPU 8 组通过。未验证 CUDA/真实训练。

### M3：分离 fresh 初始化，缩小 DQ 运行状态

对应 R06、R07；R21 只在本轮必要格式变更中顺带处理。

**M3a — 模型构造与来源分离**

- 持久模型配置描述结构；模型构造不执行下载、查找预训练文件或 fresh 权重加载。
- fresh 入口执行明确的初始化调用。Uni3D 使用一个可空来源路径即可，不并存 bool 和 path；不要用一个通用初始化框架统一不同方法。
- HF/RGB 的实际 architecture 继续随 checkpoint 保存，离线恢复根据它构造；禁止恢复时偷偷查询远程模型配置。
- 只有在上述路径成立后，才删除 DP/SAT/DQ 的临时 RGB 恢复开关及相关 override。
- 初始权重来源可以留在一次实验记录中；它不是恢复时必须存在的文件依赖。

**M3b — DQ 码本**

- 运行态保留算法实际需要的有序原型表和确定形状信息。PCA 排列、训练层权重等导出/诊断信息不应为运行模块增加动态加载协议。
- **导出产物继续携带 hand affine 和实际训练划分来源**；fresh 导入时将 affine 写入 Policy 的唯一 action normalizer。删除运行态重复 affine 前，先迁移 VQ 导出、分析及 DQ 导入的所有消费者。
- VQ 与 DQ 必须使用同一组实际 train 索引。共享 affine 不能代替 split 一致性判断；VQ 当前默认训练集上限与主策略并不自动一致。
- 按构造信息建立固定形状 buffer，用标准 `strict=True` 恢复，移除通用 `BUFFER_NAMES` 搬运、动态 `_load_from_state_dict`、`is_loaded` 和散落的重复守卫。
- fresh 导入或 restore 完成之前，不把模型交给训练/推理。内部暂存的未初始化 buffer 不能成为可被正常调用的模型；不要另建 Ready/Loaded 状态机。
- 保留量化前 finite 检查、最近原型距离、half-up 量化和物理动作解码。NaN 转整数再 clamp 可以得到合法索引，不能只检查最终输出。
- **本轮保持 NPZ 交换格式**。格式扩展名不是复杂度根因，先解决运行态所有权；不同时维护 NPZ/PT 两套正常加载协议。

**格式策略**

必要的 checkpoint 字段变化集中处理；保持 `strict=True`。旧实验默认由对应旧提交读取。只有确有旧权重继续使用需求时，提供针对已知旧格式的一次性、精确转换；不引入运行时猜版本、自动降级或 `strict=False`。

**验收**

- 断网且没有原初始化文件时，checkpoint 能恢复与推理；fresh 仍正确加载原预训练来源。
- DQ fresh、raw restore、EMA restore 的原型和 affine 一致；在同一物理手部输入下 encode/decode 对照一致；缺失/错误产物在导入边界失败。
- 严格恢复发生在 optimizer 建立之前，避免 `assign=True` 替换 Parameter 后 optimizer 仍持旧对象。
- VQ 导出/诊断仍可运行；不把“运行时不用”误判为“科研记录无价值”。

**主要文件**：`agents/obs_encoder/rgb/`、`agents/obs_encoder/pointcloud/uni3d.py`、`common/checkpoint.py`、`agents/hand_prior/{artifacts,codebook}.py`、`agents/policies/dqrise.py`、`training/pretrain/hand_prior/`。

M3 完成：fresh 来源移到 initialization，RGB/HF/Uni3D 构造无外部加载；恢复 hook 退出。DQ 运行仅固定原型 buffer，NPZ 继续保留 affine/PCA/权重/split，hand affine 在 fresh 唯一安装。tiny DINO/SigLIP × freeze/full/LoRA 的真实本地初始加载及删源后 strict 恢复输出/梯度一致；ResNet 默认、DQ物理编解码/raw与EMA删源恢复、坏产物/split/strict错误边界、InterAct10项与agent_checks8组通过。实际数据准入和fresh组装通过；真实Uni3D初始包、CUDA及完整训练未验证。保护文件hash一致。

### M4：续训直达原 run，预训练各管各的生命周期

对应 R04、R11、R12、R21、R22、R23、R24；为 R17 提供正式路径。

**M4a — 续训**

- fresh 使用现有 Hydra 方法配置；resume 使用一个小型显式入口，从原 run 的 `checkpoints/latest.pt` 读取训练配方与完整状态。
- 默认在原 run 续写。仅暴露运行参数：设备、worker 数、日志间隔、本次最多 updates、compile 开关/模式、是否启用日志后端；如数据搬家，允许显式给新路径，并做原数据/split 适用性核对。
- 不允许 resume 偷改 batch、训练总预算、数据划分、模型或优化配方。改配方或从旧点另开实验，使用显式的新 run/fresh 初始化流程；本轮不开发自动分叉、迁移历史或 lineage 功能。
- 删除 Hydra `OverridesParser` 白名单系统、新输出目录要求、`archive_resume_milestones` 及相关硬链接/复制流程。不要只是换一个更复杂的 override 合并器。
- 恢复 raw/EMA、optimizer、scheduler、RNG 和 update 位置；`max_updates` 是本次执行上限，不改已保存的学习率预算。
- 原目录续训须处理日志晚于 checkpoint 的情况：仅在确有超出恢复点的记录时，先保留一份带时间戳的原日志，再按 `step` 保留 `metrics.jsonl` 中不晚于恢复点的记录并继续追加；使用已有原子写工具，不能把原有日志整体清空。没有超出记录就直接追加，不产生备份。W&B 可以开启新的日志 session，不做跨 session 修复平台。
- 本轮按单进程使用一个训练 run；不新增分布式锁/租约。fresh 仍应拒绝覆写已有实验。不要为支持任意旧点原目录恢复而增加冲突修复分支。
- Python/NumPy/CPU RNG 始终恢复；仅在目标使用 CUDA 且有已保存 CUDA 状态时恢复对应部分。去掉“设备类型不同就拒绝”的门禁，不宣称因此保证跨设备 optimizer 兼容或逐位一致。

**M4b — 少量训练状态**

- 主训练/观测预训练中由 `global_step` 能可靠重建的 epoch/EMA 计数，可随必要格式修改删掉；保持 EMA 更新的先后顺序。VQ 的 epoch 被导出流程使用，不能一起删。
- 只为实际执行前向的 EMA 模型建立 compile 包装；ManiFlow 的 teacher 保留。compile 是惰性的，不以删除包装宣称省掉完整编译时间。
- clip ratio 在日志处推导；进度条用局部上下文管理。没有验证步骤的 epoch 之间无需反复切 eval；若保持当前 `train()` 返回模式，则在退出时切一次 eval。
- 保留 finite loss/gradient、梯度裁剪、optimizer 参数覆盖/去重及参数名顺序检查、checkpoint 原子保存和必要资源关闭。
- **保留梯度累积**。对个人单 GPU 研究它有实际价值；不为拉直循环改变 AFRO/CordViP 的尾 batch 和默认 `drop_last=False`。

**M4c — 预训练保存与诊断**

- AFRO/CordViP 周期保存完整恢复状态；`encoder.pt` 在结束或显式导出时生成。保留两种产物的实际消费者，只拆写入时机。
- 两个方法保留各自训练循环：AFRO 的同 batch VICReg 和 teacher momentum，不与 CordViP 的监督/update 语义强行统一。
- VQ 训练保留 `lookup_mse` 等真实选模指标和必要标量；完整码本 usage、CPU 分析和 matplotlib 绘图移到显式分析入口，绘图库在该入口按需导入。
- 保留 best 与 last 的区别。某轮命中多个保存条件时避免重复构造同一 payload；本轮不未经需求确认删掉历史归档能力，也不自动删除已有周期 checkpoint。
- 全配方记录一次。`training_config` 等字段没有程序读取，不等于无人需要复现实验。

**验收**

- 在 CPU、无增强、`num_workers=0` 的可控条件下比较连续 N 次更新与 K 次后恢复到 N 次：参数、optimizer/scheduler、EMA、下一 batch 位置和日志 step 一致。
- 正常训练不中断，或信号中断后续训，均能完成；恢复不复制历史 checkpoint，不再次拟合 affine 或加载初始来源。
- 单独覆盖“日志多于 checkpoint step”、已有目标目录的 fresh 拒绝和 W&B 关闭路径。
- 观测预训练 checkpoint 可恢复，显式导出的 encoder 可用于下游 fresh；VQ 无绘图库也能完成训练主流程，显式诊断正常。

**主要文件**：`train.py`、`training/trainer.py`、`training/common/{resume,logging,optimization,data}.py`、`common/{checkpoint,pytorch_util}.py`、`training/pretrain/`。

M4 完成：原目录latest续训及显式运行参数入口上线；退出override白名单/历史搬运/epoch与EMA镜像计数，无用EMA compile退出；日志仅超恢复点时备份截尾。AFRO/CordViP周期仅保存resume、结束/显式导出encoder；VQ诊断/绘图改显式分析。CPU主训练9步vs3+恢复的raw/EMA/optimizer/scheduler/RNG逐值一致，日志step/备份/fresh拒绝/实际SIGTERM恢复共4项通过；AFRO/CordViP真实4vs2+2（跨epoch/累积/尾batch）及删数据后导出逐值一致；禁止matplotlib的VQ两epoch训练与显式分析出图通过。未验证CUDA/跨设备optimizer与编译逐位重放。checkpoint格式已收敛，旧实验使用旧提交读取。

### M5：明确内部输入，撤掉无实际生产者的协议

对应 R13、R14、R15、R16；本轮明确选择 C01、C02、C03 和 C10 的下述范围。

**M5a — Tensor 与 RGB**

- 外部 NumPy/历史输入在 Dataset/Runner/Runtime 边界转换；内部模型接收明确 Tensor，不再在 `BasePolicy.preprocess` 中探测并透传任意非 Tensor。
- B/T 只在约定位置合并一次；RGB backbone 的输入固定为 `[B*T,3,H,W]`。`ImageProcessor` 不再猜任意前导维和通道位置。
- 不能直接删现有 HWC/CHW 适配：DP 当前确实经过 CHW 预处理，Spatial 当前使用原始 HWC。先把两条生产路径改为明确的共同模型内布局，再删推测分支。
- 共同边界使用 CHW float32 `[0,1]` 表示进入 RGB processor 的图像；保留各方法原有 resize/crop/增强顺序、uint8 量化时机、半像素坐标变换和标定对应关系。必要的 uint8 舍入先执行，再转换为共同表示，不能靠提前 `/255` 改写原配方。
- 外部 Runtime 的 uint8 HWC 接口保持；浮点图像只在明确约定 `[0,1]` 的内部入口接收，不靠数值范围猜单位。
- 点云模型通道选择收敛到一个明确位置；读取阶段只取所需列以减少 I/O 可以保留。

**M5b — 内部输出与固定封装**

- PointNet/MultiStagePointNet 只有一个真实 Tensor 输出时直接返回 Tensor；FeatureAlignment 返回实际消费的 tokens。实际多输出保留小型明确结构，不为了统一风格新造 dataclass 层。
- 删除无人消费的 DQ 诊断键及 attention/geometry 返回接口，但保留内部数学和真正的消费者。
- **HandScene 的 `hand_query`、`relation_update`、`geometry_descriptor` 必须保留**，GatedFusion 消费 `geometry_descriptor[...,3:]`。
- SceneAggregator 的 mass/centers/covariance 参与内部权重和 descriptor，不能按返回键无人读取删除计算。
- **本轮保持公开 `future_actions` 动作字典和 Runtime 返回接口**。将公开单键字典改 Tensor 的收益不足以抵消未审阅 Real 的迁移风险；内部输出可以独立收缩。
- 展平 CordViP 固定一层 encoder/decoder 的单元素容器、SublayerConnection 和 lambda 转发；直接写清 attention、FFN 和 residual 顺序。保留自定义 sample-std LayerNorm、eps 在根号外的行为和 clone 初始化关系，不直接换 `nn.Transformer`。

**M5c — 当前输入范围**

- FeatureAlignment/projection 按当前单相机调用收窄 V 轴及无调用的外部 confidence/image-token-valid 协议。保留双线性插值、coverage、视锥边界、quality 数值及其 gate 输入列；不把 confidence 近似为常数。
- 当前触觉输入要求实际五指 Tensor。删除无真实 Dataset/Runner/Runtime producer 的 `None` / `tactile_valid` 缺测协议及仅服务该协议的测试；有缺测需求时未来作为独立研究功能实现。**合法零力始终有效，未来力标签的有效 mask 完整保留**。
- 非空点云、有限 patch 的主路径中撤掉恒真 patch mask、整云 sentinel 和无用 null K/V 协议，前提是 M5d 建立有效边界。局部邻域仍可能没有点：保留 `local_radius` mask、proximity、near support，以及“手指附近为空”的合法语义。
- 删除当前未启用的 PointColorNoise、DP noise/blur 菜单及颜色强度的多种等价输入表示；保留实际启用的颜色、grayscale、dropout 和坐标/状态增强。同步配置、调用和文档；若实施时出现真实使用者，保留该具体项并记录，不推测它无用。

**M5d — finite 检查的真实边界**

- 当前 `check_data` 是独立工具，Dataset **不会自动做完整扫描**。不能因为工具存在就删除所有输入保障。
- 新数据准备时使用独立检查，检查训练读取所用的**目标 dtype 转换之后**的有限性与非空形状；源 float64 有限不保证 FP32 转换后有限。更新现有 `check_data` 即可，不添加证书、缓存、manifest 状态或每次训练强制全量扫描。
- 研究工作流明确约定：数据版本准备检查后不就地修改；实际增强和 affine 的合法设置在组装处检查；外部 Runtime 保留每次输入的有限性/形状检查。
- 在该边界成立的内部路径中删除 Sparse3D 每 batch 的 rank/dtype/全量 finite 扫描；不把相同扫描搬到另一个模型层。若当前生产路径不能建立此前提，保留这一处检查并暂缓相关 mask 删除。
- 保留 loss/gradient finite、DQ 量化前 finite 和几何必要精度。减少检查位置不能变成用合法默认值掩盖坏输入。

**验收**

- 固定权重和输入，主要输出与关键梯度按原 dtype/数值容差对照；RGB 训练/验证/部署的实际预处理一致，投影坐标不漂移。
- 覆盖全出视锥、手指局部邻域为空、合法零力、未来标签边界、FP64→FP32 溢出和错误外部布局。
- CordViP 同一组权重精确映射后前向/梯度一致。没有实际旧权重迁移需求时，不因此在运行时留下兼容层。
- 此批不改变 protected decoder，不以改测试期望的方式掩盖科研语义变化。

**主要文件**：`agents/policies/base.py`、`agents/obs_encoder/{spatial,xhand_tactile}.py`、`agents/obs_encoder/{rgb,pointcloud,interaction}/`、`common/image_util.py`、`datasets/{demonstration,transforms}.py`、`datasets/tools/check_data.py`、`deployment/runtime.py`、`env_runner/sim_runner.py`。

M5 完成：内部Tensor/CHW FP32 RGB、PointNet输出、单相机与五指输入明确；无producer缺测/诊断/恒真mask/sentinel协议退出；CordViP展平；数据准备检查目标FP32 finite及非空。固定权重输出/梯度、PointNet ties、Scene消融/空邻域、单相机quality/coverage与双线性梯度、RGB量化/半像素变换、现用增强对照通过。实际fresh identityPC列数补齐并覆盖真实Zarr组装回归；InterAct14项（含FP32溢出/空云/Runtime布局与负stride）及agent_checks CPU8组通过。CUDA与真实闭环未验证。

### M6：评测用函数和记录，部署用同一个事实来源

对应 R05、R09；与 R10 的结果同步部分联动。

**M6a — 内部先收敛**

- `evaluate` 接收 checkpoint、raw/EMA、inference steps、action steps、seeds 和输出目录等明确参数；不通过 `EvalProtocol`、嵌套 config merge、source identity 和 frozen request 多层组装。
- 单 checkpoint 评测和选点/测试共用实际 rollout 函数。保留 metrics、seed 处理、视频和资源关闭这些有真实职责的短模块。
- 选点显式传入候选列表，不要求必须凑齐五个 milestone 文件才允许运行。20/40/60/80/100% 仍可作为论文实验的默认候选方案；使用子集时记录实际列表，不把它标为完整预算对照。
- 同一比较中使用相同 selection seeds、episode 预算、推理和执行参数；保留当前 success rate、avg steps、global step 的排序/并列规则，除非单独提出研究变更。
- selection/test seeds 从开始就确定且不相交；选择过程不读取 test 成绩。胜者直接传给 test 调用，取消通过 `test_request` 再序列化/反序列化驱动流程。
- 一份选模记录写清候选路径/step/分数、胜者、selection/test seeds、实际推理参数、各结果位置；科研记录保留，禁止用删记录代替删协议。
- 轻量的失败结果、异常传播和原子 JSON 写入可以保留；不为了削减状态强行失去长评测的失败定位。

**M6b — 产物与公开消费者一起切换**

- 最终每次 benchmark 使用唯一输出目录；胜者由该次 selection 记录给出。取消 root `best_ckpt.json` 作为第二权威状态，取消依赖共享发布状态的 `.pipeline.lock`。
- 删除锁前必须先消除所有竞争写入的共享文件；不能“先删锁，再设计输出目录”。不引入新的全局 latest-selection 指针。
- 评测 CLI 最终只保留单点 `evaluate` 与选点加独立测试 `eval_pipeline`；`select_checkpoint` 留作普通内部函数。旧的重复 CLI/shell 转发在调用方更新后退出。
- 部署从 checkpoint 读取实际配置，减少 `PolicyInfo` 中可派生的内部镜像和 `config.yaml` 二次读取。外部需要的控制周期、相机/点云加工、手模板/指序、触觉来源仍必须能取得；不能用形状推导物理语义。
- **先保留公开 `inspect_policy`、`load_policy`、`PolicyRuntime.predict` 及动作窗口语义**。删除字段、移除 `best` 别名或改变默认 checkpoint 选择，必须核对并同步 Real 和实际私人调用脚本。
- Real 源码不在本仓库的审计范围。若实施环境拿不到它，完成 M6a 与不改变公开行为的同源读取；M6b 的破坏性公开切换明确标为待迁移，保留现有必要 best/lock/同步逻辑，不新增自动兼容框架，也不把该子项标为已完成。

**验收**

- 单点评测可直接运行；候选相同预算、确定并列规则、selection/test 无交叉；记录与实际调用参数一致。
- 输出目录独立后两个评测任务不覆盖彼此；不以格式/签名/evidence 测试替代真实结果验证。
- raw/EMA 均可离线加载；完整未来动作窗口、动作起点、control dim、RGB 预处理和触觉来源在现有 API 上保持正确。
- 如执行 M6b，Real、PolicyRuntime、`best` 选择调用和同步脚本必须在同一可用版本切换；缺消费者验证时不能宣称部署兼容完成。

**主要文件**：`evaluate.py`、`eval_pipeline.py`、`select_checkpoint.py`、`evaluation/{protocol,selection,rollout,artifacts,seeds}.py`、`common/experiments.py`、`deployment/{metadata,runtime}.py`、`scripts/eval/`、`scripts/remote/sync_down.sh`。

M6a 完成：直接checkpoint评测与显式同run候选共用rollout；protocol/merge/test_request链退出；inspect从同一checkpoint读取，公开字段/动作窗口保持。编排2项回归（相同预算、seed隔离、tie、显式EMA覆盖raw默认、唯一输出/失败记录、删除config.yaml后inspect）通过。M6b部分待迁移：Real源码存在且有用户改动，未完成跨仓实际联调，保留best/lock及同步发布链；因此selection输出仍在原run内，selection-only薄CLI暂保留。真实SimRunner已尝试，SAPIEN RenderSystem报Vulkan ErrorExtensionNotPresent，失败详情落盘后进程exit139；真实闭环未通过，CUDA不可用。

### M7：把工具和文档也变薄

对应 R10、R17、C20，及其余批次的配置/说明清理。

**任务**

- 远程执行使用显式 host、repo/data/run 路径和 GPU；保留确有复用的 launch、code/data/run sync，stop/tail 可以在文档中给直接 SSH/tmux 命令。
- 删除复杂的目录发现、状态推断和重复参数翻译；保留正确 shell quoting 和运行源码快照。不会用 `eval` 拼用户参数，不把本地删除传播到远程实验目录。
- 结果同步默认针对停止/完成的 run 做单向 rsync；依赖 best 发布链的多轮同步只有在 M6b 完成后才删。运行中查看日志用 tail，不开发新的在线一致性同步系统。
- smoke 收敛为**一个配置、短训、保存、恢复**，调用正式训练/恢复路径；多配置用 shell 循环。已有独立数据检查继续使用 `check_data`；必要配置组合测试使用现有加载函数，不再造一套平行配置解释器。
- 保留 InterAct 和 agents 的数学/梯度/恢复测试及受保护组件测试；只删除专门维护已退出内部机制的测试。不要为了保住旧测试，重新加入被删协议。
- 文档只保留当前入口、真实参数、重要数学/数据语义和可复现要求。同步 README、架构、仿真评测、观测预训练、手部动作、SSH 文档，删除过时的正式协议和新目录 resume 描述。
- 明确 `pyproject.toml` 负责直接依赖，环境固定文件负责可复现安装；同一版本约束不再双份手工维护。保留经过验证的版本组合，不顺带升级依赖；文本组件的依赖不能被清理掉。
- matplotlib 只在分析路径导入。PCA 仍使用现有 sklearn 实现，本轮不为删一个依赖重写原型排序算法。

**验收**

- README 中最短路径能完成数据准备、新训练、原目录续训、单点评测和选模/测试；预训练与部署命令对应最终代码。
- 脚本 `bash -n` 通过，参数含空格/引用的相关路径实际传递正确；依赖环境缺失时如实记录。
- 没有新增“清理框架”，没有双轨旧新接口长期并存，没有无人使用的新配置或辅助文件。
- 最终报告列出完成批次、真实删除/合并内容、运行过的检查和结果、阻塞/未验证项。没有测量就不报告提速、内存或删减百分比。

M7 完成：单config正式smoke、显式远程路径/解释器/GPU及源码快照、无删除传播的code/data同步、保留best分轮发布的单run同步上线；stop/tail脚本退出，文档给直接命令。依赖职责收敛且requirements版本未升级，文本依赖保留。实际CPU smoke2+1（总预算9不变）、全部shell语法、远端Bash本地传输替身+真实rsync的特殊字符argv/前后台/resume/保留目标/配置冲突/best最后发布/可变预训练产物重复同步检查通过。README及日常文档同步。未连接真实SSH主机，未验证远端服务/CUDA。最终InterAct14项、训练恢复4项、评测编排2项、agent_checks CPU8组各自通过；集成中发现并修复显式EMA和候选结果变量问题后重跑评测2项通过，保护文件hash一致。

## 5. 条件项的最终取舍

以下给出默认决定，Codex 不需要对每项再次询问。若新代码中出现具体使用者，按事实保留该项并记录。需要另开研究决策的项目不会阻塞核心清理。

| 编号 | 本轮决定 | 理由/边界 |
| --- | --- | --- |
| C01 单相机协议 | 按 M5 收窄 | 当前唯一真实调用为单相机；quality 和几何数值保持 |
| C02 触觉缺测协议 | 按 M5 删除无 producer 的协议 | 不删零力语义和 future label mask；这是明确的支持范围收窄 |
| C03 整云/patch 可用性协议 | 边界建立后收窄 | 局部邻域为空仍合法；未建立边界时暂缓 |
| C04 SceneAggregator 消融 | 保留声明的算法消融；可删无人传的 SceneAggregator `query_context` 参数 | HandScene 接收 `scene.mean` 的 `query_context` 有真实用途，保留 |
| C05 ConsistencyFlow absolute 时间目标 | 保留 | 算法选项；本轮不进入 `flow.py` |
| C06 ConsistencyDiTX pre_norm_modality | 保留 | 不为单个备用分支改动 decoder/cache 语义 |
| C07 UNet cond_predict_scale | 保留能力，减少重复默认项 | bias-only 是真实消融，分支本身很小 |
| C08 Diffusion v_prediction/RTC 通用包装 | 保留目标能力；已固定调用下的重复 scheduler 拒绝检查可顺带删除 | sample 与 epsilon 真实存在，RTC VJP/精度边界保留 |
| C09 modality dropout | 删重复零配置，保留简短非零能力 | 手工 override 仍可使用；不是不可达代码 |
| C10 闲置增强菜单/多输入表示 | 按 M5 的当前范围收窄 | 保留现用增强与原处理顺序 |
| C11 梯度累积 | 保留 | 单 GPU 显存受限时有用，不能悄悄改变尾 batch 权重 |
| C12 RGB 可选投影/token 模式 | 保留小型研究能力；R3DBlock 的无人覆盖 activation 参数可固定为现用 ReLU | 不删除 RGB backbone 种类或 LoRA |
| C13 VQ 周期/best/last | 保留归档能力，消除重复 payload 工作 | best 不等于 last；只改变以后写入，不清理已有实验 |
| C14 NPZ→PT | 本轮不做 | M3 不依赖格式替换；避免双读取和额外迁移 |
| C15 sklearn→Torch SVD | 本轮不做 | 需处理主轴符号、近并列/退化情况，收益不足以优先修改算法 |
| C16 split 身份/输入形式 | 合并重复解释；保留必要实际索引与一次适用性判断 | 数据不可变版本制度未证实覆盖所有来源，不批量删除身份字段，不加 hash 框架 |
| C17 冻结入口 | 保留现有范围，文档写清 | backbone freeze+eval 与 `obs_lr=0` 全观测参数冻结不同；正 obs LR 与冻结 DINO 并不冲突 |
| C18 Uni3D 无用图像 stem | 本轮暂缓结构/权重迁移 | 可以后续精确提取，但不使用 `strict=False`；pos_drop/blocks/norm/fc_norm 有用途 |
| C19 EMA/scheduler 配置 | 合并共享实际默认，保留调研入口 | EMA power 保持 0.75；不改变曲线 |
| C20 依赖清单 | 按 M7 收敛维护职责 | 直接依赖与环境锁定目的不同；不根据没有直接 import 就删依赖 |

R25（ConsistencyDiTX 的重复初始化）**本轮暂缓**：前一次初始化随后被覆盖属实，但删除会改变 RNG 消耗和同种子从头初始化轨迹。为几行清理触及 decoder 不值得；需要时作为单独明确接受随机轨迹变化的修改处理。

## 6. 必须保留的正确性成本

这些机制保护实际研究结果，不以名字中带 `validate`、`contract`、`artifact` 为删除依据。

| 保留内容 | 原因 |
| --- | --- |
| trial 隔离、train-only 统计、真实 split 索引 | 防数据泄漏和不公平比较 |
| source rows、padding、future offsets/mask、合法零标签 | 决定时序监督；不能把复制的末帧当真实未来 |
| sim/real 力来源、单位、坐标与指序 | 是物理语义，不能用 shape 检查替代或自动转换 |
| FP64 统计、有限可逆 affine、metric 几何、FP32 时间/RoPE | 防数值或空间语义静默改变 |
| PointNet 的 max 梯度语义、分组 action loss | 不能以更短 API 或全维 MSE 替换 |
| loss/gradient finite、DQ 量化前 finite | 坏值可能被后续离散化掩盖 |
| strict state、完整 optimizer/scheduler/EMA/RNG、原子 checkpoint | 长训练的正确恢复；不应用兼容 fallback 掩盖键错配 |
| optimizer 参数覆盖/去重、param_names、实际活动 BN 约束 | 防漏训、错绑和 EMA 不一致；对应能力仍在时保留 |
| 单次采样内 K/V cache、ManiFlow EMA teacher | 有实际算法/前向用途；不是跨请求状态平台 |
| Runtime 历史长度、外部 finite、负 stride 转换 | 实际外部输入边界 |
| EpisodeStore 进程本地句柄和分块读取 | DataLoader/大数据的实际生命周期 |
| AdapTac DiffusionForceHead 与 identity 分支 | force loss 权重为零也不等于该头无动作梯度 |
| AFRO/CordViP 各自训练目标 | 语义不同不强行抽象成通用 Trainer |
| RTC autograd/VJP、FP32 系数、inference-mode 和 scheduler device 处理 | 真实计算要求；不按 clone/to 的数量清洗 |
| 短小的 signals、metrics、EpochBatchSampler、dict_apply、scheduler、资源关闭工具 | 实际复用且理解成本低，不需要另起结构改造 |

## 7. 验证计划与停止条件

### 7.1 验证矩阵

只对受影响路径运行相应检查。每项有正确性风险的修改应有能发现该风险的对照；纯属性删除和文案修改不额外创造测试。一次必要检查通过后，不在没有新变化/疑点时反复扩大矩阵。

| 变化 | 最小有意义的验证 | 不算通过的替代品 |
| --- | --- | --- |
| 配置/normalizer | 解析配方对照；train-only 拟合；joint/EEF/R3D/触觉关键数学 | 只验证 YAML 能解析 |
| Dataset/sampler | episode 边界、padding source rows、验证无增强、未来 mask | 只检查 batch shape |
| 初始化/码本/恢复 | fresh 与 raw/EMA strict restore；断网无原文件推理；DQ 坐标一致 | `strict=False` 加载成功 |
| Trainer/resume | 小规模连续/恢复对照及日志回退 | 只检查 global_step 增加 |
| Encoder/封装/RGB | 固定权重的输出与梯度、像素变换与几何对照 | 只修改测试期望使之通过 |
| 评测 | 相同候选预算、seed 隔离、并列规则、实际参数记录、输出不覆盖 | 新的 protocol/evidence 字段校验 |
| 部署公开变更 | 实际调用方更新与一次实际路径调用 | 在 Policy 仓库构造一个假 Real 就宣称兼容 |
| 工具/文档 | `bash -n`、实际参数传递、最终 README 最短路径 | 只做文本搜索 |

当前已有入口，可按受影响范围复用并随正式接口更新：

```bash
python -m unittest dexmani_policy.test_tools.test_interact_dex -v
python -m dexmani_policy.test_tools.agent_checks --device cpu
python -m dexmani_policy.datasets.tools.check_data --zarr-path <实际数据>
```

`agent_checks` 的点云路径仍依赖实际 PyTorch3D 后端；InterAct CPU 测试里的 Torch 替代采样不验证原生 CUDA/FPS/kNN。缺环境时记录缺项，不下载大模型/启动长训练来“凑通过”，也不把 mock 通过等同真实后端通过。

保护范围用最终 diff 加现有相关测试检查。普通 DiT/文本的未接入状态不影响保留；不要求为了本次清理把它们接入策略。

### 7.2 目标命令形态

以下是**实施后的目标接口**，不是声称当前已存在的新参数。以少量明确参数实现，必要时沿用更简洁的既有名字；M7 必须把最终可运行命令写回日常文档。

```bash
# 新训练继续使用已有方法配置与 Hydra overrides。
bash scripts/training/train.sh dp3 'task_name=<task>'

# 续训使用现有 resume 模块扩展出的直接入口，不再重新选一份方法配方。
python -m dexmani_policy.training.common.resume \
  --run <run> --device cuda:0 --num-workers 4

# 单个 checkpoint 的直接评测。
python -m dexmani_policy.evaluate \
  --checkpoint <run>/checkpoints/latest.pt --weights ema \
  --inference-steps 10 --episodes 100 --seed 1024

# 选模和独立测试；候选是显式文件列表，输出为本次独立目录。
python -m dexmani_policy.eval_pipeline \
  --checkpoints <checkpoint-a.pt> <checkpoint-b.pt> \
  --weights ema --inference-steps 10 \
  --selection-episodes 25 --test-episodes 100 --seed 1024 \
  --output-dir <new-evaluation-directory>
```

预训练可继续保留各方法的薄入口和算法专属参数，不要求套同一个通用 CLI。小规模 smoke 的执行上限与研究训练总预算分开，不能为测试改坏 scheduler。

### 7.3 何时算完成

- 已实施项的所有真实 producer/consumer、配置、脚本和文档在同一可用版本一致。
- M2 的重复事实与 normalizer 回指、M3 的重复运行状态、M4 的目录/配方搬运、M6a 的协议转发确实退出，没有改名重建。
- 受保护组件及依赖能力未变；保留项的数学/数据含义未变。
- 已执行的检查通过；缺环境和待外部迁移的子项分别明确列出。M6b 未完成时不能写“全部迁移完成”。
- 最终 diff 无无关改动、无新增抽象框架、无新旧运行协议并行维护。
- 不设删行数/KLOC/校验数量指标，不承诺未经测量的加速。调用路径更短、所有权更清楚、用户要填的重复信息更少才是完成依据。

## 8. 结论追踪与事实限定

### 8.1 前次 25 项行动结论的去向

| 批次/决定 | 对应结论 |
| --- | --- |
| M1 | R08 冗余恢复 hook；R18 VQ 参数入口；R19 sampler 包装；R20 out_shape；R14 的独立小项 |
| M2 | R01 配置唯一来源；R02 normalizer 解释一次；R03 Dataset 去缓存/回指 |
| M3 | R06 DQ 状态/坐标来源；R07 fresh 初始化分离 |
| M4 | R04 续训；R11 预训练导出；R12 VQ 诊断；R21 计数；R22 RNG 门禁；R23 EMA compile；R24 局部状态 |
| M5 | R13 Tensor/布局；R14 内部返回结构；R15 CordViP 外壳；R16 Sparse3D 检查边界 |
| M6 | R05 评测状态；R09 部署信息；R14 公开动作 Tensor 改写明确不做 |
| M7 | R10 远程脚本；R17 smoke |
| 暂缓 | R25 ConsistencyDiTX 重复初始化 |

### 8.2 不再沿用的推断

1. “让 PyTorch 自然报错即可”不适用于广播、切片、单位、时间或 affine；必要边界保留一次。
2. 不全面清洗 `getattr`/`isinstance`；不靠新增空基类方法满足统一接口。`RectifiedFlow` 能力探测受保护。
3. 单相机不意味着 FP32 confidence 精确恒定；不做常量化，也不删除 gate 的对应输入列。
4. 固定码本 buffer 的未初始化风险是**拟议重构需要防止的风险**，不是已确认的当前 bug。
5. VQ `best` 与 `last` 一般不同，不能互换。
6. 源码可证明某分支/包装存在，不能证明完整编译时间、性能损失或清理收益；需实测才报告。
7. 不同动作/辅助/点云维度不能合并成一个通用维度。
8. 当前 split 身份比较不是内容 hash/provenance 平台；削减重复不等于删除科研划分约束。
9. 前次 44 个关联文件、88 个关联函数/方法、23 个关联类是关键词关联范围，**不是可删数量**；`evidence` 仅有清理脚本注释中的一次出现。
10. 视觉 backbone 冻结与整个观测编码器冻结作用域不同，不新增二者与正学习率冲突的伪规则。
11. 格式变化不能用 `strict=False`、缺键默认值或自动版本回退掩盖。
12. 没有程序消费者不等于没有人类科研用途；将记录集中、诊断改按需执行，比删除全部记录更合适。

### 8.3 关键事实入口

实施时阅读对应源文件及真实调用方；以下固定提交链接便于核对本文事实，不要求每批重新读取全仓。

- [配置等值检查与组装](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/training/common/config.py#L36)
- [Dataset normalizer 缓存/回指](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/datasets/demonstration.py#L283)
- [Policy 配方验证与预处理](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/agents/policies/base.py#L99)
- [续训 override 与 milestone 迁移](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/training/common/resume.py#L40)
- [码本动态 buffer 与加载协议](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/agents/hand_prior/codebook.py#L45)
- [DQ 训练来源与 affine 导入](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/agents/policies/dqrise.py#L111)
- [选模、test_request、best 与锁](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/evaluation/selection.py#L34)
- [PolicyInfo 和独立配置读取](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/deployment/metadata.py#L10)
- [Sparse3D 热路径检查与恒真 mask](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/agents/obs_encoder/pointcloud/sparse3d_encoder.py#L170)
- [GatedFusion 对 geometry_descriptor 的真实使用](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/agents/obs_encoder/interaction/gated_tactile_fusion.py#L79)
- [CordViP 的自定义 norm 和固定容器](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/agents/obs_encoder/interaction/correspondence/cross_attention_transformer.py#L12)
- [预训练的双产物保存](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/training/pretrain/obs/common.py#L48)
- [受保护的 RectifiedFlow](https://github.com/haoyangzhanglab/dexmani_policy/blob/58e29d91ddbd056054b5b5537e43226e6dc5009d/dexmani_policy/agents/action_decoders/flow.py#L21)

## 9. 可复制给 Codex 的实施 prompt

推荐使用下面的主 prompt。任务书已经作出范围选择，执行时不需要重新做一遍开放式全仓 review。

### 9.1 主 prompt：按批次连续完成

```text
请实施仓库根目录 CLEANUP_TASKBOOK.md 中的清理方案。先读 AGENTS.md 和任务书，按 M0→M7 的依赖顺序连续推进；这是编码实施任务，不要停在分析、重述方案或询问是否开始。

项目定位是个人机器人策略研究。优先删除重复配置、重复状态、重复校验、无消费者的内部协议和无意义转发；保持实际研究数学、数据语义、现用算法和默认实验配方。以任务书中已定的取舍为范围，不自行扩大为重写训练/模型框架。

硬约束：保留文本组件、未接入的普通 DiT 及必要依赖/测试；不改写 RectifiedFlow，也不通过 TimeSampler、DiTX return_features/cache 等依赖缩减其能力。遵守任务书的默认保护文件范围。R25 暂缓，其他条件项按第 5 节执行。不要按关键词或默认配置未启用批量删代码。

执行方式：
1. 检查当前分支、工作区和实施起点 SHA；尊重用户已有改动。在独立任务分支工作，若当前已是该任务分支则沿用。只复核当前批次涉及的实现、真实调用方、配置和必要测试，不重做全仓审计。
2. 用少量普通函数和显式数据流实现；不新增 schema/registry/manager/通用 Trainer/回调/自动兼容框架。producer、consumer、配置、文档在同一可用批次一起修改，删掉失去用途的旧路径。
3. 每批完成后运行风险相称的检查。优先复用现有测试；关键数学用固定权重输出/梯度对照，续训用短程连续/恢复对照。纯删除不补镜像测试。通过后继续下一批，不反复扩大无关检查。
4. 按完整批次形成可审阅的本地提交，只提交本任务文件；不要 push 或创建 PR。简短更新任务书中的进度、实际验证和未完成子项，便于中断后继续。
5. 缺真实数据、CUDA/PyTorch3D、初始权重或 Real 源码时，不捏造通过结果，不靠 mock 宣称真实后端/外部兼容；继续不依赖它们的工作。M6b 需要实际消费者才能破坏性切换，条件不足则保留现有公开行为并明确标记，不能另造兼容平台。
6. 不删除已有实验/数据/权重；不使用 strict=False 或静默 fallback 掩盖迁移错误。必要格式变化集中处理，旧实验默认使用旧提交；仅有真实需求才做一次性精确转换。

持续推进到所有可执行批次完成。只有新的研究范围冲突、用户保护边界冲突或缺失信息确实决定无法安全实施的行为时才提问；普通实现选择自行解决。最终报告完成批次、关键删减、真实验证结果、未验证/待外部迁移项和提交列表。没有测量不要宣称性能提升或删减百分比。
```

### 9.2 中断后的续做 prompt

```text
继续 CLEANUP_TASKBOOK.md 的实施。先读 AGENTS.md、任务书进度、当前 git status/diff 和本任务已有提交，确认已完成内容；不要撤销或重复实现。从第一个未完成且依赖满足的子项继续，沿用主 prompt 的保护范围、测试和提交要求。缺外部环境只挂起依赖它的子项，完成其余工作；不要重新做全仓开放式 review。
```

### 9.3 实施结束后的定向复核 prompt

```text
按 CLEANUP_TASKBOOK.md 定向复核本次清理的最终 diff 和实际调用链，不另开一轮全仓重构。重点检查：保护组件及依赖能力是否被间接删减；数学/数据/默认配方是否漂移；normalizer/split/checkpoint 是否真正只有明确来源；fresh/resume/inference 是否完整；producer/consumer/文档是否同版；外部 Real 依赖是否被误判；是否新增了替代旧复杂度的框架或兼容层。

只报告有具体代码依据、会影响本次目标的问题，按严重程度给文件位置、触发条件和最小修复。修复本轮引入的明确问题并运行对应验证，不以同种子随机轨迹、真实后端或外部兼容的未执行检查冒充通过。满足任务书验收后停止，不为减少几行再扩展范围。
```
