# Codex 任务书：精简 dexmani_policy，完成策略解耦与公开加载边界

> 核对日期：2026-10-27  
> 代码基线：`haoyangzhanglab/dexmani_policy@e384a268116c6a04bc1352769bc664c7658bb2f2`（main）  
> 配套消费者核对范围：`dexmani_real@d5ce61145a10f8d426bc97445379613f8acfb3cb` 的策略加载与 warmup 调用边界。  
> 本文是待执行任务书。编写时完成了源码、提交差异和调用关系核查，没有运行训练、推理或仿真；下述验收项不是已通过结果。

## 1. 目标和执行原则

在最新代码上继续做有证据的减法：正确、简洁、好用、模块清晰，策略可独立增删。保留各方法的官方来源与已声明的本地适配，减少公共代码中的方法特例、重复配置、重复解析和无关兼容检查。

开始前阅读根目录 `AGENTS.md`，检查实际 HEAD、工作区改动及相关调用方。若代码晚于本文基线，先核对对应差异，已完成项直接跳过，不机械重复任务。

本轮主体是本仓代码、配置、入口和文档。可以在同一可访问工作区内，对 `dexmani_real` 做公开加载和 warmup 接线的最小配套修改；不修改机器人驱动、控制调度、传感器同步或硬件动作语义。若 Real 工作区不可访问，完成 Policy 内可独立完成的部分，保持现有消费者仍可运行，并明确列出尚需配套的具体调用点。

优先使用现有文件、小函数、普通 dict 和直接 Hydra `_target_`。不新增 Registry、Factory、Manager、通用 Context 生命周期、统一 HandPrior/Codec、Validator 框架、来源认证/哈希系统、配置迁移器或强制推理 bundle。不为少量重复创建新基础设施。

## 2. 最新版本已经完成的内容：必须保留

与前一轮讨论的 `e717f746` 相比，当前数据层已完成一轮重构。以下不是待办：

| 已完成内容 | 当前位置 / 处理要求 |
| --- | --- |
| 普通 Dataset 已去掉大型 `describe_training_data` 和 `data_recipe` 生产逻辑 | `datasets/common/robot_dataset.py`；不要换个名字重新建立大 metadata 装配器 |
| `RobotDataset.get_normalizer(spec)` 已落地，validation 复用训练 normalizer | 保留该接口，不改回前轮曾讨论的 `build_normalizer` |
| 普通统计已使用选定 train episodes 的全部真实帧 | 保持；不要再次改变统计总体 |
| 坏窗口过滤已改为对选定 train/val 实际消费字段、列的一次分块检查 | `datasets/common/validation.py`；保持直接报错，不恢复静默过滤 |
| Sampler 已收敛到索引、padding、keys、`key_first_k` | 保留；`source_rows` 仍被 LAMP/LF 使用，不能因名字相似误删 |
| normalizer 已接受显式 feature shape / `last_n_dims` | 保持现有 FP64 合并、FP32 affine、Gaussian N−1、近常量处理和保存键 |
| R3D auxiliary 已拆为短 `R3DAuxDataset` | 保留独立类，不塞回普通 Dataset |
| DQ/LAMP 预训练已直接读取需要的手部字段 | 不再实例化完整视觉 Dataset；下一步解决它们彼此之间的依赖 |
| LAMP 离线历史已抽出 `load_hand_history`、`hand_history_for_rows` | prior 与 policy 数据继续复用，不重新实现历史规则 |
| 普通 split 已取消路径绑定，新产物不再输出 exclusions | 保留实际 train/val、episode 边界及已有 source/episode/trial 身份 |
| 实际划分已内嵌于 `dataset.split_manifest` | resume 直接复用 |
| Real 运行元信息仍由实际训练数据提取并随有效配置保存 | `deployment/metadata.py::describe_real_runtime` → `assemble_effective_config` → inspect/load；无需新增部署 manifest |
| 旧任务书及 `tests/test_data_paths.py`、`tests/test_evaluation.py` 已删除 | 不引用这些文件作为现有测试入口，不整套恢复已删除测试 |

当前 normalizer 静态形状路径是自洽的：动作/状态保留 D，点云保留 C，指尖/指级触觉保留 (5,3)。没有证据要求重写其数学实现。全帧统计与旧窗口覆盖统计可能不同，但该协议变化已经发生，是本轮基线。

## 3. 目标职责与必须保持的行为

| 模块 | 保留职责 |
| --- | --- |
| Dataset | 所需字段读取、实际 split、窗口、增强、统计样本及必要 query reference |
| Policy | 编码、归一化、方法表示、loss、采样、解码、输出物理 `future_actions` |
| Trainer | DataLoader、autocast、反传、optimizer/scheduler/EMA、日志、checkpoint、既定 resume |
| evaluation / SimRunner | 模型选择、冻结评测协议、环境交互、观测历史、动作执行和指标 |
| deployment | 无 Torch 的需求预览、公开加载、现有 NumPy 推理边界 |
| dexmani_real | 观测构造、控制执行和方法确实需要的执行反馈；不复制模型恢复逻辑 |

保持现有目录和模型模块路径，避免无必要的全仓搬家。共享 UNet、Flow、DiTX、点云编码器等实际计算组件；具体 Policy 不互相导入。

保持当前动作时间与物理语义：`obs[t]_before_action[t]`；关节模式为 7 arm + 12 hand、radians；EEF 模式沿用现有 xyz/rot6d/hand 布局。只在既有位置做模型空间转换与反转换，Runner 不再反归一化。R3D auxiliary 维度不能进入控制输出。

## 4. T1：优先消除当前真实的跨策略依赖

### 当前问题

1. `training/common/data.py` 顶层导入 `LatentFlowDataset`，以 `isinstance` 组装 hand_data。删除 LF Dataset 会使普通 DP3 训练入口无法导入。
2. `training/common/config.py::validate_config` 无条件导入 LF 和 `R3DAuxDataset`，用 `issubclass` 判断方法配对。删除其中一个会阻塞其他策略配置检查。
3. `datasets/hand_prior/data.py` 同时包含 DQ/LAMP 准备代码，顶层导入 LAMP 历史函数；删除 LAMP 会连带破坏 DQ 预训练导入。
4. `smoke_test.py` 顶层导入 LAMP/LF Dataset，并通过类型分支检查 context、打印 LF 专属恢复诊断。

### 修改要求

- 公共训练、配置检查和 smoke 不直接导入具体方法 Dataset，也不改成策略名称字符串白名单、动态注册表或一组新的 `needs_*` 标志。
- 方法特有的配对、历史和辅助目标约束移回已有方法初始化或数据实现。公共代码保留确实必要的 action width、观测字段与动作窗口一致性。
- 涉及未来动作起点的真实一致性，使用实际数据语义和现有模型 `action_start_index` 等值检查；若需要补一个直接的数据属性，应表达真实含义，不创建能力描述体系。不能只删除类型检查而允许 LF 使用错误的监督时间窗口。
- DQ、LAMP 数据准备各归自己的现有预训练/数据文件。真正共享的短读取/split 函数不导入任何方法。允许少量直观代码重复，不保留一份按方法分支的大准备器。
- 继续使用 `initialize_training(data_description)`；不传 Dataset、ReplayBuffer 或 Zarr 句柄。
- 公共装配只转交实际 split、normalization dims、action key、原始 attrs 和确实需要的少量数据事实。LF 已解析的语义可直接从其普通属性取值；公共代码不识别 LF 类、不构造其 contract/provenance，也不新增通用 metadata 钩子。
- 同步收窄 `assemble_effective_config` 当前对 data_description 的整体 update：初始化临时 facts 不全量写回有效配置。只保存现有必要的 normalization_dims、real_runtime、实际 split 和方法确实需要恢复/解释的语义；不要把大 attrs/来源描述重新复制进 checkpoint 并参与严格 resume 比较。
- 删除仅用于驱动公共手部装配的 `requires_hand_pretraining`、`requires_split_manifest` 及已无消费者的分支。保留仍承担动作窗口、normalization、EMA teacher 等实际含义的现有接口。
- 通用 smoke 保留公共 batch、dtype、有限值、参数更新、保存恢复一致性。方法专属 history/reference 语义由局部实现或少量定向检查覆盖；LF 专属 MAE/RMSE 诊断从公共 roundtrip 移出或删除。

完成标准：普通策略不依赖另一方法的 Dataset/预训练文件；删除方法时只需删除该方法自己的实现、配置、专用检查和必要接线。

## 5. T2：按方法收窄预训练产物检查

新版 `validate_pretraining_data` 已取消整份 manifest 全等和路径比较，但 `describe_hand_data` 仍产生 dt、H、obs horizon、padding、history，调用方仍逐项比较。对于单帧 DQ 码本，这些时间窗口绑定仍过度。

| 方法 | 必须保留 | 本轮应删除 / 避免 |
| --- | --- | --- |
| DQ-RISE | hand 维度、顺序、单位、原型表；实际同源 split；policy 与表的 hand affine 一致 | 单帧码本与 policy dt、H、obs horizon、padding、history、窗口数的相等要求 |
| LAMP | 原始 radians、历史长度、历史与目标定义、必要时间语义、实际 split | 无关诊断计数和通用整份配置全等 |
| LatentFlow / HandVAE | codec 结构、完整 H、dt、query reference、相对命令语义、同源 holdout | 统一为 DQ/LAMP 的兼容函数，或强制独立语料必须与任务数据完全相同 |

检查放在方法产物首次加载/绑定处，内部代码不重复验证同一事实。DQ 保留“先加载码本，再 `load_normalizer` 核对 hand scale/offset”的顺序。DQ 单帧预训练也应停止调用会强制要求 dt 存在的通用 `describe_hand_data`；LAMP 的时间条件和主策略/Real 的实际控制周期要求继续保留。

保留旧 artifact 的实际统计总体风险判断：旧产物使用过滤后源行，新产物使用全部训练帧，不能仅因字段变少就宣称可互换。不新增自动转换、自动重拟合、隐式 warm start 或版本迁移框架；不匹配时提示重新生成对应产物即可。

普通 split 已经不按路径认定身份，不再修改为强路径绑定。LF 当前用已有 `source_id` 或路径识别同源；相同数据跨路径使用时应保留已有 source_id。不同路径不是“独立语料”的证明；无稳定身份时明确当前判断边界，不新建数据认证系统。

## 6. T3：减少 DQ/LAMP 常规使用步骤

### 6.1 让显式 manifest 成为可选输入

- DQ/LAMP 预训练和 Policy 都调用现有 `split_episodes`，使用同一组明确的 `dataset.seed / val_ratio / max_train_episodes`。当前 `prepare_hand_source` 只传 manifest，必须同步转交上述三个参数，不能只删除 manifest 必填检查而继续使用函数隐藏默认值。
- 在对应 YAML 中明确默认划分参数；沿用文档常用的 seed=42、val_ratio=0.1、max_train_episodes=null。这是此前必须手工传 manifest 时不存在的默认路径，应在文档说明。
- 显式 manifest 优先，继续按其实际 indices 工作，不再把它作为所有用户的必经手工步骤。
- 两阶段都保存实际 split，并在加载 prior 时核对实际结果；只用相同 seed 不能代替结果一致。
- 不从主训练 seed 隐式改变数据划分，不让 VQ 与 Policy 使用各自不同的隐藏默认值。单 episode smoke 显式使用 val_ratio=0。

### 6.2 VQ 结束自动保存最终表

`training/pretrain/hand_prior/vqvae.py` 结束时已加载 best 并构造 `runtime_codebook` 用于 diagnostics。直接用已有 `save` 方法和现有文件写入工具保存同一对象为 `codebook.npz`，打印产物路径。

不再重复构建或重新选择 best；独立 exporter 继续作为显式重导出工具。正常流程应成为两步：

1. VQ 预训练，得到 best checkpoint 与 codebook.npz。
2. DQ Policy 训练，指定 codebook_path。

### 6.3 放开 DQ 的 H/C 实验默认数字

删除构造器对 H=20、n_action_steps=16 的硬锁；默认 YAML 数值不变。保留当前 n_obs_steps=1、7+12 关节布局、UNet 长度要求及合法动作窗口。

同步检查 `dqrise.yaml` 的 `pad_after=15`：若它的含义是 n_action_steps−1，用直接插值消除该重复常量；新组合仍由已有窗口/网络条件约束。不要自动猜测补齐方式，也不顺带增加未实现的观测设计。

保留当前 learned-weight 导出、half-up 索引解码、实际 M−1 缩放和本地 Sparse3D 编码器。它们已有适配说明，不静默切回官方另一数值实现，不增加 official/legacy 双模式。

## 7. T4：补全现有公开加载入口，共享一份恢复实现

### 7.1 目标接口

```python
from dexmani_policy.deployment import inspect_policy, load_policy

info = inspect_policy(
    experiment,
    checkpoint="best",
    weights=None,
    inference_steps=None,
)
runtime = load_policy(info, device="cuda:0", seed=0)
```

- 扩展已有 `inspect_policy`，接受 best/latest/现有 milestone selector/显式路径，继续兼容当前显式路径加 weights/NFE 的调用方式。
- 在现有 `common/experiments.py` 等合适文件内整理一个内部解析函数，普通 evaluation 复用；不增加 ResolvedPolicy、Bundle 或 loader registry。
- 保留已有 inspect 显式路径调用的路径语义；evaluation 的目录约束仍由自身边界负责，不因复用扩大或收紧其他调用方。
- `inspect_policy` 解析一次；`load_policy(info)` 不重新读取 best 或另选权重。
- 显式 weights/NFE 覆盖默认值；best 默认取 selection 结果，其余默认沿用已保存 eval 设置。缺失 EMA 继续明确失败，不自动换 raw。
- 保持 inspect 不导入 Torch、不构造模型。保存的 YAML 负责预览需求；checkpoint.effective_config 与 state 决定实际模型。
- 加载时保留一次影响推理语义的 YAML/checkpoint 一致性检查，不扩展为训练来源/路径/完整配置全等。

### 7.2 恢复函数只保留一份

将已有 `restore_policy_agent` 及确有必要的推理配置投影移入 `common/checkpoint.py`，同步更新：

- `evaluation/rollout.py`
- `deployment/runtime.py::load_policy`
- `smoke_test.py`

删除旧位置的实现，不长期保留两份包装。推理恢复与完整训练 resume 继续分别处理，不增加一个带大量布尔参数的总加载函数。

清理推理恢复中重复维护 prior/codec/codebook 初始化路径的名单，优先复用已有 `prepare_restore_config` 扩展点对构造配置副本做处理。当前 DQ/LAMP/LF 尚未重写该方法，若删除公共名单，必须同步在对应方法内处理初始化路径，不能假定空 hook 已完成这件事。保留原始保存配置；外部初始化文件移动或不可用不影响完整权重恢复，架构、预处理和动作语义差异仍须拒绝。

### 7.3 保持仿真与 Real 的边界

- Sim 直接调用 Tensor agent；Real 使用现有 NumPy Runtime。
- 普通仿真可使用 selection 中的 action_steps；Real 的执行长度仍来自 `execution.action_steps`，未指定时用模型 n_action_steps。
- frozen test 消费保存的具体 checkpoint、seeds 和推理设置，不重新解析 best。
- 20/40/60/80/100% milestone、完整候选要求、成功条件和 selection/test 协议不变。
- 已解析 latest 路径不等于冻结文件字节；正式结果继续使用既有 milestone，不新增快照/哈希系统。
- 保留实际 canonical 数据提取的 real_runtime（dt、点云/深度配方、fingertip names 等），不从 encoder FPS 点数反推采集点数。

T4 的 Real 加载配套仅涉及 `deployment/run.py`：删除对 Policy 内部 common/evaluation 模块及 best_ckpt.json 的重复解释。Policy 接口扩展应保持尚未更新的显式调用可用；T5 的 warmup 配套另按其调用边界处理。

## 8. T5：把剩余方法特例收回局部实现

### 8.1 LAMP 在线历史与 warmup

当前离线历史函数已完成复用，剩余问题在 SimRunner 和 Runtime：

- 在现有方法局部文件中用小函数表达在线 history 初始化和追加；不要在线构造一份伪离线 Dataset 来复用离线索引函数。
- SimRunner 保留一个必要接线点，按已有 metadata 接收/传递 action_context，避免顶层导入所有方法。
- 保留过去命令 → 当前命令的本地定义；初始实测手姿补齐；每 episode 清空；只追加环境实际处理后的 `executed_joint_action`。
- 不把尚未执行的预测加入历史。传给一次推理的 context 不应被随后更新原地污染。
- warmup 增加调用方提供的可选 action_context，并保留 RNG 状态恢复。LAMP 使用临时历史，正式 episode 重新初始化。
- 删除 warmup 内部的 LAMP 历史构造，必须与 Real 调用方一起更新。Real 工作区不可访问时，先保持该现有分支，明确配套位置；不新增静默零历史或兼容工厂来假装完成。

不向 BasePolicy 添加 initial/update context 钩子，不增加 LampRuntime、ContextManager 或跨方法状态协议。几行明确接线或数组操作的重复可以接受。

保持 LAMP 原始 radians、identity action / Gaussian state、mu+delta_z、decoded hand/drift loss；冻结 prior 参数但允许 decoder 对 delta_z 传梯度。本轮不把现有 19D state 改成 arm-only，也不改当前两层 prior-feature MLP。

### 8.2 R3D auxiliary 损失

当前 `agents/action_decoders/diffusion.py::compute_loss` 仍按 group 名字是否为 `"joint"` 决定权重，公共 decoder 因此认识 R3D。

将“哪一组加什么权重”放回 R3D；共享 decoder 只处理显式数值参数或普通损失计算。选择直接、改动最少的实现，不新增 LossRegistry、loss 插件或通用训练 callback。

保持每组分别求 mean 后加权相加，不能换成所有维度一次 mean；控制输出仍只取 joint 维度。已有 `R3DAuxDataset` 保持。

## 9. T6：只删除已证明重复的恢复检查

当前 resume 主链是清楚的，不立项重写整个恢复系统：

- `prepare_training` / `resolve_input_recipe` 使用 checkpoint 保存配置。
- Dataset 使用保存的实际 split。
- normalizer 从保存权重恢复，不 refit。
- Trainer 恢复优化器、调度器、EMA、RNG 与计数。

`validate_resume_protocol` 已在 `resolve_input_recipe` 调用，`build_dataset_and_normalizer` 又重复调用。核对当前全部调用方后，保留入口检查，删除装配层重复检查。

清理公共恢复代码中的方法产物路径名单时，必须同步实际使用者。fresh init 属于具体方法；full restore 使用内嵌参数，不应要求重新提供旧 prior 文件。

不要因为“减少校验”直接删除所有 `validate_source_recipe` 调用。Trainer 对本次实际数据及构造配方的核对，与 `archive_resume_milestones` 对候选 checkpoint 来源的核对用途不同。本轮只删除已经证明重复的 `validate_resume_protocol` 调用；其余 `validate_source_recipe` 调用默认保留，除非能指出某项比较已被哪个具体检查完整覆盖。归档候选仍需保持预算、步数、方法及训练实验设置一致，不新增递归比较框架或来源证书。

兼容范围必须准确：

| 对象 | 保持的行为 |
| --- | --- |
| 带旧 data_recipe 的 checkpoint 做 full resume | 继续明确拒绝，使用旧代码续训或新开实验 |
| 新协议 full resume | 保存的 split、affine、优化状态继续生效 |
| 旧 checkpoint 纯推理 | 保持使用保存参数和 affine 的路径；有代表性旧产物则验证，没有则注明未验证 |
| 旧独立 prior/codebook | 按该方法实际坐标和统计条件处理，不由普通推理兼容推导 |

不要自动迁移、自动 refit、隐式 warm start，或把 full-resume 拒绝扩散为所有推理都拒绝。现有 epoch 级恢复语义、optimizer-update 计数和 milestone 规则不变。

## 10. T7：做一层以内的配置去重，更新使用文档

已核对 8 个策略 YAML 的 workspace/hydra 段相同，可在现有 configs 中最多抽一个浅层公共 YAML，用直接 defaults 合并，保留清楚的覆盖顺序。

只提取真正相同的运行配置；DataLoader 如要合并，只合并相同执行项，batch_size 保持策略显式设置。EMA、训练预算、NFE、action window、增强和评测设置存在真实差异，不整体归并。不引入多级 config groups、策略配置注册表或自动参数推导。

此项是删除实际重复，不是强制整理所有 YAML。如实际 HEAD 已去重，则跳过。对独立 hand_vae 配置按自身需求处理，不为了形式一致强行继承。

同步 README、现有架构/手部表示/评测文档和相关脚本示例：

- 普通训练入口不变；
- DQ/LAMP 可直接使用明确 split 默认值，也可传已有 manifest；
- DQ 正常使用两步，独立 exporter 为可选重导出；
- 公开 inspect/load、Real 参数来源、旧 full resume 限制清楚；
- 删除已不存在 API、重复参数说明和旧任务步骤；
- 不重新生成多份相同架构文档。

## 11. 验证：覆盖改动风险，避免扩成测试平台

当前已无前述两个 tests 文件。优先复用现有 smoke；必要的确定性小样本检查可以保留为少量定向测试，不恢复整套旧测试，也不写镜像实现或为新抽象配套的测试。

| 改动 | 最小验收 |
| --- | --- |
| T1 解耦 | 在临时副本/隔离导入环境分别屏蔽 LF、R3D 私有 Dataset，DP3 配置与训练入口仍可导入；屏蔽 LAMP 后 DQ 预训练仍可导入。不在用户工作仓库真实删除策略 |
| 初始化与 split | 同一数据、同一 split 参数，两阶段实际 indices 一致；显式 manifest 优先；真实不匹配仍报错 |
| DQ | 极小预训练产出 best 与可加载 codebook；自动保存与 exporter 用同一构建函数；合法改 H/C 不再因无关元数据拒绝，hand affine 改变仍拒绝 |
| loading | 复用 raw/EMA roundtrip；selector 和显式覆盖正确；load 不重选；缺 EMA/推理语义冲突明确失败；内嵌 prior 不依赖原文件 |
| LAMP / LF / R3D | 仅对实际改动路径检查历史、query reference、冻结 decoder 梯度、分组损失或控制切片；不跑整套算法重训 |
| resume | 新协议短恢复一次 update，split/affine 不重算，计数和优化状态正确；旧 full resume 仍拒绝 |
| 配置去重 | 对比展开后的有效配置，除运行目录/run_id 及 T3 明确新增的 split 默认路径外，不改变算法/评测设置 |
| Runner 配套 | 只有触及环境交互时才做短 episode 检查；frozen test 和 Real action_steps 来源不变 |

现有命令示例（需要相应依赖；data/training 模式还需要可用数据和 prior）：

```bash
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise lamp latent_flow maniflow r3d sat
python dexmani_policy/smoke_test.py --data-only --max-batches 2 dp3
python dexmani_policy/smoke_test.py --max-updates 2 dp3
```

smoke 当前接收配置名或显式测试 YAML 路径，不接受任意 Hydra key=value 作为配置名。需要小数据/CPU/关闭 compile 的验证时，用临时 YAML 指向 fixture；不要修改正式默认配置来让 smoke 通过。

未触及 normalizer 时，不要求重跑大规模统计测试。若实际修改了其数值实现，再用 [T,D]、[T,N,C]、[T,5,3] 的分块统计与直接统计做一次对照。保持 RGB、EEF rot6d 和 LF 相对目标规则。

不安装整套训练环境来替代源码工作，不启动长训练或真机动作。缺依赖、GPU、数据、prior 或旧 checkpoint 时，完成其他可执行任务，并报告具体未验证项。不能用 mock 输出声称真实训练/仿真已经通过。

数值检查只保留在实际边界或会掩盖错误的有损变换前。例如 DQ float index 转整数/clamp 前不能盲删 finite 检查；同一输入在 Dataset/Trainer/Policy 的重复检查则应合并。不要增加一层通用验证系统。

## 12. 实施与交付

推荐顺序：T1 → T2/T3 → T4 → T5 → T6/T7 → 必要验证与文档。每个范围内的调用方一起更新，不留下临时双实现。

最终交付应包括：

1. 实际修改和删除项，尤其公共代码不再依赖哪些方法。
2. 已完成保持项，以及因 HEAD 已完成而跳过的项目。
3. 实际执行的验证命令和结果；未验证内容及具体原因。
4. 对新 split 默认路径、接口配套等用户可见变化的简短说明。
5. 若 Real 不可访问，准确列出仍需配套的调用点，不能宣称其启动已验证。
6. 核对没有新增无必要抽象、兼容模式、元数据重复、配置开关或校验层。

完成标准是依赖和调用链更直接、公共职责更少、操作步骤更少。不能只移动文件而保留原依赖；不能只更新这份任务书而不完成已具备条件的实现。

## 13. 官方对照：采用具体实现经验，不照搬全部工程

以下固定版本作为已读参考；它们是来源证据，不表示要求迁入全部算法或运行系统。

| 项目 | 参考版本 / 主要代码 | 本轮采用的原则 |
| --- | --- | --- |
| Diffusion Policy | [5ba07ac：Dataset](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/real_pusht_image_dataset.py) | 直接 Dataset/Sampler/Policy/Runner 分工；不照搬其所有统计总体 |
| ManiFlow | [ef2f116：Dataset](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/dataset/adroit_dataset.py) | 简单数据与策略边界；consistency 所需 EMA teacher 保留 |
| RDT-1B | [cd79363：模型包装](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/cd79363a1387e8f81c7724d070ef7e45fd23150f/scripts/agilex_model.py) | embodiment 输入输出映射局部化，不放进共享 Trainer |
| AFRO | [283024a：Policy](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/policy/afro_policy.py) | 表示预训练与策略训练分开，权重由实际消费者接收 |
| CordVIP | [ad6d441：Dataset](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/dataset/robot_dataset.py) | 数据、几何与模型输入有明确位置，不复制整套真机脚本 |
| TransDex | [a415187：训练](https://github.com/LFGfg/TransDex/blob/a4151879701bfb85a45447fa6d99fea07dd8c7dc/policy_ws/src/VTFusion/src/training.py) | 阶段与模态处理按实际任务组织，不引入其整套运行依赖 |
| T-Rex | [ba4047c：数据与动作约定](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/utils/lerobot_common.py) | 明确坐标、时间与转换位置；慢快分支和服务结构不通用化 |
| DQ-RISE | [2889d27：Policy](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/policy.py)、[码本导出](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_vqvae.py) | 离线原型表、在线连续索引；不搬入硬件驱动或每次查询读表 |
| LAMP | [0467ba3：BC Policy](https://github.com/dex-lamp/LAMP/blob/0467ba3504e0d199727a0586ba525c5fe14f8138/imitation_learning/common/model/policy_jax.py)、[历史数据](https://github.com/dex-lamp/LAMP/blob/0467ba3504e0d199727a0586ba525c5fe14f8138/vae/model/data_io.py) | 方法局部 prior/history；官方含当前 pose 预测下一 pose，不等同本地过去命令预测当前命令 |
| UniHM | [ae8bf39：SFT](https://github.com/Zhenhao-Zhang/UniHM/blob/ae8bf39761d796e6ced99c6cc4880c6d810bf68d/UniHM/train_sft.py)、[推理](https://github.com/Zhenhao-Zhang/UniHM/blob/ae8bf39761d796e6ced99c6cc4880c6d810bf68d/UniHM/inference.py) | 阶段与权重交接明确；SFT decoder 可训练，不能推导所有 codec 一律冻结或统一 token runtime |

补充正确性约束：

- LAMP 的冻结 decoder 仍允许 loss 对 delta_z 反传；LF 的监督目标编码可 no_grad，而 decoded loss 必须对 flow 保留梯度。
- LF 的 query q0 是实测参考状态，与 UniHM 的跨手型 reference encoder 不是同一概念。当前默认 HandVAE 数据指向任务数据；独立语料是允许的用法，不是默认事实。
- UniHM 发布 inference 入口使用 hidden → from_qwen → decoder；不能因存在其他 autoregressive 辅助函数就要求本仓建立统一 token 接口。
- 对官方严格复现、LAMP arm-only 输入、新 VLM/多手型支持、硬件时序修改，另开算法或系统任务，本轮不混入。

## 14. 可直接复制给 Codex 的执行 Prompt

```text
请在当前 dexmani_policy 工作区实际执行根目录 CODEX_POLICY_SIMPLIFICATION_TASK.md 的任务，先完整阅读 AGENTS.md 和任务书。

任务书基于 e384a268116c6a04bc1352769bc664c7658bb2f2。先检查当前 HEAD、工作区改动和相关调用方；若 HEAD 已变化，核对差异，保留已经完成的简化，不能机械套用旧方案。不要只给计划或只修改任务书。

目标是正确、简洁、好用、模块清晰解耦，并删除已有冗余。沿用独立 Policy、直接 Hydra _target_、共享 Trainer、现有 Dataset/get_normalizer、checkpoint、评测及 Runtime。不要新增 Registry/Factory/Manager、通用 context 生命周期、统一 HandPrior/Codec、schema/校验框架、文件指纹、迁移器或强制推理 bundle。

优先处理 T1 的真实依赖：公共 training/config/smoke 对 LF/LAMP/R3D 私有 Dataset 的导入，以及 DQ 预训练对 LAMP 的间接依赖。继续完成 T2–T7：收窄各方法产物约束，DQ 自动保存码本与合法 H/C 配置，明确且可选的共享 split 输入，公开 inspect/load 和单一恢复实现，局部 LAMP/R3D 特例，已证明重复的 resume 检查及浅层配置去重。按实际依赖顺序修改所有调用方，不留下临时双实现。

已经完成的全训练帧统计、数据边界检查、简单 Sampler、R3DAuxDataset、离线手部读取与历史函数、real_runtime 保存链路必须保留，不重新设计。保持现有算法、动作时间/单位、LAMP 过去命令历史、LF 未增强 query reference 和 decoder 梯度、R3D 分组损失及控制切片。保留 milestone、selection/frozen test、新协议 resume；旧 data_recipe full resume 继续明确拒绝，纯推理兼容单独验证。

主要修改本仓。若同工作区可访问 dexmani_real，只做公开加载和 warmup 调用的必要配套，不修改硬件或执行时序；不可访问时保持现有消费者可用并准确记录配套点，继续完成其他任务。

验证使用现有 smoke 和与改动风险匹配的少量定向检查，不恢复整套已删除测试、不建立测试平台、不启动长训练或真机。没有依赖/数据/产物的项目明确标为未验证，不能伪称通过。常规实现选择按最小完整改动自行推进，发现当前代码已满足某项则跳过并说明。

完成后用中文报告：修改与删除内容、为何更解耦、保持的行为、用户可见变化、实际验证命令与结果、未验证项及原因。最后检查是否引入了可删除的抽象、配置、重复逻辑或校验，并完成清理。
```
