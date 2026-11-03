# DexMani Policy 轻量化改造任务书

日期：2026-11-03。审查与方案基线：[`156b6bfa17a4d0edca31cea913ba8ec8975d1b27`](https://github.com/haoyangzhanglab/dexmani_policy/tree/156b6bfa17a4d0edca31cea913ba8ec8975d1b27)。

本文汇总最近三轮代码审查，并给出一套默认执行方案。**本文是待实施任务书，不表示修改或训练验证已经完成。** 实施者先读 `AGENTS.md`，再核对当前代码；若基线之后已有等价修改，直接跳过，不重复实现。

目标是让个人机器人研究代码正确、直观、易改、好用：少维护一套状态、少走一层流程、少暴露一个无实际用途的选项。允许收窄支持范围、减少提前诊断和历史兼容，不以生产级可靠性、任意组合、逐位重放为目标。

## 1. 最终决策

本次执行 T01—T10。第 7 节的研究变更和跨仓库任务不纳入本次实施。

| 方向 | 本次决策 | 明确接受的代价 |
| --- | --- | --- |
| Resume | 保存的配置与完整训练状态拥有恢复语义；只在 fresh 装配有效配置 | 不支持换数据、换 batch/累积步数、改总预算后仍声称原实验无缝续训 |
| 输入字段 | `dataset.sensor_modalities` 是配方输入列表；删除模型独立声明和多层自动过滤 | 不承诺自动忽略多余模态；错误覆盖可在 forward 直接报 `KeyError` |
| 精度与 EMA | 只维护 FP32 master parameters、同设备 EMA；BF16 仅作 autocast | 不维护历史低精度 master parameters 的原精度续训轨迹 |
| Checkpoint 选择 | 保留 `latest` 和明确文件路径；删除百分比 selector 与 `best` 自动发布链 | 选模后手动使用获胜文件及记录的 weights/NFE/执行长度 |
| 保存 | 单 run 单写者；保留同目录临时文件加 `os.replace` | 不承诺并发创建排他、断电持久性、多文件事务、边训练边同步的一致快照 |
| 热路径诊断 | loss 有限性合并到 update 边界；保留已有裁剪；不裁剪时不逐参数扫描梯度 | 非有限 loss 最迟在本 accumulation group 结束发现；不穷尽检测有限 loss 对应的异常梯度 |
| 测试 | 检查算法、数据和真实使用结果；删除对象回收时序与人工协议矩阵 | 不把每个内部调用顺序、异常组合和 PyTorch 原生行为变成项目承诺 |
| 部署与研究能力 | 保留薄的 `PolicyInfo` / Runtime、RTC、选点与独立测试、可选预训练验证 | 不为了减少行数而把真实消费者需要的功能拆散或复制到设备项目 |

这些是有意的范围选择，不需要实施时逐项再次征求确认。发现新的实际消费者或数学冲突时，先选择保持正确的最小实现，并明确报告该项为何不能按原方案删除；不要补兼容平台来掩盖冲突。

## 2. 三轮审查的合并结论

### 2.1 已确认的问题与维护负担

| 发现 | 性质与触发条件 | 处理 |
| --- | --- | --- |
| Resume 再次调用 `assemble_effective_config` | 同一保存配方重新派生、比较、复制；还会用当前数据 attrs 改写保存的 dt/点云配方 | T01：删除恢复装配回路，保留保存事实 |
| 主策略与观测预训练长期持有完整 CPU checkpoint | 恢复完成后调用栈仍保留不再消费的原始状态；主策略使用 mmap，不能一概称为堆内存泄漏 | T02：缩短引用生命周期 |
| EMA 按 device/dtype/cast 分组 | 当前标准入口没有混合 master dtype 或跨设备 EMA 需求 | T03：收窄精度支持后删除分组与转换 |
| AFRO resume 无条件重算总更新预算 | 数据批数改变时覆盖保存预算；同数据正常 resume 不因此必然出错 | T04：仅 fresh 推导预算，不建立数据身份校验 |
| 关闭训练 EMA 后默认评测仍请求 EMA | 多份 YAML 的两处默认值独立，用户只改一处就会加载失败 | T05：默认插值，显式缺失 EMA 仍报错 |
| 主策略短训没有 metrics | 小于日志周期的 fresh/resume 可能没有任何指标；进度条只在日志时推进，退出时才校正 | T05：首个实际 update、周期、已知末次 update 记录 |
| fresh 输出目录冲突发现太晚 | 在模型、外部权重和统计初始化之后才失败 | T05：在昂贵初始化之前检查一次 |
| 未消费输入的声明与过滤链 | 14 处声明/property 与多个消费者维护同一字段集合；13 种默认组合没有利用额外输入能力 | T06：整链移除，不用另一套 schema 替代 |
| 每个 micro-batch 的 CPU 有限性判定及无裁剪梯度扫描 | 源码可见 CUDA 同步和额外 kernel；没有实测吞吐百分比 | T07：缩减诊断频率和范围 |
| 百分比 selector 与 best 发布、同步、测试相互依赖 | 文件名与读端耦合；同一文件经 best/明确路径可能采用不同默认推理参数 | T08：删除附属工作流，保留实验结果 |
| 通用保存工具承担 hardlink no-clobber、强制 fsync | 超出单写者个人研究需要，且并不构成完整断电事务 | T09：收窄为简单原子替换 |
| 实现时序测试、人工无消费者字段、单一巨大检查入口 | 支持义务部分由测试自行制造；无关依赖妨碍局部验证 | T10：删除附属测试、按职责组织剩余检查 |

### 2.2 不应误报为 bug 或过度工程的内容

三轮静态审查未确认默认算法路径上的 P0/P1 数值错误。更新级计步、尾组累积、数据窗口与归一化、多阶段统计迁移等有已核对的正确实现；不应为交付“更多问题”而改坏它们。

- “五参考均未见同等机制”是进一步审查的线索，不是删除判据。按功能与真实消费者比较，不能只比名称或一次搜索结果。
- 带配置 checkpoint、输入 shape 元数据、strict load、优化器参数覆盖检查、EMA dtype 转换、best/latest 概念、独立预训练和轻量 encoder，在参考中有全部或局部等价物。它们不能被整体列为本地独有负担。
- 参考实现也有反射遍历 workspace、dill、后台保存线程、TopK 管理、重复统计拟合等复杂度或局限；本项目不照搬。
- 允许额外键、只读取所需键，在参考中也存在。本次删除的是本地默认配方没有需要的整套声明、投影、过滤、再裁剪链。
- best 当前只是按顺序写结果与指针，不是事务系统；训练和单点评测也没有被强制要求经过选模。本次删除是主动收窄便利功能。
- 当前 `PolicyInfo`、Runtime 与 RTC 已有 `dexmani_real` 实际消费者。模拟器部分构造清理也有外部源码依据，不能当成纯测试制造的需求。

## 3. 实施原则与支持边界

1. **保留直接的数据流。** 配方 → 数据/模型 → 训练状态 → checkpoint → 推理。继续使用现有小函数、显式字典和少量方法分支，不新增 Manager、Context、Contract、Registry、Adapter、StateMachine、schema version 或迁移框架。
2. **一次事实只有一个所有者。** fresh 配方决定构造；checkpoint 保存实际模型、统计和进度；恢复不重新获取“证明”。同源派生值不需要相互验证。
3. **允许可信内部调用失败得朴素。** 缺键、形状不符、非法研究组合可由实际消费点或依赖库抛错。不要为了统一错误格式重复扫描所有配置和张量。
4. **保护会静默改变实验的边界。** 模型权重 strict load、优化器位置映射、train-only 统计、物理坐标/尺度、未来标签时序、真实设备输入与执行窗口仍需正确。
5. **恢复限于同数据、同配方、同训练布局。** 研究者保证数据未改写、未重排；可原样搬迁并改路径。保留已有 device/worker/日志/compile 等明确执行覆盖，不改为任意 `OmegaConf.merge`。改变 batch、累积、split、算法或 scheduler 预算属于新实验，不在本次扩展。
6. **不承诺逐位重放。** 保存必要进程 RNG 和样本位置即可；不保存 worker RNG、预取队列、所有设备 RNG，不建立 compiler RNG 守卫或重放模式。
7. **不净化小配置快照来制造大架构。** 初始化路径及顶层便捷参数可以作为实验笔记继续存在，但不能成为恢复准入条件；不要拆成 fresh/resume/deploy 三套 schema。
8. **不追求机械去重和行数指标。** 主策略与观测预训练保留两个直接循环；少量有语义差异的重复优于通用 hook 系统。简化的收益应体现在支持义务和阅读成本，而非删除 assert 的数量。

## 4. 必须完成的任务

### T01 — 让 resume 直接恢复，不重新装配配方

主要位置：`training/common/config.py::assemble_effective_config`、`training/trainer.py::__init__`、`training/common/resume.py`。以下路径省略共同前缀 `dexmani_policy/`。

实施要求：

- `assemble_effective_config` 只用于 fresh；删除其 `checkpoint` 参数和同源 `action_start_index` / `control_action_dim` 比较。恢复时使用已从 checkpoint 创建并应用明确执行覆盖的 cfg。
- Resume 不刷新 `deployment_metadata`，不重新用数据 attrs 推导控制周期、点云配方、depth 尺度或指尖名称。也不为此新增 attrs/hash/来源一致性校验。
- Resume 的 `normalization_dims` 采用保存值；normalizer buffer 形状仍由实际 state 得到，随后 strict load。当前 Store 实际字段宽度与保存输入表的比较可保留**一次**，放在数据绑定位置；不比较时间戳、导出版本、路径身份或全量数据内容。
- 保留数据主动作宽度与 Policy 输出的真实边界，以及辅助 action 宽度的现有约束；它们不是同源 metadata 自证。
- 原始外部 RGB/Uni3D/手先验/码表/encoder 权重加载、架构解析和统计拟合继续只在 fresh 执行。完整 resume 不访问这些初始化来源。
- `max_updates` 只约束本次调用，不进入新保存的有效配方；不要用它改总训练预算或 scheduler。读取旧配置时的一行 `pop` 可以保留，不扩展成兼容层。
- 本次保留小型 `policy_metadata` 快照及 `PolicyInfo` 的既有字段。为删三项小字段而让离线 inspect 导入完整 Policy 类，净收益不足；删除的是其恢复自证回路。

完成判据：保存 checkpoint 后修改当前数据的非数值 attrs，再 resume/save，保存的部署事实不被悄悄改写；缺失初始化来源不妨碍完整恢复；统计不重拟合。此检查不代表支持更换数值数据。

### T02 — 恢复完成即结束原 checkpoint 的生命周期

主要位置：`training/common/resume.py::main`、`train.py::run_training`、`training/trainer.py`、`training/pretrain/obs/common.py`、`test_tools/smoke_test.py`。

- 主策略把“加载 checkpoint、构造并恢复 Trainer”放在一个短生命周期阶段，返回已就绪的 Trainer 后再进入长循环。可以用一个普通构造函数；不建立恢复管理器或所有权协议。
- 不仅在 `PolicyTrainer.__init__` 末尾 `del checkpoint`：外层 `main`、`run_training`、smoke 调用帧也不能继续无用地持有完整 payload。
- 观测预训练先保留一个 `resuming` 布尔及恢复时真正需要的小状态，完成模型/optimizer/scheduler/RNG 恢复后删除完整 checkpoint 引用。
- 不 `deepcopy` 已保存状态来“释放”它，不强制 `gc.collect()`，不每轮 `empty_cache()`，不增加 weakref 计数或内存所有权测试。CPU 模型实际复用的 storage 自然继续存活。
- 不把推理专用 `_restore_policy` 用作训练恢复；它会切 eval、统一推理精度并合并 LoRA。

完成判据：读取源码调用链即可确认训练循环不持有无用完整 payload；正常 fresh/save/resume 通过。没有真实 RSS 测量时，只报告生命周期改善，不报告内存节省数字。

### T03 — 收窄到 FP32 参数与简单 EMA

主要位置：`training/common/optimization.py::EMA`、`training/trainer.py`。

- fresh 与 resume 的训练参数统一为 FP32；可选 BF16 只作用于 autocast。加载、迁移/转换完成后再建立 optimizer 和 EMA 所持 Parameter 引用。
- EMA 保留两份对应的 trainable 参数列表、现有 decay 公式与批量平均调用；删除按 device/dtype/needs_cast 分组、EMA `.data` 重绑定和逐 update source 转换。
- 保留 raw/EMA 独立加载、冻结参数筛选、从 `global_step` 恢复 EMA warmup 计数。`mmap=True`、`assign=True` 本身不需因“非参考默认”而删除。
- 保留 optimizer 完整覆盖及 `param_names` 顺序核对。AFRO/CordViP 的部分参数组使用注册顺序，仅仅 strict load 或“另一些方法排序了名称”不能防止 Adam moments 静默错绑。
- 保留当前活动 BatchNorm 与 EMA 的约束。`obs_lr=0` 只冻结参数，不自动冻结 running stats/dropout；不能据此删除 BN 检查。
- 不新增历史 dtype 迁移器、precision 选项、异设备 EMA 支持。旧低精度参数 checkpoint 不再承诺保持原精度训练轨迹。

完成判据：标准 FP32 fresh/resume 与 BF16 autocast 路径保持有效；raw/EMA 各自状态与下一次 EMA decay 正确；优化器更新的是 forward 使用的当前参数。

### T04 — 整理 AFRO/CordViP 预训练恢复，保留方法差异

主要位置：`training/pretrain/obs/common.py`、`datasets/pretrain/__init__.py`。

- AFRO 的 `num_epochs * batches_per_epoch` 仅在 fresh 计算并保存；resume 使用保存的 `total_train_steps`。不因这个修正宣称支持改变数据批数后无缝续训。
- RNG 在 logger/config 初始化完成后、进入训练之前恢复。保持严格原地 `load_state_dict`：此路径的 optimizer 已绑定参数，不能照搬主策略的 `assign=True`。
- 对已完成预算的 resume 直接返回，不重写 `resume.pt` 和 `encoder.pt`；独立导出沿用现成导出入口。能在昂贵构造前由保存 step/预算判断时就提前返回，不新增 completed 标记。
- 停止规则与主训练一致：第一次信号完成当前 update 后保存；第二次允许主动放弃保存并退出。用简单布尔和现有 handler helper，不建 StopController、超时或重试机制。
- 保留 `resume.pt` 与 `encoder.pt` 两种产物：前者有完整训练状态；后者含 encoder 与输入 affine。周期只保存恢复状态，结束/受控中断再导出 encoder，不建立两文件事务。
- 保留可选离线 validation：默认关闭，但 held-out 预训练损失有直接研究用途，现有实现很短。保留 no-grad、模式/RNG 恢复、按样本数加权及中断不报部分均值。可删除仅用来阻止“验证视图再生成验证视图”的 `training` 标志与分支，由训练调用方只创建一次视图。
- 不强行合并主策略/预训练 Trainer；不统一 AFRO teacher-before-optimizer 与主策略 EMA-after-optimizer；AFRO VICReg 继续要求 accumulation=1。
- 本次不为 CLI 外观重写配置层。`--config`、显式运行选项及 W&B 习惯差异不是已证算法问题，保留现有小入口即可。

完成判据：两种预训练均可短训、保存、恢复、导出；恢复预算不重算，统计不重拟合；已完成 run 不改写产物；开启 validation 不改变后续训练的进程 RNG 状态。

### T05 — 让常用配置、短跑和启动顺序自然工作

主要位置：`configs/*.yaml`、`train.py`、`training/common/config.py`、`training/common/logging.py`、`training/trainer.py`。

- 普通配方 `eval.use_ema` 默认引用 `${training.use_ema}`，仍允许显式 raw/EMA 选择。需要 EMA 参与 loss 的 ManiFlow 继续拒绝关闭 EMA；请求不存在的 EMA 不自动回退 raw。
- 没有研究含义的重复推理步数默认值改为引用已有 agent 值；保留 ManiFlow 等有意不同的训练/常规采样/评测步数。AdapTac 的 force 与 action 两阶段步数不是一组可机械去重的参数。
- 主训练在**本次调用的第一个实际 update**、日志周期、已知 `stop_step` 的最后一个 update 记录指标。resume 的第一步不能只用 `global_step == 0` 判断。已知末次包括 bounded run；不为未知信号时刻强制缓存全部历史指标。
- 进度条可按已完成 update 轻量推进，指标仍按上述频率计算；不为显示进度增加 CUDA 同步。
- fresh 输出冲突检查移到配置装配之后、HF 架构访问/数据读取/模型构造/统计拟合之前；真实 fresh 入口及 smoke 共用相同简单检查。空目录可用于 smoke。不要提前启动 W&B 来检查目录。
- 检查的所有权只有一处，logger 不再重复同一 fresh 检查；预训练把现有产物检查也提前。单写者前提下不建立目录预约、锁或运行 ID 冲突协调器。
- 日志继续 append-only；不解析历史恢复进度，不修复/裁剪坏行，不要求 step 单调或与 checkpoint 一一对应。当前为残缺末行补一个换行可保留。

完成判据：EMA 可选配方关闭 EMA 后默认评测选 raw；4+2 短跑两段都有有效指标；误用已有输出目录在昂贵初始化前失败；不改变正常优化步数、损失权重和调度预算。

### T06 — 删除未消费输入的声明、投影和过滤协议

主要位置：`agents/policies/`、`agents/obs_encoder/`、`common/config.py`、`training/trainer.py`、`training/common/config.py`、`evaluation/rollout.py`、`env_runner/sim_runner.py`。

默认配方的唯一输入列表是 `cfg.dataset.sensor_modalities`。fresh 的实际字段宽度由 `build_dataset` 取得，保存为 `normalization_dims`；resume/部署使用这张保存表。

具体删除链：

1. 删除 `consumed_observation_fields` 的静态声明、Base 转发、AdapTac 两层 union，以及 `validate_observation_fields` 与全部调用；不新增 `required_fields` 等同义替身。
2. Trainer fresh 直接使用 `data_description['field_dims']` 配置 normalizer，不再按模型字段投影；resume 仍遵循 T01 的保存表与一次实际宽度核对。
3. 删除 `train_microbatch` 的 obs 过滤；删除 `Policy._preprocess_obs` 的字段丢弃，只保留实际预处理、归一化、点云通道裁剪、modality dropout、B/T flatten。
4. fresh 部署 metadata 已由 `build_dataset` 按唯一输入列表取得，删除第二次按 consumed 集合裁剪。
5. evaluation/Runner 从保存的 `normalization_dims` 去掉 `action` 得到输入；Runner 以自己已有的 `sensor_modalities` 做来源判断。离线 `PolicyInfo.observation_fields` 已使用保存表，不必导入模型探测输入。
6. normalization 的配方字段检查收敛到装配处一次；继承 encoder affine/identity 后再确定最终模式。不要新增所有 encoder 的独立字段 schema 来恢复早期错误诊断。
7. 删除证明“额外合法字段不拟合、不传模型、不部署”的能力测试及文档承诺；普通继承、正常输入和数值测试保留。

必须保留：

- 触觉模块真正用于索引的 `input_key`。
- AdapTac taxel/patch 的 **aggregate 与 dense 两类输入**：触觉 encoder 读 dense，current guide/未来力任务仍用 aggregate；不能只看 encoder 就删 aggregate。
- 顶层 future-force 辅助标签、有效 mask 与 R3D 辅助 action 拼接；它们不是多余 observation。
- `normalization_dims` 与从 state 推导 affine shape 的分工。identity 输入没有 affine buffer；DexLatent 的内部 `hand_latent` 不是外部观测，不能靠遍历全部 normalizer 字段反推输入。
- 实际物理语义：几何 identity、触觉零点/共享尺度、sim/real 来源、点云通道宽度。

完成判据：13 种默认组合的输入字段仍正确，正常 forward/推理不因删协议而变化；AdapTac taxel/patch 字段集合相同但声明顺序不同，不应误要求顺序一致或重排模型 token。错误自定义配方允许晚到实际索引时报错，不增加补偿性校验。默认原本没有额外输入，不能宣传默认 I/O 提速。

### T07 — 删除高频防御扫描，保留一层有用诊断

主要位置：`training/trainer.py::train_microbatch/optimizer_step/train`、`training/pretrain/obs/common.py::train`。

采用明确而有限的诊断策略：

- 不在每个 micro-batch 把 GPU loss 有限性转换为 Python bool。把该 accumulation group 中各次 detached loss 的有限性在设备端合并，在 **optimizer.step 之前**判定一次。实现只需一个局部标量/标志；不保存计算图或 per-batch 历史。
- 不只检查最后一个 micro-batch，也不依赖只有日志步才存在的 metrics；较早 micro-batch 出错仍应在本组更新前失败。
- `max_grad_norm > 0` 保留原有 `clip_grad_norm_(..., error_if_nonfinite=True)` 与裁剪阈值。删除 `max_grad_norm == 0` 分支的逐参数 `isfinite(...).all()` 扫描，不用 `max_norm=inf` 重新引入另一轮全参数扫描。
- 观测预训练采用相同的 update 级 loss 判断；保持 AFRO teacher 更新相对 optimizer 的原时序，检查失败时两者都不前进。不为预训练额外增加梯度扫描。
- 发生异常不抢救当前半步状态，不保存已污染或部分更新的 checkpoint；最后完整保存点仍可用。
- 不增加 `strict_checks`、`check_every`、诊断等级、异步 assert 或自动修复/跳过坏 batch。

明确放宽：检查可以发生在该组 backward 之后；没有裁剪的配方不保证在参数更新前检测所有“loss 有限、梯度非有限”的情形。保持日志可观察、遇到已检测错误直接失败即可。此策略不降低部署输出边界的检查要求。

**不要动 DQ 的量化前 finite 检查。** 连续码索引的 Inf 可能经 clamp 变为合法索引并查出有限手姿态，最终动作检查不能补救这一信息丢失。也不删除 Runtime 的真实外部 shape/dtype/finite 与执行范围检查。

完成判据：正常有限值输入下的梯度与更新语义不变；早期坏 micro-batch 不进入 optimizer/EMA/scheduler；代码中不再有每 micro-batch 的 CPU 有限性判定及无裁剪逐参数扫描。GPU 同步与日志/裁剪仍可能存在，不宣称整步零同步。

### T08 — 删除 selector 与发布工作流的附属复杂度

主要位置：`deployment/metadata.py`、`evaluation/selection.py`、`eval_pipeline.py`、`scripts/remote/sync_down.sh`、相关测试与文档。

采用本次明确的简化选择，不另留开关维持旧路径：

- `_resolve_checkpoint` 只保留 `latest` 与明确文件路径；删除 `20pct` 等百分比解析。训练仍可在 20/40/60/80/100% 保存五个研究检查点，文件名可以保留说明性百分比，但读端不再解析其 grammar。
- 删除 `best_ckpt.json` 写入、读取、推理默认值分流及其发布顺序承诺。`inspect_policy` 默认改为 `latest`；旧 `best` / `20pct` 调用不静默回退到其他权重。
- `eval_pipeline` 保留显式候选、统一 selection 设置、选出胜者、独立 test 与 `selection_result.json`。记录继续包含获胜 checkpoint 路径及 weights/use_ema、inference_steps、action_steps；不增设新 manifest。
- 删除只服务发布的 `--run-dir` 及相应参数；评测根目录沿用第一候选所在 run，任意结果位置用现有 `--output-dir`。更新调用方和 CLI 文档。
- `sync_down.sh` 改成一次普通 rsync；删除 best-last 第二阶段、`--ignore-times` 指针协议与专用失败注入测试。保留实际需要的路径参数、dry-run、W&B 排除、符号链接和临时文件排除，不增加镜像数据库。
- 同步针对已停止/完成的 run，传输成功后再使用；不支持同步过程中读取完整快照。不要用 `--inplace` 覆写大 checkpoint，也不新增默认删除目标文件的选项。
- 文档给出明确文件的评测/部署用法。复现胜者成绩必须同时使用结果记录的 weights、NFE 和执行长度；只换 checkpoint 文件而回到默认推理设置不算相同实验。Real 的执行长度通过其运行配置设置，不为此新增 Policy 参数恢复协议。

兼容边界：不删除用户已有的 `best_ckpt.json` 和历史结果；新代码不再消费它们。当前已核对的 `dexmani_real.run_policy` 默认传 `latest`，其正常接口不需改动；手动使用 `best` 的旧脚本须改为明确路径。此项属于有意的使用方式变更，需在交付说明中列明。

完成判据：同一 checkpoint 经 latest/明确路径、配同一显式设置得到一致的加载选择；选点与 test seeds 不相交；结果能指出胜者和实际推理配方；没有残留 best 发布/双阶段同步实现。不要删除选点统计来顺带简化论文实验。

### T09 — 简化 IO 与无默认消费者的时间采样选项

主要位置：`common/io_util.py`、其保存调用方、`agents/action_decoders/sampling/time_sampling.py`。

保存工具：

- `atomic_path` 保留同目录临时文件、写入成功后 `os.replace`、失败清理。删除强制 `fsync`、hardlink no-clobber 分支和工具层 `overwrite` 协议。
- fresh 目录、评测空目录和码表导出的“默认不覆盖”由各自用户入口已有的单次检查负责；尤其保留 codebook 导出的显式 `--overwrite` 使用语义，但不承诺并发竞态排他。
- 同步调整 `write_json`、logger、evaluation、codebook 等调用，别把旧的通用 overwrite 分支复制到每个底层 writer。
- 保留 latest 的小型相对 symlink、明确 checkpoint 字段、`close_resources`、persistent worker 的单点关闭与 handler 恢复。它们有真实消费者，不为几行代码退回裸写大文件或依赖析构。

时间采样：

- 正式 YAML 当前使用 uniform、beta、discrete；保留这三种及真正使用的 Beta 参数、time shift、Euler grid。
- 删除没有当前默认配方使用的 lognorm/mode/cosmap/discrete_pow 分支、专属参数、`sample_dyadic` 及对应无用测试/文档。
- 这是收窄研究接口，不是默认算法修复；依赖被删分布的旧实验使用其原源码版本，本次不建别名或迁移层。

完成判据：失败的写入不替换上次完整文件，正常 checkpoint/JSON/NPZ/symlink 均可写；单写者防误覆盖仍成立；保留分布的采样定义不变。不要为以上工具新建事务/缓存/发布框架。

### T10 — 测试与文档一起瘦身，保证方案真正收敛

主要位置：`test_tools/agent_checks.py`、`test_tools/pretrain_checks.py`、`test_tools/remote_checks.py`、`test_tools/smoke_test.py`、README 与现有 `docs/`。

删除：

- checkpoint 在某个函数调用之前必须被回收的 weakref/patch/call_count 协议。
- 仅确认 PyTorch strict load 能拒绝缺 scale/offset、错误 shape 的重复矩阵；保留本项目真实恢复结果的检查。
- 人为给 DP3 塞入无消费者 `hand_latent`、并要求 raw/EMA 该字段形状不同的 fixture。DexLatent 的真实 latent 统计仍用它自己的 roundtrip 检查。
- 未消费输入能力、百分比 selector、best 发布事件序列、双阶段 rsync 与已删除选项的测试。

保留并整理：

- 数据窗口、train-only 统计、affine/identity 继承、future 标签与有效 mask、动作表示、raw/EMA 选择、正常训练保存恢复、真实部署边界。
- 把巨大 `agent_checks.py` 按 decoder、观测/几何、手先验、训练恢复/部署等实际职责拆成少量可独立运行的检查模块，避免每次纯训练检查顶层导入全部视觉/点云依赖。沿用普通函数和现有运行方式即可，不强制迁移 pytest，不创建测试插件或动态注册表。
- `smoke_test` 继续走正式训练/恢复路径，作为实际数据短训入口；不另写一套配置加载或恢复算法供测试使用。
- 本次暂留模拟器部分构造清理及其最小回归，原因见第 7 节；不能只因 fake fixture 难看就撤掉当前真实保护。
- 同步更新 `README.md`、`docs/项目架构.md`、`docs/仿真评测机制.md`、`docs/观测预训练.md`、`docs/手部动作表示.md`、`docs/SSH服务器训练部署.md` 中受影响的段落。删掉失效承诺与重复说明，不新增一套“审计证据”文档体系。

文档还应准确说明：AdapTac 零 force loss weight 只是“无显式 force MSE”；AFRO/CordViP 的来源假设与本地适配；未实测的优化不写提升百分比。检查入口、README 命令与实际文件名一致。

## 5. 必须保留的正确性骨架

这些条目是防止简化破坏既有研究结果的边界，不是要求再添加一套运行时验证。

| 保留项 | 具体理由 |
| --- | --- |
| `global_step` 表示已完成 optimizer updates | accumulation、scheduler、EMA、保存点必须用同一种进度；不另存 epoch/batch 作为第二真值 |
| 尾组按实际 group_size 缩放；不等 micro-batch 的现有限制 | 多个均值直接平均在样本数不同时会改变损失权重；本次不重写为新的样本加权机制 |
| 当前轻量 epoch sampler 与独立 worker generator | 从保存 update 重建 epoch/前缀，跳过索引而不重新加载旧 batch；避免 loader 创建扰动模型 RNG |
| raw/EMA、optimizer、scheduler、RNG、有效配置、实际统计 | 完整 resume 的必要状态；不能降级成只加载权重和手填 epoch |
| `strict=True` 与 optimizer 参数名/顺序核对 | 权重名称匹配不等于 optimizer 位置匹配；小检查能防止静默错误 |
| normalizer 实际 affine 与输入维度表 | identity 字段可能没有 buffer；外部输入与内部 latent 字段也不同 |
| train-only 原始源帧统计 | 不把验证、unused、padding、重叠窗口或增强引入统计总体 |
| 预训练 encoder 的 affine 继承 | 迁移的是 `E(scale*x + offset)`；只载 encoder 后重新 fit 会改变其输入函数 |
| action/几何/触觉的真实数学约束 | EEF rot6d identity、触觉 RMS/MaxAbs 保零与共享尺度、sim/real 坐标语义、future 有效 mask 均有数值作用 |
| DQ 码表与量化前检查；DexLatent 冻结 FP32 先验 | 不能把无效输出量化成合法姿态；也不改 learned RVQ weights、手部 affine 或确定性 mean |
| Flow/ConsistencyFlow 分离及现有数学细节 | 不随手 clamp teacher target time、不按采样比例重复加权两个 loss，不强行合并不同目标 |
| 请求内 K/V 复用、SDPA、FP32 关键计算 | 已有明确计算意义；不回退成参考的多余 attention weights 或更低精度噪声计算 |
| Runtime/PolicyInfo、RTC 与外部输入边界 | 已有 Real 消费者；控制窗口、输入配方、来源与非有限输出关系到实际执行 |
| 原子替换及资源清理 | 保住最后完整 checkpoint、关闭实际 worker/环境/日志、避免次要清理错误覆盖主异常 |

补充：`encoder.pt` 中 normalization 模式字符串不是前向数学必要信息；已知 inherited fields 下缺 affine 可表示 identity。但现有几个字符串能准确解释继承模式，保留比改出新 schema 更简单。

## 6. 实施顺序与适量验证

### 6.1 分三批完成，不重新开启无限审查

1. **恢复与常用路径：T01—T05。** 先修保存配方所有权、生命周期、EMA、预训练恢复和启动/日志体验，形成可验证的训练闭环。
2. **删除整条机制：T06—T09。** 移除输入协议、热路径扫描、selector/best 工作流与多余 IO/采样支持；每删一项同步清理其调用、配置、测试和文档。
3. **收敛：T10。** 整理检查入口，核对实际消费者和最终 diff，完成受影响路径验证后停止。

可以拆成少量逻辑清楚的提交；不要为“分阶段”保留新旧机制并行、临时兼容层或功能旗标。

### 6.2 验证以实际结果为目标

| 范围 | 足够的验证 |
| --- | --- |
| 配置与输入 | 实际 compose 当前 13 种默认组合；检查保留字段/模式及 EMA 默认联动。重点覆盖 AdapTac force/taxel/patch、R3D 辅助目标、DexLatent 内部 latent，不要求把全部模型正式训练一遍 |
| 主训练恢复 | 在一个确定性小配置中检查连续执行与中断恢复的下一次 update、raw/EMA、optimizer moments、LR、更新计数和样本位置；关闭编译/随机增强并设 `num_workers=0`。比较合理数值容差，不把这个局部检查扩展成所有设备逐位重放承诺 |
| 真实使用 | 有实际数据时用正式 smoke 做 fresh 4 updates + resume 2 updates；确认两段日志、checkpoint 和统计，覆盖一个普通策略及受改动影响较大的方法。无法取得数据就明确未运行，不伪造通过 |
| 初始化自包含与 metadata | 恢复时不访问外部初始化路径、不重新统计；保存的部署 facts 不被当前 attrs 改写。用正常项目 checkpoint/模型，避免创造任意未知字段兼容需求 |
| 观测预训练 | AFRO/CordViP 各一次小规模 save/resume/export；检查 fresh 才推导 AFRO 预算、完成 run 不写产物、实际 affine 继承、可选 validation 的 RNG/加权语义 |
| 有限性与累积 | 一组正常输入保持更新语义；一组较早 micro-batch 的非有限 loss 在 optimizer/EMA/scheduler 前失败。无需穷举所有 NaN、Inf、dtype、异常组合 |
| 文件与评测 | 一次写入中途失败保住旧文件；正常 NPZ/checkpoint/latest 可用；显式候选能选点、独立 test 并记录设置；一次普通 rsync 的基本传输/symlink 行为。不用恢复发布故障矩阵 |
| 部署 | 保留 raw/EMA 与 normalizer/output roundtrip、缺失 EMA 明确失败、外部输入边界及现有 RTC 核心数值检查；只读核对 Real 调用接口，不把静态核对称作真机验证 |

GPU 性能只在有合适环境时测：固定模型、batch、数据、warmup 与测量区间，比较完整 update / `predict` 时间，报告实际配置。AdapTac 必须计入 force 与 action 两阶段。没有硬件时可以确认移除了哪处同步/扫描，不能声称吞吐、显存或成功率提升。

验证依赖不可用时，完成可执行的静态/轻量检查并列明限制。不要为了本次清理新建 benchmark 平台、安装一大套无关环境或编写只模仿实现的测试。

## 7. 已审查但本次不实施的项目

以下均有明确结论，不是留给实施者无限扩张的候选池。

| 项目 | 审查发现与默认决策 | 后续真正需要时的最小方向 |
| --- | --- | --- |
| 改 split 以固定不同预算的验证集 | 当前 train 前缀后取 val，改变训练预算会改变 held-out 成员；符合当前文档，不是单次 split 泄漏。本次保留，避免旧 checkpoint 按保存 seed/预算恢复到另一划分 | 学习曲线另开明确实验，先固定 held-out 再取训练前缀，或使用独立固定验证数据；不建身份 manifest |
| VQ-VAE 完整 resume | 默认长达 1500 epochs，当前没有 optimizer/scheduler/RNG 完整恢复，是功能缺口；不为统一而在精简任务里新增 | 确有长跑中断成本时，在现有单文件循环补完整状态与简单恢复；不新建 Trainer 框架 |
| AFRO 单 token attention | Q/K 不影响单 token attention 输出，四层共 526,336 个相关参数；来自来源设计，不是本地默认 bug | 单独做轻量模型变体/消融，再比较实际速度与任务结果；不伪装成整次训练逐位等价重构 |
| AFRO 随机 teacher、更新时序、epoch 300 后动量 | 原版行为，本次保留 | 复制初始化或改 EMA 时序属于实验配方变化 |
| CordViP 固定 1024 接触槽 | 全局 PointNet 对点排列不变，固定槽监督依赖稳定点序；来源也有该假设 | 核实实际 producer 的索引语义；需要任意点序预测时让 head 读取对应点特征，不能用 metadata contract 代替算法处理 |
| CordViP 重建捷径 | cross-attention 可读目标模态，不能解释成严格遮挡预测；本次仅准确说明 | 遮挡目标作为独立消融 |
| AdapTac `force_loss_weight=0` | 仍通过未来真值 force 的加噪输入和 action loss 训练 force predictor；只是关闭显式 MSE | 真正无未来标签基线单独设计，不随手 detach/删分支冒充等价清理 |
| InterActDex 辅助权重为零 | 会去辅助 head/未来标签，但交互 token 仍存在；不是移除整个交互机制 | 保持现有准确语义 |
| 共享冻结 backbone 以减少 EMA 副本 | 可能省显存，但增加可变 buffer/冻结状态/保存恢复复杂度；当前 deepcopy 易懂 | 先测真实峰值，只有瓶颈明确再做针对性共享 |
| 数据缓存、Beta 缓存、更大 compile 范围 | 没有实测瓶颈，本次不做 | 真实 profile 后优先调整简单数据布局或局部实现，不建通用缓存 |
| SAT/DQ/DexLatent 构造器合并 | 有真实重复，但参数和动作表示语义不同；非本次主要负担 | 只有小函数能明显减少维护点时提取，不引入 ActionRepresentation/工厂协议 |
| 删除预训练离线 validation 或 RTC | 都是可选研究能力；validation 很小，RTC 有 Real 消费者。本次保留 | 研究范围明确放弃时才整组删，不能只删正确性保护而留下功能 |
| 任意 milestone resume、分支 run、延长 scheduler | 有研究用途，但属于新增操作能力，不是消除现有复杂度的前提 | 明确需要时只支持明确文件/输出路径，不建立 selector、分支归档或日志协调框架 |
| 删除 Policy 中的特殊模拟器构造 | 当前 `dexmani_sim` 创建 scene/viewer 后的构造链缺少自身失败清理，Policy 手动 `__new__/__init__` 因而有现实依据 | 单独在 sim 资源创建处实现失败清理，覆盖其内部部分创建，再把 Policy 改为普通构造并删相应故障 fixture。本次不跨仓库改动、不提前删除 |

初始化 NPZ 的必要 shape/finite/scale 检查、RGB crop/resize、真实窗口边界等短检查，本次也不按行数清除。R3D 个别重复校验可以在触及相关代码时留一个合适边界，但不启动全仓“删除所有 ValueError”的清扫。

## 8. 参考与核查范围

五个主参考使用固定源码版本；“未见等价机制”限定其实际第一方训练/恢复/评测调用链，不泛指所有 vendored 代码、作者私有代码或未来版本。

| 项目 | 固定版本 | 本次主要借鉴与警惕 |
| --- | --- | --- |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f) | `5ba07ac6661db573af695b419a7947ecb704690f` | 窗口、动作对齐、diffusion/FiLM；不照搬反射 workspace 和 batch/update 混用 |
| [ManiFlow](https://github.com/allenai/maniflow/tree/e4dc24d62a6c91825813308b9926e921b4bb9ef6) | `e4dc24d62a6c91825813308b9926e921b4bb9ef6` | flow/consistency 目标、时间语义与 DiTX；不把 target time 或两组 loss 按直觉改写 |
| [DQ-RISE](https://github.com/rise-policy/DQ-RISE/tree/2889d27fce823288e8dd12ec6e91506b52ed086a) | `2889d27fce823288e8dd12ec6e91506b52ed086a` | RVQ/手部码表；保留本地使用实际 groups 与 learned weights 的导出改进 |
| [AFRO](https://github.com/KolaKivy/AFRO/tree/283024a49e2c14bc030debf15383a4e7ee73836d) | `283024a49e2c14bc030debf15383a4e7ee73836d` | 独立预训练、随机 teacher 与更新时序；单 token 问题独立做研究变体 |
| [CordViP](https://github.com/xuanxuanzzzii/cordvip/tree/ad6d441abe7b0480cf79b0311db266965c1313b6) | `ad6d441abe7b0480cf79b0311db266965c1313b6` | 预训练迁移与交互重建；固定点槽假设需与真实 producer 对照 |

关键本地定位（均指审查基线，可按符号在新版本查找）：

- [有效配置装配](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/training/common/config.py#L138)；[训练恢复与循环](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/training/trainer.py)；[显式 resume 入口](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/training/common/resume.py)。
- [优化器与 EMA](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/training/common/optimization.py)；[数据、统计与更新位置](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/training/common/data.py)；[观测预训练](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/training/pretrain/obs/common.py)。
- [输入预处理](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/agents/policies/base.py#L104)；[checkpoint 信息与 selector](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/deployment/metadata.py)；[选点流程](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/evaluation/selection.py)。
- [原子 IO 与清理](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/common/io_util.py)；[检查工具](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/dexmani_policy/test_tools/agent_checks.py)；[远程同步](https://github.com/haoyangzhanglab/dexmani_policy/blob/156b6bfa17a4d0edca31cea913ba8ec8975d1b27/scripts/remote/sync_down.sh)。
- 外部实际消费者：[Real 策略入口](https://github.com/haoyangzhanglab/dexmani_real/blob/85b6174920025833d6e300548789a246a1ad6076/dexmani_real/deployment/run.py#L53)、[Real 部署检查](https://github.com/haoyangzhanglab/dexmani_real/blob/85b6174920025833d6e300548789a246a1ad6076/dexmani_real/deployment/preflight.py)。
- 跨仓库清理前提：[Sim 构造链](https://github.com/haoyangzhanglab/dexmani_sim/blob/13582770fa354cbfc65380a7b17a946d31076a2e/dexmani_sim/envs/base_env.py#L127)、[部分构造 close](https://github.com/haoyangzhanglab/dexmani_sim/blob/13582770fa354cbfc65380a7b17a946d31076a2e/dexmani_sim/envs/base_env.py#L1353)。

三轮实际完成的是源码/调用链对照、13 组默认字段集合静态核对、37,343 个 NumPy 采样窗口/源行比较、3,610 个恢复位置边界检查，以及抽取真实配置装配函数的最小 metadata 覆盖复现。该复现使用最小配置替身，未跑真实 OmegaConf 端到端。

审查环境没有可用的 PyTorch/Hydra/Zarr 训练依赖，未完成真实训练、GPU profile、仿真闭环或真机测试。上述检查不能替代实施后的针对性验证，也不能据此保证任务成功率或量化加速。

## 9. 交付与停止条件

实施完成后提交代码、必要的现有文档更新及简洁说明：

1. T01—T10 各项已完成内容；如有实际阻碍，只报告具体代码事实和最小未完成范围。
2. 删除了哪些整条机制及对应调用/配置/测试；确认没有用新抽象或兼容层重新实现它们。
3. 使用方式变化：默认 latest、删除 best/百分比 selector、显式使用选模参数、单写者保存、FP32 master、输入配方范围和诊断放宽。
4. 实际运行的验证、结果与未验证项；性能结论只给实测结果。
5. 确认研究算法、数据划分及第 7 节暂缓项没有被顺手修改，现有用户改动未被覆盖。

完成上述范围和适量验证后停止。不要继续寻找能新增的 contract、evidence、异常处理、配置参数或抽象层，也不要为了实现本任务再建立任务状态管理系统。
