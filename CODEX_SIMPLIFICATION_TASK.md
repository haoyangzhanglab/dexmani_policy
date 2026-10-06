# Codex 任务书：DexMani Policy 最终简化整改

日期：2026-10-07（Asia/Tokyo）  
仓库：`haoyangzhanglab/dexmani_policy`  
设计审查基线：`7a27b35e48123c04a5ad04a071b9283cf652e80d`  
状态：**设计终审通过，实施与运行验收待本地执行。**

## 0. 任务目标、权限与事实边界

把现有个人研究仓库做得简洁高效、正确好用。实施方法是删除重复状态、重复计算、隐藏依赖、伪兼容及危险自动化，保留真正保护学习、动作、数据和实验结论的机制。不是重建框架，不是逐项机械删除，也不是再做一轮只输出报告的 review。

本文件自包含，不需要此前对话、ChatGPT 下载目录或外部审查附件。既有审查去重为58项：30项确认整改、12项条件处理、16项保留/纠正；对应处置见末尾表格。数字是范围索引，不是缺陷数量或验收分数。以下API名字属于设计建议，执行前读取真实实现。

执行时先读根及受影响目录适用的 `AGENTS.md`，检查 `git status`、HEAD、已有diff和实际调用者。基线SHA用于比较，不要求把本地checkout退回此提交；本地有后续实现时，按当前事实适配，已解决项不要重复改。旧 `CODEX_*TASK*.md` 是历史任务来源，不自动叠加为本轮义务；本轮具体取舍以本任务书和用户后续指令为准，不覆盖协作与安全规则。

授权范围是修改相关源码、配置、测试及必要使用文档，执行现有环境下的定向验证。默认不自行commit/push，不拉取/合并/重置用户分支，不stash或覆盖用户改动，不修改Codex权限配置，不绕过沙盒。修改同一处有实质冲突时保留现场、列明冲突，继续独立工作。

不得删除或改写既有数据、实验、权重、normalizer、视频、W&B记录或预训练缓存；不得启动真机、远程训练/停止/同步、正式长训练、批量仿真、闭环消融、视频流水线、自动资产迁移或依赖大升级。可以改这些工具并用隔离fixture验证，但不能在真实资产/远程会话上试运行破坏性操作。

历史CPU小实验仅支持warmup、点集排列不变性、split规则、分组loss等有限结论，不是本地验收。不要沿用历史PASS数量、性能百分比或“全部通过”。

## 1. 终审修订：避免简化本身引入错误

1. **把调度事实从resume合同中移出，再收缩合同。** 当前Trainer用 `resume_contract['batches_per_epoch']` 控制尾部accumulation和epoch归一化。改为构建时确定的完整epoch batch数；不能使用应用cursor后变短的 `len(loader)` 代替，更不能直接删除字段后改变组边界。
2. **不改变global_step含义。** warmup首个lr=0的optimizer调用仍按现有定义计步。短运行截止只限制本次执行，不重写总计划、warmup或milestone，不伪造100%训练完成。
3. **正视state_dict和optimizer布局变化。** 删除注册text_encoder不是天然兼容。参数对象、分组顺序、dtype和优化状态映射必须正确；不能靠shape相同或 `strict=False` 蒙混恢复。
4. **fresh EMA与resume EMA分开。** fresh副本必须对应同步后的raw初始状态；DDP不会同步独立EMA对象。resume必须读取历史EMA及updater，不用raw覆盖。训练teacher仍需要各rank本地副本。
5. **显式split的新语义不是等价重构。** 新训清单决定最终episode集合；旧manifest+cap实验必须恢复保存的actual IDs，不扩充、不重抽。保留窗口资格过滤和trial隔离。
6. **数据选择与训练seed可分开，但不新增别名。** 使用现有 `dataset.seed`；不另加重复 `data_seed`，也不批量改历史配方或所有默认seed。
7. **工程与研究分开。** 本轮不删除SAT action shuffle或EJC，不自动跑其消融；不全局关闭compile/static_graph。PointNet特定配方的shuffle收缩单独交付，并明确RNG消耗变化。
8. **只有承诺原配置的smoke才使用原资产和计算配置。** CPU/tiny fixture可独立证明机制，但不能通过换资产、降精度或关闭compile来宣称生产配方通过。没有有效参数更新不能打印训练smoke PASS。

## 2. 目标结构：沿用目录，仅重分职责

```text
首次训练配置 / 来源实验的保存配置
    -> Dataset：固定真实索引的读取、现有预处理
    -> Sampler：task配比、epoch顺序、rank分片、cursor
    -> Agent：实际共享流程 + 方法特有loss/预测
    -> Trainer：唯一micro-batch与optimizer更新路径
    -> 现有原子checkpoint、resolved config、轻量日志

完整checkpoint -> 保存架构构造 -> 严格加载指定raw/EMA
                   推理检查必要权重与语义 / resume检查训练状态

实际task->seeds计划 -> 一个不可变selection result
                       best只引用 -> 数值eval
                       demo独立显式执行
```

“单一来源”指单一控制权，允许小型不可变证据快照。不要为了消除几项重复数字而迁移全部历史产物。保持主要目录、模型名和用户入口；已有小registry、基类、optimizer分组工具有真实复用则保留。

禁止新增RuntimeManager、ArtifactRegistry、ValidationPipeline、万能launcher、迁移引擎、共享缓存服务、测试等级系统、自动配置依赖分析器或“公平性认证器”。不整体迁移到LeRobot/Accelerate/FSDP，不新增框架依赖。不为追求代码行数百分比删除行为保护。

## 3. 执行顺序与阶段退出规则

依次推进P1-P6；同阶段中可拆成更小的自洽修改。每个修改包括实现、真实调用者更新、旧路径删除及对应定向测试，不留下默认双实现等待以后清理。某一受限项目不得阻断所有独立整改，也不得被偷偷移出完成清单。

构造/恢复顺序调整应先于文本子模块删除；数据选择来源明确后才删除全配置resume比较；先直接执行task->seeds，再删除旧位置映射校验。底层变更影响前阶段时必须补跑相关测试，不能把阶段次序当作不再回看的理由。

如果会话/资源限制导致无法全部完成，保留可运行的自洽子集；不要留下半套入口切换或宽松fallback。结束时明确已完成、未完成、阻塞、下一入口。不因任务大而只输出计划，也不为凑齐完成率扩大权限或研究范围。

## P1. 真实训练smoke与明确的安全边界

主要文件：`dexmani_policy/train.py`、`train_ddp.py`、`smoke_test.py`、`training/trainer.py`、`training/build_utils.py`、`training/workspace.py`、`agents/core/base.py`；`scripts/utils/clean_experiments.sh`、`scripts/remote/stop_remote.sh`、`train_remote.sh`。

### P1.1 一个训练计算实现

- 单卡、DDP、smoke复用现有构造函数和Trainer。必要共享装配放既有training模块，不造第二个Trainer或伪实现workspace。单卡/DDP只保留真正不同的进程/设备/包装职责。
- 给现有训练循环增加可选、正整数的本次更新上限，建议 `max_updates`。语义为 `stop_step=min(total_train_steps,start_step+max_updates)`；未设置时保持正式训练行为。
- 截止发生在完整accumulation边界，所有rank使用相同截止。保存真实step/epoch/next_micro_step，使用既有checkpoint写法及latest更新；同step已有合法保存则复用，不重复写大状态。中途截止不伪造milestone、不中途丢半个梯度组，不打印全计划完成。已经到总计划末端时不再训练或覆盖历史产物。
- 完整epoch batch数在cursor应用前确定，Trainer直接拥有这个事实；必要时沿用现有小字段，不从旧合同反向驱动实时循环。保持原尾组归一化、mixed micro-batch尺寸限制、scheduler/EMA推进顺序和global_step计数。
- smoke使用临时输出、真实配置资产及对应AMP/compile/accumulation路径，少量更新（例如4次）后预测、保存和raw/EMA恢复。W&B允许不创建；生产数值路径不因此换实现。读取缺失资产时报准确原因，合成DQ码本只留定向测试，不覆盖实际codebook_path。
- 检查至少一个预期可训练主体的浮点参数发生有限变化。step counter、动量、BN buffer变化不是替代证据；基础smoke不要求所有参数第一批非零。没有观察到有效更新则失败或明确未验证，不自动提高LR、重拟合统计、无限延长预算。
- 不需要持有多份全模型GPU状态来比较；可逐个恢复、在CPU比较并释放临时对象。不得通过缩水真实batch后声称原配置通过。

### P1.2 参数覆盖与运维

- 在optimizer创建边界要求应训参数恰好覆盖一次；遗漏/重复报错。有意冻结用 `requires_grad=False`，保留LR/decay分组、参数排序和dtype。不要用“每批必须有梯度”取代optimizer覆盖。
- 清理脚本缩为只读事实报告；删除基于未完成、toy阈值、mtime自动删除实验的主流程。旧破坏性选项明确拒绝并给出新用法；不新增智能回收器或自动隔离/移动历史实验。
- 普通stop只发优雅退出请求。等待上限到达后仍运行，返回明确未完成状态/非零结果，不自动kill；只有显式force才允许强停。保留具体session约束、正确shell quoting、SSH错误传播。session消失不宣称checkpoint完整，其他GPU进程不推断为本任务残留。
- 预检分别报告连接/数据检查、信息展示、失败或未知；命令提交不等于训练成功。不要为了总括性PASS补建更多探测服务。

**验收：** 原warmup下权重实际变化；短运行中途/epoch末端/尾accumulation组的cursor正确；保存恢复后的学习状态正确；optimizer漏参/重复明确拒绝；隔离shell fixture证明普通扫描不删除、慢停止不升级强杀、SSH未知状态不假成功。不得调用真实远程脚本验证这些行为。

## P2. 热路径与诊断去重

主要文件：`training/trainer.py`、`training/logging.py`、`training/checkpoint.py`、`train_ddp.py`。

- 根据即将完成的global update确定该group是否输出日志。非记录group不进行日志 `.item()` / Python float转换；记录group只聚合detached tensor，再在输出边界转CPU。保持当前单group口径和字段，不悄悄改成多步平均，不持有跨step计算图。
- grad_norm供日志的转换同样延后；loss/gradient有限值检查和必要rank协调不删除。不要以消除一切同步为目标，不新建异步logger线程/队列，不在训练热路径加入测量用CUDA synchronize。
- 删除默认全状态NaN checkpoint及其轮换代码，保留正常checkpoint、异常rank/step和已有必要摘要。没有现成样本ID时不要新增血缘系统；特定故障的重放需求另用定向工具。
- 将打印的 `Actual observation fields` 改为声明含义，不建设自动图追踪来认证字段确实影响输出。
- 删除DDP构造后的normalizer专用二次broadcast及重载；保留DDP正常初始同步和梯度通信。依据是基线Torch 2.4.1正常DDP同步已覆盖当前注册normalizer参数；执行前确认没有新的ignore配置。
- 删除checkpoint保存中的无条件 `torch.cuda.empty_cache()`；不全仓库机械删除生命周期清理。
- 修正“默认NCCL无限等待”的错误说明。固定版本官方默认NCCL超时为10分钟，30分钟是延长；无明确运行证据需要延长时采用上游默认，不加watchdog。timeout配置改变单独记录，不和纯文案改动混淆。
- 本阶段不改compile/static_graph默认及其首backward/no_sync特殊处理，不把二次normalizer广播删除推广成删除其他collective。

**验收：** 受控输入、状态和噪声下更新一致；非日志步无日志标量化，日志步口径一致；保留数值异常退出；在可用的短DDP验证中检查初始化normalizer一致。未有目标GPU测量，不宣称吞吐提升百分比。

## P3. 固定数据集合与采样顺序各归其位

主要文件：`datasets/multi_task_dataset.py`、`resumable_sampler.py`、`base_dataset.py`、`split.py`、`training/resume.py`中的loader构造、`scripts/training/train_vq_hand.py`及其实际调用入口。

### P3.1 移走MultiTaskDataset的epoch服务

- 把现有目标任务计数、任务内permutation、完整遍历加余数、最终shuffle移到采样侧纯函数。保留精确配比/舍入和RNG算法，使用固定全局样本索引。
- 等价迁移必须保留两层现有顺序的组合：多任务映射 `M_e` 与rank逻辑位置序列 `Q_e,r`。实际输出为 `child_offset[task(M_e[Q_e,r[j]])]+local_index(M_e[Q_e,r[j]])`。
- 保持原DistributedSampler补齐、rank slicing及DataLoader drop_last；按原完整序列应用cursor。不在恢复时生成缩短的新epoch，不直接换WeightedRandomSampler，也不顺手消除两次shuffle。
- Dataset仅由累计长度解析真实task/local index并读取。删除Manager、每样本epoch proxy读取、worker完整epoch表及其专属pickle/close/析构逻辑。Zarr自身进程本地句柄/缓存处理不动。
- deterministic/validation固定顺序也要有真实调用者覆盖；更新其loader/sampler调用，不能只迁移训练分支导致直接验证读取改变分布。不新增跨进程全局采样缓存。

### P3.2 清单、实际IDs与恢复

- 无清单继续保留现有dataset.seed、val_ratio、cap；需要固定数据时独立设置已有dataset.seed，不批量引入新seed字段或改现有实验配方。
- 新显式清单路径让清单的最终episode IDs决定训练/验证集合，不在Dataset中再次随机cap。预算在准备清单时落实；非空cap或冲突val_ratio明确拒绝，提示显式去除/生成所需子集清单，不静默忽略继承自YAML的设置。
- 新清单契约改变须在使用文档明确。旧manifest+cap实验恢复使用保存的actual_train_ids和val集合，不重新抽样、不扩大、不重写原清单。区分首次训练请求和保存配方恢复，而不是全局改变旧文件的解释。
- resume从保存的清单内容和实际划分恢复；原外部文件可不存在。检查revision、episode顺序、完整互斥分区、trial不跨集、actual IDs有效性。hash证明内容一致，不证明物理trial标注真实。
- 清单控制的是进入窗口构造的episode集合。horizon/padding、finite和真实dispatch资格过滤继续保留；选中episode数、实际有效窗口数分开记录。训练split与仿真eval manifest不合并成通用格式。
- 保留现有小型data_recipe数量/mask证据，必要事实只在一处决定；不要为了字段去重迁移所有旧数据。新训统计只用合格训练源行；恢复只加载保存normalizer。

### P3.3 DQ的唯一主数据配方

DQ主入口使用policy-driven数据准备及normalizer。移除其默认standalone全数据统计旁路，更新脚本帮助和测试。若当前确有独立VQ研究消费者，隔离保留该独立用途而不提供默认fallback，并明确列出消费者；不能凭猜测删除活跃工作流。不增加legacy_mode或通用码本迁移器。

**验收：** 对旧算法的独立期望序列逐项比较(task,local index)，覆盖balanced/weighted/proportional、validation、短任务重复、rank补齐、drop_last、cursor、不同worker数。清单跨trial/revision/顺序错误拒绝，旧actual集合恢复一致，删除原清单后保存配方可恢复，统计源行不变。worker增强随机流与索引序列保证分开，不伪称bit-exact。

## P4. 生命周期、checkpoint与resume的闭合整改

主要文件：`training/build_utils.py`、`resume.py`、`checkpoint.py`、`agents/loader.py`、`core/multi_task.py`、相关RGB/text构造、`deployment/runtime.py`及真实消费者。

### P4.1 构造、首次初始化、完整恢复

- 沿用已有 `initialize_training()` 的思想：fresh训练加载所需预训练初始化；完整恢复仅根据保存架构构造，再严格加载权重。不先加载一份马上被覆盖的预训练权重。
- 把必要HF结构信息放现有resolved config，保持LoRA设置/dtype、特殊token、projection及预处理语义。仅改AutoModel而仍远端读取AutoConfig不算完成；不把所有构造强制包装成新工厂。
- 在compile/DDP前优先deepcopy已有模型建立EMA。验证无可变参数别名，dtype、normalizer、codebook、hooks、cache及buffer正确。fresh EMA要对应同步后的raw；必要时在DDP初始同步后对本地EMA一次校准，不在每步广播。不得对DDP已包装raw重新注册参数。
- resume分别加载历史raw/EMA、EMA updater；teacher仍按实际算法需求每rank持有。评测EMA继续允许rank0-only。不能因为某个冻结参数没梯度，就把所有冻结状态复制删除。
- 先完成决定学习参数对象/布局的构造与load，再建立最终optimizer及恢复状态；核对分组名称/顺序/形状/dtype关系。避免新ParameterDict或被替换模块让optimizer指向旧对象。恢复optimizer、scheduler、EMA后，在随机构造结束且第一次训练取数前恢复本rank主RNG；编译warmup如被额外运行须控制其状态影响。

### P4.2 闭集文本真正去掉生产embedding的模型

- 固定已知文本只保留映射、固定embedding buffer、text_proj。准备阶段/首次初始化生成一次并保存，未知文本明确拒绝，不自动在线fallback。
- 真实开放文本用途存在才保留对应在线路径；固定任务不常驻完整CLIP/tokenizer。保留相同CLIP特征值，不替换成随机任务ID。
- 这是state_dict布局变化。转换旧产物只能针对已识别格式、已保存完整embedding表及已知删去的冻结子模块；保留raw/EMA学习参数、normalizer和dtype。不可凭前缀大范围丢弃未知key，不用strict=False掩盖。
- 优先只提供所需格式的窄读取边界/转换函数及fixture；仅为具体已确认资产需要时增加一个小转换命令，不批量执行迁移，不覆盖原件，不为所有历史版本建立框架。
- 推理转换通过不等于full resume兼容。旧optimizer参数顺序或布局无法证明正确映射时，保留旧代码恢复方式或明确拒绝该续训；不要自动丢弃优化状态。历史缺结构/embedding信息时不给出伪造的冷缓存保证。

### P4.3 一种容器，两种必需字段

- 保留现有.pt及原子写。公共低层读取payload，推理不必先构造包含全部训练状态的TrainCheckpoint；分别验证推理和resume的必需字段。
- 版本受支持、必需字段类型/语义正确即可，容忍无关额外metadata。learned state保持strict=True；指定EMA缺失就失败，不换raw/latest。
- cfg与checkpoint按所属实验成套解析，不根据同shape推断任意cfg可用。仅分离校验并不会让torch.load跳过同文件optimizer字节，不夸大IO/内存节省；本轮不新增必需的模型bundle或推理导出协议。

### P4.4 saved-config拥有恢复控制权

- 来源saved config必须在validate/build dataset/model/optimizer之前确定。不能先按当前默认YAML构建，再试图修补历史实验。
- 使用一个窄的操作性override白名单，例如输出位置、等价设备定位、日志；逐项说明worker/compile覆盖是否在既有恢复承诺内。world_size、batch/accumulation、dtype/AMP、总计划、模型/数据配方不是随意操作性覆盖。
- 检查用户实际显式override，不把当前默认值当用户要求，也不能静默忽略禁止的override。复用现有Hydra override信息，不自建CLI解析语言。数据路径迁移可显式指定，但revision相同只是声明，不是内容hash证明。
- runtime只保留必要事实检查：实际数据身份/集合/窗口资格、loader完整几何、cursor、训练步数和优化/EMA状态。完成P3及P1调度事实移交后，再删除重复保存完整agent/dataset/training配置的递归比较。
- 小的既有字段可以先不改名。真有不兼容语义变化时用现有格式版本机制区分；旧格式在一个入口解释，不让每个消费者维护双分支。保留有实际需求的窄reader，停止扩张没有用户的历史兼容。
- 完整resume失败不得自动退回weights-only。另有明确新实验初始化需求时沿用独立语义、重置训练状态，不能自动重拟合normalizer或把旧global_step带入新优化状态。

**验收：** 模型新格式禁网/无初始化缓存恢复，raw/EMA在固定输入和受控噪声下预测一致；关键tensor缺失拒绝、extra metadata允许、EMA缺失拒绝；EMA无别名；optimizer仍绑定正确参数，保存恢复后下一次受控更新一致；旧布局有明确映射证据或明确拒绝。窄tiny模型测试和真实资产恢复分别报告，不能互相冒充。

## P5. 直接评测计划与单一发布结果

主要文件：`evaluation/protocol.py`、`env_runner/multi_task_sim_runner.py`、`select_best_ckpt.py`、`eval_best_ckpt.py`、`record_demo.py`、`agents/loader.py`、`deployment/runtime.py`、`scripts/eval/`及真实消费者。

- 新runner直接消费实际 `{task: [seeds]}`；先展开旧reference/ordinal映射，在内存中得到同样的实际列表，保持任务执行顺序、预算和macro/micro指标含义，再删除旧映射。不能先删其顺序/hash防护再保留位置依赖。
- 检查所用task/seed合法、role互斥、请求与完成的(task,seed)相符。环境身份和必要来源继续记录；不因取消全池位置依赖就取消真实语义保护，不在本次开放不同任务预算或换统计主指标。
- 保留现有两阶段tie-break、success/avg_steps/global_step排序、全零选点行为。删除recorded-only initial_episodes和batch_size配置/参数及其专属测试；保留确有作用的预算上限，打印实际计数。旧被删除CLI参数明确报错并说明清单来源，不静默忽略。
- 一个不可变selection_result包含具体milestone、raw/EMA、NFE、实际计划和候选结果/证据。best_ckpt.json只存相对引用；handoff传同一结果路径，不复制整份内容。候选全部正常完成才发布，模型/环境异常不是普通0%结果。
- 写完必要结果/证据后原子发布selection_result，最后原子更新best。失败不改旧best；解析best一次后固定权重，后续不反复读取可变alias。已有轻量并发锁、路径边界和原子写保留，不新增事务服务。
- 每episode明细和每次eval实际设置快照可保留，但不是第二个可修改的控制源。普通显式checkpoint推理不读selection审计；正式held-out评测必须有足够选择证据。明确的NFE曲线报告不冒称未经选择的单点最佳结果。
- 旧inline best/selection仅在一个加载边界解释为当前内存表示；实际seed无法重建时不冒称held-out可用，但具体权重仍能独立推理。先迁移训练评测和deployment/demo消费者，再删除旧副本互认证。
- 默认pipeline只selection+数值eval，demo保留独立命令。关闭视频不关闭RGB Policy必要相机/离屏渲染。视频编码失败不抹掉数值结果；动作、模型和环境异常仍失败。

**验收：** 旧新实际task/seed序列、episode预算、原排名和指标一致；候选异常/缺权重/半写入不更新best；alias变化不改变已解析调用；选点/测试重叠拒绝；旧记录证据不足不会假held-out；demo失败与数值结果分开。使用fake env测试协议不是仿真rollout验证。

## P6. 小而准确的算法接口、定向配方清理与收尾

主要文件：`agents/action_decoders/`、相关Agent及 `configs/`，受影响测试与README/机制说明。

- 保留共有接口的None；不支持非空dim_groups就明确拒绝，删除无消费者且吞掉错误的宽泛kwargs。不能把Diffusion分组loss无声换成Flow整体平均，也不要构建能力注册系统。查完调用者后只保留真正需要的局部model_kwargs。
- consistency训练实际micro-batch不足以形成合法flow/consistency分组就拒绝；梯度累积后的总batch大不能补救。纯flow直接用现有RectifiedFlow；不通过改flow_batch_ratio或降目标来让smoke通过。保留真实teacher EMA、时间分布和训练网格/NFE独立性。
- 活动配方不暴露inactive参数：例如标准cosine不把lr_min_ratio当实际控制，连续beta采样不声称离散grid决定训练。共享构造只在相关分支传参数。清理当前配置和打印，不写自动参数依赖分析器；旧saved config由窄reader保留原含义。
- 单独收缩明确使用PointNet/MultiStagePointNet对称全局池化的配方：关闭“同一已选点集采样后shuffle”，不修改模型权重形状、不全局删共享FPS选项。保留随机起点与子集变化。旧保存配置不改写，新run RNG消耗变化明确记录，不能声称训练轨迹逐位相同。
- **本轮不实施SAT shuffle/EJC删除或新消融。** 非退化权重等变性是后续候选证据，不足以宣称dropout训练和优化轨迹不变；不以两个零初始化输出相等作为证明。不因为PointNet结论推广到SAT/PointNext/Uni3D。
- 不全局关闭compile/static_graph，不移除其首backward特殊处理，除非当前阶段的真实变更已取消该前提且相应测试覆盖。SAT compile覆盖不自动撤销。
- 方法名/路径保留可用，只纠正“本地适配=官方原样复现”“相同步数=公平预算”“Wilson=训练seed方差”等不准确说明。预算、数据集合和最终test的使用属于实验设计，不加认证器。
- 保留normalizer、动作窗口/单位/joint-EEF/辅助维度、有限值、metric点云、RGB预处理、VQ affine、RTC输入VJP/prefix/delay和teacher语义。保留Zarr有界缓存、RGB uint8已有配方，无profile不移增强到GPU或新加prefetch stream。
- 随删除路径清理专属旧测试，保留真实LoRA/EMA/注意力/RTC/窗口测试。只更改确实受影响的文档；未确认完成的旧任务文档、历史证据及重复skills不批量删除，不将本任务书变成长期AGENTS规则。不改W&B已有唯一ID或无证据地删依赖文件。

**验收：** 不支持的语义参数/不足micro-batch明确拒绝；分组loss有独立数学期望；PointNet在同点集非退化实例上的输出/梯度检查与原配方解读准确；受影响入口和保存配置可读；未改变SAT/EJC默认、控制与已有有效性能路径。

## 4. 验证策略与资源上限

先确定已有Python/依赖、数据/模型缓存和CUDA可用性，不假定本地环境名等于远程环境名。不自动安装、升级或下载大权重。允许现有环境中有界CPU测试；短GPU smoke或必要的双卡定向检查仅在确认可用资源、明确GPU/进程数、步数和timeout后运行，不抢占或停止其他任务。多卡/真实资产不足则对应项目NOT VERIFIED，不通过降配修改原验证含义。

每个测试命令有合理timeout；超时只清理本次测试进程/临时文件，不误杀用户会话。无需给每项改动跑全模型全精度组合，更不能运行无边界“直到收敛”的检查。

优先复用并按真实风险选择：

- `tests/test_infra_training.py`、`test_infra_resume.py`、`test_infra_cuda.py`：更新、cursor、状态与分布式；
- `tests/test_streaming_dataset.py`、`test_research_split.py`、`test_policy_windows.py`：数据、划分、时间窗；
- `tests/test_infra_codebook.py`、`test_policy_vq_alignment.py`：码本与统计对齐；
- `tests/test_infra_evaluation.py`、`test_paper_protocol.py`、`test_infra_launch.py`：评测和shell边界；
- `tests/test_paper_readiness.py`、`test_review_remediation.py`、`test_policy_rtc.py`：实际机制回归。

以上是导航，不是要求无条件运行整个清单。缺少的边界优先补现有测试文件，必要时新增一个聚焦文件，不新增覆盖率目标/验证框架。

关键区分：

| 改动 | 实际验收对象 | 禁止冒称 |
|---|---|---|
| 结构/热路径简化 | 受控输入、状态、噪声下的更新/预测与统计口径 | 单元通过=吞吐提升/成功率不变 |
| 采样/划分 | 完整实际索引、role/trial、cursor和统计源行 | 比例一样=序列/增强轨迹一样 |
| 权重/恢复 | strict learned tensors、normalizer、EMA、optimizer对应关系 | 推理转换通过=完整续训兼容 |
| 冷缓存 | 禁网、隔离初始化缓存、真实架构构造与完整状态恢复 | mock禁止某函数=所有生产资产均脱离依赖 |
| DDP | 初始化、accumulation、恢复及必要单rank失败路径 | config-only或单卡=两卡通过 |
| 评测 | 实际task/seed、原排名、故障不发布 | fake env=真实rollout/物理安全 |

优化器missing/duplicate、旧key转换、select发布、shell退出等故障fixture都应独立构造，不能全部调用生产helper互相证明。测试涉及零初始化模型时使用非退化权重或足够的有效更新，保持科研语义。

## 5. 交付与结束标准

结束时在终端给出：实际HEAD及用户已有改动保护情况；按P1-P6的完成/部分/未做与原因；关键删除对象及保留不变量；实际运行命令、解释器/设备、PASS/FAIL/NOT VERIFIED；旧配置/权重的兼容边界；剩余风险和下一入口。必要复现日志放临时目录/已有忽略目录，不创建另一批大型整改报告，不把历史通过数累加进本次。

不自动commit/push。用 `git diff --check` 及受影响测试收尾，重新搜索旧入口/旧字段真实消费者；不以删函数但旧CLI仍调用为完成。发现相关回归要修复，不能删除失败测试或降低strict校验获得全绿。

验收必须回答：

1. smoke是否执行真实更新并证明学习参数变化？
2. 新完整checkpoint是否不依赖重新获取初始化资产即可恢复，且旧产物边界清楚？
3. 多任务是否不再需要worker共享epoch服务且实际索引序列保持？
4. 数值评测是否只依赖一个固定计划与选择结果，失败不污染best？
5. 普通运维是否不再自动删除实验、升级强停或夸大成功？

达到目标后停止扩大工程范围。本任务书是一次性实施依据，不是每次研究改动必须更新的管理系统。

## 6. 58项审查结果的完整落点

“整改”进入本轮实现；“条件”按本文件边界保留/局部处理，不自动升级成必删；“保留”不重复修复或删除。此表只用于核对范围，不要求建立长期状态数据库。

| ID | 审查事项 | 最终处置 |
|---|---|---|
| 01 | smoke独立训练路径 | 整改：P1，同一个Trainer |
| 02 | warmup首step零学习率 | 整改：P1，验证有效权重变化，不改计步 |
| 03 | 合成DQ替代真实资产 | 整改：P1/P3，fixture与集成分开 |
| 04 | config-only证明范围 | 保留：只声明解析/import |
| 05 | optimizer漏参只warning | 整改：P1，构建时覆盖/去重失败 |
| 06 | 未输出日志仍标量化 | 整改：P2，按输出group处理 |
| 07 | 默认巨大NaN转储 | 整改：P2，删默认全状态副本 |
| 08 | 删除全部有限值检查 | 保留：真实边界防护不撤 |
| 09 | 正整数/shape检查是主要瓶颈 | 保留：不为小检查跨目录重构 |
| 10 | Actual fields被当作测量 | 整改：P2，准确称声明 |
| 11 | 推理依赖完整训练状态 | 整改：P4，读取目的分开 |
| 12 | metadata字段集合完全相等 | 整改：P4，required字段+版本，learned strict |
| 13 | 全配置递归resume合同 | 整改：P3/P4，saved-config与必要runtime事实 |
| 14 | exact resume被当逐位轨迹 | 保留并纠正：主RNG/cursor有界承诺 |
| 15 | 所有历史分支都删 | 条件：P4，有消费者的窄reader保留 |
| 16 | 恢复前重复预训练加载 | 整改：P4，本地保存架构构造 |
| 17 | EMA再次完整初始化 | 整改：P4，复制已建模型并保证初始同步 |
| 18 | 固定文本仍携带CLIP | 整改：P4，闭集表，正视旧key布局 |
| 19 | 一切冻结状态EMA复制无用 | 条件：P4，不全局假设不可变 |
| 20 | DDP后二次normalizer同步 | 整改：P2，保留正常DDP同步 |
| 21 | 全局static_graph前提 | 条件：P2/P6，现有有效配置先保留 |
| 22 | NCCL默认无限等待说明 | 整改：P2，按固定版本事实修正 |
| 23 | 保存时无条件empty_cache | 整改：P2，仅移除保存副作用 |
| 24 | 全部compile默认关闭 | 条件：P6，不一刀切 |
| 25 | Dataset Manager/worker表 | 整改：P3，顺序归Sampler |
| 26 | data/training seed绑定 | 条件：P3，使用已有dataset.seed，不加别名 |
| 27 | 相同步数等于公平预算 | 条件：P6，实验定义说明，不改超参 |
| 28 | 清单完全固定最终训练集 | 整改：P3，新语义明确、旧actual IDs保留 |
| 29 | 窗口/normalizer/源行约束 | 保留：P3/P6，学习与动作语义 |
| 30 | finite扫描/Zarr cache直接删 | 保留：无profile不推翻 |
| 31 | RGB uint8无收益 | 保留：已有配方/预处理及证据范围 |
| 32 | VQ两种主配方 | 整改：P3，DQ主流程单一来源 |
| 33 | PointNet采样后shuffle | 整改：P6，仅对应全局池化配方 |
| 34 | SAT shuffle必定无用 | 条件：本轮不实施/不自动消融 |
| 35 | EJC三字段都无用 | 条件：本轮不改参数化 |
| 36 | decoder吞掉语义参数 | 整改：P6，不支持就拒绝 |
| 37 | consistency小batch退化 | 整改：P6，实际micro-batch检查；原有warning不等于无风险 |
| 38 | inactive参数控制/比较 | 整改：P4/P6，显式相关分支 |
| 39 | dp名字等于官方复现 | 条件：P6，说明local adaptation，不重命名全库 |
| 40 | best/summary/handoff互认证 | 整改：P5，一个权威结果 |
| 41 | 历史预算仍控制episodes | 整改：P5，删除无效参数 |
| 42 | tie-break本身错误 | 保留：原排名/预算逻辑 |
| 43 | task->seed绕位置映射 | 整改：P5，先展开再删旧层 |
| 44 | 固定test保证研究无泄漏 | 条件：P5，预先固定选择和报告口径 |
| 45 | Wilson代表训练方差 | 保留函数、纠正含义 |
| 46 | 数值pipeline强制demo | 整改：P5，已有独立命令 |
| 47 | 多任务失败偷偷缩分母 | 保留现有最终失败保护，不重修不存在缺陷 |
| 48 | 未完成/toy自动删除 | 整改：P1，只读事实报告 |
| 49 | 30秒自动强停 | 整改：P1，force必须显式 |
| 50 | Pre-flight总括成功 | 整改：P1，逐项真实/未知 |
| 51 | source.zip保证运行隔离 | 条件：快照保留；本轮不改活动远程副本 |
| 52 | 原子写/run claim/快照多余 | 保留：低成本真实保护 |
| 53 | W&B静态ID必然冲突 | 保留现有实际唯一后缀，不新造日志平台 |
| 54 | requirements/pyproject删一个 | 保留职责，不批量升级依赖 |
| 55 | 旧任务/skills永久同步 | 条件：只改受影响文档，不大规模归档/删活跃任务 |
| 56 | 测试多/零输出相等即有效 | 保留行为测试，删除只服务旧机制的测试 |
| 57 | RTC/teacher/物理语义可通用删 | 保留：P4/P6，禁止自动真机验证 |
| 58 | 所有registry/基类都删 | 保留有复用的轻逻辑，禁止框架重建 |

## 7. 参考实现与证据使用方法

基线源码链接可由仓库URL与上方SHA直接定位。重点阅读真实调用者，不需重复全文阅读所有历史任务书；上游网络不可用不阻断本地可执行工作。

- Diffusion Policy：`diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`，已审读blob `9219427ca0c71d3d2bceca746446c83f744de003`。借鉴同路径debug和EMA deepcopy，不照搬所有计数/恢复细节。https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py
- ACT：`policy.py`，已审读blob `7b091e5e0c4c73b0cce35a714ba3ccc26a242962`。保留padding/L1/KL等方法责任；不照搬重写__call__，这里用标准forward。https://github.com/tonyzhaozh/act/blob/main/policy.py
- LeRobot：`src/lerobot/scripts/lerobot_train.py`，已审读blob `7d88d5fc7271cbfb9fbbfd38e34db0c9676076b8`。借鉴共享预处理和PEFT状态范围，不引入平台迁移。https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_train.py
- RDT-1B：`cd79363a1387e8f81c7724d070ef7e45fd23150f` 的 `train/train.py`。预计算语言时不创建text encoder；不照搬完整恢复失败后weights-only fallback。https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/cd79363a1387e8f81c7724d070ef7e45fd23150f/train/train.py
- PyTorch 2.4 DDP及状态加载：https://docs.pytorch.org/docs/2.4/generated/torch.nn.parallel.DistributedDataParallel.html 、https://docs.pytorch.org/docs/2.4/generated/torch.optim.Optimizer.load_state_dict.html 。按项目实际固定版本核对，不拿新版API替换旧栈。

上游main会变化，列出的blob仅用于说明审读版本，不把它当commit SHA。官方代码也是参考而非正确性豁免。此次“终审通过”仅表示本任务设计范围、依赖与验收边界已确定；所有实现、性能、恢复和闭环结论以本地实际证据为准。
