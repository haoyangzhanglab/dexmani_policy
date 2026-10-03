# DexMani Policy 基础设施修复任务书

状态：**方案已复查，等待实现**。本文不是“代码已修复”或“GPU 验收已通过”的声明。

审查基线：[`141b0337c148d4f2dd09ade11f4db660a8ab9c4d`](https://github.com/haoyangzhanglab/dexmani_policy/tree/141b0337c148d4f2dd09ade11f4db660a8ab9c4d)。发布任务书前已确认 GitHub `main` 的实现仍为此版本。执行时以本地实际 HEAD 和工作区为准，先检查差异，**不要 reset 到审查基线**。

目标：服务个人机器人学习论文研究，让训练、恢复、数据读取、选点和评测**简单、正确、高效、好用**。完成下列局部修复，保持已有正确的算法与研究变量，不建设通用框架。

本文是自包含的执行依据，不依赖聊天记录、审查者的临时测试文件或另一份未提交方案。F01–F19 保留原审查编号，便于逐项追踪。后文的源码路径均相对于仓库根目录；函数名与基线对应，若当前代码有变化，先重新确认调用链。

## 1. 执行约束与完成标准

先读 `AGENTS.md` 和实际改动目录适用的指令，再读本文。检查 `git status`、当前分支和 HEAD，保留用户未提交修改。不要修改 `AGENTS.md` 来固化本次实验配置，也不要覆盖无关文档或产物。

本轮必做范围是 **F01–F19 的下述处置，以及 G01/G02 两项小改进**。其中“拒绝不正确组合”“为未知历史产物明确标记验证边界”也是有意选择的完整处置，不要求实现所有历史或自定义场景。第 10 节列出的研究扩展暂缓。

执行方式：

1. 从当前 config、入口和真实调用者确认问题；如果已经修复或事实不成立，记录源码证据并跳过重复修改，不机械照搬本文。
2. 按批次实现最小但端到端完整的修改。配置、构造、保存、恢复、评测和相关脚本要一起闭合，避免只改一个构造函数或只加 YAML 字段。
3. 用小型合成数据、临时目录和定向回归验证真实失败条件。优先复用现有 smoke 和辅助函数；不建立新的测试框架或庞大配置矩阵。
4. 每批完成后继续下一批，不只停在分析或计划。局部环境受限时完成不受影响的代码、检查和文档，并列出明确的待验证项。
5. 最后更新必要的 README/已有使用文档，并创建 `docs/infra_fix_report.md`：逐项记录 F01–F19、G01/G02 的状态、修改位置、真实执行的命令和结果、兼容边界、剩余阻塞。不要把设计验收条件当成已通过测试。

不自动启动真机、远程训练、长训练或大批量仿真；不删除/覆盖已有数据、实验、checkpoint、录像、W&B 日志或预训练权重。允许为当前风险做必要的短 CPU 检查；有目标 GPU 时做最小 CUDA/DDP 检查，不擅自扩成完整实验。不要更换用户全局 Python/CUDA 环境。默认不提交或推送实现代码，除非执行者另获用户指令。

保留的核心行为：现有 Trainer；按 optimizer step 推进的 scheduler/EMA；resumable sampler 及其消费游标；原子 checkpoint；保存的 normalizer；从历史配置构建的共享推理 loader；现有模型、动作表示、loss、训练预算和 checkpoint 选点排名规则。

## 2. 优先级和实施批次

P1：先处理训练阻塞、实验产物混写和主要结果风险。P2：条件性缺陷或低成本边界修复；本任务仍要求完成其规定处置。优先级不表示默认配置一定触发缺陷。

| 编号 | 已确认事实／触发条件 | 最小处置 | 优先级 | 批次 |
|---|---|---|---|---|
| F01 | MultiTaskDataset 持有不可被 spawn 序列化的 Manager owner | owner 留在创建进程，worker 只接收 proxy | P1 | A |
| F02 | 分钟级训练目录名可碰撞，workspace 会混写产物 | 更细目录身份＋原子认领，恢复写新目录 | P1 | A |
| F03 | 远程启动无条件杀同名 tmux，日志可被截断 | 唯一会话／日志名，取消启动时自动 kill | P1 | A |
| F04 | 依赖声明不一致，默认 LoRA 路径缺 peft | 一条实际验证的安装路线 | P1 | A |
| F05 | 一个累积组内 batch 大小不等时，等权 batch 均值不等于样本均值 | 只拒绝实际出现混合大小的累积组 | P2 | B |
| F06 | episode_ends 仅校验末值，回退等坏边界可进入采样 | 读取前统一验证元数据 | P2 | B |
| F07 | 同路径、同形状数据内容替换不能被现有 resume 检出 | 不可变路径约定＋可选 data_revision 检查 | P2 | F |
| F08 | 结果未保存完整的实际评测环境及最终解析参数 | 保存有效评测快照和结果引用 | P1 | C |
| F09 | 多任务非 reference seed 池重排可破坏 held-out | 记录池身份和真实 task/seed 映射，再校验无交集 | P2 | C |
| F10 | sync_down 跳过已存在的 best 等可变小文件 | 可靠更新元数据，保留大文件增量策略 | P1 | C |
| F11 | 可选的选点录像互相覆盖，候选明细未持久化 | 候选／阶段隔离，保存已有 episode_details | P2 | C |
| F12 | 失败 selection 与旧 best 的归属不清楚，best 非原子写 | 明确最近成功语义，成功后原子发布 | P2 | C |
| F13 | 相对 resume_from 被当成当前 checkpoint tag | 在训练入口解析外部路径 | P2 | A |
| F14 | VQ 验证对 batch 均值等权，尾批可反转选点 | 按样本数统计 MSE 等样本均值指标 | P2 | D |
| F15 | 裁剪前总范数溢出，旧慢路径可误推进更新 | 统一检查范数，失败时不推进训练状态 | P1 | B |
| F16 | Gaussian 动作归一化与固定 DDIM x0 裁剪不匹配 | 显式 clip_sample＋兼容性检查 | P2 | E |
| F17 | 全可见 GPU RNG 列表按旧下标恢复，映射变化有风险 | 按 rank 保存／恢复实际训练设备的 RNG | P2 | E |
| F18 | NPZ codebook 导入校验弱于恢复路径，且可半更新对象 | 共享局部校验，验证后提交对象状态 | P2 | D |
| F19 | 部署发现与 selector 假定三层路径，漏掉 DDP | 按实验内容发现，支持可变路径深度 | P2 | A |

通常按 A → B → C → D → E → F 执行；DQRise/codebook 研究可将 D 提前，Gaussian 或换 GPU 续训前必须完成 E。G01 随 C，G02 随 B。批次用于控制改动和验证，不要求每批创建 PR。

## 3. 批次 A：环境、启动和路径

### F01：spawn worker 中只保留 Manager proxy

位置：`dexmani_policy/datasets/multi_task_dataset.py` 的构造、序列化和清理；调用方为共享 DataLoader 构造及 DDP 入口。

事实：`_epoch_val` 是可跨进程使用的 proxy；真正阻碍 spawn 序列化的是 `self._manager`。不应把“共享 epoch 不可用”或“当前采样器全部错误”作为诊断。

修改：

- 在 `__getstate__()` 中移除 Manager owner，保留 `_epoch_val` 等 worker 所需状态。父进程继续持有 owner，避免提前析构。
- 清理只由创建 owner 的进程执行；worker 副本不得 shutdown 父进程 Manager。覆盖构造异常、重复清理和 fork 下的 owner 归属；需要时记录 owner PID，使用局部防护即可。
- 保留 balanced/weighted/proportional 的任务目标计数、epoch 更新、顺序和断点游标，不用上游 IID weighted sampler 替换当前分层采样。

验收：用真实 `MultiTaskDataset`、小型可序列化子数据集和两个 `spawn` persistent workers 迭代 epoch 0 → 1 → 0；每次结果与父进程生成的对应索引一致，退出无异常。覆盖 `num_workers=0`。此前 owner/proxy 原型通过此检查，但不等于完整 NCCL 训练已通过。

### F02：防止两个训练 run 混写

位置：`dexmani_policy/configs/` 中实际训练入口的 Hydra 输出路径；`training/workspace.py`；单卡/DDP 启动调用链。

修改：

- 默认目录使用秒／微秒时间加本次启动的短随机后缀，避免依赖时钟精度保证唯一。目录身份在启动父进程生成并确定一次，DDP 各 rank 共用解析结果；W&B 也使用该 run 身份。检查基础配置、DDP override 和 Hydra sweep 的子目录语义，不只改 dp.yaml。
- 在任何训练 checkpoint/log/W&B/config 写入前，由唯一 writer 原子创建永久运行标记（独占创建，例如 `open(..., 'x')`）。已有标记或旧训练 `config.yaml`、metrics、checkpoint 时报清晰冲突，不接管、不删除旧目录。
- 继续采用“旧 checkpoint → 新输出目录”的恢复方式，记录来源 checkpoint；恢复模型/优化器状态，不续写旧输出文件。已有 W&B ID 与目录身份的绑定继续使用。
- 认领失败不应启动昂贵训练；若提前到入口认领，workspace 应消费已认领结果，避免同一启动重复认领。只保留一个小型本地机制，不新增任务调度器。

关键边界：Hydra 在进入用户 `main()` 前已创建目录并可能写 `.hydra` 和启动日志。因此不能用 `mkdir(exist_ok=False)` 直接替代认领，也不能声称 workspace 的标记保护了所有 Hydra 启动文件。默认独立目录降低这类冲突；本轮硬保证是旧**训练产物**不被覆盖，不引入 Hydra callback 框架追求全局事务。

验收：默认连续启动路径不同；两个进程认领同一空目标仅一个成功；旧核心训练产物字节不变；恢复到新目录成功，旧目录不写入。用临时目录与轻量 logger 替身，不启动真实 W&B 或训练。

### F03：远程启动不终止旧实验

位置：`scripts/remote/train_remote.sh`，关联 `stop_remote.sh`。

会话名及日志名追加每次启动的独立后缀，保留 `dex_` 和停止脚本允许的字符集；删除启动中的无条件 `tmux kill-session`。最终名字冲突应失败，不覆盖旧日志。打印实际会话名、日志位置和已有停止命令。停止旧实验使用已有 `stop_remote.sh <SESSION>`，不增加自动 replace 流程。

验收：`bash -n`；用替身 ssh/tmux 和临时日志检查两次启动命令，旧会话未被终止、旧日志未截断，错误退出码保留。不要通过连接真实服务器来验证“不会误杀”。

### F04：统一实际可安装的研究环境

位置：`requirements.txt`、`pyproject.toml`、README 安装段及确实需要延迟导入的可选功能。

基线存在 Hydra 1.2 与 >=1.3 的声明冲突；默认 DINO/LoRA 路径需要 peft，但未声明。不要用升级整个软件栈来掩盖问题。

修改：保留 Torch 2.4 / torchvision 0.19 / diffusers 0.27.2 / Zarr 2.x 这一研究基线，统一相容的 Hydra 1.3.x；按真实 import 补齐直接依赖和与当前 transformers/huggingface_hub 相容的 peft。固定版本需经过解析与构造检查，不能臆造“已验证版本”。requirements 记录实测环境，pyproject 的范围与之相容；README 给出一条安装顺序和实际验证的 Python/Torch/CUDA 组合。

大型可选后端可在选择该功能后再导入，缺失时报准确依赖名称；不要以关闭默认 LoRA、替换 backbone、移除点云等方式让测试变绿，也不引入多层 extras/环境矩阵。PyTorch3D 等 CUDA 扩展的安装按目标环境实际处理。

验收：隔离环境安装、`python -m pip check`、直接依赖导入、有效 policy 配置 compose/resolve；现有权重和环境允许时检查默认 RGB/点云 Agent 最小构造。配置检查不等于权重下载、CUDA 扩展或 GPU 前向已通过。环境不允许的部分明确 `NOT VERIFIED`，保留可执行的复验命令。

### F13：明确 resume_from 的外部路径语义

位置：`dexmani_policy/train.py`、`train_ddp.py`；保留 `training/checkpoint.py::CheckpointStore.resolve_path()` 的内部 tag 用法。

`resume_from` 表示旧实验目录或 checkpoint 文件。训练父进程先展开 `~`，把相对路径转成绝对路径，再交给恢复逻辑；目录仍解析到 `checkpoints/latest.pt`；DDP 在 spawn 前完成。

**相对基准明确为项目根目录**：当前入口已调用 `set_project_root()`，Hydra 捕获的工作目录也受此影响。不要误称为“用户在任意 shell 中启动时的原始目录”；不为本修复改变全仓库 cwd 规则。内部 `latest`/tag 接口继续保留，外部 resume 不再猜作当前实验 tag。

验收：项目根下的相对实验目录、相对 checkpoint、绝对路径、`~` 都解析正确；source 不存在时在开始更新前失败；内部 checkpoint tag 的行为不变。

### F19：发现和选择 DDP 实验

位置：`dexmani_policy/deployment/runtime.py::list_experiments/resolve_experiment`。

用 `config.yaml` 加所需 checkpoint 识别实验，支持普通三层和 DDP 四层等可变深度。递归扫描时忽略新增 selection/eval 快照等非训练目录；F08 使用 `eval_config.yaml` 也有助于避免误识别。

支持 experiments 根下的可变深度 selector、已有项目相对实验目录及显式绝对实验目录；前两者解析后必须仍落在 experiments 根内，避免路径拼接歧义。显式绝对目录保持现有能力。复用已有 checkpoint 文件与范围校验，不依赖固定路径层数。

验收：临时构造 `dp/task/run` 与 `ddp/dp/task/run`，均可列出、选择并进行 config-only inspect；错误路径、缺失配置或 checkpoint 给出清晰错误。不加载模型或驱动机器人。

## 4. 批次 B：训练数值与数据边界

### F15：裁剪前检查非有限总范数

位置：`dexmani_policy/training/trainer.py::apply_gradient_step()`。

事实：每个梯度元素有限不代表计算出的 float32 总范数有限。基线慢路径可能先用 Inf 范数把梯度缩为零，再通过逐元素检查，并错误推进 optimizer/scheduler。

启用裁剪时统一使用 `clip_grad_norm_(..., error_if_nonfinite=True)`；正常路径避免额外逐参数 GPU→CPU 同步。异常时才收集参数诊断；如果总范数溢出但所有梯度元素有限，也必须失败，不能因“没有坏参数名”吞掉异常。保留非数值 RuntimeError 的原始原因。未启用裁剪时保留必要的 finite 检查。

旧 `fast_grad_finite_check` 字段可继续被历史配置接受，但不能再决定正确性；不维持两套裁剪算法，也不引入 silent skip-update 策略。

验收：正常有限梯度与原正确路径一致；NaN/Inf 元素被拒绝；构造有限的约 `1e20` 梯度使 float32 总范数溢出，也在任何更新前失败。optimizer 状态、scheduler、EMA、global_step 都不得推进。保留 DDP 已有 backward/no_sync 行为。

### F05：仅拒绝真正混合 batch 大小的累积组

位置：`training/resume.py::build_train_loader()` 或其直接调用的局部检查；`build_utils.py` 的现有累积校验；不重写 Trainer。

本次复查调整：不再统一禁止 `accumulation > 1 && drop_last=False`。dataset 整除 batch，或者小尾批单独占一个累积组时，现有算法仍正确。

对当前标准 DataLoader＋`ResumableDistributedSampler`，在恢复游标改变前使用完整 epoch 信息：令 `N = sampler.num_samples`（每 rank）、`B = batch_size`、`A = gradient_accumulation_steps`、`L = ceil(N/B)`。当且仅当下列条件同时满足时，末组包含不等大小的 micro-batch：

```python
A > 1 and not drop_last and N % B != 0 and L % A != 1
```

在开始模型更新前拒绝该情况，错误信息显示 N/B/A 和可选修正（`drop_last=True` 或 `A=1`）。空 loader 继续用已有校验拒绝。用完整 sampler 计数／`full_num_batches()`，不要用恢复后缩短的 `len(loader)` 重算分组。

保留 epoch 末不足 A 个、但每个 batch 等大的组：现有 `group_size` 处理正确。当前 sampler 各 rank 样本数相同；若未来引入自定义 batch sampler/IterableDataset，需重新证明其组内大小约束，本文不扩展支持范围。

验收：N=5/B=2/A=3（2、2、1 同组）拒绝；N=4/B=2/A=3 允许；N=5/B=2/A=2（2、2 一组，1 单独一组）允许；A=1 和 drop_last=True 不受此新增限制。等大累积与等效大 batch 更新一致，尾组和恢复游标语义不变。上述判定已在复查中与 8,640 种 sampler/batch/accumulation/world_size 组合的分组大小核对一致，执行者仍需测试实际集成。

暂不实现跨 rank 样本加权累积；那需要同时处理 loss 分母、DDP 平均、裁剪顺序和日志，不是本轮必要能力。

### F06：读数据前检查 episode_ends

位置：`dexmani_policy/datasets/replay_buffer.py`。

增加一个局部复用的元数据检查：episode_ends 非空、一维、整数且非 bool；第一项 >0；严格递增；末值与所有实际读取字段的时间维长度一致。拒绝回退、重复、越界、浮点和空边界；字段缺失／无时间维给出可定位的错误。

在 `copy_from_path()` 大数组 `arr[:]` 之前用 Zarr metadata/shape 检查；直接内存 root 构造也检查。比较相邻值可避免无符号整数差分下溢；不要在每次 `__getitem__` 中扫描。

验收：`[10, 5, 20]` 等非法边界在 sampler 建立前失败；合法短 episode、前后 padding、train/val 划分和读取 dtype 不变。回退边界确实可制造帧重叠，但这不等于用户现有数据已损坏。

### G02：避免 joint action:auto 的全量拼接

位置：`training/build_utils.py::build_normalizer()`，复用 `fit_field`/`fit_field_chunks`。

当前 action:auto 无条件 concatenate；joint action 的 auto 实际解析为 limits。单 chunk 直接 fit，多 chunk 复用已有分块统计，避免额外全量数组。不改变统计数据范围，不改变 action_ee 的 mixed normalizer、旋转维 identity 或其他动作表示。

验收：单／多 chunk、常量维和一般数据下的 scale/offset、归一化及反归一化与原实现数值相符（使用明确容差）；路径不再分配完整拼接副本。不得在没有 profile 时宣称吞吐提升百分比。

## 5. 批次 C：选点、评测与结果同步

F08/F09/F10/F11/F12 合并设计；复用已有结果 JSON、selection summary、best 指针和共享 loader。建议目录为 `eval_ckpt_selector/<run_id>/` 与已有 `eval_dexsim/<run_id>/`，根目录 `best_ckpt.json` 仍是用户入口。无需另外建立 manifest 数据库。

### F08：保存真正执行的评测配置

位置：`evaluation/protocol.py`、`select_best_ckpt.py`、`eval_best_ckpt.py`，检查脚本和 `record_demo.py` 对共同逻辑的使用。

最终评测已通过 `_prepare_result_dir()` 分配独立目录并拒绝重用非空目录，**保留并复用**；为 selection 补同等独立目录。每次运行保存 `eval_config.yaml`，至少包含：

- 历史模型配置来源，以及允许的 environment/eval override 合并后的有效配置；模型输入、normalizer 和权重仍来自保存实验。
- CLI、dotlist、best 记录和配置默认值按现有优先级解析后的实际 EMA/raw、NFE、policy seed 模式、shuffle seed、请求 episodes 和实际 task/seed 列表。
- Runner 构建后由保存的数据配置覆写的模型输入参数也应与快照一致，不能只保存表面上的 YAML。
- policy 代码和模拟器可取得的 commit/version、dirty 状态；无法取得时写 unknown，不捏造版本。dirty=true 本身不能还原源码。

在现有结果中保存配置引用、解析后的真实 checkpoint 路径（优先实验相对路径，便于同步）、global step、逐 episode 结果和实际分母。NFE sweep 的父快照记录完整请求，每个子结果记录实际 NFE。**单纯 `OmegaConf.save(cfg)` 可能漏掉独立 CLI 参数，不足以验收。**

不修改训练 `config.yaml`，不从当前数据重拟合评测 normalizer；不为每个候选反复 hash 大 checkpoint。遇到 `latest` 先解析到具体权重，不以可变别名作为唯一证据。

验收：改变合法的 table/instance randomization override，快照准确反映；CLI 覆盖 best 的 EMA/NFE 时快照与调用实参一致；结果能定位具体权重；两次运行不覆盖；单次与 sweep 均覆盖。替身 runner 只证明协议，不证明真实仿真成功率。

### F09：以真实 (task, seed) 保证 held-out

位置：`env_runner/multi_task_sim_runner.py`、selection/final-eval 的 seed 选择逻辑和 best 元数据。

保留当前“reference seed 的序号 → 各任务有序池相同序号”的映射算法。记录任务顺序、各任务去重后的有序池身份（确定性摘要及长度即可），以及选择阶段所有候选、所有实际启用阶段映射出的 task/seed 排除集合。

可以在派发每次候选评测前记录其映射后的请求集合，作为保守排除集合；完成的 episode 结果另外保存，不把尚未执行的请求伪称为完成 episode。排除集合必须涵盖普通失败 episode、非最终 best 候选和 tie-break，不能只取 best 的成功样本。候选 fatal error 仍使本次 selection 失败，不发布新 best。

最终 `best` held-out 评测：先检查任务顺序与池身份一致，再使用既有 reference seed 排除，最后检查**映射后的** task/seed 与 selection 排除集合无交集；在 rollout 前完成。池重排或任务变化时明确要求重新建立 selection 记录，不猜测旧 reference seed 的含义。

旧多任务记录缺少足够池身份时，严格 held-out 入口要求重新选点；显式加载 checkpoint 做普通推理仍允许，但不能标记已证明 held-out。旧单任务的明确 `selection.seeds` 可继续使用。不同 selection 或人工调参看过的全部测试样本，不在本局部检查的证明范围内；正式比较另固定公共协议。

验收：A 池 `[0,1,2,3]`、B 池 `[100,101,102,103]`，选 reference 0/1；把 B 池改为 `[102,103,100,101]` 后原先 final reference 2/3 会复用 B 的 100/101，必须提前拒绝。固定池正常通过，任务顺序变更拒绝，单任务兼容，最终集合真实无交集。

### F11：保存候选证据并隔离录像

位置：`select_best_ckpt.py::CkptEvalAccum`、两阶段候选调用及持久化逻辑。

把已收集的 `episode_details` 保存到对应候选／阶段结果，含 task、seed、success、steps、checkpoint、阶段，能重建成功数、分母和选点排名。保持普通失败与基础设施异常的既有区别，不把模型/CUDA 异常改成 0% 并继续。

选点默认不录像（当前 eval_pipeline 已显式关闭，不能称其默认已触发覆盖）。显式录像时路径至少按 run、候选、阶段、task 分开；seed 继续用于文件名。最终 best demo 从选定权重独立录制，不复用混杂的候选录像。

验收：两个候选同 seed、同候选不同阶段的输出不覆盖；候选统计可从明细重建；关闭录像不创建不必要的视频产物；调用参数与目录引用一致。

### F12：best 指向最近一次成功发布的 selection

位置：`select_best_ckpt.py`、`agents/loader.py::read_best_ckpt_json()` 和相关输出说明。

每次 selection 使用独立 run_id 和 summary。满足已有选点成功条件后，先完成成功 summary，再以**唯一临时文件＋同目录 `os.replace()`** 原子更新根 `best_ckpt.json`。指针包含对应 selection ID/summary 相对路径；并发时不能共用一个临时文件名。

全零或 fatal error 时尽力保存失败 summary，包含原因和已完成的候选结果，然后保留原始异常/非零退出；不吞掉异常，不发布新 best。之前成功的 best 保留，日志明确本次未更新。无法处理 SIGKILL/断电时也至少保证旧 best 完整；不承诺所有崩溃都能生成失败日志。

新格式读取时校验所引用 summary 的成功状态和 checkpoint/global_step 等对应关系；同步不全应准确报缺文件，不悄悄换权重。旧格式 best 仍能作为历史记录加载，显式标记来源信息缺失，不伪造 selection ID；F09 的 held-out 要求单独适用。

验收：成功 A → 失败 B 后仍能明确辨认 A；成功 C 后指针/summary 一致；发布中断不产生半个 best JSON；原 pipeline 遇到 selection 失败仍终止。**保留旧成功 best 本身不是 bug，不删除它。**

### F10：更新可变的小文件

位置：`scripts/remote/sync_down.sh`。

保留第一遍 `--ignore-existing` 下载新文件的策略，且**不要给这一遍加 `--partial`**：残缺 checkpoint 不能在下一次因“已存在”被跳过。继续增量更新 metrics/latest。

对 best、selection summary、result JSON、有效评测 YAML 等实验小型元数据增加可靠更新规则，即使内容等大小、mtime 粒度相同也能更新。小文件可用 checksum；不要把 checksum 扩展到所有大 checkpoint 或巨大的持续增长日志。按需要拆一个小文件 rsync pass 即可，不引入同步框架。

保留 latest 的 symlink 语义，不默认解引用复制大权重；不使用 `--delete`。指针与其目标短暂不同步时，读取端报缺失而非 fallback。

验收：使用本地目录跑真实 rsync fixture，两轮 best 从 old.pt 切到 new.pt，同大小同 mtime 的 JSON 也更新；summary/快照正确；已完成大 checkpoint 无重复传输；检查中断策略不会把部分文件当成完整文件。shell 参数与 dry-run 行为保持可用，不连接真实服务器做破坏性检查。

### G01：每任务成功率附带基本统计

位置：复用 `evaluation/protocol.py` 的现有统计归约及结果序列化。

保留每任务成功数和实际分母，增加一个小型、无新依赖的 95% Wilson 区间函数，参考 LeRobot 的实现。n=0 返回明确的无结果（如 null），不要产生假的 0% 置信区间。保留 micro/macro 定义和现有选点规则。

区间描述给定模型与当前 episode 抽样协议下的统计不确定性，不能替代多训练 seed 方差；不要给 macro 平均直接套二项区间，也不要借此改 ranking。验收覆盖 n=0、0/n、n/n 和一般成功数，并核对公式；真实 rollout 不必要。

## 6. 批次 D：VQ/codebook

### F14：验证指标按样本数归约

位置：`scripts/training/train_vq_hand.py` 的 train/val 聚合与 best 选择。

对实际定义为逐样本均值的 enc/vq/reconstruction MSE，累加 `batch_mean * batch_size` 再除以总样本数；验证 MSE 必修，train 日志采用相同口径。usage/perplexity 等按自身定义计算，不机械加权。保留 loss 权重、数据 split、优化更新和选点指标的选择。

验收：同一组 257 个样本，前 256 个预测误差为 0，最后一个误差为 10；全局 MSE 应为 `100/257 ≈ 0.3891`，batch=256 与 batch=257 一致，不再把两个 batch 均值平均得到 50。加入 MSE=1 的候选验证选点排名不受验证 batch size 影响；无需完整 VQ 训练。

### F18：codebook 内容验证后再导入

位置：`dexmani_policy/agents/vq_hand/codebook_manager.py`。

事实边界：现有 state_dict 路径已检查 pose/weights、幂关系、排列和 hand range，但不能据此声称 affine normalizer 的数值已经全部校验。NPZ 路径存在额外缺口，且先修改部分成员后才完成读取。

提取局部校验，供 NPZ 和 state_dict 路径复用其共同语义：

- pose/weights 非空、有限、shape 正确；pose 数量与 `codebook_size ** num_groups` 一致；维度和组数为合法整数。
- permutation 原始 dtype 必须为整数且非 bool，长度正确、无重复、完整覆盖；在 `.long()` 或 state_dict copy 类型转换前检查，避免浮点坏输入被截断成合法排列。NPZ 的整数元数据同样不能靠 `int(坏浮点值)` 蒙混通过。
- hand_min/max 为有限标量且 max>min；已加载运行时 codebook 的 affine scale/offset 维度匹配、有限，scale 非零，保留已有坐标系语义。
- 如存在 per-group poses，检查全部组和对应 shape/有限性；metadata JSON 在提交对象状态前解析并检查必要结构。

NPZ 在局部变量中读完并验证，全部通过后再更新成员，失败时对象保持原状。state_dict 保留已有严格恢复规则，并补齐共同数值检查；区分合法的未初始化全空 manager 与已加载 runtime 状态，不意外禁止原本正常的初始化/EMA 流程。不改变 NPZ 格式版本或建立通用 schema 系统，除非实际源码证明必需并说明理由。

验收：NaN/Inf、错误 shape、幂关系错误、重复/浮点 permutation、坏整数元数据、非法 hand range、零/NaN scale 均拒绝；坏 NPZ 不改变已有 manager；合法 NPZ 与 state_dict roundtrip 的查表/解码一致；空初始化与正常 EMA 复制不回归。默认 EMA/state_dict 已能拦住部分坏 pose，不把问题夸大为默认训练必然吞入所有坏文件。

## 7. 批次 E：采样配置与 RNG 恢复

### F16：显式控制 DDIM 的 x0 裁剪

主要路径：`agents/action_decoders/diffusion.py`；`agents/core/base.py` 中 UNetDiffusionAgent（DP/DP3 经 kwargs）；`core/dqrise.py`、`core/r3d.py`、`core/multi_task.py`；normalization helper、训练 builder、共享 loader、resume contract；五个 diffusion 基础 YAML 及其 DDP 继承配置。

事实：基线 DDIMScheduler 固定 `clip_sample=True`。Gaussian 动作的非退化维可超过 ±1，固定裁剪会把 x0 限制在约原始均值 ±1 标准差。DDIM 本身支持关闭此裁剪；这不是“DDIM 不支持 Gaussian”。DP3/CordViP 的公开主配置配合 limits 动作归一化，默认 true 合理；normalizer 存在 Gaussian 分支不代表主路径在使用它。

| normalization.action | agent.clip_sample | 本轮支持语义 |
|---|---|---|
| auto / limits | true | 保持当前有界动作默认行为 |
| auto / limits | false | 允许裁剪消融 |
| gaussian | false | 支持 Gaussian 动作实验 |
| gaussian | true | 在 rollout/训练更新前明确报错 |

实施要求：

1. 在 `Diffusion` 构造参数**末尾**追加 `clip_sample: bool = True`，检查严格 bool 类型，用关键字传给 DDIMScheduler。Agent 构造新增参数也避免移位已有位置参数。
2. 透传所有真实 Diffusion 构造点；MultiTask 仅 diffusion 分支使用，rectified_flow 分支不受此规则误伤。基础 `dp/dp3/dqrise/r3d/multitask_dit` 配置显式 true；检查实际 compose 后的 DDP 配置，不重复复制无必要字段。
3. 建一个小型共享兼容性判断。配置预检查在能明确确认 diffusion 分支时提前报错；模型构造后必须根据**实际 decoder 与 scheduler** 再检查。可用 `BaseAgent.set_normalization_spec()` 收口，由训练 model、EMA 和共享推理 loader 调用。不要通过散落的类名字符串猜测所有 decoder，也不为此造 registry。
4. 仅校验 Gaussian **动作**与 x0 裁剪；Gaussian 观测不受限制；flow/SAT 等无此 scheduler 的路径不受限制。不 clamp 训练标签，不自动改变 normalization。
5. saved config 决定推理开关；raw/EMA loader 一致。resume 严格比较只对已知 Diffusion 语义的新增 `agent.clip_sample` 做“缺失 ≡ true”规范化，false 保持真实差异；比较副本，不修改历史配置或 checkpoint，不把其他字段缺失一律忽略。

验收：默认配置输出回归；用实际 decoder/scheduler 的 oracle x0=±3 检查 false 能保留 ±3、true 最终裁为 ±1；覆盖 sample/epsilon/v_prediction（各自 oracle 需按其参数化正确构造）。配置拒绝矩阵、Gaussian 观测和 flow 不误拦、各构造点透传、raw/EMA 恢复、旧 missing 与 true 等价、true 与 false 不等价均覆盖。可以用轻量 backbone，避免下载大模型。

历史 Gaussian＋固定 true 的 checkpoint 不被自动“修正”为 false。新严格入口应解释不匹配及重新评测要求；需要忠实复现旧错误行为时使用其旧代码版本。修正后的结果是新实验记录，不能冒充原结果。默认 limits 历史实验应兼容。

不增加动态阈值、自动 clip range 或新的 scheduler 体系。DP3/CordViP 的 diffusers 版本与本仓库不同，本次只借鉴范围匹配原则，不承诺逐位 sampler 等价。Flex-pi 的 z-score 实际裁到 ±5，也不是“完全不裁剪”的证据。

### F17：RNG 随 rank 的实际训练设备保存和恢复

位置：`utils/random.py`、`training/trainer.py::_save_checkpoint()`、`training/resume.py::restore_training_state()` 及两条入口传参。

新记录保留 Python/NumPy/CPU Torch RNG，并保存**该 rank 实际训练设备的一份** CUDA RNG；沿用现有 rank-ordered `rng_states` 外层。capture/restore 显式传入 `Trainer.device`／目标 device，不能只调用默认 current_device：单卡 `cuda:1` 不一定经过 `set_device()`。

恢复时将 rank 对应随机流应用到该 rank 当前目标 GPU，使相同 world_size 下的可见 GPU 数量或设备映射调整不再依赖旧列表下标。保持 checkpoint 顶层格式和 world_size 检查，不宣称可改变 world_size 精确续训。新单设备 Tensor 与旧 list 必须可明确区分；CPU-only 状态仍可处理。

旧 checkpoint 的多设备 list：优先从**来源实验保存的配置**确定该 rank 原 logical CUDA 槽位（单卡 device／DDP gpu_ids 或原 rank 映射），提取对应旧状态后恢复到新目标。只有一个合法槽位时可直接确定；信息不足或冲突时拒绝声称精确恢复，不能按当前新映射猜。若需新增参数，在单卡与 DDP 两条加载路径一起传递来源信息，禁止用新实验配置替代历史证据。

不因无法恢复 RNG 阻止正常的权重推理。worker augmentation/prefetch 未保存的边界继续写明；该修复不保证不同硬件、非确定性算子和 worker 增强的逐位一致。

验收：CPU RNG 回归；替身 device setter 验证参数流；有 CUDA 时验证 cuda:0→cuda:1、单卡非零 device、同 world_size 的 gpu_ids 重排／可见设备减少，比较恢复前后实际设备随机序列和短更新。只有 mock 通过时必须标 `NOT VERIFIED: actual CUDA remapping`，不能据此声称 GPU 恢复通过。

## 8. 批次 F：数据身份

### F07：轻量数据身份约定

位置：`datasets/replay_buffer.py` 已保留的 root attrs、单／多任务数据集、`training/build_utils.py` 的元数据捕获、保存配置、`training/resume.py` 的比较；数据生成端只在本仓库确有对应入口时修改。

默认研究约定：训练数据路径不可变；重新生成/修正数据时使用新路径和新 `data_revision`。从 Zarr attrs 读取可选的非空字符串 revision，单任务／每个子任务分别记录到训练快照和恢复元数据；读取发生在加载阶段，保存 checkpoint 时不重新扫描数据。不要把记录字段随意塞进 Hydra dataset 构造 kwargs 导致构造失败。

新旧兼容规则需显式实现：

- 保存值和当前值都已知且不同：拒绝恢复。
- 保存值已知而当前身份丢失：拒绝，不能通过移除 attr 绕过已建立的约束。
- 旧 checkpoint 或旧数据没有可比较的身份：保留既有路径/长度等检查，允许原有恢复能力，但日志/报告明确“数据身份未验证”；不能把当前新读到的 revision 反填成旧 checkpoint 的历史身份。
- 多任务逐 task 比较；缺省兼容仅限这项可选字段，不放松 dataset 配置、长度、world_size 等已有严格检查。

revision 是生产者声明的身份，**不是内容 hash**。同路径同形状数据被改而 revision 未更新，仍不能检测；不要用 mtime/shape/episode_ends 摘要伪称内容校验。若确需内容审计，在生成时计算一次摘要或提供一次性显式验证，不默认每个 rank 全量扫描 Zarr。

验收：revision 一致通过、不同拒绝、已知变缺失拒绝、旧身份缺失准确报告；多任务覆盖；训练快照和 checkpoint 一致。缺少本仓库内的数据转换器时只补约定/读取端，不臆造转换 pipeline。此处完成的是最小身份约束，不是完全防篡改。

## 9. 必须共同检查的兼容性

| 历史或边界输入 | 预期行为 |
|---|---|
| 旧 limits 配置缺 clip_sample | 按 true 重建；与新显式 true 的 resume 等价 |
| 同一 Diffusion 实验 true→false | 属于采样配置变化，严格 resume 不静默等价 |
| 旧 Gaussian＋true | 不静默修正；新入口明确拒绝不匹配，旧行为用旧版本复现 |
| 旧 RNG list 有可信来源设备配置 | 选择原 rank 活跃槽位，再写入当前目标设备 |
| 旧 RNG list 无法确定活跃槽位 | 不猜精确续训；普通权重推理仍可使用 |
| 无 revision 的旧 checkpoint | 原有检查继续；明确数据身份未验证 |
| 旧单任务 best 有 selection.seeds | 可继续使用既有 held-out 规则 |
| 旧多任务 best 缺 seed 池身份 | 可推理；严格 held-out 需重新选点 |
| selection 失败后仍有旧 best | 保留最近成功记录，显示其真实身份 |
| 新 best 引用的 summary/权重尚未同步 | 清楚报缺失，不换成 latest/raw 等其他结果 |

“兼容”只针对上述可证明的情况，不构建自动迁移框架，也不把所有新字段都从严格检查中删除。

## 10. 参考项目、机制选择与暂缓事项

以下均为审查时读取的固定版本。需要移植具体代码时重新阅读相应文件，并遵守原仓库许可证；机制参考不要求复制代码或升级到上游依赖版本。

| 项目／固定提交 | 已核对的机制与源码 | 本轮使用 |
|---|---|---|
| Diffusion Policy `5ba07ac6661db573af695b419a7947ecb704690f` | [SequenceSampler 与 key_first_k][R-DP] | 保留窗口/normalizer 思路；部分观测读取仅按 profile 决定 |
| DexUMI `acddb8f8a89a8f0186868bbec44306eb7808114a` | [数据集动作构造][R-DU-DATA]、[相对 SE(3) 变换][R-DU-MAT] | 用于动作表示研究的正确性约束，当前不改动作空间 |
| Flex-pi `20c1b2b71ea35a415d5d47c39b04443cfadad7a1` | [采样器][R-FLEX-SAMPLE]、[normalizer][R-FLEX-NORM]、[有效样本归约][R-FLEX-LOSS]、[模态 helper][R-FLEX-MOD] | 借鉴分母/范围匹配原则，保留现有分层采样分布 |
| LeRobot `ff71cae1ae2d09fd035553c35da65888ed6c8304` | [输出目录约束][R-LR-TRAIN]、[RNG][R-LR-RNG]、[sampler][R-LR-SAMPLE]、[Wilson][R-LR-STATS] | F02/F17 的机制参考及 G01，小范围接入 |
| DP3 `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97` | [DDIM 配置][R-DP3-CFG]、[训练入口][R-DP3-TRAIN]、[默认 limits 数据集][R-DP3-DATA] | 支持 F16 保持有界动作默认裁剪 |
| CordViP `ad6d441abe7b0480cf79b0311db266965c1313b6` | [DDIM 配置][R-CORD-CFG]、[训练入口][R-CORD-TRAIN]、[默认 limits 数据集][R-CORD-DATA] | 同 F16；encoder 预训练不代表动作使用 Gaussian |

以下不纳入本轮代码修改：

- **公共跨方法选点／测试池**：正式论文比较时固定简单 task→seed 列表，保证各方法相同协议；当前“排除本实验选点 seed”不自动等于公共测试集。先记录研究协议，不建设 experiment registry。
- **全局 batch／训练预算对齐**：复用已有 `print_training_recipe`；单卡与 DDP 默认全局 batch 不同是配置事实，不自动改 LR、batch 或总步数。
- **DP key_first_k、Zarr lazy/cache**：当前 BaseDataset 已先截到 obs_horizon 再增强；RAM slice 可能只是 view，不能宣称 horizon/obs_horizon 就是吞吐提升倍数。先测 data wait、各 rank/worker RSS 和吞吐，再处理真实瓶颈。
- **Flex-pi weighted sampler 替换**：IID weighted multinomial 与现有每任务目标计数不同，不能作为“等价优化”直接替换。
- **模态 anchor、presence、缺失模态有效分母**：改变训练目标/条件分布，作为独立研究实验。
- **DexUMI 相对 SE(3) 动作**：`inv(T0) @ Tt` 不是世界坐标逐元素相减；表示、normalizer、逆变换和控制端必须一起验证，当前不自动改。
- **train-only normalizer**：当前 full-buffer 统计明确存在，类似路径也见于 DP/DP3；严格 few-shot 研究需单独选择并记录统计范围，不混入本轮默认修复。
- **worker 增强逐位恢复、全内容 hash、多环境矩阵、全量重构**：没有当前研究必要性，不增加默认维护成本。

## 11. 验收和最终交付

按风险选择检查，不在每个纯文档或局部改动后重复跑全部重测试。

基础入口（以当前 CLI 为准）：

```bash
python -m pip check
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise r3d multitask_dit
bash -n scripts/remote/train_remote.sh scripts/remote/stop_remote.sh scripts/remote/sync_down.sh
git diff --check
```

再按实际影响检查其余 policy/DDP 配置。`--config-only` 仍需要相关 Python import，但不要求真实数据/GPU；不要给这个入口追加它不支持的 Hydra override 参数。需要组合测试时用小型 compose fixture。目标环境可用时，再运行相关 `python dexmani_policy/smoke_test.py <config_name>`，并把实际使用的数据/权重/设备写入报告。

必须保留的定向回归：

| 风险 | 最低充分验证 |
|---|---|
| 训练/恢复共同路径 | 无 worker 随机增强的小 CPU 模型，连续训练 vs 中断恢复；权重、EMA、Adam、LR、step、样本顺序一致 |
| F01 | 真实 spawn persistent workers＋epoch 0/1/0，不只测试 pickle.dumps |
| F05/F06/F15 | 不等批反例被精确拒绝；合法等大累积不变；非法 episode 提前失败；非有限范数不推进 |
| F08–F12/G01 | 轻量 runner 记录实际参数；真实 task/seed 隔离；明细可重算；best 失败/发布行为；本地 rsync 两轮更新 |
| F14/F18 | 257 样本尾批排名；NPZ/state_dict 合法 roundtrip 和坏输入拒绝；坏 NPZ 不半更新 |
| F16 | 实际 scheduler 的裁剪 oracle、所有构造透传、旧默认等价、raw/EMA loader 一致 |
| F17 | CPU/mock 只是参数流证据；实际 GPU 映射和最小 DDP 恢复需目标 CUDA 环境验证 |
| F07/G02 | revision 兼容矩阵；分块统计与原归一化等价且避免拼接 |

先前审查已用 CPU/合成数据或源码确认这些失败机制，但那些临时审计文件不是本仓库的测试依赖。执行者应将必要反例整理成少量可重跑的仓库检查，真实调用修改后的实现；不能只复制一段“正确算法”自测，也不要求复制上游全部测试。

`docs/infra_fix_report.md` 最少包含：

1. 实际起始 HEAD、最终改动摘要、范围调整及原因。
2. F01–F19、G01/G02 逐项状态：IMPLEMENTED／ALREADY FIXED／BLOCKED，并附代码依据；测试另列 PASS／FAIL／NOT VERIFIED，避免“实现完成＝环境验收通过”。
3. 实际执行命令、关键结果、GPU/数据/权重/仿真缺失项及复验命令；不写推测的成功率或性能提升。
4. 新用户操作：安装、resume 路径、冲突处理、Gaussian 开关、selection/best 语义、数据 revision、历史产物边界。
5. 对论文对比的影响：修复前后提交需要区分；F14/F15/F16 或数据版本变化可能改变结果，旧实验不会自动被修正。

任务通过标准：当前能验证的检查通过；所有编号有完整处置；未验证项诚实列明；默认有界动作训练和历史正常推理不回归；没有无关算法改动、泛化框架或实验产物损坏。真实数据完整训练、模拟器成功率、外部动作单位/关节顺序/成功判据、真机时序仍需对应环境验证，不能在代码任务完成时宣称全部已证实。

[R-DP]: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/sampler.py
[R-DU-DATA]: https://github.com/real-stanford/DexUMI/blob/acddb8f8a89a8f0186868bbec44306eb7808114a/dexumi/diffusion_policy/dataloader/dexumi_dataset.py
[R-DU-MAT]: https://github.com/real-stanford/DexUMI/blob/acddb8f8a89a8f0186868bbec44306eb7808114a/dexumi/common/utility/matrix.py
[R-FLEX-SAMPLE]: https://github.com/geyan21/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/utils/samplers.py
[R-FLEX-NORM]: https://github.com/geyan21/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/datasets/lerobot/utils/normalizer.py
[R-FLEX-LOSS]: https://github.com/geyan21/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/flexpi.py
[R-FLEX-MOD]: https://github.com/geyan21/flex-pi/blob/20c1b2b71ea35a415d5d47c39b04443cfadad7a1/src/flexpi/models/helpers/flex_joint.py
[R-LR-TRAIN]: https://github.com/huggingface/lerobot/blob/ff71cae1ae2d09fd035553c35da65888ed6c8304/src/lerobot/configs/train.py
[R-LR-RNG]: https://github.com/huggingface/lerobot/blob/ff71cae1ae2d09fd035553c35da65888ed6c8304/src/lerobot/utils/random_utils.py
[R-LR-SAMPLE]: https://github.com/huggingface/lerobot/blob/ff71cae1ae2d09fd035553c35da65888ed6c8304/src/lerobot/datasets/sampler.py
[R-LR-STATS]: https://github.com/huggingface/lerobot/blob/ff71cae1ae2d09fd035553c35da65888ed6c8304/src/lerobot/utils/eval_stats.py
[R-DP3-CFG]: https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml
[R-DP3-TRAIN]: https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/train.py
[R-DP3-DATA]: https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/dataset/adroit_dataset.py
[R-CORD-CFG]: https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/config/CordViP_train.yaml
[R-CORD-TRAIN]: https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/load_pretrain.py
[R-CORD-DATA]: https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/dataset/robot_dataset.py
