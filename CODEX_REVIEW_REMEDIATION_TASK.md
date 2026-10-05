# DexMani Policy 全量审查整改任务书（Codex CLI）

状态：**方案已复审；本文是待执行任务，不是实现或测试通过报告。**

审查日期：2026-10-05。代码基线：[`650f1e3cfc63c64678b5bb69e7258b548800d053`](https://github.com/haoyangzhanglab/dexmani_policy/tree/650f1e3cfc63c64678b5bb69e7258b548800d053)。发布前重新核对了目标仓库和五个参考项目的 HEAD。执行时以实际 HEAD、保存的实验配置及工作区为准，**不要 reset 到此基线**。

目标：以最少机制完成真实缺陷修复、降低可证实的资源开销、保持论文实验可解释。覆盖前面各轮归并后的 **62 个编号**；这不表示存在 62 个待修 bug。本文自包含，不依赖聊天记录、外部审查报告或审查者的临时探针。

`CODEX_INFRA_TASKS.md` 是已实施的历史任务书，本轮不重新执行其 F/G 清单；其中过时的事实不覆盖当前代码。先读 `AGENTS.md` 及改动目录适用的指令。本文是一次性任务，不把模型配方写入 `AGENTS.md`、`CLAUDE.md` 或项目级 skill。

## 1. 执行范围与优先级

1. 检查 `git status`、当前分支、HEAD、适用指令和实际运行入口，保留用户已有修改。若源码已变化，按真实调用链重新判断；已修复的项目记录依据，不再修改。
2. 按下述批次推进到代码、配置、脚本、保存/恢复和定向验证闭合，不只提交分析或计划。局部环境受限时，继续完成不受阻的项目，留下明确的复验方法。
3. 尽量沿用已有函数和数据结构。不要引入新的 registry、配置 schema 框架、实验管理平台、通用导出协议、另一套 Trainer 或第二条长期维护的训练链。
4. 不修改动作含义、关节顺序、单位、loss 权重、默认 NFE、模型容量、训练预算和默认时间分布来让测试通过。确实改变数值配方、训练样本或随机流时，明确记录变化和适用新实验的边界。
5. 不删除/覆盖旧数据、实验、checkpoint、视频、日志及预训练权重。不得自动启动真机运动、远程停任务、长训练或批量仿真。可运行覆盖当前风险的短 CPU/CUDA/DDP 验证；不改用户全局环境。
6. 默认在本地完成实现和报告，不自动提交、推送或合并实现代码，除非执行时用户另有指令。本文的发布提交仅包含任务书。

处置类型决定实际工作量，优先级决定先后，二者不能混用：

| 类型 | 本轮要求 |
|---|---|
| FIX | 必做最小修复/机制简化；已修复则保留。不能因为默认配置暂未触发就跳过已明确的小型边界缺陷。 |
| CHECK | 必做调用路径与证据检查；环境可用且路径相关时做有界诊断/基准。只有正确性、适用性和收益证据充分才落地优化；否则记录暂缓原因，不加入新开关占位。 |
| USE | 只有当前工作流确有该用途才实现；没有需求或缺少必要实物条件时记录不适用/暂缓，继续主线。 |
| DOC | 必做事实披露及后续实验设计；本轮不自动改变模型或运行论文实验。 |
| KEEP | 本地正确、已规避或并非缺陷。保留；邻近代码受修改影响时做必要回归。 |
| DEFER | 研究选择，明确暂缓；不作为本轮代码完成的阻塞条件。 |

P1 优先消除产物覆盖、DQ 数据空间不一致、无效选模与消融不生效，并建立论文比较的可靠入口。P2 完成边界正确性与低成本资源简化。P3 完成较外围的小修复；可选研究和发布功能不挤占主线。

推荐顺序：**W01 的 B10 + W02 → W03 的 B06 → W08 → W03/W04/W05/W06/W07 的剩余 FIX → W01 的 E03 → CHECK/USE → W11 与最终报告**。W05 的恢复整理与任何训练入口改动一起核对；R08 的验证贯穿相关批次。W 编号仅为归并索引，不要求创建 11 个 PR。

## 2. 本次复审对上一版方案的调整

- **VQ 对齐数据合同，不强制统一训练/验证比例。** 策略的有效训练行和 hand affine 是权威来源；允许 VQ 在其内部保留用于选点的小 holdout，但必须标明其不是整条 pipeline 的独立测试集。不得为“统一”把 policy 默认 `val_ratio=0` 改成 0.05。
- **`discrete_pow` 小批次先拒绝不支持的组合。** 独立同分布地采指数是新训练分布，本轮不把它包装成等价修复。
- **精度调整先做数值诊断。** BF16 可训练参数/EMA 是风险证据，不是任务失败的实验证据。取消无条件把所有可训练 backbone 和 EMA 切成 FP32 的要求；优先检查实际 LoRA 配方和更新损失。
- **优化按真实瓶颈落地。** foreach、KNN、日志同步、RTC、FPS 和多任务 sampler 均有数值/时序/随机性边界。已有实现足够时不增加缓存层、共享内存系统或兼容框架。
- **代码追溯保持局部。** 保存实际源码及身份即可；不强制每个 run 新建 checkout，不以源码 hash 代替原有 resume 数学合同，也不因纯文档变化拒绝恢复。
- **官方代码也是待判断的实现。** 保留本地已修正的 EMA 时钟、累积边界、teacher eval、DQ 索引取整、learned-weight 码本和单次采样 KV cache；不为了外观接近官方而退回缺陷。
- **明确不做的范围。** 没有实际需求不迁移 torchrun、不引入推理产物新格式、不扩大真机支持、不实现新的时间目标或时序点匹配算法。

## 3. W01：数据、VQ 坐标与统计

主要位置：`dexmani_policy/datasets/base_dataset.py`、`datasets/sampler.py`、`datasets/multi_task_dataset.py`、`training/build_utils.py`、`agents/normalization.py`、`scripts/training/train_vq_hand.py`、`dexmani_policy/configs/dqrise.yaml`。未写前缀的目录沿用该节明确的包路径，实施时以实际文件为准。

### B10 / R01：一个有效数据口径，一份 hand affine

成因：policy 已按有效训练窗口引用的去重源行统计，VQ 仍用全量 hand 数据拟合、用独立 episode 划分取帧，且遗漏观测有效性/dispatch 过滤；两阶段不再处于相同坐标和数据预算中。R01 的旧 policy 全量统计问题已经关闭。

最小方案：

1. VQ 读取**完整目标 policy 配置**并解析现有 OmegaConf resolver，复用该 Dataset/sampler 的有效性规则、episode 选择和 action 源行。不能只读取 `vq_vae` 小节或只检查 hand 是否有限；观测无效可能使某些动作不再可用于训练。
2. 令 `P` 为 policy 有效训练窗口引用的**唯一 action 源行集合**。VQ 优化样本必须取自 `P`；不是简单的所有训练 episode 全部帧，也不要求按 policy 窗口重复频次采样。
3. hand normalizer 直接取 policy action normalizer 对应 hand 维度的 affine 参数及统计；在 VQ 中保存为现有 hand 字段。禁止再次独立 fit、重新推测 min/max 或绕过 codebook 一致性检查。当前支持的 `action`/`action_ee` 分解分别核对；不支持的 normalizer/动作组合提前说明并拒绝，勿自动退回 limits。
4. 保留 VQ 内部验证的明确用途：若其验证比例大于零，在 `P` 所属训练 episode 内按既定 seed 做 episode 级内部 holdout，先划 episode 再取各自的有效行；VQ 训练行与内部验证行不交叉。policy 的验证/测试 episode 不进入 VQ 优化或归一化。不要二次无意降低 `max_train_episodes`；policy cap 定义可用池，内部 holdout 只再分配该池。
5. hand affine 为与 policy 一致而使用整个 `P`，故上述内部验证共享该 affine，且其 episode 以后可用于 policy 训练。明确命名/记录为 **VQ 内部选点诊断**，不能宣称整个策略都未见过它。若论文需要全流程严格 held-out，由外层数据协议划分，不能偷偷缩减默认 policy 训练预算。
6. 显式 `val_ratio=0` 才使用无验证集模式。要求非零验证比例却无法产生非空训练/验证集合时，在训练前报错。记录 policy eligible pool、VQ train/val 的 episode/行数、数据 identity/revision、有效性配方和 normalizer 来源。
7. 使用已有配置加载能力和一个明确入口。可保留现有 argparse wrapper，但对共享字段的覆盖必须先作用于目标 policy 配置再统一解析；冲突的 `action_key/tcp_dim/hand_dim/seed` 不能各自生效。VQ 优化器 seed 与 policy 数据划分 seed 的角色分清。删除不再有独立语义的重复字段，保留确属 VQ 的模型/优化/内部验证参数；不要同时维护两套默认配方。

验收：小型多 episode 数据中放置验证极值、无效观测、dispatch、NaN 和被窗口过滤的行；断言各集合、hand affine 与 policy 完全对应。走一次 **VQ 小步训练 → checkpoint → codebook 导出 → DQ agent 加载/预测** 的真实链路，不用手工伪造匹配参数替代。原有正常码本继续严格校验；旧不匹配码本明确要求新 run 重训，不静默改写。

### E03：真正分块的统计输入（FIX）

沿现有 iterator 做小改动：BaseDataset 按 source-row 索引分块读取；MultiTask 逐块 yield；`build_normalizer` 和 `fit_field_chunks` 不把整条 iterator 转为 list，不在 `action:auto` 上全量 concatenate。用一套 count/min/max/mean/M2 累积生成原 affine，避免另造 normalizer 类。保持样本标准差 `ddof=1`、原 `range_eps`、常量维退化处理、mixed action 的 rot6d 恒等与 aux 切片语义。

验收：空块、单样本限制、常量列、不同 chunk 大小、多任务及 action_ee 与原统计在合理浮点容差内一致；不能要求 FP64 归约重排后 bitwise 一致。用小型内存检查证明无 O(全部选中行) 的额外拼接；大数据 RSS/启动时间只报告实际测量值。这不能消除底层 replay buffer 本身的全量内存。

### E02：DDP 只拟合一次（CHECK）

先区分全 rank 读数据、重复 fit、构造模型及广播各自的成本。若重复拟合值得优化，fresh run 在 rank 0 fit，构造 agent 前分发小型 normalizer state；resume 直接用保存状态，不重拟合。沿现有 DDP 初始化和广播机制完成，确保当前 CUDA device、rank 生命周期和失败传播正确；rank 0 拟合失败时，其余 rank 不得永远等待新的 collective。单卡入口不依赖 distributed 初始化。

每 rank 仍会读数据，不能宣称实现了数据共享。mmap、lazy Zarr、跨进程缓存不在本轮默认范围。任何改动都验证各 rank affine 一致、fresh/resume 正确和失败无挂起；性能环境不可用时只记录候选方案。

## 4. W02：VQ 产物、选点与 usage

位置：`scripts/training/train_vq_hand.py`、`train_vq_hand.sh`、`extract_vq_codebook.py`、`measure_vq_usage.py`、`dexmani_policy/training/run_identity.py`、`dexmani_policy/agents/vq_hand/`。

- **B01（FIX）**：默认输出改为任务下独立 run 目录，并复用已有原子 run 认领。当前 `claim_run` 只查 policy 产物，VQ 接入前须拒绝已存在的旧 `vqvae_hand_*.pt`/同名 VQ 产物，不能把已有目录误当空目录。临时文件 replace 只保证完整写入，不保证实验隔离。保留 shell 显式输出路径的可用性，但已有 run 要报冲突。导出码本记录源 checkpoint、split、normalizer 身份，推荐输出到该 run；指定已存在的导出目标需明确的覆盖意图，否则拒绝。不要为此新建完整 VQ resume 功能。
- **B03（FIX）**：训练开始前固定选点指标，有内部验证集用 `val_mse`，无验证集才用 `train_mse`。被选指标为 NaN/Inf 时保留旧 best 并明确失败，不回退到另一个指标混排；best 变量/保存 metadata 使用真实指标名。保持已经修正的按样本数归约，不能退回 batch 均值等权。
- **E06（FIX）**：nearest-prototype usage 与 encoder tuple usage 分块计算，累计 count/距离统计，不构造全量 `N×K×D` 差值或收集全量 indices。先保留原平方差表达式、原型排序和 argmin tie 规则；不要为了公式看起来简短改成有抵消误差的新距离表达式。
- **A08/U01/U02/D06（KEEP）**：保留 learned softmax layer weights、encoder tuple 与 runtime 最近 prototype 两种 usage 的区分、模型内缓存码本，以及按 `K−1` 缩放后的 half-up/clamp。上游等权导出、每 query 读 NPZ 和直接整数截断的条件缺陷不属于当前待修本地缺陷。

验收：新 run/重复 run/旧式目录冲突；有限/非有限验证与无验证三种选点；257 样本等尾批反例；非等权码本导出重建一致；所有码字往返及边界；usage 在重复原型、近 tie 和多个 chunk 大小时与原式一致。覆盖真实保存/导出/加载路径，旧产物字节不变。

## 5. W03：配置生效与局部边界

### B06（P1，FIX）

位置：`dexmani_policy/agents/obs_encoder/pointcloud/registry.py`、`uni3d.py`、`dexmani_policy/datasets/multi_task_dataset.py` 及真实调用者。

现有 factory 显式消费它支持的参数；消费后剩余键报清楚的错误。`hidden_channels/num_layers/include_global_token/fps_random_config` 等参数，必须确实传入并生效或明确拒绝。分支不使用 FPS 时不能先 pop 再静默丢弃；不要通过删除用户配置掩盖失效消融。对 Uni3D 固定 LN 的合同明确处理 `norm` 并拒绝不支持值；移除吞掉未知参数的无用途 `**kwargs`。若当前合法配置显式写了受支持默认值，应继续接受，不以“一律拒绝”破坏已有入口。

MultiTask 检查每个 child 的实际 action key 与共享动作合同，不能只检查顶层。无需创建反射 schema 或扩大所有 agent 的配置接口；例如不能假设 DP3Agent 已有不存在的 `pc_encoder_config` 入口。

验收：对真实暴露该参数的构造链改键，断言实际模块结构变化或前置报错；拼错键被捕获。默认已支持配置通过；配置摘要反映实际层、维度、模态和可训练参数，而不是只复述 YAML。

### 其余小型正确性修复（FIX）

| 编号 | 最小修改与不能改变的行为 | 定向验收 |
|---|---|---|
| B05 | `agents/action_decoders/backbone/unet1d.py` 末端 Conv1dBlock 传 `n_groups`；检查当前保长卷积的正奇数 kernel、相关通道整除，以及按实际 stride-2 次数 S 的 `H % 2^S == 0`。不自动 pad/crop。 | H=15/16/18/20、两层/三层及 `[12,24,48]`、groups=4；合法 H 输出形状不变。约束放在实际有该限制的模型边界。 |
| B08 | `time_sampler.py` 对 `discrete_pow` 实际子批次 `B < floor(log2 K)+1` 提前给出不支持错误；调用层能预判时提早校验，运行时仍兜底。保留合法旧分配及默认 beta/discrete。 | K=10 时 B=1/2/3 明确拒绝；B≥4 与旧 helper 同输入/RNG 一致；覆盖 flow/consistency 分流后的 B，不只总 batch。禁止静默切新分布、跳过样本或重加权 loss。 |
| B13 | 复用 `utils/validation.py` 的 `positive_int` 校验真正要求正整数的累积、步数、日志/保存间隔，覆盖 CLI 与直接 Trainer 入口。区分已有允许 0 的“禁用”字段，别一刀切。禁止先 `int()` 截断再验证。 | bool、小数、NaN/Inf、非法 0 尽早失败；合法 10 个 micro-batch、G=4 仍是 4/4/2，并按实际尾组缩放。 |
| B09 | `datasets/augmentation.py` 定值区间使用该值；brightness/hue=0、contrast/saturation=1 才是恒等。保留概率、顺序、HSV 和 clamp 语义。 | 固定非恒等值确实改变颜色；恒等值保持；默认随机范围不被重定义。 |
| A02 | `backbone/attention.py` 保留全无效 mask 的安全 softmax，并在输出 projection/dropout 后清零全无效 context 对应输出，避免 bias 重新注入。 | 有限输入、非零 proj bias、fused/manual、全无效/部分有效的输出与梯度；不把未接入 mask 的默认 ManiFlow 说成已失效。 |
| A03 | `rgb/geometry_processor.py` 有效深度包含 isfinite；无效深度投影前用 where 替换，变换后显式清零；patch 有效须 count>0 且 ratio 达标。标定非有限/明显非法应报错，保留 valid_mask。 | NaN/Inf、全无效、`min_valid_ratio=0`、合法几何；零乘 NaN 不作为处理方法。仅修局部合同，不把几何分支接入现有模型。 |

## 6. W04：RGB dtype 与数值/效率边界

位置：`dexmani_policy/agents/obs_encoder/rgb/{base,dino,clip,siglip,image_processor,utils,r3m}.py`、`dexmani_policy/training/ema_model.py`。

- **B02（FIX）**：局部 projection helper 将特征转为非 Identity projection 的权重 dtype 再投影；patch、CLS、pooler 等所有有效分支统一调用。Identity 不新增 cast。不要只修 patch 路径，也不能要求所有调用方靠 autocast 避免崩溃。验收自定义 out_dim 的 DINO/CLIP/SigLIP 支持模式、无 autocast 推理和 raw/EMA 恢复；原默认 Identity 路径保持。缺权重时小型真实 projection 检查只代表局部合同，不能冒充完整 backbone 通过。
- **R04（CHECK，数值诊断优先）**：列出当前真实 RGB 配方中 trainable/frozen/LoRA/projection/EMA 的 dtype，检查当前 PEFT 行为及代码强制回 BF16 的影响。在目标环境用固定短 batch 测梯度、参数/EMA 更新非零比例、loss、显存及存取恢复。若证据支持，优先做**新实验的局部 LoRA FP32 + 冻结 backbone BF16 + autocast** 配方；不要顺带把 full backbone 全转 FP32。任何采用的精度选择须保存到构造/恢复配置，旧配置缺字段时按旧语义重建，严格续训不能悄悄切 dtype。没有充分数值/资源证据就保持现状并写明诊断结论；不以“社区通常如此”宣布已有实验无效。
- **E04（CHECK）**：优先测已有 foreach EMA，不另写 EMA 实现。不先缓存参数对或跳过 frozen copy；`requires_grad=False` 不证明 tensor/buffer 永不变。只有证明静态且收益明显时才减少复制；时钟、动态 buffer、teacher eval 语义保留。比较多步 loop/foreach 及中断恢复的结果，报告整步收益而非只报 kernel 数。
- **E05（CHECK）**：首先考虑 ImageProcessor 按当前 device/dtype 缓存 mean/std 两个常量；它目前不是 nn.Module，不为常量重构继承结构。范围检查只能在明确的数据合同边界消除重复：uint8 可依赖类型范围；外部任意 float 输入仍校验，检查首个 batch 不能证明所有后续数据合法。内部 float 来源无法证明时保留检查。uint8 传输是单独优化，不与 crop/resize/augment 顺序或 float 标度变更混在一起。验证数值、同步次数及端到端收益后再采用。
- **A09（KEEP）**：R3M 的 BN→GN 是既定适配，不自动恢复 BN。论文中披露，不能据此直接判 bug。

## 7. W05：训练状态、恢复与日志

位置：`dexmani_policy/train.py`、`train_ddp.py`、`training/{build_utils,trainer,resume,checkpoint,ema_model}.py` 和现有 `tests/`。

- **S08（FIX）**：每进程只读取一次完整 TrainCheckpoint，共用对象提取 normalizer 并调用既有恢复逻辑；恢复完成后释放不再需要的 CPU 载荷，不把完整对象挂到 Trainer 跨训练持有。不把全 optimizer checkpoint 改成 rank 0 object broadcast。还原 optimizer、scheduler、EMA updater、RNG 和 sampler 消费游标的顺序不变；不因函数签名简化去掉校验。
- **B12（FIX）**：里程碑预先按百分比 p 用整数式 `(p * total_steps + 99) // 100` 映射到 step，碰撞保留最大百分比；最后 step 必有 100pct，每 step 最多保存一份大 checkpoint。使用 global_step 判断已过里程碑，不扫描目录猜进度。恢复跨过的标签不补写到旧 run；历史已完成短 run 不自动改写。selector 已按实际发现文件工作，核验 `100pct` 查询和候选发现能接纳去重后的集合，不为凑五个候选重复同一权重。
- **E01（CHECK）**：日志诊断先累积 detached device tensor，到真实日志边界统一转 host；DQ 的额外诊断按日志需要计算。保持当前平均口径和尾组处理，不保留计算图。不删除 loss/gradient finite、总范数和全 rank 协调，不能用日志“懒求值”延迟必要的失败处置。实测日志同步占比后改动，低成本标量已被安全检查需要时不造第二套绕过路径。
- **R08（FIX：验证缺口）**：扩充现有 smoke/定向测试，验证实际受影响策略的一批 forward/backward/predict、raw/EMA 保存恢复，以及当前生产 DDP flags。没有指定本轮论文策略时，按实际修改到的默认配置选最小代表，不扩展成所有 encoder×decoder×模式矩阵。已有真实 RTC CPU input-VJP 测试保留；不能再称仓库没有真实模型测试。
- **D03/D04/D05/D09（KEEP）**：teacher 始终 eval；EMA 权重与更新时钟共同恢复；optimizer/scheduler/EMA 按 optimizer step 前进；实际尾组缩放；ManiFlow B=1 的 flow-only 和其余样本的余数分配保持。不把上游独立 floor 分流、首 micro-batch 提前 update、按 micro-batch EMA 或 epoch 后 teacher.train() 移植回来。

验收：总步数 1/2/3/4/5 和常规里程碑；连续训练 vs 中断恢复的小模型；原始样本顺序、在线/EMA 权重、Adam/LR/时钟/游标一致；每进程 checkpoint load 次数为 1。精确恢复检查使用可控随机性，不承诺任意 persistent-worker 随机增强都能逐位恢复。DDP 用生产 `static_graph` 等实际组合做必要的最短验证，并检查一 rank 失败不会留下其它 rank 挂起。

## 8. W06：R3D 点云

位置：`dexmani_policy/agents/obs_encoder/pointcloud/{uni3d,ops,r3d_obs_encoder}.py`。

- **B07（FIX）**：PatchDropout 输出保留索引，token、center 和输出位置编码使用相同索引及顺序。默认 dropout=0 保持现有直接路径；eval 不误用训练抽样。R3D 消费者当前要求 pointsam token 协议，明确拒绝不支持的 cls/max_pooling 组合，不为修复而扩展新输出协议。若底层 Uni3D 仍有其它合法独立调用，不无故删除它们。
- **E09（CHECK，优先测量）**：KNNGrouper 在 no_grad 下只用索引，可复用已有 PyTorch3D/`ops.knn_point`，避免完整 cdist 矩阵；不新增 FAISS 等依赖。核对 `N >= num_groups/group_size` 等实际固定形状要求，不能让 helper 的 `min(K,N)` 静默造成后续 reshape 失配。无 tie 比邻居集合及 patch 输出；有重复 XYZ、不同 RGB/特征时单独评估 tie 改变。接受条件包含真实显存/整步时延收益及数值结果；不承诺两个后端在 tie 时逐位一致。
- **E07（CHECK）**：逐云 first-point dropout 可批量生成 ratio/mask 并用三参数 where 替换**整行**。注入同一 ratio/mask 应精确对齐旧循环；还需核对分布。不得改成补零、删点或改变默认概率。新随机调用顺序不保证旧 seed 的逐位续训；若采用，只用于明确记录的新配方/新 run，不能以“分布相同”抹掉恢复边界。
- **A01（CHECK）**：只对零噪声 FPS 路径评估原生 `random_start_point` 替换 clone/swap/remap。保留所需输出 shuffle、评测确定性和从原始点 gather 的语义；非零噪声、输入顺序和 tie 要单独核对。没有净收益则不改。
- **A06（USE）**：未使用 timm classifier head 不等于生产 DDP 必崩。若整理可训练参数确有用途，最小方案为冻结并保留 state keys，不做权重迁移系统。检查 strict load、optimizer 组和生产 DDP；否则保留即可。

验收：B07 用带可识别 center 的数据确认“同一索引”而不只是 shape 相同；默认输出不回归。性能比较使用实际 B×观测帧、N、分组数、dtype，注明训练/评测模式及库版本；理论中间张量大小不是实测峰值显存。

## 9. W07：RTC 的真实分支与性能

位置：`dexmani_policy/deployment/runtime.py`、`dexmani_policy/agents/action_decoders/rtc.py`、`tests/test_policy_rtc.py`。

- **B11（FIX）**：最新版本已把正延迟传给 warmup，剩余问题是用 `rtc_delay` 真值决定是否准备 prefix，导致 delay=0 时未热身实际引导。按已配置的 RTC 模式和 `guidance_cap>0` 决定引导预热；delay=0 也准备非空 prefix 并进入实际 input-VJP。优先利用现有运行状态，不增加与 mode/cap 不一致的冗余开关。sync/async/cap=0 保持普通路径，所有 delay 边界继续校验。
- **E08（CHECK）**：每次采样准备 CPU timestep 整数、模型使用的 device timestep 和同 dtype 系数，减少每步 GPU→Python 标量及重复迁移。按 prefix 的已知约束区间构造 residual，保留 detach 与 VJP。不能用 `0 * (target-f)` 掩盖未约束 NaN；如果约束区间存在实际为零的权重，也应维持原“零权重不参与误差运算”的语义。预计算保持原 cast/sqrt 顺序、当前 scheduler 的步进/裁剪/epsilon、eta=0 和最后 clamp，不以 `inference_mode` 禁掉 input-VJP。

验收：实际 LoadedPolicy 的 delay=0/正值/cap=0 调用分支；相同 cond/noise/prefix 的逐步 output、VJP 和最终动作；sample/epsilon/v_prediction、aux/控制维边界。源码或局部算术不能证明 GPU 首次时延，目标 GPU 上分别报告首次和稳态测量；E08 无充分数值及性能证据就保留原实现。

## 10. W08：论文评测与代码追溯

位置：`dexmani_policy/evaluation/protocol.py`、`select_best_ckpt.py`、`eval_best_ckpt.py`、`record_demo.py`、`agents/loader.py`、`training/workspace.py`、`scripts/eval/eval_pipeline.sh`、`scripts/remote/sync_code.sh`。

### R03（FIX：建立显式论文协议，保留历史语义）

当前已排除实际选点 seed 并记录实际测试分母；问题是默认 eval seed 依赖训练 seed，且平局追加样本会改变测试池。不要重复修已经存在的防泄漏逻辑。

新增一个简单、显式的 task→seed 清单输入，包含完整 selection 池、预留 tie-break 池、固定 test 池及必要任务/池身份；selection 与 eval 复用已有 protocol helper 读取和校验。清单独立于训练 RNG，在选点之前确定；无论是否触发平局，预留池不进入测试。按真实 `(task, seed)` 检查重复与跨角色相交，核对实际任务映射和池身份，不因两个不同任务使用同一个整数 seed 就误报泄漏。保存实际清单或其不可变副本到 selection/eval 证据。

新论文比较显式使用同一清单；旧 run 未给清单时仍按原协议读取并注明 legacy，不能把历史 75/70 个测试样本重新标成统一 100 个。只有 100 个可用 seed 而选择+预留占 30 个时，独立测试最多 70 个；需要 100 个测试必须补足真实独立 seed。不要编造生产 seed；使用现有有效池生成/验证，小 fixture 只用于测试。

验收：改变训练 seed、模型、是否平局后 test task/seed 列表仍相同；真实映射无交集；数量不足清楚拒绝/按显式协议处理，不悄悄补选点样本。成功率保持实际分母和已有统计口径。

### S03（FIX）：三阶段固定同一次选择结果

selector 成功时直接交出该次不可变 summary/selection_id，例如写入调用方指定的唯一结果文件；失败不得交出旧 best 当成功。eval/demo 显式接收这份记录，并固定具体不可变 checkpoint、raw/EMA、NFE、selection 及 seed 协议。不要在 selector 退出后再读可变 `best_ckpt.json` 猜“自己的结果”。`best` 保留便捷入口，既有单次调用内固定结果的机制继续使用；保留三个进程的隔离。

权重引用不能停留在 `latest` 等可变别名；用实际 milestone 文件与 global_step/来源身份，并沿用现有校验。无需为每个预测计算权重 hash，也不重写整个选点平台。验收在三阶段之间模拟另一成功 selector 更新 best，当前流水线仍用原结果；缺失/身份不一致报错，不回退到其它模型。

### R06（FIX）：保存实际训练源码，而非只记评测 commit

复用 workspace 的产物保存点，记录训练开始时可获得的 commit、dirty 状态、代码内容身份和关键依赖版本；评测另记评测代码身份。每 run 保存一次小型源码归档及文件清单，限仓库运行源码、配置和构建/依赖文件，纳入这些源码位置的实际未提交/未跟踪文件，排除数据、权重、实验、缓存、凭据和虚拟环境。使用标准库和现有保存机制即可，不建立 artifact 平台或另一套同步工具。

远端没有 `.git` 时，以实际归档内容为准，commit 可写 unknown；不得拿本地 commit 无条件冒充远端执行版本。归档失败明确报告，不能伪称已可追溯。保存源码不会自动隔离后续 lazy import，运行中的源码目录不要原位同步；只在使用文档写明这一边界，不强制实现每 run checkout/容器调度。源码元数据不直接加入严格 resume 相等合同；数据/模型/优化器等现有合同保留。

验收：clean、dirty、未跟踪源码和无 `.git` 的小目录都能核对归档与文件身份；新增纯文档不应无故改变训练恢复合同；不会打包 robot_data/experiments/权重或环境文件。历史 run 无训练源码身份时如实标 unknown，不以当前代码补写历史。

### C03 / D08（DOC）

记录实际动作表示/单位、感知、容量、训练预算、NFE、query/replan interval、执行窗口、ensemble、raw/EMA 与 seed 协议。完整系统比较允许各方法原配方，但不能由此单独归因于某个机制；机制消融须固定有关变量。DQ 官方 query 间隔参数不是 NFE；解码后重叠 chunk 平均可离开码本，默认无重叠条件下不自动发生。论文协议明确是否允许该平滑，不一律改所有方法的 NFE 或动作空间。

## 11. W09：多任务机制

位置：`dexmani_policy/datasets/multi_task_dataset.py`、`resumable_sampler.py`、`training/resume.py`、`agents/core/multi_task.py`、`agents/obs_encoder/text/clip.py`、`deployment/runtime.py`。

- **S01（CHECK）**：先确认每样本 Manager 访问在实际数据开销中的比例。有净收益才将每 epoch `(task_idx, local_idx)` 映射移到父进程的现有 sampler 流程，让 Dataset 只取样。必须保留原配额、无放回轮次、全局随机顺序、DistributedSampler rank 分片/补齐/drop_last 及消费游标；不能双重 shuffle，不能直接换 WeightedRandomSampler。证明旧新全局/各 rank 序列及恢复边界一致后再删除 Manager；persistent worker、validation/smoke 的调用一起闭合。不为优化新增 sampler 抽象层。
- **S05（USE）**：固定任务集合的实际发布需求出现时，可导出 task→embedding 表并在未知任务明确报错；此时才允许删除产物中的 encoder/tokenizer。当前训练和开放语言输入保留现有冻结 encoder 与缓存 fallback。查表不是开放语言泛化。
- **A04（KEEP）**：当前 Real runtime 只支持具备必要 metadata 的策略，并未承诺任意 MultiTask/仿真 checkpoint 都能部署。保持明确入口报错和支持说明；本轮不自动增加 task_text/传感器接入，不宣称外部 dexmani_real 或真机已验证。

## 12. W10：发布与清理的条件范围

- **B04（USE）**：当前 README 的 editable 工作流不受 wheel 缺 YAML 的同样影响。需要 wheel 分发时，在现有 setuptools 中补 configs package data，隔离安装验证实际读取配置；不换包管理框架。
- **R05（USE）**：实际需要完整 checkpoint 的离线 RGB/文本恢复时，沿本地 Uni3D 的初始化/完整恢复分离方式，从保存的 HF 配置构造骨架后 strict load，保存必要 tokenizer/预处理信息，不先下载同一预训练权重。不要扩展所有未用 encoder，不允许缺依赖静默变随机初始化。用清空缓存的断网目标路径验证；旧产物缺构造信息时说明边界，不猜模型结构。
- **S04（USE）**：只有发布体积/冷启动确成需求，才导出选定 raw 或 EMA 的轻量推理产物，包含构造配置、来源及必要资源。normalizer/codebook 已在模型 state 中，不重复外置。训练恢复继续使用完整 TrainCheckpoint；推理产物不能伪装可 resume。若实施，复用现有 loader，明确区分用途，验证固定输入/noise 的预测及大小/加载时间。不要为了先删 CPU 载荷就同时开辟 R05/S04 两个未完成新格式。
- **S02（KEEP）**：保留单机 mp.spawn；它是官方支持方式，不因“主流”之名改 torchrun。有多机/集群需求再单独决定。
- **S07（USE）**：`agents/obs_encoder/text/t5.py`、`agents/action_decoders/backbone/ditx_rms.py`、`agents/obs_encoder/plugins/token_compressor.py` 暂无已发现的活跃调用。先搜索 import、Hydra `_target_`、动态调用及已有外部使用线索，确认可删才移除；未知外部 API 不臆断不存在。Git 保留历史，不另建“待用插件”目录。不以删除行数声称提速；验证 import/配置即可，不写镜像实现的测试。
- **A05（FIX）**：`scripts/remote/stop_remote.sh` 在远端先检查 tmux 查询状态，再解析输出；区别正常无 session、tmux 命令故障与 SSH 失败，避免 `tmux list-sessions | cut` 掩盖退出码。不新增调度器。用本地受控命令/返回码 fixture 验证，不连接真实服务器执行 stop。

## 13. W11：参考项目和研究机制的保留边界

以下 DOC 项写入最终报告的实际配方表；“以后做消融”不等于现在自动跑训练。以下 DEFER/KEEP 项不能成为强制改模型的理由。

| 编号 | 已核对的差异/边界 | 本轮最佳处置 |
|---|---|---|
| C01 | 本地 SAT 使用 PointNext patch/learned global token、有序状态 MLP、不同 token 数/宽度；不能等同官方完整结构。EJC 字段求和思想并非因此错误。 | DOC：说明 structured-action 思想的适配。若后续证明 EJC 效果，在相同感知/动作/容量下只改 EJC；不先重写整套 SAT。 |
| C02 | 本地 DQ 是 iDP3＋状态/多帧观测适配，非官方 RISE 感知；12 维手部使用 16 原型是否不足，源码不能判定。 | DOC：先修 B10/B03；以后固定感知、动作与预算，对比连续手部和量化手部，结合 held-out 重建误差、码字 usage/跳变和闭环成功率判断。不要自动扩大码本。 |
| C04 | SAT 时间分布、ManiFlow slot PE/通用或 Dex 配方、DP3 宽度/模态、R3D 评测 FPS 等存在明确适配。上游当前未锁定依赖不能代表论文历史环境。 | DOC：逐项披露实际配置，不一律回退。只有支撑论文因果主张的差异才设计单变量对照。 |
| D07 | DQ VQ 的 kmeans/dead code/实际 hidden 层数、policy AdamW/EMA 与官方有差异；VQ 字典 EMA 与 policy EMA 是不同机制。 | DOC：固定活跃配方再比较。官方未使用的内部 Adam 不是训练循环优化器；不要据参数名误判层数。 |
| R02 | 本地 SAT 跨帧按 patch 序号拼接，上游也逐帧采样；不保证物理点对应，但不能从中直接推出无效。 | DEFER：共享锚点/匹配是新方法，影响可见性和覆盖，不作为普通修复。 |
| R07 | ManiFlow absolute teacher target 可超过 1，上游有同类写法；默认 relative 不触发该反例。 | DEFER：使用 absolute 研究分支时先定义外推与一致性目标；不能只 clamp target 而保留不匹配分母。 |
| S06 | SAT 动作与 EJC 同步 shuffle 在特定条件下置换等变；NumPy toy 不覆盖原模型/dropout/梯度。 | CHECK：只有原实现 eval、反传、位置/掩码与随机分布检查通过且有净收益才删 shuffle/inverse；否则保留，不新增永久兼容开关。 |
| D01 | 官方无 joint-ID 的共享标量状态 attention 对关节置换不敏感。 | KEEP：本地有序 MLP 对固定机器人保留身份；不换回该上游表示。变长跨机器人状态是另一个研究任务。 |
| D02 | 官方构造 padding mask 却未传入状态 attention，数值 −100 不是 mask。 | KEEP：本地无需此 padding，不引入该路径。未来独立运行官方基线时另做最小修补并披露。 |
| A07 | ManiFlow KV cache 已限定单次观测的采样过程；SAT/R3D 条件含时间因素。 | KEEP：保留现有缓存生命周期和 student 合并前向，不跨观测复用，不无条件把缓存移到其它模型。 |

正式实验表还应写明本轮代码修复前后版本；B10 的样本/坐标修复等可能要求重训待比较模型。已有历史分数不会因代码修复自动有效，也不必无证据全部作废，应按是否触发相应差异判断。

## 14. 验证与完成标准

### 验证顺序

先复用现有测试，补能复现失败且调用真实实现的最少反例。不要把方案中的 NumPy/公式验证当成修改后 PyTorch 模型已通过，也不要为了简单改动写大型参数矩阵或全套新框架。

当前基线确有以下入口；执行前核对当前源码。按改动面选取，不要求每一批重复运行全部命令：

```bash
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat maniflow r3d multitask_dit
python -m pytest -q tests/test_policy_windows.py tests/test_policy_rtc.py tests/test_infra_codebook.py
python -m pytest -q tests/test_infra_training.py tests/test_infra_resume.py tests/test_infra_evaluation.py tests/test_infra_launch.py
bash -n scripts/training/train_vq_hand.sh scripts/eval/eval_pipeline.sh scripts/remote/stop_remote.sh
git diff --check
```

`--config-only` 仍需要相关 Python import，不代表无需依赖；它不接受任意 Hydra override，组合检查用现有 compose 能力。新增定向测试后执行实际新文件/用例。CUDA 条件具备时再使用 `tests/test_infra_cuda.py` 及必要的实际模型验证；读取其中 skip/flags，不能把跳过统计当 GPU PASS。

涉及真实模型路径时，在数据、权重和设备具备的环境运行相应 `python dexmani_policy/smoke_test.py <config_name>`，并补它未覆盖的 raw/EMA/恢复或生产 DDP 风险。缺数据/GPU/权重/仿真时标 **NOT VERIFIED**，写明缺什么和复验命令；继续其余工作，不改默认模型绕过限制。

### 性能项目的接受条件

- 先有相同 shape、dtype、配置和输入的基线，warmup 后重复测量；使用正确 CUDA 同步/计时，报告统计而非单次数字。
- 分别记录所改部分和整次训练 step/预测的时延、CPU RSS/峰值 CUDA 内存等实际相关指标。不要因为 kernel 更少、理论中间量更小就宣称端到端已提速。
- 行为等价的改动做数值/恢复对照；改变 RNG 消耗但只保证分布、或改变等距邻居选择的改动单独披露，不宣称旧实验逐位可续训。需要严格复现旧随机流的 run 使用保存的原代码版本；不得仅凭 resume 结构校验通过就承诺等价。若本轮必须保持该路径精确续训且无法兼顾，暂缓优化，不造长期双路径。无收益、精度不符或明显增加复杂度时撤回候选实现，不留下永久开关。
- E03/E06/S08 是明确消除整块中间量/重复读取的局部简化，可以用小型定向证据完成实现；实际大模型速度仍需真实测量，不能造结论。

### 最终交付

创建 `docs/review_remediation_report.md`（这是执行本任务后的报告，不要预填成功），至少包括：

1. 实际起始 HEAD、工作区边界、变更摘要及与本文方案不同的必要调整。
2. 第 15 节全部 62 个编号的最终状态：`IMPLEMENTED / ALREADY_FIXED / PRESERVED / DEFERRED / NOT_APPLICABLE / BLOCKED`，并给实现/调用路径或暂缓原因。FIX 未修复不能写成已完成；CHECK 无环境或证据不足可以明确暂缓。研究项处置完成不等于研究假设被证实。
3. 真实执行的命令、`PASS / FAIL / NOT VERIFIED`、关键结果及环境；实现状态与验证状态分开。失败未解决、环境未验收的部分如实列出。
4. 新的 VQ 训练/导出/加载操作、输出冲突行为、配置错误提示、论文 seed 清单使用、selection 交接及代码追溯的最短示例；只为真实改变的用户入口更新 README/现有文档。
5. 性能测量和采用/撤回理由；需要新 run、重训或保持旧版本恢复的具体边界；参考方法配方表与暂缓实验，不填写虚构成功率。

结束前逐条检查本文的 FIX、所有编号覆盖、diff、路径和命令。能够继续完成的授权工作不要留在“建议下一步”。只有 FIX 均已修复或证实已修复、当前可运行的必要检查通过、其余处置与未验证边界完整时，才可报告本轮实现通过；环境受阻应报告部分完成及阻塞，不使用“全部验证通过”。

## 15. 全部 62 项处置索引

每个编号只在下面的覆盖表中出现一次；编号链接指向审查固定版本的主要证据，正文给出完整方案及约束。表中“保留”也属于完成处置，不生成重复修复提交。

| 编号 / 固定证据 | 批次 | 优先级 | 处置 | 复审后的结论 |
|---|---|---|---|---|
| [B10](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/datasets/base_dataset.py#L303-L318) | W01 | P1 | FIX | 复用有效训练源行与hand affine；内部VQ holdout不改变policy预算 |
| [R01](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/datasets/base_dataset.py#L90-L147) | W01 | — | KEEP | policy训练行统计已修正，保持现有角色合同 |
| [E03](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/datasets/base_dataset.py#L303-L318) | W01 | P2 | FIX | 分块索引与单次流式统计，去掉list/全量拼接 |
| [E02](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/datasets/replay_buffer.py#L45-L76) | W01 | P2 | CHECK | 先量化重复fit成本，再决定rank0拟合与小状态广播 |
| [B01](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/scripts/training/train_vq_hand.sh#L29-L37) | W02 | P1 | FIX | 独立VQ run及旧式产物冲突保护；不新增VQ恢复系统 |
| [B03](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/scripts/training/train_vq_hand.py#L311-L324) | W02 | P1 | FIX | 预先固定选点指标，非有限值不能回退混排 |
| [E06](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/vq_hand/codebook_manager.py#L605-L622) | W02 | P2 | FIX | usage分块归约，保留距离和tie规则 |
| [A08](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/vq_hand/vqvae.py) | W02 | — | KEEP | 保留加权RVQ及两种usage的不同含义 |
| [U01](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_vqvae.py#L88-L114) | W02 | — | KEEP | 本地已按真实learned weights导出，不引入上游等权假设 |
| [U02](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_rise_vae_2cam.py#L230-L240) | W02 | — | KEEP | 本地码本已缓存，不重复读取NPZ |
| [D06](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/train_dqrise.py#L155-L163) | W02 | — | KEEP | 本地half-up/clamp已规避上游截断错误 |
| [B06](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/pointcloud/registry.py#L48-L131) | W03 | P1 | FIX | 配置键必须生效或报错；逐child核对动作合同 |
| [B05](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/action_decoders/backbone/unet1d.py#L10-L36) | W03 | P2 | FIX | 补n_groups和当前UNet的明确形状约束 |
| [B08](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/action_decoders/time_sampler.py#L40-L57) | W03 | P2 | FIX | 拒绝过小discrete_pow子批次，不默认更换分布 |
| [B09](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/datasets/augmentation.py#L46-L62) | W03 | P3 | FIX | 定值非恒等颜色变换必须执行 |
| [B13](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/build_utils.py#L222-L250) | W03 | P2 | FIX | 在实际入口严格校验整数控制量 |
| [A02](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/action_decoders/backbone/attention.py#L43-L102) | W03 | P3 | FIX | 全无效attention在最终投影后清零 |
| [A03](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/rgb/geometry_processor.py#L91-L106) | W03 | P3 | FIX | 无效深度用where处理，空patch不能标有效 |
| [B02](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/rgb/dino.py#L25-L48) | W04 | P2 | FIX | 所有ViT投影分支显式匹配输入和权重dtype |
| [R04](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/rgb/base.py#L67-L90) | W04 | P1 | CHECK | 诊断BF16参数/EMA更新；新精度配方须证据与恢复边界 |
| [E04](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/ema_model.py#L17-L36) | W04 | P2 | CHECK | 先测已有foreach，勿预先加入参数缓存或删buffer更新 |
| [E05](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/rgb/utils.py#L35-L54) | W04 | P2 | CHECK | 常量缓存与可证明的边界校验；不删除外部float检查 |
| [A09](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/rgb/r3m.py) | W04 | — | KEEP | R3M的GN是既定适配，披露即可 |
| [S08](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/build_utils.py#L112-L132) | W05 | P2 | FIX | 每进程读取一次checkpoint并及时释放恢复载荷 |
| [D03](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/model/diffusion/ema_model.py#L30-L42) | W05 | — | KEEP | 保留本地EMA teacher的eval模式 |
| [D04](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/workspace/train_maniflow_dex_workspace.py#L105-L152) | W05 | — | KEEP | 保留本地EMA权重与更新时钟共同恢复 |
| [D05](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/train.py#L217-L230) | W05 | — | KEEP | 保留optimizer-step累积边界、尾组和EMA语义 |
| [D09](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py#L506-L552) | W05 | — | KEEP | 保留ManiFlow小batch分流与样本守恒 |
| [B12](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/trainer.py#L27-L47) | W05 | P2 | FIX | 合并碰撞里程碑，确保最终100pct且不重复存大权重 |
| [R08](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/tests/test_policy_rtc.py) | W05 | P2 | FIX | 补真实受影响策略及生产DDP的最小验证，诚实记录环境缺口 |
| [E01](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/trainer.py#L173-L186) | W05 | P2 | CHECK | 日志边界再转host；必要非有限检查与rank协调保留 |
| [B07](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/pointcloud/uni3d.py#L151-L178) | W06 | P2 | FIX | patch dropout同步选择token、center与位置编码 |
| [E09](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/pointcloud/uni3d.py#L19-L26) | W06 | P2 | CHECK | 评估已有KNN后端，核对固定K、tie、显存及端到端收益 |
| [E07](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/pointcloud/uni3d.py#L29-L44) | W06 | P2 | CHECK | 批量first-point dropout；披露随机流变化 |
| [A01](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/pointcloud/ops.py#L73-L128) | W06 | P3 | CHECK | 原生FPS仅在语义边界和收益证实后替换 |
| [A06](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/pointcloud/uni3d.py) | W06 | P3 | USE | 需要整理时冻结未用head；不是默认DDP崩溃修复 |
| [B11](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/deployment/runtime.py#L159-L171) | W07 | P2 | FIX | 零延迟RTC也热身真实引导分支 |
| [E08](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/action_decoders/rtc.py#L40-L76) | W07 | P2 | CHECK | 减少RTC标量同步与索引开销，保留VJP及scheduler语义 |
| [R03](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/evaluation/protocol.py#L27-L36) | W08 | P1 | FIX | 显式固定论文selection/tie/test清单，保留旧协议可读 |
| [R06](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/scripts/remote/sync_code.sh#L33-L54) | W08 | P1 | FIX | 记录实际训练源码、身份与依赖，不建设追踪平台 |
| [S03](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/scripts/eval/eval_pipeline.sh#L81-L89) | W08 | P2 | FIX | 流水线交接selector自己的不可变结果，不重新猜best |
| [C03](https://github.com/haoyangzhanglab/dexmani_policy/commit/650f1e3cfc63c64678b5bb69e7258b548800d053) | W08 | P1 | DOC | 披露动作、感知、容量和预算，区分系统比较与机制归因 |
| [D08](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_rise_vae_2cam.py#L191-L212) | W08 | P1 | DOC | 分开NFE、查询间隔、执行窗口和解码后ensemble |
| [S01](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/datasets/multi_task_dataset.py#L79-L105) | W09 | P2 | CHECK | 只有序列/恢复等价且有收益才移除每样本Manager访问 |
| [S05](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/core/multi_task.py#L126-L149) | W09 | P3 | USE | 仅固定任务发布可用embedding表替代语言encoder |
| [A04](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/deployment/runtime.py) | W09 | — | KEEP | 保持Real runtime明确支持边界，不扩成任意策略部署 |
| [B04](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/pyproject.toml#L37-L39) | W10 | P3 | USE | 需要wheel时补YAML package data，不迁移构建框架 |
| [R05](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/loader.py#L108-L119) | W10 | P2 | USE | 实际离线恢复需求下保存HF构造资源并strict load |
| [S04](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/checkpoint.py#L62-L107) | W10 | P3 | USE | 实际发布需求下才导出轻量推理产物 |
| [S02](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/train_ddp.py#L44-L72) | W10 | — | KEEP | 单机mp.spawn保留，非必须迁移torchrun |
| [S07](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/obs_encoder/text/t5.py) | W10 | P3 | USE | 确认无活跃/外部用途后删预留模块，不按删行数称提速 |
| [A05](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/scripts/remote/stop_remote.sh) | W10 | P3 | FIX | 保留远端tmux和SSH真实退出状态，用本地fixture验证 |
| [C01](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/policy/sat.py#L90-L112) | W11 | P1 | DOC | SAT适配不等同完整官方复现；暂不重写感知 |
| [C02](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/policy.py#L12-L44) | W11 | P1 | DOC | DQ感知适配与码本容量通过后续受控实验判断 |
| [C04](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/policy/sat.py#L90-L112) | W11 | P1 | DOC | 分项披露配方差异，不把上游默认统一强制回退 |
| [D07](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/vqvae_rise/vector_quantize_pytorch/vector_quantize_pytorch.py#L686-L718) | W11 | P1 | DOC | 区分DQ的VQ配方、policy优化器及online/EMA |
| [R02](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/core/sat.py#L78-L131) | W11 | — | DEFER | 跨帧物理对应是新研究选择，暂不做共享锚点 |
| [R07](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/action_decoders/consistency_flow.py#L113-L164) | W11 | — | DEFER | absolute时间外推需重审目标，不做孤立clamp |
| [S06](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/action_decoders/backbone/sat.py#L421-L472) | W11 | P3 | CHECK | 原SAT等变性/梯度/随机分布与收益均支持才删shuffle |
| [D01](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/vision/obs_tokenizer.py#L18-L75) | W11 | — | KEEP | 本地有序状态MLP保留关节身份 |
| [D02](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/vision/obs_tokenizer.py#L641-L661) | W11 | — | KEEP | 本地不引入官方无效padding mask路径 |
| [A07](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/action_decoders/consistency_flow.py) | W11 | — | KEEP | 保留ManiFlow单次采样内KV缓存，禁止跨观测复用 |

## 16. 固定参考版本

以下版本在任务书发布前重新核对。它们用于解释机制和边界，不要求执行者为了本文升级本仓库依赖或复制上游全部实现。需要移植时阅读具体调用链并遵守原项目许可证。

| 项目 | 固定提交 | 本轮使用原则 |
|---|---|---|
| SAT | [`cd7c0a8877d6090a9a85ebee0ceca961830b3654`](https://github.com/XiaohanLei/SAT/tree/cd7c0a8877d6090a9a85ebee0ceca961830b3654) | 区分 structured-action 思想与完整感知/状态/训练配方；不引入状态 padding/身份缺陷。 |
| ManiFlow_Policy | [`ef2f116f1f90163ed36e657b8c5503740bb468af`](https://github.com/geyan21/ManiFlow_Policy/tree/ef2f116f1f90163ed36e657b8c5503740bb468af) | 明确通用/Dex 配方，保留本地 teacher eval、batch 分配和 KV 生命周期。 |
| 3D-Diffusion-Policy | [`47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97) | 借鉴既有 DDIM/UNet/数据机制，披露宽度/模态，不退回累积和 EMA 恢复缺口。 |
| R3D-Policy | [`e637c0148376ddc4b5e667fa8f8e108cb8ff7a85`](https://github.com/Wushr-Lance/R3D-Policy/tree/e637c0148376ddc4b5e667fa8f8e108cb8ff7a85) | 保持单向机制；点云 grouping/dropout/FPS 改动分别验收，避免把不同 tie/RNG 当等价。 |
| DQ-RISE | [`2889d27fce823288e8dd12ec6e91506b52ed086a`](https://github.com/rise-policy/DQ-RISE/tree/2889d27fce823288e8dd12ec6e91506b52ed086a) | 保留量化手部思想和本地 learned-weight/rounding/cache 修正，分别控制感知、VQ 与 policy EMA。 |

最终原则：**先让一个真实工作流正确闭合，再删除重复工作；未证明错误的研究选择不以工程修复之名替换，未测得收益的优化不包装成性能结论。**
