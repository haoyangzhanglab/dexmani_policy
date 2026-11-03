# 预训练精简与优化任务书

状态：待实现。本文是最近两轮审查的合并决策与实施要求，不表示代码已经修改或验收通过。

审查基线：[`b2fde5c54c6f5fbb0a770bad38b156fd38c67eb2`](https://github.com/haoyangzhanglab/dexmani_policy/commit/b2fde5c54c6f5fbb0a770bad38b156fd38c67eb2)，2026-11-03。实施前核对当前代码；已解决的问题不重复修改，行号变化以函数及调用关系为准。

## 1. 目标与明确授权

将 AFRO、CordViP 观测预训练与 DQ-RISE 手部 VQ 预训练维护为个人研究代码：入口直接、数据流清楚、参数少而有用，训练与导出正确，常见的小数据实验容易运行。

本轮允许直接改变内部接口、删除冗余字段和旧产物格式支持。**不兼容旧 checkpoint、旧配置或旧导出调用方式，不编写迁移器、版本分派、兼容别名或回退加载。** 需要时重新训练、重新导出；不删除用户已有实验文件。“不兼容旧产物”不等于取消新版本自身的保存、加载与续训。

“放宽”落实为：允许 VQ 小数据及尾批训练；不要求数据来源身份、evidence、fingerprint 或多阶段清单一致；信任项目内部已经明确的张量与状态约定。仍阻止实际会导致错误数学结果、错误索引、错误优化器恢复或覆盖已有实验的输入。

不以删行数为验收指标。优先删除不参与研究或运行的机制；有必要的十行直接实现优于引入新抽象的一行调用。

## 2. 两轮审查合并结论

参考版本固定为：

| 项目 | 参考版本 | 主要对照内容 |
| --- | --- | --- |
| [DQ-RISE](https://github.com/rise-policy/DQ-RISE/tree/2889d27fce823288e8dd12ec6e91506b52ed086a) | `2889d27fce823288e8dd12ec6e91506b52ed086a` | residual VQ、EMA、损失、原型解码 |
| [AFRO](https://github.com/KolaKivy/AFRO/tree/283024a49e2c14bc030debf15383a4e7ee73836d) | `283024a49e2c14bc030debf15383a4e7ee73836d` | teacher、双向动力学、VICReg、时间采样 |
| [CordViP](https://github.com/xuanxuanzzzii/cordvip/tree/ad6d441abe7b0480cf79b0311db266965c1313b6) | `ad6d441abe7b0480cf79b0311db266965c1313b6` | 接触标签、交叉注意力、协调损失、归一化 |

两轮未确认默认配置存在阻断训练的算法错误，但发现配置行为错误、日志偏移、无用计算，以及几处可以直接移除的接口与测试复杂度。参考项目中的实现也有局限，不机械回退本地已修正的数值问题。

| 发现 | 判断 | 本轮决定 |
| --- | --- | --- |
| 未传 `--no-wandb` 时把已有 `wandb_enabled=false` 覆盖为 true | 确定的配置错误，fresh/resume 都受影响 | 修复覆盖规则 |
| `scheduler.step()` 后记录 LR | 记录的是下一次 update 的 LR | 记录本次 optimizer 使用的 LR |
| AFRO 在保留 online 计算图后才运行 teacher | 可减少中间激活重叠；不是算法错误 | teacher 前向提前，核对数值与梯度 |
| VQ 的 `act_scale` 默认 1，输入已有唯一 hand affine | 可删除的第二套缩放状态 | 从配置、模型和导出链完整删除 |
| VQ 拒绝不足一个 batch 的训练集，且丢弃尾批 | 对逐样本 MLP/VQ 不必要的限制 | 固定保留尾批，不加配置开关 |
| CordViP 权重为 0 仍运行相应 head 和 loss | 无用计算；`0 * NaN` 仍可能污染总 loss | 用现有权重直接控制分支执行 |
| 导出器构造运行时 codebook 再取回数组保存 | 不必要的对象与复制 | 直接保存数组，返回导出数量 |
| `codebook_chunk_size` 的 requested 参数只有常量调用 | 无实际可变需求 | 收为一个参数的预算函数 |
| 预训练检查绑定触觉配置与 Open3D 预处理 | 职责及依赖混杂 | 迁出触觉检查，独立预处理检查 |
| 测试锁定调用次数、已删除元数据键 | 测试实现形式多于用户行为 | 改为结果与必要不变量检查 |
| 文档重复解释内部生命周期与工程边界 | 阅读成本大于帮助 | 收敛为用法、产物、方法差异及限制 |
| 点序假设、目标信息可被 cross-attention 访问、边界 padding | 研究语义或既有配方问题 | 说明限制，不在精简中改算法 |
| mixed-size 梯度累积、single-token attention、worker 调优 | 涉及额外语义或收益尚未测量 | 本轮不做，理由见第 7 节 |

当前主链没有需要补建的 evidence/contract/source 管理框架，也不应先建立这种框架再声称完成精简。保留源码出处与参考 commit；它们是简短说明，不参与运行。

前两轮的源码对照与局部数值、分支验证不等于完整训练验证。本文列出的 Torch、续训、下游加载和 GPU 验收均须由实施者实际执行；未执行必须注明。

## 3. 保留的最小结构与算法要点

继续沿用现有目录及职责：

- `training/pretrain/obs/common.py` 负责两个观测方法的训练、保存、恢复和导出；方法文件保留各自网络、目标与 teacher 逻辑。
- `training/pretrain/hand_prior/vqvae.py` 保留独立 VQ 训练；`codebook.py` 负责原型构建与 NPZ 导出。
- `EpisodeStore`、`SequenceSampler`、约 48 行的 `ObservationPretrainDataset` 保留。它们已有实际复用，不为“薄”而复制展开。
- 不创建 `BasePretrainer`、注册表、通用 loss router、统一 artifact 对象、配置转换层或新的插件机制。

以下实现不能被当成冗余工程删除：

| 范围 | 必须保持的行为 |
| --- | --- |
| DQ-RISE / VQ | 加权 L1 + commitment；residual 减去 detached quantized；learned softmax layer weights 同时用于训练与导出；当前网络深度含义不变 |
| VQ EMA | 随机初始化时 `cluster_size=1`，`embed_avg=embed*cluster_size`；不能回到零计数配非零累计和。保留 Laplace smoothing、空簇处理以及已有 kmeans/dead-code 实验能力 |
| VQ 距离 | 保留关闭 autocast 的 direct FP32 `cdist`，含 `donot_use_mm_for_euclid_dist`；不能为省代码改成可能消减误差的平方距离展开 |
| VQ 状态 | 保留 initialized buffer、主机端初始化缓存与加载 hook 的一致性，避免每次 forward 的 CUDA 同步 |
| VQ 产物 | 保留独立加载所需的 `model_config`、权重及 hand affine。NPZ 仍仅需 poses、scale、offset；PCA 保持在解码手姿空间中执行 |
| AFRO | teacher 独立随机初始化、无梯度；仍在 backward 后、online optimizer 前更新。双向预测、噪声采样顺序、分别计算 VICReg 25/25/1、AdaLN-Zero、单 token FDM 和 momentum 配方不变 |
| CordViP | 保留交叉注意力与外部 residual、克隆初始化、sample std 及根号外 epsilon 的自定义 LayerNorm；不替换成 `nn.LayerNorm`。默认三个 mean MSE 与 arm7+hand12 适配不变 |
| 归一化 | 新统计只拟合真实 train 源帧；不包括 val、unused、padding 或随机增强。下游继承预训练输入 affine/identity，只拟合新字段；hand affine 继续随码本传递 |
| 导出/验证 | 观测预训练导出最终 online encoder；VQ 有 val 时按 `lookup_mse` 选 best，无 val 时导出 last，不额外扫描训练集选点。eval 不更新 VQ EMA 或 AFRO teacher |
| 数据 | 保留自包含 CordViP 派生 Zarr，保留帧序、点序、contact 对齐及可选 action；不引入 sidecar、来源关联表或数据指纹 |

## 4. 必做任务

下列任务为确定实施范围，不是任选菜单。可按依赖分成少量清楚的提交；不要求逐条建立新文件、测试类或报告。

### T1：修正观测预训练的配置与日志

文件：[`common.py`](../dexmani_policy/training/pretrain/obs/common.py)，必要测试放在现有预训练检查中。

1. `load_run()` 仅在显式传入 `--no-wandb` 时将 `cfg.workspace.wandb_enabled` 设为 false；未传时保留配置或恢复文件中的值。字段不存在时继续使用 `ExperimentLogger` 的现有默认行为，不引入三态配置、`--wandb` 新开关或通用 CLI 合并器。
2. 在 `optimizer.step()` 前读取本次 `optimizer.param_groups[0]["lr"]`，日志记录该值。保留 scheduler 调用顺序、global_step 和总预算语义。
3. fresh 与 resume 使用同一规则；保存的 effective config 反映最终值。

验收：覆盖配置 true/false、无覆盖与显式关闭、fresh/resume；无 flag 时 false 不再变 true。用实际发生变化的 warmup/decay 验证日志等于本次 update 的 LR，不能用恒定 LR 测试掩盖错位。

### T2：AFRO 先运行 teacher 前向

文件：[`afro.py`](../dexmani_policy/training/pretrain/obs/afro.py) 的 `AFRO.forward()`。

在同一 normalized points 上先以 `torch.no_grad()` 运行 teacher，再构建 online encoder 的有梯度计算图。teacher 更新位置保持不动。

当前 PointNet 路径没有 Dropout、BatchNorm 或其它依赖前向顺序的随机/可变状态，这是等价交换的依据；实施时复查该前提。不要引入 copy、checkpointing、缓存、额外模型或 CUDA stream。

验收：相同初始状态、输入与 RNG 下，对照旧顺序，检查 loss、各训练参数梯度、teacher 无梯度及前向后的 RNG。尽量复用 `check_afro_directions`。GPU 可用时测完整 forward/backward 的峰值 allocated memory 与耗时，包含 warmup、测量前同步和峰值重置；记录 batch/frames/points。没有 GPU 时只报告尚未测量，不承诺显存或速度提升比例。

### T3：删除 VQ 的 `act_scale`，保持一个归一化坐标系

文件：[`HandVQVAE`](../dexmani_policy/agents/hand_prior/vqvae.py)、[VQ 训练](../dexmani_policy/training/pretrain/hand_prior/vqvae.py)、[导出](../dexmani_policy/training/pretrain/hand_prior/codebook.py)、[VQ 配置](../dexmani_policy/configs/pretrain/hand_prior/vqvae.yaml) 及真实调用方。

1. 删除构造参数、合法性检查、buffer、`model_config` 字段、配置键和参数传递。
2. `forward()` 直接使用已由 hand normalizer 归一化的输入；移除除法。解码直接返回 decoder 的 normalized hand 输出；移除乘法。
3. 若 `HandVQVAE.decode()` 删除缩放后仅剩对 decoder 的转发，直接删除该包装，并将原型构建调用改为 `model.decoder(latent)`；不要影响运行时 `HandStateCodebook.decode()`。
4. 最终导出顺序为：learned softmax layer weights → decoder → FP32 finite check → clamp `[-1,1]` → 解码手姿 PCA 排序。验证的原型构建可不排序，最近距离不受排序影响。
5. 原始数据至 normalized hand 的 limits affine、以及下游 inverse affine 都保留。验证数据允许超过训练 limits，不在输入处追加 clamp。

不读取旧 `act_scale` 字段、不忽略旧 state_dict 多余键、不设置 `strict=False`。旧产物重新生成；不新增 format version。

验收：以旧实现 `act_scale=1` 为数值基准，检查 reconstruction/commitment、梯度和导出原型；新 checkpoint 可以独立导出，新 NPZ 可被下游 DQ-RISE 加载。测试不要求旧文件兼容。

### T4：允许 VQ 小数据与尾批训练

文件：[`training/pretrain/hand_prior/vqvae.py`](../dexmani_policy/training/pretrain/hand_prior/vqvae.py)。

- 删除“至少一个完整 batch”的限制，训练 DataLoader 固定 `drop_last=False`，不增加 `allow_small_batch` 或 `drop_last` 配置项。
- 继续按实际 `len(train_loader)` 推导 scheduler 总 updates；epoch loss 按实际样本数加权。
- 保持空训练集的明确失败；沿用已有数据划分/拟合边界，不为此增加多层检查。
- 不把 VQ 的放宽照搬到 AFRO。AFRO VICReg 的 batch 统计及 accumulation=1 限制保持。

这是有意的训练配方变化：非整除时每 epoch 从 floor(N/B) 变成 ceil(N/B) 次更新，尾批参与 EMA，轨迹不会与旧训练逐位一致。小尾批仍执行标准 EMA 更新，不引入按尾批比例改 decay 的新规则。

验收：至少覆盖 `N < batch_size`、非整除尾批、刚好整除；一个样本的 VQ 短训应可运行。确认每个 epoch 样本未被丢弃，updates/调度长度及样本加权指标正确；有/无验证的 best/last 规则不变。

### T5：CordViP 的零权重分支停止执行

文件：[`CordViP`](../dexmani_policy/training/pretrain/obs/cordvip.py)。

1. 继续以现有三个 loss weight 作为唯一控制量，构造时做一次有限、非负检查，并要求至少一个权重大于 0。
2. 保留三个 head 的直接定义。forward 中只对正权重项构造其 head 输入、执行 head、读取对应目标并计算 MSE。不要先计算所有损失再乘 0。
3. 指标字典包含总 loss 与实际计算的分项，关闭分项直接省略；不伪造零值，也不为获得诊断指标重新计算。
4. 共享 encoder 仍执行完整既有路径。无需按 loss 重组 encoder、动态移除模块、引入开关注册表或自定义 optimizer 分组。
5. 数据集本轮仍使用原有统一格式并要求 `contact_map`。关闭 contact loss 仅关闭模型分支，不新增“无 contact 数据”模式或第二个 dataset 开关。

验收：默认全正权重的 loss/梯度与原实现一致；分别关闭三个分支时该 head 不调用、梯度为 None，执行 optimizer 后其参数不因 AdamW weight decay 改变。测试关闭 contact 时模型不读取坏的或缺失的 contact target，但明确这只是模型边界测试，不代表数据集已支持缺少 contact 字段。至少一条分支仍给共享 encoder 产生有效梯度；负数、非有限和全零权重在构造处失败。

### T6：收短 VQ 导出与验证接口

文件：[VQ 导出](../dexmani_policy/training/pretrain/hand_prior/codebook.py)、[VQ 训练](../dexmani_policy/training/pretrain/hand_prior/vqvae.py)、相关测试和文档。

- `export_codebook()` 直接用已构建、排序的 FP32 poses 与 checkpoint 中的 hand affine 写 NPZ，不实例化 `HandStateCodebook`，不复制到运行时 buffer 后再取回。
- 返回 `int` 导出原型数量，供 CLI 打印。训练调用方无需返回值；测试通过真正读取新 NPZ 验证结果，不依赖导出函数返回模型对象。
- 继续使用现有同目录临时文件与 `atomic_path`，通过已打开的文件对象调用 `np.savez`；保留 `.npz` 后缀检查及显式 `--overwrite`。
- 保留消费外部 NPZ 的 shape、finite、affine 可逆性检查。不要因导出端不再构造运行时对象而复制整套加载校验进导出器；自产完整 checkpoint 直接按当前结构消费。
- `codebook_chunk_size(num_codes)` 内部固定最多 4096 行与 16 MiB FP32 `[N,K]` 距离矩阵预算，至少一行。移除只有固定调用的 `requested` 参数与其检查，不扩展成配置项。

验收：新 checkpoint → export → `load_codebook()` → 策略加载完整贯通；poses、affine 与 lookup MSE 一致。保留 PCA、learned layer weights、近重合原型索引及 K=1 行为测试。输出已存在时明确失败，显式 overwrite 才覆盖。

### T7：让预训练检查只依赖它实际检查的功能

文件：[`pretrain_checks.py`](../dexmani_policy/test_tools/pretrain_checks.py)、[`hand_prior_checks.py`](../dexmani_policy/test_tools/hand_prior_checks.py)、[`training_checks.py`](../dexmani_policy/test_tools/training_checks.py)；允许新增一个有明确职责的 `test_tools/contact_map_checks.py`。

1. 将 `check_tactile_configs` 合入已有 `training_checks.py::check_configs`，去除重复组合但保留关键配置行为；预训练测试不再绑定 adaptac/interact_dex。
2. 将接触公式、串行/spawn 生成及失败清理检查放入独立 `contact_map_checks.py`。它负责 Open3D 预处理回归，新增入口接受现有实际所需参数即可，不创建测试注册机制。
3. 核心 `pretrain_checks` 直接创建小型合成 Zarr，含明确对齐的 contact labels，供采样、短训、验证、续训与导出使用；不先调用 `generate()`。为少量 fixture 允许直接重复几行，不造通用 fixture 框架。
4. 将 VQ 中 loader/read/evaluate 的机械调用次数断言改为有意义的结果验证；删除 `split_metadata` 不存在这类历史实现断言。保留“无 val 时不做训练集选点/无用验证”“identity 不读取拟合数据”“导出不依赖 metrics/epoch”的行为检查，必要时仍可使用针对性的 mock。
5. 不全仓迁移 pytest、不扩建 CI、不新增依赖。已有短训和恢复检查是主要回归资产，优先复用。

验收：基础预训练检查无需安装 Open3D 即可通过；独立接触预处理检查在安装 `[preprocess]` 后通过。AFRO/CordViP 短训、同版本续训、验证状态/RNG 恢复及 encoder 导出加载仍被覆盖。

### T8：删除附属复杂度，压缩使用文档

主要更新 [`观测预训练.md`](观测预训练.md)、[`手部动作表示.md`](手部动作表示.md)；只有存在实际过期引用时才更新 [`项目架构.md`](项目架构.md) 或启动脚本。

- 删除 `act_scale`、完整训练 batch 限制、旧导出返回值、旧 chunk helper 签名和旧测试入口描述。
- 文档以可直接执行的命令、少量研究参数、产物内容、下游使用、续训语义及方法限制为主。
- 将重复的 checkpoint 释放顺序、对象生命周期、事务能力、来源声明等实施说明收为必要的一句话或删除；不写成新的长篇治理规范。
- 明确零权重只跳过模型分支、VQ 尾批改变更新次数、旧产物需重新生成。只需一处简短说明，无旧版操作教程。
- 对确实已无调用的包装、参数和附属测试同步删除；清理范围限于本次链路。代码中若没有对应机制，不为了满足“删除”字样制造修改。

## 5. 校验、异常与状态的取舍

| 处理 | 具体内容 |
| --- | --- |
| 保留：输入边界 | 实际所需字段、非空样本/窗口、关键 shape、FP32 转换后的有限性；码本 affine 有限且 scale 非零；必要的明确单位/关节顺序冲突检查 |
| 保留：数学语义 | AFRO accumulation=1；CordViP object/contact 1024 槽一致；观测长度与 horizon 关系；split 合法性；prototype 索引转换前的 finite 检查 |
| 保留：当前版本恢复 | model strict load、optimizer 参数名顺序检查、optimizer/scheduler/global_step/RNG、保存的有效配置。保持同版本同数据条件下的现有恢复能力，不增加跨设备/随机 worker 逐位重放承诺 |
| 保留：文件和资源 | 已有输出冲突提示、原子文件替换、finally 中的资源释放及 RNG/模式恢复；主异常不能被清理异常覆盖 |
| 不增加 | 每层 shape/schema/finite 检查、每步逐参数梯度扫描、strict/relaxed 双模式、自动重试、静默 fallback、失败后改用训练集、artifact 版本系统、evidence/source/hash 检查 |
| 可删除 | 同一条内部链中无独立风险的重复校验、已无调用的包装与字段，以及依附这些机制的测试/文档；须指明真实重复位置，不全局批量删 `raise`/`assert` |

观测预训练现有 loss finite 检查发生在该 update 的 backward 之后、teacher/optimizer 之前；VQ 在 backward 前检查 loss，启用 clipping 时才由 clip 检查 gradient。它们不保证检测所有“loss 有限而梯度非有限”的情况。本轮接受这一有限保证，不为未观测故障追加全参数扫描。

生产者与消费者各做一次 shape 检查可能分别面对不同输入边界，不能仅凭重复字样删除。接触预处理中的少量重开文件、提交前输出存在检查也不值得引入复用打开句柄或事务框架来消除。

## 6. 最小验收与执行顺序

建议顺序：T1/T2 → T3/T4/T6 → T5 → T7/T8。每一组检查实际行为；完成有关检查后不为凑数量扩展测试。

| 验收组 | 最少覆盖 |
| --- | --- |
| 配置/日志 | fresh/resume 不覆盖 W&B false；显式关闭有效；日志 LR 对应本次 update |
| AFRO | 换序前后 loss/gradient/RNG；teacher 无梯度；更新时机未变；验证不改 teacher |
| VQ | 删除 scale 的默认数值等价；小数据/尾批；真实样本加权；EMA 初始化；新 checkpoint 独立导出；有/无 val 的 best/last |
| CordViP | 默认损失/梯度等价；三个零权重分支跳过；未执行 head 不更新；active loss 可反传 |
| 产物 | encoder+normalizer 继承包括 identity；NPZ poses+affine 下游加载；strict 当前格式加载，不用外部数据清单 |
| 回归边界 | 同版本短训续训、源行/contact 对齐、验证状态/RNG 恢复、预处理公式及失败清理 |

在项目已有依赖环境中，核心命令为：

```bash
python -m dexmani_policy.test_tools.hand_prior_checks --device cpu
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
```

T7 完成后的独立预处理检查：

```bash
python -m pip install -e ".[preprocess]"
python -m dexmani_policy.test_tools.contact_map_checks
```

触觉配置迁移后运行 `training_checks` 中对应的配置检查；若相关初始化/共享训练代码被改动，再运行相应下游短训与恢复检查。不要仅为运行一个配置检查而强制启动无关仿真或整套重依赖测试。

有 CUDA 时可对上述核心命令追加相同的 `--device cuda:0` 检查，AFRO 性能测量独立记录。无 CUDA、无数据或依赖未满足时，明确列出未验证项及原因；静态检查不能替代实际 Torch 测试。

提交前核对：

- [ ] T1–T8 已完成；当前 HEAD 已解决的项目注明而不重复实现。
- [ ] 正常训练、短训、验证、导出和当前版本恢复链路成立。
- [ ] 无旧格式迁移/容错分支、无新增研究无关配置、无 evidence/contract/source 框架。
- [ ] 删除和接口变化同步到真实调用方、测试、文档；未留下旧参数示例。
- [ ] 等价修改与 VQ 尾批这类配方变化分别说明，未把研究假设变化混入重构。
- [ ] 结果仅报告实际运行的命令与结果；未测性能不写提升比例。

## 7. 本轮明确不实施的研究与扩展项

这些是两轮审查发现的限制或候选，不应在落地时遗失，也不应扩张成本轮重构：

1. **CordViP 接触槽与点序。** 全局 PointNet 对排列不变，固定 1024 输出槽依赖稳定点序语义；同时打乱点与标签也不能自动解决。当前生产链保持输入顺序，但这不证明上游不同帧拥有稳定模板对应。真实数据生成链需研究者确认；不添加模板哈希或 source 合同。point-wise head 是独立方法实验。
2. **协调预测的目标可见性。** cross-attention 已可能读取被重建模态；不能宣称严格 masked prediction。若研究遮挡，应在编码和交叉注意力前设计 mask，并单独评估，不只遮 head 输入。
3. **采样与 EMA 配方。** AFRO padding 后的重复帧、stride 的实际间隔，CordViP horizon 对窗口起点的限制都影响训练分布。保留现值；`n_obs_steps` 是 `obs_horizon` 的配置来源，不是重复参数。AFRO teacher 的 `0.996 + 0.004*min(epoch,300)/300` 不随任意新预算自动重标定，也不新增 schedule 参数。
4. **混合大小 micro-batch 的梯度累积。** 现有共享 loader 拒绝一个 accumulation group 中混合大小 batch，是对当前等权实现的必要保护，不能只删报错。未来若支持，应核实每个损失的统计含义，再按样本数权重累积并同步 metrics、主训练与观测训练；AFRO VICReg 不能因此等价成大 batch。本轮 VQ 使用独立 loader，不需动这条共享限制。
5. **单 token Q/K、worker 和恢复构造优化。** AFRO attention 的单 token 冗余可作独立性能实验，但本轮不改网络；不为一次 checkpoint 加载建立 meta/skip-init 路径；不默认开启 persistent_workers、添加 autotuner 或更换距离内核。先有实际测量再决定。
6. **额外功能。** 不增加 VQ 完整续训平台、无 contact 数据模式、通用 loss 开关系统、跨版本恢复、数据迁移工具或新的实验追踪服务。

## 8. 完成后的交付说明

实施者最终用中文简要报告：实际删除/修改了什么；哪些算法与数据语义保留；VQ 尾批带来的配方变化；运行过的验证及未验证项；GPU 测量结果（若实际执行）。有具体冲突时解释依据和最小调整，不用一份新审查报告替代已授权的实施工作。
