# 数据集模块收敛任务书：轻、薄、正确、好用

> 状态：P0—P5 已实施；当前行为以正文为准，第 17、18 节保留既往实施与验证记录，不作为当前兼容行为的依据。第 14 节为验收要求，不表示所有研究或实机验证均已执行。
> 仓库：`haoyangzhanglab/dexmani_policy`  
> 首轮审查基线：`f9a8e5134ad3a23788a2dc6f7107cff9fb8174bb`
>
> 本次小型收敛基线：`5698be1d5010230001210e2c887bb3c557ae123e`
>
> 定位：个人机器人策略研究项目。优先正确、直观、易改和实验可解释，不建设数据平台。  
> 本文汇总数据链路审查及 KineDex、LingBot-VLA、CordViP、AdapTac、FingerViP、DP3 的源码核查，并修正早期审查中的过强判断。

执行前阅读 [AGENTS.md](../AGENTS.md)。本文是当前数据集收敛任务的实施依据；[第一阶段](dataset_refactor_task.md)和[第二阶段](research_simplification_task.md)保留为历史实施记录。涉及本轮范围的冲突以本文明确决定为准；历史验证结果不得写成本轮验证结果。

## 1. 最终方案

保留当前 `EpisodeStore → SequenceSampler → Dataset → PyTorch DataLoader` 的直接结构。Dataset 负责取样、轻量 CPU 预处理和方法专属标签；训练装配负责拟合或继承 normalizer；模型负责输入归一化及模型相关处理；外部推理入口负责输入边界。各层只保存本层实际使用的状态。

本轮完成剩余的配置、数据描述、重复计算和检查工具收敛。已经正确、简洁的核心不重写，也不把算法消融混入工程重构。

| 事项 | 最终决定 |
| --- | --- |
| 存储 | 继续使用只读 Zarr 2、`data/*`、`meta/episode_ends`；不增加第二套训练格式 |
| 读取 | 按实际字段、点云通道和观测长度读取；保持进程内句柄与有界分块统计 |
| 采样 | 保持 episode 内窗口和边界重复；主策略动作起点仍为 `n_obs_steps - 1` |
| 训练数据量 | 新训练统一默认 `max_train_episodes=null`；100 条等论文预算由实验显式设置 |
| 留出与上限 | `val_episodes` 先占数量，训练数为 `min(M, N-V)`；选择顺序仍为同一排列上的 train、val、unused |
| 主策略验证 | 保留共享数量接口；主策略的 val 仅为留出，不增加离线验证循环 |
| RGB | 新训练的 Dataset/Runner/Runtime CPU 输出统一为 CHW uint8；模型内部转浮点 |
| 归一化 | 新字段只拟合选中 train episodes 的真实源帧；预训练 affine/identity 优先继承；resume 不重新拟合 |
| 方法差异 | 保留 R3D、未来力监督、AFRO、CordViP 的小型专用实现；不引入通用目标描述语言 |
| 信息传递 | 删除整包 attrs 的跨层转发；保留实际消费的手部坐标事实和部署数值事实 |
| 运行效率 | 优先去掉确定的无用读取、通道传输和可证明冗余的统计；不增加缓存管理层 |
| 校验 | 外部输入和真实数学边界检查一次；内部直接调用；不吞错、不随机换样本 |
| 文档与测试 | 一个当前任务入口；轻量数据检查与模型集成检查分开，复用已有工具 |

完成标准是使用路径更清楚、用户参数更少、重复机制消失、必要行为可验证。没有代码行数、类数、测试数量或加速百分比指标。

## 2. 必须先接受的审查结论

### 2.1 已经正确的实现

基线已经完成以下工作，继续保留，不重复建立一套新机制：

- Store 只打开选中字段；进程变化后重开 Zarr 句柄，支持 DataLoader workers。
- 普通采样的观测只读取 `O` 帧，动作读取 `H` 帧；窗口不跨 episode。
- 统计使用 train mask 下的原始帧，不使用 padding 后、增强后或重复窗口中的样本。
- AFRO/CordViP 可以继承 encoder 输入的 affine 和 identity；完整策略 checkpoint 包含推理所需权重与 normalizer。
- 主策略在搬到设备之前按 **Policy** 的消费字段过滤观测。多声明字段仍可能增加 CPU 读取、增强、整理及锁页内存开销，但不能说它们全部已传到 GPU。
- RGB/点云几何、触觉标量缩放、未来标签有效 mask、动作窗口与控制维度有真实语义，不属于可批量删除的工程负担。
- 数据身份 manifest、来源 hash 门槛、Canonical 格式身份门槛、持久化 normalizer shape 副本等已经移除。当前没有独立 evidence 验证子系统。

### 2.2 三项必须纠正的判断

| 议题 | 已核实事实 | 本轮处理 |
| --- | --- | --- |
| CordViP 预训练窗口 | 上游以完整策略 horizon 枚举窗口，再取前 `O` 帧做预训练；本项目延续这种取样方式 | 不称为索引 bug；默认保留。观察窗口专用采样单列为后续实验 |
| AdapTac 未来力 | 上游启用未来力分支时，训练把真实力加噪后的一步预测用于 guide，推理使用完整采样；action loss 可通过 guide 回传 | 不称为本项目独有实现错误；默认保留。不能直接改成回归头或 detach 并宣称等价 |
| DDIM 裁剪 | FingerViP、DP3 都存在范围归一化到 `[-1,1]` 配合 `clip_sample=true` 的默认路径 | 保留默认。风险在不匹配的非默认归一化组合，而非 DDIM 本身 |

这些事实来自代码阅读，不等于已证明对应算法的实机表现。既不保证上游实现绝对最优，也不把未经实验的推测当作重构理由。

## 3. 范围与执行边界

### 3.1 本轮实施

1. 明确并实现更直观的数量上限语义及新训练默认值。
2. 收拢现有配置，统一新训练 RGB 输出路径，缩小跨层数据描述。
3. 完成少量可证明等价的读取、统计和点云传输优化。
4. 保持各方法的时间、坐标、标签、归一化、checkpoint 行为。
5. 整理现有检查工具与用户文档，执行与实际改动相称的验证。

### 3.2 本轮不实施

- 不改策略骨干、action loss、未来力生成算法、contact loss、VICReg、码本算法、EMA、AMP、RTC 或学习率调度。
- 不改变各策略默认 `H/O/A`、动作起点、padding、预测与执行长度关系。
- 不改写原始 Zarr，不删除数据、权重或实验目录，不自动重采样控制频率。
- 不新增 DataModule、registry、adapter、通用 schema/contract、通用 TargetSpec、验证等级、格式版本系统、来源证明链、统计缓存服务、数据审计报告或任务运行器。
- 不引入 LeRobot/Arrow/视频解码框架、多机器人拼接和分布式数据恢复体系。
- 不建立缺测补零、错误样本跳过、随机替换样本、自动重试或静默回退。
- 不执行第 12 节的算法实验；它们不是本任务完成条件。

可以调整直接相关调用点。不要把“全面”理解成重写所有策略、扫描所有无关模块或重新跑所有历史实验。

## 4. 模块职责与接口

表中代码路径相对 `dexmani_policy/`。

| 模块 | 唯一主要职责 | 不应承担 |
| --- | --- | --- |
| `datasets/episode_store.py` | 只读数组、episode 边界、进程句柄、字段/列读取、分块迭代 | 配方解析、样本语义、normalizer、数据来源认证 |
| `datasets/split.py` | 数量参数到 train/val bool mask 的纯函数 | 路径、attrs、模型、manifest、自动分组 |
| `datasets/sampler.py` | episode 内窗口索引、padding 对应的真实行、按字段长度取样 | 数据增强、动作解释、未来标签协议 |
| `datasets/demonstration.py` | 普通 `obs/action` 样本、CPU 增强与图像整理 | 模型构造、归一化拟合、设备传输、checkpoint |
| `datasets/variants/` | R3D 拼接、AdapTac/InterAct 的专用监督 | 通用标签注册或多个算法的统一状态机 |
| `datasets/pretrain/` | AFRO/CordViP 样本及现有轻量验证视图 | 伪造 action 以服从主策略接口 |
| `training/common/data.py` | 数据装配、实际维度与必要事实、normalizer 装配、DataLoader | 数据身份判定、额外统计持久化、泛化管理器 |
| `training/common/config.py` | 已知配方派生关系及真正会改变含义的配置冲突 | 冻结所有配置、重复每层校验、维护方法能力矩阵 |
| `common/normalizer.py` | 流式统计、affine 张量与 normalize/unnormalize | 认识具体 Dataset、任务来源或部署环境 |
| `common/image_util.py` | 训练/推理共用的 CPU 图像处理原语 | 模型特定几何投影、第二套归一化体系 |
| `agents/.../image_processor.py` | 模型侧图像处理、像素坐标变换、投影一致性 | 重新读取数据或猜测相机标定 |
| `deployment/`、`env_runner/` | 外部观测整理、历史、动作输出与实际控制语义 | 拟合 normalizer、读取训练集、认证来源 |

保留现有普通函数和小类。不要为了与表格一一对应新增文件，也不要为省文件强行合并语义不同的组件。

## 5. 存储、字段与批数据

### 5.1 存储约定

- 必需结构仍是 `data/<field>` 和 `meta/episode_ends`。每个选中字段的时间轴长度等于最后一个 episode end。
- `episode_ends` 为非空、严格递增的正整数累计边界；不把 bool 当整数边界。
- 只检查所选字段；未使用字段及采集端其他 meta 记录不影响训练启动。
- RGB 存盘为 `[T,H,W,3]` uint8；连续物理量读取为 FP32。不要把磁盘数据全量转换或复制进内存。
- 保留 `pc_dim` 的真实作用：消费 XYZ 或 XYZRGB。需要保留全部存储通道的内部调用仍可用现有 `None`，不扩大成通用通道映射系统。
- 不强制填写 task/source/evidence/format 等身份标签。缺失必要数组直接暴露原生或一条明确的边界错误。
- `check_finite` 等离线定位工具可保留；不在每次训练启动或每个 batch 额外全库扫描。

### 5.2 统一的是含义，不是所有方法的字典形状

| 数据用途 | 单样本输出 | 时间/数值含义 |
| --- | --- | --- |
| 主策略 | `obs`、`action` | 各 obs 为 `O` 帧；action 为 `H` 帧，原始物理坐标 |
| RGB obs | `obs.rgb[O,3,Himg,Wimg]` uint8 | CPU 处理结束后连续存储；模型再转 FP32 |
| R3D 辅助目标 | 普通 obs 与拼接后的 action | joint 19 维加 `action_ee` 前 9 维；控制输出仍为主动作 19 维 |
| AdapTac | 普通样本，加顶层 `future_force[F,5,3]`、`future_force_valid[F]` | `F=H-O+1`；标签不混入普通 obs 字典 |
| InterAct-Dex | 辅助启用时加 `[1,5,3]` 与 `[1]` mask | 单个偏移 `interaction_horizon`，与执行长度独立 |
| AFRO | 点云序列 | 按现有 stride 和真实源行读取，不生成无用 action |
| CordViP 预训练 | 三类 obs 与 `contact_map` | 前 `O` 帧；contact 标签必须保持物体点槽对应 |

DataLoader 只增加 batch 维。无需再定义 Batch dataclass、通用 schema、字段版本或包装容器。

### 5.3 字段声明与读取

- `sensor_modalities` 是 Dataset 的读取声明；Policy 的 `consumed_observation_fields` 是实际模型消费定义。两者用途不同，不另加第三份 required-fields 注册表。
- 检查当前配方声明的额外字段是否确有辅助标签、策略层或几何消费者；确认无用后直接删 YAML 声明及对应 normalization 项。
- 不先构造整个模型来推导 Dataset 的读取字段，不引入自动字段发现系统。
- 保留缺少必需字段时的组装检查。主策略搬运前过滤必须看 Policy，不能只看 obs encoder：AdapTac 的策略层还消费 aggregate finger force。
- 如果现有配方没有多余字段，记录“无须修改”，不要为达成精简数量制造改动。

## 6. 采样与时序：保留稳定的核心

设 `H=horizon`、`O=n_obs_steps`、`A=n_action_steps`，episode 长度为 `L`，相对窗口起点为 `s`。

- 普通样本的源行是 `clip(s + arange(H), 0, L-1)`，再加 episode 的全局起点。
- 观测取这些源行的前 `O` 项；query 是源行序列的第 `O-1` 项。
- 预测后可执行动作从位置 `O-1` 开始；可用未来长度为 `F=H-O+1`；要求 `1 <= A <= F`。
- `validate_action_window(H, O, A)` 直接推导 F 并检查可执行性，不再接收重复派生的动作起点；H/O/A 的正整数检查保留。
- `s` 从 `-pad_before` 到 `L-H+pad_after`，窗口数为 `max(0,L-H+pad_before+pad_after+1)`。
- 主策略当前派生 `pad_before=O-1`、`pad_after=A-1`，继续保留。这不保证每个 episode 的每个 query 都被覆盖，也不应自动解释成 bug。

实现要求：

1. 保留现有紧凑索引和 `sample_source_rows()`；不要新增 query manifest、时间戳索引服务或窗口对象层。
2. 普通读取继续为每字段读取一个所需连续片段，然后在必要时作边缘重复；不要先读取完整 `H` 帧视觉数据再丢弃。
3. 不跨 episode 取 padding 或未来标签。零力是合法值，不能用 `value != 0` 推断有效性。
4. 个别短 episode 无窗口时保持原规则；整份训练选择无样本时明确失败，不换 episode 补数。
5. AFRO 的 stride、CordViP 的点槽对应、R3D 的辅助切片保留专用代码，不为了统一都改成相同 horizon。

### 6.1 未来力标签

`read_future_force()` 的有效性根据**未裁剪**目标行判断：`query + offset < episode_end`。越界标签以本 episode 最后一行占位，loss 使用 bool mask 排除。

- AdapTac 偏移为 `1..F`，动作从 query 开始；这种定义可表示每次动作后的力，并不能仅凭偏移差一就认定 off-by-one。本轮保持。
- AdapTac 当前过滤没有真实后继的 query，保证样本至少有一个有效未来力标签。不能删过滤后又保留依赖该前提的分母。
- InterAct 保留只有一个 `q+K` 标签、允许全无效样本的现有语义；辅助权重为 0 时继续使用普通 Dataset，完全不读未来标签。
- InterAct 仿真残差先在世界系作差，再用 query 时刻的 EEF 旋转转换；不读取未来位姿，不另拟合残差统计。

不要为了表面统一强制两种方法采用相同的过滤或全无效处理。

## 7. Episode 数量与默认值：本轮明确的行为变更

### 7.1 新语义

保留 `split_episodes(n_episodes, seed, max_train_episodes, val_episodes)` 及两个 bool mask 的接口。`M` 是训练数量的上限，`V` 是明确留出的数量：

```python
# 伪代码；沿用现有函数，不增加参数对象。
require N > 0 and 0 <= V < N
require M is None or M > 0
available = N - V
T = available if M is None else min(M, available)
order = np.random.default_rng(seed).permutation(N)
train = order[:T]
val = order[T:T + V]
# 其余为 unused；输出对应 bool masks。
```

数值类型仍在该函数入口检查一次。不缩小请求的 `V`，不重复采样补数，不用 `~train_mask` 代表 val，不增加 split manifest。

这是相对基线的**明确设计调整**：基线要求先满足 `min(M,N)` 的训练数量，再检查 val 是否放得下；新语义让“上限”与留出数量自然组合。此前已成功的相同 `N/M/V/seed` 组合仍给出同一 train/val 选择；此前数量冲突的一部分组合变为合法。

| N | M | V | 新结果 train / val / unused | 与基线关系 |
| --- | --- | --- | --- | --- |
| 150 | 100 | 20 | 100 / 20 / 30 | 不变 |
| 80 | 100 | 10 | 70 / 10 / 0 | 原先报错，现采用上限语义 |
| 110 | 100 | 20 | 90 / 20 / 0 | 原先报错，现采用上限语义 |
| 100 | null | 10 | 90 / 10 / 0 | 不变 |
| 150 | null | 0 | 150 / 0 / 0 | 不变 |
| 1 | null | 0 | 1 / 0 / 0 | 不变 |
| 1 | null | 1 | 错误：没有训练 episode | 保留错误 |
| 合法 N | 0、负数或错误类型 | 合法 V | 参数错误 | 保留错误 |

依然使用局部 RNG。`dataset.seed` 与 `training.seed` 独立。同一 seed 下改变训练上限时，val 成员可能随之变化；本轮不承诺跨预算固定验证集。

### 7.2 新训练默认与主策略留出

- `DemonstrationDataset` 的默认值与 `configs/common.yaml` 改为 `max_train_episodes=null`；独立预训练已有的 null 保持。
- `val_episodes=0`、数据 seed=42 保持。论文对照需要 100 条时显式设置 `dataset.max_train_episodes=100`。
- 新默认会使超过 100 条的数据集使用更多数据。这是训练配方变化，必须在 README/实验示例中说明，不宣称为等价重构。
- 不把新默认覆盖到已有 checkpoint 的 resolved config；resume 沿用保存的具体值。
- 主策略保留现有共享 `val_episodes` 参数，避免为同一种数量选择复制第二套接口。启动摘要明确写“留出但未验证”；它不是已运行的验证集指标。
- AFRO/CordViP/VQ 的显式 val 继续运行已有验证。V=0 不构造 val loader，不额外遍历 train 来选 best。
- 统计范围仍是选中 train episodes 的原始帧；某个被选 episode 因窗口过短未产生样本，也不自动改写划分或引入按窗口加权统计。

## 8. Normalizer：一个所有者、一条数据来源路径

### 8.1 必须保留

1. Dataset 输出物理数据，不持有 normalizer；训练装配拟合一次，Policy 保存最终 affine。
2. 新统计只遍历 train mask 下的真实源帧。排除 val、unused、padding、增强与重叠窗口重复。
3. 流式按块累计；gaussian/rms 的矩使用 FP64，最终 affine 为 FP32。保持当前常数通道处理、零力处理和数值约定。
4. 已继承的字段不重拟合；identity 也是预训练输入变换的有效决定，不能因没有 affine tensor 而偷偷改成 limits。
5. Resume 与完整 checkpoint 推理直接使用保存的 normalizer；不要依赖原始预训练目录或为了恢复去扫描训练数据。
6. 不增加 stats.json、可选统计 scope、缓存指纹、缓存失效协议或重复的 affine shape metadata。

### 8.2 不能被“统一归一化”破坏的数学含义

| 输入 | 要求 |
| --- | --- |
| 普通动作 | 保持当前默认 limits；反归一化回实际物理动作 |
| `action_ee` 的 rot6d | 保持现有 `[3:9]` identity 处理；不要把姿态分量改成任意偏移缩放 |
| 用于几何投影的点云/内外参/身体锚点 | 按当前方法保留物理坐标及 identity；不套普通特征缩放 |
| 触觉向量 | 保留整个字段共享标量、零 offset 的 rms/max_abs；合法零力有效 |
| DQ-RISE / DexLatent 的 hand action | 使用预训练 hand affine；不能用本阶段新统计覆盖 |
| DexLatent latent | 当前训练集 latent 统计仍有独立作用；不能因为有 hand affine 就删除 |

触觉共享标量满足旋转交换关系 `c(RF)=R(cF)`。逐轴不同的 scale 或非零 offset 一般不满足该关系，因此不能用通用逐通道标准化替代。

### 8.3 手部统计优化的精确边界

基线先拟合完整 action，再以预训练 hand affine 覆盖手部。这在默认 limits 下包含可以去掉的手部统计。

目标是在现有训练装配中用一小段直接代码完成，不新增 PartialNormalizer、规则注册表或用户配置：

- `limits`、`gaussian` 为逐特征统计：只拟合 arm 部分，再拼入固定 hand scale/offset；其余结果应与“全量拟合后覆盖”数值一致。
- `identity`：直接建立 arm identity，再拼 hand affine，无需扫描 action 统计。
- **`rms`、`max_abs` 当前跨整个字段使用共享标量。只读取 arm 会改变 arm 的尺度。对于这些非默认模式保留原有全 action 统计再覆盖 hand 的路径。**
- 列切片能减少计算和中间数组；底层 Zarr chunk 仍可能被整体解压，不能承诺磁盘读取等比例减少。
- 优先复用现有 chunk iterator，必要时在局部 generator 上切列。若为了省这部分统计需要重构所有 Dataset 或引入多层抽象，保留现实现并说明原因；不能牺牲易读性。

### 8.4 DDIM 与动作范围

默认 `limits + clip_sample=true` 保持。`agent.clip_sample` 是有研究意义的参数，不删除，也不在后台自动改写。

- gaussian/rms 不能保证动作落入 `[-1,1]`；identity 要看物理范围，不能一律判错。
- max_abs 可把拟合数据限制在单位范围内，但仍不保证分布外动作；limits 同样不是全世界动作的保证。
- 非默认无界动作实验应显式配对关闭动作 DDIM 的 sample clipping。真实配置键沿用 `agent.clip_sample`；Hydra 中尚未声明时使用既有的添加键语法，不发明第二个参数。
- epsilon/sample/v_prediction 只是预测参数化；不能通过改 prediction type 消除对重建 clean sample 的范围裁剪。
- 本轮不新增自动扫描动作范围、通用 decoder-normalizer 兼容矩阵或“安全归一化”模式。文档给出配对示例，数值回归验证裁剪确实发生在预期空间。
- 调整现有检查中孤立的 `normalization.action=gaussian` 示例：若它表示可训练推荐配方，应显式配对关闭 clipping；不能把“YAML 可解析”写成该组合已数值正确。

当前 DP3 配方已显式声明 `agent.clip_sample`，配对覆盖写为 `normalization.action=gaussian agent.clip_sample=false`，不使用 `+`。这个例子用于说明参数关系，不改变默认配方，也不表示已经完成该研究实验。

## 9. RGB 与几何：统一输出，保留必要处理阶段

### 9.1 新训练的固定输出

新训练统一路径：存盘 HWC uint8 → CPU resize/crop/颜色增强 → 连续 CHW uint8 → 模型内部浮点转换与 mean/std。

- 新训练和推理均使用固定 uint8 输出，不提供输出 dtype 开关。
- 更新 Dataset、`rgb_preprocessing_kwargs()`、SimRunner、PolicyRuntime 及相关测试的直接调用点。原来默认 RGB 配方已经为 true，应保持这些配方的数值处理次序。
- 颜色增强可能需要浮点中间值，最终再量化；不能为强制 uint8 而在每个中间步骤重复量化。
- 保留训练随机 crop、推理 center crop 和同一观察窗口共享裁剪参数。
- 保留 `float_spatial_before_uint8` 对齐训练/推理量化顺序的实际含义。它可以是内部派生值，不新增用户开关。
- 模型内部确有浮点输入的调用可以继续接受浮点，不把本任务扩大成禁止所有浮点图像。

### 9.2 checkpoint 输入边界

checkpoint 保存当前固定 uint8 配方，查看、续训和推理共用直接读取路径，不改写预处理配置或磁盘文件，不维护历史配方迁移、版本判断或兼容开关。

### 9.3 不误删几何处理

纯视觉 DP 在 CPU 缩放以降低传输成本；标定 RGB+点云路径保持 Dataset 原图，由 ImageProcessor 在模型侧变换像素坐标并保持投影一致。这两类处理有不同职责，不能仅因都叫 resize 就删除一处。

- 保持 `align_corners=False` 下的像素中心仿射，包括半像素偏移。
- 不能独立对标定点云 XYZ 加噪而不保持 RGB 对应关系。
- CordViP 的物体点云与 contact 标签必须使用同一槽顺序；不能单独 shuffle/FPS 一边。
- 不把这些约束抽成 geometry contract 系统；保留当前局部检查和真实数值测试。

## 10. Metadata、source、校验与异常的最终取舍

### 10.1 按消费者缩小信息传递

`build_dataset()` 的 description 只包含实际消费者需要的少量值；DQ-RISE 从中读取必要的手部坐标声明：

- 字段维度、`action_key`、`control_action_dim`；Dataset/Policy 的动作起点均由同一 `n_obs_steps` 派生为 `O−1`，description 不再传递，Trainer 不再比较这两个同源值。各自用于采样、标签及动作输出的属性保留。
- 必要手部坐标声明的小字典，例如已有 `hand_joint_order`、`hand_units`；同步 `initialize_training()` 调用。
- 已经白名单提取的部署事实。只保存模型真实消费模态对应的项目。

Store 内部读取少量 Zarr attrs 本身不是主要开销，不必为其增加懒加载管理器。重点是不要把原始 attrs 整包传给各层或 checkpoint。

| 信息 | 决定及理由 |
| --- | --- |
| `normalization_dims` | 保留；identity 字段未必有 affine 张量，输入宽度仍需明确 |
| affine shape 的额外配置副本 | 不恢复；从选中 state_dict 推导 |
| `policy_metadata` 中动作起点、控制维度、执行模式 | 保留，R3D 预测维度与控制维度确实不同 |
| `control_dt_s`、点云构造参数、depth 尺度、指尖名称 | 有值且有实际消费者时保留；缺失不伪造 |
| encoder 的输入 normalization | 保留，包含 identity；不扩展成来源凭证 |
| hand 单位与顺序 | 已声明且明确矛盾时拒绝；缺省不升级为来源认证要求 |
| `tactile_source` | 保留 real/sim 的数值解释与现有外部检查；不能因名称含 source 删除 |
| 初始化权重路径 | fresh 初始化确实需要，保留；完整 checkpoint 恢复不访问它 |
| source ID、hash、evidence、采集工具标签 | 不作为训练、导入或推理门槛；不重新引入已删除机制 |
| 实验 seed、预算、推理设置、结果 | 保留用于研究复现；不要求另写强制证据清单 |

只有先证明 real/sim 的触觉单位、轴向和坐标语义被统一，才能讨论替换 `tactile_source`。本任务不执行该算法/数据定义变化。

### 10.2 检查只放在真实边界

| 边界 | 保留 | 可以清理 |
| --- | --- | --- |
| 打开 Zarr | 所选字段存在、时间轴、episode ends、数值 dtype | 对无关字段/身份标签的要求 |
| Dataset 构造 | 所需 shape、非空点云、窗口可用、实际几何约束 | 重复 Store 已保证的结构检查 |
| 参数消费 | 数量范围、动作窗口可执行、所需字段齐全 | 上下游反复检查同一内部派生值 |
| 外部 affine/权重/码表 | strict 加载、有限可逆 affine、真正使用的维度和物理布局 | 同任务/同来源/同 split 的凭证 |
| Runtime 外部输入 | dtype、shape、有限性、必需历史与实际坐标语义 | 追查训练数据身份或重新读取统计 |
| 训练数值 | 非有限 loss 明确中止 | 每 batch 额外遍历所有观测做重复全量验证 |

保留原生错误的上下文。不要把 `KeyError`、文件不存在或形状错误逐层 catch 后重新包装；确需增加信息时只在拥有该信息的边界处理。资源关闭的 `finally` 和保留主异常的清理逻辑不因“减少异常处理”而删除。

## 11. 小而确定的运行效率优化

### 11.1 SimRunner / Runtime 提前裁剪点云通道

SimRunner 与 PolicyRuntime 在 H2D 前均直接依据已加载 `agent.normalization_dims[name]` 裁剪各点云字段。Runtime 的通道数检查也使用同一宽度；`PolicyInfo.pc_dim` 仅保留为展示字段，不再充当所有点云字段的统一裁剪依据。

- 复用已有 `normalization_dims` 或等价的实际消费信息，不再从任意环境默认值猜 `pc_dim`。
- 直接读取各字段消费宽度，不创建新的持久化维度映射或 observation adapter。
- 三类点云字段的处理遵循现有字段常量，支持无点云的策略。
- 结果应与“传全部通道后取前 C 维”一致，历史顺序与初始重复首帧不变。
- `O=2,N=1024` 时，从 6 维 FP32 缩为 3 维，单次点云载荷由 48 KiB 变为 24 KiB；这只是载荷量，不是端到端加速测量。

### 11.2 现有性能机制继续保留

- DataLoader 的 `num_workers/pin_memory/persistent_workers/prefetch_factor` 有实际用途，不统一删除；workers=0 时保持现有无效项处理。
- 进程重开句柄及 persistent workers 退出清理保留。
- 不新增全量 replay buffer、样本缓存、预取队列封装或每 epoch 重算 normalizer。
- 不承诺 worker 增加必然提速；沿用现有训练步耗时即可，只有观察到瓶颈才做小范围 profiling。
- `EpochBatchSampler` 的 epoch 排列和恢复位置服务现有 resume，保留；不要为了数据模块精简重写训练恢复。
- 不承诺多 worker 随机增强在中断恢复后逐 bit 一致，除非真实实现与测试已经保证。

### 11.3 控制周期的边界

基线保存/暴露 `control_dt_s`，但 SimRunner 不自动据此设置环境周期。这是非默认控制频率实验的条件风险，未证明默认 16 Hz 路径已经错配。

本轮在部署说明中明确：训练和执行控制周期需匹配，动作数不能替代秒数。若后续接入其他频率，先核实 `dexmani_sim` 的真实 API，再在环境创建边界显式设置或核对一次；没有核实 API 时不猜 `dt/control_dt/frame_skip` 参数，不加入自动重采样或频率兼容层。

## 12. 独立研究事项：只记录，不在本轮实现

### 12.1 CordViP 的观察窗口采样

当前默认 `H=12,O=4,pad_before=3,pad_after=5`，每个 episode 有 `max(0,L-3)` 个窗口；实际预训练只用前 4 帧。

可以另开实验，把 sampler 改为 `sequence_length=O,pad_before=O-1,pad_after=0`，得到每个真实 query 一个观察窗口，共 L 个。旧有 query 的观察历史可保持一致，但会加入尾部 query，且短 episode 从无窗口变为有窗口。这改变训练分布和训练步所见样本，不是等价删除无用参数。

本轮保留当前默认及其参数。若以后做该实验，单独报告样本数量、短 episode、训练预算与结果；不要新增 `sampling_mode` 多模式框架来预置尚未做的实验。

### 12.2 AdapTac 的未来力 guide

上游未来力启用时具有与当前项目相似的训练/推理条件差异。当前项目扩展为五指 15D 未来力，并非上游 3D aggregate force 的逐字复刻。

- 保留当前扩散未来力分支和 action-loss 梯度路径。
- 显式未来力 loss 权重设为 0，不代表 guide 不再使用依赖未来标签的训练预测，也不代表梯度路径完全消失。
- 优先的后续消融是“未来力仍作辅助预测，guide 只使用当前观测力”，然后比较完整模型。
- 改回归头、detach guide、训练时完整采样、移除未来力均是独立算法实验，不能以精简 Dataset 为名直接替换。
- 辅助 loss mask 的计算正确，不足以证明未来预测占位对 guide 的全部影响；如果研究这个影响，应单独控制有效 horizon 和条件构造，不混入工程验收。

### 12.3 DDIM 的非默认归一化实验

保持 FingerViP/DP3 所支持的默认范围归一化思路；如需 gaussian/rms 或超出单位范围的 identity 动作，单独比较显式关闭 clipping 的配方。不能通过截断物理动作或修改训练数据来伪造与默认范围的兼容。

## 13. 实施顺序与文件清单

按依赖顺序完成，允许把相邻小任务合并为少数可审查提交。不要求为每个条目创建 commit、issue 或产物报告。

### P0：核对当前代码与边界

- [x] 阅读 AGENTS、本文和实际调用方；确认当前分支相对审查基线的变化，尊重已有工作区修改。
- [x] 已经满足的条目标记无需修改；不要重新建立已删除的来源/manifest/shape 机制。
- [x] 先记录本轮两类变化：明确改变使用行为的数量默认/上限；应保持数值行为的工程收敛。

### P1：数量与配置收敛

- [x] `datasets/split.py` 实现第 7 节的上限语义，复用到所有当前入口。
- [x] `datasets/demonstration.py`、`configs/common.yaml` 的新训练上限改为 null；保留显式论文预算示例。
- [x] 主策略摘要准确区分留出与实际验证；预训练 V=0/正值语义保持。
- [x] 核对配置派生：主策略 H/O/A 与 action layout 继续在现有装配处派生，避免新增可独立覆盖的副本。
- [x] 核对字段列表和 normalization 声明；只删除已证明无消费的项。

### P2：数据内容与统计收敛

- [x] `training/common/data.py` 缩小 description；同步 `agents/policies/dqrise.py` 等真实消费者。
- [x] `training/trainer.py` / 现有 normalizer 装配去掉逐特征模式下被覆盖的 hand 统计；保留 rms/max_abs 的全字段尺度语义。
- [x] 不增加 normalizer 持久化文件或旧来源依赖；检查 inherited identity 与 resume 路径。

### P3：RGB 与推理输入收敛

- [x] `datasets/demonstration.py`、`common/image_util.py`、RGB 配方统一新训练 uint8 输出。
- [x] 同步 `deployment/runtime.py`、`deployment/metadata.py`、`env_runner/sim_runner.py`、`evaluation/rollout.py` 的实际调用点。
- [x] checkpoint 直接读取当前固定 uint8 配方，不迁移配置或改写原文件。
- [x] SimRunner H2D 前裁剪点云消费通道；不改变历史、动作切片和控制长度。
- [x] 保留几何仿射、训练/推理量化顺序和各路径的职责。

### P4：检查工具与文档

- [x] `test_tools/dataset_checks.py` 聚焦 split、source-frame 统计、sampler、标签与数据读取；避免顶层导入整个 PolicyTrainer、视觉主干和外部初始化资源。
- [x] 将已有 checkpoint/raw-EMA、恢复日志、完整策略短训等检查移到现有 `agent_checks.py` / `pretrain_checks.py` 的适当位置；消除重复，不复制同一套测试。
- [x] 保留本次受影响的最小集成检查；不新建测试框架、测试发现协议或参数组合矩阵。
- [x] 更新根 README、`data/README.md`、`docs/项目架构.md`、`docs/仿真数据集格式.md`、`docs/观测预训练.md` 及受影响的手部/部署说明。
- [x] 当前任务书只记录本轮实际状态；旧任务书保留历史标记和入口链接，不复制为第三份用户使用文档。

### P5：验证与收尾

- [x] 执行第 14 节中受改动影响的检查；说明已运行、失败、未运行及原因。
- [x] 检查配置、CLI 示例、调用引用与文档是否同步；删除本轮引入而未使用的参数和 helper。
- [x] 对新默认使用全部数据、上限新语义、固定 RGB 输出作简短说明。
- [x] 未执行长训、真实闭环、硬件测试或性能 profiling 时明确写出；不把合成数据结果扩写成任务成功率结论。

## 14. 验收：验证真实风险，复用已有检查

先运行轻量 CPU 检查，再针对涉及的模型边界做最小集成。没有修改某项内部数学时，复用已有检查即可，不要求再造一套穷举或镜像实现测试。

| 检查对象 | 最小有意义的验证 |
| --- | --- |
| split | 覆盖第 7 节边界表；互斥/数量正确；同 seed 稳定；不消费全局 RNG；旧有合法组合成员不变 |
| 窗口 | 短 episode、两段相邻 episode、前后 padding 与 O<H；用小型真实行编号数组确认不跨界、query/action 起点不变 |
| 按需读取 | 小 Zarr 中加入未声明字段，确认不读取；obs 只取 O 帧、action 取 H 帧，所选点云通道正确 |
| train-only stats | 在 val/unused 放入极端值，确认 train affine 不变；不同 padding/重叠窗口不改变源帧统计 |
| 统计数学 | 分块与直接统计在合理浮点误差内一致；常数列和零力有效；normalize/unnormalize 可逆 |
| hand affine | limits/gaussian 优化与旧拟合后覆盖结果一致；identity 不扫描；rms/max_abs 用含大 hand 值的样本确认 arm 尺度未改变 |
| 继承/恢复 | affine 与 identity 均继承；恢复时不重新拟合，不访问原 encoder/codebook；必要时复用已有模拟缺失来源测试 |
| 未来力 | mask 由未裁剪索引产生，零值合法；AdapTac 无后继过滤、InterAct 全无效 mask 与权重 0 路径保留 |
| RGB | 新训练/推理为连续 CHW uint8；相同确定性设置下预处理数值一致；颜色增强量化顺序不变 |
| 几何 | 使用真实 torch resize/投影路径核对像素中心与标定；仅检查矩阵形状或纸面公式不算完整图像验证 |
| SimRunner / Runtime | H2D 前按各字段实际消费宽度裁剪；统一宽度配方数值不变，不同字段宽度独立生效；历史、FP32、连续布局及有限性保持 |
| DDIM | 小型 scheduler 数值例子验证 clipping 与动作 affine 的关系；默认 limits 路径保持；非默认例子显式关闭 clipping |
| 当前 checkpoint | 当前产物 fresh→save→resume/inference；读取不迁移配置或改写原文件 |
| 轻量入口 | Dataset 检查不要求下载权重、加载实机环境或构造大视觉模型；测试失败保留明确异常 |

优先使用仓库已有入口，修改后同步其实际能力：

```bash
python -m dexmani_policy.test_tools.dataset_checks
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
python -m dexmani_policy.test_tools.agent_checks --device cpu
```

这些是现有命令，不表示每轮必须完整运行三个入口。若重排检查后提供更小的现有调用方式，在交付中列出实际执行命令。CUDA/图像模型/预训练集成只对被修改且环境具备的路径运行，不为了文档精简启动长训。

本任务书编写阶段仅核对仓库源码与文档，**没有运行上述实施后验收，没有获得性能提升或训练成功率结果**。此前采样/历史索引的数值审查也不替代本轮真实依赖下的验证。

## 15. 交付要求

完成后用中文简要报告：

1. 实际修改及对应文件，哪些建议经核对无需修改。
2. 数量默认/上限语义的明确变化，以及保留的算法和时序。
3. 实际验证命令、结果、未覆盖项与具体原因。
4. 仅在确有测量时报告性能；区分传输载荷减少和端到端吞吐改善。
5. 如有无法完成的必要项，说明阻碍和最小后续动作；不能悄悄跳过，也不因一个无关依赖停止其他独立工作。

不要另交强制 source/evidence 清单、迁移 manifest、全面审计数据库或一组没有消费者的配置。本文的源码链接用于理解设计，不是训练或推理的运行条件。

## 16. 参考源码与取舍依据

以下固定提交仅用于定位本次阅读依据，实施时仍需核对当前仓库。引用上游是为了理解真实机制，不要求复刻其目录、缓存、异常或配置体系。

| 项目 | 阅读定位 | 采用与保留距离 |
| --- | --- | --- |
| KineDex | [KinesDataset](https://github.com/DinoMini00/KineDex_code/blob/9246f77d7b969b181b0f1b600601b9a9c8083011/diffusion_policy/dataset/kines_dataset.py) | 学习直接 obs/action 字典与短观测读取；不复制 MemoryStore 全量 replay、shape hash 缓存/文件锁或全 buffer 统计 |
| LingBot-VLA | [VLADataset](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/data/vla_data/base_dataset.py)、[transform](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/data/vla_data/transform.py) | 借鉴读取与特征变换的职责划分；不引入多机器人最大维度 padding、LeRobot 平台依赖或失败后随机换样本 |
| CordViP | [预训练目标](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/policy/CordViP_pretrain.py)、[配方](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/config/CordViP_pretrain.yaml)、[Dataset](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/dataset/robot_dataset.py)、[padding 配置](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/config/task/all_tasks_160.yaml) | 确认完整 horizon 枚举、前 O 帧监督；本项目已不为预训练加载无用 action，继续保留 |
| AdapTac | [FFG policy](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/policy/FFG_policy.py)、[扩散实现](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/model/transformer_diffusion.py) | 确认未来力条件与梯度路径；不把算法改造当作数据集修复 |
| FingerViP | [训练配方](https://github.com/zhzhen0626/FingerViP/blob/aba80110bf5777cd926f394ac6e870c877e21c6c/DiffusionPolicy/diffusion_policy/config/train_fingervip_workspace.yaml)、[Dataset normalizer](https://github.com/zhzhen0626/FingerViP/blob/aba80110bf5777cd926f394ac6e870c877e21c6c/DiffusionPolicy/diffusion_policy/dataset/fingervip_dataset.py) | 确认 epsilon DDIM、clip=true 与 range action normalization；不复制为统计遍历并缓存窗口的实现 |
| DP3 | [默认配方](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml)、[RealDex Dataset](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/dataset/realdex_dataset.py) | 确认 sample DDIM、clip=true 与 limits 的默认组合；保持本项目自己的 train-only 分块统计 |
| Diffusers | [v0.27.2 DDIMScheduler](https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/schedulers/scheduling_ddim.py) | 当前依赖版本的 clip 作用于重建的 `pred_original_sample`；不能通过更换 prediction type 回避范围约束 |
| 本项目 | [数据目录](https://github.com/haoyangzhanglab/dexmani_policy/tree/f9a8e5134ad3a23788a2dc6f7107cff9fb8174bb/dexmani_policy/datasets)、[训练装配](https://github.com/haoyangzhanglab/dexmani_policy/blob/f9a8e5134ad3a23788a2dc6f7107cff9fb8174bb/dexmani_policy/training/common/data.py)、[Normalizer](https://github.com/haoyangzhanglab/dexmani_policy/blob/f9a8e5134ad3a23788a2dc6f7107cff9fb8174bb/dexmani_policy/common/normalizer.py) | 在已经精简的主线上收敛剩余问题，保持物理含义与现有研究能力 |


## 17. 上一轮实施与验证记录（2026-11-03，历史）

实施起点为 `88f2856`，工作区干净；相对首轮审查基线仅有三份任务文档变化。以下保留截至 `5698be1` 的原实施及验证记录，其中“包含 RGB 键一律拒绝”的行为已由第 18 节替换，不能作为新行为已通过验证的依据。

### 实际变更与无需修改项

- 新训练默认上限为 null，T 为 `N−V` 或 `min(M,N−V)`；主策略明确报告 val 留出但未验证。100 条预算需显式设置，这是训练配方变化；恢复保持 checkpoint 保存的具体参数。
- 跨层 description 改为必要维度、动作事实、hand_coordinates 和已有部署事实，不再转发完整 attrs。DQ-RISE 消费者及检查同步。
- hand affine 在现有 normalizer 装配中拼接：limits/gaussian 仅统计 arm，identity 不扫描，rms/max_abs 保留完整字段共享尺度。CPU 五模式与旧版“完整拟合后覆盖 hand”逐位对照通过，DexLatent latent 统计保留。
- RGB 固定连续 CHW uint8，移除配方开关；checkpoint 不做参数迁移，包含废弃参数时明确拒绝。颜色增强、最终量化、标定原图和模型侧几何保持。
- SimRunner 按恢复 Policy 的实际 normalization_dims 在 H2D 前裁剪点云，PolicyInfo 公开实际消费宽度；Runtime 和 rollout 的公共 kwargs 调用自动同步，无需额外修改。
- Dataset 检查与模型集成分开；sampler 检查迁入 Dataset 入口，恢复/日志/策略短训迁入 agent 入口，未建立新框架。README、数据、架构、格式、预训练、手部及部署说明已同步。
- 经核对无需修改：Store 进程句柄、Sampler 窗口/按需读取、train 原始帧统计、预训练 affine/identity 继承、resume 不拟合、H/O/A 装配、字段声明、预训练 V=0/正值分支。CordViP 默认采样、contact 点槽、AdapTac 算法/梯度、未来 mask、R3D 与默认 DDIM 语义均保留。

### 实际执行的验证

解释器为 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`，Torch 2.4.1+cu124。CUDA 在用户授权的 sandbox 外执行，设备为 RTX 4090。

```bash
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m dexmani_policy.test_tools.dataset_checks
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m dexmani_policy.test_tools.pretrain_checks --device cpu
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m compileall -q dexmani_policy
git diff --check
```

上述命令均通过。Dataset 检查覆盖数量/RNG、sampler、train-only 统计、hand 五模式、按需读取、未来标签、R3D、RGB 连续布局和增强量化；预训练检查覆盖 AFRO stride、CordViP contact 行/点槽对齐、串行/spawn 标签处理、有无 val、恢复、导出及相关异常清理。另独立导入轻量入口，确认不导入 trainer、transformers、timm 或 dexmani_sim。

复查阶段运行了完整 CUDA agent_checks，全部通过；日志为 `/tmp/dexmani-dataset-review-cuda.log`：

```bash
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m dexmani_policy.test_tools.agent_checks --device cuda:0
```

完整入口覆盖各策略合成前后向、保存/加载/预测对照、RGB raw/EMA 与 LoRA、R3D 有无辅助目标、点云传输前裁剪与历史、真实 resize/投影、DDIM clipping，以及 AFRO/CordViP/DQ-RISE 各 2 次 fresh updates 与 1 次 resume update。恢复检查删除原 encoder/codebook，并禁止重新拟合；DQ 确认保存的 100 条预算不受新默认影响。包含废弃 RGB 参数的配置明确拒绝，不再维护自动删键及兼容恢复检查。CPU 对固定 uint8 预处理的数值及增强量化顺序也已验证。

本轮沿调用链复查了 Dataset、统计装配、配置、策略消费、checkpoint、Runtime、SimRunner、评测和 shell 入口；未发现额外 SHA/hash、来源证明或版本迁移机制。保留现用几何、触觉和恢复边界，未为清理数量改写稳定核心。轻量 Dataset 检查固定在 CPU，移除了接受却不使用的 device 参数；脚本无该参数调用，当前文档命令已同步。相关训练、预训练和评测 shell 入口的 `bash -n` 检查通过；轻量入口帮助及导入隔离复核通过。

类型检查针对 7 个相关源文件通过：

```bash
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m mypy --follow-imports=silent --ignore-missing-imports --check-untyped-defs dexmani_policy/common/checkpoint.py dexmani_policy/common/image_util.py dexmani_policy/datasets/split.py dexmani_policy/training/common/data.py dexmani_policy/deployment/metadata.py dexmani_policy/deployment/runtime.py dexmani_policy/env_runner/sim_runner.py
```

构建检查在临时目录复制当前 pyproject 和 Python 包，使用 `pip wheel --no-deps --no-build-isolation --no-index` 成功构建 wheel，并确认全部 18 份 YAML 包资源存在。环境不能运行 `python -m build`，因此使用已安装的 pip 构建前端，没有安装新依赖或向仓库写入构建产物。

### 未执行及限制

- 未执行长训、真实仿真闭环、实机测试或性能 profiling；合成短训和模拟 rollout 不代表真实任务训练成功、成功率或吞吐改善。
- 不支持带废弃 RGB 参数的产物；本轮按清理要求移除自动适配，读取会明确失败，不提供迁移工具。
- DexLatent 手部 affine 的公共装配已作五模式数学回归；本轮未修改先验与 latent 算法，因此未扩展到依赖官方初始化权重的完整 DexLatent 训练集成。
- agent_checks 完整入口已在 CUDA 通过；未重复运行 CUDA 预训练全集，预训练完整入口使用 CPU。
- 本轮不实施第 12 节研究事项，也不猜测非默认控制频率的仿真 API。最终检查未发现本轮冗余参数、无消费者 helper 或无关文件变更。

## 18. 三项小型收敛及验证（2026-11-03，历史）

起点为 `5698be1d5010230001210e2c887bb3c557ae123e`，HEAD 与指定 review 基线一致，工作区干净。此前已完成的数量语义、统计优化及数据主线保留。

- `load_checkpoint()` 替换“包含 RGB 键一律拒绝”：无键直接读取，true 仅在内存删键，false 明确说明浮点预处理等价性无法保证并提示旧提交。没有增加迁移工具或第二条数值路径。
- `build_dataset()` description 删除 `action_start_index`，Trainer 删除同源等值比较。Dataset/Policy 自身属性、checkpoint/部署动作信息、真实目标/控制宽度和动作窗口检查保留。
- Runtime 直接按 `agent.normalization_dims[name]` 检查和裁剪各点云字段，与 SimRunner 同源；`PolicyInfo.pc_dim` 仍供展示，没有新增持久化映射。外部格式、历史顺序、FP32、连续布局和有限性保持。
- README 与项目架构的相关使用说明已同步；第 17 节保留历史结论，本节仅记录本次执行结果。

使用 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`（Torch 2.4.1+cu124）运行，以下均在 CPU 通过：

```bash
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m dexmani_policy.test_tools.dataset_checks
/home/zhanghaoyang/miniconda3/envs/policy/bin/python - <<'PY'
import tempfile
from pathlib import Path
import torch
from dexmani_policy.test_tools.agent_checks import (
    check_codebook_training, check_deployment_facts, check_sim_observations,
)
torch.manual_seed(7)
torch.set_num_threads(2)
device = torch.device('cpu')
check_sim_observations(device)
with tempfile.TemporaryDirectory(prefix='dexmani-targeted-checks-') as directory:
    root = Path(directory)
    check_deployment_facts(root)
    check_codebook_training(root, device)
PY
```

Dataset 入口覆盖既有 split、sampler、源帧统计、hand 五模式、未来标签、R3D 及 RGB 检查。定向检查覆盖 description 不再传递动作起点；三类点云统一 3 通道时与原设备侧切片及 Runtime 一致；混合 3/6 通道时按各字段消费宽度处理，保留最后 O 帧、连续 FP32，拒绝通道不足及 FP32 转换溢出。

复用 DQ-RISE 小模型与 Zarr/codebook fixture，实际执行 2 次 fresh updates 和 1 次 resume update。无旧键与 true 配方分别经过元信息查看、加载与预测，固定 seed 的输出逐值一致；true 配方还经过 `prepare_resume()` 与 `run_training()` 恢复，保持保存的 100 条预算，删除原 codebook 后不重新读取或拟合。false 在加载、查看、恢复模型和准备续训四个入口均明确失败。true/false 原 checkpoint 文件均核对字节不变；续训写入另一输出目录。

另对四个修改的实现文件运行 `mypy --follow-imports=silent --ignore-missing-imports --check-untyped-defs` 通过；五个修改的 Python 文件 `py_compile` 通过，`git diff --check` 通过。未重跑完整 agent/pretrain 全集、CUDA、长训、仿真闭环、实机、性能或算法实验；上述合成检查不代表任务成功率结论。

### 调用链复查与清理

在上述改动上继续检查 Dataset → 装配 → Policy → checkpoint → resume/inspect/Runtime/评测，以及直接配置、脚本和检查工具：

- `validate_action_window()` 的唯一调用方原来额外传入 `O−1`，现只接收 H/O/A；保留正整数和完整执行窗口约束。已有 `check_policies()` 补上恰好可执行、单步、超长执行窗口、O>H 和非正值边界。
- 点云 Runtime 检查改用实际 `Policy` 小实例提供已声明的 `normalization_dims`，删除给普通 Linear 临时添加字段的类型不一致。删除 DQ-RISE 检查内重复的 `load_codebook` import。
- 扩展类型检查发现并修正了编译返回类型、投影图像尺寸 tuple，以及 DDIM 检查的 lazy import 类型不一致；DDIM 使用与实际 decoder 相同的具体模块导入，算法和数值路径保持。
- 原有 raw/EMA 恢复 fixture 加入 true 配方，复用 `check_normalizer_restore()` 验证部署和评测读取边界均可用，仍只加载一次、释放未选权重、strict 恢复 affine。没有增加模型、适配器、回退或迁移流程。
- 相关链路未发现 SHA/hash、来源凭证或通用版本迁移机制；旧 RGB true 的单点内存删键继续满足本任务的明确读取要求，false 仍拒绝。

本次复查实际重新运行 CPU `dataset_checks`，并定向运行 `check_policies`、`check_sim_observations`、`check_output_boundaries`、`check_evaluation`、`check_image_geometry`、`check_ddim_clipping`、`check_deployment_facts`、`check_normalizer_restore`、`check_codebook_training`，全部通过。真实模型加载和预测使用 CPU；评测检查中的 rollout 为模拟实现，没有运行真实仿真环境。DQ-RISE 再次完成 2 次 fresh updates 和 1 次 resume update。

13 个相关 Python 文件（含完整 `agent_checks.py`）的 mypy 检查通过，未添加 type ignore；`compileall -q dexmani_policy`、相关四个 shell 入口的 `bash -n`、resume/evaluate/eval_pipeline/smoke_test 的 `--help` 及 `git diff --check` 通过。在临时目录使用 `pip wheel --no-deps --no-build-isolation --no-index` 构建通过，wheel 的 18 份 YAML 资源齐全，未安装依赖或修改仓库构建产物。CUDA、完整 agent/pretrain 全集、长训、闭环、实机及性能验证仍未运行。
