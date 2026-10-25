# DexMani Policy 轻量化重构：Codex CLI 任务书

> 状态：四批核心改动已实施；实际验证与未覆盖范围见[验收记录](验证指南.md#本轮轻量化重构验收)。下文保留设计依据与约束，不表示所有方法、设备或性能均已验证。
>
> 审阅基线：[dexmani_policy@332cf9167debb75fee14b02125bd69a06569066b](https://github.com/haoyangzhanglab/dexmani_policy/commit/332cf9167debb75fee14b02125bd69a06569066b)。执行时以实际 HEAD、调用链与保存产物为准。本轮接续[上一轮实施状态](infra_simplification_status.md)，不重复已完成的迁移。[配套 prompt](lightweight_refactor_prompt.md) 可直接交给 Codex CLI。

## 1. 目标与取舍

让仓库适合论文研究：方法容易修改，数据和动作含义清楚，实验可以解释与复现，常见工作流直接可用。优先减少重复事实、跨层知识和隐藏行为；代码行数只是辅助指标。

本轮直接实施四项：**best 加载与选点验真分离；配置事实收敛；Dataset 内部细节停止外泄；方法约束回到方法所属组件。** 异常处理随相关链路一起清理。性能优化须有测量证据，不能用结构重构替代性能验证。

明确保留当前共享 Trainer、单卡/DDP 薄入口、完整 checkpoint、已有 PolicyInfo 和 EvaluationRequest、Zarr 读取与方法差异。不要求新增策略基类、统一 PolicySpec、配置 schema、另一套权重格式、通用 Adapter/Registry、工作流引擎、RPC 或分布式训练框架。没有必要重新组织整个目录。

实施按依赖顺序推进，每批端到端更新调用者。普通实现选择自行解决；只有证据不足且会改变研究语义的关键歧义才询问。环境缺失不阻塞其他独立工作。

## 2. 当前问题与明确落点

以下是审阅基线的源码事实，不把旧方案的“现状”当成当前问题。表内源码路径相对 `dexmani_policy/`。

| 源码位置 | 当前事实 | 本轮处理 |
| --- | --- | --- |
| `common/experiments.py` | 普通 best 解析会进入 selection record，读取全部候选结果并重新排名 | 普通加载读取直接指针；选点和显式验真负责完整证据 |
| `training/validation.py`、`training/builder.py` | 根配置与 agent/dataset 重复表达布局；校验用具体 target 字符串和 Dataset 构造默认值推断行为 | 一个输入来源，一次派生；方法约束委托给实际组件 |
| `training/data.py` | 构建层深入 `replay_buffer`、sampler、子 Dataset 收集数据事实 | 数据事实由 Dataset 提供，训练层只组合并保存 |
| `agents/policies/base.py`、`latent_flow.py` | Policy 接收 Dataset 并读取 split、validation Dataset、sampler；基类了解手部预训练、Sparse3D、Diffusion 特例 | Policy 只接收需要的数据描述；具体约束移到具体方法/组件 |
| `evaluation/rollout.py`、`selection.py` | 异常落盘或 cleanup 的二次异常可能覆盖原始失败 | 保留原始异常，明确记录与资源所有者 |
| `deployment/metadata.py`、`common/checkpoint.py` | 已有无 Torch 检查、保存配置、完整恢复与一致性核对 | 复用现有边界，不再加一层元信息或导出格式 |

已经存在且本轮保留的机制：完整 `future_actions`、共享 RGB 预处理、实际 split manifest、按有效窗口消费源行的统计、评测请求提前解析、固定预算选点、Zarr 惰性访问和块缓存、分块有限值扫描、ConsistencyFlow 的合并计算与条件复用。不要把这些重新列成待开发功能。

## 3. 目标数据流与所有权

| 链路 | 数据流 | 唯一职责 |
| --- | --- | --- |
| 新训练 | 输入配置 → 构建 Dataset/模型 → 保存实际有效配置 → Trainer 调用 Policy | Dataset 解释数据；组件解释自身约束；Trainer 管更新与状态 |
| 完整恢复 | checkpoint 的有效配置与状态 → 恢复构建 → 原采样/更新进度 | 当前默认配置不能重新解释历史实验 |
| 普通推理 | 保存配置 + 明确 checkpoint → 一次加载核对 → Policy → 完整未来动作 → 消费端 | best 只是启动时的选择便利；推理不读取候选评测日志 |
| 正式测试 | 已完成 selection record → 固定 test request → rollout | 研究条件与 seeds 冻结；完整候选验真是独立操作 |

模块内部继续使用现有 dict、Tensor 和小型 dataclass。不要为了这张表建立新的层级。训练与推理共享数值变换的实际实现；训练随机增强只在训练数据路径启用。现场传感器整理、时间对齐、执行队列和运动授权由 `dexmani_real` 管理。

## 4. 必须保持的研究语义

以下是重构验收依据；其他未涉及的模型与配方沿用当前代码。

- **时间与动作**：设 H 为完整预测长度，N 为观测长度，s 为实际动作起点，P=H−s，C 为执行长度。普通路径通常 s=N−1，LatentFlow 为 s=0。必须先完成 H 步方法解码，再输出物理空间 `future_actions[B,P,D_control]`，最后由消费端取 C；不能用 C 改训练 padding、截短 codec 解码或再次裁剪历史前缀。辅助预测维度不发送给执行器。
- **数据与统计**：保留 episode 边界、训练/验证划分、有效窗口、归一化总体及计算精度。普通绝对量按有效训练窗口实际消费的唯一源行统计；LatentFlow 相对量按窗口与原始 reference/q0 统计。q0 不能由增强后的观测反推。多任务零采样权重子任务当前仍参与统计，本轮不改变。
- **物理语义**：不改坐标系、单位、关节顺序、RGB/depth/点云预处理、metric XYZ、action_ee 的 rot6d 处理、控制周期或现有边界 dtype；Real 输出保持 CPU NumPy float64。
- **方法与训练**：不改损失、噪声/时间采样、teacher/EMA 规则、冻结参数、梯度裁剪组或 optimizer 参数覆盖。DDP 训练仍经过包装模型的 forward。保留实际尾部累积组除数、同步位置、每 optimizer update 的 scheduler/EMA 次序及逐 rank 非有限值协调。
- **恢复**：完整保存和恢复 raw/EMA、optimizer、scheduler、更新计数、采样游标、RNG、参数组名称与有效配置。`max_updates` 只截短本次运行，不重算正式训练预算或学习率曲线；不夸大多 worker/异步预取下的逐位恢复保证。
- **配方**：`ddp/dp` 当前有效 batch 为 192，单卡 dp 为 64；这是既有实验配方，不借去重改成相同值。训练、数据划分、选点、正式测试 seed 的职责不合并。
- **评测**：selection/test seeds 不相交，候选使用相同预算与研究条件。排序保持 micro success rate 高优先、成功 episode 的原始首次成功步数均值低优先、global_step 大优先。技术异常是失败的请求，不能伪装成普通任务失败、跳过或换 seed。
- **部署**：普通仿真可采用 best 保存的 C；Real 的执行 C 仍由 Real 执行配置及原有默认规则决定。保持 LAMP/LatentFlow 当前仅 sync、多任务 Real 尚未支持等真实能力，不能靠声明扩充支持范围。加载完整策略时不重新读取原始预训练资产。

## 5. 批次 A：让普通 best 真正成为轻量指针

这是有意的产物协议变化，单独提交并说明。落点包括 `common/experiments.py`、`evaluation/{selection,config,rollout}.py`、有关测试与评测文档。

1. 新 `best_ckpt.json` 直接保存四个字段：`ckpt_relpath`、`global_step`、`inference`、`selection_result`。`inference` 的键沿用 `use_ema`、`inference_steps`、`action_steps`。这些值由完成选点的实际结果写入，不让用户维护第二份配置，不添加版本号、摘要或指纹。
2. 保留 `resolve_best_checkpoint` 的返回形态 `(info, checkpoint_path)`，盘点所有调用者真正使用的字段。普通加载只读直接指针并解析权重路径；`selection_result` 是来源记录，缺少候选日志甚至该来源文件不应阻止普通推理。best 不因此获得“正式 held-out test”资格。
3. 保留必要检查：指针字段/数值有效，权重存在且位于当前 experiment/checkpoints 内，禁止路径逃逸和将 latest/符号链接当作不可变选点结果。实际权重加载沿用配置、normalizer、EMA 等一致性核对；普通仿真 best 将预期 global_step 传入既有 request.source 检查，不额外加载一遍权重。
4. `resolve_selection_record` 负责读入完整、内部一致的选点结论和冻结测试计划：校验状态、选中项、checkpoint、推理设置、任务/seeds、预算及 selection/test 分离。它不在每次正式测试前重新打开所有候选 episode 文件。使用现有种子检查函数；需要复用的小段逻辑才提取函数。
5. 完整候选检查与重排名留在选点生成时和显式离线验真路径。将这些选点专属逻辑移到 evaluation 的自然位置，通用 checkpoint 路径解析不导入它。沿用现有纯排名计算，不新增 audit 管理器。
6. 提供一个小型离线入口：只验证已有 selection record 时只读；用户显式选择“发布 best”时，完整验真后写出新直接指针。它同时解决现有仅含 `selection_result` 的旧指针迁移。命令与参数由实际 CLI 风格确定并写入文档；不要维护永久双协议自动回退，不自动批量改写历史产物。旧指针应给出明确的迁移/显式 checkpoint 使用提示。
7. 先保存完成的 selection record，再原子发布 best。`scripts/remote/sync_down.sh` 现有“权重/事实 → selection → best”顺序保留并核对；普通加载依赖减少不等于可以不保存科研证据。

验收：一份新 best 在候选日志缺失时仍能普通加载；缺权重或非法路径失败；显式验真缺候选证据失败；冻结测试拒绝不一致计划与研究参数覆盖；旧指针不能悄悄使用当前默认推理参数。不得把普通 evaluate 标为 held-out test。

## 6. 批次 B：收敛配置与数据事实，移除跨层窥探

落点：`common/config.py`、`training/{validation,data,builder}.py`、Dataset 及现有 hand representation 工具。先逐字段画清实际读写，再删除重复检查。

### B1. 配置输入只定义一次

- 沿用现有公开配置键与 Hydra/OmegaConf。H、N、训练动作窗口、action_dim 等根字段作为用户输入，agent/dataset 中同义值由引用或构建阶段派生。若派生字段仍接受外部 override，就必须在入口拒绝冲突；不能删校验后让两个值悄悄分歧。
- 合并与默认值绑定集中在现有准备/构建流程，构建完成后仍保存实际采用的默认值和有效值。避免在训练、smoke、评测、部署各实现一套默认规则；不要求把所有 YAML 改成新的 model/data/train 层级。
- 静态配置检查不读取数据、权重或构造昂贵模型；实际数据检查在数据打开处，实际模型/normalizer 检查在构建或加载处。组件只接收检查所需的窄配置，不通传整个实验对象再任意修改。
- 去掉通过完整 `_target_` 字符串认定 LatentFlow 语义等做法。利用现有组件方法/少量明确属性表达动作起点、数据要求与能力；没有需要时不引入工厂或注册表。Python/Hydra 自然报告的参数错误不再重复反射验证多遍。
- 完整恢复继续以 checkpoint 的 `effective_config` 为准。`config.yaml` 保留无 Torch 检查用途；加载时一次核对它与 checkpoint 中真正影响推理的内容，不删成任意 YAML 都能配任意权重。派生事实不能重新变成可编辑输入。

### B2. Dataset 提供事实，Policy 不接收 Dataset

- Dataset 负责产生自身实际数据描述：布局/时间、split manifest、有效窗口/源行统计、动作统计总体，以及必要的手部/真机数值配方。优先复用 `data_recipe`、`hand_data` 和已有函数；确需聚合时只用一个小型只读视图或普通 dict，不创建通用 DatasetSpec。
- 将 `training/data.py` 里读取 buffer、sampler、子 Dataset 内部状态的事实提取移到相应 Dataset。训练构建层保留数据创建、normalizer 拟合、loader 和保存配置等职责；不得为生成描述再次扫描全量 Zarr、额外构造验证 Dataset 或重复统计。
- 将 `training_data_config(dataset, ...)` 改为接收已经提取的最小描述；同步 DQ-RISE、LAMP、LatentFlow 和 builder。Policy 不再知道 Zarr 路径组织、sampler 或 `get_validation_dataset()`，仍检查实际需要的先验/codec 兼容性。
- MultiTaskDataset 在自身构建处验证共享布局并汇总子任务事实；输入任务顺序、采样权重、normalizer 总体和训练增强语义保持不变。不要给单任务和多任务各加一套相似协议。
- 逐字段核对 `data_identity` 与 `data_recipe` 的使用。完全重复的部分合并；独有的防止静默错配事实须保留。路径用于定位，物理/时间/布局与 split 用于语义比较，不能因为数据搬家就发明哈希身份，也不能仅凭路径相同认定数据没变。重生成数据仍使用新路径。

验收：Policy 不再收到 Dataset；构建层不再遍历其内部实现收集同一批事实；重复配置只有一个可修改来源；config-only 保持轻量；新训练、保存后推理、完整恢复读取的是同一套实际事实。保持历史 checkpoint 恢复所需的字段与参数名；若某字段无从等价保留，明确限制并保留原实现，不能按当前默认值猜测。

## 7. 批次 C：基类变薄，数值路径不重写

落点：`agents/policies/base.py`、相关具体 Policy、encoder/decoder、已有 hand representation 函数及少量构建调用者。

- BasePolicy 保留已经共用的预处理、normalizer、损失/预测调用与动作出口。不要为了“纯接口”把同一算法复制到九个 Policy，也不强制增加 ContinuousPolicy 等中间父类。
- Sparse3D 的 metric XYZ 约束归所属 encoder/方法；Diffusion clipping/RTC 条件归 decoder 及真正支持的策略；DQ/LAMP 的先验数据约束归手部方法。复用现有组件检查入口，不建立层层转发的 validator 体系。
- 执行模式由真实支持路径明确声明，替换通过 `type(self).method is BasePolicy.method` 猜能力的逻辑；能力仍写入既有 `policy_metadata`。核对每个实现，不能仅因继承同一父类就启用 RTC/async。
- 训练与部署的确定性输入变换使用同一实现；避免新增训练专用和部署专用的 normalization/RGB/relative-action 副本。数据增强和现场原始输入整理保留各自职责。不强行将全部模态变成一个庞大的 FeatureTransform 类。
- 静态布局只在合适的构建/加载边界确认；公开 batch/obs 入口仍保留必要的实际输入形状检查，同一调用链不在每一层重复检查。动态非有限值和现场输入/输出约束不能用“首次检查通过”缓存跳过。
- 不移动导致 `state_dict` 键变化的已注册子模块，不改变继承关系来制造 checkpoint 迁移。normalizer、codec、teacher、action_context 和推理随机数语义保持原样。

验收：基类不再直接识别 Sparse3D 或手部预训练数据结构；能力判断不再比较方法对象身份；原配置与保存权重可用；固定输入/权重/随机状态下，相关损失、动作及梯度与基线一致。容差按实际 dtype/算子说明，不能以放宽容差掩盖算法变化。

## 8. 批次 D：异常与易用性收尾；性能按证据处理

### D1. 只保留有职责的异常处理

- 配置/数据/权重错误在首次读取的边界清楚失败；内部操作依赖已成立的约定，不包多层 try/except，不补静默默认值。
- Dataset 不因坏样本随机换索引；评测不因异常跳过或替换 seed；模型不因缺 EMA 悄悄用 raw。
- 允许顶层任务为记录失败或处理退出捕获异常，但必须保留原始 traceback 并重新抛出。记录失败或 cleanup 再失败时，原始计算异常仍是主要失败，二次失败须可见；仅有 cleanup 失败时不能报告成功。不要依赖 Python 3.11 专属机制解决 Python 3.10 项目的问题。
- 环境、视频、loader、worker、模型资源各由创建它的层关闭。保留必要的 `finally`/上下文管理器，不为减少行数删除关闭行为。不要为短脚本新建异常类层级、自动重试器或资源管理框架。
- 结果只有预期 episode 全部完成才能是 completed；失败保留已有 episode 事实，不发布部分成功率。

### D2. 只修实际不便，不扩展功能矩阵

- 同步 README、架构、评测和验证文档中的已变入口，保留一处权威说明，其他位置链接。明确新 best 与旧指针的使用方式、普通评测与正式测试的差别。
- 盘点 `policy_name=ddp/dp` 这样的“方法名包含启动方式”问题。本轮不强制重命名所有实验，也不在本仓库单方面改变 Real 的路径契约；没有配套工作区时保留现有公共接口并记录实际缺口。若进行新 run 命名修复，必须核对目录、远端脚本、日志与真实消费者，保持历史路径身份和 batch 配方；禁止简单把 `/` 替换成可能碰撞的名字。
- 不把训练日志采样值改名成区间均值而不改计算。发现口径错误就如实修正说明，改变统计计算须单独说明。

### D3. 条件性性能工作

先测相关瓶颈，再决定是否改代码。区分冷启动、数据准备、稳定训练 update、推理编码和采样耗时；记录模型、数据规模、batch、H/N/C、NFE、dtype、设备与 warmup。GPU 计时同步只用于测量，不插进每步训练热路径。

优先调查已观察到的重复读取/复制、无用 CPU/GPU 同步或重复编码；一次采样内可复用不变条件，不能跨观测复用过期特征，训练时也不能误缓存需梯度的结果。已有块缓存、合并前向和 KV 复用先核对，别重复开发。没有硬件/数据或没有稳定收益就保持实现，报告 NOT VERIFIED；不要求新增 benchmark 平台，不承诺未经测量的加速比例。

## 9. 验证与完成标准

先检查工作区、环境和现有测试，建立小的改动前后对照。优先扩展已有测试，覆盖真实行为变化；不要用大量 mock 或源码字符串断言代替实际执行，也不要为删除冷路径重复检查写一套镜像测试。

| 改动风险 | 最小有意义的验证 |
| --- | --- |
| best 与 selection 分离 | 普通加载不依赖候选；缺权重/非法路径失败；显式验真拒绝不完整证据；冻结计划拒绝越权覆盖；排序和平局保持 |
| 配置与数据描述 | 代表性配置、冲突 override、单/多任务、split 和 normalizer 总体；LatentFlow raw q0；保存事实与实际对象一致 |
| Policy/组件边界 | 受影响方法的真实小模型 loss/predict；H/P/C 与辅助维度；raw/EMA 保存恢复；缺预训练原始路径仍可恢复完整策略 |
| 训练与恢复受到影响 | 累积尾组、参数覆盖、teacher/EMA、短更新与恢复后的计数/LR/状态；改动 DDP 时补有资源支持的实际双 rank 验证 |
| 异常与资源关闭 | 计算失败、记录失败、cleanup 失败的主次错误；失败结果不伪装完整；已有资源关闭测试 |
| 部署公共接口受到影响 | 无 Torch inspection、Runtime 输入输出/维度/dtype、Real 离线 preflight；不启动设备运动 |

已有命令从仓库根目录运行，执行前对照[验证指南](验证指南.md)及当前参数：

```bash
python -m compileall -q dexmani_policy
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat r3d maniflow multitask_dit lamp latent_flow
python dexmani_policy/smoke_test.py --config-only ddp/dp ddp/dqrise ddp/maniflow ddp/multitask_dit ddp/r3d ddp/sat
python -m unittest discover -s tests -p test_infrastructure.py -v
python -m unittest discover -s tests -p test_training_contracts.py -v
```

按影响范围补充现有方法测试；公共链路改动完成后，有可用依赖时再运行一次完整现有单元测试集：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

有相应数据/权重/设备时，再针对受影响代表方法运行：

```bash
python dexmani_policy/smoke_test.py --data-only dp3 --max-batches 2
python dexmani_policy/smoke_test.py --data-only latent_flow --max-batches 2
python dexmani_policy/smoke_test.py dp3 --max-updates 4
```

smoke 不接收 Hydra 字段 override；data-only 的 batch 上限不限制前置扫描/统计；DDP config-only 不等于多卡运行验证。不要通过改正式数据、精度、NFE、总预算或有效 batch 来让检查通过。短训练最多 4 个更新；实际 DDP 验证仅在改动需要且现有资源可用时进行，不启动完整实验。Real 工作区可用时按其自身说明运行 `python scripts/check_deployment.py` 等离线检查，不连接或驱动设备。

受依赖、数据、权重、GPU、模拟器或另一个仓库限制的项目标为 **NOT VERIFIED**，写清缺口与最短补验命令；只有实际执行并通过的检查才标 **PASS**。语法检查、mock 或本次文档审阅不能冒充训练/数值/成功率/性能证明。

完成判据：四项核心改动端到端闭合；公开调用者同步；研究不变量未变；重复事实和跨层依赖实际减少；相关验证有真实结果；协议变化有明确使用说明。若某项保留原实现，说明具体依赖和证据，不能把未完成工作标成完成。仅移动文件而未减少依赖不算简化；如果新增转发和状态超过消除的复杂度，应采用更小实现。无需再创建庞大的状态/规则文档，最终用一张简短执行表记录结果即可。

## 10. 执行边界

先读当前 AGENTS.md 与相关局部规则，检查实际 HEAD、dirty diff 和依赖；必要时使用独立分支/worktree，不能 reset、覆盖或丢弃用户改动。每批先查真实调用者再动手；保持行为的重构与 best 产物协议变化分开提交。可以创建本地提交；本任务的代码执行阶段不自动 push/merge。

`dexmani_real` 不是本轮默认重构对象。只核对相关调用边界；若确需破坏其公共接口，须有可用、获授权的配套工作区并成对实现验证，否则保留接口，继续独立部分。不要因一个外部仓库不可访问而停下全部工作。

不修改无关模型/损失/数据配方，不引入依赖升级或预训练权重下载，不启动 SSH 任务、完整训练、长评测、录像或真机运动，不改写历史实验与权重。禁止自建哈希、指纹、兼容框架、运行时环境准入体系或失败自动恢复。必要的路径保护、输出防覆盖、科学语义检查和动态输入检查保留。

交付用中文说明：实际删除/合并的责任与依赖、对外行为变化、改动文件、已运行命令及结果、NOT VERIFIED 与后续补验。不能仅给建议、停在计划或泛化调研。

## 11. 官方参考：借鉴机制，不照搬规模

以下固定源码版本用于解释设计取舍，不作为新增依赖，也不要求执行者再次完整调研五个仓库。

| 官方仓库与核对文件 | 借鉴 | 不照搬 |
| --- | --- | --- |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/base_image_policy.py)；[训练 workspace](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py) | 少量明确 Policy 接口，normalizer 随策略保存 | 按每个方法再复制一份训练循环 |
| [ManiFlow Policy](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py) | teacher、目标与条件在方法内显式表达 | 复制全部 workspace 或退回已被本仓库优化的计算路径 |
| [R3D Policy](https://github.com/Wushr-Lance/R3D-Policy/blob/e637c0148376ddc4b5e667fa8f8e108cb8ff7a85/R3D/r3d/policy/dp3.py) | 顺着实际配置与输入处理追踪动作生成 | 根据方法名猜契约，或把其数值归一化直接套到 metric XYZ |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/e08ba78888b233b77ed2bb2d92886a52360a1081/model_template.py)；[DP 部署](https://github.com/XPolicyLab/XPolicyLab/blob/e08ba78888b233b77ed2bb2d92886a52360a1081/policy/DP/deploy.py) | 外部评测接口处保持薄适配，动作消费与模型职责清楚 | 把 benchmark adapter 变成内部平台，统一全部仿真/真机执行 |
| [LingBot-VLA v1 数据变换](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/data/vla_data/utils.py)；[部署](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/deploy/lingbot_vla_policy.py)；[训练入口](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/tasks/vla/train_lingbotvla.py) | 训练和推理实际复用 FeatureTransform；模型/数据/训练职责可辨；调用链直接可追踪 | 大模型 FSDP/模型并行/分布式 checkpoint 设施，以及它们对小型研究仓库的额外负担 |

此处核对的是 **LingBot-VLA v1 官方仓库**，不混用 LingBot-VA 或 LingBot-VLA v2。其[采样实现](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/models/vla/pi0/modeling_lingbot_vla.py#L1744)在一次采样中复用图像/语言 prefix KV，可借鉴“只算一次不变条件”；DexMani 已有相关复用，应先核对而不是重建缓存层。其[Dataset](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/data/vla_data/base_dataset.py#L176)捕获取样异常后随机换样本的做法不适用于本项目的固定实验总体；训练中每步同步和大模型吞吐宣称也不能直接转化为 DexMani 的优化或收益结论。
