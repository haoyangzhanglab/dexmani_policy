# 第二阶段任务书：研究代码的简化与收敛

> 仓库：`haoyangzhanglab/dexmani_policy`  
> 审查基线：`e4196bf7f04ad6b409434cbc37f218dbe9c5c944`  
> 制定日期：2026-11-02  
> 状态：待实施；本文的验收项尚未执行。  
> 前置工作：[数据集重构任务书](dataset_refactor_task.md)。前一阶段已完成的行为继续保留，本文只规定两轮后续审查中值得实施的收敛工作。

## 1. 目标与范围

把项目保持为个人机器人研究代码：轻而薄、正确好用、简洁高效统一。主要减少恢复、评测、数据加工和预训练衔接中的重复状态、来源门槛、目录限制、重复校验和失败记录流程，而不重写已可用的学习算法。

实施后的主要变化：

1. 训练恢复只依赖 checkpoint 中的训练状态，不解析、修复或截断历史指标日志。
2. 数值事实按实际消费关系传递，不再用 `format == "dexmani.real.canonical"` 决定是否保留部署信息。
3. `contact_map.py` 只产生 CordViP 实际使用的自包含数据，保存固定的 XYZ/FP32 表示和必要事实，不复制通用来源元数据。
4. normalizer 的张量形状从已加载的权重取得，不在配置中重复保存；删除多余的恢复标记和配置冻结。
5. 评测使用直接的顺序流程，只保存完成结果；独立测试完成后再原子更新 best。
6. 检查留在承担其语义的边界，内部直接传值；删除没有消费者的参数、字段和文档要求。

本轮不增加数据格式版本、迁移框架、验证等级、配置开关矩阵、产物注册表、来源证明链或通用任务运行器。不以行数下降为唯一目标，也不承诺未经测量的训练加速或成功率提升。

## 2. 已完成设计：继续保留，不重新推翻

| 行为 | 本轮要求 |
| --- | --- |
| 数据链路 | 保持 `EpisodeStore → SequenceSampler → Dataset → DataLoader`；不全量加载视觉 replay buffer |
| 数量划分 | 保持 `max_train_episodes` / `val_episodes`；150 条、M=100、V=20 得到 100 train、20 val、30 unused |
| 数量不足 | M 非空时不为 val 缩小 train；组合不可满足时明确报错；val 不等于 `~train_mask` |
| 归一化 | fresh 只从实际 train episodes 的原始消费字段拟合；不计 padding、重复窗口、val、unused 或增强后的样本 |
| 继承与恢复 | 预训练 affine 与 identity 均继承；resume 使用保存的 normalizer，不重新拟合 |
| 验证集 | 主策略不建立 val loader；预训练默认 `val_episodes=0`；无 val 不额外遍历 train 来选 best |
| 预训练产物 | AFRO/CordViP 继续导出 encoder、normalizer 和输入 normalization；运行时不加载预训练 teacher、预测头或 optimizer |
| 手部码表 | NPZ 保持 `sorted_hand_poses`、`hand_normalizer_scale`、`hand_normalizer_offset` 三数组；实际表决定 K |
| 完整策略恢复 | checkpoint 包含实际模型结构、raw/EMA、normalizer、码表及训练状态；恢复不访问最初的 encoder/NPZ |
| 时间和物理含义 | 保留 action 窗口、R3D 辅助目标、未来触觉有效 mask、关节顺序、单位、坐标系和点槽对应关系 |

这些要求是可检查的行为，不需要新增一个“契约系统”来实现。旧任务书中的历史实施记录不是本轮测试结果。

## 3. 校验与信息的取舍

### 3.1 每类检查有清楚的归属

| 边界 | 保留的检查 | 不再做的工作 |
| --- | --- | --- |
| Zarr 打开 | `episode_ends` 合法、所选数组时间轴一致、支持的数值 dtype | 不扫描无关数组、不要求来源 ID 或导出工具身份 |
| Dataset / 具体预训练输入 | 本方法实际需要的 shape、非空点云、动作布局、几何与图像一致性 | 不重复验证 Store 已保证的 episode/time 结构 |
| 数量划分 | 正确的数量范围与是否有足够 episodes | 调用方不重复校验由算法保证的互斥性 |
| 配置装配 | 实际输入是否齐全、动作/坐标语义、会改变结果的冲突 | 不冻结整个配置，不依赖额外 resume 布尔/路径标记 |
| 外部权重 / 码表导入 | 原生 strict 加载；affine 可逆且有限；实际 K、维度与表值合法 | 不要求同任务、同数据来源、同 train split 或预训练选模记录 |
| 训练恢复 | strict 模型加载、optimizer 参数组顺序、必要动作/输入语义一致 | 不审计日志，不比较来源证明或复制一份张量 shape metadata |
| 推理消费 | 实际输入/输出窗口、物理动作有限性、真实使用的坐标语义 | 不因元数据缺少 `format` / `source_id` 拒绝纯推理 |

不同外部入口仍各自承担自己的边界。例如 contact 导出器和独立读取任意 CordViP Zarr 的 Dataset 都可以检查各自输入的 shape；这不是同一调用栈中无意义地重复检查。

不要为压缩代码把清楚的错误改成静默跳过、补零、默认 identity、`strict=False`、返回成功率 0 或捕获异常后继续训练。由 Python、NumPy、PyTorch、Zarr 已能清楚报告的缺键、文件不存在和张量不匹配，优先直接传播。

### 3.2 不按名字批量删除 metadata / source / evidence

| 信息 | 决策与理由 |
| --- | --- |
| `normalization_dims` | 保留。identity 字段没有 affine tensor，仍需要输入宽度和实际消费字段 |
| `normalizer_shapes` | 删除持久化副本；从选中的 state_dict 中的 scale tensors 直接推导 |
| `policy_metadata` | 保留实际消费的动作窗口、控制维度、执行模式等，不因为名字叫 metadata 就删除 |
| `deployment_metadata` | 保留实际控制周期、点云配方、depth 尺度和指尖名称；删除 Canonical 身份门槛 |
| encoder 的 `normalization` | 保留小字典，表达预训练输入变换；不扩展为 provenance 协议 |
| `source_id`、`episode_ids`、`trial_ids` | 不进入划分、恢复、导入和部署判定；contact 产物不继续传播 |
| `initialization.encoder/rgb/codebook/...` | 保留 fresh 初始化所需的真实文件或模型位置；它们不是来源认证 |
| `tactile_source` | 保留模型/Runner 对 real、sim 的数值解释；删除 Dataset 中只校验和转发的副本 |
| 手部单位和顺序声明 | 已存在且明确与实际输入约定矛盾时，保持边界拒绝；不把声明缺省升级成来源证明要求 |
| seeds、推理设置、候选成绩 | 保留在实验结果中，用来解释和复现实验；不当作产物导入凭证 |
| commit/source/evidence 手工表格 | 不作为正常使用的必填清单；复现笔记按研究需要记录即可 |

基线中没有独立的 evidence 验证子系统，也没有待删除的通用数据 hash/manifest 框架。不要为了“统一删除 evidence”发明一个新的分类器、审计表或扫描设施。

## 4. 恢复和日志：checkpoint 是唯一进度来源

涉及 `training/common/logging.py`、`training/common/resume.py`、`training/common/config.py`、`training/trainer.py` 及预训练调用方。

### 4.1 删除历史日志修复

当前 `ExperimentLogger` 在 resume 时读取整个 `metrics.jsonl`，逐行解析，保留 `step <= resume_step` 的记录，并备份、重写文件。它使日志长度影响恢复开销，也会让一条损坏记录阻止恢复。此前审查已复现 malformed JSON 导致 logger 初始化抛出 `JSONDecodeError`。

目标实现：

- 恢复直接追加 metrics，模型进度只来自 checkpoint 的 `global_step`。
- 删除历史全文读取、JSON 逐行校验、截断、备份和恢复时的原子重写。
- logger 只需要知道 fresh / resume，可改为一个 `resume: bool = False` 参数；不要继续传入一个不参与任何工作流的 `resume_step`。
- 为避免已有文件最后一行缺少换行时与新记录粘连，可在 resume 打开时检查最后一个字节并补一个换行。只维护追加边界，不解析或修复旧记录，不引入恢复状态文件。
- 历史损坏行原样保留；新产生的每条记录保持独立有效 JSON。训练不以日志内容是否完整为启动条件。
- 允许恢复后出现重复 step 或较小 step。文档明确这是追加式运行历史；不为强制单调性增加 run UUID、dedup 或日志迁移系统。
- fresh 仍防止覆盖已有 `config.yaml`、指标和 checkpoint；W&B 继续沿用当前每次继续训练建立新 logging run 的行为。

旧指标不是 checkpoint 的事务日志。删除修复流程后，也删除只为证明修复、备份、截断正确而存在的测试。

### 4.2 删除重复恢复标记和全局冻结

- `run_training(..., checkpoint=...)` / 预训练的实际 checkpoint 参数决定 fresh 或 resume。
- 删除 `cfg.resume_from` 的内部注入和只检查该标记的防御分支；已有恢复 CLI 仍负责真正读取 checkpoint。
- 保留观测预训练的 `--resume-from` 文件参数。它是实际 I/O 入口，与冗余的 `cfg.resume_from` 标记不同。
- 保留当前恢复入口从 checkpoint 配方重建训练，允许设备、worker、日志间隔、原样数据搬迁路径等执行参数覆盖。不要把“继续训练”扩展成任意改算法的启动器。
- 删除 `OmegaConf.set_readonly(effective, True)`；仍在保存前解析配置，并生成独立的实际运行配置快照，避免之后的对象修改污染已保存状态。
- 从新保存的配置中移除已退休的内部 marker 和重复字段；不新增 schema version、迁移器或逐版本 fallback。

继续使用同一、未重写和未重排的数据是研究者的实验责任。原样搬迁可改路径；改变内容或 episode 顺序应开启新实验。不引入数据指纹证明这一责任。

## 5. normalizer、输入和部署事实只保留一条数据流

### 5.1 从权重推导 normalizer 形状

涉及 `common/normalizer.py`、`common/checkpoint.py`、`training/common/config.py`、`training/trainer.py` 和相关检查。

- 新 checkpoint 的 effective config 不再保存 `normalizer_shapes`。
- 训练恢复从 `checkpoint.model_state`，推理恢复从实际选择的 raw 或 EMA state，提取 `normalizer.fields.<field>.scale` 的 shape，再创建 `FieldNormalizer`，最后执行原生 strict 加载。
- 两处真实复用可以共享一个小函数；它只做已知前缀与后缀的键提取，不尝试推断多种历史布局。
- 必须包含 `hand_latent` 等真实存在于 normalizer、但不属于观测字段的条目；不能仅遍历 `normalization_dims`。
- 保留 `normalization_dims`。不能把“没有 affine”误解为“没有该输入”，尤其是 identity 点云和相机几何字段。
- strict 加载继续发现缺失的 scale/offset、两者形状不匹配、模型结构不匹配和多余参数；它不证明数据的单位或坐标语义。外部 affine 的有限性与非零 scale 检查保留在已有导入边界。
- e4196bf 的 checkpoint 若仍有额外的旧 `normalizer_shapes` 配置键，读取时无需使用它；不需要为了一个不再消费的字段开发兼容机制。

### 5.2 移除 Canonical 格式门槛

修改 `training/common/data.py:build_dataset`。当前有 `dt` 等真实事实但没有 `format` 标签的数据，会得到空的 `deployment_metadata`；这是需要修正的行为。

按已有事实和实际消费模态保存：

| checkpoint 字段 | 来源 |
| --- | --- |
| `control_dt_s` | 存在的 Zarr `dt` |
| `pointcloud_config` | 消费 `point_cloud` 时，已有的同名 attr |
| `hand_pointcloud_config` | 消费 `hand_pointcloud` 时，已有的同名 attr |
| `fingertip_link_names` | 消费 `fingertip_xyz` 时，已有的同名 attr |
| `depth_config.shape` | 消费 depth 时，实际训练数组的非时间维 shape |
| `depth_config.depth_scale_m` | 已有的 `depth_scale_m_per_unit`，缺失则保持缺失/None |

删除 `format == "dexmani.real.canonical"` 条件。不要用另一个 source/schema 标签替代它，不猜测缺失的周期、单位、坐标系或模板配方。

数据装配可先读到声明字段事实，在知道模型 `consumed_observation_fields` 后只保存相关部署项；用小段直接筛选完成，不新建 metadata manager。无用模态的事实不应变成真实部署的必需输入。

纯 inspect 和离线推理可以缺少硬件事实。只有实际硬件消费者需要某个事实时才在该边界检查。保留 `PolicyInfo` 作为现有公开接口，不把这次任务扩展为重写 Real 项目或新的部署配置体系。

### 5.3 解开通用归一化函数与预训练 Dataset 的耦合

- `fit_normalizer` 不再 import / `isinstance(..., ObservationPretrainDataset)`。
- 它依赖实际使用的 `iter_normalization_data`、模式、维度和初始 affine；这些普通参数已经足够。
- 预训练入口在 fresh 装配处检查一次 normalization 声明是否覆盖该方法字段；训练调用方始终传入 train Dataset。不要把同一检查复制到 Dataset、模型和 fitter 三层。
- 删除内部 `dataset.training` 防御判断，不加入新的 Protocol、抽象基类、normalizer provider 或方法注册表。
- identity 零统计读取、已有 affine 优先、action identity 的显式可逆 affine、`action_ee` 旋转部分处理保持不变。

## 6. `contact_map.py`：单一、干净的 CordViP 数据产物

### 6.1 唯一默认产物

保留独立自包含 Zarr。当前 `configs/cordvip.yaml` 的主策略默认读取 `.cordvip.zarr`，因此只导出 contact 标签会破坏现有主策略训练。不要引入 overlay、sidecar、两个 Store 联接或 `pretrain-only` / `full` 模式来解决本来不存在的需求。

输出固定为：

| 路径 | shape / dtype | 含义 |
| --- | --- | --- |
| `meta/episode_ends` | `[E]`, int64 | 原 episode 边界 |
| `data/hand_pointcloud` | `[T, Nh, 3]`, float32 | 手点云 XYZ，Nh > 0 |
| `data/object_pointcloud` | `[T, 1024, 3]`, float32 | 物体 XYZ，点序原样保持 |
| `data/joint_state` | `[T, 19]`, float32 | 本项目 arm7 + hand12 的当前状态 |
| `data/contact_map` | `[T, 1024]`, float32 | 与同一帧物体点槽一一对应 |
| `data/action` | `[T, 19]`, float32，仅源数据存在时 | 保留主策略使用的原动作及原时间对齐 |

源点云允许 C≥3，只读并保存前三个 XYZ 通道。删除复制无用颜色/附加列、原 dtype、每个数组 attrs 和原压缩器的通用逻辑。使用当前 Zarr 环境的统一默认压缩设置和固定行 chunk，不新增 codec 选择器。

源数据有 action 就保留；没有 action 的产物可用于 CordViP 预训练，主策略仍会在读取所需 action 时明确失败。不要从 joint_state 伪造 action，也不要复制官方转换器中不适用于本项目的关节排序或时间移位。

### 6.2 最小 attrs

在第 5.2 节去掉 Canonical gate 后，contact 产物只按存在情况保留当前 CordViP 部署链路实际消费的 `dt` 和 `hand_pointcloud_config`。不复制其余根 attrs 或数组 attrs。

特别删除当前 `retained_attrs` 中的 source/episode/trial 身份、`format`、task 标签和未被此产物消费的描述性字段。输入的单位、关节顺序和坐标关系由本项目数据约定描述，导出器不推测或转换它们；只改变存储通道与 dtype。

不得把任意源 attrs 透传作为“以后可能需要”的替代方案。若实施时发现当前实际消费者确实读取某个遗漏字段，定位调用方并保留这一字段即可，不推广为元数据协议。

### 6.3 数学与时间含义不变

保持现有几何计算：Open3D KNN=30 法线、`gamma=1`、`theta=10`、距离分母 `+1e-8`、法线点积绝对值、每个物体点对手点的最小加权距离，再映射 contact 值。官方函数签名中的 `gamma=0.2` 不是官方数据处理入口实际使用值，不能据此改成本项目默认值。

- 不增加法线朝向修正：当前绝对点积对法线符号不敏感。
- 不改为普通最近点距离，不改为二值标签，不改变物体点顺序，不做 FPS 或重新采样。
- 保留逐帧标签与原 `episode_ends`；不移动 action、状态或 contact 的时间轴。
- 几何使用实际输出的 FP32 XYZ；核心距离累积沿用现有 FP64，最终标签转 FP32。不要为了减少几行 cast 顺带修改数值精度。

### 6.4 保留真正有用的资源和内存控制

- 保留有限行 chunk 和物体点块计算，避免全序列或巨大点对矩阵常驻内存。
- 保留默认串行；显式 `num_workers > 1` 时继续使用 spawn。Open3D/OpenMP 不安全的 fork 继承是真实问题，不应为了“统一 multiprocessing”改回 fork。
- 保留单个临时目录、成功后 rename、失败清理；既有输出不覆盖。不增加锁、重试、恢复 manifest 或进度数据库。
- 输入结构在打开/构造时检查一次；实际 XYZ 分块转为 FP32 后检查有限性，避免异常坐标进入 Open3D。不要每个内层距离函数再重复同一检查。
- 单次发现源字段并限定真正读取的数组即可。不要为了消除一次轻量 Zarr 打开而重构整个 `EpisodeStore` 或加入句柄池。

目标是让脚本内容直观地对应“读所需字段 → 计算标签 → 写固定产物”，不是强制单函数、删光 `try/finally` 或牺牲有界内存。

## 7. 评测与 best 发布：直线流程，不维护任务状态机

涉及 `evaluation/rollout.py`、`evaluation/selection.py`、`eval_pipeline.py`、`deployment/metadata.py`、`common/io_util.py`。

### 7.1 完成结果才落盘

- 单 checkpoint 评测：加载模型 → 运行 episodes → 汇总 → 原子保存一次 `result_details.json`。
- 删除 running/completed/failed 状态转换、预写空结果和失败 traceback JSON；错误保留原异常传播到终端/已有任务日志。
- 保留结果中的 checkpoint、step、实际推理设置、seeds、逐 episode 成绩和汇总。这些是实验结果，不是冗余 evidence。
- Torch/CUDA/dtype 等一次性运行信息不作为验收或导入条件；无需为删掉几项静态信息扩大本次改动。
- 失败可以留下本次目录和已生成的视频，不声称结果完整。不要再开发失败目录回收或断点续评机制。
- 保留环境、episode generator 等资源关闭；失败不能变成成功率 0。

### 7.2 流水线的固定顺序

1. 解析明确的候选 checkpoint；解析一次公用推理设置和互斥 seed 集合。
2. 每个候选在相同 selection seeds、推理和执行参数下评测。
3. 沿用当前排序规则选出胜者：success_rate 较高，avg_steps 较低，global_step 较大。
4. 对胜者使用独立 test seeds 运行一次测试。
5. 保存完整的 `selection_result.json`，包含候选结果、胜者、推理设置、两组 seeds 和 test 结果位置。
6. 最后原子更新 `best_ckpt.json`。

测试成功执行但得分低，仍发布原 selection 胜者；不依据 test 分数重新选模或添加发布阈值。test 抛异常或完整结果保存失败时，不更新旧 best，也不写出一个看似完整的 selection 结果。之前已完成的候选结果可以保留。

`select_checkpoint` 当前是内部函数，可以收敛到 `run_pipeline` 内的一段直接代码或一个无状态小 helper；不要为了保留历史调用层级继续维护跨函数 record 状态。

### 7.3 删除目录归属证明，明确输出位置

- `inspect_policy(experiment, checkpoint=...)` 保留 best/latest/百分比快捷 selector。
- 显式 checkpoint 路径允许位于 experiment/checkpoints 之外，且允许指向真实文件的 symlink；删除 `is_relative_to(checkpoint_dir)` 限制。仍要求最终目标是存在的文件。
- 显式绝对路径直接使用；已有相对文件名快捷方式继续相对 experiment/checkpoints 解释。不要尝试多目录递归搜索或根据文件名猜格式。
- 评测 `--output-dir` 可在任意指定位置。保留防止覆盖非空结果目录的检查，删除必须位于训练 run 内的检查。
- selection 接受明确候选路径，不用“位于同一 checkpoints 目录”证明候选可比较。此流程仍用于研究者明确选定的同任务、同评测协议候选，不扩展成跨任务 benchmark。
- 给 `eval_pipeline` 增加一个用途单一的可选 `--run-dir`：指定 best 发布目录及默认结果目录的锚点。未指定时沿用当前从首个候选父目录的父目录取得 run 的默认；使用任意外部候选布局时显式指定。这个参数解决输出归属，不参与来源认证或候选身份检查。
- run 目录按普通输出目录创建；其位置不决定候选是否合法。默认结果仍放在 run 的独立子目录，显式 `--output-dir` 则优先使用指定位置。
- 用一个明确基准解释保存路径：best 中 `ckpt_relpath` 相对 run；selection 记录内路径相对 selection 结果目录。可用 `os.path.relpath` 表达外部位置的 `..`，不要再用 `Path.relative_to` 强制包含关系。文件关系改变后的定位由用户更新，不加入搜索或路径修复。
- `best_ckpt.json` 只保留现有部署实际读取的 `ckpt_relpath` 与 `inference`。global_step、selection_result 自引用等留在完整实验结果即可，不重复放入运行时指针。

best 的最小结构仍沿用已有字段名：

```json
{
  "ckpt_relpath": "checkpoints/epoch=0001-step=00000100-example.pt",
  "inference": {
    "use_ema": true,
    "inference_steps": 10,
    "action_steps": 6
  }
}
```

不要因为路径现在更宽松就新增配置指纹或完整 effective_config 等价检查。推理窗口、EMA 是否存在、seed 池可用性和实际 Runner 输入语义仍由既有消费者检查。不同候选是否属于同一可比实验，是明确选择候选时的研究责任，不能靠目录归属证明。

### 7.4 不提供同一 run 的并发发布协调

删除 `.pipeline.lock`、`fcntl` 依赖及专用锁异常处理。个人研究使用一次 pipeline 发布一个 run；不同任务使用独立结果目录。

best 仍原子替换，因此读者不会读取到半个 JSON；这不等于为同一 run 的并发实验定义优先级。如果用户并发启动同一发布目录，最后一个成功写入者覆盖指针。文档说明一次即可，不为防止这种误用再添加 ownership、lease 或 transaction 系统。

## 8. 去掉无消费的参数和额外数据搬运

### 8.1 tactile_source 只交给真正解释它的模块

- 删除 `DemonstrationDataset` 的 `tactile_source` 参数和 `require_tactile_source` 调用。
- 删除 `AdapTacDataset` 仅为转发而要求的同名参数，以及配置装配中的 `cfg.dataset.tactile_source`。
- 保留顶层配置传给 agent；保留模型对 real/sim、坐标旋转和编码类型的真实约束。
- 保留 SimRunner 在创建环境前拒绝 real-source 触觉输入的检查。
- 保留 Dataset 对 future force 所需数组、形状、时间偏移和有效 mask 的处理；`tactile_source` 删除不等于可以删除触觉数据语义。
- 更新现有 tactile 配置测试：检查参数到达 agent 且实际坐标行为正确，不再断言 agent/dataset 存在两个相同副本。

### 8.2 只处理实际消费的观测，避免为此搭框架

基线已经在归一化与模型内过滤额外观测，但 `train_microbatch` 会先把整个 batch 搬到设备。

- 默认策略配置只声明真实需要的传感器；保持 Dataset variant 额外读取的 R3D 和未来触觉监督字段。
- 在训练 batch 搬到设备前，按 **Policy 的** `consumed_observation_fields` 筛选 `batch["obs"]`；不得只看 obs_encoder，AdapTac 的策略层还会消费 aggregate force。
- 其他 batch 项如 `action`、`future_force`、`future_force_valid` 原样保留。
- 模型公开预测入口仍可忽略调用方额外观测；不要用一个新的 registry 替代现有字段 tuple/property。
- 本轮不重排整个 trainer 构造生命周期，不增加可变 Dataset 视图或通用字段依赖规划器。用户显式在 Dataset 中多声明字段时，Dataset 可能仍读取/增强它们；上述改动只保证默认配置和设备搬运不做无用工作，不虚称消除了全部磁盘 I/O。

## 9. 预训练产物保持小，去掉明显重复保存

### 9.1 保持当前运行时边界

AFRO/CordViP 的小导出和 DQ-RISE 三数组码表已经符合目标，本轮不再改格式。独立预训练完整 checkpoint 与下游小产物服务不同用途，不必强行合为一个“统一 Artifact”。

保留：

- 完整预训练保存其继续训练或重新导出所需状态；小 encoder 只交付下游真正需要的内容。
- VQ `V=0` 导出 last；`V>0` 才建立 val loader、以 validation lookup MSE 选 best。
- 下游不要求预训练数据和主策略 train episodes 同源或同集合。
- 不加载到预期字段时明确失败，不用宽松 state_dict 或默认 affine 掩盖错误。

### 9.2 VQ 最后一轮不再保存两份等价命名快照

当前最后一轮固定保存 `vqvae_hand_epoch=....pt` 和 `vqvae_hand_last.pt`，若最佳还会保存 best。

- 非最后一轮继续按现有 `save_epochs` 保存周期快照。
- 最后一轮保存 `vqvae_hand_last.pt`，不额外生成仅名称不同的 epoch 快照。
- 有 val 时 best 仍表示独立的选择结果；它与 last 同轮时允许各自存在，不为了省一次写盘引入 alias/manifest/去重存储系统。
- 默认导出来源不变；手动导出仍可接收任意实际完整 VQ checkpoint。
- 保留当前 loss/gradient 非有限值处理。若顺手合并裁剪与有限性实现，只能复用原生数值操作并验证等价；不以“删除校验”为理由让错误梯度进入 optimizer。

本轮不重写 EMA 的 dtype 分组、foreach 更新、BF16 策略或信号保存。这些属于独立的训练数值与性能问题，收益尚不足以支持扩大本次重构。

## 10. 异常与 I/O：删除工作流，保留小的正确工具

删除评测失败状态写入后，检查 `write_failure` 是否还有实际调用；没有则删除函数及无用 import。不要保留无人使用的“通用失败记录能力”。

继续保留：

- checkpoint / best / 完成结果的临时文件写入和原子替换；不把部分写入文件暴露成成功产物。
- contact 的临时目录发布与失败清理。
- DataLoader worker、W&B、信号 handler、仿真环境等实际资源的关闭。
- `close_resources` 在存在主异常时尽量完成其他清理、报告次要异常并保留主异常的行为。

`except BaseException` 不按文本匹配批量删除。在包围整个评测业务、只为生成失败 JSON 的位置删除；在确保 KeyboardInterrupt 下也关闭真实资源的位置可以保留。不要把精简变成异常被 cleanup 覆盖或后台进程泄漏。

以下检查明确不能作为“多余校验”删除：

1. optimizer 参数组与参数名顺序的恢复检查。PyTorch 的 optimizer state 可能按位置匹配；strict 模型加载不保证 optimizer 对齐。
2. DQ 连续码索引在量化之前的有限性检查。NaN 经整数转换和 clamp 可能变成有效索引，再解码成有限动作；末端动作 finite 无法替代这一检查。
3. point/RGB/相机几何的物理一致性、action/future window 边界和触觉有效 mask。
4. affine 的有限非零 scale、实际表维度，以及不存在 EMA 时拒绝隐式改用 raw。
5. 原子写入、防止覆盖已有真实数据，以及真实资源清理。

不要为这些保留项再添加额外报告文件、验证级别或跨层重复断言。

## 11. 实施次序和文件范围

按依赖完成，每步结束后继续下一步，不停留在再写一份方案：

| 顺序 | 工作 | 主要位置 |
| --- | --- | --- |
| 1 | 日志追加、恢复标记、normalizer shape 单一来源 | `training/common/{logging,resume,config,data}.py`、`training/trainer.py`、`common/{normalizer,checkpoint}.py` |
| 2 | 按事实提取部署信息，去掉格式门槛 | `training/common/data.py`、effective config 装配、`deployment/metadata.py` |
| 3 | contact 固定字段、XYZ/FP32、最小 attrs | `datasets/preprocess/contact_map.py`、`test_tools/pretrain_checks.py` |
| 4 | 评测完成结果、测试后发布、路径与锁简化 | `evaluation/{rollout,selection}.py`、`eval_pipeline.py`、`deployment/metadata.py`、`common/io_util.py` |
| 5 | tactile 参数清理、设备前筛选、VQ 保存去重 | Dataset/variant、配置装配、trainer、VQ 训练入口及调用方 |
| 6 | 清理旧断言、死代码、脚本和文档 | 相关 `test_tools`、README、data README、docs、真实受影响的 scripts |

第 2 步先于第 3 步，避免 contact 删除 `format` 后仍被训练阶段丢弃 `dt` 等事实。新 normalizer shape 恢复同时修改 fresh/resume/inference 及 synthetic checkpoint 构造，不能只改 writer。

文档以实际行为为准：

- README 和架构文档删除 mandatory source_id/commit/template 记账要求，更新 tactile 参数去向和 shape 恢复方式。
- 仿真评测文档同步路径、完成结果、锁移除和 best 的新发布顺序。
- 观测预训练文档描述 contact 最小输出、action 可选以及新旧点云通道均可按 XYZ 消费。
- 旧任务书是历史记录，保留其已执行验证事实；若旧的规范描述与本轮冲突，加一处简短后续说明指向本文即可，不改写历史为“已执行本轮”。

只修改本任务涉及的调用方。不要顺带整理所有模型、改变 checkpoint 通用封装、引入新依赖或格式化整个仓库。

## 12. 验证：覆盖真实风险，使用现有工具

### 12.1 必要回归

| 项目 | 最小有效验证 |
| --- | --- |
| 日志恢复 | 正常、含损坏行、末行无换行三种日志都不参与恢复进度；旧字节不被截断/备份；新记录是独立 JSON；fresh 不覆盖已有实验 |
| normalizer 恢复 | 非 identity、identity 和含 `hand_latent` 的 state 恢复正确；新 cfg 不含 shapes 副本；raw/EMA 使用各自选中 state；scale/offset 缺键或形状不一致仍由 strict 加载报告 |
| 初始化独立性 | AFRO/CordViP/DQ 完整策略保存后移走外部初始化文件，仍可恢复；fresh 仍实际导入必要文件 |
| 部署事实 | 有 dt/配方但没有 format 的数据仍保存事实；没有事实不臆造；未消费模态不变成部署必需项 |
| contact 产物 | 六通道源只输出前三通道 FP32；关节/action FP32、episode 边界和点序不变；无 action 可预训练，有 action 可走当前主策略输入 |
| contact 数值 | 对少量确定性点云比较现有公式/逐点参考，覆盖不同 block 大小和法线符号翻转；输出 finite 且在 [0,1]；不要求浮点跨平台逐位相同 |
| contact I/O | 源文件不改写、已有输出不覆盖、中途失败不留下成功目录；有 Open3D 时串行与 spawn 结果一致到合理浮点误差 |
| 评测发布 | 用已有函数的轻量替身跑候选和 test；成功顺序正确；test 抛异常时旧 best 完全不变；低 test 分仍发布 selection 胜者 |
| 路径 | 显式外部 checkpoint 和 symlink 可用，外部 output_dir 可用；best 相对 run 解析，selection 路径相对结果目录解析 |
| tactile 与额外输入 | Dataset 不接收 source；模型/Runner 的 real/sim 语义仍在；设备前过滤不丢 AdapTac force 或未来监督目标 |
| VQ | 无 val 不建立验证遍历；最后一轮不产生重复 epoch 快照；best/last 与默认导出语义不变；DQ 量化前 NaN 拒绝仍有效 |

已有 `dataset_checks`、`pretrain_checks`、`agent_checks` 足以承载大部分回归。根据实际修改增加少量行为检查即可；不另建 contract test framework、evidence 生成器或要求每项产出日志证明文件。

更新与本轮行为冲突的旧断言，例如 contact 复制六通道、cfg.dataset.tactile_source、配置含 normalizer_shapes、恢复自动修复日志、失败 JSON 状态等；不是让错误的历史断言重新决定实现。

### 12.2 执行方式

在仓库已有环境中，先核对当前 CLI，再运行适用检查，例如：

```bash
python -m compileall -q dexmani_policy
python -m dexmani_policy.test_tools.dataset_checks --device cpu
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
git diff --check
```

对 checkpoint、DQ 和触觉变动按需调用已有 `agent_checks` 相关函数。环境具备依赖时，复用 `smoke_test` 或现有合成数据检查，完成少量 fresh → save → resume updates；不用完整训练来验收一次机制简化。

Open3D/spawn、CUDA、仿真和真实硬件分别依赖其环境。缺少时继续完成可运行项，准确列出未执行项；不得把 compileall、stub 流程测试或旧任务书中的 CPU/CUDA 记录当成本轮真实训练、闭环评测通过。

若比较性能，只做对应路径的小测量，例如恢复是否不再随日志全文长度增长、contact 输出字节和峰值内存、额外 obs 是否未搬入 GPU。没有 profiling 时只描述删除了哪些工作，不给加速倍数。

## 13. 完成条件

- [ ] 上一阶段数量划分、train-only、identity/affine 继承和预训练导出语义没有回退。
- [ ] 历史 metrics 不再影响训练恢复，重复 marker 与 shapes 配置副本不再被新代码消费/保存。
- [ ] 部署数值事实不依赖 Canonical/source 标签；实际消费的事实仍能到达公开接口。
- [ ] contact 仅写固定的 XYZ/FP32 数据和必要 attrs；action、时间和点槽含义正确。
- [ ] 评测不再维护失败状态机或文件锁；完整 test 后才更新最小 best 指针。
- [ ] 显式文件和结果路径不再被实验目录包含关系限制；路径基准写清楚。
- [ ] Dataset 的 tactile_source 空转参数退出；无用观测不再先搬上设备再被丢弃。
- [ ] VQ 去掉最后一轮无意义的同内容 epoch 快照，产物接口不扩张。
- [ ] 原生 strict、optimizer 对齐、量化前 finite、物理语义和小的资源清理仍有效。
- [ ] 已同步实际调用方、测试和操作文档，无新增来源协议、兼容层或通用框架。
- [ ] 已报告真实执行的验证和未验证项，没有宣称未经测量的效果。

最终用中文简要报告：主要改动与删除、保留的重要检查、接口/产物变化、实际测试结果和未执行项。无需额外生成 evidence 报告或重复整份任务书。

## 14. 官方代码的参考范围

参考的是这些项目实际使用的模块和数据流，不是复制它们所有实现细节。下列链接固定到两轮审查采用的版本。

| 项目与代码 | 已核实事实 | 本轮采用的判断 |
| --- | --- | --- |
| [Diffusion Policy / PushT Dataset](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/pusht_image_dataset.py) | Dataset 直接组织 replay buffer、sampler 和 normalizer；这个具体 Dataset 在完整 buffer 上拟合 | 保持直接的数据流；train-only 是本项目明确选择，不宣称官方项目一致采用它 |
| [CordViP / contact_map.py](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/script/contact_map.py) | 逐点计算法线对齐距离，处理入口使用 gamma=1，theta=10；直接改写输入 pickle | 保留数学，保留本项目有界内存和独立输出；不照搬原地改写和脚本入口缺陷 |
| [CordViP / pretrain.py](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/pretrain.py) | 小导出保存 obs encoder；有验证损失日志，checkpoint 按周期保存 | encoder 与预训练训练器分离；不把验证 loss 等同闭环成功率 |
| [AFRO / AFRO_Pretrain.py](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/AFRO_Pretrain.py)；[下游 Policy](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/policy/afro_policy.py) | 入口创建 val loader，但训练循环没有用它选模，保存 pretrain_latest；下游提取视觉 encoder 并冻结 | 默认不需要验证选模基础设施；本项目额外保留输入 normalizer 是有意的数值要求 |
| [DQ-RISE / train_vqvae.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/train_vqvae.py)；[eval_vqvae.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_vqvae.py)；[train_dqrise.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/train_dqrise.py) | VQ 入口无验证循环，周期/最后保存；离线解码并排序成表，下游直接读取 | 保持固定码表和显式路径；不引入跨阶段 split、source、选模身份校验 |

这些官方仓库也有宽松权重加载、未使用管理器、整量加载、分布式同步或脚本缺陷；不作为照搬依据。此前 DexPort 等更广泛的归一化调研见旧任务书，本轮不重复更改已确定的数据统计政策。

## 15. 执行约定

先阅读 `AGENTS.md` 和本任务书，再看当前实现与调用方。若 HEAD 已变化，按本文最终行为在现状上局部适配，不回滚后续工作、不重新开始开放式方案评选。

普通命名、局部函数拆合、测试放置由执行者判断；不需要为每一项删改另行确认。只有真实代码证据表明目标存在数学、时序或外部接口冲突时，说明具体影响并采用最小正确修正，不以新增模式、验证体系或兼容平台绕开问题。

任务完成的依据是行为正确、工作流更直接、必要验证完成；不是零 metadata、零异常处理、最少文件或固定行数。
