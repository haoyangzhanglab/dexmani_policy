# Codex 任务书：正确性修复、归一化对齐与等价推理优化

## 1. 目标与执行范围

本任务将已核实的问题收敛为一组小而完整的工程修改。优先保证正确、好用、可恢复、易于维护，再减少明确的重复计算。无需通过完整策略训练验收本任务。

- 审查基线：`haoyangzhanglab/dexmani_policy@98fd816a84256820f3078d9b86eb0334c413345a`。
- 任务书日期：2026-10-21。它描述待实施方案，不表示下面的修改已经完成。
- 开始前读取根目录 `AGENTS.md`，检查当前 HEAD 和工作区。如果代码已演进，重新确认相关调用链，跳过已修复项；不要回退代码以匹配审查基线。
- 默认实施 T1—T6。第 8 节是明确延后的工作，不属于本轮交付。
- 可使用合成张量、临时小型 Zarr、无预训练权重的小模型和少量 CPU optimizer step 做回归；不启动完整训练、批量闭环评测或真机运动。

### 必须保留的行为

1. 现有 observation/action 字段顺序、物理单位、坐标系、旋转表示和执行时序。
2. 模型结构、损失权重、训练预算、数据增强、采样比例、默认 NFE 和手部表示容量。
3. strict resume 以 checkpoint 的 `effective_config` 和已保存状态为准；推理不依赖训练数据或外部 prior 文件。
4. raw/EMA、optimizer、scheduler、RNG、数据采样位置和内嵌手部表示的保存恢复。
5. 已有 selection/test 隔离机制，以及 Sparse3D 米制坐标、rot6d、RGB 的专用处理。

允许的明确行为变化：拒绝此前漏检的非法配置；规范连续运行输入 dtype；fresh training 使用参考一致的 normalizer 拟合总体和正确的 EMA 累积精度；新实验显式采用独立的评测种子。旧实验的统计、权重和历史结果不原地改写。

## 2. 实施顺序与完成条件

| 顺序 | 任务 | 交付重点 | 完成条件 |
| --- | --- | --- | --- |
| 1 | T1：配置与输入契约 | 在边界发现 shape/window/dtype 错误 | 合法配置通过，已知反例被明确拒绝；float64 连续输入可正常推理 |
| 2 | T2：Normalizer 对齐 | 拟合不再依赖有效窗口；保留流式统计 | 参考对照、窗口独立性、旧 checkpoint 恢复通过 |
| 3 | T3：EMA 精度 | 可训练低精度参数不丢失微小 EMA 更新 | 数值基准、保存恢复、下一步更新等价通过 |
| 4 | T4：评测种子 | 新实验评测 seed 与 training seed 解耦 | CLI/配置优先级、任务 seed 清单、held-out 排除通过 |
| 5 | T5：Temporal 条件缓存 | 单次采样内重复的 context/KV 投影只计算一次 | 非退化模型输出等价；调用次数从 NFE 次降为一次 |
| 6 | T6：回归与文档收口 | 小型测试集和准确的使用说明 | 相关检查通过或有具体 NOT VERIFIED 原因；无半套接口 |

按这一顺序实施，但把每项的定向测试与代码一起完成，不要等到 T6 才验证。一次只解决一个可解释问题。普通实现选择自行判断；缺少 GPU、数据或权重时继续完成可独立验证的工作。

## 3. T1：统一配置校验与运行输入 dtype（P1）

### 证据和入口

- `dexmani_policy/training/validation.py`：`validate_config`、`validate_hand_config`。
- `dexmani_policy/utils/config.py`：已有 `validate_action_window` 等校验函数。
- `dexmani_policy/agents/core/base_policy.py`、实际 U-Net 实现及其配置。
- `dexmani_policy/deployment/runtime.py`：`PolicyRuntime.predict`。

审查基线中，普通策略的 root/agent/dataset 窗口一致性不如手部策略完整。实例：root 配置合法时，单独将 DP3 的 `agent.n_action_steps` 改为 16，可能通过配置检查，但 H=16、Nobs=2 只有 15 个未来动作。identity 归一化字段没有经过 affine transform 的 dtype 转换，float64 proprio 输入会进入 float32 网络并报错。

### 方案

1. 将共有窗口一致性检查从手部专用逻辑中提取或提升到现有公共验证位置；不要复制两套规则。
2. 校验 resolved root、agent、各子 dataset 的 H/Nobs/action_dim 等实际对应关系，以及 `1 <= C <= H - Nobs + 1`。不要为不具有某字段的对象伪造字段。
3. 按实际 backbone 的下采样结构检查 horizon 合法性；不要对所有策略一律硬编码“必须为 4 的倍数”。同时保留模型运行时的必要防线。
4. 检查 MultiTaskDataset 各子数据集的契约；支持现有 auxiliary EE 尾部，不把 policy 完整动作宽度误认成控制输出宽度。
5. 在 NumPy → policy tensor 的运行边界，将连续数值观测转换为 `torch.float32`，再进入模型自身混合精度流程。RGB 使用现有预处理；离散索引、mask、语言字段遵循原契约。核对 RTC prefix 等同类连续入口，避免同一种遗漏。
6. 保留有限值、维度和历史长度检查；不要通过截断、补零或修改用户配置来“修复”不一致。

### 验收

- 覆盖合法边界 `C=H-Nobs+1`、越界 C、root/agent Nobs 不一致、子 dataset H/Nobs 不一致、U-Net 不合法 horizon。
- 使用真实轻量 policy 路径，证明相同 float32/float64 连续观测都可执行，并在规定容差内产生相同输出。
- identity 不改变数值归一化语义；RGB uint8 路径、action/action_ee、auxiliary EE 的 shape 不变。

## 4. T2：Normalizer 按参考配方对齐（P1）

### 4.1 已确定的设计决策

本轮对 DP 系通用 affine normalizer 采用以下固定规则，不增加多种拟合策略开关：

> Fresh fit 统计传入 dataset 的完整 replay buffer 中、该字段原始且未经增强的数据；不通过 SequenceSampler 选择统计行。每个原始样本按字段定义统计一次，结果不依赖 H、Nobs、C、padding、batch size、shuffle 或采样重加权。

这与已检查的 DP RealPushT、ManiFlow RoboTwin、CordViP、AFRO Adroit、XPolicyLab DP 的 `get_normalizer()` 路径一致。它们的 sampler 使用 train/validation mask，但拟合函数直接读取完整 buffer。因此：

- `max_train_episodes`、train/validation mask 继续控制训练采样，不隐式改变该参考配方的拟合总体。
- 如果加载的 buffer 同时包含训练和验证 episode，拟合会使用两者；必须在文档与运行记录中如实标注，不能称为严格 train-only 统计。
- “完整 buffer”仅指本次 dataset 加载的数据，不扫描其他数据目录、评测轨迹或外部数据。
- 如未来需要严格 train-only 的实验，应显式固定其拟合数据集合并另立实验协议；本轮不再增加一个自动切换的双轨框架。

### 4.2 代码落点与最小实现

- `dexmani_policy/datasets/base_dataset.py`：`iter_normalization_data`。
- `dexmani_policy/datasets/multitask_dataset.py`：各任务统计数据的拼接。
- `dexmani_policy/agents/core/normalizer.py`：`fit_field`、`affine_parameters`。
- `dexmani_policy/training/builder.py`：`build_normalizer`、`build_dataset_and_normalizer`。

1. `iter_normalization_data` 按 replay buffer 的 chunk 顺序读取字段，不再使用 `sampler.observation_source_rows/action_source_rows`。训练窗口有效性过滤本身继续保留。
2. 保留懒加载、分块读取、`cache=False`、point-cloud 通道选择和 auxiliary EE 拼接；不整体物化点云，也不通过 DataLoader 拟合。
3. 多任务按各 child buffer 的原始样本拼接统计；保持样本数自然权重，不使用训练 batch 的任务重采样权重。
4. 同一拟合字段完成通道选择和拼接后必须全部有限。遇到 NaN/Inf，报告字段和源数据区间并失败；不 `nan_to_num`、不夹断、不静默改成按有效窗口拟合。空数据也明确失败。这是拟合输入的要求，不删除训练 sampler 原有有效性过滤。
5. 保留 FP64 流式统计与最终 FP32 affine 参数，保留既有 feature axis 语义。普通向量和点云按最后一维特征统计；手指向量 `[5,3]` 保留现有独立位置语义，不误当作仅 3 个通道。
6. limits 模式保留参考默认 `[-1,1]`、`range_eps=1e-4`；对 `max-min < eps` 的维度使用单位 scale，并设置 `offset=(output_max+output_min)/2-input_min`。
7. Gaussian 保持当前 `_fit` 对应的样本标准差语义；不因另一条参考 helper 使用 `np.std` 就全局改为 population std。FP64 分块累计无需与 FP32 一次性归约逐 bit 相同，需解释并限制误差。
8. 在已有 effective config/data recipe 的自然位置记录拟合总体与公式版本，例如 `population=full_replay_buffer`、`formula=dp_affine_v1`。这是拟合行为记录，不是新开关；不增加独立 manifest、全数据哈希或每次保存时扫描数据。

### 4.3 字段语义与历史兼容

- `action_ee` 的 rot6d 部分保持 identity；平移与手关节继续使用既有字段规则。
- Sparse3D 的 XYZ 保持米制 identity；RGB 遵循已有 backbone 预处理。不要因为 ManiFlow 某路径拟合点云，就修改所有 encoder 的坐标输入。
- identity 字段仍不需要伪造 affine 参数；其 dtype 问题由 T1 的输入边界解决。
- 新规则只替换上述通用 normalizer 的 fresh-fit 过程。DQ codebook、LAMP prior、Temporal codec 的专用坐标变换和统计保持自身定义。尤其不要把 Temporal 的相对动作统计改成绝对动作全 buffer min-max。
- 保留现有 prior/codec 与 policy 坐标一致性校验；如果新实验的统计与已有 artifact 不兼容，给出明确错误，不修改 artifact、不放宽检查。
- Resume、评测、部署只加载 checkpoint 的 scale/offset，绝不重新拟合。旧 checkpoint 缺少新增拟合说明时，按保存的参数使用；不要猜测其 population，更不要强制迁移。

### 4.4 必须通过的测试

| 检查 | 期望 |
| --- | --- |
| 参考 oracle | 固定小数组按 DP/ManiFlow 公式得到预期 scale/offset；普通 limits 与参考一致 |
| 统计边界 | 常数、近常数、阈值两侧、非对称 `[0,1]` 输出范围均正确 |
| 分块与布局 | 多种 chunk 大小、向量/点云/手指字段、auxiliary EE 和多任务拼接结果正确 |
| 窗口独立 | 同一 buffer 改 H/Nobs/C/padding，统计不变；使用真实 dataset 迭代入口验证 |
| 拟合总体 | 验证 episode 和 `max_train_episodes` 之外的 buffer 行仍参与参考配方拟合；使用极值反例防止误测 |
| 无增强 | 更改增强概率或采样顺序不会改变拟合参数 |
| 非有限/空数据 | 明确失败；不输出 NaN affine 参数或静默回退 |
| roundtrip | normalize/unnormalize 在合理浮点容差内可逆，无新 clipping |
| 旧 checkpoint | 保存加载参数不变；resume 不调用 fit；缺新 metadata 的旧状态可恢复 |

审查时已用 40 帧单 episode 验证旧问题：Nobs=2、pad_after=7 时，H=12 的观测统计行是 0–36，H=16 是 0–32。修复后两者均应按同一完整拟合总体统计。该案例应作为回归依据。

## 5. T3：修复低精度 EMA 累积（P1）

### 证据和入口

- `dexmani_policy/training/common/ema.py`：`EMA.step`。
- `dexmani_policy/training/builder.py`：`build_model_and_ema`。
- `dexmani_policy/training/resume.py`、`training/common/checkpoint.py`。
- `dexmani_policy/agents/obs_encoder/rgb/dino.py`、`siglip.py`、`lora.py`。

问题只涉及可训练 BF16/FP16 参数的 EMA：DINO/SigLIP full tuning 或低精度 LoRA 可触发。全 FP32 参数和冻结参数不应被描述为同一种缺陷。只把单次运算临时转 FP32、每步仍把唯一累积状态写回 BF16，不能解决更新丢失。

### 方案与兼容决策

1. Fresh training 的可训练 EMA 参数以 FP32 保存累积结果。首选直接让已有 EMA 参数承担 FP32 累积，不额外复制整份模型；冻结大 backbone 保持原精度和复制语义。
2. 真正需要独立 accumulator 时，只覆盖低精度可训练部分，并将其纳入 checkpoint。不能留下仅在 Python 内存中存在、恢复后丢失的状态。
3. EMA 累积精度与前向计算精度分开验证：不能假设 `.float()` 后混合精度 DINO/SigLIP/LoRA 自动可用。检查 autocast、projection、teacher、评测以及恢复后的前向 dtype。
4. 若直接使用 FP32 EMA 参数，checkpoint 恢复必须按保存的精度构造/恢复它们，防止 `load_state_dict` 静默拷入新建 BF16 参数导致精度再次丢失。以现有 state/config 记录精度，避免引入第二套 checkpoint 管理器。
5. Fresh 配方明确使用 FP32 累积。旧 checkpoint 的 strict resume 保持原累积规则和已保存状态；必要的旧字段缺省处理集中在一个加载边界。不能以“修 bug”为理由静默改变旧实验的下一步更新轨迹。
6. 不修改 decay、warmup、更新频率、优化器、冻结参数和 BatchNorm 约束。不宣称修复了历史 EMA；从最终 checkpoint 无法重建已舍入丢失的累积过程。
7. 优先沿用现有 tensor 路径和容器。只有确需保存独立 accumulator 时，才增加必要状态并提供旧格式的明确读取规则；不要仅为这个任务统一重写 checkpoint 格式。

### 验收

- 在高 decay 下以小变化驱动 BF16 可训练参数，EMA 累积结果跟随 FP32 oracle；测试必须能让旧实现失败。
- 比较 foreach 与普通更新；覆盖冻结参数和原 FP32 参数，验证无意外退化。
- 在更新中途保存恢复，后续 EMA、优化器和调度器状态与不中断路径一致。使用小型 CPU 模型，包含真正的下一步 optimizer/EMA 更新，而不只是预测 roundtrip。
- 新旧精度状态各做恢复测试；旧 BF16 恢复保留旧规则，新 FP32 恢复保留新规则。
- 使用本地小 architecture、`load_pretrained=False` 验证相关视觉分支；需 CUDA 才能覆盖的路径明确记录，不下载大权重或伪称完整验证。

## 6. T4—T5：评测可比性与等价推理优化（P2）

### T4：独立评测 seed，复用已有计划和记录

入口：`env_runner/evaluation.py`、`env_runner/selection.py`、`env_runner/common/seeds.py`、`evaluate.py`、`select_checkpoint.py`、相关 eval 配置和脚本。

1. 新配置显式设置 `eval.seed: 1024`；该值不随 `training.seed` 变化。多个训练 seed 的实验采用同一评测协议时应得到相同的任务 episode 计划。
2. 统一选择与评测入口的有效 seed 解析。优先级：显式 CLI seed > 显式 dotlist `eval.seed` > 已保存 `eval.seed` > 旧配方兼容回退。相同优先级冲突明确处理，不形成第二套隐式覆盖规则。
3. 旧配置缺少 `eval.seed` 时，保留原 `training.seed+1024` 回退以重现历史默认行为，并在结果中记录实际使用值。比较旧模型时可显式指定共同 eval seed，无需重训。
4. CLI 与流水线应传递同一个显式 seed，不通过修改 `training.seed` 间接调整评测。若保留旧 override 行为，仅做集中兼容并在帮助中说明。
5. 继续使用现有 selection record、每任务 seed plan 和 episode 明细。最终测试仍排除所有已用于 selection 的 seeds，包括落选候选使用的 seeds。
6. 同一个 seed 只有在任务池、预算和排除集合相同时才保证相同计划。文档说明：跨实验比较应核对实际清单；不能只看 seed 数字就宣称公平，也不能绕过 selection 排除来强行匹配。
7. 不增加 `eval.seed_manifest` 框架、全局评测数据库或新的结果目录体系。

验收使用合成任务池即可：改变 training seed 不影响新配方计划；优先级正确；旧配方回退不变；同协议计划一致；selection/test 无交集；池不足明确失败。无需仿真 rollout。

### T5：普通 DiTX 复用已有静态 context/KV 缓存机制

入口：`agents/action_decoders/backbones/ditx.py`、`agents/action_decoders/flow.py`、`agents/core/temporal_latent_flow_policy.py`。

当前 `ConsistencyDiTX` 已有 `prepare_context_kv`，普通 `DiTX` 在每个 Euler step 重复 context embedding 和逐层 K/V 投影。采用现有接口形状做小幅复用，不新建 cache manager。

1. 在一次 `RectifiedFlow.predict_action` 内准备静态 context/KV，由该调用局部持有，并传给每个 step。兼容没有缓存能力的现有 backbone。
2. 仅缓存与 x、timestep、target_t 无关的值。带时间条件的 modality normalization 不可复用为静态值；训练路径不使用推理缓存。
3. 不把缓存存成跨请求的 module 属性或持久 buffer；不跨 observation、episode、设备或 dtype 复用。这样无需额外失效框架。
4. 保留参数名、state_dict、权重 shape、Euler 网格、随机噪声生成次数和默认 NFE。`torch.compile` 路径若受影响，针对实际 backbone 检查，不为缓存扩大编译范围。
5. 测试使用非零输出头、非零 modulation gate 的小模型，避免初始化输出全零掩盖错误。固定初始噪声，比较 NFE=1/2/4/10 的缓存/非缓存输出；覆盖连续两次不同 context 调用，排查串用。
6. 用 hook 验证 context projection 和每层 K/V projection 每次采样只执行一次。若有 GPU，分别报告 warmup 后 p50/p95 和测试条件；无 GPU 时仅报告计算次数变化，不承诺加速百分比。

这项优化是利用条件不随采样步变化的等价重用，并复用本仓库已有代码；不要误称 ManiFlow 官方仓库已经实现同样缓存。

## 7. T6：少量高价值回归与文档收口

### 测试组织

- 优先复用已有测试工具；没有现成框架时可用标准库 `unittest` 加 PyTorch，建立少量按行为分组的 `tests/test_*.py`。不要只为任务引入大型测试依赖。
- 测试针对外部契约与已复现故障，不逐行镜像实现。不要恢复此前删除的整套大量旧测试，也不要为了 coverage 数字堆积用例。
- 临时数据在测试中构造；不把大模型、外部数据或参考项目副本提交到仓库。参考公式 oracle 可由少量固定数组和明确预期值实现，标注来源即可。
- 将 T1—T5 的关键定向测试纳入统一 CPU 命令；测试 GPU 算子或实际外部资产时按事实标记 skip/NOT VERIFIED。

### 建议执行命令

先核对当前 README 和入口的真实参数，再执行适用命令：

```bash
python -m unittest discover -s tests -p 'test_*.py'
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat r3d maniflow lamp temporal_codec_flow multitask_dit
python dexmani_policy/evaluate.py --help
python dexmani_policy/select_checkpoint.py --help
git diff --check
```

若仓库已有测试框架，使用其对应命令，不同时维护两套 runner。DDP 配置若被改动，补相应 config-only 检查；不能把 config-only 通过等同于 DDP 训练通过。仅在 CUDA、数据和权重齐备时，对实际受影响的代表配置运行现有 `smoke_test.py <config> --max-updates 4`；不修改生产配方以让 smoke 假通过，不遍历所有模型启动训练。

### 文档与交付

1. 同步 README 和现有相关 docs 中 normalizer 拟合总体、历史 checkpoint 使用边界、EMA 精度、评测 seed 用法。
2. 明确普通 affine 统计与手部 codec 专用统计的区别；将新增字段放在自然位置，避免散落多个配方来源。
3. 说明 EJC 在当前默认设置下的 joint-ID 参数化语义；对确实没有进入 active path 的研究模块准确说明状态。只改与本任务结论有关的说明，不顺手重命名模型或清理整棵目录。
4. 保留 `AGENTS.md` 的通用规则，不把本次实验和具体模型结论写成新的长期约束。
5. 最终按 T1—T6 报告：修改文件、行为变化、兼容性、实际运行的检查及结果、未验证项目和具体原因。
6. 如某项已在当前 HEAD 修复，给出代码证据与检查结果后标记已满足。遇到有证据的设计冲突，完成不受影响项，说明最小剩余缺口；不要悄悄跳过高优先级项。

## 8. 明确延后：本轮不做

| 项目 | 延后原因及后续条件 |
| --- | --- |
| SAT 跨帧 token 改造、Sparse3D 新模块、LAMP/Temporal 容量改变 | 改变模型假设，需新训练和公平消融；不混入等价工程修复 |
| DQ codebook 重选/替换、PCA 排序或 KMeans 替代 | 可先离线诊断，但替换后影响策略动作标签；不自动修改旧 artifact |
| teacher 端点分支裁剪 | 数学上有冗余，但动态子 batch、同步和 RNG 影响需测量；先完成收益更明确的 T5 |
| EJC 推理折叠 | 固定权重可等价折叠，但当前收益小；已有导出需求和性能证据时再做，不为它新建导出框架 |
| XPolicyLab adapter、批量仿真或网络服务 | 已有 PolicyRuntime 边界；没有具体集成目标时不引入第二套 runner/server。未来只做薄 adapter |
| Dexora 式数据质量判别器、AFRO/CordViP 的额外学习阶段 | 需要数据和训练；episode 级质量报告可另立小任务 |
| resume 目录整体重构或历史产物重命名 | 当前主要是命名可读性问题；保留显式 output_dir，不为其重写 Hydra 日志和目录生命周期 |
| 新 registry/factory、大型基类、通用 migration/cache/benchmark 框架 | 不解决本轮必要问题，会增加维护面 |

XPolicyLab 借鉴的是接口边界和显式字段语义，T1 已落实这一原则。其 EE pose 使用 `[x,y,z,qw,qx,qy,qz]`，本项目对应 rot6d 布局不同；未来接入必须显式转换。不要复制其 DP adapter 从当前 YAML 和 checkpoint 分别读取窗口/模型配置的做法，本项目应保留 saved effective config 的单一来源。

## 9. 核实过的参考来源

以下链接固定审查版本，供核对具体语义，不要求下载或复制整个参考项目。当前项目行为仍以执行时 HEAD 为准。

- [Diffusion Policy normalizer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/normalizer.py)：limits/Gaussian 与近常数处理。
- [Diffusion Policy RealPushT dataset](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/real_pusht_image_dataset.py)；[分字段 normalization helpers](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/normalize_util.py)。
- [ManiFlow RoboTwin dataset](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/dataset/robotwin_dataset.py)：完整 replay buffer 拟合。
- [CordViP robot dataset](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/dataset/robot_dataset.py)；[AFRO Adroit dataset](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/dataset/adroit_dataset.py)。
- [XPolicyLab DP dataset](https://github.com/XPolicyLab/XPolicyLab/blob/5d0be1499634d3176258c8f2e31f5fe816497b0a/policy/DP/diffusion_policy/dataset/robot_image_dataset.py)；[ModelTemplate](https://github.com/XPolicyLab/XPolicyLab/blob/5d0be1499634d3176258c8f2e31f5fe816497b0a/model_template.py)；[状态打包](https://github.com/XPolicyLab/XPolicyLab/blob/5d0be1499634d3176258c8f2e31f5fe816497b0a/utils/process_data.py)；[DP adapter](https://github.com/XPolicyLab/XPolicyLab/blob/5d0be1499634d3176258c8f2e31f5fe816497b0a/policy/DP/model.py)。
- [DQ-RISE hand pretraining](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/dataset/pretrain.py)；[固定范围](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/utils/constants.py)：不能当作通用数据 min-max 拟合规则直接套用。
- [Dexora 官方论文 §III-C/III-D](https://arxiv.org/html/2605.18722v1)：区分数据质量检查、回放验证与需要训练的判别器/加权策略。

## 10. 发给 Codex CLI 的执行 prompt

```text
请在当前 dexmani_policy 仓库实施根目录 CODEX_TASK_BRIEF.md。

先读取 AGENTS.md 和任务书，检查当前 HEAD、工作区及相关实际调用链。任务书中的 SHA 是审查基线，不要回退当前代码。保留已有用户修改，已修复的问题核实后跳过。

按 T1→T6 完成代码、定向回归和必要文档，不要停留在建议或计划，也不要执行第 8 节明确延后的研究与集成任务。以正确、简洁、高效、易维护为准，复用现有函数和边界，不引入无必要的框架或依赖。

重点：normalizer 的 fresh fit 按完整 replay buffer 的原始字段对齐参考，保留流式统计和字段语义；旧 checkpoint 继续加载原统计。EMA 修复必须覆盖保存恢复和下一步更新，不能只做临时 FP32 运算。评测 seed 与训练 seed 解耦但保留旧配方回退。Temporal 缓存只在单次推理调用内复用，保持权重和采样语义。

先运行小型 CPU 定向测试与 config-only smoke。完整模型训练、批量评测和真机运动不在本次范围；只有环境资产齐备时才运行任务书规定的有界 smoke。缺少 GPU、数据或权重时继续完成独立工作，把未覆盖项标为 NOT VERIFIED，不通过修改生产配方绕过检查。

一般实现选择自行处理，不逐项请求确认。如发现任务书与当前代码冲突，以可复核证据调整最小实现，并说明影响。完成后按 T1—T6 列出变更、兼容性、实际执行的检查结果及剩余限制。不要自动提交或推送，保留清晰的 diff 供我审阅。
```
