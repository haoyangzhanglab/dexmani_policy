# DexMani Policy 第二阶段轻量化任务书

日期：2026-11-03。代码基线：[`98c318f9c3b4e313196fdc3e12cb3bf178a874c2`](https://github.com/haoyangzhanglab/dexmani_policy/tree/98c318f9c3b4e313196fdc3e12cb3bf178a874c2)。

本文汇总最近两轮审查的八项发现，给出直接实施的方案。**这是待实施任务书，不表示代码已修改或训练测试已通过。** 先读 [`AGENTS.md`](../AGENTS.md)，再核对当前实现和实际调用方；已有等价修改的项目直接跳过。

目标是让个人机器人研究代码轻、薄、直观，减少没有真实消费者的能力和为诊断而存在的热路径分支，同时保持训练、恢复和推理正确。用户已明确授权：**不考虑旧版本配置、checkpoint 或 Python 调用接口兼容，允许收窄支持范围和适度放宽诊断。** 不需要为以下删除逐项再次征求确认。

本次只执行 F01—F08。上一阶段的 [`simplification_task.md`](simplification_task.md) 保留为历史记录，不重新执行 T01—T10；与本次八项决策冲突的内容以本文为准。此前完成的预训练局部状态收敛、整模态 dropout 删除、本地 Zarr 目录收窄等，不重新引入。

## 1. 合并结论与支持范围

| 任务 | 当前维护负担或具体问题 | 本次决策 | 明确接受的变化 |
| --- | --- | --- | --- |
| F01 | VQ 默认不裁剪时，每步仍逐参数扫描梯度有限性 | 删除这条扫描分支，保留 loss 和实际数值边界检查 | 不保证在 optimizer 前发现所有“loss 有限、梯度非有限”情况 |
| F02 | 精确 update 计时需要 CUDA 同步，主循环还为最后一个 micro-batch 特殊归约 | 删除 `train/step_time` 及专属计时逻辑 | 常规训练日志不再承担性能测量 |
| F03 | 已完成 run 的 resume 仍构造数据、模型、logger，甚至准备 compile | 读取 checkpoint 进度后立即判断完成并返回 | 已完成 run 不再检查或初始化无须使用的数据与执行资源 |
| F04 | sampling dtype 参数贯穿三种 decoder 和 RTC，正式路径统一使用 FP32 | 采样状态固定 FP32，删除 dtype 选择接口 | 不再支持调用方选择 BF16 采样状态；训练 BF16 autocast 保留 |
| F05 | 普通 DiTX 和 SAT 传播无消费者的 `return_features` 联合返回类型 | 普通输出仅返回 Tensor；特征接口限于 CoupledDiTX/InterAct | 不再支持普通 DiTX/SAT 特征返回 |
| F06 | `CLIPTextEncoder` 无当前策略、配置或训练入口消费者 | 删除孤立文件和保留说明 | 不预留未接入的文本条件组件 |
| F07 | Flow 对确定存在的 KV 接口做 `getattr` 探测和可选参数拼装 | 直接调用现有接口，保留实际缓存 | 不兼容任意缺少该接口的第三方 backbone |
| F08 | FPS 选点噪声跨实现、默认配置和八份 YAML，正式值全部为零 | 整条删除，不做历史兼容 | 旧配置中的 `random_noise_scale` 可以自然报错 |

这些发现以支持范围和维护成本为主。F03 是可具体触发的无效初始化问题；不能把其余各项统一宣传为已确认的训练数值 bug 或显著吞吐收益。

实施边界：

- **直接删除，不换一种封装。** 不新增 Manager、Context、RunState、能力注册表、结果 dataclass、配置版本、迁移器或兼容适配器。必要的局部分支和已有小函数可以保留。
- **不兼容旧版本，仍保证新版本闭环。** 本次修改后的 fresh → save → resume → inference 必须成立；不要求旧配置、旧 checkpoint 或外部私有脚本继续可用，不改写历史实验文件。
- **允许朴素失败。** 不为删除的参数添加忽略逻辑、告警、`**kwargs` 接收器、自动迁移或 TypeError 回退；也不另建“拒绝旧格式”检测层。自然的缺键、非法参数或依赖库报错即可。
- **不扩大任务。** 不借机清理全仓兼容代码、统一两个训练循环、重写 checkpoint 格式、增加 VQ resume 或搭建性能测试平台。若最新代码出现真实新消费者，说明证据并保留满足该消费者的最小实现，不凭空维护通用能力。

## 2. 必须实施的八项任务

以下代码路径省略共同前缀 `dexmani_policy/`；配置路径也位于该目录下。

### F01 — 删除 VQ 无裁剪路径的全梯度扫描

主要位置：`training/pretrain/hand_prior/vqvae.py` 的 backward/optimizer 段；`configs/pretrain/hand_prior/vqvae.yaml`。

当前正式配方 `max_grad_norm: 0.0`，因此每步执行 `torch.isfinite(p.grad).all()` 列表及 `torch.stack(...).all()`。它比主策略、观测预训练已经接受的诊断边界更重。

- 删除 `max_grad_norm > 0` 对应 `else` 中的整个逐参数有限性扫描。
- 保留 backward 前的标量 `total_loss` 有限性判断；启用裁剪时继续使用现有 `clip_grad_norm_(..., error_if_nonfinite=True)`。
- 保留输入手部数据、导出码表及 DQ 量化前的真实数值检查。不得把“少做梯度诊断”扩展成删除数据或量化边界。
- 不新增 `check_gradients` 配置、调试扫描函数或告警机制。不更改 VQ 损失、码表更新、scheduler、训练/验证划分或导出选择。

完成判据：现有小型 VQ 训练与码表 roundtrip 仍通过；无裁剪训练不再扫描所有参数梯度。接受有限 loss 对应的异常梯度可能在后续计算才暴露，不为这一放宽补一套监控。

### F02 — 删除训练精确计时，恢复直接的指标累积

主要位置：`training/trainer.py::{train,log_step}`、`training/pretrain/obs/common.py::train`；当前说明见 `docs/项目架构.md`、`docs/观测预训练.md`。

- 删除 `train/step_time` 指标、`elapsed` 参数、起止时钟变量、专用于计时的 CUDA synchronize，以及因此不再使用的 import。
- 主策略在每个需要记录的 micro-batch 后统一调用一次 `accumulate_metrics`，组末按实际 `group_size` 求均值。删除“先累积前几个，结束计时后再累积最后一个”的特殊路径；不再使用的 `group_pos` 改为 `_`。
- 观测预训练同步删除计时，不改变现有正常指标累积。两处仍分别保留直观训练循环，不抽取公共训练框架。
- 保留本次调用首个实际 update、日志周期及已知末次 update 的记录；保留 loss、LR、epoch、现有梯度指标及每个成功 update 的进度条推进。
- 保留所有 micro-batch 的有限性合并、组末更新前判断和尾组实际除数。必要的数值检查/日志标量转换可能同步设备，不在本项盲目删除。
- 不用未同步的 CPU 计时替代同名指标，不新增计时开关、采样计时器或 profiler 包装。

完成判据：现有训练/预训练检查保持 step、组均值、尾组、首次 resume 日志及停止行为；当前文档不再承诺该耗时指标。只删除计时所需同步，不能声称普通非日志步因此获得明显加速。

### F03 — 已完成的主策略 resume 在构造前结束

主要位置：`training/common/resume.py::{prepare_resume,main}`；实际调用方见 `train.py`、`test_tools/smoke_test.py` 和 `test_tools/training_checks.py`。

当前 `prepare_resume()` 总会构造 Trainer，而完成判断位于后面的 `PolicyTrainer.train()`。已完成 run 因而仍可能访问不存在的数据、初始化模型或产生空日志会话。

采用一个局部、明确的返回约定：**未完成返回 Trainer，已完成返回 None。** 示意顺序如下，变量名与实际代码保持一致：

```python
checkpoint = load_checkpoint(source)
cfg = OmegaConf.create(checkpoint.effective_config)
if checkpoint.global_step >= cfg.training.loop.total_train_steps:
    print(...)  # 简短说明已经完成及保存的进度。
    return None
# 然后应用现有执行覆盖，并 build_trainer(cfg, checkpoint=checkpoint)。
```

- 判断放在读取保存配置之后、执行覆盖和 `build_trainer` 之前；尤其不先 resolve 新数据路径，不访问 dataset、外部初始化权重、模型、logger 或 compile。
- 同文件 `main()` 先接收结果，为 None 就正常返回，否则调用 `run_trainer`。**`run_trainer` 继续只接受真实 Trainer**，不要把 Optional 分支传播到整个训练框架。
- 其余实际调用方同步适配。已有 smoke 要求尚有剩余预算；如需表达这一前提，一条直接断言足够，不建立统一结果包装。
- 保留未完成 resume 的原有执行覆盖、状态加载、参数/EMA/optimizer 绑定顺序及短 checkpoint 生命周期；不重新装配配置、不重新拟合统计、不重复读 checkpoint。
- 保留 `PolicyTrainer.train()` 对已构造 Trainer 的预算早退。本项不顺带重写所有预算校验或更改总训练预算语义。
- 已完成路径不新建 logger/W&B 会话、不修改 checkpoint 或日志，不要求原数据仍存在。不新增完成标志文件、状态枚举或单独 completion API。

完成判据：复用现有微型训练 fixture 训练到真实预算终点，再调用 resume，可在数据不可用的情况下正常结束；确认没有进入 Trainer 构造和没有新产物。可用一个局部 `build_trainer` 替身证明不进入构造，不为每个构造器分别 mock。未完成的分段训练/恢复仍通过。不要篡改 checkpoint 中的 step 冒充真实完成来替代这项行为验证。

### F04 — 采样状态固定 FP32，删除 dtype 参数链

主要位置：`agents/action_decoders/diffusion.py::predict_action`、`flow.py::{RectifiedFlow,ConsistencyFlow}.predict_action`、`sampling/rtc.py::{guided_ddim,guided_flow,soft_mask}`。

正式调用均为默认 FP32 或显式 `dtype=torch.float32`；唯一显式 BF16 采样调用来自检查代码。这里收窄的是采样状态接口，训练 autocast 和模型内部计算精度不变。

- 删除三个 `predict_action` 的 `dtype` 参数，以及 RTC helper 中仅用于这条选择链的参数和转发。
- 初始噪声/采样状态显式使用 `torch.float32`；RTC prefix 转换、soft mask 及既有时间/系数计算保留 FP32 数值语义。`soft_mask` 仅有内部消费者，可以直接创建 FP32 tensor，不保留 dtype 选项。
- 删除 `policies/dqrise.py`、`dexlatent.py`、`adaptac.py` 中冗余的显式 FP32 参数，查齐同类实际调用。
- 不修改全局 default dtype，不强制把所有 condition 或模型参数转成 FP32，不删除 BF16 训练/autocast。保留现有算法公式、随机数调用顺序、噪声分布和 timestep。
- RTC 为引导求导保留的 `inference_mode(False)`、`enable_grad`、detach/clone 不能当作多余保护删除。
- 删除显式 dtype override/BF16 采样测试，以及“默认等于显式 FP32”的同义测试；保留 BF16 condition 下默认输出 FP32、普通采样与零引导一致、非零 RTC 有限且求导可用的实际检查。

完成判据：三个 decoder 的普通/RTC 采样仍工作；默认输出保持 FP32，BF16 训练检查不受影响。无需兼容旧 Python 调用，不新增采样精度配置。

### F05 — 特征返回只留给 CoupledDiTX / InterAct

主要位置：`agents/action_decoders/backbones/ditx.py::{DiTXOutput,DiTX,CoupledDiTX}`、`agents/hand_prior/action_io.py::SATActionIO.decode_velocity`。

实际特征消费者是 `InterActDex → RectifiedFlow.compute_loss(return_features=True) → CoupledDiTX`。普通 DiTX、SAT、DQ-RISE、DexLatent 没有对应消费者。

- `DiTXOutput.forward` 只返回预测 Tensor，删除开关与 tuple 返回分支。
- 普通 `DiTX.forward` 删除 `return_features` 参数、联合返回类型与转发，直接返回普通输出或 `action_io.decode_velocity` 的结果。
- `SATActionIO.decode_velocity` 改为直接读取 `self.final_layer(tokens)` 并转置为动作速度，删除主动请求并丢弃 features 的路径。
- 保留 CoupledDiTX 自己的特征返回、`RectifiedFlow.compute_loss` 的对应分支及 InterAct 辅助损失。保留现有耦合特征、注意力隔离、非退化梯度及缓存测试。
- 保持 module/parameter 的含义、名称、布局和初始化不变，不移动实际输出所需的 normalization。不为此次接口收窄新增输出容器、抽象基类或特征提取器。
- 保留 SATActionIO 本身：它承载真实的关节轨迹 token 语义。删除的是未使用的通用出口，不是 SAT 算法。
- 更新 `docs/手部动作表示.md`、`docs/项目架构.md` 中 SAT 的特征返回描述，保留 InterAct 的准确说明。

完成判据：SAT/DQ-RISE/DexLatent 的预测形状与训练语义不变；CoupledDiTX 的 velocity/features 和 InterAct 辅助梯度路径仍正确。不要把少分配一个 tuple 宣传成显著吞吐优化。

### F06 — 删除未接入的 CLIPTextEncoder

主要位置：`agents/obs_encoder/text.py`、`docs/项目架构.md` 的文本组件保留说明。

- 删除该孤立实现，清理实际发现的导出、引用和说明。基线没有策略、Hydra 配方或训练入口调用它。
- 不迁移到 experimental/archive，不保留空类或弃用别名，不添加未来文本扩展接口。
- `transformers` 仍被 RGB 组件使用，不因删除此文件移除共享依赖。

完成判据：现有观测检查和正式配置的构造引用仍正确。无需新增“类不存在”测试，也不声称删除一个默认未加载的文件能改善训练吞吐。

### F07 — Flow 直接调用已有 KV 接口，保留缓存收益

主要位置：`agents/action_decoders/flow.py` 两个采样循环、`agents/action_decoders/sampling/rtc.py::guided_flow`。

当前 Flow 正式使用的 DiTX、CoupledDiTX、ConsistencyDiTX 均有 `prepare_context_kv`，现有 decoder 检查中的 Head 也提供它。因此无需每次预测用 `getattr(..., None)` 猜测能力。

- 在 **eval 且采样步数大于 1** 时直接调用 `model.prepare_context_kv(cond)`，否则令局部 `context_kv = None`；网络调用显式传 `context_kv=context_kv`。
- 同步修改两种 Flow 的普通路径和 RTC 路径，删除仅服务于可选缓存能力的探测、字典拼装与回退。
- 保留 `guided_flow` 因 RectifiedFlow/ConsistencyFlow 的 `target_t` 差异而需要的最小参数分支或字典。不要为形式统一把无关参数强塞给另一算法。
- 缓存只在一次预测内复用，不跨观测或跨请求持久化；不新增 cache manager、supports-cache 开关、协议类或统一 backbone 基类。
- 保留真正的 KV 复用以及 RTC 的正确求导边界。CoupledDiTX 每次速度估计仍须重建动态 body/interaction 状态，不能把它们误当作固定 context 一起缓存。

完成判据：现有缓存/非缓存等价、Coupled 动态状态变化和 RTC 检查通过；训练/单步路径不提前建立多步缓存。无需为删除的“无缓存接口 backbone”回退继续造测试。

### F08 — 删除 FPS 选点噪声及所有正式配置键

主要位置：`agents/obs_encoder/pointcloud/ops.py::farthest_point_sample`、`agents/obs_encoder/spatial.py` 的 FPS 默认值，以及以下八份 YAML：

`configs/{dp3,r3d,maniflow,adaptac,interact_dex,sat,dqrise,dexlatent}.yaml`。

基线全部正式值为 `random_noise_scale: 0.0`。此前对旧 checkpoint 所存配置的顾虑，在本次“不考虑兼容”授权下不再构成保留理由。

- 删除 `random_noise_scale` 参数、`noisy` / `sampling_xyz` 和扰动 XYZ 的分支，始终用原始 XYZ 运行 FPS。
- 删除 Spatial 默认值、八份 YAML 键及已失效的当前说明/检查。调用方直接使用删减后的配置，不 `pop` 旧键、不悄悄忽略旧参数、不迁移旧 checkpoint。
- XYZ 输入继续复用原生 sampled XYZ；XYZRGB 等带额外通道的输入继续按原始索引 gather 全部通道。原 `not noisy and C == 3` 分支简化为 `C == 3`，保留已有快路径。
- 保留随机起点、输出 shuffle、eval 确定性及 training/eval 开关语义；默认噪声原本为零，不应额外消耗或减少随机数。保留原有索引顺序与 XYZ/RGB 对齐。
- 保留 Dataset 使用中的坐标、颜色和状态增强；它们不同于仅改变 FPS 选点的噪声。保留 Uni3D/Sparse3D 实际使用的不同 kNN 实现，不顺带删除后端。

完成判据：固定 seed 的小型 XYZ/XYZRGB 样例保持采样点、索引和通道对应；eval 仍确定。优先补入现有 observation 检查，不建新测试框架。新版本配置经 fresh/save/resume 不重新出现该键；旧版本因该键不兼容可自然报错，无需专门测试所有旧格式。默认选点噪声本来关闭，不承诺吞吐改善。

## 3. 保护已有的真实边界

以下不是本次删除对象，不能为了“更薄”破坏其语义：

- FP32 master parameters 与 EMA、BF16 autocast；optimizer 参数覆盖/顺序核对、strict load、参数设备迁移和 EMA/optimizer 引用建立顺序。
- 保存的 `normalization_dims`、train-only 统计、预训练 affine 继承、DQ 量化前检查，以及输入数据真实数值/物理含义的检查。
- 主策略与预训练的 update 计步、尾组累积、全部 micro-batch 有限 loss 合并、scheduler/EMA 时序；AFRO teacher 位于 online optimizer 更新前的算法时序。
- 观测预训练的 fresh-only AFRO 预算、logger 初始化后恢复 RNG、已完成早退、验证 RNG/模式恢复和可选 held-out 验证。F03 不把主策略方案扩展成统一 resume 框架。
- Runtime/PolicyInfo/RTC 的外部实际用途、动作窗口和采样数学。保留 RTC 求导、Flow KV 缓存，以及 InterAct 的耦合特征和动态状态。
- 已有正式配方和真实消融使用的 `fusion_stage`、`semantic_pool_sigma` 等研究参数。不能仅以参考项目没有同名字段为理由删除。

不改模型结构、损失权重、采样分布、时间离散、数据 split 或恢复预算。AdapTac force predictor 保持当前 eager、仅 action decoder compile 的支持边界，不在本任务重新挑战历史非有限梯度组合。

## 4. 实施顺序与适度验证

建议顺序：先 F01/F02/F03 收敛训练路径，再在同一轮接口修改中处理 F04/F05/F07，最后 F06/F08 和当前文档。顺序仅方便阅读和验证，不要求拆成八个 PR 或建立新的执行流程。

优先使用已有可用训练环境，不为本任务批量升级依赖。复用现有检查入口；具备所需依赖时，CPU 验证命令如下：

```bash
python -m dexmani_policy.test_tools.training_checks --device cpu
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
python -m dexmani_policy.test_tools.hand_prior_checks --device cpu
python -m dexmani_policy.test_tools.decoder_checks --device cpu
python -m dexmani_policy.test_tools.observation_checks --device cpu
```

| 验证范围 | 使用现有覆盖，并只补确实缺失的行为 |
| --- | --- |
| 主训练 / resume | 连续训练与分段恢复；raw/EMA/optimizer/scheduler/RNG；尾组、非有限早期 micro-batch、首步/周期/末步日志；新增一例真实完成后的构造前早退 |
| 观测预训练 | AFRO/CordViP 原有恢复、teacher 时序、RNG、验证和日志行为，确认删计时没有改变算法顺序 |
| 手先验 | 小型 VQ 更新、有限 loss、现有码表导出和真实 roundtrip；不为不裁剪时故意放宽的梯度诊断补机制 |
| Decoder | FP32 采样状态、BF16 condition/训练、零/非零 RTC、KV 缓存等价、Coupled 动态状态与 InterAct 特征语义 |
| 观测 / FPS | 原有观测检查，少量固定 seed 的 XYZ/XYZRGB 索引与输出对照；检查正式 YAML 已删除噪声键 |

若有合适 GPU 环境，再运行受影响的 decoder/observation CUDA 检查及一个已有的短训→恢复 smoke。使用现有小规模入口/配方，不新增命令参数，不要求所有策略 × 精度 × compile × worker 的组合矩阵，也不要求全量真实数据训练。

验证应与改动风险相称：

- 已有检查覆盖的行为不重复建立测试；不测试 PyTorch 自带 strict-load 行为，不为孤立类删除、旧参数报错或删除后的代码缺席写专门测试。
- 搜索残留参数和实际调用，检查最终 diff、语法及文档一致性即可；不要把每次清理的 grep 结果做成永久 CI 门禁。
- 测试失败时定位相关真实问题；依赖/GPU 不具备时，明确列出未执行项和限制。不能把语法检查、mock 检查或静态阅读写成 GPU 训练验收通过。
- 不要求在本任务中测性能；没有测量就不报告加速比例。F02 的收益仅能表述为减少日志步人为同步和特殊控制流。
- 相关检查通过且没有未解决的正确性疑点后结束，不继续发散审查、扩大重构或反复跑无关测试。

## 5. 五个参考项目与取舍依据

以下是两轮审查使用的固定源码，不要求实施者重新逐文件审查五个仓库。参考用于判断什么是必要算法、什么是本地额外支持义务，**“参考没有”本身不构成删除理由**。

| 参考与固定版本 | 重点对照 | 对本方案的意义 |
| --- | --- | --- |
| [Diffusion Policy · 5ba07ac](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f) | `diffusion_policy/policy/diffusion_unet_image_policy.py` 的条件采样入口 | 不需要额外暴露 runtime dtype 选择；其内部 dtype 来源不等于本项目统一 FP32 的决策 |
| [ManiFlow · e4dc24d](https://github.com/allenai/maniflow/tree/e4dc24d62a6c91825813308b9926e921b4bb9ef6) | `maniflow/model/diffusion/ditx.py`、`maniflow/model/vision_3d/point_process.py`、Dex workspace 与 pointcloud policy | 直接输出与原始 XYZ 选点可作为简化参照；其语言条件有真实消费者，不能据此声称五参考都没有文本能力 |
| [DQ-RISE · 2889d27](https://github.com/rise-policy/DQ-RISE/tree/2889d27fce823288e8dd12ec6e91506b52ed086a) | `train_vqvae.py`、`policy/diffusion.py` | VQ 可使用直接 backward/optimizer 路径；其采样 dtype 跟随内部表示，不应误称为所有参考都固定 FP32 |
| [AFRO · 283024a](https://github.com/KolaKivy/AFRO/tree/283024a49e2c14bc030debf15383a4e7ee73836d) | `AFRO_Pretrain.py`、`afro_workspace/policy/afro_policy.py` | 保留 teacher 更新等算法内容，避免把预训练变成通用生命周期框架 |
| [CordViP · ad6d441](https://github.com/xuanxuanzzzii/cordvip/tree/ad6d441abe7b0480cf79b0311db266965c1313b6) | `train_policy/policy/CordViP/CordViP/CordViP/policy/CordViP_train.py` | 保护真实预训练目标、数据和表征语义，不以机械去重消除必要差异 |

本方案依据当前实际消费者主动收窄范围；不复制参考实现中的大型 Workspace、反射保存、dill 或其他平台设施，也不把参考项目没有本地优化当作删除 KV 缓存等有效机制的理由。

## 6. 交付要求

交付代码、同步后的正式配置/当前说明，以及必要的小型检查修改。保持历史任务书作为记录，不把本任务变成新一轮 contract/evidence 文档体系，不增加单独的验收框架或兼容清单文件。

最终报告用中文简要说明 F01—F08 的完成/已等价完成/有真实阻碍状态、实际删除的能力、保留的关键边界、运行的命令和结果，以及未验证的限制。说明本次不承诺旧版本兼容，但新版本自身的保存与恢复是否已验证。不要把待执行计划写成通过，不把源码推断写成实测性能。

尊重已有工作区修改，不覆盖或提交无关内容；完成相关实现和验证后结束，不自动合并 PR 或扩展到设备端仓库。
