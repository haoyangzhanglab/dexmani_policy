# 预训练后续精简与优化任务书

状态：待实施。本文汇总上一轮实施后的两轮新审查，2026-11-03。

审查基线：[`156fe679e6825fb306d0488e1f945d9ee9f60b2f`](https://github.com/haoyangzhanglab/dexmani_policy/commit/156fe679e6825fb306d0488e1f945d9ee9f60b2f)。[上一份任务书](pretraining_simplification_task.md)的 T1–T8 已落地，保留其实施记录；本任务书只补充新问题与后续决策。开始实施时核对最新代码和真实调用方，已解决项不重复修改。

## 1. 目标、授权和收敛方式

将项目维护为个人研究代码：入口直接、接口只暴露实际使用的结果、配置服务当前实验、正确性检查集中在真正的输入边界。减少理解与修改成本，同时保持训练、导出和下游使用正确。

- 允许删除内部类、参数、返回值、重复检查及其附属测试；允许新 checkpoint 键名和码本编号变化。
- **不考虑旧产物兼容。** 不加迁移器、旧字段别名、版本分派、`strict=False`、双格式或自动回退；需要时重新训练或导出，但不删除用户已有实验文件。
- 同一新版本自身的保存、加载与已有续训能力仍须成立；独立 VQ 不因此扩建完整续训功能。
- “放宽”指信任已绑定的数据维度与内部接口、不追求跨平台逐位一致、不为假想输入建立防御层；不允许把实际的非有限输出、错误 affine 或恢复状态静默转换为可用结果。
- 基础工作 F1–F8 为确定实施范围。E1/E2 是独立、按条件决定的优化；没有必要环境时注明未测，不阻塞基础工作，也不留下半成品分支。
- 不用删行数、报告数量或覆盖率指标驱动重构；不新增 evidence/contract/source 框架、注册表、通用校验器、配置平台、性能基准平台或 CI 体系。

## 2. 两轮新结论与参考依据

| 发现 | 判断与决定 |
| --- | --- |
| 主训练 `PolicyTrainer` 在 scheduler 更新后记录 LR | 确定的日志错误；预训练已修复，主训练仍遗漏。记录本次更新实际使用的 LR |
| epoch 边界日志读取已递增 step 推导出的 epoch | 日志与产生该 loss 的 epoch 不一致；使用训练循环已有的 epoch，不改变恢复位置语义 |
| `HandStateMLP` 只包装已有 MLP builder | 删除类，保持层深、ReLU、输出线性层和正交初始化 |
| VQ/运行时码本暴露调用方丢弃的索引与分层 loss | 收短生产接口；算法内部仍保留必需的最近邻索引和 EMA 计算 |
| `evaluate_vq` 切换模式并用 finally 恢复 | 当前只读取码本并经过 Linear/ReLU decoder，不调用量化器 forward；删除无必要的状态切换 |
| `load_codebook → set_prototypes` 再做相同 shape 检查 | 校验集中到 NPZ 加载边界，删除 setter 及内部重复检查 |
| sklearn 只用于一次一维 PCA 及对应测试 | 改为 NumPy 中心化 SVD，保留 PCA 语义，移除直接依赖 |
| 独立 encoder 导出普通加载完整 resume checkpoint | 使用现有 Torch 的 `mmap=True`，避免立即读取未使用的大块 tensor storage |
| contact 公式的 gamma/theta 没有可变调用 | 收为现有配方常量；保留真正影响内存的分块参数 |
| 观测训练 AdamW 可用现有配置传入 fused | 用已有 override 测量，不另建 optimizer 机制，不盲改默认值 |
| AFRO FDM 只有一个 token，Q/K 不影响输出 | 数学上可移除；初始化、训练参数化及整步收益需独立验证 |

**修正 PCA 风险的表述：** `requirements.txt` 固定 sklearn 1.7.2，不能称推荐环境默认存在随机排序 Bug。`pyproject.toml` 未限定 sklearn 版本，较旧版本的 auto solver 在较大码本上可能选 randomized；例如 sklearn 1.4.2 对 512×12、一个主成分会进入该路径。默认 16 个原型不受这一特定分支影响。改用 SVD 的主要价值是删除单一用途依赖和 solver 选择分支，不是增加随机种子或配置选项。

对照版本和可直接查看的源码：

| 参考 | 源码 | 本轮采用的边界 |
| --- | --- | --- |
| DQ-RISE `2889d27fce823288e8dd12ec6e91506b52ed086a` | [VQ-VAE](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/vqvae_rise/vqvae.py)、[导出](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_vqvae.py)、[ResidualVQ](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/vqvae_rise/vector_quantize_pytorch/residual_vq.py) | 保持加权 residual VQ 和解码手姿 PCA；不继承其多功能通用量化器接口。参考导出设置了 seed，不能据此称它每次随机；也不照搬其硬编码层权重 |
| AFRO `283024a49e2c14bc030debf15383a4e7ee73836d` | [预训练方法](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/policy/afro.py)、[FDM](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/afro_workspace/model/transformer/DiT1D.py) | 保留 teacher、双向目标与 AdaLN 训练机制；单 token 可约简不等于任意替换网络 |
| CordViP `ad6d441abe7b0480cf79b0311db266965c1313b6` | [接触公式](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/script/contact_map.py)、[预训练方法](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/train_policy/policy/CordViP/CordViP/CordViP/policy/CordViP_pretrain.py) | 实际生成路径使用 gamma=1、theta=10；保留接触公式、交叉注意力、自定义归一化及现有 arm7+hand12 适配 |

参考项目用于判断数学与数据语义，不是软件架构模板。不把参考项目的通用 workspace、快照、top-k 管理等机制搬入本项目。

## 3. 确定实施的任务

### F1：主训练日志对齐实际更新

位置：[`training/trainer.py`](../dexmani_policy/training/trainer.py) 的 `train()`、`log_step()`；必要回归放在现有 `training_checks.py`。

1. 在本次 `optimizer_step()` 前，用局部变量取得 `optimizer.param_groups[0]["lr"]`，传给 `log_step()`。
2. `train/epoch` 使用外层循环已有的 `epoch`，代表产生本次 loss 的逻辑 epoch。继续使用现有的零起点约定。
3. 日志的 step 仍为完成更新后递增的 `global_step`；scheduler 顺序、checkpoint 命名、`current_epoch` 和恢复位置计算保持原样。
4. 不引入 `last_lr`、`last_epoch` 等持久状态，也不扩大到重写日志系统或多参数组日志功能。

验收：用会变化的 warmup/decay，记录实际 optimizer 更新所用 LR，与同一步日志逐项比较；覆盖 epoch 最后一组和恢复后的第一组。沿用 `check_training_resume` 的短训/恢复，验证改日志没有改变训练状态。恒定 LR 或只比较 scheduler 当前值不能发现此问题。

### F2：删除 HandStateMLP 包装

位置：[`agents/hand_prior/vqvae.py`](../dexmani_policy/agents/hand_prior/vqvae.py)，使用已有 [`build_mlp`](../dexmani_policy/agents/modules.py)。

- encoder 直接使用 `[hand_dim] + [hidden_dim] * num_layers + [latent_dim]`；decoder 直接使用 `[latent_dim] + [hidden_dim] * num_layers + [hand_dim]`。
- 继续指定 `activation=nn.ReLU, normalize=False`；输出层不激活。保持当前 `num_layers` 的含义，不按参考代码的参数名字重新解释深度。
- 每个 MLP 构建后应用现有 `orthogonal_init`；保留 Linear bias 为零。尽量保持 encoder 构建/初始化、decoder 构建/初始化的顺序，不对整个 VQ 模型调用会误改其它模块的初始化。
- `num_layers` 的必要正整数检查移到 `HandVQVAE` 构造处一次；删除 `HandStateMLP` 类。保留有实际算法职责的 `ResidualVectorQuantizer`。
- 同步 checkpoint/测试引用，不为新的 Sequential 键名编写转换逻辑。

验收：对简单深度和默认深度，用逐层对应的相同权重一次性对照输出、loss 和梯度，确认没有多/少一层或多余输出激活；新版本 checkpoint strict roundtrip 成立。不永久保留一份旧 MLP 实现当兼容层或测试框架。

### F3：生产接口只返回真正使用的结果

位置：VQ 模型、[运行时 codebook](../dexmani_policy/agents/hand_prior/codebook.py)、[DQ-RISE policy](../dexmani_policy/agents/policies/dqrise.py)、[VQ 训练](../dexmani_policy/training/pretrain/hand_prior/vqvae.py)及真实测试调用方。

| 接口 | 新返回值 | 必须保持 |
| --- | --- | --- |
| `EMAVectorQuantizer.forward` | `(quantized, commitment)` | 内部最近邻 indices 继续用于 embedding/EMA；量化值与 straight-through 梯度不变 |
| `ResidualVectorQuantizer.forward` | `(quantized, total_commitment)` | commitment 仍是各层之和；可直接在原有 `torch.stack(losses)` 后 `.sum()`，不必为删列表增加逐层标量 GPU 运算 |
| `HandStateCodebook.encode` | continuous indices Tensor，保留末尾单维 | 连续索引映射和 K=1 行为不变；不返回离散索引 |
| `HandStateCodebook.decode` | decoded hand poses Tensor | 保留 clamp、half-up rounding 和输入维度对应的输出形状；不返回离散索引 |
| `evaluate_vq` | `float lookup_mse` | 调用方自行放入已有 `val_lookup_mse` 指标，不保留单键字典 |

不删除实际计算需要的索引，不增加 `return_indices`、调试模式、结果 dataclass 或另一套 encode/decode 方法。`HandVQVAE.forward` 的三个 loss/metric 均被训练使用，保持不变。

验收：更新现有检查，使其直接验证选中的原型值、连续索引、half-up/K=1、EMA 数值、commitment 和梯度；不因测试曾读取离散索引而保留无生产用途的返回值。仍覆盖近重合原型下的最近邻正确性。

### F4：删除 VQ lookup 评估的无用模式切换

位置：`training/pretrain/hand_prior/vqvae.py::evaluate_vq()`。

删除 `training = vqvae.training`、`eval()`、`try/finally` 和模式恢复；保留 `@torch.no_grad()`、原型构建、分块距离、按真实样本数统计及空集合/非有限结果检查。

成立依据是当前调用链只读取 codebooks、softmax 权重并运行 Linear/ReLU decoder，不执行 quantizer forward、EMA、Dropout 或 BatchNorm。实施时核对这一前提；只简化本函数，不批量删除其它验证过程的模式/RNG/资源恢复。

验收：从 train 与 eval 两种起始模式调用，lookup MSE 一致，参数、EMA buffers 和各子模块模式均不改变；保留现有不干扰训练随机流的行为。结合 F3 返回单个标量。

### F5：将 codebook 校验集中在 NPZ 边界

位置：`agents/hand_prior/codebook.py::load_codebook()` 和 `HandStateCodebook`。

1. 删除 `set_prototypes()`；当前只有 loader 和测试使用它。loader 根据 NPZ 的 `[K,12]` 创建表后直接复制 buffer，不再重复比较由同一 shape 构造出的 shape。
2. 把 setter 中真实需要的 dtype/finite 检查移到 loader：拒绝 complex/bool；转换 FP32 后检查有限性，防止有限的高精度值转换溢出。保留非空 K 的既有边界，不另加第二套检查。
3. affine 继续在同一加载边界检查 shape、实数类型、FP32 finite 和 scale 非零；保留 `allow_pickle=False`。不要对模型每步重复这些检查。
4. `encode` 的手宽检查可随本任务删除，依赖已有 action/data/model 绑定和张量操作；不扩展为通用任意手模型接口。测试准备可信表时可直接填 buffer，加载边界测试通过真实 NPZ 进入。
5. **保留 `decode` 在转整数索引前的 finite 检查。** NaN/Inf 经转换和 clamp 可能变成合法编号，后续输出 finite 检查无法发现这一错误；这是有实际作用的检查。

验收：复用新 NPZ 加载/策略推理检查，保留对真实坏输入的少量覆盖（非有限/转换溢出、无效 affine）；不创建独立 validator、schema 或负面用例矩阵。shape 错误由边界一次明确拒绝即可。

### F6：用 NumPy SVD 保留 PCA 排序，移除 sklearn

位置：[`training/pretrain/hand_prior/codebook.py`](../dexmani_policy/training/pretrain/hand_prior/codebook.py)、`test_tools/hand_prior_checks.py`、`pyproject.toml`、`requirements.txt`。

在现有 `len(poses) > 1` 分支内直接实现：

```python
centered = poses - poses.mean(axis=0)
_, _, axes = np.linalg.svd(centered, full_matrices=False)
axis = axes[0]
if axis[np.abs(axis).argmax()] < 0:
    axis = -axis
poses = poses[np.argsort(centered @ axis)]
```

- 保留 learned layer weights → decoder → FP32 finite → clamp `[-1,1]` → 解码手姿主成分排序的顺序；不得改成 latent 排序或取消排序。
- 最大绝对值分量为正只是明确轴符号约定。K=1 继续跳过 SVD；重复/全相同原型允许存在，不加去重、微扰、随机回退或分支配置。
- 同环境相同输入可复现即可；不承诺退化主空间、近似相等投影在不同 BLAS/平台逐位同序。无需维护旧码本编号；重新导出后使用新码本重新训练相应策略。
- 删除生产及测试 sklearn import，移除两个依赖声明中的 scikit-learn；若其它依赖间接安装它，不以卸载传递依赖为任务。
- 不用 `PCA(svd_solver="full")` 加保留依赖作为最终方案，不新增 solver/seed/排序配置项或通用 PCA helper。

验收：用第一主轴明确的手工构造 fixture 验证排序方向与结果，不能把同一段 SVD 再执行一遍当唯一预期值。验证排序前后原型多重集合、affine、最近邻 lookup MSE 不变；覆盖 K=1、常见 K=16、较大表及重复原型。端到端测试仍应验证新 NPZ 的下游加载。无需为测试保留 sklearn。

### F7：独立 encoder 导出使用内存映射

位置：[`training/pretrain/obs/common.py::main`](../dexmani_policy/training/pretrain/obs/common.py)。

现有 `torch.load(args.checkpoint, map_location="cpu", weights_only=False)` 增加 `mmap=True`，沿用仓库主 checkpoint loader 已有做法。当前 Torch 2.4.x 与本项目保存格式支持该路径；只需这一处直接修改，不引入新格式、专用 reader、meta 初始化或失败后重试普通加载。

作用是延迟读取 optimizer、teacher、heads 等未使用的 tensor storage；pickle 元数据仍会解析，不能描述为只读取了两个字典键或保证某个 RSS 降幅。

验收：使用新版本实际保存的含 optimizer 状态的 resume checkpoint，经独立导出入口生成 encoder，并与训练结束时直接导出的 encoder/normalizer 比较。无实测不宣称内存或耗时改善比例。

### F8：收掉无实际调用需求的 contact 参数，收尾测试与文档

位置：[`datasets/preprocess/contact_map.py`](../dexmani_policy/datasets/preprocess/contact_map.py) 的 `compute_contact_map()` 及相关使用文档。

- 当前生产调用固定 gamma=1、theta=10，测试也不改变二者；删除这两个形参，公式内直接使用既有常量。若实施时最新代码确实已有对应消融调用，保留该实际需求并简要说明，不为潜在未来实验保留参数。
- 保留 `point_chunk_size`：它有明确的内存控制用途，现有测试也验证跨块结果。保留接触公式、FP64 中间计算、normal 的符号不变性与最终 FP32 标签。
- 保留 Open3D 的独立预处理入口、spawn 和失败清理；不要重新把预处理依赖引入核心预训练检查。
- 同步实际变化的 API/依赖/用法，删除过时示例；只更新相关文档段落和本任务的完成状态，不增加审计报告目录或运行时 provenance 字段。上一份任务书作为已完成历史，不重写其历史结果。

验收：复用已有接触公式及分块检查；若文件已改，运行独立接触检查。依赖不具备时如实报告，不为删除两个参数建立新的测试设施。

## 4. 独立优化：满足条件才合入

### E1：通过现有 override 实测 fused AdamW

观测预训练已将 `**cfg.optimizer` 传给 AdamW，可直接用 `+optimizer.fused=true` 进行 CUDA 实验，不需要生产代码分支、optimizer 工厂、新 CLI 或自动设备回退。

在支持的 GPU/Torch 环境中对 AFRO/CordViP 测完整训练 update（含 backward、teacher 更新、optimizer），使用相同输入规模、warmup、同步与重复测量；检查短训数值和同版本恢复。不能只看 isolated optimizer microbenchmark，也不要求 fused 与普通实现逐位相同。使用已有记录方式即可。

若观察到可重复收益且训练/恢复正确，在使用文档写一条推荐 override；默认配置保持现状，以保留直接 CPU 运行体验。无 GPU 或收益不明确时记录未采用，不为此添加自动探测或开关矩阵。

### E2：AFRO 单 token Q/K 的局部约简

仅针对 `training/pretrain/obs/afro.py::ForwardDynamics`。当前每个 block 的 attention 输入长度恒为 1，attention/projection dropout 为 0；softmax 恒为 1，因此输出可写为 `output_proj(value_proj(x))`。默认 H=256、depth=4，删除 Q/K 含 bias 共减少 `4 × 2 × (256² + 256) = 526336` 个参数。

合入条件与实现边界：

1. 将 AFRO 局部 attention 约简为独立的 V 与 output 两个 Linear，去掉无效 Q/K、head reshape 和 SDPA。删除因此无用途的局部 `num_heads` 参数；不修改其它多 token policy 的 shared attention，不增加 `single_token_mode` 或按 token 数运行时分派。
2. 保留两个 Linear 的参数化。合并成一个 Linear 虽可匹配一次前向，却改变优化动力学和 weight decay，不属于本项等价精简。保留 LayerNorm、AdaLN、gate、residual、MLP、位置/条件投影。
3. 保留 V 初始化的分布：旧 fused QKV 的 Xavier fan-out 为 3H，新 H×H V 不能直接用默认 Xavier。可在现有初始化后仅对 V 使用 Xavier gain=`1/sqrt(2)`、bias=0，以匹配旧 V 切片方差；output projection 仍按原方式初始化。不需要模拟已删除 Q/K 的 RNG 消耗，因此不承诺相同 seed 从头逐位一致。
4. 用同一组映射权重比较前向、输入与保留参数梯度、optimizer 更新；映射只用于一次性验证，不进入 checkpoint loader。测试必须包含非零 AdaLN gates，不能只用初始全零 gates 得到无意义的“等价”。
5. 验证新模型的同版本保存/续训。GPU 上测完整 AFRO update 的时间和峰值内存，说明 batch/frames/points/dtype；不能把 FDM 局部节省说成 PointNet 主导的整网同比例提升。

基础任务完成后再做本实验。只有验证通过、实现仍直观且有实测价值时合入；环境不足或收益不足则保持原实现，在交付中简述结论。不留双路径、实验开关或通用专用化框架。

## 5. 本轮保留的正确性边界与不扩张项

| 范围 | 决定 |
| --- | --- |
| 已落地改进 | 保留 W&B 覆盖规则、预训练 LR 修复、teacher 前向提前、删除 act_scale、VQ 尾批、CordViP 零权重跳过、直接 NPZ 导出及测试依赖拆分 |
| VQ 算法 | 保留加权 L1、commitment 求和、detached residual、learned softmax layer weights；保留 EMA 的单位伪计数/累计和一致性、kmeans/dead-code 能力、初始化 buffer/cache/load hook |
| 数值与选择 | 保留 direct FP32 cdist 及 `donot_use_mm_for_euclid_dist`；有 val 按 lookup_mse 选 best，无 val 导出 last；不额外扫描训练集选点 |
| 归一化与产物 | train 源帧拟合；保留 hand affine、encoder+normalizer 继承和 identity、独立加载所需 model_config；NPZ 保持 poses+scale+offset，不加身份/hash/evidence 字段 |
| AFRO/CordViP | 保留 teacher 独立初始化、backward 后且 online optimizer 前更新、momentum 配方、双向 VICReg；保留 CordViP 自定义 sample-std LayerNorm、cloned initialization、交叉注意力和既有损失 |
| 当前版本恢复与文件 | strict model load、optimizer 参数顺序、scheduler/step/RNG、有效配置；原子写入、输出冲突、真实资源释放和必要 RNG/模式恢复继续保留 |
| 不扩张的研究项 | CordViP 点序/固定槽位、目标信息可见性、采样 stride/padding/horizon、AFRO schedule、无 contact 数据模式、混合大小 microbatch 累积均另属研究/功能变更 |

特别说明：混合大小 microbatch 的限制保护当前等权 loss/metrics，不能只删除报错。AFRO VICReg 的 batch 统计也不能通过普通样本权重累积等价为大 batch。VQ 尾批已放宽，不需再动共享 loader。

上一轮 teacher 前向换序的仓库记录在 RTX 4090、B=128/F=17/P=1024 下，完整 forward/backward 峰值仅从 12133.0 MiB 到 12129.8 MiB；不能再宣传显著显存下降。新优化同样以实际测量为准。

## 6. 最小验证、执行顺序与交付

建议顺序：F1 → F2/F3/F4/F5/F6 → F7/F8 → 按环境决定 E1/E2。测试随相关改动完成，优先扩展已有检查，不一条任务新建一套测试。核心命令：

```bash
python -m dexmani_policy.test_tools.hand_prior_checks --device cpu
python -m dexmani_policy.test_tools.pretrain_checks --device cpu
python -m dexmani_policy.test_tools.contact_map_checks
```

主训练至少运行 `training_checks.check_training_resume(torch.device("cpu"))` 及此次改动关联的日志检查；接口改动需贯通新 VQ checkpoint → NPZ → DQ-RISE 训练/推理，利用已有 `check_codebook_training` 等检查，不强制为此启动无关仿真环境。具备 CUDA 时对核心预训练检查运行相同的 `--device cuda:0`，E1/E2 的性能测量单独说明。

验收以以下行为为准：

- [ ] F1：变化 LR 与实际 update 对齐，epoch 边界与恢复日志正确；训练状态未变。
- [ ] F2–F5：MLP/量化损失/梯度/EMA 正确，接口消费者全部更新，lookup 评估无状态副作用，必要外部边界与索引 finite 检查保留。
- [ ] F6：一维 PCA 几何语义正确，原型集合与 lookup MSE 不变；生产/测试/直接依赖中不再需要 sklearn，新码本可供下游使用。
- [ ] F7：mmap 独立导出与直接导出的 encoder/normalizer 一致。
- [ ] F8：接触公式/分块结果未变，相关文档和测试同步。
- [ ] E1/E2：分别注明采用、未采用或未测及原因；若合入，提供实际数值/恢复/性能验证，未引入双模式或兼容层。
- [ ] 最终 diff 无无关变更、无新工程框架、无旧接口残留；`git diff --check` 通过。

两轮审查已完成源码对照和局部验证：日志控制流显示实际使用 0.001 而记录 0.0005；NumPy 单 token 对照、MLP 结构对照及 SVD 与 full PCA 的样例比较支持本方案。**这些不等于完成 Torch 梯度、CUDA 性能或真实任务训练验收。** 上一份任务书中的 CPU/CUDA 通过记录属于旧实现，不能当成本轮修改后的验证。

实施者最终用中文报告：实际删除/修复内容、算法或产物语义变化、运行的命令和结果、未验证项、E1/E2 取舍。可在本文末尾追加简短完成记录；不另建 evidence 清单或大篇审查文档。基础任务完成、必要验证通过后收敛，不继续无边界重构。
