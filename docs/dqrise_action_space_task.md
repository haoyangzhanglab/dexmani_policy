# DQ-RISE 动作空间复现：DiTX-8 + 标准 Flow Matching 实施任务书

## 1. 目标与实施边界

在 `dexmani_policy` 中复现 DQ-RISE 的动作空间设计：先离线学习单步手部状态码本，再将手部动作替换为有序码本的一个连续索引，与手臂动作联合生成。观测编码器沿用本仓库的指定实现；策略主干使用普通 8 层 DiTX；训练与推理使用标准直线 Flow Matching。

本任务以官方代码的实际计算为主要实现依据，用论文解释设计意图；允许针对 xArm7/XHand 和本仓库训练框架做必要适配。它不要求恢复官方观测编码器、Diffusion scheduler、机器人平台或复现其成功率数字。

核对基准：

- 论文：[Learning Dexterous Manipulation with Quantized Hand State，v2，2026-03-16](https://arxiv.org/pdf/2509.17450v2)，重点为 §III-B/C/D。
- 官方代码：[rise-policy/DQ-RISE @ 2889d27fce823288e8dd12ec6e91506b52ed086a](https://github.com/rise-policy/DQ-RISE/tree/2889d27fce823288e8dd12ec6e91506b52ed086a)。
- 本仓库审阅版本：[b817ad01e17b6fb4972f0c48683c32bac78caff7](https://github.com/haoyangzhanglab/dexmani_policy/tree/b817ad01e17b6fb4972f0c48683c32bac78caff7)。执行时先读 `AGENTS.md` 和当前相关代码，保留此版本之后的无关改动；已经完成的事项不重复实现。

**现状结论：本仓库已经实现下述主要架构与训练流程。本次应收敛为两个 Python 文件中的精度修正，以及现有说明文档的少量更新，无须重写 VQ-VAE、DiTX、Flow 或训练入口。** 第 2–5 节用于固定应保留的设计，第 6 节才是本次必须实施的改动。

硬性约束：不新增配置开关、依赖、抽象层、异常处理、运行时校验、兼容/迁移机制、测试文件、测试组件、诊断工具或批量实验框架。不恢复已经删除的 `artifacts.py`、`analyze_codebook.py`、`export_codebook.py`。现有与本任务无关的校验和测试代码不增删。不要把规范中的每条性质实现成新的断言或检查器。

## 2. 固定的模型与动作契约

| 项目 | 本次确定的实现 | 处理方式 |
| --- | --- | --- |
| 数据动作 | `action[..., :7]` 为 arm，`action[..., 7:19]` 为 hand；绝对关节目标，radians | 保留 |
| 手部顺序 | thumb×3、index×3、middle×2、ring×2、pinky×2 | 保留 |
| 观测 | 当前 `SpatialObservationEncoder` 及其配置 | 保留用户指定替换 |
| 窗口 | `n_obs_steps=2`、`horizon=16`、`n_action_steps=8` | 保留 |
| 策略动作表示 | 7D 归一化 arm + 1D 有序手部索引，共 8D | 保留 |
| Backbone | 共享 `DiTX` 的普通时间 token 路径，`action_io=None`；8 层、hidden 256、8 heads、MLP ratio 4 | 保留；不接入 SATActionIO，不复制一份 DiTX |
| 策略生成 | 一个网络联合预测 `[B,16,8]` 的速度 | 不拆 arm/hand head，不增加分类 loss |
| 手部码本 | 两层 RVQ，每层 4 个 code，笛卡尔积共 16 个 decoded prototypes | 保留默认值 |
| 训练阶段 | 独立 VQ 预训练 → 导出固定表 → Policy 训练 | 不联合训练 VQ，不在 Policy 中运行 VQ encoder/decoder |
| 策略输出 | lookup 后为 `[B,16,19]`，逆归一化后由 `format_action_output` 返回 `future_actions=[B,15,19]` | 保留 `action_start_index=1`，运行时执行前 8 步 |

“单步量化”指 VQ-VAE 每个样本仅含一个时刻的 12D 手部目标；Policy 仍联合生成整个动作窗口。不能把它改为手部 chunk VQ、19D 全动作 VQ、两个分别预测的 RVQ 索引、one-hot token 或直接预测 latent。

## 3. 归一化与 VQ-VAE：保留当前正确适配

### 3.1 归一化决策

官方 `dataset/pretrain.py` 对 ROHand 的设备位置值使用：

```text
hand_norm = 2 * clip(hand_raw, 0, 65536) / 65536 - 1
```

其采集文档给出的设备值范围是 0–65535。这是固定量程映射，**没有从整个 replay buffer 或训练子集拟合 min/max**；预训练 loader 的 `split` 参数也没有在该文件中实施数据划分。不能据此把本仓库归一化改为扫描 train+validation。

本仓库输入为 12D radians，不能直接套用 65536。当前采用的训练集逐关节 `limits` 是本任务接受的 XHand 适配，继续使用：

```text
hand_norm[j] = hand_raw[j] * scale[j] + offset[j]
```

非近常量关节的 `scale=2/(max_train-min_train)`、`offset=-1-scale*min_train`；近常量维沿用 `LinearNormalizer` 现有处理，不另写一套公式或分支。

- VQ 仅用实际 train episodes 的真实 `action[:,7:19]` 行拟合 affine；不统计验证数据，不统计窗口重复/padding，不读取视觉数据来训练 VQ。
- 验证样本直接使用训练 affine，可以超出 `[-1,1]`；不要额外 clamp 验证输入来压低误差。
- VQ 导出表、PCA 排序、最近原型标签和 lookup 误差统一在 **normalized hand-pose 坐标**中计算。导出 prototypes 仍 clamp 到 `[-1,1]`。
- fresh Policy 组装时，把码本保存的 hand affine 安装到唯一 action normalizer 的后 12 维；完整 Policy checkpoint 保存该 normalizer 和已排序 prototype 表。保留当前训练组装顺序与 raw/EMA 保存方式。
- 两阶段使用同一份实际 `split_manifest`。VQ 默认不限训练回合，Policy 默认最多 100 回合；按第 7 节显式共用清单即可，不增加自动匹配或重新划分逻辑。

官方各通道使用相同尺度，硬件坐标与其归一化坐标之间的共同平移、正比例缩放不改变欧氏最近邻或 PCA 排序。本地逐关节 affine 则改变了距离度量，相当于按各关节训练范围缩放。这是明确接受的适配，不宣称与官方 6D 硬件空间数值等价。为保持本轮基线简洁一致，不改成 raw-radian PCA，不添加 normalized/raw 双模式，也不猜测 XHand 硬件限位。

### 3.2 模型与损失

保留 `agents/hand_prior/vqvae.py` 的现有实现：

- encoder：`12 → 512 × 6 个隐藏层 → 256`；decoder：`256 → 512 × 6 个隐藏层 → 12`。隐藏层 ReLU，输出线性，Linear 使用现有 orthogonal 初始化。官方训练脚本 `vae_layer_num=5` 对应“首个隐藏层 + 循环 5 层”，即本地 `num_layers=6`，不要误减为 5 层。
- 两层独立欧氏 RVQ，各 4 code；第一层量化 encoder latent，下一层量化 `residual - quantized.detach()`。
- 训练使用 straight-through；各层 commitment 是到 detached quantized vector 的 MSE，两层求和。
- decoder 输入为各层 quantized vector 的 **learned softmax weights 加权和**。残差扣除的仍是未加权的 quantized vector，不能把权重乘进残差更新。
- codebook 用 EMA 更新，`decay=0.8`、`eps=1e-5`；默认 `kmeans_init=false`、`threshold_ema_dead_code=0`。保留当前随机初始化尺度和一致的单位伪计数 `cluster_size=ones`、`embed_avg=embed*cluster_size`，不要恢复官方零计数配随机累积和的初值。
- VQ 预训练保持 FP32；`act_scale=1.0`。不添加 KL、可学习 codebook 的额外梯度损失或新的正则项。

实际优化目标保持：

```text
L_recon = mean(abs(reconstruction - target) * joint_weight)
L_commit = sum_g MSE(quantized_g.detach(), residual_g)
L_vqvae = 3 * L_recon + 5 * L_commit
joint_weight = [1,1,1, 1,1,1, 1,1, 0.5,0.5, 0.5,0.5]
```

重建 MSE 只用于现有日志，不再加到优化目标。论文给出标准 VQ-VAE 的 L2 表达式；官方实际训练使用 weighted L1 + EMA quantizer + 上述 3/5 系数，本任务跟随实际代码。官方手部顺序为 thumb、index、middle、ring、pinky、thumb rotation，权重为 `[1,1,1,0.5,0.5,1]`；本地降低 ring/pinky 的 4 个关节权重是合理的解剖映射，不要求改成全 1。

### 3.3 训练配方与选点

保留 `configs/pretrain/hand_prior/vqvae.yaml` 与现有入口：batch 256、1500 epochs、AdamW、LR `3e-4`、betas `(0.95,0.999)`、weight decay `1e-6`、150 optimizer steps warmup 后 cosine、默认不裁剪梯度；shuffle、`drop_last=true`。这些取自官方实际训练配方；数据、seed 和设备组织沿用本仓库。

选点继续使用固定模型 `eval/no_grad` 下的最终 prototype 表 lookup MSE：

```text
lookup_mse = sum_n min_k ||hand_norm[n] - prototype[k]||² / (N * 12)
```

有 holdout 时按 `val_lookup_mse` 选 best；无 holdout 时保留现有 `train_lookup_mse / train_no_holdout` 标记。验证与导出不更新 VQ EMA。训练结束自动从 best 导出 NPZ，保留现有 best/last/周期 checkpoint 和日志，不增加选模指标或分析文件。

## 4. 码本导出、训练标签与最终解码

继续复用 `training/pretrain/hand_prior/codebook.py`，按以下唯一流程处理：

1. 从选定 VQ checkpoint 恢复模型并 `eval()`。
2. 枚举两层 codebook 的全部 4×4 组合；使用 checkpoint 中 learned softmax weights 合成 latent。
3. decoder 解码，应用现有 `act_scale`，得到 FP32 normalized hand poses，clamp `[-1,1]`。
4. 对这 16 个 **decoded hand poses** 做一维 PCA，按投影升序排列；保存最终排序表和同一 hand affine。PCA 投影只用于排序，不作为 Policy 的非等距标签。
5. Policy 对每个 normalized hand target，在最终表中取欧氏最近原型的排序下标 `i`，再映射为 `s=2*i/(K-1)-1`。继续复用已有 FP32 direct `codebook_distances`，不改为 VQ encoder 产生的 RVQ code tuple。
6. 推理只在 Flow 完成后，将 `s` 按下式取整，查表得到 normalized hand，再与 arm 拼接并逆 action affine。

```text
scaled = (clamp(float32(s), -1, 1) + 1) * (K - 1) / 2
i = clamp(floor(scaled + 0.5), 0, K - 1)
hand_norm = sorted_hand_poses[i]
```

不按官方导出脚本硬编码 `0.5/0.5`：官方训练会学习 layer weights，导出跟随已训练权重才能与 decoder 的训练输入一致。不用整数截断替代最近整数取整，不用 `arm_dim+hand_dim` 替代 `K-1`。也不对重复原型去重或在加载时重做 PCA：排序表是 Policy 标签语义的一部分，直接使用已保存表。

现有 NPZ 字段和 Policy checkpoint 格式均不变。Policy 学习及部署不需要加载完整 VQ-VAE；完整 Policy 恢复不依赖原 NPZ 路径。更换码本才需要重新训练下游 Policy；本次纯精度修改不要求重训 VQ 或迁移 checkpoint，也不承诺混合精度下与旧实现逐位一致。

## 5. 标准 Flow Matching 的精确定义

保留 `DQRISEPolicy` 已显式指定的 `RectifiedFlow(t_sample_mode="uniform", time_shift_alpha=1.0)`，不修改公共 Flow 默认值。

训练时，令 `x1=[arm_norm,s]`，形状 `[B,H,8]`：

```text
x0 ~ N(0, I)
t ~ Uniform(0, 1)                 # 每个样本一个 t，广播到 H×8
xt = (1 - t) * x0 + t * x1
v_target = x1 - x0
loss = mean((DiTX(xt, t, obs) - v_target)²)
```

8 个维度直接做统一 mean MSE；不添加 hand 放大系数、arm/hand 分组权重、分类损失或额外重建 loss。手部从 12D 降到 1D 后在联合目标中的维度占比变化，正是本动作表示要保留的效果。

推理从 `x(0)~N(0,I)` 出发，默认 `N=10`，均匀显式 Euler：

```text
dt = 1 / N
for j in 0 .. N-1:
    x = x + dt * v_theta(x, j/N, obs)
```

沿用已有观测条件缓存、`generator`、`inference_steps` 和设备行为。积分过程中不 clamp/round 手部索引，不注入 VQ lookup；完成后才按第 4 节解码。不引入 DDPM/DDIM、FlowMatchEulerDiscreteScheduler、时间偏移、consistency flow、蒸馏、RTC 或新 solver。

## 6. 本次必须实施的最小改动

### A. 连续索引离散化使用 FP32

文件：`dexmani_policy/agents/hand_prior/codebook.py`，`HandStateCodebook.decode()`。

保留现有 finite 检查、half-up 规则和整数边界 clamp；**仅在缩放之前把 continuous 转为 FP32**，例如将现有计算改为：

```python
scaled = (continuous.float().clamp(-1.0, 1.0) + 1.0) * 0.5 * (self.num_codes - 1)
indices = torch.floor(scaled + 0.5).long().clamp(0, self.num_codes - 1)
return self.sorted_hand_poses[indices], indices
```

这样避免 BF16/FP16 在加 1、缩放、加 0.5 时引入额外的索引边界舍入。转换不能恢复输入此前已经丢失的精度，因此需要同时完成 B。不要为不同 K 添加分支、epsilon、特殊取整方法或新检查。

### B. DQ 推理显式选择 FP32 积分状态

文件：`dexmani_policy/agents/policies/dqrise.py`，`predict_action_from_condition()`。

调用已有 `RectifiedFlow.predict_action()` 时传入它已经支持的 `dtype=torch.float32`：

```python
reduced_action = self.action_decoder.predict_action(
    condition, (condition.shape[0], self.horizon, self.flow_action_dim),
    inference_steps=inference_steps, dtype=torch.float32, generator=generator,
)
arm_pred = reduced_action[..., : self.tcp_dim]
idx_pred = reduced_action[..., -1]
hand_pred, _ = self.hand_codebook.decode(idx_pred)
normalized_full_action = torch.cat([arm_pred, hand_pred], dim=-1)
prediction = self.normalizer["action"].unnormalize(normalized_full_action)
return self.format_action_output(prediction)
```

删除随之冗余的 `hand_pred = hand_pred.to(arm_pred.dtype)`。保持现有 FP32 model buffers/normalizer，避免将 prototype 降精度后再逆 affine。

这固定的是噪声和 Euler 累计状态的 dtype；网络前向仍遵循调用方的 autocast，可以使用 BF16，未要求所有网络运算和速度增量乘法都改成 FP32。不关闭整个策略的 AMP，不强转观测条件，不调用整个模型的 `.half()`/`.bfloat16()`，不改变公共 `RectifiedFlow`、DiTX 或 SAT 路径。当前公共积分器的加法会保留 FP32 状态，直接复用即可。

### C. 同步现有使用说明

文件：`docs/手部动作表示.md`。

只更新 DQ 相关说明：明确 A/B 的精度行为，补充官方固定设备量程归一化与本地 train-only limits 的区别，以及 normalized 坐标中 PCA/最近邻是已接受适配。现有训练命令、窗口、返回值、产物字段和选点规则已正确的部分保留；不新增平行说明文件或诊断步骤。

预计实现 diff 仅涉及上述三个文件。若当前分支已完成 A/B，只保留仍缺失的说明；不要为了达到文件数制造修改。其余第 2–5 节的代码已经符合约定时保持原样。发现当前分支的新问题时，先用实际调用证明它与本任务直接相关，再做必要的局部修正，不扩展为通用框架。

## 7. 使用流程：复用现有入口

以下是用户完成代码修改后的训练用法，不要求实现任务自动启动完整训练。路径由使用者替换；`HAND_RUN` 使用尚不存在的新目录。

```bash
conda activate policy
DATA_ZARR=/absolute/path/to/dataset.zarr
SPLIT_PATH=/absolute/path/to/episode_split.json
HAND_RUN=experiments/vq_hand/dqrise_baseline

python -m dexmani_policy.datasets.split \
  --zarr-path "$DATA_ZARR" --output "$SPLIT_PATH" \
  --val-ratio 0.1 --seed 42 --max-train-episodes 100

python -m dexmani_policy.training.pretrain.hand_prior.vqvae \
  --config pretrain/hand_prior/vqvae \
  "zarr_path=$DATA_ZARR" "dataset.split_manifest=$SPLIT_PATH" \
  "vq_vae.output_dir=$HAND_RUN"

bash scripts/training/train.sh dqrise \
  "zarr_path=$DATA_ZARR" "dataset.split_manifest=$SPLIT_PATH" \
  "initialization.codebook=$HAND_RUN/codebook.npz"
```

已有共用划分时直接使用该清单。默认预训练结束会导出 best，通常无需单独导出；确需从指定 checkpoint 导出时，现有入口为：

```bash
python -m dexmani_policy.training.pretrain.hand_prior.codebook \
  --checkpoint /absolute/path/to/vqvae_hand_best.pt \
  --output /absolute/path/to/codebook.npz --device cpu
```

不要另建 export CLI 或统一调度脚本，也不要把代码任务扩展为参数搜索、K 消融或长期训练。

## 8. 完成标准与适度验证

本节是人工审阅和现有工具验证标准，不要求新增任何测试或运行时校验组件。

1. 当前有效配置仍为普通 DiTX-8、uniform/无偏移 Flow、16 个有序手部原型、7+1 联合生成；两阶段的归一化和固定表语义未改变。
2. FP32/BF16/FP16 continuous 输入都先在 FP32 中完成索引缩放与 half-up；对同一已表示输入，结果与其 `.float()` 后的参考公式一致。不能要求已被低精度合并的不同实数恢复为不同索引。
3. 默认 K=16 的精确码中心往返正确，端点映射为 0 和 15；DQ 推理在普通 FP32 和可用的 BF16 autocast 下维持 FP32 action 状态及最终物理动作。
4. shape、`future_actions` 切片、normalizer、模型参数键和 checkpoint 格式不变；没有额外配置、依赖、分类 head 或测试源文件。

先检查最终 diff 和语法。依赖可用时，直接调用现有 `test_tools.agent_checks` 中的 `check_codebook`、`check_codebook_pipeline` 即可覆盖已有码本与 Policy 链路，不扩展这些函数或遍历全部无关检查。对于此次新增的 dtype 行为，可用一次性 Python 命令验证码中心/舍入边界以及 autocast 推理，不保存成测试组件。只做与此次数行改动匹配的验证，不启动完整训练来证明代码可导入。

缺少依赖、GPU、数据或真实 checkpoint 时如实记录未执行项，不用伪造结果或新增 mock/回退框架补齐环境。交付说明用中文简述实际改动、保留的关键设计、运行过的验证及未验证项；不要宣称已证明任务成功率或量化重建质量最优。

## 9. 官方源码定位

以下固定版本链接用于核对已有决定，无需在实施时重新设计整套方法：

| 依据 | 官方源码 |
| --- | --- |
| 固定设备量程、预训练输入 | [dataset/pretrain.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/dataset/pretrain.py)、[采集顺序与范围](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/assets/docs/COLLECT.md) |
| MLP、weighted L1、模型结构 | [policy/vqvae_rise/vqvae.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/vqvae_rise/vqvae.py) |
| residual、learned weights | [residual_vq.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/vqvae_rise/vector_quantize_pytorch/residual_vq.py) |
| EMA、commitment、初始化 | [vector_quantize_pytorch.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/policy/vqvae_rise/vector_quantize_pytorch/vector_quantize_pytorch.py) |
| 实际优化器、loss 系数、训练预算 | [train_vqvae.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/train_vqvae.py)、[command_train_vqvae.sh](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/scripts/command_train_vqvae.sh) |
| 枚举组合、decoded-pose PCA | [eval_vqvae.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_vqvae.py)；本地有意修正其固定合成权重 |
| 最终表最近邻标签、arm/索引拼接 | [train_dqrise.py](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/train_dqrise.py) |

