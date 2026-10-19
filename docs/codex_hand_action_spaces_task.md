# Codex CLI 任务书：三种灵巧手动作空间的唯一实现

> 状态：待实施的设计与验收规范；提交本文件不表示算法代码已经修改或测试通过。
>
> 审计基线：`haoyangzhanglab/dexmani_policy@ef65cf7848e19bbf3e84cb7ad785840705a0741f`（2026-10-19，`1019 temp`）。CLI 执行时必须重新确认实际 HEAD、工作区修改及调用链，保留此后已经正确完成的工作。
>
> 配套启动文本：[hand_action_spaces_codex_cli.txt](prompts/hand_action_spaces_codex_cli.txt)。本任务书是实施时的完整规范，prompt 只负责启动和执行约束。

## 1. 目标、边界与最终决定

把 DQ-RISE、LAMP、Temporal hand codec 做成可训练、可恢复、可评测的动作空间 baseline：保留论文的关键机制，明确机器人、观测与训练框架的适配，每种方法只有一个实际策略实现。目标是机制正确、运行链路完整和使用简单，不承诺未经实验的最优成功率。

| 方法 | 当前判断 | 唯一目标实现 | 本次变化 |
| --- | --- | --- | --- |
| DQ-RISE | RVQ、原型排序、单标量标签与查表解码主链成立 | 保留 `DQRISEPolicy` 与 `dqrise.yaml` | 补齐数据来源校验、checkpoint 元信息和针对性验证；不重写量化算法 |
| LAMP | 历史先验成立；当前逆解标签 + chunk Flow 缺少论文式 policy offset 约束 | `agents/core/lamp_policy.py::LAMPPolicy`，配置 `lamp.yaml` | 替换为单步直接 BC、冻结历史先验、decoded hand loss + latent offset 正则；删除旧 Flow/逆解缓存链路 |
| Temporal | 有状态条件的时序 VAE、posterior mean 标签和 FM 主链成立 | 保留 `TemporalLatentFlowPolicy` 与 `temporal_codec_flow.yaml` | 固定损失语义、完善选点及产物记录、收紧同步执行契约；不重写 codec |

本次只实现动作空间及其 IL/监督训练接口。**不新增 LAMP 残差 RL，不复现完整 VLA 预训练/在线 RL，不改 SAT 算法，不引入新的视觉主干。** 仓库已有 SAT 仅做受共享接口修改影响时的回归检查。

“唯一实现”指删除被替代的算法分支、类、配置、入口和调用；不要求把三种本来不同的方法抽象成同一个模型。允许同一实现的必要参数，不保留 `legacy`、`v2`、`paper_mode`、`flow_lamp` 等并行选择。历史 checkpoint 使用对应历史提交，不做自动迁移。

## 2. 证据与取舍

先阅读以下论文的机制章节，再核对固定提交的真实训练路径；README、类名或未激活的实验分支不能代替证据。

| 证据 | 要核对的机制 | 使用方式 |
| --- | --- | --- |
| [LAMP 论文 v1](https://arxiv.org/html/2607.06323v1)，§3.1–3.2、Eq.1–6、Appendix B | 历史条件先验；冻结 E/D；预测相对 prior mean 的 offset；解码后监督与 offset 正则 | 本任务的 LAMP 目标采用论文 Eq.6 的 latent offset 正则 |
| [LAMP 官方代码固定提交](https://github.com/dex-lamp/LAMP/tree/0467ba3504e0d199727a0586ba525c5fe14f8138)，[train_jax.py](https://github.com/dex-lamp/LAMP/blob/0467ba3504e0d199727a0586ba525c5fe14f8138/imitation_learning/behavior_clone/scripts/train_jax.py) | 活跃 BC 路径使用 decoded-action supervision；`reg_drift` 约束解码动作相对 prior 解码的漂移 | 官方代码的 decoded drift 与论文 latent L2 不相同；记录差异，只实现本文选定的 latent L2，不把两者叠加 |
| [DQ-RISE 论文 v1](https://arxiv.org/html/2509.17450v1)，§III；[官方代码固定提交](https://github.com/rise-policy/DQ-RISE/tree/2889d27fce823288e8dd12ec6e91506b52ed086a) | residual VQ、组合手姿原型、PCA 排序、标量化、查表 | 对照 `train_dqrise.py` 的最近原型标签及 `eval_vqvae.py` 的导出逻辑；保留本地训练/导出一致的 learned weights |
| [Towards High-DoF Dexterous Manipulation through VLA Post-Training v1](https://arxiv.org/html/2609.19666v1)，§3、Eq.2–10、Appendix A/B | q0 条件时序 codec、相对动作、差分重建、KL、latent FM 与解码监督 | 只复现其动作表示和监督学习机制；12-DoF 单手、H=16、现有视觉编码器均是显式适配 |
| [Wuji 官方发布代码固定提交](https://github.com/wuji-technology/wuji-openpi/tree/be6b32540546fb4fdd274f5cc86a7f9a67148a4c) | 发布范围与实际 action head | 已核对 README、`src/openpi/models/pi0.py`、`src/openpi/policies/wuji_policy.py`、training config；这些路径不能作为 Temporal codec 已逐行复现官方实现的证据。若后续发布新实现，先说明版本差异再取舍 |

项目入口：[DQ-RISE](https://rise-policy.github.io/DQ-RISE/)、[LAMP](https://dex-lamp.github.io/)、[Wuji](https://wuji.tech/blog/post-training-0/)。网页不可访问时使用上述论文和固定代码，不凭摘要补造实现细节。

纠正范围必须准确：当前 LAMP 逆解目标与解码坐标的代数关系是自洽的，不能把它说成必然存在未来信息泄漏；问题是偏离选定的论文 BC 目标并增加额外逆问题。Temporal 当前 `Nobs=2` 的时间切片也不是天然 off-by-one。不要为“修错误”改掉已经成立的机制。

## 3. 共同信号契约

设 `t` 为本次查询对应的最后一帧 observation：`obs[t]` 在执行 `action[t]` 之前采集。`q_t` 是该帧**未做数据增强、未归一化**的实测手关节位置；`a_t^h` 是该时刻下发的绝对手关节目标。两者不是同一信号。

| 信号 | 单位与形状 | 必须保持的含义 |
| --- | --- | --- |
| joint action | `[B,H,19]`，arm 7 + hand 12 | arm/hand 物理关节目标；模型内部归一化不改变外部协议 |
| EEF action，仅 DQ 已有支持 | `[B,H,21]`，arm 9 + hand 12 | 保留已有表示，不能与 joint baseline 的指标直接混作动作空间消融 |
| LAMP history | `[B,K,12]`，物理手关节目标 | 严格过去 `a^h[t-K:t]`，不包含 `a_t` 或预测但未执行的尾部 |
| Temporal reference | `[B,12]`，物理实测手姿 | 本次 query 的 `q_t`，整段编码/解码共用；不能用命令历史替代 |
| `pred_action` | `[B,H,A]` | 已恢复物理坐标的完整预测 |
| `control_action` | `pred_action[:, Nobs-1:Nobs-1+C]` | `C=n_action_steps`；先完整解码，再按统一约定切片 |

继续保留 source alignment、有限值、shape、正 `dt`、关节顺序、训练集统计、episode 边界及 artifact/source 一致性检查。元数据缺失时不能仅根据数组索引猜测时序。显式 `obs_before_action` 断言只在核实数据采集方式后使用。

不要改变全仓 `SequenceSampler` 规则。对 Temporal 当前 `H=16,Nobs=2,C=8`，动作窗口为 `a[t-1:t+15]`，`q0=q_t`，执行索引 `[1:9]` 对应 `a[t:t+8]`；它是包含一个历史位置的窗口适配。episode 边缘 padding 与源行映射沿用实际 sampler，并由定向测试锁定。

## 4. LAMP：单步直接 BC，移除逆解链路

### 4.1 唯一窗口与结构

- 配置和构造器均要求 `horizon=n_obs_steps=n_action_steps=1`；不支持隐式退回 chunk 版本。`K=8` 是独立的手部命令历史长度，不等于 observation horizon。
- 保留 `action_key=action`、19 维 joint action、现有 12 维手与 `latent_dim=2`、prior hidden 256。12→2 是本仓基于论文思想的适配，不能声称原文已证明该机器人上的容量足够。
- 复用现有 `SATObservationEncoder` 的 Sparse3D + proprio 计算；取其已存在的 global token `cond[:,0,:]`。在当前实现中该 token 是 patch 均值与 state feature 的拼接，不需另写注意力聚合器。保留本仓完整 19 维 proprio 是相对论文 RGB + arm proprio 的观测适配；它不能代替独立的 command history。
- 拼接 `[obs_global, mu, sigma]`，其中 `sigma=exp(0.5*log_var)`；MLP hidden `[512,512,256]`、SiLU，输出 `[arm_normalized(7), delta_z(2)]`。MLP 宽度和激活是明确的本地结构选择，不标注为论文原值。
- 在 `lamp_policy.py` 内放置一个小型、真实的回归 head 容器，令其 `.model` 持有全部 head 参数并作为 `action_decoder` 接入 BasePolicy。它只执行一次前向；不包装假的 Flow/Diffusion，不接收噪声、时间或迭代采样。这样复用现有 optimizer、compile、EMA 和 smoke 的 `.action_decoder.model` 路径，无需大改 BasePolicy。
- `LAMPPolicy` 自己实现 loss 和 physical decode，复用 `format_action_output`。`.model` 调用必须经过 compile 包装；不得在外部另放遗漏 optimizer 的 trainable projection。`inference_steps` 保留公共入口形参，但只接受 `None` 或 `1`；配置、eval、PolicyInfo 均设为 `1`，防止旧的 `10` 被静默忽略。

### 4.2 先验、目标与梯度

先验预训练保留当前历史预测下一命令的 VAE：训练用重参数采样，重建为下一手部命令 MSE，加对标准高斯的 KL；保留当前归一化、20k steps、beta `1e-3`、2k warmup、历史归一化空间噪声 `0.01` 的生产默认值。只修必要契约与选点记录，不借机调参。

Policy 阶段取冻结 encoder 的均值和标准差，**不采样 prior**：

```text
mu, log_var = frozen_E(physical_past_command_history)
arm_hat_norm, delta_z = head(obs_global, mu, exp(0.5 * log_var))
hand_hat_norm = frozen_D(mu + delta_z)
L = mean((arm_hat_norm - arm_target_norm)^2)
  + hand_loss_weight * mean((hand_hat_norm - hand_target_prior_norm)^2)
  + offset_reg_weight * mean_batch(sum_latent(delta_z^2))
```

`hand_loss_weight=1.0`、`offset_reg_weight=1e-3` 为固定基线起点，写入配置与 effective config。arm/hand 的组内 mean 是本仓尺度适配，offset 项保留逐样本 latent L2；它们不是已验证的最优超参数。禁止继承旧 Flow 的 `7/19`、`12/19` 权重后又声称完全相同的目标。

- 不引入 `sigma*delta_z`、latent whitening、`latent_scale`、tanh 限幅、离线 `z*` 监督或 decoded drift 第二正则。必要的非有限值检查直接报错，不靠任意 clamp 掩盖问题。
- E/D 参数 `requires_grad=False` 且始终 eval。E 的条件计算可在 `no_grad`；D 在 policy loss 内**不能**放入 `no_grad`，必须保留 `dD/d(delta_z)`。
- prior 手部 affine 是 hand target 和 decode 的唯一归一化来源；加载时核对其与该 policy 数据配方的手部 affine 相符。arm 使用 policy action normalizer 的前 7 维。输出分别逆变换一次，再拼成 `[B,1,19]`；不能把完整输出再次统一 unnormalize。
- 记录分项 loss、`delta_z` L2/RMS、prior-only 与 policy-decoded hand error、physical hand MAE；这些诊断不能变成新增训练项。

### 4.3 删除内容与历史闭环

删除 `lamp_flow_policy.py`、`configs/lamp_flow.yaml`、`training/export_lamp_latents.py` 及所有活跃引用；删除 `HandMotionPrior.invert`、`latent_scale`、`latent_path`、`hand_latent_target`、`require_latents` 和 cache readiness。不是把旧分支挪到别处或留 alias。

`LampDataset` 最终只负责 observation/action、严格过去历史、源数据与划分契约；不读取 latent cache。先验预训练可直接实例化该 dataset，避免“要先有 prior 才能生成训练 prior 的数据”的循环依赖。原来夹在 `load_latents()` 中的 prior/source/split 检查，移动到 fresh policy training 的 dataset–artifact 绑定边界；检查本身必须保留。

删除 buffer 会改变 state schema，使用新的 `lamp_prior.v2` 格式并重新生成先验；不静默接受旧 v1 或旧 LAMPFlow policy。版本校验必须给出历史 checkout/重新训练的明确说明。policy checkpoint 内嵌 E/D、hand affine、ready state 与来源信息，部署无需外部 prior。

训练历史从真实源行构造，episode 开始缺少的 K 个历史位置用该 episode 初始 `q_0` 补齐。SIM 在 `env.step` 成功后逐步追加 `info.executed_joint_action[-12:]`，含执行端裁剪/处理后的实际目标；不能追加未执行尾部或实测位置。保持 reset/bootstrap/每个实际执行步一次的现有正确行为，并更新类名和入口。

## 5. Temporal：保留时序模型，明确唯一损失语义

保留当前 `TemporalLatentFlowPolicy`、`TemporalHandVAE`、`H=16,Nobs=2,C=8`、12→9、3 层双向 Transformer、hidden 128、4 heads、现有 positional/q0 token 结构。不要为了形式统一改成单帧 VAE、PCA、离散码本或整段一个 latent。

### 5.1 Codec 与 policy

```text
r = (physical_hand_chunk - q0[:,None] - train_mean) / train_std
mu, log_var = E(r, q0)                        # [B,H,9]
codec pretraining: z = mu + std_z * epsilon
policy target: z_star = mu                    # frozen, deterministic, no_grad
x_s = (1-s) * noise + s * concat(arm_norm, z_star)
v_target = concat(arm_norm, z_star) - noise
x_clean_hat = x_s + (1-s) * v_hat
hand_hat = D(x_clean_hat.hand_latent, q0) * train_std + train_mean + q0[:,None]
```

- `q0` 使用 query 源帧未增强的实测手姿。训练由 `action_context.hand_reference` 显式传入；推理可从原始 observation 的最后一帧取得。不能从经噪声增强的 proprio、上个 action 或 batch 内混合来源重建。
- Codec 保留带关节权重的 Huber 位置、一阶差分、二阶差分**重建误差**，默认权重 `1,0.2,0.05`。这是匹配目标运动，不是惩罚预测速度趋零；差分不跨窗口/episode。手指权重 `[1]*8 + [0.5]*4` 对应本地已确认的 thumb/index/middle/ring/pinky 顺序。
- 保留 KL 前 4k 为零、后 4k ramp 到 `3e-4`，以及 30k steps 默认值；保留 train-only relative statistics、对重复窗口/padding 的现有统计口径并记录。
- Policy 保留 `L_arm_FM + 0.25 L_latent_FM + 0.75 L_decoded`，各组按自己的维度与全 H 求 mean。唯一 `L_decoded` 在 codec 的 standardized relative 空间计算 MSE；去除 `decoded_loss_space=physical` 训练分支及选择参数，physical MAE/RMSE 仅作为解释性指标。这是明确的本地尺度选择，不声称与 raw-radian loss 数值相同。
- 保留 noise→data 的 `s ~ 0.999*Beta(1,1.5)` 与 10-step Euler；这是论文反向时间约定的变量替换，不能再次翻转 Beta 参数或目标速度符号。
- 冻结 E/D；target encoder `no_grad`；decoded loss 必须穿过 D 回传到 flow head。完整解码 H 个 token 后再取 `[Nobs-1:Nobs-1+C]`；双向 decoder 不能只解码执行前缀。
- 整段物理动作只加一次固定 `q0`。已预测 chunk 在执行中不得用新 observation 重新锚定。新 query 才使用新 q0。
- 禁止缓存“每个源帧唯一 z”：同一帧处在不同 chunk、位置和 q0 下可以有不同编码。继续在线用冻结 E 的 mean 生成目标。

### 5.2 先验选点

LAMP 与 Temporal 的预训练都保存 `selection_step`、当时实际 beta、selection split/metric、reconstruction、KL 和物理误差。用于 policy 的选中模型须来自 KL 达到目标系数后的候选步骤；否则可能只选中了尚未正则化的早期模型。这是本仓实验协议改进，不写成论文硬性规定。

存在 held-out validation 时，仅在符合上述条件的候选中按验证指标选点；LAMP 继续其确定性 prior-mean 重建指标，Temporal 继续 physical MAE。没有 validation 时保存最终合格步骤，标记 `final_no_holdout`，不要把最小训练误差称为验证最优。检查 `steps` 足以覆盖非零 beta 的 warmup/ramp，失败应在训练前说明；微型测试可显式使用缩短的测试 schedule，不改生产默认值。

正式 baseline 实验必须提供 episode 级固定 held-out split；开发 smoke 允许无 held-out 数据，但不能据此报告泛化结论。保持 DQ 现有 VQ 选点方式，此次不混入另一轮量化超参数研究。

## 6. DQ-RISE：保留正确核心，补全来源闭环

保留单帧 residual VQ、2 组 × 每组 4 个 code、16 个组合原型、训练得到的 softmax layer weights、解码后 PCA 一维排序。枚举与训练使用同一权重，不照抄官方导出中的固定 `0.5+0.5` 而造成训练/导出不一致。保留当前 weighted L1 reconstruction + EMA commitment 的作者代码口径，记录与论文公式表述的差异，不强改 MSE。

训练标签取**实际部署原型表**的最近手姿，而不是用 encoder index 冒充部署原型的最近邻：`i*=argmin ||hand_norm-prototype_norm[i]||²`；`s=2*i*/(M-1)-1`。推理使用 `floor((clamp(s,-1,1)+1)*(M-1)/2+0.5)` 后查表。`M-1` 是码本基数决定的分母，与 arm/hand 维数无关。

`affine_raw` 的 `[0,65535]` 是已有码本内部存储坐标，不能直接下发为关节命令；查表后必须完整恢复到物理手部坐标，保持单次变换。保持 16 原型，不因误差大就自动增大码本或加连续残差。

目前 `train_vq_hand --policy-config` 已复用 policy 的 split、eligible source rows 和 action normalizer；不要再新增独立数据入口。重点补齐：即使 affine 数值相同，来自不同数据 revision、训练 episode 或源行集合的码本也不能无提示混入本次基线。继续报告真正 NN→lookup 路径的归一化/物理误差、per-joint/p95、利用率和相邻原型跳变；训练 VQ 重建误差不能替代这些诊断。

## 7. Artifact、恢复与数据划分

复用已有 metadata、data_recipe、split 和 hash 工具，缺什么补什么。不要建立新的 registry、统一 codec 插件系统或任意层级的 manifest 框架。

1. 在产物中记录 method/schema、模型配置、动作键与手关节顺序/单位、训练数据 identity/revision、episode 边界、实际 train/val masks 及 eligible source rows/window 的指纹、相关源数组指纹、normalizer/statistics、窗口参数、alignment/dt、训练/选点设置。LAMP 还需 history/bootstrap；Temporal 还需 q0/relative statistics/窗口统计口径；DQ 还需原型数、排序、layer weights 与 source_summary。
2. 身份校验依赖数据内容和语义，不依赖机器上的绝对路径；复制相同数据/产物到新目录应可用。源码版本作为追溯信息，不因无关文档 commit 改变就拒绝相同产物。
3. Fresh training 在 dataset/normalizer 和 artifact 已就绪的绑定边界校验一次，开始 optimizer update 前完成；strict resume 对 saved source contract 与当前训练数据再次校验。不要在每次 forward、每个 batch 或每个 epoch 重新读 NPZ/pt、扫描整个 Zarr 或计算全量 SHA。
4. 将 compact artifact metadata（kind/schema、artifact SHA、source contract/hash、选点信息）写进 checkpoint 的 effective config，例如单一 `action_space_artifact` 字段；大张量、码本、E/D 与 scaler 仍放 state_dict。不能依赖未序列化的 Python `artifact_metadata` 属性。使用现有 checkpoint 格式承载，只有确有 schema 不兼容才升级，不全仓无差别换版本。
5. Fresh training 必须显式提供所选 artifact；完整 checkpoint 的推理依靠内嵌状态与保存的配置，raw/EMA 都不需要外部 prior/codebook/codec 或训练数据。Strict resume 仍按现有机制恢复数据、优化器、步数等；不得把重新加载另一份外部 artifact 当成恢复模型。若 dataset 构造当前额外需要路径，只能加载并验证同一个 artifact，绝不能替换 checkpoint 内嵌坐标系。
6. 老产物若缺少此次必需的来源契约，明确报错并要求重新导出/预训练；不填假 hash、放宽 strict load 或自动替换。兼容性影响写进 README，旧实验仍可 checkout 历史代码复现。

正式多 seed 对比固定同一 episode split manifest，并在 representation 预训练及 policy 训练全部复用。manifest 是最终划分：使用时设 `dataset.max_train_episodes=null`、`dataset.val_ratio=0`，不要再按 seed 划一次。方法间因 history/window 合法性导致的样本数差异要报告，不能声称完全相同样本集合。

## 8. 部署与使用体验

### 8.1 LAMP 的显式历史接口

把 `LoadedPolicy.predict` 扩展为 `predict(observation, *, action_context=None, rtc_prefix=None, delay_steps=0)`。LAMP 的外部 context 接受有限的、物理坐标 NumPy `hand_history[K,12]`，runtime 转为 `[1,K,12]` tensor 后传入 agent；其他既有调用不变，未知 context key 或错误 shape 明确报错。不要改变 `LoadedPolicy.predict` 返回**全部未来 P=H-Nobs+1 步**的现有约定，core 的 C 步输出也保持原约定。

只有实际执行端知道命令是否执行：调用者在 reset 后用初始实测手姿填满 K 帧，之后每个成功执行步追加实际下发的手目标，下一 query 传入当前历史。历史必须按保存的控制 dt 对齐本次 observation，只覆盖 query 之前的执行步。runtime 不根据 `predict` 的返回值自动推进历史。缺少 LAMP context 时 fail fast，禁止回退为“重复当前实测手姿”或“上次预测当成执行”。

完成上述 API、warmup 与测试后，同时替换当前 `inspect_policy` 与 `LoadedPolicy.__init__` 对 LAMP 的无条件拒绝，不能只修改其中一个。warmup 只显式构造合成历史，不更新真实 episode 状态。此项完成表示本仓提供可调用的历史接口；外部 `dexmani_real` 执行端若不在当前工作区或未适配，必须记录为外部集成 `NOT VERIFIED`，不能宣称真机部署已完成。

### 8.2 执行模式

LAMP 与 Temporal 本轮只支持 sync。在 `configure_execution` 的入口显式拒绝二者的 async/RTC，保留 policy 侧对 RTC prefix/delay 的拒绝；不能在完成加载后默默忽略模式。不要全局禁用其他已有支持的方法。无需为两类策略新建通用调度器或 RTC 算法。

### 8.3 唯一准备入口

实施后 README 只保留以下算法准备链路，并给出相同数据覆盖、独立 artifact 输出目录的完整示例：

| 方法 | 准备与训练 |
| --- | --- |
| DQ-RISE | 现有 `train_vq_hand --policy-config ...` → `export_hand_codebook` → `train.sh dqrise` |
| LAMP | `python -m dexmani_policy.training.train_lamp_prior --config lamp` → `bash scripts/training/train.sh lamp` |
| Temporal | `python -m dexmani_policy.training.train_temporal_codec --config temporal_codec_flow` → `bash scripts/training/train.sh temporal_codec_flow` |

这些是最终用户入口，不是要求本次 CLI 自动运行完整生产训练。pretraining 与 policy 必须使用同一份有效配置覆盖；`--output` 等产物路径覆盖也要同步到 policy。LAMP 不再有 export latents 步骤。沿用现有“不覆盖已有文件，除非明确 overwrite”的保护。

## 9. 可比性与不改动项

默认配置保持 DQ 和 Temporal 已有 action mode、H/Nobs/C、模型主干和优化器；LAMP 的单步窗口、MLP 与损失变化是本次明确修改。不要顺手把 DQ 改成 Flow、把 Temporal 改成 LAMP head，或统一 latent 维度。

正式结果必须注明这是“论文机制适配后的方法 baseline”，不是已经隔离所有变量的纯动作空间消融。若需同条件对比，在实验配置覆盖中固定 joint `action`、19 维输出、相同观测/数据划分/dt、`Nobs=1,C=1`；DQ/Temporal 可保留 H=16 并明确 H 的差异，LAMP H=1。此时为覆盖 episode 末端 query，可显式设 `pad_before=0,pad_after=H-1`，并重新生成匹配的表示产物；不要改全仓 pad 默认公式。尾部 padding 比例及误差统计口径必须记录。

以上覆盖仍不能消除模型/优化目标差异，不能把成功率差异全部归因于动作空间。**不要为此另建三套 canonical policy 或额外实验模式。** 对齐所有网络和训练目标是另一个研究任务。

本轮不新增统一接口框架、artifact 数据库、latent 搜索器、后处理平滑器、关节限位策略或自动容量搜索。不把重建损失下降、低维 latent 或 L2 正则当作安全性/成功率保证。

## 10. 文件级实施顺序

下面路径相对仓库根目录；修改前用 `rg` 检查实际引用和当前内容，清单不是凭空增加文件的理由。

| 顺序 | 主要路径 | 需要闭合的改动 |
| --- | --- | --- |
| A | `dexmani_policy/agents/core/lamp_flow_policy.py` → `lamp_policy.py`；`configs/lamp_flow.yaml` → `lamp.yaml` | 单步直接 head、显式损失/梯度/归一化、旧参数与配置删除 |
| B | `agents/hand_prior/motion_prior.py`；`datasets/lamp_dataset.py`；`training/train_lamp_prior.py`；删除 `training/export_lamp_latents.py` | 去 inverse/cache，v2 prior、历史/数据契约、选点与唯一准备命令 |
| C | `agents/core/temporal_latent_flow_policy.py`；`training/train_temporal_codec.py`；`configs/temporal_codec_flow.yaml` | 固定 standardized decoded loss、完整 chunk 与 q0、选点记录；仅必要时动 codec/dataset |
| D | `agents/core/dqrise_policy.py`；`agents/hand_prior/codebook.py`；`training/train_vq_hand.py`；`training/export_hand_codebook.py` | 增强 source provenance 绑定与保存，保持 RVQ/NN/PCA/lookup 核心 |
| E | `training/build_utils.py`、`checkpoint.py`、`resume.py`、`workspace.py` 与 `utils/config.py` 中实际相关路径 | 在最小边界接通 artifact 校验/持久化、fresh/resume/raw/EMA/compile，更新 `SPARSE3D_POLICY_TARGETS` 等实际类型引用，避免重复读入 |
| F | `env_runner/base_runner.py`、`sim_runner.py`；`deployment/runtime.py` | 更新 LAMP 引用，保留 executed history，显式 runtime context 与模式拒绝 |
| G | `smoke_test.py`、必要定向测试、README、相关 docs | 唯一入口、删除失效命令、可复核验收记录；不把实现配置写进 AGENTS.md |

可拆成数个可解释的本地提交，但每个阶段最终必须闭合到真实调用方。提交前扫描 `lamp_flow`、`LAMPFlowPolicy`、`export_lamp_latents`、`hand_latent_target`、`latent_path` 等旧引用；历史讨论或本任务书中的删除说明可保留，活跃代码/配置/用户命令不得残留。

## 11. 验收：针对会改变结论的风险

优先少量定向测试和现有 production smoke，不写只复述实现的测试，不启动大规模训练。允许测试构造小型、非线性的可控 decoder 与临时数据；验证必须调用生产代码的实际接口，而不是复制其公式后自证正确。

| 验收项 | 必须验证的行为 |
| --- | --- |
| LAMP 因果性 | 修改当前/未来 target 不改变当前 history；episode reset 无跨段污染；实际执行 1 步只追加 1 条，取消预测不更新；bootstrap 来源是初始 q0 |
| LAMP loss/梯度 | `delta_z=0` 时 hand 等于 `D(mu)` 且正则为零；非零 offset 的正则有正确值/梯度；非线性 D 下 decoded loss 能回传 head，E/D 无 grad 且 train/eval 后仍冻结 |
| LAMP 数值坐标 | 使用非 identity、各关节不同的 affine，验证 arm/hand 物理输出只逆变换一次；没有 latent labels 也能前向、反向与推理 |
| Temporal q0/时序 | obs augmentation 不改 reference；窗口/控制切片与源行正确；保存/恢复 `q0` 后逐点物理解码正确；整 H 解码先于执行切片，验证 decoder 收到全 H |
| Temporal 梯度/噪声 | target mean 在固定输入下确定；Flow noise→data 的端点与 clean estimate 正确；decoded loss 回传 flow head，codec 无 grad；目标差分一致时差分损失为零 |
| 参考变换测试边界 | 可以验证 `normalize_actions` 的代数逆变换；**不能**要求有损 E→D 精确还原，也不能要求显式 q0 条件 Transformer 对平移严格等变 |
| DQ 完整链路 | 所有 index 的 scalar→index 回环、端点/越界/半整数 round；标签来自实际原型 NN；非 identity affine 下 lookup 恢复物理命令；导出权重与训练权重一致 |
| 来源拒绝 | 维度和 affine 相同，但 split/源数据/窗口被换掉时 fresh binding 或 strict resume 报错；相同内容迁移目录不报错；旧 schema 有明确诊断 |
| checkpoint | raw 与 EMA 各自 round-trip；冻结模块/normalizer/来源信息保留；暂时隐藏外部表示产物后仍可 restore + infer；测试不删除用户原文件 |
| 部署 | LAMP 缺 history、错 shape、非有限 context 均报错；warmup 不推进真实历史；LAMP/Temporal async/RTC 在配置入口拒绝；返回 P 步协议不变 |
| 选点 | warmup/ramp 未完成的步骤不能成为 policy 用产物；有/无 validation 的选择与元信息一致；不足 steps 在训练前报错 |
| 优化/编译 | trainable 参数恰好入 optimizer 一次，冻结参数不进入；LAMP 新 head 被正确 compile；在具备环境时验证 eager/compiled 保存恢复与有限 loss |

最小命令（执行前核对实际入口；不得为通过检查而改研究语义）：

```bash
python dexmani_policy/smoke_test.py --config-only dqrise lamp temporal_codec_flow
# 涉及共享训练/部署路径时，加一个已有连续策略与 SAT 的 config-only 回归
python dexmani_policy/smoke_test.py --config-only dp sat
```

完整 smoke 使用临时微型数据/产物或用户已经提供的有效产物，每种方法做少量 production Trainer updates、保存/恢复和一次推理。当前完整 smoke 有真实依赖要求；若默认配置绑定的数据/权重不存在，使用合法的测试配置或定向集成检查，不伪造正式 artifact 的 metadata。必要时只对 smoke 添加最小测试夹具入口，不能新增训练框架。

CUDA、依赖、数据、权重或外部执行器缺失时，完成仍可做的实现与检查，明确 `NOT VERIFIED`、具体缺口和复跑命令；不要把 source/AST 检查当作训练通过。禁止为验收而自动启动长训练、批量仿真或真机。

## 12. 完成定义与 CLI 最终交付

- [ ] DQ-RISE、LAMP、Temporal 各一个有效策略实现，配置、准备、训练、恢复、推理入口互相一致。
- [ ] LAMP 单步 direct BC + frozen history prior + decoded supervision + raw latent offset L2，逆解/缓存/旧 Flow 链路彻底退出。
- [ ] Temporal 保持 q0-conditioned full-chunk codec 和正确 FM，不发生误锚定、截断解码或梯度断路。
- [ ] 表示产物的来源、坐标、划分和选点记录可追溯；fresh、strict resume、独立推理的边界分明。
- [ ] SIM 与 runtime 的 history/reference/execution 语义已落到调用方，外部未验证部分明确披露。
- [ ] README 和机制文档说明论文要求、本地适配、生产命令及不兼容点；不得声称完整复现论文系统或取得论文成绩。
- [ ] 针对风险的检查有命令和结果；仅实际运行通过标 `PASS`，缺条件标 `NOT VERIFIED`；不留下绕过校验的临时代码。

最终用中文报告：实际起始/最终提交、改动与删除清单、三种动作空间的有效配置和数据流、与论文的对应及适配、测试命令/结果、剩余环境或外部集成缺口、用户接下来如何生成产物并开始训练。遇到非阻塞实现细节按本任务选择简单方案继续；只有真实冲突、无法访问的必要输入或破坏性操作才请求用户决策，不停在再次提出方案。
