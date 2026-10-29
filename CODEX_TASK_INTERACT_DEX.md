# Codex 任务书：InterAct-Dex 的训练期逐指触觉响应监督

本任务以手—物交互为主线，对现有 `interact-dex` 做最小、完整的代码修改：让动作 DiTX 的共享隐状态同时接受执行窗口内的逐指未来触觉监督。辅助预测只参与训练损失，部署继续使用现有动作生成路径。

代码核对基准：[`main@f2267b276fd15b06a7c0587660acc7a5306d0eab`](https://github.com/haoyangzhanglab/dexmani_policy/tree/f2267b276fd15b06a7c0587660acc7a5306d0eab)。实施时先读取当前工作区及 [AGENTS.md](AGENTS.md)，保留已有改动；如当前接口已变化，以真实调用关系完成等价的最小实现，不回退仓库版本。

这是一份待执行任务书。方案与默认损失权重尚未经闭环训练验证，不将其写成已经证实的最优结果。

## 1. 目标与机制

灵巧操作需要协调各手指的接触建立、支撑维持与释放。现有策略已经编码手部身份、局部几何、当前触觉和历史，但动作表示仅由模仿损失训练。本次让同一动作表示还需解释接下来逐指触觉的演化，促使动作计算保留与手—物交互结果有关的信息。

只实现一条机制：

历史观测经现有交互编码器和动作 DiTX 得到共享归一化隐状态。该状态同时送入原动作输出 MLP 和小型逐指读出头，分别计算原 FM 损失与未来触觉损失。

不另建预测骨干，不将预测触觉反馈为动作条件，不加入 EMA teacher、潜状态展开、视觉重建、接触分类或在线规划。保留现有整策略 EMA。

监督目标称为“逐指触觉响应”。它不是接触真值、滑移真值或完整因果动力学。DiTX 保留双向动作注意力，因此辅助预测可能利用整段带噪动作；本次不额外改变注意力结构，也不据此宣称反事实预测能力。

## 2. 严格范围

- 保留 `InterActDexPolicy`、`policy_name: interact-dex` 和当前训练/部署入口。
- 保留观测编码器、六个身体槽、几何与触觉融合、历史编码和动作空间。
- 保留 FM 噪声/时间采样、动作损失及其归约、动作窗口、推理积分步数、KV 缓存和公开输出格式。
- 不增加测试组件、测试文件、测试夹具、测试数据、benchmark、消融模式或实验配置矩阵；不改造现有测试工具。
- 除本任务书外，不新增或修改 Markdown 文档，不补研究报告、README、架构说明或实施记录。
- 不增加依赖、注册表、通用辅助任务框架、forward hook、隐状态缓存或额外训练入口。
- 不启动正式长训或机器人执行；仅做本次代码正确性所需的轻量检查。

预计修改 **4 个现有文件、新增 1 个必要的数据集 Python 文件**：

| 文件 | 修改内容 |
| --- | --- |
| [`dexmani_policy/agents/action_decoders/backbones/ditx.py`](dexmani_policy/agents/action_decoders/backbones/ditx.py) | 显式、可选地返回动作输出 MLP 前的归一化隐状态 |
| [`dexmani_policy/agents/action_decoders/flow.py`](dexmani_policy/agents/action_decoders/flow.py) | 让 FM 损失可返回同一次前向得到的隐状态 |
| [`dexmani_policy/agents/policies/interact_dex.py`](dexmani_policy/agents/policies/interact_dex.py) | 小型读出头、辅助损失、参数组和有效配置 |
| `dexmani_policy/datasets/aux/interact_dex_dataset.py`（新增） | 保留原采样窗口，只额外读取真实后继触觉及掩码 |
| [`dexmani_policy/configs/interact_dex.yaml`](dexmani_policy/configs/interact_dex.yaml) | 切换数据集目标、绑定执行长度、启用辅助权重 |

优先保持 `BasePolicy`、`RobotDataset`、`SequenceSampler`、`AdapTacDataset`、trainer、normalizer、checkpoint 工具和其他策略不变。不要为了减少文件数，把策略专属的未来标签逻辑塞入公共数据集。

## 3. 时序与数据契约

定义：

- `s = policy.action_start_index = n_obs_steps - 1`；
- `K = n_action_steps`；
- `q = sampler.sample_source_rows(idx)[dataset.action_start_index]`，表示当前观测的真实源行；
- `end` 为包含 `q` 的 episode 的排他结束行，按现有 `episode_ends` 与 `searchsorted(..., side="right")` 定位。

对 `j = 0, ..., K-1`：

```text
hidden[:, s+j]  →  原动作位置 a[q+j]
                →  触觉标签 F[q+j+1]
valid[j]       =  q+j+1 < end
```

默认配置为 `horizon=16, n_obs_steps=4, n_action_steps=4`，对应：

| 隐状态位置 | 动作行 | 触觉监督行 |
| --- | --- | --- |
| 3 | q | q+1 |
| 4 | q+1 | q+2 |
| 5 | q+2 | q+3 |
| 6 | q+3 | q+4 |

只读出 `features[:, s:s+K]`，不能使用 `[:, :K]`，不能把 16 个位置都当成从当前开始执行的动作。数值 3 和 4 仅用于说明，不写死在实现中。

新增 batch 顶层字段：

| 字段 | 单样本 | batch | 语义 |
| --- | --- | --- | --- |
| `future_force` | `[K,5,3]` float32 | `[B,K,5,3]` | 下一观测行开始的原始 aggregate 触觉 |
| `future_force_valid` | `[K]` bool | `[B,K]` | 同一 episode 中确实存在该后继行 |

五指顺序沿用数据的 thumb/index/middle/ring/pinky。保留三个通道，不取模长、不跨指池化、不二值化。

### 数据集实现

新增小型 `InterActDexDataset(RobotDataset)`，构造参数接收 `n_action_steps`，复用原有加载、split、增强、normalizer 和 validation dataset 逻辑。

1. 先调用 `super().__getitem__(idx)` 取得现有观测与动作，再按真实源行读取未来触觉。
2. 使用 `replay_buffer.read("finger_tactile_force", ...)` 进行一次有界连续读取，不读取未来点云或其他重型模态。
3. 未来目标不放入 `batch["obs"]`，不传入观测编码器，不参与观测增强。
4. 不覆盖 `build_sampler` 来删样本，不改变 `sampler.indices`、padding、episode split、训练窗口数量或动作标签。
5. 可参考 [AdapTacDataset](dexmani_policy/datasets/aux/adaptac_dataset.py) 的源行定位方式，**不要直接继承它，也不要复制其删除无后继样本的过滤规则**。
6. 无效后缀使用该 episode 最后一行的有限值占位，掩码为 false；即使完全没有后继行也保留样本，不能对空数组取最后元素。
7. 标签有效性只由真实后继行是否存在决定。合法零值、力相消、reset 零值、SDK hold 都不能据数值被标成无效。
8. 数据集与策略的执行长度通过同一个 `n_action_steps` 绑定；在必要边界确认长度合法、执行切片不超过 horizon，不增加层层重复校验。

## 4. 单次前向与共享隐状态

### DiTX

为 `DiTX.forward` 增加显式的 keyword-only `return_features=False`：

- 默认保持返回原动作向量场张量，所有既有调用方无需修改。
- 开启时返回 `(pred_v, features)`，其中 `features` 为 `[B,horizon,hidden_dim]`。
- 特征取自现有 `final_layer.norm_final` 之后、`ffn_final` 之前；直接复用同一份归一化结果，不新增 LayerNorm/RMSNorm，不重复执行最终归一化。
- 可对 `DiTXOutput.forward` 做相同的可选返回扩展；保留原模块名、参数名和默认返回行为，使 `ConsistencyDiTX` 等调用方不受影响。
- 不 detach、不 clone 留存计算图、不使用 hook 或成员变量缓存特征。返回行为由显式参数决定，不由 `self.training` 隐式决定。

### RectifiedFlow

为 `RectifiedFlow.compute_loss` 增加 keyword-only `return_features=False`：

- 默认仍返回 `(loss, metrics)`。
- 开启时仅执行一次 `sample_training_pair` 和一次 `self.model(..., return_features=True)`，返回 `(loss, metrics, features)`。
- 保持原 FM MSE、时间采样、噪声、原有指标及其含义。
- 默认路径不要把新关键字传给不支持它的其他 backbone。
- `log_metrics=False` 时仍应按 `return_features` 正确返回特征，metrics 为 `{}`；不要让提前返回丢失特征。
- 不把仍带梯度的特征塞入 metrics，不进行第二次模型前向，不在 policy 中复制一套 FM 实现。
- `predict_action`、`ConsistencyFlow` 和推理缓存代码不改。

## 5. 策略读出、损失与配置

### 读出头

在 `InterActDexPolicy` 中注册 `interaction_head`：

```text
Linear(hidden_dim, 128) → SiLU → Linear(128, 15)
```

同一小头在 K 个时间位置共享，输出 reshape 为 `[B,K,5,3]`。默认 hidden_dim=768 时约 10 万参数。中间宽度固定 128，不新增可调 head-depth/head-width 配置。

只从请求到的共享特征执行切片和读出。未来真值只能用于损失；不进入 DiTX 输入、condition、额外 token 或推理采样。

### 损失

沿用现有动作与观测预处理；对原始 `future_force` 使用训练集已拟合的：

```python
self.normalizer["finger_tactile_force"].normalize(future_force)
```

不另拟合未来标签统计，不混合 sim/real 统计，不对标签做额外标准化或单位/坐标变换。sim 是世界系合力，real 是 SDK 通道值；同一 episode 的后继观测也不等于严格同步的瞬时动作后真值。

定义 `target`、`prediction` 为归一化标签与预测，`m` 为有效掩码：

```text
element_loss = smooth_l1_loss(prediction.float(), target.float(),
                              reduction="none", beta=1.0)
L_interaction = sum(element_loss * m[..., None, None])
                / (15 * clamp_min(sum(m.float()), 1))
L_total = L_FM + interaction_loss_weight * L_interaction
```

要求：

- reduction 和分母使用 FP32，避免 bfloat16 归约失真；不改变原 FM 计算约定。
- 全无效 batch 的辅助损失为有限、可微的 0，动作损失继续正常计算。
- 在损失输入边界确认目标为 `[B,K,5,3]`、mask 为 `[B,K]`，防止广播掩盖配置错误。
- 辅助梯度经 `interaction_head → final norm → DiTX → observation encoder` 回传；不 stop-gradient，不用 no_grad 包裹读出或共享特征。原动作输出 MLP 仍由 FM 损失训练。
- `log_metrics=True` 时沿用现有日志通道，保留 `loss_action` 表示纯 FM，将 `loss` 更新为总损失，仅增加 `loss_interaction`、`loss_interaction_weighted`、`interaction_valid_fraction`；指标均 detach。
- `log_metrics=False` 返回 `(total_loss, {})`。不创建额外监控/诊断组件。

### 最少配置与 checkpoint

新增唯一独立研究超参数 `interaction_loss_weight`：

- Python 构造默认值为 `0.0`，保证旧配置缺少该字段时保持旧结构。
- 在现有 `interact_dex.yaml` 中显式设为 `0.1`，使新的标准训练配方启用本方案。这是初始工程取值，不声称已经调优。
- 要求有限、非负。为 0 时不创建有参数的辅助头，不读取 loss 中的未来标签，直接走原 `super().compute_loss`，不请求特征、不计算辅助损失。
- 权重大于 0 才注册小头；覆写本策略的 `optimizer_groups`，在原参数组后复用 `parameter_groups` 加入该头，使用动作侧 lr/weight_decay，保证每个可训练参数恰好一次。不要修改公共优化器。
- 覆写 `effective_config`，合并 `super().effective_config` 并保存实际权重；保留观测编码器已有配置。
- 复用整策略现有 EMA 与严格 state_dict 保存/恢复。新配置及小头可正常保存、续训与离线恢复；旧配置缺字段时不会出现小头缺失键。不要引入 `strict=False`、键名迁移或“旧 checkpoint 自动升级新目标”的兼容框架。
- YAML 只新增必要内容：dataset target 改为 `dexmani_policy.datasets.aux.interact_dex_dataset.InterActDexDataset`，增加 `dataset.n_action_steps: ${n_action_steps}`，以及 `agent.interaction_loss_weight: 0.1`。不增加独立的预测 horizon 或 enabled 开关，不改变其他默认配方。

### 部署约束

`predict_action`、`predict_action_from_condition` 和 `format_action_output` 继续复用原实现。推理不请求 features、不调用 interaction_head、不要求未来标签、不新增 token、不增加 DiTX 前向次数或积分步数。

辅助头可以随新 checkpoint 保留，部署时不执行；不为删除约 10 万参数另建导出管线。公开 `future_actions` 的长度与维度保持现有契约，**不要为 K 步监督而将返回值截断到 K**；K 只限定本次监督窗口。

## 6. 必要验证与交付

只使用现有入口或不落盘的一次性检查，不新增测试代码/组件，不开展对比消融。

有依赖时先运行现有配置检查：

```bash
python -m dexmani_policy.test_tools.smoke_test --config-only --override tactile=sim_force interact_dex
python -m dexmani_policy.test_tools.smoke_test --config-only --override tactile=real_force interact_dex
```

有真实可用数据与设备时，用对应来源运行一次现有短训/续训检查；以下路径为占位符，应替换为环境中的实际路径：

```bash
python -m dexmani_policy.test_tools.smoke_test --max-updates 2 --resume-updates 1 --override 'zarr_path=<actual_zarr_path>' --override tactile=sim_force --override dataloader.batch_size=2 --override dataloader.num_workers=0 interact_dex
```

结合必要的一次性检查和代码审查，确认：

- 时间切片对应 `s:s+K → q+1:q+K+1`，末尾 mask、全无效样本及合法零值处理正确，窗口集合不变。
- 动作和辅助损失共用一次 DiTX 前向；共享特征未 detach，小头和共享动作主干参与反向传播，优化器无遗漏/重复。注意 DiTX 的零初始化门控可能使个别层在首步没有非零梯度，不能据此误判并重写初始化。
- loss 与 metrics 契约正确，缺少未来标签时不被静默降级；权重 0 的旧路径不要求这些标签。
- 新 checkpoint 严格保存/恢复和 EMA 路径可用；旧配置采用默认 0 时保留原 state_dict 结构。
- 推理没有调用辅助头，公开输出契约和默认 DiTX/RectifiedFlow 调用方未改变。
- 最终 diff 仅包含必要的 Python/YAML 实现，无新增测试、文档、实验配置或运行产物；执行 `git diff --check`。

如果缺少依赖、数据、GPU 或旧 checkpoint，只报告真实完成的检查和具体未验证项，不伪造通过、不为补足验证新建基础设施、不以此扩展任务。

提交前按 [AGENTS.md](AGENTS.md) 清理冗余。最终用中文简要交付：修改文件与目的、关键数据/梯度路径、实际运行的验证和结果、未验证项。训练成功率和闭环收益留待真实训练验证，本任务不交付性能提升结论。
