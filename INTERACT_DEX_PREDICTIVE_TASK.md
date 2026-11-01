# InterAct-Dex 动作耦合交互状态预测实施任务书

## 1. 任务目标与执行边界

在 `haoyangzhanglab/dexmani_policy` 中就地升级现有 `InterActDexPolicy`：让 wrist＋五指共 6 个交互状态 token 与动作共享 DiTX 计算，并以单个未来力残差目标监督这些状态。未来力监督应经过共享动作主干，交互状态在推理时继续参与动作生成。

本任务已经收敛到一个实现方案。实施者应完成可训练、可恢复、可推理的最小实现，不重新展开架构选型或多方案实验。

遵守根目录 [AGENTS.md](AGENTS.md)，保留当前工作树已有改动。默认中文说明，沿用现有 Python 排版、命名、参数分组和配置组装方式。

只实施这一项主要优化：

- 保留 `horizon=16`、`n_obs_steps=4`、`action_start_index=3`，对外仍输出 13 个未来动作。
- 保留 768 维、8 层、8 heads 的 DiTX，以及当前观测编码器和历史编码器。
- 保留当前 Flow 时间采样、10 步 Euler、动作分组损失、优化器、EMA 与训练配方。
- 不实施 13 步纯未来训练窗口、运动／力增量输入、尾部全覆盖、额外 action-valid 机制、512 维缩容或部署功能修复。
- 不新增验证器、运行时防御分支、测试组件、测试文件、通用 mask 框架、兼容迁移层或新的依赖。
- 这份任务书是用户明确要求的唯一文档交付；后续代码实施不新增或顺带修改 Markdown 文档。

“确定方案”指在以上约束下确认计算合同与工程落点。机器人控制收益、反事实预测准确度和实测时延尚未被训练或闭环实验验证，不能写成已经证明的全局最优。

## 2. 版本与源码 fact-check

核对日期：2026-11-01（Asia/Shanghai）。

源码基线：[main@`b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba`](https://github.com/haoyangzhanglab/dexmani_policy/commit/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba)，提交说明为 `fix: align update timing and streamline RGB and timestep processing`。相比先前讨论的 `8011f035`，已有 DexLatent 接入、FP32 推理和时间编码等后续改动；不得为了照抄旧方案回退这些改动。

实施时以当前工作树为准。若相关代码已更新，重新核对对应调用点后适配；若已经实现本任务的一部分，只补齐缺失部分。

| 核对项 | 当前源码事实 | 本任务结论 |
| --- | --- | --- |
| 当前辅助 head | [读取最后观测帧的五指 condition](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/policies/interact_dex.py#L221)，未读取 DiTX 特征 | 把同名 head 改接共享主干的五指交互特征 |
| 特征返回 | [RectifiedFlow.compute_loss](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/action_decoders/flow.py#L66) 已支持 `return_features=True`，一次模型调用返回 velocity 与 features | 直接复用，不增加 `return_interaction` 或修改 Flow 接口 |
| 观测布局 | [ObservationEncoder](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/policies/interact_dex.py#L22) 拼接六个身体与十二个场景 token，再按帧展平；[场景默认数量为 12](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/obs_encoder/interaction/scene_aggregator.py#L98) | 保持 `[B,72,192]` |
| 动作起点 | [BasePolicy](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/policies/base.py#L81) 返回 `n_obs_steps - 1` | 首四个执行动作是索引 `3:7`；不能误用 `0:4` |
| 注意力 | [DiTXBlock](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/action_decoders/backbones/ditx.py#L72) 使用 `nn.MultiheadAttention`，目前不接收 mask | 只增加默认 `None` 的 `attn_mask` 透传；布尔 `True` 表示禁止读取 |
| 静态缓存 | [DiTX.prepare_context_kv](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/action_decoders/backbones/ditx.py#L237) 在 eval 下准备各层静态观测 K/V | 原样继承；动态交互状态不放入缓存 |
| 推理精度 | [RectifiedFlow.predict_action](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/action_decoders/flow.py#L124) 的默认积分 dtype 已明确为 FP32 | 保留最新行为，不改回跟随 condition.dtype |
| 未来标签 | [Dataset](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/datasets/variants/tactile.py#L46) 已读取 offsets 并输出对应有效性；[有效性在裁剪标签行号之前计算](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/datasets/variants/tactile.py#L8) | 配置改成单元素 `[4]` 即可复用 |
| 配置传递 | [assemble_policy_config](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/training/common/config.py#L79) 同时把 offsets 传给 agent 与 dataset | 保留这条单一来源，不增加可独立覆盖的第二个 K 配置 |
| 力归一化 | [FieldNormalizer](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/common/normalizer.py#L16) 使用乘法 scale；[RMS/MaxAbs 为共享标量尺度、零 offset](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/common/normalizer.py#L86) | 残差只乘现有 scale，不另拟合统计 |
| checkpoint | [restore_policy](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/common/checkpoint.py#L41) 按保存的 effective_config 构造，strict load，恢复为 FP32 | 新结构重新训练；旧结构用旧代码恢复，不静默放宽 strict load |
| 已有检查覆盖 | [agent_checks 当前入口](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/test_tools/agent_checks.py#L565) 没有 InterAct-Dex 专项 | 不能把通用检查通过表述为本任务已获完整验证 |

本次优化调整相对先前方案主要是：明确最新 FP32 推理合同；补齐当前 sampler 的真实覆盖范围；锁定单元素 offset 的解包语义；明确辅助损失关闭时仍保留交互 token；限定计算估算的算子范围；把初始化退化与缓存一致性的有限验收写清楚。

## 3. 固定配方和张量合同

| 项目 | 最终选择 |
| --- | --- |
| policy 类／名称 | 沿用 `InterActDexPolicy`／`interact-dex` |
| 专用 backbone | 新增 `InterActDiTX(DiTX)` |
| 观测帧数 | 4 |
| 模型动作长度 | 16 |
| 动作维数 | 19，arm7＋hand12 |
| 对外动作起点 | 3，保持 BasePolicy 原行为 |
| 默认实际执行长度 C | `n_action_steps=4` |
| 交互预测边界 K | 唯一 `future_force_offsets=[4]` 中的 4 |
| 交互 token | 6，顺序为 wrist、五指 |
| 主干 | hidden_dim=768、n_layers=8、n_head=8 |
| Flow | Uniform 训练时间；time_shift_alpha=1.0；10 步 Euler |
| 力目标 | 单个 `Delta F_4`，SmoothL1(beta=1.0)，权重 0.1 |
| 力读出 | 五指共享 `768 -> 128 -> 3`，SiLU，末层权重和 bias 为零 |
| 训练／推理精度 | 沿用当前 BF16 autocast 训练、FP32 默认推理积分及恢复方式 |

batch 保持现有数据结构，只把未来力的 offsets 轴长度改为 1：

```python
{
    "obs": {
        "joint_state":          [B, 4, 19],
        "point_cloud":          [B, 4, 1024, 6],
        "eef_pose":             [B, 4, 9],
        "fingertip_xyz":        [B, 4, 5, 3],
        "finger_tactile_force": [B, 4, 5, 3],
    },
    "action":             [B, 16, 19],
    "future_force":       [B, 1, 5, 3],
    "future_force_valid": [B, 1],       # bool
}
```

当前窗口对应的理想时间位置为：

| 分组 | 动作／动态序列索引 | 时间含义 |
| --- | --- | --- |
| 历史动作预测 H | `0:3` | `t-3, t-2, t-1` |
| 首四个执行动作 P | `3:7` | `t, t+1, t+2, t+3` |
| 后续动作 T | `7:16` | `t+4, ..., t+12` |
| 交互状态 Z4 | 动态序列的 `16:22` | 执行前四个动作后的交互预见表示 |

episode 起点不足的观测／历史行仍按现有 sampler 重复首帧。不要把这些历史动作预测位置当成推理时额外提供的真实历史动作。

仿真数据已有 [obs[t]_before_action[t] 合同](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/datasets/demonstration.py#L47)，所以 `F[t+4]` 对应四次动作之后的观测。真实数据沿用 canonical 数据的行时间关系与 SDK 通道，不新增对传感器延迟、物理坐标或力单位的标定假设。

### 3.1 当前 padding 的真实边界

保留现有配置派生出的 `pad_before=3`、`pad_after=3`。对长度 L 的 episode，当前 [window_indices](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/datasets/sampler.py#L6) 的 query 为 `0 ... L-10`，窗口数为 `max(0, L-9)`。

直接执行当前源码的 `window_indices` 所得结果：

| episode 长度 | query 数 | 最后一个 query | 有效 +4 标签数 |
| --- | ---: | ---: | ---: |
| 1、4、5、9 | 0 | 无 | 0 |
| 10 | 1 | 0 | 1 |
| 13 | 4 | 3 | 4 |
| 16 | 7 | 6 | 7 |
| 20 | 11 | 10 | 11 |
| 100 | 91 | 90 | 91 |

因此，固定配方下有样本的窗口中，P 的四个动作及 +4 力标签都是真实行。现有后续动作 T 仍可能包含重复尾帧监督；本任务不修改该旧语义，也不宣称已覆盖全部真实 query。结构 mask 将 H/T 排除在 P/Z4 的输入之外，但它不是 action padding mask。

保留已有 `future_force_valid` 的损失处理，不能因为默认配方中 +4 有效就删除 mask，也不能利用标签有效性控制 backbone。

## 4. 专用主干：复用 DiTX，只增加动态状态

新增文件：

```text
dexmani_policy/agents/action_decoders/backbones/interact_dex.py
```

实现 `InterActDiTX(DiTX)`，继承父类已有的输入投影、动作位置编码、观测投影、时间编码、8 层 blocks、输出层和 `prepare_context_kv()`。本任务只用普通 19D action I/O，不为 SATActionIO、latent 动作或其他 policy 扩展泛化接口。

### 4.1 构造与初始化

- 通过父类构造完成原有参数初始化。
- 增加一个 `future_body_embedding`，形状 `[1,6,hidden_dim]`，采用已有 `WEIGHT_INIT_STD` 初始化。
- 增加一个下面定义的布尔 attention mask，使用 `register_buffer(..., persistent=False)`。
- 不新增独立的 body projector；复用 `context_embedder`。
- 不覆盖父类初始化方法去访问尚未创建的子类属性，也不在创建新 embedding 后重新初始化整套主干。
- 输入合法性处理沿用当前 DiTX 已有边界；不增加新的 shape／finite 校验层。

### 4.2 每次 forward 的计算合同

每次调用从当前观测编码的最后一帧取出六个身体 token，重新初始化 Z4。这里的身体 token 已包含现有几何、触觉与历史编码结果：

```python
body = context.unflatten(1, (self.n_obs_steps, -1))[:, -1, :6]
interaction = self.context_embedder(body)
interaction = interaction + self.future_body_embedding.to(interaction.dtype)
```

不使用未来力、未来观测、额外未加噪的真实动作或上一轮 Z4。六个身体 token 的小投影每次重算即可，不为它扩展缓存协议。

forward 的核心顺序如下。沿用当前文件的类型标注和已有输入边界处理，下面只展示计算主体：

```python
def forward(
    self, x, timestep, context, *,
    context_kv=None, return_features=False,
):
    context_c = self.embed_context(context) if context_kv is None else None
    timestep = batch_timesteps(timestep, x, dtype=torch.float32)
    time_c = self.timestep_embedder(timestep)

    body = context.unflatten(1, (self.n_obs_steps, -1))[:, -1, :6]
    interaction = self.context_embedder(body)
    interaction = interaction + self.future_body_embedding.to(interaction.dtype)

    actions = self.input_embedder(x)
    actions = actions + self.input_pos_embed.to(actions.dtype)
    hidden = torch.cat((actions, interaction), dim=1)

    for index, block in enumerate(self.ditx_blocks):
        hidden = block(
            hidden, time_c, context_c,
            context_kv=None if context_kv is None else context_kv[index],
            attn_mask=self.interaction_mask,
        )

    features = self.final_layer.norm_final(hidden)
    velocity = self.final_layer.ffn_final(features[:, :self.horizon])
    return (velocity, features) if return_features else velocity
```

必须保持以下输出语义：

- 默认只返回 `[B,16,19]` velocity。
- `return_features=True` 返回 `(velocity, features)`，其中 features 为归一化后的 `[B,22,hidden_dim]` 全部动态 token。
- policy 读取 features 的 `horizon+1:horizon+6`，恰好是五指，排除 wrist。
- 不把 6 个交互 token 送入动作输出 MLP，也不把它们追加到 Flow 积分状态。
- 不复制 Transformer、CrossAttention、RmsNorm 或输出 MLP 的实现。
- 复用 [batch_timesteps／时间编码](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/action_decoders/backbones/modules.py#L13)，不重建已经缓存的频率，也不改变当前 FP32 时间输入习惯。

## 5. 所有层共享同一固定注意力边界

物理顺序保持 `[H, P, T, Z4]`。矩阵行是 query，列是可读 K/V：

| Query / K,V | H | P | T | Z4 |
| --- | ---: | ---: | ---: | ---: |
| H | 允许 | 允许 | 允许 | 允许 |
| P | 禁止 | 允许 | 禁止 | 允许 |
| T | 允许 | 允许 | 允许 | 允许 |
| Z4 | 禁止 | 允许 | 禁止 | 允许 |

每一组都可以 cross-attend 到全部静态观测。该结构不是逐动作 causal mask：P 内部四个候选动作可以双向读取，P 与 Z4 也可以双向交换信息。

构造方式：

```python
start = self.n_obs_steps - 1
core = torch.zeros(self.horizon + 6, dtype=torch.bool)
core[start:start + interaction_horizon] = True
core[self.horizon:] = True

self.register_buffer(
    "interaction_mask",
    core[:, None] & ~core[None, :],
    persistent=False,
)
```

只有 core 中的 query 被禁止读取 core 外的 K/V。这里 `nn.MultiheadAttention` 的布尔 `True` 表示禁止；不要套用 scaled_dot_product_attention 直接调用时的布尔语义。

在现有 `DiTXBlock.forward()` 中：

```python
def forward(self, x, time_c, context_c, *, context_kv=None, attn_mask=None):
    ...
    self_attn_output = self.self_attn(
        normed_x, normed_x, normed_x,
        attn_mask=attn_mask,
        need_weights=False,
    )[0]
```

默认值为 None，其他 DiTX/ConsistencyDiTX 调用方无需改动。不要扩展 key_padding_mask、动态 mask 生成器或 attention 后端。

该边界必须同时作用于 P 和 Z4：只屏蔽 Z4 的一跳读取，无法阻止两层中的 `T -> P -> Z4` 或 `H -> P -> Z4`。由于 cross-attention 的观测是静态的、LayerNorm/RmsNorm/MLP 按 token 计算，完整边界可以逐层保持；也使 Euler 循环中 P 的更新不通过 H/T 回流。

已做的布尔图可达性核对：22×22 mask 禁止 120 条 query-key 边，每行至少有可读 key；8 层后 H/T 到 P/Z4 不可达。只屏蔽 Z4 的错误版本在第 2 层已经出现回流。这是结构核对，不替代实现后的非退化数值检查。

## 6. 训练目标：从共享五指状态预测力残差

### 6.1 policy 与读出 head

在现有 `InterActDexPolicy` 内替换 backbone，不新增 policy 类、版本开关或新 policy_name。

继续使用 `future_force_head` 属性，输入维数从观测的 192 改为 `hidden_dim`：

```python
self.future_force_head = nn.Sequential(
    nn.Linear(hidden_dim, 128),
    nn.SiLU(),
    nn.Linear(128, 3),
) if future_force_loss_weight > 0 else None
```

head 存在时，仅将最后一个 Linear 的 weight 与 bias 置零。保留其余层正常初始化及 DiTX 原有零 gate／零 velocity 读出的初始化方式，不增加 warmup、额外 gate 或特殊梯度处理。

保持当前 `compute_loss(..., log_metrics=True, **kwargs)` 约定，复用同一次 Flow 采样和 forward：

```python
condition = self.encode_observation(batch["obs"])
actions = self.normalizer["action"].normalize(batch["action"])
kwargs["dim_groups"] = self.loss_groups()
kwargs["return_features"] = True

action_loss, metrics, features = self.action_decoder.compute_loss(
    condition, actions, log_metrics=log_metrics, **kwargs,
)

fingers = features[:, self.horizon + 1:self.horizon + 6]
prediction = self.future_force_head(fingers).unsqueeze(1)
```

prediction 为 `[B,1,5,3]`。不 detach，不重复编码观测，不再做第二次 backbone forward，不再保留观测端并行的未来力 head。

辅助梯度应能够进入共享 DiTX 和观测编码器；不要求它经过单独的 action velocity 输出 MLP。动作损失通过中间层的交互状态与共享参数共同塑造表示。

### 6.2 目标与坐标

使用原始 batch 中的当前力与未来力，不能从已归一化的 condition 中还原或重复归一化：

```text
sim:  target = ((F_world[t+4] - F_world[t]) @ R_world_from_eef[t]) * scale
real: target =  (F_sdk[t+4]   - F_sdk[t])                                * scale
```

这里 `scale = self.normalizer["finger_tactile_force"].scale`，是归一化乘数，即数学上的 `1 / S_F`，不是除数。仿真先在世界系作差，再将差旋转到 query 的 EEF 系；始终使用 `R_t`，不得读取 `R_{t+4}` 或把两帧分别旋转到各自的坐标系再相减。

实现按现有 FP32 几何习惯完成：

```python
force = batch["future_force"]
valid = batch["future_force_valid"]

with torch.autocast(device_type=force.device.type, enabled=False):
    current = batch["obs"]["finger_tactile_force"][:, -1:].float()
    delta = torch.where(valid[..., None, None], force.float() - current, 0.0)
    if self.obs_encoder.tactile_source == "sim":
        rotation = _rotation_6d_to_matrix(
            batch["obs"]["eef_pose"][:, -1, 3:].float()
        )
        delta = delta @ rotation[:, None]
    target = delta * self.normalizer["finger_tactile_force"].scale
```

复用既有旋转函数和既有力统计，不增加新 normalizer、未来力统计或残差 offset。真实力只解释为 SDK 通道变化，不添加未经标定的物理解释。

### 6.3 损失、mask 与日志

动作 loss 完全沿用当前 RectifiedFlow 的 16 步训练目标和分组权重：

```text
L_action = 0.5 * MSE_arm7 + 0.5 * MSE_hand12
```

保持 Uniform 时间采样以及 `A_tau = (1-tau) * noise + tau * action`、`target_v = action - noise`，不修改 Flow 公式或重新加权时间。

力损失沿用当前写法，只替换 prediction 和 target 的来源：

```python
element_loss = F.smooth_l1_loss(
    prediction.float(), target.float(), reduction="none", beta=1.0,
)
mask = valid.float()
future_force_loss = (element_loss * mask[..., None, None]).sum() / (
    15 * mask.sum().clamp_min(1)
)
weighted_loss = self.future_force_loss_weight * future_force_loss
loss = action_loss + weighted_loss
```

默认 `future_force_loss_weight=0.1` 由已有 YAML 提供。

- 有效零残差样本正常参与监督；不把“零力”当无效标签。
- `future_force_valid` 只进入目标处理和 loss，不进入 observation encoder、backbone、attention mask 或 Z4 初始化。
- 全无效标签的 batch 仍计算 Z4 与动作 loss，力 loss 为零。
- 保留当前 `loss`、`loss_action`、`loss_future_force`、`loss_future_force_weighted`、`future_force_valid_fraction` 语义；其中 future_force 指本版的残差预测误差。按既有 `log_metrics` 开关记录，不增加诊断指标体系。
- 现有未来标签形状边界按单元素 offsets 使用，不另建校验函数。
- 当权重为 0 时，保留现有 `head is None -> super().compute_loss(...)` 路径，不读取 future_force 字段；新 backbone 中的 6 个交互 token 仍然存在并参与动作计算。

末层零初始化使初始预测为零残差。第一次反向时，力 loss 对 head 前级的梯度为零；DiTX 也有原有零 gate 初始化。不要把这个预期初始化状态误判为梯度路径断开，也不要为此引入新的初始化方案。

## 7. 配置、参数归属、推理与恢复

### 7.1 单一预测边界

YAML 顶层唯一变更：

```yaml
future_force_offsets: [4]
```

policy 构造函数默认值同步改为 `(4,)`，保持已有 offsets 校验与 effective_config 的序列化。使用直接解包表达本版单偏移合同：

```python
future_force_offsets = validate_future_force_offsets(future_force_offsets)
interaction_horizon, = future_force_offsets
```

随后把 `interaction_horizon` 传给新 backbone。不要使用 `future_force_offsets[0]` 静默忽略多余偏移，也不要维护另一个可以独立覆盖的 K。保留有效配置中的 `future_force_offsets=[4]`，由它恢复相同 mask。

本任务固定配方为 K=4；不开发任意 offsets 组合、任意 K 扩展或新的配置校验。C 来自 `n_action_steps`，K 来自唯一力偏移，二者不能互相代替。运行时改变执行动作数量不得改变 Z4 的预测边界。

### 7.2 optimizer 和 EMA

复用现有 `optimizer_groups()`：

- 新 future_body_embedding 归属于 `action_decoder.model`，由 [parameter_groups](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/agents/policies/base.py#L12) 自动覆盖。
- 力 head 继续通过现有 policy 中的独立 head 分组加入。
- 不手动重复追加 embedding 或整个 backbone，不修改优化器、EMA 和训练器。
- 梯度和 optimizer 参数唯一覆盖在现有机制中核对，不新增运行时校验。

### 7.3 推理生命周期

沿用现有 `BasePolicy.predict_action()` 和 `RectifiedFlow.predict_action()`：

1. 一次真实查询编码观测得到 condition。
2. eval 且多次 NFE 时，通过继承的 prepare_context_kv 准备静态观测 K/V。
3. 每次 velocity evaluation 从同一个 condition 重建 Z4，与当次带噪动作共同运行所有层。
4. 只使用 `[B,16,19]` velocity 更新 FP32 动作积分状态。
5. 本次 Z4 随 forward 结束丢弃；下一次 evaluation 重新构造。
6. 输出保持 BasePolicy 的 `future_actions=[B,13,19]`，执行端仍按 C 选择动作数量。

不跨去噪步、真实查询或 batch 保存动态 Z4／动态 self-attention K/V。不修改静态缓存的返回类型、缓存关闭条件或作用域。缓存与非缓存两条路径使用完全相同的 body 初始化公式。

普通 predict_action 不调用力 head，也不返回额外状态；六个交互 token 必须仍参与计算。不新增未来力推理 API、force-guidance、在线搜索、RTC、异步状态缓存或新的执行模式。

### 7.4 checkpoint

新结构需要新训练。旧 head 的输入维数和输出维数已经改变，且旧模型没有 future_body_embedding，不能承诺旧 checkpoint 在新类上 strict load。

- 旧 checkpoint 用对应旧代码版本恢复。
- 新 checkpoint 继续保存完整 effective_config、model／EMA state、optimizer state 等现有内容。
- 新模型自身的保存、恢复、继续训练应沿用当前机制。
- 不引入 `strict=False`、参数截断、自动迁移、双版本 policy 或新的恢复分支。
- 保留最新 restore_policy 的 FP32 恢复逻辑。

## 8. 逐文件实施范围

下表路径相对 `dexmani_policy/`。本次代码实施以这 4 个文件为目标：

| 文件 | 具体改动 |
| --- | --- |
| 新增 `agents/action_decoders/backbones/interact_dex.py` | 实现小型 InterActDiTX 子类：继承已有网络，增加 future_body_embedding、固定 mask、拼接与输出切分 |
| `agents/action_decoders/backbones/ditx.py` | 仅给 DiTXBlock.forward 增加默认 None 的 attn_mask，并透传给已有 self_attn |
| `agents/policies/interact_dex.py` | 改用 InterActDiTX；单偏移解包；head 输入/输出与零初始化；读取共享特征；生成力残差目标；保留现有日志、权重 0 路径和 optimizer 分组 |
| `configs/interact_dex.yaml` | 把 future_force_offsets 改为 [4] |

已经确认可复用而无需修改：Flow、BasePolicy、ObservationEncoder、HandHistoryEncoder、Dataset、sampler、normalizer、配置组装、训练器、EMA、部署与 checkpoint 模块。

不改其他 policy，不接入 DexLatent，不维护新的导出／注册表，不顺带重排 import 或格式化无关代码。若实施时当前源码已变化，只做当前调用链所需的最小适配，最终说明实际超出目标文件范围的原因；不得为满足固定文件数而保留真实接口错误。

## 9. 效率预算与能力表述

以当前默认 768 维、8 层和启用四偏移辅助 head 的原模型作比较：

| 算子／参数项 | 新旧关系 |
| --- | --- |
| 动态 token 数 | 16 -> 22 |
| 共享 block 中按 token 计算的 self QKV、cross query/output 与 FFN | 主要 token 项比例 22/16 = 1.375 |
| self-attention 的 QK/AV 二次项 | (22/16)^2 = 1.890625 |
| 静态观测编码与各层 cross K/V | 沿用原数量与请求内缓存 |
| 动作输入位置数、动作 velocity 输出 MLP | 仍只处理 16 个动作位置 |
| 新增 body 初始化 | 复用投影，仅对 6 个身体 token 进行小投影 |
| 新增 future_body_embedding 参数 | 6 × 768 = 4,608 |
| 原辅助 head 参数 | 192->128->12，共 26,252 |
| 新辅助 head 参数 | 768->128->3，共 98,819 |
| 参数净增加 | 4,608 + 98,819 - 26,252 = 77,175 |

以上比例不是总 FLOPs 或完整 predict_action 时延比例。固定 mask 也不自动意味着稀疏执行；attention kernel 选择和设备会影响实际结果。不要宣称零推理开销、已满足某个毫秒预算或取得确定控制提升。

Z4 是动作生成内部的交互预见表示。高噪声时的候选动作缺乏明确行动信息；观察式演示监督也不等于干预式动力学识别。因此不能仅凭结构或力误差下降宣称得到校准的反事实世界模型、接触分类器或显式规划器。

## 10. 有限验收：只运行必要检查，不新增组件

### 10.1 本任务书写入前已完成的核对

- 通过 GitHub 读取上述基线的真实源码、调用方、配置、归一化和恢复路径，并对比了先前讨论基线后的相关提交。
- 从检索到的 sampler 源码提取并执行原始 window_indices，得到第 3.1 节的结果。
- 使用 NumPy 做 22-token 布尔注意力图的 8 层可达性检查，并确认“只屏蔽 Z4”会在两层泄漏。
- 核算算子项比例与 77,175 个净新增参数。
- 本次环境没有 PyTorch/timm，因此没有运行拟议模型的 PyTorch forward/backward、checkpoint 恢复、真实数据训练、时延测量或机器人闭环。

这些是任务设计的证据，不代表拟议源码实现已经通过运行验收。

### 10.2 实施者完成代码后的必要检查

使用已有工具或临时内联命令；不把检查脚本、夹具或测试模块提交到仓库，不修改既有 test_tools 来扩充测试体系。按实际环境执行，缺失依赖或数据时明确报告，不伪造通过，也不因此顺带安装／重构依赖体系。

1. **静态与配置合同。** 检查最终 diff 与语法；组装 interact_dex 配方，确认 agent 与 dataset 的 offsets 都为 [4]，horizon=16、action_start_index=3。新增参数被既有 optimizer 分组覆盖一次。
2. **一次训练链路。** 以适当小的合成 batch 或可用真实 batch，确认 action loss、残差 loss、backward 均有限；future head 输出 [B,1,5,3]；权重 0 时无 future_force 字段也能训练。Sim 的 query 旋转和 real 的 SDK 差值按已知数值核对；零残差与无效标签语义正确。
3. **非退化信息边界与梯度。** 在至少两层的临时模型上使用非零 self-attention gates、非零 action velocity 输出末层与非零力 head 末层，关闭 dropout。固定 condition、timestep，分别扰动 H/T，P velocity 和 Z4 features 不应改变；扰动 P 时 Z4 应能响应。仅从力 loss 反向时，共享主干与观测表示应能收到梯度。只看零初始化模型的“相等”或零梯度不能作为通过依据；临时权重不能写入最终模型初始化。
4. **缓存与无残留。** eval 下固定同一输入，缓存／非缓存结果在合理浮点容差内一致；用非退化临时权重执行 x1、x2、x1 的调用次序，前后两次 x1 一致。检查没有保存动态 Z4；训练时不复用推理 K/V。
5. **新 checkpoint 与对外接口。** 对新模型做一次保存／strict restore／继续训练的有限检查；普通 predict_action 返回 [B,13,19] 且不调用力 head。保持最新 FP32 积分和恢复方式。

可以复用 [现有 smoke_test](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0a32aa8a18b4ce9afba1076d6bffcdcd496e1ba/dexmani_policy/test_tools/smoke_test.py#L26)，在真实数据和环境具备时运行少量 fresh/resume updates；无需另造训练脚手架。通用 agent_checks 不覆盖本任务，不能以其结果替代上面的针对性核对。

控制收益与完整 predict_action 时延需要后续训练、真实环境和固定测量条件；本次代码实施不开展新的多因素消融、长时间训练或机器人实验。不得为了生成性能结论而扩大任务范围。

## 11. 完成交付

交付应包含这项核心优化的最小代码差异，并在最终回复中说明：

- 实际改动文件、共享交互状态与力残差的数据流。
- 已运行的有限检查、结果和未运行项目的具体原因。
- 新结构需要重新训练，旧 checkpoint 的恢复边界。
- 如当前源码变化导致必要适配，简要说明理由。

实现完成前检查没有引入第二套 Transformer、额外预测分支、重复参数分组、动态状态残留、未来标签输入或无关代码修改。达到上述合同并完成环境允许的必要验证后结束，不继续扩展方案。
