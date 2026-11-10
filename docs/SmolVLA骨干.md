# SmolVLA 风格的独立 DiT backbone

`dexmani_policy/agents/action_decoders/backbones/dit.py` 提供 `DiT`，与现有
`ditx.py` 并列，供普通动作 token 的 `RectifiedFlow` 使用。现有 policy 仍使用原主干。

## 源码依据与移植边界

参照 LeRobot 提交
[`45a0756`](https://github.com/huggingface/lerobot/commit/45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6)：

- [`modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6/src/lerobot/policies/smolvla/modeling_smolvla.py#L698)：action/time 输入融合、动作 mask、输出 velocity。
- [`smolvlm_with_expert.py`](https://github.com/huggingface/lerobot/blob/45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6/src/lerobot/policies/smolvla/smolvlm_with_expert.py)：逐层 SA/CA、RoPE、GQA、prefix K/V 的二次投影。
- [`configuration_smolvla.py`](https://github.com/huggingface/lerobot/blob/45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6/src/lerobot/policies/smolvla/configuration_smolvla.py)：默认 16 层、每两层一次 SA、时间 period 为 0.004–4.0。
- [`vla_utils.py`](https://github.com/huggingface/lerobot/blob/45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6/src/lerobot/policies/common/vla_utils.py)：时间编码和累计分组 attention mask。

SmolVLA 的 action expert 是语言模型式 Transformer，不能视作现有 DiTX 改名。
原模型同时运行 VLM prefix 和 action expert；每个 expert 层读取对应 VLM 层的
prefix K/V，联合 SA 时两条分支各自投影 Q/K/V。CA 层则再次投影 VLM 的
post-RoPE K 和 V。

本项目的 observation encoder 只返回一个 `[B, P, C]` tensor，没有逐层 VLM
状态。因此本实现给每个 expert 层建立独立的 observation RMSNorm 和 K/V
投影，以固定的观测特征生成该层 prefix K/V；不再增加一个从零训练的
observation Transformer。它保留 action expert 的计算结构，但不复现 VLM
prefix 的逐层更新，也不提供视觉语言预训练或原 SmolVLA checkpoint 加载。

## 与 DiTX 的区别

| 项目 | 现有 DiTX | 新 DiT |
| --- | --- | --- |
| 时间条件 | 每层 AdaLN-Zero 的 shift/scale/gate | 输入 action 与 sin/cos time 拼接后经过两层 MLP |
| 每层注意力 | 每层 SA，再 CA | 第 0、2、4…层 joint SA；第 1、3、5…层 CA |
| 动作可见性 | chunk 内双向 | SA 为 causal；CA 仅读 prefix |
| SA 的 memory | 动作 token | 所有 observation prefix 加当前及较早动作 |
| 归一化 | LayerNorm，输出 RMSNorm | pre-norm RMSNorm |
| token 位置 | 可学习 action position | Q/K 的 RoPE；观测另保留 frame embedding |
| FFN | GELU MLP | SiLU gated MLP |
| head 结构 | MHA | GQA，可独立指定 head_dim |
| 输出 | DiTXOutput 两层 MLP，末层零初始化 | final RMSNorm + Linear，普通随机初始化 |

时间编码不是 action token 位置编码：前者表示 flow 的连续时间，后者表示
序列位置，二者同时存在。新实现没有 AdaLN 或 zero gates，避免把两种结构混在一起。

## 张量流与 mask

输入 `x: [B, H, A]`，`context: [B, P, C]`，
`timestep` 为标量、单元素 tensor 或 `[B]`。观测使用与 DiTX 相同的 frame-major
顺序，`P = n_obs_steps * tokens_per_frame`，不接受 padding token。

1. `Linear(A,D)` 投影动作。几何间隔的 period 生成 time sin/cos，默认维度为 D。
2. 广播 time 到 H 个动作，与 action embedding 拼接，经过 Linear → SiLU → Linear。
3. 每层执行 `h += attention(RMSNorm(h))`，再执行
   `h += gated_mlp(RMSNorm(h))`。
4. final RMSNorm + Linear 返回 `[B,H,A]` velocity。

SA 层 action Q/K 的 RoPE 位置为 `P + arange(H)`；context K 位置为
`arange(P)`。第 i 个动作可读取全部 P 个观测和动作 0…i。这里需要显式的
矩形 mask；直接对 `[H, P+H]` 使用 SDPA `is_causal=True` 会遮掉一部分观测。

CA 层仅 action Q 使用局部位置 `arange(H)`；context K 在观测投影后已经应用
RoPE，再 flatten 后经过 expert K 投影，不重复旋转。context V 也经过二次投影。

Q heads 数为 `n_head`，K/V heads 数为 `n_kv_head`，要求前者是后者的整数倍。
`head_dim` 是独立参数：Q projection 为 `D → n_head * head_dim`，因此允许
`D != n_head * head_dim`，这对于 SmolVLA 缩窄 expert width 的设计很重要。
默认省略 `head_dim` 时才使用 `D // n_head`。

## Flow 时间方向

仓库 `RectifiedFlow` 使用：

```text
x_t = (1-t) * noise + t * action
target_velocity = action - noise
Euler: t=0 → 1，dt=+1/N
```

原 SmolVLA 使用相反约定：噪声在 t=1，target 为 noise-action，从 1 积分至 0。
新 backbone 按项目 decoder 传入的 t 学习目标，既不反转 t，也不取负输出。
不能把原 SmolVLA 的 velocity、时间分布或预训练权重直接接到此 decoder。
若未来引入原预训练函数，才需要显式使用 `tau=1-t` 和 `v_project=-v_original`。

## 使用方式

```python
from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow

backbone = DiT(
    horizon=16,
    action_dim=19,
    n_obs_steps=2,
    obs_token_dim=obs_encoder.out_dim,
    hidden_dim=256,
    n_layers=8,
    n_head=8,
    n_kv_head=2,
    self_attn_every_n_layers=2,
)
decoder = RectifiedFlow(model=backbone, num_inference_steps=5)
```

新类保留 `forward(x, timestep, context, *, context_kv=None)`、
`prepare_context_kv(context)` 和 `forward_prefix_features(...)`。默认深度 8
是本项目的实验规模；若要对齐 upstream 深度可设置 `n_layers=16`。
FFN 中间维度沿用 SmolVLA 的约 8D/3 并向上取整到 256，默认不使用 attention dropout。
`timestep_embed_dim` 可显式设置；默认与 hidden_dim 相等。

它没有 DiTX 的 `qk_norm`、SAT `action_io` 或 ConsistencyDiTX 的 `target_t`。
SAT 按关节组织 token，不能直接采用此处的时间 causal 假设；ManiFlow 的一致性目标
需要另一套双时间条件。因此不应批量替换所有 policy 的 import。普通动作 token 的
policy 接入时，需要同步改构造器参数，再进行训练和评测。

## 缓存与验证

`prepare_context_kv` 在 eval 下返回逐层 tuple；训练时返回 None，使所有观测投影
保持可微。缓存包含 CA 的二次投影结果，只属于当前观测、权重、设备和计算精度。
SA 每次临时拼接当前 noisy action 的 K/V，不把它们写回缓存，避免跨 Euler 步污染。
调用方负责在同一精度环境下构造和使用缓存；现有 `RectifiedFlow` 已覆盖此调用模式。

RMSNorm 和 RoPE 中间运算使用 FP32；SDPA 显式关闭 autocast 后使用 FP32，之后
恢复激活 dtype。它使用 SDPA 的 FP32 attention，数值实现与 upstream eager
softmax 后降精度的路径并非逐位一致。

运行针对该主干的 CPU 回归：

```bash
python -m dexmani_policy.test_tools.dit_test
```

验证范围包括时间与观测依赖、梯度、因果性、缓存等价性、多步 flow、RTC 和 BF16。
这些检查只能证明实现契约；策略收益还需在相同 encoder、动作表示、训练预算和 NFE
下与 DiTX 对照实验，不能仅凭结构判断优劣。
