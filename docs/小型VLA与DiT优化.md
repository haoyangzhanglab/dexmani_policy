# 小型 VLA 源码调研与 DiT 优化建议

本次调研固定下面列出的源码提交，目的是评估当前独立 `dit.py` 的改进方向。
下文区分已发布代码的事实、对本项目的推断和待做的实验；没有把跨项目的成功率或
吞吐数据当作本项目的收益证据。本轮更新设计说明，尚未修改模型计算或训练配置。

## SAT action_io：上一版说法的修正

`ditx.py` 已经支持 `action_io`，SAT 没有结构上不可兼容的问题。当前 `dit.py`
没有移植这条分支，上一版“SAT 需要另行适配”的表述没有清楚交代这是新增实现的
接口缺口。

现有协议只有 `encode(x)` 与 `decode_velocity(hidden)`，应该继续复用，不需要
另外建立 action I/O 框架。SAT 的实际张量流是：

| 阶段 | 形状 | 含义 |
| --- | --- | --- |
| flow 动作输入 | `[B,H,19]` | H 步、19 个关节 |
| `x.transpose(1,2)` | `[B,19,H]` | 每个关节的整段轨迹 |
| `action_io.encode` | `[B,19,D]` | 轨迹特征 + robot/joint descriptor |
| Transformer | `[B,19,D]` | 关节 token 间的交互 |
| `action_io.decode_velocity` | `[B,H,19]` | 恢复 flow velocity |

因此移植时需要同时处理以下细节：

1. 自定义 action_io 提供输入和输出，不重复创建普通动作的 Linear input/output。
2. time embedding 广播到编码后的 token 数，而不是原始 H；SAT 中二者为 19 和 16。
3. 跳过对 action_io 的递归重初始化，保留 SATActionIO 自己的零 velocity head。
4. SAT 的关节 token 应双向交互。因果 mask 与 action_io 必须解耦，不能把关节顺序
   当成时间顺序。robot/joint descriptor 已提供身份信息；一维 action RoPE 不是必需。
5. `forward_prefix_features` 不能把短时间前缀直接送入当前 SATActionIO：其轨迹
   Linear 的输入维度固定为 H。保持 DiTX 的显式拒绝行为即可；这不妨碍 decoder RTC
   对整段动作输入求梯度。
6. SAT 的构造器还向 backbone 传 `qk_norm`，需要明确保留或删除该配置，不能只换 import。

项目依据：[SAT 构造器](https://github.com/haoyangzhanglab/dexmani_policy/blob/177537e5cbba9e66a1b700537a11945f83e629f3/dexmani_policy/agents/policies/sat.py)、
[SATActionIO](https://github.com/haoyangzhanglab/dexmani_policy/blob/177537e5cbba9e66a1b700537a11945f83e629f3/dexmani_policy/agents/hand_prior/action_io.py)、
[DiTX](https://github.com/haoyangzhanglab/dexmani_policy/blob/177537e5cbba9e66a1b700537a11945f83e629f3/dexmani_policy/agents/action_decoders/backbones/ditx.py)。

## 调研对象与真实动作头

| 项目、固定提交 | 规模口径 | 源码事实 | 对本项目的启发 |
| --- | --- | --- | --- |
| TinyVLA `94f4418` | VLM 约 400M/700M/1.3B，并非精确总参数 | ConditionalUnet1D；sin/cos time、FiLM；对称卷积处理动作 chunk | 动作联合去噪不要求 causal；深层时间调制可作为消融项 |
| Octo `241fb35` | Small 27M、Base 93M；小型通用机器人策略 | Transformer 编码一次，默认 3-block、256 宽 MLP diffusion head，20 steps；整个 H*A chunk 联合建模 | 固定条件计算移出 NFE；观测历史 causal 与动作 chunk causal 要分开 |
| VLA-Adapter `23fa0c9` | Qwen2.5-0.5B 语言 backbone，非总参数 | L1RegressionActionHead，24 个 cross-conditioned MLPResNet blocks；不是已发布的 DiT/flow head | action/context 分开投影合理；真正的逐层 VLM features 需要 encoder 支持 |
| X-VLA `6c4ecfb` | 约 0.9B 总模型 | SoftPromptedTransformer；action/proprio/time 拼接，双向 self-attention；原仓库回归 clean action 后迭代更新 | 不必为 DiT 强行加入 AdaLN；也不能原样移植其 loss/采样循环 |
| MiniVLA `0822b36` | 支持 Qwen2.5-0.5B 语言 backbone，非总参数 | Residual VQ 将 chunk 编码为若干离散 codewords，再自回归生成 | token 轴不一定等于动作时间；action_io 应保持明确的表征边界 |

具体源码：

- TinyVLA：[README](https://github.com/liyaxuanliyaxuan/TinyVLA/blob/94f441827b45e4f76316ef6a0ae443736dc93a5d/README.md#L70)、
  [UNet 与 FiLM](https://github.com/liyaxuanliyaxuan/TinyVLA/blob/94f441827b45e4f76316ef6a0ae443736dc93a5d/policy_heads/models/droid_unet_diffusion.py#L118)、
  [DDIM 循环](https://github.com/liyaxuanliyaxuan/TinyVLA/blob/94f441827b45e4f76316ef6a0ae443736dc93a5d/llava-pythia/llava_pythia/model/language_model/pythia/llava_pythia.py#L363)。
- Octo：[动作头](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/model/components/action_heads.py#L386)、
  [一次运行 Transformer](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/model/octo_model.py#L204)、
  [time/obs/action concat](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/model/components/diffusion.py#L141)。
- VLA-Adapter：[实际 head](https://github.com/OpenHelix-Team/VLA-Adapter/blob/23fa0c9c159e2aa04341cdd3e924f44061311060/prismatic/models/action_heads.py#L21)、
  [Pro attention](https://github.com/OpenHelix-Team/VLA-Adapter/blob/23fa0c9c159e2aa04341cdd3e924f44061311060/prismatic/models/action_heads.py#L287)。
  注意其 FiLM 调用被注释，不能把定义存在当成实际有效机制；README 把 diffusion/flow
  列为未来计划，也不能称它已经发布了 U-DiT。
- X-VLA：[双向 attention](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/models/transformer.py#L139)、
  [输入融合](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/models/transformer.py#L370)、
  [clean-action target 和迭代](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/models/modeling_xvla.py#L148)。
- MiniVLA：[官方设计说明](https://github.com/Stanford-ILIAD/openvla-mini/blob/0822b36227b5a771be4eb2680e34c559734c8fdc/README.md#L488)、
  [生成与动作解码](https://github.com/Stanford-ILIAD/openvla-mini/blob/0822b36227b5a771be4eb2680e34c559734c8fdc/prismatic/models/vlas/openvla.py#L81)。

补充对照 DexVLA/ScaleDP `fc21a82`：其 2B VLM 加 400M/1B action head 不属于本表的
小型完整 VLA，但 [ScaleDP block](https://github.com/juruobenruo/DexVLA/blob/fc21a822f4c774e242eb6f1ab4a235788de7aba9/policy_heads/models/transformer_diffusion/modeling_scaledp.py#L148)
提供深层调制参考，[forward](https://github.com/juruobenruo/DexVLA/blob/fc21a822f4c774e242eb6f1ab4a235788de7aba9/policy_heads/models/transformer_diffusion/modeling_scaledp.py#L447)
明确使用 `attn_mask=None`。构造器中的 `initialize_weights()` 被注释，不能误报它
所有路径已启用 AdaLN-Zero 初始化。

## 优化优先级

### 1. 先修 token 语义与接口，再比较网络结构

优先复用 action_io，显式允许 causal/bidirectional 动作交互，并分开 action 与
context 的位置策略。对 SAT 建议双向关节 attention、使用已有 robot/joint descriptor，
不强行加入按时间解释的一维 RoPE。普通时间动作 token 则保留 causal 参照组，
在同一训练预算下测试 bidirectional，之后再决定默认值。

当前 SAT 配置有 2 帧、每帧 128 个空间 patch 和 1 个 joint-state token，合计 P=258；
backbone 配置是 D=768、6 层、8 heads、5 NFE。
[配置](https://github.com/haoyangzhanglab/dexmani_policy/blob/177537e5cbba9e66a1b700537a11945f83e629f3/dexmani_policy/configs/sat.yaml)、
[观测 encoder](https://github.com/haoyangzhanglab/dexmani_policy/blob/177537e5cbba9e66a1b700537a11945f83e629f3/dexmani_policy/agents/obs_encoder/rgb_point.py)。

点云 patch 序号不是稳定的几何位置。当前 context `arange(P)` RoPE 会让同帧空间
token 的排列影响输出；已有 3D geometry features、frame embedding 与 token-type
embedding，应先评估是否还需要该一维序号。建议对点云 context 禁用这类序号 RoPE，
保留 frame/geometry 信息；动作的时间 RoPE 独立控制。

关闭 context 序号 RoPE 后，普通动作位置也宜从 `P + arange(H)` 改为局部
`arange(H)`，避免 P 通过无关的绝对偏移影响 action→context 分数。空间 context K
不旋转、动作 Q 随时间旋转仍可构成置换不变的 context attention；SAT 则可以同时
关闭 action 序号旋转，继续使用已有 joint descriptor。

用未训练的小型 DiT 进行过一次结构检查：只重排每帧 3 个空间 token、保持 joint token
位置和所有 token 值不变，输出最大差异约 3.89e-4。这只证明当前模型依赖该顺序，
不证明差异在训练后有害。移除序号编码后的置换不变性可作为语义回归检查，最终仍需
验证策略效果。

### 2. 让 BF16 SDPA 成为可测的工程候选

当前 attention 无条件关闭 autocast 并将 Q/K/V 全部转 FP32。这有稳定性价值，
但无法利用通常要求 FP16/BF16 的 CUDA FlashAttention 路径。建议在保持 RMSNorm
等归一化 FP32 的同时，比较原生激活 dtype 的 SDPA 与当前 FP32 参照路径。

项目 `common.yaml` 当前 `use_bfloat16: false`，因此必须连同训练/推理精度环境一起
评估，不能只改一个 `.float()` 就宣称端到端加速。测试目标应包含目标 GPU 的 kernel
选择、prefill 和单步/NFE 延迟、显存、loss/梯度有限性、RTC VJP、cache parity。
本轮没有 GPU 基准或训练结果，不报告速度提升。

本项目安装的 PyTorch 2.4.1 SDPA 没有 `enable_gqa` 参数，不能直接搬用新版示例。
当前 repeat_interleave 可运行；是否升级依赖应另行评估。

### 3. 保留只读 context cache，按目标 batch 决定进一步预计算

已有 cache 将逐层观测 K/V 与 CA 二次投影移出了 NFE 循环，应该保留。
若动作 token 改成双向，只开放 action/action 交互；不要顺便让 prefix hidden states
依赖 noisy actions，否则原 cache 将不再成立。

可测的后续工作包括减少每层每步重复的 positions、sin/cos、mask 构造，以及 eval
阶段是否提前展开 GQA K/V。按 SAT 的 D=768、L=6、P=258、head_dim=96、
新 DiT 默认 Hkv=2 计算，紧凑 FP32 context cache 每个样本约 2.27 MiB，展开到
8 个 heads 则约 9.07 MiB。提前展开是显存换重复计算，不能无条件称为优化。

### 4. 简化静态 observation adapter 的二次 K/V 投影，做独立消融

SmolVLA 原来的二次投影解决 VLM K/V 与 expert 的接口适配。当前没有真实逐层 VLM
状态，固定 context 本来就由自己的 adapter 生成，额外的方阵投影未必是最佳参数预算。
建议比较当前路径与直接 context→expert K/V 的简化路径。

这里不是简单删除两行：K 的 RoPE 处于两次投影之间，通常不能代数合并；V 在没有
中间非线性时可以组合线性映射，但有限精度仍可能产生差异。这个消融应与位置策略
一起明确设计，并检查实际参数量、吞吐和任务效果。

### 5. 深层 time modulation 和 context 压缩属于后续实验

TinyVLA 的 FiLM 和 ScaleDP 的调制为深层注入 time 提供实验动机，但 Octo 和
X-VLA 也说明输入融合可以工作。建议先比较稳定的 SmolVLA-style 输入 time concat，
再单独实验轻量的每层 time conditioning；不要一次叠加 AdaLN、QK norm 和 gates，
否则难以解释收益。

P=258 相对 H=16 或 SAT 的 19 个动作 token 较长，可以实验按帧压缩空间 context，
保留单独的 proprio/joint token。直接全局 mean pooling 可能损失局部接触与几何信息，
不应只因 TinyVLA 使用了它就默认采用。X-VLA 的全序列双向 Transformer 会逐步更新
视觉/action token 混合状态，与当前固定观测 KV 的缓存语义不同，也不适宜直接替换。

## 建议的实验顺序

| 次序 | 改动 | 应证明什么 |
| --- | --- | --- |
| A | action_io 支持、SAT 双向、正确位置语义 | shape/梯度/初始化/RTC 正确，关节间能双向传递信息 |
| B | 原生 BF16 SDPA 与 FP32 对照 | 目标 GPU 的实际速度与稳定性，不改变 loss/表示 |
| C | 时间动作 token 的 causal 与 bidirectional 对照 | 相同参数/encoder/NFE 下的任务成功率与轨迹质量 |
| D | 静态 K/V adapter 简化、context 压缩分别测试 | 能否在质量可接受时减少延时/显存 |
| E | 深层 time conditioning | 前述问题收敛后，是否仍有可重复的任务收益 |

各组固定数据、动作表示、encoder、训练步数、采样 NFE 与评测 seed；报告 encoder、
context prefill、单次 expert forward 和完整 action chunk 延迟。跨论文的模型总参数、
NFE、动作维度和硬件不同，不能直接据其 FPS 排名选结构。
