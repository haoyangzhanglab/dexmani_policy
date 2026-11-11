# 小型 VLA 源码调研

本文记录固定源码版本的动作头设计，供独立实验参考。当前 DiT/SAT 的实现、
位置语义、精度与缓存约定以[SmolVLA 骨干方案](SmolVLA骨干.md)为准。
跨项目参数量、吞吐和成功率不作为本项目的收益证据。

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

进一步的研究候选见[2026 顶会源码复核](2026顶会VLA与WAM优化.md)，均未作为默认组件加入。
