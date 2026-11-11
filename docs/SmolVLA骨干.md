# SmolVLA 风格 DiT：收敛设计与使用

本实现以 LeRobot 的 SmolVLA action expert 为依据，为本项目已有 observation encoder
和 RectifiedFlow 提供独立骨干。当前推荐方案是：**固定观测 K/V、交替 joint SA / CA、
输入 action/time 融合、GQA、RMSNorm、SwiGLU，并原生复用 SATActionIO。**

这是从头训练的研究基线。接口和数值正确性已经测试；“任务质量最优”仍需要相同训练
预算下的闭环评测。它不包含 SmolVLA 的预训练 VLM，也不能加载原 SmolVLA checkpoint，
更不能把 action expert 的参数量当成完整 VLA 参数量。

## 设计依据与取舍

固定的 upstream 源码版本为
[huggingface/lerobot@45a0756](https://github.com/huggingface/lerobot/tree/45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6/src/lerobot/policies/smolvla)。
本项目只接收 encoder 最终的 `[B,P,C]` token，没有 upstream VLM 每层 hidden states/K/V。

| 机制 | 当前方案 | 原因 |
| --- | --- | --- |
| 时间条件 | 原 SmolVLA 的 sin/cos time 与 action concat，再过 Linear–SiLU–Linear | 结构简洁，先保留有源码依据的基本路径 |
| Transformer | pre-RMSNorm、SiLU gated MLP；每两层一次 joint SA，其余为 CA | 保留动作间通信，同时减少每层都做 SA+CA 的开销 |
| GQA | 默认 8 个 Q heads、2 个 KV heads，head_dim 可独立设置 | 减少 K/V 参数和缓存；适配项目 PyTorch 2.4.1 |
| 观测 K/V | 每层独立 RMSNorm 和直接 K/V 投影 | 没有真实 VLM cache，不再模拟两级 K/V 适配 |
| 观测位置 | 保留 frame embedding；空间和 token-type 信息由 encoder 提供 | 点云 patch 的数组序号不是物理空间位置 |
| 普通动作 token | temporal causal mask + 局部 `arange(H)` RoPE | 保留 SmolVLA 的因果动作参照和 prefix-feature 语义 |
| SAT / 自定义 action_io | token 双向交互，不额外加序号 RoPE | I/O 自己提供关节等身份信息；token 轴不等于时间 |
| 输出 | 普通动作 final RMSNorm + Linear；SAT 直接复用自己的 readout | 不重复归一化/解码；保留 SAT 零 velocity head |
| 精度 | RMSNorm、RoPE 中间计算 FP32；SDPA 使用 Q/K/V 原 dtype | 让 BF16/FP16 输入具备选择融合 attention kernel 的条件 |
| 缓存 | 每次采样预计算一次、逐层只读 tuple | prefix 不依赖 noisy action 和 flow time，可跨 NFE 重用 |

原 SmolVLA CA 的二次 K/V 投影用于适配 VLM 与 expert；上一版 standalone 实现
也模拟了它。本版关闭 context 序号 RoPE 后，直接 observation→expert K/V 更直观，
不再保留两个连续的线性适配层。这是对本项目的设计取舍，不声称与 upstream
或上一版参数化逐位等价，也不兼容上一版 DiT 的 state_dict。

`qk_norm` 为兼容现有 SAT 构造器而保留，默认关闭。开启时，对所有层的 Q/K
采用逐 head 的 RMSNorm，位于 action RoPE 之前；它不同于 DiTX 仅在 CA 使用
LayerNorm 的具体实现。主对照使用现有 YAML 的 `qk_norm: false`。

## 实际张量流

SAT 使用既有协议，不增加 action I/O 框架：

| 阶段 | 形状与含义 |
| --- | --- |
| flow 输入 | `x_t: [B,H,19]`，H 步的 19 维动作 |
| SATActionIO.encode | 转为 19 个关节轨迹 token，输出 `[B,19,D]`，含 robot/joint descriptor |
| 时间融合 | 把 flow time embedding 广播到 19 个 token，concat 后投影到 D |
| 第 0、2、4…层 | action Q 读取所有 context K/V 和所有 action K/V |
| 第 1、3、5…层 | action Q 只读取 context K/V |
| SATActionIO.decode_velocity | 输出 `[B,H,19]` velocity |

只有 action hidden states 更新，context 不被 noisy action 反向更新。因此双向
**关节**交互与静态观测缓存可以同时成立。普通动作模式的 joint SA 则让动作 i
读取全部观测和动作 0…i；显式矩形 mask 避免 SDPA 非方阵 causal mask 屏蔽观测。

SAT 自定义 I/O 的约定是：`encode(x)` 返回带身份信息的 token，
`decode_velocity(hidden)` 恢复动作坐标。构造 DiT 时跳过 I/O 的递归初始化，
也不创建普通动作 input/output projection。SAT 的 readout 已包含归一化。

SAT 的短时间前缀不能直接编码：轨迹 MLP 的输入宽度固定为 H，因此
`forward_prefix_features` 明确拒绝自定义 action_io。RTC 对完整 `[B,H,19]`
求梯度后约束执行前缀，不依赖这个短前缀接口，已验证可用。

## Flow 约定与数值边界

沿用项目已有约定，不改 loss、noise、时间采样或 Euler 积分：

```python
x_t = (1 - t) * noise + t * action
target_velocity = action - noise
# Euler: t = 0 → 1, dt = 1 / NFE
```

原 SmolVLA 的噪声位于 t=1、积分方向相反。这里不直接移植它的速度符号、
采样循环或 checkpoint。当前接口仍是单时间 RectifiedFlow，不接受
ConsistencyDiTX 的 `target_t`。

RMSNorm 和 RoPE 保留 FP32 中间计算；连续时间编码采用原来的高精度频率计算，
随后转换为激活 dtype。SDPA 不再无条件把 Q/K/V 上转为 FP32。具体 CUDA kernel
取决于设备、dtype、mask、head_dim、dropout 和 PyTorch 版本，不能据此直接宣布
FlashAttention 已启用或 GPU 已加速。参见
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

项目锁定 PyTorch 2.4.1，GQA 使用显式 repeat_interleave，不使用较新版本的
`enable_gqa` 参数。缓存保持紧凑 KV heads；仅 attention 前展开。训练时
`prepare_context_kv` 返回 None，保留观测路径梯度；eval 缓存由调用方为同一观测、
权重、设备和精度环境构造。SA 临时拼接 action K/V，绝不写回缓存。

## 配置与接入

SAT 增加 `backbone_type: ditx | dit`，默认 ditx，已有配方和 checkpoint 可继续使用。
`effective_config` 会记录所选骨干。新实验只需：

```bash
bash scripts/training/train.sh sat agent.backbone_type=dit
```

沿用当前 `sat.yaml` 的 D=768、6 层、8 heads、time_dim=128、H=16、
2 帧观测、5 NFE、time_shift_alpha=3 和原训练预算。DiT 默认 2 KV heads、
每两层一次 SA。先在同精度下比较；评估 BF16 时可另加
`training.use_bfloat16=true`。不要同时缩宽度、压缩 context、降低 NFE，
否则无法判断质量和速度变化的来源。

直接构造的等价示例：

```python
from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.flow import RectifiedFlow
from dexmani_policy.agents.hand_prior.action_io import SATActionIO

backbone = DiT(
    horizon=16, action_dim=19, n_obs_steps=2, obs_token_dim=384,
    hidden_dim=768, n_layers=6, n_head=8, n_kv_head=2,
    timestep_embed_dim=128, mlp_ratio=4.0,
    p_drop_attn=0.1, qkv_bias=False, qk_norm=False,
    action_io=SATActionIO(horizon=16, hidden_dim=768),
)
decoder = RectifiedFlow(
    model=backbone, num_inference_steps=5,
    t_sample_mode="uniform", time_shift_alpha=3.0,
)
```

对于普通 `[B,H,A]` 时间 token，省略 `action_io` 即可；不用改 decoder。
自定义 I/O 接口在本版明确面向自带身份信息的集合 token，不用于隐式切换其他
时间序列表示。新的结构需要新训练，不能恢复旧 DiTX 权重/优化器为 DiT。

## 已测量的结构成本

按上面 SAT 配置实际构造并统计，以下均包含 SATActionIO，不含 observation encoder：

| 动作骨干 | 参数 |
| --- | ---: |
| DiTX，D=768、6 层 | 89,950,800 |
| 本版 DiT，D=768、6 层、2 KV heads | 40,266,576 |

参数约减少 55.2%。这是精确的结构统计，不代表成功率相同，也不等于运行时间减少
55.2%。P=258、6 层、2 KV heads、head_dim=96 时，batch=1 的紧凑 context cache
占 BF16 1,188,864 bytes（约 1.13 MiB），FP32 约 2.27 MiB；不包含临时 GQA 展开、
attention workspace 或整个模型显存。

保留全部 258 个观测 token，避免在没有训练证据时先丢失小物体和接触几何。
宽度压缩与 learned resampler 留给独立实验。

## 验证与下一步

```bash
OMP_NUM_THREADS=1 python -m unittest dexmani_policy.test_tools.dit_test -v
```

本次整合 main 后，在 Python 3.10.20、PyTorch 2.4.1+cu124、torchvision 0.19.1+cu124、
timm 1.0.22 环境中使用 CPU 验证：10 项回归通过，覆盖普通动作 flow 梯度、时间/观测依赖、因果性与 prefix feature、
缓存等价和只读、多步 Euler、BF16 autocast/整模 BF16、SAT 初始化与两步训练、
双向关节梯度、关节置换等变、帧内观测置换不变与跨帧身份、RTC。新增 SAT 的 Hydra
默认/显式骨干选择和 checkpoint 保存恢复回归；使用真实的小尺寸随机初始化 encoder，
检查严格权重恢复与非零 velocity 输出一致，旧配方缺少 backbone_type 时仍恢复为 DiTX。

本次另通过 14 项 correctness 回归、现有 DP3/CordViP CPU smoke（各训练 2 步后续训
2 步，并检查部署输出）、相关 Python 编译及 git diff --check。

历史检查（本次未重跑）：实际 H=16/P=258/D=768 配置的 BF16 有限输出与缓存等价，以及小型 SAT
模型在 CPU `torch.compile(backend="aot_eager", fullgraph=True)` 下的缓存 forward
与 action gradient 等价。AOT 图捕获不等于 GPU Inductor 验证。

本次未运行 SAT 完整 RGB-point encoder 前向、端到端训练/续训/部署流程、GPU kernel/延迟测量或机器人
闭环评测。下一步固定数据、encoder、动作表示、训练步数、NFE 和评测 seed，分别
比较 DiTX 与本版；报告成功率、轨迹质量、encoder/prefill/单步/整 chunk 延迟及峰值显存。
目标是以更小的动作主干达到可接受或更好的任务表现，不能先验保证“最佳”。

只有这个基线收敛后，再分别实验：

1. 每帧空间 resampler（128→32，保留 joint token，prefill 一次完成）。
2. FFN 前轻量逐层时间注入。
3. 空间通道适配、模态 attention 偏置，或训练期未来几何辅助监督。

这些候选分别来自前期源码调研，当前没有默认叠加。
依据与局限见[小型 VLA 调研](小型VLA与DiT优化.md)和
[2026 顶会 VLA/WAM 复核](2026顶会VLA与WAM优化.md)。
