# ActionFlow-vNext：实现冻结与代码修改计划

> 基线：最后确认的 `main@bd6ca600ee92876c71e228708265ed4e4a310c88`（commit: `action_flow exp`）。
>
> 目标：在不修改 action space / action representation / BaseAgent / Trainer / Dataset / rollout 接口的前提下，将当前 ActionFlow 升级为简洁、高效、可解释的 8-layer DiT-X，并保持 Rectified Flow 训练目标极简。

---

## 1. 最终冻结的模型设计

### 1.1 不修改的部分

- `horizon = 16`
- `n_obs_steps = 2`
- `n_action_steps = 8`
- action representation 仍为 `[B, 16, action_dim]`
- `action_dim = 19`（joint）/ `21`（EE，根据现有配置）
- PointNext tokenizer
- `num_patches = 128`
- action hidden width `512`
- `depth = 8`
- `8Q / 4KV GQA`
- QK-Norm
- GEGLU
- zero-gated residual
- learned absolute action position embedding
- observation time embedding
- KV cache
- Rectified Flow target `v* = action - noise`
- Euler / Midpoint inference solver
- 默认 Midpoint NFE=2

### 1.2 vNext 核心变化

1. `4 SA + 4 CA` alternating layers  
   → `8 × (SA → CA → FFN)` 完整 DiT-X blocks。

2. observation/context width  
   `512 → 256`。

3. Cross-Attention 采用 asymmetric bottleneck：
   - action residual stream: 512D
   - query projection: 512 → 256
   - geometry/state memory: 256D
   - 8 query heads / 4 KV heads
   - head_dim = 32
   - output projection: 256 → 512

4. State 不再作为独立 token；采用 R3D-style **state-conditioned geometry memory**：
   - 每个 observation frame 的 `joint_state` 经 StateMLP；
   - state embedding 复制到该帧所有 global/patch token；
   - 与 point feature concat 后投影到 256D；
   - 每帧 state 只调制本帧 geometry tokens；
   - state-conditioned context 仍是静态 memory，可完整 KV cache。

5. Flow timestep 继续作为 AdaRMS / zero-gate 的唯一全局 modulation condition。

6. Flow time sampler：
   - F1: `NoiseShift(alpha=4)`
   - F2: `75% NoiseShift(alpha=4) + 25% Uniform`
   - Flow target / loss / solver 完全不改。

---

## 2. 最终数据流

```text
PointCloud_{t-1}, PointCloud_t
        │
        ▼
PointNext
        │
        ├── patch tokens:  [2B, 128, 128]
        └── global token:  [2B,   1, 128]

JointState_{t-1}, JointState_t
        │
        ▼
StateMLP (per frame)
        │
        └── state emb: [2B, D_s]
                 │
                 ├──────── expand to patch/global tokens
                 ▼
       concat(point feature, state emb)
                 │
                 ▼
       patch/global projection → 256
                 │
       + type embedding
       + observation-time embedding
                 │
                 ▼
State-conditioned Observation Memory
           [B, 258, 256]
                 │
                 │ static K/V cache
                 ▼

Noise Action x_t [B,16,A]
        │
        ▼
Linear A → 512
        │
+ learned action position
        │
        ▼
Action stream [B,16,512]

Flow timestep t
        │
        ▼
TimestepMLP → 512
        │
        ▼
8 × ActionFlowDiTXBlock
    ├── AdaRMS(t)
    ├── Self-Attn 512D, 8 heads
    ├── zero-gated residual
    │
    ├── AdaRMS(t)
    ├── GQA Cross-Attn
    │     Q: 512 → 256
    │     KV: context 256
    │     8Q / 4KV / head_dim=32
    ├── zero-gated residual
    │
    ├── AdaRMS(t)
    ├── GEGLU 512 → 896 → 512
    └── zero-gated residual

        ▼
Adaptive RMSNorm(t)
        ▼
Linear 512 → A (zero initialized)
        ▼
velocity v_theta [B,16,A]
```

默认 128 patches 时 observation token 数：

`2 × (128 patch + 1 global) = 258`

不再存在独立 state token。

---

## 3. 分阶段实施

### Stage B0 — 冻结当前基线

保持当前代码不变。

必须记录：
- parameter count
- training forward latency
- Midpoint-2 inference latency
- peak VRAM
- 20k/100k closed-loop success
- solver sensitivity

---

### Stage B1 — 完整 8-layer DiT-X

#### 文件

`dexmani_policy/agents/action_decoders/backbone/action_flow_dit.py`

#### 修改

删除：
- `attention_pattern`
- `attention_type`
- “一层只含 SA 或 CA”的逻辑

新增统一 block：

```python
class ActionFlowDiTXBlock(nn.Module):
    def __init__(...):
        self.self_attn = SelfAttention(...)
        self.cross_attn = GQACrossAttentionWithCache(...)
        self.ffn = GEGLU(...)
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim),
        )
```

modulation 顺序固定：

```text
scale_sa, shift_sa, gate_sa,
scale_ca, shift_ca, gate_ca,
scale_ffn, shift_ffn, gate_ffn
```

Forward：

```python
# SA
h = modulate_rms(x, scale_sa, shift_sa)
x = x + gate_sa[:, None] * self.self_attn(h)

# CA
h = modulate_rms(x, scale_ca, shift_ca)
x = x + gate_ca[:, None] * self.cross_attn(h, context)

# FFN
h = modulate_rms(x, scale_ffn, shift_ffn)
x = x + gate_ffn[:, None] * self.ffn(h)
```

KV cache：

```python
def setup_kv_cache(self, context):
    for block in self.blocks:
        block.cross_attn.setup_kv_cache(context)

def clear_kv_cache(self):
    for block in self.blocks:
        block.cross_attn.clear_kv_cache()
```

#### 不修改

- ObsEncoder
- context shape
- Flow decoder
- config

#### 验收

- `smoke_test.py action_flow` 通过
- cached / uncached forward 一致
- 初始化 output ≈ 0
- finite backward
- 20k screening 不明显劣于 B0

---

### Stage B2 — 256D Asymmetric Cross-Attention

#### 文件

- `dexmani_policy/agents/core/action_flow.py`
- `dexmani_policy/agents/action_decoders/backbone/action_flow_dit.py`
- `dexmani_policy/configs/action_flow.yaml`

#### 配置

只新增一个 width：

```yaml
hidden_dim: 512
context_dim: 256
```

不要额外新增 `cross_attn_dim`；Cross-Attention 内部维度直接等于 `context_dim`。

#### ObsEncoder

Stage B2 仍保留独立 state token，以隔离变量。

```text
patch/global/state:
→ 256D
```

context：

`[B,260,256]`

#### Cross-Attention

推荐接口：

```python
GQACrossAttentionWithCache(
    query_dim=512,
    context_dim=256,
    num_heads=8,
    num_kv_heads=4,
)
```

内部：

```text
q_proj:   512 → 256
k_proj:   256 → 128
v_proj:   256 → 128
out_proj: 256 → 512
```

shape：

```text
Q: [B,8,16,32]
K: [B,4,260,32]
V: [B,4,260,32]
```

先继续使用当前 `repeat_interleave` GQA 实现。Native GQA 必须留给后续纯工程 commit。

#### 验收

- parameter count < B1
- latency < B1
- VRAM < B1
- 成功率不显著下降

---

### Stage B3 — R3D-style State-Conditioned Geometry

这是核心 conditioning 修改。

#### 文件

`dexmani_policy/agents/core/action_flow.py`

原则：**不修改 BaseAgent，不改 cond 类型。**

#### State 时间语义

保持与当前 ActionFlow 完全一致：

- 当前输入已是 `[B*T, state_dim]`
- 每个 observation frame 独立经过 StateMLP
- 不改成 `[q_t, Δq]`
- 不做跨帧 state history 重参数化

这样 B2 → B3 唯一核心变化就是：
“state 独立 token” → “state 融入同帧 geometry tokens”。

#### 推荐实现

```python
state_emb = self.state_encoder(obs["joint_state"])  # [BT, D_s]
```

对 patch：

```python
patch_state = state_emb[:, None, :].expand(
    -1, patch_tokens.shape[1], -1
)
patch_tokens = torch.cat(
    [patch_tokens, patch_state], dim=-1
)
patch_tokens = self.patch_proj(patch_tokens)
```

对 global：

```python
global_tokens = torch.cat(
    [global_token, state_emb[:, None, :]], dim=-1
)
global_tokens = self.global_proj(global_tokens)
```

然后：

```text
+ patch/global type embedding
+ obs_time_embed
```

最终：

`[B, 258, 256]`

删除：
- `state_embed`
- 独立 state token concatenation

#### state_out_dim：严格隔离变量

B3 主实验必须保持与 B2 相同的 state representation width：

```yaml
state_out_dim: 256
```

原因：B2 中独立 state token 已经是 256D。B3 若同时改为 64D，会同时改变：

1. state 的融合位置；
2. state representation capacity。

这样无法判断收益来自 R3D-style fusion 还是 state bottleneck。

因此：

```text
B3:
state_out_dim = 256
只验证：
state token → state-conditioned geometry
```

若 B3 成立，再单独增加：

```text
B3b:
state_out_dim = 64
```

验证 compact proprio branch 是否能进一步降参数/延迟而不掉性能。

最终部署配置是否采用 64D，由 B3b 实验决定，而不是预先锁死。

#### 重要

State-conditioned context 是静态 observation memory，因此 inference 时仍然：

```text
Point + State → K/V
```

一次 projection 后 cache；不会因为 NFE 增加而重新编码 state。

#### 验收

- context token 数 `260 → 258`
- state 不再以独立 token 存在
- 每个 frame 只融合对应 frame state
- KV cache parity
- observation encoder latency / params 记录
- 20k success 与 B2 比较

---

### Stage B4 — Shared AdaRMS（可选，不是主线必选）

只在 B3 已经稳定后执行。

目标：减少每层 `512 → 9×512` modulation MLP 的重复参数。

主模型一次：

```python
shared_modulation = Linear(
    512, 9 * 512
)
```

每个 block 保留一个小型：

```python
modulation_table = nn.Parameter(
    torch.zeros(9, 512)
)
```

使用：

```python
layer_mod = base_mod + block.modulation_table[None]
```

Final AdaptiveRMSNorm 保持当前独立实现，不并入 shared modulation。

若 B4 性能下降，直接回滚 B3；不应为了参数量强行保留。

---

## 4. Flow Stage

### F0

当前：

```yaml
noise_shift_alpha: 2.0
noise_shift_ratio: 1.0
```

### F1

```yaml
noise_shift_alpha: 4.0
noise_shift_ratio: 1.0
```

### F2

```yaml
noise_shift_alpha: 4.0
noise_shift_ratio: 0.75
```

### MixtureTimeSampler

使用 i.i.d. mixture，而不是 batch 强制 96/32 划分：

```python
class MixtureTimeSampler:
    def __init__(
        self,
        alpha: float = 4.0,
        shifted_ratio: float = 0.75,
    ):
        ...

    def _shift(self, u):
        return u / (
            1
            + (self.alpha - 1)
            * (1 - u)
        )

    def sample(self, batch, device):
        u = torch.rand(batch, device=device)

        if self.shifted_ratio == 0.0:
            return u

        shifted = self._shift(u)

        if self.shifted_ratio == 1.0:
            return shifted

        use_shifted = (
            torch.rand(batch, device=device)
            < self.shifted_ratio
        )

        return torch.where(
            use_shifted,
            shifted,
            u,
        )
```

保持：

```text
x_t = (1-t) noise + t action
target = action - noise
loss = MSE(pred_v, target_v)
```

不增加：
- consistency
- teacher
- Δt
- endpoint loss
- MeanFlow
- distillation

### Sampler sanity check

`alpha=4`：

`E[t_shift] ≈ 0.283`

75/25 mixture：

`E[t] ≈ 0.337`

训练日志已有 `t_mean`，直接监测即可。

---

## 5. 最终配置目标

```yaml
agent:
  # Observation
  encoder_type: pointnext_tokenizer
  pc_dim: 6
  state_dim: ${action_dim}
  # B3 主实验保持 256；B3b 再独立测试 64
  state_out_dim: 256
  num_points: 1024

  pc_encoder_config:
    num_patches: 128
    token_channels: 128
    stem_channels: 64
    patch_radii: [0.04, 0.08]
    patch_neighbors: [16, 32]

  # Action DiT-X
  hidden_dim: 512
  context_dim: 256
  depth: 8
  num_heads: 8
  num_kv_heads: 4
  ffn_hidden_dim: 896
  timestep_embed_dim: 128
  qk_norm: true
  attn_drop: 0.0

  # Rectified Flow
  denoise_steps: 2
  solver: midpoint
  noise_shift_alpha: 4.0
  noise_shift_ratio: 0.75
```

---

## 6. 不修改的公共代码

原则上不得修改：

```text
dexmani_policy/agents/core/base.py
dexmani_policy/training/trainer.py
dexmani_policy/datasets/base_dataset.py
dexmani_policy/datasets/pc_dataset.py
dexmani_policy/env_runner/*
dexmani_policy/common/normalizer.py
```

vNext 的修改范围应集中为：

```text
dexmani_policy/agents/core/action_flow.py
dexmani_policy/agents/action_decoders/backbone/action_flow_dit.py
dexmani_policy/agents/action_decoders/action_flow_flowmatch.py
dexmani_policy/configs/action_flow.yaml
```

以及 ActionFlow-specific smoke/example checks。

---

## 7. Git commit 建议

```text
af-b1-full-ditx
af-b2-asymmetric-cross-attn
af-b3-state-conditioned-geometry
af-b3b-compact-state-64      # optional efficiency ablation
af-b4-shared-adarms          # optional
af-f1-noise-shift-alpha4
af-f2-mixture-time-sampler
af-opt-native-gqa            # optional engineering
```

一个 commit 只解决一个核心问题。

---

## 8. 实验验收顺序

每阶段：

1. syntax/import
2. ActionFlow synthetic example
3. generic smoke test
4. zero-init
5. cached/uncached parity
6. parameter count
7. bf16 forward/backward
8. compile
9. inference latency
10. peak VRAM
11. 20k closed-loop screening

只有通过 20k screening 才进入 100k。

Architecture 阶段固定：
- 128 patches
- Midpoint-2
- alpha=2

Flow 阶段固定最佳 architecture，只改变 sampler。

---

## 9. 最终研究问题

### Architecture

是否：

`8×完整 DiT-X + 256D asymmetric CA + state-conditioned geometry`

能够同时获得：
- 更深 action temporal reasoning
- 更充分 observation grounding
- 不显著增加 latency / VRAM

### Flow

是否：

`75% high-noise + 25% full-path coverage`

能够将：

`Success@Euler1`

推近：

`Success@Midpoint2`

并降低：

`NFE / solver sensitivity`

而无需 consistency / distillation。

---

## 10. 最终目标模型

```text
ActionFlow-vNext
=
PointNext
+ state-conditioned geometry memory
+ 8-layer full DiT-X
+ asymmetric 256D GQA cross-attention
+ timestep-conditioned AdaRMS
+ standard Rectified Flow
+ 75/25 high-noise mixture training
+ Midpoint-2 default inference
```

设计原则：

**不改变 action 表征；不复杂化 Flow objective；把计算预算用在完整的 action reasoning 与 observation grounding 上。**
