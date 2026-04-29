# Agents 设计文档

## 概述

`dexmani_policy/agents` 实现了 4 种模仿学习策略的统一架构：

- **DPAgent** - RGB Diffusion Policy
- **DP3Agent** - Point Cloud Diffusion Policy  
- **MoEAgent** - Mixture of Experts Diffusion Policy
- **ManiFlowAgent** - Flow Matching with Consistency Distillation

所有 agent 继承自 `BaseAgent`，遵循统一的接口：`compute_loss()` 用于训练，`predict_action()` 用于推理。

---

## 架构设计

### 三层结构

```
BaseAgent
├── obs_encoder: 观测编码器 (点云/RGB/proprio → embedding)
├── action_decoder: 动作解码器 (embedding + noise → action)
│   ├── model: 骨干网络 (UNet1D / DiT-X)
│   └── compute_loss / predict_action
└── configure_optimizer: 优化器配置 (支持 encoder/decoder 独立学习率)
```

### 数据流

**训练**：
```
batch['obs'] → normalize → obs_encoder → cond (B, cond_dim)
batch['action'] → normalize → action_decoder.compute_loss(cond, actions) → loss
```

**推理**：
```
obs_dict → preprocess → obs_encoder → cond
randn(B, horizon, action_dim) → action_decoder.predict_action(cond, noise) → actions
→ unnormalize → control_action
```

---

## 核心模块

### 1. BaseAgent (`agents/core/base.py`)

所有策略的基类，定义统一接口。

#### 关键方法

**`configure_optimizer(lr, weight_decay, obs_lr, obs_weight_decay, betas)`**

为 encoder 和 decoder 配置独立的优化器参数组。

```python
# decoder 参数组（带 weight_decay 分组）
action_groups = self.action_decoder.model.get_optim_groups(weight_decay)
for g in action_groups:
    g['lr'] = lr

# encoder 参数组
obs_lr = obs_lr if obs_lr is not None else lr  # 支持 obs_lr=0.0 冻结 encoder
obs_wd = obs_weight_decay if obs_weight_decay is not None else weight_decay
obs_params = list(self.obs_encoder.parameters())

groups = action_groups + [{'params': obs_params, 'weight_decay': obs_wd, 'lr': obs_lr}]
return torch.optim.AdamW([g for g in groups if g['params']], lr=lr, betas=betas)
```

**设计说明**：
- 全局 `lr` 参数是 PyTorch API 必需的，但所有 param group 都有显式 `lr`，全局 lr 仅作 fallback
- `obs_lr is not None` 检查支持 `obs_lr=0.0` 冻结 encoder（`obs_lr=0` 会被误判为 False）

**`compute_loss(batch)`**

调用 `action_decoder.compute_loss()`，子类可扩展（如 MoEAgent 添加辅助损失）。

**`predict_action(obs_dict, action_template)`**

调用 `action_decoder.predict_action()`，处理归一化/反归一化。

---

### 2. Action Decoders

#### Diffusion (`agents/action_decoders/diffusion.py`)

DDIM 扩散模型，用于 DP/DP3/MoE。

- **训练**：`x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise`，预测 `x_0` 或 `noise`
- **推理**：DDIM 采样，支持 `num_inference_steps < num_train_timesteps` 加速

#### FlowMatchWithConsistency (`agents/action_decoders/flowmatch.py`)

Flow Matching + Consistency Distillation，用于 ManiFlow。

**核心思想**：
- **Flow Matching**：学习从噪声到数据的速度场 `v_t`，使 `x_1 = x_0 + ∫v_t dt`
- **Consistency Distillation**：用 EMA 模型预测未来状态，指导当前模型学习全局趋势

**Batch 切分**（L134-156）：
```python
flow_batchsize = max(1, min(B - 1, int(B * self.flow_batch_ratio)))

# Flow Matching 部分
flow_targets = self.get_flow_velocity(actions[:flow_batchsize])
pred_vt_flow = self.model(flow_targets["xt"], flow_targets["t"], ...)
loss_flow = F.mse_loss(pred_vt_flow, flow_targets["vt_target"])

# Consistency Distillation 部分（需要 EMA 模型）
if ema_model is not None:
    consistency_targets = self.get_consistency_velocity(
        actions[flow_batchsize:], cond[flow_batchsize:], ema_model
    )
    pred_vt_consistency = self.model(consistency_targets["xt"], ...)
    loss_consistency = F.mse_loss(pred_vt_consistency, consistency_targets["vt_target"])
```

**设计说明**：
- `flow_batchsize` 在数据处理前计算，确保 flow 和 consistency 使用不同的数据
- `B < 2` 时返回零损失（保留梯度图）：`zero = actions.sum() * 0.0`
- `ema_model=None` 时 `loss_dict` 包含 `loss_consistency: torch.zeros_like(loss_flow)` 保持键集稳定

**Timestep 采样**（L47-88）：
- `target_t_sample_mode="relative"`：模型输入 `(t, dt)`，预测 `v_t`
- `target_t_sample_mode="absolute"`：模型输入 `(t, t+dt)`，预测 `v_t`
- 所有 `t + dt` 都经过 `torch.clamp(max=1.0)` 限制在 [0,1] 范围内

---

### 3. Backbones

#### ConditionalUnet1D (`agents/action_decoders/backbone/unet1d.py`)

1D U-Net，用于 DP/DP3/MoE。

**条件注入**：
- `condition_type="film"`：全局向量通过 FiLM (scale, bias) 调制特征
- `condition_type="cross_attention_film"`：序列 token 通过 Cross-Attention + FiLM 调制

**优化器分组** (L228-230)：
```python
def get_optim_groups(self, weight_decay):
    return get_optim_group_with_no_decay(self, weight_decay=weight_decay)
```

**设计说明**：
- 使用 `get_optim_group_with_no_decay` 排除 GroupNorm/LayerNorm 参数的 weight_decay
- 旧实现 `get_default_optim_group` 会对所有参数应用 weight_decay，导致归一化层被错误正则化

#### DiTXFlowMatch (`agents/action_decoders/backbone/ditx.py`)

Transformer 骨干，用于 ManiFlow。

**条件注入**：
- Timestep `t` 和 `target_t` 通过 AdaLN-Zero 调制 (scale, shift, gate)
- Context 通过 Cross-Attention 注入

**输入格式**：
```python
x: (B, horizon, action_dim)
timestep: (B,) - 当前时刻 t
target_t: (B,) - 目标时刻 (relative 模式为 dt，absolute 模式为 t+dt)
context: (B, n_tokens, cond_dim) - 观测 embedding
```

---

### 4. Observation Encoders

#### Point Cloud Encoders (`agents/obs_encoder/pointcloud/`)

- **PointNet** - 简单 MLP + max pooling
- **iDP3** - DP3 论文的点云编码器
- **PointPN Tokenizer** - PointNext + Perceiver 生成 token 序列
- **PointNext Tokenizer** - PointNext 直接输出 token 序列

#### RGB Encoders (`agents/obs_encoder/rgb/`)

- **ResNet** - torchvision 预训练模型
- **DINO** - 自监督视觉 Transformer
- **CLIP** - 对比学习视觉-语言模型
- **SigLIP** - Sigmoid Loss CLIP

所有 RGB encoder 使用 `ImageProcessor` 预处理（resize, normalize）。

#### Proprio Encoder (`agents/obs_encoder/proprio/state_mlp.py`)

简单 MLP，编码 joint state。

#### Plugins

**MoE** (`agents/obs_encoder/plugins/moe.py`)

Mixture of Experts 路由器，动态选择专家网络。

**Load Balance Loss** (L73-77)：
```python
def aux_loss(self, probs):
    topk_idx = torch.topk(probs, self.top_k, dim=-1)[1]  # (B, top_k)
    dispatch = topk_idx.reshape(-1)                       # 展平所有 top_k
    f_i = torch.bincount(dispatch, minlength=self.num_experts).to(probs.dtype)
    f_i = f_i / dispatch.numel()                          # 除以 B*top_k
    P_i = probs.mean(dim=0)
    return self.num_experts * torch.sum(f_i * P_i)
```

**设计说明**：
- `topk_idx.reshape(-1)` 展平 (B, top_k) 为 (B×top_k,)，统计所有被选中的专家
- `dispatch.numel() = B × top_k`，正确归一化频率
- 当 `top_k=2` 时，两个专家都会被统计，避免次选专家坍塌

---

### 5. Common Utilities

#### LinearNormalizer (`common/normalizer.py`)

对 joint state 和 action 进行归一化。

**透传行为** (L285-294)：
```python
def _normalize_impl(self, x, forward=True):
    if isinstance(x, dict):
        result = dict()
        for key, value in x.items():
            if key not in self.params_dict:
                result[key] = value  # 未 fit 的 key 原样返回
                continue
            result[key] = self._normalize_impl(value, forward)
        return result
```

**设计说明**：
- 支持多模态输入（如 `{'joint_state': ..., 'point_cloud': ...}`）
- 点云不需要归一化，直接透传
- 只对 `fit()` 过的 key 进行归一化

#### Timestep Sampling (`agents/action_decoders/common/sample.py`)

`SampleLibrary` 提供多种 timestep 采样策略：

- `uniform` - 均匀分布 U(0,1)
- `lognorm` - Logit-Normal 分布（中间时刻密集）
- `beta` - Beta 分布（可控偏向早期/晚期）
- `mode` - 自定义模式采样
- `cosmap` - Cosine mapping
- `discrete` - 离散时间网格 {0, 1/K, ..., (K-1)/K}
- `discrete_pow` - 幂次离散采样（早期密集）

**默认配置**：
- Flow Matching: `t ~ Beta(1.0, 1.5)`（偏向早期）
- Consistency: `t ~ discrete`，`dt ~ uniform`

---

## 关键设计决策

### 1. 为什么 Flow Matching 需要切分 batch？

**原因**：Flow 和 Consistency 的训练目标不同，必须使用不同的数据。

- **Flow Matching**：学习 `v_t = x_1 - x_0`（数据到噪声的速度）
- **Consistency**：学习 `v_t = (pred_x1 - x_t) / (1-t)`（EMA 模型预测的全局趋势）

如果使用相同数据，Consistency 会覆盖 Flow 的梯度，导致训练不稳定。

### 2. 为什么 B<2 时返回零损失？

**原因**：`flow_batchsize = max(1, min(B-1, ...))` 确保至少有 1 个样本用于 consistency。

当 `B=1` 时，`flow_batchsize=0` 会导致空 tensor 错误。返回 `actions.sum() * 0.0` 保留梯度图，避免 DDP 崩溃。

### 3. 为什么 GroupNorm 不应该被 weight_decay？

**原因**：归一化层的参数（scale, bias）不是模型容量的一部分，正则化会干扰归一化效果。

PyTorch 官方建议：只对卷积/线性层的权重应用 weight_decay，排除 bias 和归一化层。

### 4. 为什么 obs_lr 使用 `is not None` 而不是 `if obs_lr`？

**原因**：支持 `obs_lr=0.0` 冻结 encoder。

- `if obs_lr:` 会将 `0.0` 判断为 `False`，错误地使用全局 `lr`
- `if obs_lr is not None:` 正确区分 `0.0`（显式冻结）和 `None`（使用默认值）

### 5. 为什么 target_t 需要 clamp？

**原因**：`t + dt` 可能超过 1.0，超出训练分布 [0,1]。

Flow Matching 假设 `t ∈ [0,1]`，超出范围会导致外推误差。所有 `t + dt` 都应 `clamp(max=1.0)`。

---

## 已知限制

### 死代码（不影响功能）

- `agents/action_decoders/backbone/dit.py` - 整文件未使用（372 行）
- `agents/action_decoders/backbone/ditx.py:397` - `DiTXDiffusion` 类未使用（70 行）
- `agents/obs_encoder/plugins/token_compressor.py` - 四个类未使用（345 行）
- `agents/action_decoders/common/sample.py:7` - `logit_normal_density` 函数未使用

**建议**：保留作为实验性功能，或移入 `experimental/` 目录。

### 简化实现

- `flowmatch.py:78` - `dt2 = dt1.clone()`（注释说明可独立采样，但当前简化为相同）
- `ditx.py:392` - `x[:, -self.horizon:]` 冗余切片（为未来 obs+action 拼接预留）

---

## 测试建议

### 单元测试

```bash
# 测试各 agent 的前向传播
python -m dexmani_policy.agents.core.dp3
python -m dexmani_policy.agents.core.maniflow

# 测试 obs encoder
python -m dexmani_policy.agents.obs_encoder.pointcloud.idp3

# 测试 action decoder
python -m dexmani_policy.agents.action_decoders.flowmatch
```

### 集成测试

```bash
# 单卡训练（快速验证）
python dexmani_policy/train.py --config-name=dp3 num_epochs=2

# DDP 训练（验证梯度同步）
python dexmani_policy/train_ddp.py --config-name=maniflow_ddp num_epochs=2
```

### 关键检查点

1. **Flow/Consistency 数据分离**：检查 `loss_flow` 和 `loss_consistency` 是否都有梯度
2. **小 batch 稳定性**：测试 `batch_size=1` 是否崩溃
3. **MoE 负载均衡**：检查 `load_balance_loss` 是否约束所有 top-k 专家
4. **Encoder 冻结**：设置 `obs_lr=0.0`，检查 encoder 参数是否更新
5. **Timestep 范围**：检查 `target_t` 是否始终在 [0,1] 范围内

---

## 参考

- **Diffusion Policy**: [Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/)
- **DP3**: [Duan et al., CoRL 2024](https://3d-diffusion-policy.github.io/)
- **Flow Matching**: [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)
- **Consistency Models**: [Song et al., ICML 2023](https://arxiv.org/abs/2303.01469)
