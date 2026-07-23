# DQ-RISE 知识体系：从论文到代码的完整梳理

> **Paper**: *Learning Dexterous Manipulation with Quantized Hand State* (ICRA 2026, arXiv:2509.17450)
> **Code**: https://github.com/RISE-Policy/DQ-RISE
> **Authors**: Ying Feng\*, Hongjie Fang\*, Yinong He\* et al. (MVIG, SJTU)
> **基于**: RISE (IROS 2024) — 3D Perception Makes Real-World Robot Imitation Simple and Effective
> **本项目实现**: `dexmani_policy/agents/core/dqrise.py` DQRISEAgent

---

## 目录

1. [问题定义与动机](#1-问题定义与动机)
2. [架构全景图](#2-架构全景图)
3. [VQ-VAE 手部量化系统](#3-vq-vae-手部量化系统)
4. [扩散策略动作解码器](#4-扩散策略动作解码器)
5. [训练管道](#5-训练管道)
6. [推理管道](#6-推理管道)
7. [评估体系](#7-评估体系)
8. [策略变体对比](#8-策略变体对比)
9. [代码模块详解](#9-代码模块详解)
10. [数据管道](#10-数据管道)
11. [关键设计决策](#11-关键设计决策)
12. [论文-代码交叉验证](#12-论文-代码交叉验证)
13. [与本项目实现的完整对比](#13-与本项目实现的完整对比)
14. [已知局限与改进方向](#14-已知局限与改进方向)

---

## 1. 问题定义与动机

### 1.1 核心问题

灵巧操作中的**臂-手动作空间不平衡**：

```
完整动作空间: [手臂TCP(9维) + 手部关节(6维)] = 15维
                ↑                    ↑
           负责到达目标         负责抓取/操作
```

- **手臂**（3平移 + 6D旋转 = 9维）：低频、大范围、决定任务空间可达性
- **手部**（6个手指关节）：高频、精细、变化模式多

**直接拼接的问题**：高维手部动作在 MSE 损失中占主导，手部"噪声"淹没了手臂的关键信号，导致手臂定位精度被牺牲。

### 1.2 DQ-RISE 的核心洞察

> **不是解耦臂和手，而是将手部离散化 + 连续松弛，与手臂在统一扩散过程中联合建模。**

三步走：
1. **VQ-VAE 量化**：6DOF 连续手部姿态 → 2组码本 × 4码字 = 16种离散组合
2. **PCA 重排序 + 连续松弛**：16种手部姿态按 PCA 第一主成分排序 → 索引具有连续语义 → 归一化到[-1,1]作为连续值
3. **联合扩散**：`[TCP(9) + VQ索引(1)]` 作为10维联合动作，在 DDIM 条件 UNet 中端到端去噪

### 1.3 为什么这个方案有效？

**梯度流一致性 (Gradient Flow Consistency)** 是 DQ-RISE 成功的根本原因：

| 方法 | 臂梯度 | 手梯度 | 梯度流一致性 | 结果 |
|------|--------|--------|-------------|------|
| RISE (原始15维) | 统一扩散MSE | 统一扩散MSE | ✅ 一致 | 手部主导，55% |
| RISE-S (双扩散头) | 独立扩散MSE | 独立扩散MSE | ❌ 独立 | 协调性丧失，62% |
| DQ-RISE-C (分类头) | 扩散MSE | 交叉熵 | ❌ **不一致** | **2.5%崩溃** |
| **DQ-RISE** | 联合扩散MSE | 联合扩散MSE | ✅ **一致** | **85.83%** |

DQ-RISE-C 的惨败验证了关键理论：手部分类头（交叉熵 loss）与手臂扩散头（MSE loss）的梯度流不一致，导致训练不稳定和 rollout 崩塌。

---

## 2. 架构全景图

### 2.1 端到端数据流（官方实现）

```
┌─────────────────────────────────────────────────────────────────┐
│                    2× RealSense RGB-D (720×1280)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  RGB-D → 点云生成        │
              │  - 双相机→Base坐标系     │
              │  - 工作空间裁剪          │
              │  - Voxel降采样(0.005m)  │
              │  - ImageNet颜色归一化    │
              └────────────┬────────────┘
                           │ [N, 6] = (x,y,z,r,g,b) 稀疏张量
              ┌────────────▼────────────┐
              │  MinkowskiEngine        │
              │  Sparse ResNet14        │
              │  in=6, out=512          │
              │  + SparsePosEncoding    │
              └────────────┬────────────┘
                           │ [B, 100, 512] tokens + pos_emb + mask
              ┌────────────▼────────────┐
              │  DETR Transformer       │
              │  Encoder: 4层, Decoder: 1层 │
              │  d=512, h=8, ff=2048    │
              │  1× learnable readout   │
              └────────────┬────────────┘
                           │ [B, 512] readout token
              ┌────────────▼────────────┐
              │  ConditionalUnet1D      │
              │  FiLM conditioning      │
              │  DDIM (100 train, 20 inf)│
              │  down=[256,512], k=5    │
              └────────────┬────────────┘
                           │ [B, 20, 10] = [TCP(9)+VQ_idx(1)]
              ┌────────────▼────────────┐
              │  后处理                  │
              │  - TCP反归一化           │
              │  - VQ索引→码本查表→6DOF │
              │  - 时间集成(Ensemble)    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Flexiv Rizon 4 +       │
              │  OyMotion RoHand        │
              └─────────────────────────┘
```

### 2.2 核心模块

| 模块 | 官方文件 | 本项目文件 | 输入 | 输出 |
|------|----------|-----------|------|------|
| 3D编码器 | `policy/tokenizer.py` | `agents/core/dp3.py` DP3ObsEncoder | ME.SparseTensor / PC(1024,6) | (B,N,C) tokens |
| Transformer | `policy/transformer.py` | _(无，直接用编码器输出)_ | tokens+pos | (B,512) readout |
| 扩散解码器 | `policy/diffusion.py` | `agents/action_decoders/diffusion.py` | cond (B,cond_dim) | (B,horizon,action_dim) |
| UNet骨干 | `diffusion_modules/conditional_unet1d.py` | `action_decoders/backbone/unet1d.py` | sample+cond | noise pred |
| VQ-VAE | `policy/vqvae_rise/vqvae.py` | `agents/vq_hand/vqvae.py` | (B,hand_dim) | loss, codes |
| CodebookManager | `eval_vqvae.py` (脚本) | `agents/vq_hand/codebook_manager.py` | VQ-VAE模型 | 排序码本 |

---

## 3. VQ-VAE 手部量化系统

### 3.1 架构细节

```
EncoderMLP                      ResidualVQ                    EncoderMLP
(编码器)                         (量化层)                      (解码器，同结构)

hand_pose [B,hand_dim]          latent [B,256]               recon [B,hand_dim]
      │                              │                          ↑
      ▼                              ▼                          │
 flatten                            │                          │
      │                     project_in → [B,1,256]             │
      ▼                              │                          │
 Linear(→hidden_dim)        ┌────────▼────────┐               │
 ReLU                       │ VQ Layer 0      │               │
 [重复 num_layers 次]        │ codebook[4,256] │               │
 Linear(→hidden_dim)        │ quantize →残差→  │               │
 ReLU                       └────────┬────────┘               │
      │                               │                         │
      ▼                       ┌──────▼──────────┐              │
 Linear(→latent_dim)          │ VQ Layer 1      │              │
                              │ codebook[4,256] │              │
 latent [B,256]               │ quantize →残差→  │              │
                              └────────┬────────┘              │
                                       │                        │
                              weighted sum (softmax)            │
                                       │                        │
                              project_out → [B,1,256]          │
                                       │                        │
                                       └────────────────────────┘
                                                  │
                                            EncoderMLP 解码
                                            (latent_dim→hidden_dim→hand_dim)
```

### 3.2 关键配置

| 参数 | 官方值 | 本项目值 | 位置 |
|------|--------|---------|------|
| `hand_dim` | 6 (ROHand) | 12 (XHand) | 取决于灵巧手 |
| `latent_dim` | 256 | 256 | 量化潜在空间 |
| `hidden_dim` | 512 | 512 | MLP隐藏层 |
| `num_groups` (码本组数) | 2 | 2 | ResidualVQ层数 |
| `codebook_size` (每组码字数) | 4 | 4 | 每层码字数量 |
| 总组合数 | 4² = **16** | 4² = **16** | `codebook_size ^ num_groups` |
| `num_layers` (MLP层数) | 1+1=2 (官方实际) | 3 (等价于官方vae_layer_num=5时) | 隐藏层数(不含首尾Linear) |
| `act_scale` | 1.0 | 1.0 | 数据已在[-1,1] |

### 3.3 ResidualVQ 工作流程

**文件**: 官方 `policy/vqvae_rise/vector_quantize_pytorch/residual_vq.py`
本项目 `dexmani_policy/agents/vq_hand/residual_vq.py`

```
输入 x [B, 1, latent_dim]

Layer 0:
  quantized_0 = VQ(x)          # 找到最近码字: argmin ||x - codebook[0,:,:]||
  residual = x - quantized_0   # 计算残差 (detach阻断梯度)

Layer 1:
  quantized_1 = VQ(residual)   # 对残差再量化

输出 = softmax([0.5, 0.5])[0] * quantized_0 + softmax([0.5, 0.5])[1] * quantized_1
```

**关键实现** (`residual_vq.py:48`):
- `layer_weights` 是**可学习参数**，初始化为 `[0.5, 0.5]`
- 通过 `F.softmax(layer_weights, dim=0)` 归一化为权重
- 每层的 `quantized` 通过 `residual = residual - quantized.detach()` 传播（`detach` 阻止梯度流过残差链，符合 VQ-VAE 的 stop-gradient 设计）

### 3.4 EuclideanCodebook EMA 更新机制

**文件**: 官方 `policy/vqvae_rise/vector_quantize_pytorch/vector_quantize_pytorch.py:436-510`

VQ 码本使用 **EMA (Exponential Moving Average)** 更新，而非梯度下降：

```python
# 训练时 (self.training=True, ema_update=True):
cluster_size = embed_onehot.sum(dim=1)             # 每个码字被使用的次数
ema_inplace(self.cluster_size, cluster_size, decay=0.8)

embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
ema_inplace(self.embed_avg, embed_sum, decay=0.8)

# Laplace 平滑防止死码字（除零保护）
cluster_size = laplace_smoothing(cluster_size) * cluster_size.sum()
# laplace_smoothing: (x + eps) / (denom + n_categories * eps), eps=1e-5

embed_normalized = embed_avg / cluster_size
```

**死码字复活机制** (`vector_quantize_pytorch.py:424-434`):
- `threshold_ema_dead_code=2`：如果某码字的 `cluster_size < 2`，标记为死亡
- 从当前 batch 随机采样替换死亡码字
- 本项目配置默认使用 `kmeans_init=True` 降低码本崩溃风险

### 3.5 损失函数

**官方**: `train_vqvae.py:158`:
```python
total_loss = Encoder_Loss * encoder_loss_multiplier * 3 + VQ_loss_state * 5
```

**本项目**: `configs/dqrise.yaml:236-237`:
```yaml
enc_loss_weight: 3.0         # L1 reconstruction multiplier
vq_loss_weight: 5.0          # VQ commitment multiplier
```

| 损失项 | 计算方式 | 官方权重 | 本项目权重 | 含义 |
|--------|----------|----------|-----------|------|
| `Encoder_Loss` (加权L1) | `(hand_pose - decoded).abs() × loss_weight` 的均值 | ×3 | ×3 | 手部姿态重建保真度 |
| `VQ_loss_state` (承诺损失) | 编码器输出与量化后向量的 MSE，由 VQ 层内部计算 | ×5 | ×5 | 编码器承诺到码字 |
| `Recon_Loss` (MSE) | `MSELoss(hand_pose, decoded)` | 仅日志 | 仅日志 | 辅助监控指标 |

**手指维度权重**（官方 6DOF ROHand）:
```python
loss_weight = [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
# 拇指三关节(0-2): 1.0  # 食指(3): 1.0   # 中指(4): 0.5
# 无名指(5): 0.5         # 小指/拇指旋转: 1.0
```

**手指维度权重**（本项目 12DOF XHand）:
```yaml
loss_weight: [1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0,  0.5, 0.5,  0.5, 0.5]
# thumb×3: 1.0  # index×3: 1.0  # middle×2: 1.0  # ring×2: 0.5  # pinky×2: 0.5
```

**与论文损失函数的对应关系**：

论文公式: `L = ||s^(h) - ŝ^(h)||₂² + β||sg[z_e] - z_q||₂² + γ||z_e - sg[z_q]||₂²`

代码对应：
- `||s^(h) - ŝ^(h)||₂²` → `Encoder_Loss`（官方用加权L1，本项目同）
- `β||sg[z_e] - z_q||₂²` → VQ 层内部 commitment loss（`commitment_weight=1.0`）
- `γ||z_e - sg[z_q]||₂²` → EMA 更新替代了梯度（码本通过 EMA 而非梯度更新）

### 3.6 码本提取与 PCA 排序

**官方**: `eval_vqvae.py` (脚本式，硬编码权重 0.5)

```python
# Step 1: 枚举所有码字组合
for i in range(4):
    for j in range(4):
        latent = codebooks[0,i,:] * 0.5 + codebooks[1,j,:] * 0.5  # 硬编码！
        action = decoder(latent)   # → 6DOF

# Step 2: PCA 排序
pca = PCA(n_components=1)
proj_1d = pca.fit_transform(actions.reshape(16, 6))
sorted_index = np.argsort(proj_1d[:, 0])
```

**本项目**: `dexmani_policy/agents/vq_hand/codebook_manager.py:247-345` (CodebookManager.reindex_by_pca)

关键改进：
- 使用**学到的 `layer_weights`**（`softmax(layer_weights)`）而非硬编码 0.5
- 内置解码器输出范围验证（检测超出[-1,1]的异常值）
- 保存 PCA 解释方差比等诊断信息
- 支持 `.npz` 格式，包含 hand normalizer 元数据以确保坐标一致性

**为什么 PCA 排序至关重要？**
- 排序后相邻索引的手部姿态在 PCA 空间中接近 → 形成"平滑遍历"
- 索引归一化到 [-1,1] 区间时才有连续语义意义
- 扩散模型可以在连续索引空间中平滑插值
- 论文消融实验：去除 PCA 排序导致"显著的性能下降"

**关键注意事项**：PCA 必须在**原始手部姿态空间**（非 VQ 潜在空间）上进行。论文消融验证了在 256 维潜在特征上做 PCA 不能保证有意义的排序。

---

## 4. 扩散策略动作解码器

### 4.1 DiffusionUNetPolicy（官方）

**文件**: `policy/diffusion.py`

```python
class DiffusionUNetPolicy(nn.Module):
    def __init__(self, action_dim=10, horizon=20, n_obs_steps=1, obs_feature_dim=512):
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim * n_obs_steps,  # 512
            down_dims=(256, 512), kernel_size=5, n_groups=8,
            cond_predict_scale=True     # FiLM modulation
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100, beta_start=1e-4, beta_end=0.02,
            beta_schedule="squaredcos_cap_v2", prediction_type="epsilon"
        )
        self.num_inference_steps = 20
```

### 4.2 ConditionalUnet1D 架构

**文件**: `policy/diffusion_modules/conditional_unet1d.py`

```
输入: sample [B, T, input_dim] + timestep + global_cond

1. 时间步编码:
   SinusoidalPosEmb(256) → Linear(256→1024) → Mish → Linear(1024→256)
   → concat(时间嵌入, global_cond) → [B, 768]

2. 下采样路径 (2层，down_dims=[256,512]):
   [B,input_dim,T] → Conv1dBlock → FiLM ResBlock ×2 → Downsample(×2)
   input_dim→256 → 256→512 → [B,512,T/4]

3. 中间瓶颈 (2个FiLM ResBlock):
   [B,512,T/4] → FiLM ResBlock ×2 → [B,512,T/4]

4. 上采样路径 (2层):
   concat(skip) → FiLM ResBlock ×2 → Upsample(×2)
   1024→256 → 256→input_dim → [B,input_dim,T]

5. 输出: Conv1dBlock → Conv1d(1×1) → [B,input_dim,T]
```

**FiLM 条件注入** (`conditional_unet1d.py:56-62`):
```python
# 条件编码器将 cond [B, cond_dim] 映射为 per-channel scale 和 bias
embed = Linear(cond_dim → out_channels*2)  # cond_predict_scale=True
scale = embed[:, 0, ...]   # per-channel scale
bias  = embed[:, 1, ...]   # per-channel bias
out = scale * out + bias   # FiLM modulation
```

### 4.3 训练时的 DDPM 前向

```python
def compute_loss(self, readout, actions):
    global_cond = readout.reshape(B, -1)          # [B, 512]
    noise = torch.randn_like(actions)
    timesteps = torch.randint(0, 100, (B,))
    noisy_actions = scheduler.add_noise(actions, noise, timesteps)
    pred_noise = UNet(noisy_actions, timesteps, global_cond=global_cond)
    loss = MSE(pred_noise, noise)
    return loss
```

### 4.4 推理时的 DDIM 采样

```python
def predict_action(self, readout):
    trajectory = torch.randn(B, 20, action_dim)    # 从纯噪声开始
    for t in scheduler.timesteps:                  # 100→0, 步长=5 (20步)
        pred_noise = UNet(trajectory, t, global_cond=global_cond)
        trajectory = scheduler.step(pred_noise, t, trajectory).prev_sample
    return trajectory  # [B, 20, action_dim]
```

### 4.5 DQ-RISE 的特殊动作处理（核心创新）

**文件**: `train_dqrise.py:157-163`

```python
# 1. 从完整动作中分离手部姿态
handpose = action_data[..., TCP_DIM:TCP_DIM+HAND_DIM]  # [B, 20, 6]

# 2. 找到最近码字（连续松弛的关键步骤）
code_book_actions = torch.from_numpy(sorted_hand_actions)  # [16, 6]
distances = torch.cdist(handpose, code_book_actions)       # [B, 20, 16]
indices = distances.argmin(dim=-1).float()                 # [B, 20] ∈ {0..15}

# 3. 索引归一化到 [-1, 1]（使扩散模型能平滑操作）
indices = indices / (TCP_DIM + HAND_DIM) * 2 - 1           # [B, 20] ∈ [-1, 1]

# 4. 拼接为联合动作
processed_action = torch.cat([
    action_data[..., :TCP_DIM],    # TCP [B, 20, 9]
    indices.unsqueeze(-1)          # VQ索引 [B, 20, 1]
], dim=-1)                         # → [B, 20, 10]
```

**索引归一化公式**: `index / (TCP_DIM + HAND_DIM) * 2 - 1`
- 16 个离散索引 0..15 → [-1, 1]
- 步长 = 2/15 ≈ 0.133
- 使离散索引"看起来像"连续值，扩散模型可以对其进行平滑去噪

### 4.6 本项目的 Diffusion 实现

**文件**: `dexmani_policy/agents/action_decoders/diffusion.py`

与官方的关键差异：
- 支持 `prediction_type="sample"` (预测 x0) 和 `"epsilon"` (预测噪声) 两种模式
- DQ-RISE 配置使用 `prediction_type: epsilon`
- `ddpm_training_steps=100`, `ddim_inference_steps=20`（与官方一致）
- UNet 骨干的 `context_dim` = `obs_encoder.out_dim * n_obs_steps`（如 128*2=256）

---

## 5. 训练管道

### 5.1 三阶段流程

```
Stage 1: VQ-VAE 预训练        Stage 2: 码本提取+排序       Stage 3: DQ-RISE 联合训练
───────────────────────       ──────────────────────      ─────────────────────────
输入: 手部姿态 [T, hand_dim]   输入: 训练好的VQ-VAE         输入: 点云 + 完整动作序列
                             枚举所有码字组合              动作 = [TCP(tcp_dim) + hand(hand_dim)]
编码: hand → latent(256)      解码: code_size^groups种姿势  手部 → VQ码本索引 (连续松弛)
量化: 2×4码本 ResidualVQ     PCA降维到1D → 排序           扩散头输入 = [TCP + VQ_index]
解码: latent → hand           保存排序码本 + 元数据         输出: diffusion_action_dim = tcp_dim+1

损失: L1_recon*3 + VQ_commit*5 (无需训练)                 损失: DDIM noise/sample MSE
优化器: AdamW(lr=3e-4)                                    优化器: AdamW(lr=3e-4)
Epochs: 1500                                               Epochs: 1000
```

### 5.2 VQ-VAE 训练详情

**官方入口**: `train_vqvae.py`（DDP torchrun）
**本项目入口**: `dexmani_policy/scripts/train_vq_hand.py`（单卡，读取 `dqrise.yaml` 的 `vq_vae` 段）

| 超参数 | 官方值 | 本项目值 |
|--------|--------|---------|
| 优化器 | AdamW, lr=3e-4, betas=[0.95,0.999], wd=1e-6 | 同 |
| LR 调度 | Cosine + warmup 150 steps | 同 |
| Batch size | 256 (总计) | 256 |
| Epochs | 默认1000/论文1500 | 1500 |
| 损失权重 | L1×3 + VQ×5 | `enc_loss_weight: 3.0, vq_loss_weight: 5.0` |
| 码本使用监控 | 每200 epoch输出柱状图 | 每 `codebook_report_epochs: 200` |

**码本使用监控**：诊断码本崩溃（codebook collapse）——如果某些码字从未被使用，说明码本未充分利用。

### 5.3 DQ-RISE 训练详情

**官方入口**: `train_dqrise.py`（DDP torchrun）
**本项目入口**: `train.py`（单卡 Hydra）+ `train_ddp.py`（多卡 DDP）

| 超参数 | 官方值 | 本项目值 |
|--------|--------|---------|
| 优化器 | AdamW, lr=3e-4, wd=1e-6 | AdamW, lr=3e-4, wd=1e-6, betas=[0.95,0.999] |
| LR 调度 | Cosine + warmup 2000 steps | Cosine + warmup 2000 steps |
| Batch size | 240 (总计) | 128 |
| Epochs | 1000 | 1000 |
| 动作维度 | 10 (9 TCP + 1 VQ索引) | tcp_dim + 1 (8 或 10) |
| 预测 horizon | 20 | 16 |

**官方训练循环关键步骤** (`train_dqrise.py:144-173`):
```python
for data in dataloader:
    # 1. 稀疏点云批处理 + 数据增强
    cloud_coords, cloud_feats, action_data = ...
    cloud_data = ME.SparseTensor(feats, coords)

    # 2. 手部姿态 → VQ 码本索引（连续松弛）
    distances = cdist(handpose, sorted_hand_actions)
    indices = argmin(distances) / 15 * 2 - 1

    # 3. 拼接联合动作 → 扩散损失
    processed_action = cat([tcp, indices], dim=-1)
    loss = policy(cloud_data, processed_action)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
```

**DDP 配置**（官方）:
- `find_unused_parameters=True`：因排序码本是普通 tensor（非 nn.Parameter），不产生梯度
- `NCCL_P2P_DISABLE=1`：单卡训练兼容

### 5.4 数据增强

**训练时在线增强** (在 Dataset 层执行，normalize 前):

1. **3D 空间增强** (`aug=True`):
   - 随机平移: [-0.2, 0.2]m（各轴独立）
   - 随机旋转: [-30°, 30°]（各轴独立）
   - 以点云中心为原点旋转

2. **颜色增强** (`aug_jitter=True`):
   - 官方：HSV 空间增强（非 RGB ColorJitter）
   - 本项目：`color` augmentation (brightness/contrast/saturation/hue)
   - 在 ImageNet 归一化之前执行

3. **ImageNet 颜色归一化**（官方）:
   - `(color - [0.485,0.456,0.406]) / [0.229,0.224,0.225]`

---

## 6. 推理管道

### 6.1 双相机真机推理（官方）

**文件**: `eval_rise_vae_2cam.py`

```
每 num_inference_step=20 控制步执行一次策略推理:

1. 采集 RGB-D (双相机)
2. 点云融合:
   - 各相机 RGB-D → Open3D PointCloud
   - 相机2 → Base坐标系 → 相机1坐标系
   - 合并 → 工作空间裁剪(SAFE_WORKSPACE) → Voxel降采样(0.005m)
3. 策略推理:
   colors → ImageNet norm → MinkowskiEngine → ME.SparseTensor
   → Sparse3DEncoder → Transformer → DiffusionUNetPolicy
   → [20, 10] = 20步 × [TCP(9) + VQ_index(1)]
4. 后处理:
   TCP: translation [-1,1]→[TRANS_MIN,TRANS_MAX]; 6D rotation→quaternion
   Hand: VQ_index → round→clamp→sorted_hand_actions[index] → 6DOF [0,65535]
5. 时间集成 (EnsembleBuffer)
6. 发送机器人指令:
   手臂: send_tcp_pose (可选 discretize_rotation: 大旋转→π/16小步)
   手部: set_finger_pos (仅当变化 > ROHAND_THRESHOLD 时发送)
```

### 6.2 时间集成模式

**文件**: `utils/ensemble.py`

| 模式 | 策略 | 适用场景 |
|------|------|----------|
| `new` | 使用最新预测 `action[t] = pred[-1]` | 快速响应 |
| `old` | 使用最早预测 `action[t] = pred[0]` | 平滑但滞后 |
| `avg` | 等权平均 `mean(preds)` | 平衡 |
| `act` | ACT风格指数衰减 `w_i = exp(-k*i*Δt)`, k=0.01 | 历史优先 |
| `hato` | HATO风格幂衰减 `w_i = τ^(i*Δt)`, τ=0.5 | 快速遗忘 |

### 6.3 VQ 索引解码

**训练时**（官方 `train_dqrise.py`）:
```python
# 连续手部姿态 → 最近码字 → 归一化
distances = cdist(handpose, codebook)    # [B, T, 16]
index = argmin(distances) / 15 * 2 - 1   # → [-1, 1]
```

**推理时**（官方 `eval_rise_vae.py:204-207`）:
```python
# VQ索引 → 码本查表
codebook = np.load("sorted_actions_task_0001.npy")  # [16, hand_dim]
vq_index = action[..., -1]          # 扩散模型输出的连续值
# 反归一化 + 最近整数取整
discrete_index = round((vq_index + 1) / 2 * (num_codes - 1))
discrete_index = clip(discrete_index, 0, num_codes - 1)
hand_pose = codebook[discrete_index]
```

**本项目** (`CodebookManager.continuous_index_to_hand_pose`):
```python
# half-up 取整（而非 Python round），确保边界行为一致
scaled = (continuous_index.clamp(-1, 1) + 1) * 0.5 * (num_codes - 1)
discrete_idx = torch.floor(scaled + 0.5).long().clamp(0, num_codes - 1)
```

---

## 7. 评估体系

### 7.1 VQ-VAE 提取评估 (`eval_vqvae.py` / `extract_codebook.py`)

**目的**: 加载训练好的 VQ-VAE → 提取码本 → PCA 排序 → 保存

**本项目额外功能**:
- 内嵌 hand normalizer 到 `.npz` 以确保训练/推理坐标一致
- SHA256 校验源 checkpoint
- 解码器输出范围诊断（检测异常值）
- 支持 per-group codebook 构建（实验性）

### 7.2 策略评估

**真实机器人**（官方 `eval_rise_vae_2cam.py`）:
- `num_inference_step`: 20, `max_steps`: 300, `num_action`: 20
- 安全：工作空间裁剪 + 旋转离散化 + 手部阈值 + 力限制

**仿真评估**（本项目 `env_runner/sim_runner.py`）:
- `default_eval_episodes: 25`
- 支持 `texture_random/instance_random/table_random` 域随机化
- 自动计算 success_rate 和 avg_steps

### 7.3 论文实验结果

**6 个真实世界任务**（每任务 50 个演示）:

| 任务 | 类别 | DQ-RISE | RISE | RISE-S | DQ-RISE-C |
|------|------|---------|------|--------|-----------|
| Pull Tissue | 抓取-放置 | **95%/85%** | 75%/45% | 75%/55% | 15%/10% |
| Open Jar | 关节物体 | **95%/90%** | 80%/55% | 60%/45% | 0%/0% |
| Collect Toy | 抓取-放置 | **95%/80%** | 60%/60% | 75%/70% | 0%/0% |
| Pour Rice | 大旋转 | **100%/100%** | 90%/80% | 95%/85% | 0%/0% |
| Open Oven | 关节物体 | **100%/100%** | 100%/90% | 95%/95% | 20%/5% |
| Toast Bread | 长时序 | **100%/65%/60%** | 80%/20%/0% | 75%/25%/20% | 0%/0%/0% |
| **平均** | — | **85.83%** | 55.00% | 61.67% | 2.50% |

---

## 8. 策略变体对比

代码库实现了论文的完整消融实验：

### 8.1 RISE — 原始基线

**文件**: `policy/policy.py`

```python
class RISE(nn.Module):
    # 动作空间: 15维 (9 TCP + 6 hand)
    # 统一扩散头: 一个 ConditionalUnet1D 预测 15 维
    def forward(self, cloud, actions):
        loss = self.action_decoder.compute_loss(readout, actions)  # 单一 MSE
```

### 8.2 RISE-S — 独立双扩散头

**文件**: `policy/separate_diff_baseline_policy.py`

```python
class separate_diff_baseline_policy(nn.Module):
    # tcp_decoder: 9维扩散头 | handPose_decoder: 6维扩散头
    # 两个独立的 DiffusionUNetPolicy，各自预测各自的动作
    def forward(self, cloud, actions):
        loss_tcp = self.tcp_decoder.compute_loss(readout, tcps)
        loss_handPose = self.handPose_decoder.compute_loss(readout, handPoses)
        loss = (loss_tcp + loss_handPose) / 2
```

### 8.3 DQ-RISE-C — 分类解耦（几乎完全失败的基线）

**文件**: `policy/seperate_diff_vae_baseline_policy.py`

```python
class seperate_diff_vae_baseline(nn.Module):
    # TCP: 扩散头 (9维) + Hand: 分类头 (16类), 仅从readout预测
    # key failure: 无TCP条件, 梯度流不一致
    def forward(self, cloud, actions):
        loss_tcp = self.tcp_decoder.compute_loss(readout, tcps)
        logits = self.handPose_head(self.handPose_tfmr(readout_seq))
        loss_handPose = cross_entropy(logits, handPoses)
        loss = loss_tcp * 0.5 + loss_handPose * 0.5
```

### 8.4 TCP-Conditioned — 带TCP条件的分类

**文件**: `policy/tcp_cond_handPose_policy.py`

```python
class tcp_cond_handPose_policy(nn.Module):
    # TCP: 扩散头 (9维)
    # Hand: 从 readout + TCP特征 条件分类 (16类)
    # 改进: TCP MLP 编码 → fuse → 4层 Transformer 分类
    def forward(self, cloud, actions):
        loss_tcp = self.tcp_decoder.compute_loss(readout, tcps)
        tcp_feat = self.tcp_mlp(tcps)
        seq_in = self.fuse(cat([tcp_feat, readout_seq]))
        seq_out = self.handPose_tfmr(seq_in)
        logits = self.handPose_head(seq_out)
        loss_handPose = cross_entropy(logits, handPoses)
        loss = loss_tcp * 0.5 + loss_handPose * 0.5
```

### 8.5 DQ-RISE — 联合扩散（论文方法）

```python
# 与 RISE 共用 policy.py 的同一个类
# 差异仅在训练脚本中的动作预处理:
#   RISE:       action_dim=15, 直接用完整动作
#   DQ-RISE:    action_dim=10, 手部转为连续VQ索引后拼接
handpose = action_data[..., TCP_DIM:TCP_DIM+HAND_DIM]
distances = cdist(handpose, codebook)
indices = argmin(distances) / (TCP_DIM+HAND_DIM) * 2 - 1
processed_action = cat([action_data[...,:TCP_DIM], indices], dim=-1)
```

### 对比总结

| 策略 | 手部表示 | TCP预测 | 手部预测 | 臂-手联合 | 成功率 |
|------|----------|---------|----------|-----------|--------|
| RISE | 连续6DOF | 统一扩散(15d) | 统一扩散(15d) | ✅隐式 | 55% |
| RISE-S | 连续6DOF | 独立扩散(9d) | 独立扩散(6d) | ❌ | 62% |
| DQ-RISE-C | 离散16类 | 扩散(9d) | 分类头(obs only) | ❌梯度流不一致 | 2.5% |
| TCP-Cond. | 离散16类 | 扩散(9d) | 分类头(obs+TCP) | 🟡弱条件 | — |
| **DQ-RISE** | **离散索引(松弛)** | **联合扩散(10d)** | **联合扩散(10d)** | **✅显式** | **85.83%** |

---

## 9. 代码模块详解

### 9.1 官方文件组织

```
DQ-RISE/
├── train_dqrise.py                    # DQ-RISE 训练入口 (DDP torchrun)
├── train_vqvae.py                     # VQ-VAE 训练入口 (DDP torchrun)
├── eval_vqvae.py                      # 码本提取 + PCA 排序 (DDP)
├── eval_rise_vae_2cam.py             # 双相机真机评估
├── eval_rise_vae.py                   # 单相机真机评估
├── eval_agent_2cam.py                # 双相机评估 Agent (Flexiv+RoHand+RealSense)
├── preprocess_data.py                 # 数据预处理 (RGB-D→点云, 保存.h5+.npy)
├── data_filter.py                     # 数据过滤 (静止帧+双相机时间对齐+pairs.pth)
├── process_pointcloud.py              # RGB-D→点云转换工具函数
│
├── policy/                            # 核心策略模块
│   ├── policy.py                      # RISE 主策略 (也用于 DQ-RISE 模式)
│   ├── tcp_cond_handPose_policy.py    # TCP条件手部分类 (baseline)
│   ├── seperate_diff_vae_baseline_policy.py   # 扩散+VAE分类解耦基线
│   ├── separate_diff_baseline_policy.py       # 双独立扩散头基线
│   ├── diffusion.py                   # DiffusionUNetPolicy (DDIM+ConditionalUNet1D)
│   ├── transformer.py                 # DETR Transformer (4enc+1dec, return_intermediate)
│   ├── tokenizer.py                   # Sparse3DEncoder + SparsePositionalEncoding
│   ├── diffusion_modules/             # 扩散模型组件
│   │   ├── conditional_unet1d.py      # 1D UNet + FiLM条件注入
│   │   ├── conv1d_components.py       # Conv1dBlock, Downsample1d, Upsample1d
│   │   ├── positional_embedding.py    # SinusoidalPosEmb (时间步编码)
│   │   └── mask_generator.py          # 扩散条件掩码生成 (LowdimMaskGenerator)
│   ├── minkowski/                     # MinkowskiEngine 稀疏3D ResNet
│   │   ├── resnet.py                  # ResNet14/18/34/50/101 (BasicBlock/Bottleneck)
│   │   ├── resnet_block.py            # 稀疏3D BasicBlock / Bottleneck
│   │   └── common.py                  # MinkowskiEngine conv/norm/pool 包装
│   └── vqvae_rise/                    # VQ-VAE 子模块
│       ├── vqvae.py                   # VqVae 主模型 (EncoderMLP + ResidualVQ + 内置optimizer)
│       ├── vqvae_utils.py             # 正交初始化 (orthogonal_) + tensor转换
│       ├── pretrain_vqvae.py          # Hydra 预训练脚本 (备用，与train_vqvae.py功能重叠)
│       ├── vector_quantize_pytorch/   # 向量量化库 (来自 lucidrains)
│       │   ├── vector_quantize_pytorch.py  # VectorQuantize (EMA Euclidean/CosineSim)
│       │   └── residual_vq.py         # ResidualVQ + GroupedResidualVQ
│       └── dataset/rohand.py          # VQ-VAE 数据集骨架 (不完整)
│
├── dataset/                           # 数据集模块
│   ├── riseVAE_2cam.py               # 双相机训练数据集 (主力)
│   ├── riseVAE.py                     # 单相机数据集 (备用)
│   ├── pretrain.py                    # VQ-VAE 预训练数据集 (HDF5格式)
│   ├── projector.py                   # 坐标变换 Projector (相机↔Base↔TCP)
│   ├── constants.py                   # 相机内参 INTRINSICS, INHAND_CAM_TCP 等
│   └── cleandata.py                   # (空文件)
│
├── utils/                             # 工具模块
│   ├── constants.py                   # TCP_DIM=9, HAND_DIM=6, TRANS_MIN/MAX, 工作空间
│   ├── training.py                    # set_seed, plot_history, sync_loss (DDP barrier)
│   ├── ensemble.py                    # EnsembleBuffer: new/old/avg/act/hato 时间集成
│   ├── transformation.py              # 旋转/位姿变换 (quaternion↔6d↔matrix 等)
│   └── rotation_utils.py              # 9D/10D 旋转表示工具
│
├── device/                            # 机器人硬件控制
│   ├── robot/flexiv.py               # Flexiv Rizon 4 (Flexiv RDK v0.9)
│   ├── camera/realsense.py           # Intel RealSense D415/D435 (pyrealsense2)
│   └── OyMotion/                      # OyMotion RoHand 灵巧手
│       ├── ROHand.py                  # Modbus RS485 (pyserial+modbus_tk)
│       ├── USB_Glove_ctrl.py          # 手套遥操作
│       └── lib_gforce/gforce.py       # GForce SDK
│
└── scripts/                           # Shell 启动脚本
    ├── command_preprocess_data.sh
    ├── command_train_vqvae.sh
    ├── command_eval_vqvae.sh
    ├── command_train_dqrise.sh
    └── command_eval_rise_vae.sh
```

### 9.2 关键类详解

#### Sparse3DEncoder (`policy/tokenizer.py`)

```
输入: ME.SparseTensor(features=[N,6], coordinates=[N,4])  # [xyz, batch_idx]
  ↓
MinkowskiEngine ResNet14 (3D稀疏卷积):
  conv1: 6→64, stride=1
  pool: sum_pool(2×2×2)
  layer1: BasicBlock(64→64)
  layer2: BasicBlock(64→128, stride=2)
  layer3: BasicBlock(128→256, stride=2)
  layer4: BasicBlock(256→512, stride=2)
  final: conv(512→512, k=1)
  ↓
SparsePositionalEncoding: 3D正弦位置编码
  - x/y/z 三分量独立编码，dim_t = temperature^(2i/d) 方式
  - max_pos=800 (对应 0.005m×800=4m 范围)
  ↓
按 batch 分组 + padding 到 [B, max_num_token=100, 512]
  ↓
返回: tokens [B,100,512], pos_emb [B,100,512], padding_mask [B,100]
```

#### Transformer (`policy/transformer.py`)

```python
# DETR-style Transformer，来自 Facebook Research DETR + ACT
# Input: src [B,100,512], pos_emb [B,100,512], query_embed [1,512]
#   → permute(1,0,2) → [100,B,512] (DETR convention: seq_len first)

# Encoder (4 layers, post-norm):
#   Self-Attention(Q=K=src+pos, V=src) + FFN(d→2048→d)
#   → memory [100,B,512]

# Decoder (1 layer, return_intermediate_dec=True):
#   Self-Attention(Q=K=tgt+query, V=tgt)
#   + Cross-Attention(Q=tgt+query, K=memory+pos, V=memory)
#   + FFN
#   → hs [1,B,1,512] → squeeze → [B,512]

# 注意: 位置编码直接传入 attention 的 Q/K（而非加到 token 上）
```

#### MinkowskiEngine ResNet14 (`policy/minkowski/resnet.py`)

```
ResNet14: BLOCK=BasicBlock, LAYERS=(1,1,1,1)
  PLANES = (64, 128, 256, 512)
  OUT_PIXEL_DIST = 32

BasicBlock (稀疏3D版):
  Conv3×3×3 → BN → ReLU → Conv3×3×3 → BN
  + shortcut (1×1×1 Conv if dim mismatch)
  → ReLU

特征: 使用 MinkowskiEngine 的稀疏张量操作
  - conv(..., D=3): 3D稀疏卷积
  - sum_pool: 稀疏求和池化
  - BN: ME.MinkowskiBatchNorm (bn_momentum=0.02)
```

#### VectorQuantize (`vector_quantize_pytorch.py`)

核心 VQ 实现，**1,051 行代码**。关键机制：
- 两种距离度量：`EuclideanCodebook`（L2距离）和 `CosineSimCodebook`（余弦相似度）
- EMA 更新码本（`decay=0.8`）
- K-Means 初始化码本（可选，`kmeans_init=True`）
- Gumbel-Softmax / Straight-Through 采样
- 死码字复活（`threshold_ema_dead_code`）
- DDP 同步码本更新（`use_ddp=True` 时 `all_reduce`）

### 9.3 坐标变换系统

**文件**: `dataset/projector.py`

```python
class Projector:
    def project_tcp_to_camera_coord(tcp, cam_id):
        """Base Frame → Camera Frame (TCP 动作在相机坐标空间预测)"""
    def project_tcp_to_base_coord(tcp, cam_id):
        """Camera Frame → Base Frame (点云在基座坐标空间处理)"""
```

**双相机融合** (`riseVAE.py:284-290`):
```python
# 相机2 → Base → 相机1
points_cam2_base = projector.project_tcp_to_base_coord(points_cam2, cam2_id)
points_cam1_space = projector.project_tcp_to_camera_coord(points_cam2_base, cam1_id)
# 合并两个相机的点云
cloud = np.concatenate([cloud_cam1, points_cam1_space], axis=0)
```

### 9.4 data_filter.py 详解

**文件**: `data_filter.py`

数据过滤脚本执行三个关键操作：

1. **找第一个有效 TCP 帧**（`data_filter.py:29-34`）：
   ```python
   for ts in timestamp:
       if not ((tcp[ts][3:])**2).sum()==0:  # 旋转分量非零
           first_idx = ts; break
   ```

2. **过滤静止帧**（`data_filter.py:39-55`）：
   ```python
   # 检查相邻帧的运动
   trans = norm(trans2 - trans1)
   dot = clip(abs(dot(rot1, rot2)), -1, 1)
   rot = 2 * arccos(dot)
   moved = (trans > 1e-4) or (rot > 0)
   # 保留 moved=True 的帧
   ```

3. **时间对齐双相机**（`data_filter.py:96-101`）：
   ```python
   # 为相机1的每个时间戳找到相机2最接近的时间戳
   pairs = {}
   for ts in first_timestamps:
       closest = min(second_timestamps, key=lambda x: abs(x - ts))
       pairs[str(ts)] = closest
   torch.save(pairs, f"{color_dir}/pairs.pth")
   ```

### 9.5 本项目文件组织

```
dexmani_policy/
├── agents/
│   ├── core/dqrise.py                  # DQRISEAgent (BaseAgent子类)
│   └── vq_hand/
│       ├── vqvae.py                    # VqVaeHand (重构版，from_checkpoint支持)
│       ├── codebook_manager.py         # CodebookManager (nn.Module, 自包含)
│       ├── residual_vq.py              # ResidualVQ (与官方同源)
│       └── vector_quantize.py          # VectorQuantize (与官方同源)
├── configs/dqrise.yaml                 # Hydra完整配置 (含vq_vae段)
├── scripts/
│   ├── train_vq_hand.py               # VQ-VAE 预训练脚本
│   ├── extract_codebook.py            # 码本提取脚本 (使用CodebookManager)
│   └── measure_vq_usage.py            # 码本使用率分析
└── docs/
    ├── DQ-RISE-完整分析.md             # 旧版分析文档
    └── DQ-RISE-知识体系.md             # 本文档
```

---

## 10. 数据管道

### 10.1 原始数据采集

```
遥操作采集 (DQ-RISE 用户研究):
  - Meta Quest 3 VR 手柄: 手臂控制 (带"暂停"机制用于大旋转重定位)
  - OyMotion GForce 手套: 手部关节控制
  - 2× RealSense D415/D435: 720×1280 @ 30fps
  - 每任务 ~50 个遥操作演示
```

### 10.2 原始数据格式（官方）

```
data/task_XXXX/
├── calib/
│   └── [timestamp]/
│       ├── extrinsics.npy       # 相机→标记物外参
│       ├── intrinsics.npy       # 相机内参
│       └── tcp.npy             # 标定时刻TCP位姿
└── train/
    └── [episode_id]/
        ├── metadata.json       # {start_time, finish_time}
        ├── timestamp.txt       # 使用的标定时间戳
        ├── cam_[serial_1]/
        │   ├── color/*.png     # RGB (uint8)
        │   ├── depth/*.png     # Depth (uint16, 毫米单位)
        │   └── lowdim/
        │       ├── tcp.npz     # {timestamp: [x, y, z, qx, qy, qz, qw]}
        │       └── pos.npz     # {timestamp: [6DOF手指关节, 0-65535]}
        └── cam_[serial_2]/
            └── ... (同上)
```

### 10.3 预处理管道

```
Step 1 — 数据过滤 (data_filter.py):
  - 找到第一个有效TCP帧 (旋转非零)
  - 过滤静止帧 (translation < 1e-4 且 rotation = 0)
  - 双相机时间对齐 → pairs.pth

Step 2 — 预处理 (preprocess_data.py):
  - 加载 RGB-D → process_pointcloud.process():
      Open3D PointCloud → Base坐标变换 → 工作空间裁剪 → Voxel降采样(0.005m)
  - 双相机点云合并
  - 保存:
      processed_data.h5: handPose_data + tcp_data (HDF5 + gzip压缩)
      processed_pc_*.npy: 点云 (N,6) = (xyz + rgb)

Step 3 — Zarr 转换 (本项目):
  原始 .npz 数据 → Zarr replay buffer
  normalizer 拟合 (全量, mode='limits')
```

### 10.4 在线数据加载

**官方 DQ-RISE 训练** (`RiseVAEDataset`, `dataset/riseVAE_2cam.py`):

```
预处理点云 .npy → 在线增强:
  - 3D 旋转/平移 (aug)
  - HSV 颜色抖动 (aug_jitter)
  - ImageNet 归一化: (color - IMG_MEAN) / IMG_STD

TCP 动作归一化:
  平移: (trans - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1 → [-1,1]
  旋转: quaternion → 6D rotation representation

手部动作归一化:
  原始 [0,65535] → decode: /65536 → [0,1] → *2-1 → [-1,1] (VQ-VAE用)
  → 最近码字查找 → index/(TCP_DIM+HAND_DIM)*2-1 → [-1,1] (DQ-RISE用)
```

### 10.5 序列采样

```python
# 滑动窗口: num_obs=1, num_action=20 (官方) / horizon=16 (本项目)
obs_frame_ids = [frame_ids[cur_idx]]                    # 1个观测帧
action_frame_ids = frame_ids[cur_idx+1 : cur_idx+1+horizon]  # 动作序列
# 边界: 复制首尾帧填充
```

---

## 11. 关键设计决策

### 11.1 为什么单步量化而非动作块量化？

论文明确讨论了此设计选择：
- **动作块量化被拒绝**：对 `[20,6]` 块做量化 → 码本膨胀（需要巨大码本覆盖 20 步组合）
- **单步量化**：仅量化单步 `[1,6]` → 16 种组合 → 通过联合扩散自然产生时序轨迹

### 11.2 为什么 PCA 在原始手部姿态上排序？

- 在**原始 6/12 DOF 手部姿态**上做 PCA → 排序有物理语义（如手指开合程度）
- 在 **VQ-VAE 256 维潜在特征**上做 PCA → 不能保证排序语义
- 论文消融验证了这一点

### 11.3 为什么连续 VQ 索引？

- 连续索引使扩散模型能平滑地在码本空间中去噪
- one-hot 表示不可微，无法利用扩散的平滑特性
- 推理时通过 `round(clip(index, -1, 1))` 取整

### 11.4 为什么用 FiLM 而非 Cross-Attention？

- FiLM 通过 per-channel scale+bias 注入条件
- 比 Cross-Attention 轻量，适合 UNet 卷积结构
- Diffusion Policy 系列工作标准做法

### 11.5 为什么 CodebookManager 是 nn.Module？

本项目设计决策（与官方脚本式方法的关键差异）：
- **自包含 checkpoint**: 排序码本存入 `state_dict`，checkpoint 加载即用
- **设备透明**: `model.to(device)` 自动移动码本
- **Normalizer 一致性校验**: 内置 `_validate_codebook_normalizer()` 确保训练/推理坐标空间一致
- **持久化元数据**: `.npz` 包含 hand_min/max、layer_weights、PCA 诊断信息

---

## 12. 论文-代码交叉验证

### 12.1 匹配项 ✅

| 论文声明 | 代码实现 | 文件:行 |
|----------|----------|---------|
| VQ-VAE 2组×4码字=16种 | `ResidualVQ(num_quantizers=2, codebook_size=4)` | `vqvae.py:74-78` |
| PCA排序使索引连续 | `PCA(n_components=1)` + `argsort` | `eval_vqvae.py:106-109` |
| DDIM 100/20步 | `DDIMScheduler(100)`, `num_inference_steps=20` | `diffusion.py:43-48,68` |
| MinkowskiEngine ResNet14 | `ResNet14(in=6, out=512)` | `tokenizer.py:11` |
| DETR (4enc+1dec) | `num_encoder_layers=4, num_decoder_layers=1` | `transformer.py` |
| FiLM条件注入 | `cond_predict_scale=True` | `conditional_unet1d.py:56-62` |
| 50个演示/任务 | README确认 | — |
| 三阶段训练管道 | train_vqvae → eval_vqvae → train_dqrise | 三个独立脚本 |
| 手指加权L1 | `loss_weight=[1,1,1,0.5,0.5,1]` | `vqvae.py:189` |

### 12.2 差异项 ⚠️

| 方面 | 论文 | 代码 | 影响 |
|------|------|------|------|
| VQ-VAE重建损失 | MSE `||s-ŝ||₂²` | 加权L1 `abs(state-dec_out)·weight` | L1对异常值更鲁棒 |
| VQ-VAE损失权重 | `β=γ=1.67` | `L1×3 + VQ×5` | 比例接近，数值不同 |
| VQ-VAE Epochs | 1500 | 代码默认1000 (README说1500) | README和代码不一致 |
| 码本提取权重 | 未说明 | 硬编码0.5 (官方) / 学习权重 (本项目) | 本项目更一致 |
| `layer_weights` | 未提及 | `softmax([0.5,0.5])` 可学习参数 | `residual_vq.py:48` |
| 死码字复活 | 未提及 | `threshold_ema_dead_code=2` | `vector_quantize_pytorch.py:424` |

### 12.3 代码独有细节（论文未提及）📝

| 细节 | 位置 | 说明 |
|------|------|------|
| EncoderMLP 正交初始化 | `vqvae_utils.py:9-19` | `nn.init.orthogonal_` |
| act_scale=1.0 (不缩放) | `vqvae.py:172` | 数据已在[-1,1] |
| ImageNet颜色归一化 | `riseVAE.py:282` | `(color-mean)/std` |
| HSV颜色增强(非RGB) | `riseVAE_2cam.py` | 保留颜色语义 |
| 双相机时间对齐 | `data_filter.py` | 保存 `pairs.pth` |
| NCCL_P2P_DISABLE=1 | `train_dqrise.py:55` | 单卡DDP兼容 |

### 12.4 已知 Bug 🐛

| Bug | 文件:行 | 说明 | 严重程度 |
|-----|---------|------|----------|
| Dead code: `idx == len(self.up_modules)` | `conditional_unet1d.py:229` | 永假条件，`h_local[1]` 永不注入 | 中等 (local_cond当前不用) |
| 硬编码 `range(5)` | `preprocess_data.py:67` | 仅处理前5个样本，非全部 | 高 (需手动修改) |
| 硬编码标定路径 | `dataset/riseVAE.py:74` | `data/task_0006/calib/1753091226804` | 高 (需手动修改) |
| `torch.concatenate`(不存在) | `separate_diff_baseline_policy.py:52` | 应为 `torch.cat` | 致命 (baseline崩溃) |
| `import pdb; pdb.set_trace()` | `vqvae.py:253` | 调试代码残留 | 低 |
| `import pdb; pdb.set_trace()` | `data_filter.py:77` | 调试代码残留 | 低 |

---

## 13. 与本项目实现的完整对比

### 13.1 架构对比

| 维度 | 官方 DQ-RISE | 本项目 DQRISEAgent |
|------|-------------|-------------------|
| **3D编码器** | MinkowskiEngine Sparse ResNet14 | iDP3 / PointNeXT (dense point cloud) |
| **输入** | 2× RGB-D → 融合稀疏点云 (双相机实时融合) | 预采样 (1024, 3\|6) 密集点云 |
| **Transformer** | DETR Transformer (4enc+1dec+readout) | 无 (直接用编码器输出) |
| **扩散骨干** | ConditionalUnet1D + DDIM | ConditionalUnet1D + DDIM (同架构) |
| **动作空间** | joint(19) 或 ee(21) + VQ | joint(19) 或 ee(21) + VQ (同) |
| **VQ-VAE** | ResidualVQ 2组×4码字 | ResidualVQ 2组×4码字 (同架构) |
| **CodebookManager** | `eval_vqvae.py` 脚本 (一次性) | `CodebookManager(nn.Module)` (自包含) |
| **手部维度** | 6 (ROHand) | 12 (XHand) |
| **训练框架** | DDP torchrun | Hydra + DDP (兼容单卡/多卡) |
| **数据格式** | 原始 RGB-D + .npz | Zarr replay buffer |
| **推理** | 真机 (Flexiv+RoHand) | 仿真 (dexmani_sim) |
| **VQ-VAE 训练** | 独立脚本 (DDP) | `train_vq_hand.py` (单卡, Hydra配置) |

### 13.2 CodebookManager: 从脚本到 nn.Module 的进化

**官方** (`eval_vqvae.py`):
```python
# 一次性脚本: 加载checkpoint → 提取码本 → PCA排序 → 保存.npy
codebooks = vq_layer.codebooks.cpu()  # [2, 4, 256]
latent = codebooks[0,i,:]*0.5 + codebooks[1,j,:]*0.5  # 硬编码权重!
action = decoder(latent)
# → sorted_hand_actions.npy
```

**本项目** (`codebook_manager.py`):
```python
# nn.Module: 可持久化到policy checkpoint中
class CodebookManager(nn.Module):
    # Persistent buffers (随state_dict保存):
    sorted_hand_poses:   torch.Tensor  # [16, hand_dim]
    pca_permutation:     torch.Tensor
    layer_weights:       torch.Tensor  # 学习到的, 非硬编码

    # 关键方法:
    reindex_by_pca(vqvae)              # 使用学习权重+解码器范围验证
    hand_pose_to_continuous_index()    # 训练时: hand→index
    continuous_index_to_hand_pose()    # 推理时: index→hand
    save/load()                        # .npz 持久化 + 兼容旧版.npy

    # 内置 Normalizer 一致性检查:
    _validate_codebook_normalizer()    # 确保 VQ 和 policy 坐标系一致
```

### 13.3 VqVaeHand vs 官方 VqVae

| 方面 | 官方 VqVae | 本项目 VqVaeHand |
|------|-----------|-----------------|
| `from_checkpoint` 支持 | ❌ | ✅ 从state_dict推断架构 |
| 内置优化器 | ✅ | ❌ (外部管理) |
| 层数推断 | 配置参数 (可能不准) | 从state_dict自动推断 |
| act_scale | 参数 | register_buffer |
| loss_weight | 硬编码在forward | register_buffer (可配置) |
| 代码行数 | ~262 | ~239 |

### 13.4 本项目 DQRISEAgent 训练流程

```python
# compute_loss (dqrise.py:172-210):
def compute_loss(self, batch):
    # 1. 编码观测
    cond = self._build_cond(batch["obs"])

    # 2. 归一化动作
    normed = self.normalizer["action"].normalize(batch["action"])

    # 3. 分离臂/手，手部→连续VQ索引
    tcp_part = normed[..., :self.tcp_dim]
    hand_part = normed[..., self.tcp_dim:]
    index = self.codebook_manager.hand_pose_to_continuous_index(hand_part)

    # 4. 联合动作 → 扩散损失
    joint_action = torch.cat([tcp_part, index], dim=-1)
    loss = self.action_decoder.compute_loss(cond, joint_action)

    # 5. 监控: 码本使用熵和利用率
    entropy = -(p * log(p)).sum()          # batch_nn_code_entropy
    used_1pct = (probabilities > 0.01).sum() # batch_nn_code_used_1pct
```

### 13.5 本项目配置层级

```yaml
# configs/dqrise.yaml
policy_name: dqrise          # → DQRISEAgent
action_key: action_ee        # → 21维 (tcp_dim=9, hand_dim=12)
tcp_dim: 9
codebook_path: robot_data/sorted_hand_poses_${task_name}.npz

agent:                       # DQRISEAgent 构造参数
  encoder_type: idp3         # iDP3 点云编码器
  down_dims: [256, 512]
  num_inference_steps: 20

vq_vae:                      # VQ-VAE 预训练配置 (Stage 1)
  num_groups: 2
  codebook_size: 4
  num_epochs: 1500
  enc_loss_weight: 3.0
  vq_loss_weight: 5.0
```

---

## 14. 已知局限与改进方向

### 14.1 论文指出的局限

1. **计算开销**: 3D 重建（双相机 RGB-D → 点云）增加推理延迟
2. **数据需求**: 每任务 50 个演示，更复杂任务可能需要更多
3. **码本大小固定**: 16 种手部姿态可能不足以覆盖所有手势
4. **单任务训练**: 无跨任务泛化

### 14.2 架构改进方向

| 方向 | 当前 | 改进 | 优先级 |
|------|------|------|--------|
| 码本大小 | 4×4=16 | 8×8=64, 或更多组 | P1 |
| 多任务码本共享 | ❌ | 跨任务共享VQ码本 | P2 |
| 端到端VQ训练 | 冻结 | 联合微调VQ-VAE | P2 |
| DDIM步数 | 20 | 10或更少 (加速推理) | P1 |
| Transformer | DETR (官方) / 无 (本项目) | FlashAttention / Perceiver | P3 |
| K-Means初始化 | 官方无/本项目有 | 启用减少码本崩溃 | ✅已实现 |

### 14.3 本项目已验证的无效优化

参见 memory: `[[dqrise-60pct-baseline]]` 和 `[[dqrise-vq-quality-checklist]]`：
- `vq_idx_used` < 8 → 判死（~0% 成功率），≥ 12 → 健康（~60% 成功率）
- VQ-VAE 质量是下游成功的决定性因子

---

## 附录 A: 超参数速查表

### VQ-VAE 训练

| 参数 | 官方值 | 本项目值 |
|------|--------|---------|
| `hand_dim` | 6 | 12 |
| `latent_dim` | 256 | 256 |
| `hidden_dim` | 512 | 512 |
| `num_groups` | 2 | 2 |
| `codebook_size` | 4 | 4 |
| `num_layers` | 1+1=2 | 3 |
| `lr` | 3e-4 | 3e-4 |
| `batch_size` | 256 | 256 |
| `num_epochs` | 1000/1500 | 1500 |
| `vq_decay` | 0.8 | 0.8 |
| `threshold_ema_dead_code` | 2 | 2 |
| `kmeans_init` | False | **True** |
| 损失权重 | L1×3 + VQ×5 | `enc_loss_weight:3.0, vq_loss_weight:5.0` |

### DQ-RISE 训练

| 参数 | 官方值 | 本项目值 |
|------|--------|---------|
| `action_dim` | 10 (9+1) | 8 (7+1) 或 10 (9+1) |
| `tcp_dim` | 9 | 7 或 9 |
| `horizon` | 20 | 16 |
| `n_obs_steps` | 1 | 2 |
| `n_action_steps` | — | 8 |
| `down_dims` | [256, 512] | [256, 512] |
| `kernel_size` | 5 | 5 |
| `n_groups` | 8 | 8 |
| `lr` | 3e-4 | 3e-4 |
| `batch_size` | 240 | 128 |
| `num_epochs` | 1000 | 1000 |
| DDIM train/inf | 100/20 | 100/20 |

### 硬编码常量（官方）

| 常量 | 值 | 文件 |
|------|-----|------|
| `TCP_DIM` | 9 | `utils/constants.py` |
| `HAND_DIM` | 6 | `utils/constants.py` |
| `max_num_token` | 100 | `tokenizer.py:15` |
| `max_pos` | 800 | `tokenizer.py:44` |
| UNet down_dims | [256, 512] | `diffusion.py:19` |
| UNet kernel_size | 5 | `diffusion.py:20` |
| VQ finger weights | [1,1,1,0.5,0.5,1] | `vqvae.py:189` |
| VQ layer_weights init | [0.5, 0.5] | `residual_vq.py:49` |
| VQ EMA decay | 0.8 | `vector_quantize_pytorch.py:248` |
| VQ dead code threshold | 2 | `vector_quantize_pytorch.py:250` |

---

## 附录 B: 命令速查

### 官方 DQ-RISE

```bash
# Step 1: 数据预处理
bash scripts/command_preprocess_data.sh

# Step 2: VQ-VAE 预训练
torchrun --nproc_per_node=1 train_vqvae.py \
    --preprocess_data_path processed_data.h5 \
    --input_dim_h 1 --input_dim_w 6 \
    --n_latent_dims 256 --vqvae_n_embed 4 --vqvae_groups 2 \
    --num_epochs 1500 --batch_size 256

# Step 3: 码本提取 + PCA 排序
torchrun --nproc_per_node=1 eval_vqvae.py \
    --ckpt_path logs/vqvae_ckpt.pt \
    --codebook_path logs/sorted_actions.npy

# Step 4: DQ-RISE 联合训练
torchrun --nproc_per_node=1 train_dqrise.py \
    --vae_codebook logs/sorted_actions.npy \
    --action_dim 10 --num_epochs 1000

# Step 5: 真机评估
bash scripts/command_eval_rise_vae.sh
```

### 本项目 dexmani_policy

```bash
# Step 1: VQ-VAE 预训练
bash scripts/train_vq_hand.sh pour '--num_epochs 1500'

# Step 2: 码本提取
python dexmani_policy/scripts/extract_codebook.py \
    --checkpoint experiments/vq_hand/pour/.../policy_last.ckpt \
    --output robot_data/sorted_hand_poses_pour.npz

# Step 3: DQ-RISE 训练 (单卡)
bash scripts/train.sh dqrise

# Step 4: DQ-RISE 训练 (多卡 DDP)
bash scripts/train_ddp.sh ddp/dqrise

# Step 5: 仿真评估
bash scripts/eval_sim.sh dqrise pick_apple_messy <exp_dir>
```

---

## 附录 C: 论文核心贡献

1. **手部姿态量化**: VQ-VAE 将连续手部动作压缩为离散编码，解决臂-手动作空间不平衡
2. **连续松弛 + PCA 排序**: 使离散编码可被扩散模型平滑处理（关键：PCA on raw pose, not VQ feature）
3. **联合扩散**: 手臂 TCP + VQ 索引在统一扩散过程中联合建模，保持臂-手协调
4. **实验验证**: 6 个真实世界任务，85.83% 平均成功率，远超基线 (RISE 55%, RISE-S 62%, DQ-RISE-C 2.5%)
5. **梯度流一致性理论**: DQ-RISE-C 的惨败 (2.5%) 验证了扩散 MSE + 分类 CE 梯度流不一致导致的训练崩塌

> **一页纸总结**: DQ-RISE = **V**Q-VAE 手部量化 + **P**CA 排序 + **D**DIM 联合扩散。三阶段管道：预训练 VQ-VAE → 提取排序码本 → 联合扩散训练。tcp_dim+1 维动作空间替代完整 action_dim 维。16 种手部姿态 (2组×4码字) + 连续索引松弛 = 平滑的臂-手联合去噪。
