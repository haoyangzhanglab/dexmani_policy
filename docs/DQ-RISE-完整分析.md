# DQ-RISE 完整分析

> 论文精读 / 官方代码走读 / 本项复现差异 / 实验踩坑全纪录
>
> **论文**: Feng et al., *Learning Dexterous Manipulation with Quantized Hand State*, ICRA 2026 (arXiv:2509.17450)
> **官方仓库**: <https://github.com/RISE-Policy/DQ-RISE> (CC BY-NC-SA 4.0)
> **前置工作**: RISE (IROS 2024) — 3D Perception Makes Real-World Robot Imitation Simple and Effective

---

## 目录

1. [论文精读](#1-论文精读)
   - 1.1 [核心问题：动作空间失衡](#11-核心问题动作空间失衡)
   - 1.2 [三部曲论证弧线](#12-三部曲论证弧线)
   - 1.3 [VQ-VAE 手部状态量化](#13-vq-vae-手部状态量化)
   - 1.4 [PCA 连续松弛 — 核心创新](#14-pca-连续松弛--核心创新)
   - 1.5 [联合扩散与实验结果](#15-联合扩散与实验结果)
2. [官方代码精读](#2-官方代码精读)
   - 2.1 [项目结构](#21-项目结构)
   - 2.2 [核心架构逐模块](#22-核心架构逐模块)
   - 2.3 [三阶段训练管道](#23-三阶段训练管道)
   - 2.4 [Baseline 策略对比](#24-baseline-策略对比)
   - 2.5 [关键硬编码常量](#25-关键硬编码常量)
3. [本项目实现与官方代码的差异](#3-本项目实现与官方代码的差异)
   - 3.1 [动作空间](#31-动作空间)
   - 3.2 [感知架构 ★最大差异](#32-感知架构-最大差异)
   - 3.3 [VQ-VAE](#33-vq-vae)
   - 3.4 [CodebookManager 独创封装](#34-codebookmanager-独创封装)
   - 3.5 [归一化与训练流程](#35-归一化与训练流程)
   - 3.6 [保持一致的设计](#36-保持一致的设计)
   - 3.7 [架构对比图](#37-架构对比图)
4. [实验踩坑记录与优化方向](#4-实验踩坑记录与优化方向)
   - 4.1 [60% 基线](#41-60-基线-pour-任务)
   - 4.2 [决定性因子：vq_idx_used](#42-决定性因子vq_idx_used)
   - 4.3 [已验证无效的优化](#43-已验证无效的优化)
   - 4.4 [E1 教训：指标-奖励错配](#44-e1-教训指标-奖励错配)
   - 4.5 [VQ 质量诊断 Checklist](#45-vq-质量诊断-checklist)
   - 4.6 [优化方向按优先级](#46-优化方向按优先级)
5. [附录](#5-附录)
   - 5.1 [术语速查](#51-术语速查)
   - 5.2 [关键文件索引](#52-关键文件索引)
   - 5.3 [官方代码已知问题](#53-官方代码已知问题)
   - 5.4 [参考文献](#54-参考文献)

---

## 1. 论文精读

### 1.1 核心问题：动作空间失衡

灵巧手操作的核心矛盾——**高维手部动作主导耦合的臂-手动作空间，损害臂部定位精度**。

以"开罐子"为例，任务同时考验三个能力：
1. **臂部定位**：末端执行器准确对齐罐盖
2. **手部灵巧**：手指精确勾住盖子边缘
3. **臂-手协调**：手指"勾"与臂部"压"在时间上精确对齐

当臂部动作（6-9 维）和手部动作（6-12 维）拼接在一起让策略学习时，高维手部信号在损失函数中占据更大比例，优化过程"忽略"臂部定位——而臂部负责"到达"目标，是任务成功的基础。

### 1.2 三部曲论证弧线

论文的核心论证围绕三个递进的策略设计展开：

```
统一动作空间（RISE，现有方法）
  ↓ 问题：高维手部动作主导，臂部学不好（55.00% avg）

天真解耦 — 分离臂手预测（RISE-S）
  ↓ 问题：破坏了臂-手协调（开罐子任务失败，61.67% avg）

DQ-RISE 方案：量化 + 连续松弛 + 联合扩散
  ↓ 优势：压缩手部空间 + 保持联合可微 + 保留协调（85.83% avg）
```

三个 Baseline 对应了三个位置：

| 基线 | 策略 | 臂-手关系 | 失败原因 |
|---|---|---|---|
| **RISE** | 全动作空间统一扩散 (15-dim) | 统一但未分离 | 手部主导损失，臂部定位受损 |
| **RISE-S** | 两个独立扩散头（臂 + 手） | 完全分离 | 臂-手协调断裂（开罐子失败） |
| **DQ-RISE-C** | 臂扩散 + 手分类（16 类） | 有条件依赖 | 梯度流不一致（扩散 loss + 交叉熵 loss 在共享 backbone 中冲突），**2.50% avg** |
| **DQ-RISE** | 臂扩散 + 手连续索引联合扩散 | 联合可微 | ✅ 统一梯度 + 压缩手部空间 |

**关键洞察**：不是将臂和手完全解耦（丢失协调性），也不是简单拼接（手部支配损失），而是通过**量化和联合扩散**找到平衡——压缩手部动作对损失的支配，同时保留臂-手联合建模能力。

### 1.3 VQ-VAE 手部状态量化

#### 两个"不"：量化设计原则

| 不做 | 原因 | 后果 |
|---|---|---|
| 量化臂-手全动作 | 臂和手功能本质不同 | 臂部连续控制被离散化，定位精度受损 |
| 量化动作块（时间序列） | codebook 指数爆炸 + 生成方法不一致 | 臂（扩散回归）vs 手（分类）→ 梯度流冲突 |

核心原则：**"model hand states, not motion trajectories"** — 直接建模单步手部姿态，而非运动轨迹。

#### VQ-VAE 结构

```
hand_state (B, 6) → EncoderMLP → (B, 256)
  → ResidualVQ Layer 0 (4 codes) → residual → ResidualVQ Layer 1 (4 codes)
  → weighted sum (0.5 × layer0 + 0.5 × layer1)
  → DecoderMLP → reconstructed (B, 6)
```

| 组件 | 配置 |
|---|---|
| 编码器 | MLP (6 → 512 → 256)，正交初始化，`layer_num=1` |
| VQ 层 | ResidualVQ，2 groups × 4 codes = **16 种离散手部姿态** |
| 解码器 | MLP (256 → 512 → 6)，复用 EncoderMLP |
| 损失 | `L1_recon(per-finger weighted) × 3 + commitment_loss × 5` |
| 优化器 | AdamW，lr=3e-4，betas=[0.95, 0.999]，weight_decay=1e-6 |
| 训练 | 1500 epochs，batch_size=256（分布式 per-GPU 值） |
| 手指权重 | `[1.0, 1.0, 1.0, 0.5, 0.5, 1.0]`（拇指3+食指+中指=1.0，无名指+小指+拇指旋转=0.5） |

#### 后验量化 vs 先验量化

DQ-RISE 的关键设计选择：**训练时自动量化（后验），而非采集时预设手势（先验）**。
- 采集时操作者自由控制手部 → VQ-VAE 自动从数据中发现手势模式
- 保留了遥操作的灵活性，同时获得量化的训练好处
- Codebook 的 16 个条目是从数据中学到的，不是人工定义的

### 1.4 PCA 连续松弛 — 核心创新

#### 动机：从"分类"回到"回归"

自然做法（DQ-RISE-C）：臂扩散 + 手分类 head → 2.50% 成功率，彻底失败。失败源于梯度流不一致——扩散 loss（MSE）和分类 loss（交叉熵）在共享 backbone 中反向传播时产生冲突。

DQ-RISE 的方案：受**夹爪**启发——
- 夹爪语义上是二值的（开/闭），但物理上是连续的过渡
- 16 个离散手部姿态如果排成连续谱 → 可以像夹爪一样用连续值预测

#### PCA 排序算法

对应官方代码 `eval_vqvae.py:88-111`：

```
1. 枚举所有 4×4=16 个 codebook 组合
   latent = 0.5 × codebook[0, i] + 0.5 × codebook[1, j]

2. 解码为手部姿态（raw servo space [0, 65535]）
   hand_pose = decoder(latent) → clip to [0, 65535]

3. 对 16×6 手部姿态矩阵做 PCA(n_components=1)

4. 按 PC1 投影值排序 → [code_0, code_1, ..., code_15]
   相邻索引 = 在 PCA 方向上相邻 → 语义相似的手部姿态
```

**为什么在手部状态空间做 PCA，不在潜空间做？**
高维 VQ-VAE latent space（256 维）中的欧式距离结构和语义相似性之间没有简单对应。256 维中"近"的两个向量解码后可能得到差异很大的手部姿态。在原始手部关节角空间（6 维）做 PCA，"近"天然意味着关节配置相似，语义一致。

#### 连续松弛的数学形式

**训练时**（将 GT 手部姿态转为连续标签）：
```python
distances = cdist(hand_pose, codebook_poses)   # L2 in raw [0, 65535] space
index = argmin(distances)                       # [0, 15] discrete
continuous_index = index / 15 * 2 - 1           # → [-1, 1]
action = [tcp(9), continuous_index(1)]          # 10-dim training target
```

**推理时**（将扩散输出转回手部姿态）：
```python
tcp_pred, idx_pred = split(diffusion_output)    # idx_pred ∈ [-1, 1]
idx_discrete = round((idx_pred + 1) / 2 * 15)   # → {0, ..., 15}
hand_pose = codebook[idx_discrete]              # 查表
```

**误差容忍**：预测值 3.4 vs 3.6 都 round 到索引 3 → 语义上（PCA 排序后）相似的两个手部姿态。这是连续松弛允许扩散模型存在一定预测误差而不导致灾难性失败的关键。

### 1.5 联合扩散与实验结果

#### 动作空间

DQ-RISE 训练时的有效动作空间为 **10 维**：`[TCP(9) + continuous_VQ_index(1)]`。
扩散头输出 10 维，使用单一的 DDIM MSE loss。与 DQ-RISE-C 的关键区别：统一回归损失 → 无梯度冲突。

#### 主实验结果

6 个真实世界任务，50 demos/task，20 trials/task：

| 策略 | Pull Tissue | Open Jar | Collect Toy | Pour Rice | Open Oven | Toast Bread | **平均** |
|---|---|---|---|---|---|---|---|
| RISE | 75%/45% | 80%/55% | 60%/60% | 90%/80% | 100%/90% | 80%/20%/0% | **55.00%** |
| RISE-S | 75%/55% | 60%/45% | 75%/70% | 95%/85% | 95%/95% | 75%/25%/20% | **61.67%** |
| DQ-RISE-C | 15%/10% | 0%/0% | 0%/0% | 0%/0% | 20%/5% | 0%/0%/0% | **2.50%** |
| **DQ-RISE** | **95%/85%** | **95%/90%** | **95%/80%** | **100%/100%** | **100%/100%** | **100%/65%/60%** | **85.83%** |

多阶段任务用 `a/b` 格式（例：Pull Tissue = Grasp/Place，Toast Bread = Grasp/Insert/Press）。

#### 消融关键发现

| 消融因素 | 发现 |
|---|---|
| PCA 重排序 vs 无排序 | 移除重排序 → **显著性能下降**，非连续编码使学习"困难和不可稳定" |
| 单步 vs 动作块量化 | 块量化 → codebook 膨胀 + 与臂扩散解耦 |
| PCA on 原始状态 vs VQ 特征 | 必须在原始手部状态空间做 PCA——高维 VQ 特征不保证排序语义 |
| 联合扩散 vs 解耦分类 | 梯度流一致性是 DQ-RISE-C 失败的根本原因（2.5% vs 85.83%） |

#### 硬件平台

- **机械臂**: Flexiv Rizon 4
- **灵巧手**: OyMotion RoHand (6-DOF)
- **相机**: 2× Intel RealSense D415 (720×1280 RGB-D)
- **遥操作**: Meta Quest 3 VR 手柄（臂）+ OyMotion GForce 数据手套（手），含暂停机制

---

## 2. 官方代码精读

> 代码路径：`/home/zhanghaoyang/Desktop/DQ-RISE/`

### 2.1 项目结构

```
DQ-RISE/
├── policy/                            # 核心策略模块
│   ├── policy.py                      # RISE 主策略类
│   ├── tokenizer.py                   # Sparse3DEncoder (Minkowski ResNet14)
│   ├── transformer.py                 # DETR-style Transformer
│   ├── diffusion.py                   # DiffusionUNetPolicy (DDIM UNet)
│   ├── tcp_cond_handPose_policy.py    # Baseline: TCP-条件手部分类
│   ├── seperate_diff_vae_baseline_policy.py  # Baseline: 独立扩散+VAE
│   ├── separate_diff_baseline_policy.py      # Baseline: 独立双扩散头
│   ├── diffusion_modules/             # ConditionalUnet1D, conv blocks, pos emb
│   ├── minkowski/                     # MinkowskiEngine ResNet14/18/34/...
│   └── vqvae_rise/                    # VQ-VAE 子模块
│       ├── vqvae.py                   # VqVae + EncoderMLP
│       ├── pretrain_vqvae.py          # VQ-VAE 预训练入口 (Hydra)
│       └── vector_quantize_pytorch/   # ResidualVQ, VectorQuantize
├── dataset/                           # 数据加载 + 预处理
│   ├── riseVAE_2cam.py               # 双相机数据集（主力）
│   └── pretrain.py                    # VQ-VAE 预训练数据集
├── utils/
│   ├── constants.py                   # TCP_DIM=9, HAND_DIM=6, TRANS_MIN/MAX 等
│   ├── training.py                    # set_seed, plot_history, sync_loss
│   └── ensemble.py                    # 时间集成 (ACT/HATO/avg)
├── device/                            # 机器人设备驱动 (Flexiv/RealSense/ROHand)
├── train_dqrise.py                    # Stage 3: DQ-RISE 训练入口
├── train_vqvae.py                     # Stage 1: VQ-VAE 训练入口
├── eval_vqvae.py                      # Stage 2: 码本提取 + PCA 重排序
├── eval_rise_vae_2cam.py             # 双相机真机评估
└── scripts/                           # Shell 训练脚本
```

### 2.2 核心架构逐模块

#### RISE 主策略 (`policy/policy.py`)

```python
class RISE(nn.Module):
    def __init__(self):
        self.sparse_encoder = Sparse3DEncoder(input_dim=6, obs_feature_dim=512)
        self.transformer = Transformer(d_model=512, nhead=8,
                                       num_encoder_layers=4, num_decoder_layers=1)
        self.action_decoder = DiffusionUNetPolicy(action_dim=10, num_action=20,
                                                   num_obs=1, obs_feature_dim=512)
        self.readout_embed = nn.Embedding(1, 512)  # 1 个可学习 readout query

    def forward(self, cloud, actions=None):
        src, pos, mask = self.sparse_encoder(cloud)
        readout = self.transformer(src, mask, self.readout_embed.weight, pos)[-1][:, 0]
        if actions is not None:
            return self.action_decoder.compute_loss(readout, actions)
        else:
            return self.action_decoder.predict_action(readout)
```

**数据流**: Sparse Point Cloud → Sparse3DEncoder (per-point tokens) → Transformer (cross-attn → readout) → UNet (FiLM cond) → `[TCP(9)+VQ_idx(1)] × 20 steps`

#### Sparse3DEncoder (`policy/tokenizer.py`)

- **骨干**: MinkowskiEngine ResNet14，4 个残差块，通道 [64, 128, 256, 512]
- **输入**: 稀疏张量 (坐标 + RGB 特征, 6 通道)，voxel_size=0.005m
- **输出**: 每个点的特征 token (512 维)
- **位置编码**: 3D 正弦编码 (x/y/z 各 ~21 维)，max_pos=800 (~4m 范围)
- **批次化**: padding 到 `[B, max_token=100, 512]`

#### Transformer (`policy/transformer.py`)

- **来源**: 改编自 DETR (Facebook Research) 和 ACT
- **架构**: 4 层 encoder + 1 层 decoder，d_model=512，nhead=8，dim_ff=2048，dropout=0.1
- **特点**: 位置编码直接传入 multi-head attention（非加到 token 上）；1 个可学习的 readout query；Xavier 初始化

#### DiffusionUNetPolicy (`policy/diffusion.py`)

- **骨干**: `ConditionalUnet1D` (1D 卷积 UNet)，down_dims=[256, 512]，kernel_size=5，n_groups=8
- **条件注入**: 全局 FiLM 调制（从 readout token 展平）
- **DDIM 调度**: 100 train / 20 inference steps，beta [1e-4, 0.02]，`squaredcos_cap_v2`，epsilon 预测
- **输出**: 20 步动作轨迹，每步 10 维

#### VqVae (`policy/vqvae_rise/vqvae.py`)

```python
class VqVae(nn.Module):
    def __init__(self, input_dim_h=1, input_dim_w=6, n_latent_dims=256,
                 vqvae_n_embed=4, vqvae_groups=2, layer_num=1, act_scale=1.0):
        self.encoder = EncoderMLP(input_dim_w, n_latent_dims, hidden_dim=512,
                                  layer_num=layer_num)   # 6 → 512 → ... → 256
        self.vq_layer = ResidualVQ(dim=n_latent_dims, num_quantizers=vqvae_groups,
                                   codebook_size=vqvae_n_embed)
        self.decoder = EncoderMLP(n_latent_dims, input_dim_w, hidden_dim=512,
                                  layer_num=layer_num)   # 256 → 512 → ... → 6

    def forward(self, state):
        state = state / self.act_scale
        state_rep = self.encoder(self.preprocess(state))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        dec_out = self.decoder(state_vq)

        loss_weight = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 1.0])
        encoder_loss = (state - dec_out).abs().mul(loss_weight).mean()   # L1 weighted
        vqvae_recon_loss = F.mse_loss(state, dec_out)                   # MSE (logging only)

        return encoder_loss, vq_loss_state, vq_code, vqvae_recon_loss
```

**关键实现细节**：
- EncoderMLP 使用正交初始化（`weights_init_encoder`）
- ResidualVQ 层权重 `layer_weights` 初始化为 `[0.5, 0.5]`（softmax-normalized，在训练中是**可学习的**）
- 内置独立优化器（Adam，lr=1e-3，weight_decay=1e-4）— 但实际训练使用外部 AdamW
- `act_scale=1.0` — 不对输入做额外缩放
- 编码器和解码器复用同一个 `EncoderMLP` 类

### 2.3 三阶段训练管道

#### Stage 1: VQ-VAE 预训练 (`train_vqvae.py`)

```python
# 核心训练循环 (line 148-158)
Encoder_Loss, VQ_loss_state, VQ_code, Recon_Loss = policy(action_data)
total_loss = Encoder_Loss * encoder_loss_multiplier * 3 + VQ_loss_state * 5
# default: encoder_loss_multiplier=1.0 → enc_loss×3 + vq_loss×5
```

| 参数 | 值 |
|---|---|
| 训练 epochs | 1500 |
| Batch size | 256 (总) |
| 优化器 | AdamW，lr=3e-4，betas=[0.95, 0.999]，weight_decay=1e-6 |
| LR 调度 | Cosine + warmup 150 steps |
| DDP | torchrun，find_unused_parameters=True |
| 数据 | 从预处理 HDF5 加载手部姿态，归一化到 [-1, 1] |
| 码本统计 | 每 epoch 记录 per-group code usage 柱状图 |

#### Stage 2: 码本提取与 PCA 重排序 (`eval_vqvae.py`)

```python
# 核心算法 (lines 88-111)
codebooks = policy.module.vq_layer.codebooks  # (L, N, D) = (2, 4, 256)

actions = []
for i in range(4):       # codebook_size
    for j in range(4):
        latent = codebooks[0,i] * 0.5 + codebooks[1,j] * 0.5    # 硬编码 0.5 权重
        action = decoder_action(policy.module.get_action_from_latent(latent))
        # decoder_action: clip((action_norm+1)/2*65535, 0, 65535)
        actions.append(action)

actions_array = np.array(actions)               # (16, 6)
pca = PCA(n_components=1)
proj_1d = pca.fit_transform(actions_array.reshape(16, 6))[:, 0]
sorted_index = np.argsort(proj_1d)
sorted_hand_actions = actions_array[sorted_index]  # (16, 6) in raw [0, 65535]
np.save(codebook_path, sorted_hand_actions)
```

**注意**：PCA 使用硬编码 0.5 权重，而非训练后的 `softmax(layer_weights)`。这是刻意的——官方代码保证提取时的一致性，不依赖训练可能偏移的权重。

#### Stage 3: DQ-RISE 联合训练 (`train_dqrise.py`)

```python
# 训练标签生成
distances = cdist(hand_pose, code_book_actions)    # L2 in raw [0,65535] space
indices = argmin(distances) / 15 * 2 - 1           # [0,15] → [-1,1]
action = concat([tcp(9), indices(1)])              # 10-dim training target

# 扩散训练
loss = policy(cloud, action)                       # epsilon-prediction MSE
```

| 参数 | 值 |
|---|---|
| 训练 epochs | 1000 |
| Batch size | 240 (总) |
| 优化器 | AdamW，lr=3e-4，weight_decay=1e-6 |
| LR 调度 | Cosine + warmup 2000 steps |
| DDP | torchrun，find_unused_parameters=True |
| 数据增强 | 3D 空间增强（平移 + 旋转）+ HSV 颜色抖动 |
| 推理 | 20 DDIM steps，action horizon=20 |

### 2.4 Baseline 策略对比

官方代码库实现了 4 种策略变体：

| 策略 | 文件 | 手部空间 | 臂预测 | 手预测 | 联合建模 |
|---|---|---|---|---|---|
| **RISE** | `policy.py` | 连续 6-DOF | 统一扩散 | 统一扩散 | ✅ 隐式 |
| **RISE-S** | `separate_diff_baseline_policy.py` | 连续 6-DOF | 独立扩散 | 独立扩散 | ❌ 两个 UNet，loss=(L_tcp+L_hand)/2 |
| **DQ-RISE-C** | `seperate_diff_vae_baseline_policy.py` | 离散 16 类 | 扩散 | 分类 (仅 obs) | ❌ loss=0.5×L_tcp+0.5×L_hand_ce |
| **TCP-Cond.** | `tcp_cond_handPose_policy.py` | 离散 16 类 | 扩散 | 分类 (obs+TCP) | 🟡 TCP 特征经 MLP 后与 readout 融合 |
| **DQ-RISE** | `policy.py` (train_dqrise 模式) | 连续索引 (松弛) | 联合扩散 | 联合扩散 | ✅ 同梯度流 |

### 2.5 关键硬编码常量

| 常量 | 位置 | 值 |
|---|---|---|
| `HAND_DIM` | `utils/constants.py` | 6 |
| `TCP_DIM` | `utils/constants.py` | 9 (3 trans + 6 rot) |
| `num_action` | `policy/policy.py` | 20 (预测步数) |
| `voxel_size` | 多处 | 0.005m |
| `down_dims` | `diffusion.py:19` | [256, 512] |
| `loss_weight` | `vqvae.py:189` | [1.0, 1.0, 1.0, 0.5, 0.5, 1.0] |
| `layer_weights` init | `residual_vq.py:49` | [0.5, 0.5] (softmax normalized) |
| `act_scale` | `vqvae.py` | 1.0 |
| `DDIM num_train_steps` | `diffusion.py` | 100 |
| `DDIM num_inference_steps` | `diffusion.py` | 20 |
| `vqvae loss multipliers` | `train_vqvae.py:158` | enc×3 + vq×5 |
| `num_epochs` (vqvae) | `train_vqvae.py` | 1500 |
| `num_epochs` (dqrise) | `train_dqrise.py` | 1000 |
| VQ-VAE `lr` | 训练脚本 | 3e-4, betas=[0.95, 0.999] |
| 连续索引归一化 | `train_dqrise.py:161-162` | `idx / (TCP_DIM+HAND_DIM) * 2 - 1` |

**巧合说明**：`action_dim=15 = num_codes-1=15`，所以 `idx/15*2-1` 和 `idx/(action_dim)*2-1` 在官方配置下等价。但这依赖于 `codebook_size^groups == action_dim+1` 的数值巧合（4²=16=15+1），在扩展到 12-DOF hand 时不再成立。

---

## 3. 本项目实现与官方代码的差异

### 3.1 动作空间

这是最根本的差异——不同的灵巧手硬件导致不同的动作维度。

| 维度 | 官方 DQ-RISE | 本项目 (DexMani) |
|---|---|---|
| 硬件手 | OyMotion RoHand (6-DOF) | XHand (12-DOF) |
| 臂控制模式 | 末端执行器位姿 (9-dim) | **joint** 或 **end-effector** 两种 |
| joint 模式 action_dim | — | 19 = **7 个臂关节角 + 12 个手部关节** |
| EE 模式 action_dim | 15 = 9 + 6 | 21 = 9 (pos3+rot6d6) + 12 |
| 扩散输出维数 | 10 (9+1) | EE: 10 (9+1) / joint: 8 (7+1) |

> **⚠️ 命名澄清**：配置中 `tcp_dim` 在 `action` (joint) 模式下取值为 7，它实际是**机械臂的 7 个关节角**，不是 TCP 位姿。在 `action_ee` 模式下 `tcp_dim=9`，它是末端执行器位姿（pos3+rot6d6）。文档中避免笼统说"TCP 维度"。

### 3.2 感知架构 ★最大差异

```
官方 DQ-RISE:
  RGB-D (2× D415) → Open3D 点云 → 坐标变换 + 裁剪 + 降采样
    → Minkowski Sparse Tensor (voxel 0.005m)
    → Sparse3DEncoder (ResNet14, 6→512) + 3D Sine PosEnc
    → (B, max_token=100, 512)
    → DETR Transformer (4enc+1dec, 1 readout token, cross-attn)
    → (B, 512) readout → FiLM → ConditionalUnet1D
    → [arm + VQ_idx] × 20 steps

本项目 (DexMani DQ-RISE):
  Zarr PC (预计算, 1024 points via FPS, xyz+rgb 6d)
    → iDP3/PointNeXT + StateMLP
    → (B, out_dim×2) global cond → FiLM → ConditionalUnet1D
    → [arm_ctrl + VQ_idx] × 16 steps
```

| 差异维度 | 官方 | 本项目 | 影响 |
|---|---|---|---|
| 点云表示 | MinkowskiEngine 稀疏体素 | 稠密 FPS 采样 1024 点 | 官方需 MinkowskiEngine（安装复杂），本项目无额外依赖 |
| 3D Backbone | Sparse ResNet14 | iDP3 / PointNeXT | 不同的归纳偏置，官方稀疏卷积对稀疏数据更高效 |
| 中间层 | DETR Transformer (cross-attn readout) | 无（编码器输出直接作为 cond） | 官方 Transformer 可建模点间关系，本项目更简洁 |
| 条件注入 | readout token → FiLM | obs cond → FiLM | 两者都是全局 FiLM 条件，机制相同 |
| Action horizon | 20 steps | 16 steps | 项目统一使用 horizon=16 |

**差异原因**：
1. 官方需要真机实时推理 → MinkowskiEngine 稀疏卷积效率高
2. 本项目仿真环境点云已预处理为稠密格式 → 复用 DP3 的点云 encoder 基础设施
3. 跳过 Transformer 减少参数量和训练复杂度，但可能损失跨点建模能力

### 3.3 VQ-VAE

| 维度 | 官方 | 本项目 |
|---|---|---|
| hand_dim | 6 | 12 (action_ee) / 12 (action, 与臂控制模式无关) |
| layer_num (MLP 深度) | 1 (默认) | **2** (实测 1 层欠拟合 → vq_idx_used=9/16) |
| 手指权重 | 硬编码 `[1,1,1,0.5,0.5,1]` | 可配置，默认 `[1,1,1, 1,1,1, 1,1, 0.5,0.5, 0.5,0.5]` |
| 优化器 | 内置 Adam（代码中）+ 外部 AdamW（训练时） | 训练脚本 AdamW（无内置优化器） |
| EncoderMLP 初始化 | 正交初始化 (`weights_init_encoder`) | 正交初始化 (相同模式) |
| 训练 epoch | 1500 | 1500 (可配) |

**num_layers=2 的调优历程**：
- layer_num=1（匹配官方）→ vq_idx_used=9/16，欠拟合，下游 28%
- layer_num=5（加深）→ 过拟合坍缩 → vq_idx_used=4/16，下游 0%
- **layer_num=2 → vq_idx_used=13/16，下游 60%（★甜点）**

### 3.4 CodebookManager 独创封装

官方代码在 `eval_vqvae.py` 中以脚本方式完成 PCA 重排序（内联循环 + `np.save`），没有抽象。

本项目封装了 **`CodebookManager`** 类 (`dexmani_policy/agents/vq_hand/codebook_manager.py`)，提供：

| 功能 | 说明 |
|---|---|
| `extract_from_vqvae()` | 从训练好的 VqVaeHand 提取 codebook 向量 + layer_weights |
| `reindex_by_pca()` | 枚举所有 K^G 组合 → 解码 → PCA 排序（**硬编码 0.5 权重**，匹配官方 `eval_vqvae.py:96`） |
| `save()` / `load()` | `.npz` 格式（含 metadata 校验：hand_dim, num_groups, codebook_size） |
| `continuous_index_to_hand_pose()` | 推理查表：连续索引 → hand pose |
| `hand_pose_to_continuous_index()` | 训练标签生成：hand pose → 连续索引 |
| `build_per_group_codebooks()` | Per-group 独立码本（为多索引实验保留） |
| v2 raw-space 格式 | 手部姿态存储在 raw `[0, 65535]` 空间，API 边界做归一化转换 |
| 向后兼容 | 支持加载官方 `.npy` 格式和 v1 legacy 格式（自动转换） |

**动机**：
- 训练-推理分离：VQ-VAE 训练后提取码本 → DQ-RISE 训练时直接加载，不依赖 VQ-VAE 模型
- Checkpoint 一致性：`.npz` 文件携带 metadata，加载时校验配置不匹配
- 可测试性：独立于 VQ-VAE 模型，可单独单元测试

### 3.5 归一化与训练流程

| 维度 | 官方 | 本项目 |
|---|---|---|
| 动作归一化 | per-task 硬编码 TRANS_MIN/MAX | LinearNormalizer `limits` 模式，全量数据拟合 |
| 手部姿态归一化 | `clip((x+1)/2*65535, 0, 65535)` | 同 official raw servo space `[0, 65535]` |
| DDP 方式 | `torchrun` | `mp.spawn` (与 DP3/DP 统一) |
| find_unused_parameters | `True` (VQ-VAE 编码器的条件使用) | `False` (VQ-VAE 冻结，所有参数都参与梯度) |
| Batch size | 240 (总) | 128 (单卡) |
| 数据格式 | HDF5 + .npy 点云文件 | Zarr（预合并，直接加载） |
| 数据增强 | 3D 空间 + HSV 颜色抖动 | PC coord noise + color jitter + state noise（Hydra 配置） |

### 3.6 保持一致的设计

以下设计严格忠实于官方 DQ-RISE：

- **ResidualVQ 结构**: 2 groups × 4 codes = 16 组合，EMA codebook update (decay=0.8)
- **PCA 重排序算法**: 硬编码 0.5 权重枚举所有组合 → 解码 → 在手部状态空间做 PCA → 排序
- **连续松弛公式**: L2 最近邻（raw space）→ `idx/(K-1)*2-1` → Diff 输出 → round → 查表
- **DDIM 调度器**: 100 train / 20 inference，beta [1e-4, 0.02]，`squaredcos_cap_v2`，epsilon 预测
- **VQ-VAE 损失权重**: `encoder_loss × 3 + commitment_loss × 5`
- **训练超参**: AdamW lr=3e-4，betas=[0.95, 0.999]，weight_decay=1e-6，cosine schedule
- **三阶段管道**: VQ-VAE 预训练 → 码本提取+PCA 排序 → 联合扩散训练

### 3.7 架构对比图

```
═══════════════════════════════════════════════════════════
                    官方 DQ-RISE 前向
═══════════════════════════════════════════════════════════
  RGB-D → Open3D PC → Minkowski SparseConv(ResNet14)
    → [B, 100, 512] tokens + 3D PosEnc
    → DETR Transformer(4enc+1dec, x-attn)
    → [B, 512] readout
    → ConditionalUnet1D(FiLM) → [B, 20, 10]
    → split: TCP_pred(9) + idx_pred(1)
    → idx → round → codebook[raw_hand_pose]
    → cat(TCP_pred, hand_pred) → 15-dim action

═══════════════════════════════════════════════════════════
                  本项目 DQ-RISE 前向
═══════════════════════════════════════════════════════════
  Zarr PC(1024×6, preprocessed) → iDP3/PointNeXT + StateMLP
    → [B, out_dim×2] obs cond
    → ConditionalUnet1D(FiLM) → [B, 16, tcp_dim+1]
    → split: arm_ctrl_pred + idx_pred
    → CodebookManager.continuous_index_to_hand_pose(idx)
    → cat(arm_ctrl_pred, hand_pred) → full action
      (19-dim joint / 21-dim EE)
```

---

## 4. 实验踩坑记录与优化方向

### 4.1 60% 基线 (Pour 任务)

**最佳实验**: `experiments/dqrise/pour/2026-07-08_01-14_42`

**关键 VQ-VAE 参数**:

| 参数 | 值 | 备注 |
|---|---|---|
| `num_layers` | **2** | ★甜点：1→9码/28%，5→4码/0%，2→13码/60% |
| `vq_decay` | **0.8** | ★反直觉：0.99 压低利用率→32%，勿"稳定化" |
| `threshold_ema_dead_code` | **2** | ★必须 >0 才能复活死码 |
| `kmeans_init` | **true** | k-means 初始化 codebook |
| `num_groups` | 2 | 16 组合，pour 用 13 未饱和 |
| `codebook_size` | 4 | |
| `enc_loss_weight` | 3.0 | 与官方一致 |
| `vq_loss_weight` | 5.0 | 与官方一致 |

**主策略参数**: `down_dims=[256, 512]`（★2 层 UNet，加深→[256,512,1024] 掉分），`num_inference_steps=20`

**⚠️ 复现警告**: 原始 60% 码本（vq_idx_used=13）已丢失，无法字节级复现。当前重训码本 util=12（压线过 ≥12 门槛），下游成功率未验证。

### 4.2 决定性因子：vq_idx_used

`vq_idx_used` = 16 个 PCA 排序后码本组合中，训练数据实际命中（>1% 概率）的数量。

**与成功率的单调关系**（pour 任务）：

```
vq_idx_used: 13 → 60%  ← 基线
             10 → 32%
              9 → 28%
              8 → 32%
              4 →  0%  ← 码本坍缩
```

**VQ 质量判死门槛**: `vq_idx_used < 8` → **直接否决码本**，不浪费下游 250 epoch 训练 + 仿真评测。该值在 VQ-VAE 训练后即可计算（通过 `measure_vq_usage.py`），无需下游训练。

### 4.3 已验证无效的优化

以下优化在 pour 任务上经过实验验证，**导致成功率下降或持平**：

| 尝试 | 预期效果 | 实际结果 | 失败原因 |
|---|---|---|---|
| `decay=0.99` (从 0.8) | 更稳定的码本更新 | 32% 成功率 | EMA 过高 → 码本失去适应性 |
| EMA 预热 (start EMA after N steps) | 更好的初始阶段 | 码本坍缩 | 预热期内无动量 → 码本发散 |
| per-group 多索引预测 | 更丰富的表征 | **60→32%** | 破坏 PCA 单主轴连续性假设 |
| 加深 UNet (down_dims +1 层) | 更强表达能力 | 掉分 | 过拟合小数据集 |
| 加深 Encoder MLP (num_layers=5) | 更好 VQ 重建 | 0% | 过拟合 → 码本坍缩 (vq_idx_used=4) |
| val/loss 目标调参 | 更好收敛指标 | 成功率腰斩 | 指标-奖励错配（第一次教训） |

### 4.4 E1 教训：指标-奖励错配

**实验** (2026-07-09): vq_loss_weight sweep (3.0 / 2.0 / 1.0) 以最大化 `vq_idx_used`。

| 配置 | vq_idx_used | recon_mse | 下游成功率 |
|---|---|---|---|
| 基线 (vqw=5.0) | 13 | 0.013 | **60%** |
| vqw=3.0 | **14** (>基线) | 0.022 (+69%) | **16%** (~3×退化) |
| vqw=2.0 | 12 | 0.020 | — |
| vqw=1.0 | 12 | 0.025 | — |

**核心发现**: `vq_idx_used` 是好的**跨 run 预测器**，但**不能当独立优化目标去堆**。为铺满第 14 码牺牲了每码保真度（recon_mse 退化 69%）→ 手部姿态变粗糙 → 拖垮成功率。

**真正目标 = 利用率 × 每码保真度的联合**，不是利用率单值。这是指标-奖励错配的第二次教训（第一次是压 val/loss 腰斩成功率）。

### 4.5 VQ 质量诊断 Checklist

#### P0 硬门槛（任一不过即否决码本，不浪费下游训练）

| 指标 | 阈值 | 工具 |
|---|---|---|
| `vq_idx_used` (16 桶中 >1%) | **≥12** 健康；7-11 边缘；**≤6 判死** (~0% 成功率) | `dqrise.py:compute_loss` |
| 归一化熵 `entropy/ln(16)` | ≥0.75 健康；<0.5 异常 | `dqrise.py:compute_loss` |
| Per-group 活码数 (每组 4 码) | 每组 ≥3；单组坍缩→16 退化为 4 | `train_vq_hand.py` |
| `recon_mse` (scaled, unweighted) | **<0.03**（须与利用率联合看，低 MSE 可被码本坍缩伪造） | `measure_vq_usage.py` |
| 样本→码字 L2 p99/mean | **<3.0** | `analyze_vq_pour.py` Section 2 |

#### P1 质量门槛（表征够精细）

| 指标 | 阈值 |
|---|---|
| Per-finger 逐指 L1（拇/食/中/无名/小） | 各 <0.05，拇食优先 |
| 重建误差 / 关节范围比 | <2% 优；>10% 异常 |
| PCA 比率 PC1/std_min | <50×（>50 退化为 1-D 开合，丢指间协调） |
| 扩散 VQ 索引 top-1 acc（验证集） | >70%；<40% 问题在扩散头 |

#### Codebook 扩展触发条件

仅当以下两个条件**同时满足**时才考虑扩展码本容量：
1. `vq_idx_used == 16/16`（当前码本饱和）
2. `p99/mean > 3.0`（存在稀有手姿被吸附到远距离码字）

优先扩 `codebook_size`（4→6→8）而非 `num_groups`（2→3），后者加重扩散离散维联合误差。Pour 当前 13/16 未饱和 → 暂不扩展。

### 4.6 优化方向（按优先级）

#### P0 — 防止 Codebook 坍塌

1. **死码替换** `threshold_ema_dead_code=2`（已启用）
   - 当 cluster size < 2 时用 batch 中随机样本替换死码
   - 官方默认 0（禁用），可能是疏忽

2. **K-Means 初始化** `kmeans_init=true`（已启用）
   - 首次 forward 时对第一个 batch 跑 k-means 初始化 codebook

#### P1 — 可能带来收益

3. **Layer Weights 训练-推理一致性**
   - 训练时 `layer_weights` 是 softmax 可学习参数
   - PCA 重排序时硬编码 0.5（匹配官方）
   - 修复方案：将 `layer_weights` 改为固定 `[0.5, 0.5]` buffer（非可学习），或 PCA 时使用训练后的 softmax 值

4. **更大 MLP** `num_layers=3-5`
   - 仅当基线稳固且确认 12-DOF 欠拟合时尝试
   - 风险：过拟合 → 码本坍缩

#### P2 — 系统化监控

5. **Per-epoch codebook perplexity**
   - `perplexity = exp(-sum(p*log(p)))`，最大值 = codebook_size
   - 低 perplexity 是坍缩的早期信号，远比每 200 epoch 的柱状图灵敏

6. **推理手部姿态 jerk 监控**
   - 连续帧间 hand pose 变化平滑度 ≤ 2× 示范

#### P3 — 探索性

7. **Codebook 容量扩展**
   - 仅当饱和触发（见 4.5 触发条件）

8. **多任务共享码本**
   - 跨任务训练 VQ-VAE → 观察码本是否泛化到任务间的手势模式
   - 可能减少 per-task VQ-VAE 训练成本

---

## 5. 附录

### 5.1 术语速查

| 术语 | 说明 |
|---|---|
| VQ-VAE | Vector Quantized Variational Autoencoder — 将连续手部姿态量化为离散编码 |
| ResidualVQ | 残差向量量化 — 每层量化上一层的残差，多层叠加得到最终量化结果 |
| K=16 | `codebook_size^num_groups = 4^2 = 16` 种离散手部姿态组合 |
| PCA re-indexing | 在主成分方向排序 16 个码本条目，实现从离散到连续的松弛 |
| Continuous relaxation | 通过 PCA 排序使离散编码可被扩散模型作为连续值 [-1,1] 预测 |
| Joint diffusion | 臂部动作和连续手部索引在同一个扩散头中联合去噪 |
| vq_idx_used | 16 个 PCA 排序码本中训练数据实际命中（>1%）的数量 — 最强单一预警指标 |
| Metric-reward mismatch | 指标（val loss / vq_idx_used）改善但下游成功率下降 |

### 5.2 关键文件索引

**本项目 (DexMani_Policy)**:

| 文件 | 内容 |
|---|---|
| `dexmani_policy/agents/core/dqrise.py` | DQRISEAgent — 主策略类 |
| `dexmani_policy/agents/vq_hand/vqvae.py` | VqVaeHand — VQ-VAE 模型 |
| `dexmani_policy/agents/vq_hand/codebook_manager.py` | CodebookManager — 码本 PCA + 查表 |
| `dexmani_policy/agents/vq_hand/residual_vq.py` | ResidualVQ — 残差向量量化（移植自官方） |
| `dexmani_policy/agents/vq_hand/vector_quantize.py` | VectorQuantize — 底层 VQ 组件 |
| `dexmani_policy/configs/dqrise.yaml` | Hydra 训练配置（含 VQ-VAE 段） |
| `dexmani_policy/scripts/train_vq_hand.py` | VQ-VAE 预训练入口 |
| `dexmani_policy/scripts/extract_codebook.py` | 码本提取脚本 |
| `dexmani_policy/scripts/measure_vq_usage.py` | VQ 利用率检测工具 |
| `dexmani_policy/scripts/analyze_vq_pour.py` | Pour 任务 VQ 深度分析工具 |
| `dexmani_policy/scripts/analyze_hand_joints.py` | 手部关节→指尖映射分析 |
| `dexmani_policy/scripts/analyze_hand_data.py` | 手部数据综合统计分析 |
| `dexmani_policy/smoke_test.py` | 冒烟测试（含 dqrise 模式） |
| `scripts/train_vq_hand.sh` | VQ-VAE 训练 Shell 脚本 |
| `scripts/train_ddp.sh` | DDP 训练脚本（支持 `ddp/dqrise`） |

**官方 DQ-RISE** (`/home/zhanghaoyang/Desktop/DQ-RISE/`):

| 文件 | 内容 |
|---|---|
| `policy/policy.py` | RISE 主策略类 |
| `policy/tokenizer.py` | Sparse3DEncoder + 3D PosEnc |
| `policy/transformer.py` | DETR Transformer |
| `policy/diffusion.py` | DiffusionUNetPolicy |
| `policy/vqvae_rise/vqvae.py` | VqVae + EncoderMLP |
| `policy/vqvae_rise/vector_quantize_pytorch/residual_vq.py` | ResidualVQ |
| `train_vqvae.py` | VQ-VAE 训练入口 |
| `eval_vqvae.py` | 码本提取 + PCA |
| `train_dqrise.py` | DQ-RISE 训练入口 |
| `utils/constants.py` | 全局常量 (TCP_DIM=9, HAND_DIM=6) |

### 5.3 官方代码已知问题

1. `preprocess_data.py:67`: 循环硬编码为 `range(5)`，应处理全部数据
2. `dataset/riseVAE.py:74`: 标定路径硬编码为 `data/task_0006/calib/1753091226804`
3. `policy/separate_diff_baseline_policy.py:52`: `torch.concatenate` 应为 `torch.cat`
4. `policy/vqvae_rise/dataset/rohand.py`: 数据集骨架类不完整
5. `eval_rise_vae.py:19`: 导入外部未包含的依赖 `baseline.RISE.eval_agent`
6. `policy/vqvae_rise/vqvae.py:253`: `import pdb; pdb.set_trace()` 残留调试断点

### 5.4 参考文献

```bibtex
@article{feng2025learning,
  title     = {Learning Dexterous Manipulation with Quantized Hand State},
  author    = {Feng, Ying and Fang, Hongjie and He, Yinong and Chen, Jingjing
               and Wang, Chenxi and He, Zihao and Liu, Ruonan and Lu, Cewu},
  journal   = {arXiv preprint arXiv:2509.17450},
  year      = {2025},
  note      = {ICRA 2026}
}

@inproceedings{wang2024rise,
  title     = {RISE: 3D Perception Makes Real-World Robot Imitation Simple and Effective},
  author    = {Wang, Chenxi and Fang, Hongjie and Fang, Hao-Shu and Lu, Cewu},
  booktitle = {IEEE/RSJ IROS},
  year      = {2024},
  pages     = {2870--2877}
}
```

---

> **文档版本**: v2.0 | **创建**: 2026-07-09 | **取代**: docs/10, 11, 12, 13（已删除）
