# FAAS 集成方案：将 UniDex 统一手部动作空间引入 DexMani

> **FAAS**: Function-Actuator-Aligned Space，UniDex (CVPR 2026) 提出的跨灵巧手统一动作空间
> **目标**: 在 DexMani_Policy 中以最小代价引入 FAAS，为多手数据训练、跨手泛化、UniDex 数据复用奠定基础
> **前置阅读**: `docs/UniDex-知识体系.md`（§5 FAAS 统一动作空间）

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [现状分析](#2-现状分析)
3. [FAAS 核心机制](#3-faas-核心机制)
4. [XHand ↔ FAAS 精确映射](#4-xhand--faas-精确映射)
5. [集成方案设计](#5-集成方案设计)
6. [关键决策分析](#6-关键决策分析)
7. [实施路线图](#7-实施路线图)
8. [风险与缓解](#8-风险与缓解)
9. [附录](#9-附录)
   - 9.1 [关键文件索引](#91-关键文件索引)
   - 9.2 [术语对照](#92-术语对照)
   - 9.3 [参考资料](#93-参考资料)
   - 9.4 [UniDex vs DexMani Normalizer 对比](#94-unidex-vs-dexmani-normalizer-对比)
   - 9.5 [腕部动作表示差异](#95-腕部动作表示差异)
   - 9.6 [Agent 兼容性矩阵](#96-agent-兼容性矩阵)
   - 9.7 [DDP 兼容性验证总结](#97-ddp-兼容性验证总结)

---

## 1. 背景与动机

### 1.1 当前问题

DexMani_Policy 目前仅支持 **XHand** 单手操作，动作空间固定为：

| 模式 | 总维度 | 手臂 | 手部 | 手部关节含义 |
|------|--------|------|------|-------------|
| `action` (joint) | 19D | 7D 臂关节角 | 12D XHand 关节 | 隐式（仅靠索引顺序约定） |
| `action_ee` (end-effector) | 21D | 9D 末端位姿 | 12D XHand 关节 | 同上 |

这带来三个局限：

1. **无跨手泛化能力** — 模型将 12 个手部维度视为无结构平坦向量，无法理解"第 7 维是食指近端关节"这一事实
2. **无法复用外部数据** — UniDex 的 50K+ 轨迹分布在 8 种手上，格式为 FAAS 82D，无法直接注入 DexMani 训练
3. **扩展成本高** — 若未来支持第二只手，需重新设计动作空间、修改模型架构、重新训练

### 1.2 FAAS 的价值

FAAS 定义了一个 **32 维功能对齐的手部关节空间**，核心思想是：

> 不同灵巧手的**拇指屈曲关节**虽然名称不同、索引不同、运动学不同，但**功能相同**。将它们映射到 FAAS 的**同一索引**，模型就能学到"拇指屈曲"这一抽象概念，而非"XHand 第 0 号关节"。

UniDex 已验证的效果：
- 仅在 **Inspire Hand** 上训练，零样本迁移到 **Wuji (20DoF)** 成功率达 40%
- 仅在 **Inspire Hand** 上训练，零样本迁移到 **Oymotion (11DoF)** 成功率达 60%
- 基线方法（π₀、DP3）在未见手上几乎完全失效（0-10%）

### 1.3 文档目标

本方案的设计原则：

1. **最小侵入** — 不破坏现有架构，FAAS 作为可选开关（`use_faas: true/false`）
2. **向后兼容** — 提供 checkpoint 迁移工具，历史权重可转换
3. **渐进实施** — 分 Phase 推进，每个 Phase 独立可验证

---

## 2. 现状分析

### 2.1 DexMani 动作数据流

```
Zarr (N, 19)
  │  data/action: float32, [arm(7) | hand(12)]
  │  data/joint_state: float32, [arm(7) | hand(12)]
  ▼
BaseDataset.sample_to_data()
  │  提取 action_key (action | action_ee)
  │  可选: use_aux_ee → 拼接 ee_pose 到 action
  ▼
LinearNormalizer (mode='limits')
  │  全量拟合 → [-1, 1]
  ▼
Agent.compute_loss()
  │  拆分 arm/hand (仅 DQ-RISE)
  │  Diffusion/FlowMatch 去噪
  ▼
Agent.predict_action()
  │  DDIM/Euler 采样 → unnormalize
  ▼
SimRunner / SimEvaluator
  │  env.step(action_19d)
  ▼
XArm7_XHand.apply_action()
  │  mapping[] 用户序 → SAPIEN 序
  │  set_drive_target() 逐关节
```

### 2.2 XHand 关节定义

**DexMani_Sim 定义** (`dexmani_sim/robots/_urdf_config.py`):

```
索引  关节名                        类型      范围              手指
─────────────────────────────────────────────────────────────────────
 0    right_hand_thumb_bend_joint   revolute  [0, 1.832]       拇指
 1    right_hand_thumb_rota_joint1  revolute  [-0.698, 1.57]   拇指
 2    right_hand_thumb_rota_joint2  revolute  [0, 1.57]        拇指
 3    right_hand_index_bend_joint   revolute  [-0.174, 0.174]  食指
 4    right_hand_index_joint1       revolute  [0, 1.919]       食指
 5    right_hand_index_joint2       revolute  [0, 1.919]       食指
 6    right_hand_mid_joint1         revolute  [0, 1.919]       中指
 7    right_hand_mid_joint2         revolute  [0, 1.919]       中指
 8    right_hand_ring_joint1        revolute  [0, 1.919]       无名指
 9    right_hand_ring_joint2        revolute  [0, 1.919]       无名指
10    right_hand_pinky_joint1       revolute  [0, 1.919]       小指
11    right_hand_pinky_joint2       revolute  [0, 1.919]       小指
```

**关键特征**：
- 拇指 3 关节（bend + 2 rota），其余四指各 2 关节（无远端关节）
- `index_bend_joint` 轴为 `(-1, 0, 0)`，范围极小（±0.174 rad ≈ ±10°），本质上是食指的侧摆自由度
- 中指/无名指/小指关节轴全部为 `(0, 1, 0)`（纯屈曲），且范围完全相同 [0, 1.919]
- **无欠驱动联动**（与 Inspire 不同），12 个关节全部独立驱动

---

## 3. FAAS 核心机制

### 3.1 空间定义

```
82D FAAS = 右手腕(9D) + 左手腕(9D) + 右手关节(32D) + 左手关节(32D)

手腕 9D  = pos(3) + rot6d(6)           # 6D 连续旋转表示 [Zhou et al. 2019]
单手 32D = 27 功能关节 + 5 预留槽       # MAPPED_JOINT_DIM=32, JOINT_DIM_IN_USE=27
```

**DexMani 仅需右手**，即关注 `41D = 右手腕(9D) + 右手关节(32D)`（`action_ee` 模式）或 `39D = 臂关节(7D) + 右手关节(32D)`（`action` 模式）。

### 3.2 FAAS 32D 的语义结构

FAAS 索引的组织规律：**按手指分组，每指 5 个槽位（但大多手只填 2-4 个）**。

```
FAAS 索引范围   手指     功能语义                         XHand 占用
──────────────────────────────────────────────────────────────
 0-4           拇指     CMC_Abd, CMC_Flex, MCP, Int, Dist    [1,2,3]
 5-9           食指     Abd_base, Spread, MCP, PIP, DIP      [6,7,8]
10-14          中指     Abd_base, Spread, MCP, PIP, DIP      [12,13]
15-19          无名指   Abd_base, Spread, MCP, PIP, DIP      [17,18]
20             空位     (所有手均不使用)
21-25          小指     Spread, MCP, PIP, DIP, Extra         [22,23]
26             拇指Extra (Shadow THJ3)
27-31          预留     (手部特有额外关节)
```

**为什么 XHand 的索引不连续？** 因为 FAAS 按**功能完备性**定义（Shadow Hand 22DoF 几乎填满），XHand 缺少的远端关节（Distal）和侧摆关节（Abd_base/Spread）在 FAAS 中留空。

### 3.3 FAAS 的关键不变量

1. **功能对齐** — 所有 8 种手的拇指屈曲关节都映射到 FAAS[1]，无论原生关节名叫什么
2. **零填充** — 未使用的 FAAS 索引恒为零，模型自然学会忽略
3. **符号约定** — retarget scale 统一关节旋转的正方向（旋转正方向因 URDF 定义而异）
4. **归一化独立** — 每个 FAAS 维度独立归一化，零填充维度 `scale=1.0, offset=0.0`（恒等映射），与 rot6d 处理一致

---

## 4. XHand ↔ FAAS 精确映射

### 4.1 正向映射：Native 12D → FAAS 32D

```
DexMani 索引  XHand 关节名               Scale   →  FAAS 索引  FAAS 功能语义
─────────────────────────────────────────────────────────────────────────────
 0            thumb_bend_joint            1.0     →  [1]        Thumb CMC Flexion
 1            thumb_rota_joint1           1.0     →  [2]        Thumb MCP Pitch
 2            thumb_rota_joint2           1.0     →  [3]        Thumb Intermediate
 3            index_bend_joint           -1.0 ⚠️  →  [6]        Index Spread
 4            index_joint1                1.0     →  [7]        Index Proximal (MCP)
 5            index_joint2                1.0     →  [8]        Index Intermediate (PIP)
 6            mid_joint1                  1.0     → [12]        Middle Proximal (MCP)
 7            mid_joint2                  1.0     → [13]        Middle Intermediate (PIP)
 8            ring_joint1                 1.0     → [17]        Ring Proximal (MCP)
 9            ring_joint2                 1.0     → [18]        Ring Intermediate (PIP)
10            pinky_joint1                1.0     → [22]        Pinky Proximal (MCP)
11            pinky_joint2                1.0     → [23]        Pinky Intermediate (PIP)
```

**Scale 矩阵**（来自 `hand_utils.json`）:

```python
# 仅 index_bend_joint 需要符号翻转，其余均恒等
NATIVE_TO_FAAS_SCALES = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

### 4.2 逆向映射：FAAS 32D → Native 12D

逆向是正向的逆操作：从 32 个 FAAS 索引中 gather 出 12 个活跃位置，再乘以相同的 scale 向量（因为 scale∈{1,-1}，`scale⁻¹ = scale`）。

### 4.3 符号翻转的验证方法

`index_bend_joint` 的 scale=-1 需要通过实验验证。验证方案：

1. 从 DexMani Zarr 取一段轨迹，提取手部关节值
2. 正向映射到 FAAS 32D，再逆向映射回 12D
3. 检查 `roundtrip_error = |native - inverse(faas(native))|` 是否 < 1e-7
4. 单独检查 `index_bend_joint` 维度：确认 `faas_dim_6 = -native_dim_3`

### 4.4 各手 FAAS 覆盖率对比

```
手        原生DoF   FAAS占用数   FAAS占用率   主要缺失
──────────────────────────────────────────────────────
Shadow     22      22/32       68.8%        几乎填满
Wuji       20      20/32       62.5%
Allegro    16      16/32       50.0%        无小指 (FAAS[21-25])
Leap       16      16/32       50.0%        无小指 (FAAS[21-25])
Xhand      12      12/32       37.5%        无远端+侧摆关节
Inspire    12      12/32       37.5%        与 XHand 类似
Oymotion   11      11/32       34.4%        缺 thumb_rota2
Ability    10      10/32       31.3%        最少关节数
```

---

## 5. 集成方案设计

### 5.1 架构总览

```
                          训练数据流
┌─────────────────────────────────────────────────────────────┐
│  Zarr (N, 19)                                               │
│    │  data/action: [arm(7) | hand_native(12)]               │
│    ▼                                                        │
│  BaseDataset.sample_to_data()                               │
│    │  拆分 arm/hand                                        │
│    │  FAASHandMapper.native_to_faas(hand_12d) → hand_32d   │
│    │  cat([arm_7d, faas_hand_32d]) → action_39d            │
│    ▼                                                        │
│  LinearNormalizer (limits, 39D)                             │
│    │  活跃维度: scale/offset 正常拟合                         │
│    │  补零维度: scale=1.0, offset=0.0 (恒等映射)              │
│    ▼                                                        │
│  Agent.compute_loss()                                       │
│    │  Diffusion/FlowMatch 在 39D FAAS 空间去噪               │
│    │  可选: sparse loss mask (仅对 19 个活跃维度计算 MSE)     │
│    ▼                                                        │
│  optimizer.step()                                           │
└─────────────────────────────────────────────────────────────┘

                          推理数据流
┌─────────────────────────────────────────────────────────────┐
│  Agent.predict_action()                                     │
│    │  Diffusion/FlowMatch 采样 → 39D FAAS action             │
│    │  unnormalize                                           │
│    ▼                                                        │
│  拆分: arm(7D) + faas_hand(32D)                             │
│    │  FAASHandMapper.faas_to_native(hand_32d) → hand_12d   │
│    │  cat([arm_7d, hand_12d]) → action_19d                 │
│    ▼                                                        │
│  env.step(action_19d)  ← 仿真器接口不变                      │
└─────────────────────────────────────────────────────────────┘
```

**核心原则：模型在 FAAS 空间训练和推理，仅在 I/O 边界做格式转换。仿真器接口完全不变。**

### 5.2 新增模块：`common/faas_mapper.py`

```python
"""
FAAS (Function-Actuator-Aligned Space) mapper for DexMani_Policy.

Maps between native XHand joint space (12D) and UniDex FAAS unified hand
joint space (32D). The mapping is derived from UniDex's hand_utils.json.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FAASHandMapper(nn.Module):
    """XHand native joint space ↔ FAAS unified hand space.

    FAAS is a 32-dim functional alignment space where each index represents
    a specific joint role (e.g., index [1] = thumb CMC flexion across ALL
    hand types). XHand uses only 12 of 32 indices; the remaining 20 are
    zero-padded and the model learns to ignore them.

    This is an nn.Module so it can be included in checkpoint state, but
    all its parameters are buffers (non-trainable).
    """

    MAPPED_JOINT_DIM: int = 32
    JOINT_DIM_IN_USE: int = 27
    NATIVE_HAND_DIM: int = 12

    # native XHand index → FAAS index (ordered per DexMani joint convention)
    _NATIVE_TO_FAAS_INDICES: tuple = (1, 2, 3, 6, 7, 8, 12, 13, 17, 18, 22, 23)

    # Sign corrections: index_bend_joint rotates opposite direction in FAAS
    _NATIVE_TO_FAAS_SCALES: tuple = (1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # Per-joint offsets (all zero for XHand; non-zero for Inspire, etc.)
    # Formula (matching UniDex _apply_scale_shift): faas_value = native * scale + offset
    _NATIVE_TO_FAAS_OFFSETS: tuple = (0.0,) * 12

    def __init__(self):
        super().__init__()
        idx = torch.tensor(self._NATIVE_TO_FAAS_INDICES, dtype=torch.long)
        scales = torch.tensor(self._NATIVE_TO_FAAS_SCALES, dtype=torch.float32)
        offsets = torch.tensor(self._NATIVE_TO_FAAS_OFFSETS, dtype=torch.float32)
        self.register_buffer('_faas_indices', idx, persistent=True)
        self.register_buffer('_scales', scales, persistent=True)
        self.register_buffer('_offsets', offsets, persistent=True)

    def native_to_faas(self, native_hand: torch.Tensor) -> torch.Tensor:
        """12D native XHand → 32D FAAS (zero-padded on unmapped indices).

        Transformation (matching UniDex _apply_scale_shift):
            faas_value = native_value * scale + offset
        For XHand, all offsets are 0.0 (hand_utils.json verified).

        Args:
            native_hand: (..., 12) in DexMani XHand joint order.
        Returns:
            (..., 32) in FAAS order.
        """
        assert native_hand.shape[-1] == self.NATIVE_HAND_DIM, \
            f"Expected last dim {self.NATIVE_HAND_DIM}, got {native_hand.shape[-1]}"
        shape = native_hand.shape[:-1]
        transformed = native_hand * self._scales + self._offsets
        faas = native_hand.new_zeros(*shape, self.MAPPED_JOINT_DIM)
        faas[..., self._faas_indices] = transformed
        return faas

    def faas_to_native(self, faas_hand: torch.Tensor) -> torch.Tensor:
        """32D FAAS → 12D native XHand.

        Inverse transformation: native = (faas_value - offset) / scale
        Since scale ∈ {1, -1} for XHand: native = faas_value * scale - offset*scale

        Args:
            faas_hand: (..., 32) in FAAS order.
        Returns:
            (..., 12) in DexMani XHand joint order.
        """
        assert faas_hand.shape[-1] == self.MAPPED_JOINT_DIM, \
            f"Expected last dim {self.MAPPED_JOINT_DIM}, got {faas_hand.shape[-1]}"
        native = faas_hand[..., self._faas_indices]
        # Inverse: (x - off) / s = x/s - off/s = x*s - off*s (since s ∈ {1,-1})
        return native * self._scales - self._offsets * self._scales

    def get_active_mask(self) -> torch.Tensor:
        """Return bool mask of shape (32,) — True where XHand has a joint."""
        mask = torch.zeros(self.MAPPED_JOINT_DIM, dtype=torch.bool)
        mask[self._faas_indices] = True
        return mask

    def get_active_count(self) -> int:
        """Return number of active FAAS dimensions for XHand (=12)."""
        return self.NATIVE_HAND_DIM
```

### 5.3 配置变更

**所有策略 YAML**（`dp.yaml`, `dp3.yaml`, `maniflow.yaml`, `moe_dp.yaml`, `r3d.yaml`, `dqrise.yaml`）:

```yaml
# === 新增: FAAS 开关 ===
use_faas: true                                   # 启用 FAAS 手部动作空间
faas_hand_dim: 32                                # FAAS 单手维度

# === 修改: 动作空间维度 ===
# 原值:
#   action_dim: ${eval:'21 if ${eq:${action_key},action_ee} else 19'}
#   state_dim: 19
# 新值:
action_dim: ${eval:'41 if ${eq:${action_key},action_ee} else 39'}
state_dim: ${eval:'41 if ${eq:${action_key},action_ee} else 39'}
# state_dim 也需要扩大: joint_state 观测同样需要 FAAS 转换

# === 保持不变的字段 ===
tcp_dim: ${eval:'9 if ${eq:${action_key},action_ee} else 7'}  # 臂部分不变
hand_dim: 32                                         # 从 12 改为 32
```

**向后兼容**: `use_faas: false` 时所有维度保持原值，行为与现有代码完全一致。

### 5.4 数据管线变更

**`datasets/base_dataset.py`** — `sample_to_data()` 尾部追加：

```python
def sample_to_data(self, sample):
    # ... 现有逻辑（增强、normalize 等） ...

    if self.use_faas:
        data = self._apply_faas_mapping(data)
    return data

def _apply_faas_mapping(self, data: dict) -> dict:
    """Convert native hand joint space to FAAS space."""
    arm_dim = self.tcp_dim  # 7 (joint) or 9 (ee)

    # Action: [arm(tcp_dim) | hand(12)] → [arm(tcp_dim) | FAAS_hand(32)]
    arm_action = data['action'][..., :arm_dim]
    hand_action = data['action'][..., arm_dim:]
    faas_hand = self.faas_mapper.native_to_faas(hand_action)
    data['action'] = torch.cat([arm_action, faas_hand], dim=-1)

    # Joint state: same mapping
    if 'joint_state' in data:
        arm_state = data['joint_state'][..., :arm_dim]
        hand_state = data['joint_state'][..., arm_dim:]
        faas_hand_state = self.faas_mapper.native_to_faas(hand_state)
        data['joint_state'] = torch.cat([arm_state, faas_hand_state], dim=-1)

    return data
```

### 5.5 Agent 推理变更

**`agents/core/base.py`** — `predict_action()` 返回前追加：

```python
def predict_action(self, obs_dict, denoise_timesteps=None):
    # ... 现有逻辑: normalize obs, denoise diffusion, unnormalize ...

    if self.use_faas:
        arm_dim = self.tcp_dim
        arm_action = pred[..., :arm_dim]
        faas_hand = pred[..., arm_dim:]
        native_hand = self.faas_mapper.faas_to_native(faas_hand)
        pred = torch.cat([arm_action, native_hand], dim=-1)

    # pred 恢复为 (B, n_action_steps, 19|21) → 仿真器无感知
    return pred
```

### 5.6 可选优化：Sparse Loss Mask

仅对活跃 FAAS 维度（19 个中的 19 个——等等，39D 中有 7 arm + 12 active hand + 20 padding = 19 active out of 39）计算 MSE：

```python
# 在 Agent.__init__() 中构建
if self.use_faas:
    arm_dim = self.tcp_dim
    hand_mask = self.faas_mapper.get_active_mask()  # (32,) bool
    full_mask = torch.cat([
        torch.ones(arm_dim, dtype=torch.bool),       # arm: always active
        hand_mask                                     # hand: 12 active + 20 padding
    ])                                                # (39,) or (41,)
    self.register_buffer('_faas_loss_mask', full_mask.float())

# 在 compute_loss() 中使用
if self.use_faas:
    dim_losses = F.mse_loss(pred, target, reduction='none')
    loss = (dim_losses * self._faas_loss_mask).sum() / self._faas_loss_mask.sum()
```

**建议**: 初始实现不启用 mask，让模型自然学习补零维度的零预测。仅在观测到补零维度振荡时启用。

---

## 6. 关键决策分析

### 6.1 为什么模型必须在 FAAS 空间训练（而非仅在数据边界转换）？

| 方案 | 模型 sees | 训练后补零维度 | 跨手扩展 |
|------|----------|---------------|---------|
| **A: 模型原生 FAAS** | 39D (20 dims 恒为零) | 自然学会预测零 | 添加新手数据即可，同维度 |
| B: 数据边界转换 | 19D (12 hand dims) | N/A | 需改模型架构、重新训练 |

**方案 A 的代价**：各 backbone 类型参数增量：

| Backbone | 使用策略 | 受影响的层 | 参数增量 | 总参数占比 |
|----------|---------|-----------|---------|-----------|
| ConditionalUnet1D | DP, DP3, MoE, DQRISE | `down_modules[0]` Conv1d + `final_conv` Conv1d | ~50K | <0.5% |
| DiTXFlowMatch | ManiFlow | `input_embedder` Linear + `final_layer` Linear | ~20K | <0.3% |
| DiTDiffusion | MultiTask | `x_embedder` Linear + `final_layer` Linear | ~20K | <0.3% |
| OneWayTransformer | R3D | 输入/输出投影层 | ~15K | <0.3% |

**可忽略**。

**方案 B 的代价**：每次加新手需重构模型。**不可接受**。

→ **选择方案 A**。

### 6.2 为什么使用 nn.Module 而非纯函数？

`FAASHandMapper` 作为 `nn.Module` 子类，其 `_faas_indices` 和 `_scales` 注册为 `persistent=True` 的 buffer，自动随 checkpoint 保存和加载。这确保了：

1. **checkpoint 自包含** — 加载旧模型时无需额外配置文件
2. **映射参数不可训练** — `requires_grad=False` 由 `register_buffer` 自动保证
3. **设备透明** — `.to(device)` 自动迁移

### 6.3 补零维度在 Normalizer 中的行为

Normalizer 对补零维度的处理是理解 FAAS 安全性的关键。下面逐行追踪 `fit_params()` 的实际代码路径（`common/normalizer.py`）：

```python
# 对于补零维度: min=0.0, max=0.0

# Line 93: input_range = input_max - input_min     = 0.0
# Line 94: ignore_dim = (input_range < range_eps)   = True   (0.0 < 1e-4)
# Line 95: input_range[ignore_dim] = output_max - output_min  = 1.0 - (-1.0) = 2.0  ← 关键!
# Line 96: scale = (output_max - output_min) / input_range    = 2.0 / 2.0 = 1.0
# Line 98: offset[ignore_dim] = -input_mean[ignore_dim]       = -0.0 = 0.0
```

**正确结果：`scale=1.0, offset=0.0`（恒等映射）。**

`range_eps` 仅用作 `ignore_dim` 的**布尔判断阈值**，不作为分母。`input_range` 被裁剪到 `output_max - output_min = 2.0`，而非 `range_eps`。

**这意味着：**

| 场景 | 计算 | 结果 |
|------|------|------|
| 训练: normalized(0) | `(0 + 0) × 1.0` | **0** (恒等) |
| 推理: 模型预测 z=0.01 | `unnormalize(0.01) = 0.01/1.0 - 0` | **0.01** (原值通过) |
| 推理: 模型预测 z=1.0 | `unnormalize(1.0) = 1.0/1.0 - 0` | **1.0** (无衰减!) |
| 逆映射: faas→native | `gather(12 个活跃维度)` | **补零维度被丢弃** |

**关键结论**：

1. **Normalizer 不提供任何安全衰减** — 它是恒等映射，模型预测什么值就输出什么值
2. **唯一的安全机制是 `faas_to_native()` 的 gather 操作** — 32D 中仅 12 个活跃维度被提取，20 个补零维度**直接丢弃**，无论其值是多少
3. **模型必须学会在补零维度上输出零** — 无法依赖 normalizer 来"压制"错误预测。但这是可行的：补零维度在所有训练数据中恒为零，是扩散模型能遇到的最简单学习目标
4. 这一行为与 rot6d 在 `build_mixed_action_normalizer()` 中的恒等映射处理完全一致（`normalizer.py:373-374`），是系统范围内的设计惯例

**为什么这反而是好事**：因为模型在补零维度上学到的"恒输出零"能力，在引入新手的训练数据时会自然迁移——那些维度从"恒为零"变为"有时非零"，模型平滑过渡，无需修改 normalizer。

> **与 UniDex 对比**: UniDex 使用 `minmax` 模式，补零维度 `min=max=0`，归一化后同样保持零值。两种 normalizer 模式在补零维度上等价——都是恒等映射。详见附录 §9.4。

### 6.4 扩散过程对补零维度的影响

扩散训练时，前向过程对所有 39 维加等量高斯噪声：

```
x_t = √(ᾱ_t) × x_0 + √(1-ᾱ_t) × ε

补零维度: x_0=0, 所以 x_t = √(1-ᾱ_t) × ε  (纯噪声)
活跃维度: x_0≠0, 所以 x_t 是信号+噪声的混合
```

模型需要同时学会：
- 活跃维度：从噪声中恢复信号（困难）
- 补零维度：从噪声中恢复零（简单，因为 target 恒定）

实际上，模型学习"补零维度总是零"非常容易——它只需要在这些维度上输出一个很小的值。UniDex 的实践已经证明了这一点（它们在 32D 空间训练 8 种手，每种手有不同的稀疏模式，模型自动适应）。

### 6.5 `index_bend_joint` 符号翻转的正确性

UniDex 的 `hand_utils.json` 中 XHand 的 `retarget_joint_map_scale` 为 `index_bend_joint: -1.0`（且仅此一个非 1.0）。这是因为：

- DexMani XHand URDF 中 `index_bend_joint` 的旋转轴为 `(-1, 0, 0)`
- FAAS 约定食指侧摆的正方向与 URDF 定义相反
- 翻转仅影响 FAAS 内部表示，逆向映射时自动翻转回来

**验证方法**：对同一帧数据，比较 `faas_to_native(native_to_faas(x))` 与 `x` 的差异。若 roundtrip error < 1e-7，则映射正确。

### 6.6 `state_dim` 为何也需要扩大

`joint_state` 观测包含手部关节角（后 12 维），作为 Diffusion/FlowMatch 模型的条件输入。如果模型在 FAAS 空间预测动作，条件也应使用 FAAS 空间的关节状态，保证：

1. **表示一致性** — 条件和目标在同一空间
2. **obs_encoder 复用** — StateMLP 输入维度和 action_decoder 输入维度使用相同的 FAAS 约定

### 6.7 与 DQ-RISE 的特殊交互

**结论：DQ-RISE 在 Phase 1 不启用 FAAS。** 原因并非架构不兼容（`hand_dim = action_dim - tcp_dim` 自动推导，`32 = 39 - 7` 在数学上成立），而是 **VQ-VAE codebook 与 FAAS 32D 存在根本性的数据不兼容**。

**4 个具体阻断点**：

| # | 组件 | 当前 (12D native) | FAAS (32D) | 不兼容原因 |
|---|------|-------------------|------------|-----------|
| 1 | **Codebook 存储** | `sorted_hand_poses: (codebook_size^num_groups, 12)` = e.g. `(16, 12)` | 需要 `(16, 32)` 或更大 | Codebook 维度硬编码为 `hand_dim=12` |
| 2 | **Per-finger loss 权重** | `[1.0]*12`（thumb×3, index×3, middle×2, ring×2, pinky×2） | 需要 32 元素权重 | 12 维对应 5 指的具体 DOF 分布，32 维仅 12 个非零位置 |
| 3 | **VQ groups + codebook_size** | `num_groups=2, codebook_size=4 → 16 codes` | 可能需要更大 codebook | 32D 空间的覆盖需求可能不同于 12D |
| 4 | **CodebookManager API** | `hand_pose_to_continuous_index(12D) → 1D` | 期望 `hand_pose_to_continuous_index(32D) → 1D` | 函数签名依赖 `hand_dim`，但 codebook 数据本身是 12D |

**如果未来需要在 DQ-RISE 上启用 FAAS，必须重跑完整三阶段管道**：

```
Stage 1: VQ-VAE 预训练 (train_vq_hand.py)     — hand_dim=32, new loss weights
Stage 2: Codebook 提取+PCA 排序 (extract_codebook.py) — 32D codebook
Stage 3: 联合扩散训练 (train.py dqrise)          — action_dim=39
```

这相当于从头训练一个 DQ-RISE 模型，工作量与训练新策略相同。

**其他 5 种 Agent（DP/DP3/ManiFlow/MoE/MultiTask）均无此限制** — 它们不拆分 action 空间，仅需修改 config 中的 `action_dim` 和 `state_dim`。

---

## 7. 实施路线图

### Phase 1: 核心基础设施（1-2 天）

```
目标: FAAS 管线跑通，smoke test 通过

文件清单:
  ✏️ common/faas_mapper.py               (新建, ~80 行)
  ✏️ configs/dp.yaml                      (新增 use_faas 等字段)
  ✏️ configs/dp3.yaml
  ✏️ configs/maniflow.yaml
  ✏️ configs/moe_dp.yaml
  ✏️ configs/r3d.yaml
  ✏️ configs/dqrise.yaml                  (仅加字段, 默认 use_faas: false)
  ✏️ datasets/base_dataset.py             (追加 _apply_faas_mapping)
  ✏️ agents/core/base.py                  (追加 predict_action 逆向转换)
  ✏️ smoke_test.py                        (适配 39/41D)

验证标准:
  ☐ FAASHandMapper roundtrip: |x - faas⁻¹(faas(x))| < 1e-7
  ☐ smoke_test dp3 通过
  ☐ 5 策略 × smoke_test 全部通过 (dqrise 除外)
  ☐ 单任务训练 1 epoch 无 NaN
```

### Phase 2: 训练等价性验证（2-3 天）

```
目标: 证明 FAAS 模式不降低成功率

实验设计:
  A组: dp3 use_faas=false (baseline, 19D)
  B组: dp3 use_faas=true  (FAAS, 39D)
  C组: dp3 use_faas=true + sparse_loss_mask (FAAS+mask, 39D)

  在 pick_apple_messy 上各训练 200 epochs, 对比:
  ☐ 收敛速度 (loss 曲线)
  ☐ 最终 success_rate (差距 < 2% 视为等价)
  ☐ 补零维度预测值 (应稳定在 0±0.01 范围内)

  扩展到 3 个任务验证一致性
```

### Phase 3: UniDex 数据接入（1-2 周，可选）

```
目标: UniDex retargeted XHand 数据 → DexMani Zarr, 混合训练

步骤:
  1. 编写 scripts/convert_unidex_to_dexmani.py
     - 输入: UniDex HDF5 (retarget_RGBD)
     - 输出: DexMani Zarr
     - 挑战:
       a. 腕部 Δpose(9D) → arm_joint(7D): 需要 IK
          → 或使用 action_ee 模式 (pos3+rot6d6), 9D 直接对应
       b. 相机坐标系对齐 (UniDex CV→CAM vs DexMani SAPIEN)
       c. 点云格式转换 (UniDex RGBD投影 vs DexMani 仿真传感器)

  2. 混合训练: Zarr(XHand_sim) + Zarr(XHand_unidex)
     - FAAS 空间天然对齐, 无需额外处理

  3. 验证: 混合训练后 success_rate 是否提升
     - 预期: 数据量增加 → 泛化性提升
     - 风险: UniDex retargeting 精度不足 → 引入噪声
```

### Phase 4: 多手扩展（长期）

```
目标: 支持 ≥2 种手, 验证跨手泛化

前提:
  - DexMani_Sim 集成第二只手 (如 Inspire)
  - 采集新手的仿真数据
  - FAASHandMapper 扩展为多手注册表

实现:
  class MultiHandFAASMapper:
      def __init__(self, hand_type: str):
          self.mapper = FAAS_REGISTRY[hand_type]  # 加载对应手的映射参数
```

---

## 8. 风险与缓解

### 8.1 技术风险

| # | 风险 | 概率 | 影响 | 缓解措施 |
|---|------|------|------|---------|
| R1 | `index_bend_joint` 翻转方向错误 | 中 | 食指侧摆行为反向，可能降低抓取成功率 | Phase 1 roundtrip 测试 + Phase 2 对比实验 |
| R2 | 补零维度在训练中漂移 | 低 | 被 normalizer 和 gather 双重过滤，实际无影响 | 监控 Phase 2 中补零维度的 variance |
| R3 | 39D 模型收敛慢于 19D | 低 | 需要更多 epoch 达到同等成功率 | Phase 2 对比 loss 曲线；必要时启用 sparse loss mask |
| R4 | DDP 中 FAAS buffer 广播失败 | 无 | 多卡训练挂起 | 已验证安全: DDP broadcast 是 shape-agnostic 逐 tensor 操作; `register_buffer(persistent=True)` 确保 buffer 包含在 `state_dict` 中; config 通过 `mp.spawn` 正确传播; 所有 per-rank 操作均使用动态 `self.action_dim`; `find_unused_parameters=False` 仅检查可训练参数（buffer 不受影响） |
| R5 | 历史 checkpoint 加载失败 | 高 | 已有权重无法复用 | 提供 `scripts/migrate_checkpoint_to_faas.py` 迁移工具（权重矩阵扩展 + `train_params.action_dim` 更新） |

### 8.2 Checkpoint 迁移工具设计

```python
# scripts/migrate_checkpoint_to_faas.py
"""
将 19D/21D native checkpoint 迁移为 39D/41D FAAS checkpoint。

原理:
  - 19D action: [arm_7d | hand_12d]
  - 39D action: [arm_7d | FAAS_hand_32d] = [arm_7d | 0...native_0@idx1...native_3@idx6*(-1)...0]
  - UNet 首层 Conv1d(19→256) 权重: 前 19 列复制, 后 20 列零初始化
  - UNet 末层 Conv1d(256→19) 权重: 前 19 行复制, 后 20 行零初始化

CRITICAL: 还需更新 checkpoint payload 中的元数据:
  checkpoint['train_params']['action_dim']: 19 → 39
  checkpoint['train_params']['state_dim']:  19 → 39 (如果 joint_state 也做了 FAAS 转换)

否则 SimEvaluator._load_for_inference() 的严格校验 (sim_evaluator.py:60-68)
会因 action_dim 不匹配而拒绝加载已迁移的 checkpoint。
"""
```

**受影响的参数量（以 UNet1D 为例）**：

| 参数名 | 旧形状 | 新形状 | 迁移操作 |
|--------|--------|--------|---------|
| `down_modules.0.0.block.0.weight` | `(256, 19, k)` | `(256, 39, k)` | 列 [0:19] 复制，列 [19:39] 零初始化 |
| `down_modules.0.residual_conv.weight` | `(256, 19, 1)` | `(256, 39, 1)` | 同上 |
| `final_conv.1.weight` | `(19, 256, 1)` | `(39, 256, 1)` | 行 [0:19] 复制，行 [19:39] 零初始化 |
| `final_conv.1.bias` | `(19,)` | `(39,)` | 元素 [0:19] 复制，[19:39] 零初始化 |
| **所有中间层** | 不变 | 不变 | 无需操作（channel dim 256/512/1024 不变） |

对于 DiTX/DiT backbone（ManiFlow/MultiTask），迁移模式相同（input_embedder + final_layer 的输入/输出维度扩展）。

### 8.3 回滚策略

- `use_faas: false` 时所有代码路径与原行为一致
- 若 Phase 2 发现 FAAS 模式性能退化 >2%，可暂缓推进，仅保留 FAASMapper 代码不动
- FAAS 管线不影响现有 checkpoint 的推理（`use_faas: false` 路径不变）

---

## 9. 附录

### 9.1 关键文件索引

| 文件 | 角色 |
|------|------|
| `docs/UniDex-知识体系.md` | UniDex 完整分析，§5 详述 FAAS 机制 |
| `UniDex/src/assets/utils/hand_utils.json` | FAAS 完整映射表（32D 索引、scale、offset、8 手） |
| `UniDex/src/utils/inspire_utils.py` | Inspire MIMIC_RELATION 欠驱动联动 |
| `UniDex/src/dataset/base_retarget.py` | HDF5 → FAAS 数据加载管线 |
| `DexMani_Sim/dexmani_sim/robots/_urdf_config.py` | XHand 12 关节名定义 |
| `DexMani_Sim/dexmani_sim/assets/robots/xhand/xhand_right.urdf` | XHand URDF（关节限位、轴） |

### 9.2 术语对照

| 术语 | UniDex 定义 | DexMani 对应 |
|------|-----------|-------------|
| FAAS | 82D 统一动作空间 (18+64) | 右手 41D (9+32) 或 39D (7+32) |
| MAPPED_JOINT_DIM | 32 (单手 FAAS 维度) | 同 |
| JOINT_DIM_IN_USE | 27 (活跃槽位数) | XHand 使用 12/27 |
| retarget scale/offset | IK retargeting 后的归一化参数 | 仅 `index_bend: -1.0` 非平凡 |
| MIMIC_RELATION | 欠驱动联动约束 | XHand 不适用（全独立驱动） |
| wrist pose | pos3 + rot6d (9D per hand) | action_ee 模式对应 |

### 9.3 参考资料

- **UniDex 论文**: Zhang et al., *UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control*, CVPR 2026, arXiv:2603.22264
- **FAAS 原始设计**: UniDex `hand_utils.json` + `base_retarget.py` + `base.py` § `_apply_action_map`
- **6D 旋转表示**: Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019
- **DexMani 架构**: 本项目 `CLAUDE.md` + `docs/` 下各知识体系文档

### 9.4 UniDex vs DexMani Normalizer 对比

两种 normalizer 模式在 FAAS 补零维度上的**数值行为不同但安全等效**：

| 属性 | UniDex (`minmax`) | DexMani (`limits`) |
|------|-------------------|-------------------|
| 归一化目标 | `[mid_val-min, mid_val+max]` | `[-1, 1]` |
| 补零维度 min/max | `min=0.0, max=0.0`（预计算全局统计） | `min=0.0, max=0.0`（从 replay buffer 拟合） |
| 补零维度 normalize scale | `2/(max-min+1e-6) ≈ 2e6` | `ignore_dim` 触发 → `input_range=2.0` → `scale=1.0` |
| 补零维度 **unnormalize factor** | `1/scale ≈ 5e-7`（隐式安全衰减） | `scale=1.0`（真正恒等映射） |
| 补零维度 offset | `mid_val = (max+min)/2 = 0` | `-input_mean = 0` |
| 安全机制 | unnormalize 将非零预测值压缩到 ≈0 | `faas_to_native()` gather 丢弃补零维度 |

**关键差异**: UniDex 的 minmax 对补零维度产生 **unnormalize factor ≈ 5e-7**，模型预测的非零值会被极度压缩。DexMani limits 产生**真正的恒等映射** (scale=1.0)，预测值原样通过。两者**数值不等价**。

**为什么 DexMani 的恒等映射反而更优**: 
1. 推理时 `faas_to_native()` 的 gather 操作丢弃 20 个补零维度，无论其值如何
2. 未来引入新手训练数据时，那些维度从"恒为零"变为"有时非零"，恒等映射的 normalizer 支持平滑过渡，无需重拟合
3. UniDex minmax 的 5e-7 unnormalize factor 在引入新手后会破坏新手的非零值，需要重拟合 normalizer

**结论**: 不需要为 FAAS 切换 normalizer 模式。DexMani 现有的 `limits` 模式对 FAAS 补零维度的处理正确且更适应未来扩展。强制要求 FAAS 模式下使用 `limits` 模式（gaussian 模式在补零维度上数值不稳定）。

### 9.5 腕部动作表示差异：相对 Delta vs 绝对关节角

影响 Phase 3（UniDex 数据接入），Phase 1-2 不受影响。

| 属性 | UniDex action | DexMani action (joint模式) | DexMani action_ee |
|------|-------------|--------------------------|-------------------|
| 腕部表示 | **相对 Δ** (当前帧→目标帧) | **绝对关节角** | **绝对末端位姿** |
| 公式 | `action_wrist = mat_to_pose9d(inv(T_cur) @ T_tgt))` | `action_arm = joint_angles` | `action_ee = [pos3, rot6d6]` |
| 物理量 | 位移 + 旋转增量 | 关节位置 | 末端位姿 |

**Phase 3 转换策略**:
- `action_ee` 模式直接兼容（UniDex 的相对 Δpose 与 DexMani 的 ee action 语义一致）
- `action` 模式需要通过 IK 将相对腕部位姿转为臂关节角增量，再叠加当前关节角

### 9.6 Agent 兼容性矩阵

经代码审查，各 Agent 对 FAAS（`action_dim=39`, `state_dim=39`）的兼容性：

| Agent | 兼容性 | 改动范围 | 备注 |
|-------|--------|---------|------|
| DPAgent / DP3Agent | ✅ 直接兼容 | config only | 无 action-dim 特化逻辑 |
| ManiFlowAgent | ✅ 直接兼容 | config only | DiTXFlowMatch 无 dim 假设 |
| MoEAgent | ✅ 直接兼容 | config only | 双 backbone 切换不涉及 action_dim |
| MultiTaskAgent | ✅ 直接兼容 | config only | 文本条件与 action 结构独立 |
| R3DAgent (`use_aux_ee=False`) | ✅ 直接兼容 | config only | dim_groups 机制天然支持任意维度分组 |
| R3DAgent (`use_aux_ee=True`) | ⚠️ 需适配 | backbone 改造 | OneWayTransformer 仅支持 2-head (joint/ee)，FAAS 32D hand 需 N-head 泛化 |
| DQRISEAgent | ❌ 不兼容 | 完整三阶段重训 | VQ-VAE codebook 与 32D FAAS 数据不兼容（详见 §6.7） |

### 9.7 DDP 兼容性验证总结

所有 DDP 机制经代码路径分析验证为 FAAS-safe：

| 机制 | 验证结果 |
|------|---------|
| Normalizer broadcast (`dist.broadcast`) | ✅ 逐 tensor 操作，shape-agnostic |
| Config 传播 (`mp.spawn`) | ✅ `OmegaConf.resolve(cfg)` 在 spawn 前完成，`use_faas` boolean 正确传播 |
| Per-rank 操作 | ✅ 所有维度敏感代码使用动态 `self.action_dim`/`self.tcp_dim`/`self.hand_dim` |
| 梯度累积 (`model.no_sync()`) | ✅ 与 tensor size 无关 |
| `find_unused_parameters=False` | ✅ 仅检查 `requires_grad=True` 参数，FAASHandMapper buffer 不受影响 |
| DDP config overlays | ✅ 通过 `defaults:` 继承 base config 的 FAAS 字段 |
| 硬编码 buffer size | ✅ 无 — 全部通过 config 或模型属性动态确定 |

**结论: DDP 训练对 FAAS 无任何阻断风险。**
