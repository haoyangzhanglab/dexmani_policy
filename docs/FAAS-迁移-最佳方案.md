# FAAS 迁移方案：Joint Space | DQ-RISE | FAAS 三轨对比

> **目标**: 将 UniDex FAAS 以最小侵入方式迁移到 DexMani_Policy
> **前置阅读**: `docs/FAAS-集成方案.md`（原始设计）、`docs/UniDex-知识体系.md`（§5 FAAS）
> **策略**: 独立 FAAS config 文件，Hydra `defaults` 继承，**零侵入现有 config**
> **状态**: ✅ 已审查（v5，ultracode 深度审查 + 4 P0/8 P1 修复）— 待执行
> **最后更新**: 2026-07-27

---

## 目录

1. [三轨对比总览](#1-三轨对比总览)
2. [设计决策](#2-设计决策)
3. [实施步骤](#3-实施步骤)
4. [文件变更清单](#4-文件变更清单)
5. [实验设计](#5-实验设计)
6. [向后兼容性](#6-向后兼容性)
7. [风险与缓解](#7-风险与缓解)
8. [附录](#8-附录)

---

## 1. 三轨对比总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    三轨动作空间对比                                │
├────────────┬────────────────┬────────────────┬──────────────────┤
│            │  Joint Space   │   DQ-RISE      │     FAAS         │
│            │  (现有基线)     │  (现有基线)     │   (新增)         │
├────────────┼────────────────┼────────────────┼──────────────────┤
│ 动作维度    │  19 (joint)    │  19 (joint) ¹  │  39 (7+32)       │
│  (config)  │  21 (ee)       │  21 (ee)       │  41 (9+32)       │
├────────────┼────────────────┼────────────────┼──────────────────┤
│ 手部表示    │  12D 连续关节角 │  1D 连续VQ索引  │  32D 功能对齐     │
│ 跨手泛化    │  ❌            │  ❌             │  ✅              │
│ 臂-手解耦   │  ❌            │  ✅ (VQ)        │  ❌              │
├────────────┼────────────────┼────────────────┼──────────────────┤
│ 配置方式    │  dp3.yaml      │  dqrise.yaml   │  dp3_faas.yaml   │
│ 启动命令    │  train.sh dp3  │  train.sh      │  train.sh        │
│            │                │  dqrise         │  dp3_faas        │
└────────────┴────────────────┴────────────────┴──────────────────┘

¹ DQ-RISE 扩散模型实际输出 tcp_dim+1=8/10 维（连续 VQ 索引），
  经 CodebookManager 查表还原为 19/21 维。config action_dim 仍为 19/21。
```

**核心设计原则**: FAAS 和 Joint Space 共享同一 Agent 代码路径（仅数据管线不同），DQ-RISE 是独立 Agent 子类。Joint vs FAAS 的对比是"纯动作空间消融"。

---

## 2. 设计决策

### 2.1 配置策略：独立 FAAS config + Hydra defaults 继承

**不修改任何现有 config 文件**。创建 `configs/dp3_faas.yaml` 和 `configs/ddp/dp3_faas.yaml`，通过 Hydra `defaults` 从 `dp3.yaml` 继承后仅覆写维度相关字段。

DDP config 已使用相同模式（`defaults: [- /dp, - _self_]`），经过充分验证。

```yaml
# configs/dp3_faas.yaml (~25 行，新建)
# @package _global_
# DP3 + FAAS (Function-Actuator-Aligned Space) unified hand action space.
#
# Inherits all training hyperparameters, encoder config, UNet backbone,
# and env settings from dp3.yaml.  Only overrides action-space dimensions.
#
# Usage:
#   bash scripts/train.sh dp3_faas
#   bash scripts/train.sh dp3_faas 'training.loop.num_epochs=400'
#   bash scripts/train.sh dp3_faas 'action_key=action_ee'

defaults:
  - /dp3
  - _self_

policy_name: dp3_faas

# ── FAAS ──
use_faas: true
faas_hand_dim: 32

# ── Dimension Overrides ──
# tcp_dim is INHERITED from dp3.yaml (same expression). Listed here for explicitness.
# action_dim overrides the global inherited from dp3.
# FAAS hand: 32D (12 active + 20 zero-padded).
# IMPORTANT: joint_state arm is ALWAYS 7D (arm joint angles), regardless of action_key.
# The action's arm part is 7D (joint) or 9D (action_ee). These are DIFFERENT concepts.
# NOTE: action_dim is at GLOBAL level — dp3.yaml:68 references ${action_dim}, so
# overriding the global value auto-propagates into agent.action_dim.
tcp_dim: ${eval:'9 if ${eq:${action_key},action_ee} else 7'}   # action arm dim
action_dim: ${eval:'${tcp_dim} + 32'}   # 39 (joint) / 41 (action_ee)

agent:
  # state_dim is hardcoded as a literal inside agent: in dp3.yaml:73 —
  # must override inside agent: to take effect.  Deep merge preserves all
  # other agent fields (_target_, encoder_type, down_dims, etc.).
  # joint_state = 7 arm joints (fixed) + 32 FAAS hand = 39 (independent of action_key).
  state_dim: ${eval:'7 + ${faas_hand_dim}'}   # 39 (7+32)
```

```yaml
# configs/ddp/dp3_faas.yaml (~15 行，新建)
# @package _global_
# DP3-FAAS DDP multi-GPU training overlay.
# Usage: bash scripts/train_ddp.sh ddp/dp3_faas

defaults:
  - /dp3_faas
  - _self_

policy_name: ddp/dp3_faas

training:
  num_gpus: 4
  gpu_ids: null

dataloader:
  batch_size: 32
  num_workers: 4

val_dataloader:
  batch_size: 32
  num_workers: 4
```

**关键发现（来自 Hydra 审查）**:
- `dp3.yaml:68` 的 `agent.action_dim` 引用 `${action_dim}` → 全局覆写自动传播 ✅
- `dp3.yaml:73` 的 `agent.state_dim: 19` 是**硬编码字面量** → 必须在 `agent:` 块内覆写，全局 `state_dim` 无效
- `joint_state` arm 部分**固定 7D**（关节角），与 `action_key` 无关 → `state_dim` 恒为 `7 + faas_hand_dim = 39`
- `tcp_dim` 仅用于 action 的 arm 分割（7 或 9），**不可**用于 joint_state 的 arm 分割

### 2.2 FAAS 属性注入：共享 helper

`build_model_and_ema` 和 `eval_sim.py` 都需要注入 FAAS 属性。提取为共享 helper 避免重复和漂移：

```python
# training/build_utils.py — 新增
def inject_faas_into_agent(agent, cfg):
    """Post-construction FAAS injection for any entry point."""
    agent.use_faas = cfg.get('use_faas', False)
    if not agent.use_faas:
        return
    from dexmani_policy.common.faas_mapper import FAASHandMapper
    agent.tcp_dim = cfg.tcp_dim
    agent.hand_dim = cfg.get('hand_dim', cfg.get('faas_hand_dim', 32))
    agent.faas_mapper = FAASHandMapper()
```

### 2.3 数据流：模型在 FAAS 空间训练，I/O 边界全量转换

```
训练:
  Zarr (19D/21D native)
    → Dataset.__getitem__
      → sample_to_data() (native)
      → apply_augmentation() (native)     ← 增强在 native 空间执行
      → ensure_tensor() (numpy→torch)
      → _apply_faas_mapping()             ← FAAS 转换在 torch 空间执行
        action:  [arm(tcp_dim) | hand_12D] → [arm(tcp_dim) | FAAS_32D]
        joint_state (in obs): [arm_7D | hand_12D] → [arm_7D | FAAS_32D]
    → Normalizer.fit(FAAS数据)             ← get_normalizer() 内部做 FAAS 转换
    → Agent.compute_loss(FAAS cond, FAAS action)

推理 (predict_action):
    → obs_dict from env (native 19D joint_state)
    → _convert_obs_to_faas(obs_dict): native→FAAS  ← P0-1 修复
    → normalize(FAAS obs) → encode → denoise → unnormalize
    → inverse_transform_action → native (19D/21D)
    → pred_action: native    ← temporal ensembling 直接可用
    → control_action: native ← env.step 直接可用

compute_action_mse:
  gt_action (FAAS from dataset) → inverse_transform → native
  pred_action (native) → MSE in native joint space
```

### 2.4 关键边界情况处理

| 边界 | 方案 |
|------|------|
| **eval_sim.py** | 调用 `inject_faas_into_agent(agent, cfg)`（与 `build_model_and_ema` 共享 helper） |
| **推理 obs 转换** | `predict_action()` 开头调用 `_convert_obs_to_faas()` — env native 19D joint_state → FAAS 39D（P0-1 修复） |
| **get_normalizer()** | 内部对 replay_buffer 数据做 FAAS 转换后再拟合（P0-2 修复） |
| **FAAS 转换时机** | `__getitem__` 中 `ensure_tensor`（numpy→torch）之后、`return` 之前执行（P0-3 修复） |
| **joint_state 键路径** | `data['obs']['joint_state']` 非 `data['joint_state']`（P0-4 修复） |
| **joint_state arm_dim** | 固定 `state_arm_dim=7`（关节角），不可复用 `tcp_dim`（action_ee 下为 9）（P1-2 修复） |
| **action_ee state_dim** | 恒为 39（7+32），非 41（P1-1 修复） |
| **MultiTaskDataset** | Phase 1 加 guard：`use_faas=true` + MultiTask → `NotImplementedError`。Phase 2 递归注入子数据集 |
| **Smoke test roundtrip** | 显式测试 `|native - faas⁻¹(faas(native))| < 1e-6` |
| **train_params** | 同时存 `action_dim`(FAAS 39/41)、`use_faas`(bool)、`control_action_dim`(native 19/21) |
| **SimEvaluator** | `_load_for_inference` 新增 `use_faas` 一致性校验 |
| **Dataset.tcp_dim** | `BaseDataset.__init__` 新增 `tcp_dim` 参数（use_faas 时必传） |

### 2.5 DQ-RISE 豁免

DQRISEAgent 完全覆写 `compute_loss()` 和 `predict_action_from_cond()`（不调用 `super()`）。BaseAgent 层 FAAS 转换对其无影响。三轨对比仅需 `dp3` | `dqrise` | `dp3_faas`。

**注意**: DQ-RISE 豁免仅限于 Phase 1。若未来需在 DQ-RISE 上启用 FAAS，需重跑完整三阶段管道（VQ-VAE 预训练 → Codebook 提取+PCA 排序 → 联合扩散训练）。特别地，PCA 排序算法中的 `layer_weights` 权重矩阵需从头学习（12D→32D），不可简单复用。

### 2.6 Sparse Loss Mask：暂不启用

补零维度在所有训练数据中恒为零，模型自然学会预测零。UniDex 同样不使用 sparse loss mask。补零维度的 gradient 自然收敛到极小值，无需干预。Phase 2 等价性验证中监控补零维度 mean/std，设定启用 mask 的触发阈值（|mean| > 0.05 或 std > 0.1）。

### 2.7 腕部动作表示差异（Phase 1 不受影响）

UniDex 的腕部 action 是**相对 Δ 位姿**（`inv(T_cur) @ T_tgt`），而 DexMani 使用**绝对关节目标**（`action` 模式）或**绝对末端位姿**（`action_ee` 模式）。这一差异**仅影响手部关节部分以外的臂部表示**。

Phase 1 的 FAAS 迁移**仅影响手部关节部分**（12D→32D），不改变臂部动作表示方式。与 UniDex FAAS 的手部关节设计完全一致（均为绝对角度）。差异仅在 Phase 3（UniDex 数据接入）时需要处理：`action_ee` 模式天然兼容（UniDex 相对 Δpose 与 DexMani ee action 语义一致），`action` 模式需通过 IK 将相对腕部位姿转为臂关节角增量。

### 2.8 与 UniDex Normalizer 的数值差异

### 2.8 与 UniDex Normalizer 的数值差异

**UniDex `minmax` 模式**对补零维度 (min=max=0) 产生 `scale ≈ 2e6`，unnormalize factor ≈ 5e-7，模型预测值被几乎清零。**DexMani `limits` 模式**对补零维度产生恒等映射 (`scale=1.0, offset=0.0`)，预测值原样通过。

两种行为**数值不同但安全等效**：推理时 `faas_to_native()` 的 gather 操作丢弃 20 个补零维度，无论其值是多少。DexMani 的恒等映射行为反而是优势——未来加入新手训练数据时，那些维度从"恒为零"变为"有时非零"，恒等映射的 normalizer 支持平滑过渡，无需修改 normalizer 参数。UniDex 的 minmax 模式下的 5e-7 unnormalize factor 在引入新手数据后需要重新拟合 normalizer。

**强制要求**: FAAS 模式下 normalizer mode 必须为 `limits`（`_validate_faas_config` 中校验）。`gaussian` 模式在补零维度上存在数值不稳定风险（std≈0 → scale→∞）。

---

## 3. 实施步骤

### Step 1: 核心基础设施

**目标**: `dp3_faas` smoke test 通过 + roundtrip 验证

#### 3.1 新建 `common/faas_mapper.py` (~120 行)

`FAASHandMapper(nn.Module)`:
- `register_buffer(persistent=True)` → checkpoint 自包含
- `native_to_faas(hand_12d) → hand_32d`（scatter + scale/offset，公式: `faas = native * scale + offset`）
- `faas_to_native(hand_32d) → hand_12d`（gather + 通用逆变换，公式: `native = (faas - offset) / scale`）
- `transform_action(action, arm_dim)` — 一站式正向（仅转换 hand 部分）
- `inverse_transform_action(action, arm_dim)` — 一站式逆向（仅转换 hand 部分）
- `transform_joint_state(state_19d) → state_39d` — arm 固定 7D

映射参数来自 UniDex `hand_utils.json`（已验证）:
- indices: `(1,2,3,6,7,8,12,13,17,18,22,23)`
- scales: 全 1.0 除 `index_bend_joint=-1.0`
- offsets: 全 0.0（XHand 无 offset）

**逆变换使用通用公式** `(faas - offset) / scale` 而非简化的 `faas * scale`。当前所有 scale ∈ {1,-1}，
两者数值等价 (1/s = s)，但通用公式对未来非 ±1 scale 具有正确性。

#### 3.2 修改 `datasets/base_dataset.py` (~50 行)

**A. `__init__` 新增参数**:
- `tcp_dim: int | None = None` — action arm 维度（7 or 9），use_faas 时必传

**B. `__getitem__` 中插入 FAAS 转换**（在 `ensure_tensor` 之后）:
```python
def __getitem__(self, idx):
    sample = self.sampler.sample_sequence(idx)
    data = self.sample_to_data(sample)          # native numpy
    data = self.apply_augmentation(data)         # augmentation on native
    if self.rgb_preprocess_size is not None and 'rgb' in data['obs']:
        data['obs']['rgb'] = self._preprocess_rgb_cpu(data['obs']['rgb'])
    data = dict_apply(data, ensure_tensor)       # numpy → torch
    if getattr(self, 'use_faas', False):
        data = self._apply_faas_mapping(data)    # FAAS on torch Tensors ← HERE
    return data
```

**C. `_apply_faas_mapping()` — 在 torch Tensor 上操作，正确键路径**:
```python
def _apply_faas_mapping(self, data: dict) -> dict:
    """Convert native action & joint_state to FAAS. Operates on torch.Tensor."""
    # Action: [arm(tcp_dim) | hand(12)] → [arm(tcp_dim) | FAAS_hand(32)]
    arm_action = data['action'][..., :self.tcp_dim]
    hand_action = data['action'][..., self.tcp_dim:]
    data['action'] = torch.cat(
        [arm_action, self.faas_mapper.native_to_faas(hand_action)], dim=-1)

    # Joint state in data['obs'] — arm is ALWAYS 7D joint angles (not tcp_dim!)
    if 'joint_state' in data.get('obs', {}):
        js = data['obs']['joint_state']
        arm_state = js[..., :7]   # state_arm_dim = 7 (fixed)
        hand_state = js[..., 7:]  # hand starts at index 7
        data['obs']['joint_state'] = torch.cat(
            [arm_state, self.faas_mapper.native_to_faas(hand_state)], dim=-1)

    return data
```

**D. `get_normalizer()` — 对 replay buffer 做 FAAS 转换后拟合**:
```python
def get_normalizer(self, mode='limits'):
    joint_state = self.replay_buffer['joint_state']  # numpy (N, 19)
    action = self.replay_buffer[self.action_key]      # numpy (N, 19|21)

    if getattr(self, 'use_faas', False):
        import torch
        # Convert to torch for FAAS mapping, then back to numpy for fitting
        js_t = torch.from_numpy(joint_state).float()
        a_t = torch.from_numpy(action).float()
        # Joint state: arm(7) is always 7D arm joints
        joint_state = torch.cat([
            js_t[..., :7],
            self.faas_mapper.native_to_faas(js_t[..., 7:])
        ], dim=-1).numpy()
        # Action: arm(tcp_dim) depends on action_key
        action = torch.cat([
            a_t[..., :self.tcp_dim],
            self.faas_mapper.native_to_faas(a_t[..., self.tcp_dim:])
        ], dim=-1).numpy()

    normalizer = LinearNormalizer()
    ...  # existing fitting logic (unchanged)
```

#### 3.3 修改 `agents/core/base.py` (~40 行)

**A. `predict_action()` — 开头添加 obs native→FAAS 正向转换**:
```python
@torch.no_grad()
def predict_action(self, obs_dict, denoise_timesteps=None):
    if getattr(self, 'use_faas', False):
        obs_dict = self._convert_obs_to_faas(obs_dict)
    cond, _ = self._build_cond(obs_dict)
    return self.predict_action_from_cond(cond, denoise_timesteps)
```

**B. `_convert_obs_to_faas()` — 新方法**:
```python
def _convert_obs_to_faas(self, obs_dict):
    """Convert env-native joint_state (19D) to FAAS (39D) for model consumption.

    joint_state arm is ALWAYS 7D arm joint angles; action's tcp_dim is irrelevant here.
    """
    if 'joint_state' not in obs_dict:
        return obs_dict
    js = obs_dict['joint_state']
    arm_state = js[..., :7]          # state_arm_dim = 7 (fixed)
    hand_state = js[..., 7:]
    faas_hand = self.faas_mapper.native_to_faas(hand_state)
    return {**obs_dict, 'joint_state': torch.cat([arm_state, faas_hand], dim=-1)}
```

**C. `predict_action_from_cond()` — unnormalize 后全量 inverse_transform**:
```python
@torch.no_grad()
def predict_action_from_cond(self, cond, denoise_timesteps=None):
    template = torch.zeros(cond.shape[0], self.horizon, self.action_dim, ...)
    pred = self.action_decoder.predict_action(cond, template, denoise_timesteps)
    pred = self.normalizer['action'].unnormalize(pred)
    if getattr(self, 'use_faas', False):
        pred = self.faas_mapper.inverse_transform_action(pred, self.tcp_dim)
    # ... rest unchanged (control_action slicing, tail, etc.)
```

**D. `compute_action_mse()` — gt_action 同步逆转换**:
```python
@torch.no_grad()
def compute_action_mse(self, batch):
    obs = batch["obs"]
    gt_action = batch["action"]
    if getattr(self, 'use_faas', False):
        gt_action = self.faas_mapper.inverse_transform_action(gt_action, self.tcp_dim)
    pred_action = self.predict_action(obs)["pred_action"]
    return F.mse_loss(pred_action, gt_action).item()
```

**E. `control_action_dim` property — FAAS 模式返回 native dim**:
```python
@property
def control_action_dim(self):
    """Native control-space dimension (after FAAS inverse_transform)."""
    if getattr(self, 'use_faas', False):
        return self.tcp_dim + self.faas_mapper.NATIVE_HAND_DIM  # e.g. 7+12=19 or 9+12=21
    return self.action_dim
```

> **注意**: `predict_action_from_cond` 的 `inverse_transform_action` 已将 32D FAAS hand gather 为 12D native hand，输出 pred_action/control_action/tail 均为 native dim。`control_action_dim` slicing（行 130-132）对 FAAS 模式是 no-op（因为 pred 已是 native dim = control_action_dim）。

#### 3.4 修改 `training/build_utils.py` (~50 行)

新增:
- `STATE_ARM_DIM = 7` — 模块级常量，joint_state 臂部分固定 7D（关节角）
- `inject_faas_into_agent(agent, cfg)` — 共享 helper（train/eval/smoke-test 统一入口）
- `_validate_faas_config(cfg)` — 维度一致性校验：
  - `use_aux_ee` 互斥
  - `tcp_dim` 必须存在且为 7 或 9
  - `action_dim == tcp_dim + hand_dim`
  - `state_dim == STATE_ARM_DIM + hand_dim`（joint_state arm 固定 7D，与 action_key 无关）
  - MultiTaskDataset + `use_faas` → `NotImplementedError`
  - normalizer mode 强制为 `limits`（gaussian 模式补零维度数值不稳定）

修改:
- `build_dataset_and_normalizer()`:
  - 向 dataset 注入 `use_faas`、`faas_mapper`、`tcp_dim`（**在 `get_normalizer()` 调用之前**）
  - `get_normalizer()` 内部对 replay buffer 做 FAAS 转换后拟合
- `build_model_and_ema()`: 调用 `inject_faas_into_agent(model, cfg)` + EMA 同步

**`inject_faas_into_agent` 详细实现**:
```python
def inject_faas_into_agent(agent, cfg):
    """Post-construction FAAS injection for any entry point (train/eval/smoke-test)."""
    agent.use_faas = cfg.get('use_faas', False)
    if not agent.use_faas:
        return
    from dexmani_policy.common.faas_mapper import FAASHandMapper
    agent.tcp_dim = cfg.tcp_dim       # action arm dim (7 or 9)
    agent.hand_dim = cfg.get('hand_dim', cfg.get('faas_hand_dim', 32))
    agent.faas_mapper = FAASHandMapper()
```

> **建议**: 提供 `build_agent(cfg)` 工厂函数封装 `hydra.utils.instantiate(cfg.agent)` + `inject_faas_into_agent`，train/eval/smoke-test 统一调用，消除遗漏风险。

#### 3.5 新建 `configs/dp3_faas.yaml` + `configs/ddp/dp3_faas.yaml`

见 §2.1。

#### 3.6 修改 `eval_sim.py` (~10 行)

```python
# eval_sim.py run_eval() — 在 agent = hydra.utils.instantiate(cfg.agent) 之后:
from dexmani_policy.training.build_utils import inject_faas_into_agent
inject_faas_into_agent(agent, cfg)
```

#### 3.7 修改 `smoke_test.py` (~35 行)

- 行 132 `ctrl_shape` 断言：FAAS 感知（`native_action_dim = cfg.tcp_dim + 12 if cfg.use_faas else cfg.action_dim`）
- `_prepare_dqrise_codebook`: 优先使用 `cfg.hand_dim`（FAAS config 显式定义 `hand_dim: 12`），fallback `action_dim - tcp_dim`
- **新增 roundtrip 测试**:
  ```python
  if getattr(cfg, 'use_faas', False):
      from dexmani_policy.common.faas_mapper import FAASHandMapper
      mapper = FAASHandMapper()
      # Action roundtrip: 用通用逆变换公式验证
      native_action = torch.randn(4, 16, cfg.tcp_dim + 12)
      faas_action = mapper.transform_action(native_action, cfg.tcp_dim)
      native_rt = mapper.inverse_transform_action(faas_action, cfg.tcp_dim)
      assert torch.allclose(native_action, native_rt, rtol=1e-6), \
          f"FAAS action roundtrip error: {(native_action - native_rt).abs().max():.2e}"
      # Joint state roundtrip (arm=7D fixed)
      native_state = torch.randn(4, 19)
      faas_state = mapper.transform_joint_state(native_state)
      native_state_rt = mapper.faas_to_native(faas_state[:, 7:])
      assert torch.allclose(native_state[:, 7:], native_state_rt, rtol=1e-6), \
          f"FAAS hand roundtrip error: {(native_state[:, 7:] - native_state_rt).abs().max():.2e}"
      print("      ✓ FAAS roundtrip OK")
  ```
- `train_params` 追加 `use_faas` 和 `control_action_dim`

#### 3.8 修改 `training/trainer.py` (~3 行)

`finish_epoch` 的 `train_params` 追加:
```python
'use_faas': getattr(self.raw_model, 'use_faas', False),  # use raw_model for DDP safety
'control_action_dim': self.raw_model.control_action_dim,
```

#### 3.9 修改 `training/sim_evaluator.py` (~5 行)

`_load_for_inference` 追加 `use_faas` 一致性校验:
```python
ckpt_use_faas = train_params.get('use_faas', False)
agent_use_faas = getattr(agent, 'use_faas', False)
if ckpt_use_faas != agent_use_faas:
    raise ValueError(
        f"Checkpoint use_faas={ckpt_use_faas} but agent use_faas={agent_use_faas}. "
        f"Use matching config (dp3_faas for FAAS checkpoints, dp3 for native)."
    )
```

#### 3.10 修改 `scripts/train.sh` + `CLAUDE.md`

- `train.sh` 行 14: 配置列表追加 `dp3_faas`
- `CLAUDE.md`: 新增 §FAAS 段落

#### 3.11 验证

```bash
# 现有基线不受影响
python dexmani_policy/smoke_test.py dp3

# FAAS smoke test（含 roundtrip + normalizer FAAS 拟合验证）
python dexmani_policy/smoke_test.py dp3_faas

# 1 epoch 训练无 NaN（验证训练/推理管道完整）
bash scripts/train.sh dp3_faas 'training.loop.num_epochs=1'
```

---

### Step 2: 等价性验证

| 组 | 命令 | 动作空间 | 任务 | Epochs |
|----|------|---------|------|--------|
| A | `train.sh dp3` | 19D native | pour | 400 |
| B | `train.sh dp3_faas` | 39D FAAS | pour | 400 |

扩展到 `place_milk_box`。验证：loss 可比、SR 差距 < 3%、补零维度 std < 0.05。

### Step 3: 三轨对比

| 实验 | 命令 | 任务 | Epochs |
|------|------|------|--------|
| Joint | `train.sh dp3` | pour, place_milk_box | 400×2 |
| DQ-RISE | `train.sh dqrise` | pour, place_milk_box | 400×2 |
| FAAS | `train.sh dp3_faas` | pour, place_milk_box | 400×2 |

---

## 4. 文件变更清单

### 新建

| 文件 | 行数 | 说明 |
|------|------|------|
| `common/faas_mapper.py` | ~120 | FAASHandMapper (nn.Module) |
| `configs/dp3_faas.yaml` | ~25 | FAAS config，Hydra defaults 继承 dp3 |
| `configs/ddp/dp3_faas.yaml` | ~15 | FAAS DDP overlay |

### 修改

| 文件 | 变更 | 说明 |
|------|------|------|
| `datasets/base_dataset.py` | +50 行 | `__getitem__` 中 FAAS 转换 (ensure_tensor 后) + `get_normalizer` FAAS 适配 + `tcp_dim` 参数 |
| `agents/core/base.py` | +40 行 | `_convert_obs_to_faas` + 全量 inverse_transform + `compute_action_mse` + `control_action_dim` |
| `training/build_utils.py` | +50 行 | `STATE_ARM_DIM` 常量 + `inject_faas_into_agent` + `_validate_faas_config` (含 normalizer mode 校验) |
| `smoke_test.py` | +35 行 | FAAS 感知断言 + roundtrip 测试（通用逆变换公式） |
| `eval_sim.py` | +5 行 | 调用 `inject_faas_into_agent` |
| `training/trainer.py` | +3 行 | `train_params.use_faas` + `control_action_dim`（通过 raw_model） |
| `training/sim_evaluator.py` | +10 行 | `use_faas` 一致性校验 + 维度校验 |
| `scripts/train.sh` | +1 行 | help 文本追加 `dp3_faas` |
| `CLAUDE.md` | +25 行 | §FAAS 段落 |

### 不受影响

- **所有现有 config**（7 个 YAML）：**零修改**
- **所有 Agent 子类**：通过 `base.py` 统一处理
- **DQRISEAgent**：完全覆写 + `use_faas` 不触发
- **SimRunner / BaseRunner**：全量转换后 pred_action 已是 native
- **ChunkOverlapBlender**：零修改
- **Normalizer**：`build_mixed_action_normalizer` 自动适配 32D hand

---

## 5. 实验设计

### 5.1 对比协议

```
固定条件:
  任务: pour, place_milk_box
  数据: 同一 Zarr (80 train episodes)
  Encoder: iDP3 (pc_dim=6, pc_out_dim=128)
  Backbone: ConditionalUnet1D (down=[256,512,1024])
  lr: 1e-4 | batch: 128 | EMA | seed: 42
  Epochs: 400 per config per task | eval: 100 episodes

变量:
  A: dp3       (19D native)
  B: dqrise    (21D, action_ee; diffusion outputs 10D VQ index)
  C: dp3_faas  (41D FAAS, action_ee)
```

### 5.2 假设

| # | 假设 | 预期 | 验证 |
|----|------|------|------|
| H1 | FAAS 不降低性能 | SR ≥ native - 3% | pour + place_milk_box |
| H2 | DQ-RISE VQ 先验有效 | SR ≥ native | 逐任务对比 |
| H3 | 补零维度稳定 | std < 0.05 | 训练中监控 |
| H4 | 收敛不慢于 native | 80% SR epoch ≤ native + 20% | 收敛曲线 |

---

## 6. 向后兼容性

| 场景 | 结果 | 原因 |
|------|------|------|
| 旧 config (全部) + 新代码 | ✅ 正常 | `cfg.get('use_faas', False)` → 原路径 |
| 新 FAAS config + 新代码 | ✅ 正常 | 全链路 FAAS |
| 旧 checkpoint + 新代码 (native config) | ✅ 正常 | `action_dim` 校验通过 |
| 旧 checkpoint + FAAS config | ✅ 安全拒绝 | `action_dim` 不匹配 + `use_faas` 校验 |
| FAAS checkpoint + native config | ✅ 安全拒绝 | 同上 |

所有新增代码使用 `getattr(..., 'use_faas', False)` 或 `cfg.get('use_faas', False)` 守卫。

---

## 7. 风险与缓解

| # | 风险 | 概率 | 缓解 |
|---|------|------|------|
| R1 | FAAS 成功率显著低于 native | 中 | roundtrip 测试 + index_bend 符号翻转验证 |
| R2 | Hydra `defaults` 覆写不完整 | 低 | smoke test 覆盖完整构建链；DDP 已有先例 |
| R3 | 旧 checkpoint 误用 | 低 | SimEvaluator `action_dim` + `use_faas` 双重校验 |
| R4 | `use_aux_ee` + FAAS 同时启用 | 低 | `validate_config` 互斥检查 |
| R5 | 评测入口遗漏 FAAS 注入 | **已修复** | `inject_faas_into_agent` 共享 helper |

---

## 8. 附录

### A. FAAS 维度速查

```
                     Native                FAAS
                     ──────                ────
action (joint):      19 =  7+12           39 =  7+32
action (ee):         21 =  9+12           41 =  9+32
joint_state:         19 =  7+12           39 =  7+32

FAAS 活跃索引: (1,2,3, 6,7,8, 12,13, 17,18, 22,23)
              thumb×3  index×3  mid×2   ring×2  pinky×2
仅 index_bend (native[3]→FAAS[6]) scale=-1.0
```

### B. 数据流不变量

```
训练: Zarr native → FAAS transform → Normalizer.fit → Agent(FAAS空间)
推理: Agent(FAAS采样) → unnormalize → inverse_transform → native
      所有对外接口（pred_action/control_action/tail）均为 native
```

### C. v5 审查修订记录（2026-07-27）

Ultracode 深度审查（3 专家 × 3 维度 + 1 综合仲裁）。发现 4 P0 + 8 P1 + 10 P2。全部 P0/P1 已修复：

| # | 严重度 | 发现 | 修复 |
|---|--------|------|------|
| 1 | **P0** | 推理路径缺失 obs joint_state native→FAAS 正向转换 | `predict_action()` 开头新增 `_convert_obs_to_faas()` |
| 2 | **P0** | `get_normalizer()` 绕过 FAAS 转换，在 native 数据上拟合 | `get_normalizer()` 内对 replay buffer 做 FAAS 转换 |
| 3 | **P0** | `_apply_faas_mapping` 使用 torch API 但数据仍是 numpy | 移至 `__getitem__` 中 `ensure_tensor` 之后执行 |
| 4 | **P0** | joint_state 键路径错误 (`data['joint_state']` → `data['obs']['joint_state']`) | 修正为正确嵌套路径 |
| 5 | **P1** | action_ee 下 `state_dim` 设为 41 实际应为 39 | `state_dim` 恒为 `${eval:'7 + ${faas_hand_dim}'}` = 39 |
| 6 | **P1** | `tcp_dim` 错误用于 joint_state arm 分割（action_ee 下取 9D 而非 7D） | 引入 `STATE_ARM_DIM=7` 常量，`_apply_faas_mapping` 中 arm 分割用 7 |
| 7 | **P1** | 腕部相对 Delta 差异仅在附录提及 | 新增 §2.7 主体设计说明 |
| 8 | **P1** | normalizer 等价性声明事实性错误 | 新增 §2.8 精确描述差异 + `validate_config` 强制 limits 模式 |
| 9 | **P1** | `faas_to_native` 使用简化逆变换公式 | 改用通用公式 `(faas-offset)/scale` |
| 10 | **P1** | DQ-RISE PCA 权重重训需求未标注 | §2.5 新增标注 |
| 11 | **P1** | checkpoint `train_params` 缺 `use_faas` | trainer 通过 `raw_model` 存储 |
| 12 | **P1** | `tcp_dim` 未传递给 Dataset | `BaseDataset.__init__` 新增参数 |

### D. v4 审查修订记录（2026-07-27）

多 agent 审查发现的 3 个 P0 阻断项均已修复：

| # | 发现 | 修复 |
|---|------|------|
| 1 | `eval_sim.py` 绕过 `build_model_and_ema`，FAAS 属性未注入 | 新增 `inject_faas_into_agent()` 共享 helper；`eval_sim.py` 调用 |
| 2 | `MultiTaskDataset` 子数据集 FAAS 不传播 | Phase 1 加 `NotImplementedError` guard；Phase 2 递归注入 |
| 3 | Smoke test 缺少 roundtrip 测试 | 显式实现 action + joint_state roundtrip 断言 |

P1 修复：`train_params` 新增 `control_action_dim`；`SimEvaluator` 新增 `use_faas` 校验；新增 `ddp/dp3_faas.yaml`；`_validate_faas_config` 完整维度校验。

### E. 执行优先级

```
P0: Step 1 — faas_mapper.py + 管线代码 + dp3_faas.yaml + smoke test (含 roundtrip + normalizer FAAS 拟合)
    修复清单: predict_action obs转换 / get_normalizer FAAS转换 / __getitem__ 时序 / 键路径
P1: Step 2 — dp3 vs dp3_faas 等价性验证 (pour + place_milk_box, 400ep)
P2: Step 3 — dp3 vs dqrise vs dp3_faas 三轨对比
P3: 扩展 maniflow_faas.yaml / r3d_faas.yaml / MultiTask+FAAS（如验证通过）
```
