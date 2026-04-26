# UMI vs GalaxeaDP 相对动作设计机制深度对比

> 核心结论：两者存在**本质不同** — UMI 的相对动作在**数据集加载时（`__getitem__`）**完成转换，而 GalaxeaDP 的相对动作在**模型内部（`SignalTransform.forward()`）**延迟转换。此外，UMI 支持三种相对模式（`rel` / `relative` / `delta`），GalaxeaDP 仅支持一种（等价于 UMI 的 `relative`）。

---

## 一、转换发生的位置

### 1.1 GalaxeaDP：模型内延迟转换（Lazy Transform）

```
LeRobot磁盘(绝对) ──► Dataset.__getitem__(绝对) ──► Policy.compute_loss()
                                                          │
                                                    SignalTransform.forward()
                                                    此处才转换为相对
                                                          │
                                                    网络训练
```

**代码证据**：

```python
# src/data/galaxea_lerobot_dataset.py — __getitem__
action = lerobot_sample[f"action.{key}"]  # 原始绝对位姿，无转换
sample["action"][key] = action             # 直接返回

# src/models/policy/diffusion_unet_image_policy.py — compute_loss
batch = self.signal_transform.forward(batch)  # ← 在模型内转换
nobs = self.normalizer.normalize(batch["obs"])
```

**设计特点**：
- 数据在磁盘和 DataLoader 中始终保持绝对格式
- 转换发生在 `forward()` 中，紧贴网络输入
- 归一化统计量通过独立的 `SignalTransform` 实例在相对空间上计算（`get_norm_stats()`）

### 1.2 UMI：数据集加载时即时转换（Eager Transform）

```
Zarr磁盘(绝对) ──► Sampler.sample_sequence(绝对) ──► Dataset.__getitem__()
                                                          │
                                                    convert_pose_mat_rep()
                                                    此处转换为相对
                                                          │
                                                    返回相对动作给 DataLoader
```

**代码证据**：

```python
# diffusion_policy/dataset/umi_dataset.py — __getitem__, 第343-353行
pose_mat = pose_to_mat(...)           # 当前观测位姿
action_mat = pose_to_mat(...)         # 动作位姿

# 观测转换
obs_pose_mat = convert_pose_mat_rep(
    pose_mat, base_pose_mat=pose_mat[-1],
    pose_rep=self.obs_pose_repr, backward=False)

# 动作转换 — 使用同一个 base_pose（最后一帧观测）
action_pose_mat = convert_pose_mat_rep(
    action_mat, base_pose_mat=pose_mat[-1],
    pose_rep=self.action_pose_repr, backward=False)
```

**设计特点**：
- 转换在 `__getitem__()` 中完成，DataLoader 返回的已是相对动作
- 归一化统计量直接在相对数据上计算（`get_normalizer()` 遍历 dataset 采样）
- 推理时反向转换也在**数据集外部**完成（`get_real_umi_action()`）

### 1.3 本质差异

| 维度 | GalaxeaDP（Lazy） | UMI（Eager） |
|------|-------------------|-------------|
| 转换位置 | `Policy.compute_loss()` 内部 | `Dataset.__getitem__()` 内部 |
| 磁盘存储 | 绝对位姿 | 绝对位姿 |
| DataLoader输出 | 绝对位姿 | 相对位姿 |
| 归一化计算基础 | 需额外 `SignalTransform` 预转换 | 直接遍历 dataset 采样（已是相对） |
| 推理反向转换 | `SignalTransform.backward()` 在模型内 | `get_real_umi_action()` 在模型外 |
| 可切换性 | 通过 `use_relative_control` 配置开关 | 通过 `obs_pose_repr` / `action_pose_repr` 配置切换 |
| 数据可调试性 | DataLoader 输出是绝对，需进入模型才能看到相对 | DataLoader 输出直接是相对，易于外部检查 |

---

## 二、相对模式的种类与语义

### 2.1 GalaxeaDP：单一相对模式

GalaxeaDP 只有一种相对模式，通过 `use_relative_control` 布尔开关控制：

```python
# SignalTransform.forward()
if self.use_relative_control:
    base_pose = batch["obs"][obs_key][:, -1:, :]  # 当前位姿
    cur_action = self.relative_pose_transformer.forward(cur_action, base_pose)
```

数学：`T_rel[k] = T_obs^{-1} @ T_abs[k]`，整个 chunk 的 32 步动作都相对于**同一个**当前观测位姿 `T_obs`。

**NOT 逐帧增量**：第 k 步动作不是相对于第 k-1 步，而是全部相对于同一基准。

### 2.2 UMI：三种相对模式

UMI 通过 `pose_repr` 配置支持三种模式：

```yaml
# diffusion_policy/config/task/umi.yaml
pose_repr:
  obs_pose_repr: relative      # abs / rel / relative / delta
  action_pose_repr: relative   # abs / rel / relative / delta
```

#### 模式 1：`abs`（绝对）

```python
# pose_rep == 'abs': 不做任何转换
return pose_mat  # 直接返回绝对位姿
```

#### 模式 2：`rel`（逐元素差分，非矩阵运算）

```python
# pose_rep == 'rel':
output_pos = pos - base_pos                              # 位置逐元素相减
output_rot = rot_transformer.forward(rot_mat @ inv(base_rot_mat))  # 旋转矩阵右乘逆
```

**注意**：这是 UMI 的 legacy 实现，位置和旋转的处理方式不对称。位置是直接相减（欧氏空间差分），旋转是矩阵乘法（李群差分）。

#### 模式 3：`relative`（齐次变换，与 GalaxeaDP 等价）

```python
# pose_rep == 'relative':
out = inv(base_pose_mat) @ pose_mat
```

**这与 GalaxeaDP 的 `absolute_to_relative` 完全相同**：`T_rel = T_base^{-1} @ T_abs`。

#### 模式 4：`delta`（逐帧增量，frame-to-frame）

```python
# pose_rep == 'delta':
# 训练时：将绝对序列转换为逐帧差分
all_pos = concat([base_pos, pos], axis=0)
out_pos = diff(all_pos, axis=0)          # Δp[k] = p[k] - p[k-1]

all_rot_mat = concat([base_rot_mat, rot_mat], axis=0)
out_rot = rot_mat[1:] @ inv(rot_mat[:-1])  # ΔR[k] = R[k] @ R[k-1]^{-1}
```

**这是与 GalaxeaDP 最大的本质区别**：

| | GalaxeaDP | UMI `relative` | UMI `delta` |
|---|---|---|---|
| 基准 | 所有步骤共享 `T_obs` | 所有步骤共享 `T_obs` | 每步相对于前一步 |
| 数学 | `T_rel[k] = T_obs^{-1} @ T[k]` | 同上 | `ΔT[k] = T[k-1]^{-1} @ T[k]` |
| 推理恢复 | `T[k] = T_obs @ T_rel[k]` | 同上 | `T[k] = T[k-1] @ ΔT[k]`（累积） |
| 误差传播 | 无（每步独立） | 无（每步独立） | 有（误差沿链累积） |

#### 推理时的反向转换对比

**GalaxeaDP 恢复绝对动作**：
```python
# 所有步骤共享同一 base_pose
cur_action = self.pose_rotation_transformer.backward(cur_action)
if self.use_relative_control:
    base_pose = batch["obs"][obs_key][:, -1:, :]
    cur_action = self.relative_pose_transformer.backward(cur_action, base_pose)
# T_abs[k] = T_obs @ T_rel[k]，所有 k 并行计算
```

**UMI `relative` 模式恢复**：
```python
# convert_pose_mat_rep(..., pose_rep='relative', backward=True)
out = base_pose_mat @ pose_mat  # 与 GalaxeaDP 相同
```

**UMI `delta` 模式恢复**：
```python
# convert_pose_mat_rep(..., pose_rep='delta', backward=True)
output_pos = cumsum(pos) + base_pos          # 累积和
curr_rot = base_rot
for i in range(len(rot_mat)):
    curr_rot = rot_mat[i] @ curr_rot         # 链式累积
    output_rot_mat[i] = curr_rot
```

**关键差异**：`delta` 模式使用**逐帧累积**（`cumsum` + 链式矩阵乘法），而 `relative` 模式和 GalaxeaDP 使用**单步并行**（每个时间步独立乘基准）。

---

## 三、观测的相对表示

### 3.1 GalaxeaDP：双表示拼接

GalaxeaDP 对 qpos（观测状态）生成**两种**相对表示并拼接：

```python
# 1. 相对于 episode 起点（带噪声增强）
base_pose = batch["obs"][f"episode_start_{key}"]
if self.training:
    base_pose = self.pose_rotation_transformer.add_noise(base_pose)
ee_pose_wrt_start = relative_transformer.forward(cur_qpos, base_pose)

# 2. 相对于当前位姿
base_pose = cur_qpos[:, -1:, :]
cur_qpos = relative_transformer.forward(cur_qpos, base_pose)

# 拼接：[wrt_start, wrt_current, gripper, ...]
batch["obs"]["qpos"] = torch.cat([ee_left_wrt_start, ee_left_wrt_current, gripper_left, ...])
```

### 3.2 UMI：多表示独立字段

UMI 在 `__getitem__()` 中生成多种相对观测，但作为**独立字段**存在：

```python
# 1. 相对于另一台机器人（bimanual 场景）
rel_obs_pose_mat = convert_pose_mat_rep(pose_mat, base=other_pose_mat[-1], pose_rep='relative')
obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]

# 2. 相对于 episode 起点（带噪声）
start_pose_mat = pose_to_mat(start_pose + noise)
rel_obs_pose_mat = convert_pose_mat_rep(pose_mat, base=start_pose_mat, pose_rep='relative')
obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

# 3. 相对于当前位姿（由 obs_pose_repr 控制主观测）
obs_pose_mat = convert_pose_mat_rep(pose_mat, base=pose_mat[-1], pose_rep=obs_pose_repr)
```

### 3.3 对比

| 维度 | GalaxeaDP | UMI |
|------|-----------|-----|
| episode_start 相对观测 | 是，拼接到 qpos 中 | 是，独立字段 `_wrt_start` |
| 跨机器人相对观测 | 无 | 是，`_wrt{other_robot_id}` |
| 噪声增强 | `add_noise()` 函数（四元数空间） | `np.random.normal(scale=0.05)`（6D 空间） |
| 拼接方式 | `torch.cat` 合并为单一向量 | 独立 key，由 `shape_meta` 控制哪些进入网络 |

---

## 四、旋转表示的差异

### 4.1 GalaxeaDP

- **输入**：四元数（x, y, z, i, j, k, r）共 7 维
- **支持转换**：quaternion ↔ rotation_6d ↔ rotation_9d
- **相对变换**：在 4x4 齐次矩阵空间进行（四元数 → 矩阵 → 相对 → 四元数）

### 4.2 UMI

- **输入**：axis-angle（旋转向量）共 3 维 + 位置 3 维 = 6 维
- **支持转换**：axis-angle → rotation_6d（用于网络输入，`mat_to_pose10d`）
- **相对变换**：在 4x4 齐次矩阵空间进行（axis-angle → 矩阵 → 相对 → pose10d）
- **输出表示**：`pose10d` = position(3) + rotation_6d(6) + gripper(1) = 10 维/臂

### 4.3 对比

| 维度 | GalaxeaDP | UMI |
|------|-----------|-----|
| 原始旋转表示 | 四元数（4D） | 旋转向量 axis-angle（3D） |
| 网络输入旋转表示 | quaternion(4) / 6D(6) / 9D(9) | rotation_6d(6) |
| 单臂动作维度 | 7+1=8（quaternion）或 9+1=10（6D） | 10（pose10d） |
| 双臂总维度 | 2×8+2=18 或 2×10+2=22 | 2×10=20 |
| 插值方式 | 无（PyTorch tensor） | Slerp（scipy.spatial.transform） |

---

## 五、推理执行策略

### 5.1 GalaxeaDP

```
每步重新推理：
  obs(t) → SignalTransform.forward → 网络 → SignalTransform.backward → 32步绝对动作
  执行第 t 步，丢弃其余 31 步
  obs(t+1) → 重新推理...
```

- 每步都利用最新观测
- 无开环执行，误差不会沿 chunk 累积

### 5.2 UMI

```
执行多步后重新推理：
  obs(t) → 网络 → 16步相对动作 → get_real_umi_action() → 16步绝对动作
  执行 steps_per_inference 步（默认 6 步）
  obs(t+6) → 重新推理...
```

- 存在开环执行窗口（`steps_per_inference=6`）
- 在 `relative` 模式下，误差不会沿执行链累积（每步都是相对于同一基准）
- 在 `delta` 模式下，误差会沿执行链累积

---

## 六、总结：本质差异一览

| 差异点 | GalaxeaDP | UMI | 影响 |
|--------|-----------|-----|------|
| **转换时机** | 模型内（Lazy） | 数据加载时（Eager） | UMI 更易于外部调试，GalaxeaDP 更灵活（可在模型内切换开关） |
| **相对模式** | 单一（单基准并行） | 三种（abs / relative / delta） | UMI 的 `delta` 模式支持逐帧增量，是 GalaxeaDP 不具备的 |
| **Delta 模式** | 不存在 | `ΔT[k] = T[k-1]⁻¹ @ T[k]`，逐帧累积 | **最大的本质差异** — 这是两种完全不同的相对语义 |
| **观测相对表示** | 双表示拼接（wrt_start + wrt_current） | 多字段独立（wrt_start, wrt_other, wrt_current） | UMI 支持跨机器人相对观测（bimanual），GalaxeaDP 不支持 |
| **旋转表示** | 四元数 ↔ 6D/9D | axis-angle → 6D（pose10d） | 不同的原始表示，但相对变换都在矩阵空间进行 |
| **噪声增强** | `add_noise()` 在四元数→轴角空间 | `np.random.normal` 直接在 6D 向量上 | GalaxeaDP 的噪声更符合旋转几何 |
| **推理执行** | 每步重新推理（32步输出，执行1步） | 多步执行后推理（16步输出，执行6步） | UMI 有开环执行窗口，推理延迟要求更低 |

### 一句话总结

**GalaxeaDP 的相对动作是"所有步骤共享同一基准"的并行相对（等价于 UMI 的 `relative` 模式），在模型内延迟转换；UMI 额外提供了 `delta` 模式——逐帧增量的链式相对，在数据集加载时即时转换。两者在数学上都是 `T_rel = T_base^{-1} @ T_abs`，但 UMI 的 `delta` 模式的 `T_base` 随时间步变化（`T_base[k] = T[k-1]`），而 GalaxeaDP 的 `T_base` 固定为当前观测帧。**
