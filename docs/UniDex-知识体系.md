# UniDex 知识体系：从论文到代码的完整梳理

> **Paper**: *UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos* (CVPR 2026, arXiv:2603.22264)
> **Code**: https://github.com/unidex-ai/UniDex
> **Authors**: Gu Zhang, Qicheng Xu, Haozhe Zhang, Jianhan Ma et al. (Tsinghua, Shanghai Qizhi, SYSU, UNC)
> **Project Page**: https://unidex-ai.github.io/
> **Models**: https://huggingface.co/UniDex-ai/UniDex
> **本地副本**: `/home/zhanghaoyang/Desktop/UniDex`

---

## 目录

1. [问题定义与动机](#1-问题定义与动机)
2. [架构全景图](#2-架构全景图)
3. [各数据集实现详解](#3-各数据集实现详解)
4. [Pose 表示与手部机制](#4-pose-表示与手部机制)
5. [FAAS 统一动作空间](#5-faas-统一动作空间)
6. [点云编码器 Uni3D](#6-点云编码器-uni3d)
7. [VLA 模型架构 (PointCloudUniDex)](#7-vla-模型架构-pointcloudunidex)
8. [Flow Matching 动作解码](#8-flow-matching-动作解码)
9. [多 Mixture 联合推理 (JointModel)](#9-多-mixture-联合推理-jointmodel)
10. [训练管道](#10-训练管道)
11. [推理管道与 KV Cache 优化](#11-推理管道与-kv-cache-优化)
12. [HandAdapter 手部 Retargeting](#12-handadapter-手部-retargeting)
13. [数据处理管线](#13-数据处理管线)
14. [关键设计决策](#14-关键设计决策)
15. [与 DexMani_Policy 的完整对比](#15-与-dexmani_policy-的完整对比)
16. [已知问题与代码审查发现](#16-已知问题与代码审查发现)
17. [可借鉴设计建议](#17-可借鉴设计建议)

---

## 1. 问题定义与动机

### 1.1 核心问题

灵巧手操作面临的三大挑战：

```
挑战 1: 数据稀缺          → 遥操作数据昂贵，难以规模化
挑战 2: 具身异构          → 不同灵巧手 DoF (6-24)、形态、运动学差异巨大
挑战 3: 控制高维度        → 多指协调 + 臂手联合控制
```

### 1.2 UniDex 的核心洞察

> **个人类视频充足、多样化、廉价 → 通过手部 Retargeting 转为机器人数据 → 大规模预训练 → 小样本真机微调。**

四步法：
1. **UniDex-Dataset**: 4 个人类自我中心视频数据集 → IK Retargeting → 8 种灵巧手的 50K+ 轨迹
2. **FAAS (Function-Actuator-Aligned Space)**: 不同手的**功能相似关节映射到统一 82D 空间**的相同索引
3. **UniDex-VLA**: PaliGemma 3B (Gemma 语言模型) + Uni3D 点云编码器 → Flow Matching 动作生成
4. **UniDex-Cap**: Apple Vision Pro + RealSense → 人-机器人协同数据采集

### 1.3 核心结果

| 模型 | 5 任务平均进度 | 每任务所需机器人 Demo |
|------|---------------|---------------------|
| DP | 29.0% | ~200+ |
| DP3 | 35.0% | ~200+ |
| π₀ | 38.0% | ~200+ |
| **UniDex-VLA** | **81.0%** | **50** |

- 零样本跨手迁移: Wuji→40%, Oymotion→60%（仅 Inspire Hand 训练）
- 人类 demo 替代比: ~2 人类 demo ≈ 1 机器人 demo，采集速度快 5.2×

### 1.4 消融实验与 Scaling 分析

**预训练是关键**。No Pretrain 变体（仅在 50 个 per-task demo 上训练）的平均进度为 **32.5%**，完整模型为 **81.0%**。差距最大的是 Cut Bags 任务（32.5% → 90.0%，相对提升 84.6%）。

**人类-Robot 协同训练 Scaling**:
- 0 机器人 demo + 纯人类 demo → 任务进度为 0（人类数据单独无效）
- 10 机器人 demo + 人类 demo → 性能开始稳步提升
- ~2 人类 demo ≈ 1 机器人 demo（高效边界斜率约 2）
- 人类 demo 采集速度快 **5.2×**

**零样本跨手迁移** (策略仅在 Inspire Hand 上训练):

| 目标手 | π₀ | UniDex-VLA (No Pretrain) | UniDex-VLA |
|--------|-----|--------------------------|------------|
| Wuji (20 DoF) | 0% | 0% | **40%** |
| Oymotion (11 DoF) | 10% | 5% | **60%** |

> 基线方法在未见手上几乎完全失效。FAAS 空间 + 多手预训练是实现零样本迁移的关键。

**数据规模对比**:

| 数据集 | 轨迹数 | 手部种类 | 语言标注 | 场景多样性 | 点云质量 |
|--------|--------|---------|---------|-----------|---------|
| **UniDex-Dataset** | **52K** | **8** | ✓ | ✓ | 高 |
| ActionNet | 30K | 2 | ✓ | ✗ | 低 |
| RoboMind | 19K | 1 | ✓ | ✗ | 无 |
| RealDex | 2K | 2 | ✓ | ✗ | 高 |

内部统计: **9M 配对图像-点云-动作帧**，跨越 6-24 个主动 DoF。

---

## 2. 架构全景图

### 2.1 端到端数据流（预训练）

```
┌─────────────────────────────────────────────────────────────────┐
│          4× Egocentric Human Video Datasets                     │
│    H2O / HOI4D / Hot3D / Taco (原始 RGBD + MANO 手部姿态)       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  HandAdapter            │
              │  - MANO 手模型 → IK求解  │
              │  - 指尖轨迹对齐          │
              │  - 人手→机器人手关节映射  │
              │  - RGBD → 点云投影       │
              │  → HDF5: frames/{rgb,   │
              │    depth, joints, poses} │
              └────────────┬────────────┘
                           │ 8 种手 × 4 数据集 = 50K+ 轨迹
              ┌────────────▼────────────┐
              │  BaseDataset            │
              │  - pickle 缓存          │
              │  - FPS 降采样到 1024 点 │
              │  - MixtureDataset 拼接   │
              └────────────┬────────────┘
                           │ batch: {pointcloud, state, action, prompt}
              ┌────────────▼────────────┐
              │  VLAProcessor           │
              │  - PaliGemma tokenizer  │
              │  - <image> tokens +     │
              │    text prompt 拼接      │
              └────────────┬────────────┘
                           │ input_ids, attention_mask
              ┌────────────▼────────────┐
              │  PointCloudUniDexTrain  │
              │  ┌────────────────────┐ │
              │  │ Uni3D 点云编码器    │ │
              │  │ Group(FPS+KNN)+    │ │
              │  │ PointConv+ViT      │ │
              │  └────────┬───────────┘ │
              │           │ pcd_tokens  │
              │  ┌────────▼───────────┐ │
              │  │ Projector          │ │
              │  │ Linear(768→2048)   │ │
              │  └────────┬───────────┘ │
              │           │             │
              │  ┌────────▼───────────┐ │
              │  │ JointModel (18层)  │ │
              │  │ ┌────────────────┐ │ │
              │  │ │ vlm mixture    │ │ ← text + pcd tokens
              │  │ │ proprio mixt.  │ │ ← state tokens
              │  │ │ action mixture │ │ ← noisy action tokens
              │  │ └────────────────┘ │ │
              │  │ Block Attention    │ │
              │  │ GQA (8Q/1KV) + RoPE│ │
              │  └────────┬───────────┘ │
              │           │             │
              │  ┌────────▼───────────┐ │
              │  │ Action Decoder     │ │
              │  │ Linear(1024→82)    │ │
              │  └────────────────────┘ │
              │  Flow Matching Loss:   │
              │  MSE(v_pred, x₁-x₀)    │
              └────────────────────────┘
```

### 2.2 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| PointCloudUniDex | `src/unidex/unidex.py` | 主模型：embed_tokens + Uni3D + Projector + JointModel + Action Decoder |
| JointModel | `src/unidex/joint_model.py` | 多 Mixture Transformer：vlm/proprio/action 三流并行 + Block Attention |
| Mixture | `src/unidex/mixture.py` | 单 Mixture = 18× MixtureDecoderLayer (GQA + GemmaMLP + Adaptive Norm) |
| Uni3D | `src/pointcloud_encoder/uni3d.py` | 点云分层编码：FPS+KNN Grouping → PointConv → ViT |
| ActionEncoder | `src/unidex/modules.py` | 动作嵌入 + 时间条件注入 |
| AdaptiveRMSNorm | `src/unidex/modules.py` | adaLN / adaLN-Zero 时间条件 Norm |
| VLAProcessor | `src/utils/processing.py` | PaliGemma tokenizer wrapper + `<image>` token 插入 |
| BaseDataset | `src/dataset/base.py` | 抽象数据集基类：序列发现、窗口构建、FPS 采样、pickle 缓存 |
| MixtureDataset | `src/dataset/mixture.py` | 多数据集拼接 + 维度 padding + Normalizer 合并 |
| Normalizer | `src/utils/normalizers.py` | meanstd / minmax / identity 归一化 + 跨数据集合并 |

### 2.3 类继承关系

```
nn.Module
├── PointCloudUniDex                         (unidex.py:20)
│   ├── PointCloudUniDexTrain                (unidex.py:883)
│   └── PointCloudUniDexInference            (unidex.py:814)
├── JointModel                               (joint_model.py:308)
│   └── .mixtures: ModuleDict[str, Mixture]  → {vlm, proprio, action}
│       └── Mixture                          (mixture.py:17)
│           └── .layers: ModuleList[MixtureDecoderLayer]
│               ├── MixtureAttention          (GQA: 8头Q, 1头KV, head_dim=256)
│               ├── GemmaMLP                  (gate/up/down, GeLU approx=tanh)
│               └── AdaptiveRMSNorm / GemmaRMSNorm
├── Uni3D                                    (uni3d.py:167)
│   ├── Group                                (FPS + KNN grouping, → 6D特征)
│   ├── Encoder                              (PointConv: Conv1d→MaxPool→Conv1d)
│   ├── PatchDropout                         (训练时随机丢弃 patch token)
│   └── .visual: ViT (timm)                  (eva02_{tiny/small/base/large/giant})
├── ActionEncoder                            (modules.py:26)
├── SinusoidalPosEmb                         (modules.py:10)
└── PaliGemmaMultiModalProjector             (modules.py:122)
    └── Linear(pc_feat_dim → 2048)           (可选 LoRA/4bit量化)

L.LightningModule
└── LightningTrainingWrapper                  (train.py:46)
```

---

## 3. 各数据集实现详解

### 3.1 H2oDataset — 动作标签驱动分割

**文件**: `src/dataset/H2o.py` (290 行)

H2o 是唯一有**逐帧动作标签**的数据集。37 种操作类别（grab/place/open/close/pour 等）。

**序列发现** (`_find_sequences`):
```
1. 扫描 retarget_RGBD/{subject}/{session}/{seq}/{camera}/{hand_type}.h5
2. 解析 action_label/ 目录下的 .txt 标签文件
3. 使用 skip sampling (step=64) + 二分查找检测标签变化点
4. 按 action label 将长序列拆分为子序列（过滤 background 标签）
5. prompt 模板: "Use {hand_type} hands to {ACTION_LABELS[label]}."
```

**标签解析算法** (`_parse_action_labels`, 行 33-78):
- 每隔 64 帧采样一个标签（减少 I/O）
- 相邻采样点的标签变化时，二分查找精确变化点
- 过滤全零标签序列（无操作片段）

**窗口构建** (`_build_window`, 行 190-241):
- 滑动窗口步长: `chunk_size // 6`（与所有数据集一致）
- 最短要求: `chunk_size + max(state_horizon, pcd_horizon)` 帧
- 如为子序列（含 start_frame/end_frame），窗口范围限制在子序列内

**场景 RGBD 加载** (`_load_scene_rgbd`, 行 247-290):
- 路径: `all_img/{relative_path}/rgb/{frame:06d}.png`
- 深度: mm 转 m (`/ 1000.0`)
- 可选 mask 过滤（手掌区域）

### 3.2 Hot3DDataset — Prompt 文件驱动分割

**文件**: `src/dataset/Hot3D.py` (246 行)

Hot3D 使用**人工标注的 prompt.txt** 文件定义每个子序列。

**Prompt 文件格式**:
```
action_description start_timestamp end_timestamp
# 例: "grasp the cup 59211074860121 59212241516144"
```

**序列发现** (`_find_sequences`, 行 36-63):
- 查找每个 H5 文件同级目录的 `prompt.txt`
- 解析格式: `action_description start_ts end_ts`
- 时间戳 → 帧索引转换（逆序遍历匹配）
- prompt 模板: `"Use {hand_type} hands to {action_description}."`

**坐标系转换** (类常量):
```python
TRANSFORM = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])  # Hot3D 特有: x→y, y→-x 旋转
```

**场景数据格式**: RGB 用 jpg，深度用 tiff，时间戳作为文件名。

### 3.3 TacoDataset — 目录名驱动 Prompt

**文件**: `src/dataset/Taco.py`

Taco 从**目录命名约定**中提取操作描述。

**Prompt 提取** (`_find_sequences`, 行 28-57):
```python
# 目录格式: "(verb, object, tool)"
# 例: "(cut, bag, scissors)" → "cut the bag"
prime_relative_path = str(relative_path).split('/')[0]
words = prime_relative_path[1:-1].split(',')
prompt_content = f'{words[0].strip()} the {words[2].strip()}'
```

**点云裁剪** (PCD_MASK):
```python
PCD_MASK = np.array([[-0.4, -0.3, -0.9], [0.3, 0.4, -0.45]])
# 裁剪桌面和远处背景
```

### 3.4 RealDataset — 真机数据双格式支持

**文件**: `src/dataset/Real.py` (814 行，最复杂的数据集)

支持两种 Zarr 格式和生成增强数据:

**V1 格式** (遥操作采集):
```
data/
├── action (N, 26)                    # 完整动作
├── camera0_pointcloud (N, 10000, 6)  # 预计算点云
├── camera0_rgb (N, 224, 224, 3)      # RGB 图像
├── robot0_eef_pos (N, 3)             # 末端位置
├── robot0_eef_rot_axis_angle (N, 3)  # 轴角旋转
├── gripper0_gripper_pose (N, 20)     # 手部关节
└── ...
meta/
└── window_ends (K,)                  # 窗口边界
```

**V2 格式** (retarget 数据):
```
data/
├── pointcloud (N, 10000, 6)
├── right_joint_poses (N, 4, 4)      # 4×4 变换矩阵
├── right_joint_values (N, J)         # 关节值
└── ...
```

**生成增强数据** (可选 `use_generated_data`):
- 独立 Zarr 文件，含 `pointcloud_camera0_base` + `state_camera0_base`
- 保留原始关节值但替换腕部位姿（实现空间泛化）
- 通过 `sequence_num` 参数限制使用数量

**单侧手处理** (hand_side):
- 仅激活侧手（如 'right'）使用真实数据
- 非激活侧手使用单位矩阵（腕部）和零值（关节）

**点云预处理**:
```python
CV_TO_CAM = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]  # CV→Camera 坐标
EXTRINSICS = [...]  # 相机外参 (标定获得)
PCD_MASK = [-0.04, 0.33, 0.89, 0.78]  # 裁剪桌面
```

### 3.5 四数据集 Prompt 模板对比

| 数据集 | Prompt 来源 | 模板示例 |
|--------|-----------|---------|
| H2o | ACTION_LABELS 字典 | "Use Inspire hands to grab book." |
| HOI4D | action_label 文件 | "Use Leap hands to open the bottle." |
| Hot3D | 人工标注 prompt.txt | "Use Inspire hands to grasp the cup." |
| Taco | 目录命名约定 | "Use Wuji hands to cut the bag." |
| Real | 配置中直接指定 | "Use Wuji hands to spray." |

**多样性**: H2o 的标签是枚举值（37 类），Hot3D 是自由文本，Taco 是动词-宾语结构。这为 VLM 提供了丰富的语言监督信号。

---

## 4. Pose 表示与手部机制

### 4.1 Pose 表示约定

**文件**: `src/utils/pose.py` (109 行)

UniDex 使用 **9D 姿态表示** = pos(3) + rot6d(6)，支持 numpy 和 torch:

```python
def mat_to_pose9d(mat):        # 4×4→9D: 提取 pos + 旋转矩阵前两列
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = rotmat[...,:2,:].reshape(6)  # 6D 连续旋转表示 [Zhou et al. 2019]
    return [pos, d6]                  # (9,)

def pose9d_to_mat(d9):         # 9D→4×4: Gram-Schmidt 正交化
    pos, d6 = d9[...,:3], d9[...,3:]
    b1 = normalize(d6[...,:3])
    b2 = normalize(d6[...,3:] - proj(b1))
    b3 = cross(b1, b2)
    return [[b1,b2,b3,pos],[0,0,0,1]]

def mat_to_rot6d(mat):         # 3×3→6D: 取前两列
def rot6d_to_mat(d6):          # 6D→3×3: Gram-Schmidt
def pose7d_to_mat(pose_7d):    # 四元数→4×4
```

**与 DexMani 的对比**:
- DexMani 使用 `action_ee` 模式中的 `pos3+rot6d6`，与 UniDex 一致
- DexMani `action_key=action` 模式使用 **关节角**（非 6D 旋转），与 UniDex FAAS 的绝对关节角部分对应
- 关键差异: UniDex 的 action 是**相对位姿**（当前帧→目标帧的 Δ 变换），而 DexMani 的 action 是**绝对关节目标**

---

### 4.2 Inspire 手部联动机制 (MIMIC_RELATION)

**文件**: `src/utils/inspire_utils.py` (24 行)

Inspire 手存在**欠驱动联动**——远端关节的运动由近端关节驱动:

```python
MIMIC_RELATION = {
    "thumb_intermediate_joint":  ["thumb_proximal_pitch_joint", 1.334, 0],
    "thumb_distal_joint":        ["thumb_proximal_pitch_joint", 0.667, 0],
    "index_intermediate_joint":  ["index_proximal_joint",  1.064, -0.04545],
    "middle_intermediate_joint": ["middle_proximal_joint", 1.064, -0.04545],
    "ring_intermediate_joint":   ["ring_proximal_joint",   1.064, -0.04545],
    "pinky_intermediate_joint":  ["pinky_proximal_joint",  1.064, -0.04545],
}
# 格式: "从动关节": ["主动关节", scale, offset]
# q_distal = scale * q_proximal + offset
```

- 拇指联动比: 1.334 (intermediate), 0.667 (distal)
- 四指联动比: 1.064, offset=-0.04545
- 这意味着 Inspire 的 12 个关节中仅 **6 个独立 DoF**（1 thumb yaw + 5 proximal）
- HandAdapter IK 求解时使用 `mimic_iterations=50, mimic_step=5` 迭代施加联动约束

**对 DexMani 的意义**: DexMani 的 Inspire 手 12D action 空间实际只有 6 个主动 DoF + 6 个被动联动关节。这解释了为什么 12D 手部控制在实践中比理论上更稳定——真正需要学习的自由度减半。

---

## 5. FAAS 统一动作空间

### 5.1 设计原理

> **功能对齐**而非形态对齐。不同灵巧手的拇指屈曲关节可能名称不同、索引不同，但功能相同。

```
82D FAAS = [左腕9D | 右腕9D | 左手关节32D | 右手关节32D]

手腕 9D = pos(3) + rot6d(6)          # 连续6D旋转表示
手部 32D = 27 活动关节 + 5 预留槽    # 仅 27 维被实际使用 (JOINT_DIM_IN_USE)
```

### 5.2 各手部 DoF 与映射

| 手部 | 原生 DoF | → FAAS 32D | 关节命名示例 |
|------|---------|-----------|-------------|
| Inspire | 12 | 12/27 映射 | `thumb_proximal_pitch_joint→2`, `index_proximal_joint→7` |
| Leap | 16 | 16/27 映射 | `12→1, 13→0, 14→2, 15→3` (拇指) |
| Shadow | 22 | 22/27 映射 | `THJ5→0, THJ4→1, THJ3→26, FFJ4→6` |
| Allegro | 16 | 16/27 映射 | `joint_12.0→1, joint_13.0→0` (拇指) |
| Ability | 10 | 10/27 映射 | `thumb_q1→1, thumb_q2→2` |
| Oymotion | 11 | 11/27 映射 | `th_joint_1→1, th_joint_2→2` |
| Xhand | 12 | 12/27 映射 | `thumb_bend_joint→1, thumb_rota_joint1→2` |
| Wuji | 20 | 20/27 映射 | `F1J1→1, F1J2→2` (拇指) |

**映射规律**: 所有手的拇指关节→FAAS 索引 [0,4] 区域，食指→[5,9]，中指→[10,14]，以此类推。

### 5.3 核心常量

**文件**: `src/assets/utils/hand_utils.json`

```python
MAPPED_JOINT_DIM = 32   # 单手 FAAS 关节维度
JOINT_DIM_IN_USE  = 27  # 实际使用的关节数（部分手 DoF < 27 时补零）
# state_dim = 18 + 2 × MAPPED_JOINT_DIM = 82
# action_dim = 82
```

### 5.4 与 DexMani 动作空间对比

| 维度 | UniDex FAAS | DexMani joint | DexMani action_ee |
|------|------------|---------------|-------------------|
| 总维度 | 82 | 19 | 21 |
| 手臂 | 18D (9D/手 × 2, pos3+rot6d6) | 7D (臂关节角) | 9D (pos3+rot6d6) |
| 手部 | 64D (32D/手 × 2) | 12D (单手 Inspire) | 12D (单手) |
| 跨手支持 | ✅ 8种手 | ❌ 固定 Inspire | ❌ 固定 Inspire |
| 功能对齐 | ✅ FAAS 索引映射 | ❌ 无 | ❌ 无 |
| 双手支持 | ✅ 左右手同时 | ❌ 单手 | ❌ 单手 |

---

## 6. 点云编码器 Uni3D

### 6.1 架构

**文件**: `src/pointcloud_encoder/uni3d.py`

```
输入: pointcloud (B, N, 6) [xyz + rgb]
        │
        ▼
┌───────────────────────┐
│ Group (FPS + KNN)     │
│ - FPS 采样 G=512 中心  │
│ - KNN 取每组 M=64 近邻 │
│ - 归一化: neighborhood - center
│ - 拼接: xyz(3) + rgb(3) = 6D
│ → (B, G, M, 6)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Encoder (PointConv)   │
│ - Conv1d(6→128→256)   │
│ - MaxPool              │
│ - Conv1d(512→512→enc)  │
│ → (B, G, pc_enc_dim)   │  (默认 512)
└───────────┬───────────┘
            ▼
    Linear(enc_dim → pc_feat_dim)   (pc_feat_dim 默认 192)
            │
            ▼
    + CLS token + Position Embedding (MLP: 3→128→pc_feat_dim)
            │
            ▼
┌───────────────────────┐
│ PatchDropout          │  训练时随机丢弃 p% patch token
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ ViT (timm)            │
│ eva02_{ti/s/b/l/g}    │
│ + pos_drop + blocks   │
└───────────┬───────────┘
            ▼
输出: 
  - trans2embed=True:   (B, embed_dim)         ← CLS token → Linear→embed
  - trans2embed=False:  (B, num_group, feat_dim) ← 全部 patch tokens
```

### 6.2 模型变体

| 变体 | ViT backbone | pc_feat_dim | 配置文件 |
|------|-------------|-------------|---------|
| Uni3D-ti | eva02_tiny_patch14_224 | 192 | `uni3d_ti.yaml` |
| Uni3D-S | eva02_small_patch14_224 | 384 | `uni3d_s.yaml` |
| Uni3D-B | eva02_base_patch14_224 | 768 | `uni3d_b.yaml` |
| Uni3D-L | eva02_large_patch14_224 | 1024 | `uni3d_l.yaml` (默认) |
| Uni3D-G | eva02_giant_patch14_224 | 1408 | `uni3d_g.yaml` |

### 6.3 与 DexMani R3D 中 Uni3D 的对比

| 维度 | UniDex Uni3D | DexMani R3D Uni3D |
|------|-------------|-------------------|
| **用途** | 通用点云→token (作为 VLM 输入) | PC→特征 (作为 Diffusion 条件) |
| **Grouping** | ✅ FPS+KNN (6D 特征) | ❌ |
| **PointConv** | ✅ 2层 Conv1d + MaxPool | ❌ |
| **ViT backbone** | eva02 全系列 (ti→Giant) | ViT-tiny only |
| **PatchDropout** | ✅ 正则化 | ❌ |
| **输出** | CLS token 或全部 patch | 全部 patch tokens |
| **Position Encoding** | MLP(3→128→feat_dim) | RelativePositionalEncoding3D |
| **可训练** | 端到端参与训练 | 冻结/可选 LoRA |
| **参数规模** | ti≈6M→G≈1B | ti≈6M |

**关键差异**: UniDex 的 Uni3D 是一个完整的分层点云处理器（Grouping → PointConv → ViT），DexMani R3D 中更轻量（仅 ViT），因为它们将"分组+编码"留给了 `R3DObsEncoder` 的其他组件。

---

## 7. VLA 模型架构 (PointCloudUniDex)

### 7.1 完整前向流程

**文件**: `src/unidex/unidex.py` (947 行)

```
Batch:
  pointcloud: (B, 1, 1024, 6)     # 1帧观测
  state:      (B, 82)             # 当前关节状态
  action:     (B, 30, 82)         # 未来 30 步动作序列
  prompt:     List[str]           # 语言指令

Step 1: Tokenize
  VLAProcessor(prompt, pointcloud)
    → input_ids:      (B, 540)      # <image>×N + BOS + text + padding
    → attention_mask: (B, 540)

Step 2: Embed text + pointcloud
  text_emb = embed_tokens(input_ids)       # (B, 540, 2048)
  pcd_feat = Uni3D(pointcloud)             # (B, 256, 1024)  [L variant]
  pcd_feat = Projector(pcd_feat)           # (B, 256, 2048)
  → merge: pcd_feat 填入 <image> token 位置

Step 3: Encode proprio
  proprio_emb = Linear(82→1024)(state)     # (B, 1, 1024)

Step 4: Flow Matching
  t ~ U(0, 1)
  x0 ~ N(0, I)                            # (B, 30, 82)
  ψ_t = (1-(1-σ)t)*x0 + t*action           # noisy action

Step 5: Encode action
  action_emb = ActionEncoder(ψ_t, time_emb) # (B, 30, 1024)

Step 6: JointModel forward
  embeds   = {vlm: (B,540,2048), proprio: (B,1,1024), action: (B,30,1024)}
  → Block Attention → 18 层 Transformer
  → action_hidden: (B, 30, 1024)

Step 7: Decode action & loss
  v_pred = Linear(1024→82)(action_hidden)  # (B, 30, 82)
  target_vel = action - (1-σ)*x0
  loss = MSE(v_pred, target_vel)
```

### 7.2 关键配置参数

**文件**: `config/model/unidex.yaml`

| 参数 | 值 | 含义 |
|------|-----|------|
| `cond_steps` | 1 | 观测条件步数 |
| `horizon_steps` | 30 | 动作预测步数 |
| `action_dim` | 82 | FAAS 动作维度 |
| `proprio_dim` | 82 | 本体感知维度 |
| `max_seq_len` | 540 | 最大序列长度 (512 pcd + 28 text) |
| `num_inference_steps` | 10 | Flow Matching 推理步数 |
| `flow_sig_min` | 0.001 | 最小信号量 (≈直线路径) |
| `action_expert_adaptive_mode` | null (禁用) | adaLN/adaLN-Zero 时间条件注入模式 |
| `time_hidden_size` | 256 | 时间嵌入维度 |
| `pcd_token_index` | 257152 | `<image>` 的 token ID |
| `vocab_size` | 257216 | PaliGemma 词表大小 |
| `pad_token_id` | 0 | 填充 token ID |

### 7.3 Mixture 配置

| Mixture | hidden_size | intermediate_size | layers | KV Cache | 用途 |
|---------|------------|-------------------|--------|----------|------|
| vlm | 2048 | 16384 | 18 (共享) | ✅ | 点云+文本→跨模态理解 |
| proprio | 1024 | 4096 | 18 (共享) | ✅ | 关节状态→动作条件 |
| action | 1024 | 4096 | 18 (共享) | ❌ | 噪声动作→去噪 |

> **注意**: 三者在同一 18 层 Transformer 中通过 Block Attention 交互。参数是分离的（独立 Q/K/V/O/MLP 权重），但通过拼接 Q/K/V 实现跨 mixture 注意力。

### 7.4 冻结策略

**文件**: `src/unidex/unidex.py:227-232`

```python
def freeze_unused_weights(self):
    self.embed_tokens.weight.requires_grad = False   # 冻结词嵌入
    for name, param in self.joint_model.mixtures["vlm"].named_parameters():
        if self._check_gemma_unused_parameter_by_name(name):
            param.requires_grad = False  # 冻结 VLM 最后一层的 post_attn/mlp/o_proj/v_proj
```

冻结的 VLM 参数（最后一层 = layer 17）:
- `{17}.post_attention_layernorm` 后的所有参数
- `{17}.mlp` (gate/up/down)
- `{17}.self_attn.o_proj`
- `{17}.self_attn.v_proj`

> 注意：LoRA 参数即使在这些层中也不被冻结（`load_pretrained` 时显式保留 lora_ 前缀的参数）。

---

## 8. Flow Matching 动作解码

### 8.1 Conditional Flow 公式

```python
# unidex.py:747-755
def psi_t(x, x1, t, flow_sig_min=0.001):
    """Conditional Flow: 直线插值 + 微小偏移"""
    t = t[:, None, None]  # (B, 1, 1)
    return (1 - (1 - flow_sig_min) * t) * x + t * x1
    # σ=0.001 → 路径几乎是 x₀→x₁ 的直线

# 目标速度
d_psi = x1 - (1 - flow_sig_min) * x0  # 瞬时速度 / 直线路径的解析导数

# Loss
loss = torch.mean((v_pred - d_psi) ** 2)
```

**与 DexMani FlowMatch 的对比**:

| 维度 | UniDex | DexMani FlowMatch |
|------|--------|-------------------|
| 路径 | ψ_t = (1-(1-σ)t)·x₀ + t·x₁ | x_t = (1-t)·x₀ + t·x₁ |
| σ | 0.001 (≈直线) | target_t=0 (严格直线) |
| Loss | MSE(v_pred, x₁-(1-σ)x₀) | MSE(v_pred, x₁-x₀) + consistency |
| 推理 | Euler 前向: x += Δt·v | Euler ODE: x += Δt·v |
| 推理步数 | 10 (默认) | 10 (默认) |
| Backbone | Gemma Transformer (18层) | DiTX (cross-attn Transformer) |
| Consistency | ❌ | ✅ (可选, ManiFlow 使用) |

### 8.2 时间条件注入

UniDex 支持两种模式（由 `action_expert_adaptive_mode` 控制）:

**模式 1: pi0 风格 (action_expert_adaptive_mode=null)**
```python
# time_emb 拼接到 action 特征
action_emb = ActionEncoder(action, time_cond=time_emb)
# ActionEncoder: Linear(82→1024) + cat(time_emb) + Linear(2048→1024) + SiLU + Linear(1024→1024)
```

**模式 2: adaLN / adaLN-Zero**
```python
# time_emb 通过 AdaptiveRMSNorm 注入各层
action_emb = ActionEncoder(action)  # 不拼接 time
# 各层: AdaptiveRMSNorm(x, time_cond)
# adaLN-Zero 附加: AdaptiveLayerscale(x, time_cond) 控制残差缩放
```

### 8.3 Guided Inference (ReAct)

**文件**: `unidex.py:515-640`

实现了 [ReAct (Real-Time Execution of Action Chunking Flow Policies)](https://arxiv.org/pdf/2506.07339):

```
输入: previous_action (已执行的动作), delay, execution_horizon, beta (引导强度)

1. 将 previous_action 填充到 target_action 序列头部
2. 计算 inpaint_attention: 对过去帧赋权1.0，对当前执行窗口用指数衰减权重
3. 每步推理:
   v_pred = model(action)
   v_corrected = v_pred + gradient_guidance(action, target_action, inpaint_attention, beta)
   action += Δt * v_corrected
```

这使得在线部署时可以**结合已执行的动作进行条件引导**，减少 rollout 累积误差。

---

## 9. 多 Mixture 联合推理 (JointModel)

### 9.1 Block Attention 掩码

**文件**: `unidex.py:250-335` (build_causal_mask_and_position_ids)

```
                  pcd/text   proprio   action
pcd/text            ●          ✗         ✗
proprio             ●          ●         ✗
action              ●          ●         ●

● = 可以 attend (0), ✗ = 屏蔽 (-inf)
```

- pcd/text: **双向** self-attention (PaliGemma 特性)
- proprio: attend to pcd/text + self (causal)
- action: attend to pcd/text + proprio + causal self

### 9.2 前向传播

**文件**: `joint_model.py:308-384`

```python
def forward(embeds_all, attention_mask, position_ids_all, time_cond, ...):
    # 1. 输入归一化: embeds *= sqrt(hidden_size)
    for name in active_mixture_names:
        embeds_all[name] *= sqrt(hidden_size)

    # 2. 逐层处理
    for layer_idx in range(num_hidden_layers):  # 18层
        embeds_all = forward_mixture_layers(
            mixtures, attention_mask, position_ids_all,
            embeds_all, layer_idx, time_cond, ...
        )

    # 3. 最终 Norm (仅对 active mixtures)
    final_norm = mixture.norm(embeds)
```

### 9.3 单层前向 (forward_mixture_layers)

```
1. Input LayerNorm (可含 adaLN 时间条件)
2. MixtureAttention:
   - 各 mixture 独立计算 Q/K/V
   - 拼接所有 Q/K/V → 一次联合 scaled dot-product attention
   - 拆分结果 → 各 mixture 独立 O_proj
3. Post-Attention Adaptive Scale (adaLN-Zero 模式)
4. Residual Add
5. Post-Attention LayerNorm
6. GemmaMLP (gate/up/down, GeLU(tanh))
7. Final Adaptive Scale (adaLN-Zero)
8. Residual Add
```

### 9.4 Mixture Attention 关键实现

**文件**: `joint_model.py:130-304` (forward_mixture_attn)

```python
# 1. 独立计算 Q/K/V（各 mixture 不同 hidden_size + head 配置）
for name in active_mixture_names:
    Q[name] = mixture.attn(name).q_proj(x)
    # K/V: 从 KV Cache 取（推理）或新计算（训练）

# 2. GQA: K/V repeat 到 Q 的 head 数
for name in key_states_all:
    key, val = repeat_kv(key, val, num_key_value_groups)

# 3. 跨 mixture 拼接 → 联合 attention
Q = cat([Q[vlm], Q[proprio], Q[action]])   # 沿 seq 维拼接
K = cat([K[vlm], K[proprio], K[action]])
V = cat([V[vlm], V[proprio], V[action]])

attn = softmax(Q @ K^T / sqrt(head_dim) + attn_mask)
out = attn @ V

# 4. 拆分 + 独立 O_proj
out_vlm, out_proprio, out_action = split(out)
for name: out[name] = mixture.attn(name).o_proj(out[name])
```

### 9.5 KV Cache 策略

| 模式 | VLM KV | Proprio KV | Action KV | 使用场景 |
|------|--------|-----------|-----------|---------|
| 训练 | 不使用 | 不使用 | 不使用 | 一次 forward 过所有层 |
| 推理 Step 0 | 计算+缓存 | 计算+缓存 | 不使用 | 首次 pass |
| 推理 Step 1-9 | 读取缓存 | 读取缓存 | 不使用 | 后续 denoising 复用 |
| Naive 推理 | 不使用 | 不使用 | 不使用 | 每步重算全部（已废弃） |

> **这是 UniDex 推理加速的关键**: VLM+proprio 的 KV 仅在第一步计算并缓存，后续 9 步仅 forward action tokens，避免重复计算 3B 参数模型的 VLM 部分。

---

## 10. 训练管道

### 10.1 预训练

**文件**: `train.py` (212 行)

```yaml
# config/train.yaml 核心配置
train:
  load_checkpoint: null
  optimizer:
    _target_: torch.optim.AdamW
    lr: 1e-4
    betas: [0.9, 0.95]
    eps: 1e-8
    weight_decay: 1e-10
  scheduler:
    _target_: src.utils.schedulers.CosineDecaySchedule
    warmup_steps: 2000
    decay_steps: 200000
    decay_lr: 0.1       # = 0.1 × peak_lr = 1e-5 终值
    peak_lr: 1.0         # multiplier
  dataloader:
    batch_size: 4        # 每卡
    num_workers: 1
    val_ratio: 0.05
  trainer:
    max_epochs: 32
    devices: [0,1,2,3,4,5,6,7]  # 8 × H800
    strategy: ddp_find_unused_parameters_true
    precision: 32-true          # 无混合精度!
    accumulate_grad_batches: 4
    gradient_clip_val: 1.0
# 有效 batch size = 4 × 4(accum) × 8(GPU) = 128
```

训练流程:
1. `LightningTrainingWrapper.setup()` → `hydra.utils.instantiate(config.model)` → `PointCloudUniDexTrain`
2. 可选: `load_pretrained()` 加载 PaliGemma + Uni3D 预训练权重
3. `trainer.fit()` → PyTorch Lightning 管理 DDP/checkpoint/logging
4. 每个 training_step: `loss, _ = policy(batch)` → Flow Matching MSE
5. 每个 validation_step: `pred = policy.infer_action(batch)` → MSE(pred, target)

### 10.2 微调 (Real-World Post-Training)

**文件**: `finetune.py` (193 行)

与预训练几乎相同，配置差异:
```yaml
# config/finetune.yaml
train:
  load_checkpoint: <pretrained_ckpt_path>  # 必填
  dataloader:
    batch_size: 4
  trainer:
    max_epochs: 20        # 更少 epochs
    devices: [0, 1]       # 2 GPU
    strategy: ddp_find_unused_parameters_true
```

### 10.3 与 DexMani 训练对比

| 维度 | UniDex | DexMani_Policy |
|------|--------|---------------|
| 框架 | PyTorch Lightning + Hydra | 纯 PyTorch + Hydra |
| DDP | Lightning 内置 | mp.spawn 手动 |
| AMP | 32-true (无) | bfloat16 |
| EMA | ❌ | ✅ |
| Scheduler | CosineDecaySchedule (warmup+cosine) | OneCycleLR / Cosine |
| 梯度裁剪 | 1.0 | 无 |
| Checkpoint | Lightning ModelCheckpoint | 自实现 (epoch-step-score, topk=3, atomic) |
| NaN 诊断 | Lightning 默认 | 三层 NaN 检测 + debug checkpoint |
| LoRA | 4bit + Linear | Linear only |
| compile | ✅ 多子模块 | ✅ use_compile 开关 |

---

## 11. 推理管道与 KV Cache 优化

### 11.1 标准推理流程

**文件**: `unidex.py:431-505` (infer_action)

```
Step 0: 预处理
  - VLAProcessor: text + pcd → input_ids, attention_mask
  - build_causal_mask: 生成 Block Attention 掩码
  - split_mask: 拆分为 pcd_text_proprio_mask 和 action_mask

Step 1: VLM + Proprio forward (缓存 KV)
  inputs_embeds = embed_tokens(text) + projector(Uni3D(pcd))
  proprio_embeds = Linear(state)
  _, kv_caches = JointModel(
      mask=pcd_text_proprio_mask,
      embeds={vlm, proprio},
      return_caches=True
  )

Step 2: Action denoising (循环 num_inference_steps=10 次)
  t = 0
  action = randn(B, 30, 82)
  for step in range(10):
      time_emb = SinusoidalPosEmb(t)
      action_emb = ActionEncoder(action, time_emb)
      action_emb = JointModel(
          mask=action_mask,
          embeds={action},
          kv_caches=kv_caches,        # ← 复用 VLM+proprio 缓存!
          cache_mode="append_non_active"
      )
      v_pred = Linear(action_emb)
      action += (1.0 / 10) * v_pred   # Euler step
      t += 1.0 / 10
```

### 11.2 Naive 推理（已废弃但保留）

**文件**: `unidex.py:642-707` (infer_action_naive)

每步都重新前向所有 mixture（包括 VLM），比标准推理慢约 **3-5×**（因为 VLM 占模型 60%+ 参数）。保留此方法可能是为了对比测试。

### 11.3 推理子类封装

```python
class PointCloudUniDexInference(PointCloudUniDex):
    def process(self, pointcloud, state, prompt):
        # 预处理: tokenize + build masks
        return input_ids, pointcloud, masks, ...

    def forward(self, previous_action, delay, execution_horizon, beta, ...):
        # Guided inference (ReAct)
        return super().guided_inference(...)

    @torch.inference_mode()
    def infer_action(self, pointcloud, state, prompt):
        # 标准推理 (无引导)
        return super().infer_action(...)
```

---

## 12. HandAdapter 手部 Retargeting

**文件**: `HandAdapter/hand_processor.py`

### 12.1 核心思想

> **从个人类视频 → 机器人手关节角**，通过 PyBullet IK 求解对齐指尖轨迹，再用 Open3D 离屏渲染生成手部 RGB-D，与场景 RGB-D 融合。

```
原始 MANO 手部姿态 (21 关节, 三维)
  + 人类手部 RGB-D (含手部区域)
  + 场景 RGB-D (含物体+背景)
        │
        ▼
┌──────────────────────────┐
│ 指尖位置提取              │  MANO_TIP_INDEX_MAP (数据集特定)
│  H2o/HOI4D: [4,8,12,16,20]│
│  Hot3D: [16,17,18,19,20]  │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ PyBullet IK 求解          │
│ - max_iterations=1000     │
│ - residual_threshold=1e-3 │
│ - ik_damping=0.1          │
│ - mimic_iterations=50     │ ← 联动关节约束迭代
│ - mimic_step=5            │
└───────────┬──────────────┘
            ▼
    ┌───────────────────┐
    │ 联动关节 (Mimic)   │  ← 对 Inspire 等欠驱动手施加
    │ q_distal = scale   │    远端关节=联动比×近端关节+offset
    │   × q_prox + off   │
    └───────────────────┘
            │
            ▼
┌──────────────────────────┐
│ 手部 RGB-D 渲染           │  Open3D OffscreenRenderer
│ - URDF mesh → 三角面片   │  + 相机内参 (fx,fy,cx,cy) 对齐
│ - PyBullet 关节角 → 姿态  │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ 手+场景 RGB-D 融合        │  hand_depth < scene_depth →
│ + RGBD → 点云 (Open3D)   │  使用手部像素覆盖场景
│ + 工作空间裁剪 (PCD_MASK) │  深度截断 >1.15m
│ + CV→CAM 坐标变换         │
└───────────┬──────────────┘
            ▼
    HDF5 存储: frames/{rgb_images, depth_images, joint_values/left+right, poses/left+right}
              metadata/{dataset, hand_type, total_frames, camera_intrinsics, ...}
```

### 12.2 IK 求解器详解

**配置**:
```python
max_iterations = 1000      # 最大迭代次数
residual_threshold = 1e-3  # 残差收敛阈值
ik_damping = 0.1           # 阻尼系数（防奇异）
mimic_iterations = 50      # 联动关节迭代
mimic_step = 5             # 每步联动更新幅度
```

**联动关节约束 (Inspire 特有)**:
每次 IK 迭代后，强制执行联动约束:
```python
# 对每个 mimic pair:
distal_joint = scale * proximal_joint + offset
# 例: thumb_intermediate = 1.334 * thumb_proximal_pitch + 0
```

**URDF 加载** (`_load_hand`, 行 204-267):
1. 读取 `config.json` (tips/poses 关节名称列表)
2. `p.loadURDF(urdf_path, useFixedBase=True)` — 固定基座
3. 过滤可驱动关节 (`JOINT_REVOLUTE | JOINT_PRISMATIC`)
4. 构建 mesh_bank: 每个 link → (TriangleMesh, T_local)
5. 解析 visual offsets → 渲染用局部变换

### 12.3 使用方法

```bash
# 对 H2O 数据集生成 Inspire 手的 retargeted 数据
python HandAdapter/hand_processor.py \
    --hand_type Inspire \
    --dataset H2o \
    --cont

# --randperm 随机排列顺序 → 多进程并行处理
# 输出: data/{dataset}/retarget_RGBD/{seq_path}/{hand_type}.h5
```

### 12.4 新增手部

1. 放置 URDF: `HandAdapter/urdf/base/{hand_name}/{left,right}/main.urdf`
2. 配置: `HandAdapter/urdf/{hand_name}/config.json` (tips, poses 关节名)
3. 坐标约定: X 轴指向掌心, Z 轴沿手指方向
4. 注册到 `HandAdapter/visualizer.py` 的 `HAND_TYPES`
5. Web 界面调参 → 确认 retargeting 质量

### 12.5 对 DexMani 的适配潜力

**输出 → DexMani 输入转化路径**:
```
HandAdapter HDF5
  → load joint_values['right'] (N, 12) for Inspire
  → _reorder_joint_values() 重排为 FAAS 顺序
  → 去掉 MIMIC_RELATION 被动关节，仅保留 6 主动 DoF
  → 写入 DexMani Zarr: action (N, 19) [7 arm + 12 hand]
  → SequenceSampler 采样窗口
```

**主要挑战**:
- 坐标系统对齐: Bullet (Y-up)→Open3D (Z-up) 通过 `B2O` 矩阵处理
- 手部 DoF 差异: HandAdapter 输出完整 12 DoF，DexMani 需要区分主动/被动
- 场景点云缺失: HandAdapter 融合手+场景，DexMani 需要独立场景点云

---

## 13. 数据处理管线

### 13.1 缓存策略

```
第1次运行:
  _find_sequences()                → cache_dir/sequences.pkl      (序列元数据)
  _build_window(seq) × N           → cache_dir/windows.pkl         (所有训练窗口)
  filter by hand_type              → 内存过滤

第1次 __getitem__:
  _load_pointcloud_batch_and_cache() → cache_dir/pcd/{hash}.npy   (FPS 采样后的点云)
  _load_robot_data()                 → cache_dir/robot/{hash}.pkl (state + action + prompt)

后续运行:
  use_cached_metadata=True → 直接从 .pkl 加载
  pcd/robot 缓存命中 → 跳过重计算
```

### 13.2 BaseDataset 抽象

**文件**: `src/dataset/base.py` (646 行)

子类仅需实现 6 个抽象方法:
- `_find_sequences()` → 遍历数据集目录，发现所有序列
- `_build_window(seq)` → 从序列构建滑动窗口
- `_load_raw_pointcloud(pcd_paths)` → 加载原始点云
- `_load_state(window)` → 加载关节状态
- `_load_action_sequence(window)` → 加载动作序列
- `_load_prompt(window)` → 生成语言指令
- `_get_initial_action(window)` → 获取当前帧动作 (用于插值)

### 13.3 MixtureDataset

**文件**: `src/dataset/mixture.py` (43 行)

```python
class MixtureDataset(Dataset):
    def __init__(self, **kwargs):  # kwargs = {H2o: ds1, HOI4D: ds2, ...}
        self.datasets = list(kwargs.values())
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.lengths)

        # 自动合并各数据集的 Normalizer
        self.normalizer = Normalizer(normalizers=[ds.normalizer for ds in self.datasets])

        # 自动 padding: 取各数据集 shape 的最大值
        for ds in self.datasets:
            for key, value in ds.shape.items():
                self.shape[key] = max(self.shape.get(key, 0), value)

    def __getitem__(self, idx):
        data = self.datasets[i][idx]
        # 零填充到统一 shape
        for key in data:
            if data[key].shape != self.shape[key]:
                data[key] = np.pad(data[key], ..., constant_values=0)
        return data
```

### 13.4 与 DexMani 数据管线对比

| 维度 | UniDex | DexMani_Policy |
|------|--------|---------------|
| 存储格式 | HDF5 (retargeted) + pickle 缓存 | Zarr |
| 点云来源 | RGBD 投影 | 仿真传感器 |
| FPS 采样 | pytorch3d FPS (batch) | 无 (Zarr 预存) |
| 缓存策略 | 二级: 元数据 + 点云 + 机器人数据 | Zarr 直接读取 |
| 序列采样 | _build_window 子类实现 | SequenceSampler (numba) |
| 归一化 | meanstd/minmax/identity, 跨数据集合并 | limits→[-1,1], 全量拟合 |
| 数据增强 | ❌ 无 | ✅ PC/RGB/State 多类型 |
| 文本 | ✅ 语言指令 (每窗口) | ✅ 仅 MultiTaskAgent |
| 动作插值 | ✅ interpolation_factor | ❌ 无 |

---

## 14. 关键设计决策

### 14.1 为什么使用 PaliGemma 而不从头训练？

1. **跨模态对齐已内建**: PaliGemma 的 `<image>` token 嵌入空间天然支持视觉-文本对齐
2. **点云替换 <image>**: Uni3D 编码点云 → Projector → 填入 PaliGemma 预训练的 `<image>` 位置 → 无需重新训练视觉-语言对齐
3. **冻结策略**: 仅冻结 embed_tokens + VLM 最后一层 beyond attention，保留中间层可塑性

### 14.2 为什么 Flow Matching 而非 Diffusion？

1. **更少的推理步数**: 10 步 Euler vs 20-100 步 DDIM（同等质量）
2. **直线路径**: Conditional Flow 路径几乎是 x₀→x₁ 直线，速度场简单，容易学习
3. **与 π₀ 一致**: 遵循 π₀ 的设计范式

### 14.3 为什么不用 EMA？

1. Flow Matching 在 Transformer backbone 上不需要 EMA（与 Diffusion UNet 不同）
2. Lightning 的 checkpoint 机制简单：save_last + save_top_k (monitor val_loss)
3. 预训练→微调范式下 EMA 收益有限

### 14.4 为什么 32-true 精度而非 bfloat16？

可能的理由:
1. PaliGemma 预训练权重为 float32，直接用 bfloat16 会导致数值不稳定
2. 8×H800 (80GB) 充足显存使得混合精度非必需
3. 3B 参数 + batch=128 (有效) 在 float32 下仍可行

### 14.5 为什么 action_dim=82 而非动态维度？

1. **固定维度 = 固定架构 = 简化工程**: 不需要运行时改变 Linear 层
2. **32D 预留**: 目前仅 27D 被使用，预留 5D 用于未来扩展
3. **零填充**: 对于 DoF 不足 27 的手，未使用维度填零 → 注意力自然忽略

### 14.6 为什么 Inspire 手 12 DoF 实际只有 6 个独立自由度？

**MIMIC_RELATION** (`src/utils/inspire_utils.py`) 定义了 Inspire 手的欠驱动联动:

```
远端关节 = scale × 近端关节 + offset

thumb_intermediate = 1.334 × thumb_proximal_pitch + 0
thumb_distal       = 0.667 × thumb_proximal_pitch + 0
{index,middle,ring,pinky}_intermediate = 1.064 × {respective}_proximal - 0.04545
```

意味着:
- 拇指: 2 主动 DoF (yaw + proximal pitch) → 3 被动联动
- 四指: 4 主动 DoF (proximal) → 4 被动联动
- **实际 6 个独立 DoF** 控制 12 个关节

这对 DexMani 的意义重大: 如果显式建模联动关系，可以将 12D 手部动作空间降为 6D，大幅降低扩散模型的学习难度。

---

## 15. 与 DexMani_Policy 的完整对比

### 15.1 架构层面

| 维度 | UniDex | DexMani_Policy |
|------|--------|---------------|
| 模型范式 | VLA (Vision-Language-Action) | Imitation Learning (vision→action) |
| 骨干网络 | PaliGemma 3B + Uni3D | UNet1D / DiTX (轻量) |
| 参数量 | ~3B | ~10M-100M |
| 观测模态 | 点云+文本+proprio | PC/RGB+state (+text: MultiTask) |
| 点云编码器 | Uni3D (Group+Conv+ViT) | iDP3/PointNeXT/Uni3D (轻量) |
| 动作解码器 | Flow Matching + Gemma Transformer | Diffusion (DDPM/DDIM) / FlowMatch |
| 动作空间 | 82D FAAS (8手统一) | 19D joint / 21D action_ee (1手) |
| 条件注入 | 多 Mixture Block Attention + adaLN | FiLM (UNet) / Cross-attention (DiTX) |
| 推理优化 | KV Cache 复用 | DDIM 采样步数减少 |

### 15.2 数据处理层面

| 维度 | UniDex | DexMani_Policy |
|------|--------|---------------|
| 数据来源 | 人类视频 retargeting | DexMani_Sim 仿真 |
| 数据规模 | 50K+ 轨迹, 80TB+ | 取决于任务 |
| 缓存方式 | pickle + numpy (.npy) | Zarr (直接读取) |
| 归一化方法 | meanstd / minmax / identity | limits→[-1,1] |
| 数据增强 | 无 | PC/RGB/State 多类型增强 |
| 多数据集 | MixtureDataset (自适应 padding) | 单数据集 |
| 文本监督 | 每窗口语言指令 | 仅 MultiTaskAgent |

### 15.3 训练策略层面

| 维度 | UniDex | DexMani_Policy |
|------|--------|---------------|
| 训练框架 | PyTorch Lightning | 纯 PyTorch (手动训练循环) |
| DDP 实现 | Lightning 内置 | mp.spawn + dist |
| 精度 | 32-true (float32) | bfloat16 AMP |
| EMA | 无 | 有 (训练+推理) |
| 优化器 | AdamW (lr=1e-4, β=0.9/0.95) | AdamW (配置可调) |
| 学习率调度 | CosineDecaySchedule | OneCycleLR / Cosine |
| 梯度累积 | 4 steps | config 可配 |
| 梯度裁剪 | 1.0 | 无 |
| 冻结策略 | embed_tokens + 末层VLM | obs_lr=0 |
| LoRA | ✅ Linear + 4bit | ✅ Linear only |
| torch.compile | ✅ | ✅ |

### 15.4 评测方式

| 维度 | UniDex | DexMani_Policy |
|------|--------|---------------|
| 评测环境 | 真实机器人 (Inspire, Wuji, Oymotion) | DexMani_Sim 仿真 |
| 任务类型 | 5 工具操作 (咖啡/清扫/浇花/切袋/鼠标) | 多种抓取+操作 |
| 主要指标 | Task Progress (0-100%) | Success Rate (0-1) |
| 跨手泛化 | ✅ 零样本 | ❌ 未支持 |
| 人类 demo | ✅ 50 机器人 demo + 可选人类 demo | ❌ |

---

## 16. 已知问题与代码审查发现

### 16.1 Config `_target_` 路径错误 (高优先级 Bug)

**文件**: `config/model/unidex.yaml:64,72`, `config/model/unidex_inference.yaml:64,72`

```yaml
# 错误引用
projector:
  _target_: src.openmodel.modules.PaliGemmaMultiModalProjector  # src/openmodel/ 不存在!
joint:
  _target_: src.openmodel.joint_model.JointModel                # 同上
```

**实际类位置**:
- `PaliGemmaMultiModalProjector` → `src/unidex/modules.py:122`
- `JointModel` → `src/unidex/joint_model.py:308`

修复: 将 `src.openmodel` 替换为 `src.unidex`

### 16.2 遗留测试代码

**文件**: `joint_model.py:386-472` (`__main__` block)

```python
cfg = OmegaConf.load("config/train/bridge.yaml")  # bridge.yaml 不存在!
```

这是开发阶段的测试代码，引用了不存在的配置文件。应删除或更新为有效配置。

### 16.3 Config 中注释与实际值不一致

**文件**: `unidex.yaml:27`
```yaml
# ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224
```
注释中的路径格式是旧的 HuggingFace cache 结构，新版使用 `models--google--paligemma-3b-pt-224` 格式。

### 16.4 无数据增强

UniDex 完全依赖数据集多样性而**无任何数据增强**（无点云抖动、颜色扰动、dropout 等）。这意味着:
- 训练集必须覆盖足够多的场景变化
- 对采集条件变化（光照、相机位置）的鲁棒性依赖预训练规模
- DexMani 的增强管线可互补

### 16.5 `MixtureDataset` 零填充可能引入注意力偏差

小数据集（DoF 较小的手）的零填充维度在 Block Attention 中仍然可被 attend，理论上可能引入噪声。实践中归一化后的零值 token 在 softmax 中权重极小。

---

## 17. 可借鉴设计建议

### 17.1 P0 — 可直接迁移

**PatchDropout 正则化** (`uni3d.py:58-96`):
40 行纯 PyTorch 模块，无外部依赖，可直接复制到 DexMani PC encoder 增强管线。

**FAAS 关节映射表** (`hand_utils.json`):
8 种手的完整关节→FAAS 映射（scale/offset/map），可为 DexMani 未来多手支持提供参考。

**MixtureDataset 归一化合并** (`normalizers.py:37-78`):
`merge_normalizers()` 的跨数据集 minmax 合并逻辑可直接采用。

**Inspire MIMIC_RELATION** (`inspire_utils.py`):
联动关节约束表可直接用于 DexMani：
- 训练时: 仅预测 6 个主动 DoF，通过联动公式推导被动关节
- 推理时: Action Decoder 输出 6D → MIMIC_RELATION 展开为 12D
- 预期收益: 将 12D 手部控制降为 6D，降低学习难度

**HDF5→Zarr 转换工具**:
基于 `base_retarget.py` 的 `_load_state`/`_load_action_sequence` 逻辑，可构建将 HandAdapter 输出转为 DexMani Zarr 格式的转换脚本。

### 17.2 P1 — 需要适配

**HandAdapter → DexMani_Sim 对接**:
- 输出: HDF5 frames → 读取关节值 → 转为 Zarr
- 挑战: 坐标系对齐 (Bullet Y-up→Open3D Z-up 的 `B2O` 矩阵: `[[1,0,0],[0,0,1],[0,-1,0]]`)、关节命名映射

**KV Cache 推理加速** (DiTX/ManiFlow):
- UniDex: VLM+proprio KV 缓存, action 使用 `append_non_active` 模式
- DiTX: Cross-attention 条件天然可缓存 → 类似策略可减少重复编码

**多数据集 MixtureDataset**:
- UniDex 的 `MixtureDataset` (~43 行) 提供自动维度 padding + Normalizer 合并
- DexMani 的多任务训练可借鉴此模式

**RealDataset 双格式支持模式**:
- V1 (遥操作) + V2 (retarget) 双 Zarr 格式，通过 `_validate_zarr_format` 自动检测
- 可借鉴为 DexMani 的仿真/真机数据统一接口

### 17.3 P2 — 长期方向

**多手联合训练**:
- 借助 FAAS 映射，在 action_ee 模式下扩展为多手统一空间
- 需要: 手部 URDF 模型 + IK retargeting pipeline + 多手数据采集

**人类 Demo 协同训练**:
- 基于论文发现: ~2 人类 demo ≈ 1 机器人 demo, 采集快 5.2×
- DexMani 路径: 通过 DexMani_Sim + HandAdapter → 生成人类 retargeted 数据 → 混合训练
- 前提: 需先验证 retargeting 精度

**语言条件控制**:
- UniDex 使用 PaliGemma tokenizer + Gemma backbone 的自然语言理解
- DexMani MultiTaskAgent 已支持 CLIP text encoder → 可扩展到指令跟踪

**Guided Inference (ReAct)**:
- 在线执行时结合已执行动作进行条件引导
- 对 DexMani 的 rollout 稳定性可能有帮助

**生成增强数据 (DemoGen 风格)**:
- UniDex RealDataset 支持 `use_generated_data` + `camera_angle_peturb`/`camera_pos_peturb`
- DexMani 可通过类似方式生成空间泛化数据

### 17.4 优先路线图

```
短期 (1-2周): 集成 PatchDropout + 参考 FAAS 映射表
中期 (2-4周): MixtureDataset 多任务支持 + DiTX KV-cache 推理加速
长期 (1-2月): HandAdapter 对接 → 多手联合训练 → 语言条件扩展
```

---

## 参考文献

- **UniDex 论文**: Zhang et al., *UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos*, CVPR 2026. arXiv:2603.22264
- **PaliGemma**: Beyer et al., *PaliGemma: A versatile 3B VLM for transfer*, arXiv:2407.07726
- **π₀**: Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164
- **Uni3D**: Zhou et al., *Uni3D: Exploring Unified 3D Representation at Scale*, ICLR 2024
- **ReAct**: *Real-Time Execution of Action Chunking Flow Policies*, arXiv:2506.07339
- **LoRA**: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- **Flow Matching**: Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023
