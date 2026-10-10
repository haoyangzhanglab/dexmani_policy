# robot_data 数据格式与模态说明

`robot_data/` 当前包含两类数据：**真机数据**与 **sim 仿真数据**。两者都使用 **Zarr v2 目录存储**，共享 `data/` 和 `meta/` 的组织方式，但模态字段、部分形状与物理含义不同。

本文以真机任务 `pick_place_toy` 和 sim 任务 `pick_apple_messy` 为例。字段、shape、dtype 与数量依据 **2026-10-09 本地实际数据**核对；数据重新导出后，帧数和 episode 数应以新的 Zarr 为准。

## 1. 目录与命名

```text
robot_data/
├── pick_place_toy.zarr/       # Real：真机数据
├── pick_apple_messy.zarr/     # Sim：本文的仿真示例
├── multi_grasp.zarr/         # Sim
├── place_milk_box.zarr/      # Sim
└── pour.zarr/                # Sim
```

数据集目录命名为 `<task_name>.zarr`，例如 `pick_place_toy.zarr` 对应 `task_name: pick_place_toy`。`.zarr` 是包含元数据和数组分块的目录，不是单个文件。当前配置通常通过 `robot_data/${task_name}.zarr` 定位数据集。

**当前这些任务中，只有 `pick_place_toy` 是真机，其余均为 sim。** 下文只展开一个 sim 任务；其他 sim 任务的帧数、episode 数仍需读取各自的数据，不能套用示例数值。

| 项目 | Real：`pick_place_toy` | Sim：`pick_apple_messy` |
| --- | --- | --- |
| 总帧数 `T` | 13,991 | 21,293 |
| episode 数 `E` | 60 | 125 |
| 单个 episode 长度 | 165–331 帧 | 151–205 帧 |
| 记录步长 `dt` | 0.0625 s，即 16 Hz | 约 0.0625 s，即 16 Hz |
| RGB / depth 分辨率 | 高 480 × 宽 640 | 高 240 × 宽 320 |
| 每帧场景点云 | 1,024 点，每点 6 维 | 1,024 点，每点 6 维 |
| 关节状态 / 关节动作 | 19 维 | 19 维 |
| 末端动作 | 21 维 | 21 维 |

## 2. 共同存储结构：帧数组与 episode 边界

```text
<task_name>.zarr/
├── .zgroup                   # Zarr group 信息，zarr_format = 2
├── .zattrs                   # 数据集级属性，如 task_name、dt
├── data/
│   ├── joint_state/          # 各 episode 按时间顺序拼接后的数组
│   ├── action/
│   ├── action_ee/
│   ├── rgb/
│   ├── depth/
│   ├── point_cloud/
│   └── ...                   # 两类数据各自的其他模态
└── meta/
    ├── episode_ends/         # 每个 episode 的累计结束位置
    └── ...                   # sim 还有 seed、成功帧等信息
```

每个数组目录中的 `.zarray` 记录 `shape`、`dtype`、`chunks` 和压缩方式。`shape` 是完整逻辑形状；`chunks` 是磁盘分块形状，不能当作 episode 长度或训练窗口长度。例如，真机 `action` 的 shape 为 `(13991, 19)`，chunks 为 `(42, 19)`。

本文使用以下符号：

| 符号 | 含义 |
| --- | --- |
| `T` | 一个数据集中所有 episode 的总帧数；`data/*` 的首维 |
| `E` | episode 数量；`meta/episode_ends` 的长度 |
| `H, W` | 图像的高、宽 |
| `P` | 点云点数 |

数组不额外保留 episode 轴：`joint_state` 是 `(T, 19)`，而非 `(E, T, 19)`。episode 的长度可以不同，通过 `meta/episode_ends` 切分。

`episode_ends[i]` 是第 `i` 个 episode 的**右开结束位置**，不是该 episode 的最后一帧索引：

```python
start = 0 if i == 0 else episode_ends[i - 1]
end = episode_ends[i]
episode = array[start:end]
```

真机数据前几个边界为 `[272, 485, 730, ...]`，所以第 0 条轨迹是 `[0:272]`，第 1 条是 `[272:485]`。所有 `data/*` 的首维都应等于 `episode_ends[-1]`；当前两个示例均已核实满足这一条件。

## 3. Real：`pick_place_toy.zarr`

该数据集是由真机 Raw 记录导出的 canonical 多模态数据，根属性 `format` 为 `dexmani.real.canonical`。下表中的 `T = 13991`，所有 shape 均为**磁盘存储形状**。

### 3.1 帧级数据 `data/`

| 字段 | Shape | Dtype | 内容与维度解释 |
| --- | --- | --- | --- |
| `joint_state` | `(T, 19)` | `float32` | 实测关节位置：机械臂 7 维 + 灵巧手 12 维 |
| `action` | `(T, 19)` | `float32` | 关节目标：机械臂 7 维 + 灵巧手 12 维 |
| `action_ee` | `(T, 21)` | `float32` | 目标末端位置 3 维 + `rot6d` 6 维 + 手关节目标 12 维 |
| `eef_pose` | `(T, 9)` | `float32` | 由实测机械臂关节经 FK 得到的末端位置 3 维 + `rot6d` 6 维 |
| `arm_qvel` | `(T, 7)` | `float32` | 机械臂 7 个关节的速度反馈 |
| `arm_effort` | `(T, 7)` | `float32` | 机械臂 7 个关节的 effort 反馈，保留 Raw 字段数值 |
| `hand_current` | `(T, 12)` | `float32` | 灵巧手 12 个关节的电流反馈，保留 Raw 字段数值 |
| `rgb` | `(T, 480, 640, 3)` | `uint8` | RGB 图像，通道在最后，像素值为 0–255 |
| `depth` | `(T, 480, 640)` | `uint16` | 深度原始整数值；转为米需要乘根属性中的深度尺度 |
| `point_cloud` | `(T, 1024, 6)` | `float32` | 场景点云，每点为 `[x, y, z, r, g, b]` |
| `fingertip_points` | `(T, 5, 3)` | `float32` | 5 个指尖的位置，每个位置 3 维 |
| `contact_force` | `(T, 5, 3)` | `float32` | 每根手指的三轴聚合触觉力读数 |
| `tactile_force` | `(T, 5, 120, 3)` | `float32` | 5 根手指，每根 120 个触觉点，每点 3 维 |

`joint_state` 是观测到的状态，`action` 是控制目标，二者即使形状相同也不能互换。真机 `action_ee` 的前 9 维由机械臂**目标关节**经过 FK 计算；`eef_pose` 则由**实测关节**计算。

指尖位置的 5 指顺序由 `fingertip_link_names` 明确保存：

| 索引 | 手指 | Link 名称 |
| --- | --- | --- |
| 0 | 拇指 | `right_hand_thumb_rota_tip` |
| 1 | 食指 | `right_hand_index_rota_tip` |
| 2 | 中指 | `right_hand_mid_tip` |
| 3 | 无名指 | `right_hand_ring_tip` |
| 4 | 小指 | `right_hand_pinky_tip` |

### 3.2 Episode 信息与根属性

`meta/` 中只有一个数组：

| 路径 | Shape | Dtype | 含义 |
| --- | --- | --- | --- |
| `meta/episode_ends` | `(60,)` | `int64` | 60 条 episode 的累计结束位置，最后一个值为 13,991 |

episode 的文本标识和处理配置保存在根属性 `.zattrs`，不在 `data/` 中：

| 属性 | 当前值或结构 | 用途 |
| --- | --- | --- |
| `format` | `dexmani.real.canonical` | 标识真机 canonical 格式 |
| `task_name` | `pick_place_toy` | 任务名 |
| `dt` | `0.0625` | 记录步长，单位 s |
| `episode_ids` | 长度为 60 的字符串列表 | 与 episode 顺序对应，如 `episode_20260827_165510` |
| `data_revision` | 非空字符串 | 标识当前导出数据版本 |
| `depth_scale_m_per_unit` | `0.0002500000118743628` | 每个深度整数单位对应的米数 |
| `fingertip_link_names` | 长度为 5 的字符串列表 | 指尖几何的 link 选择和顺序 |
| `pointcloud_config` | 配置字典 | 点云生成参数，包括 `num_points: 1024`、深度范围、workspace、去桌面与采样参数 |

此 canonical Zarr 没有保存 sim 的 `done`、`segmentation`、相机内外参数组或 `episode_seeds`；也没有单独的 `eef_pos` 和 `eef_rot6d` 数组。

## 4. Sim：`pick_apple_messy.zarr`

该数据集由仿真 episode 转换并拼接而成。下表中的 `T = 21293`。

### 4.1 帧级数据 `data/`

| 字段 | Shape | Dtype | 内容与维度解释 |
| --- | --- | --- | --- |
| `joint_state` | `(T, 19)` | `float32` | 当前关节位置：机械臂 7 维 + 灵巧手 12 维 |
| `action` | `(T, 19)` | `float32` | 关节目标：机械臂 7 维 + 灵巧手 12 维 |
| `action_ee` | `(T, 21)` | `float32` | 目标末端位置 3 维 + `rot6d` 6 维 + 手关节目标 12 维 |
| `eef_pos` | `(T, 3)` | `float32` | 当前末端位置 |
| `eef_rot6d` | `(T, 6)` | `float32` | 当前末端姿态的 6D 旋转表示 |
| `rgb` | `(T, 240, 320, 3)` | `uint8` | RGB 图像，通道在最后，像素值为 0–255 |
| `depth` | `(T, 240, 320)` | `uint16` | 深度，单位为 mm；转换为米时除以 1000 |
| `segmentation` | `(T, 240, 320)` | `uint8` | 每个像素的分割标签，不是 RGB 图像 |
| `point_cloud` | `(T, 1024, 6)` | `float32` | 场景点云，每点为 `[x, y, z, r, g, b]` |
| `imagine_point_cloud` | `(T, 512, 6)` | `float32` | 根据手部模型表面采样并随 link 位姿变换的几何点云，每点为 XYZ + RGB |
| `fingertip_points` | `(T, 15)` | `float32` | 5 个指尖的 XYZ 位置，展平为 `5 × 3 = 15` 维 |
| `contact_force` | `(T, 15)` | `float32` | 5 个手指接触 link 的三轴净接触力，展平为 15 维 |
| `camera_intrinsic` | `(T, 9)` | `float32` | 相机 `3 × 3` 内参矩阵展平后的 9 维 |
| `camera_extrinsic` | `(T, 12)` | `float32` | 相机 `3 × 4` 外参矩阵展平后的 12 维，采用仿真导出的 `extrinsic_cv` 约定 |
| `done` | `(T,)` | `bool` | 每一步记录的环境完成标记；episode 边界仍由 `episode_ends` 定义 |

`fingertip_points` 与 `contact_force` 都按拇指、食指、中指、无名指、小指排列，每根手指占连续 3 维。需要显式的手指轴时，可将 `(T, 15)` reshape 为 `(T, 5, 3)`。`imagine_point_cloud` 来自手部模型几何，不能当作相机采到的场景点云。

### 4.2 Episode 信息与根属性

| 路径 | Shape | Dtype | 含义 |
| --- | --- | --- | --- |
| `meta/episode_ends` | `(125,)` | `int64` | 累计结束位置，最后一个值为 21,293 |
| `meta/episode_seeds` | `(125,)` | `int64` | 每条 episode 对应的仿真随机种子 |
| `meta/success_frame_idx` | `(125,)` | `int64` | episode 内首次成功的帧索引；导出逻辑以 `-1` 表示未记录成功帧 |
| `meta/physics_static_friction` | `(125,)` | `float64` | 静摩擦参数；当前示例全部为 `NaN` |
| `meta/physics_dynamic_friction` | `(125,)` | `float64` | 动摩擦参数；当前示例全部为 `NaN` |
| `meta/physics_mass_scale` | `(125,)` | `float64` | 质量缩放参数；当前示例全部为 `NaN` |

`success_frame_idx` 是 **episode 内的局部索引**，不能直接作为拼接数组的全局索引。物理参数中的 `NaN` 表示缺失值，不能解释为零或某个默认物理参数。

| 根属性 | 当前值 | 含义 |
| --- | --- | --- |
| `task_name` | `pick_apple_messy` | 任务名 |
| `action_space` | `joint` | 数据采集使用关节动作；Zarr 同时保存 `action_ee` |
| `dt` | 约 `0.0625` | 控制 / 记录步长，单位 s |
| `physics_dt` | 约 `0.00416667` | 仿真物理步长，约 1/240 s |
| `frame_skip` | `15` | 一个控制步对应 15 个物理步，`dt ≈ physics_dt × frame_skip` |

该 sim 示例没有真机的 `format`、`data_revision`、`episode_ids` 或 `depth_scale_m_per_unit` 属性，也没有 `arm_qvel`、`arm_effort`、`hand_current`、`tactile_force` 和 `eef_pose` 数组。

## 5. 两种格式之间最容易混淆的地方

### 5.1 状态、动作与姿态表示

两类数据的动作维度分解相同，以下切片均采用 Python 的右开区间：

```text
joint_state / action  (19) = arm[0:7] + hand[7:19]
action_ee             (21) = position[0:3] + rot6d[3:9] + hand[9:21]
Real eef_pose          (9) = position[0:3] + rot6d[3:9]
Sim 末端观测          (9) = concatenate(eef_pos, eef_rot6d)
```

关节位置 / 目标使用弧度，位置使用米。`rot6d` 是旋转矩阵前两列依次拼接的 6D 表示，不是欧拉角、四元数，也不是“位置 + 旋转”共 6 维。

几何量还需区分坐标系：真机导出的末端位姿、指尖位置与点云 XYZ 使用 `xarm_base` 坐标系；sim 使用仿真世界坐标系。**形状一致不代表坐标系已经对齐。**

### 5.2 图像、深度与点云

| 项目 | Real | Sim |
| --- | --- | --- |
| RGB 存储布局 | `(T, H, W, 3)`，`uint8` | 同左，但 `H, W` 不同 |
| 深度转米 | `depth.astype(float32) × depth_scale_m_per_unit` | `depth.astype(float32) / 1000` |
| 当前深度尺度 | 约 0.25 mm / unit | 1 mm / unit |
| 点云 6 通道 | XYZ（米）+ RGB（浮点 0–1） | 同左 |
| 当前点云形状 | `(T, 1024, 6)` | `(T, 1024, 6)` |

深度图虽然同为 `uint16`，不能共用未经确认的换算系数。点云中的 RGB 已为 0–1；原始 `rgb` 图像为 0–255，两者不要重复或遗漏归一化。

### 5.3 指尖与力传感

| 项目 | Real | Sim |
| --- | --- | --- |
| `fingertip_points` | `(T, 5, 3)` | `(T, 15)` |
| `contact_force` | `(T, 5, 3)`，来自硬件聚合触觉通道 | `(T, 15)`，来自仿真接触冲量计算并在物理子步间取平均的净接触力 |
| `tactile_force` | `(T, 5, 120, 3)`，硬件稠密触觉通道 | 无此字段 |

reshape 只能统一布局，不能统一测量含义。真机触觉值保留硬件通道的数据约定；当前 Zarr 属性没有给出其力单位与轴向标定，不能仅凭 `contact_force` 这个同名字段就将其当作 sim 世界坐标系下的力直接比较。

### 5.4 触觉有效性与归一化

有效零接触读数与传感器缺测不同，不能根据力值是否为零推断有效位。本文两个 Zarr 示例均未提供 `tactile_valid`；启用 interaction 的 `agent.use_tactile_valid=true` 时，数据与推理调用方必须提供 bool `[T,5]` 有效位，并在 `dataset.sensor_modalities` 和 `normalization` 中分别声明该字段和 `identity`。

有效位进入 encoder 后可以屏蔽无效触觉，但当前 dataset 仍会拒绝含 NaN 的观测窗口，normalizer 也不会排除有限占位值。训练集有缺测污染时，需要处理数据过滤和统计拟合，不能只依靠 encoder mask。`tactile_dropout_prob` 在规范化后的编码路径模拟缺测，不会修复原始数据中的污染。

当前 [normalizer 构建](../dexmani_policy/training/build_utils.py)使用默认 `last_n_dims=1`，因此触觉存储布局还决定统计粒度：

| 原始布局 | Gaussian 统计通道 | 汇总范围 |
| --- | --- | --- |
| `(T,15)` | 15 个手指 × 轴通道 | 时间 / 训练源行 |
| `(T,5,3)` | 3 个轴通道 | 训练源行和手指 |
| `(T,5,120,3)` | 3 个轴通道 | 训练源行、手指和 taxel |

因此，拟合统计前后的 reshape 不可视为等价操作；不同布局的统计不能直接替换。训练只使用有效训练窗口的唯一源行拟合，验证、推理和恢复复用该份统计。

interaction 要求点云、腕部和指尖处于同一米制坐标系，几何字段使用 `identity`。Real/SIM 的字段选择、完整 normalization 替换方式及开关见 [interaction 数据与配置](./interaction_representation.md#6-数据与配置)。

## 6. 从 Zarr 到训练样本

上面的表格描述的是磁盘数据。训练时，[BaseDataset](../dexmani_policy/datasets/base_dataset.py) 依据 `sensor_modalities` 选择观测字段，依据 `action_key` 选择监督目标，并在 episode 内采样时间窗口。

```text
Zarr data/<观测字段>  → sample["obs"][<观测字段>]
Zarr data/<action_key> → sample["action"]
```

这里的 `sample["action"]` 是统一的输出键：当 `action_key: action_ee` 时，其内容来自 Zarr 的 `data/action_ee`。

设观察长度为 `N = obs_horizon`，监督长度为 `L = horizon`，DataLoader batch size 为 `B`：

| 字段 | Dataset 单个样本 | DataLoader batch |
| --- | --- | --- |
| `obs.joint_state` | `(N, 19)` | `(B, N, 19)` |
| `obs.point_cloud` | `(N, 1024, 6)` | `(B, N, 1024, 6)` |
| `obs.rgb`，启用 RGB 预处理后 | `(N, 3, H_out, W_out)` | `(B, N, 3, H_out, W_out)` |
| `action`，选择关节动作 | `(L, 19)` | `(B, L, 19)` |
| `action`，选择末端动作 | `(L, 21)` | `(B, L, 21)` |

例如，当前 [dp.yaml](../dexmani_policy/configs/dp.yaml) 使用 `N = 2`、`L = 16`，RGB 先 resize 到 `240 × 240` 再 crop 到 `224 × 224`，单个样本中的 RGB 为 `(2, 3, 224, 224)`。当前该配置启用 `rgb_keep_uint8: true`，所以 Dataset 输出图像仍是 `uint8`；后续模型视觉路径再处理浮点转换与归一化。未启用 RGB 预处理时，Dataset 不会自动把原始 HWC 改为 CHW。

上表的动作形状按未启用辅助末端监督计算。若设置 `use_aux_ee=True`，Dataset 会额外拼接 `action_ee[..., :9]`，监督张量最后一维增加 9；Zarr 中原始数组的形状不变。

`RGBDataset` 和 `PCDataset` 分别默认使用 `joint_state + rgb`、`joint_state + point_cloud`，它们是**观测选择方式**，不是 Real / Sim 两种数据格式。数据中存在的模态也不等于当前模型一定消费该模态；例如 `depth` 和 `tactile_force` 不会因存在于 Zarr 就自动进入模型。

## 7. 只读检查示例

在项目 Python 环境中，从仓库根目录运行以下代码，可查看当前属性、字段、形状与第一条 episode 的范围。它只读取元数据与 episode 边界，不加载完整图像或点云数组。

```python
import zarr

root = zarr.open_group("robot_data/pick_place_toy.zarr", mode="r")
print("attrs:", dict(root.attrs))

for group_name in ("data", "meta"):
    for name in sorted(root[group_name].array_keys()):
        array = root[group_name][name]
        print(f"{group_name}/{name}: shape={array.shape}, dtype={array.dtype}")

ends = root["meta/episode_ends"][:]
print(f"episodes={len(ends)}, total_frames={int(ends[-1])}")
print(f"first_episode=[0:{int(ends[0])}]")
```

将路径替换为 `robot_data/pick_apple_messy.zarr` 即可检查 sim 示例。本文的 shape、dtype、属性和 episode 边界以本地 Zarr 为依据；字段生成含义同时对照同级 `dexmani_real` 的 `dataset/processing.py`、`dataset/contracts.py`，以及 `dexmani_sim` 的 `envs/base_env.py`、`sensors/`、`mimic_gen/utils/env_recorder.py` 和 `data_process/hdf5_to_zarr.py`。本仓库的读取与维度校验见 [ReplayBuffer](../dexmani_policy/datasets/replay_buffer.py)。
