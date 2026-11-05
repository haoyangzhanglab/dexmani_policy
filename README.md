# DexMani Policy

灵巧手操作策略学习研究仓库，用于在统一的数据、训练和评测流程中研究观测表征、动作表示与生成式策略。项目覆盖示教学习、仿真闭环评测，以及供真实机器人项目调用的策略推理接口。

## 总体思路

训练时，策略从示教中学习“给定观测历史，可以采取怎样的动作序列”。执行时，它将图像、点云、机器人状态或触觉编码为条件，通过扩散或 Flow 模型把噪声逐步变成动作窗口。执行端先执行窗口前部的动作，再用新观测重新预测，形成闭环。

各方法共用数据时序、归一化、训练恢复和评测约定，主要在三处形成差异：

- **观测表征**：将场景汇聚为全局特征、保留局部空间特征序列（token），或围绕手部组织交互特征。
- **动作表示**：直接生成归一化动作，或借助手部码本、冻结先验生成动作。
- **学习目标**：动作生成之外，可加入观测预训练、一致性约束或未来触觉监督。

完整的数据流和模块关系见[项目架构](docs/项目架构.md)。

## 策略与研究方向

| 方法 | 主要关注点 |
| --- | --- |
| DP / DP3 | 分别以 RGB＋关节状态、点云＋关节状态为条件的扩散策略 |
| R3D | 使用预训练三维编码器与空间条件的扩散策略 |
| ManiFlow | 保留逐点特征，以 Flow Matching 和参数滑动平均教师的一致性目标学习动作生成 |
| SAT / DQ-RISE / DexLatent | 使用共同的几何与图像语义编码，分别研究关节轨迹组织、离散手姿码本的连续索引和连续手部潜表示 |
| AFRO / CordViP | 先学习时序动力学或接触与协调表征，再迁移到动作策略 |
| AdapTac | 预测未来触觉力，并用其引导视觉与触觉融合及动作扩散 |
| InterAct-Dex | 结合身体、局部场景、触觉与历史表征，以未来力增量提供训练辅助监督 |

这些实现包含面向本项目机器人、数据和训练流程的适配。比较时需同时考虑观测、动作空间、先验和监督目标。

## 数据与模型的关系

示教数据以 episode 组织，每条样本由观测历史与对应动作窗口构成。数据的时间对齐、单位和坐标系定义见[仿真数据集格式](docs/仿真数据集格式.md)。

部分方法需要外部视觉权重或独立预训练产物。来源与用途见[data 说明](data/README.md)，方法机制见[手部动作表示](docs/手部动作表示.md)和[观测预训练](docs/观测预训练.md)。完整策略 checkpoint 保存模型、构造配置和最终归一化统计，推理无需原训练数据或初始权重文件。

恢复直接加载 checkpoint 中的 affine 数值，不根据历史 normalization 标签重新拟合或解释。触觉训练支持 `identity`、`limits`、`gaussian`、`rms` 和 `max_abs`，默认仍为 `limits`；identity 保存对应形状的单位 affine，未来力损失也使用该数值。normalization 可包含未使用字段，仅校验和拟合实际观测与动作；缺少所用字段仍报错。InterAct-Dex 接受满足实际实例的输入、token 数与输出维度且配置可重建的触觉编码器；模型配置 normalizer 时检查数据字段与编码器输入匹配。

resume 更换数据路径时，部署周期与预处理 recipe 按当前数据刷新，恢复的模型结构、raw/EMA 权重、affine、optimizer、scheduler 和 RNG 仍沿用 checkpoint。仿真触觉表示接触力 N；Real 的 SDK 通道、尺度和坐标须按其实际含义使用，同名同形状不能证明跨域等价。

仿真由 `dexmani_sim` 提供环境；真实观测构建、硬件控制与执行调度由独立的 `dexmani_real` 项目负责。本仓库负责策略学习与推理。

## 快速检查

在 `policy` 环境从仓库根目录运行最小 CPU 检查，无需真实数据或预训练权重：

```bash
python -m dexmani_policy.test_tools.core_checks
```

使用 CUDA 时加 `--device cuda:0`。检查覆盖 DP3 的数据采样、归一化、短训练、保存恢复，以及删除原数据后的 checkpoint 加载与推理，并核对恢复的归一化参数。数据与训练产物均在临时目录中自动清理。

对已有示教数据，可使用指定配置做短训练与恢复检查，默认先训练 4 步，再恢复训练至多 2 步（不超过配置的训练总步数）：

```bash
python -m dexmani_policy.test_tools.smoke_test dp3 --device cpu \
  --override zarr_path=/path/to/dataset.zarr \
  --override training.use_compile=false
```

通过 `--max-updates` 和 `--resume-updates` 调整两阶段的步数，配置的训练总步数须大于第一阶段步数。通过 `--output-dir` 指定新的训练目录以保留产物；未指定时自动清理。其他策略可替换配置名，并提供各自需要的数据和初始权重。

## 阅读导航

建议先读项目架构，再按研究问题阅读手部动作表示、观测预训练或触觉与交互建模；核对数据与实验结果时查阅数据格式和评测机制。远程操作集中在 SSH 文档。

| 文档 | 回答的问题 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 系统如何组成，数据和动作如何流动，各策略在哪里形成差异？ |
| [仿真数据集格式](docs/仿真数据集格式.md) | 观测和动作分别表示什么，时间、单位与坐标如何约定？ |
| [手部动作表示](docs/手部动作表示.md) | SAT、DQ-RISE 和 DexLatent 如何表示和生成手部动作？ |
| [观测预训练](docs/观测预训练.md) | AFRO 和 CordViP 学习什么，表征如何迁移到策略？ |
| [触觉与交互建模](docs/触觉与交互建模.md) | 当前触觉如何融合，未来力在 AdapTac 和 InterAct-Dex 中分别起什么作用？ |
| [仿真评测机制](docs/仿真评测机制.md) | 如何闭环执行、选模，并进行独立测试？ |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 如何连接服务器、同步数据、远程训练和取回结果？ |

## 真机部署接口

`load_policy(POLICY, *, weights=None, inference_steps=None, device="cuda:0", seed=0)` 一次读取 checkpoint 并返回带 `.info` 的 runtime。接受 checkpoint 文件或实验目录；目录只读取 `checkpoints/latest.pt`。支持 `~`，相对路径基于调用时工作目录。weights 和 inference_steps 缺省时沿用保存配置。

```python
from dexmani_policy.deployment import load_policy

runtime = load_policy("/path/to/checkpoints/latest.pt", weights="ema", device="cuda:0", seed=0)
info = runtime.info
```

`PolicyInfo` 保留部署使用的输入字段、动作表示、观测与预测长度、RTC 能力、控制周期和预处理参数；`task_name` 继续用于 Raw 任务标签。`policy_name` 仍用于训练配置、模型选择和 checkpoint，不进入 `PolicyInfo` 或 `to_dict()` 快照。

`PolicyInfo.future_steps` 是物理未来动作长度，`supports_rtc` 表示模型的实际 RTC 能力；sync/async 由执行端调度。SimRunner 和部署共用观测准备与动作输出边界，`predict` 返回连续 NumPy `float32 [future_steps, control_action_dim]`，已反归一化并由模型裁掉历史及辅助维度；Sim 执行端再取前缀。RTC 接收同一物理表示的未消费前缀。`warmup(observation)` 只预测一次；传入 `rtc_guidance_cap` 时再使用首预测预热一次 RTC，cap=0 同样生效。预热恢复 RNG，不返回计时统计；调用方负责每段 `reset_episode()` 和最终 `close()`。
