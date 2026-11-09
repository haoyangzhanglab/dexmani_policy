# DexMani Policy

灵巧手操作策略学习研究仓库，围绕示教数据、观测表征和动作生成，提供统一的训练与仿真评测流程。

## 整体流程

示教轨迹被组织为观测历史与动作窗口。策略编码图像、点云、机器人状态或触觉，通过 Diffusion、Flow 或 ODE 模型生成动作序列。执行端执行配置的动作步数，再根据新观测重新预测，形成闭环；ODE 要求完整执行训练时的 chunk 长度。

研究主要围绕三个部分展开：

- **观测表征**：从全局场景特征、局部空间特征到手部交互历史。
- **动作表示**：直接生成关节目标，或借助手姿码本、连续手部潜表示生成动作。
- **学习目标**：在动作学习之外，引入观测预训练、一致性学习或未来触觉监督。

本仓库负责策略学习与推理；`dexmani_sim` 提供仿真环境，`dexmani_real` 负责真实观测构建和硬件执行。模块分工见[项目架构](docs/项目架构.md)。

## 策略方向

| 方法 | 核心思路 |
| --- | --- |
| DP / DP3 | 分别以 RGB、点云结合关节状态，学习扩散策略 |
| ODE | 点云历史作为 condition，最新实测状态进入 FiLM 向量场，从上一条已下发指令积分绝对关节目标 |
| R3D | 使用预训练三维表征和空间条件生成动作 |
| ManiFlow | 结合 Flow Matching 与一致性学习 |
| SAT / DQ-RISE / DexLatent | 共享空间观测编码，研究关节轨迹、手姿码本与连续手部潜表示 |
| AFRO / CordViP | 先学习场景动力学或接触与协调表征，再迁移到策略 |
| AdapTac / InterAct-Dex | 融合触觉与场景，研究未来力引导和手部交互建模 |

各方法均接入本项目的数据与机器人设置，具体结构和监督方式有所不同。

## ODE 使用

```bash
bash scripts/training/train.sh ode task_name=pick_apple_messy
bash scripts/training/train.sh ode task_name=pick_apple_messy agent.solver=rk4
bash scripts/training/train.sh ode task_name=pick_apple_messy agent.side_head=12
```

默认 `n_obs_steps=2, n_action_steps=8, horizon=9`，支持 19 维绝对关节目标。
原始数据第 t 行保存观测后的指令 `u_t`；Dataset 另取同回合 `u_(t-1)`，首帧取未增强的实测状态。
动作起点和 target 使用 action affine，最新状态使用 joint_state affine。`perturb_std=0.01` rad
与 `vel_scale=0.05` 是迁移初值，尚未调优。`eval.inference_steps` 表示每个控制周期的积分子步数，
不改变输出长度或控制周期。diversity 默认按 mixture 权重抽分量并取其均值；
`agent.eps_center=false` 才在分量内采样。ODE 不支持 RTC。

真机调用方负责承接实际指令。以下函数由客户端提供；`send_joint_target` 必须按数据中的控制周期逐条下发，并返回实际提交目标：

```python
from contextlib import closing
from dexmani_policy.deployment import load_policy

with closing(load_policy(checkpoint_path)) as runtime:
    prev_action = None  # 每个回合重新初始化；warmup 不提交任何动作。
    runtime.warmup(read_history())
    while not episode_finished():
        chunk = runtime.predict(read_history(), prev_action=prev_action)
        for target in chunk:  # 正常执行完整 H 步，然后再编码观测。
            sent_target = send_joint_target(target)
            # 返回客户端裁剪后、同一关节坐标下的实际提交目标。
            prev_action = sent_target.copy()
            if episode_finished():
                break
```

`prev_action` 为物理单位 `[19]`；省略参数仅用于初始化或无执行副作用的 warmup。
不能在执行前将预测末帧当作上一条指令。中途终止立即停止，重新启动时显式初始化。
`dexmani_real` 已按 `PolicyInfo.requires_prev_action` 接入上述承接，并在连接设备前检查 ODE 的同步、完整 H 步执行约束。
真机承接值取自 `Robot.send_action()` 正常返回后的裁剪目标；这表示高层指令提交，不表示 SDK 回执或物理到位。实机闭环仍未验证。

定向验收（在 `policy` Conda 环境中）：

```bash
python -m dexmani_policy.test_tools.smoke_test --device cpu --max-updates 2 --resume-updates 2
python -m dexmani_policy.test_tools.check_ode
python -m dexmani_policy.test_tools.check_ode --device cuda:0 --compile
```

ODE 检查仅使用临时合成数据，每种 head 训练 2 次、恢复 2 次；不启动完整训练。
独立官方数值/梯度对照需将固定版本
[`ode_image_policy.py`](https://github.com/NTUMARS/ODE-POLICY/blob/bcea95a299dc3fa8b2a571603a5e8b50e1c449e9/ode_policy/policies/ode/ode_image_policy.py)
下载到临时目录，并加 `--upstream /tmp/ode_image_policy.py`；脚本仅提取连续算法核心进行对照。
不提供该参数时明确跳过官方对照。移植来源和主要调整记录在 decoder 文件头。

## 阅读导航

| 文档 | 内容 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 数据流、核心模块与策略组成 |
| [仿真数据集格式](docs/仿真数据集格式.md) | 回合组织、主要模态与观测动作关系 |
| [手部动作表示](docs/手部动作表示.md) | SAT、DQ-RISE 与 DexLatent 的表示和解码 |
| [观测预训练](docs/观测预训练.md) | AFRO 与 CordViP 的学习目标及迁移 |
| [触觉与交互建模](docs/触觉与交互建模.md) | 当前触觉融合与未来力预测的作用 |
| [仿真评测机制](docs/仿真评测机制.md) | 闭环执行、候选选择与独立测试 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远程同步、训练、续训与结果取回 |
