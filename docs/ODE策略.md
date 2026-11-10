# ODE 策略

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

移植来源和主要调整记录在 decoder 文件头。

