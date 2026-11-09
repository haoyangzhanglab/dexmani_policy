# ODE Action Head：Codex 实施任务书

审查日期：2026-11-10。状态：**实现已接入，CPU 与 GPU compile 定向验收通过**。
本文件保留设计约定；当前使用与验收命令见 [README](README.md#ode-使用)。真实数据完整训练及真机闭环尚未验证。

本任务是在 `dexmani_policy` 中接入 ODE Policy 的关节动作头。按根目录 `AGENTS.md` 的个人研究代码标准实施：正确、直观、易读易改，优先复用现有机制与官方算法代码，完成最小但完整的训练和执行闭环。

## 1. 目标与范围

实现一份可以通过现有训练、恢复、仿真评测和部署入口使用的 `ode` 配方：

- 点云历史作为 condition；最新实测关节状态直接进入向量场。
- 对 19 维绝对关节目标积分：手臂 7 维、手部 12 维均为连续动作。
- 官方 FiLM-MLP、路径监督、起点收缩扰动、跨 chunk 动作承接。
- 默认 plain ODE；同一实现提供官方 detached diversity head 的开关。
- 复用现有 Dataset、PointNet、normalizer、trainer、EMA、checkpoint 和同步 chunk 执行方式。

这是官方机制在本项目点云与关节控制上的迁移，不是原论文 LIBERO 实验的逐项复现；不得把迁移默认值写成已验证的最优超参数。

本任务不扩展 `action_ee`、二值夹爪、手部 latent、RTC、异步/流式服务、多机器人控制协议或新的观测融合架构。不增加 `torchdiffeq`、adjoint、solver registry、policy factory、通用 adapter、自动调参或额外稳定性损失。不启动完整训练或大规模评测来代替针对性验证。

## 2. 官方来源

本任务书基于以下版本：

| 来源 | 已审查版本 |
| --- | --- |
| 官方主分支 | [`bcea95a299dc3fa8b2a571603a5e8b50e1c449e9`](https://github.com/NTUMARS/ODE-POLICY/tree/bcea95a299dc3fa8b2a571603a5e8b50e1c449e9) |
| 官方关节控制分支 | [`cc3e08b2cf95fe2b4545fa34a68c51bb9cec9d89`](https://github.com/NTUMARS/ODE-POLICY/tree/cc3e08b2cf95fe2b4545fa34a68c51bb9cec9d89) |
| 论文 | [ODE Policy, arXiv:2611.01729v1](https://arxiv.org/abs/2611.01729v1) |

优先阅读这些官方文件：

- [ode_image_policy.py](https://github.com/NTUMARS/ODE-POLICY/blob/bcea95a299dc3fa8b2a571603a5e8b50e1c449e9/ode_policy/policies/ode/ode_image_policy.py)：`_FiLMLayer`、`VectorField`、积分、loss、diversity head。
- [PORTING.md](https://github.com/NTUMARS/ODE-POLICY/blob/bcea95a299dc3fa8b2a571603a5e8b50e1c449e9/docs/PORTING.md)、[METHOD_NOTES.md](https://github.com/NTUMARS/ODE-POLICY/blob/bcea95a299dc3fa8b2a571603a5e8b50e1c449e9/docs/METHOD_NOTES.md)。
- [build_ode_zarr_real.py](https://github.com/NTUMARS/ODE-POLICY/blob/cc3e08b2cf95fe2b4545fa34a68c51bb9cec9d89/ode_policy/real/build_ode_zarr_real.py)：绝对关节目标与时间对齐。
- [eval_ode_policy.py](https://github.com/NTUMARS/ODE-POLICY/blob/bcea95a299dc3fa8b2a571603a5e8b50e1c449e9/ode_policy/libero/eval_ode_policy.py)：执行方显式传入 `a0`，承接已执行步对应的积分目标。

## 3. 文件与职责

以下路径均相对于 `dexmani_policy/`：

| 文件 | 当前职责 |
| --- | --- |
| `agents/action_decoders/ode.py` | `_FiLMLayer`、`VectorField`、`ODE` decoder；条件投影、积分、loss、可选 diversity head |
| `agents/policies/ode.py` | 薄的点云 encoder 与 `ODEPolicy(Policy)`；组装、归一化及训练/预测接口 |
| `datasets/variants/ode.py` | `ODEDataset(DemonstrationDataset)`；只补充 `prev_action` |
| `configs/ode.yaml` | 一份配方，沿用现有 Hydra 组装与训练入口 |
| `agents/policies/base.py`、`agents/policies/adaptac.py` | 公共 `prev_action` 签名；基类优化器收集整个 decoder |
| `deployment/metadata.py` | 从策略读取 `requires_prev_action`，供执行端判断承接需求 |
| `deployment/runtime.py`、`env_runner/sim_runner.py` | 传入物理单位的上一条已下发指令，执行端负责承接 |
| `test_tools/check_ode.py` | 复用 smoke test，完成算法与执行约定的定向验收 |

`common/inference.py:predict_actions()` 已支持 `**kwargs`，优先直接利用。`SequenceSampler` 已提供 `sample_source_rows()`，不新增 sampler 或重新生成 Zarr。Dataset 的 `_target_` 在 YAML 中直接指定，无需在配置组装器新增按策略名切换 Dataset 的分支。

移植来源、固定 commit 和主要调整记录在 decoder 文件头；针对性验收放在现有 `test_tools/` 中。

## 4. 必须正确的数据时间约定

### 4.1 以本项目原始数据为准

本项目第 `t` 行表示：动作执行前观测 `obs[t]`，以及随后下发的绝对目标 `action[t] = u_t`。实测 `joint_state[t] = q_t` 与目标 `u_t` 是不同物理量。

官方预处理把连续目标后移一帧，故其 `action[t]` 保存 `u_(t-1)`。本项目保持原 Zarr，不照搬官方的动作切片。

令 `N = n_obs_steps`、`H = n_action_steps`，本配方满足：

```python
horizon = N - 1 + H
start = N - 1
target = batch["action"][:, start:start + H]  # u_t ... u_(t+H-1)
prev_action = batch["prev_action"]           # u_(t-1)
```

默认 `N=2, H=8, horizon=9`：观测为 `[o_(t-1), o_t]`，原动作窗口为 `[u_(t-1), u_t, ..., u_(t+7)]`，监督目标是 `action[:, 1:9]`。

### 4.2 Dataset 只补充上一条指令

继承现有 `__getitem__()` 以复用读数、增强、点采样和 Tensor 转换。利用 `sample_source_rows(idx)[N-1]` 找到全局当前行 `t`，通过 `episode_ends` 判断当前回合起点：

- 非首帧：读取同一回合原始 `action[t-1]`。
- 首帧：读取原始 `joint_state[t]`，作为没有历史指令时的初始化约定。
- 首帧初始化必须在状态噪声增强之外取值，不能使用已增强的 `data["obs"]["joint_state"]`。
- 不读取上一回合的末帧；不把左侧 padding 复制出的 `u_0` 当作上一条真实指令。
- 保留末尾重复最后动作的现有 padding 规则，不新增 mask 体系。

返回顶层 `prev_action: float32[19]`，批处理后为 `[B,19]`。它不进入 `sensor_modalities`、`normalization_dims`，也不单独拟合统计；继续使用 `action` affine。

## 5. 模型与数值实现

### 5.1 观测、状态与向量场

复用 `PointNet`、`preprocess_point_cloud()` 和现有 DP3 点云采样配置。每帧编码后按时间顺序展平，得到 `[B, N*pc_out_dim]`；通过官方两层 `cond_net` 投影为 `[B,cond_dim]`。

不要直接使用包含 `StateEncoder` 拼接逻辑的整个 `DP3Encoder`。状态只使用最新帧，经 state normalizer 得到 `s: [B,19]`。condition 与 `s` 在整个 chunk 内固定。

`VectorField` 采用官方 FiLM residual MLP：输入拼接 `[a,s]`，输出 19 维变化率；保留输出层零初始化。向量场是 autonomous 的，不添加 diffusion/flow time 或动作步位置 embedding。

`action_decoder.model` 保持指向 `VectorField`；`cond_net` 和可选 `side_head` 直接注册在 decoder 上。无需额外的模型容器。

### 5.2 归一化边界

Policy 负责原始物理量与归一化坐标之间的转换，decoder 只处理归一化张量：

- `a0` 与所有动作 target：使用 `normalizer["action"]`。
- 首帧实测 `q_0` 作为 `a0` 时，也使用 action normalizer。
- 向量场条件 `s`：使用 `normalizer["joint_state"]`。
- 物理扰动标准差转为归一化标准差：`std_norm = std_rad * action_scale`，不加 offset。

不要把两个 affine 混用，也不要把数据集的 limits 范围当作硬关节限位向积分状态新增 `[-1,1]` 裁剪。

### 5.3 积分与时间单位

移植官方固定步长 Euler / RK4，直接通过短 rollout 反向传播。训练每个控制周期使用一个积分步；预测的 `inference_steps=m` 表示每个控制周期的数值子步数，默认 1。每个子步长度为 `1/m`，累计积分 H 个控制周期，且只在每个控制周期结束后输出一条动作。

`vel_scale * VectorField(...)` 的单位是“归一化动作 / 控制周期”。不再额外乘真实秒数。`inference_steps` 不改变 H 或观测刷新频率，也不是整个 chunk 的 NFE：Euler 的主 rollout 为 `H*m` 次场计算，RK4 为 `4*H*m` 次；启用 diversity head 另有起点特征读取。

观测编码、`cond_net` 和 FiLM 调制参数每个 chunk 计算一次；FiLM 预计算不得 detach。保留 FP32 积分累加与 loss，默认关闭 BF16。用明确循环实现，不新建 solver 抽象层。

### 5.4 路径监督与扰动

沿用官方算法。训练时以概率 `perturb_prob` 对每个样本加入独立高斯起点扰动，未选中的样本扰动为零：

```python
a0_perturbed = a0 + delta
target_perturbed[:, k] = target[:, k] + perturb_decay ** (k + 1) * delta
prediction = integrate(a0_perturbed, state, condition, H)
path_loss = mse(prediction, target_perturbed) / vel_scale**2
```

起点之后各步使用模型自己的积分结果，不做逐步 teacher forcing，不回归有限差分速度，也不加入额外平滑/jerk/stability loss。`eval()` 下的 loss 不随机扰动，沿用官方 `self.training` 语义。

按项目约定返回 `(loss, metrics)`，尊重 `log_metrics`；至少提供总 `loss`、路径损失，启用 diversity 时提供对应 loss。logging 只记录 detached 标量，不改变梯度。

### 5.5 Diversity head

在同一个 decoder 中直接移植官方 velocity-offset 分支。`side_head=0` 不创建和调用该 head；`side_head=12` 启用官方的 K 分量模型。保留官方 `side_hidden=256`、`spread_min=0.02`、`mix_min=0.05` 的内部默认值，无需把每个常量都放进 YAML。

必须保留以下数值与梯度关系：

- head 读取起点处的 detached 场特征；拟合 detached 路径残差。
- 每步非负 rate 经过 `cumsum` 得到累计尺度；保留原实现的 `vel_scale`、floor、mean/sum 与 eps 拟合细节，不凭公式印象重写。
- 同一个 chunk 只采样一次方向 epsilon；每个物理控制步内所有 RK4 子阶段/数值子步使用同一个该步 offset。
- 默认 `eps_center=true`：按 mixture 权重抽一个分量后取其均值，不是取最大权重分量，也不是混合均值。`eps_center=false` 才在分量内部采样。
- 随机抽分量和高斯采样都使用传入的 `torch.Generator`；遵循项目 PyTorch 2.4 的 API，必要时用 `torch.randn(shape, ..., generator=generator)`，不要假设 `randn_like` 支持该参数。
- 不引入对 NaN 权重静默替换为均匀分布的回退；沿用已有输入输出检查和明确失败行为。

公式等价性按固定输入与固定扰动验证。detach 保证 diversity loss 不给主场/encoder 额外梯度；不能据此宣称两种训练配置的参数更新轨迹完全相同，随机数消耗与全局梯度裁剪仍可能影响更新。无需为追求轨迹完全一致改造现有 trainer。

全连续关节版本不创建 gripper head，也不计算空 gripper 张量上的 BCE；不能只给官方整类设置 `n_grippers=0`。

## 6. Policy、执行端与兼容性

### 6.1 Policy 输出契约

`ODEPolicy` 声明 `supported_action_keys = ("action",)`、`supports_rtc = False`，覆写 `compute_loss()` 和 `predict_action()`，继承现有训练/normalizer 机制。

decoder 输出已经是 `[B,H,19]` 的未来序列。Policy 反归一化后直接返回 `{"future_actions": prediction}`，不再调用会裁去历史部分的 `_action_output()`。保持 `future_steps == H`，使 `predict_actions()` 的既有 shape 检查自然成立。

初始化时校验 `horizon - n_obs_steps + 1 == n_action_steps`。ODE 的预测方法保留公共推理签名；若给出 `rtc_prefix`，显式报不支持，不能因覆写基类而静默吞掉 RTC 请求。

### 6.2 显式传递上一条动作

统一增加 keyword-only `prev_action=None`：

- `Policy.predict_action()` 与 AdapTac 的独立覆写：接受并忽略，原有算法行为不变。
- `ODEPolicy.predict_action()`：接收物理单位的 `[B,19]`；None 仅用于回合初始化或无执行副作用的 warmup，此时读取最新实测关节状态作为起点。
- `PolicyRuntime.predict()`：接收物理单位的 `[19]`，利用已有数值 Tensor 准备函数批处理为 `[1,19]` 后转发；只在这个外部输入边界做一次必要 shape 检查。
- `common/inference.py` 利用已有关键字转发即可。基类不需要把 `prev_action` 再转入不使用它的 `_predict_from_condition()`，故 DQ-RISE/DexLatent 的这些覆写无需为此改签名。

不要新增 hidden carry buffer、execution state object、capability registry 或预测自动提交机制。

### 6.3 仿真与真机边界

`SimRunner._run_episode()` 中初始化局部 `prev_action=None`，将它传过 `predict_action_chunk()`；每执行一条 `env.step(action)` 后取 `info["executed_joint_action"].copy()`，承接环境按硬关节限位裁剪后实际下发的目标，与数据记录器一致。绝对关节执行保留角度分支，等价解由 IK/规划阶段选择。下一个回合重新初始化。

无裁剪时，承接目标等于官方 carried execution 的积分终点；裁剪生效时，这是本项目在 chunk 边界对起点的执行约束适配，不能声称整段执行仍是无约束 ODE 的同一条积分曲线。decoder 保留官方积分核心。

常规执行要求完整 H 步后再编码观测。`Policy.requires_prev_action` 默认为 False，ODE 声明为 True；仿真执行与部署元数据直接读取该标志。仿真共同入口检查 ODE 的 `action_steps == agent.n_action_steps` 一次。其他策略不读取承接字段，继续支持其已有的前缀执行行为。

终止或截断发生在 chunk 中间时立即结束，不能为了执行满 H 而继续控制环境。若系统中断后要继续任务，应重新定义一次明确的初始化，不能暗中视为已执行整段。

`PolicyRuntime` 仅提供预测，不知道外部硬件执行了多少动作。因此真机调用方必须每次传入上一条实际下发的关节目标，并按本配方执行 H 步。只新增 runtime 参数、而客户端一直省略它，会变成每个 chunk 从实测状态重启，不能称为已完成 carried execution。

`PolicyInfo.requires_prev_action` 读取加载后策略的声明，不写入 checkpoint。`dexmani_real` 在连接设备前检查同步、完整 H 步约束；每条 `Robot.send_action()` 正常返回后保存裁剪后的 19 维高层目标，再传入下一次预测。提交失败不推进承接值，中途停止清空承接。这里不承诺 SDK 回执、两设备原子执行或物理到位；Ruckig 中间参考仍由驱动管理。实机闭环尚未验证。

模型不持有执行状态，因此 `warmup()` 不推进承接动作；保留 runtime 已有的 generator 状态恢复。更新调用示例时不要把预测 chunk 的最后一帧提前当作已执行指令。

## 7. 优化器、EMA 与编译

基类 `optimizer_groups()` 收集整个 `action_decoder`，obs 参数组逻辑不变；特殊策略继续使用各自的参数组规则。

参数组覆盖 ODE 的 `model`、`cond_net` 与启用时的 `side_head`，由 `build_optimizer()` 检查每个可训练参数恰好出现一次。

现有 Diffusion/Flow 的可训练 decoder 参数位于 `.model` 下；对它们验证修改前后全局参数名列表、分组与组内顺序一致。仅检查集合不够，因为恢复训练会核对有序 `param_names`。不为本任务改 checkpoint 格式或添加兼容迁移逻辑。

继续使用已有 `compile_backbone()` 编译 `.model`，无需编译整个 Policy 或引入另一条编译路径。cond_net、预计算调制和 side head 保持正常 autograd。EMA、编译后的 `_orig_mod` 清理、保存与恢复沿用已有机制。

## 8. 默认配方

`configs/ode.yaml` 继承 `common`，沿用 DP3 的点云处理、batch、优化器和训练预算。保留所有真实入口需要的配置；下表是关键取值，不是完整 YAML：

| 项目 | 默认值/约定 |
| --- | --- |
| `policy_name` / `action_key` | `ode` / `action` |
| Dataset / Agent | `ODEDataset` / `ODEPolicy` |
| `horizon`, `n_obs_steps`, `n_action_steps` | `9`, `2`, `8` |
| 传感输入 | `joint_state`, `point_cloud` |
| 动作与状态维度 | `action_dim` 沿用组装器推导的 19；`state_dim=19` |
| 点云 | `pc_dim=6`, `pc_out_dim=128`, `num_points=1024` |
| affine | `action: limits`, `joint_state: limits`, `point_cloud: limits`，与所选 DP3 基线一致 |
| `cond_dim`, `hidden_dim`, `num_layers` | `512`, `512`, `4` |
| `solver`, `vel_scale` | `euler`, `0.05` |
| `perturb_prob`, `perturb_decay` | `0.5`, `0.7` |
| `perturb_std` | `0.01` rad 初值；支持标量或每关节值 |
| `side_head`, `eps_center` | `0`, `true`；完整 diversity 变体用 `side_head=12` |
| `eval.inference_steps` | `1`，每控制周期的积分子步数 |
| EMA / compile / BF16 | 沿用 DP3 的 EMA 与 compile 开关；BF16 默认 false |

perturb_std 与 vel_scale 需要按实际训练数据检查，不能宣称已调优。无需新增自动统计/估计流水线。普通测试关闭 compile；GPU 可用时再做一次相关编译检查。

训练命令：

```bash
bash scripts/training/train.sh ode task_name=pick_apple_messy
bash scripts/training/train.sh ode task_name=pick_apple_messy agent.solver=rk4
bash scripts/training/train.sh ode task_name=pick_apple_messy agent.side_head=12
```

无需创建新 launcher。用户给定真实数据后再训练；验收不能自动发起上述完整训练。

## 9. 必要验收与完成标准

复用现有 `test_tools/smoke_test.py` 的训练/恢复流程；需要新断言时，用一个短的 ODE 检查脚本或函数承载。不要建设新测试框架、测试矩阵平台或长期保存上游完整实现。

1. **数据语义**：用具有明确行号的两段以上合成轨迹，检查普通帧、各回合首帧、末尾 padding；覆盖 `n_obs_steps=1` 与默认 2。状态增强后初始 prev_action 仍等于原始状态，且从不跨回合取动作。
2. **数值移植**：用固定、非零权重与输入，对照固定版本官方核心的 Euler/RK4 rollout、路径 loss、梯度；关闭扰动或显式固定同一 delta，采用 FP32 合理容差。单独核对扰动衰减和 action/state affine 不同的情况。可在临时目录提取官方核心作为参考，不能以完整图像 policy 的无夹爪空 BCE 当作参考。
3. **diversity**：`side_head=0/12` 均可前向和反向且 loss 有限。单独对 diversity loss 反传，主场/condition encoder 没有该分支梯度，side head 有梯度；随机采样使用指定 generator，固定 seed 可复现。数值对照保留官方 rate/floor/eps 计算。
4. **参数与生命周期**：所有 ODE 可训练参数恰好入组一次；代表性已有 Diffusion/Flow 的有序参数组与旧规则一致。运行现有 DP3 smoke，运行 ODE 的少量训练更新、EMA、保存、恢复和 checkpoint-only runtime 加载；验证 `[1,H,19]` / runtime `[H,19]` 输出。至少在 side_head=12 配置覆盖一次保存/恢复。
5. **执行承接**：用轻量 fake env 或 predictor spy 覆盖有、无指令裁剪，检查 runner 传入的是环境实际执行的末条关节目标；第二回合首调用为 None；中途 done 或 truncated 不继续执行；warmup 不改变外部 prev_action，且恢复 generator 状态。ODE 拒绝不一致执行长度和 RTC prefix。此类控制流检查无需完整机器人仿真环境。
6. **编译**：现有环境具备对应 GPU 条件时，验证 `.model` compile 后训练/预测与保存恢复可用；缺少条件则明确列为未验证，不建立新编译回退系统。

CPU 定向验收和 RTX 4090 上的 compile、少量训练、保存恢复已通过；真机执行仅完成离线测试。真实数据完整训练、任务成功率、延迟和平滑性收益，以及真机闭环仍需独立实验验证。验收命令见 [README](README.md#ode-使用)。
