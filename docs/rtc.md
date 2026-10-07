# Real 数据配方与推理式 RTC

## 数据与保存统计

BaseDataset 对实际启用的输入生成 obs_valid，对完整监督（含启用的 action_ee 前九维）生成 action_valid。Real canonical 额外读取 row_info/dispatch_status，默认要求两设备均为 ACCEPTED=1；缺少 dispatch 明确报错。CRC_UNCONFIRMED=2 不是 accepted。普通仿真数据不要求硬件 row_info，辅助 NaN 和未使用模态不会自动删除整段。

训练划分可由 `dataset.split_manifest` 显式指定。JSON 包含：

- 与缓存相等且已知的 `data_revision`，以及按缓存顺序排列的 `episode_ids`；
- episode→已确认 trial 的 `trial_ids` 映射，以及 `seed`、`group_unit`；
- 完整、互斥的 `train_ids`、`val_ids`、`exclusions`，同 trial 不跨 train/val。

新训练的清单直接决定最终 episode 集合，要求 `val_ratio=0`、`max_train_episodes=null`，冲突在打开 ReplayBuffer 和扫描数据前报错；预算应在清单准备阶段落实。不从目录名、暂停或 HOME 推断 trial，排除项不会进入任一侧。旧 manifest+cap 续训读取保存的完整清单和 `actual_train_ids`，不重新抽样，外部清单文件可以不存在。实际读取的完整规范化清单、SHA-256、最终 mask/子集及 episode/trial/有效窗口数量进入现有 `data_recipe` 并随 resolved config 保存；无验证侧时明确 `holdout=false`。这是训练划分，与仿真评测的 `eval.seed_manifest` 独立。

未提供清单时保留原 episode split、配置和 data_recipe 结构；默认配置不添加 `split_manifest: null`。新清单不接受 unknown revision，可从 Raw 导出到新路径后使用；不向旧缓存回填身份。修改清单内容或实际划分属于新实验，strict resume 会拒绝配方变化。

在划分和 max_train_episodes 后建立 train/val 窗口。对 H 长窗口，sampler 的 padding 映射及数值/dispatch 资格是（启用时间筛选时还须满足下述时间条件）：

```text
r = clip(buffer_start + arange(H) - sample_start, buffer_start, buffer_end - 1)
obs_valid[r[:N]].all() & action_valid[r[:H]].all() & dispatch_valid[r[:H]].all()
```

不删除内部坏行后重拼、不跨 episode、不改变 padding loss。观察统计用有效训练窗口 r[:N] 的唯一源行，动作统计用 r[:H] 的唯一源行；不同角色不互相扩大取值范围。验证复用训练统计，零有效窗口明确失败，日志输出候选/有效数和可重叠的拒绝原因。新训练的 resolved config 保存 data_recipe 与统计源行数。

部署从 checkpoint 恢复 normalizer，不读取训练数据拟合。训练恢复也直接读取保存统计；数据配方加入现有续训一致性检查，新配方不是旧实验的无缝续训。精确复现旧训练请使用其源版本；不迁移旧缓存、不覆盖历史模型。推理式 async/rtc 不要求为了算法本身重新训练。

Dataset 的窗口仍按实际记录行索引（recorded_rows）构造，不修补时间、不压紧坏行。新 Real 训练默认 `max_time_gap_ratio=1.5`，要求 H 窗口内真实源行时间为正、相邻不同源行满足 0<Δt≤ratio×canonical 保存的 dt；H=1 同样拒绝未知时间，padding 重复源行不构成零间隔。通过 Hydra 可用 `+dataset.max_time_gap_ratio=null` 显式禁用，data_recipe 记录 unfiltered；这是一项研究资格选择，不是安全阈值或 QA 自动过滤。仿真配方不增加时间规则。

恢复在 Dataset 构造前解析 checkpoint 的 data_recipe：保存的新规则直接恢复；已知 role_finite_v1/unique_train_source_rows 旧规则缺键时维持无时间筛选及原 recipe 表示，不注入新计数或默认值。证据不足拒绝 full resume，推理不受影响。normalizer 继续从 checkpoint 严格恢复。Policy-aligned VQ 与 usage 采用同一保存规则；统计改变时需重新建立对齐码本，不放宽 hand affine 一致性。旧 Raw、配置、码本与 checkpoint 均不原地改写。

Zarr reader 按进程重开句柄，只保留选中字段和每字段一个有界当前 chunk；训练期间不得替换当前读取的缓存。normalizer 单遍合并统计，mixed action 的 xyz/hand 使用 limits、rot6d 保持 identity，辅助 EE 切片与唯一训练源行权重不变。多任务 Dataset 读取固定真实索引，训练与 deterministic validation 都由 `ResumableDistributedSampler` 产生顺序；不创建 Manager，也不在 worker 中维护 epoch 表。

## 推理接口

`PolicyInfo.horizon` 提供 H，N=n_obs_steps，P=H-N+1。LoadedPolicy.predict(observation, *, rtc_prefix=None, delay_steps=0) 返回物理控制子空间的 (P,C) NumPy 数组，起点是模型索引 N-1，C 为 joint19 或 EEF21。内部 Agent 的 pred_action/control_action/tail 约定不变，不额外运行网络获取 tail。

Real 只提供物理数组和时序；Policy 负责 checkpoint affine 归一化。joint+aux 的 28 维统计只取实际 19 个控制维给前缀归一化一次，不向辅助输出填造目标。Policy 检查 pred_action 的 tensor、batch/horizon/control 结构及浮点类型，返回完整 (P,C) CPU float64 future，包括 NaN/Inf。Real 在准入执行前拒绝非有限值，并将归属正确且未结束 attempt 的合法浮点数组留存 NPZ；严格 JSON 不保存非有限数。独立 warmup 则检查初始化、测量和测试 prefix 的所有输出，发现非有限值直接失败。这不代表真机安全或闭环质量已验收。

warmup 的 `rtc_delay` 必须为非 bool 整数且满足 0≤delay≤P。`guidance_cap=0` 时不构造 prefix，内部按 delay=0 执行普通采样；正 cap 使用配置 delay。bootstrap 普通路径和 steady 路径各自先初始化一次，再返回命名的少量耗时集合；sync/async/beta=0 复用同一普通测量，正 cap 的 prefix 从初始化结果取得并复用。初始化不计入测量；预热前后 reset，保证相同 seed 的首次实际推理不变。测量包含 RGB 预处理、模型及 CPU 返回，不包括 Real 观测构建、IK、owner 与 SDK。公共 predict 仍拒绝无 prefix 时传非零 delay。

`configure_execution(mode, guidance_cap)` 只配置推理模式；Real 串行 worker 随后独立调用 `warmup(samples=..., rgb_hw=..., rtc_delay=...)`，取得 bootstrap/steady 测量。`configure_execution('rtc', beta)` 显式配置 RTC，未支持的 agent/decoder 报 NotImplementedError。当前为连续动作 BaseAgent + Diffusion 的 DDIM 路径；DP/DP3/R3D 的 UNet、OneWayTransformer 已有 CPU VJP 测试。SAT、DQRise、flow 等未适配路径仍可使用原有 sync/async，不会将 RTC 请求默默降级。模式调度和硬件预算由 dexmani_real 管理。

## DDIM-adapted guidance

使用 diffusers==0.27.2 的 DDIM，eta=0；支持 sample、epsilon、v_prediction。dynamic thresholding 等未适配配置拒绝。沿用 Gaussian action normalization 与 clip_sample=True 的不兼容检查。

无前缀、空前缀或 beta=0 直接调用原采样，保持随机数消耗和输出。编码条件计算一次并 detach；每个 denoising step 在关闭外层 inference mode 的局部范围先使 noisy action requires_grad，再运行 denoiser。

前缀长度 L、承诺长度 d 满足 0≤d≤L≤P。时间权重为 i<d 时 1；软区 z=(L-i)/(L-d+1)，w=z*expm1(z)/(e-1)；其余位置为0。模型位置 N-1+i 对应 future i，历史位置和辅助维 endpoint residual 为0。

令 a=sqrt(alpha_bar_t)，b=sqrt(1-alpha_bar_t)。未裁剪 clean estimate f：sample 为 m，epsilon 为 (x-b*m)/a，v_prediction 为 a*x-b*m。所有实际 denoising t 要求 a,b>0。

```text
residual = detach(W * (Y_normalized - f))  # 仅实际约束位置求值
g = VJP(f, x, residual)
base = scheduler.step(m, t, x).prev_sample
x_next = base + (a_prime*b - a*b_prime) * min(beta, 1/(a*b)) * g
```

previous timestep 严格使用 0.27.2 的 t - num_train_timesteps // num_inference_steps，末步使用 final_alpha_cumprod。base 保留原 clip 和默认不重算 epsilon 的行为；VJP 使用未裁剪 f。最终一步仅当 clip_sample=True 才把 guided sample 投影至相同 clip_sample_range。

W 只乘一次。虽然辅助维没有 endpoint 目标，网络耦合产生的辅助 latent gradient 仍保留。grad_outputs detach，create_graph=False，每步结果 detach，不积累模型参数梯度。冻结承诺由 Real 调度器保证，有限 guidance 不保证网络前 d 项精确相等；这些项不会重复执行。此实现是推理式 DDIM 适配，不宣称完整复现 flow 论文的实验性质。

## 离线验证

```bash
python -m pytest -q tests/test_research_split.py tests/test_policy_windows.py tests/test_policy_rtc.py \
  tests/test_streaming_dataset.py tests/test_infra_training.py \
  tests/test_infra_resume.py tests/test_infra_evaluation.py
python dexmani_policy/smoke_test.py --config-only dp dp3 r3d sat dqrise maniflow
```

测试覆盖生产 Dataset/sampler/normalizer、三种 prediction_type、RTC-off 逐元素等价与 RNG、非平凡 Jacobian 有限差分、辅助维耦合、history offset、clip/base/最终投影、28→19 维统计和实际 backbone 的 input VJP。临时 checkpoint 仅用于验证保存统计不重拟合。

这些测试不使用真实设备，也不证明真实训练权重的闭环质量。实际 GPU 延迟/内存与新 sync/async/rtc 的任务收益必须使用适用 checkpoint 在同权重、seed、NFE、观察协议及预算下另行验证。

单任务训练在模型构建时核对 Dataset 声明与实际 `consumed_observation_fields`；部署在模型恢复后、连接设备前复用同一核对。辅助监督与训练元数据不属于输入相等关系。MultiTask 保留 Agent 外层消费的 task_text，训练/仿真不受真机支持范围限制；当前真机尚无任务选择和 child 数值配方恢复，明确拒绝。
