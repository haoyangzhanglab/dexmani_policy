# Real 数据配方与推理式 RTC

## 数据与保存统计

BaseDataset 对实际启用的输入生成 obs_valid，对完整监督（含启用的 action_ee 前九维）生成 action_valid。Real canonical 额外读取 row_info/dispatch_status，默认要求两设备均为 ACCEPTED=1；缺少 dispatch 明确报错。CRC_UNCONFIRMED=2 不是 accepted。普通仿真数据不要求硬件 row_info，辅助 NaN 和未使用模态不会自动删除整段。

先 episode split、max_train_episodes，再建立 train/val 窗口。对 H 长窗口，原 sampler 的 padding 映射仍是：

```text
r = clip(buffer_start + arange(H) - sample_start, buffer_start, buffer_end - 1)
obs_valid[r[:N]].all() & action_valid[r[:H]].all() & dispatch_valid[r[:H]].all()
```

不删除内部坏行后重拼、不跨 episode、不改变 padding loss。观察统计用有效训练窗口 r[:N] 的唯一源行，动作统计用 r[:H] 的唯一源行；不同角色不互相扩大取值范围。验证复用训练统计，零有效窗口明确失败，日志输出候选/有效数和可重叠的拒绝原因。新训练的 resolved config 保存 data_recipe 与统计源行数。

部署从 checkpoint 恢复 normalizer，不读取训练数据拟合。训练恢复也直接读取保存统计；数据配方加入现有续训一致性检查，新配方不是旧实验的无缝续训。精确复现旧训练请使用其源版本；不迁移旧缓存、不覆盖历史模型。推理式 async/rtc 不要求为了算法本身重新训练。

## 推理接口

`PolicyInfo.horizon` 提供 H，N=n_obs_steps，P=H-N+1。LoadedPolicy.predict(observation, *, rtc_prefix=None, delay_steps=0) 返回物理控制子空间的 (P,C) NumPy 数组，起点是模型索引 N-1，C 为 joint19 或 EEF21。内部 Agent 的 pred_action/control_action/tail 约定不变，不额外运行网络获取 tail。

Real 只提供物理数组和时序；Policy 负责 checkpoint affine 归一化。joint+aux 的 28 维统计只取实际 19 个控制维给前缀归一化一次，不向辅助输出填造目标。完整返回 chunk 的有限性准入在 Real；Policy 保留形状/浮点 dtype 检查，独立 warmup 检查自身输出。

`configure_execution('rtc', beta)` 显式配置 RTC，未支持的 agent/decoder 报 NotImplementedError。当前为连续动作 BaseAgent + Diffusion 的 DDIM 路径；DP/DP3/R3D 的 UNet、OneWayTransformer 已有 CPU VJP 测试。SAT、DQRise、flow 等未适配路径仍可使用原有 sync/async，不会将 RTC 请求默默降级。模式调度和硬件预算由 dexmani_real 管理。

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
python -m pytest -q tests/test_policy_windows.py tests/test_policy_rtc.py \
  tests/test_infra_training.py tests/test_infra_resume.py tests/test_infra_evaluation.py
python dexmani_policy/smoke_test.py --config-only dp dp3 r3d sat dqrise maniflow
```

测试覆盖生产 Dataset/sampler/normalizer、三种 prediction_type、RTC-off 逐元素等价与 RNG、非平凡 Jacobian 有限差分、辅助维耦合、history offset、clip/base/最终投影、28→19 维统计和实际 backbone 的 input VJP。临时 checkpoint 仅用于验证保存统计不重拟合。

这些测试不使用真实设备，也不证明真实训练权重的闭环质量。实际 GPU 延迟/内存与新 sync/async/rtc 的任务收益必须使用适用 checkpoint 在同权重、seed、NFE、观察协议及预算下另行验证。
