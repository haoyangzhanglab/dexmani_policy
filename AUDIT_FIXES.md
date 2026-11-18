# 审查修复记录

本页记录 `d87b937` 对 `725ff2dcb88cd8c10ef25a37cb33e287ae05e138` 基线的审查修复。下文的验证命令对应当时三个仓库的版本。当前使用和验证入口见 [README](README.md)。

## 结论与验证

| 编号 | 决策与改动位置 | 验证与适用范围 |
| --- | --- | --- |
| P0-01 | 修复无效未来力污染。 `ForceDiffusion` 在加噪前用固定零占位，仍按有效元素计力损失。 | 固定观测、有效标签和 RNG，两种触觉路径的整段动作条件、动作/力损失和参数梯度均不受无效尾部有限值、NaN 或 ±Inf 影响。全有效时复现原计算和梯度，显式力权重为零时仍有动作梯度。 |
| P0-01 | 保留官方的训练与采样机制。训练力预测器对真实有效未来力加噪，部署从独立噪声采样。 | 原审查对信息路径的证据成立。当前修复消除了无效值影响，训练与部署的分布差异及任务影响仍待验证。 |
| P0-02 | `training/data.py` 在手臂切片和动作量化前流式检查完整动作，沿用先验手部 affine。 | 在直接导入的 Zarr 注入手部 NaN 和正负 Inf，确认报错含数据集、字段、来源、行和通道。验证正常编码和物理坐标往返，另用实际 DQ-RISE checkpoint 验证 warm start 和 resume 的拒绝路径。 |
| P1-01 | Sim 记录实际局部点序和分块，共享模板只计算一次摘要。Zarr 根属性沿用 Real 的几何身份定义。`contact_map.generate` 先检查固定槽，再一起发布点云和标签。 | SAPIEN 录制、HDF5 转换、Open3D 标签和 CordViP 读取实测通过。覆盖点序改变、混合模板、零接触权重和缺模板 HDF5 拒绝。用该链路产出的数据实测过预训练、续训及下游继承。缺少 `object_geometry` 的仿真 HDF5 须重新录制。 |
| P1-02 | 记录 CordViP 官方协调目标的解释边界。 | 同时刻完整关节状态可影响重建输出。重建损失衡量该条件下的拟合，跨肢体预测能力待单独验证。 |
| P1-03 | AFRO 默认使用完整窗口，`pad_before=pad_after=0`，`frame_stride=2`。归类为数据采样协议调整。 | 覆盖同回合、精确帧间隔、短回合与空批次。CUDA 预训练、续训和编码器继承通过。保留独立冻结 teacher 及其 EMA 更新顺序。 |
| P1-04 | VQ 和观测预训练通过 `--policy-config` 共享同源分组与预算。当前训练产物必须保存 `training_data`，同源继承检查训练组和验证组。 | 覆盖 Real 多段分组、训练组统计、超预算先验及换源续训拒绝。派生接触数据沿用原始 Zarr 来源。独立外部权重可注明来源，或显式记录来源未知。 |
| P1-05 | R3D 在配置和恢复边界要求点云 `limits`，与 Uni3D 和固定区间 clamp 配套。 | 默认配方及发布权重通过 CUDA 冒烟。配置、warm start、resume 和推理恢复拒绝 `mean_std`。 |
| P1-06 | 明确 resume 恢复模型、optimizer、scheduler、EMA 和步数，再重新播种和创建迭代器。 | 短训及恢复测试检查状态张量、学习率和步数。原审查的随机序列差异归类为恢复协议边界。 |
| P1-07 | 文档固定评测池、拆分 seed 和数量，先选模再在独立测试集报告。 | 回归覆盖拆分互斥和胜者参数。单次 SAPIEN 真闭环验证了执行和结果记录，正式多 seed 对照仍待实验。 |
| P1-08 | Real 逐行记录提交接受状态，导出 `action_ode` 并在失败时保持有效目标和时间轴。 | Raw→Zarr→ODEDataset 离线验证成功 A、失败 B、成功 C、连续失败、段首、HOI 联合导出的关节对齐和缺失证据。C 的上一目标为 A。已接受表示主机提交边界，硬件实际运动另有时序。 |
| P1-09 | 当前 leading DDIM 拒绝 1 NFE，覆盖构造、配置、评测、部署加载与 RTC。 | 独立检查 100 个训练时间步下的默认 10、20 步序列及 RTC 更新，Flow 和 ODE 的 1 步语义仍可用。 |
| P1-10 | 保留各动作表示的既有损失定义。 | 同等逐维误差下，SAT、DQ-RISE、DexLatent 的手臂损失份额分别为 7/19、7/8、7/39。在该目标下比较表示时，结果同时受损失配比影响。 |
| P1-11 | `data/README.md` 补齐已确认的上游源码和权重条款。 | AdapTac 上游声明冲突、Uni3D 发布权重的单独许可及整库授权仍待确认。 |
| P2-01 | 可选预训练 ResNet 的 BN→GN 复制 affine。 | 单元测试确认 γ、β 和 dtype 保留。运行统计行为的差异已写入权重文档。 |
| P2-02 | 提供 `.[dev]`，将仿真环境清理单元测试与仿真包解耦，另设原生集成入口。 | 无仿真环境回归通过，原生 CUDA FPS/KNN、Open3D 和 SAPIEN 单独实测。数值验证覆盖 Flow 方向、Consistency teacher target、ODE 解析轨迹、RTC 及 K/V 输出与梯度。 |
| P3-01 | `train_remote.sh` 将远程源码位置写进运行配置和 checkpoint。 | Shell dry run 及短训、续训证明快照路径随实验记录保存。未连接远端服务器执行同步。 |

## 验证记录

- 基线在已安装仿真包的环境通过 128 项。无仿真环境中 126 项通过，两项失败于仿真包导入。修复后无仿真环境通过 151 项。合并重复回归、移除一次性探针后，当前 Policy 149 项、Real 735 项和原生集成 2 项通过。运行方式见 [README](README.md#开发验证)。
- RTX 4090 上 11 个策略完成小规模 CUDA BF16 训练前向、反向和物理动作推理。AdapTac 五指力及密集 taxel 分别验证，后者使用合成密集触觉。R3D、DINOv3 和 DexLatent 测试加载了发布权重。验证已有 raw/EMA 权重恢复，AFRO 配方关闭 EMA。VQ、AFRO、CordViP 均走过实际预训练入口。
- 两项原生集成通过。运行顺序与部署一致，先初始化 Torch CUDA，再运行 SAPIEN。早期反序在同一进程中触发 CUDA illegal memory access。使用短训 DP3 实际完成 `pick_apple_messy` 的 seed 32 单回合闭环，记录 240 步并未完成任务。
- 本地 61 个历史 Real Raw 回合只有 `assumed_accepted`，另 5 个带单调时钟的回合未记录提交状态。两类均无法恢复 Real ODE 所需证据。需要新采集数据才能验证真机训练和闭环。完整评测池上的成功率与 AdapTac 训练/采样分布差异尚未实测。
- 审查期间三仓通过 mypy、变更文件的 Python 3.10 和 Shell 语法检查及离线 wheel 构建。解包的 Policy wheel 成功加载 15 个配方组合。

```bash
# 在 dexmani_real 根目录使用 real_robot 环境
python -B -m pytest -q -p no:cacheprovider tests/test_export.py tests/test_profile.py \
  tests/test_rollout_hoi_recovery.py tests/test_experiment_pipeline.py \
  tests/test_teleop_session.py tests/test_cordvip.py
python scripts/export_zarr.py "$RAW_ROOT" --output "$REAL_ODE_ZARR" \
  --fields joint_state point_cloud action_ode

# 在准备评测池和对应 checkpoint 后，在 dexmani_policy 根目录运行
python -m dexmani_policy.eval_pipeline --checkpoints "$CKPT" \
  --seed 1024 --selection-episodes 25 --test-episodes 100
```

## GPU 测量

在同一 RTX 4090 上比较基线和本轮源码。使用默认模型尺寸与 NFE、FP32、batch 1、2 帧观测、每帧 1024 点、16 帧动作窗口。取固定示教输入和随机种子。每组预热 10 次，测量 30 次并同步 CUDA。各阶段报告中位延迟和峰值已分配显存，整策略还报告 P95。整策略从设备上的观测张量到 CPU 物理动作，耗时包含该边界的传输，仿真和数据加载各自独立。

| 策略（动作/力 NFE） | 观测 ms 基线→本轮 / MiB | 生成 ms 基线→本轮 / MiB | 整策略 ms 中位数 基线→本轮 | 整策略 ms P95 基线→本轮 / MiB |
| --- | ---: | ---: | ---: | ---: |
| DP3（10/0） | 0.122→0.127 / 285.54 | 16.487→16.122 / 293.84 | 16.086→16.421 | 17.101→17.635 / 293.85 |
| ManiFlow（4/0） | 0.133→0.136 / 704.91 | 26.143→24.714 / 847.91 | 26.491→24.992 | 27.753→26.643 / 849.66 |
| AdapTac 五指力（20/20） | 1.135→1.134 / 144.58 | 44.203→43.407 / 144.98 | 44.993→45.216 | 47.130→46.309 / 145.00 |

三条路径的物理动作与基线逐值一致。当前 ManiFlow 保留 K/V 时生成中位延迟为 24.714 ms、峰值 847.91 MiB，关闭缓存时为 30.095 ms、717.32 MiB。回归另外比较缓存输出、参数梯度和新观测时的重建结果。推理实现没有新增性能改造，表中波动用于记录本次测量范围。
