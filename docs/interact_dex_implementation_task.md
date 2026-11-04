# InterAct-Dex 实施记录

本文件记录已完成的有限验证，不再作为待执行任务书。当前接口与数据流以[项目架构](项目架构.md#handscenefusion-触觉融合与历史)为准，配方、命令、日志及离线诊断见 [README](../README.md#interact-dex-身体表征与未来力监督)。

## Scene12 表征

scene12 实施基于 `c302959` 工作区，完成后的代码基线为 `a3f5f13`。共享一次 Sparse3D，Encoder 返回 `interaction[B,T,6,192]` 与 `scene[B,T,12,192]`，Policy 按帧打包；默认条件为 `[B,72,192]`，在线动作为 `[B,13,19]`。SceneTokenizer 使用真实三维 FPS anchors，一次共同原点 3D cross-attention 读取全部 patches，再经过小 FFN。

当时的 encoder 模式默认模型为123,483,129参数，其中 SceneTokenizer 为445,824、future head 为35,331。该参数量对应当时默认配置，不适用于 action_prefix 模式或非默认跨度、宽度。

已有验证环境为 Python3.10.20、PyTorch2.4.1+cu124、PyTorch3D0.7.8，设备为CPU及沙箱外RTX4090。scene12 实施记录包含：

- CPU observation 检查与20项 InterAct 配置检查通过。
- CPU/CUDA 定向检查覆盖 FP32、BF16 autocast、96→12压缩、非 seed memory、共同 RoPE 原点、梯度、一次观测编码、零辅助权重、非默认 horizons、K/V 缓存及 RTC。
- 合成 Trainer 检查覆盖 limits/rms/max_abs、零权重不读未来标签、raw/EMA strict恢复、normalizer不重拟合、optimizer/scheduler恢复及连续训练对照。
- 正式宽度、batch2、FP32、真实 `pick_apple_messy.zarr` 的 fresh4步与原目录resume2步 smoke 通过；保存预算为100,000步。产物的 raw/EMA strict恢复、72-token条件、离线未来力、在线13步动作及RTC检查通过。该记录仅验证 scene12 的 encoder 配方。

真实数据 smoke 的复现入口如下；输出目录须尚不存在：

```bash
OMP_NUM_THREADS=4 python -m dexmani_policy.test_tools.smoke_test interact_dex \
  --device cuda:0 --max-updates 4 --resume-updates 2 \
  --output-dir experiments/train/interact-dex/scene12_smoke \
  --override zarr_path=robot_data/pick_apple_messy.zarr \
  --override task_name=pick_apple_messy --override dataloader.batch_size=2
```

## 共享动作前缀监督

动作前缀增强基于 `a3f5f13`。保留 scene12 观测结构和主动作目标，以一个读取头从共享 DiT-X 合法动作前缀预测未来力变化；每批仅抽一个有有效标签的跨度。`soft_flow` 是该辅助分支的可选时间权重，默认模式仍为 `encoder + none`。

初次实施的有限验证记录：

- 25项 InterAct 配置检查通过，覆盖模式与时间加权约束。
- CPU 前缀/encoder 检查及完整 CPU decoder 检查通过。
- RTX4090 上的前缀/encoder、Flow时间和DiT-X精度检查通过；前缀检查覆盖非退化信息边界、辅助梯度与detach对照、权重/mask、单次前缀调用、推理隔离、BF16和随机跨度恢复。
- 合成小模型通过正式训练入口的连续3步与2+1步恢复对照，比较 raw/EMA、optimizer和scheduler，核对 effective config 与 strict加载。

清理后的验证记录：

- 全部82项配置检查、完整 CPU training/decoder/dataset 检查通过；数据检查包含 fork/spawn DataLoader，训练检查包含其他共享 Flow/DiT-X 策略的保存恢复回归。
- CPU 与 RTX4090 前缀/encoder 检查通过，新增核对各跨度标签对齐、唯一有效跨度、一次动作归一化；正式训练恢复比较实际跨度选择序列及真实 EMA。
- 三份核心源码的 mypy、受影响 Python 文件的 compileall 和本地 wheel 构建通过。
- CPU `torch.compile(backend="aot_eager")` 下主 forward 与前缀特征、梯度、导出参数名对照通过。这不覆盖默认 CUDA Inductor 后端。

## 实验边界

尚未完成完整100k-step训练、真实任务闭环对照、泛化实验、完整宽度吞吐/峰值显存基准或 action_prefix 的默认 CUDA Inductor 编译检查。有限 smoke 和合成检查不能证明成功率或效率提升。

前缀版本应使用相同验证窗口、noise、tau、模型权重类型和推理设置，报告共同h=4未加权误差，多跨度版本额外报告h=1、2。最终比较依据是共同闭环协议下的任务表现、失败类型和实际成本。时间加权仍是待验证假设，没有额外闭环收益时关闭 `soft_flow`，保留多跨度版本。
