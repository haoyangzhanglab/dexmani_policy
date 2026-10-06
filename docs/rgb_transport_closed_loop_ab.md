# RGB Transport Closed-Loop A/B

当前状态：**实验运行中，尚未完成验收；INCONCLUSIVE（证据未齐）**。不得将本页理解为质量等价或生产配方 PASS。2026-10-06 启动，未修改模型或训练代码。

## Reproducibility

- Policy commit：`3e74a99426795071bcad2c2f5f8743c8618a5188`。已重新 fetch 并确认当时最新 main；实验使用独立 detached worktree，后续 main 变化不进入实验。
- Simulator：桌面 `dexmani_sim`，commit `c3d56fd58f402fc6f67a635b9452e10eee935afe`；源码归档、资产副本和 seed 文件均已固定。
- 硬件：同一台 RTX 4090 24 GiB，driver 580.173.02；Python 3.10.20、Torch 2.4.1+cu124、torchvision 0.19.1+cu124。BF16=true、compile=true、compile mode 沿用生产默认 reduce-overhead。
- 环境：使用现有 policy 解释器和包；policy 缺少的 simulator 依赖从现有 sim 环境解析，policy 包优先。固定相同搜索路径用于全部 runs；未安装/升级依赖。已有环境完成真实 GPU headless reset/step，尚非策略闭环结果。
- Task：`pick_apple_messy`；数据：`robot_data/pick_apple_messy.zarr`，现有 125 episodes，未生成或修改数据。
- `data_revision`：**缺失/null**，没有新增或伪造 revision。对现有 Zarr 文件记录完整只读 SHA256 清单，并在每个阶段启动及统计前复核；内容身份 `4b183673de5b243303be47ac31521718b17bd12495f9b7f569c77d0bfd61e4c9`。
- Training seeds：42、43、44。`max_train_episodes=80`。生产 recipe 的 dataset.seed 随 training.seed 改变，所以有效窗口依次为 12994、13044、13156；同一 seed 的 A/B episode indices、窗口及 sampler epoch-0 顺序逐项一致。未人为固定或扩大跨 seed 的 episode subset。
- 每组 100000 optimizer updates；B64、gradient accumulation=1；milestones 为 20000/40000/60000/80000/100000。六组均 fresh，不跨 recipe resume。
- 同一实际 200-seed runner 池以固定 shuffle seed 1066 划分：25 selection、5 tie-break、100 held-out，其余不使用。六组共用同一 manifest，test 永不包含预留 tie-break seeds。
- 正式 selection：五个 milestone，现有 fixed two-stage selector，EMA=true，NFE=10；当前 recipe 不联合搜索 NFE。成功发布后用该次不可变 `--selection-record` 交接 final eval，每组完整 100 episodes，不录 demo。
- Success 判定沿用固定 simulator 的 apple placement、clutter unmoved、hand release 和成功保持条件，runner 使用 `info.success`；未改变 randomization 或 success 定义。

本地实验目录：[完整实验产物](../experiments/rgb_transport_ab/20261006_195505/)。[实时状态](../experiments/rgb_transport_ab/20261006_195505/status.json)、[预注册协议](../experiments/rgb_transport_ab/20261006_195505/protocol.json)、[eval seed manifest](../experiments/rgb_transport_ab/20261006_195505/seed_manifest.json)、[数据身份](../experiments/rgb_transport_ab/20261006_195505/evidence/data_identity.json)、[软件版本](../experiments/rgb_transport_ab/20261006_195505/evidence/software.json)。这些大实验产物位于 Git 忽略的 experiments 下，需随研究结果另行保管。

## Controlled Difference

唯一算法/数据输入 recipe 差异为：**`dataset.rgb_keep_uint8=false vs true`**。

六组已解析完整配置并执行现有 `validate_config` 和 config-only target 检查；逐对 diff 仅为 transport 开关与独立输出目录，结果见 [机器可读 diff](../experiments/rgb_transport_ab/20261006_195505/evidence/config_diff.json) 和 [六组检查日志](../experiments/rgb_transport_ab/20261006_195505/evidence/preflight_configs.log)。当前 smoke CLI 不接收 Hydra dotlist，因此没有把 `--config-only dp training.seed=...` 当作有效命令；六组调用同一实际配置检查逻辑，另执行了原生 `smoke_test.py --config-only dp`。

训练结束还必须检查**实际保存**的 resolved configs、source hash、100000 step、EMA counter、milestones、dataset length 与最终 seed identity；当前预检查不能替代结束审计。

固定运行顺序：float42 → uint8_42 → uint8_43 → float43 → float44 → uint8_44；串行使用同一 GPU。完整训练完成后，逐 run 执行 selection → 固定 handoff → final held-out eval。任一技术错误停止队列并保留证据，不按早期 loss 淘汰 run。实验编排脚本、完整命令与日志均在实验目录。

## Training Results

| train seed | float best step | uint8 best step | float train time | uint8 train time | notes |
|---|---:|---:|---:|---:|---|
| 42 | 未产生 | 未产生 | 未完成 | 未开始 | 先启动 float32；进度以 status.json 为准 |
| 43 | 未产生 | 未产生 | 未开始 | 未开始 | 配置已验证 |
| 44 | 未产生 | 未产生 | 未开始 | 未开始 | 配置已验证 |

复用 Trainer metrics.jsonl 的 loss、gradient norm、samples/sec、milestone/checkpoint 与 EMA 状态；外部执行包装记录整次 wall time、主进程 peak RSS 和 CUDA allocated/reserved 峰值，不修改训练循环，不插入 CUDA synchronize。主进程 RSS 不是并发进程树总峰值。最终需要比较完整 loss curves、milestone/最终 loss 和 best step；loss 不是 primary metric。

## Closed-Loop Results

| train seed | float success | uint8 success | delta |
|---|---:|---:|---:|
| 42 | NOT RUN | NOT RUN | N/A |
| 43 | NOT RUN | NOT RUN | N/A |
| 44 | NOT RUN | NOT RUN | N/A |
| mean | N/A | N/A | N/A |

正式 final eval 尚未运行，不能填写成功率、Wilson 95% interval 或 training-seed SD。

## Paired Episode Results

| train seed | both pass | both fail | float-only | uint8-only |
|---|---:|---:|---:|---:|
| 42 | N/A | N/A | N/A | N/A |
| 43 | N/A | N/A | N/A | N/A |
| 44 | N/A | N/A | N/A | N/A |

预注册 hierarchical paired bootstrap：先有放回重采样 3 个 training seeds，再在每个被抽中的 training seed 内重采样 100 个 paired eval seeds；20000 replicates，分析 RNG seed=20261006，two-sided percentile 95% CI。仅 3 个 training seeds，CI 稳定性有限。当前没有 episode 数据，不计算或伪造区间；不将 p>0.05 解释为等价。

## Efficiency

**历史工程证据**：payload 73.5 → 18.375 MiB/batch；整批 H2D 约 2.95 → 0.79 ms；短时吞吐仅小幅差异。此前 benchmark 为 BF16=true、compile=false，见 [工程报告](rgb_transport_report.md)。

**本次完整训练**：BF16=true、compile=true；wall time、paired wall-time difference、实际吞吐和内存峰值尚待收集，不能引用旧短测替代。GPU 系统负载等噪声限制小幅 wall-time 差异的外推。

## Acceptance

**INCONCLUSIVE — 实验进行中，完整 multi-seed paired closed-loop evidence 尚未齐备。**

预定义 δ=0.05。PASS 至少要求平均差 ≥−5pp、hierarchical CI 下界 ≥−5pp、无明确持续 seed-wise/discordant 退化及训练异常。CI 跨越 margin 或证据不足则 INCONCLUSIVE；CI 上界仍低于−5pp、3/3 seeds 均下降超过5pp，或确认 candidate-specific 数值不稳定则 FAIL。仅满足数值门槛也需完成训练动态复核，不自动宣布生产 PASS。

本轮不改 DP 默认或 RGB 实现，不外推至其它 task、backbone、多相机。六组完整证据与人工异常复核完成后再更新正式结论。

## Preflight Notes

初始 policy 环境找不到 dexmani_sim，添加固定源码与现有 sim 依赖搜索路径后导入和 reset/step 通过。首个诊断探针缺少 saved config，修正探针后成功，未改生产代码。队列首次在训练前被 clean-source guard 拦住：数据根 symlink 被 Git 视为未跟踪文件；仅将链接移入忽略的数据目录后重新启动，原失败日志保留在 evidence/prelaunch_*，当时尚无训练 run 或 optimizer update。
