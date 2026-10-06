# RGB transport 与同步优化验证

日期：2026-10-06。起始分支 `main`，HEAD `9d568112fc40d9c411499d2152902d4aa2f770f5`，工作区干净；读取根 AGENTS.md 与实际调用链后增量修改。未提交、推送、启动正式训练、仿真或真机运动，未安装依赖。历史整改证据不计入本轮通过数。

## 改动与控制变量

| 文件（包内路径省略 dexmani_policy/） | 原因与实现 |
|---|---|
| `datasets/base_dataset.py` | 保留已有 uint8 无增强 fast path、float32 默认路径；显式 keep_uint8 且有颜色增强时，仅在最终 clamp 后量化。 |
| `datasets/preprocessing.py` | keep_uint8 不再受颜色增强存在与否限制；从保存的 augmentation recipe 派生内部 `float_spatial_before_uint8`，验证/评测/Real 保留 float 插值再最终量化。无第二个用户开关。 |
| `agents/obs_encoder/rgb/utils.py`、`image_processor.py`、`agents/core/dp.py` | 默认严格 float 范围检查；DPObsEncoder 对仓库内部输入显式跳过 reductions/item。mean/std 以 `(device.type, device.index, dtype)` 缓存，无输入 batch 引用；pickle/deepcopy 丢弃 cache，Agent state_dict 不变。 |
| `configs/dp.yaml` | 显式启用 `rgb_keep_uint8: true`，保留 240 resize、224 crop、ImageAug 配方及 RGB identity。 |
| `training/build_utils.py`、`datasets/multi_task_dataset.py` | 配置层要求 RGB uint8 transport 使用 identity；MultiTask 配置层和实际 child 构造层拒绝混合 keep_uint8。MultiTask 配置本轮全部保持原默认 false。 |
| `tests/test_streaming_dataset.py`、`test_review_remediation.py`、`test_infra_evaluation.py` | 扩展已有夹具，验证三条路径、旧路径数值/RNG、共享 T-frame augmentation、验证/eval/Real recipe、范围检查、cache 和 resume 合同。 |
| `tests/benchmark_rgb_transport.py` | 独立有界 CUDA 对照与 profiler，不改 Trainer。README、仿真评测机制与整改报告同步描述并链接本记录。 |

当前 DP 数据路径：

```text
Zarr uint8 → CPU CHW float32 /255 → resize → shared random crop
→ existing ImageAug → clamp [0,1] → final round-to-uint8
→ DataLoader / pinned memory → non-blocking H2D
→ GPU float32 /255 → ImageNet normalization → visual backbone
```

resize → random crop → ImageAug 的顺序、同样本多帧共享 crop/颜色增强的调用和 RNG 消耗保持不变。只在增强完成后引入一次 `mul(255).round_().clamp_(0,255).to(torch.uint8)`，round 使用 PyTorch 的 ties-to-even；相对最终 clamped float 的量化误差约不超过 `0.5/255`（另有 float 舍入误差）。不承诺新旧 recipe 的训练结果 bit-exact。

- 未设置或 false：继续原 float32 输出；测试与独立旧路径计算逐元素精确相等，RNG state 相等。
- true、无颜色增强：继续已有 uint8 resize/crop fast path；没有改成 float 插值。
- true、有颜色增强：训练在 float 中完成所有 spatial/photometric 操作，再量化。验证/eval/Real 不执行增强，但按保存的训练 recipe 使用 float resize/center crop 再量化，避免隐含改变插值。未配置 resize 时原有 raw HWC uint8 行为保留。

没有修改 point-cloud、Zarr streaming/cache、normalizer 数学、action window、模型、训练预算、AMP 边界、Trainer `non_blocking=True` 或 DataLoader 参数。uint8 恢复为 float32，未提前转 BF16。通用 ImageProcessor 的越界 float 仍报错；跳过检查仅用于内部可信边界。

dataset config 已属于 strict resume contract，本轮未增加兼容层：旧缺省/false 与新 true 配方会被拒绝静默续训。历史实验仍应使用其保存配置；若历史曾显式配置 true＋color aug，却由旧代码实际输出 float，则当前代码会落实该 flag 的 uint8 语义，精确复现实验应保留原代码版本。

## 实际验证

解释器 `$PY=/home/zhanghaoyang/miniconda3/envs/policy/bin/python`，Python 3.10、Torch 2.4.1+cu124，已有 DINOv2-small 缓存。命令统一设置：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/zhanghaoyang/miniconda3/envs/policy/bin/python

timeout 240 "$PY" -m pytest -q tests/test_streaming_dataset.py tests/test_review_remediation.py tests/test_infra_evaluation.py tests/test_infra_resume.py tests/test_infra_training.py tests/test_policy_windows.py
timeout 90 "$PY" dexmani_policy/smoke_test.py --config-only dp
timeout 180 "$PY" -u dexmani_policy/smoke_test.py dp
```

| 验证 | 实际结果与证据 |
|---|---|
| 受影响回归 | **101 passed，8 subtests passed**；[原始日志](review_remediation_evidence/rgb_regressions.log)。两个 warning 是受控 resume 夹具缺少历史 data revision，不代表真实数据身份已验证。此前 RGB 子集 9 passed 与之重叠，不累加。初版测试曾因既有通用 normalization 校验抢先报错、Hydra struct 夹具不能添键而失败，已分别调整校验顺序和夹具后复跑。 |
| config-only | PASS，仅配置/target/import；[日志](review_remediation_evidence/rgb_config_smoke.log)。 |
| 完整 DP smoke | **真实 RTX 4090 PASS**：现有 pick_apple_messy 数据的 12,994 个训练窗口，forward/backward、一次 optimizer/scheduler/EMA 更新、预测、raw/EMA checkpoint strict restore 后预测及损坏权重拒绝；[日志](review_remediation_evidence/rgb_cuda_smoke.log)。此 smoke 不启用生产 AMP/compile/DDP。 |
| CUDA profiler/cache | 四次独立 benchmark 均完成：默认 float 检查各一次 amin/amax、两次 item/_local_scalar_dense；可信 float 路径没有这些操作。CUDA mean/std cache 复用且与直接数学计算精确相等。CPU 测试另覆盖不同 dtype/device、pickle 清空 cache、uint8 与 exact uint8/255 float 的精确相等。 |
| eval/Real recipe | 使用真实 preprocessing、build_eval_runner 与 LoadedPolicy，替换模拟器和 Agent 进行 CPU 入口回归；不是仿真 rollout 或真实机器人运行。 |
| 静态复核 | `git diff --check`、修改 Python 的 AST、文档本地链接、整改表 62 个编号唯一性、benchmark JSON 控制变量一致性均 PASS；不替代上述运行验证。 |

沙箱内 CUDA 不可见，multiprocessing Manager socket 受限；使用获准的沙箱外执行完成需要这些能力的命令。未降低测试校验要求。日志中的 DINO preset 警告来自原有 small 模型配合 dino preset，不是本轮替换了 backbone。

## 独立 CUDA benchmark

RTX 4090 24 GiB；B=64、T=2、3×224×224；workers=8、prefetch_factor=2、pin_memory=true、persistent_workers=true。同数据、DP DINOv2-small LoRA＋UNet、seed=42、sampler 顺序 hash、CPU ImageAug 参数，**BF16=true、compile=false**。两边使用相同的 spawn worker context（仅 benchmark），每进程 8 warmup＋64 measured batches，含真实 loss/backward/optimizer/scheduler/EMA；全部 loss 有限。主进程测量开始/结束同步只存在于独立脚本，不进入 Trainer。

顺序为 float32 A → uint8 A → uint8 B → float32 B，每次全新进程。四份 JSON 中 loader/model/seed/data/窗口/顺序/BF16/compile/warmup/measured 配置已对比一致，dataset 仅 keep_uint8 不同。计时排除构造、worker 启动、warmup、profiler、保存与预测，不能代表完整训练 wall time；这里 baseline 也包含本轮可信 float 和 cache 修复，故不能独立归因这些修复的加速。

| 测量 | float32 A | uint8 A | uint8 B | float32 B |
|---|---:|---:|---:|---:|
| DataLoader RGB dtype | float32 | uint8 | uint8 | float32 |
| RGB payload MiB/batch | 73.500 | 18.375 | 18.375 | 73.500 |
| 整批 H2D CUDA Event 均值 ms | 2.947 | 0.786 | 0.786 | 2.946 |
| DataLoader wait p50/p95 ms | 0.260/0.313 | 0.245/0.315 | 0.243/0.308 | 0.267/0.341 |
| 整体 samples/sec | 434.47 | 438.34 | 441.68 | 433.50 |
| 主进程 lifetime peak RSS MiB | 4093.1 | 2497.1 | 2530.7 | 4092.6 |
| 各 worker VmHWM 范围 MiB | 777.0–780.7 | 644.7–654.8 | 644.3–654.3 | 772.1–782.1 |
| 测量段 peak CUDA allocated MiB | 9235.0 | 9177.0 | 9177.0 | 9235.0 |

原始结果：[float32 A](review_remediation_evidence/rgb_float32_a.json)、[uint8 A](review_remediation_evidence/rgb_uint8_a.json)、[uint8 B](review_remediation_evidence/rgb_uint8_b.json)、[float32 B](review_remediation_evidence/rgb_float32_b.json)。RSS 来自主进程 getrusage 与 worker /proc VmHWM，各自是 lifetime high-water，不是同一时刻的进程树总峰值，不能直接求和。CUDA allocated 不等于 reserved/设备总用量。H2D 对整批 tensor 计时，包含 RGB 之外的小型状态和动作。

实际 payload 为 77,070,336 → 19,267,584 bytes，精确减少 75%；这也符合理论字节量。H2D 明显减少，两组吞吐相对差分别约 +0.9% 和 +1.9%，短测未观察到回退；没有统计置信区间，不宣称稳定训练加速，更不宣称 4× training speedup。初次 3 warmup＋12 measured 的预跑未纳入表格。

复验命令（沿用上方环境，每次指定新的 output，已有文件会被拒绝覆盖）：

```bash
timeout 180 "$PY" -u tests/benchmark_rgb_transport.py --transport float32 --warmup 8 --batches 64 --output /tmp/rgb_float32_a.json
timeout 180 "$PY" -u tests/benchmark_rgb_transport.py --transport uint8 --warmup 8 --batches 64 --output /tmp/rgb_uint8_a.json
timeout 180 "$PY" -u tests/benchmark_rgb_transport.py --transport uint8 --warmup 8 --batches 64 --output /tmp/rgb_uint8_b.json
timeout 180 "$PY" -u tests/benchmark_rgb_transport.py --transport float32 --warmup 8 --batches 64 --output /tmp/rgb_float32_b.json
```

## 剩余边界

生产 compile/AMP 组合、双卡与单 rank 故障传播、其它 RGB backbone 的真实更新、冷缓存恢复、收敛/成功率及真实闭环均 **NOT VERIFIED**；R08 保持 PARTIAL。本轮只新增 DP eager BF16 短测证据，不替代这些检查。

后续已启动同一 commit、3 paired training seeds 的完整预算实验，进度见 [Closed-loop A/B 报告](rgb_transport_closed_loop_ab.md)。这是独立的后续验证；启动或早期训练正常不等于 closed-loop 验收 PASS，以上工程证据不回填为质量结果。

GPU photometric augmentation、CUDA prefetch stream 均未实施。只有未来 profiling 明确 CPU ImageAug 成为瓶颈，再单独研究 GPU batched、per-sample 且 temporal-consistent augmentation，并重新验证 RNG 与训练效果；不在此次 transport 优化中混入该研究变量。

## 后续清理（2026-10-06）

在上述未提交改动上继续清理，未覆盖原有实现和证据：删除仅被构造参数引用的 `DEFAULT_RGB_KEEP_UINT8`，直接保留 `False` 默认值；移除 dp.yaml 中整段注释掉的备选模型及无证据的效果判断，配置解析值前后完全一致。RGB-D 已有实现与示例调用，保留接口并纠正“尚未实现”的注释；删除 DP channels-last 的固定加速百分比，将布局说明改为保持 NCHW shape 的 channels-last memory format；去掉 crop 输出必然 contiguous 的错误断言。README 与架构说明补齐三个 RGB 分支、未配置 resize 的行为和可信 float 边界。

本次实际执行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 60 $PY -m pytest -q tests/test_streaming_dataset.py tests/test_review_remediation.py tests/test_infra_evaluation.py -k 'rgb or dp_encoder'`：**9 passed、65 deselected**，一个受控夹具的历史 data revision warning；[日志](review_remediation_evidence/rgb_cleanup_tests.log)。配置值对比、Python AST、文档链接与 diff 检查通过。此子集与前述 101 项重叠，不累加；没有重跑 GPU smoke/benchmark，也没有将历史结果标为本次执行。
