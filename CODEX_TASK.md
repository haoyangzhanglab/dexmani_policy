# CODEX_TASK — P0-2：重构 ActionFlow Inference Profiler，使延迟测量可信、轻量、可比较

> 临时 Codex 实现任务书。开始工作前必须先阅读并遵守仓库根目录 `AGENTS.md`。
> 完成实现、独立检查和验证后，**在最终变更中删除本文件**，不要把一次性任务计划长期留在仓库。
>
> 本任务按 `haoyangzhanglab/dexmani_policy@89e81d422d08b808112af23b28c3e287cff82c7c` 编写。
> 如果实际 worktree 已前进，以当前 config/code 为 source of truth；先重新 fact-check 再实施，禁止机械套 patch。

---

## 1. 任务定位

本任务只处理：

```text
dexmani_policy/tools/profile_action_flow.py
```

目标是把它变成一个可信的、B=1、steady-state、GPU-side ActionFlow inference profiler，用于研究中比较：

```text
Architecture
NFE
precision
compile / eager
parameter count
GPU latency
GPU memory
```

本任务**不是**：

```text
deployment benchmark framework
real-robot end-to-end profiler
TensorRT integration
CUDA Graph project
artifact benchmark system
W&B/CSV benchmark database
```

也与 `dexmani_real` 的 replanning task 没有实现依赖；不要修改相邻仓库。

---

## 2. Research question

P0-2 需要回答的是：

> 对同一个 ActionFlow architecture，在明确的 NFE / precision / compile setting 下，B=1 steady-state GPU inference 的主要计算段分别多慢、总共多慢、占多少显存？

最终应该可以低成本比较：

```text
baseline DiTX
vs smaller DiTX
vs GQA / attention change
vs FFN change
vs obs encoder change
vs NFE 1/2/4/...
vs eager/compile
vs fp32/bf16
```

而不是把 DataLoader、EMA copy、optimizer、H2D batch、重复 forward 等无关因素混入结果。

---

## 3. 当前脚本存在的已确认问题

实施前重新打开 `dexmani_policy/tools/profile_action_flow.py` 核对当前 HEAD。

### 3.1 `--mode infer` 仍构建 training-only objects

当前 `main()` 无论 mode 都会：

```text
build dataset + normalizer
build model + EMA model + EMA updater
build optimizer + scheduler
build training DataLoader
```

当前 ActionFlow config 设有：

```yaml
training:
  use_ema: true
```

所以仅测 inference 时 GPU 上仍可能驻留完整 EMA copy，直接污染 inference memory footprint，也做了无意义初始化。

### 3.2 Inference 先把训练 batch 整体搬到 GPU，再 slice B=1

当前逻辑类似：

```python
batch = dict_apply(next(batches), lambda x: x.to(device, non_blocking=True))
obs = {k: v[:1] for k, v in batch["obs"].items()}
```

也就是：

```text
CPU B=training_batch_size
    ↓ H2D whole batch
GPU B=training_batch_size
    ↓ slice
B=1 inference
```

虽然 H2D 不一定落在 CUDA event 内，但会污染：

```text
resident memory
peak memory
allocator/cache state
benchmark stability
```

### 3.3 一次 measurement 重复执行模型

当前每轮分别调用：

```python
model._build_cond(obs)
model.predict_action_from_cond(cond)
model.predict_action(obs)
```

因此为了得到：

```text
encoder
decoder
total
```

实际上执行了大约：

```text
condition build × 2
action generation × 2
```

而且当前 `CudaTimer.__exit__()` 每个 section 都 `torch.cuda.synchronize()`，人为切断自然 pipeline。

### 3.4 当前 infer path 没有明确 runtime precision / compile control

ActionFlow training config 当前可能启用：

```yaml
use_bfloat16: true
use_compile: true
```

但真实 deployment runtime 当前默认是 eager + `torch.inference_mode()`，并没有自动继承 training `use_compile/use_bfloat16`。

因此 profiler 不应把 training setting 默认为 deployment/inference setting。

### 3.5 当前 peak memory 同时覆盖 warmup 和 training-side allocations

现有一个笼统的：

```text
peak CUDA memory during warmup+measurement
```

无法清晰表示单模型 steady-state inference footprint。

---

## 4. 最终工具语义

`profile_action_flow.py --mode infer` 的终态必须明确是：

```text
one ActionFlow model only
one fixed real dataset observation
batch size = 1
model.eval()
torch.inference_mode()
optional bf16 autocast
optional existing compile_backbone protocol
warmup excluded from timing
one condition-build + one action-generation per measured iteration
CUDA event timing
steady-state allocated / peak allocated memory
```

### 默认 runtime

Inference profiler 默认必须是：

```text
precision = fp32
compile = false
NFE = cfg.agent.denoise_steps
```

理由：这是当前 public deployment model execution 最接近的 baseline，不应暗中继承 training recipe。

候选优化必须由用户显式打开，例如：

```bash
--precision bf16
--compile
--nfe 2
```

---

## 5. Expected edit surface

原则上只修改：

```text
MODIFY  dexmani_policy/tools/profile_action_flow.py
```

不应修改：

```text
dexmani_policy/agents/**
dexmani_policy/training/**
dexmani_policy/deployment/**
dexmani_policy/datasets/**
dexmani_policy/configs/action_flow.yaml
README.md
docs/**
```

允许复用现有 helper，例如：

```text
compile_models
count_params
set_seed
build_dataset_and_normalizer
```

但不要为 profiler 新建通用 framework/module。

如果当前 HEAD 已变化，只有在同一 measurement contract 确实要求时才扩大 edit surface，并在最终 handoff 说明。

---

# 6. Task A — 将 train / infer build path 分离

当前脚本有：

```text
--mode train|infer|both
```

本任务要求删除：

```text
both
```

最终保留：

```text
--mode infer   # 建议默认
--mode train
```

原因：

```text
train mode 需要 model + EMA + optimizer + training batch
infer mode 只应有 one model + one B=1 obs
```

如果继续支持 `both`，要么 inference memory 被 training objects 污染，要么必须写复杂 cleanup/rebuild 逻辑；都不值得。

### 6.1 `mode=train`

保留当前 training profiler 的已有职责，尽量少改。

不要趁本任务重构 trainer benchmark，也不要为 training mode 新增复杂 precision/compile semantics。

### 6.2 `mode=infer`

必须走独立 lightweight path：

```text
build dataset + normalizer
    ↓
instantiate exactly one agent
    ↓
load normalizer
    ↓
set action_key
    ↓
move model to GPU
    ↓
model.eval()
    ↓
construct one fixed B=1 observation
    ↓
optional compile
    ↓
warmup
    ↓
measurement
```

禁止在 infer path 创建：

```text
EMA model
EMA updater
optimizer
scheduler
training DataLoader workers
persistent workers
prefetch queue
```

---

# 7. Task B — Infer path 只实例化一个 Agent

推荐直接使用当前 build logic 的最小组成，而不是 `build_model_and_ema()`：

```python
import hydra

model = hydra.utils.instantiate(cfg.agent)
model.load_normalizer_from_dataset(normalizer)
model.action_key = cfg.action_key
model.to(device)
model.eval()
```

`normalizer` 仍然来自：

```python
dataset, normalizer = build_dataset_and_normalizer(cfg)
```

原因：profiling 应保持当前 dataset-derived normalization 和真实 Agent 构造方式，但不需要 EMA/optimizer。

### 参数统计必须在 compile 前完成

在任何 `torch.compile()` wrapper 之前统计：

```text
total parameters
obs_encoder parameters
action backbone parameters
```

ActionFlow action backbone 对应当前：

```python
model.action_decoder.model
```

直接复用仓库已有 `count_params()`；不要复制新的 count helper。

建议只报告总参数数目，不需要 trainable/frozen 细表。

---

# 8. Task C — 构造一个固定的真实 B=1 observation

不要使用 training DataLoader 来制造 profile input。

最简单且推荐的方式：

```python
sample = dataset[0]
obs = {
    name: value.unsqueeze(0).to(device)
    for name, value in sample["obs"].items()
}
```

当前 ActionFlow observation 应为 tensor modalities：

```text
joint_state
point_cloud
```

得到的 batch dimension 必须是：

```text
B = 1
```

### 为什么直接 `dataset[0]`

这样自动消除：

```text
training batch-size H2D
DataLoader workers
prefetch
shuffle
persistent workers
per-iteration data loading
```

`set_seed(cfg.training.seed)` 保留即可。

数据 augmentation 即使对第一次 sample 产生一次随机值，也不会改变模型 workload shape；本任务不需要为 profiler 新建 augmentation-disable mechanism。

### Model-side FPS 已经 deterministic

ActionFlow point-cloud preprocessing 在 `model.eval()` 时会通过现有 `resolve_fps_random_config(..., training=False)` 强制关闭随机 FPS。

不要另加 FPS freeze/seeding 机制。

---

# 9. Task D — 明确 inference runtime CLI

为 inference profiling 增加三个轻量参数。

### 9.1 `--nfe`

```text
--nfe N
```

默认：

```text
None -> cfg.agent.denoise_steps
```

解析后得到：

```python
nfe = cfg.agent.denoise_steps if args.nfe is None else args.nfe
```

可以在 CLI 层只检查 positive integer。

不要复制 solver-specific validity logic；ActionFlow decoder 当前 `_resolve_nfe()` 已经拥有：

```text
positive NFE
midpoint requires even NFE
```

让 model/decoder contract 继续做 source of truth。

### 9.2 `--precision`

```text
--precision {fp32,bf16}
```

默认：

```text
fp32
```

不要设计 `auto`，不要从 `cfg.training.use_bfloat16` 隐式推导。

### 9.3 `--compile`

使用简单 boolean flag：

```text
--compile
```

默认：

```text
false
```

不需要额外 `--no-compile`，因为默认就是 eager。

Compile 时直接复用：

```python
compile_models(
    model,
    None,
    mode=cfg.training.get("compile_mode", "reduce-overhead"),
)
```

不要在 profiler 重新设计 compile scope。

ActionFlow 当前 `compile_backbone()` 已经知道自己的边界：

```text
ActionFlowDiT -> compile
GeoFormer -> compile
PointNeXT/FPS ops -> eager
```

不要新增：

```text
--compile-encoder
--compile-dit
--compile-mode
```

---

# 10. Task E — Inference context 必须匹配显式 precision

测量和 warmup 都统一使用：

```python
with torch.inference_mode(), torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
    enabled=(precision == "bf16"),
):
    ...
```

FP32 时 autocast disabled。

不要使用 `torch.no_grad()` 作为最终 inference profiler context。

不要主动调用：

```python
torch.set_float32_matmul_precision("high")
```

因为当前 public deployment runtime 没有设置这一全局 policy；P0-2 默认 baseline 不应自己改变被测 runtime。

---

# 11. Task F — Warmup 与 measurement 明确分离

推荐顺序：

```text
model.eval()
parameter counting
optional compile
fixed B=1 obs
    ↓
warmup under requested precision × args.warmup
    ↓
torch.cuda.synchronize()
    ↓
record steady allocated memory
reset peak memory stats
    ↓
measurement × args.measurement
```

### Warmup

Warmup 可以调用完整：

```python
model.predict_action(obs, denoise_timesteps=nfe)
```

因为：

```text
compile graph initialization
kernel/cache warmup
ActionFlow KV-cache setup path
```

都应该在正式 measurement 前发生。

Compile time 不计入 steady-state inference latency。

默认 `--warmup 50` / `--measurement 500` 可以继续沿用当前值。

---

# 12. Task G — 每轮 measurement 只执行一次 inference pipeline

不要保留当前：

```text
_build_cond
predict_action_from_cond
predict_action again
```

最终每轮只执行：

```python
start.record()

cond, _ = model._build_cond(obs)

mid.record()

model.predict_action_from_cond(
    cond,
    denoise_timesteps=nfe,
)

end.record()

torch.cuda.synchronize()
```

只需要三个 CUDA events：

```text
start
mid
end
```

然后：

```python
condition_ms = start.elapsed_time(mid)
generate_ms = mid.elapsed_time(end)
total_ms = start.elapsed_time(end)
```

这样一次真实 segmented inference 同时得到三项数据，没有重复 forward，也没有 encoder/generator 中间 host synchronization。

### 为什么不要复用当前 `CudaTimer` nested pattern

当前 `CudaTimer.__exit__()` 会立刻：

```python
torch.cuda.synchronize()
```

如果对 encoder/generator 分别嵌套计时，会在 pipeline 中间人为同步。

Inference measurement 建议直接使用上述三 event pattern；training profiler 是否继续用现有 `CudaTimer` 不在本任务范围内。

---

# 13. Metric 命名必须准确

不要把 `_build_cond(obs)` 简单标成严格的 `obs_encoder`，因为 `_build_cond()` 实际包含：

```text
BaseAgent.preprocess
normalization
point-cloud clamp / observation reshape
obs_encoder forward
```

推荐输出：

```text
condition_build
action_generate
total
```

定义：

```text
condition_build
    = model._build_cond(obs)

action_generate
    = model.predict_action_from_cond(cond, denoise_timesteps=nfe)

total
    = condition_build + action_generate on the same measured pass
```

`action_generate` 比旧的 `decoder_denoise` 更准确，因为 ActionFlow generation 包含：

```text
initial noise creation
KV cache setup
solver / NFE model evaluations
KV cache clear
action unnormalization
```

不要宣传成“纯 DiT kernel latency”。

---

# 14. Summary statistics

可以复用当前 `_summarize()` 逻辑。

对每项：

```text
condition_build
action_generate
total
```

至少报告：

```text
mean
median (p50)
p95
```

`samples_per_sec` 对单请求 latency profiler 不是关键，可以保留已有输出，也可以删除；不要为此增加复杂统计库。

不需要：

```text
p99
histogram
confidence interval
benchmark database
```

除非当前脚本已经有且保留成本为零。

---

# 15. GPU memory measurement

Warmup 完成并 `torch.cuda.synchronize()` 后：

```python
resident_mib = torch.cuda.memory_allocated(device) / 1024**2
torch.cuda.reset_peak_memory_stats(device)
```

Measurement 完成后：

```python
peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
```

建议只报告：

```text
resident allocated MiB
peak allocated MiB
```

如果要报告差值，可以叫：

```text
peak_delta
```

**不要**把：

```text
peak - resident
```

命名为 `activation memory`，因为其中可能混有 temporary buffers / solver intermediate / compile workspace 等。

不要引入 reserved-memory/caching allocator 的复杂分析。

---

# 16. Expected output

输出建议保持纯 console、简洁可复制，例如：

```text
ActionFlow Inference Profile
────────────────────────────────
Runtime
  device            NVIDIA ...
  precision         fp32
  compile           false
  solver            midpoint
  NFE               2
  batch             1

Model
  total             75.7 M
  obs_encoder       xx.x M
  ActionDiT         xx.x M

Latency
  condition_build   mean=...  median=...  p95=...
  action_generate   mean=...  median=...  p95=...
  total             mean=...  median=...  p95=...

Memory
  resident          ... MiB
  peak              ... MiB
```

参数名 `ActionDiT` 可根据当前类名/实际 backbone 保持简洁，不需要动态做通用 profiler taxonomy。

不生成：

```text
JSONL
CSV
W&B run
plots
artifact registry
```

---

# 17. 不要加入 deployment/system latency

P0-2 只测 GPU-side Policy compute。

不要在本脚本加入：

```text
NumPy -> Torch conversion breakdown
H2D timing
D2H timing
Real IPC timing
observation assembly timing
network timing
executor timing
```

这些属于实际 deployment/runtime 的系统级 latency，不属于本工具。

因此最终结果必须在 docstring/output wording 中避免写成：

```text
real robot end-to-end inference latency
```

准确说法是：

```text
ActionFlow model-side B=1 steady-state GPU inference latency
```

---

# 18. 不要新增 runtime optimization feature

Profiler 可以**测**：

```text
bf16
compile
NFE
```

但本任务不负责把这些选项加入：

```text
dexmani_policy.deployment.load_experiment
dexmani_real runtime
PolicySpec
artifact contract
```

也不要做：

```text
TensorRT
FP8
manual CUDA Graph
attention backend flags
TRT engine export
```

这是 measurement task，不是 deployment optimization task。

---

## 19. CLI target behavior

建议最终：

```bash
python dexmani_policy/tools/profile_action_flow.py action_flow
```

默认等价于：

```text
mode=infer
precision=fp32
compile=false
NFE=config default
```

显式候选测试：

```bash
python dexmani_policy/tools/profile_action_flow.py action_flow \
    --mode infer \
    --precision bf16 \
    --compile \
    --nfe 2
```

Training profiler 仍可：

```bash
python dexmani_policy/tools/profile_action_flow.py action_flow --mode train
```

删除：

```text
--mode both
```

不要为 train/infer mode 新建 subcommand framework。

---

## 20. Validation requirements

### 20.1 开始前

必须：

```bash
git status --short
```

保护用户已有改动。

重新 inspect：

```text
dexmani_policy/tools/profile_action_flow.py
dexmani_policy/configs/action_flow.yaml
ActionFlowAgent.compile_backbone
SimpleRectifiedFlowDecoder.predict_action/_resolve_nfe
BaseDataset.__getitem__
```

只在事实仍成立时实施。

### 20.2 低成本检查

按 `AGENTS.md`，至少运行：

```bash
conda run -n policy python -m compileall -q dexmani_policy
conda run -n policy python dexmani_policy/smoke_test.py --config-only action_flow
git diff --check
git status --short
```

如果 Conda 环境不可用，使用当前可用 Python 做等价 syntax/import check，并明确说明限制。

### 20.3 CLI/parser check

至少检查 `--help` 或 parser-level behavior，确认：

```text
mode choices only train / infer
infer is default
--precision choices fp32/bf16
--compile exists
--nfe exists
```

不要为了 parser 检查触发 dataset/model build。

### 20.4 GPU/data validation（有环境才做）

如果 `policy` 环境、CUDA 和 configured ActionFlow dataset 都可用，运行一个短 inference profile：

```bash
conda run -n policy python dexmani_policy/tools/profile_action_flow.py action_flow \
    --mode infer \
    --warmup 3 \
    --measurement 10
```

验证：

```text
B=1
finite latency
condition_build + action_generate roughly equals total
resident <= peak
no EMA/optimizer allocation path
```

再根据环境可选检查：

```bash
conda run -n policy python dexmani_policy/tools/profile_action_flow.py action_flow \
    --mode infer \
    --precision bf16 \
    --compile \
    --nfe 2 \
    --warmup 3 \
    --measurement 10
```

如果 compile 第一次启动耗时较长，这是 warmup，不应算入 measured latency。

### 20.5 缺 GPU/data 时

明确报告：

```text
GPU profiler execution: NOT VERIFIED
```

不要为了让 profiler 在无数据/无 CUDA 环境通过而修改生产模型或构造 fake architecture path。

---

## 21. Code review acceptance criteria

### Build isolation

- [ ] `mode=infer` 不构建 EMA model。
- [ ] `mode=infer` 不构建 EMA updater。
- [ ] `mode=infer` 不构建 optimizer/scheduler。
- [ ] `mode=infer` 不启动 training DataLoader workers。
- [ ] `mode=train` 保留现有独立用途。
- [ ] `mode=both` 已删除。

### Input

- [ ] profile input 来自真实 configured dataset。
- [ ] batch size 从一开始就是 1。
- [ ] observation 只搬一次 GPU，并在 warmup/measurement 中复用。
- [ ] `model.eval()` 在采样/measurement 前生效。

### Runtime controls

- [ ] default precision = fp32。
- [ ] default compile = false。
- [ ] default NFE = ActionFlow config。
- [ ] `--precision bf16` 使用 autocast。
- [ ] `--compile` 复用 ActionFlow 现有 compile protocol。
- [ ] `--nfe` 传到 `predict_action_from_cond(..., denoise_timesteps=nfe)`。
- [ ] 不隐式读取 `training.use_bfloat16/use_compile` 作为 inference default。

### Timing

- [ ] 每个 measured iteration 只执行一次 `_build_cond()`。
- [ ] 每个 measured iteration 只执行一次 `predict_action_from_cond()`。
- [ ] 使用 start/mid/end 三个 CUDA events。
- [ ] encoder/generator 中间没有 host synchronize。
- [ ] warmup 不计入 measurement。
- [ ] metric 名称准确表达 measurement boundary。

### Memory

- [ ] warmup 后才 reset peak stats。
- [ ] 报告 allocated resident 和 peak。
- [ ] 不把 peak delta 错称 activation memory。
- [ ] inference memory 不被 EMA/training batch/optimizer 污染。

### Scope

- [ ] 不修改 Agent/Decoder architecture。
- [ ] 不修改 training/deployment contracts。
- [ ] 不增加 TensorRT/FP8/CUDA Graph。
- [ ] 不增加 benchmark database/framework。
- [ ] 不修改 `dexmani_real`。

---

## 22. Suggested implementation order for Codex

```text
1. Read AGENTS.md and git status.
2. Re-read current profiler + ActionFlow config + compile/inference implementation.
3. Remove `both`; make infer the default mode.
4. Split main into lightweight infer build path and existing train path.
5. Infer path: build dataset+normalizer, instantiate one model only.
6. Construct one fixed dataset[0] B=1 observation; move once to GPU.
7. Count params before compile.
8. Add --nfe / --precision / --compile.
9. Add optional existing compile_models() call.
10. Warm up full predict_action under selected precision.
11. Replace inference timers with start/mid/end events and one segmented forward.
12. Reset/report steady allocated + peak allocated memory.
13. Clean imports and remove dead infer/training shared plumbing made obsolete.
14. Run low-cost validation; run short GPU profile only if environment supports it.
15. Inspect final diff for duplicated forward, hidden EMA/optimizer path, or scope creep.
16. Delete CODEX_TASK.md in the same final change set.
```

---

## 23. Final handoff requirements

Codex 最终回复必须说明：

1. `profile_action_flow.py` 的 inference path 如何被简化；
2. infer mode 是否还会创建 EMA/optimizer/DataLoader workers；
3. 最终 CLI 的 `nfe / precision / compile` 默认与显式语义；
4. 最终 timing boundary：`condition_build / action_generate / total`；
5. 最终 memory 指标是什么；
6. 执行了哪些 validation 以及真实结果；
7. 如果未实际执行 CUDA profiler，明确标记 **NOT VERIFIED**；
8. 若当前 HEAD 与本任务假设不同，说明做了哪些必要适配。

不要把未执行的 GPU 测试写成 PASS，不要给出未经实际测量的 latency/VRAM 数字。
