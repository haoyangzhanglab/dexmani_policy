# DexMani_Policy SSH 服务器训练部署

> 本文是 `scripts/remote/` 的**远程训练与实验同步 runbook**。它描述稳定的操作流程和 destructive-operation 边界，不维护服务器 IP、用户名、端口、硬件 inventory、网络 benchmark、当前 Policy/task 列表等基础设施快照。
>
> 当前脚本参数、远端路径和 Python executable 以 `scripts/remote/*.sh` 顶部配置区为准；如果本文与脚本冲突，以脚本为准。
>
> 训练架构见 [`项目架构.md`](./项目架构.md)，checkpoint selection / final evaluation 见 [`仿真评测机制.md`](./仿真评测机制.md)。

---

## 1. Remote Workflow Overview

```text
Local repository
      │
      ├── sync_code.sh ─────────────► Remote source tree
      │
Local data / pretrained assets
      │
      ├── sync_data.sh ─────────────► Persistent remote data root
      │                                  │
      │                                  ▼
      │                            train_remote.sh
      │                                  │
      │                                  ▼
      │                           Remote experiments
      │                                  │
      ◄──────────── sync_down.sh ─────────┘
      │
      ▼
Local checkpoint selection / evaluation
```

核心职责：

| 脚本 | 方向 | 作用 |
|---|---|---|
| `sync_code.sh` | local → remote | 高频同步源码，远端保持源码镜像 |
| `sync_data.sh` | 双向 | 同步 `robot_data/` 与 `data/` |
| `sync_down.sh` | remote → local | 拉取 experiment artifacts，同时保护本地评测产物 |
| `train_remote.sh` | remote execution | pre-flight + foreground/tmux 训练启动 |
| `tail_log.sh` | read-only | 追踪 `metrics.jsonl` |
| `stop_remote.sh` | control | SIGINT → wait → force kill fallback |

---

## 2. Prerequisites and First-time Bootstrap

### 2.1 Prerequisites

远程 workflow 假设：

1. 本机已经配置可用的 SSH alias；
2. `DEX_SERVER` 指向目标 SSH alias，或使用脚本默认 alias；
3. 远端存在脚本期望的项目目录和持久数据 root；
4. 远端存在脚本配置的 Python/Conda environment；
5. `dexmani_policy` 能在该远端环境中 import；
6. 需要远端 simulator 操作时，`dexmani_sim` 也已在对应环境中安装；
7. 训练数据最终位于 `train_remote.sh` 检查的数据 root，并能通过 repository-relative `robot_data/...` 路径访问。

推荐只在本机 `~/.ssh/config` 保存真实 HostName / User / Port，不把基础设施 identity 写进仓库文档。

```sshconfig
Host <ssh-alias>
    HostName <host>
    User <user>
    Port <port>
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

可选覆盖：

```bash
export DEX_SERVER="<ssh-alias>"
```

后续手工 SSH 示例统一可使用：

```bash
SERVER="${DEX_SERVER:-dexserver}"
```

这里的 `dexserver` 只是当前脚本默认 alias；真实 host identity 由本机 SSH config 管理。

### 2.2 本地与远端 Python 环境

本地 README 约定的研究环境与远端训练脚本使用的 Python executable 不要求同名。远端实际 executable 由 `train_remote.sh` 配置区的 `CONDA_PYTHON` 决定。

不要根据环境名推断依赖一致性；需要确认时：

1. 查看 `scripts/remote/train_remote.sh` 中的 `CONDA_PYTHON`；
2. 使用该 executable 在远端检查 import / version。

例如将脚本中的值代入 `<remote-python>` 后：

```bash
SERVER="${DEX_SERVER:-dexserver}"
python -c 'import torch, dexmani_policy; print(torch.__version__)'
ssh "$SERVER" '<remote-python> -c '\''import torch, dexmani_policy; print(torch.__version__)'\'''
```

`<remote-python>` 是占位符，不应原样执行。

### 2.3 Runtime Directory Boundary

当前 remote scripts 将三类数据分开管理：

```text
source tree
    = Python / YAML / shell / docs

persistent data
    = robot_data/ + data/

experiment artifacts
    = experiments/
```

远端通常通过 symlink 让 repository root 下的：

```text
data/
robot_data/
experiments/
```

指向持久数据盘，使代码中的相对路径不需要感知物理存储位置。

具体远端 root path 不在本文复制，查看 `sync_code.sh`、`sync_data.sh`、`sync_down.sh`、`train_remote.sh` 顶部配置区。

### 2.4 First-time Bootstrap Checklist

换新服务器、容器重建或首次初始化时，不要只运行 `train_remote.sh`；先确认 remote path contract 已建立。

推荐顺序：

```text
1. 查看 remote scripts 顶部配置
   ├── SSH alias / DEX_SERVER
   ├── remote project root
   ├── persistent data root
   └── remote Python executable

2. 在远端创建 project parent / persistent data directories

3. bash scripts/remote/sync_code.sh

4. 在远端 Python 环境执行 editable install
   <remote-python> -m pip install -e <remote-project-root>

5. 在 remote project root 建立 data / robot_data / experiments symlink
   → 指向 persistent data root 下对应目录

6. bash scripts/remote/sync_data.sh --dry-run
7. bash scripts/remote/sync_data.sh

8. 做 config-only / 最小训练验证后再启动长训练
```

为什么 symlink 是重要 contract：训练 config 通常使用 repository-relative `robot_data/<task>.zarr`，Hydra experiment 输出也使用 repository-relative `experiments/...`。remote scripts 把持久数据放在独立 data root 时，缺少这些 symlink 会导致 dataset 找不到或 experiment 写入非持久 source tree。

Bootstrap 中的真实 path 应从当前脚本读取，不在本文维护第二份硬编码副本。

---

## 3. Code Sync — `sync_code.sh`

### 3.1 Usage

```bash
bash scripts/remote/sync_code.sh
bash scripts/remote/sync_code.sh --dry-run
```

`train_remote.sh` 在正常启动前也会执行 code sync，因此手动调用主要用于：

- 独立同步代码；
- 启动前检查 rsync diff；
- 远端调试。

### 3.2 Semantics

`sync_code.sh` 使用源码镜像语义：

```text
local source tree
      │
      └── rsync --delete
              │
              ▼
remote source tree
```

生成物、缓存、数据和 experiment directory 被排除。

关键区别：

- `--delete` 用于清理**远端源码树中的 stale source files**；
- persistent `data/robot_data/experiments` 通过 exclude/protect 规则与源码镜像隔离；
- 删除语义只应该作用于 source ownership boundary。

如果修改 `sync_code.sh` 的 exclude/filter 规则，必须先执行：

```bash
bash scripts/remote/sync_code.sh --dry-run
```

确认不会触碰持久数据或实验结果。

---

## 4. Data Sync — `sync_data.sh`

`sync_data.sh` 管理训练数据和预训练/阶段性 data artifacts，而不是 experiments。

### 4.1 Usage

```bash
# 默认：local → remote
bash scripts/remote/sync_data.sh

# 只同步一类数据
bash scripts/remote/sync_data.sh robot_data
bash scripts/remote/sync_data.sh data

# 精确 checksum compare
bash scripts/remote/sync_data.sh --checksum

# remote → local
bash scripts/remote/sync_data.sh --pull

# destructive mirror
bash scripts/remote/sync_data.sh --prune
bash scripts/remote/sync_data.sh --pull --prune

# 所有 destructive 操作前先预览
bash scripts/remote/sync_data.sh --pull --prune --dry-run
```

### 4.2 Safe Default

默认同步：

```text
push: local → remote
no --delete
```

即：本地缺失某个远端文件不会导致远端删除。

注意这只是**删除安全**：普通 rsync 仍可能根据 size/mtime 更新已存在的目标文件；如果需要先确认覆盖行为，使用 `--dry-run`。

### 4.3 Pull Mode

`--pull` 反转方向：

```text
remote data
    ↓
local data
```

适合：

- 远端 Stage 1 训练产生的 codebook/checkpoint；
- 服务器生成后需要本地分析或二阶段准备的 data artifact。

默认 pull 同样不删除本地独有文件，但可能更新同名且被 rsync 判定为变化的本地文件；需要保护本地修改时先 dry-run。

### 4.4 `--prune` Is Destructive

`--prune` 打开 rsync `--delete`：

```text
push --prune
    remote-only files may be deleted

pull --prune
    local-only files may be deleted
```

任何 `--prune` 操作都应先执行同参数 `--dry-run`。

---

## 5. Experiment Pull — `sync_down.sh`

`sync_down.sh` 专门处理：

```text
remote experiments/
        ↓
local experiments/
```

而不是 dataset/pretrained data。

### 5.1 Usage

```bash
# 全部 experiments
bash scripts/remote/sync_down.sh

# 指定 policy/task 或具体 run
bash scripts/remote/sync_down.sh <policy>/<task>
bash scripts/remote/sync_down.sh <policy>/<task>/<run>

# 预览
bash scripts/remote/sync_down.sh --dry-run

# 查看服务器实验
bash scripts/remote/sync_down.sh --list

# 同时拉 W&B offline artifacts
bash scripts/remote/sync_down.sh --with-wandb <optional-subpath>
```

### 5.2 Two-pass Protection Strategy

`sync_down.sh` 的核心设计是**保护本地已有 artifact**。

#### Pass 1 — New files only

```text
remote file does not exist locally
    → download

remote file already exists locally
    → leave local copy untouched
```

这使本地生成的 evaluation/demo artifacts 不会因为之后重复 pull 而被远端覆盖。

Pass 1 不保留 partial destination，以避免下一次 `--ignore-existing` 把未完成 checkpoint 当成完整文件。

#### Pass 2 — Mutable training metadata

训练过程中少数文件会持续变化，因此第二趟只更新脚本显式 allowlist 中的 mutable entries。

具体 allowlist 以当前 `sync_down.sh` 为准，不在文档复制文件数量，避免脚本演进后形成静态 drift。

### 5.3 rsync Exit Code 24

训练运行过程中可能出现文件在 rsync 扫描后被轮换/消失。脚本将 rsync exit code 24 视为可接受的并发变化；其他 rsync error 仍然失败。

### 5.4 Offline W&B

如果训练使用 W&B offline mode，需要把对应 `wandb/` artifact 拉回本地：

```bash
bash scripts/remote/sync_down.sh --with-wandb <optional-subpath>
```

随后可按工具脚本执行：

```bash
# 先预览
bash scripts/utils/wandb_sync.sh --dry-run --all

# 同步 experiments/ 下所有 offline runs
bash scripts/utils/wandb_sync.sh --all
```

上传需要本地 W&B credential / network；`wandb_sync.sh` 使用本地受管 `policy` 环境。

---

## 6. Remote Training — `train_remote.sh`

### 6.1 Canonical Usage

```bash
bash scripts/remote/train_remote.sh <config> <task> [hydra_overrides...]
```

常用模式：

```bash
# foreground：适合 debug
bash scripts/remote/train_remote.sh --fg <config> <task> [overrides...]

# 指定 GPU visibility
bash scripts/remote/train_remote.sh --gpus 0 <config> <task> [overrides...]

# DDP
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/<config> <task> [overrides...]

# 第一次上机时同时同步数据
bash scripts/remote/train_remote.sh --sync-data <config> <task> [overrides...]

# 只预览 command + code sync
bash scripts/remote/train_remote.sh --dry-run <config> <task> [overrides...]
```

`--sync-data` 不能替代 Section 2.4 的 first-time directory/symlink bootstrap；它只是在 launch 前调用当前 `sync_data.sh`。

当前可用 config 不从本文枚举，使用：

```bash
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml
```

### 6.2 Hydra Overrides

remote script 将 `<task>` 转成 Hydra `task_name=<task>`，其余 positional arguments 按独立 Hydra overrides 转发。

例如：

```bash
bash scripts/remote/train_remote.sh --gpus 0 <config> <task> \
  'training.seed=42' \
  'training.loop.total_train_steps=1000'
```

有 shell 特殊字符、列表或空格的 override 应整体引用。

### 6.3 DDP Contract

当 config 以 `ddp/` 开头时，remote script 选择 DDP entry point。

`CUDA_VISIBLE_DEVICES` 暴露的 GPU 数与 resolved `training.num_gpus` 应一致。修改 world size 时要显式检查：

- `training.num_gpus`；
- per-rank batch size；
- effective/global batch size；
- resume contract compatibility。

不同 world size 或 loader contract 的 checkpoint 能被读出权重，并不等价于能够 strict resume。

### 6.4 Pre-flight

正常 launch 前执行 fail-fast checks，包括：

```text
SSH reachable
→ code sync
→ optional data sync
→ task dataset exists
→ GPU query / requested GPU validation
→ disk-space query
```

其中 dataset/source/SSH 失败会阻止 launch；GPU/disk status 中部分 query 属于 diagnostic，但如果能够查询 GPU 数且显式 requested GPU id 越界，launch 会失败。

### 6.5 Foreground vs Tmux

`--fg`：

```text
SSH session
   ↓
training process
```

适合短 smoke/debug；终端中断会直接影响训练进程。

默认后台模式：

```text
train_remote.sh
    ↓
detached tmux session
    ↓
training stdout/stderr → remote logs/
```

训练进程退出后 tmux session 自动结束；日志文件保留 crash traceback / exit status。

脚本启动成功后会打印实际 session name。后续 attach/stop 应使用该输出，不要在文档或外部脚本重新实现 session-name 规则。

---

## 7. Monitoring — `tail_log.sh`

### 7.1 Usage

```bash
bash scripts/remote/tail_log.sh <policy> <task>
bash scripts/remote/tail_log.sh <policy> <task> <run-timestamp>
```

逻辑：

```text
server reachable
    → tail remote metrics.jsonl

server unreachable
    → fallback to downloaded local experiment
```

它用于看 scalar metrics，不等价于查看完整 process stdout/stderr。需要 crash traceback 时使用 `train_remote.sh` 输出的 remote log path。

### 7.2 Policy Validator Note

`tail_log.sh` 当前包含显式 policy-name validator。新增 config/Policy 时，需要确认该 validator 是否同步支持新名称；不要仅因为训练入口能启动就假设 monitoring helper 一定接受新 Policy。

---

## 8. Stop and Cleanup — `stop_remote.sh`

### 8.1 Usage

```bash
# 列出远端 tmux sessions
bash scripts/remote/stop_remote.sh --list

# 停指定训练 session
bash scripts/remote/stop_remote.sh <session-name>

# 停所有由 remote trainer 创建的训练 session
bash scripts/remote/stop_remote.sh --all
```

### 8.2 Graceful Stop Protocol

```text
1. send Ctrl+C / SIGINT to tmux pane
        ↓
2. Trainer completes the current logical optimizer-step boundary
        ↓
3. if interrupted after at least one completed optimizer step:
       attempt interrupt checkpoint
        ↓
4. wait for session to exit
        ↓
5. timeout → force kill tmux
        ↓
6. best-effort GPU-memory check
```

第二次 signal 或 force kill 可能绕过完整的 graceful save。当前 Trainer 只在 interrupted 且 `global_step > 0` 时尝试 interrupt checkpoint，因此“收到 SIGINT”本身不保证一定产生 checkpoint。

`--all` 只处理 remote trainer 命名空间内的 training sessions，而不是无差别终止所有 tmux 会话。

---

## 9. Recommended Experiment Workflow

### 9.1 Normal Training Cycle

```bash
# 1. 本地修改并做低成本检查
python dexmani_policy/smoke_test.py --config-only <config>

# 2. 可选：先看 remote sync diff
bash scripts/remote/sync_code.sh --dry-run

# 3. 启动远端训练（会再次同步源码）
bash scripts/remote/train_remote.sh --gpus <ids> <config> <task> [overrides...]

# 4. 监控
bash scripts/remote/tail_log.sh <policy> <task>

# 5. 训练结束后拉取 experiment
bash scripts/remote/sync_down.sh <policy>/<task>

# 6. checkpoint selection + held-out evaluation + demo
bash scripts/eval/eval_pipeline.sh <policy> <task> <exp_name>
```

这里 `<config>` 是 Hydra config path；`<policy>` 是 experiment `policy_name` 路径。DDP overlay 可以让二者都带 `ddp/...`，最终以保存的 `config.yaml` 与实际 `experiments/...` 目录为准。

`eval_pipeline.sh --no-videos` 只关闭 Step 2 final-eval 的视频；Step 3 demo 仍然录制视频。如果需要完全不录 demo，应分步运行 selection/eval，而不是依赖该 flag。

### 9.2 Multi-seed / Ablation Runs

并行实验的原则：

- 每个 run 使用显式 `training.seed`；
- GPU partitions 不重叠；
- 不共享会被写入的 experiment directory；
- DDP world size 与 config 保持一致；
- 不依赖 run directory 名推断实验配置，最终以每个 run 保存的 `config.yaml` 为准。

### 9.3 Two-stage Artifacts

如果 Stage 1 在服务器产生 Stage 2 所需 artifact：

```text
remote Stage 1 output
      │
      ▼
sync_data.sh --pull
      │
      ▼
local inspection / preparation
      │
      ▼
sync_data.sh
      │
      ▼
remote Stage 2 training
```

先确认产物属于 `data/` ownership 还是 `experiments/` ownership，再选择 `sync_data` 或 `sync_down`，避免把两类同步职责混用。

---

## 10. Troubleshooting

### 10.1 SSH Unreachable

先检查：

```bash
SERVER="${DEX_SERVER:-dexserver}"
ssh "$SERVER" 'echo ok'
```

如果失败，remote script 无法可靠判断训练状态。不要在网络不可达时把“session 查不到”解释为训练已经停止。

### 10.2 Dataset Missing

`train_remote.sh` 会在启动前检查对应 task 的 Zarr directory。

处理顺序：

```bash
bash scripts/remote/sync_data.sh robot_data --dry-run
bash scripts/remote/sync_data.sh robot_data
```

如果任务数据命名与 Hydra `task_name` 不一致，应修 config/data contract，而不是跳过 pre-flight。

如果数据已在 persistent root 但训练仍找不到，还要检查 remote project 的 `robot_data` symlink 是否正确。

### 10.3 GPU OOM / Wrong GPU Set

检查：

```text
CUDA_VISIBLE_DEVICES
training.num_gpus
per-rank batch size
gradient accumulation
model-specific activation/token memory
```

不要只通过“减少 GPU 数”处理 DDP OOM，因为 world-size 改动同时改变 global batch / resume contract。

### 10.4 Disk Full

训练 checkpoint 和 logs 都需要写空间。当前 `train_remote.sh` 的 disk-space query 主要是 diagnostic，不包含通用的自动 free-space threshold；看到空间不足时应在启动长训练前人工处理。

### 10.5 Partial Experiment Pull

checkpoint transfer 中断后重新运行 `sync_down.sh`。其 Pass-1 设计避免把 partial checkpoint 长期当成已存在的完整 artifact。

评测前应确认目标 checkpoint 实际存在并能通过 `CheckpointStore.load()`。

### 10.6 Stale / Unknown Session

优先：

```bash
bash scripts/remote/stop_remote.sh --list
```

然后使用 `train_remote.sh` 启动时打印的 session name 操作。

不要根据旧文档中的 session naming example 猜 session ID。

---

## 11. Safety Boundaries

### Code ownership

`sync_code.sh --delete` 只应删除 remote source tree 中对应的 stale source files。

### Data ownership

`sync_data.sh --prune` 是显式 destructive mirror。任何 prune 先 dry-run。

### Experiment ownership

`sync_down.sh` 默认保护本地已存在 artifact。不要随意把它改成通用 `rsync --delete`，否则可能覆盖/删除本地 selection、eval、demo 结果。

### Process ownership

`stop_remote.sh --all` 应只处理本项目 remote trainer 创建的 session namespace。

---

## 12. Script Map

```text
scripts/remote/
├── sync_code.sh      # source mirror: local → remote
├── sync_data.sh      # data/pretrained: push / pull
├── sync_down.sh      # experiments: remote → local
├── train_remote.sh   # pre-flight + launch
├── tail_log.sh       # metrics monitor
└── stop_remote.sh    # graceful stop / cleanup
```

当脚本行为变化时，文档只维护这些**操作 contract**，不复制容易变化的：

- server identity；
- hardware inventory；
- transfer speed benchmark；
- session-name implementation；
- current Policy/task list；
- mutable-file allowlist 数量。

统一原则：

> **脚本是 executable source of truth；本文解释 ownership、safe defaults、destructive boundaries 和推荐工作流。**
