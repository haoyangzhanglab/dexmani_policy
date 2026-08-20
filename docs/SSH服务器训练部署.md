# DexMani_Policy 服务器训练部署

> **服务器**: 192.168.88.230 (8×H200) | **脚本目录**: `scripts/remote/` | **更新**: 2026-08-11
>
> **文档导航**: [README](../README.md) — 项目概览 · [CLAUDE.md](../CLAUDE.md) — AI 工作速查 · [同步机制分析](同步机制分析.md) — 内部深度分析 · [仿真评测](仿真评测机制.md)

---

## 1. 快速开始

### 1.1 SSH 配置

在 `~/.ssh/config` 添加（已配置 ✅）：

```
Host dexserver
    HostName 192.168.88.230
    Port 51822
    User zjurobot
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

之后 `ssh dexserver` 即可登录。可通过 `DEX_SERVER` 环境变量覆盖别名。

### 1.2 服务器初始化（一次性）

```bash
# 1. 创建目录
ssh dexserver 'mkdir -p ~/ZHY/dexmani_policy'
ssh -t dexserver 'sudo mkdir -p /data_ssd/ZHY && sudo chown zjurobot:zjurobot /data_ssd/ZHY'
ssh dexserver 'mkdir -p /data_ssd/ZHY/{robot_data,experiments,data}'

# 2. 上传代码
bash scripts/remote/sync_code.sh

# 3. 安装项目包
ssh dexserver 'cd ~/ZHY/dexmani_policy && ~/.conda/envs/dex_policy/bin/pip install -e .'

# 4. 创建 symlink（让代码中相对路径透明访问大数据）
ssh dexserver 'cd ~/ZHY/dexmani_policy && \
    ln -sfn /data_ssd/ZHY/data data && \
    ln -sfn /data_ssd/ZHY/robot_data robot_data && \
    ln -sfn /data_ssd/ZHY/experiments experiments'

# 5. 上传数据（首次 ~6 分钟，后续增量秒级）
bash scripts/remote/sync_data.sh
```

### 1.3 验证

```bash
# 冒烟测试
ssh dexserver 'cd ~/ZHY/dexmani_policy && ~/.conda/envs/dex_policy/bin/python dexmani_policy/smoke_test.py dp3'

# 短训练（10 步，前台）
bash scripts/remote/train_remote.sh --fg dp3 pour 'training.loop.total_train_steps=10'
```

---

## 2. 架构概览

### 2.1 目录布局

```
本地机器                                服务器
────────                                ──────
dexmani_policy/                         ~/ZHY/dexmani_policy/         ← sync_code.sh (源码)
├── dexmani_policy/                     ├── dexmani_policy/
├── scripts/                            ├── scripts/
├── configs/                            ├── configs/
├── pyproject.toml                      ├── pyproject.toml
├── data/  (预训练权重)                   ├── data → /data_ssd/ZHY/data/       ← symlink
├── robot_data/  (.zarr 数据集)           ├── robot_data → /data_ssd/...       ← symlink
└── experiments/  (本地评测产物)          └── experiments → /data_ssd/...      ← symlink

                                        /data_ssd/ZHY/                ← sync_data.sh + sync_down.sh
                                        ├── data/
                                        ├── robot_data/
                                        └── experiments/
```

### 2.2 核心设计

| 决策 | 原因 |
|------|------|
| 代码放 `~/ZHY/` (home) | 轻量（~60 MB），Git 备份，容器重建成本低 |
| 大数据放 `/data_ssd/ZHY/` (NFS) | 持久化，容器重建不丢失 |
| 服务器 symlink `data` → `/data_ssd/ZHY/data` 等 | 代码中 `robot_data/pour.zarr` 等相对路径无需修改 |
| 3 个独立 sync 脚本 | 单一职责，各自按数据特性优化 flags |

### 2.3 脚本角色

| 脚本 | 方向 | 传输内容 | 频率 | 触发 |
|------|------|---------|------|------|
| `sync_code.sh` | 本地→服务器 | 源码 | 高（每次改代码） | 手动 / train_remote 自动 |
| `sync_data.sh` | **双向** | robot_data/ + data/ | 低（新增数据/二阶段回传） | 手动 |
| `sync_down.sh` | 服务器→本地 | experiments/ | 中（训练后） | 手动 |
| `train_remote.sh` | — | 启动训练 | 每次训练 | 手动 |
| `tail_log.sh` | — | 实时日志流 | 训练中 | 手动 |
| `stop_remote.sh` | — | 停止训练 | 训练中 | 手动 |

---

## 3. 同步机制

### 3.1 设计哲学

| 原则 | 体现 |
|------|------|
| **单一职责** | 3 个脚本各自处理一类数据（源码/数据集/实验产物） |
| **安全默认** | 数据同步不加 `--delete`（防误删）；pull 不加 `--prune`（防覆盖本地） |
| **按数据特性优化** | 源码 `-z` 压缩（文本 3-4x）；数据免压缩（.zarr 已内置）；checkpoint 利用 immutability |

### 3.2 sync_code.sh — 源码上传

```bash
bash scripts/remote/sync_code.sh              # 同步（增量，2-3 秒）
bash scripts/remote/sync_code.sh --dry-run    # 预览变更
```

**rsync flags**: `-avz --partial --progress --delete`

| Flag | 作用 | 设计理由 |
|------|------|---------|
| `-a` | 归档模式（保留权限、时间戳） | 依赖 mtime 做增量检测 |
| `-z` | 压缩传输 | `.py`/`.yaml` 压缩比 3-4x |
| `--partial` | 断点续传 | 中断后从断点继续 |
| `--delete` | 删除远端残留 | 本地删文件 → 服务器同步删除 |

**排除项**:
- **递归排除**（任意深度）: `.git/`, `__pycache__/`, `*.pyc`, `*.pyo`, `*.egg-info`, `.DS_Store`
- **根锚定排除**（仅项目根目录）: `/data`, `/robot_data`, `/experiments`, `/wandb`, `/outputs`, `/logs` 等

**`--delete` 安全保护**: 3 个 `protect` filter 确保 `--delete` 不会删除服务器独有的 symlink 目录：

```
--filter='protect /data'
--filter='protect /robot_data'
--filter='protect /experiments'
```

**触发时机**: 每次 `train_remote.sh` 启动时必定自动调用（pre-flight 步骤 2）。也可手动执行。

### 3.3 sync_data.sh — 数据集双向同步

```bash
bash scripts/remote/sync_data.sh                  # push: 本地→服务器（默认）
bash scripts/remote/sync_data.sh --prune          # push + 删除服务器独有文件
bash scripts/remote/sync_data.sh --pull           # pull: 服务器→本地（安全，不删本地文件）
bash scripts/remote/sync_data.sh --pull --prune   # pull + 删除本地独有文件
bash scripts/remote/sync_data.sh --dry-run        # 预览
bash scripts/remote/sync_data.sh -c               # checksum 模式（精确但慢）
```

**4 种模式**:

| 模式 | 方向 | --delete | 安全性 | 用途 |
|------|------|----------|--------|------|
| （默认） | local → server | 否 | 安全 | 上传新数据 |
| `--prune` | local → server | 是 | 需确认 | 上传 + 清理服务器冗余 |
| `--pull` | server → local | 否 | **安全** | 下载服务器独有的数据 |
| `--pull --prune` | server → local | 是 | 需确认 | 完整镜像服务器数据 |

**rsync flags**: `-av --partial --progress`（无 `-z` — `.zarr` 和 `.safetensors` 已内置压缩；无 `--delete` — 安全默认）。

**与 sync_down.sh 的分工**:

| | sync_data.sh | sync_down.sh |
|------|------|------|
| 同步内容 | `data/` + `robot_data/`（数据集、预训练权重） | `experiments/`（训练产物） |
| 方向 | 双向（push 为主，pull 为辅） | 仅下载 |
| 策略 | 单趟 rsync | 两趟 rsync（存在性保护） |

**典型 Pull 场景 — DQ-RISE 二阶段训练**:

```bash
# 1. 服务器完成 Stage 1（VQ-VAE 预训练），产出 codebook/checkpoint 到 /data_ssd/ZHY/data/

# 2. 拉回本地
bash scripts/remote/sync_data.sh --pull --dry-run   # 预览
bash scripts/remote/sync_data.sh --pull              # 下载

# 3. 本地准备 Stage 2（DQ-RISE agent），推回服务器
bash scripts/remote/sync_data.sh                     # 推本地新增文件
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/dqrise pour
```

> Pull 模式下本地目录不存在时会自动 `mkdir -p`。始终先 `--dry-run` 预览。

### 3.4 sync_down.sh — 实验结果下载

```bash
bash scripts/remote/sync_down.sh                          # 全部实验
bash scripts/remote/sync_down.sh dp3/pour                 # 特定 policy/task
bash scripts/remote/sync_down.sh dp3/pour/2026-08-03_12   # 特定 run
bash scripts/remote/sync_down.sh --dry-run                # 预览
bash scripts/remote/sync_down.sh --list                   # 列出服务器实验
bash scripts/remote/sync_down.sh --with-wandb             # 含 wandb 离线数据
```

#### 两趟 rsync 策略

这是下载链路的核心设计 — 不依赖文件名或目录名来判断哪些是本地评测产物，而是基于**文件是否已存在**来决策：

**Pass 1 — 只拉新文件** (`-av --ignore-existing`)

| 本地状态 | 行为 | 效果 |
|----------|------|------|
| 文件**不存在** | 下载 | 新 checkpoint、新实验 run |
| 文件**已存在** | **跳过** | 保护所有本地文件不被覆盖 |

**Pass 2 — 更新可变文件** (`-av --existing` + 文件过滤)

Pass 2 仅更新 3 种训练中持续变化的文件：

| 文件 | 为何需要更新 | 大小 |
|------|-------------|------|
| `metrics.jsonl` | 训练中每步追加 | ~KB |
| `checkpoints/latest.pt` | symlink 目标随训练推进变化 | ~几十字节 |
| `checkpoints/scores.json` | top-k tracker 更新 | ~KB |

#### 为什么比基于名称排除更健壮

| | 基于名称排除（脆弱） | 基于存在性保护（当前设计） |
|------|------|------|
| 需要预知本地文件 | 是 — 必须硬编码排除列表 | **否** — 任何本地已有文件自动跳过 |
| 未来评测代码改变输出路径 | 需同步更新 sync 脚本 | **无需修改** |
| 误删本地评测产物 | 可能（exclude 与 delete 不匹配时） | **从不** |

> Checkpoint `.pt` 文件是 immutable 的（写一次不改），`--ignore-existing` 完美处理：新的下载，已有的跳过。本地评测产物（`eval_dexsim/`、`demo_videos/`、`best_ckpt.json` 或任何未来新目录）只要存在就被 Pass 1 自动保护。

退出码 24（"some files vanished during transfer"）被捕获为良性 — 发生在训练运行中 checkpoint 被轮换时。

### 3.5 路径映射

```
本地                                    服务器
────                                    ────
experiments/                            /data_ssd/ZHY/experiments/
robot_data/                             /data_ssd/ZHY/robot_data/
data/                                   /data_ssd/ZHY/data/
dexmani_policy/                         ~/ZHY/dexmani_policy/dexmani_policy/
scripts/                                ~/ZHY/dexmani_policy/scripts/
```

所有远程脚本通过 `DEX_SERVER` 环境变量（默认 `dexserver`）定位服务器。路径常量定义在各脚本顶部配置区。

### 3.6 同步时机

| 事件 | sync_code | sync_data | sync_down |
|------|-----------|-----------|-----------|
| `train_remote.sh` 每次调用 | **自动** | 可选（`--sync-data`） | — |
| 改代码后 | 手动 | — | — |
| 新增数据集 | — | 手动 (push) | — |
| 训练完成后 | — | — | 手动 |
| 二阶段产物回传 | — | 手动 (`--pull`) | — |
| tail_log / stop | — | — | — |

---

## 4. 训练控制

### 4.1 train_remote.sh — 一键启动

```bash
bash scripts/remote/train_remote.sh <config> <task> [hydra_overrides...]

# 常用选项
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 <config> <task> [...]   # 指定 GPU
bash scripts/remote/train_remote.sh --fg <config> <task> [...]             # 前台（调试用）
bash scripts/remote/train_remote.sh --sync-data <config> <task>            # 含数据上传
bash scripts/remote/train_remote.sh --dry-run <config> <task>              # 预览
```

**Pre-flight checks（任一失败则退出）**:

| 步骤 | 检查 | 阻塞 |
|------|------|------|
| 1. SSH 可达 | `ssh -o ConnectTimeout=5` | 是 |
| 2. 代码同步 | 自动调用 `sync_code.sh` | 是 |
| 2b. | 数据同步（仅 `--sync-data` 时） | 是 |
| 3. Dataset 存在 | `{task}.zarr` 在服务器上 | 是 |
| 4. GPU 状态 | nvidia-smi 查询（仅打印） | 否 |
| 5. 磁盘空间 | `/data_ssd` df -h（仅打印） | 否 |

**Session 命名规则**: `config_task` (如 `dp3_pour`)，指定 seed 时追加 `_s<seed>` (如 `dp3_pour_s42`)，确保同 config+task 不同 seed 可并行。

**单卡 vs DDP**:

```bash
# 单卡
bash scripts/remote/train_remote.sh dp3 pour
bash scripts/remote/train_remote.sh --gpus 0 maniflow pour 'training.seed=42'

# DDP 多卡
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour 'training.seed=99'
```

支持的全部 config 名见 [CLAUDE.md](../CLAUDE.md#命令速查)。

### 4.2 tail_log.sh — 实时日志

```bash
bash scripts/remote/tail_log.sh <policy> <task>              # 自动找最新 run
bash scripts/remote/tail_log.sh <policy> <task> <timestamp>  # 指定 run
```

自动尝试服务器 `tail -f` → 不可达时回退本地文件。Ctrl+C 退出。

### 4.3 stop_remote.sh — 优雅停止

```bash
bash scripts/remote/stop_remote.sh <session>    # 停止指定 session
bash scripts/remote/stop_remote.sh --all        # 停止所有训练 session
bash scripts/remote/stop_remote.sh --list       # 查看活跃 session
```

**三阶段停止流程**:

| 阶段 | 操作 | 超时 |
|------|------|------|
| 1. SIGINT | `tmux send-keys C-c` → 训练代码捕获信号，优雅保存 checkpoint | — |
| 2. 轮询等待 | 每 2s 检查 `tmux has-session` | 30s |
| 3. Force kill | `tmux kill-session`（SIGHUP→SIGKILL） | — |
| 4. GPU 验证 | `nvidia-smi --query-compute-apps` 检查显存释放 | — |

`--all` 模式只停匹配 `*_*` 命名的 session（训练命名规则），不会误杀其他 tmux 会话。

---

## 5. 日常工作流

### 5.1 典型训练周期

```bash
# 1. 改代码 → 同步（2-3 秒）
bash scripts/remote/sync_code.sh

# 2. 启动训练（自动再同步一次代码 + pre-flight checks）
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour

# 3. 监控
bash scripts/remote/tail_log.sh maniflow pour        # Ctrl+C 退出

# 4. 训练完成 → 拉取结果
bash scripts/remote/sync_down.sh maniflow/pour

# 5. 本地评测
bash scripts/eval/eval_pipeline.sh maniflow pour experiments/maniflow/pour/<timestamp>

# 6. (可选) 录制 demo
bash scripts/eval/record_demo.sh maniflow pour experiments/maniflow/pour/<timestamp>
```

### 5.2 多实验并行

```bash
# 同步一次代码，启动多个训练
bash scripts/remote/sync_code.sh

# 两个 4 卡 DDP 并行（不同 GPU 分区，不同 seed）
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour 'training.seed=42' &
bash scripts/remote/train_remote.sh --gpus 4,5,6,7 ddp/sat pour 'training.seed=42' &
wait

# 8 个单卡 seed sweep
for seed in 0 1 2 3 4 5 6 7; do
    bash scripts/remote/train_remote.sh --gpus $seed dp3 pour "training.seed=$seed" &
done
```

### 5.3 二阶段训练（DQ-RISE）

```bash
# Stage 1: 服务器产 VQ-VAE 权重到 /data_ssd/ZHY/data/

# Stage 2 准备: 拉取产物
bash scripts/remote/sync_data.sh --pull --dry-run     # 预览
bash scripts/remote/sync_data.sh --pull                # 下载

# 本地准备 Stage 2 代码 → 推回 → 训练
bash scripts/remote/sync_code.sh
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/dqrise pour
```

### 5.4 紧急操作

```bash
# 立即停止所有训练
bash scripts/remote/stop_remote.sh --all

# 检查 GPU 是否清理干净
ssh dexserver 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'

# 查看磁盘空间
ssh dexserver 'df -h /data_ssd'

# 强制杀 session（跳过 30s 优雅等待）
ssh dexserver 'tmux kill-session -t dp3_pour'
```

---

## 6. 参考手册

### 6.1 服务器硬件

| 资源 | 规格 |
|------|------|
| GPU | 8× NVIDIA H200 SXM (141 GB × 8 = 1.125 TB), NVSwitch 全互联 |
| CPU | 2× Xeon Platinum 8558 (96C/192T) |
| RAM | 2.0 TiB DDR5 |
| 数据盘 (NFS) | `/data_ssd`: 35 TB | NAS: 96 TB |
| 网络 | 千兆 LAN, 上传 ~25 MB/s / 下载 ~32 MB/s |
| OS | Ubuntu 22.04 (Docker 容器) |
| CUDA | Driver 565.57.01 / CUDA 12.7 |

### 6.2 传输速率

> 实测: zhy-MS-7E06 ↔ 192.168.88.230，千兆 LAN，2026-08-06。

| 方向 | 文件 | 耗时 | 速率 |
|------|------|------|------|
| ⬆️ 上传 | 100 MB / 1 GB | 3.6s / 40.4s | 27.6 / 25.4 MB/s |
| ⬇️ 下载 | 100 MB / 1 GB | 3.2s / 31.2s | 31.3 / 32.9 MB/s |

**实际场景预估**（基于 25 MB/s 保守值）:

| 操作 | 数据量 | 耗时 |
|------|--------|------|
| sync_code | ~60 MB | 2-3 s |
| sync_data 首次 | ~8.5 GB | ~6 min |
| sync_data 增量（无变化） | 0 | <1 s |
| sync_down 首次 | ~7-10 GB | ~5-7 min |
| sync_down 增量（无新 ckpt） | ~500 KB | <1 s |

### 6.3 脚本清单

```
scripts/remote/
├── sync_code.sh        # 上传源码（频繁，2-3s）
├── sync_data.sh        # 双向数据同步（增量，秒级）
├── sync_down.sh        # 下载实验（两趟 rsync，保护本地）
├── train_remote.sh     # 一键训练（pre-flight checks + tmux）
├── tail_log.sh         # 实时日志流（远程 + 本地 fallback）
└── stop_remote.sh      # 优雅停止（SIGINT → poll → kill → GPU 验证）
```

### 6.4 常用运维命令

```bash
# GPU 状态
ssh dexserver 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'

# 磁盘
ssh dexserver 'df -h /data_ssd /home'

# tmux 会话
ssh dexserver 'tmux list-sessions'                      # 所有会话
bash scripts/remote/stop_remote.sh --list                # 训练会话

# 训练进程
ssh dexserver 'ps aux | grep train'

# 查看实验列表
bash scripts/remote/sync_down.sh --list

# 找大文件（清理磁盘用）
ssh dexserver 'find /data_ssd/ZHY/experiments -type f -size +1G -exec ls -lh {} \;'
```

### 6.5 Wandb

```bash
# 在线模式（需服务器可访问 wandb.ai）
bash scripts/remote/train_remote.sh dp3 pour 'workspace.wandb_cfg.mode=online'

# 离线模式（默认）→ 事后同步
bash scripts/remote/sync_down.sh --with-wandb dp3/pour
bash scripts/utils/wandb_sync.sh
```

---

## 附录: SSH 终端操作常识

> 面向不熟悉 Linux 终端的同学。已熟悉的可以跳过。

### A.1 登录与退出

```bash
ssh dexserver          # 登录（需先配置 ~/.ssh/config）
exit                   # 退出（或 Ctrl+D）
```

### A.2 目录与文件

```bash
mkdir -p path/to/dir           # 递归创建目录
ls -la                         # 详细列表
cd ~/ZHY/dexmani_policy        # 进入项目目录
pwd                            # 显示当前路径

cp -r src/ dst/                 # 复制
mv old new                      # 移动/重命名
rm -rf dir/                     # 删除（⚠️ 无回收站，确认后再执行）

cat file.txt                    # 查看内容
less file.txt                   # 分页查看（q 退出）
head -20 file.txt               # 前 20 行
tail -f metrics.jsonl           # 实时追踪（Ctrl+C 退出）
```

### A.3 Tmux 速查

```bash
tmux new -s name                # 创建会话
tmux attach -t name             # 重新连接
tmux ls                         # 列出会话
tmux kill-session -t name       # 终止会话

# 在 tmux 内部（前缀键 Ctrl+B）
Ctrl+B D    # 断开（训练继续运行）
Ctrl+B [    # 滚动模式（PgUp/PgDn 翻页，q 退出）
```

### A.4 训练命令结构

```
train.py <策略> <任务> [Hydra覆盖参数...]
```

**策略（policy）** — 可用值:

| 策略 | 单卡写法 | DDP 写法 |
|------|---------|---------|
| DP3 | `dp3` | — |
| DP3 + FAAS | `dp3_faas` | `ddp/dp3_faas` |
| ManiFlow | `maniflow` | `ddp/maniflow` |
| MultiTask | `multitask_dit` | `ddp/multitask_dit` |
| R3D | `r3d` | `ddp/r3d` |
| DQ-RISE | `dqrise` | `ddp/dqrise` |
| SAT | `sat` | `ddp/sat` |
| DP | `dp` | `ddp/dp` |
| MoE DP | `moe_dp` | — |

**任务（task）** — 当前可用: `pour`（倒水）

**常用 Hydra 覆盖参数**:

```bash
training.seed=42                           # 随机种子
training.loop.total_train_steps=50000      # 训练步数（默认 100000）
action_key=action_ee                       # EE 动作空间（默认 action）
workspace.wandb_cfg.mode=online            # Wandb 在线模式
```

**示例**:

```bash
# 本地（原有脚本）
bash scripts/training/train.sh dp3 pour

# 远程（train_remote.sh）
bash scripts/remote/train_remote.sh --gpus 0 dp3 pour 'training.seed=42'
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour

# 直接 SSH（调试用）
ssh dexserver 'cd ~/ZHY/dexmani_policy && export DATA_DIR=/data_ssd/ZHY && \
    ~/.conda/envs/dex_policy/bin/python dexmani_policy/train.py --config-name=dp3 task_name=pour'
```

> `train_remote.sh` 已自动在 tmux 里启动，SSH 断开训练继续。直接 `python train.py` 需要用 `tmux` 或 `nohup` 保护。

### A.5 进程与监控

```bash
ps aux | grep python             # 查看 Python 进程
htop                             # 进程监控（或 top）

nvidia-smi                       # GPU 状态
watch -n 1 nvidia-smi            # 每秒刷新

df -h /data_ssd                  # 磁盘空间
du -sh experiments/              # 目录大小
```

### A.6 网络传输

```bash
# scp（单文件/目录）
scp -P 51822 local_file zjurobot@192.168.88.230:~/ZHY/      # 上传
scp -P 51822 zjurobot@192.168.88.230:~/ZHY/file ./          # 下载

# rsync（增量，更高效）
rsync -avz local_dir/ dexserver:~/ZHY/dir/                   # 上传
rsync -avz dexserver:~/ZHY/dir/ local_dir/                   # 下载
# --dry-run: 预览  -z: 压缩  --delete: 删目标端多余文件
```
