# DexMani_Policy 服务器训练部署方案（v4 重构版）

> **更新日期**: 2026-08-06 | **服务器**: 192.168.88.230 (8×H200) | **设计原则**: 清晰分离 × 高效同步 × 易于维护
>
> **文档导航**: [README](../README.md) — 项目概览 · [CLAUDE.md](../CLAUDE.md) — AI 工作速查 · [项目架构](项目架构.md) — 架构全景 · [评测机制](评测机制.md) — 评测全链路

---

## 设计原则

| # | 原则 | 实现 |
|---|------|------|
| 1 | **数据/代码分离** | `robot_data` + `experiments` → NFS 数据盘；源码 → home overlay |
| 2 | **三向同步** | 代码上传（频繁）、数据上传（增量）、实验下载（保护本地评测产物） |
| 3 | **稳定高速** | rsync 默认 size+mtime 去重，源码 `-z` 压缩，数据免压缩 |
| 4 | **清晰可靠** | 每个脚本单一职责，`--dry-run` 预览，pre-flight checks 快速失败 |

---

## 服务器硬件速查

| 资源 | 规格 |
|------|------|
| CPU | 2× Intel Xeon Platinum 8558 (Emerald Rapids), 96C/192T |
| GPU | **8× NVIDIA H200 SXM**, 141 GB × 8 = 1.125 TB 总显存, NVSwitch 全互联 |
| RAM | 2.0 TiB DDR5 |
| 系统盘 | Docker overlay (`/`): 11 TB (1.6 TB 可用) |
| 数据盘 (NFS) | `192.168.88.10:/data_ssd`: 35 TB (20 TB 可用) |
| NAS (NFS) | `192.168.88.17:/nasPublic`: 96 TB (59 TB 可用) |
| 网络 | 千兆 LAN，实测上传 ~25 MB/s / 下载 ~32 MB/s |
| OS | Ubuntu 22.04.5 LTS (Docker 容器) |
| CUDA | Driver 565.57.01 / CUDA 12.7 / toolkit 12.1 |

> **Docker 注意**: `/home/` 在 overlay FS 上，容器重建可能丢失。大数据（robot_data、experiments、pretrained weights）**必须**放 `/data_ssd/ZHY/`（NFS 持久化）。代码和 conda 环境放 `/home/` 即可（Git 备份，重建成本低）。

---

## 目录布局

### 设计逻辑

```
本地机器                                服务器
────────                                ──────
dexmani_policy/                         ~/ZHY/dexmani_policy/         ← sync_code.sh (源码)
├── dexmani_policy/                     ├── dexmani_policy/           ← rsync -avz --delete
├── scripts/                            ├── scripts/
├── configs/                            ├── configs/
├── docs/                               ├── docs/
├── pyproject.toml                      ├── pyproject.toml
├── data/  (53 MB, Uni3D 权重)          ├── data → /data_ssd/ZHY/data/    ← symlink
├── robot_data/  (8.5 GB, .zarr)        ├── robot_data → /data_ssd/...    ← symlink
└── experiments/  (本地评测产物)         └── experiments → /data_ssd/...   ← symlink
                                         
                                        /data_ssd/ZHY/                ← sync_data.sh (数据集+权重)
                                        ├── data/                     ← rsync -av (免压缩, 不删)
                                        ├── robot_data/               ← rsync -av (增量, 免重复上传)
                                        └── experiments/              ← sync_down.sh (下拉)
                                            └── <policy>/<task>/<ts>/
```

### 目录职责

| 目录 | 存储位置 | 大小 | 同步方向 | 频率 |
|------|---------|------|---------|------|
| 源码 (`dexmani_policy/`, `scripts/`, `configs/`, ...) | `~/ZHY/dexmani_policy/` (home) | ~60 MB | 本地→服务器 | **频繁**（每次改代码） |
| 预训练权重 (`data/`) | `/data_ssd/ZHY/data/` (NFS) | ~53 MB | 本地→服务器 | 极少（权重不变） |
| 数据集 (`robot_data/`) | `/data_ssd/ZHY/robot_data/` (NFS) | ~8.5 GB | 本地→服务器 | 偶尔（新增任务） |
| 实验输出 (`experiments/`) | `/data_ssd/ZHY/experiments/` (NFS) | ~GB 级 | 服务器→本地 | 训练后 |

> **关键**: 服务器上 `~/ZHY/dexmani_policy/` 下的 `data`、`robot_data`、`experiments` 是 **symlink**，指向 `/data_ssd/ZHY/` 下的实际目录。这样代码中 `robot_data/pour.zarr` 等相对路径无需任何修改即可工作。

---

## Phase 0: SSH 配置

**状态**: SSH 密钥已配置 ✅

在本地 `~/.ssh/config` 添加便捷别名：

```
Host dexserver
    HostName 192.168.88.230
    Port 51822
    User zjurobot
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

之后所有脚本中 `dexserver` 等价于 `ssh zjurobot@192.168.88.230 -p 51822`。

> 可通过环境变量 `DEX_SERVER` 覆盖服务器别名：`DEX_SERVER=myhost bash scripts/remote/sync_code.sh`

---

## Phase 1: 服务器初始化（一次性）

### 1.1 创建目录结构

```bash
# 代码目录（home，个人 workspace）
ssh dexserver 'mkdir -p ~/ZHY/dexmani_policy'

# 数据目录（/data_ssd NFS，持久化）
# 首次需 sudo 创建（zjurobot 在 sudo 组）
ssh -t dexserver 'sudo mkdir -p /data_ssd/ZHY && sudo chown zjurobot:zjurobot /data_ssd/ZHY'
ssh dexserver 'mkdir -p /data_ssd/ZHY/{robot_data,experiments,data}'
```

### 1.2 补装缺失的包

```bash
# dex_policy 环境（已确认存在，Python 3.10.19, torch 2.4.1+cu124）
# 缺少 zarr 和 dexmani_policy 本身
ssh dexserver '~/.conda/envs/dex_policy/bin/pip install zarr'
# 等代码首次上传后再执行:
# ssh dexserver 'cd ~/ZHY/dexmani_policy && ~/.conda/envs/dex_policy/bin/pip install -e .'
```

### 1.3 首次上传

```bash
# 1. 上传代码
bash scripts/remote/sync_code.sh

# 2. 安装包
ssh dexserver 'cd ~/ZHY/dexmani_policy && ~/.conda/envs/dex_policy/bin/pip install -e .'

# 3. 创建 symlink（代码透明访问大数据）
ssh dexserver 'cd ~/ZHY/dexmani_policy && \
    ln -sfn /data_ssd/ZHY/data data && \
    ln -sfn /data_ssd/ZHY/robot_data robot_data && \
    ln -sfn /data_ssd/ZHY/experiments experiments'

# 4. 上传数据（8.5 GB，约 6 分钟，后续增量秒级）
bash scripts/remote/sync_data.sh
```

### 1.4 验证链路

```bash
# 冒烟测试
ssh dexserver 'cd ~/ZHY/dexmani_policy && ~/.conda/envs/dex_policy/bin/python dexmani_policy/smoke_test.py dp3'

# 短训练（10 步，前台观察）
bash scripts/remote/train_remote.sh --fg dp3 pour 'training.loop.total_train_steps=10'
```

---

## 同步架构（核心设计）

### 三向同步全景

```
┌─────────────────────────────────────────────────────────────────┐
│                         sync_code.sh                            │
│  本地源码 ──── rsync -avz --delete ────→ ~/ZHY/dexmani_policy/  │
│  频率: 高（每次改代码）  耗时: ~2-3秒   方向: 本地→服务器       │
├─────────────────────────────────────────────────────────────────┤
│                         sync_data.sh                            │
│  本地 robot_data/ + data/ ── rsync -av ──→ /data_ssd/ZHY/      │
│  频率: 低（新增数据时）  耗时: 增量秒级  方向: 本地→服务器       │
├─────────────────────────────────────────────────────────────────┤
│                        sync_down.sh                             │
│  /data_ssd/ZHY/experiments/ ── rsync -av ──→ 本地 experiments/  │
│  频率: 中（训练后评测前） 耗时: 增量秒级  方向: 服务器→本地      │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么这样设计

| 决策 | 理由 |
|------|------|
| 源码用 `-z` 压缩 | .py/.yaml/.sh 文本压缩比 3-4x，千兆网下省时间 |
| 数据不用 `-z` | .zarr 和 .safetensors 已内置压缩，再压缩浪费 CPU |
| 源码用 `--delete` | 本地删文件 → 服务器也删，保持一致 |
| 数据不用 `--delete` | 安全：本地误删不影响服务器数据 |
| rsync 默认 size+mtime 比较 | 文件未变 → 秒级跳过；文件变了 → 只传差异 |
| `data/` 放 NFS 不随源码同步 | 53 MB 权重文件极少变动，与 60 MB 源码分开传输 |

### 实验下载的本地保护机制（两趟 rsync）

`sync_down.sh` 不依赖文件/目录名来判断哪些是本地评测产物，而是用**两趟 rsync**，基于文件**是否存在**来决策：

**Pass 1 — `rsync --ignore-existing`**（只拉新文件）
- 本地**没有**的文件 → 下载（新的 checkpoint、新的实验 run）
- 本地**已有**的文件 → **跳过**（不论它叫什么名字、放在哪个目录）

**Pass 2 — `rsync --existing`（精确更新可变文件，仅更新已有文件）**
- `metrics.jsonl` — 训练中持续增长
- `checkpoints/latest.pt` — symlink 目标随训练推进变化
- `checkpoints/scores.json` — top-k tracker 更新

> **为什么这样更健壮**：
> - 本地评测产物（`eval_dexsim/`、`demo_videos/`、`best_ckpt.json` 或未来任何新目录）只要存在于本地，就会被 Pass 1 的 `--ignore-existing` 自动跳过。**不需要预先知道它们的名字**。
> - Checkpoint `.pt` 文件是 immutable 的（写一次不改），`--ignore-existing` 完美处理：新的会下载，已有的跳过。
> - 如果未来评测代码改了输出路径，sync 脚本不需要同步修改。

---

## 脚本参考

### `sync_code.sh` — 上传源码

```bash
bash scripts/remote/sync_code.sh              # 同步代码（增量，2-3秒）
bash scripts/remote/sync_code.sh --dry-run    # 预览变更
```

**传输内容**: `dexmani_policy/`, `scripts/`, `configs/`, `docs/`, `pyproject.toml`, `setup.py`, `requirements*.txt` 等所有非数据源码。

**排除**: `.git/`, `__pycache__/`, `data/`, `robot_data/`, `experiments/`, `wandb/`, `outputs/`, `logs/` 等生成/数据目录。

**rsync flags**: `-avz --delete --partial` — 压缩传输 + 删远端残留 + 断点续传。

---

### `sync_data.sh` — 上传数据

```bash
bash scripts/remote/sync_data.sh              # 上传全部数据（robot_data + data）
bash scripts/remote/sync_data.sh robot_data   # 仅上传数据集
bash scripts/remote/sync_data.sh data         # 仅上传预训练权重
bash scripts/remote/sync_data.sh --dry-run    # 预览
```

**传输内容**: `robot_data/` → `/data_ssd/ZHY/robot_data/`, `data/` → `/data_ssd/ZHY/data/`。

**关键行为**:
- 文件未变（同 size + mtime）→ **秒级跳过**
- 文件新增或变更 → 仅传差异
- 不含 `--delete`：服务器数据安全优先

> **首次上传 8.5 GB 约 6 分钟**。后续增量上传仅传输新增/变更的文件，通常秒级完成。

---

### `sync_down.sh` — 下载实验

```bash
bash scripts/remote/sync_down.sh                          # 全部实验
bash scripts/remote/sync_down.sh dp3/pour                 # 特定 policy/task
bash scripts/remote/sync_down.sh dp3/pour/2026-08-03_12   # 特定 run
bash scripts/remote/sync_down.sh --dry-run                # 预览
bash scripts/remote/sync_down.sh --list                   # 列出服务器上的实验
bash scripts/remote/sync_down.sh --with-wandb             # 含 wandb 离线数据
```

**关键行为**:
- 仅传输新增/变更的文件（rsync 默认 size+mtime 比较）
- **排除本地评测产物**（见上表），不会被覆盖
- 5 个 milestone checkpoint .pt 文件（各 ~1.5-2GB）只在首次下载时传输；后续仅 `latest.pt` symlink 和 `metrics.jsonl` 等小文件更新
- 不含 `--delete`：本地删了实验不会被远端删

---

### `train_remote.sh` — 一键训练

```bash
bash scripts/remote/train_remote.sh <config> <task> [hydra_overrides...]
bash scripts/remote/train_remote.sh --fg <config> <task> [...]     # 前台
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 <config> <task> [...]  # 指定 GPU
bash scripts/remote/train_remote.sh --sync-data <config> <task>    # 含数据同步
bash scripts/remote/train_remote.sh --dry-run <config> <task>      # 预览
```

**Pre-flight checks（任一失败则退出）**:

| 步骤 | 检查内容 | 失败提示 |
|------|---------|---------|
| 1 | 服务器可达 | "Check VPN/network" |
| 2 | 代码同步 (`sync_code.sh`) | rsync 错误 |
| 3 | `{task}.zarr` 存在于服务器 | "Run sync_data.sh" |
| 4 | GPU 占用情况（打印，不阻塞） | — |
| 5 | `/data_ssd` 磁盘剩余空间 | — |

**示例**:

```bash
# ── 单卡 ──
bash scripts/remote/train_remote.sh dp3 pour

# 单卡 + 指定 GPU + 指定 seed
bash scripts/remote/train_remote.sh --gpus 0 sat pour 'training.seed=123'
bash scripts/remote/train_remote.sh --gpus 0 maniflow pour 'training.seed=42'

# 单卡 + 多个覆盖参数
bash scripts/remote/train_remote.sh --gpus 0 sat pour \
    'training.seed=123' 'training.loop.total_train_steps=50000'

# ── DDP 多卡 ──
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour

# DDP 多卡 + 指定 seed
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour 'training.seed=99'
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/r3d pour 'training.seed=7'

# ── 其他 ──
# 前台调试（看完整输出，Ctrl+C 可中断）
bash scripts/remote/train_remote.sh --fg dp3 pour 'training.loop.total_train_steps=10'

# 首次在新服务器上训练（含数据上传）
bash scripts/remote/train_remote.sh --sync-data dp3 pour
```

---

### `tail_log.sh` — 查看训练日志

```bash
bash scripts/remote/tail_log.sh dp3 pour                    # 自动找最新 run
bash scripts/remote/tail_log.sh dp3 pour 2026-08-03_12-34   # 指定 run
```

自动尝试服务器 → 本地 fallback。Ctrl+C 退出。

---

### `stop_remote.sh` — 停止训练

```bash
bash scripts/remote/stop_remote.sh dp3_pour          # 停止特定 session
bash scripts/remote/stop_remote.sh --all             # 停止全部
bash scripts/remote/stop_remote.sh --list            # 查看活跃 session
```

---

## 日常工作流

### 典型训练周期

```bash
# 1. 改完代码 → 同步到服务器（2-3秒）
bash scripts/remote/sync_code.sh

# 2. 启动训练（自动再同步一次代码 + pre-flight checks）
bash scripts/remote/train_remote.sh dp3 pour

# 3. 监控进度
bash scripts/remote/tail_log.sh dp3 pour                    # Ctrl+C 退出

# 4. 训练完成 → 拉取实验结果
bash scripts/remote/sync_down.sh dp3/pour                   # 只拉这个 task
# 或
bash scripts/remote/sync_down.sh --list                     # 先看有哪些实验
bash scripts/remote/sync_down.sh dp3/pour/2026-08-03_12-34  # 拉特定 run

# 5. 本地评测
bash scripts/eval/select_best_ckpt.sh dp3 pour experiments/dp3/pour/2026-08-03_12-34
bash scripts/eval/eval_best_ckpt.sh dp3 pour experiments/dp3/pour/2026-08-03_12-34

# 6. 可选: 录制 demo 视频
bash scripts/eval/record_demo.sh dp3 pour experiments/dp3/pour/2026-08-03_12-34
```

### 改代码 + 训多个策略

```bash
# 改代码后，训练 3 个策略
bash scripts/remote/sync_code.sh

bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour
bash scripts/remote/train_remote.sh --gpus 4,5,6,7 ddp/dp3_faas pour
bash scripts/remote/train_remote.sh --gpus 0 sat pour          # 单卡

# 拉全部 pour 实验结果
bash scripts/remote/sync_down.sh maniflow/pour
bash scripts/remote/sync_down.sh dp3_faas/pour
bash scripts/remote/sync_down.sh sat/pour
```

### 数据上传（新增任务后）

```bash
# 新加了数据集到本地 robot_data/，只上传新增的
bash scripts/remote/sync_data.sh robot_data --dry-run    # 先看会传什么
bash scripts/remote/sync_data.sh robot_data              # 实际传输
```

---

## GPU 多租户并行

8 张 H200 通过 `CUDA_VISIBLE_DEVICES` 分区，`train_remote.sh --gpus` 已集成：

```bash
# 两个 4 卡 DDP 并行（各自指定 seed）
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour 'training.seed=42'
bash scripts/remote/train_remote.sh --gpus 4,5,6,7 ddp/dp3_faas pour 'training.seed=99'

# 8 个单卡 seed sweep
for i in 0 1 2 3 4 5 6 7; do
    bash scripts/remote/train_remote.sh --gpus $i dp3 pour "training.seed=$i" &
done

# 检查 GPU 占用
ssh dexserver 'nvidia-smi --query-gpu=index,memory.used,name --format=csv,noheader'
```

---

## Wandb

```bash
# 在线模式（需服务器能访问 wandb.ai）
bash scripts/remote/train_remote.sh dp3 pour 'workspace.wandb_cfg.mode=online'

# 离线模式（默认），事后同步
bash scripts/remote/sync_down.sh --with-wandb dp3/pour       # 拉 wandb 数据
bash scripts/utils/wandb_sync.sh                            # 本地同步到云端
```

---

## 传输速率基准

> 实测环境: zhy-MS-7E06 ↔ 192.168.88.230，千兆局域网，scp 单文件，2026-08-06。

| 方向 | 文件大小 | 耗时 | 速率 |
|------|------|------|------|
| ⬆️ 上传 | 100 MB | 3.6s | **27.6 MB/s** (221 Mbps) |
| ⬆️ 上传 | 1 GB | 40.4s | **25.4 MB/s** (203 Mbps) |
| ⬇️ 下载 | 100 MB | 3.2s | **31.3 MB/s** (250 Mbps) |
| ⬇️ 下载 | 1 GB | 31.2s | **32.9 MB/s** (263 Mbps) |

**实际场景预估**（基于 25 MB/s 保守值）:

| 操作 | 数据量 | 耗时 |
|------|--------|------|
| 同步源码 | ~60 MB | ~2-3 秒 |
| 同步 robot_data（首次） | ~8.5 GB | ~6 分钟 |
| 同步 robot_data（增量，无变化） | 0 | <1 秒 |
| 下载 1 个实验（5 个 checkpoint .pt） | ~7-10 GB | ~5-7 分钟 |
| 下载 1 个实验（增量，无新 checkpoint） | ~500 KB (metrics+logs) | <1 秒 |

---

## 运维速查

```bash
# GPU 状态
ssh dexserver 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader'

# 磁盘空间
ssh dexserver 'df -h /data_ssd /home'

# tmux 会话
ssh dexserver 'tmux list-sessions 2>/dev/null || echo "no sessions"'
# 或
bash scripts/remote/stop_remote.sh --list

# 训练进程
ssh dexserver 'ps aux | grep train'

# conda 环境
ssh dexserver 'ls ~/.conda/envs/'

# 服务器上的实验列表
bash scripts/remote/sync_down.sh --list
```

---

## 脚本清单

```
scripts/
├── training/
│   ├── train.sh            # 本地单卡训练（原有）
│   ├── train_ddp.sh        # 本地 DDP 训练（原有）
│   └── train_vq_hand.sh    # VQ-VAE 预训练（原有）
├── remote/
│   ├── sync_code.sh        # 上传源码（频繁，2-3秒）
│   ├── sync_data.sh        # 上传数据（增量，秒级跳过）
│   ├── sync_down.sh        # 下载实验（保护本地评测产物）
│   ├── train_remote.sh     # 一键训练（pre-flight checks + tmux）
│   ├── tail_log.sh         # 实时日志流
│   └── stop_remote.sh      # 停止训练
├── eval/
│   ├── select_best_ckpt.sh # 最优 checkpoint 筛选（原有）
│   ├── eval_best_ckpt.sh   # 离线评测（原有）
│   ├── eval_pipeline.sh    # 评测管道（原有）
│   └── record_demo.sh      # Demo 录制（原有）
└── utils/
    ├── clean_experiments.sh # 清理实验（原有）
    ├── download_pretrained.sh # 下载预训练权重（原有）
    └── wandb_sync.sh        # Wandb 离线同步（原有）
```

> 原有 `train.sh`、`train_ddp.sh`、`eval_best_ckpt.sh` 等脚本保持不变，仅新增了 6 个远程部署脚本。

---

---
## 附录 A: SSH 终端操作常识

> 面向不熟悉 Linux 终端的同学。已熟悉的可以跳过。

### A.1 登录 SSH 服务器

**基本命令**：

```bash
# 完整格式
ssh 用户名@服务器地址 -p 端口号

# 本项目的实际登录命令（等价于文档中用到的 ssh dexserver）
ssh zjurobot@192.168.88.230 -p 51822
```

**配置 SSH 别名（推荐，一次配置长期省事）**：

编辑本地 `~/.ssh/config`，添加：

```
Host dexserver
    HostName 192.168.88.230
    Port 51822
    User zjurobot
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

之后直接用别名登录：

```bash
ssh dexserver          # 等价于上面一长串
```

**退出 SSH**：

```bash
exit                   # 或按 Ctrl+D
```

**免密登录（SSH 密钥）**：

```bash
# 1. 本地生成密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 把公钥复制到服务器
ssh-copy-id -p 51822 zjurobot@192.168.88.230

# 3. 之后 ssh dexserver 不再需要输入密码
```

> 本项目 SSH 密钥已配置 ✅。

---

### A.2 目录操作

**新建目录**：

```bash
# 新建单个目录
mkdir my_folder

# 递归新建多级目录（推荐，不会因父目录不存在而报错）
mkdir -p ~/ZHY/dexmani_policy/experiments/dp3/pour
```

**删除目录**：

```bash
# 删除空目录
rmdir empty_folder

# 删除非空目录（递归强制删除，谨慎！没有回收站）
rm -rf my_folder

# 更安全：先列出会删什么
ls my_folder/           # 确认内容
rm -rf my_folder        # 确认无误再删
```

> ⚠️ **`rm -rf` 没有确认提示，删了就没了。** 删除前务必确认路径正确。尤其在服务器上不要 `rm -rf /` 或 `rm -rf ~`。

**查看目录内容**：

```bash
ls                     # 列出当前目录
ls -la                 # 详细列表（含隐藏文件、权限、大小）
ls -lh                 # 人类可读的文件大小（KB/MB/GB）
ls experiments/dp3/    # 列出指定目录
```

**切换目录**：

```bash
cd ~/ZHY/dexmani_policy    # 进入项目目录
cd ..                      # 返回上一级
cd -                       # 返回上一次所在的目录
cd                         # 返回 home 目录
pwd                        # 显示当前所在完整路径
```

**创建符号链接（symlink）**：

```bash
# 让 ~/ZHY/dexmani_policy/experiments 指向 /data_ssd/ZHY/experiments
ln -sfn /data_ssd/ZHY/experiments ~/ZHY/dexmani_policy/experiments

# -s: 符号链接（类似快捷方式）
# -f: 强制覆盖已有链接
# -n: 如果目标是目录链接，替换链接本身而非链接内部
```

---

### A.3 训练命令参数详解

训练命令的核心结构只有三个位置：

```
train.py <策略类型> <任务名> [Hydra覆盖参数...]
```

#### 参数 1：策略类型（policy）

决定用哪个 Agent 架构。可用值：

| 策略 | 命令写法 | 说明 |
|------|---------|------|
| DP | `dp` | RGB+UNet+Diffusion |
| DP3 | `dp3` | 点云+UNet+Diffusion |
| DP3 + FAAS | `dp3_faas` | DP3 + 功能对齐动作空间 |
| ManiFlow | `maniflow` | 点云+DiTX+FlowMatch |
| MoE DP | `moe_dp` | RGB+MoE 多专家+Diffusion |
| MultiTask | `multitask_dit` | 多任务 DiT |
| R3D | `r3d` | 点云+OneWayTransformer |
| DQ-RISE | `dqrise` | 点云+VQ 码本手势 |
| SAT | `sat` | 结构动作 Transformer |

**DDP 多卡版本**（用 `ddp/` 前缀）：

| DDP 策略 | 命令写法 | GPU 数 |
|----------|---------|--------|
| DDP ManiFlow | `ddp/maniflow` | 4 |
| DDP DP3 FAAS | `ddp/dp3_faas` | 4 |
| DDP MultiTask | `ddp/multitask_dit` | 4 |
| DDP R3D | `ddp/r3d` | 4 |
| DDP DP | `ddp/dp` | 4 |
| DDP DQ-RISE | `ddp/dqrise` | 4 |

> `dp3`（非 FAAS）、`moe_dp`、`sat` 只支持单卡，无 DDP 版本。

#### 参数 2：任务名（task）

即数据集名，对应 `robot_data/<task>.zarr`。当前可用任务：

```bash
# 查看服务器上有哪些任务
ssh dexserver 'ls /data_ssd/ZHY/robot_data/*.zarr/ 2>/dev/null'
```

| 任务 | 命令写法 | 说明 |
|------|---------|------|
| 倒水 | `pour` | 主要任务 |

> 后续新增任务（如 `grasp`、`place` 等）直接写对应的 `.zarr` 名前缀即可。

#### 参数 3：Hydra 覆盖参数（可选，可多个）

用于覆盖配置文件中的任意值。格式：`key.path=value`（点号分隔层级）。训练最常用的几个：

```bash
# ---- 种子 ----
training.seed=42                     # 设置随机种子（默认在各 config 中定义）

# ---- 训练步数 ----
training.loop.total_train_steps=50000 # 覆盖总训练步数（默认 80000）

# ---- 动作空间 ----
action_key=action_ee                 # 切换到末端执行器空间（默认 action 即关节空间）

# ---- Wandb ----
workspace.wandb_cfg.mode=online      # 启用在线 wandb（默认 offline）
```

#### 完整示例

```bash
# === 使用 train_remote.sh（推荐，自动同步代码+tmux 后台） ===

# ── 单卡 ──

# 最基本：DP3 训练 pour，全部默认参数
bash scripts/remote/train_remote.sh dp3 pour

# 指定 seed
bash scripts/remote/train_remote.sh dp3 pour 'training.seed=42'

# 指定 GPU + seed
bash scripts/remote/train_remote.sh --gpus 0 sat pour 'training.seed=123'
bash scripts/remote/train_remote.sh --gpus 0 maniflow pour 'training.seed=7'

# SAT 训练，自定义 seed + 步数
bash scripts/remote/train_remote.sh --gpus 0 sat pour \
    'training.seed=123' 'training.loop.total_train_steps=50000'

# ManiFlow 用 EE 空间 + 自定义 seed
bash scripts/remote/train_remote.sh --gpus 0 maniflow pour \
    'action_key=action_ee' 'training.seed=7'

# ── DDP 多卡 ──

# DDP 4 卡训练（默认参数）
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/dp pour

# DDP 4 卡 + 指定 seed
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour 'training.seed=99'
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/r3d pour 'training.seed=7'

# DDP 4 卡 + seed + 步数
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour \
    'training.seed=99' 'training.loop.total_train_steps=50000'

# === 直接 SSH 手动启动（调试用） ===

ssh dexserver
cd ~/ZHY/dexmani_policy
conda activate dex_policy
export DATA_DIR=/data_ssd/ZHY

python dexmani_policy/train.py --config-name=dp3 task_name=pour
python dexmani_policy/train.py --config-name=sat task_name=pour training.seed=42 training.loop.total_train_steps=30000
python dexmani_policy/train.py --config-name=maniflow task_name=pour action_key=action_ee training.seed=7
```

#### seed sweep（多 seed 并行扫描）

```bash
# 单卡 seed sweep：8 个 GPU 各跑一个 seed
for seed in 0 1 2 3 4 5 6 7; do
    bash scripts/remote/train_remote.sh --gpus $seed dp3 pour "training.seed=$seed" &
done

# 单卡 seed sweep：不同策略+seed 组合
for seed in 42 123 999; do
    bash scripts/remote/train_remote.sh --gpus 0 sat pour "training.seed=$seed"
    bash scripts/remote/train_remote.sh --gpus 1 maniflow pour "training.seed=$seed"
done

# DDP 多卡 seed sweep：每 4 卡一组跑不同 seed
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/maniflow pour 'training.seed=42' &
bash scripts/remote/train_remote.sh --gpus 4,5,6,7 ddp/maniflow pour 'training.seed=99' &
wait  # 等两组都完成
```

**确保训练在断开 SSH 后继续运行**：

如果用直接 `python train.py` 而非 `train_remote.sh`，SSH 断开后训练会终止。需要用 `tmux` 保护：

```bash
ssh dexserver
tmux new -s train_dp3                # 创建会话
cd ~/ZHY/dexmani_policy && conda activate dex_policy && export DATA_DIR=/data_ssd/ZHY
python train.py dp3 pour training.seed=42
# Ctrl+B, D 断开 → 训练继续跑
# 重连: ssh dexserver && tmux attach -t train_dp3
```

> `train_remote.sh` 已自动在 tmux 里启动，直接用即可，不需要手动操作 tmux。

---

### A.4 其他常用终端命令

#### 进程管理

```bash
# 查看 Python 训练进程
ps aux | grep python
ps aux | grep train

# 查看进程树（父子关系）
pstree -p

# 终止进程
kill <PID>                # 优雅终止
kill -9 <PID>             # 强制杀死（最后手段）
pkill -f "train.py dp3"   # 按名字杀进程

# 实时进程监控
htop                       # 比 top 更友好（可能需要安装）
top                        # 系统自带
```

#### GPU 监控

```bash
# 实时 GPU 状态（每 1 秒刷新）
watch -n 1 nvidia-smi

# 精简输出
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader

# 持续监控 GPU 利用率
nvidia-smi dmon -s u
```

#### 磁盘与文件

```bash
# 磁盘空间
df -h                     # 各挂载点使用情况
df -h /data_ssd           # 只看特定目录所在分区

# 目录大小
du -sh *                  # 当前目录下各文件/文件夹大小
du -sh experiments/       # 某个目录总大小
du -h --max-depth=1       # 一层深度的各子目录大小

# 文件操作
cp -r source/ dest/       # 递归复制
mv old_name new_name      # 移动/重命名
cat file.txt              # 查看文件内容
less file.txt             # 分页查看（q 退出，/ 搜索）
head -20 file.txt         # 前 20 行
tail -20 file.txt         # 后 20 行
tail -f metrics.jsonl     # 实时追踪文件末尾增长（Ctrl+C 退出）
wc -l file.txt            # 统计行数
```

#### 网络与传输

```bash
# scp 传输（单文件/目录）
# 上传：本地 → 服务器
scp -P 51822 local_file.txt zjurobot@192.168.88.230:~/ZHY/
scp -P 51822 -r local_dir/ zjurobot@192.168.88.230:~/ZHY/

# 下载：服务器 → 本地
scp -P 51822 zjurobot@192.168.88.230:~/ZHY/file.txt ./
scp -P 51822 -r zjurobot@192.168.88.230:~/ZHY/dir/ ./

# rsync（增量传输，比 scp 更高效）
rsync -avz local_dir/ dexserver:~/ZHY/dir/        # 上传
rsync -avz dexserver:~/ZHY/dir/ local_dir/         # 下载
# -a: 归档模式（保留权限、时间戳）
# -v: 详细输出
# -z: 压缩传输
# --dry-run: 只预览不实际传输
# --delete: 删除目标端多出的文件
```

#### Tmux 速查

```bash
# 会话管理
tmux new -s name         # 创建会话
tmux attach -t name      # 重新连接会话
tmux ls                  # 列出所有会话
tmux kill-session -t name # 终止会话

# 在 tmux 内部（默认前缀键: Ctrl+B）
Ctrl+B  D    # 断开（detach），训练继续运行
Ctrl+B  C    # 创建新窗口
Ctrl+B  N    # 下一个窗口
Ctrl+B  [    # 滚动模式（PgUp/PgDn 翻页，q 退出）
```

#### Conda 环境

```bash
conda env list                    # 列出所有环境
conda activate dex_policy         # 激活环境
conda deactivate                  # 退出环境
pip list | grep torch             # 查看安装了哪些包
which python                      # 确认用的是哪个 Python
```

#### 实用组合技

```bash
# 看 GPU 显存 + 对应进程
nvidia-smi && echo "---" && ps aux | grep python | grep -v grep

# 从本地看远程 GPU（不登录）
ssh dexserver 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'

# 统计服务器上实验数量
ssh dexserver 'ls /data_ssd/ZHY/experiments/*/*/ 2>/dev/null | wc -l'

# 找大于 1GB 的文件（清理磁盘用）
ssh dexserver 'find /data_ssd/ZHY/experiments -type f -size +1G -exec ls -lh {} \;'

# 后台运行任意命令（不依赖 tmux）
nohup python train.py dp3 pour > train.log 2>&1 &
# nohup: 忽略 hangup 信号
# > train.log: 标准输出到日志
# 2>&1: 错误输出也到同一日志
# &: 后台运行
# 查看: tail -f train.log
# 停止: ps aux | grep train → kill <PID>
```

---
## v4 更新记录（2026-08-06，架构重构）

| 变更 | v3 | v4 |
|------|-----|-----|
| 同步架构 | 1 个 `sync_up.sh` 同时处理代码+数据 | **3 个独立脚本**：`sync_code.sh`（源码）、`sync_data.sh`（数据）、`sync_down.sh`（实验） |
| 代码同步 | 混在 sync_up 中 | **独立 `sync_code.sh`**，`-z` 压缩 + `--delete`，轻量高频 |
| 数据上传 | 混在 sync_up 中 | **独立 `sync_data.sh`**，免压缩 + 免 `--delete`，增量安全 |
| 实验下载 | `sync_down.sh` 简单 rsync | **重写**：自动排除本地评测产物（`eval_dexsim/`, `demo_videos/`, `best_ckpt.json` ...），保护本地文件不被覆盖 |
| `data/` 目录 | 随源码同步 | **数据盘独立存储**，symlink 透明访问 |
| `train_remote.sh` | 调用 `sync_up.sh` | 调用 `sync_code.sh`，新增 `--sync-data` flag |
| 目录初始化 | 手动 `mkdir` | **结构化 Phase 1 流程**，含 symlink 创建 |
| 可视化 | 无 | **同步架构全景图** + 职责表 |
