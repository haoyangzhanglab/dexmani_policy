# DexMani Policy · SSH 服务器训练部署

命令从**本地仓库根目录**执行；常规训练与验证见 [README](../README.md#快速开始)，预训练见 [手部动作表示](手部动作表示.md)。远端训练使用独立源码副本，数据和实验产物通过共享目录访问。

## 准备环境

默认 SSH host 为 `dexserver`，先配置 `~/.ssh/config` 并确认 `ssh dexserver` 可用；其他主机用 `export DEX_SERVER=<ssh_host>` 指定。

本地需要 Bash、SSH、rsync、Python 3；远端需要 Bash、rsync、tmux、训练依赖。远端解释器**不必**叫 `policy`。首次使用或更换服务器时，核对脚本中的源码目录、Conda Python、数据与实验目录：

| 脚本 | 需核对的配置 |
| --- | --- |
| `train_remote.sh` | `SERVER_PROJ`、`CONDA_PYTHON` |
| `sync_code.sh` | `REMOTE_CODE` |
| `sync_data.sh` | `REMOTE_DATA` |
| `sync_down.sh` / `tail_log.sh` | `REMOTE_EXP` / `SERVER_EXP` |

提前创建持久存储，将远端源码目录的 `data/`、`robot_data/`、`experiments/` 接到相应位置；launcher 不会准备存储。Uni3D 权重可随 `data/pretrained/uni3d/` 同步，其他 pretrained_models/HF 缓存须单独准备。迁移机器时统一调整相关路径，避免只改一处。

## 启动与源码副本

launcher 选项位于 config/task **之前**，Hydra overrides 位于其后：

```bash
bash scripts/remote/train_remote.sh --dry-run dp3 <task>
bash scripts/remote/train_remote.sh --gpus 0 dp3 <task> 'training.seed=42'
bash scripts/remote/train_remote.sh --fg --gpus 0 dp3 <task> '+max_updates=4'
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/sat <task>
```

流程：SSH 检查 → 创建唯一 `outputs/remote_sources/<id>/` → 同步源码（可选 `--sync-data`）→ 链接共享目录 → 检查 Dataset 路径 → 查询 GPU/磁盘 → 启动训练。`--gpus` 必须在查询得到的 GPU index 内；查询失败中止。**目录与资源检查通过，不代表权重、显存或剩余空间足够。**

远端使用副本中的 Python module 入口训练；副本共享 `robot_data/`、`data/`、`experiments/`、`pretrained_models/` 和 `logs/`。主源码后续同步**不会改变**运行中的副本，但修改共享数据会影响实验；副本不自动清理，失败后也保留。`--dry-run` 仍连接 SSH 并预览 rsync，但不创建副本、不验证数据/GPU。`--fg` 为前台 SSH，适合排查启动错误。

### 配置与并行运行

DDP 需使可见 GPU 数与 `training.num_gpus` 一致，有效 batch = 每卡 batch × world size × gradient accumulation。`ddp/sat` 选择 DDP 配置，但实验、W&B group、评测和同步路径均按基础方法 `sat` 命名：

```bash
bash scripts/remote/tail_log.sh sat <task> <run>
bash scripts/remote/sync_down.sh sat/<task>/<run>
```

后台使用 `tmux -L dexmani_policy`，每次创建唯一 session 和 `logs/<session>.log`；任务退出后 session 消失但日志保留。多任务以 `+` 连接，复杂 overrides 应整体引用。

### 续训

```bash
bash scripts/remote/train_remote.sh --gpus 0 dp3 <task> \
  '+resume_from=experiments/dp3/<task>/<run>'
```

相对路径经共享 `experiments/` 解析；恢复到**新 run**，任务身份须匹配，允许更改的字段见 [完整恢复边界](项目架构.md#完整恢复边界)。

## 同步边界

### 源码

```bash
bash scripts/remote/sync_code.sh --dry-run
bash scripts/remote/sync_code.sh
```

主源码同步使用 `--delete`，但排除 Git/工具配置、缓存、数据、实验、权重和运行产物，并保护数据/实验根目录。`--dest <absolute_directory>` 只接受空目录，用于独立副本；launcher 已自行同步副本，通常无需先同步主源码。

### 数据

```bash
bash scripts/remote/sync_data.sh --dry-run robot_data
bash scripts/remote/sync_data.sh robot_data
bash scripts/remote/sync_data.sh --checksum data
bash scripts/remote/sync_data.sh --pull --dry-run robot_data
```

目标为 `all`、`data` 或 `robot_data`；默认 push，按大小/mtime 更新，不删目标独有文件。`--checksum` 按内容比较，`--pull` 反转方向。**`--prune` 会删除目标独有文件、不可撤销**，使用前必须以相同方向/参数先执行 `--dry-run`。已用于实验的数据不应原地重建，须使用新路径。

### 实验结果

```bash
bash scripts/remote/sync_down.sh --list
bash scripts/remote/sync_down.sh <policy>/<task>/<run> --dry-run
bash scripts/remote/sync_down.sh <policy>/<task>/<run>
```

只从远端拉取，**不删除本地结果**；不带子路径时拉取全部实验。`--list`/`--dry-run` 也会连接 SSH。双方已有 `config.yaml` 必须 checksum 完全一致，否则中止；同名大权重不做完整内容身份校验。

同步分阶段完成：① 新文件及不可变产物；② 按 size/mtime 更新可变的 `metrics.jsonl`、`latest.pt` 与先验权重，并无条件传输 `result_details.json`、最后传 `selection_result.json`；③ 无条件更新 `best_ckpt.json`。**先同步候选事实/选点记录，再同步 best 指针**，避免指针领先证据。

同步排除 `.dexmani-publish-*` 临时文件；任一 rsync 非零返回（包括 24）都中止后续阶段。整个目录不是事务快照，发布过程中可能需要重试；使用前检查 Checkpoint 和 selection 证据齐全。先验权重在 `experiments/` 时由此下载，在 `data/` 时用数据同步。

### W&B

实验结果默认不拉取 W&B；上传 offline run 须先用 `sync_down.sh --with-wandb` 拉取，再运行：

```bash
bash scripts/utils/wandb_sync.sh --dry-run --all
bash scripts/utils/wandb_sync.sh --all
```

上传需要本地凭据和网络；拉取并不代表已经上传。

## 监控与停止

```bash
bash scripts/remote/tail_log.sh <policy> <task> <run>
bash scripts/remote/stop_remote.sh --list
bash scripts/remote/stop_remote.sh <session>
bash scripts/remote/stop_remote.sh --all
```

`tail_log.sh` 跟随 `metrics.jsonl`，省略 run 时按 `config.yaml` 修改时间选择；只有 SSH 不通时回退本地。启动异常查看 launcher 的控制台日志或 tmux Attach 命令。

停止只作用于项目 tmux socket：先发一次 SIGINT，约 30 秒仍存活则报告错误并保留进程，只有显式 `--force` 才强制结束。Trainer 尝试在 update 边界保存，重复中断/强杀可能导致未保存；**session 消失不等于 Checkpoint 完整**。清理脚本 `scripts/utils/clean_experiments.sh` 仅做只读盘点。后续按 [仿真评测机制](仿真评测机制.md) 选点与评测。
