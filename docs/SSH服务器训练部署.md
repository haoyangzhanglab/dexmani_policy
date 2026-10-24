# DexMani_Policy SSH 服务器训练部署

命令从本地仓库根目录执行。训练与数据接口见 [README](../README.md)，先验准备见 [手部动作表示](手部动作表示.md)。

## 准备环境

默认 SSH host 为 `dexserver`，在 `~/.ssh/config` 配置并确认 `ssh dexserver` 可用；其他别名通过 `export DEX_SERVER=<ssh_host>` 指定。

本地需 Bash、SSH、rsync、Python 3；远端需 Bash、rsync、tmux 和训练依赖。远端解释器不必与本地 Conda `policy` 环境同名。首次使用核对脚本顶部配置：

| 配置位置 | 当前默认远端路径 |
| --- | --- |
| `train_remote.sh`：`SERVER_PROJ` | `$HOME/ZHY/dexmani_policy` |
| `train_remote.sh`：`CONDA_PYTHON` | `$HOME/.conda/envs/dex_policy/bin/python` |
| `sync_code.sh`：`REMOTE_CODE` | `~/ZHY/dexmani_policy/` |
| `sync_data.sh`：`REMOTE_DATA` | `/data_ssd/ZHY/` |
| `sync_down.sh`：`REMOTE_EXP`、`tail_log.sh`：`SERVER_EXP` | `/data_ssd/ZHY/experiments` |

准备持久存储，并将主源码目录的 `data/`、`robot_data/`、`experiments/` 连接到对应位置；launcher 不自动配置这些存储。Uni3D 默认位于 `data/pretrained/uni3d/`，可随 data 同步；独立 `pretrained_models/` 和 HF 缓存需另行准备。换服务器时同步调整全部相关路径。

## 启动与源码副本

launcher 选项在 config/task 前，Hydra overrides 在后：

```bash
bash scripts/remote/train_remote.sh --dry-run dp3 <task>
bash scripts/remote/train_remote.sh --gpus 0 dp3 <task> 'training.seed=42'
bash scripts/remote/train_remote.sh --fg --gpus 0 dp3 <task> '+max_updates=4'
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/sat <task>
```

`--sync-data` 在启动前同步数据。正式流程为：检查 SSH → 创建唯一 `outputs/remote_sources/<id>/` → 同步源码/可选数据 → 链接共享目录 → 检查 Dataset 目录 → 显示 GPU/磁盘并提交训练。前置检查失败不启动训练；数据检查只验证目录，GPU/磁盘输出不保证权重、OOM 或剩余空间满足训练。

训练以副本中的 `python -m dexmani_policy.train` / `train_ddp` 启动。副本共享 `robot_data`、`data`、`experiments`、`pretrained_models`、`logs`；同步主源码不影响运行中的副本，修改共享数据仍会影响实验。副本不自动清理。

`--dry-run` 仍访问 SSH 并预览 rsync，不创建副本或验证数据/GPU。本地检查见 [验证指南](验证指南.md)，路径解析见 [数据准备](../README.md#数据准备)。

### 配置与并行运行

`ddp/` 选择多卡入口，可见 GPU 数需与 `training.num_gpus` 一致。有效 batch = 每卡 batch × world size × gradient accumulation。config 名与输出 `policy_name` 可不同，多任务名用 `+` 连接，复杂 override 整体引用。

后台使用 `tmux -L dexmani_policy`，每次创建唯一 session 和 `logs/<session>.log`；任务退出后 session 消失，日志保留。`--fg` 使用前台 SSH，便于查看启动错误。

### 续训

```bash
bash scripts/remote/train_remote.sh --gpus 0 dp3 <task> \
  '+resume_from=experiments/dp3/<task>/<run>'
```

相对实验路径通过共享目录解析，恢复写入新 run，task 必须匹配保存身份。允许覆盖项见 [完整恢复边界](项目架构.md#完整恢复边界)。

## 同步边界

### 源码

```bash
bash scripts/remote/sync_code.sh --dry-run
bash scripts/remote/sync_code.sh
```

主源码同步使用 `--delete`；Git、工具配置、缓存、数据、实验、权重和运行输出被排除，数据/实验根受删除保护。`--dest <absolute_directory>` 只接受空目标，用于新副本。launcher 已自动同步运行副本，无需每次先同步主源码。

### 数据

```bash
bash scripts/remote/sync_data.sh --dry-run robot_data
bash scripts/remote/sync_data.sh robot_data
bash scripts/remote/sync_data.sh --checksum data
bash scripts/remote/sync_data.sh --pull --dry-run robot_data
```

目标可选 `all`、`data`、`robot_data`。默认 push，按 size/mtime 更新，不删除目标独有文件；`--checksum` 比内容，`--pull` 反转方向。`--prune` 删除目标独有文件且无回收站，先用同参数加 `--dry-run` 核对方向与删除清单。已用于实验的数据应保留，重新生成使用新路径。

### 实验结果

```bash
bash scripts/remote/sync_down.sh --list
bash scripts/remote/sync_down.sh <policy>/<task>/<run> --dry-run
bash scripts/remote/sync_down.sh <policy>/<task>/<run>
```

只从远端拉取，不删除本地；不传子路径时拉取全部 experiments。`--list` / `--dry-run` 也访问 SSH。双方已有 `config.yaml` 按 checksum 比较，任何字节差异均中止；不验证同名大权重的内容身份。

| 顺序 | 更新内容 |
| --- | --- |
| 1 | 下载新文件，保留已有不可变产物，排除临时文件及后续可变项 |
| 2 | 按 size/mtime 更新 `metrics.jsonl`、`checkpoints/latest.pt`、`vqvae_hand_best.pt`、`prior.pt`、`codec.pt`、`codebook.npz`，再无条件传输 `result_details.json`，最后传 `selection_result.json` |
| 3 | 无条件传输 `best_ckpt.json`，确保它引用的候选事实和选择记录已传输成功 |

checkpoint/选择记录先于指针拉取。发布中的临时文件统一使用 `.dexmani-publish-*`，同步时排除；不识别旧版临时文件命名。新文件不组合 `--partial` 与 `--ignore-existing`；任一 rsync 非零退出（含 24）中止后续阶段。整个目录不是事务快照，远端发布中可能需重试；使用前确认 checkpoint 和选择证据齐全。先验产物放在 experiments 时由此拉取，放在 data 时使用数据同步。

### W&B

实验拉取默认排除 W&B，上传 offline run 时先加 `--with-wandb` 拉取：

```bash
bash scripts/utils/wandb_sync.sh --dry-run --all
bash scripts/utils/wandb_sync.sh --all
```

上传需本地凭据与网络；拉取文件不等于已上传。

## 监控与停止

```bash
bash scripts/remote/tail_log.sh <policy> <task> <run>
bash scripts/remote/stop_remote.sh --list
bash scripts/remote/stop_remote.sh <session>
bash scripts/remote/stop_remote.sh --all
```

`tail_log.sh` 跟随 `metrics.jsonl`。省略 run 时按 `config.yaml` 修改时间选择，名称打破并列；仅 SSH 不可达才回退本地。启动错误和 traceback 查看 launcher 打印的控制台日志或 Attach 命令。

停止脚本只管理项目 tmux socket：先发一次 SIGINT，等待约 30 秒；仍存活则报错并保留进程，显式 `--force` 才允许随后强制结束。Trainer 在更新边界尝试保存，再次中断/强制结束可能跳过保存；session 消失不保证 checkpoint 完整。空列表正常返回，SSH/tmux 查询失败明确报错。

`scripts/utils/clean_experiments.sh` 只读盘点，不删除实验或源码副本。训练完成后按 [仿真评测机制](仿真评测机制.md) 选点和评测。
