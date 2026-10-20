# DexMani_Policy SSH 服务器训练部署

本文说明远端启动、源码和数据同步、监控、停止与实验拉取。模型及数据接口见 [项目架构](项目架构.md)，本地训练和先验准备见 [README](../README.md)。命令从本地仓库根目录执行，`<...>` 替换为实际值。

## 准备环境

远端脚本使用 `DEX_SERVER` 指定 SSH host，默认 `dexserver`。在本地 `~/.ssh/config` 配置对应的 HostName、User、Port 和 IdentityFile，并先确认：

```bash
ssh dexserver
```

使用其他别名时设置：

```bash
export DEX_SERVER=<ssh_host>
```

本地需要 Bash、SSH、rsync 和 Python 3；远端需要 Bash、rsync、tmux、训练 Python 环境及模型依赖。launcher 的 Python 小工具在本地运行，训练解释器由 `train_remote.sh` 中的 `CONDA_PYTHON` 指定；它不必与本地 Shell 使用的 Conda `policy` 环境同名。

首次使用时核对各脚本顶部的路径配置：

| 配置位置 | 需要对应的路径 |
| --- | --- |
| `train_remote.sh` | `SERVER_PROJ`、`CONDA_PYTHON` |
| `sync_code.sh` | 主源码目录 `REMOTE_CODE` |
| `sync_data.sh` | `data/` 与 `robot_data/` 的远端存储根 |
| `sync_down.sh`、`tail_log.sh` | 同一实验存储根 |

先在远端准备持久数据、实验和预训练权重目录，并将主源码目录下的 `data/`、`robot_data/`、`experiments/` 连接到相应位置。依赖安装方法见 [安装与配置](../README.md#安装与配置)。`sync_data.sh` 不上传 `pretrained_models/`，所需权重需另行准备。

主源码与持久数据分开管理：

```text
远端主源码目录/
├── dexmani_policy/、scripts/、配置与文档
├── data/、robot_data/、experiments/、pretrained_models/ → 持久存储
├── logs/                                             → 训练控制台日志
└── outputs/remote_sources/<唯一标识>/                 → 每次启动的源码副本
```

目录和解释器位置以脚本为准。换服务器时应同时调整这些入口，避免 launcher、数据同步和实验拉取指向不同存储。

## 启动与源码副本

### 常用命令

launcher 选项放在 config 和 task 前面，Hydra overrides 放在后面：

```bash
bash scripts/remote/train_remote.sh --dry-run dp3 <task>
bash scripts/remote/train_remote.sh --gpus 0 dp3 <task> 'training.seed=42'
bash scripts/remote/train_remote.sh --fg --gpus 0 dp3 <task> '+max_updates=4'
bash scripts/remote/train_remote.sh --gpus 0,1,2,3 ddp/sat <task>
bash scripts/remote/train_remote.sh --sync-data dp3 <task>
```

`--dry-run` 仍访问 SSH，解析远端路径并预览 rsync；它不启动训练、不创建本次源码目录，也不完成数据/GPU 验证。只想在本地检查配置时使用：

```bash
python dexmani_policy/smoke_test.py --config-only dp3
python -m dexmani_policy.utils.config \
  --config-name dp3 'task_name=<task>'
```

resolver 输出实际 Dataset 路径；相对数据路径与训练入口一致，按当前源码的项目根目录解析，不依赖调用目录。加 `--check` 只验证目录存在，不加载 Zarr 或证明训练可运行。多任务和 resume 均按实际配置解析，不能仅凭 task 名推断数据位置。

### 启动过程

正式启动依次执行：

1. 检查 SSH，原子创建新的 `outputs/remote_sources/<唯一标识>/`。
2. 同步源码；指定 `--sync-data` 时再上传或更新数据。
3. 在副本中连接共享 `robot_data`、`data`、`experiments`、`pretrained_models` 和 `logs`。
4. 使用副本中的 resolver 检查新训练或续训所需数据目录。
5. 显示 GPU、磁盘状态，提交训练。

源码或数据检查失败时不启动训练。GPU 查询和磁盘输出主要用于诊断；可获取 GPU 数量时会检查显式设备编号，但不会预估 OOM 或保证剩余磁盘足够。

训练从副本目录以 `python -m dexmani_policy.train` 或 `train_ddp` 启动，使导入绑定到本次源码。之后同步主源码不改变已运行副本；数据和权重仍共享，修改共享数据会影响引用它的运行。源码副本保留供追踪，不自动清理。

### 配置与并行运行

`ddp/` config 选择多卡入口，实际可见 GPU 数必须与 `training.num_gpus` 一致。有效 batch 为每卡 batch × world size × gradient accumulation；改变 GPU 数不应被误认为保持了实验配方。

config 名用于启动，`policy_name` 决定实验目录，两者可能不同。多任务名称以 `+` 连接；带列表、空格或特殊字符的 override 应整体引用。

后台模式使用项目专用 `tmux -L dexmani_policy`，每次创建唯一 session 和控制台日志，不复用或杀掉旧 session。任务退出后 session 消失，日志保留；实际名称和路径以 launcher 打印值为准。`--fg` 通过前台 SSH 运行，适合查看启动错误。

### 续训

```bash
bash scripts/remote/train_remote.sh --gpus 0 dp3 <task> \
  '+resume_from=experiments/dp3/<task>/<run>'
```

相对路径通过共享实验目录解析，恢复仍写入新 run。显式 task 必须匹配保存身份；模型、数据配方、world size 和训练计划沿用 checkpoint，允许覆盖项见 [Checkpoint 与续训](项目架构.md#checkpoint-与续训)。

## 同步边界

### 源码

```bash
bash scripts/remote/sync_code.sh --dry-run
bash scripts/remote/sync_code.sh
```

默认同步更新远端主源码，使用 `--delete` 移除目标中已经不在本地的源码。`.git`、本地工具配置、缓存、数据、实验、预训练权重和运行输出被排除；数据与实验根还受到删除保护。

`--dest <absolute_directory>` 用于新的独立副本，实际目标必须为空；不能用它覆盖已填充的运行副本。dry-run 可预览尚不存在的目标。日常训练由 launcher 自动调用源码同步，无需每次手动更新主源码。

### 数据

```bash
bash scripts/remote/sync_data.sh --dry-run robot_data
bash scripts/remote/sync_data.sh robot_data
bash scripts/remote/sync_data.sh --checksum data
bash scripts/remote/sync_data.sh --pull --dry-run robot_data
```

默认 local → remote，可选择 `all`、`data` 或 `robot_data`。默认按 size/mtime 更新已有文件，不删除目标独有文件；`--checksum` 改为内容比较，`--pull` 反转方向。

`--prune` 会删除目标独有文件且没有回收站；push 时删除远端，pull 时删除本地。需要镜像清理时，先用完全相同参数加 `--dry-run` 核对方向和删除清单。

这些目录是共享存储，不属于运行源码快照。已经用于实验的数据应保留，生成新数据时使用新路径和 `data_revision`。

### 实验结果

```bash
bash scripts/remote/sync_down.sh --list
bash scripts/remote/sync_down.sh <policy>/<task>/<run> --dry-run
bash scripts/remote/sync_down.sh <policy>/<task>/<run>
bash scripts/remote/sync_down.sh <policy>/<task>/<run> --with-wandb
```

不提供子路径时拉取全部 experiments。该入口只从远端拉取，不删除本地实验；`--list` 和 `--dry-run` 仍访问 SSH。

同步先按 checksum 核对双方已有 `config.yaml`。任何字节差异，包括仅格式变化，均视为冲突并中止；它不验证同名大权重的内容身份。通过后分三阶段：

| 阶段 | 更新内容 |
| --- | --- |
| 1 | 下载新文件，已有不可变产物不覆盖，排除临时文件和后续可变项 |
| 2 | 按 size/mtime 更新 `metrics.jsonl`、`checkpoints/latest.pt`、`vqvae_hand_best.pt` |
| 3 | 按 checksum 更新 `best_ckpt.json` 和各次 selection 的 `best_ckpt_selection.json` |

checkpoint 和选择结果先于引用它们的指针拉取。新文件传输不组合 `--partial` 与 `--ignore-existing`，避免残缺正式文件在下次同步时被跳过。任一 rsync 非零退出都会中止后续阶段，包括源文件消失的退出码 24。

这不是整个实验目录的事务快照。远端正在发布产物时可能需要重试；恢复、评测前仍应确认被引用 checkpoint 和选择证据齐全。VQ、LAMP 和 Temporal 产物若放在 experiments 中也由此拉取；位于 data 中的产物使用 `sync_data.sh`。

### W&B

实验拉取默认排除 W&B。需要上传 offline run 时，先带 `--with-wandb` 拉取，再在本地预览并同步：

```bash
bash scripts/utils/wandb_sync.sh --dry-run --all
bash scripts/utils/wandb_sync.sh --all
```

上传需要本地 W&B 凭据和网络。该脚本先完整发现所选 offline runs，再开始上传；拉取产物本身不等于已上传到 W&B。

## 监控与停止

### 训练指标与控制台

```bash
bash scripts/remote/tail_log.sh <policy> <task>
bash scripts/remote/tail_log.sh <policy> <task> <run>
bash scripts/remote/stop_remote.sh --list
```

`tail_log.sh` 跟随 `metrics.jsonl`。未指定 run 时，按实验 `config.yaml` 的修改时间选择候选，名称用于稳定打破并列；不是简单按目录名判断最新。指定 run 时目录必须存在，指标文件可以稍后生成。

只有 SSH 不可达时才回退读取本地实验；远端查询出错会报告失败。metrics 不包含全部 stdout/stderr，启动异常和 traceback 应查看 launcher 打印的 `logs/<session>.log`，或按其 Attach 命令连接 tmux。

### 停止训练

```bash
bash scripts/remote/stop_remote.sh <session>
bash scripts/remote/stop_remote.sh --all
```

脚本只管理项目专用 tmux socket 中的训练 session。默认发送一次 SIGINT，轮询等待约 30 秒，SSH 耗时另计；Trainer 在逻辑更新边界退出，已有 optimizer update 时尝试保存恢复点。

超过等待时间仍在运行会报错并保留进程。显式 `--force` 才允许等待后强制结束所选 session；强制终止或再次中断可能跳过保存。session 消失不证明 checkpoint 已完整保存，应查看日志和产物。

空 session 列表正常返回；SSH/tmux 查询失败不会被当成空列表。`scripts/utils/clean_experiments.sh` 只读盘点实验，不会删除或移动文件，也不清理源码副本。

## 常见排查

| 现象 | 核对内容 |
| --- | --- |
| SSH 不可达 | Host 别名、网络、身份文件与非交互登录；可用 `ssh -v` 查看连接过程 |
| Dataset 缺失 | resolver 输出、override、共享链接和同步方向；不要按 task 名猜路径 |
| 缺少预训练权重 | 配置路径与共享 `pretrained_models/`；数据同步不包含该目录 |
| GPU 不匹配或 OOM | 可见设备、`training.num_gpus`、每卡 batch 与其他进程；launcher 状态输出不等于训练验证 |
| 找不到新指标 | 实际 `policy_name`、run 目录、控制台启动日志和首次 update 是否完成 |
| 实验拉取失败 | config 冲突、空间、rsync 退出码；不要强行覆盖冲突配置或接受残缺 checkpoint |
| session 已结束 | 检查控制台退出码和保存日志，区分正常完成、报错与强制终止 |

## 脚本导航

| 脚本 | 用途 |
| --- | --- |
| `scripts/remote/train_remote.sh` | 源码副本、数据检查、单卡/DDP 启动 |
| `scripts/remote/sync_code.sh` | 主源码镜像或空目录副本同步 |
| `scripts/remote/sync_data.sh` | 数据 push/pull 与可选镜像清理 |
| `scripts/remote/sync_down.sh` | 实验与可选 W&B 产物拉取 |
| `scripts/remote/tail_log.sh` | 跟随训练指标 |
| `scripts/remote/stop_remote.sh` | 列出和停止项目训练 session |

训练完成后按 [仿真评测机制](仿真评测机制.md) 进行选点与最终评测。源码副本、实验配置、checkpoint 和评测记录共同用于追踪结果；单独保存命令或控制台日志不足以重建完整实验。
