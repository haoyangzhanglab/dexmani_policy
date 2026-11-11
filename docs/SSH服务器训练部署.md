# SSH 服务器训练部署

本文按“准备 → 连接 → 同步 → 启动 → 监控 → 续训 → 取回”的顺序说明远程训练。除 SSH 配置外，所有命令均在**本地仓库根目录的同一个 Bash 会话**中执行。

## 1. 使用前提

| 位置 | 所需条件 |
| --- | --- |
| 本地 | Bash、SSH、rsync，以及待同步的源码、示教数据和初始化权重 |
| 远端 | Bash、rsync、tmux、可用 GPU，以及已安装项目依赖的 Python 环境 |

下文以 DP3 的 `pick_apple` 任务为例。主机、用户、密钥和路径均需替换；任务名应与实际数据集对应。脚本不会自动安装训练环境。

## 2. SSH 连接与路径

### 配置连接

在本地 `~/.ssh/config` 中添加别名，已有可用配置时直接复用：

```sshconfig
Host dexserver
    HostName server.example.com
    User user
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

### 设置本次实验路径

```bash
HOST=dexserver
REPO=/home/user/dexmani_policy
DATA_ROOT=/data/project
RUN=/data/project/experiments/train/dp3/pick_apple/run01
PYTHON=/home/user/miniconda3/envs/policy/bin/python
```

| 变量 | 含义 |
| --- | --- |
| `HOST` | SSH 主机别名 |
| `REPO` | 远端源码快照与启动日志的存放位置 |
| `DATA_ROOT` | 远端数据根目录，下设 `robot_data/` 和 `data/` |
| `RUN` | 本次实验的远端输出目录；新训练使用新目录，续训复用原目录 |
| `PYTHON` | 远端训练环境的 Python 解释器 |

远端路径均使用绝对路径。测试连接：

```bash
ssh "$HOST"
```

连接成功后，在远端输入 `exit` 返回本地，再执行后续步骤。

## 3. 同步数据与初始化权重

### 创建远端目录

```bash
printf -v REMOTE_DIRS '%q ' "$DATA_ROOT/robot_data" "$DATA_ROOT/data"
ssh "$HOST" "mkdir -p -- $REMOTE_DIRS"
```

### 预览并同步

先预览示教数据和权重的同步内容：

```bash
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local robot_data --remote "$DATA_ROOT/robot_data" --dry-run
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local data --remote "$DATA_ROOT/data" --dry-run
```

确认后执行同步：

```bash
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local robot_data --remote "$DATA_ROOT/robot_data"
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local data --remote "$DATA_ROOT/data"
```

每次同步一个目录，目标独有文件会保留；需要按内容比较时加 `--checksum`。这里的 `--dry-run` 会连接远端比较文件，但不传输文件内容。

新训练按策略准备[初始化权重](../data/README.md)。项目训练生成的编码器或码本需另外同步，并在训练参数中指定远端绝对路径。完整 checkpoint 续训不需要初始化文件，但仍需训练数据。

训练期间保持数据内容和顺序不变；重新导出数据时使用新路径和新实验目录。

## 4. 启动新训练

### 公共参数与启动预览

```bash
REMOTE=(--host "$HOST" --repo "$REPO" --data-root "$DATA_ROOT" \
  --run "$RUN" --python "$PYTHON" --gpu 0)

bash scripts/remote/train_remote.sh "${REMOTE[@]}" --dry-run -- \
  dp3 task_name=pick_apple training.seed=42
```

启动脚本的 `--dry-run` 打印包含 `CUDA_VISIBLE_DEVICES` 的启动命令，不连接远端、不同步源码，也不检查远端环境。方法名和训练配置覆盖放在 `--` 后。

`REMOTE` 保存定义时的变量值；更换实验路径后，需要重新执行上面的数组定义。

### 后台启动

```bash
bash scripts/remote/train_remote.sh "${REMOTE[@]}" -- \
  dp3 task_name=pick_apple training.seed=42
```

脚本自动同步源码，为本次启动保留独立快照，并链接到已准备的数据和权重目录。默认通过 tmux 在后台运行，训练产物写入 `RUN`，终端输出写入日志。启动信息会打印 Source、Session 和 Log。一个实验目录同时只启动一个训练进程。

### 前台运行与 GPU 选择

需要前台运行时，在 `--` 前加入 `--fg`：

```bash
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --fg -- \
  dp3 task_name=pick_apple training.seed=42
```

后台与前台命令二选一。前台输出直接显示在终端，SSH 连接需保持。

`--gpu` 设置远端 GPU 可见性。前台和后台启动都会将本次 `CUDA_VISIBLE_DEVICES` 显式传给训练进程；复用已有 tmux 服务时也会使用本次设置。修改公共参数中的 GPU 编号即可切换，进程内部的 `cuda:0` 表示第一张可见 GPU。

## 5. 查看进度与停止

### 查看后台会话和日志

用启动信息中实际打印的 Log 路径替换占位符：

```bash
ssh "$HOST" 'tmux -L dexmani_policy list-sessions'
ssh "$HOST" 'tail -F -- "/absolute/path/to/<session>.log"'
```

退出 `tail` 只结束日志查看，不停止训练。

### 正常停止

用实际 Session 替换 `<session>`，向后台训练发送一次 `Ctrl-C`：

```bash
ssh "$HOST" 'tmux -L dexmani_policy send-keys -t "=<session>:" C-c'
```

等待当前更新结束并保存。重复中断或强杀可能丢失最近更新；续训或下载前，确认进程已退出，并检查日志与 checkpoint。

## 6. 原目录续训

保持原来的 `RUN`，确认已有 `RUN/checkpoints/latest.pt`，且没有训练进程继续写入该目录。

先预览续训命令，再启动：

```bash
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --resume --dry-run -- \
  --num-workers 4
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --resume -- \
  --num-workers 4
```

续训恢复 checkpoint 保存的训练配方与状态，不再次传入方法名或模型配置。`--` 后传续训入口的运行参数，例如 `--num-workers`；可追加 `--max-updates 100` 限制本次执行步数，不改变实验总预算。延长预算使用 `--total-steps 150000`，已完成的实验也可继续；本次数据从新一轮遍历开始，不恢复旧 batch 或 RNG。

`metrics.jsonl` 只追加当前指标，恢复时不读取、修复或截断历史。旧坏行、崩溃残行、重复或回退后的 step 均保留，不保证整个历史可完整解析；实际写入失败仍报错。初始 config.yaml 不被覆盖，checkpoint.effective_config 是恢复配方。

需要前台续训时，在 `--` 前加入 `--fg`。源码仍由启动脚本同步，数据沿用已准备的远端目录。

## 7. 取回实验结果

仅下载已停止或完成的实验。先指定本地目录并预览：

```bash
LOCAL_RUN=experiments/train/dp3/pick_apple/run01
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run "$LOCAL_RUN" --dry-run
```

确认后下载：

```bash
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run "$LOCAL_RUN"
```

| 同步行为 | 说明 |
| --- | --- |
| 预览 | `--dry-run` 会连接远端比较文件，不下载内容 |
| 实验内容 | 同步实验目录，默认排除 `wandb/` 和发布临时文件 |
| W&B 数据 | 如需包含，在预览与下载命令中均加入 `--with-wandb` |
| 同名文件 | 可能更新，本地独有文件保留 |
| 符号链接 | 按原样复制，外部链接目标需另行同步 |

传输成功后再使用本地结果。启动日志位于 `REPO/logs/`，不属于 `RUN`；训练运行中通过远端日志查看进度。

后续选模与测试见[仿真评测机制](仿真评测机制.md)。复现选模胜者时，应同时保留 checkpoint、raw/EMA 权重选择、推理步数与执行长度。
