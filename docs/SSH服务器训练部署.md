# SSH 服务器训练部署

以下命令从本地仓库根目录执行。本地需要 Bash、SSH、rsync；远端需要 Bash、rsync、tmux，以及已安装项目依赖的 Python 环境。示例中的主机、用户、端口、密钥与路径均需替换为实际信息。

## 连接与路径

可在本地 `~/.ssh/config` 中配置别名；已有可用配置时直接复用：

```sshconfig
Host dexserver
    HostName server.example.com
    User user
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

在同一个本地 Bash 会话中设置路径并确认连接：

```bash
HOST=dexserver
REPO=/home/user/dexmani_policy
DATA_ROOT=/data/project
RUN=/data/project/experiments/train/dp3/pick_apple/run01
PYTHON=/home/user/miniconda3/envs/policy/bin/python

ssh "$HOST"
```

确认后用 `exit` 回到本地。`REPO` 是远端源码与日志的存放位置，`DATA_ROOT` 下存放 `robot_data/` 和 `data/`，`RUN` 指定本次实验目录，`PYTHON` 指向远端训练环境的解释器。

## 同步数据与初始化权重

```bash
printf -v REMOTE_DIRS '%q ' "$DATA_ROOT/robot_data" "$DATA_ROOT/data"
ssh "$HOST" "mkdir -p -- $REMOTE_DIRS"
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local robot_data --remote "$DATA_ROOT/robot_data" --dry-run
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local robot_data --remote "$DATA_ROOT/robot_data"
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local data --remote "$DATA_ROOT/data"
```

同步前先用 `--dry-run` 预览；可加 `--checksum` 按内容比较。每次指定一个目录，目标独有文件会保留。

新训练按所选策略准备[初始化权重](../data/README.md)。项目自产的编码器或码本需另外同步，并在训练参数中使用对应的远端绝对路径。完整 checkpoint 续训不需要初始化文件。

训练期间保持数据内容和顺序不变；重新导出数据时使用新路径和新实验目录。

## 启动与原目录续训

```bash
REMOTE=(--host "$HOST" --repo "$REPO" --data-root "$DATA_ROOT" \
  --run "$RUN" --python "$PYTHON" --gpu 0)

bash scripts/remote/train_remote.sh "${REMOTE[@]}" --dry-run -- \
  dp3 task_name=pick_apple training.seed=42
bash scripts/remote/train_remote.sh "${REMOTE[@]}" -- \
  dp3 task_name=pick_apple training.seed=42
```

启动脚本自动创建 `REPO` 所需目录并同步本地源码，每次保留独立副本；数据和初始化权重使用前一步准备的远端目录。`--gpu` 指定远端物理 GPU 编号，默认在 tmux 后台运行，加 `--fg` 可前台运行。启动脚本的 `--dry-run` 只打印命令，不连接或同步远端。

新训练的方法名及配置覆盖放在 `--` 后，输出统一写入新的 `RUN` 目录。一个实验目录同时只启动一个训练进程。

中断后确认原进程已退出，使用相同 `RUN` 恢复：

```bash
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --resume -- \
  --num-workers 4
```

续训从 `RUN/checkpoints/latest.pt` 恢复保存的训练配方与状态，不再次传入方法名或模型配置。需要限制本次执行时可追加 `--max-updates 100`；它只限制本次调用，不改变实验总预算。

## 监控与停止

后台训练的输出写入启动时打印的 Log，使用 `tail` 查看；tmux 用于管理会话。用实际 Session 和 Log 替换以下占位符：

```bash
ssh "$HOST" 'tmux -L dexmani_policy list-sessions'
ssh "$HOST" 'tail -F -- "/absolute/path/to/<session>.log"'
ssh "$HOST" 'tmux -L dexmani_policy send-keys -t "=<session>:" C-c'
```

退出 `tail` 只结束日志查看；停止训练使用上面的 `send-keys` 命令发送一次 `Ctrl-C`，等待当前更新结束并保存。重复中断或强杀可能丢失最近更新。恢复前检查日志与 checkpoint。

## 取回实验结果

仅下载已停止或完成的实验：

```bash
LOCAL_RUN=experiments/train/dp3/pick_apple/run01
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run "$LOCAL_RUN" --dry-run
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run "$LOCAL_RUN"
```

同步完整实验目录，传输成功后再使用。已有同名文件可能更新，本地独有文件会保留；链接按原样复制，外部链接目标或其他实验目录需自行同步。运行中查看进度使用远端日志。

下载后的选模与测试遵循[仿真评测机制](仿真评测机制.md)。使用选模胜者时，同时保留其 checkpoint、权重选择、推理步数与执行长度。
