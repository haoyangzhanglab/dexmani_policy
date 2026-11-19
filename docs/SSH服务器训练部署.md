# SSH 服务器训练部署

以下命令在本地仓库根目录的同一个 Bash 会话中运行，以 DP3 的 `pick_apple_messy` 为例。本地需有 SSH 和 rsync，远端需有 Bash、rsync、tmux、GPU 和已安装项目依赖的 Python 环境。

## 准备与同步

将主机和远端绝对路径替换为实际值。已有 SSH 别名时可以直接赋给 `HOST`。

```bash
HOST=user@server.example.com
REPO=/home/user/dexmani_policy
DATA_ROOT=/data/project
RUN=/data/project/experiments/train/dp3/pick_apple_messy/run01
PYTHON=/home/user/miniconda3/envs/policy/bin/python

printf -v REMOTE_DIRS '%q ' "$DATA_ROOT/robot_data" "$DATA_ROOT/data"
ssh "$HOST" "mkdir -p -- $REMOTE_DIRS"
for name in robot_data data; do
  bash scripts/remote/sync_data.sh --host "$HOST" \
    --local "$name" --remote "$DATA_ROOT/$name"
done
```

同步命令加 `--dry-run` 可连接远端预览。按策略准备权重，将另行训练的编码器或码本同步到远端，并通过初始化参数传入其绝对路径。 [初始化权重](../data/README.md)

## 启动与日志

新训练选用新的 `RUN`。设置 `REMOTE` 后若更改变量，重新定义该数组。

```bash
REMOTE=(--host "$HOST" --repo "$REPO" --data-root "$DATA_ROOT" \
  --run "$RUN" --python "$PYTHON" --gpu 0)

bash scripts/remote/train_remote.sh "${REMOTE[@]}" -- \
  dp3 task_name=pick_apple_messy training.seed=42
```

在 `--` 前加 `--dry-run` 可在本地预览启动命令，无需连接远端。加 `--fg` 在当前 SSH 连接前台运行。`--gpu` 选择远端 GPU，训练进程用 `cuda:0` 访问这张卡。

脚本同步本次训练的源码，默认在 tmux 后台运行。实验产物写到 `RUN`。用启动信息中的 Session 和 Log 替换下面的示例值。

```bash
SESSION=dex_20260101_120000_1234_5678
LOG=/absolute/path/to/training.log
ssh "$HOST" "tail -F -- $(printf '%q' "$LOG")"
```

退出 `tail` 只结束日志查看。需要停止训练时，单独运行以下命令发送一次 Ctrl-C。

```bash
ssh "$HOST" "tmux -L dexmani_policy send-keys -t '=$SESSION:' C-c"
```

等待训练保存并退出后再续训或下载。每个实验目录同时只启动一个训练进程。

## 续训与取回

续训沿用原来的 `RUN`、首次训练的源码和 `checkpoints/latest.pt`。训练和续训期间保持 Zarr 内容与顺序不变。重新导出的数据应使用新路径和新实验目录。

```bash
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --resume
```

需要延长训练预算时，在命令末尾加 `-- --total-steps 150000`。

训练结束后取回结果。同步命令可加 `--dry-run` 预览，或加 `--with-wandb` 包含 W&B 数据。启动日志可在远端另行查看。

```bash
LOCAL_RUN=experiments/train/dp3/pick_apple_messy/run01
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run "$LOCAL_RUN"
```

复现选模结果需保留胜者的 checkpoint、raw/EMA 权重选择、推理步数和执行长度。 [仿真评测机制](仿真评测机制.md)
