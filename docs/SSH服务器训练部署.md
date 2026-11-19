# SSH 服务器训练部署

在本地仓库根目录的同一个 Bash 会话中执行。示例使用 DP3 训练 `pick_apple_messy`。本地需要 SSH 和 rsync，远端需要 Bash、rsync、tmux、GPU 和已安装项目依赖的 Python 环境。

## 准备与同步

将 `HOST` 改为 SSH 地址或别名，其余变量改为远端绝对路径。新训练使用新的 `RUN`。

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

同步命令加 `--dry-run` 可连接远端预览。策略若需要另行训练的编码器或码本，将权重同步到远端，再用初始化参数指定绝对路径。 [初始化权重](../data/README.md)

## 启动与日志

新训练会同步一份源码快照，默认在远端 tmux 后台运行，产物写入 `RUN`。

```bash
REMOTE=(--host "$HOST" --repo "$REPO" --data-root "$DATA_ROOT" \
  --run "$RUN" --python "$PYTHON" --gpu 0)

bash scripts/remote/train_remote.sh "${REMOTE[@]}" -- \
  dp3 task_name=pick_apple_messy training.seed=42
```

在 `--` 前加 `--dry-run` 可离线预览启动命令，加 `--fg` 可在 SSH 前台运行。`--gpu 0` 选择远端卡号，进程内用 `cuda:0` 访问所选 GPU。查看日志时，将 `LOG` 替换为脚本打印的路径。

```bash
LOG=/absolute/path/to/training.log
ssh "$HOST" "tail -F -- $(printf '%q' "$LOG")"
```

停止训练时，用脚本打印的 Session 替换示例值，向训练进程发送一次 Ctrl-C。等待训练保存并退出后再续训或取回结果。

```bash
SESSION=dex_20260101_120000_1234_5678
ssh "$HOST" "tmux -L dexmani_policy send-keys -t '=$SESSION:' C-c"
```

## 续训与取回

续训沿用原来的 `RUN`，读取首次训练的源码快照及 `checkpoints/latest.pt`。每个实验目录同时只运行一个训练进程。训练和续训使用同一份 Zarr，重新导出的数据用新路径和新实验目录。

```bash
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --resume
```

延长训练预算时，在命令末尾加 `-- --total-steps 150000`。训练退出后可取回结果，加 `--dry-run` 可预览，加 `--with-wandb` 可同时取回 W&B 数据。

```bash
LOCAL_RUN=experiments/train/dp3/pick_apple_messy/run01
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run "$LOCAL_RUN"
```

[仿真评测机制](仿真评测机制.md)
