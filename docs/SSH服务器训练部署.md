# DexMani Policy · SSH 服务器训练部署

所有脚本从本地仓库执行，显式传入 SSH host 和绝对远端路径，不修改脚本中的机器配置。本地需要 Bash、SSH、rsync；远端需要 Bash、rsync、tmux 和完整训练环境。

## 数据与来源准备

```bash
HOST=dexserver
REPO=/home/user/dexmani_policy
DATA_ROOT=/data/project
RUN=/data/project/experiments/dp3/pick_apple/run01
PYTHON=/home/user/miniconda3/envs/policy/bin/python

ssh "$HOST" 'mkdir -p /data/project/robot_data /data/project/data'
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local robot_data --remote "$DATA_ROOT/robot_data" --dry-run
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local robot_data --remote "$DATA_ROOT/robot_data"
bash scripts/remote/sync_data.sh --host "$HOST" \
  --local data --remote "$DATA_ROOT/data" --checksum
```

每次同步一个明确目录，默认上传；`--pull` 反转方向，`--checksum` 改用内容比较，`--dry-run` 预览。同步不删除目标独有文件。训练使用的数据版本不就地修改；重新导出或修改加工配方时使用新路径。

Uni3D 初始权重随 `data/pretrained/uni3d/` 上传；HF 初始模型缓存须在远端单独准备，来源设置见 [数据说明](../data/README.md)。从完整 checkpoint 续训或推理不需要初始来源文件。

## 启动与原目录续训

```bash
REMOTE=(--host "$HOST" --repo "$REPO" --data-root "$DATA_ROOT" \
  --run "$RUN" --python "$PYTHON" --gpu 0)
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --dry-run -- \
  dp3 task_name=pick_apple
bash scripts/remote/train_remote.sh "${REMOTE[@]}" -- \
  dp3 task_name=pick_apple training.seed=42

# 中断后从同一 RUN/checkpoints/latest.pt 直接恢复保存配方。
bash scripts/remote/train_remote.sh "${REMOTE[@]}" --resume -- \
  --num-workers 4 --max-updates 100
```

每次启动保留一个新的 `$REPO/outputs/remote_sources/<session>/` 源码副本，代码同步不覆盖已有副本。副本中的 `robot_data/`、`data/`、`pretrained_models/` 指向显式 `DATA_ROOT`，run 单独由绝对路径指定。预训练 encoder 等其他初始来源使用对应远端绝对路径。

`--gpu` 指定一个物理 GPU，通过 `CUDA_VISIBLE_DEVICES` 映射为进程内 `cuda:0`；脚本不查询或自动选择设备。默认后台 `tmux -L dexmani_policy`，输出 Host、Source、Run、Session 和 Log；`--fg` 前台执行。`--dry-run` 只打印已构造命令，不连接远端、不复制源码。训练返回错误可在控制台或 Log 中查看，不以 session 消失推断 checkpoint 完整。

fresh 的方法名和 Hydra overrides 放在 `--` 后；resume 只接受正式恢复 CLI 的运行参数，不再指定方法配方或新输出目录。参数含空格、引号或特殊字符时整体引用。源码和实验产物不自动删除。

需要单独上传代码时，为明确的新空目录执行：

```bash
bash scripts/remote/sync_code.sh --host "$HOST" --dest /home/user/source_snapshot --dry-run
bash scripts/remote/sync_code.sh --host "$HOST" --dest /home/user/source_snapshot
```

源同步排除数据、实验、缓存和运行产物，无删除传播；目标父目录须存在。

## 监控与停止

使用 launcher 打印的确切 session 和 Log 路径：

```bash
ssh "$HOST" 'tmux -L dexmani_policy list-sessions'
ssh -t "$HOST" 'tmux -L dexmani_policy attach -t <session>'
ssh "$HOST" 'tail -F -- "/absolute/repo/logs/<session>.log"'
ssh "$HOST" 'tmux -L dexmani_policy send-keys -t "=<session>:" C-c'
```

收到一次 SIGINT 后，Trainer 在 update 边界保存并退出；重复中断或强杀可能使最近更新未保存。恢复前确认目标 run 与日志状态。

## 下载完成的 run

```bash
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run experiments/dp3/pick_apple/run01 --dry-run
bash scripts/remote/sync_down.sh --host "$HOST" \
  --remote-run "$RUN" --local-run experiments/dp3/pick_apple/run01
```

默认同步停止或已完成的单个 run，单向 rsync，不删除本地结果。已有 `config.yaml` 不同则在传输前失败。保留现有 best 发布链所需顺序：先传新权重与不可变记录，再更新训练状态、观测预训练的 `resume.pt/encoder.pt` 和选点结果，最后更新 `best_ckpt.json`；任一阶段失败即停止后续发布。临时 `.dexmani-publish-*` 文件不传输。同名大权重不做完整内容身份验证，整个目录不是事务快照；运行中查看日志用 tail。

部署公开字段和 best/lock 消费者仍待实际 Real 源码迁移，因此这里继续保留分阶段同步。下载后直接评测或显式选模：

```bash
python -m dexmani_policy.evaluate --checkpoint <run>/checkpoints/latest.pt \
  --weights ema --inference-steps 10 --episodes 100 --seed 1024
python -m dexmani_policy.eval_pipeline \
  --checkpoints <run>/checkpoints/<a>.pt <run>/checkpoints/<b>.pt \
  --weights ema --inference-steps 10 --selection-episodes 25 --test-episodes 100 \
  --output-dir <run>/eval_ckpt_selector/<new-evaluation>
```

训练默认使用 offline W&B，实验下载默认不拉取 W&B；需要 offline run 时给 `sync_down.sh` 加 `--with-wandb`。在本地安装 W&B 的 `policy` 环境中直接同步所需 run，上传凭据与网络由该环境提供：

```bash
conda activate policy
wandb sync "experiments/dp3/pick_apple/run01/wandb/<offline-run目录>"
```
