# DexMani_Policy

面向灵巧手操作研究的个人策略仓库，提供数据读取、策略训练、离线模型选择、仿真评测和已保存策略的推理接口。方法名表示本地研究适配，具体行为以配置和源码为准。

当前实验从 `dexmani_policy/configs/` 进入；历史实验以 checkpoint 内的 `effective_config`、模型状态及对应产物为准，实验目录的 `config.yaml` 用于阅读与交叉核对。

## 安装与配置

要求 Python 3.10+。本地 Shell 入口使用 Conda 的 `policy` 环境。先安装与设备匹配的 Torch / torchvision，再安装项目依赖；版本约束见 [requirements.txt](requirements.txt) 和 [pyproject.toml](pyproject.toml)。

```bash
conda activate policy
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat r3d \
  maniflow multitask_dit lamp_flow temporal_codec_flow
```

部分点云算子需要单独安装与 Torch/CUDA 匹配的 PyTorch3D；仿真评测需要 `dexmani_sim`。模型所需的外部预训练权重由配置指定，可查看 `scripts/utils/download_pretrained.sh`。

```bash
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml
```

配置名用于定位入口，实际策略由 `agent._target_` 决定。当前类路径见 [策略接口](docs/项目架构.md#策略接口)。以下命令从仓库根目录执行，`<...>` 替换为实际值。

## 训练与续训

```bash
bash scripts/training/train.sh dp3 'task_name=<task>' 'training.seed=42'
bash scripts/training/train_ddp.sh ddp/sat 'task_name=<task>' 'training.seed=42'
```

Shell 入口转发 Hydra overrides。数据通常位于 `robot_data/<task>.zarr`，但实际路径由 resolved `dataset` 配置决定；多任务使用各 child 的路径。数据字段、划分和归一化见 [数据接口](docs/项目架构.md#数据接口)。

新 run 默认写入 `experiments/<policy_name>/<task_name>/<时间戳>_seed<seed>/`。已有训练产物的输出目录拒绝重用；并行任务必须使用不同目录。

续训需要显式指定来源，恢复结果写入新目录：

```bash
bash scripts/training/train.sh dp3 \
  '+resume_from=experiments/dp3/<task>/<run>'
```

`resume_from` 接受实验目录或 checkpoint，支持 `~`，相对路径按项目根目录解析。完整恢复使用保存的配方、normalizer、raw/EMA、optimizer、scheduler、RNG 和采样位置；只允许规定的运行参数和数据路径覆盖。显式 `task_name` 必须与保存值一致，不能借续训改变模型、数据配方、world size 或训练预算。详见 [Checkpoint 与续训](docs/项目架构.md#checkpoint-与续训)。

训练与推理均要求绑定配置的 `simple.v4`。旧类路径和历史格式使用对应源码快照，不自动迁移实验产物。

常见配置约束：

- 重新生成数据时使用新路径和新的 `data_revision`；revision 是生产者声明，不是内容哈希。
- 显式 `dataset.split_manifest` 定义最终 episode 集合，需同时设置 `dataset.max_train_episodes=null`、`dataset.val_ratio=0`。
- Diffusion 使用 Gaussian 动作归一化时需设置 `agent.clip_sample=false`；DQ-RISE/VQ 仅支持动作 `auto/limits`。
- `dataset.rgb_keep_uint8=true` 要求 `normalization.rgb=identity`；评测沿用保存的 RGB 预处理配方。
- SAT、LAMP 和 Temporal 策略的 Sparse3DEncoder 要求 `normalization.point_cloud=identity`。

## 手部先验准备

下列多阶段流程必须使用一致的数据、split、窗口和归一化配置。更换任务或数据时，将相同覆盖传给每个阶段，并为新产物选择新路径。

### DQ-RISE

先按目标 Policy 的数据配方训练 VQ，再导出码本：

```bash
VQ_RUN="experiments/vq_hand/pick_apple_messy/$(date +%Y%m%d_%H%M%S)_${RANDOM}"
python -m dexmani_policy.training.train_vq_hand \
  --policy-config dexmani_policy/configs/dqrise.yaml \
  --policy-override task_name=pick_apple_messy \
  --output_dir "$VQ_RUN"
python -m dexmani_policy.training.export_hand_codebook \
  --checkpoint "$VQ_RUN/vqvae_hand_best.pt" \
  --output "$VQ_RUN/codebook.npz"
bash scripts/training/train.sh dqrise \
  task_name=pick_apple_messy codebook_path="$VQ_RUN/codebook.npz"
```

额外覆盖用重复的 `--policy-override` 传入，并同步到 Policy 训练命令。VQ 使用有效训练窗口引用的唯一 action 源行；验证集复用训练统计。有验证集按 `val_mse` 选点，否则按 `train_mse`。支持 joint `7+12` 和 EEF `9+12`，不支持辅助动作尾部。

VQ checkpoint 使用 v4，码本 NPZ 使用 v3；导出已有文件需要显式 `--overwrite`。`codebook_diagnostics.json` 提供查表误差和原型利用率。Policy 加载时核对手部 affine 参数，完整 Policy checkpoint 内嵌码本，推理无需原 NPZ。

### LAMP

```bash
python -m dexmani_policy.training.train_lamp_prior --config lamp_flow
python -m dexmani_policy.training.export_lamp_latents --config lamp_flow
bash scripts/training/train.sh lamp_flow
```

通过 `lamp.prior_path`、`lamp.latent_path` 指定产物，默认位于 `experiments/lamp_prior/<task>/`。训练入口要求 dataset 与 Policy 使用同一路径，缓存还需匹配 prior 哈希、数据身份和划分。

数据须声明 `obs_alignment="obs[t]_before_action[t]"` 和有效 `dt`。只有确认记录时序后，才能为上述三个入口追加 `dataset.obs_alignment=obs_before_action`。历史使用 query 之前的手部命令，episode 开始以初始实测姿态补齐。

当前实现支持单手 joint `7+12` 和同步 SIM；SIM 使用实际执行命令更新历史。Real bridge 尚未提供该历史，加载 LAMP 会明确报错。完整 Policy checkpoint 内嵌 prior 和 latent 统计，推理不依赖外部 prior/cache。

### Temporal hand codec

```bash
python -m dexmani_policy.training.train_temporal_codec --config temporal_codec_flow
bash scripts/training/train.sh temporal_codec_flow
```

通过 `codec.path` 指定产物，默认位于 `experiments/temporal_codec/<task>/codec.pt`。两个入口使用相同的 `codec.*` 和数据配方；数据也须声明上述 observation/action 时序，确认后可在两个入口追加同样的显式断言。

冻结 encoder 的 posterior mean 在线生成 latent 标签，无需离线缓存；手部参考姿态取 query 源帧未增强的实测值。Policy checkpoint 内嵌完整 codec。更换 codec 后应重新训练 Policy，不能直接替换已训练策略中的权重。

先验结构、损失和实验选项见 [手部动作表示](docs/项目架构.md#手部动作表示)；默认超参数以配置和各预训练入口的 `--help` 为准。

## 验证

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
python dexmani_policy/smoke_test.py <config_name> --max-updates 4
```

`--config-only` 检查配置、Hydra target 和策略导入，不加载数据或执行 GPU 算子。完整 smoke 需要 CUDA、真实数据和所需权重，按原训练配方运行有界更新，再检查预测和 raw/EMA 保存恢复；临时目录退出时清理，不创建 W&B run。

smoke 支持多个配置名，默认运行 4 次 optimizer update，不接受 Hydra 字段覆盖。若需覆盖配置或保留短跑产物，使用正式训练入口：

```bash
bash scripts/training/train.sh dp3 'task_name=<task>' '+max_updates=4'
```

`max_updates` 只限制本次执行，不改变总训练计划或 warmup。配置检查、单卡 smoke、多卡训练和闭环性能是不同层次的验证。

## 仿真评测

```bash
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <run>
bash scripts/eval/record_demo.sh <policy_name> <task_name> <run>
```

流水线执行 checkpoint selection 和不录视频的 held-out eval，使用本次选择的固定记录交接；demo 单独运行。也可分别使用 `select_best_ckpt.sh`、`eval_best_ckpt.sh`，按各入口的 `--help` 调整预算、EMA/raw、NFE 和录像。

评测读取已保存实验，不重新组合当前 Policy YAML。`best_ckpt.json` 使用 `best.v1` 推理快照，详细选择证据使用 `selection.v3`；最终 best 评测排除实际用于选点的 `(task, seed)`。正常全零结果仍可完成选择，技术异常不会记成普通任务失败。

结果按调用保存到独立目录，包含 `eval_config.yaml`、实际 checkpoint、推理参数、seed 计划及 episode 明细。协议、参数优先级、指标和结果路径见 [仿真评测机制](docs/仿真评测机制.md)。

## 远端与其他工作流

```bash
bash scripts/remote/train_remote.sh --dry-run <config> <task>
bash scripts/remote/train_remote.sh --gpus 0 <config> <task> 'training.seed=42'
bash scripts/remote/tail_log.sh <policy> <task>
bash scripts/remote/sync_down.sh <policy>/<task>/<run>
```

远端 launcher 为每次运行准备独立源码副本，共享数据、预训练权重和实验目录；dry-run 仍会访问 SSH。环境准备、数据同步、停止训练和结果拉取见 [SSH 服务器训练部署](docs/SSH服务器训练部署.md)。

- `scripts/utils/clean_experiments.sh`：只读实验盘点，不删除或移动产物。
- `scripts/utils/wandb_sync.sh`：同步 W&B offline runs，支持先用 `--dry-run --all` 预览。
- `dexmani_policy/deployment/`：检查和加载已保存策略；真机执行由 `dexmani_real` 管理，接口见 [部署边界](docs/项目架构.md#部署边界)。

比较实验时同时记录数据集合、batch/accumulation、训练预算、选点与最终评测协议。相同 optimizer steps 不代表相同样本量或计算量；episode 的 Wilson 区间也不代表跨训练 seed 方差。

## 仓库导航

| 路径 | 职责 |
| --- | --- |
| `dexmani_policy/configs/` | 当前实验配置与 DDP overlay |
| `dexmani_policy/agents/` | Policy、观测编码器、动作解码器和手部先验 |
| `dexmani_policy/datasets/` | Zarr 读取、窗口采样、增强和划分 |
| `dexmani_policy/training/` | 构建、训练循环、EMA、保存与恢复 |
| `dexmani_policy/evaluation/` | 离线评测共享协议 |
| `dexmani_policy/env_runner/` | 仿真环境与动作执行 |
| `dexmani_policy/deployment/` | 已保存策略的检查与 Real 推理接口 |
| `scripts/` | 本地训练、评测、远端运行和运维入口 |

观测交互组件 `HandSceneEncoder`、`SceneAggregator` 位于 `agents/obs_encoder/interaction/`，当前没有专属策略配置；不要仅凭模块存在推断默认模型已接入。

| 文档 | 内容 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 数据流、接口、归一化、先验和恢复边界 |
| [仿真评测机制](docs/仿真评测机制.md) | selection、held-out seeds、指标与结果 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远端启动、同步、监控和停止 |
| [AGENTS.md](AGENTS.md) | AI 研究协作规则 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 的规则加载入口 |
