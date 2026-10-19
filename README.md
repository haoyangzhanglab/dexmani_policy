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
  maniflow multitask_dit lamp temporal_codec_flow
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

三种方法各保留一个策略实现，均为论文机制的机器人/训练框架适配，不能视作已隔离模型、窗口和目标差异的纯动作空间消融。生产默认值见配置；以下示例统一使用 joint action、同一数据 revision 和固定 episode split，各自保存独立产物。

```bash
conda activate policy
DATA_ZARR=/path/to/revision.zarr
SPLIT=/path/to/episode_split.json
HAND_RUN="experiments/hand_spaces/$(date +%Y%m%d_%H%M%S)_${RANDOM}"
DATA=("zarr_path=$DATA_ZARR" "dataset.split_manifest=$SPLIT"
      dataset.max_train_episodes=null dataset.val_ratio=0)
```

正式 baseline 必须提供 episode 级 held-out manifest，并在表示预训练、policy 和各 seed 中复用。manifest 所需 `data_revision`、`episode_ids`、trial 分组见[数据与划分](docs/项目架构.md)；缺少这些信息时应核实并重新导出数据，不补造身份。开发 smoke 可无 holdout，但不据此报告泛化。各方法 history/window 合法性不同，需报告实际样本数与 padding 比例。

LAMP/Temporal 要求数据声明 `obs_alignment="obs[t]_before_action[t]"`，三种表示要求有限正 `dt`。只有确认采集时序后，才可向 LAMP/Temporal 的两个阶段同时追加 `dataset.obs_alignment=obs_before_action`；不要从索引猜测。手部物理坐标为本仓 XHand thumb×3、index×3、middle×2、ring×2、pinky×2，单位 radians。

### DQ-RISE

```bash
VQ_DATA=()
for item in "${DATA[@]}"; do VQ_DATA+=(--policy-override "$item"); done
python -m dexmani_policy.training.train_vq_hand \
  --policy-config dexmani_policy/configs/dqrise.yaml \
  "${VQ_DATA[@]}" --policy-override action_key=action \
  --output_dir "$HAND_RUN/dqrise"
python -m dexmani_policy.training.export_hand_codebook \
  --checkpoint "$HAND_RUN/dqrise/vqvae_hand_best.pt" \
  --output "$HAND_RUN/dqrise/codebook.npz"
bash scripts/training/train.sh dqrise "${DATA[@]}" action_key=action \
  codebook_path="$HAND_RUN/dqrise/codebook.npz"
```

配置默认仍为 EEF `9+12`；示例的 joint `7+12` 是显式覆盖。VQ 使用有效训练窗口引用的唯一 action 源行，有验证集按 `val_mse` 选点，否则按 `train_mse`。VQ checkpoint v4、码本 NPZ v3 必须包含 source contract；旧产物缺少契约时需用当前代码重新准备，不能只复制 affine。`codebook_diagnostics.json` 报告实际 NN→lookup 的误差、利用率和相邻跳变。

### LAMP

```bash
python -m dexmani_policy.training.train_lamp_prior --config lamp "${DATA[@]}" \
  lamp.prior_path="$HAND_RUN/lamp/prior.pt"
bash scripts/training/train.sh lamp "${DATA[@]}" \
  lamp.prior_path="$HAND_RUN/lamp/prior.pt"
```

唯一结构为 `H=Nobs=C=1`、K=8 历史先验、单步 arm/latent offset 回归；训练用 decoded hand MSE 与 latent offset L2，不需要离线 latent 标签。prior 为 `lamp_prior.v2`，旧 v1 和旧 Flow policy 不自动迁移，历史实验使用历史提交。

历史严格来自 query 之前的实际执行命令，reset 用初始实测手姿补齐。SIM 每个成功执行步追加处理后的命令。runtime 要求调用者显式传 `action_context={"hand_history": history}`，其中 `history` 是物理 NumPy `[K,12]`；`predict` 不更新历史。外部 `dexmani_real` 尚未接入该历史，真机闭环 **NOT VERIFIED**。

### Temporal hand codec

```bash
python -m dexmani_policy.training.train_temporal_codec --config temporal_codec_flow "${DATA[@]}" \
  codec.path="$HAND_RUN/temporal/codec.pt"
bash scripts/training/train.sh temporal_codec_flow "${DATA[@]}" \
  codec.path="$HAND_RUN/temporal/codec.pt"
```

保留 H=16、Nobs=2、C=8、12→9 时序 VAE。冻结 encoder 的 posterior mean 在线生成标签；q0 取 query 源帧未增强的实测值，整段固定，完整解码后切片。decoded loss 唯一使用 standardized relative 空间；物理 MAE/RMSE 只作诊断。codec 保持 `temporal_hand_codec.v1` 张量格式，旧文件缺来源/选点信息时必须重新预训练。

LAMP/Temporal 的候选选点必须在 KL 达到目标系数后：LAMP 默认 20k steps、2k warmup、beta=1e-3；Temporal 默认 30k、4k 零 KL + 4k ramp、beta=3e-4。有 holdout 按各自验证指标选点，无 holdout 保存最终合格步骤并标记 `final_no_holdout`。不足 schedule 在训练前报错。

所有阶段的数据覆盖必须一致；若使用 LAMP `--output`，policy 的 `lamp.prior_path` 必须同步。三种预训练入口与 policy 共用配置校验，在读取数据前检查窗口、动作布局和产物路径是否对齐；三份配置均声明了可直接覆盖的 `dataset.split_manifest`。已有文件默认不覆盖。

Fresh training 校验来源、窗口、划分和归一化；strict resume 使用 checkpoint 内嵌表示并重新核对当前数据，不重新加载外部 artifact。`simple.v4` 的 raw/EMA 均内嵌表示、normalizer 和 `action_space_artifact` 元信息，独立推理无需原 artifact 或训练数据。来源以单一 source contract 记录，源行/窗口保留计数与指纹，不重复保存完整索引列表。LAMP/Temporal 仅支持 sync。

机制公式、论文对应与固定官方提交差异见[手部动作表示](docs/项目架构.md#手部动作表示)。

## 验证

Smoke 也接受显式 `.yaml` 路径，与预训练入口共用 Hydra 配置组合规则（包括 `defaults`），便于使用临时微型数据/产物。定向测试通过真实预训练/导出入口创建临时产物，默认 CPU；设 `DEXMANI_TEST_DEVICE=cuda:0` 可验证实际 GPU Trainer 构建入口。


```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
python dexmani_policy/smoke_test.py <config_name> --max-updates 4
python -m pytest -q tests/test_hand_action_spaces.py
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
| [三种手部动作空间 Codex CLI 任务书](docs/codex_hand_action_spaces_task.md) | DQ-RISE、LAMP、Temporal 实施约束、验收证据及未验证项 |
| [配套 Codex CLI prompt](docs/prompts/hand_action_spaces_codex_cli.txt) | 复制给 Codex CLI 执行上述任务书 |
| [AGENTS.md](AGENTS.md) | AI 研究协作规则 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 的规则加载入口 |

