# DexMani_Policy

面向灵巧手操作研究的个人策略仓库，提供数据读取、策略训练、离线模型选择、仿真评测和已保存策略的推理接口。方法名表示本地研究适配，具体行为以配置和源码为准。

当前实验从 `dexmani_policy/configs/` 进入；历史实验以 checkpoint 内的 `effective_config`、模型状态及对应产物为准。实验目录的 `config.yaml` 供阅读及评测、部署入口显式核对，checkpoint 加载器不隐式读取旁路配置。

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

`resume_from` 接受实验目录或 checkpoint，支持 `~`，相对路径按项目根目录解析。完整恢复以 checkpoint 的 `effective_config` 为唯一配方来源，恢复 normalizer、raw/EMA、optimizer、scheduler、RNG 和采样位置；只允许规定的运行参数和数据路径覆盖。显式 `task_name` 必须与保存值一致，不能借续训改变模型、数据配方、world size 或训练预算。详见 [Checkpoint 与续训](docs/项目架构.md#checkpoint-与续训)。

训练日志记录标量指标。每次启动（含续训）创建新的 W&B run，默认 `offline`；`workspace.wandb_cfg` 只配置 `project/group/name/mode`。评测入口共用 `eval.video.fps` 设置录像帧率，不经训练日志上传；录像开关与优先级见[仿真评测机制](docs/仿真评测机制.md)。

设置 `workspace.wandb_cfg=null` 可关闭 W&B。续训可覆盖 W&B 的 `project/group/name/mode`，见[完整恢复边界](docs/项目架构.md#完整恢复边界)；独立推理不构造日志组件。

训练与推理均要求绑定配置的 `simple.v4`。旧类路径和历史格式使用对应源码快照，不自动迁移实验产物。

常见配置约束：

- 单卡入口要求 `training.num_gpus=1` 或不设置；多 GPU 配置使用 `train_ddp.py`。`training.loop` 显式声明总更新步数、日志间隔和梯度累积步数。
- 重新生成数据时使用新路径和新的 `data_revision`；revision 是生产者声明，不是内容哈希。
- 显式 `dataset.split_manifest` 定义最终 episode 集合，需同时设置 `dataset.max_train_episodes=null`、`dataset.val_ratio=0`。
- Diffusion 使用 Gaussian 动作归一化时需设置 `agent.clip_sample=false`；DQ-RISE/VQ 仅支持动作 `auto/limits`。
- `dataset.rgb_keep_uint8=true` 要求 `normalization.rgb=identity`；评测沿用保存的 RGB 预处理配方。
- SAT、LAMP 和 Temporal 的 Sparse3DEncoder 要求 `normalization.point_cloud=identity`：原始米制 centers 经位置 MLP 加入 token 内容，相对几何与 RoPE 保留局部结构和相对位移。训练和推理必须沿用同一坐标系与原点。
- Uni3D 的 `pc_in_channels` 固定为整数 `6`，表示内部 patch 宽度；原始 XYZ 输入由 encoder 补零 RGB，不应改为 `3`。这不提供完整 R3D XYZ-only 数据/normalizer 配方。

缺少 `position_embedding.*` 的旧 Sparse3D 策略 checkpoint 会在严格加载时失败；请使用对应历史 checkout，或重新训练新策略。视觉分支变化不要求重训 `hand_data` 和 normalizer 坐标仍匹配的手部 prior/codec/codebook。机制和加载边界见[项目架构](docs/项目架构.md#观测编码)。

## 手部先验准备

三种方法各保留一个策略实现，均为论文机制的机器人/训练框架适配，不能视作已隔离模型、窗口和目标差异的纯动作空间消融。生产默认值见配置；以下示例统一使用 joint action、同一数据 revision 和固定 episode split，各自保存独立产物。

动作空间预训练与码本导出入口位于 `dexmani_policy/training/pretrain/hand_prior/`，共享表示模型位于 `agents/hand_prior/`。观测表征预训练目录为 `training/pretrain/obs/`，当前尚无实现。VQ 的便捷 Shell 入口为 `bash scripts/pretrain/vqvae.sh <task> [VQ options...]`。

```bash
conda activate policy
DATA_ZARR=/path/to/revision.zarr
SPLIT=/path/to/episode_split.json
HAND_RUN="experiments/hand_spaces/$(date +%Y%m%d_%H%M%S)_${RANDOM}"
DATA=("zarr_path=$DATA_ZARR" "dataset.split_manifest=$SPLIT"
      dataset.max_train_episodes=null dataset.val_ratio=0)
```

正式 baseline 必须提供 episode 级 held-out manifest，并在表示预训练、policy 和各 seed 中复用。manifest 所需 `data_revision`、`episode_ids`、trial 分组见[数据与划分](docs/项目架构.md)；缺少这些信息时应核实并重新导出数据，不补造身份。开发 smoke 可无 holdout，但不据此报告泛化。各方法 history/window 合法性不同，需报告实际样本数与 padding 比例。

只支持当前 dexmani_sim/dexmani_real 采集与导出的 Zarr：每行观测在该行动作提交前取得，时序由生产端保证。训练不读取或配置 `obs_alignment`，不支持旧版错位数据，也不执行自动移位。三种表示要求有限正 `dt`；手部物理坐标为本仓 XHand thumb×3、index×3、middle×2、ring×2、pinky×2，单位 radians。预训练与策略训练通过 `hand_data` 核对数据设置，不计算内容哈希；修改数据内容时需更新 `data_revision`。详见[预训练数据与选点](docs/项目架构.md#预训练数据与选点)。

### DQ-RISE

```bash
VQ_DATA=()
for item in "${DATA[@]}"; do VQ_DATA+=(--policy-override "$item"); done
python -m dexmani_policy.training.pretrain.hand_prior.vqvae \
  --policy-config dexmani_policy/configs/dqrise.yaml \
  "${VQ_DATA[@]}" --policy-override action_key=action \
  --output_dir "$HAND_RUN/dqrise"
python -m dexmani_policy.training.pretrain.hand_prior.export_codebook \
  --checkpoint "$HAND_RUN/dqrise/vqvae_hand_best.pt" \
  --output "$HAND_RUN/dqrise/codebook.npz"
bash scripts/training/train.sh dqrise "${DATA[@]}" action_key=action \
  codebook_path="$HAND_RUN/dqrise/codebook.npz"
```

配置默认仍为 EEF `9+12`；示例的 joint `7+12` 是显式覆盖。VQ 使用有效训练窗口引用的唯一 action 源行，有验证集按 `val_mse` 选点，否则按 `train_mse`。VQ checkpoint v5、码本 NPZ v4 保存 `hand_data` 和归一化参数。训练指标保留在 VQ checkpoint，码本仅记录选点及导出诊断；无 holdout 也可导出。`codebook_diagnostics.json` 报告实际 NN→lookup 的误差、利用率和相邻跳变。

### LAMP

```bash
python -m dexmani_policy.training.pretrain.hand_prior.lamp --config lamp "${DATA[@]}" \
  lamp.prior_path="$HAND_RUN/lamp/prior.pt"
bash scripts/training/train.sh lamp "${DATA[@]}" \
  lamp.prior_path="$HAND_RUN/lamp/prior.pt"
```

唯一结构为 `H=Nobs=C=1`、K=8 历史先验、单步 arm/latent offset 回归；训练用 decoded hand MSE 与 latent offset L2，不需要离线 latent 标签。prior 为 `lamp_prior.v3`。

历史严格来自 query 之前的实际执行命令，reset 用初始实测手姿补齐。SIM 每个成功执行步追加处理后的命令。runtime 要求调用者显式传 `action_context={"hand_history": history}`，其中 `history` 是物理 NumPy `[K,12]`；`predict` 不更新历史。执行端按实际提交的命令维护历史，真机闭环 **NOT VERIFIED**。

### Temporal hand codec

```bash
python -m dexmani_policy.training.pretrain.hand_prior.temporal_vae --config temporal_codec_flow "${DATA[@]}" \
  codec.path="$HAND_RUN/temporal/codec.pt"
bash scripts/training/train.sh temporal_codec_flow "${DATA[@]}" \
  codec.path="$HAND_RUN/temporal/codec.pt"
```

保留 H=16、Nobs=2、C=8、12→9 时序 VAE。冻结 encoder 的 posterior mean 在线生成标签；q0 取 query 源帧未增强的实测值，整段固定，完整解码后切片。decoded loss 唯一使用 standardized relative 空间；物理 MAE/RMSE 只作诊断。codec 为 `temporal_hand_codec.v2`。

LAMP/Temporal 的候选选点必须在 KL 达到目标系数后：LAMP 默认 20k steps、2k warmup、beta=1e-3；Temporal 默认 30k、4k 零 KL + 4k ramp、beta=3e-4。有 holdout 按各自验证指标选点，无 holdout 保存最终合格步骤并标记 `final_no_holdout`。不足 schedule 在训练前报错。

所有阶段的数据覆盖必须一致。LAMP/Temporal 默认写入实际 policy 配置的 `agent.prior_path` / `agent.codec_path`，默认分别由 `lamp.prior_path` / `codec.path` 插值。若使用 LAMP `--output`，policy 的 `agent.prior_path` 也应指向该产物。三种预训练入口与 policy 共用窗口、动作布局校验；训练构建时核对数据设置与坐标，不要求两处路径字符串相同。三份配置均声明了可直接覆盖的 `dataset.split_manifest`。已有文件默认不覆盖。

Fresh training 核对数据版本、窗口、划分和手部表示坐标，观测归一化可在 encoder 允许范围内另选；strict resume 使用 checkpoint 内嵌表示并重新核对 `hand_data`，不重新加载外部 artifact。`simple.v4` 的 raw/EMA 均内嵌表示和 normalizer，独立推理无需原 artifact、训练数据或额外表示记录。选点规则在预训练循环中执行，诊断保存在预训练产物；不再保存 Git 状态、内容哈希或参数指纹。旧手部预训练产物及缺少 `hand_data` 的手部训练恢复需重新准备或使用历史提交，普通策略 checkpoint 格式不变。LAMP/Temporal 仅支持 sync。

机制公式、论文对应与固定官方提交差异见[手部动作表示](docs/项目架构.md#手部动作表示)。

## 验证

Smoke 也接受显式 `.yaml` 路径，与预训练入口共用 Hydra 配置组合规则（包括 `defaults`），便于使用临时微型数据/产物。

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
python dexmani_policy/smoke_test.py <config_name> --max-updates 4
python -m unittest discover -s tests -p 'test_*.py'
```

`--config-only` 检查配置、Hydra target 和策略、训练、Dataset、增强类的导入，不加载数据或执行 GPU 算子。完整 smoke 需要 CUDA、真实数据和所需权重，按原训练配方运行有界更新，再检查预测和 raw/EMA 保存恢复；临时目录退出时清理，不创建 W&B run。

CPU 回归使用临时合成数据：`test_utils.py` 检查路径/续训配方、原子写入、RNG 和手部产物恢复；`test_deployment.py` 检查部署配置快照与动作输出；`test_evaluation.py` 检查观测窗口、多任务输入对齐、seed 证据，以及选点、评测和 demo 入口。不替代完整训练 smoke 或仿真、真机闭环。

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

评测读取已保存实验，不重新组合当前 Policy YAML。`best_ckpt.json` 仅引用本次成功的 `selection_result.json`，推理参数与 episode 证据只保存一份；最终 best 评测排除实际用于选点的 `(task, seed)`。正常全零结果仍可完成选择，技术异常不会记成普通任务失败。

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
| `dexmani_policy/utils/` | `config.py` 配置与路径、`runtime.py` 运行工具、`hand_representation.py` 手部数据设置与 KL schedule |
| `dexmani_policy/agents/` | Policy、观测编码器、动作解码器和手部先验 |
| `dexmani_policy/datasets/` | 通用 Dataset、RGB、点云和多任务组合 |
| `dexmani_policy/datasets/hand_prior/` | LAMP 历史命令与 Temporal Codec 参考姿态的数据集 |
| `dexmani_policy/datasets/common/` | 只读存储、窗口与分布式采样、划分、增强和 RGB 预处理 |
| `dexmani_policy/training/` | 组件构建、训练循环、实验管理、状态恢复与校验 |
| `dexmani_policy/training/common/` | EMA、学习率调度、日志与 checkpoint |
| `dexmani_policy/training/pretrain/hand_prior/` | 动作空间预训练：VQ、LAMP、Temporal 及码本导出 |
| `dexmani_policy/training/pretrain/obs/` | 观测表征预训练（当前尚无实现） |
| `dexmani_policy/env_runner/` | 仿真执行；`evaluation.py` 评测、`selection.py` 选点 |
| `dexmani_policy/env_runner/common/` | `seeds.py` 采样与计划校验、`metrics.py` 指标、`video.py` 录像 |
| `dexmani_policy/deployment/` | `metadata.py` 检查保存配置，`runtime.py` 提供 Real 推理接口 |
| `scripts/` | 本地训练、评测、远端运行和运维入口 |

观测交互组件 `HandSceneEncoder`、`SceneAggregator` 位于 `agents/obs_encoder/interaction/`，当前没有专属策略配置；不要仅凭模块存在推断默认模型已接入。

| 文档 | 内容 |
| --- | --- |
| [项目架构](docs/项目架构.md) | 数据流、接口、归一化、先验和恢复边界 |
| [仿真评测机制](docs/仿真评测机制.md) | selection、held-out seeds、指标与结果 |
| [SSH 服务器训练部署](docs/SSH服务器训练部署.md) | 远端启动、同步、监控和停止 |
| [AGENTS.md](AGENTS.md) | AI 研究协作规则 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 的规则加载入口 |

