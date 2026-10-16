# DexMani_Policy

面向灵巧手操作研究的个人策略仓库。目标是让新的机器人学习想法能够以尽量小的改动快速落地、验证和比较，而不是建设一个通用、长期兼容的软件平台。

当前研究内容会持续变化，可能涉及 3D / 点云、多模态感知、灵巧操作、生成式策略等方向。具体模型结构、输入输出和训练方式始终以**当前 config 与源码**为准，不在 README 中固化。

> **事实来源**：当前实现以 config 和源码为准；新 checkpoint 的 `effective_config` 绑定该次运行的配方，旁边 resolved `config.yaml` 用于阅读和交叉核对。旧产物的支持边界见下文。README 只维护稳定入口、常用命令和研究工作流。

## 研究定位

这个仓库优先服务于快速研究迭代：

- **问题驱动**：先明确研究假设和主要变量，再改代码。
- **控制变量**：尽量保持 baseline 其他部分不变，只修改真正需要验证的机制。
- **直接实现**：研究特有逻辑优先放在最自然的位置，不为潜在复用提前搭框架。
- **结果可追溯**：config、实验目录、日志和 checkpoint 应足以还原一次实验。

## 快速开始

项目要求 Python 3.10+，研究环境使用 `policy`。依赖以 `requirements.txt` 的实测版本为准，`pyproject.toml` 与之相容。当前已验证的本地组合为 Python 3.10.20、Torch 2.4.1+cu124、torchvision 0.19.1+cu124；CUDA 构建版本不代表 GPU 驱动已通过验证。

已有 `policy` 环境时，安装顺序为：

```bash
conda activate policy
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python dexmani_policy/smoke_test.py --config-only dp dp3 sat maniflow dqrise r3d multitask_dit interaction_flow
```

新环境先安装与目标设备匹配的 Torch 2.4.1 / torchvision 0.19.1，再按上面的 requirements → editable install 顺序安装。PyTorch3D 0.7.8 是单独的编译后端，须针对实际 Torch/CUDA 安装；点云采样会明确报告缺失依赖。R3M 自动下载额外需要 `gdown`，已有本地权重不需要它。

仿真评测需要另外安装 `dexmani_sim`。训练数据路径由当前 config 决定，常见位置为 `robot_data/<task>.zarr`；字段、坐标系与 SIM/Real 差异见 [数据与模态说明](docs/data_modality.md)。

查看当前可用配置：

```bash
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml
```

配置名只用于定位实验入口，不代表内部模型一定采用某种固定结构；理解实现时应继续进入实际源码。

交互表征策略使用 `interaction_flow`，多卡配置为 `ddp/interaction_flow`。当前模型机制、模态开关与 Real 数据配方见 [逐指交互表征](docs/interaction_representation.md)。

## 推荐研究流程

一次典型迭代保持为一个短闭环：

```text
研究问题 / 假设
      ↓
选择当前 baseline
      ↓
确认真实实现与执行路径
      ↓
定义主要实验变量
      ↓
完成最小可运行修改
      ↓
最低成本验证
      ↓
训练 / 评测 / 对比
```

模型层数、宽度、学习率、NFE、batch size、具体模块组合和实验结论等易变化信息，应留在 config、源码、实验目录与日志中，不写成仓库级长期约定。

仓库方法名表示本地适配实现，不意味着官方代码或论文结果的原样复现。相同 `total_train_steps` 不自动代表相同样本量、计算量或公平预算；比较时仍需固定数据集合并记录 batch/accumulation 与选择、test 协议。Wilson 区间描述 episode 抽样不确定性，不是跨训练 seed 的方差。

## 训练

单卡：

```bash
bash scripts/training/train.sh <config_name> 'task_name=<task>'
```

DDP：

```bash
bash scripts/training/train_ddp.sh ddp/<config_name> 'task_name=<task>'
```

Hydra override 可直接追加，例如：

```bash
bash scripts/training/train.sh <config_name> \
  'task_name=<task>' \
  'training.seed=42'
```

需要续训时显式指定 `resume_from`；重复运行同一训练命令不会自动续训。

```bash
bash scripts/training/train.sh dp3 '+resume_from=experiments/dp3/<task>/<old-run>'
# 远程续训：<task> 必须与来源实验保存值完全一致
bash scripts/remote/train_remote.sh dp3 <task> '+resume_from=experiments/dp3/<task>/<old-run>'
```

`resume_from` 接受当前格式的实验目录或 checkpoint 文件，展开 `~`，相对路径以**项目根目录**为基准。恢复始终写入新输出目录。新 run 默认命名为 `YYYY-MM-DD_HH-MM-SS_seed<seed>`，例如 `2026-10-11_02-00-52_seed1011`，已有实验目录保持原名。已有训练配置、metrics 或 checkpoints 的目录会拒绝使用；同一秒内启动相同 Policy、任务和 seed 可能重名，遇到冲突请指定新目录。目录检查不提供并发锁，不要让多个训练写入同一目录；Hydra 在进入训练前可能已写入自己的启动文件。

完整续训先读取 checkpoint 内绑定的 resolved `effective_config`，并核对来源 `config.yaml` 的配方，再校验和构造。仅允许显式覆盖新输出位置、W&B、等价设备定位、日志间隔、worker 参数、compile 开关/模式、当次执行上限 `+max_updates=N`，以及数据目录迁移；worker/compile 覆盖沿用有界恢复承诺，不承诺增强或算子逐位相同。`max_updates` 仅作用于当次启动，不从来源实验继承。`workspace` 使用当前字段 `_target_`、`output_dir` 和 `wandb_cfg`，不自动清理已废弃字段。world_size、batch/accumulation、训练 seed、AMP/dtype、总计划、optimizer/scheduler/EMA 和模型/数据配方不能覆盖。路径迁移使用 `dataset.zarr_path=...` 或 `dataset.datasets.<index>.zarr_path=...`，训练时仍校验 revision 与保存集合；相同 revision 不是内容 hash 证明。

续训时显式 `task_name` 仅断言来源任务身份，必须与保存值完全相同（复合任务包含顺序）；冲突重复参数也会拒绝。未显式指定时使用保存任务，不受当前 YAML 默认任务影响。远程 `train_remote.sh <config> <task> +resume_from=...` 的数据预检与训练共享配方解析，检查保存的单任务/child 数据路径或显式迁移路径；`--check` 只保证目录存在，不保证数据完整或训练成功。checkpoint 绑定配置缺失或损坏时拒绝续训，不回退到当前配方；存在的 sidecar 必须与绑定配方一致。

保存配置记录 HF 结构及闭集文本维度；完整权重恢复要求 HF backbone 的 `architecture` 为有效映射，闭集 `text_embed_dim` 与保存的 embedding 表一致，不再加载外部初始化权重。固定文本保留任务映射、embedding 表和 projection，未知文本报错；所有 checkpoint 权重严格加载。推理读取完整 `.pt` 文件，不跳过 optimizer I/O。

新训练 checkpoint 使用 `simple.v4`；未绑定来源的 `simple.v3` 拒绝严格续训，仍可在配置足够明确时警告后推理，不自动降级或补写绑定。VQ checkpoint/NPZ 保持 v3。`best_ckpt.json` 使用 `best.v1`，EMA、NFE 和 policy seed mode 必须完整有效。新 selection 使用 `selection.v3`，每个 stage 保存一次完整 episode 证据；读取端保留 `selection.v2` 支持。`--selection-record` 接收交接引用、上述 selection 结果或 `best.v1` 快照，不重生成旧计划。版本边界、回归命令和源码对照见 [修复与验证记录](docs/修复与验证记录.md)。

完整续训要求 `resume_contract.facts_format=1`，CUDA RNG 为每个 rank 的单个 Tensor；CPU RNG 的 CUDA 字段为 null。配置、contract 和状态字段必须满足当前约定，恢复失败不会自动退回 weights-only。

数据路径应保持不可变。重新生成数据时使用新路径和 Zarr root attrs 中新的非空字符串 `data_revision`；训练保存单任务/逐任务身份。完整续训必须包含数据身份映射；缺失映射或已知 revision 改变、丢失均报错。未版本化 Dataset 显式保存 `revision:null`，提示身份未验证。revision 是生产者声明，不是内容 hash。

显式训练 `dataset.split_manifest` 决定最终 episode 集合，训练和恢复均须设置 `dataset.max_train_episodes=null`、`dataset.val_ratio=0`；更小预算应提前写入清单。无清单时使用 seed/比例/cap。恢复使用保存的清单内容，actual IDs 必须与清单训练集合及 canonical 顺序完全一致，mask 也须一致，外部清单文件可以不存在；窗口长度和所选 observation/action 的 finite 检查仍生效。多任务 Dataset 现在读取固定全局索引；通过 `ResumableDistributedSampler`（训练使用 `build_train_loader`）取得原任务配比和 epoch 顺序，validation 的 deterministic 配方也须使用该 sampler（`shuffle=False`）。索引顺序恢复不承诺多 worker 增强逐位相同。训练 Real canonical 数据可用 `+dataset.split_manifest=/path/split_manifest.json` 指定清单。

Diffusion 默认 `agent.clip_sample=true` 保持有界动作行为。支持 Gaussian **动作**归一化的 diffusion 策略必须同时设置 `agent.clip_sample=false`；DQ-RISE/VQ 手部原型仅支持 `action:auto/limits`，关闭裁剪也不支持 Gaussian。Gaussian 观测和 flow 不受 diffusion 裁剪限制。true→false 属于实验变化，不能静默严格续训。

RGB 可通过 `dataset.rgb_keep_uint8=true` 使用 uint8 transport（要求 `normalization.rgb=identity`）；当前 DP 显式启用。配置 CPU resize 时，含 ImageAug 的 recipe 在 float resize/crop/增强后最终量化，无颜色增强的 uint8 recipe 保留 uint8 spatial 路径；视觉入口恢复 float32。缺省/false 保留原 float preprocessing；未配置 CPU resize 时仍返回原始 HWC uint8。recipe 变化不能跨越 strict resume；MultiTask 的 RGB child 必须一致。确定性评测、验证与 Real preprocessing 从保存配置恢复。

DP3/DQRISE 的 PointNet/MultiStagePointNet 配方在执行 FPS 时关闭采样后 shuffle，随机起点与子集选择开关保持不变。该预处理仅在输入点数大于 `agent.num_points` 时采样；当前默认数据与目标均为 1024 点，不执行这条 FPS 路径，也不会因该 shuffle 开关改变 RNG 消耗。对于实际进入 FPS 的配方，关闭 shuffle 会改变 RNG 消耗，后续随机轨迹不保证相同。切换为 PointNext 的配置保留采样后 shuffle。SAT action shuffle、EJC、ManiFlow dense tokenizer、Uni3D 和 compile/static_graph 不受此结论影响。

### DQ-RISE 码本与 Policy 对齐

从仓库根目录，在 `policy` 环境中执行。先确定目标 Policy 配置，所有数据与窗口覆盖在两个训练入口保持一致：

```bash
VQ_RUN="experiments/vq_hand/pick_apple_messy/$(date +%Y%m%d_%H%M%S)_${RANDOM}"
python -m dexmani_policy.training.train_vq_hand \
  --policy-config dexmani_policy/configs/dqrise.yaml \
  --policy-override task_name=pick_apple_messy \
  --output_dir "$VQ_RUN"
python -m dexmani_policy.agents.vq_hand.export_codebook \
  --checkpoint "$VQ_RUN/vqvae_hand_best.pt" \
  --output "$VQ_RUN/codebook.npz"
bash scripts/training/train.sh dqrise task_name=pick_apple_messy codebook_path="$VQ_RUN/codebook.npz"
```

`--policy-config` 使用目标 Dataset 的 split、有效窗口与唯一 action 源行，码本和 Policy 共用训练统计，验证集不参与拟合。支持 joint `7+12` 和 EEF `9+12`，不支持辅助 action 布局；动作归一化必须为 `auto/limits`。VQ 显式请求验证集，已分配 holdout 但无有效窗口时会报错。VQ 的 `--seed` 只控制优化随机性；数据配方以 Policy 配置为准。额外覆盖通过重复的 `--policy-override` 传入，并在 Policy 训练时传入相同覆盖。

省略 `--output_dir` 会自动生成任务下独立 run。已有训练产物的目录拒绝重用。导出目标存在时拒绝写入，只有显式 `--overwrite` 才允许覆盖。选点在开始时固定：有验证集为 `val_mse`，无验证集为 `train_mse`；非有限值报错，旧 best 保留。

导出生成 Policy 使用的完整 residual-code 组合码本。NPZ 必须严格符合 v3 schema，额外字段会被拒绝；Policy checkpoint 同时保存推理需要的内嵌码本。

训练过程保留分组码本使用图和损失曲线；结束后另存 `codebook_diagnostics.json`，对已选 best 的实际部署查表链路报告 train/held-out 误差及原型利用率，不参与 best 排名。导出权重、epoch、SHA256 取自同一字节快照，运行 NPZ 只留短来源摘要，完整源行列表保留在 VQ checkpoint。码本沿用严格 affine 兼容校验。`train_vq_hand.sh` 同样使用目标 Policy 配方。独立 VQ 研究仍可显式指定 Python `--config` 使用全量 hand 统计，但不作为 DQ 默认路径，不保证与目标 Policy 的 affine 参数一致。Policy 和 standalone VQ checkpoint 均须保存 `split_metadata.normalization_spec`；导出前检查结构和动作归一化模式，缺失声明、Gaussian、identity 或非法模式均在解码前拒绝。

### LAMP-style 手部动作空间

`lamp_flow` 使用冻结的历史 MLP prior 和逐帧 hand decoder，DiTX/Flow Matching 预测整段 latent offset。该实现是 LAMP 的 chunk FM 适配；latent 标签来自离线 decoder 反演，采用每个真实帧一份 absolute latent。当前支持单手 joint `7+12` 与同步 SIM 执行。

从仓库根目录执行，三个入口必须使用相同的数据、split、窗口与 `lamp.*` 配置：

```bash
python -m dexmani_policy.training.train_lamp_prior --config lamp_flow
python -m dexmani_policy.training.export_lamp_latents --config lamp_flow
bash scripts/training/train.sh lamp_flow
```

数据须声明 `obs_alignment="obs[t]_before_action[t]"` 和有效 `dt`。旧数据只有在确认记录时序后，才能为三个入口都追加 `dataset.obs_alignment=obs_before_action`。历史取 query 之前的手部命令，episode 开始用初始测得姿态补齐；SIM 使用实际执行后的命令更新历史，不写入未执行预测。

Prior 默认训练 20k 步；有验证集时按重建 MSE 选点，无验证集时保存最终权重。导出只处理有效 train/validation 源帧，并报告关节单位下的重建误差和 latent 跳变。`--device cpu --steps 2` 可用于小样本入口检查，不代表 prior 已有可用精度。默认产物写入 `experiments/lamp_prior/<task>/`，已有文件拒绝覆盖；不同配方应为三个入口指定新的 `lamp.prior_path` 和 `lamp.latent_path`。

反演默认 `--center-penalty 0`，只优化重建。可显式设置 `--center-penalty 1e-4`，在原始 latent 坐标中加入到源帧 history mean 的均方距离惩罚；每帧按同一目标选最佳迭代，包含初始化。该选项用于研究反演多解，不是原论文算法：它可能降低极端分支跳变，也可能增加标签对历史的依赖。导出同时报告完全相同相邻动作的 latent 跳变和重建不劣于初始化的比例；应结合重建误差和 policy 使用的 latent scale 比较，不能只看 raw latent 最大跳变。

Policy checkpoint 内嵌 prior、归一化和 latent scale，推理无需外部 prior/cache；训练和续训仍校验缓存的数据、split 和权重身份。`agent.decoded_loss_weight=0` 可关闭 decoded hand 辅助损失；整段动作不施加向当前 prior 姿态回缩的 drift 惩罚。[LAMP 论文](https://arxiv.org/abs/2607.06323) 的单步 BC 直接监督解码动作，Equation 6 正则化 latent offset；[开源实现](https://github.com/dex-lamp/LAMP) 则使用 decoded-action drift。这里的反演标签与 chunk FM 均为本地适配，decoded supervision 参考 [latent FM](https://arxiv.org/abs/2609.19666)。

`agent.latent_loss_weight` 默认 `1.0`，只缩放 latent FM 项：`(7 * arm_MSE + 12 * latent_loss_weight * latent_MSE) / 19`。它用于隔离反演标签监督强度，不改变 arm 或 decoded hand 的权重、latent 坐标及推理解码；非默认值属于消融实验。

`lamp_flow` 与 `temporal_codec_flow` 的 SIM 配置均使用 `env_runner.clear_cache_freq=1`，每回合重建环境，避免前一回合的仿真状态影响比较。已有实验仍以各自保存的评测配置为准。

### Temporal hand codec 动作空间

`temporal_codec_flow` 根据 [Towards High-DoF Dexterous Manipulation through VLA Post-Training](https://arxiv.org/abs/2609.19666) 的 codec 与 SFT 描述实现单手 BC baseline。时序 VAE 用实测手部姿态 `q0` 条件化 E/D，将 `H×12` 手部命令编码为 `H×Z`，保持时间长度。冻结 encoder 的 posterior mean 在线提供 FM 标签，无需反演或 latent 缓存；冻结 decoder 仍向 policy 回传 decoded hand loss 的梯度。

```bash
python -m dexmani_policy.training.train_temporal_codec --config temporal_codec_flow
bash scripts/training/train.sh temporal_codec_flow
```

两个入口使用相同的数据、split、窗口和 `codec.*` 配置。数据需声明 `obs_alignment="obs[t]_before_action[t]"`；旧数据确认时序后，在两个入口都追加 `dataset.obs_alignment=obs_before_action`。默认产物为 `experiments/temporal_codec/<task>/codec.pt`，已有文件拒绝覆盖；改变配方时应指定新的 `codec.path`。Policy checkpoint 内嵌完整 codec，部署不依赖外部 codec 文件。

codec 使用双向时序 Transformer、相对手部命令的 train-only 标准化、位置/一阶/二阶 weighted Huber 和分阶段 KL。默认 `Z=9`，可通过 `codec.latent_dim` 改动后重新预训练。保留本项目的数据划分和 `H=16` 窗口：`q0` 取 query 时刻（`n_obs_steps-1`）未增强的实测手部状态，整个训练窗口均相对此姿态表示，包含窗口前部的历史帧；推理先解码完整窗口再截取执行段。手臂沿用原来的归一化绝对关节命令。

这是论文方法的本地适配。block 未公开的细节采用 pre-LN、GELU、可学习位置编码和零 dropout；KL 取 batch/time/latent 全均值，论文未明确其实现中的 reduction。BC 的 arm/latent/decoded 三项分别取均值，权重为 `1/0.25/0.75`，decoded loss 在标准化相对动作空间计算。预训练默认 30k 步；有验证集时按原关节单位 MAE 选 checkpoint，包括 KL ramp 期间的候选。默认 `dataset.val_ratio=0` 沿用现有策略配置，此时按 train MAE 选点，不能报告 held-out 重建性能；需要验证集时为两个入口设置相同的 `dataset.val_ratio`。

`agent.decoded_loss_space=physical` 可单独检验 BC decoded loss 的单位选择：对逆标准化后的关节误差计算 MSE（rad²）。默认 `standardized` 保持已有实验语义；该选项不改变 latent 标签、其他损失或推理解码，需要重新训练 BC 才能比较效果。论文 Eq.10 没有充分展开此处的标准化边界，两种设置应明确标注为复现选择，不能仅凭公式相似就宣称与作者实现等价。

预训练可选 `--reference-noise-std 0.05`（rad）：仅对训练窗口的 `q0` 加独立 Gaussian 扰动，整段共用该 reference，physical action targets 保持原样，E/D 和 relative targets 使用同一扰动值。统计量和验证仍使用原始数据，增强使用独立 RNG 并写入 artifact 元信息。默认 `0`；这是针对少量数据下状态偏移的鲁棒性实验，不属于论文 baseline，可能牺牲干净数据重建精度。改变 codec 后需指定新的 `codec.path` 并重新训练 BC，不能直接替换旧策略内的 codec。

## 验证

修改 config 或研究实现后，优先执行轻量检查：

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

当改动进入实际数据、模型构造、训练或推理链路时，再运行完整 smoke：

```bash
python dexmani_policy/smoke_test.py <config_name>
```

配置与训练链路统一使用 `smoke_test.py`，Agent 模块不再维护独立的合成数据演示入口。安装 `pytest` 后，定向回归检查使用：

```bash
python -m pytest -q tests
```

pytest 使用合成数据和临时产物；CUDA 不可见时会跳过对应检查，CPU 检查通过不代表 GPU 检查通过。完整训练、DDP 和长时间评测不是普通代码改动后的默认验证步骤。

`--config-only` 只检查配置和 target。完整 smoke 通过正式 Trainer 使用原资产、batch、BF16/compile/accumulation 配置，默认运行 4 次 optimizer update（可用 `--max-updates N` 设置本次上限），要求实际学习参数发生有限变化，再验证预测及 raw/EMA 保存恢复。执行预算不修改 warmup 和总训练计划，`global_step` 随实际更新增长；预算内没有有效变化会失败，不提高学习率。输出使用临时目录且不创建 W&B run；此入口不验证多卡运行。

保存恢复的预测比较使用相同 eager 执行模式，保持原误差阈值；compile 预测仍检查形状/有限性，并单独报告与 eager 的数值差异。DDIM 推理将 scheduler 系数统一放在采样设备，避免训练曾调用 add_noise 与新恢复实例之间的 CPU/GPU 标量运算差异。

smoke 接受配置名、`--config-only` 和 `--max-updates`，不接受 `task_name=...` 等 Hydra 字段覆盖。需要修改数据配方或保留产物时，使用正式单卡/DDP 入口及 `+max_updates=N`；有界停止会保存真实训练进度和采样游标。具体命令见 [interaction 运行检查](docs/interaction_representation.md#运行检查)。配置检查不证明 GPU 算子、真实数据链路或闭环性能已通过。

## 仿真评测

常用一键入口：

```bash
bash scripts/eval/eval_pipeline.sh <policy_name> <task_name> <exp_name>
```

也可以分步执行：

```bash
bash scripts/eval/select_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/eval_best_ckpt.sh <policy_name> <task_name> <exp_name>
bash scripts/eval/record_demo.sh <policy_name> <task_name> <exp_name>
```

评测从 checkpoint 的绑定配方恢复模型，并核对实验 `config.yaml`；旧 v3 推理仅有上文所述的 sidecar 保证。`eval_pipeline.sh` 执行 selection 和不录视频的 held-out eval；demo 用独立的 `record_demo.sh`，不作为 held-out 结果。

评测从 runner 的 seed 池按 evaluation seed 确定性抽样，不需要额外清单。selection 使用 `--initial-episodes`、`--batch-size` 和 `--max-episodes` 控制预算；最终评测使用 `--episodes` 控制每任务数量，并排除实际用于选点的 seeds。具体规则见 [仿真评测机制](docs/仿真评测机制.md#42-seed-池与采样预算)。

`best_ckpt.json` 保存最近一次成功选择的 `best.v1` 推理快照和详细结果引用。普通 demo/部署读取快照即可；held-out best 评测和显式 handoff 仍需详细 selection 证据。流水线通过 `--result-file` / `--selection-record` 固定本次 checkpoint、raw/EMA 和 NFE；冲突覆盖或未同步的产物会报错。普通 best 调用允许显式覆盖 EMA/NFE。正常全零结果仍可发布，技术异常以非零状态退出并保留旧 best。

每次 selection、eval 和 demo 保存独立结果目录及 `eval_config.yaml`。协议、seed 隔离、当前记录格式和结果字段见 [仿真评测机制](docs/仿真评测机制.md)。

### 源码追溯

远程 `train_remote.sh` 每次在 `outputs/remote_sources/<launch_uuid>/` 准备独立源码副本，以 module 方式启动，复用原有数据、权重和实验目录；后续启动不会同步到旧副本。副本保留供追溯。恢复行为见 [项目架构](docs/项目架构.md)。

远端代码和数据都可持续更新：`sync_code.sh` 默认更新主源码镜像，`sync_data.sh` 更新共享数据，`train_remote.sh --sync-data` 也可在每次启动前同步数据。独立副本只隔离源码，数据更新会对所有引用同一路径的副本可见；直接从主源码目录运行的手工任务，其后续文件读取和 lazy import 也可能看到源码更新。

## 其他工作流

- 训练工作流 Shell 入口：`scripts/training/`
- 远端训练、数据同步与日志管理：`scripts/remote/`
- 仓库级辅助脚本：`scripts/utils/`
- Real policy inspection / inference：`dexmani_policy/deployment/`

远程启动会输出独立 session 名，后台任务使用 `tmux -L dexmani_policy`。`stop_remote.sh` 的具名停止、`--all` 和 `--list` 均只管理这个 socket。普通停止有30秒轮询等待预算（SSH 查询耗时另计），仅显式 `--force` 允许超时强停。session 消失不证明 checkpoint 完整，数据预检也不代表训练通过。

`tail_log.sh <policy> <task> [run_name]` 支持当前及自定义 run 名，不枚举 Policy。默认按 `config.yaml` 文件时间选择；候选须有配置和 metrics。它使用 `tail -F` 跟随同步替换后的日志。W&B 批量同步只搜索 run 根目录，先完成发现再上传。

`sync_down.sh` 保留已有不可变产物，只更新指定训练文件与可变引用；配置或来源身份冲突时停止。参数、同步顺序和失败处理见 [SSH 服务器训练部署](docs/SSH服务器训练部署.md)。实验盘点使用 `bash scripts/utils/clean_experiments.sh`，只报告事实，不删除或移动实验。

Dataset 按需读取 Zarr 窗口，normalizer 使用有效训练窗口的唯一源行；数据与恢复合同见 [项目架构](docs/项目架构.md)。Real 真机运行入口由 `dexmani_real` 提供，执行长度、预取、运行模式和预算由 Real 管理；模型配置与点云生产 recipe 的职责见 [部署边界](docs/项目架构.md#24-deployment-boundary)。真机运动需明确授权。

## 项目结构

```text
dexmani_policy/
  configs/       当前实验与运行配置
  agents/        策略与研究实现
  datasets/      数据读取、采样与预处理
  training/      训练、状态保存与恢复
  evaluation/    离线评测共享逻辑
  env_runner/    仿真环境执行
  deployment/    已保存策略检查与 Real runtime
  utils/         少量跨模块通用工具

scripts/
  training/      训练工作流 Shell 入口
  eval/          仿真评测入口
  remote/        远端训练与同步
  utils/         仓库辅助脚本
```

## 文档分工

三份根目录文档只承担三个稳定职责：

- **README.md：怎么使用这个研究仓库。** 面向人，维护定位、入口、常用命令和项目地图。
- **AGENTS.md：AI 应该怎样和我一起做研究。** 维护研究协作原则、改动方式与验证方法，不描述某个具体模型。
- **CLAUDE.md：Claude 去哪里读取这些规则。** 只作为加载入口，不维护第二份规范。

较长的背景和机制说明放在 `docs/`：

- [`docs/项目架构.md`](docs/项目架构.md)
- [`docs/interaction_representation.md`](docs/interaction_representation.md)：逐指交互表征、模态开关与运行检查
- [`docs/hand_spatial_encoding.md`](docs/hand_spatial_encoding.md)：腕部与指尖位置编码的输入契约和研究边界
- [`docs/point_image_fusion.md`](docs/point_image_fusion.md)：独立的 2D–3D patch 对应与语义融合模块（当前策略未接入）
- [`docs/data_modality.md`](docs/data_modality.md)：Zarr schema、SIM/Real 模态语义与归一化边界
- [`docs/仿真评测机制.md`](docs/仿真评测机制.md)
- [`docs/SSH服务器训练部署.md`](docs/SSH服务器训练部署.md)

当背景文档与当前 config/code 不一致时，以当前实现为准。

## AI Skills

仓库提供两个**模型无关**的研究工作流 skill：

- `research-iterate`：研究假设 → 识别真实改动面 → 最小实现 → 验证 → 下一步实验。
- `preflight-experiment`：昂贵实验前检查当前配置、数据、依赖、产物与运行条件是否一致。

Skill 只规定研究工作方法，不写死当前模型的模块划分、输入模态、动作表示或训练算法；模型变化时无需随之重写。

Codex skill 位于 `.codex/skills/`，Claude Code skill 位于 `.claude/skills/`。
