# DexMani_Policy

面向灵巧手操作研究的个人策略仓库。目标是让新的机器人学习想法能够以尽量小的改动快速落地、验证和比较，而不是建设一个通用、长期兼容的软件平台。

当前研究内容会持续变化，可能涉及 3D / 点云、多模态感知、灵巧操作、生成式策略等方向。具体模型结构、输入输出和训练方式始终以**当前 config 与源码**为准，不在 README 中固化。

> **事实来源**：当前实现以 config 和源码为准；已有实验以其保存的 resolved `config.yaml` 与 checkpoint 为准。README 只维护稳定入口、常用命令和研究工作流。

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
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise r3d multitask_dit
```

历史全新 venv 验证覆盖 Python 3.10.20、Torch 2.4.1+cpu / torchvision 0.19.1+cpu 的安装、依赖导入、13 个配置解析和默认 DP/DP3 构造（使用已有权重缓存，保留 LoRA），见 [基础设施修复报告](docs/infra_fix_report.md)。后续在已有 `policy` 环境的 RTX 4090 上完成 DP3/DQ 真实数据单步更新、预测及 raw/EMA 保存恢复，见 [整改报告](docs/review_remediation_report.md)。这不代表全新 CUDA 安装、全部策略或生产 AMP/compile/DDP 已验收。

新环境先安装与目标设备匹配的 Torch 2.4.1 / torchvision 0.19.1，再按上面的 requirements → editable install 顺序安装。PyTorch3D 0.7.8 是单独的编译后端，须针对实际 Torch/CUDA 安装；点云采样会明确报告缺失依赖。R3M 自动下载额外需要 `gdown`，已有本地权重不需要它。

仿真评测需要另外安装 `dexmani_sim`。训练数据路径由当前 config 决定，常见位置为 `robot_data/<task>.zarr`。

查看当前可用配置：

```bash
ls dexmani_policy/configs/*.yaml
ls dexmani_policy/configs/ddp/*.yaml
```

配置名只用于定位实验入口，不代表内部模型一定采用某种固定结构；理解实现时应继续进入实际源码。

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

七种方法与六个 DDP overlay 的当前数值配方、固定评测协议及验证边界见 [论文 recipe 与实施记录](docs/paper_recipes.md)。

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
```

`resume_from` 接受旧实验目录或 checkpoint 文件，展开 `~`，相对路径以**项目根目录**为基准。恢复始终写入新输出目录。默认 run 使用微秒时间和随机后缀；已有 `.training_run.json`、训练配置、metrics 或 checkpoints 的目录会拒绝认领。遇到冲突请指定新目录，不要删除旧标记来续写产物。Hydra 在进入训练前可能已写入自己的启动文件，训练认领保护不等于整个 Hydra 启动过程的事务。

数据路径应保持不可变。重新生成数据时使用新路径和 Zarr root attrs 中新的非空字符串 `data_revision`；训练保存单任务/逐任务身份。恢复时已知 revision 改变或丢失会报错，历史身份缺失会明确提示“数据身份未验证”。revision 是生产者声明，不是内容 hash。

Diffusion 默认 `agent.clip_sample=true` 保持有界动作行为。Gaussian **动作**归一化必须同时设置 `agent.clip_sample=false`；Gaussian 观测和 flow 不受此限制。历史缺失开关等价于 true，true→false 属于实验变化，不能静默严格续训。旧 Gaussian＋true 结果需用旧代码复现，修正后重新评测。

RGB 可通过 `dataset.rgb_keep_uint8=true` 使用 uint8 transport（要求 `normalization.rgb=identity`）；当前 DP 显式启用。配置 CPU resize 时，含 ImageAug 的 recipe 在 float resize/crop/增强后最终量化，无颜色增强的 uint8 recipe 保留 uint8 spatial 路径；视觉入口恢复 float32。缺省/false 保留原 float preprocessing；未配置 CPU resize 时仍返回原始 HWC uint8。recipe 变化不能跨越 strict resume；MultiTask 的 RGB child 必须一致。确定性评测、验证与 Real preprocessing 从保存配置恢复，详见 [RGB transport 验证报告](docs/rgb_transport_report.md)。

### DQ-RISE 码本与 Policy 对齐

从仓库根目录，在 `policy` 环境中执行。先确定目标 Policy 配置，所有数据与窗口覆盖在两个训练入口保持一致：

```bash
VQ_RUN="experiments/vq_hand/pick_apple_messy/$(date +%Y%m%d_%H%M%S)_${RANDOM}"
python -m scripts.training.train_vq_hand \
  --policy-config dexmani_policy/configs/dqrise.yaml \
  --policy-override task_name=pick_apple_messy \
  --output_dir "$VQ_RUN"
python -m scripts.training.extract_vq_codebook \
  --checkpoint "$VQ_RUN/vqvae_hand_best.pt" \
  --output "$VQ_RUN/codebook.npz"
python -m scripts.training.measure_vq_usage \
  --checkpoint "$VQ_RUN/vqvae_hand_best.pt" \
  --zarr robot_data/pick_apple_messy.zarr \
  --codebook "$VQ_RUN/codebook.npz"
bash scripts/training/train.sh dqrise task_name=pick_apple_messy codebook_path="$VQ_RUN/codebook.npz"
```

`--policy-config` 使用目标 Dataset 的 split、有效窗口与唯一 action 源行，码本和 Policy 共用训练统计，验证集不参与拟合。支持 joint `7+12` 和 EEF `9+12`，不支持辅助 action 布局。VQ 的 `--seed` 只控制优化随机性；数据配方以 Policy 配置为准。额外覆盖通过重复的 `--policy-override` 传入，并在 Policy 训练时传入相同覆盖。

省略 `--output_dir` 会自动生成任务下独立 run。新 run 原子认领；已认领目录或已有 VQ 产物均拒绝重用。导出目标存在时拒绝写入，只有显式 `--overwrite` 才允许覆盖。选点在开始时固定：有验证集为 `val_mse`，无验证集为 `train_mse`；非有限值报错，旧 best 保留。

使用率工具对新 checkpoint 默认测训练源行；有验证 split 时可加 `--split validation`。checkpoint 保存目标配置、实际 Dataset 配置、统计范围和源行；码本沿用严格 affine 兼容校验。旧 `--config` 独立调用和 `train_vq_hand.sh` 保留全量 hand 统计配方，不保证与目标 Policy 兼容；旧 checkpoint 不会被静默重拟合。

## 验证

修改 config 或研究实现后，优先执行轻量检查：

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

当改动进入实际数据、模型构造、训练或推理链路时，再运行完整 smoke：

```bash
python dexmani_policy/smoke_test.py <config_name>
```

完整训练、DDP 和长时间评测不是普通代码改动后的默认验证步骤。

`--config-only` 只检查配置和 target；完整 smoke 包含一次参数更新、预测及保存恢复，不等同于正式训练收敛或生产 DDP 验证。

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

评测已有实验时，使用该实验保存的 resolved config 与 checkpoint 恢复其真实语义，不用当前默认配置推测历史实验。

每次 selection、final eval 和 demo 都保存独立目录及 `eval_config.yaml`，包括实际推理参数、环境、task/seed 和代码版本；候选明细可重算选点指标。selection 默认不录像，显式加 `--videos` 才记录候选/阶段隔离的视频。demo 保存各 NFE 的结果明细，但不属于 held-out 评测。

`best_ckpt.json` 指向最近一次**成功发布**的 selection；失败保留旧 best，并以非零状态结束。评测、demo 和 policy inspection 在一次调用中固定同一份 best 记录与具体权重。普通 best 调用允许显式覆盖 EMA/NFE 并保留来源证据；使用 `--selection-record` 的流水线交接则拒绝冲突的 checkpoint/EMA/NFE 覆盖。新 best 的 summary 或权重尚未同步时会报缺失，不替换为其他权重。多任务 held-out 校验任务顺序、seed 池身份与真实 `(task, seed)` 无交集；旧多任务记录缺证据需重新选点，普通权重推理仍允许。

远程启动需要本地可用的 `python3` 或 `python`，通过标准库生成独立 session/log 名，不依赖本地 `/proc`，不自动终止旧会话。停止时使用启动输出中的 `stop_remote.sh <SESSION>`。`sync_down.sh` 对小型评测 JSON/YAML 使用 checksum 更新，checkpoint 继续增量下载。

### 固定论文评测与训练源码追溯

新论文比较先从实际 runner seed 池确定同一份 task→seed JSON 清单，通过 `eval.seed_manifest=/absolute/path/seeds.json` 交给 selector。它包含 `pool_id`、`selection`、`tie_break`、`test` 四个字段，后三者均为 task→整数 seed 列表。预留 tie-break 池始终不进入 test，测试数量由清单固定，与 `episodes` 不同时提示实际数量；`max_episodes` 必须容纳完整 selection 加预留 tie-break，否则报错。多任务清单必须匹配现有 paired runner 的映射；缺失、重复、相交或映射不一致直接报错。新 selector 必须使用有效清单；七个基础配置提供任务对应的默认路径，生成方式和已发布清单见 [论文 recipe](docs/paper_recipes.md)。后续 eval/demo 使用 selection record 内嵌的协议，显式冲突报错。历史无清单的已发布记录仍可按 legacy 协议复现。完整正常的全零 selection 仍发布并继续固定 test；空结果、缺失 episode 和技术异常失败并保留旧 best。

`select_best_ckpt.py --result-file <新文件>` 导出该次选择记录；`eval_best_ckpt.py` 与 `record_demo.py` 用 `--selection-record <该文件>` 固定 checkpoint、raw/EMA、NFE 和 seed 协议。`eval_pipeline.sh` 自动使用唯一交接文件，可通过 `SEED_MANIFEST=/absolute/path/seeds.json` 指定论文清单。

新训练 run 保存 `source.zip` 和 `source_manifest.json`，记录实际运行源码（含源码目录中的未提交/未跟踪文件）、内容 SHA256、Git 身份和关键依赖。远端没有 `.git` 时 commit 为 `unknown`，以内容归档为准。**不要向正在训练的源码目录原位运行 `sync_code.sh`**：归档不能隔离后续 lazy import。源码身份不纳入严格 resume 相等合同。

全部整改状态、验证范围、兼容边界与复验命令见 [整改报告](docs/review_remediation_report.md)。

## 其他工作流

- 辅助训练与研究脚本：`scripts/training/`
- 远端训练、数据同步与日志管理：`scripts/remote/`
- 仓库级辅助脚本：`scripts/utils/`
- Real policy inspection / inference：`dexmani_policy/deployment/`

Dataset 只读打开选中 Zarr 字段，按进程管理句柄，窗口按需读取；资格按块扫描，normalizer 从有效训练窗口的唯一源行分块消费；高维 payload 由每字段当前块、窗口和 worker 预取量约束，资格 mask/源行索引仍随行数增长。时间窗口采用实际记录行索引（recorded_rows），不自动插值或把不规则间隔改成固定网格。

Real 新训练配方使用角色化有效窗口、两设备 ACCEPTED dispatch 和去重 train-only normalizer；旧 checkpoint 推理继续使用保存统计。部署桥接返回完整 future，Real 默认 sync，可选择 async 或 DDIM-adapted RTC。支持范围、数学与定向测试见 [Real 数据与 RTC](docs/rtc.md)。真机预算和运行入口由 `dexmani_real` 提供。

真机运动不属于普通开发验证；只有在明确需要时才进入 Real 流程。

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
  training/      训练与辅助研究流程
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
- [`docs/仿真评测机制.md`](docs/仿真评测机制.md)
- [`docs/SSH服务器训练部署.md`](docs/SSH服务器训练部署.md)

当背景文档与当前 config/code 不一致时，以当前实现为准。

## AI Skills

仓库提供两个**模型无关**的研究工作流 skill：

- `research-iterate`：研究假设 → 识别真实改动面 → 最小实现 → 验证 → 下一步实验。
- `preflight-experiment`：昂贵实验前检查当前配置、数据、依赖、产物与运行条件是否一致。

Skill 只规定研究工作方法，不写死当前模型的模块划分、输入模态、动作表示或训练算法；模型变化时无需随之重写。

Codex skill 位于 `.codex/skills/`，Claude Code skill 位于 `.claude/skills/`。
