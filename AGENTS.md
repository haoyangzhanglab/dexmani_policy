# AGENTS.md — DexMani_Policy

本文件是 Codex 在本仓库中的项目级工作约定。详细背景以 `CLAUDE.md` 和
`README.md` 为准；若文档与代码不一致，先核对实际配置和调用链，再做最小化修改。

## 项目概览

- 本项目是灵巧手操作模仿学习框架，使用 Hydra 配置、Zarr replay buffer、
  Diffusion/FlowMatch 动作解码，并通过 `dexmani_sim` 进行仿真评测。
- Python 包位于 `dexmani_policy/`；训练入口为 `train.py` / `train_ddp.py`，
  评测入口为 `eval_best_ckpt.py`，构建验证入口为 `smoke_test.py`。
- 配置位于 `dexmani_policy/configs/`；脚本位于 `scripts/training/` 和
  `scripts/eval/`；详细架构文档位于 `docs/`。

## 环境与依赖

- 使用 Python 3.10+ 和 Conda 环境 `policy`。
- 安装项目：`pip install -e .` 仅安装项目声明的核心依赖，并不是覆盖所有策略的完整环境；
  RGB、Uni3D 等策略仍需由受管 Conda 环境提供额外依赖。
- 评测还需要以 editable 模式安装相邻的 `dexmani_sim` 项目。
- 训练数据由 config 的相对路径 `robot_data/<task>.zarr` 定位（运行时 chdir 到仓库根目录），
  不读取 `DATA_DIR` 环境变量；评测种子文件由相邻 `dexmani_sim` 包导出的 `DATA_DIR` 常量提供。
- 训练、评测和大多数 smoke test 依赖 CUDA/GPU。不要仅因当前环境缺少 GPU、
  数据集、预训练权重、显示服务或 `dexmani_sim` 就改写核心逻辑。
- 在非交互 shell 中优先使用 `conda run -n policy <command>`，避免依赖
  `conda activate` 的 shell 状态。
- 本机有 GPU；sandbox 内 CUDA 不可见时，使用命令级提权进行 GPU 检查和必要的
  smoke test，不据此判定本机无 GPU，不修改核心逻辑或全局关闭 sandbox。

## 修复 workflow

用户要求执行 v2 修复 workflow 时，先读 `docs/repair_workflow.md` 和
`docs/repair_progress.md`。按 workflow 自动完成阶段验收、精简交接与后续阶段，
无需逐阶段确认。允许按任务难度调用 `sol-high`、`terra-xhigh`、`luna-max`
三档 agent；具体分工、验证节奏及本轮优先约束以 workflow 为准。

## 常用命令

```bash
# 单卡训练（task 用 task_name= 覆盖）
bash scripts/training/train.sh dp3 'task_name=pour'
bash scripts/training/train.sh dp3 'task_name=pour' 'training.seed=42'

# 多卡训练
bash scripts/training/train_ddp.sh ddp/maniflow 'task_name=pour'

# 构建冒烟测试
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py dp3 maniflow sat

# 评测管道（<exp_name> = experiments/<policy>/<task>/ 下的时间戳目录名）
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_name>
bash scripts/eval/eval_pipeline.sh dp3 pour <exp_name> --no-videos
```

- 单卡配置：`action_flow`、`dp`、`dp3`、`dqrise`、`maniflow`、
  `multitask_dit`、`r3d`、`sat`。
- DDP 配置：`ddp/action_flow`、`ddp/dp`、`ddp/dqrise`、`ddp/maniflow`、
  `ddp/multitask_dit`、`ddp/r3d`、`ddp/sat`。
- 不要自动启动完整训练、DDP、长时间评测或视频录制；除非用户明确要求，
  验证应从受影响策略的 smoke test 开始。

## 修改原则

- 优先修复根因，保持改动小而集中，不顺带重构无关模块。
- Hydra 通过 `_target_` 直接导入 Agent，不存在需要同步维护的显式注册表。
- 新增或修改策略时同步检查 Agent 类、主配置、可适用的 DDP overlay、评测参数
  和相关文档；从结构最接近的现有策略复制模式。
- 保持公共基类的兼容性；策略特有行为应留在对应 Agent、encoder、backbone 或
  decoder 中，避免为了复用而抹平有意的架构差异。
- 不修改或提交 `robot_data/`、`experiments/`、checkpoint、视频、W&B 日志、
  预训练权重等生成物或大文件。
- 不覆盖工作区中与当前任务无关的用户改动。

## 核心不变量

除非任务明确要求改变接口并同步所有耦合点，否则保持：

- `horizon=16`、`n_obs_steps=2`、`n_action_steps=8`。
- `pad_before=1`、`pad_after=7`，且
  `n_obs_steps - 1 + n_action_steps <= horizon`。
- `joint_state` 固定为 19 维（7 臂 + 12 手）；`action` 为 19 维（7 臂 + 12 手），
  `action_ee` 为 21 维（9 位姿 + 12 手）。
- 动态动作维度公式为
  `${eval:'21 if ${eq:${action_key},action_ee} else 19'}`。
- 优化器保持 `AdamW(fused=torch.cuda.is_available())`；UNet conditioning 保持
  `cond_predict_scale=True`；StateMLP hidden 保持 `[64]`。
- DDIM scheduler 保持 `beta_start=0.0001`、`beta_end=0.02`、
  `beta_schedule='squaredcos_cap_v2'`。
- DINO/CLIP/SigLIP ViT backbone 使用 `bfloat16` 和
  `attn_implementation="sdpa"`。
- milestone checkpoint 仅为 20/40/60/80/100%，`latest.pt` 是 symlink。

## 有意设计，不要误修

- Normalizer 使用完整 replay buffer（包括验证集）拟合。
- `tcp_dim` 是历史命名：joint 模式为 7，ee 模式为 9。
- DQ-RISE 直接继承 `BaseAgent`，其 `diffusion_action_dim=tcp_dim+1`，不要强行
  改为 `UNetDiffusionAgent`。
- R3DObsEncoder 的 patch/state/position 表示按 feature 维组合，保持现有语义。
- EMAModel 的 BatchNorm affine 参数直接复制，不做 EMA 平均。
- `FlowMatchWithConsistency.target_t`：flow 分支训练为 0，consistency 分支训练为 `dt1(>0)`，推理为 `dt(>0)`。
- ActionFlow 的 flowmatch、DiT 和采样逻辑是独立实现；不要与通用
  `flowmatch.py`、`ditx.py` 或 `time_sampler.py` 合并或交叉引用。
- `dp3` 有意没有 DDP overlay。
- Uni3D 预训练权重 fail-closed：`use_pretrained_weights=true` 时权重缺失/下载失败/key
  匹配率 <0.5 直接抛异常，除非显式 `allow_random_init=true`（默认 false）。
- `__init__.py` barrel 已清空为纯文档字符串（不 re-export）；Hydra 走 `_target_` 直接
  模块导入，新增 Agent 无需在 `__init__.py` 注册。
- 不要意外启用 modality dropout、TokenCompressor 或 T5TextEncoder 预留功能。

## 验证要求

- Python 改动先做语法/导入级检查，再运行最接近受影响策略的 smoke test。
- Agent、decoder、encoder 或配置改动至少运行对应单策略：
  `python dexmani_policy/smoke_test.py <config_name>`。
- 跨公共基类的改动应选取不同架构代表验证，例如 `dp3 maniflow sat`。
- 配置改动核对维度链：`action_dim`、`state_dim`、`tcp_dim`，以及主配置与
  DDP overlay 的继承结果。
- 评测改动优先使用小 episodes 或 `--no-videos`；视频录制需要 X11/Wayland。
- 如果受环境限制无法执行 GPU 验证，明确报告未执行的命令和缺失条件，
  不把环境失败表述为代码通过。

## 文件导航

- UNet + Diffusion 参考：`dexmani_policy/agents/core/dp3.py` 和
  `dexmani_policy/configs/dp3.yaml`。
- DiTX + FlowMatch + Consistency 参考：`dexmani_policy/agents/core/maniflow.py`
  和 `dexmani_policy/configs/maniflow.yaml`。
- 完全自定义 Agent 参考：`dexmani_policy/agents/core/sat.py`、`r3d.py` 或
  `action_flow.py`。
- 数据增强：`dexmani_policy/datasets/base_dataset.py`。
- 训练循环：`dexmani_policy/training/trainer.py`。
- 评测逻辑：`dexmani_policy/env_runner/sim_runner.py`。
- Normalizer：`dexmani_policy/common/normalizer.py`。

## 项目技能

- 添加新 Agent/策略时使用 `dexmani-agent-integration`。
- PR 前审计或“review/check before PR”时使用 `dexmani-pr-check`。
- 训练出现 NaN、梯度 NaN 或 debug checkpoint 时使用
  `dexmani-training-debug`。

过程性 checklist 以对应 skill 为准；项目事实、命令和设计约定以本文件、
`CLAUDE.md`、`README.md` 及实际代码为准。
