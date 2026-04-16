# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Env & Commands

Conda env: `policy`. External dep: `dexmani_sim`.

```bash
python train.py --config-name dp3 seed=233 task_name=multi_grasp
python eval_sim.py --agent-name dp3 --task-name multi_grasp --exp-name 2026-04-01_11-18_233
python -m dexmani_policy.agents.obs_encoder.backbone_2d.resnet
wandb sync ./wandb --include-offline --sync-all
```

## Workflow: 改什么去哪里

| 目标 | 路径 |
|------|------|
| 改网络结构（backbone/encoder/decoder） | `agents/` — `base_agent.py` 是基类，`dp3_agent.py`/`moe_dp3_agent.py` 是具体策略 |
| 改观测编码（RGB/PC/文本/融合） | `agents/obs_encoder/` — registry 模式，新增 encoder 需注册到 `rgb/registry.py` 或 `pointcloud/registry.py` |
| 改动作解码（diffusion/flowmatch） | `agents/action_decoders/` — backbone 在 `backbone/unet1d.py` |
| 改数据加载与采样 | `datasets/base_dataset.py` — zarr replay buffer + `SequenceSampler` |
| 改训练循环逻辑 | `training/trainer.py` — `train_one_step()` / `validate()` / `evaluate()` |
| 改日志与 checkpoint | `training/common/workspace.py` — Wandb+JSONL 日志，Top-K checkpoint |
| 改仿真评估 | `env_runner/sim_runner.py` — 动态导入 `dexmani_sim.envs.<task>` |
| 改配置 | `configs/*.yaml` — Hydra 配置，`eval:` resolver 支持动态表达式 |

## Data Flow

```
zarr data → BaseDataset (SequenceSampler) → DataLoader
  → Trainer.train_one_step() → BaseAgent.compute_loss(batch)
    → encode_obs_as_condition() → obs_encoder + state_mlp → condition
    → action_expert.compute_loss(condition, norm_action) → loss
  → optimizer.step() + scheduler.step() + EMA update
```

## Key Conventions

- Config: `horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19`
- 优化器分离学习率: `lr` (backbone) vs `obs_lr` (obs_encoder)
- EMA: `use_ema: true` 时评估默认用 EMA 权重
- 实验输出: `experiments/<policy_name>/<task_name>/<timestamp>_<seed>/`
