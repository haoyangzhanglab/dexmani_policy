#!/bin/bash
# 训练示例速查

# ── 单卡训练 ──────────────────────────────────────────────────────────────────
bash scripts/train.sh dp3
bash scripts/train.sh moe_dp3
bash scripts/train.sh dp
bash scripts/train.sh maniflow

# 覆盖任务和种子
bash scripts/train.sh dp3 task_name=sim/pick_apple_messy seed=42

# ── DDP 多卡训练（需使用含 training.num_gpus 的 DDP config）─────────────────
bash scripts/train_ddp.sh maniflow_ddp

# 指定 GPU（例如 GPU 1,2,3,4）
bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[1,2,3,4]

# 使用非连续 GPU（例如 GPU 0 和 7）
bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=2 training.gpu_ids=[0,7]

# 使用环境变量选卡（gpu_ids=null 时按 CUDA_VISIBLE_DEVICES 顺序）
CUDA_VISIBLE_DEVICES=2,3,5,6 bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=4 training.gpu_ids=null

# ── 仿真评估 ──────────────────────────────────────────────────────────────────
python dexmani_policy/eval_sim.py \
    --policy-name dp3 \
    --task-name pick_apple_messy \
    --exp-name 2026-04-01_11-18_233

# ── 单文件 smoke test ─────────────────────────────────────────────────────────
python -m dexmani_policy.agents.core.dp3
python -m dexmani_policy.agents.core.maniflow
