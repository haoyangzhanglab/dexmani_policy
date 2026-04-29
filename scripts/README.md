# 训练脚本使用指南

## 快速开始

### 单卡训练

```bash
# 基础用法（使用 GPU 0）
bash scripts/train.sh dp3
bash scripts/train.sh maniflow
bash scripts/train.sh moe_dp3

# 指定 GPU（多卡服务器上选择特定卡）
bash scripts/train.sh dp3 training.device=cuda:1
bash scripts/train.sh dp3 training.device=cuda:3

# 或使用环境变量
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh dp3

# 覆盖其他配置
bash scripts/train.sh dp3 task_name=sim/pick_apple_messy
bash scripts/train.sh dp3 training.seed=42
```

### 多卡训练

```bash
# 使用默认 GPU (0,1,2,3)
bash scripts/train_ddp.sh maniflow_ddp

# 指定 GPU
bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[0,1,2,3]
bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[1,2,3,4]

# 使用非连续 GPU
bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=2 training.gpu_ids=[0,7]

# 通过环境变量选卡
CUDA_VISIBLE_DEVICES=2,3,5,6 bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=4
```

---

## 常用配置覆盖

### 任务和数据

```bash
# 切换任务
bash scripts/train.sh dp3 task_name=sim/pick_apple_messy

# 修改数据路径
bash scripts/train.sh dp3 dataset.zarr_path=/path/to/data.zarr

# 调整 batch size
bash scripts/train.sh dp3 dataloader.batch_size=64
```

### 训练参数

```bash
# 修改训练轮数
bash scripts/train.sh dp3 training.loop.num_epochs=500

# 修改学习率
bash scripts/train.sh dp3 optimizer.lr=5e-5

# 修改随机种子
bash scripts/train.sh dp3 training.seed=42

# 调整评估频率
bash scripts/train.sh dp3 training.loop.eval_interval_epochs=50

# 指定 GPU（多卡服务器）
bash scripts/train.sh dp3 training.device=cuda:2
```

### 输出目录

```bash
# 自定义输出目录
bash scripts/train.sh dp3 workspace.output_dir=outputs/my_experiment

# 自定义 wandb 名称
bash scripts/train.sh dp3 workspace.wandb_cfg.name=my_run
```

---

## 多卡训练注意事项

### GPU 选择方式

**方式 1: 使用默认 GPU**
```bash
# 使用 GPU 0,1,2,3
bash scripts/train_ddp.sh maniflow_ddp
```

**方式 2: 通过 gpu_ids 指定**
```bash
# 使用 GPU 1,2,3,4
bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[1,2,3,4]

# 使用 GPU 0 和 7
bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=2 training.gpu_ids=[0,7]
```

**方式 3: 通过环境变量**
```bash
# CUDA_VISIBLE_DEVICES 重映射 GPU 编号
# 物理 GPU 2,3,5,6 → 逻辑 GPU 0,1,2,3
CUDA_VISIBLE_DEVICES=2,3,5,6 bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=4
```

### 配置要求

多卡训练必须使用包含 `training.num_gpus` 的 DDP 配置文件：
- ✅ `maniflow_ddp.yaml`
- ✅ `dp3_ddp.yaml` (如果存在)
- ❌ `maniflow.yaml` (单卡配置)

---

## 评估和测试

### 仿真评估

```bash
python dexmani_policy/eval_sim.py \
    --policy-name dp3 \
    --task-name pick_apple_messy \
    --exp-name 2026-04-01_11-18_233
```

### 单文件测试

```bash
# 测试策略模块
python -m dexmani_policy.agents.core.dp3
python -m dexmani_policy.agents.core.maniflow
python -m dexmani_policy.agents.core.moe_dp3
```

---

## 常见问题

### Q: 如何断点续训？

**A**: 训练会自动从 `latest.pt` 恢复，无需额外参数。

### Q: 如何在多卡服务器上使用特定 GPU？

**A**: 
- **单卡训练**: `training.device=cuda:N` 或 `CUDA_VISIBLE_DEVICES=N`
- **多卡训练**: `training.gpu_ids=[...]` 或 `CUDA_VISIBLE_DEVICES=...`

示例:
```bash
# 单卡使用 GPU 3
bash scripts/train.sh dp3 training.device=cuda:3
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh dp3

# 多卡使用 GPU 4,5,6,7
bash scripts/train_ddp.sh maniflow_ddp training.gpu_ids=[4,5,6,7]
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_ddp.sh maniflow_ddp training.num_gpus=4
```

### Q: 多卡训练时 GPU 空闲？

**A**: 主进程执行评估时其他 GPU 会等待。建议增大 `training.loop.eval_interval_epochs`。

### Q: 如何查看所有可配置参数？

**A**: 查看 `dexmani_policy/configs/` 目录下的 YAML 文件。

---

## 配置文件说明

| 配置文件 | 策略 | 观测 | 用途 |
|---------|------|------|------|
| `dp.yaml` | DPAgent | RGB | 单卡训练 |
| `dp3.yaml` | DP3Agent | 点云 | 单卡训练 |
| `moe_dp3.yaml` | MoEAgent | 点云 | 单卡训练 |
| `maniflow.yaml` | ManiFlowAgent | 点云 | 单卡训练 |
| `maniflow_ddp.yaml` | ManiFlowAgent | 点云 | 多卡训练 |

---

## 更多信息

- 训练机制设计: `docs/训练机制设计.md`
- 项目文档: `CLAUDE.md`
- 配置文件: `dexmani_policy/configs/`
