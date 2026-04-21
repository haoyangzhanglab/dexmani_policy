# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

运行环境：conda env `policy`（`conda activate policy`）。所有脚本必须从项目根目录运行（Hydra 以 CWD 解析 config 路径）。

## Commands

```bash
# 安装（可编辑模式）
pip install -e .

# 训练
python dexmani_policy/train.py --config-name dp3 seed=233 task_name=multi_grasp
# 可用 config：dp, dp3, moe_dp3, maniflow
# 常用覆盖：task_name=<task> seed=<int> training.loop.num_epochs=2000 dataloader.batch_size=64

# 评估
python dexmani_policy/eval_sim.py --config-name dp3 task_name=multi_grasp \
  eval.sim.ckpt_tag_or_path=best eval.sim.eval_episodes=100
# 或通过 exp-name 指定实验目录（见 eval_sim.py 参数）

# WandB 同步
wandb sync ./wandb --include-offline --sync-all

# 单模块 smoke test（验证 import 和基本 shape）
python -m dexmani_policy.agents.core.dp3
python -m dexmani_policy.agents.obs_encoder.backbone_2d.resnet
python -m dexmani_policy.datasets.pointcloud_dataset
```

## Architecture

### 整体流程

```
obs_dict {point_cloud: (B,T,N,3), joint_state: (B,T,D)}
    └─► BaseAgent.preprocess()          # normalize + slice n_obs_steps + flatten B*T
            └─► ObsEncoder.forward()    # → cond: (B, out_dim) [film] 或 (B, T, out_dim) [cross_attn]
                    └─► ActionDecoder.compute_loss / predict_action
                                └─► action chunk (B, horizon, action_dim)
```

**关键维度约定**（`preprocess` 之后）：
- `film` 模式：encoder 输出 `(B, out_dim * n_obs_steps)` 展平向量
- `cross_attn` 模式：encoder 输出 `(B, n_obs_steps, out_dim)` 序列
- `predict_action` 返回 `pred_action (B, horizon, action_dim)` 和 `control_action (B, n_action_steps, action_dim)`，后者从 `pred[:, n_obs_steps-1:]` 截取

### Agents（`agents/core/`）

所有 agent 继承 `BaseAgent`（`agents/core/base.py`），通过 `hydra.utils.instantiate(cfg.agent)` 构建。

| Agent 类 | config | 观测 | 解码器 |
|---|---|---|---|
| `DP3Agent` | `dp3` | point cloud + joint_state | ConditionalUnet1D + DDIM |
| `DPAgent` | `dp` | RGB + joint_state | ConditionalUnet1D + DDIM |
| `MoEDP3Agent` | `moe_dp3` | point cloud + joint_state + MoE | ConditionalUnet1D + DDIM |
| `ManiFlowAgent` | `maniflow` | point cloud + joint_state | DiTXFlowMatch + consistency |

`BaseAgent` 提供：`preprocess()`、`compute_loss()`、`predict_action()`、`configure_optimizer()`、`load_normalizer_from_dataset()`。

`ManiFlowAgent.compute_loss()` 额外接收 `ema_model` kwarg 作为 consistency teacher。

### Observation Encoders（`agents/obs_encoder/`）

**点云全局编码器**（`pointcloud/registry.py: build_pc_global_encoder`）：
- `dp3` → `PointNet`（原版 DP3）
- `idp3` → `MultiStagePointNet`（iDP3 多阶段）
- `pointnext` → `PointNextEncoder`（分层 SA 模块）
- 输出：`{'global_token': (B*T, pc_out_dim)}`

**点云 patch tokenizer**（`build_pc_patch_tokenizer`）：
- `pointpn` → `PointPNTokenizer`
- `pointnext_tokenizer` → `PointNextPatchTokenizer`
- 输出 token 序列，供 DiTX 使用

**RGB 编码器**（`backbone_2d/`）：ResNet / CLIP / DINO / SigLIP，均输出 512-dim `global_token`。

**Proprio**：`StateMLP`，输出 64-dim（默认）。

**Plugins**（`obs_encoder/plugins/`）：`MoE`（sparse top-k router，loss 中含 load-balance 项）、`TokenCompressor`（cross+self attention 降维）。

### Action Decoders（`agents/action_decoders/`）

- **`Diffusion`**（`diffusion.py`）：包装任意 backbone + `diffusers.DDIMScheduler`。`prediction_type`: `epsilon` 或 `sample`。
- **`FlowMatchWithConsistency`**（`flowmatch.py`）：rectified flow + consistency distillation。batch 按 `flow_batch_ratio`/`consistency_batch_ratio`（默认 0.75/0.25）拆分；consistency 分支用 EMA 模型作 teacher。
- **Backbones**：`ConditionalUnet1D`（FiLM 或 cross-attn 条件化）、`DiTXFlowMatch`（AdaLN-Zero、RMSNorm、cross-attn context）。

### Data Pipeline

- **Zarr 数据集**：`robot_data/sim/<task_name>.zarr`。必需 key：`point_cloud (N,3)`、`joint_state (D,)`、`action (A,)`；可选：`rgb`、`depth`、`camera_intrinsic`、`camera_extrinsic`。
- **`ReplayBuffer`**（`datasets/common/replay_buffer.py`）：内存映射 Zarr，懒加载。
- **`SequenceSampler`**（`datasets/common/sampler.py`）：Numba-JIT 索引构建，采样长度 `horizon` 的窗口，episode 边界用零填充（`pad_before = n_obs_steps-1`，`pad_after = n_action_steps-1`）。
- **Dataset 类**：`BaseDataset`（点云+state）、`RGBDataset`、`SemGeoDataset`，均通过 `get_normalizer()` 返回 `LinearNormalizer`。

### Training Infrastructure

- **`train.py`**：入口。`build_train_components(cfg)` 构建所有组件，`Trainer.train(resume_tag="latest")` 启动训练。
- **`Trainer`**（`training/trainer.py`）：主循环。按 `TrainLoopConfig` 控制 log / val / eval / checkpoint 频率。`eval_interval_epochs > 0` 且有 `env_runner` 时触发仿真评估。
- **`TrainWorkspace`**（`training/common/workspace.py`）：管理输出目录、WandB（offline 模式）、JSONL 日志。
- **`CheckpointIO`**（`training/common/checkpoint_io.py`）：原子保存，`latest.pt` 软链接，JSON manifest top-k 追踪（自动删除低分 checkpoint）。监控 key 默认 `test_mean_score`（max 模式，top-3）。
- **`EMAModel`**（`training/common/ema_model.py`）：EMA 权重，用于 eval 推理和 ManiFlow consistency teacher。
- **`LinearNormalizer`**（`common/normalizer.py`）：`limits`（min-max → [-1,1]）或 `std` 模式。在 dataset 上 fit 一次，存入 agent，checkpoint 中随模型一起保存。

### Configuration System

Hydra configs 在 `dexmani_policy/configs/`，通过 `--config-name` 选择。

**关键默认值**（dp3.yaml）：
- `horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19`
- `batch_size=256, lr=1e-4, num_epochs=1000, lr_warmup_steps=500`
- `eval_interval_epochs=250, val_interval_epochs=25`
- `num_training_steps=100, num_inference_steps=10`（DDIM）

**输出目录**：`experiments/<policy_name>/<task_name>/<date>_<time>_<seed>/`
- `config.yaml`：resolved Hydra config
- `checkpoints/`：`.pt` 文件（含 model、ema_model、optimizer、scheduler、epoch）
- `*.jsonl`：append-only 训练日志
- `wandb/`：offline WandB 数据

评估输出：`experiments/.../eval/<timestamp>/`，含 `eval_metrics.json` 和每个 `denoise_timesteps` 的 `.mp4` 视频。

### External Dependency

`dexmani_sim`：仿真环境（不在本仓库）。由 `env_runner/sim_runner.py` 动态导入，需单独安装。提供 `DATA_DIR` 和 `DATA_DIR/eval_seeds/<task_name>.txt`（可复现 eval 的随机种子文件）。
