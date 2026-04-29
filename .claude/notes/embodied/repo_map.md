# dexmani_policy — Embodied Repo Research Note

> Generated: 2026-04-29. Re-run `/embodied-repo-map` if repo structure changes significantly.

## TL;DR

灵巧操作模仿学习框架。4 种策略（DP3、DP、MoE-DP3、ManiFlow）× 多模态观测（点云/RGB/Proprio）× 单卡/DDP 训练。核心特色：Hydra 全驱动实例化、ManiFlow 的 Flow Matching + Consistency Distillation 双目标训练、EMA teacher 支持。

---

## Execution Skeleton

### 单卡训练
```
train.py::main(cfg)
  build_train_components(cfg)
    hydra.utils.instantiate(cfg.dataset)    → PC/RGB/RGBPCDataset
    dataset.get_normalizer()                 → LinearNormalizer (joint_state+action → [-1,1])
    hydra.utils.instantiate(cfg.agent)       → DP3/DP/MoE/ManiFlowAgent
    model.configure_optimizer()             → AdamW (obs/action 分组不同 lr)
    get_scheduler()                         → CosineAnnealing (warmup=500 steps)
    EMAModel(copy.deepcopy(model))
    SimRunner + TrainWorkspace
  Trainer.train("latest")
    workspace.load_for_resume()             → 从 latest.pt 断点续训
    for epoch in range(start_epoch, num_epochs):
      for batch in train_loader:
        model.compute_loss(batch, ema_model)
        loss.backward() → clip_grad_norm_ → optimizer.step()
        ema_updater.step(model)
      [eval_interval] env_runner.run(ema_model)  → success_rate
      [val_interval]  compute val_loss(ema_model)
      workspace.save_topk / save_latest
```

### DDP 多卡训练
```
train_ddp.py → mp.spawn(ddp_worker, nprocs=num_gpus)
  setup_ddp() → dist.init_process_group("nccl")
  model = DDP(model, device_ids=[gpu_id])   # EMA 不被 DDP 包装
  synchronize_states()                       # broadcast main/ema/normalizer from rank-0
  for epoch:
    train_sampler.set_epoch(epoch)
    [训练同单卡，DDP 自动 all-reduce]
    [rank-0 only] validate / evaluate / checkpoint
    dist.barrier()
```

### 评估流程
```
eval_sim.py → SimEvaluator.run()
  workspace.load_for_inference(agent, use_ema=True)
  for denoise_timesteps in [10, 5, 1]:
    env_runner.run(agent) → eval_one_episode()
      obs_deque(maxlen=n_obs_steps+1) 滚动队列
      predict_action(nobs, denoise_timesteps)
      execute action_chunk[n_action_steps=8 步]
  save MP4 + metrics.json
```

---

## Repository Map

### 关键文件路径

| 模块 | 路径 |
|------|------|
| 单卡入口 | `dexmani_policy/train.py` |
| DDP 入口 | `dexmani_policy/train_ddp.py` |
| 评估入口 | `dexmani_policy/eval_sim.py` |
| 基础 Agent | `agents/core/base.py` |
| DP3 Agent | `agents/core/dp3.py` |
| DP Agent | `agents/core/dp.py` |
| MoE Agent | `agents/core/moe.py` |
| ManiFlow Agent | `agents/core/maniflow.py` |
| DDIM 解码器 | `agents/action_decoders/diffusion.py` |
| Flow 解码器 | `agents/action_decoders/flowmatch.py` |
| t 采样库 | `agents/action_decoders/common/sample.py` |
| UNet1D | `agents/action_decoders/backbone/unet1d.py` |
| DiTX | `agents/action_decoders/backbone/ditx.py` |
| PC 编码器注册表 | `agents/obs_encoder/pointcloud/registry.py` |
| RGB 编码器注册表 | `agents/obs_encoder/rgb/registry.py` |
| StateMLP | `agents/obs_encoder/proprio/state_mlp.py` |
| MoE 插件 | `agents/obs_encoder/plugins/moe.py` |
| Token Compressor（未激活）| `agents/obs_encoder/plugins/token_compressor.py` |
| Dataset 基类 | `datasets/base_dataset.py` |
| PC Dataset | `datasets/pc_dataset.py` |
| Replay Buffer | `datasets/common/replay_buffer.py` |
| 序列采样器 | `datasets/common/sampler.py` |
| 单卡训练循环 | `training/trainer.py` |
| DDP 训练循环 | `training/ddp_trainer.py` |
| EMA | `training/common/ema_model.py` |
| Checkpoint | `training/common/checkpoint_io.py` |
| Workspace | `training/common/workspace.py` |
| 仿真评估器 | `training/sim_evaluator.py` |
| 环境 Runner 基类 | `env_runner/base_runner.py` |
| Sim Runner | `env_runner/sim_runner.py` |
| 归一化器 | `common/normalizer.py` |

---

## Main Entrypoints

| 命令 | 用途 |
|------|------|
| `python dexmani_policy/train.py --config-name=dp3` | 单卡训练 DP3 |
| `python dexmani_policy/train.py --config-name=maniflow` | 单卡训练 ManiFlow |
| `python dexmani_policy/train_ddp.py --config-name=maniflow_ddp` | DDP 多卡训练 |
| `bash scripts/train_ddp.sh dp3 pick_apple_messy 4` | Shell 脚本启动 DDP |
| `python dexmani_policy/eval_sim.py --policy-name dp3 --task-name ... --exp-name ...` | 仿真评估 |
| `python -m dexmani_policy.agents.core.dp3` | DP3 smoke test |
| `python -m dexmani_policy.agents.core.maniflow` | ManiFlow smoke test |

---

## Embodied Module Breakdown

### 1. 策略 Agent 继承体系

```
nn.Module
└── BaseAgent                      [agents/core/base.py]
    ├── UNetDiffusionAgent
    │   ├── DP3Agent               点云 + DDIM
    │   ├── DPAgent                RGB + DDIM
    │   └── MoEAgent               点云 + MoE路由 + DDIM
    └── DiTXFlowMatchAgent
        └── ManiFlowAgent          点云 patch token + Flow+Consistency
```

**BaseAgent 核心接口**（`agents/core/base.py`）：

| 方法 | 说明 |
|------|------|
| `preprocess(obs_dict)` | normalize → 切 n_obs_steps → flatten(B*T, ...) |
| `compute_loss(batch, **kwargs)` | obs→encoder→cond; action→normalize; decoder.compute_loss |
| `predict_action(obs_dict, denoise_timesteps)` | cond→decoder.predict_action→unnorm; s=n_obs_steps-1 切片 |
| `configure_optimizer(lr, wd, obs_lr, ...)` | obs/action 分组 AdamW，支持不同 lr |

### 2. Obs 编码器分类

**点云全局编码器**（→ DP3/MoE，输出 `(B*T, 256)` 全局 token）：

| 名称 | 类 | 文件 | 架构 |
|------|-----|------|------|
| `dp3` | `PointNet` | `pointnet.py` | MLP → max-pool |
| `idp3` | `MultiStagePointNet` | `pointnet.py` | 多阶段局部+全局融合 |
| `pointnext` | `PointNextEncoder` | `pointnext.py` | SetAbstraction + InvertedResidual + 全局位置编码 |

**点云 Patch Tokenizer**（→ ManiFlow，输出 patch 序列）：

| 名称 | 类 | out_shape (K, D) |
|------|-----|-----------------|
| `pointpn` | `PointPNTokenizer` | `(128, 512)` |
| `pointnext_tokenizer` | `PointNextPatchTokenizer` | `(96, 128)` + `(1, 128)` global |

**RGB 骨干**（→ DP，输出 `(B*T, 512)` global token）：

| 名称 | 骨干 | tune_mode |
|------|------|-----------|
| `resnet` | ResNet18 | full |
| `dino` | dinov2-base | freeze/full/lora |
| `clip` | clip-vit-base-patch32 | freeze/full/lora |
| `siglip` | siglip-base-patch16-224 | freeze/full/lora |

**Proprio**：`StateMLP`：`(B*T, 19) → Linear → ReLU → Linear → (B*T, 64)`

**插件**：
- `MoE`（`plugins/moe.py`）：`num_experts=8, top_k=2`，输出加权专家求和 + load_balance/entropy aux loss
- `TokenCompressor`（`plugins/token_compressor.py`）：Perceiver-style，N query → K latent token，**当前未激活**

### 3. Action 解码器

| 解码器 | 骨干 | 训练目标 | 推理 |
|--------|------|---------|------|
| `Diffusion` | `ConditionalUnet1D` | MSE(pred, x0)，t~Uniform(0,99) | 10步 DDIM |
| `FlowMatchWithConsistency` | `DiTXFlowMatch` | flow MSE + consistency MSE，t~Beta(1,1.5) | 10步 Euler ODE |

**FlowMatchWithConsistency 双目标**（`flowmatch.py`）：
```
Flow 部分（75% batch）：
  x_t = (1-t)*x_noise + t*x_data
  v_target = x_data - x_noise
  loss_flow = MSE(model(x_t, t, dt=0, cond), v_target)

Consistency 部分（25% batch）：
  EMA teacher 预测 v(x_{t_next}, t_next, dt)
  pred_x1 = x_{t_next} + v_teacher * (1-t_next)
  consistency_v_target = (pred_x1 - x_{t_ct}) / (1-t_ct)
  loss_consistency = MSE(model(x_{t_ct}, t_ct, dt, cond), consistency_v_target)

total_loss = loss_flow + loss_consistency
```

**UNet1D 条件注入**：`film`（全局向量 FiLM）或 `cross_attention_film`（obs 序列 CrossAttn + FiLM）

**DiTX 条件注入**：时间步 → AdaLN-Zero（9×D 调制参数，零初始化）；obs token → Cross-Attention；final layer fc2 零初始化

**t 采样模式**（`common/sample.py::SampleLibrary`）：`uniform` / `beta` / `lognorm` / `mode` / `cosmap` / `discrete` / `discrete_pow`

### 4. 数据管道

```
Zarr 文件 (episode_ends + data/{key})
  → ReplayBuffer.copy_from_path()     # 全量加载到 numpy 内存
  → SequenceSampler.create_indices()  # numba JIT，生成 horizon 窗口索引
  → Dataset.__getitem__()
      pad_before=1 / pad_after=7 边界处理（首/末帧重复）
      augmentation (PCAug / RGBAug)
  → DataLoader (num_workers=8, pin_memory)
```

**样本形状**（horizon=16, N=1024 点）：
- `point_cloud: (16, 1024, 3)`
- `joint_state: (16, 19)`
- `action: (16, 19)`

**归一化策略**：

| 数据类型 | 是否归一化 | 方法 |
|---------|----------|------|
| `joint_state` | 是 | limits → [-1, 1] |
| `action` | 是 | limits → [-1, 1] |
| `point_cloud` | 否 | 原始坐标 (m) |
| `rgb` | 否 | ImageProcessor 单独处理 |

### 5. Config 系统（Hydra）

| 配置文件 | 策略 | Dataset | Agent |
|---------|------|---------|-------|
| `configs/dp3.yaml` | DP3 | `PCDataset` | `DP3Agent` |
| `configs/dp.yaml` | DP | `RGBDataset` | `DPAgent` |
| `configs/moe_dp3.yaml` | MoE | `PCDataset` | `MoEAgent` |
| `configs/maniflow.yaml` | ManiFlow | `PCDataset` | `ManiFlowAgent` |
| `configs/maniflow_ddp.yaml` | ManiFlow DDP | 继承 maniflow | 继承 maniflow，覆盖 num_gpus=4 |

Hydra 扩展：`OmegaConf.register_new_resolver("eval", eval)` 支持 yaml 内动态计算，如 `${eval:'${n_obs_steps} - 1'}`。

### 6. 关键超参

| 参数 | 默认值 |
|------|--------|
| `horizon` | 16 |
| `n_obs_steps` | 2 |
| `n_action_steps` | 8 |
| `action_dim` | 19 |
| `num_epochs` | 1000 |
| `lr` | 1e-4 |
| `lr_warmup_steps` | 500 |
| `EMA power` | 0.75，max_value=0.9999 |
| `Diffusion train/infer steps` | 100 / 10 |
| `FlowMatch denoise_timesteps` | 10 |
| `flow:consistency ratio` | 0.75:0.25 |
| `Checkpoint topk` | 3，monitor=success_rate |

### 7. 推理时间对齐

```
训练: obs[t:t+2] 对应 pred action[t:t+16]
推理: obs_history = [obs[t-1], obs[t]] → pred[0:16]
执行: control_action = pred[1:9]  (s = n_obs_steps-1 = 1)
```
pred[0] 对应历史时刻 t-1（不执行），pred[1] 对应当前 t。

### 8. EMA 与 Checkpoint

**EMAModel** (`training/common/ema_model.py`)：
- warmup 公式：`decay = 1 - (1 + step/inv_gamma)^(-power)`（power=0.75）
- `requires_grad=False` 参数直接 copy；`requires_grad=True` 参数做 EMA 更新；buffers 直接 copy
- 约 10K optimizer steps 时 decay 达 0.999

**CheckpointStore** (`training/common/checkpoint_io.py`)：
- 原子写：先写 `.tmp` 再 `replace`
- `latest.pt`：symlink，支持断点续训
- `TopKCheckpointTracker`：维护 `topk_manifest.json`，按 score 降序，超出 k 时删除最差

---

## Reproducibility Checklist

- [x] Hydra config 完整，所有组件通过 `_target_` 实例化
- [x] EMA checkpoint 支持断点续训（`latest.pt` symlink）
- [x] episode 级 train/val 划分（val_ratio=0.05）
- [x] DDP 状态同步（main model、EMA、normalizer 均 broadcast from rank-0）
- [x] fixed eval seeds（来自 `dexmani_sim/eval_seeds/{task}.txt`）
- [ ] **dexmani_sim 外部依赖**：不在仓库内，动态 importlib 加载，仿真评估必须安装
- [ ] **smoke test 需伪造 batch 数据**：`python -m agents.core.dp3` 在无真实数据时需构造 dummy input
- [ ] ManiFlow val_loss 不含 consistency loss，训练/验证指标不完全可比

---

## Ablation Surface

| 消融维度 | 配置/文件 | 预期效果 |
|---------|---------|---------|
| PC 编码器：`dp3` vs `idp3` vs `pointnext` | `dp3.yaml::agent.obs_encoder.pc_encoder_type` | 感受野和局部结构建模能力 |
| UNet 条件注入：`film` vs `cross_attention_film` | `dp3.yaml::agent.obs_encoder.cond_type` | 序列 obs 的利用效率 |
| Flow:Consistency 比例：75%:25% → 100%:0% | `flowmatch.py::flow_batch_ratio` | Consistency distillation 的贡献 |
| ManiFlow t 采样：`beta` vs `uniform` vs `lognorm` | `flowmatch.py::t_sampler_type` | 训练分布偏差对收敛的影响 |
| EMA 开关 | `training.use_ema` | EMA 对推理稳定性的贡献 |
| RGB 骨干 freeze vs LoRA vs full | `dp.yaml::agent.obs_encoder.tune_mode` | 预训练特征迁移效果 |
| DiTX block 数量 | `ditx.yaml::num_layers` | 模型容量 vs 训练速度 |
| 点云点数：1024 → 512 | `dp3.yaml::agent.obs_encoder.num_points` | 推理速度 vs 几何精度 |
| action horizon：16 → 8 | `configs/*.yaml::horizon` | 规划 horizon 对鲁棒性影响 |

---

## Open Questions

1. **ManiFlow consistency dt 设计**（`flowmatch.py:76`）：`dt2 = dt1.clone()`，teacher 和 student 使用同一 dt，与标准 Consistency Models 解耦 t-sequence 做法不同，是否有意为之？

2. **Validation loss 可比性**：ManiFlow val_loss 只含 flow 部分，与 train_loss（flow + consistency）不可比，长期监控可能误导 early stopping。当前注释说明是有意设计（防止 teacher=student 退化）。

3. **TokenCompressor 未激活**：`agents/obs_encoder/plugins/token_compressor.py` 已完整实现但未在任何配置中启用，是否计划接入 ManiFlow 的 obs token 压缩？

4. **文本编码器未激活**：`agents/obs_encoder/text/` 存在 CLIP/T5 编码器但未集成任何 Agent，是否计划做 language-conditioned policy？

5. **DDP EMA 冗余更新**：非 rank-0 的 EMA 更新在 broadcast 后被覆盖，计算量浪费，可考虑只在 rank-0 更新。
