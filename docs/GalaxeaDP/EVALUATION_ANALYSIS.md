# GalaxeaDP 评测机制深度分析

## 一、评测体系总览

GalaxeaDP 提供**两级独立评估管线**，分别解决不同层面的验证问题：

```
┌─────────────────────────────────────────────────────────────┐
│                    训练完成后                                │
│                          │                                  │
│              ┌───────────┴───────────┐                      │
│              ▼                       ▼                      │
│  eval_open_loop.py          eval_sim.py                     │
│  开环评估                   闭环仿真评估                     │
│  数据集 val split           Gymnasium 仿真环境               │
│  离线批量推理               在线循环推理                     │
│  轨迹拟合度可视化           成功率 + 视频                    │
│  无仿真依赖                 需要 GalaxeaManipSim             │
└─────────────────────────────────────────────────────────────┘
```

两条管线完全解耦：没有共享的评估基类、没有统一的评估接口，各自通过独立的 shell 脚本启动，均基于 Hydra 解析配置。

---

## 二、开环评估（Open-Loop Evaluation）

### 2.1 评估目的

回答一个问题：**模型对数据集动作的拟合程度如何？**

- 不依赖仿真环境，纯 GPU 推理
- 训练完成后快速验证，确认模型"学懂了数据"
- 发现过拟合/欠拟合、维度级的问题

### 2.2 完整流程

```
eval_open_loop.py (Hydra @hydra.main)
    │
    ├─► hydra.utils.instantiate(model)     → DiffusionPolicyBCModule
    ├─► hydra.utils.instantiate(datamodule)→ BaseDataModule
    ├─► model.load_state_dict(ckpt_path)
    ├─► policy = model.policy.cuda().eval()
    │
    ├─► datamodule.setup('predict')
    ├─► dataloader = datamodule.val_dataloader()
    │
    ├─► for batch in dataloader:
    │       with torch.no_grad():
    │           cur_pd_action = policy.predict_action(batch)  # (B, horizon, action_dim)
    │       cur_gt_action = batch["gt_action"]
    │       收集到 gt_actions[], pd_actions[]
    │
    ├─► gt_actions = concat[:, 0, :]       # 只取第一步 GT
    │   pd_actions = concat                # 保留完整 chunk
    │
    ├─► 按 episode 分组:
    │       episode_data_index["from"] / ["to"] 确定帧范围
    │
    └─► for each episode:
            plot_result(path, gt, pd)      # Plotly HTML 交互式图表
```

### 2.3 关键设计细节

#### （1）GT 只取第一步

```python
gt_actions = np.concatenate(gt_actions, axis=0)[:, 0, :]  # (N, action_dim)
pd_actions = np.concatenate(pd_actions, axis=0)            # (N, horizon, action_dim)
```

- GT 动作只取 `[:, 0, :]`，即每个时间步对应的**第一步真实动作**
- 预测动作保留完整的 `(N, horizon, action_dim)` chunk
- 这反映了"策略从当前观测出发的预测能力"与"实际执行的第一步动作"的对比

**为什么这样设计？** 扩散策略的 action chunking 机制中，每个推理时刻产生的是一整个未来动作序列。开环评估关心的是：在任意时刻 t，策略预测的 chunk 中各个时间步的动作，与数据集中对应时间步的真实动作的偏差。GT 取第一步是因为 DataLoader 的 `delta_timestamps` 对每个采样帧只提供了 `t=0` 的 GT 动作窗口。

#### （2）按 Episode 分组可视化

```python
episode_from = dataset.episode_data_index["from"]
episode_to = dataset.episode_data_index["to"]
for idx in range(dataset.num_episodes):
    cur_path = output_dir / f"{idx:06}"
    cur_gt_action = gt_actions[episode_from[idx]: episode_to[idx]]
    cur_pd_action = pd_actions[episode_from[idx]: episode_to[idx]]
    plot_result(cur_path, cur_gt_action, cur_pd_action)
```

每个 episode 一个独立目录，包含每个 action 维度的 HTML 文件。

#### （3）Plotly 可视化（`plot_result`）

```python
# 每个 action 维度 d 一个 HTML
for d in range(dim):
    fig = go.Figure()
    # 预测轨迹：按时间步分组，不同组不同颜色（5 色循环）
    for t in range(episode_size):
        color_idx = t % len(PALETTE)  # 5 色循环
        fig.add_trace(go.Scatter(
            x=np.arange(t, t + chunk_size),
            y=pd[t, :, d],
            line=dict(color=PALETTE[color_idx]),
        ))
    # 真实动作：红色曲线叠加
    fig.add_trace(go.Scatter(x=np.arange(episode_size), y=gt[:, d],
                              name="gt", line=dict(color="red")))
    fig.write_html(path / f"{d:02}.html")
```

**可视化效果**：
- 预测动作按推理时刻分组，用 5 种绿色深浅区分不同推理时刻产生的 chunk
- 真实动作用一条红色曲线贯穿全图
- 可以直观看到策略预测的连续性和对 GT 的贴合程度
- HTML 格式支持交互式缩放、悬停查看数值

**5 色循环的设计意义**：同一次推理产生的 chunk 内动作用同一颜色，相邻推理时刻的 chunk 颜色不同。这样可以直观看到 chunk 边界处的连续性 —— 如果颜色切换处出现跳跃，说明 action chunking 的重复推理存在不连续问题。

### 2.4 启动方式

```bash
bash scripts/eval_open_loop.sh trainer.devices=[0] task=open_galaxea/<robot>/<task> \
    ckpt_path=out/open_galaxea/<robot>/<task>/<time>/checkpoints/step_20000.ckpt
```

脚本自动解析 task 参数，配置 Hydra 的输出目录和 WandB 项目/组/名称。

### 2.5 开环评估的局限性

1. **只反映拟合度，不反映策略性能**：模型可能完美拟合数据集，但在闭环中失败（因为数据集动作本身就不成功）
2. **开环假设**：每一步的观测都来自数据集，而非策略自己的执行结果。没有误差累积效应
3. **无定量评分**：目前只有可视化，没有 DTW 距离、MSE 等自动相似度指标
4. **验证集依赖**：评估质量取决于 val split 的代表性

---

## 三、闭环仿真评估（Simulation Evaluation）

### 3.1 评估目的

回答一个问题：**策略在仿真环境中实际执行时，能不能完成任务？**

- 真实的闭环控制：策略的输出影响下一步观测
- 模拟实际部署场景
- 输出可量化的成功率指标

### 3.2 完整流程

```
eval_sim.py (Hydra @hydra.main)
    │
    ├─► hydra.utils.instantiate(model)
    ├─► model.load_state_dict(ckpt_path)
    ├─► policy = model.policy.cuda().eval()
    │
    ├─► env = gym.make(cfg.env, control_freq=30, headless=True,
    │                   max_episode_steps=600, controller_type=target_controller_type)
    │
    └─► for eval_idx in range(num_evaluations):  # 默认 100 次
            │
            ├─► env.reset(seed=42)  # 固定种子，保证可复现
            ├─► if save_video: env.render(); frames.append(render())
            │
            ├─► action_seq = None, seq_idx = 0
            │
            ├─► while not done:
            │       │
            │       ├─► if action_seq is None or seq_idx >= len(action_seq):
            │       │       │
            │       │       ├─► 根据 controller_type 构建观测 batch:
            │       │       │   ├── bimanual_relaxed_ik:  make_single_frame_batch_eef()
            │       │       │   │   └─► 观测: ee_pose(7d) + gripper(1d) × 2 臂
            │       │       │   └── bimanual_joint_position: make_single_frame_batch_joints()
            │       │       │       └─► 观测: arm_joints(7d) + gripper(1d) × 2 臂
            │       │       │
            │       │       ├─► 图像: cv2.resize(224×224) → (1,1,3,224,224)
            │       │       ├─► 状态: reshape(1, 1, -1)
            │       │       │
            │       │       ├─► with torch.inference_mode():
            │       │       │       pred = policy.predict_action(batch)
            │       │       │           # (1, horizon, action_dim)
            │       │       │
            │       │       └─► action_seq = pred[:,:num_action_steps,:].squeeze()
            │       │           # 取前 num_action_steps 步 (默认 16)
            │       │           # 丢弃后 (horizon - 16) 步
            │       │
            │       ├─► action = action_seq[seq_idx]
            │       ├─► seq_idx += 1
            │       ├─► env.step(action)
            │       ├─► actions_log.append(action[:8])  # 只记前 8 维
            │       ├─► if save_video: frames.append(env.render())
            │       └─► terminated or truncated → done
            │
            ├─► terminated → num_success++
            │
            ├─► save_dict_list_to_json(infos, output_dir / "info.json")
            ├─► save_video_ffmpeg(frames, output_dir / f"rollout_{idx}.mp4")
            │       # H.264 编码, yuv420p
            │
            └─► plot_action_trajectories(actions_array, output_dir / f"rollout_{idx}.png")
                    # 8 维动作曲线图, 标注 xyzwxyzG
                    # G = Gripper
```

### 3.3 Action Chunking 推理机制

这是评估的核心循环，也是 Diffusion Policy 的标志性设计：

```
时间线: |-- 推理产生 32 步 --|-- 执行前 16 步 --|-- 推理产生 32 步 --|-- 执行前 16 步 --|
         (horizon=32)        (num_action_steps=16)

每次推理:
1. 输入: 当前单帧观测 (1, 1, 3, 224, 224) + 状态 (1, 1, state_dim)
2. 输出: pred (1, horizon, action_dim)
3. 截取: pred[:, :16, :] → action_seq
4. 逐步执行 action_seq 的每一步，每次 env.step()
5. action_seq 执行完毕 → 重新推理
```

**为什么只取一半（16/32）？**
- 扩散策略的前几步预测通常比后几步更准确（条件信息更充分）
- 取前半段可以减少累积误差
- 30Hz 控制下，16 步 = ~0.53 秒推理一次

### 3.4 两种控制模式的观测适配

| 模式 | 控制器类型 | 观测 state 键 | 动作维度 |
|------|-----------|--------------|----------|
| 末端控制 | `bimanual_relaxed_ik` | `left_ee_pose`(7d) + `left_gripper`(1d) + `right_ee_pose`(7d) + `right_gripper`(1d) | 16 |
| 关节控制 | `bimanual_joint_position` | `left_arm_joints`(7d) + `left_gripper`(1d) + `right_arm_joints`(7d) + `right_gripper`(1d) | 16 |

**观测构建差异**：
- 末端控制：从环境 obs 中提取 `ee_pose`（xyz + 四元数），通过 `SignalTransform` 在策略内部处理旋转表示
- 关节控制：直接提取关节角度，不需要旋转转换

**图像预处理**：
```python
# 环境原始图像 → cv2.resize(224×224) → (1,1,3,224,224)
# 注意：与训练时的 Resize(252×336) → CenterCrop(240×320) 不同！
```

这里存在一个**潜在的分布偏移**：训练时图像经过 RandomCrop/CenterCrop 到 240×320，而仿真评估时直接 resize 到 224×224。这意味着评估时的图像预处理与训练时不一致，可能影响策略表现。

### 3.5 评估产物

| 产物 | 格式 | 内容 |
|------|------|------|
| `rollout_N.mp4` | H.264 (libx264, yuv420p) | 仿真环境渲染视频 |
| `rollout_N.png` | PNG | 8 维动作轨迹曲线 (x,y,z,w,x,y,z,G) |
| `info.json` | JSON | 每个 episode 的奖励、终止原因等元数据 |
| 终端输出 | 文本 | `Success rate: XX.XX% (N/100)` |

### 3.6 启动方式

```bash
bash scripts/eval_sim.sh trainer.devices=[0] task=sim/<task> \
    ckpt_path=out/sim/<task>/checkpoints/step_20000.ckpt
```

可通过命令行覆盖评估参数：
```bash
# 评估 200 次（默认 100）
bash scripts/eval_sim.sh ... num_evaluations=200

# 不保存视频（节省空间）
bash scripts/eval_sim.sh ... save_video=False

# 显示仿真画面
bash scripts/eval_sim.sh ... headless=False

# 修改 action chunk 执行步数
bash scripts/eval_sim.sh ... num_action_steps=8
```

---

## 四、开环评估 vs 闭环仿真评估

| 维度 | 开环评估 (`eval_open_loop.py`) | 仿真评估 (`eval_sim.py`) |
|------|-------------------------------|-------------------------|
| **数据来源** | 数据集 val split | Gymnasium 仿真环境 |
| **策略模式** | 离线批量推理 | 在线循环推理 |
| **观测来源** | 数据集预加载 | 环境实时输出 |
| **动作使用** | 全部 chunk 用于对比 | 仅前 16 步用于执行 |
| **Action Chunking** | 不体现（一次性推理） | 核心机制（推理→执行→重新推理） |
| **评估指标** | 轨迹拟合度（视觉对比） | 成功率（百分比） |
| **输出产物** | Plotly HTML（每 episode 每维度） | mp4 + png + info.json |
| **运行速度** | 快（纯 GPU batch 推理） | 慢（含物理仿真逐帧执行） |
| **仿真依赖** | 无 | 需要 GalaxeaManipSim |
| **使用阶段** | 训练后快速验证 | 最终性能确认 |
| **随机种子** | 数据集已固定 | `env.reset(seed=42)` 每次相同 |
| **可复现性** | 完全可复现 | 完全可复现（固定 seed） |

### 4.1 两种评估的互补关系

```
开环高拟合 + 闭环高成功率 = 模型健康 ✓
开环高拟合 + 闭环低成功率 = 数据集有问题 / 仿真-现实差距
开环低拟合 + 闭环高成功率 = 偶然（可能任务太简单）
开环低拟合 + 闭环低成功率 = 训练失败 ✗
```

**开环评估的价值**：在跑仿真之前快速发现训练问题。如果开环都拟合不好，跑仿真纯属浪费时间。

**仿真评估的价值**：唯一能反映策略实际能力的指标。开环拟合再好，如果策略在闭环中因为误差累积而失败，就没有实际价值。

---

## 五、与 DexMani Policy 评估机制的对比

### 5.1 开环评估

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 独立脚本 | **有** (`eval_open_loop.py`) | **无** |
| 可视化 | Plotly HTML（交互式） | 无 |
| 定量指标 | 无（仅视觉） | 训练中 `action_mse_error` |
| Episode 分组 | 按 episode_data_index 分组 | 无 |
| 数据源 | val split DataLoader | 训练 batch 抽样 |

GalaxeaDP 的开环评估是独立、完整的评估管线；DexMani Policy 没有独立的开环评估脚本，仅在训练过程中每 `sample_interval_epochs` 轮抽样计算一次 action MSE。

### 5.2 仿真评估

| 维度 | GalaxeaDP | DexMani Policy |
|------|-----------|----------------|
| 评估框架 | `gymnasium.make()` | `SimRunner` 动态导入 `dexmani_sim.envs` |
| 种子策略 | 固定 `seed=42`（每次相同） | 从文件读取种子列表（多样本） |
| Action Chunking | 取前 16/32 步 | 取 `n_action_steps` 步（跳过 `n_obs_steps-1`） |
| 图像预处理 | `cv2.resize(224×224)` | 数据集原生尺寸 |
| 多 denoise 步测试 | 无（固定 20 步） | `denoise_timesteps_list` 测试多步数 |
| EMA 模型评估 | 无 | `use_ema_for_eval=true` |
| 评估集成 | 独立脚本，训练后手动运行 | 集成到训练循环（周期性调用） |
| 视频编码 | H.264 (libx264) | 未指定 |

### 5.3 关键差异分析

#### （1）种子策略

- **GalaxeaDP**：`env.reset(seed=42)` 每次评估都用同一个种子，100 次评估的初始条件完全相同。这保证了不同 checkpoint 之间的**可对比性**，但无法反映策略在多样化初始条件下的**泛化能力**。
- **DexMani**：从 `DATA_DIR/eval_seeds/<task_name>.txt` 读取种子列表，每个 episode 用不同种子。这更接近真实评估场景。

#### （2）图像预处理分布偏移

- **GalaxeaDP 训练**：`Resize(252×336) → CenterCrop(240×320)`
- **GalaxeaDP 仿真评估**：`cv2.resize(224×224)`
- **DexMani 训练与评估**：使用相同的预处理流程（数据增强仅在训练集启用）

GalaxeaDP 的评估图像预处理与训练不一致，这是一个潜在的 bug。策略在训练时看到的是 240×320 的裁剪图像，在评估时看到的是 224×224 的 resize 图像，长宽比和视野都不同。

#### （3）EMA 评估

- **GalaxeaDP**：不使用 EMA 模型
- **DexMani**：默认使用 EMA 模型评估（`use_ema_for_eval=true`），EMA 通常能提供更平滑、更稳定的预测

#### （4）评估集成方式

- **GalaxeaDP**：评估完全独立于训练，需要手动运行脚本
- **DexMani**：训练中周期性自动触发仿真评估，可实时追踪策略性能变化

---

## 六、GalaxeaDP 评估机制的设计权衡

### 6.1 优势

1. **两级评估管线分工明确**：开环快速验证拟合度，闭环确认真实能力
2. **交互式可视化**：Plotly HTML 支持缩放、悬停，比静态图片更易发现问题
3. **完全可复现**：固定随机种子，相同 checkpoint 多次评估结果一致
4. **评估与训练解耦**：不受 DDP 多卡训练影响，可随时独立运行
5. **丰富的评估产物**：视频 + 动作曲线 + 元数据，多维度理解策略行为

### 6.2 已知问题

1. **图像预处理不一致**（训练 240×320 vs 评估 224×224）：可能导致策略在评估时表现低于预期
2. **固定单一随机种子**（`seed=42`）：无法评估策略在多样化初始条件下的泛化能力
3. **开环评估无定量指标**：只有可视化，没有 MSE/DTW 等自动评分
4. **无 EMA 评估**：不使用 EMA 模型，可能错过更稳定的策略表现
5. **评估未集成到训练**：训练过程中无法自动追踪策略性能变化
6. **验证集未使用**：`validation_step` 为空，训练期间没有任何验证监控
7. **无多种 denoise 步数测试**：固定 20 步推理，无法找到推理步数与质量的最优平衡

### 6.3 改进建议

1. **统一图像预处理**：仿真评估使用与训练相同的 `Resize + CenterCrop` 链
2. **多样化种子策略**：从文件读取种子列表，或设置不同的随机种子
3. **开环定量评分**：添加 DTW 距离或 MSE 指标，自动化评估流程
4. **引入 EMA**：在训练中使用 EMA 模型，评估时同时对比原始和 EMA 策略
5. **训练-评估联动**：训练过程中周期性触发开环评估（轻量级），自动保存最佳 checkpoint
6. **实现 validation_step**：在训练中使用 val split 计算验证损失
7. **多 denoise 步数扫描**：评估时测试 [5, 10, 20, 50] 步推理，找到最优推理步数
