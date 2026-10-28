# AdapTac-Dex 完整方法

本实现只有一条方法路径：视觉与触觉编码 → 未来力预测 → 当前与预测力共同引导 FGAF → 动作扩散。复用项目编码器、Diffusion、训练、恢复与评测流程；点云和触觉编码器随机初始化、端到端联合训练。

## 模型与数据

| 环节 | 默认实现与输出 |
| --- | --- |
| 视觉 | 1024 点 XYZRGB → Sparse3DEncoder(384) → token 均值池化 → Linear(384,512) |
| 触觉 | 五指三维力归一化 → XHandForceEncoder → 五个 64 维 token 展平 → Linear(320,512) |
| 未来力 | 视觉/触觉拼成 1024 维条件 → 官方力 Transformer，8 层、4 heads、256 embedding → `[B,20,3]` |
| FGAF | 当前及未来合力拼成 63 维 query 输入，视觉/触觉作为两个 memory token → `[B,512]` |
| 动作 | FGAF 输出拼接 StateEncoder(19,64) → 576 维条件 → 项目 U-Net/Diffusion → `[B,20,19]` |

FGAF 对应[论文 IV-A 的 Force-Guided Attention Fusion](https://arxiv.org/html/2505.13982v2#S4.SS1)，实现为 `agents/obs_encoder/interaction/future_tactile/force_guided_attention_fusion.py` 中的 `ForceGuidedAttentionFusion`，保留官方完整 TransformerDecoderLayer 的自注意力、交叉注意力、残差、归一化与 FFN。力 Transformer 保留条件时间编码、因果 mask 和原初始化；仅保留单帧条件解码路径。状态只加入动作条件，不参与力预测。

同一 `future_tactile/` 目录中的 `diffusion_force_head.py / DiffusionForceHead` 和 `force_prediction_transformer.py / ForcePredictionTransformer` 对应[论文 IV-B 的未来力预测](https://arxiv.org/html/2505.13982v2#S4.SS2)：前者复用公共 Diffusion，提供未来力监督与采样；后者根据视觉/触觉条件执行力去噪。命名依据论文的 diffusion force head 与 Transformer 表述，Policy 负责连接预测和融合。

单帧观测编码器直接返回 `[B,512]` 视觉/触觉特征和 `[B,64]` 状态特征；Policy 仅在送入力 Transformer 时增加一个观测 token 维，FGAF 和动作头使用批量特征。

默认 `N=1、H=20、K=H-N+1=20、C=8`。数据遵循 `obs[t] → action[t] → obs[t+1]`：输入 `obs[t]`，动作标签 `action[t:t+20]`，未来力标签 `force[t+1:t+21]`。公共 sampler 长度仍为 20，`pad_before=0、pad_after=7`。

`AdapTacDataset` 额外返回顶层 `future_force[B,K,3]`。标签为世界坐标下五指原始三维力的 FP32 向量和，单位 N。允许读取当前动作窗口之后的真实帧，但不跨 episode；尾部重复真实末帧并计入 loss，没有真实后继观测的 query 被排除。train/val 使用同一规则，normalizer 统计训练 episodes 的全部真实帧，每帧一次。

## 触觉归一化与合力缩放

触觉编码器输入使用跨五指共享的逐轴 min–max，配置为 `finger_tactile_force: limits`。每个轴在训练集全部真实帧及五指上统计 min/max，映射到 `[-1,1]`；常量轴沿用公共 normalizer 规则。不统计验证集、padding 或增强，不裁剪超出训练范围的输入。相同轴的相同物理力在各手指上具有同一尺度。

共享逐轴统计参考 [KineDex 的 normalizer](https://github.com/DinoMini00/KineDex_code/blob/9246f77d7b969b181b0f1b600601b9a9c8083011/diffusion_policy/model/common/normalizer.py#L182-L223)。当前默认训练划分含 100 episodes、18,248 帧，中指仅 4 帧非零；逐指 Gaussian 会将该指最大约 0.785 N 的分量放大到约 129。共享 limits 的 XYZ 范围为 min `[-37.5025, -29.0172, -15.0326]` N、max `[31.0712, 32.1862, 12.7604]` N，中指归一化后最大绝对值约 0.1167。

Dataset 直接复用 LinearNormalizer，将逐轴 affine 和统计展开为相同的五行，符合公共 `[5,3]` 字段布局；其他字段仍由父类拟合。恢复使用保存的 affine，不重新估计。

FGAF 当前力从归一化前的原始五指力求和，再除以 `force_scale`；未来标签在 Policy 中也只除一次。预测力处于同一缩放空间，不减均值、不除标准差、不裁剪，诊断时乘回 `force_scale`。触觉输入归一化不改变这条合力路径。

官方 [compute_resultant](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/model/tactile/tactile_processer.py#L30-L58) 同样使用固定 scale；其[公开配置](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/configs/train_config.yaml#L68)为 5。本平台保留 `force_scale=1.0`、`force_loss_weight=10.0` 作为工程起点，未验证为长期训练最优值。

## 训练、推理与恢复

训练时，力头对加噪未来力标签单次去噪，使用 `sample` 目标和 FP32 MSE；预测值直接送入 FGAF，保留梯度。动作头使用 `epsilon` 目标，总损失为 `loss_action + force_loss_weight * loss_force`。动作损失可经 FGAF 回传至力预测器；视觉、触觉编码器同时接受两条损失。日志为 `loss`、`loss_action`、`loss_force`、`loss_force_weighted`。

每次查询只编码一次观测，同一 generator 顺序驱动 20 步力 DDIM 和 20 步动作 DDIM。两头 scheduler 独立，无跨 query 缓存。返回 `future_actions[B,20,19]`，Runner 执行前 8 步。`eval.inference_steps` 只覆盖动作头；力 NFE 从保存的 `agent.force_num_inference_steps` 恢复。

默认 100,000 optimizer updates，`lr=obs_lr=1e-4`、AdamW、EMA、BF16，关闭 compile；现有 compile 入口同时覆盖两个 backbone。同步仿真使用项目统一环境随机化、20/40/60/80/100% milestone、每候选 25 回合选点及 100 回合冻结测试。没有新增 RTC、独立预训练或方法变体。

FPS 只通过 `agent.fps_random_config` 配置，输入降采样与 patch 采样共用它。恢复严格使用保存配方，模型结构和权重键须一致。

## 运行命令

在仓库根目录执行，使用 Conda `policy` 环境。Sparse3D 需要匹配 Torch/CUDA 的 PyTorch3D。

```bash
conda activate policy
python dexmani_policy/smoke_test.py --config-only adaptac
python dexmani_policy/smoke_test.py --data-only adaptac

# GPU 短训练验收；产物仍由公共训练入口管理
bash scripts/training/train.sh adaptac task_name=pick_apple_messy \
  dataloader.batch_size=4 dataloader.num_workers=0 \
  workspace.wandb_cfg=null '+max_updates=4'

# 正式训练
bash scripts/training/train.sh adaptac task_name=pick_apple_messy

# 将 RUN 替换为待恢复实验的实际目录名；短跑验收可再加 '+max_updates=2'
RUN=实际目录名
bash scripts/training/train.sh adaptac \
  "+resume_from=experiments/adaptac/pick_apple_messy/${RUN}"

# 完整训练预算结束后选点与冻结测试
bash scripts/eval/eval_pipeline.sh adaptac pick_apple_messy "${RUN}"
```

## 实现验收

当前实现于 2026-10-28 验证：

- 官方 FGAF、力 Transformer：相同 FP32 权重与输入、关闭 dropout，输出及参数梯度最大绝对差均为 0，输入梯度检查通过。只排除上游不参与计算的 dummy 参数与单帧条件下全零的 memory mask。
- 已验证数据时序、episode 边界、跨窗口读取、共享归一化、原始合力/缩放、动作梯度、单次编码、20+20 NFE、generator、FPS 配方恢复及仿真/真机预热点数。
- 单帧数据流简化前后，完整 Policy 在 train/eval 下的损失、指标及参数梯度逐位一致；raw/EMA checkpoint 恢复后的 20+20 步预测逐位一致。
- 真实数据窗口为 train 17,048、val 2,528。RTX 4090、BF16/B=4 完成 4 次更新并恢复至 step 6，loss 与梯度有限；raw/EMA 预测、optimizer、scheduler、EMA 计数、epoch 和 normalizer 恢复通过，恢复不重拟合统计。Runtime 预热保持 generator 状态。
- 此前双头 compile 完成 4 次更新并恢复至 step 6；统一仿真评测器完成 51 个控制步，成功 0/1。此次目录与单帧数据流清理未重跑编译训练或仿真闭环。
- 相关模型、调用方及数据/部署源码通过 mypy。SAT、DQ-RISE、LAMP、LatentFlow 仍有此前与 HEAD 对照确认的 11 条既有签名/类型错误，其动作接口未在本次调整。
- AdapTac、CordViP、SAT、DQ-RISE、LAMP、LatentFlow 通过配置 smoke；项目 Python 源码通过语法解析，相关训练/评测脚本通过 `bash -n`，`git diff --check` 通过。
- wheel 构建及独立导入通过，包含当前 `future_tactile`、`correspondence` 模块和方法配置，无已删除目录或旧模块路径。
- 完整 100k 训练、正式选点及 100 回合冻结测试未执行，不据此报告任务性能。此次短训练验收产物为 `/tmp/adaptac-module-cleanup-20261028/{train,resume}`，不作为正式实验入口。

## 来源与边界

核心模块移植自 [kingchou007/adaptac-dex@60e2ef1](https://github.com/kingchou007/adaptac-dex/tree/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5)。

编码器、共享触觉统计、世界坐标五指合力、状态输入、19 维动作及控制时域是本平台适配。源码数值一致性只针对移植核心模块，不代表整个 Policy 或任务成功率等同论文。实现边界见 [任务书](AdapTac-Dex复现任务书.md)，公共接口见 [项目架构](项目架构.md)。
