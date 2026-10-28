# AdapTac-Dex 适配复现任务书

本文保留当前实现的范围与验收契约。实现、命令和已完成验证统一见 [AdapTac 使用说明](adaptac.md)，公共接口见 [项目架构](项目架构.md)。后续修改遵循根目录 [AGENTS.md](../AGENTS.md)。

核心来源为 [AdapTac-Dex 官方实现](https://github.com/kingchou007/adaptac-dex/tree/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5)及[论文 v2](https://arxiv.org/abs/2505.13982v2)。

## 范围

只实现完整方法：视觉与触觉编码 → 未来力预测 → 当前与预测力共同引导 FGAF → 动作扩散。保留未来力辅助监督和动作损失向力预测器的梯度连接。

- 复用 Sparse3DEncoder、XHandForceEncoder、StateEncoder；点云和触觉编码器随机初始化、端到端联合训练，不引入独立预训练。
- 复用项目 Diffusion、U-Net、sampler、normalizer、训练、EMA、checkpoint、部署和评测入口。
- 方法专属逻辑归 AdapTacPolicy、AdapTacDataset 和对应模型；公共模块不增加 AdapTac 分支。
- 不提供对比/消融变体、替代融合、算法开关、实验矩阵、多帧扩展或 RTC 新功能。
- 不维护旧产物适配、迁移、回退或双轨配置；不新增哈希、私有协议、注册表、缓存或通用预测框架。
- 只保留真实正确性边界：单帧与动作布局、有限正 force_scale、公共 U-Net 长度约束，以及现有数据和恢复校验。信任内部张量契约。

## 模型契约

默认 `N=1、H=20、K=H-N+1=20、C=8`，绝对关节目标 19 维。输入为相机可见的 XYZRGB 点云、世界坐标五指三维力和 joint_state。

| 分支 | 契约 |
| --- | --- |
| 视觉 | Sparse3D 384 channels、8 heads；均值池化后 Linear 至 512 |
| 触觉 | 归一化五指力 → XHandForceEncoder → 按 thumb/index/middle/ring/pinky 展平五个 64 维 token → Linear 至 512 |
| 力预测 | 1024 维视觉/触觉条件，官方条件因果 Transformer；输出 `[B,K,3]` |
| FGAF | 当前与预测合力展平为默认 63 维引导，投影为一个 query；两个模态 memory token；输出 `[B,512]` |
| 动作 | FGAF 拼接 64 维状态特征，形成 576 维条件；项目 U-Net/Diffusion 输出 `[B,H,19]` |

FGAF 保留完整 TransformerDecoderLayer，包括自注意力、交叉注意力、残差、归一化与 FFN。状态在融合后加入动作条件；力预测不接收状态或候选动作。每次训练前向和推理查询只编码一次观测。

## 数据与归一化契约

`obs[t] → action[t] → obs[t+1]`。动作标签从 t 开始，未来力从 t+1 开始。AdapTacDataset 继承 RobotDataset，只增加顶层 `future_force` 并拟合共享触觉统计，不扩大公共动作窗口。

未来力允许读取动作窗口之后的真实帧，但不跨 episode；超出末帧的部分重复最后真实力，计入监督。没有真实后继的 query 被排除。train/val 使用同一规则，不另存派生 Zarr 或窗口映射。

触觉编码器使用跨训练集真实帧与五指共享的逐轴 limits，映射到 `[-1,1]`，不裁剪。复用 LinearNormalizer，展开为公共 `[5,3]` 字段布局。统计不受窗口筛选、padding、增强和验证集影响，恢复不重拟合。

当前及未来合力均由原始五指力按 FP32 求向量和，保持世界坐标与 N 单位；在 Policy 中统一除以固定 force_scale。预测力已处于相同尺度。不对合力应用触觉 affine，也不把 future_force 拼进动作或公共 normalization。

## 训练与推理契约

力头使用 `sample` 目标和 FP32 MSE，训练单次去噪所得预测直接进入 FGAF，不 detach。动作头使用 `epsilon` 目标，总损失为 `loss_action + force_loss_weight * loss_force`。视觉/触觉同时接受两条损失，状态只接受动作损失。

推理按同一 generator 顺序执行力和动作采样，两头 scheduler 独立。默认 20+20 NFE；动作头采用现有裁剪设置，力头固定不裁剪。返回标准 future_actions，由 Runner 执行前 C 步；不新增推理缓存或特殊部署出口。

训练、恢复、选点和冻结测试使用项目统一流程。默认 100k optimizer updates，沿用统一的数据划分、环境随机化、milestone 和评测预算；具体配方以 adaptac.yaml 为准。恢复验证完整状态，不承诺逐 batch 重放。

## 必要验收

1. 官方 FGAF 和力 Transformer 的有效权重、输出、输入及参数梯度数值一致性。
2. 当前/未来时间偏移、跨窗口读取、episode 边界、尾部重复和 train/val 规则。
3. 训练集共享触觉统计、保存恢复、原始合力缩放不受输入归一化影响。
4. 动作损失向力预测器和两类编码器的非零有限梯度，单次编码与两阶段采样。
5. 真实数据短训练及 raw/EMA、optimizer、scheduler、normalizer、计数和 epoch 恢复。
6. 相关配置、调用点、类型、源码和打包产物检查；可用环境下验证公共推理与仿真接入。

这些属于实现验收，不是算法对比或性能实验。资源不足时完成不依赖该资源的工作，并明确未验证项；完整训练和正式评测结果以真实产物为准。
