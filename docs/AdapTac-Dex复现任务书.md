# AdapTac-Dex 适配复现：Codex 实施任务书

本文是待执行的代码实施任务书，不代表 AdapTac 已实现或已通过训练、闭环评测。执行者应直接完成代码、配置、必要文档和针对性验证，不停留在再次讨论方案。

- 核对日期：2026-10-28。
- 范围更新：仅实现官方最终完整方法，不做对比或消融实验；保留已确认的现有编码器及项目数据/平台适配。
- 目标仓库：haoyangzhanglab/dexmani_policy。
- 当前代码基线：[1e1a5702a8e5b5b07e39a8b71d59a4bb27d0ca80](https://github.com/haoyangzhanglab/dexmani_policy/tree/1e1a5702a8e5b5b07e39a8b71d59a4bb27d0ca80)。
- 官方实现基线：[kingchou007/adaptac-dex@60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5](https://github.com/kingchou007/adaptac-dex/tree/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5)。
- 论文：[Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation，arXiv v2](https://arxiv.org/abs/2505.13982v2)。
- 开始前先读根目录 [AGENTS.md](../AGENTS.md)。若执行时当前分支已有后续提交，只复核与本任务有关的接口差异，以当前实现为准，不回退代码，不机械套用旧接口。

## 1. 目标与完成范围

在 dexmani_policy 中实现使用现有编码器的 AdapTac-Dex 方法适配复现：

1. 复用现有 Sparse3DEncoder、XHandForceEncoder 和 StateEncoder。
2. 触觉编码器随机初始化，与主策略端到端联合训练。点云编码器也不新增外部预训练权重依赖。
3. 移植官方未来力 Transformer 和 ForceGuidedCrossAttention，保留核心计算。
4. 复用项目动作 U-Net、Diffusion、数据读取、归一化、训练、EMA、checkpoint、评测与推理接口。
5. 只提供官方最终完整方法这一条训练与推理路径，能够通过统一入口训练、恢复和评测；不实现对比基线或消融变体。
6. 交付可运行配置、简洁说明与实际验证结果；缺少数据、GPU、原生扩展或仿真环境的项目如实标为未验证。

本任务不包含独立触觉预训练、MAEGAT 权重移植、3DTacDex 预训练任务、taxel 编码器、多帧观测扩展、额外融合方法、RTC 新功能或真机控制。不要为这些能力预留框架。

本复现保留 AdapTac 的未来力辅助监督、当前与预测力引导融合、动作扩散及联合梯度连接。编码器、传感器表示、状态输入、动作空间、坐标系和控制时域属于明确记录的项目适配。不要宣称逐项复刻原论文实验或保证达到其绝对成功率。

## 2. 最新仓库约束与旧方案修正

本基线相对旧方案使用的 a056948 已有三次后续提交，执行时必须吸收以下变化。

| 项目 | 当前约定 | 本任务处理 |
| --- | --- | --- |
| 主策略公共配置 | configs/common.yaml 已包含数据划分、评测预算和环境随机化 | adaptac.yaml 通过 defaults 继承，不重复维护这些字段 |
| 训练数据 | dataset.seed=42，val_ratio=0.1，max_train_episodes=100 | 最多100个训练 episodes，不是总数据只允许100个 episodes |
| 选点与测试 | eval.seed=1024，每候选25回合选点，100回合冻结测试 | 删除旧方案的70回合测试值 |
| checkpoint | common/experiments.py 的 milestone_steps 给出20/40/60/80/100%五个节点 | 沿用主策略训练和选点流程，不新增保存调度 |
| 环境随机化 | common.yaml 开启 randomize_model_id、randomize_table_height | 直接继承项目统一设置 |
| sampler | 只做窗口、episode mask、边界复制和字段读取 | 不添加 filter_valid，不扩大公共动作窗口 |
| 归一化 | 训练 episodes 全部真实帧，每帧一次 | 不按辅助标签筛选后的窗口重算统计 |
| 预训练基础设施 | 冗余预训练加载层、几何校验和大型验证脚本已被清理 | 不重新引入类似结构，不借用 AFRO/CordViP 的预训练路径 |
| 恢复 | epoch 级恢复，不保存 batch 游标 | 验证状态恢复，不承诺逐 batch、逐位重放 |

显式 split_manifest 仍由现有 split.py 处理；超过训练 episode 上限也由现有逻辑处理。不得要求数据额外提供 source_id/episode_ids/trial_ids，不重复检查这些元数据，不创建 AdapTac 专用划分协议。

复用项目统一的数据预算、候选节点、选点/测试预算、环境与成功标准，保留AdapTac自己的H/N/C、两头NFE和EMA设置。本任务只训练和评测完整方法，不安排跨策略比较或消融实验；正常checkpoint选点仍沿用项目流程。

## 3. 编码与架构要求

遵循 AGENTS.md：正确、直观、易读易改；完成最小但完整的改动。

- 尽量复用当前项目和官方实现，不重写已有 sampler、normalizer、DDIM 循环、优化器、EMA、checkpoint、runner。
- 方法专属行为放在 AdapTac Policy、Dataset 和对应模型文件。公共模块不得出现 policy_name == "adaptac" 分支或未来力专属字段处理。
- 禁止防御性编程、过度校验和过度工程化：不重复检查明确的内部 shape/dtype/有限值契约，不层层包装异常，不静默回退、补零掩盖坏数据或自动替换算法。
- 只在新功能边界保留直接防止实际错误的检查，例如单帧支持范围、有限正 force_scale，以及现有 U-Net 长度约束。复用现成检查，不创建额外校验层。
- 不新增注册表、工厂、通用预测器框架、配置镜像、运行时能力探测、内容哈希、数据指纹、完整性清单、兼容迁移或缓存生命周期。
- 不为了单个方法改造公共 API，不复制一套训练/评测入口。必要的小接口适配应局部、显式。
- 配置键、参数分组、文件命名、注释与邻近实现一致。只解释时间偏移、梯度连接等不明显原因，不堆叠 docstring 和装饰性注释。
- 尊重用户已有改动，不顺带重构其他策略、不删除现有公共校验，也不恢复已删除的旧机制。
- 验证以真实风险为边界；不要为新增抽象反向制造测试需求。

## 4. 必须先看的代码与复用边界

先建立下列短调用链，再写代码；不要通读无关目录。

| 当前项目文件 | 重点 |
| --- | --- |
| agents/policies/base.py | preprocess、action_start_index、future_steps、predict_action_from_condition、format_action_output、parameter_groups |
| agents/policies/dp3.py | 普通 Diffusion Policy 的构造和代码风格 |
| agents/policies/dqrise.py | Sparse3D 观测预处理、token 池化、effective_config；不要复制其码本/量化动作机制 |
| agents/obs_encoder/pointcloud/sparse3d_encoder.py、ops.py | 米制坐标、patch 采样、RoPE 维度、预处理 |
| agents/obs_encoder/xhand_tactile.py、proprio.py | 现有五指力与状态编码 |
| agents/action_decoders/diffusion.py | scheduler、训练目标、generator、DDIM 采样及设备处理 |
| agents/action_decoders/backbones/unet1d.py、modules.py | 动作 U-Net、时间编码、长度约束 |
| datasets/common/robot_dataset.py、sampler.py、replay_buffer.py | build_sampler、sample_source_rows、只读低维读取、train/val 构造 |
| datasets/aux/r3d_dataset.py | 局部辅助 Dataset 的写法；不要把未来力拼进动作维度 |
| training/common/data.py、config.py、optimization.py | 组件构造、配置保存、参数覆盖检查 |
| training/trainer.py、common/checkpoint.py | EMA、编译、恢复、raw/EMA 权重保存 |
| configs/common.yaml、dp3.yaml、dqrise.yaml | 共享实验设置与方法局部配方 |
| smoke_test.py | 现有配置、数据、短训练与恢复检查的能力边界 |
| docs/仿真数据集格式.md、docs/仿真评测机制.md | 时间、单位、坐标系、执行窗口、选点与冻结测试 |

官方仓库只需重点读取：

- src/adaptac/model/diffusion_modules/transformer_for_diffusion.py
- src/adaptac/model/transformer_diffusion.py
- src/adaptac/model/fuse_network.py
- src/adaptac/model/unet_diffusion.py
- src/adaptac/policy/FFG_policy.py
- src/adaptac/dataset/realdex.py
- src/adaptac/configs/train_config.yaml
- LICENSE

官方默认配置并未开启完整方法；其模型默认值只能作为源码对齐的工程起点，不等于已经确认的全部论文实验超参数。

## 5. 模型结构与张量契约

默认设置：

- N=n_obs_steps=1。
- H=horizon=20。
- K=H-N+1=20，覆盖查询起点之后全部可预测动作的执行后力。
- C=n_action_steps=8。
- action_key=action，action_dim=19，绝对关节目标。
- 观测字段恰为 joint_state、point_cloud、finger_tactile_force。
- 点云取相机可见 point_cloud，默认1024点、XYZRGB；不引入完整物体几何等额外输入。

### 5.1 编码与条件

| 环节 | 处理 | 形状 |
| --- | --- | --- |
| 原始点云 | 世界坐标XYZ，单位m；RGB保持现有值域 | [B,N,1024,6] |
| 视觉特征 | Sparse3D tokens 均值池化，Linear(384,512) | [B,N,512] |
| 原始触觉 | 世界坐标下五指三维力，单位N | [B,N,5,3] |
| 触觉特征 | 已归一化五指力 → XHandForceEncoder → 五个64维token按固定顺序展平 → Linear(320,512) | [B,N,512] |
| 状态特征 | 已归一化 joint_state → StateEncoder(19,64) | [B,N,64] |
| 力预测条件 | 视觉与触觉沿特征维拼接 | [B,N,1024] |
| 未来力 | 官方力Transformer预测缩放后的三维合力 | [B,K,3] |
| 完整方法引导力 | 当前观测合力与预测未来力按时间顺序拼接、展平 | [B,3*(N+K)]，默认[B,63] |
| FGAF输出 | 视觉、触觉两个memory token | [B,N,512] |
| 动作条件 | FGAF输出与状态拼接后展平 | [B,N*576]，默认[B,576] |
| 动作窗口 | 项目动作U-Net + Diffusion | [B,H,19] |

五指顺序沿用 thumb/index/middle/ring/pinky。保留每个手指的固定位置；首版不把五指分别送入 FGAF 的 memory。

Sparse3D 保持 channels=384、num_heads=8。该实现要求每个 head 的维度可被6整除，不能为了凑512维直接改为 channels=512、heads=8。投影使用简单 Linear，不额外叠加与任务无关的编码层。

状态在 FGAF 后加入动作条件，不输入力预测器或 FGAF memory。力预测仅条件于视觉、触觉观测，不加入候选动作。

### 5.2 模块所有权

完整方法的 AdapTacPolicy 独立持有：

- obs_encoder：现有编码器、视觉/触觉投影和状态编码。
- force_predictor：力预测 backbone 与薄扩散封装。
- force_fusion：官方 FGAF。
- action_decoder：现有动作扩散头。

观测编码器只做特征提取，可局部返回视觉、触觉、状态组成的简单 tuple/dict，由 AdapTacPolicy 消费；不改 BasePolicy 的公共返回约定，不在 obs_encoder.forward 中做力采样。

每次训练前向或策略查询只编码一次观测。FGAF 不放进 obs_encoder，避免参数所有权和学习率/冻结语义混淆。

### 5.3 保留官方 FGAF 的真实计算

直接移植 ForceGuidedCrossAttention：

1. 视觉、触觉各经模块内部投影，stack 为两个 memory token。
2. 完整引导力向量投影为一个 query。
3. 使用官方完整 TransformerDecoderLayer。
4. 返回 attended_feat，必要 reshape 由 Policy 负责。

默认 hidden_dim=512、num_heads=8、dim_feedforward=512、dropout=0.1。保留官方自注意力、交叉注意力、query残差、归一化、FFN和激活设置。

默认63维引导力是一个query的输入，不是21个query token。不能用两模态加权平均或一个自写 MultiheadAttention 替换完整模块，也不能把 attention 权重解释成全部作用路径。

### 5.4 官方整套Policy不能原样复制

以下连接问题已在官方基线确认，应在新Policy的局部接入中修正，不改变核心算法：

| 官方连接点 | 本地处理 |
| --- | --- |
| 力扩散包装被传入num_obs/obs_dim，实际签名要求n_obs_steps/obs_feature_dim | 按实际签名构造，或直接用移植backbone和项目Diffusion薄封装 |
| FGAF前向调用缺少num_obs | 显式传入，默认1 |
| 开启触觉后动作条件按1024维构造，但FGAF输出512维 | 以实际融合512维加状态64维推导576维动作条件 |
| FGAF创建与num_force_prediction>0耦合 | 完整方法始终构造力预测器与FGAF，不移植开关分支 |
| 默认YAML关闭触觉与未来力预测 | 按最终完整方法明确启用两种输入、未来力监督与预测力引导 |

## 6. 数据集：只扩展未来力标签

### 6.1 时间语义

项目契约是：

    obs[t] → action[t] → obs[t+1]

action[t] 是从当前观测出发执行的绝对关节目标。每个 episode 的观测与动作数量相同，最后一次动作后的观测不单独保存。

默认窗口：

| 数据 | 索引 |
| --- | --- |
| 输入观测 | obs[t] |
| 动作标签 | action[t:t+20] |
| 未来力标签 | force[t+1:t+21] |
| 公共 sampler 长度 | 20 |
| 最多涉及的真实时间位置 | 21帧 |

保持 BasePolicy 的 action_start_index=N-1、future_steps=H-N+1。不得照搬官方数据集的动作 t+1 起点，不切换 future_action_window，不改变19维动作布局。

### 6.2 局部实现

新增 AdapTacDataset，继承 RobotDataset：

1. 复用父类构造、字段检查、split、归一化、增强及基本采样。
2. build_sampler 调用父类后，只筛选没有真实后继观测的query。
3. 当前样本的query行由 sampler.sample_source_rows(idx)[N-1] 确定。
4. 使用 replay_buffer.episode_ends 找到query所属episode的真实结束位置e，区间右端不包含e。
5. 保留 t+1<e 的query。未来标签的第j步读取 min(t+j,e-1)，j=1..K。
6. 只额外读取 finger_tactile_force 的真实后继区间，按FP32沿手指维求和，然后重复末帧补齐。
7. 返回顶层 future_force，单样本[K,3]、batch[B,K,3]、float32、物理单位N。
8. 复用父类 __getitem__ 的普通观测/动作流程后添加标签即可，避免复制整个读取与增强流程。

边界取真实episode结束位置，不能取当前窗口的buffer_end。不能跨episode补标签，也不能把不存在的后继观测当零力。

K由H和N派生，不新增独立force_horizon配置。不预生成派生Zarr、不缓存整库合力、不增加query映射持久化文件。父类构造期间会调用build_sampler，覆盖方法不能依赖super().__init__返回后才赋值的子类状态。train/val都通过同一个覆盖方法取得相同规则。

批量筛选可以只从sampler.indices计算query行，避免不必要的全库[窗口数,H]矩阵；结果须与sample_source_rows语义一致。不为此修改公共sampler或新增索引框架。

默认pad_before=N-1=0，pad_after=C-1=7。保留官方末帧重复思路，补齐力标签正常计入loss；不新增padding损失mask。本任务不实现丢弃全部尾部窗口或屏蔽补齐loss的替代路径。

future_force不是观测模态、不是action_context，也不拼进action。action_dim和control_action_dim继续为19。

### 6.3 统计总体

公共normalizer继续统计train episodes全部真实帧，每帧一次。未来力标签读取、窗口筛选和尾部重复都不能改变动作或观测统计总体。

验证集复用训练normalizer。数据既有有限值检查已覆盖消费字段，不再为未来标签扫描一遍全库，不新增重复校验。

## 7. 合力、缩放与归一化

物理合力：

    F[t] = sum_i finger_tactile_force[t,i]
    F_scaled[t] = F[t] / force_scale

当前数据的五指力都在世界坐标系，已按物理子步平均。原论文使用的坐标系和触觉布局不同，这里明确记录为平台适配。

| 路径 | 处理 |
| --- | --- |
| 触觉特征 | 原始五指力 → 现有逐指、逐分量归一化 → XHandForceEncoder |
| 当前力query | preprocess之前的原始五指力 → FP32求和 → 除以force_scale |
| 未来监督 | Dataset返回原始合力N → Policy除以同一个force_scale |
| 预测力query | 力头输出直接处于缩放空间，不再缩放 |
| 物理量诊断 | 缩放预测乘回force_scale |

约束：

- 不先逐指归一化再求和；不把三维向量和改成力模长之和。
- 对向接触可能相互抵消；不得因而擅自改变net force定义。
- Dataset不除以force_scale，Policy只除一次。
- future_force不进入公共normalization。normalization字段仍严格对应obs字段加action。
- force_scale与clip_sample独立；不得复制官方“尺度达到阈值就启用裁剪”的逻辑。
- force_scale是保存于agent配置的固定正标量；验证、恢复、推理不重新估计。

默认归一化：

| 字段 | 模式 |
| --- | --- |
| point_cloud | identity |
| joint_state | limits |
| finger_tactile_force | gaussian |
| action | auto |

默认force_scale=1.0作为明确、可运行的起跑值，单位N；这不是数据校准结论。有训练数据时，一次性查看训练合力的数量级/分位数，决定正式实验尺度并写入配置；没有数据时保留该值并说明未校准，不阻塞实现。不要新增auto-scale协议、独立normalizer、校准缓存或外部权重依赖。

首版不启用触觉噪声和触觉模态dropout。点云坐标噪声、颜色增强和状态噪声复用项目现有配置。不引入未同步处理力/状态/动作坐标的几何旋转增强。

## 8. 训练算法与梯度

设Z为视觉、触觉拼接特征，F为已缩放未来力标签；扩散时间u与物理时间t区分。

1. 从原始obs计算当前合力，然后调用现有preprocess并编码一次。
2. 为F采样独立高斯噪声和随机扩散时间u，使用现有scheduler.add_noise。
3. 力Transformer单次前向预测干净力F_hat，prediction_type固定为sample。
4. L_force = mean((F_hat-F)^2)，覆盖B、K、3；FP32归约。
5. 将当前合力和F_hat拼接展平，送入FGAF。
6. 融合特征与状态形成动作条件。
7. 现有action normalizer处理action；动作Diffusion使用epsilon目标计算L_action。
8. L_total = L_action + force_loss_weight * L_force。

F_hat进入FGAF前不得detach。动作loss必须能经FGAF回传到力预测器；视觉与触觉编码器接受动作及辅助力监督，状态编码器仅参与动作路径。

保留官方训练/推理差异：训练中的F_hat来自加噪真实标签的单次去噪；推理中的未来力来自纯噪声完整DDIM采样。不要改成训练时完整采样、直接使用真值未来力融合、停止梯度、交替训练或额外预训练。

沿用动作头当前loss归约，不自行新增手臂/手部分组权重。日志保留loss、loss_action、loss_force及加权力loss，遵守log_metrics开关；日志tensor可以detach，参与融合的F_hat不能detach。

力预测器、未来力辅助监督和当前/预测力共同引导始终存在。force_loss_weight仅是损失系数，不兼作分支开关；不新增关闭辅助监督、跳过预测力引导或替换融合方式的控制逻辑。

## 9. 力扩散封装与推理

### 9.1 最小复用

力扩散薄封装优先继承现有Diffusion：

- 复用DDIMScheduler构造、设备处理、噪声采样和DDIM循环。
- 新增局部的loss与带梯度预测值返回方法；不改变公共Diffusion.compute_loss接口。
- 力头固定sample目标，不为epsilon/v_prediction增加未使用的力预测分支。
- 可提供predict_force薄入口，内部复用现有predict_action采样实现，传入[B,K,3]形状。
- 两头各有独立scheduler实例，不能共享可变scheduler对象。

公共Diffusion调用backbone的关键字是x/timestep/context，官方Transformer前向签名是sample/timestep/cond。用局部薄适配或等价的前向参数重命名接通，不改attention、mask、时间编码或初始化，不引入通用adapter系统。

只移植必要官方代码。可复用项目已数值一致的sin/cos时间编码；移植时同步正确的初始化类型处理。官方独立优化器方法、示例test入口和无计算用途的ModuleAttrMixin/dummy参数不必照搬。移除这些代码不能改变实际前向与梯度。

### 9.2 完整方法推理

predict_action按顺序执行：

1. 原始当前合力求和与缩放。
2. 编码一次观测。
3. 用力头从纯噪声采样K步未来力，默认20步DDIM。
4. 当前力与预测力组成query，执行FGAF。
5. 拼接状态。
6. 复用predict_action_from_condition调用动作头，默认20步DDIM。
7. 复用动作反归一化与format_action_output，返回future_actions。

Runtime传入的同一个generator依次传给力和动作采样；不在分支内部重设seed或另建隐藏generator。直接调用时允许沿用现有generator=None行为。不跨query缓存特征、预测力或随机样本。

输出future_actions[B,H-N+1,19]，Runner负责执行前C步。推理只需要当前观测，不读取future_force、训练数据或任何预训练checkpoint。

首版按同步仿真路径验收，保持现有参数签名和执行模式语义，不顺便宣称或新增RTC支持。

eval.inference_steps继续只控制动作头；force_num_inference_steps保存在agent配置中，首版固定20。不要修改公共评测覆盖白名单或增加force_steps专属分支。本任务不新增力NFE扫描入口，不篡改保存配置或checkpoint。

## 10. 参数管理、编译与恢复

AdapTacPolicy主要覆盖compute_loss、predict_action、optimizer_groups，以及需要时的compile_backbone。

- obs_encoder包含全部编码器与视觉/触觉投影，使用obs_lr。
- force_predictor、force_fusion和action_decoder使用主lr。
- 使用现有parameter_groups和build_optimizer的“每个可训练参数恰好一次”检查；不复制官方优化器或添加第二套检查。
- 默认obs_lr=lr>0，不冻结触觉，不使用预训练触觉的小学习率配方。
- 复用当前EMA、state_dict和checkpoint流程，新增模块作为Policy子模块自然保存；不单独导出力头权重。
- force_scale、力NFE、网络结构由保存配置恢复；不增加额外metadata文件。
- 若开启compile，新Policy同时覆盖动作和力backbone；不编译整个采样循环，不增加缓存/探测/回退机制。
- 默认配置先关闭compile，以便使用现有smoke做eager训练与恢复验收；通过后允许用户开启已有training.use_compile。
- 当前smoke恢复比较只显式解包action_decoder.model。不要为了AdapTac重构公共smoke；编译后的双头保存/恢复可在局部验证中将两个backbone恢复到eager后比较。
- epoch级恢复不要求新增batch游标、worker状态快照或逐位重放保证。

## 11. 官方完整方法的单份配置

### 11.1 单份主配置

新增configs/adaptac.yaml，使用现有defaults组合common和_self_。只写AdapTac局部配方，不重复dataset.seed/val_ratio/max_train_episodes/split_manifest、eval.seed/selection_episodes/episodes和已共享环境随机化值。

| 设置 | 值或来源 |
| --- | --- |
| policy_name | adaptac |
| task_name / zarr_path | 沿用项目任务命名与robot_data路径习惯 |
| horizon / n_obs_steps / n_action_steps | 20 / 1 / 8 |
| action_key / action_dim | action / 19 |
| dataset._target_ | dexmani_policy.datasets.aux.adaptac_dataset.AdapTacDataset |
| agent._target_ | dexmani_policy.agents.policies.adaptac.AdapTacPolicy |
| sensor_modalities | [joint_state, point_cloud, finger_tactile_force] |
| dataset.obs_horizon | 引用n_obs_steps |
| dataset.pc_dim | 引用agent.pc_dim |
| pad_before / pad_after | N-1 / C-1 |
| pc_dim / num_points | 6 / 1024 |
| Sparse3D | channels384、patches96、group_size32、depth4、heads8，其余沿现有Sparse3D配方 |
| 视觉/触觉特征宽度 / state_out_dim | 512 / 64 |
| 动作U-Net | time embedding256、down_dims=[256,512]、kernel5、groups8、cond_predict_scale=true |
| 动作Diffusion | 100训练步、epsilon目标、20步DDIM、clip_sample=true，与action:auto配套 |
| 力Transformer | n_layer8、n_head4、n_emb256、p_drop_emb0、p_drop_attn0.3 |
| 力条件与mask | causal_attn=true、time_as_cond=true、n_cond_layers=0，条件宽度由特征推导 |
| 力Diffusion | 100训练步、sample目标、20步DDIM、force_clip_sample=false |
| FGAF | hidden512、heads8、FFN512、dropout0.1 |
| force_scale | 1.0，起跑值；正式实验按训练数据确定后固定 |
| force_loss_weight | 10.0，官方配置起点 |
| force_num_inference_steps | 20，独立于动作推理步数 |
| eval.inference_steps | 引用agent.num_inference_steps |
| eval.action_steps | null，使用保存的n_action_steps |

K、query宽度、condition宽度从已有维度推导，不要求用户重复填写。两头训练步数可以复用同一个数值配置，但scheduler实例独立；力目标固定sample。只暴露实际需要的模型参数，避免嵌套多层工厂配置或字段别名兼容。

参考训练配方明确采用当前DP3的AdamW/EMA设置，不把不同策略默认值混成“公共训练默认值”：

| 设置 | 起点 |
| --- | --- |
| batch_size | 128；显存不足通过现有gradient_accumulation_steps调整 |
| lr / obs_lr | 1e-4 / 1e-4 |
| weight_decay / obs_weight_decay | 1e-6 / 1e-6 |
| betas | [0.95,0.999] |
| total_train_steps | 100000 optimizer updates |
| lr_scheduler / warmup | cosine / 500 |
| max_grad_norm | 1.0，复用现有裁剪 |
| EMA | 启用，采用DP3的EMA配置，eval.use_ema一致 |
| use_bfloat16 | true；FP32数值一致性检查时显式关闭 |
| use_compile | false起步；独立验证后可开启 |
| loader | 沿用项目配置；小数据冒烟按实际batch与worker条件设置 |

force_scale与force_loss_weight共同决定辅助项量级。沿用任务书给定的完整方法配方，只进行必要的训练数据尺度校准与有限loss/有效梯度检查，不安排损失权重搜索或参数对比，不能在正式test上选择它们。默认值是可执行工程起点，不是经过验证的最优超参数。

### 11.2 只实现官方最终完整路径

固定启用视觉与触觉编码、未来力辅助监督、当前与预测力共同引导的FGAF，以及动作扩散：

- 训练：一次观测编码 → 未来力单次去噪预测与辅助loss → 当前/预测力FGAF → 动作loss → 联合优化。
- 推理：一次观测编码 → 未来力DDIM采样 → 当前/预测力FGAF → 动作DDIM采样。
- FGAF的force_dim始终为3*(N+K)，默认63；融合输出512维，拼接状态后的动作条件576维。
- 只维护一份adaptac.yaml和一套Policy/Dataset路径，不添加variant参数、算法分支开关或消融配置。
- 不实现直接拼接基线、仅当前力引导或仅辅助监督的替代路径，不预留这些分支的类、投影层和命令。
- 源码数值一致性、数据时序、梯度和保存恢复检查属于实现验收，不是性能对比实验，仍需按第13节完成。

“按官方最终版本”指上述完整方法机制。继续保留既定的现有编码器、19维动作、项目数据时序和训练/评测基础设施适配；不照抄官方未启用完整方法的默认YAML，也不引入官方预训练编码器依赖。

## 12. 文件组织与共享代码边界

预期六个主要实现文件，路径均位于dexmani_policy下：

| 路径 | 职责 |
| --- | --- |
| agents/policies/adaptac.py | 观测编码组合、Policy、联合loss、推理、参数分组 |
| agents/obs_encoder/interaction/force_guided_attention.py | 官方FGAF；复用已有interaction目录，不另建平行fusion层级 |
| agents/force_predictors/transformer_for_diffusion.py | 官方力backbone和必要依赖 |
| agents/force_predictors/diffusion.py | 力训练与采样薄封装、必要前向参数适配 |
| datasets/aux/adaptac_dataset.py | 未来力标签与query筛选 |
| configs/adaptac.yaml | 主策略完整配置 |

必要时添加最小__init__.py。文件数不是考核目标；可以按实际代码体量做更简单的局部组织，但不能将所有逻辑塞进共享BasePolicy或创建多层框架。

实现后添加简洁的docs/adaptac.md，说明完整方法、平台适配、输入/输出、尺度及单一路径的训练/评测/恢复命令；同步README或项目架构中与新增策略直接相关的入口说明。当前任务书保留为实施依据，不把尚未通过的验收写成完成状态。

首版预期无需修改公共sampler、RobotDataset、normalizer、Diffusion、Trainer、Runner、checkpoint或评测器。若实际接口确实阻塞，只允许最小、通用且有调用方证据的修复；在交付中解释原因，不能添加AdapTac专属公共分支。

不复制官方整套环境，不新增MinkowskiEngine/MAEGAT等不再需要的依赖。复用项目PyTorch、diffusers和现有点云后端；缺少真实依赖时明确报告，不用替代算法掩盖。

移植文件注明官方来源commit及修改范围，保留必要版权/许可声明。官方README的MIT标注与LICENSE开头的CC BY-NC-SA 4.0文本存在冲突，不直接标为MIT。按实际来源保留必要声明/许可文本即可，不新增运行时许可检查或许可管理框架。

## 13. 验证：少量、可解释、针对真实风险

先复用已有smoke，再补其没有覆盖的AdapTac机制。当前data-only检查不会验证顶层future_force，不能把它单独当作未来力时序已通过的证据。

验证可使用短小临时脚本；只有维护价值明确时才保留一个聚焦AdapTac的检查文件。不要重建被删除的scripts/validation平台、fixture生成器、生命周期测试矩阵或强制下载上游的CI链路。

### 13.1 官方模块等价性

- 在FP32、关闭dropout、相同权重和输入下，比较官方与移植版FGAF和力Transformer的输出、输入梯度及参与计算的参数梯度。
- 使用相同torch环境，按数值精度选合理容差，不能为通过而放宽到掩盖结构差异。
- 允许局部签名/导入适配和删除无计算作用的dummy参数；这些差异要明确，不能宽松加载来掩盖其他权重缺失。
- 不把整个编码器替换后的Policy宣称为官方数值等价。

### 13.2 数据时序与物理量

用带可辨认时间值和不同episode哨兵值的小数据检查：

- 标签从query后一帧开始，动作起点保持query。
- 未来标签可越过当前sample的buffer_end继续读，但不越过真实episode边界。
- 尾部重复末帧，无任何真实后继的query被排除。
- train/val使用一致的query筛选、时间偏移和边界补齐逻辑。
- 原始五指向量先求和、再缩放，未来标签只缩放一次。
- 归一化仍统计训练episode的真实帧，额外标签读取不改变动作统计。
- 普通batch仍为obs/action，辅助标签只在顶层future_force。

### 13.3 联合训练、推理与恢复

- 完整方法能构造、计算联合loss和预测；力预测器与FGAF始终参与对应路径。
- 在局部梯度检查中单独对L_action反向传播，确认梯度能到达force_predictor；这是计算图正确性检查，不改变训练配方或构造消融模型。
- 使用现有优化器覆盖检查确认无漏参、无重复；短训练后相关模块实际更新。
- 移除future_force也能调用predict_action；输出future_actions形状和物理动作契约正确。
- 在相同设备/数值设置与generator状态下，检查重复采样及raw/EMA保存恢复一致。
- 复用现有短训练、保存、恢复流程；不要求新增逐batch重放机制。
- 若开启compile，额外检查两头编译后的保存与eager恢复，不在公共smoke中加入方法分支。

### 13.4 闭环与效率

资源具备时运行完整方法的短闭环检查；正式训练和评测使用第14节统一协议。记录实际完成的训练/评测，不把配置检查、合成数据或单步去噪MSE写成闭环成功率。

如果测未来力误差，使用部署同款纯噪声DDIM采样并还原N单位；训练时单次去噪MSE不能充当真实预测精度。

## 14. 完整方法的正式评测与时间预算

继承common.yaml：最多100个训练episodes，每候选25回合selection，100回合冻结test，20/40/60/80/100%五个milestone；默认seed池至少125个不同seed。模块预训练不在此任务范围。

- 只运行完整方法，按保存的train/val划分和训练预算训练；本任务不组织多seed、超参数搜索或方法对比实验矩阵。
- 固定成功定义、回合时限、环境随机化、selection/test seeds、执行长度、EMA和动作NFE。
- 正式test冻结后不用于调checkpoint、force_scale、loss权重、C或NFE。
- 报告完整方法的成功数/总数、成功率、现有成功步数统计和完整query延迟。
- Wilson区间按现有episode成功率统计，不据此声称跨训练seed稳定性。
- 若补充论文AEL，失败episode按最大长度计入，单独计算；不把avg_steps_all改名为AEL。

完整方法每次query为20次力去噪加20次动作去噪，两者有先后依赖，不能并行，不是单头20NFE的相同计算预算。

计时覆盖预处理与传输、一次编码、力采样、融合、动作采样、反归一化与结果回传。复用Runtime.warmup做基础检查，正式延迟用真实观测与足够重复次数测量；不只报告动作头时间。

当前正式评测不启用autocast/compile。训练compile加速不能直接表述为部署加速。记录实际精度、硬件及两头NFE，不组织加速方案对比或静默改变参考配置。

按当前数据dt约0.0625秒，K=20最远预测时刻为当前之后1.25秒，C=8执行时长约0.5秒；同步重规划还受query延迟影响。记录数据实际dt，不把16Hz控制频率写成16Hz策略查询频率，不为匹配论文5Hz直接改变控制系统。

效率优化限于消除重复编码、多余高维读取并复用现有训练能力；固定参考采样步数，不增加NFE扫描或执行长度对比。不得通过单纯增大C掩盖推理成本而忽略触觉反馈更新频率。

## 15. 实施顺序与可用命令

按依赖顺序完成：

1. 阅读当前接口，落实Dataset与物理力契约，完成索引检查。
2. 移植两个官方核心模块，完成局部数值一致性检查。
3. 接入完整Policy、参数分组和单份主配置，完成eager短训练与恢复。
4. 检查完整方法的未来力监督、预测力引导、动作梯度与两阶段采样连接，清除未使用的算法开关。
5. 补充运行文档、必要验证结果和真实资源下的闭环/延迟结果，清理冗余代码。

实现后沿用现有入口。下列命令中的task/run占位符须替换为实际值；实际环境使用项目policy环境。

配置与默认数据检查：

    python dexmani_policy/smoke_test.py --config-only adaptac
    python dexmani_policy/smoke_test.py --data-only adaptac
    python dexmani_policy/smoke_test.py --max-updates 4 adaptac

smoke的config_names是配置名或显式测试YAML路径，不是任意Hydra overrides。需要自定义任务或batch时，使用临时配置或已有load_policy_config/validate_training函数，不为此扩展smoke CLI。测试配置保持输出目录、BF16/compile、batch和drop_last与实际资源一致。

使用现有训练入口短跑：

    bash scripts/training/train.sh adaptac 'task_name=<task>' \
      '+max_updates=4' training.use_compile=false \
      workspace.wandb_cfg=null

完整方法正式训练：

    bash scripts/training/train.sh adaptac 'task_name=<task>'

恢复继续使用保存配方：

    bash scripts/training/train.sh adaptac \
      '+resume_from=experiments/adaptac/<task>/<run>'

完整预算训练完成后运行统一选点与冻结测试：

    bash scripts/eval/eval_pipeline.sh adaptac <task> <run>

不要为AdapTac新建训练shell、评测shell或调度器。完整方法的100k训练和正式评测按资源条件执行，不能因为当前缺少资源而省略代码实现或虚构结果；不额外运行对比/消融实验。

## 16. 交付标准

实施完成后的中文报告应包含：

1. 实际修改的文件、功能和必要适配。
2. 官方模块保留了什么，哪些只是接口修复，哪些属于编码器/平台替换。
3. 执行过的命令、验证结果与资源条件；明确未执行项目。
4. 完整方法的配置、训练、恢复和正式评测准确入口。
5. force_scale是否经过训练数据校准，是否做过真实闭环和完整延迟测量。
6. 如修改共享代码，给出真实阻塞点及最小修改理由。
7. 完成前检查diff，移除冗余开关、重复校验、未使用分支和无关变化。

实现可用且必要验证已完成时收敛。只在真实阻塞或会改变研究目标的冲突处向用户提出具体问题；普通接口适配、文件组织和代码风格问题自行依据本任务书与当前仓库解决。

## 17. 核心来源

- [目标项目基线](https://github.com/haoyangzhanglab/dexmani_policy/tree/1e1a5702a8e5b5b07e39a8b71d59a4bb27d0ca80)
- [公共实验配置](https://github.com/haoyangzhanglab/dexmani_policy/blob/1e1a5702a8e5b5b07e39a8b71d59a4bb27d0ca80/dexmani_policy/configs/common.yaml)
- [项目数据规格](仿真数据集格式.md)
- [项目训练、恢复和部署接口](项目架构.md)
- [项目统一评测协议](仿真评测机制.md)
- [官方Policy连接](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/policy/FFG_policy.py)
- [官方力扩散训练与采样](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/model/transformer_diffusion.py)
- [官方力Transformer](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/model/diffusion_modules/transformer_for_diffusion.py)
- [官方FGAF](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/model/fuse_network.py)
- [官方动作扩散包装](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/model/unet_diffusion.py)
- [官方默认配置](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/src/adaptac/configs/train_config.yaml)
- [官方实际LICENSE](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/LICENSE)
