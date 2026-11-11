# 2026 顶会小型 VLA/WAM 源码复核与 DiT 优化

本轮调研之后的收敛实现见[SmolVLA 骨干方案](SmolVLA骨干.md)。以下保留调研时的
实现状态和实验候选；压缩、深层时间注入与 WAM 辅助任务尚未作为默认组件加入。

核验日期：2026-11-11。本文接续[上一轮调研](小型VLA与DiT优化.md)，集中检查
2026 年正式会议、官方实现和模型规模。2025 年预印本、2026 年正式发表的论文可以
纳入；只有 README、未开放实现，或仅称 efficient 却没有可靠规模口径的项目不纳入
严格小模型集合。本轮只新增调研文档，没有修改 `dit.py`、loss 或训练配置。

## 筛选结果与规模口径

“小型”必须区分完整模型、语言骨干和可训练模块。严格总参数小于 1B 的已确认代表
是 Evo-1 和 X-VLA；下面另列约 1B、采用小语言骨干、以及训练轻量的 WAM 对照。
没有将它们都写成 sub-B 完整策略。

| 项目 | 2026 年会议核验 | 参数口径与开源状态 | 本次用途 |
| --- | --- | --- | --- |
| Evo-1 | CVPR；CVF proceedings，pp.13397–13406 | 0.77B 总模型；训练、推理、flow head 可读 | 核心小型 VLA；逐层轻量 time、context 位置、动作耦合 |
| X-VLA | ICLR Poster；OpenReview 作者记录及官方项目页互证 | 约 0.9B 总模型；完整代码 | 已调研的 sub-B 参照；双向交互、domain-aware I/O |
| StableVLA | ICML；PMLR 306:32264–32283 | 0.5B 是骨干口径，IB 模块额外参数 <10M；完整代码位于 HumanNet 子目录 | 鲁棒的视觉/context adapter 对照，不能报成精确 0.5B 总模型 |
| TwinVLA | ICLR；最终论文首页及官方项目页 | 最终论文为 1.3B；完整代码。HF 约 1.0B 页面是单臂预训练配置，不代表整个 twin 系统 | 约 1B 扩展对照；分组交互、shared-token 计算与 attention reweighting |
| MergeVLA | CVPR；CVF accepted PDF，pp.22335 起 | 模型卡 0.68B 明确排除视觉骨干，包含 Qwen0.5B+约0.18B动作头；完整代码 | 小语言骨干补充；CA-only 的适用边界、跨技能合并 |
| Light-WAM | CoRL；官方 accepted 表 Poster #55，OpenReview kB0KO1Wife | 论文默认总量 1.99B、trainable 0.44B；Wan2.1-T2V-1.3B 冻结骨干；完整代码 | 紧凑 WAM 补充；query pooling、训练期未来监督 |
| Being-H0 | ICML；PMLR 306:82265–82293 | 约1B预训练模型可用，但1B配置无robot alignment；公开机器人对齐 checkpoint 为8B-Align，完整训练脚本尚有TODO | 手部动作表征对照，不作为现成1B机器人策略推荐 |

严格条件下，本轮没有确认“2026 顶会 + 完整官方代码 + sub-B 总参数”的 WAM。
Light-WAM 是相对大视频模型更紧凑、训练轻量的 WAM，不是 0.44B 完整模型。

正式会议依据：

- [Evo-1：CVF CVPR2026 June 5 proceedings](https://openaccess.thecvf.com/CVPR2026?day=2026-06-05)，
  [作者论文与0.77B摘要](https://arxiv.org/abs/2511.04555)。
- [X-VLA：ICLR作者记录](https://openreview.net/search?content=authors&group=all&sort=cdate%3Adesc&source=forum&term=~Jianxiong_Li1)，
  [官方项目](https://thu-air-dream.github.io/X-VLA/)。
- [StableVLA：PMLR](https://proceedings.mlr.press/v306/fu26x.html)。其同名 GitHub 仓库只有网页，
  [PMLR Software 链接和项目页 Code 实际指向 HumanNet](https://github.com/DAGroup-PKU/HumanNet/tree/662d3ec846796761c74c40d36031290fd0ed0843/src/model/StableVLA)。
- [TwinVLA：最终会议论文](https://openreview.net/pdf?id=jG9W6nAwVz)，
  [官方项目](https://jellyho.github.io/TwinVLA/)。
- [MergeVLA：CVF accepted PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Fu_MergeVLA_Cross-Skill_Model_Merging_Toward_a_Generalist_Vision-Language-Action_Agent_CVPR_2026_paper.pdf)，
  [排除视觉骨干的参数说明](https://huggingface.co/FYX026/MergeVLA-LIBERO/blob/main/README.md)。
- [CoRL2026 官方录用表](https://www.corl.org/program/accepted-papers)。网页 reader 只显示外壳，
  官方原始 HTML 内的转义嵌入表包含 Light-WAM 标题、Poster #55 和上述 OpenReview 链接；
  该信息已从官方 HTML 直接核对。
- [Being-H0：PMLR](https://proceedings.mlr.press/v306/luo26e.html)，
  [1B配置](https://huggingface.co/BeingBeyond/Being-H0-1B-2508/blob/main/config.json)。

## 最有价值的局部改进

以下是将源码事实迁移到 dexmani 的推断，尚无本项目训练收益或 GPU 加速结论。

### A. Light-WAM：在 prefill 前建立可控的空间 token 瓶颈

原实现的 LearnedQueryPooler 对输入做 LayerNorm，用 learned queries cross-attend
视频 tokens。配置中 query 数为16，但默认 `merge_dim=null` 路径还把16个 slots
用 softmax 权重合并成一个向量；不能误称其原版输出16个 context memory tokens。

对本项目可借重采样部分，保留 K 个 query outputs，按帧压缩 spatial tokens，保留
每帧独立 joint/proprio token。以现有 SAT 配置为例：

| 设计 | 两帧 prefix 长度 P |
| --- | --- |
| 原始128 spatial + 1 joint/帧 | 258 |
| 32 query spatial slots + 1 joint/帧 | 66 |
| 16 query spatial slots + 1 joint/帧 | 34 |

按 K=16、SAT动作token数19计算，CA的prefix score长度从258减到34；联合SA的
key长度从277减到53。这分别是约7.6倍与5.2倍的score维度缩减，不代表端到端同倍
加速：resampler、FFN、encoder和kernel launch都仍有成本。应计入一次prefill开销。

重采样放在逐层 KV cache 之前，一次执行，训练时保持可微。保留原 RectifiedFlow 的
velocity 目标和 RTC，不搬原 Light-WAM 的 direct-action MSE head。仅有最终层
context 时也不假装拥有原版的真实多层视频 states。

风险是丢失小物体、指尖接触和局部几何；K=16/32应对照原P=258，而非默认全局
mean pooling。针对每帧/空间与proprio分开处理是本项目适配建议，并非Light-WAM原版
已验证机制。原版摄像头可横向拼图，proprio追加到骨干条件，并没有这种末端独立保留。

源码：[pooler L61](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/state_fusion_action_expert.py#L61)、
[slots合并 L138](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/state_fusion_action_expert.py#L138)、
[一次观测推理 L1388](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/lightwam.py#L1388)。

### B. Evo-1：先试 FFN 前的 additive time，而不是完整 AdaLN

公开的 BasicTransformerBlock 是普通 LayerNorm、action Q到context K/V的CA、FFN。
它在每层FFN之前把 time embedding 相加，没有完整的AdaLN scale/shift/gate。
因此可增加一个小消融：保持现有输入级action/time fusion，在各层归一化后的FFN输入
处再加time条件。需要投影时使用小Linear，不同时混入多种gate和norm变更。

Evo-1的action encoder有时间序列position，context不额外加一维序号编码。这支持
上一轮对点云context位置语义的修正，但不能单凭它认定所有context都应移除位置。

Evo-1的CA-only也不能直接照搬：它在末端flatten整段H*D，经全局投影后预测chunk，
在那里实现动作耦合。SAT是逐joint decode；若删除SA又不补其他joint mixing，
不同关节在给定context后失去直接通信。建议保留双向joint交互；SA频率可以单独测。

其evo1-flash分支优化主要在VLM的BF16/FlashAttention、14层截断与checkpointing；
head仍是nn.MultiheadAttention，未设置need_weights=False，也没有NFE式KV prefill。
不能为了模仿它而移除当前DiT已有的cache，更不能将分支名当成head加速证据。

源码：[FFN前time L128](https://github.com/MINT-SJTU/Evo-1/blob/5fd14b015013c4fd0aacf5f8f48f868ca9b870a2/Evo_1/model/action_head/flow_matching.py#L128)、
[action位置 L90](https://github.com/MINT-SJTU/Evo-1/blob/5fd14b015013c4fd0aacf5f8f48f868ca9b870a2/Evo_1/model/action_head/flow_matching.py#L90)、
[chunk读出 L340](https://github.com/MINT-SJTU/Evo-1/blob/5fd14b015013c4fd0aacf5f8f48f868ca9b870a2/Evo_1/model/action_head/flow_matching.py#L340)。

### C. StableVLA：把轻量鲁棒性模块放在条件接口、缓存之前

StableVLA的IB-Adapter实现有细节MLP通路和channel-covariance通路。后者将每个
head的Q/K按channel维做相关性矩阵，sigmoid门控后作用于变换后的V；它不是普通
token-token softmax。该操作对输入token重排等变，适合保留空间集合的顺序语义。

迁移候选是视觉/几何features到context的接口，不是每个denoise block都重复一次。
可先只处理spatial tokens，在concat joint/proprio前执行，并继续缓存后续逐层K/V。
这样更适合研究背景、遮挡和图像噪声扰动下的稳健性；不能将论文鲁棒性增益当成本项目
既定提升，也不能把这个额外模块当成无成本加速。

注意实际代码：外层fusion的`gate`是值0.3的buffer，再取tanh，不是可学习的零残差门；
clean通路有dropout。若自行设计learnable residual gate，应明确它是适配选择。

源码：[HF projector L242](https://github.com/DAGroup-PKU/HumanNet/blob/662d3ec846796761c74c40d36031290fd0ed0843/src/model/StableVLA/prismatic/extern/hf/modeling_prismatic.py#L242)、
[channel attention L55](https://github.com/DAGroup-PKU/HumanNet/blob/662d3ec846796761c74c40d36031290fd0ed0843/src/model/StableVLA/prismatic/util/nn_utils.py#L55)。

### D. TwinVLA：先诊断 token 组竞争，再决定 group prior

TwinVLA用对称的跨臂mask、各分支局部位置，共享视觉/语言输入只计算一份；它不是
把左右臂简单flatten后用全局tril。SAT也应按joint/部件语义处理mask，优先建立
全关节双向baseline，以后确有需求再做对称arm/hand或指组block mask。

其shared-modality概率乘2后重新归一化，使用STE保留原梯度，是维护预训练关注比例
的机制。对从头训练的DiT，不宜复制STE或固定乘2。可先抽样诊断各组attention mass，
包括空间、proprio、action/joint。当前258个obs对19个joint token，即使logits无信息，
token组数量也会影响总mass；这本身不证明attention已退化。

若诊断发现问题，正权w乘prob再归一化，前向等价于logits加log(w)，可以用显式可微
group prior实现。若意图让group权重不随token数变化，可测试log(alpha_g)-log(N_g)
这样的组先验，但不能预设各组应等权。另一候选是各组独立softmax再做output gate，
代价是额外attention计算。诊断应抽样执行，避免为记录权重而常态化生成完整attention矩阵。

TwinVLA动作head本身是普通双向Transformer，time+condition作为附加token；base代码
显式将head转FP32。因此2026开源代码并不支持“动作头都应无条件BF16”的统一结论。
本项目的BF16 SDPA仍需单独GPU稳定性与kernel基准。

源码：[对称mask L691](https://github.com/jellyho/TwinVLA/blob/d632d9378f3dbd7a9a32854d21b1642c6939f1f6/twinvla/model/base_models.py#L691)、
[reweighting L795](https://github.com/jellyho/TwinVLA/blob/d632d9378f3dbd7a9a32854d21b1642c6939f1f6/twinvla/model/base_models.py#L795)、
[DiT head L233](https://github.com/jellyho/TwinVLA/blob/d632d9378f3dbd7a9a32854d21b1642c6939f1f6/twinvla/model/heads/DiT_policy.py#L233)。

## 需要更大范围改动的借鉴

- **WAM训练信号**：Light-WAM训练期有未来video监督，但推理只从当前观测一次提取
  states并输出action，不生成未来视频。对本项目可研究future latent/geometry/contact
  辅助监督；它涉及数据、encoder、辅助head与loss，不是加深DiT即可。若预测future
  latent也用作condition，必须由当前观测预测，且放在NFE外，不能使用GT future。
- **部件动作表征**：Being-H0的wrist/finger分开GRVQ与vocabulary对齐，支持part-aware
  action_io的研究方向。优先利用现有SAT descriptors、归一化与RectifiedFlow.dim_groups，
  比较连续部件读出；无需先重做完整离散tokenizer/AR模型。SAT本来只有19个joint tokens，
  时间压缩不一定还能缩短这条轴。
- **多控制频率**：LaWAM的动作位置用物理时间i/hz并构造有效mask，适用于跨频率数据。
  它的Qwen3-VL-2B使其不属于严格sub-B集合；单频率场景暂不必复制。物理动作时间
  tau=i*dt与flow去噪时间t是不同变量，不能用它替换RectifiedFlow的积分时间网格。
- **DINO latent动力学**：[RSS2026的LDA-1B](https://roboticsproceedings.org/rss22/p210.html)
  把policy、dynamics和visual forecasting放在结构化DINO latent空间联合学习，支持
  优先研究latent/geometry辅助目标而非像素视频重建。不过其[官方实现配置](https://github.com/jiangranlv/LDA-1B/blob/06e6a274a9086cc26635a9fe663866335eb30fc5/README.md#L96)
  另有Qwen3-VL-4B语义encoder，不能把名称中的1B当作完整系统参数。
  [MMDiT位置实现](https://github.com/jiangranlv/LDA-1B/blob/06e6a274a9086cc26635a9fe663866335eb30fc5/lda/model/modules/action_model/flow_matching_head/mmdit/mmdit/mmdit_cross_attn.py#L105)
  对image使用Rotary3D、action使用Rotary1D，也支持分离位置语义。其image 3D是
  视频时间/高/宽网格，不能直接当作FPS点云的XYZ编码。完整联合future/action去噪还会
  改变条件依赖，不能在照搬后继续假设context KV与noisy action无关。
- **跨技能合并**：MergeVLA的CA-only旨在避免任务信息经action SA扩散，服务于模型合并；
  它是L1 head，不是flow。单任务SAT没有因此删除joint SA或加入router/MoE的依据。

源码：[WAM训练loss L1445](https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/lightwam.py#L1445)、
[Being-H0 wrist/finger L453](https://github.com/BeingBeyond/Being-H0/blob/ae92e46f671f48d91161ef1c8cc956bcdffab606/beingvla/models/motion/m2m/tokenizer/model.py#L453)、
[Being-H0对齐head L366](https://github.com/BeingBeyond/Being-H0/blob/ae92e46f671f48d91161ef1c8cc956bcdffab606/beingvla/models/vla/being_vla_model.py#L366)、
[LaWAM自然时间 L51](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/starVLA/model/framework/vlas/flowmatching_expert.py#L51)、
[MergeVLA实际Pro block L281](https://github.com/MergeVLA/MergeVLA/blob/5ef0e0d09af166d6e6c7128e0189cc0c5896737c/prismatic/models/action_heads.py#L281)。

## 对当前实现的实验顺序

| 次序 | 单独改变的因素 | 主要验证 |
| --- | --- | --- |
| 0 | 先补action_io、SAT双向与正确位置语义 | 编码/解码shape、保留零head初始化、joint通信、RTC与cache正确 |
| 1 | 每帧K=16/32 query resampling，保留proprio | prefill+NFE总延迟、显存、局部接触/小目标任务质量 |
| 2 | FFN前逐层additive time | 相同训练预算/NFE下的flow误差、任务成功率，单独记录参数增量 |
| 3 | spatial context鲁棒adapter | 干净与遮挡/噪声/背景扰动任务，确认没有破坏正常几何精度 |
| 4 | attention group诊断后再加prior | 组mass、梯度和任务收益，避免常态化attention权重开销 |
| 5 | SA每2/3/4层一次 | 保留关节耦合条件下的容量/速度折中，不直接切CA-only |
| 后续 | future contact/latent辅助监督 | label可用性与时序划分、推理无未来泄漏、辅助loss是否真正改善策略 |

工程精度优化另设FP32/BF16对照，不与结构变更同时发生。所有任务比较固定数据、seed、
encoder、动作表示、训练预算和NFE；分别计时encoder、query prefill、KV prefill、单步
expert与完整chunk。跨论文的FPS/成功率不作为本项目的可预期收益。

## 本轮未纳入核心集合的项目

| 项目 | 原因 |
| --- | --- |
| E-VLA | 虽称ECCV2026、基于SmolVLA，但官方仓库目前仅README/LICENSE等，无模型实现 |
| WholeBodyVLA | ICLR2026，但官方未开放可读模型实现 |
| FLOWER | 正式发表CoRL2025，不能因后续更新归入2026 |
| FALCON | ICLR2026且代码开放，但Kosmos-2 1.6B加ESM-1B不属于sub-B策略 |
| Being-H0.7 | README声称NeurIPS2026，但对应目录仅README，未开放WAM实现；0.7也不是参数口径 |
| Faster-WAM | 28.6M仅action expert，另有5B骨干；未核到2026顶会正式录用 |
| LDA-1B | RSS2026且完整开源，但官方配置另有Qwen3-VL-4B；不属于1B完整系统 |

这些排除仅表示不满足本轮筛选，不评价方法本身价值。本文没有运行上述项目的完整训练、
GPU性能基准或机器人评测；借鉴项仍需在本项目中验证。
