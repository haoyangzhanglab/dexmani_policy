# AFRO / CordViP 实施 Prompt

在 haoyangzhanglab/dexmani_policy 的 Codex 工作区复制下文。完整任务书见 [afro_cordvip_reproduction.md](afro_cordvip_reproduction.md)。此 prompt 触发实施；任务书提交本身仅包含文档。

```text
请在当前 dexmani_policy 仓库实际实施 docs/codex/afro_cordvip_reproduction.md，完成 AFRO 与 CordViP 的“观测预训练→encoder 导出→19 维关节策略训练→checkpoint 恢复→仿真评测”流程，不要只输出计划。

先读 AGENTS.md、任务书和相关调用链。任务书核对基线为 874090b5c4ea9e3d5b50e7e317772d0185dd06d0；先检查当前 HEAD 的相关变化，保留已有工作，不回退仓库。实际接口变化时做最小必要调整，记录影响算法语义的差异。

允许 clone KolaKivy/AFRO 和 xuanxuanzzzii/cordvip 到 /tmp 或不纳入 Git 的 tmp/，分别固定到 283024a49e2c14bc030debf15383a4e7ee73836d 和 ad6d441abe7b0480cf79b0311db266965c1313b6。新增算法直接移植官方活跃代码、保留来源/许可；尽量复用本项目 PointNet、StateEncoder、U-Net、Diffusion、数据采样/split/normalizer、PolicyTrainer 和评测。不整体搬运 workspace，不建设预训练框架、registry 或机器人适配层。

预训练代码集中在 dexmani_policy/training/pretrain/obs/，允许新建必要目录。按 P0→P1→P2→P3 持续推进，AFRO 完成后继续 CordViP，不因阶段结束而停下。可以先简要列核对结果和修改范围，随后直接实施。

必须核对：最新 train-only 全真实帧统计和保存 split；AFRO 原始 33 帧跨度/stride=2、latent=16、VICReg reduction、独立 teacher 初始化与 pre-optimizer 更新；仅冻结 AFRO visual。CordViP 保留官方 Transformer/LayerNorm、时间 attention 轴、三个独立 MSE、最后观测状态监督、固定 contact 点序、gamma=1/theta=10、默认 H/O/A=12/4/6。原生动作是 arm7+hand12 的实际 action[t]，不能复制上游 +2 状态移位。恢复使用 prepare_restore_config/initialize_training，完整 checkpoint 不依赖外部预训练文件。

先做 FP32/eager 的官方同权重输出/loss/梯度核对，再做小数据 fixture、有界训练、保存恢复和推理；具备 dexmani_sim 时做最小闭环。只运行足够验收的短任务，不擅自启动完整 100000-step/数百 epoch 训练。max_updates 只限制本次执行，不改总 scheduler 预算。AFRO 预训练不能把梯度累积称为相同大 batch 的等价实现。首版限定仿真，不扩展真机感知或触觉系统。

提供实际可用配置和 CLI，更新 docs/观测预训练.md，检查嵌套 YAML 打包、Open3D 可选隔离及缺权重显式报错。仅增加必要验证。缺依赖/数据/GPU 时，继续全部可独立完成的工作，说明具体阻塞与补验命令，不将未执行验证标为通过。重大证据冲突影响方法定义时明确提出，不凭常识替作者补算法。

最终报告修改文件、官方移植/本地复用范围、论文/代码/原生适配差异、实际执行命令及结果、未验证项，以及最短可运行的预训练/训练/恢复/评测命令。清理无关改动，不提交 tmp 参考仓库、数据、标签、checkpoint 或训练输出。除确实阻塞的方法定义问题外，自主完成必要工作。
```
