# Codex CLI 执行 prompt：DexMani Policy 轻量化

将下面完整提示复制给能够访问 `dexmani_policy` 工作区的 Codex CLI。任务是实施代码；此前文档提交本身不表示重构已完成。上一轮任务已实施的部分不要重复执行。

---

请直接实施 dexmani_policy 的轻量化重构。目标是让论文研究仓库轻而薄、简洁高效、正确好用：减少重复事实、跨模块知识和隐式行为，保持实验语义与恢复能力。

先完整阅读当前 AGENTS.md、README.md、docs/lightweight_refactor_taskbook.md 和相关验证文档。任务书若不在工作区，从以下地址读取，不要为取文档覆盖工作区或切换用户分支：
https://github.com/haoyangzhanglab/dexmani_policy/blob/main/docs/lightweight_refactor_taskbook.md

任务书审阅基线为 332cf9167debb75fee14b02125bd69a06569066b。先检查实际 HEAD、dirty diff、现有依赖和真实调用者；当前源码优先，已完成项不重复迁移。保护用户改动，必要时使用独立分支/worktree。

按任务书依次完成：

1. 将普通 best 加载与完整候选证据验真分离；写直接 best 指针，保留冻结测试计划与显式验真，提供清楚的旧指针处理方式。
2. 配置只保留一个可修改来源；数据事实由 Dataset 产生，Policy 不再接收 Dataset 或遍历 buffer/sampler。复用现有 data_recipe、hand_data、effective_config 和 policy_metadata。
3. 把 Sparse3D、Diffusion/RTC、手部先验等特定约束放回所属方法/组件，使 BasePolicy 变薄，保留已共享的数值路径。
4. 清理相关异常处理和使用文档，保留主异常与资源关闭；仅在测量支持时做局部性能优化。

每批先列简短成功标准，随后直接修改、验证并推进；不要停在建议、再次泛泛调研或只完成最容易的一项。常规实现选择自行解决，只有无法从代码查明且会改变研究语义的关键歧义才询问。遇到缺数据/GPU/权重或配套仓库时，完成其他可独立交付的工作，准确记录未验证范围。

必须保持：H/N/动作起点、训练 padding、完整解码后 future_actions[B,P,D_control] 及消费 C 的语义；split 和 normalizer 统计总体；LatentFlow 原始 q0、LAMP 历史、物理单位与 dtype；损失/EMA teacher/梯度累积/DDP forward/采样游标/完整 resume；有效 batch 和正式训练预算；固定选点预算、独立 test seeds 与现有排序。max_updates 不能重设学习率计划。Real 执行 C 不自动继承仿真 best 的 C。

复用现有 Trainer、PolicyInfo、EvaluationRequest、checkpoint 和目录结构。不要新增通用基类/registry/schema/工作流、哈希/指纹、永久兼容层、第二套权重格式、自动换样本/换 seed、静默 EMA 回退或无需求配置。保持 state_dict 参数路径和公共接口；不因另一个仓库不可用发布半个破坏性迁移。

验证按风险选择，优先已有测试与实际小模型数值对照；按任务书执行语法、config-only、定向测试，公共链路闭合后运行可用的完整单元测试。数据/设备允许时做受影响方法的 data-only、最多 4 次更新 smoke 和必要的有限恢复/双 rank 检查。禁止为通过检查改正式配方，不把 config-only 当作训练/DDP 验证。只对实际通过的命令写 PASS，其他写 NOT VERIFIED、具体原因及最短补验命令；没有测量就不宣称提速。

不自动升级依赖、下载权重、启动 SSH/完整训练/长评测/真机运动，不改写历史实验。保持行为的重构与 best 协议变化分开提交；可做本地提交，不自动 push/merge。

最终用中文给出：完成/未完成项，实际减少的耦合与重复，行为及协议变化，关键文件，真实测试结果，剩余限制。同步必要文档，避免新增冗长的规则和状态文件。请现在开始检查并实施。
