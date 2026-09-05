# DexMani_Policy 修复 workflow

## 入口与优先约束

项目：`/home/zhanghaoyang/Desktop/dexmani_policy`。
任务依据：[修复执行指南 v2](<dexmani_policy 修复执行指南 v2 — Codex Optimized.md>)。
执行状态：[repair_progress.md](repair_progress.md)。本文件是执行约定，不是后台调度服务。

启动指令：`执行 docs/repair_workflow.md，从 docs/repair_progress.md 的下一步开始，自动推进至验收完成。`
一旦启动，阶段通过后直接进入下一阶段，不逐阶段请求确认。代码失败在当前阶段修复；
需要新权限或用户决策时，先完成不受阻的工作，再报告具体阻塞。

用户最新约束优先于原指南：

- Python 一律使用 `conda run --no-capture-output -n policy`。
- 不增加 legacy 兼容分支、迁移器、旧 API 别名、临时 adapter 或静默 fallback。
  Phase 3 的 best 入口要求有效 v2 selection record；缺失或字段不完整时明确报错，
  用户仍可显式选择 latest、milestone 或具体 checkpoint。旧记录重新运行 selector 生成，
  不实现原指南 §18 的 legacy 继续评测分支。
- Phase 4 同步更新当前 artifact 的生成、解析、构造调用和测试，明确写入 blending 系数
  或 null。缺字段不新增兼容回退；保持 v3 格式，不另建迁移机制。
- 清理仅限本次受影响入口和调用链中已失效的兼容/临时代码；不删除仍在使用的格式或运行时支持。
- 仅保留任务必需的边界校验；不在多层重复校验同一件事。不增加通用 schema、
  registry、配置解析框架、采样协议或新测试框架。
- 不改变指南列出的有意机制、算法范围及 Deferred；不修改相邻仓库或实验生成物。

## Agent 分档与调度

| 档位 | 模型 / reasoning | 适用任务 |
|---|---|---|
| sol-high | `gpt-5.6-sol` / `high` | DDP/梯度语义、选优与评测参数优先级、部署状态与跨模块机制 |
| terra-xhigh | `gpt-5.6-terra` / `xhigh` | 已明确契约的 runner/config 实现、定向回归测试、阶段独立检查 |
| luna-max | `gpt-5.6-luna` / `max` | 明确的局部工具修复、静态核对、测试执行与结果整理 |

主控负责分配文件、判断 gate、更新 progress 和推进阶段。难度排序是本轮用户指定的
调度规则。主控无需切换当前会话模型；实现子任务使用上述档位。

- 默认一个实现 agent。只有存在独立且有收益的工作，才额外启动一个辅助 agent。
  最多主控加两个子 agent；GPU 检查串行执行。
- 同一文件同时只有一个写入者，包含共享的 regression test 文件。辅助 agent 可只读
  核对其他模块；需要编辑时先由主控明确转移文件所有权。
- 不递归分派。简单任务由负责人直接完成，不为了使用三档模型而拆小任务。
- 子任务消息给出：阶段、目标行为、拥有的文件、所需指南章节、测试范围与交付格式。
  新阶段使用新上下文；工具支持时使用 `fork_turns="none"`，显式指定模型和 reasoning。
- 子 agent 返回：改动文件、问题证据、测试命令与结果、未验证项、下一步所需事实。
  原始日志只在故障定位需要时回传。主控复用已完成的验证结果。

## 自动阶段循环

`读交接 → 证明问题/最小回归 → 最小修复 → 定向验证 → 验收 → 写交接 → 下一阶段`

| 阶段 | 实现负责人 | 必须完成的工作 | 验收 |
|---|---|---|---|
| Preflight | 主控；独立检查可交 luna-max | HEAD、工作区、policy 环境、基线测试、宿主 GPU 可见性 | 保存实际结果和既有失败 |
| 1 Training | sol-high | BaseAgent.forward、统一 model 调用、累积尾组、空 loader/非法 K、同步 validator 调用点 | training regression + 全套；代表策略构建 smoke |
| 2 Runner | terra-xhigh | validation RGB 复用、对应 child 参数、辅助维度裁切、MultiTask 控制校验和致命异常传播 | eval regression；dp/multitask_dit smoke 与配置维度链 |
| 3 Selection | sol-high | v2 实际参数/seeds/相对路径、EMA fail-closed、held-out、best 默认与显式 override | eval regression；主控核对记录到 final eval 的链路 |
| 4 Deployment | sol-high | export/direct 共用解析、spec 系数、runtime blending/reset/warmup | deployment tests + 全套；核对 selected → export → runtime |
| 5 Local | terra-xhigh；局部独立工作可交 luna-max | absolute target、dropout、normalizer cache、FlowMatch 参数、attrs、MultiTask 参数 | algorithm regression + 最终全套；ManiFlow smoke |

Phase 1–4 按依赖串行验收，Phase 5 在核心 gate 通过后执行。
Phase 5 可将 augmentation/normalizer/replay buffer 的局部修改交给 luna-max，
terra-xhigh 负责 FlowMatch/MultiTask 和 `test_algorithm_regressions.py` 的统一集成。

关键验收语义：

- P1：`K=4,L=10` 更新三次，尾组每个 loss 均除以 2；测试 `L<K`、`K=1`、`L=0`、
  `K=0`。保持 microbatch loss 均值；同步边界覆盖 forward/backward；optimizer、
  scheduler、EMA 与 global_step 同步推进。
- P2：RGB 与 validation 数值相同；非 RGB 不变；28D 预测仅执行 19D；每个 child
  使用对应 dataset 的预处理值；fatal episode error 后不启动下一 task。
- P3：`显式 CLI > 显式 dotlist > selection record > config`，同时覆盖 section 参数
  和 NFE 列表。只有实际执行的 unique selection seeds 入记录；final 排除这些 seeds。
  剩余不足时 cap 并提示，剩余为零时报错；MultiTask 分母仍是实际 `(task, seed)` 数。
  best 路径错误直接失败。非 best 选择不继承 best record 的推理设置。
- P4：两个入口复用一个小 helper；blender 使用已验证的 CPU prediction，先裁控制维度；
  reset 清历史，warmup 临时替换 blender 并用 finally 恢复原对象和 RNG。
  模型 parity 与有状态 blender 测试分开；导出后权重不再重复携带 EMA 开关。
- P5：仅修 absolute consistency student/teacher target，relative 分支不变；
  局部校验按指南范围处理，normalizer 只修缓存失效。

## Gate 与 compact

每阶段只做一次简短的语义检查和计划中的测试。发现具体缺陷时修复，并重跑受影响测试；
没有新增修改或未解疑点时不重复全套、不安排多轮交叉评审。

验收逐项标记 `PASS / FAIL / NOT VERIFIED`。代码引入的 FAIL 必须先修复。
基线已有且无关的失败不阻塞推进。环境阻塞不能记为 PASS；可推进不依赖该检查的工作，
但依赖它的 gate 保持 NOT VERIFIED，全部必要检查完成前不能宣布最终验收完成。

每阶段结束，主控先让本阶段 agent 完成交付，再更新 progress 中的精简交接：

- 已完成阶段和实际改动文件；
- 已通过的测试及测试时的改动状态；
- 固定的接口/参数优先级/决策；
- 未验证项、阻塞与 Deferred；
- 下一阶段、负责人、文件所有权和第一条行动。

只保留恢复工作必需的事实，不复制完整聊天、日志或指南。下一阶段 agent 使用新上下文，
读取 workflow、progress、对应指南章节和相关代码，不重新探索已完成阶段。

这一步称为“阶段交接摘要”，与原生 compact 分开记录：原生 compact 是客户端操作，
当前工具未提供可调用接口，不能通过输出 `/compact` 假装执行。客户端提供原生操作时，
在写好交接后执行并确认完成；当前会话只能保证阶段摘要与新 agent 上下文，原生 compact
记为 NOT VERIFIED。不能为了实现它额外构建 app-server 调度框架或后台脚本。

## 验证命令与 GPU

全套（Preflight、P1、P4、P5 各一次）：

```bash
conda run --no-capture-output -n policy python -m unittest discover -s tests -p 'test_*.py' -v
```

定向测试使用相同 discover 入口，仅替换 pattern：
`test_training_regressions.py`、`test_eval_regressions.py`、
`test_deployment_*.py`、`test_algorithm_regressions.py`。
P4/P5 先完成定向测试再做最终全套；其他阶段复用本阶段已有测试结果。
Python 修改先做受影响文件的语法/导入检查。

GPU preflight 使用命令级 `require_escalated`，先只检查设备：

```bash
conda run --no-capture-output -n policy python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())'
```

必要 smoke 同样按需命令级提权，保持现有 sandbox/auto-review 策略：

```bash
conda run --no-capture-output -n policy python dexmani_policy/smoke_test.py dp3 maniflow sat
conda run --no-capture-output -n policy python dexmani_policy/smoke_test.py dp multitask_dit
conda run --no-capture-output -n policy python dexmani_policy/smoke_test.py maniflow
```

仅在对应阶段执行相应命令。缺依赖/数据/权重时记录原因，不安装依赖、不下载模型、
不通过修改策略绕过环境问题。真实多 GPU DDP smoke 单独记账；本 workflow 不授权
启动 DDP 训练、完整训练、长评测或视频录制。单元测试不替代 NCCL 运行验证。

## 交付

按指南 §49 输出五阶段结果、测试、改动文件、Deferred 和剩余风险；§47 验收矩阵
按实际证据填写。本轮新增严格 record/artifact 约定替代 legacy 兼容验收。
保留五个逻辑 diff 分组，不自动 commit、push、导出真实实验 artifact 或修改其他仓库。

配置依据：[Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)、
[原生 compact 命令](https://learn.chatgpt.com/docs/developer-commands?surface=cli)。
模型/effort 已与本机 Codex 0.153.4 bundled catalog 核对。
