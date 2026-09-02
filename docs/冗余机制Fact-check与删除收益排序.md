# 冗余与过度工程化机制：Fact-check 与删除收益排序

> Fact-check 日期：2026-09-02（UTC）
> 审查分支：`main`
> 审查父提交：`1b777fbcb032fe5a36919b483caacec0e8608bfc`
> 最近源码提交：`550fa8f5af20745eb3abd61e1bfd481f5a7531a8`
> 范围：训练、评测、配置、策略、脚本、文档和部署边界；不含外部私有调用方。

## 1. 结论摘要

本轮把上一轮“冗余、无用、过度工程化”结论逐条回到最新 `main` 的定义、调用点、
canonical 配置、实验记录和部署资格矩阵核验。结论分为三类：

1. **可以进入删除队列**：Top-K 训练 checkpoint、生产模块内联 example/self-test、
   Trainer 在线验证/评测残留、eager barrel import、5 个仓内零调用 helper、未支持的
   `per_task` normalizer 入口等。
2. **应简化而不是直接删除**：自适应 checkpoint selector、四套评测入口/结果协议、
   9 份重复主配置、17 个 shell 脚本、重复文档、双日志与 deployment metadata。
3. **只能条件归档**：MoE、MultiTask、RGB-D、DDP、R3D aux-EE、预留 backbone/插件和
   暂无近期实验的完整策略。静态“没有 canonical 调用”不等于没有外部或未来用途。

按**预期净删除收益**排序，最先做的是：

1. 删除训练 Top-K 状态机；
2. 将生产模块中的约 1,626 行 example/self-test 迁移为少量正式测试后删除；
3. 删除 Trainer 从不进入的在线 validation/evaluation 分支和 16 个无效
   `val_dataloader` 配置块；
4. 把 5 个 eager `__init__.py` 改为空导出或惰性导出；
5. 删除仓内零调用 helpers 和明确不可用的配置入口。

这五步先减少确定的维护面，不改变任何 canonical 训练算法。MoE/MultiTask 等大模块虽然
潜在删行更多，但必须先由实验负责人确认是否退出研究路线，不能因为静态调用少就直接删。

## 2. 核验口径与仓库量化

### 2.1 证据等级

| 等级 | 判定条件 | 本文动作含义 |
|---|---|---|
| F3 | 定义、仓内调用、canonical 配置和文档/实验证据均已交叉核验 | 可作为当前仓库事实 |
| F2 | 静态调用与配置已核验，但无法排除仓外 API 使用或未来实验 | 删除前做 owner 确认 |
| F1 | 主要依据文档或间接证据 | 仅列为调查项，不主张删除 |

### 2.2 “删除收益”的排序方式

本文按预期净收益排序，而不是只按 LOC：

- 能否消除第二套状态机、错误分支或可选依赖；
- 可减少的生产/配置/文档表面积；
- 当前不使用的证据强度；
- checkpoint、部署协议、外部调用和实验路线的兼容成本；
- 是否已有安全替代路径和回归测试。

因此，“1,000 行但需要研究决策”的模块会排在“150 行且确定无效”的机制之后。

### 2.3 当前规模快照

| 项目 | 核验值 | 说明 |
|---|---:|---|
| Python 源码 | 23,108 行 | `dexmani_policy/**/*.py` |
| 主配置 | 9 份 / 2,142 行 | `configs/*.yaml` |
| DDP overlay | 7 份 / 233 行 | `configs/ddp/*.yaml` |
| docs 文档 | 10 份 / 4,320 行 | 含上一份全仓审查文档 |
| 根级说明文档 | 777 行 | README 358、CLAUDE 286、AGENTS 133 |
| 文档总计 | 5,097 行 | docs + 三份根级说明 |
| shell 脚本 | 17 个 / 1,838 行 | training、eval、remote、utils |
| 评测 Python 入口 | 1,829 行 | eval 518 + selector 598 + demo 330 + eval_utils 383 |
| deployment 源码 | 2,478 行 | export、qualify、restore、matrix、`__init__` |
| tests | 83 个 test method | 68 个 deployment，15 个其他路径 |

这些数字是维护面指标，不等同于可直接删除的行数；各候选之间也有重叠，不能相加当作总收益。

## 3. 已确认删除队列：按预期净收益排序

| 排名 | 候选 | 事实等级 | 保守可减表面积 | 净收益 | 主要风险 | 建议 |
|---:|---|:---:|---:|---|---|---|
| 1 | 训练 Top-K checkpoint 状态机 | F3 | ≥150 行 + 9 个配置块 | 极高 | 旧外部调用 `workspace.save_topk/best` | 删除 tracker/config；保留 milestone、`latest.pt` 和离线 best 协议 |
| 2 | 生产模块内联 example/self-test | F3 | 约 1,626 行 | 极高 | 直接删会丢断言 | 先迁移少量有价值断言到 `tests/`，再删 example 与 raw `__main__` 测试 |
| 3 | Trainer 在线 validation/evaluation 残留 | F3 | ≥140 行 | 高 | 仓外代码可能直接调用 Trainer 方法 | 删除前做一次全局/外部 API 确认；保留离线 env runner |
| 4 | eager barrel imports | F3 | 约 28 行直接导出，显著减少隐式依赖面 | 高 | 两个内部调用方依赖 barrel export | 改直接导入或惰性导出；加逐策略 import smoke |
| 5 | 自适应 best-checkpoint selector | F3 | 598 行可被短小固定评测替代 | 高 | 固定全量评测算力更高 | 以固定 seed、同 policy RNG 的全候选比较替换；异常必须中止选择 |
| 6 | 5 个仓内零调用 helper | F2 | 75 行 | 中高 | 可能是仓外 API | 若无外部用户，单独删除并记录 migration note |
| 7 | `per_task` normalizer 伪入口 | F3 | 小 | 中高 | 未来 MultiTask 设想 | 当前入口必失败；删除选项比保留半实现更安全 |
| 8 | modality dropout 全仓 wiring | F3 | 约数十行 + 9 个配置块 | 中 | 可能计划做 ablation | 所有 canonical 值为 0；无近期实验则删除公开参数和分支 |
| 9 | ActionFlow step conditioning | F3 | 小；另减 214,016 个冻结参数 | 中 | strict checkpoint key 兼容 | 用一次性 checkpoint 转换或版本兼容加载后删除 |
| 10 | MoE boost/override 支线 | F3 | 约数十行 + 示例/配置 | 中 | 只在保留 MoE 时适用 | `use_boost=false`；若无 boost 实验，删除 schedule 和 production `override_idx` |

### 3.1 训练 Top-K checkpoint 状态机

**Fact-check：成立。**

- `common/checkpoint_io.py:129-243` 定义 115 行 `TopKCheckpointTracker`，维护
  `scores.json`、score cache、恢复和淘汰逻辑。
- `training/workspace.py` 仍有 `CheckpointConfig`、`topk_tracker`、`save_topk()`，
  9 份主配置都传入 `workspace.checkpoint_cfg`。
- 仓内没有 `save_topk()` 调用；唯一引用是定义和 tracker 自身 wiring。
- `training/trainer.py:434-460` 只保存 20/40/60/80/100% milestone，写入
  `monitor={}`，并明确注释 “No score, no TopK tracking”。
- 正式 best checkpoint 由离线 `select_best_ckpt.py` 写 `best_ckpt.json`，不是训练
  tracker 选择。

**删除建议：**

1. 删除 `TopKCheckpointTracker`、`CheckpointConfig`、`save_topk()` 和 9 个
   `checkpoint_cfg` 块；
2. `TrainWorkspace.resolve_checkpoint_path()` 只保留 resume 所需的 `latest`/显式路径；
3. 为兼容旧 `simple.v1` checkpoint，可暂时保留 `state.monitor` 的宽松读取，下一次格式升级再删；
4. 回归验证 milestone 原子写入、`latest.pt` 原子替换、interrupt resume 和离线 best 解析。

不要连带删除 `CheckpointStore`、临时文件 + replace 或 `latest.pt`；这些仍是有效安全机制。

### 3.2 生产模块内联 example/self-test

**Fact-check：成立，前次“约 1.5k”应精确更新为约 1.6k。** AST 核验结果：

| 组成 | 行数 |
|---|---:|
| 23 个 `example()` / `_example()` 函数 | 1,288 |
| 对应 example `__main__` guard | 46 |
| 5 个 raw `__main__` self-test | 292 |
| 合计 | **1,626** |

最大的块包括 `core/moe.py` 213 行、`plugins/moe.py` 137 行、
`action_flow_dit.py` 103 行、`token_compressor.py` 95 行、
`pointnext_tokenizer.py` 74 行和 `maniflow.py` 73 行。

这些测试和示例混在生产模块里，未被 `pytest` discovery 自动执行，还会让核心文件显著变长。
但其中包含 MoE aux-loss、ActionFlow shape/cache、backbone shape 等有价值断言，不能无差别删除。

**删除建议：**只把稳定契约迁入正式测试；打印参数量、随机 tensor 演示、重复 shape 输出直接删。
保留真正 CLI 的 `__main__` guard（train/eval/export/tools），它们不属于这 1,626 行候选。

### 3.3 Trainer 在线 validation/evaluation 残留

**Fact-check：成立。**

- `train.py` 固定构造 `val_loader=None`、`env_runner=None`；DDP 路径也不构造验证 loader。
- `Trainer.validate()` 53 行、`Trainer.evaluate()` 30 行，仓内没有调用点。
- `TrainLoopConfig.max_val_steps` 只服务于死的 `validate()`。
- 9 份主配置和 7 份 DDP overlay 都保留 `val_dataloader`，训练构建路径从不读取它。
- `env_runner` 配置不能整体删除：离线评测通过 `training/eval_utils.py` 实例化它。

**删除建议：**删除 `TrainingComponents.val_loader/env_runner`、Trainer 对应构造参数、两个方法、
`max_val_steps` 和 16 个 `val_dataloader` 配置块。保留各主配置的 `env_runner` 供离线评测使用。

### 3.4 Eager barrel imports

**Fact-check：成立。** 以下 `__init__.py` 在导入任一具体子模块前会先执行，并把同包其他实现
和可选依赖一起导入：

- `agents/core/__init__.py`：9 个 Agent；
- `obs_encoder/pointcloud/__init__.py`：PointNet、PointNext、R3D/Uni3D 等 7 个导出；
- `obs_encoder/rgb/__init__.py`：8 个 backbone/processor；
- `obs_encoder/text/__init__.py`：CLIP + T5；
- `obs_encoder/plugins/__init__.py`：3 个 TokenCompressor 导出。

结果是，Hydra 导入 `dexmani_policy.agents.core.dp3.DP3Agent` 时也会经过
`agents.core.__init__`；点云/RGB barrel 又会提前触发 PyTorch3D、Transformers、torchvision
等本策略未必需要的模块。`deployment/__init__.py` 已提供惰性导出的正确先例。

内部仍有三处需同步改写的 barrel 使用方：`core/dp.py`/`core/moe.py` 从 RGB barrel 导入，
`pointcloud/registry.py` 从 pointcloud barrel 导入。先改为具体模块导入，再清空或 lazy export。

### 3.5 自适应 best-checkpoint selector

**Fact-check：过度工程化结论成立，并且有三项 correctness 风险。**

仓库每次只有 5 个固定 milestone，却用 598 行脚本实现初筛、增量 tie-break、不同候选的
动态 episode 数、视频目录和两个 best JSON：

1. Phase 1 的 checkpoint 级异常被转换成 0 episode / 0% accumulator；如果全部失败，或正常
   候选也为 0%，失败候选仍可能按 `global_step` 获胜并写 `best_ckpt.json`。
2. Phase 2 某候选 batch 异常后继续，候选会以不同 episode 分母比较 success rate。
3. 全局 RNG 只在整次选择开始时设置一次；相同 environment seed 不保证不同 checkpoint 使用
   相同 policy sampling noise，候选顺序和 episode 长度会推进 Torch RNG。

**替代建议：**对 5 个 milestone 使用完全相同的固定 25 或 50 seeds；每个
`(checkpoint, env_seed)` 恢复相同 policy RNG；任一 checkpoint load/模型/CUDA 异常使选择整体失败，
而不是记作任务失败。排序固定为 success rate、成功 episode avg steps、global step。这样运行量
略高但结果可比较，代码和失败语义都更短。

### 3.6 仓内零调用 helper

静态搜索和 AST 调用核验只发现定义，没有生产/测试调用：

| 函数 | 文件 | 行数 | 备注 |
|---|---|---:|---|
| `resolve_best_checkpoint` | `training/eval_utils.py` | 35 | 文档仍引用；实际入口使用 `resolve_checkpoint_path` |
| `square_distance` | `obs_encoder/pointcloud/ops.py` | 18 | 无仓内调用 |
| `compare_prediction_snapshots` | `deployment/restore.py` | 9 | 无仓内调用 |
| `logit_normal_density` | `action_decoders/time_sampler.py` | 7 | 无仓内调用 |
| `get_default_optim_group` | `agents/optim_util.py` | 6 | 无仓内调用 |
| **合计** |  | **75** | 仅代表当前仓库调用图 |

这是 F2 而不是 F3 删除判断：Python 没有私有 API 边界，仍需确认没有仓外 notebook/脚本直接导入。
若无外部用户，删除 `resolve_best_checkpoint` 时同步修正文档，不要误删当前在用的
`resolve_checkpoint_path`。

### 3.7 MultiTask `per_task` normalizer 伪入口

**Fact-check：成立。** `MultiTaskDataset` 接受 `normalizer_mode="per_task"`，但标准构建函数先执行
无参数 `dataset.get_normalizer()`，因此在后续明确的 `NotImplementedError` 之前就抛
`ValueError("requires task_name")`。这不是可用功能，只是可选项表面存在。

如果没有近期 per-task normalization 实验，删除 `per_task` 选项、normalizer dict 和错误分支；
如果确实需要，则应先定义 Agent 如何按 batch task 选择 normalizer，不能只调整异常顺序。

### 3.8 Modality dropout

**Fact-check：成立。** 9 份 canonical 主配置全部设置
`modality_dropout_probs.joint_state: 0.0`，但参数从所有 Agent 传到 `BaseAgent`，训练分支仍遍历、
校验并屏蔽观测。`maniflow.yaml` 同一行还保留“20% state dropout”注释，与实际 0.0 冲突。

若下一轮实验计划不包含 modality ablation，删除配置块、构造参数和 BaseAgent 分支；至少应先修正
误导注释。未来需要时用一个小实验分支重引入，比长期维持跨 9 策略 wiring 更便宜。

### 3.9 ActionFlow step conditioning

**Fact-check：成立，但不能无迁移直接删。**

- canonical `use_step_conditioning=false`；所有 flow loss/Euler/Midpoint 调用均传 `step_size=0.0`；
- `ActionFlowDiT` 仍实例化 `step_embedder`，禁用时仅冻结参数以保留 strict-load key；
- 默认 `step_embed_dim=64`、`hidden_dim=768` 时，该 MLP 有 214,016 个冻结参数；model/EMA
  checkpoint 都会携带对应 state；
- 文档明确把它作为已关闭、无实验收益的 branch。

删除顺序应为：给旧 checkpoint 提供一次性 key-drop 转换，或在一个明确版本窗口内允许缺失
`step_embedder.*`；随后删除 config、Agent/DiT 参数和恒 0 的调用参数。不要以 `strict=False` 全局
放松加载来换兼容性。

### 3.10 MoE boost 与 `override_idx`

**Fact-check：成立。** canonical `moe_dp.yaml` 为 `use_boost=false`，但仍公开 5 个 boost 参数，
保存 epoch 驱动的 active-expert/top-k 状态机。`override_idx` 的仓内实际使用只在内联 example，
production Agent/encoder 仍逐层传递该参数。

如果 MoE 保留而 boost 没有计划，删除 boost schedule 与 production override 接口。需要同时修正
README 的“top-2 稀疏路由不增推理 FLOPs”说法：当前 `aggregate_experts()` 先执行所有 active
experts 并 `torch.stack`，之后才 gather top-k，**不是 compute-sparse MoE**。保留 MoE 时应二选一：

- 实现真正的按路由 dispatch；或
- 承认其为 dense multi-expert，删除“稀疏/不增 FLOPs”描述和无效稀疏复杂度。

## 4. 应简化而不是直接删除：按收益排序

| 排名 | 机制 | Fact-check | 简化方向 | 不应做的事 |
|---:|---|---|---|---|
| 1 | 评测入口、selector、结果协议 | 1,829 行；checkpoint/result 解析重复 | 一个 eval engine + 薄 subcommand；一个 canonical JSON | 不要继续增加第五个入口/结果文件 |
| 2 | 9 份主配置复制 | 129 个非注释行出现在 ≥5 份配置 | 只抽一个小 `_base.yaml`，策略差异留本地 | 不要拆成大量 Hydra config group |
| 3 | 17 个 shell 脚本 | 1,838 行；remote/eval 重复解析 | 少量稳定命令或单一 `remote.sh` subcommand | 不要自动按启发式删除实验目录 |
| 4 | 5,097 行说明文档 | 3 个断链；计划/完成报告并存 | README/AGENTS/短架构 + archive；从 JSON 生成表 | 不要手工同步多份文件树和参数表 |
| 5 | 双日志 | 所有主配置 W&B offline，JSONL 同时始终写 | JSONL 默认；W&B 按需启用 | 不要仅为“可能同步”强制依赖 W&B |
| 6 | EMA compile/存储 | 仅 ManiFlow 把 EMA 当训练 teacher | 只在 teacher 路径 compile EMA；再评估轻量 EMA | 不要删除 eval 所需 EMA 权重 |
| 7 | deployment metadata 重复 | payload/contract/sidecar/handoff 有字段复制 | 与 Real 协同设计 v3 单 manifest | 不要删除 strict restore/hash/parity/atomicity |

### 4.1 统一评测引擎和结果协议

当前 `eval_best_ckpt.py` 518 行、`select_best_ckpt.py` 598 行、`record_demo.py` 330 行、
`training/eval_utils.py` 383 行，共 1,829 行。它们重复处理 config、checkpoint、seed、video dir、
结果写入和报错；并产生：

- `latest.pt`、`best_ckpt.json`、可选 `best.pt`、`deployment_latest.pt`；
- `_result.txt`、`result_details.json`、`eval_summary.json`、`best_ckpt_selection.json`；
- `CheckpointStore.resolve_path`、`eval_utils.resolve_*`、deployment `_resolve_checkpoint`
  三套解析语义。

建议一个纯 Python eval engine 返回结构化 report，CLI 只保留 `select`、`evaluate`、`demo` 薄命令。
`evaluation.json` 作为唯一事实源，`_result.txt` 仅在 RoboTwin 兼容模式下由 JSON 生成；
`latest.pt` 只表示 resume，`best_ckpt.json` 只表示离线选择，`deployment_latest.pt` 只表示已发布
Real artifact，三者不互相 fallback。

### 4.2 配置去重

9 份主配置共 2,142 行；按去注释后的逐行集合统计，129 个唯一行出现在至少 5 份配置中，
这些行产生 944 次“除第一份外的重复出现”。这是结构重复指标，不是可直接删 944 行的估算。
每份配置有 64.0%–90.8% 的唯一非注释行属于这种高频公共行。

只抽取 action/horizon、通用 dataloader、milestone、workspace、eval video 等稳定公共项到一个
`_base.yaml`。Agent、dataset、optimizer、augmentation 和策略特有 eval 参数继续放在各主配置。
这样减少漂移，又避免把简单个人研究仓库改造成复杂配置产品线。

### 4.3 Shell 脚本

17 个脚本中，`clean_experiments.sh` 298 行、`train_remote.sh` 216 行、`sync_down.sh` 173 行、
`eval_pipeline.sh` 151 行。remote 脚本重复服务器路径、参数校验、SSH/tmux 和 conda 处理；eval wrapper
重复三元实验路径和环境激活。

建议保留 `train.sh`、`evaluate.sh`、`download_pretrained.sh`，把远程操作收成
`remote.sh {sync,train,status,logs,stop,pull}`。`clean_experiments.sh` 当前会根据文件 mtime、checkpoint
文件名和 `<16000` step 启发式执行 `rm -rf`；对个人实验仓库，改成只输出候选清单或移除自动删除，
实际清理由人工显式路径完成更安全。

### 4.4 文档

当前 docs 4,320 行，加 README/CLAUDE/AGENTS 共 5,097 行。至少 3 个本地链接已断：

- `docs/SSH服务器训练部署.md` → `同步机制分析.md`；
- `docs/仿真评测机制.md` → `DP3-R3D-ManiFlow测试结果0808.md`；
- `docs/项目架构.md` → 同一个已不存在的 `0808` 文件。

建议保留 README、AGENTS、精简架构、正式实验结果和本审查文档；把已完成的 deployment plan、旧
ActionFlow 执行指南移到 `docs/archive/` 并加 `Status: Completed/Archived`。文件树、支持矩阵和参数表
尽量从代码/JSON 生成，避免 README、CLAUDE、项目架构三处手工同步。

### 4.5 双日志与 EMA compile

`TrainWorkspace` 无条件构造 JSONL 和 W&B logger；9 份主配置的 W&B 都是 `mode: offline`，另有
129 行 `wandb_sync.sh`。若实际只读 `metrics.jsonl`，把 W&B 变为可选依赖和显式开关即可；如果团队
确实同步 W&B，则保留，不应把“双写”直接判死代码。

`compile_models()` 同时 compile model 和 EMA。9 个策略中只有 ManiFlow 设置
`use_ema_teacher_for_consistency=true`，其余 EMA 在训练 step 不 forward。因此其余策略 compile EMA
backbone 只增加 compile 时间/缓存，不带来训练吞吐收益。先改成“teacher 才 compile EMA”；完整 EMA
仍用于 checkpoint/eval，不能删除。当前完整 EMA 也会复制 frozen backbone 和 persistent buffer，
确有额外显存/存储面；进一步做轻量 EMA 仍需验证这些 buffer、normalizer 和 strict checkpoint 语义，
收益证据不足时不要贸然重构。

### 4.6 Deployment metadata

`export.py` 的 payload `state` 同时保存 `train_params`、`inference_config`、`data_contract`、
`producer`，并在 `deployment_contract` 中再次嵌套同一组对象；sidecar 又复制 allocation/producer，
handoff JSON/Markdown 再展示一遍。字段漂移风险真实存在，但当前 Real consumer 接受固定
`dexmani.deployment.v2`/sidecar-v2 schema，不能单仓删字段。

若需要 v3，应与 `dexmani_real` 同步定义 `{format, manifest, weights}`，sidecar 只保存 filename、
size、artifact hash、manifest hash，Markdown 从 machine-readable receipt 生成。strict restore、
`weights_only=True`、禁止网络、hash、原子 selector、direct/export parity 都是部署安全边界，不属于
冗余删除项。

## 5. 条件归档候选：按潜在收益排序

以下排序按“如果研究路线明确退出，可减少多少维护面”，不代表当前授权删除。粗略行数相互重叠，
并包含第 3 节可能先迁移的 example，不能相加。

| 排名 | 候选 | 可见专用表面积 | 当前证据 | 决策门槛 |
|---:|---|---:|---|---|
| 1 | MoE 策略 | 约 1,090 行 | canonical config 存在；deployment deferred；不是真 sparse compute | 未来 1–2 轮是否有明确 MoE hypothesis/结果 |
| 2 | MultiTask 策略 | 约 1,025 行 | canonical + DDP 存在；deployment deferred；`per_task` normalizer 不可用 | 是否继续多任务训练和 task-text contract |
| 3 | 预留/无 canonical 的独立实现 | 约 1,014 行 gross | T5/TokenCompressor 明确预留；CLIP Vision/SigLIP/PointNext global 无 canonical | 逐组件 owner 确认；可移实验分支而非永久删除 |
| 4 | DDP 支持 | 至少 508 行 + Trainer 分支 | 7 个 overlay 和 remote 文档存在；无证据证明从未使用 | 是否实际拥有/使用多 GPU 资源 |
| 5 | RGB-D 支持 | 约 458 行 gross | 无 canonical sensor modalities 包含 depth/camera | 是否有近期 RGB-D 数据与实验 |
| 6 | R3D aux-EE | 16 文件、76 个源码/测试/配置引用 | canonical false，但已有 deployment parity 测试 | 是否保留 wrist-pose auxiliary experiment/checkpoint |
| 7 | 完整非主线策略 | 潜在最大 | 9 个 canonical 策略均可构造；实验/部署证据不均 | 明确未来一个月 active strategy set 后逐个 archive |

### 5.1 MoE

`core/moe.py` 491 行 + `plugins/moe.py` 364 行 + config 235 行，约 1,090 行。它有 canonical
入口，不能叫“仓内死代码”；但 qualification matrix 为 deferred，仓库没有与 DP3/R3D/ManiFlow
同等级的正式结果文档，而且当前 dense expert 执行推翻了“不增推理 FLOPs”卖点。

如果没有新的可检验 MoE hypothesis，归档整个策略比继续修 boost、override、router 日志和部署适配
更有收益；如果保留，则优先修正 compute 语义，不要只删小支线后继续声称 sparse。

### 5.2 MultiTask

专用 Agent 229 + dataset 241 + runner 179 + config 285 + DDP overlay 34 + CLIP text encoder 57，
约 1,025 行。它有 canonical 入口，不能叫死代码；但 `per_task` normalizer 未接入标准训练，Real
deployment 也因动态 task-text contract 缺失而 deferred。

这里必须撤回一项旧判断：**当前 `MultiTaskSimRunner.run()` 已接受 `denoise_timesteps`、
`eval_episodes`、`video_save_dir`，`rates/steps` 为空时也有保护，不能再称“评测 API/空统计必坏”。**
保留或归档 MultiTask 应基于研究路线，而不是这两个已不成立的 bug。

### 5.3 预留组件和 backbone

逐项核验后，可称“无 canonical 配置”的是 CLIP Vision、SigLIP、PointNext global encoder、
T5TextEncoder、TokenCompressor；后两者在 README/CLAUDE/AGENTS 中明确标记为预留，不是意外死代码。
个人研究仓库若不需要长期保存所有设想，可将它们移到实验分支或 archive；删除前仍需检查历史
checkpoint/notebook。

不能把以下活跃实现一并归档：

- DINO：`dp.yaml` canonical 使用；
- R3M：`moe_dp.yaml` canonical 使用；
- ResNet：`multitask_dit.yaml` canonical 使用；
- Uni3D：`r3d.yaml` canonical 使用；
- PointNext **patch tokenizer**：ActionFlow/SAT 使用，和无 canonical 的 global `PointNextEncoder`
  不是同一删除对象。

### 5.4 RGB-D

9 份主配置的 `sensor_modalities` 只有 joint_state + point_cloud 或 joint_state + rgb，没有 depth/camera。
静态无 canonical 调用的 gross surface 包括 `RGBPCDataset` 60 行、`GeometryProcessor` 220 行、
`ImageProcessor.process_rgbd` 64 行和三个 38 行 `backproject` 方法，共约 458 行。

但 RGB encoder 本身仍活跃，只能删除 RGB-D 方法/processor，不能删除 DINO/R3M/ResNet。若计划
接入 RGB-D，应先有数据 contract 和一份 canonical config，否则长期保留半条链路收益低。

### 5.5 DDP、R3D aux-EE 与策略组合

DDP 有 275 行入口、233 行 overlay，以及 Trainer distributed 分支；ActionFlow 最新结果是在单张
4090 上完成，但 remote 文档明确支持多卡。没有运行记录不等于从未使用，必须由资源/实验 owner
确认后才能归档。

R3D `use_aux_ee=false`，但其数据布局、Agent 输出、diffusion loss、checkpoint train params、
deployment allocation 和 parity tests 跨 16 个文件。删除潜在收益不小，兼容风险也高；应以历史
aux-EE checkpoint 是否需要继续 restore 为门槛。

策略层面已有等等级实验记录的至少包括 DP3、R3D、ManiFlow；ActionFlow 有最新 100k 训练/NFE
结果。部署资格是 DP3 qualified，ActionFlow/DQ-RISE/R3D conditional，其余 deferred。建议先明确
个人研究的 active set，例如“DP3 baseline + ActionFlow 主线 + R3D/ManiFlow 对照”，再把一个月内
无实验计划的其他策略移到 archive branch。不要为了减少重复把不同算法强行合并。

## 6. 已纠正或驳回的旧结论

| 旧判断 | 复核结果 | 当前结论 |
|---|---|---|
| MultiTask runner 与统一 eval API 不兼容 | **已不成立** | 当前签名兼容四个参数 |
| MultiTask 空 rates/steps 会除零 | **已不成立** | 当前用 `if rates/steps else None` |
| RGB backbone 可整体视为 dormant | **错误** | DINO、R3M、ResNet 分别有 canonical 使用方 |
| Uni3D/PointNext 都未使用 | **错误** | Uni3D 用于 R3D；patch tokenizer 用于 ActionFlow/SAT |
| T5/TokenCompressor 是误留死代码 | **证据不足** | 文档明确标注预留；只能做路线决策后归档 |
| deployment 多层校验都是重复 | **错误** | 独立 restore/parity/hash/no-network/atomic selector 防 common-mode failure |
| Top-K 与 offline best 都可一起删 | **错误** | 只删训练 tracker；离线 best 选择仍有真实需求，但实现应简化 |
| DDP 无当前结果即可判无用 | **证据不足** | 有入口、overlay、remote 流程；需 owner 确认实际资源使用 |

本轮也把两个量化值校正为最新结果：高频公共配置行是 129（不是 128）；内联 example/self-test
候选是约 1,626 行（不是笼统 1.5k）。

## 7. 明确保留：不要作为“去冗余”误删

1. checkpoint 临时文件 + atomic replace、milestone 和 `latest.pt` resume；
2. deployment strict restore、`weights_only=True`、artifact/manifest hash、禁止网络/训练资产、
   direct-vs-export parity、原子 selector 与回滚；
3. ActionFlow 独立 flowmatch/DiT/sampler 实现及 KV cache 契约；
4. normalizer、完整 replay buffer 拟合和 temporal ensemble；
5. DP3/R3D/ManiFlow/ActionFlow 等策略的算法差异；
6. 当前 canonical 使用的 DINO、R3M、ResNet、Uni3D 和 PointNext patch tokenizer；
7. evaluation 的环境 seed、逐 episode 结果和正式 deployment qualification 证据。

这些机制有调用、实验或安全边界证据。即使代码看起来重复，也不能仅以 LOC 为由删除。

## 8. 推荐执行顺序与安全门

### Phase 1：纯死代码与导入面（低风险）

1. 删除 Top-K tracker/config，保留 checkpoint schema 宽松读取；
2. 迁移最小必要 self-test，删除内联 example；
3. 删除 Trainer validation/evaluation 残留和无效 val config；
4. 改 eager barrel import；
5. owner 确认后删除 5 个零调用 helper；
6. 删除 `per_task` 伪入口或正式实现，不能继续半支持。

门禁：`compileall`、逐策略 import、checkpoint latest/resume 单测、现有 83 个 test method（允许环境
相关 skip）；GPU smoke
按受影响策略选择执行，不启动完整训练。

### Phase 2：dormant feature 与 checkpoint 兼容（中风险）

1. modality dropout；
2. ActionFlow step conditioning + checkpoint 转换；
3. MoE boost/override（若 MoE 保留）；
4. 非 teacher 策略不 compile EMA。

门禁：旧 checkpoint strict restore、ActionFlow action/action_ee parity、MoE build smoke、resolved config
维度核验。

### Phase 3：评测/配置/脚本收敛（中高风险）

1. 固定 seed + 固定 policy RNG 的 selector；
2. 单一 eval engine 和 JSON protocol；
3. 单一小 base config；
4. shell/docs 合并和断链修复。

门禁：同 checkpoint 同 seeds 的新旧结果对照；故障注入必须中止而不是产出 0%；best/latest/deploy
selector 不互相 fallback；README 命令全部可解析。

### Phase 4：条件归档（研究决策）

由 owner 明确未来 1–2 个实验周期的 active set，再依次决定 MoE、MultiTask、DDP、RGB-D、aux-EE、
预留组件和完整策略。每次只归档一个相对独立子系统，记录最后可恢复 commit；不要一批删除后再
用跨模块故障猜根因。

## 9. Fact-check 边界

- 本文核验的是当前 GitHub 仓库内部调用图；无法证明私有 notebook、未提交脚本或其他仓库没有
  导入 public helper。
- 未启动完整训练、DDP、仿真或视频评测，符合 AGENTS.md 的长任务限制；本轮结论依赖静态调用、
  配置解析关系、现有测试/实验记录和部署资格证据。
- 粗略 LOC 是维护面，不是保证的 diff 规模；统一引擎、base config 和 schema migration 会新增少量
  替代代码。
- 删除前应再次确认 `main` 没有更新；一旦源码基线变化，至少重跑调用搜索、canonical 配置矩阵和
  checkpoint/deployment contract 核验。
