# Codex CLI 任务书：六项论文实验修正

状态：六项本地实现与定向验证已记录于 [论文 recipe](docs/paper_recipes.md)。GPU/闭环及真实多任务清单等未完成验收见该记录。下文保留原审查依据和验收要求，不作为全部验收通过的声明。

仓库：`haoyangzhanglab/dexmani_policy`。设计日期：2026-10-06。

设计基线：[`b0537e08f2bb7b655e7326762aae37f053f8c38b`](https://github.com/haoyangzhanglab/dexmani_policy/commit/b0537e08f2bb7b655e7326762aae37f053f8c38b)。该提交只更新 RGB transport 实验文档；六项涉及的实现与其父提交 `3e74a99426795071bcad2c2f5f8743c8618a5188` 相同。

二次方案审查（2026-10-06）：复核任务书提交 `744f14f5f5ed5d81d291ca63e415694e9fc2fffe`，当时仓库 HEAD 仍为该提交，核心实现未变化。已对照 SAT 官方当前提交 `cd7c0a8877d6090a9a85ebee0ceca961830b3654` 的有效配置、tokenizer、policy 与 backbone；本次只优化任务书。

本轮审查通过的调整：T1 保留一个显式精度字段及局部旧配置标准化；T2 以不可变选择记录作为后续协议来源、补齐首次使用的 seed 池身份检查；T3 保持零分与技术失败的边界；T4/T5 明确官方 SAT 与本地适配的差异；T6 保持单函数修正。这里的“通过”表示方案可进入实施，不表示代码、GPU 或闭环验收已完成。

## 给 Codex CLI 的执行指令

先读根目录 [AGENTS.md](AGENTS.md)，再按本文完成 T1—T6。目标是让 RA-L 论文实验的数值配方、比较协议和代码行为可解释、可复现。实现应小而完整，直接复用已有模块；不建设通用 benchmark 或生产框架。执行 T5 前必须读取下文固定版本的 SAT 官方源码并完成对应关系核验；只读本地注释或本任务书摘要不算完成官方对照。

开始时执行 `git status --short`、`git log -5 --oneline`，核对最新 HEAD 与本基线的相关差异。若某项已修复，用实际调用路径与定向检查确认后标记完成；不要机械重复补丁。保留用户已有修改和实验产物。本文限定本轮六项范围，不要求重跑 [CODEX_REVIEW_REMEDIATION_TASK.md](CODEX_REVIEW_REMEDIATION_TASK.md) 中的全部历史任务，也不覆盖或删除旧任务书。

逐项修改、逐项验证，最后完成一次交叉检查和简短实施记录。不要停在分析或建议阶段。无需为常规实现细节请求确认；缺少 GPU、数据、权重或仿真依赖时，继续完成可做部分，并如实标记受阻验收。实施与定向检查属于本任务；完整训练、长期闭环评测和在途实验变更不属于本任务。实现阶段默认保留可审查的本地变更，不自动推送代码。

## 优先级与取舍

| 任务 | 优先级 | 已核实的问题 | 本轮决定 | 预期收益 / 风险 |
| --- | --- | --- | --- | --- |
| T1 | P1 | 默认 DP 将 LoRA 参数转回 BF16，AdamW 动量和 EMA shadow 也受该存储精度影响 | 新 DP 配方显式使用 FP32 LoRA；冻结主干继续 BF16；保存并校验精度语义 | 避免小更新被舍入；涉及历史恢复，必须做兼容验证 |
| T2 | P1 | 默认七个基础配置未启用已有 seed manifest；legacy test 池受训练 seed 和是否加赛影响 | 新 selection 必须使用预先固定的 selection / tie-break / test 清单 | 比较对象使用同一测试集；主要改配置和评测入口 |
| T3 | P1 | 所有候选的正常 episode 均失败时，selector 主动报错，中止最终测试 | 按现有确定性排名发布合法全零结果，继续固定 test | 保留失败方法的完整论文结果；局部流程修改 |
| T4 | P1 | 方法名称不足以表达本地编码器、动作、batch、NFE 等配方 | 发布七种方法的实际 recipe 表，并区分当前默认与历史实验 | 降低论文比较和复现歧义；文档为主 |
| T5 | P2 | SAT 开启 patch attention、关闭 attention 内 global 前缀时，计算结果没有返回 | 让无前缀分支返回 attention 后的 patch token | 恢复消融的有效梯度；默认 SAT 分支不变，改动很小 |
| T6 | P2 | VQ 的 `--policy-config` 用配置文件父目录作为 Hydra 根；DDP overlay 会递归包含自身 | 仓库配置统一从 `dexmani_policy/configs` 组合相对配置名 | 打通 DQ-RISE 的 DDP 配置准备流程；不改 VQ 训练算法 |

T5、T6 是剩余问题中选出的两项：都有可直接复现的代码反例，改动面明确，不需要先跑大规模实验。暂不扩展到 Manager IPC、EMA 性能重写、日志同步、SDPA/dropout、SAT 时间融合或训练器合并；这些问题需要独立 profiling 或研究变量判断。

## 必须保持的实验边界

1. [最新 RGB A/B 记录](docs/rgb_transport_closed_loop_ab.md)记录了一组尚未完成质量验收的实验：policy 固定在 `3e74a99426795071bcad2c2f5f8743c8618a5188`，simulator 固定在 `c3d56fd58f402fc6f67a635b9452e10eee935afe`，配对训练 seed 为 42/43/44，每次 100000 optimizer updates，唯一配方差异是 `rgb_keep_uint8`。这是文档记录的状态；不能把它当作本次已查询的实时进程状态。
2. 不停止、改写或重启该队列，不修改其 detached worktree、保存的 config、manifest、checkpoint、协议或结果。不要占用该队列正在使用的 GPU 做本任务验收。T1 的新精度配方是另一项实验，不能混入其任何一臂。若旧提交的全零策略阻塞旧队列，先记录事实，另行安排有记录的补充评测，不能静默改协议。
3. 该 A/B 文档已记录实际 200-seed 池及固定 25/5/100 分区；T2 修复的是仓库通用默认入口，不重新生成该实验的清单。默认 fallback 池只有 100 个 seed，但不能据此推断外部 seed 文件或所有历史实验的池大小。
4. 不更改动作布局、归一化、数据拆分、增强、LoRA rank/targets、损失、optimizer update 计数、EMA 时钟、训练预算、NFE、候选 milestone 或 SAT 时间融合。只有本文明确指定的研究行为发生变化。
5. 历史实验以其保存的 resolved config 和 checkpoint 为准；不要用新默认值改写历史解释。没有真实训练/闭环证据时，不宣称成功率提高、质量等价或加速比例。

## T1 — 为新 DP 显式固定 LoRA 存储精度

**事实与范围。** [DINO 构造器](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0537e08f2bb7b655e7326762aae37f053f8c38b/dexmani_policy/agents/obs_encoder/rgb/dino.py#L25-L29)以 BF16 加载主干；[共享 LoRA 分支](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0537e08f2bb7b655e7326762aae37f053f8c38b/dexmani_policy/agents/obs_encoder/rgb/base.py#L71-L90)把 adapter 再转成主干 dtype。前轮真实小型 HF DINO + PEFT 检查确认 adapter 和 Adam 动量为 BF16；小更新及晚期 EMA 的标量反例证明舍入风险。这不是“实际 DP 已全部停止学习”或“所有策略都有此问题”的证据。

**实施位置。** `dexmani_policy/agents/obs_encoder/rgb/{base,dino,clip,siglip}.py`、`dexmani_policy/configs/dp.yaml`、`dexmani_policy/training/resume.py`；按需在现有 build/trainer 路径添加一次性 dtype 记录。检查 `training/ema_model.py`、`agents/loader.py` 的真实消费者，不重写这两个子系统。

**方案。**

1. 在三个 ViT 构造器与共享 `set_tune_mode` 路径增加一个明确字段 `lora_dtype`，仅接受 `backbone`、`float32`。**构造器缺省值为 `backbone`**，保持旧 saved config 缺字段时的构造语义。`dp.yaml` 明确配置 `agent.rgb_backbone_config.lora_dtype: float32`，`ddp/dp` 继承它。
2. `float32`：保留或显式保证 PEFT adapter 为 FP32，冻结主干保持现有 BF16。`backbone`：保留历史 adapter dtype。主干 dtype 从冻结主干确定，不依赖 wrapper 第一个参数恰好是什么。字段转换发生在 optimizer/EMA 构造之前，不增加训练步内转换。保持 projection、state encoder、decoder 的原有精度，不调用整模型 `.float()` 或 `.bfloat16()`。
3. 沿现有构造链确认：新 adapter 参数、AdamW 的 `exp_avg`/`exp_avg_sq`、EMA 对应 shadow 都为 FP32。AdamW 状态在首次 update 后才建立，不能用空 state 宣称通过。复用当前 EMA loop/foreach 算法，不添加 master-weight optimizer 或新的 EMA 实现。
4. 精度字段必须进入保存的 config 和现有 resume contract。推理继续从 checkpoint 所属实验配置重建。给 `validate_resume_contract` 增加局部、已知字段的标准化：适用的 ViT+LoRA 旧配置中“缺失”与显式 `backbone` 等价；与 `float32` 不等价。不要给无关 encoder 注入该字段或放宽其他合同比较。
5. 旧 BF16 实验用旧 saved config 推理、用显式 `backbone` 目标配置 strict resume；新 FP32 实验可同配方 strict resume。跨两种精度 strict resume 必须在模型、optimizer、EMA 状态被加载/改变前失败，并给出字段差异。旧 checkpoint 转 FP32 不能补回过去丢失的更新，不提供自动迁移。
6. 在已有启动记录中一次输出参数/adapter/冻结主干/EMA 的 dtype 与 autocast 配置；首次 optimizer step 后一次记录动量 dtype。避免新增每步 `.item()`/CPU 同步。`training.use_bfloat16` 仍控制现有 autocast，不被当成参数存储精度开关。

`freeze` / `full` 的历史语义本轮不变；本文只验收默认 DP 的 LoRA 配方，不据此宣称 BF16 full tuning 的存储精度风险已解决。

**验收。**

- 使用离线 tiny HF DINO 和真实 PEFT，调用仓库构造器。可替换网络下载/权重获取，不可把 adapter/optimizer/EMA 替换成会自行满足断言的假实现。检查新旧两种配方、无字段旧配置、非法字段值，以及 CLIP/SigLIP 的字段传递。
- 对新配方执行数次真实 forward/backward/update，检查 loss 有限、adapter 梯度连通、FP32 参数与动量 dtype、冻结主干未更新。LoRA 初始化可能使部分参数第一步梯度为零，不把它误判成断图。
- 复现小更新和晚期 EMA 的定向数值检查；loop 与 foreach 均验证 FP32 shadow 能累积更新。该检查只证明数值性质，不代替任务质量实验。
- 使用现有真实保存/恢复入口检查 raw/EMA 推理重建、同配方连续训练与恢复的一致性、旧缺字段与显式 `backbone` 的兼容性，以及跨精度拒绝且目标状态未变。
- 仅在有空闲目标 GPU、依赖和缓存权重时做有限步 BF16 autocast/现有 compile 的 DP smoke；否则标记 `NOT VERIFIED`。需要进入论文的新 DP 质量结果应另行安排重训或配对实验，本任务不启动。

## T2 — 让固定评测清单成为新论文实验的默认入口

**事实与范围。** [legacy 测试集逻辑](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0537e08f2bb7b655e7326762aae37f053f8c38b/dexmani_policy/eval_best_ckpt.py#L152-L180)会排除实际选点用过的 seed，已做到这部分互斥；问题是 test 身份随 training seed 和加赛变化。100-seed 池的 75/70 反例只适用于该池。[现有 manifest 验证与固定 test 路径](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0537e08f2bb7b655e7326762aae37f053f8c38b/dexmani_policy/evaluation/protocol.py#L497-L546)已经具备任务映射、互斥和 hash 校验，应直接使用。

**实施位置。** 七个基础 YAML 的 `eval` 段；`evaluation/protocol.py`、`select_best_ckpt.py`、`eval_best_ckpt.py`、`record_demo.py` 的小范围入口调整；检查 `scripts/eval/eval_pipeline.sh` 的既有 handoff，原则上保留其参数接口；新增小脚本 `scripts/eval/make_seed_manifest.py` 与受版本管理的 `dexmani_policy/configs/eval_protocols/`。保留当前 `MultiTaskSimRunner` 的配对映射，不重做任意任务 seed 调度。

**方案。**

1. 七个基础配置加入 `eval.seed_manifest: dexmani_policy/configs/eval_protocols/${task_name}.json`，六个 DDP overlay 继承。多任务使用该配置实际 task-set 名称；JSON 内逐任务列出 physical seeds。从仓库根运行；必要的路径解析复用现有项目根机制。不要把唯一清单放在默认忽略的 `data/*` 或 `experiments/` 下，也不要为此放开整个数据目录的忽略规则。
2. 小脚本从仓库配置根组合现有配置，只实例化轻量 runner 并读取实际 seed 池；不实例化 dataset/model，不调用 `make_env`、reset、step，也不依赖已有实验目录或 `cfg._exp_dir`。可直接组合 `cfg.env_runner`，不要调用要求 saved config 的 `build_eval_runner`。用已有 `get_seed_list` / `mapped_task_seeds` 的配对关系，以独立的 `partition_seed=1066` 一次性划分参考池。默认请求 selection=25、tie-break=5、test=100；映射到各任务后调用 `load_seed_manifest` 复验，再用独占创建方式写文件。已存在的文件报错，不能覆盖或每次训练自动重抽。
3. JSON 复用已有 `pool_id`、`selection`、`tie_break`、`test` 格式；只补充划分 seed、实际池来源、已知 simulator revision 和 `runner_pool_sha256`。池 hash 使用现有 `load_seed_manifest` 对完整配对 mapping 的相同算法：**首次 selection 也必须检查生成时声明的 hash 与当前池一致**，不能仅把 hash 当注释保存；旧 manifest 没有该可选字段时保留既有校验，不伪造缺失来源。禁止把完整文件路径/临时运行时间等易变值混入池身份。生成时显示实际池大小与三组计数，保留 seed 顺序。将 OmegaConf 容器在读取边界转成普通容器，再执行原有严格校验，覆盖文件路径、dict、DictConfig 三种输入；不创建新 schema 框架，不更改历史 manifest 的 canonical hash 算法。
4. 25+5+100 要求至少 130 个可配对 seed。池不足时必须失败；不把 test 静默改成 70/75，也不虚构额外 seed。若研究决定采用更小固定 test，须事先通过脚本显式参数生成另一份共享清单并在 recipe 中披露。无法读取实际池时，只完成实现与合成 fixture 检查，把真实清单发布标为 `NOT VERIFIED`；不得提交伪装成真实协议的样例 JSON。
5. 新 `select_best_checkpoint` 必须有有效 manifest；缺失、找不到文件、任务不匹配或不合法时给出生成/传入方法并失败。清单角色决定实际评测 seed，不再在此分支按 `training.seed + 1024` 重抽或截断。保留已实现的 episode policy seed 语义。训练/config-only 不要求清单文件已经存在；评测阶段在加载模型和 rollout 前完成协议验证。当前 `_setup_eval` 把 runner 构造和模型加载放在一起，需要局部调整顺序：只构造一次 runner，先验证完整池，再加载一次模型；sweep 仍只加载一次模型。不要为提前报错另造第二个 runner 或修改已注入子集后的池。
6. test 始终是完整、预留的 `test` role；没有使用的 tie-break seeds 也不能回流 test。manifest 是角色和计数的事实来源，现有请求值与实际生效值分别记录；`episodes` 不截断正式 test，如与其数量不同应明确提示实际数量。`max_episodes` 既有语义是 selection 的硬上限：若它小于 selection 加预留 tie-break 的总长度，应在运行前报错，不能忽略上限或截断角色。不同 NFE 若做独立研究使用同一 test；不能凭 test 结果挑论文主表的 NFE。同一可比任务集合共用一份清单；若单/多任务结果要对同一任务做直接比较，还需核对该任务的实际角色 seed 一致。现有配对映射表达不了时应披露不可直接配对，不暗中重抽或在本任务重写 runner。
7. CLI 读的是实验目录保存的 `config.yaml`，改基础 YAML 不会自动更新历史实验。历史 checkpoint 进入新论文协议时，通过现有 dot-list `eval.seed_manifest=<path>` **重新选点**；后续使用该次不可变 `selection_result.json` / `--selection-record`。不能给旧选择记录补贴新 manifest 后直接声称协议一致。
8. 保持既有 immutable handoff，并统一协议来源：**选点时读指定 manifest；之后 pinned eval/demo 从本次 selection record 的内嵌 manifest 取完整协议**，不再依赖 saved config 里的外部路径仍然存在或未被改写。保持 shell `SEED_MANIFEST` 只传 selector、后续传 `--selection-record` 的接口，避免给三个阶段重复传一份可变路径。`record_demo.py` 当前不接受 dot-list，不向它追加这类参数。复用协议模块中的一个小的读取/绑定函数即可，不增加配置管理层。
   - CLI 显式 `eval.seed_manifest=...` 是额外的一致性声明：有 pinned record 时必须与其内嵌 manifest canonical hash 相同，否则在 rollout 前报错；显式 null 不能解除该合同。不要把 saved config 的默认路径误认为本次显式 override。入口保留 override 来源，先检查再绑定内嵌内容；Python 直接调用复用已有 override 结构，或补充一个局部可选的显式 override 实参，普通 cfg 路径只作为无 pinned 协议时的默认来源。不因 `_resolve_final_eval_request` 被再次调用而丢掉检查，不为此增加 YAML 模式开关。
   - runner 构造后复验记录的 manifest hash 与 runner pool hash，再使用该有效协议写 eval/demo snapshot；不修改磁盘上的 saved config。仅当没有 pinned 显式协议时，才使用调用配置的 manifest/原有 legacy 分支。旧选择缺协议时，不允许通过新 override 直接冒充同协议选点。
   - demo 保持原有可视化 seed 选择，不把 test role 当成 demo 采样指令，也不将 demo 结果计入 held-out 指标。记录绑定发生在 runner 被 demo seed 子集覆盖之前。
9. 历史无 manifest 的已发布记录仍可沿现有 legacy 读取/评测路径复现，并明确标识 legacy；它们不自动升级成固定协议结果。新 selector 不增加默认静默回退。保留现有多任务 seed 映射所需函数，不借机删除旧映射层。

**新脚本的约定接口（实现后才能运行）。**

```bash
python scripts/eval/make_seed_manifest.py \
  --config-name dp \
  --output dexmani_policy/configs/eval_protocols/pick_apple_messy.json \
  --partition-seed 1066 --selection 25 --tie-break 5 --test 100
```

脚本需支持现有 Hydra dot-list 覆盖任务配置；先读取真正的 runner 池，不假定 `range(200)`。此命令只生成协议，不启动评测；不用于覆盖正在进行的 A/B manifest。

**验收。**

- 用真实协议函数和轻量 runner fixture 测试单任务及各任务 seed 数值不同的多任务映射；三组 `(task, seed)` 两两互斥，角色内无重复、顺序稳定，完整池 hash 可核验。
- 两个模型、训练 seed 42/43、有加赛/无加赛的组合，最终 test 的身份与顺序完全一致；无加赛时五个预留 seed 仍不进入 test。模型输出可用最小 fixture，这不能标作闭环验证。
- 空/缺失/重复/越界/跨角色重叠、任务不符、首次使用的生成池 hash 不符、选点后 pool 改变、显式不同 manifest/显式 null、已有输出文件、池不足均明确失败，不能回退或截断继续。生成时 200 个 seed、首次选点时改为 201 个但所有已选 seed 仍存在的 fixture 也必须检测到池身份变化。
- 覆盖 Python/CLI、shell handoff、单次 eval、NFE sweep 与 demo snapshot；选点完成后移动或修改原始 manifest 路径，**未传显式 override 的 pinned eval/demo 仍使用内嵌原协议**；显式传入已修改路径则失败。更换 `best_ckpt.json` 也不能影响已固定的 handoff。断言协议错误时模型加载/rollout 调用数为零；合法 sweep 的 runner 构造与模型加载各一次。对 shell 做参数捕获，对 demo 做配置/snapshot 检查，不启动 simulator/demo。
- config-only 13 个配置可在没有实际 manifest 文件时组合。历史无 manifest 的已发布记录可读；新 selector 缺 manifest 失败。修改既有 selector 测试 fixture 以满足新协议，保留原本的失败保真和 immutable handoff 断言。

## T3 — 将合法全零 selection 与技术失败分开

**事实与范围。** [当前 selector](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0537e08f2bb7b655e7326762aae37f053f8c38b/dexmani_policy/select_best_ckpt.py#L302-L323)先记录所有候选，再因 `best.success_count == 0` 抛错；旧 best 和结果证据仍保留。问题是合法的 0% 研究结果无法进入最终评测，不是结果被静默删除。

**实施位置。** `select_best_ckpt.py` 的 dispatch、最终排名和发布分支；现有评测测试。通常不需要改 agent、runner、checkpoint 格式或 shell 的退出策略。

**方案。**

1. 保留两阶段候选比较与现有 `_rank_key`：成功率优先，再比较成功步数，最后较大 `global_step`。所有有效候选全零时，该规则自然选择最后一个候选 milestone；不另加随机选择或“表现不佳则用旧 best”。
2. 删除仅因全零而抛出的 `RuntimeError`。在共同的 `selection` 数据中记录 `selection_all_zero: true/false`，以**所有有效候选实际结果**计算；随 summary、immutable selection record 和 `best_ckpt.json` 一起保存。合法全零发布 `status=success`；这里表示流程成功，不表示策略成功率大于零。
3. 合法零分要求有完整且正常完成的 episode。复用 `failed_tasks` 和现有异常传播；在 dispatch 对返回 details 做一个局部一致性检查：其 `(task_name, seed)` 多重集恰好等于请求 mapping，没有缺失、重复或错误 episode。单任务补齐当前任务名。不能把空结果、缺失字段、异常 episode 或部分任务失败当成全零成功。不要把正常 timeout/未完成任务误判为基础设施故障。
4. 新合法选择可以按现有规则更新可变 `best_ckpt.json` 指针；历史 immutable record 不变。后续 test 通过本次 handoff 加载本次选择，不因为出现全零而退回之前的 best。真实技术异常仍以非零状态退出，保存失败证据，不发布新的可用 handoff，也不改变旧成功指针。
5. 添加的标记对旧记录是可选的，不把缺失字段解释为已证明非全零，不做历史产物批量迁移。

**验收。**

- 单任务 fixture 的五个 milestone、每个 25 个正常失败 episode，并发生 5 个 seed 的全零加赛：选中最大 `global_step`，完整保留 30 条/候选的详情，summary 成功，所有发布记录一致标记全零。多任务按实际 `(task, seed)` 单元计数，不能只比较 reference seed 的数量。
- 接着调用真实 eval 控制流程与轻量 runner fixture，通过 immutable handoff 完成完整固定 test；0/n 是有效结果，episode 数与 T2 manifest 一致。
- 覆盖无 tie pool 的合法协议、非全零排名、加赛后出现成功、重复/缺失/空详情和注入异常。技术失败保留旧 pointer；合法全零更新 pointer。同步调整 `tests/test_infra_evaluation.py` 中原先要求全零抛错的断言，不能删除错误路径测试。
- 保持旧选择记录与新选择并存时的固定 handoff 测试。此修复一般需要重新选点/评测，不因 selector 流程变化要求重新训练。

## T4 — 发布与代码一致的论文 recipe 表

**事实与范围。** 本地方法具有合理的任务适配，但方法名不能证明完全复现上游。例如 ManiFlow 使用本地 dense PointNet/XYZ PE/多帧编码；DQ-RISE 有本地动作布局与 VQ/PCA 约定；SAT 按 patch 索引融合时间特征。描述这些差异不等于已证明对应算法无效。

**实施位置。** 新建 `docs/paper_recipes.md`；README 添加短入口链接；长表放 docs。必要时在该文档内加本轮实施记录，不新增结果数据库、通用配置导出系统或完整论文写作模板。

**方案。**

1. 覆盖 `dp`、`dp3`、`dqrise`、`maniflow`、`sat`、`r3d`、`multitask_dit` 七种方法及六个 DDP overlay；组合有效配置并沿 agent/data/eval 调用者核验。不能只抄 YAML 注释、类名或 fallback 参数。
2. 以两张紧凑表区分“方法机制/本地改动”和“训练/评测配方”，必要字段如下：观测与 encoder、动作 key/维度/布局、H/obs horizon/action chunk、预测目标/损失、归一化与增强、预训练来源与 tune mode、LoRA/主干/optimizer/EMA 存储精度和 autocast、per-rank batch/world size/accumulation/global batch、optimizer updates/候选步数、EMA、实际评测 NFE、manifest 身份/计数、数据拆分及代码版本。
3. 对每个方法写“保留的机制 / 本地适配 / 未核实的上游对应”，给出本地代码/config 链接。能够确认上游仓库与 revision 才填写；找不到来源或准确提交就写 `NOT VERIFIED`，不编造，也不把 UniDex、ManiFlow_Policy、DeCAL 三个风格参考都当成所有方法的直接来源。涉及上游具体机制时读取官方代码/论文并固定来源。SAT 必须使用 T5 的官方对照，披露 global token 来源、state token 融合、patch shuffle、EJC 编码以及本地无前缀消融的差异；“参考官方”不等于在本任务中恢复整套官方结构。
4. 当前默认表与历史论文实验表分开。历史表只能来自用户提供或当前环境已有的 saved config/产物；没有证据就不填“实际运行”。T1 合入后的新 DP FP32 adapter 配方，不能覆盖旧 BF16 A/B 的记载；RGB A/B 质量结论继续保持文档中的未完成状态。
5. 主表采用事先固定的 NFE/EMA/协议；若已有扫 NFE，完整披露用途，不能从 test 最高值反选主结果。多训练 seed 的波动与单次 episode 的 Wilson 区间分别解释，不能互相替代。此项只发布事实，不顺手统一各方法容量、学习率、归一化或 batch。

**必须核对的基线锚点（指本任务书基线，不是未来或历史所有实验）。**

| 项目 | 已核实的当前行为 | 容易写错之处 |
| --- | --- | --- |
| DP | DINOv2-small + LoRA；joint action 19；单卡 batch 64，DDP 4×48=192 | BF16 autocast 不等于新 FP32 adapter 存储；单卡/DDP global batch 不相等 |
| DQ-RISE | 默认 `action_ee` 21 = tcp 9 + hand 12；`prediction_type=epsilon`；eval NFE 20 | 不能按其他方法写成 joint 19 或统一 NFE 10 |
| ManiFlow | `eval.inference_steps=4`；训练离散网格 10，agent inference fallback 10 | 实际 eval 优先级使 4 生效；训练网格不是评测 NFE |
| SAT | 默认 attention 开启且有内部 global 前缀；T5 只修另一条分支 | 不能把 T5 推广成默认 SAT 全部失效；时间对应风险仍需独立消融 |
| R3D | 核对 XYZRGB/Uni3D/OneWay 路径及可选辅助 EEF 预测与控制输出的区别 | 不能把辅助预测维度直接当成实际控制维度 |
| Multi-task DiT | scratch ResNet18 + GN + full；冻结 CLIP text 与缓存；DDP 4×16=64 | 不能按 RGB backbone 的通用预训练名推断实际 recipe |
| 数据 | 单任务和多任务的 seed、episode 选择分别从 resolved config / dataset 核验 | 同为 80 条上限不保证不同 training seed 使用相同 episode；A/B 只保证配对内一致 |

**验收。**

- 13 个配置组合与代码消费者逐项对表；global batch 按 `per_rank × world_size × accumulation` 计算，训练时钟明确为 optimizer update。
- 默认值、用户 override、saved config 历史值分栏或明确标注；读不到的实验值不填推测值。检查 README/docs 的路径和链接。
- 文档提供最短使用顺序：准备数据/权重 → 一次生成固定协议 → 训练 → 固定 selection/test → 解读结果；示例使用仓库真实 CLI。T4 不需要 GPU 测试，不以重训作为文档完成的前提。

## T5 — 对照 SAT 官方实现，修复本地无前缀 attention 分支

**事实与范围。** [MultiScalePatchTokenizer.forward](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0537e08f2bb7b655e7326762aae37f053f8c38b/dexmani_policy/agents/obs_encoder/pointcloud/pointnext_tokenizer.py#L135-L154)计算 `x = self.patch_transformer(x)` 后，仅在 `prepend_global_in_attn=True` 时使用它；False 分支返回旧 `patch_token`。前轮小型真实 SAT 三步训练确认相关 attention/position 参数未获梯度。受影响条件是 `use_patch_self_attn=True, prepend_global_in_attn=False`；与外层 `include_global_token` 不是同一开关。

**必须读取的官方来源。** 正确项目是 [XiaohanLei/SAT](https://github.com/XiaohanLei/SAT)，论文为 Structural Action Transformer for 3D Dexterous Manipulation。固定 revision 为 [`cd7c0a8877d6090a9a85ebee0ceca961830b3654`](https://github.com/XiaohanLei/SAT/tree/cd7c0a8877d6090a9a85ebee0ceca961830b3654)。[官方 README](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/README.md)声明 Early Access 且可运行性尚未验证，因此只以已读源码确认机制，不把它当作已验证的完整训练基准。

执行者需只读获取并检查以下固定版本文件，无需安装官方仿真环境、下载权重或启动训练。不能用同名的 Spatial-Aware Token 等仓库替代；官方更新了 main 也不要静默换 revision。

- [sim/sat/config/sat.yaml](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/config/sat.yaml)：`policy` 的有效选择。
- [sim/sat/model/vision/obs_tokenizer.py](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/vision/obs_tokenizer.py#L370-L511)：`Obs_Tokenizer` 的构造分支、`FPSPointNetEncoderXYZ.forward` 与 `StateAttn`。
- [sim/sat/policy/sat.py](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/policy/sat.py#L476-L495)：构造、预测与训练如何使用 tokenizer 返回值；训练对应 L618—L635。
- [sim/sat/model/diffusion/sat.py](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/diffusion/sat.py#L188-L259)：动作转置、joint 描述与 shuffle/inverse 的消费者。

**二次审查已确认的对应关系。**

| 机制 | 官方该 revision 的有效代码 | 本地行为与本轮决定 |
| --- | --- | --- |
| 感知入口 | 默认 `pointnet_type=pointnet` 且不使用颜色，经 `Obs_Tokenizer` 进入 `FPSPointNetEncoderXYZ` | 本地为多尺度 PointNext tokenizer；保持该任务适配 |
| global token | `global_pn` 对输入点做 MLP 后 max-pool，产生随观测变化的 token，并在 attention 前加入 | 本地默认前缀为 learned parameter；修正本地“等同 global_pn”的误导性注释，保留计算行为 |
| attention 输出 | patch 加中心位置特征、shuffle 后与 global token 一起经过 Transformer 和 final projection；返回的是处理后的输出 | 官方支持“使用 attention 后输出”这一语义；本地丢弃输出仍是明确错误，修复局限于此 |
| 无 global 的含义 | 发布代码没有本地 `prepend_global_in_attn` 开关；policy 只有已注释的删除 tokenizer 输出首 token 的语句 | 该注释不证明消融已运行；删输出 token 与删 attention 输入前缀不同。本地 False 分支标为本地消融，不称官方无 global 复现 |
| 时间与 state | point/state 分别沿特征维合并时间，再沿 token 维拼接；官方 patch 编码还主动 shuffle | 本地将 state 特征广播到每个 point token，再合并时间；两者都不能凭槽位索引保证物理跨帧对应。本轮不改变融合或加入匹配机制 |
| 关节身份 | backbone 拼接 robot/joint 两个 embedding，并与轨迹表示拼接 | 本地 EJC 为三字段投影求和；在 recipe 中披露，不借 T5 改结构、参数维度或 checkpoint |

**最小补丁。** 保持本地已经暴露的消融接口：在有 attention、无内部前缀的分支把 `patch_token` 赋为 attention 后的 `x`，返回原有二元 tuple。有前缀分支仍返回 patch/center/attn_global 三元 tuple；attention 关闭时仍返回原 patch/center。外层在 `include_global_token=True` 时继续从修正后的 patch token 走原有聚合，输出供 `SATObsEncoder` 消费的三元 tuple。不要删除外层 global token，也不要通过关闭 attention、删开关或绕过该分支让测试通过。

同步修正直接相关的来源注释，并在 T4 表中记录以上差异。官方的 data-dependent global token、patch shuffle、StateAttn 和两字段 EJC 不在本轮迁移范围；不能以“对齐官方”为由改变默认 SAT、FPS/patch 顺序、EJC、时间融合或训练配方。本地缺陷的验收依据是本地配置承诺、返回值消费者和梯度连通，不要求两种不同结构输出数值一致。

**验收。**

- 三种分支分别覆盖 tuple 长度、shape、中心点顺序和语义：关闭 attention；开启且有前缀；开启且无前缀。
- 官方来源的 SHA、上述有效调用链与差异表已写入 recipe/实施记录。若执行环境拿不到官方源码，记录 T5 的官方复核为 `NOT VERIFIED` 并继续其他任务，不能把摘要阅读标成官方代码复核完成。
- 使用小型真实 Torch Transformer，在 dropout=0、固定 FPS 的可重复输入上，证明无前缀输出依赖 attention 与 center-position 参数，反向梯度连通且数步 optimizer update 后有可测参数更新。不能只用常量替身或检查 shape，也不要求每个参数元素每一步梯度都非零；避免用对称损失或零初始化造成的暂时零梯度作错误判据。
- 经实际 `PointNextPatchTokenizer` / SAT 外层调用验证无前缀配合 `include_global_token=True` 的接口仍成立；默认分支在同输入/状态/RNG 下保持原有输出行为。环境缺几何依赖时如实注明 fixture/reference 算子的边界，不称其为 GPU 闭环测试。
- 受影响的无前缀消融若已训练，修复后的结果需要重训；默认分支已有实验不因本补丁被自动判作无效。此处不启动重训。

## T6 — 修复 VQ policy-config 的 Hydra 搜索根

**事实与范围。** [`load_policy_config`](https://github.com/haoyangzhanglab/dexmani_policy/blob/b0537e08f2bb7b655e7326762aae37f053f8c38b/scripts/training/train_vq_hand.py#L198-L204)使用 `path.parent` + `path.stem`。对 `configs/ddp/dqrise.yaml`，[`defaults: /dqrise`](dexmani_policy/configs/ddp/dqrise.yaml)因此再次指向同一 overlay，真实函数调用出现 `RecursionError`。普通 13 配置 config-only 通过不能覆盖这个独立入口；这也不等于 `train_ddp.py` 本身无法组合配置。

**最小方案。**

1. 在原函数中以现有 `_project_root / 'dexmani_policy' / 'configs'` 为仓库配置根。对根内文件计算不含后缀的 POSIX 相对名：基础配置为 `dqrise`，overlay 为 `ddp/dqrise`；用该根与该名字调用 Hydra compose。
2. 先展开/resolve 路径并检查文件存在。对仓库外的独立 YAML，保留当前“父目录作 root、stem 作 name”的已有行为，并说明其绝对 defaults 相对于该外部 root。不要捕获递归错误后改用基础配置，不手工合并/扁平化 YAML。
3. 保留 resolver 注册、覆盖参数的顺序与语义，使用局部 Hydra 初始化上下文；重复调用不得泄漏全局状态。不要为了通过此函数清除宿主已有的 Hydra 全局配置，也不要重构整个训练启动器。
4. VQ 仍通过现有 `build_policy_dataset` 复用策略的真实 dataset、有效窗口和 normalization；不改变 VQ/PCA/codebook、split、batch 或 DQ-RISE 动作布局。生成的 config 正确不代表 VQ 已训练完成。

**验收。**

- 直接调用真实 `load_policy_config`，覆盖基础 `dqrise.yaml`、`ddp/dqrise.yaml`、absolute/relative 路径以及带 overrides 的情况；不启动 `train_vq_hand.py` 的训练 main。
- 与从正确 repo root 组合的相应配置比较稳定字段：`agent`、`dataset`、`action_key`、normalization、windows、seed 及 dataloader/DDP 值。overlay 应有 `policy_name=ddp/dqrise`、per-rank batch 32、默认 world size 4；不能为了“等同基础配置”抹掉 overlay 本来应有的差异。不比较动态 run_id/Hydra runtime。
- 仓库外临时独立 YAML 与缺失路径各测一次；连续两次调用不串状态。已有 VQ 对齐检查继续通过。无需训练 codebook、加载真实数据或运行 DDP。

## 实施顺序与验证预算

优先级表示论文影响；实际落地顺序建议如下，以减少相互覆盖：

1. 读当前源码，记录工作树状态和六项当前结论；只对尚未成立的前提调整实施细节并说明证据。
2. 完成 T1 及精度/恢复定向检查，独立形成可审查改动。
3. 完成 T2，再完成依赖固定协议的 T3；二者共同验收 selector → immutable record → fixed test。
4. 分别完成 T5、T6 的小补丁和针对性回归。
5. 最后以实际合入行为完成 T4 recipe 文档，避免记录中间默认值。
6. 做一次 13 配置组合及六项交叉检查。若拆 commit，保持每个 commit 可解释；不要把行为修复埋在格式化、重命名或大规模测试整理里。

可直接使用的低成本命令如下。先确认解释器/依赖可用；本任务不要求升级或重建共享环境。

```bash
git diff --check
python dexmani_policy/smoke_test.py --config-only dp
python dexmani_policy/smoke_test.py --config-only ddp/dp
python dexmani_policy/smoke_test.py --config-only dqrise
python dexmani_policy/smoke_test.py --config-only ddp/dqrise
python dexmani_policy/smoke_test.py --config-only sat
```

最终一次配置矩阵：`dp, dp3, dqrise, maniflow, sat, r3d, multitask_dit, ddp/dp, ddp/dqrise, ddp/maniflow, ddp/sat, ddp/r3d, ddp/multitask_dit`。当前 smoke 的 `--config-only` 接受配置名，不接受 Hydra dot-list；带 override 的检查应调用实际 compose/目标函数，不编造 smoke 参数。

定向测试优先扩展 `tests/test_infra_resume.py`、`tests/test_infra_evaluation.py`、`tests/test_review_remediation.py` 或相邻已有测试；必要时新增一个小的专项测试文件。运行新增节点及受影响既有节点，记录真实命令。不要为凑覆盖率运行完整模型/机器人矩阵，也不要用测试替身绕过本任务要验证的真实构造、参数、协议或 Hydra 入口。

供用户**以后主动评测**的已有 CLI 语法示意如下；运行前由用户设置 `EXPERIMENT_NAME`、`SELECTION_RECORD_PATH`、`EVAL_MANIFEST_PATH` 三个变量，本次验收不执行这些昂贵命令：

```bash
python dexmani_policy/select_best_ckpt.py \
  --policy-name dp --task-name pick_apple_messy --exp-name "$EXPERIMENT_NAME" \
  --result-file "$SELECTION_RECORD_PATH" --no-videos \
  "eval.seed_manifest=${EVAL_MANIFEST_PATH}"

python dexmani_policy/eval_best_ckpt.py \
  --policy-name dp --task-name pick_apple_messy --exp-name "$EXPERIMENT_NAME" \
  --selection-record "$SELECTION_RECORD_PATH" --no-videos
```

`eval.seed_manifest=...` 是 selector/eval 的位置 dot-list 参数，不存在 `--overrides` 选项。新协议下 eval 已从 handoff 取得完整清单，无需重复指定路径；若显式指定，仅允许与记录内容一致。结果文件应是尚未存在的路径，父目录已存在。正常执行时 manifest 已决定完整 test；不要通过减小 episode 参数把正式协议变成 smoke。

## 完成定义与交付

- [ ] T1：新 DP 的 FP32 LoRA/optimizer/EMA 存储合同生效；旧 BF16 构造与 strict resume 语义被保留；跨配方恢复拒绝。
- [ ] T2：新选择流程要求真实固定协议并检查生成池身份；未加赛也完整预留 tie seeds；后续阶段依赖 handoff 内嵌清单，显式冲突失败；实际清单与合成 fixture 明确区分。
- [ ] T3：完整正常的全零结果可发布并完成固定 test；真正异常仍失败且保留历史证据。
- [ ] T4：七种方法/六个 DDP 配方有逐项可追溯的实际 recipe，历史实验不被新默认覆盖。
- [ ] T5：已读取固定 SHA 的官方 SAT 并记录本地适配差异；无内部 global 前缀的 patch attention 确实影响返回值和梯度，其他分支保持原行为。
- [ ] T6：VQ 的真实 policy-config 入口能正确组合基础与 DDP 配置，原有外部独立 YAML 行为仍可用。
- [ ] 当前 A/B 队列、旧 saved config/checkpoint、旧 immutable selection record 和原始数据未被改写；没有未经任务要求启动长训练、长期评测或真机动作。

实施后在 `docs/paper_recipes.md` 末尾用一张小表记录 T1—T6 的状态、关键改动位置、实际执行的验证命令/结果以及 `NOT VERIFIED` 项。状态区分“代码完成”“实际 seed 清单待发布”“GPU/闭环未验证”“需后续重训/评测”，不能用一个总 `PASS` 掩盖缺项。最终回复给出改动摘要、验证边界及论文结果需要怎样补充；不要宣称本任务已完成实际质量验收。
