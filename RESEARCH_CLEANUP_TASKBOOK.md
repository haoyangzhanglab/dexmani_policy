# dexmani_policy 个人研究代码清理任务书

审查基线：[`5a2c7d3a1e1bb6a37ac9d7a015204c9566d59137`](https://github.com/haoyangzhanglab/dexmani_policy/tree/5a2c7d3a1e1bb6a37ac9d7a015204c9566d59137)。实施以实际工作区为准，先核对基线之后的相关改动，不回退用户工作。

本文件规定下一轮清理的目标、取舍和验收方式。创建任务书不表示清理已经实施。本轮仅覆盖最新 evidence / validate / contract / artifact / source 审查得出的六项减法；不扩展为全仓库重构。

## 1. 目标与判断依据

项目服务个人机器人研究。删除当前流程不使用的能力、只服务这些能力的状态、一次性公共封装和可明确取消的重复准入检查；保留防止错误实验结果的必要边界。优先减少理解成本和修改路径，不以关键词数量或删除行数作为验收目标。

基线核对结果：156 个 tracked 文件中有 113 个 Python 文件；evidence 无命中，contract 只有两处文字。12 个 `validate_*` 函数合计定义跨度 135 行，具有 22 个生产调用点。`source` 大多表示数据来源、权重路径、坐标系或原始采样行，不构成一套可整体删除的框架。

本轮将保留这 12 个短校验函数的定义，删除部分重复调用及外围机制。不要通过重命名、移动同一套机制、压缩排版或删除有用注释制造“清理完成”。

## 2. 实施顺序与明确取舍

下列 P1/P2/P3 表示实施优先级，不是缺陷严重程度；六项均属于本轮目标，按关联修改成组完成。

| 顺序 | 任务 | 最终结果 | 行为取舍 |
| --- | --- | --- | --- |
| P1-A | 主策略 Dataset 退出手动验证视图 | 删除主 Dataset 的验证视图、训练/验证状态分支及单次 RGB 包装 | 主 Dataset 的手动 validation-view API 退出；预训练 holdout 不变 |
| P1-B | 取消 canonical resume 的额外 source_id 门槛 | fresh/resume 使用同一套已有 split 身份匹配 | 双方无 ID 且其他身份匹配时允许 resume；不再声称能识别同结构不同内容 |
| P2-C | 撤掉独立 hand_prior/artifacts.py | 写入归 exporter，加载归已有 runtime codebook 模块 | NPZ 格式与模型状态不变；旧 helper 导入路径退出 |
| P2-D | 仿真触觉来源检查只留实际执行边界 | 删除 rollout 外层重复拒绝，保留 SimRunner.run 检查 | 无效 real 请求可能在恢复模型后才失败，并进入失败记录 |
| P3-E | 删除 seed pool 容器类型预检 | 保留整数范围、去重与 selection/test 划分约束 | 不再专门承诺非 list 的自定义错误；不扩展 iterable 协议 |
| P3-F | 信任自产预训练 checkpoint 的 affine | 删除 strict load 后两次重复数值扫描 | 自产 checkpoint 不再做这两次损坏探测；外来 affine 校验继续保留 |

预期结构收益：删除 1 个 Python 文件和 3 个函数/方法（主 Dataset 的 `get_validation_dataset`、`preprocess_rgb`，独立 `save_codebook`），减少 Dataset 状态与重复边界。不设净删行数配额；IO 仍需实现，不能把迁移的 51 行全部算作消失。

## 3. P1-A：主策略 Dataset 只承担训练数据读取

主要文件：`dexmani_policy/datasets/demonstration.py`、`dexmani_policy/training/common/data.py`。

事实：`DemonstrationDataset.get_validation_dataset()` 在基线没有仓库内生产调用；AFRO/CordViP 的两处同名调用作用于独立的 `ObservationPretrainDataset`。主 PolicyTrainer 不构建主 Dataset 的验证视图。这是主动退出当前流程不用的 API，不能据此声称外部 notebook 从未调用。

实施要求：

1. 删除 `DemonstrationDataset.get_validation_dataset()` 和不再使用的 `copy` 导入。
2. 删除主 Dataset 的 `self.training`。将 `fit_normalizer()` 的“不能在验证视图拟合”检查放入已有的 `isinstance(dataset, ObservationPretrainDataset)` 分支，与预训练字段对应检查一起保留。不要改成 `getattr(..., True)`、添加状态标记或引入新的 Dataset 基类。
3. 删除 `DemonstrationDataset.preprocess_rgb()` 单次包装。在 `__getitem__()` 的 resize 分支直接调用 `preprocess_rgb_train()`，保持 resize、随机 crop、颜色增强、uint8 舍入和随机数消耗行为。
4. 保留无 resize 分支的 `preprocess_rgb_eval(..., resize_hw=None, center_crop_hw=None, ...)`：它仍负责 dtype/layout 转换，并非遗留验证逻辑。保留 SimRunner/Runtime 的确定性 RGB 预处理。
5. 不再持久保存 `self.augmentation_cfg`；将构造参数直接传入 `build_augmentors(augmentation_cfg)`。该方法仍可承担现有较长的增强组装，不为消灭单次调用机械内联到构造函数。
6. 保留 `train_mask`、`val_mask`、`split_manifest`、`val_ratio`、训练 episode 上的 affine 拟合，以及实际排除 holdout 的行为。不要将 val episodes 并回训练。
7. 保留 `build_sampler()` 扩展点：`AdapTacDataset.build_sampler()` 调用它过滤没有未来监督的窗口。保留各 Dataset 变体的采样、目标拼接和未来力 mask。
8. 保留 `ObservationPretrainDataset.get_validation_dataset()`、其 `training` 状态及 AFRO/CordViP 的验证流程。不要对同名方法做全仓库批量删除。

验收重点：同一输入与固定 RNG 下主 Dataset 的训练样本保持一致；主 Dataset 不再依赖 `.training`；预训练验证视图仍被拒绝用于拟合；AdapTac 的终止窗口过滤仍有效。文档改成“主策略保留划分、仅构造训练 sampler；观测预训练按需构造 holdout 视图”。

## 4. P1-B：source_id 可选，但已有身份继续比较

主要文件：`dexmani_policy/training/common/data.py::build_dataset`、`README.md`。`datasets/split.py` 原有比较逻辑原则上无需修改。

删除 `build_dataset()` 中仅因 `resume_from != None`、canonical format 且 `source_id is None` 就拒绝的条件及对应注释。该限制在旧 README 已明确写出，因此本次是放宽能力，不应描述成修复未文档化 bug。

保留 `_restore_split()` 对 `episode_ends`、`episode_ids`、`source_id`、`trial_ids` 的现有相等比较，以及 train/val 索引合法、互斥、非空 train、trial 不跨集合等规则：

- 双方 source_id 均为 `None`、其余身份与划分匹配：允许 fresh/resume。
- ID 不同，或一方为 `None` 而另一方有 ID：仍拒绝。
- episode 边界、episode/trial 身份不匹配，或划分非法：仍拒绝。

不要忽略旧 manifest 字段、自动补造 source_id、重新按 seed 划分或覆盖 checkpoint 的已有划分。不要增加 `allow_missing_source_id`、迁移器、内容 hash、来源注册表或警告开关。

同步 README 的准确约定：已有 ID 保留并匹配；没有 ID 的旧数据在其他条件匹配时可续训，由研究者保证使用同一份未改写的数据。双方无 ID 时，同结构不同内容无法区分；source_id 本来也只是 producer 标签，不是内容哈希。数据重新导出后应开启新实验，不能靠删 ID 绕过匹配。

## 5. P2-C：codebook IO 回到实际所属模块

涉及文件：

- 删除 `dexmani_policy/agents/hand_prior/artifacts.py`。
- 修改 `dexmani_policy/agents/hand_prior/codebook.py`。
- 修改 `dexmani_policy/training/pretrain/hand_prior/codebook.py`。
- 更新 `dexmani_policy/agents/policies/dqrise.py` 与 `dexmani_policy/test_tools/agent_checks.py` 的导入。

实施要求：

1. 将 `load_codebook()` 作为普通函数放到已有 runtime `agents/hand_prior/codebook.py`，就近使用 `HandStateCodebook`。不增加类、adapter、注册表或兼容转发模块，不把 sklearn/训练入口引入 runtime。
2. 将 `save_codebook()` 的 payload 组装与 NPZ 写入直接放进唯一生产调用方 `export_codebook()`；删除 helper 定义和旧导入。
3. 保留 exporter 的 `atomic_path(output, overwrite=overwrite)`、显式覆盖规则，以及 `open(temporary, "wb")` 后交给 `np.savez()` 的写法，防止 NumPy 自动加后缀而破坏原子发布。
4. 保持 NPZ 八个字段：`sorted_hand_poses`、`hand_normalizer_scale`、`hand_normalizer_offset`、`pose_space`、`hand_dim`、`num_groups`、`codebook_size`、`metadata_json`。保持 JSON 编码行为和现有 metadata。此次不迁移产物格式。
5. 保持 `allow_pickle=False`，原型表形状/有限值检查，affine 形状及转为运行 dtype 后的有限、可逆检查，手部坐标顺序/弧度约定与 DQ/VQ 实际 split 相等比较。
6. 保持 `HandStateCodebook` 类路径、buffer 名称、state_dict、PCA 排序、量化规则与导出 CLI 不变。完整 Policy checkpoint 恢复仍不依赖初始 NPZ 文件。

现有 `check_codebook_pipeline()` 已覆盖导出、加载和 Policy 流程，优先更新并复用。检查旧 NPZ 可以由新加载位置读取，新导出与旧格式字段/数值/metadata 等价；不要要求 ZIP 文件字节逐位相同，也不要新增格式版本或迁移框架。

## 6. P2-D：SimRunner 负责仿真来源拒绝

修改 `dexmani_policy/evaluation/rollout.py`，删除 `evaluate_policy()` 中基于 cfg 的 real tactile 提前拒绝，并移除该文件不再使用的 `require_tactile_source` 导入。

保留 `dexmani_policy/env_runner/sim_runner.py::SimRunner.run()` 根据实际 agent 的 consumed fields 与 `tactile_source` 执行的检查。该检查必须仍位于创建环境之前；保留 Dataset 的来源/timing 检查与 InterAct 的坐标转换。

本项对合法评测不改变行为。非法 real 请求会先经历已有前置步骤，可能在模型恢复后才收到来源错误；到达来源检查后的异常进入现有 failure record 流程。这是明确的错误时机与诊断变化，不宣称完全等价或提升性能。

不要删除 `write_failure`、最终原异常重抛、原子 JSON 写入或资源关闭；也不要在别处重新复制一层 cfg 来源检查。`run()` 是 generator，验收需实际推进迭代，确认拒绝发生在 `make_env()` 之前。

## 7. P3-E/F：两项局部减法

### E. seed pool

在 `evaluation/seeds.py::validate_seed_pool()` 中只删除 `isinstance(pool, list)` 及其自定义异常两行。保留严格整数判断（包括拒绝 bool）、`[0, 2**32)` 范围、去重、所需数量、显式评测非空和 selection/test 隔离。

保留现有调用处的 `list(seeds)` 及确定性排序/抽样。不要新增任意 iterable 适配或更换随机数算法；对现有合法输入，选择结果必须完全一致。

### F. 自产 checkpoint

在 `training/pretrain/obs/common.py` 的 `restore_run()` 和重导出 `main()` 中，删除 `strict=True` 加载后的两次 `validate_normalizer_state(model.normalizer)` 调用及无用导入。它们读取本项目自产 resume checkpoint，按与主策略恢复一致的信任约定处理。

保留 strict 加载、optimizer/scheduler/RNG 恢复和原子保存。保留 `training/common/data.py::fit_normalizer()` 对导入 affine 的 `validate_normalizer_state()` 调用及函数定义，也保留拟合生成 affine 时的有限/可逆检查。这是信任边界的收窄；strict 加载只约束结构，不应宣称它能证明数值有限。仅减少两次扫描，不声称有显著训练加速。

## 8. 本轮保留与禁止扩展

以下文件相对实施起点必须无内容变化：

- `dexmani_policy/agents/obs_encoder/text.py`
- `dexmani_policy/agents/action_decoders/flow.py`
- `dexmani_policy/agents/action_decoders/sampling/time_sampling.py`
- `dexmani_policy/agents/action_decoders/backbones/dit.py`
- `dexmani_policy/agents/action_decoders/backbones/ditx.py`
- `dexmani_policy/agents/action_decoders/backbones/modules.py`

保留文本组件、尚未接入主策略的普通 DiT 与 `RectifiedFlow` 能力；不得通过调用方删掉 sampler/context cache/return_features 等能力。保留 InterAct 的 `time_shift_alpha` 传递及 effective config、默认值 1，保留 Trainer/restore_policy 的 matmul precision `highest`。

其他明确保留的内容：

| 内容 | 保留原因 |
| --- | --- |
| `validate_config()` 的普通 normalization mode 枚举 | 继承 initial_state 的 affine 字段会跳过 `fit_field()`；后者不能替代前者。不要为消除几行重复再建统一 validator |
| future offsets 在 Dataset/Policy 的检查、动作窗口、UNet horizon、观测字段与 RGB 组合约束 | 当前有独立构造边界或实际错误风险；本轮不再收窄这些接口 |
| `tactile_source`、dt/physics_dt/frame_skip、对齐规则、坐标系、合法零力、未来 target/mask | 影响物理语义与监督正确性，不能按 source/contract 关键词清理 |
| `sample_source_rows()`、几何 source 命名、初始化权重路径、上游 `Source:` 注释 | 分别承担原始行映射、坐标含义、fresh 加载或归属说明，不是来源管理框架 |
| NPZ metadata 与 VQ checkpoint `model_config` | 本轮不迁移格式；split 有运行消费者，其余有实验解释价值；VQ 重建确实使用 model_config |
| `check_data.py`、EpisodeStore 边界、train-only affine、FP32 几何、DQ 量化前有限检查 | 防止溢出、数据污染和静默改变实验含义 |
| strict/assign、restore-before-optimizer、EMA/BF16/compile、梯度累积与分组损失 | 与训练/恢复正确性相关，不属于本轮已证实冗余 |
| inspect_policy、PolicyInfo、PolicyRuntime、future_actions、best/lock/selector、分阶段同步 | 仍有 Real/外部流程消费者；本轮不更改公开部署与发布行为 |

不实施此前独立讨论的 `configs/tactile` 重排、Uni3D 共享投影、全仓配置/异常重构、远程工作流迁移。当前 Uni3D 只删除了构造期预加载，不能把未采用的共享投影写成已实现。本轮不新增参数、依赖、Python 模块、持久诊断文件或通用测试框架。

## 9. 有限且有效的验证

先运行现有相关检查；仅为改变的行为与保留边界补充最小验证。优先放入已有检查入口，或使用一次性本地验证，不为删除两行制造一套永久测试设施。禁止测试只复述实现、只断言函数不存在或用关键词计数替代行为验收。

| 改动 | 必要验证 |
| --- | --- |
| A：Dataset | 小型 Zarr/已有 fixture 上，固定 RNG 比较训练样本；覆盖 resize 与无 resize 的 RGB、train-only 统计和 AdapTac 过滤。预训练 holdout 仍可创建，拿它拟合仍拒绝 |
| B：source_id | 实际 build_dataset 路径允许无 ID 且匹配的 resume；不同 ID、单方缺失、边界/身份不匹配仍被 split 恢复拒绝。确认 resume 沿用保存的实际索引 |
| C：codebook | 现有导出/读取流程、旧格式读取、状态与 metadata 等价；保留 affine/prototype 的必要坏输入拒绝，不重写模型测试 |
| D：仿真入口 | 对 real-source agent 推进 SimRunner generator，环境创建前拒绝；验证原有异常记录/重抛路径仍存在。有效 sim 输入仍通过来源边界；无需完整仿真证明这一局部改动 |
| E：seeds | 固定输入下 selection/test 结果一致且互斥；重复、bool、越界仍拒绝 |
| F：预训练恢复 | 自产正常 checkpoint 仍 strict 恢复/重导出；确认外来 affine 校验调用保留。不要写“损坏 checkpoint 现在必须被接受”的反向测试 |

基础检查与现有入口：

```bash
git diff --check
python -m compileall -q dexmani_policy
python -m dexmani_policy.test_tools.agent_checks --device cpu
```

在已有环境和数据可用时复用短训恢复入口：

```bash
python -m dexmani_policy.test_tools.smoke_test --max-updates 2 --resume-updates 1 dp3
```

按变更实际风险选择必要的 CUDA/预训练检查，不下载大模型或跑全量训练来验证文件搬移。若依赖或数据不足，完成可执行的静态与局部检查，明确列出未执行项及原因；不得用 mock 运行冒充真实 Torch、模型恢复或仿真通过。未通过的相关必要验证须在结果中保留为未验收项。

## 10. 文档、提交与交付标准

1. 同步 `README.md` 的 source_id/resume 约定，`docs/项目架构.md` 的 Dataset/holdout 与归一化说明；搜索并更新真实存在的旧导入路径。只有发生实际说明变化才修改 `docs/手部动作表示.md`、`docs/观测预训练.md` 等文档。
2. 不把本任务书建立为长期运行依赖，不恢复已经授权删除的历史任务书，不生成新审计报表或兼容空壳。
3. 六项各给出“已完成/基线已完成/有具体阻碍”的结果。当前源码若与任务书不同，先检查实际消费者并作最小适配；不按旧行号机械修改，不对已经清理的代码重造问题。
4. 最终 diff 仅含本轮相关源码、必要文档与有限验证。删除 `artifacts.py` 后无失效运行时导入；保护文件相对实施起点不变；没有新配置开关、注册表、fallback 或异常吞噬。
5. 汇报文件/函数/状态的真实减少、主动退出的接口与信任变化、实际验证命令和结果、未验证项。不要把文件迁移等同机制删除，不以删除量代替正确性，也不声称未经测量的性能收益。

## 11. 可直接复制给 Codex 的执行 prompt

```text
请直接实施仓库根目录 RESEARCH_CLEANUP_TASKBOOK.md，先读 AGENTS.md。

目标是个人研究代码更薄、更直接。任务书基线为 5a2c7d3；先检查实际 HEAD、工作区改动和相关调用方，只适配本轮差异，不回退用户修改，也不再做一轮泛化审查。

按任务书完成 P1-A/B、P2-C/D、P3-E/F：退出主 DemonstrationDataset 的验证视图并删除相应状态；取消 canonical resume 强制 source_id 的额外门槛但保留原 split 身份比较；删除 hand_prior/artifacts.py，将保存内联 exporter、加载并回已有 codebook.py；仿真来源检查只留 SimRunner 执行边界；删除 seed pool 的 list 类型预检；删除自产观测预训练 checkpoint strict load 后两次 affine 重验。

我授权任务书明确列出的接口退出和信任变化，无需为这些取舍重复确认。保留 ObservationPretrainDataset 的 holdout、AdapTac 的 build_sampler 扩展、train-only affine、已有 source_id/split 比较、外来 affine 检查、普通 normalization mode 枚举及必要的物理/数值边界。六个保护文件不改，尤其不删除文本组件和未接入 DiT，不改写 RectifiedFlow；不扩展到 tactile 配置或 Uni3D 优化。

直接完成代码与相关文档修改，不停在建议或计划。不留兼容空壳，不加开关、模块、框架、fallback 或新的审计设施。复用已有检查，仅对改变的行为补最小必要验证；运行可用检查，明确未执行或失败项，不把静态检查当成真实训练通过。

完成后按六项简要报告修改、行为取舍和验证结果，检查最终 diff、旧导入及保护文件。不要为凑删除行数继续扩大范围；遇到真实阻碍时先完成其他独立项，再具体说明阻碍及仍未验收的部分。
```
