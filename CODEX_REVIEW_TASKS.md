# Codex 任务书：最小正确性修复与研究代码简化

> 编写日期：2026-10-09（Asia/Shanghai）。
> 审查基线：`haoyangzhanglab/dexmani_policy@184126352cc707b999c3eb005fb78841a6e62cd1`。
> 状态：**待实施**。本文件是实施任务书；提交本文件不代表下列代码已经修改或验收通过。

## 1. 目标与范围

请先读取根目录 `AGENTS.md`，再完成下表 **11 项**。目标是让论文开源代码简洁、高效、正确、好用：修复已确认的条件性问题，删除确定多余的工作，保留研究语义与现有可用接口。

| 顺序 | ID | 优先级 / 类型 | 本次交付 |
|---|---|---|---|
| 1 | F10 | P1 · 必须修复 | DQ 连续 code 在量化前拒绝 NaN/Inf |
| 2 | F07 | P1 · 必须修复 | 明确拒绝当前 DQ/VQ 不支持的 Gaussian 动作归一化组合 |
| 3 | F01 | P1 · 必须修复 | raw RGB 预处理入口统一检查 uint8、channel-last 布局 |
| 4 | F44 | P2 · 必须修复 | 按消费者需要构造 validation sampler，保留 holdout 隔离 |
| 5 | F08 | P2 · 必须修复 | R3D 分组前明确检查点数下界 |
| 6 | F12 | P2 · 限定简化 | 去掉连续、无重复行窗口的一次多余复制 |
| 7 | F38 | P2 · 限定简化 | PCA 等纯离线依赖在实际使用处导入 |
| 8 | DEP-01 | P2 · 限定简化 | 普通 best 推理解析与 selection 证据校验分开 |
| 9 | F11 | P2 · 配置显式化 | SAT YAML 写入当前已有的三个 Beta 默认值 |
| 10 | F45 | P3 · 局部简化 | padding 校验与短 episode 筛选各有一个责任方 |
| 11 | F16 | P3 · 文档整理 | 说明 RGB 策略主路径与可选 RGB-D 几何接口 |

优先级表示问题的重要程度，不要求每项独立开一个提交。F45 与 F12 可在处理 dataset 后连续完成；DEP-01 应单独提交，便于审阅和回退。

执行约束：

- 本轮不改变模型结构、损失、时间采样分布、动作表示、数据划分、normalizer 拟合范围、增强概率、训练预算或默认 NFE。
- 不启动训练消融、完整训练、长时间评测或真机运动。若某个扩展需要训练结果才能决定是否采用，将它留在本轮之外。
- 不新增 registry、通用配置/schema 框架、部署 manifest、代码生成器或资产迁移系统。局部函数和现有字段足够时，直接使用。
- 不顺带执行 F02、F03、F04、F43 或其他历史 review 项；不要把已选择的 11 项扩写成全仓清理。
- 不删除现有 checkpoint/NPZ 必要状态和数值校验，不批量改写已有实验、split manifest 或代码本。
- 实施时若 HEAD 已变化，先核对相关函数；已被修复的项目记录现状并跳过重复补丁。完整路径相对于仓库根目录；同段内的短文件名沿用该段已注明的目录。

## 2. 参考项目如何用于本方案

参考用于核对具体机制，不把“参考项目这样写”当作正确性证明，也不整体搬入其训练框架。以下链接均固定版本。

| 参考 | 已核对的机制 | 本次采用及保留边界 |
|---|---|---|
| [Diffusion Policy sampler](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/sampler.py) | 无 padding 时直接使用已读 sample，padding 时再分配/补齐；观测可只读前 k 步 | 用于 F12 的条件分配思路。DexMani 仍由 `ReplayBuffer.read` 保证独占内存，不能直接改成共享缓存 view。参考的 padding clamp 不能继续与本仓库外层另一套解释并存。 |
| [DP3 dataset](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/dataset/adroit_dataset.py) | `get_validation_dataset()` 被调用时才创建验证 sampler | 用于 F44。保留本仓库显式 `val_mask`，不照抄 `~train_mask`：训练 cap 排除的数据不等于 holdout。 |
| [ManiFlow dataset](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/dataset/adroit_dataset.py) | 同样通过显式方法构造验证视图 | 复用现有 Dataset/getter，不增加验证生命周期管理器；不照搬其全数据 normalizer 统计。 |
| [R3D KNNGrouper](https://github.com/Wushr-Lance/R3D-Policy/blob/e637c0148376ddc4b5e667fa8f8e108cb8ff7a85/R3D/r3d/model/vision/pointnet_extractor.py) | FPS 取固定组数，KNN 取固定组大小，再固定 reshape | 用于确认 F08 的固定尺寸语义；补点数前置检查，不改成动态分组。 |
| [SAT policy](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/policy/sat.py)、[SAT config](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/config/sat.yaml) | 官方 policy 调用 `TargetConditionalFlowMatcher`；配置声明其 flow 参数 | F11 只显式保存 **DexMani 本地** Beta 默认值。不能声称 `0.999/1.0/1.5` 是这里核对出的官方 SAT 参数，也不据此替换本地 flow 实现。 |
| [DQ-RISE 离线代码本导出](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/eval_vqvae.py) | 离线枚举、PCA 排序、保存手部原型；解码包含有界范围转换 | 用于 F38 的离线依赖边界；F07/F10 的修复依据仍是本仓库实际数值契约。不能把有界手部表示推广成支持任意 Gaussian 坐标。 |
| [3D Foundation Policy 数据构造](https://github.com/horipse01/3d-foundation-policy/blob/2facbb459d1a3d507baa1f00c7e6bd8dd9864f1f/droid_policy_learning/robomimic/utils/train_utils.py) | `load_data_for_training` 只在启用 validation 时构造验证数据 | 支持 F44 的按需构造；不引入其 robomimic 数据工厂与配置体系。 |
| [Diffusion Policy 真机加载入口](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/eval_real_robot.py) | 从指定 checkpoint 和保存配置恢复模型，选择 EMA 与推理参数 | 用于 DEP-01 的最小加载边界；不复制其硬编码推理步数，也不以此删除 DexMani 的 held-out 评测证据校验。 |

F01 可另对照 [DP image dataset](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/robomimic_replay_image_dataset.py)：它同样按 raw uint8 转 float32/255；这说明缩放有输入前提，不证明其已替我们检查错误输入。

## 3. 必须修复的五项

### F10：量化前拒绝非有限 code

**定位**：`dexmani_policy/agents/vq_hand/codebook_manager.py` → `continuous_index_to_hand_pose`；消费者为 `dexmani_policy/agents/core/dqrise.py` 的预测路径。

**确认事实与触发条件**：当前先 clamp，再 half-up 舍入及索引。`+Inf/-Inf` 被映射为有限边界原型，后续动作有限性检查无法识别原始错误。这不意味着默认模型已被观测到产生 Inf；上游 scheduler 也可能更早裁剪异常。

**实施**：在首次 clamp/整数转换之前，用一个 `torch.isfinite(...).all()` 检查输入；非有限值明确抛出 `ValueError`。保留 codebook 未加载检查、有限值裁剪、half-up 规则、原型顺序、返回形状和反归一化流程。检查放在最终 code 解码边界，不加入每个 denoising step；不使用 `nan_to_num` 或静默替换。

**验收**：NaN、正负 Inf（含混合 batch）都在量化前失败；有限区间内、有限越界值、端点与舍入分界的结果保持原语义。使用一个小型内存 codebook 即可，不加载训练权重、不运行机器人。

### F07：拒绝当前 DQ/VQ 不支持的 Gaussian 组合

**定位**：`dexmani_policy/agents/normalization.py`、`dexmani_policy/agents/core/dqrise.py`、`dexmani_policy/training/build_utils.py` 的归一化校验；`dexmani_policy/training/train_vq_hand.py::prepare_policy_data`；`dexmani_policy/agents/vq_hand/export_codebook.py::extract_codebook` 与 `dexmani_policy/agents/vq_hand/codebook_manager.py::_decode_valid_pose`。

**确认事实与触发条件**：通用校验允许 Gaussian + `clip_sample=false`，但 VQ 原型导出仍 clamp 到 `[-1,1]`；Gaussian 区间外可代表合法动作。默认 `auto/limits` 不属于这个反例。

**实施**：

1. 将“当前 DQ 手部原型只支持既有 `auto/limits` 有界动作归一化”写成一处小型语义检查。沿用已有 normalization spec，不从 scale/offset 数值猜测模式，不创建策略注册表。
2. DQ 的 config 校验和 `set_normalization_spec` 路径复用检查，使新训练、恢复/推理都不会继续接受显式 Gaussian。其他策略的 Gaussian + `clip_sample=false` 不受影响。
3. `prepare_policy_data` 在构造数据和拟合统计之前检查已解析 spec；独立 VQ 配方现有 `limits` 路径保持不变。
4. 直接导出旧 VQ checkpoint 时，也读取已有 `split_metadata.normalization_spec`；明确记录为 Gaussian 的产物必须在解码/PCA/写文件之前拒绝，不能绕过训练入口修复。旧 standalone 产物没有这个字段时，保留既有加载规则，不新增必填元数据或升级 NPZ 格式；不能声称仅凭 affine 参数已证明其归一化模式。

不要通过去掉原型 clamp、扩大范围、重新拟合 normalizer 或改变量化方式来“支持 Gaussian”；这些会改变研究配方。

**验收**：DQ Gaussian 在 `clip_sample=false/true` 下均明确失败；policy-config VQ 准备与已标注 Gaussian 的旧 checkpoint 导出均失败，被拒绝的导出不生成 NPZ。`auto/limits` 默认路径保持；至少核对一个非 DQ Gaussian + `clip_sample=false` 配置仍被接受。所有测试可以使用配置和小型 synthetic checkpoint。

### F01：raw RGB 预处理契约统一

**定位**：`dexmani_policy/datasets/base_dataset.py::_preprocess_rgb_cpu`、`dexmani_policy/datasets/preprocessing.py::preprocess_validation_rgb`；后续为 `dexmani_policy/agents/core/dp.py::DPObsEncoder.forward`。

**确认事实与触发条件**：训练 float/color 分支无条件 `/255`；验证已有 uint8/channel-last 检查。错误的 float raw RGB 可在训练中被再次缩放。合法 uint8 数据没有被证实存在这个问题。

**实施**：在 raw RGB 进入空间变换或缩放前复用一个小型检查，放在现有 `preprocessing.py` 内即可。检查 dtype 为 uint8，最后三维为 HWC 且通道为 3；训练采样入口还应符合它实际接收的 `(T,H,W,3)`。验证/部署函数保留现有 `[...,H,W,3]` 前导维支持。NumPy/Torch 输入按原接口处理，并维持该模块按需导入 Torch 的特性。

raw 检查与模型已处理输入要分清：DP encoder 仍可接收 dataset 输出的 CHW uint8 或 float32 `[0,1]`。不把它们误判为非法 raw 数据，不加第二次缩放。保留 uint8 快路径、resize/crop/color 顺序、验证 center crop 与原有插值/量化方式；不做整图 min/max 扫描来猜输入约定。

**验收**：合法 uint8 覆盖 float 输送、uint8 快路径及 color 后 uint8 输送；train/validation 的各自输出与修改前相同（训练对拍固定随机状态）。float raw `[0,1]`、float raw `[0,255]`、错误 rank/通道布局在变换前失败。一个非退化尺寸的 CHW 反例即可；不能宣称只靠 shape 能识别所有维度巧合的错误布局。

### F44：validation sampler 由真实消费者请求

**定位**：`dexmani_policy/datasets/base_dataset.py::__init__/get_validation_dataset`；`dexmani_policy/datasets/multi_task_dataset.py::get_validation_dataset`；`dexmani_policy/training/train_vq_hand.py::prepare_policy_data`；`dexmani_policy/training/build_utils.py::build_dataset_and_normalizer`；`dexmani_policy/training/resume.py::validate_resume_contract`。

**确认事实与触发条件**：构造 BaseDataset 时立即构造验证 sampler，但主 policy 的 `train.py/train_ddp.py/Trainer` 只消费 train loader。非零 holdout 全短或全无效时，未被消费的验证集能阻断有效主训练。默认 `val_ratio=0` 不触发。

**实施**：

1. 初始化时只构造/筛选训练 sampler，保存原 `train_mask/val_mask`，令 `_validation_dataset=None`；删除构造函数对 getter 的调用。
2. getter 保留无 holdout 返回 `None`；有 holdout 时才构造验证视图、执行现有有效窗口筛选、禁用增强并启用确定性 RGB 预处理。成功后缓存并返回同一验证视图。不能因 `copy.copy` 共用嵌套字典而反向改写训练 recipe。
3. 显式请求的 holdout 全短/全无效时，维持明确失败；不能静默转成“无验证”，否则 VQ 的 best 选择可能从验证指标退回训练指标。未请求的无效 holdout 不阻断主训练。VQ 继续显式调用 getter；MultiTask 继续按需调用子数据集 getter。
4. 从新生成的 `data_recipe.split_manifest` 中去掉仅用于诊断的 `val_windows`；不得为了填日志而触发 getter。VQ 已有顶层 `split_metadata.val_windows` 可记录实际验证 summary，无需新增诊断体系。
5. 为避免仅移除诊断计数就拒绝旧 checkpoint，strict resume 比较时在副本中仅忽略确切路径 `data_recipe[*].split_manifest.val_windows`。不修改输入 contract，不改变 `facts_format`，不忽略整个 data recipe 或任意名字相似的字段。split 内容/摘要、mask、训练窗口与源行、normalizer 语义及数据身份继续按既有逻辑检查。

不使用 `~train_mask` 重建验证集，不改变 `actual_train_ids` 的现有语义，不把 holdout 加入 normalizer，也不在此项引入新的 split 策略。

**验收**：

- 有效 train + 全短 val，以及有效 train + 全非有限窗口 val：仅构造主训练数据成功；显式请求验证明确失败。
- 合法 val 请求两次不重复构造 sampler；验证增强关闭、RGB 确定性路径不变；无 holdout 仍返回 `None`。
- train 与 holdout 源行隔离，normalizer 输入仍只来自合格训练窗口的唯一源行。
- 旧 contract 只多 `val_windows` 时可通过；改变 split/mask、训练窗口或数据身份等受检语义仍失败。不要为兼容而拷贝整个旧 recipe 覆盖当前事实。

### F08：R3D 分组点数下界

**定位**：`dexmani_policy/agents/obs_encoder/pointcloud/uni3d.py::KNNGrouper.forward`；共享 FPS 在 `dexmani_policy/agents/obs_encoder/pointcloud/ops.py`。

**确认事实与触发条件**：FPS 返回 `min(K,N)` 个中心，grouper 按固定 `num_groups` reshape；KNN 又要求 `group_size<=N`。默认 N=1024、G=512 不受影响。

**实施**：FPS/KNN 之前检查 `N >= max(self.num_groups, self.group_size)`，错误信息包含 N、组数和组大小。以实际输入 N 为准，不只校验 YAML。保留共享 FPS 的原语义，不新增补点、动态组数或“自动修正配置”。

**验收**：默认 G=512、M=32 时，N=511 明确失败，N=512/1024 输出形状不变；另用 G=4、M=8、N=7 覆盖组大小约束。边界拒绝发生在 FPS/KNN 前；合法路径可直接测试 grouper，不需要实例化完整预训练 Uni3D。

## 4. 经风险收益评估后采用的三项简化

### F12：只移除 sampler 中确定重复的一次复制

**定位**：`dexmani_policy/datasets/replay_buffer.py::ReplayBuffer.read` → `dexmani_policy/datasets/sampler.py::SequenceSampler.sample_sequence` → `BaseDataset.apply_augmentation`。

**选择理由**：`read` 已通过 `np.array(..., copy=True)` 返回独占数组；当前 sampler 对每个 key 再用高级索引复制。去掉连续窗口的这次分配有直接机制依据，端到端吞吐收益尚未测量。

**实施**：保留 `ReplayBuffer.read` 的独占返回契约。在每个 key 按 `key_lengths` 取到的源行上判断是否有重复 padding 行：当前源行由单位步长序列 clip 得到，因此 `end-start == len(rows)` 足以识别连续、无重复行的读取，此时直接返回 `values`；否则保持 `values[rows-start]`。不要增加逐元素扫描或另一套窗口索引算法。

判断必须按 key 进行：观测前 N 步可能无 padding，而同一样本的动作 H 步已有尾部 padding。保留 dtype 转换、role 长度、padding 和有效窗口筛选。`apply_augmentation` 的首次 copy 本轮保留，不能顺手把所有 copy 一起删除。

**验收**：独立的逐帧 padding oracle 覆盖中间窗口、左右边界、短 episode、H=1、N<H 及不同 key 长度。原位修改返回样本后，底层 NumPy 数据、Zarr 解码缓存、相邻/后续样本都不变；覆盖缓存命中和跨块读取。确认连续路径确实复用了 owned `read` 结果，不把“没有训练退化”当作内存隔离证明。

**收益/风险**：确定少一次数组分配与复制；风险在误共享缓存和误判 padding，通过局部样例即可覆盖。不承诺 DataLoader 吞吐或峰值内存的量化提升，不以长基准作为实施前提。

### F38：离线依赖局部导入，保持代码本接口与状态

**定位**：`dexmani_policy/agents/vq_hand/codebook_manager.py::reindex_by_pca`、`dexmani_policy/agents/vq_hand/__init__.py` 和导出入口。

**实施**：把 `from sklearn.decomposition import PCA` 从模块顶层移到实际执行 PCA 的 `reindex_by_pca` 内。仅对确认纯离线使用、且影响推理导入的依赖作同类处理；不为了行数把所有标准库 import 都搬进函数。

核对包入口不能绕路重新导入 sklearn。当前 `dexmani_policy/agents/vq_hand/__init__.py` 虽导入 VQ 类，这些类本身的 import 不包含 sklearn；仅此事实不要求清空重导出或引入 lazy-export 机制。保留 `CodebookManager`、VQ 类现有公开导入路径、方法签名、持久化 buffers、NPZ v3 与 `state_dict` 键，不拆出两套 manager。完整开发环境的 dependency 声明本轮不改成多个 extras。

**验收**：在基础推理依赖可用而 sklearn 被屏蔽的独立进程中，真实导入 `DQRISEAgent/CodebookManager` 并加载小型持久化代码本应成功；恢复不重新跑 PCA、不依赖原 VQ checkpoint。sklearn 可用时小型导出仍成功；缺 sklearn 时只有执行离线 PCA 才报告缺依赖。修改前后 state_dict 键与合法 lookup 结果一致。

**收益/风险**：推理导入不再被纯离线 PCA 库阻断；局部 import 风险低。没有测得启动加速，不把它扩展成打包或类层次重构。

### DEP-01：best 保存最小推理信息，评测单独读取证据

**定位**：

- writer：`dexmani_policy/select_best_ckpt.py` 的成功发布分支。
- 普通加载：`dexmani_policy/agents/loader.py::resolve_best_checkpoint/resolve_checkpoint` 与 `dexmani_policy/deployment/runtime.py::inspect_policy`。
- 评测/显式 handoff：`dexmani_policy/evaluation/protocol.py`、`dexmani_policy/eval_best_ckpt.py`、`dexmani_policy/record_demo.py`；联动核对 `scripts/eval/eval_pipeline.sh`。

**确认事实**：当前 `best_ckpt.json` 只引用 `selection_result.json`；共用 resolver 会验证候选、completed stages 和 selection 身份。因此缺少过程材料会阻断普通 best 加载。完整产物正常时不受影响。直接指定权重文件虽然可绕过记录，但不显式带 EMA/NFE 就可能采用与 best 不同的配置默认值。

**选定方案：沿用 best 文件，增加自包含的推理快照。** 不新增部署 bundle、manifest 或新 schema 框架。

新 writer 写出的结构示意如下；路径与参数由本次成功 selection 的实际结果填入，不使用示例常量：

```json
{
  "ckpt_relpath": "checkpoints/<selected-milestone>.pt",
  "inference": {
    "use_ema": true,
    "inference_steps": 10,
    "policy_seed_mode": "episode_seed"
  },
  "selection_result": "<selection-run>/selection_result.json"
}
```

1. **Writer 成对调整。** 原有不可变 `selection_result.json`、详细报告和 `--result-file` handoff 继续保存。成功后原子更新 `best_ckpt.json`，从同一个已完成结果复制 `ckpt_relpath` 与整个 `inference` 字典，并保留指向详细结果的引用。已有 `global_step/pct/selection_id` 可作为普通元数据保留，但不新增普通推理的过程证据要求。技术失败保持旧 best；正常全零结果的现有排名/发布语义不变。
2. **普通 resolver 只解决如何推理。** `resolve_best_checkpoint(experiment_dir)` 从快照取得具体权重路径与 EMA/NFE，保留文件存在、目录归属以及参数类型/正整数检查。字段完整时不解引用 `selection_result/selection_summary`，不要求候选、阶段或 seed 记录。`inspect_policy` 保持现有显式 override 优先级和返回接口；加载使用实验保存配置及 checkpoint 内状态。
3. **严格读取器归评测。** 在现有 `dexmani_policy/evaluation/protocol.py` 中提供明确的 `resolve_selection_checkpoint(experiment_dir, record_path=None)`（名称可依现有组织调整）；迁移现有 selection 证据检查，复用 loader 的小型路径解析逻辑。不要给普通加载增加 `strict/permissive/production` 模式矩阵。内部调用者一起改完，不保留两套半成品 resolver。
4. **held-out best 也要走证据路径。** `dexmani_policy/eval_best_ckpt.py` 的普通 best 评测需要实际 selection task/seeds，不能只改显式 `--selection-record`。普通 best 评测保留原有 EMA/NFE override 能力；显式 handoff 仍拒绝冲突的 checkpoint、EMA/raw、NFE。`dexmani_policy/record_demo.py` 普通 best 可使用快照，显式 handoff 使用严格读取器。不要把只成功加载模型标成“已验证 held-out”。
5. **快照与证据一致性。** 评测从新 best 进入详细记录时，核对二者的权重与 inference；详细记录继续检查 selection 成功状态、候选/阶段等现有事实，并检查实际 held-out seeds。不因拆分而丢失已有 global_step 对照。显式传入旧 handoff 时只跟随该 handoff，不再读当前 best 别名以替换其选择。
6. **有限兼容。** 普通读取继续识别旧“仅引用”记录：引用可读时提取其中必要推理信息，不要求加载无关候选/阶段。旧 flat best 的既有配置 fallback 语义保留并明确提示其来源；新快照必须包含有效 inference，不得悄悄回退。只含引用且目标已丢失的旧文件无法恢复权重/推理参数，应明确报错并提示使用显式 checkpoint 和对应参数，不能猜 `latest`。不批量迁移历史实验。

必须保留的最小部署边界：实际 checkpoint、保存的模型/输入配置、EMA/raw、NFE、动作/观测维度和归一化。不要顺带修改 Real 的控制接口、RTC、动作长度、单位、关节映射或时序检查，也不需要改另一个机器人仓库。

**验收矩阵**：

| 场景 | 预期 |
|---|---|
| 新 best，详细 selection 报告完整 | 与旧成功解析结果的 checkpoint、EMA、NFE 相同 |
| 新 best，删除/不复制候选、阶段或整个详细报告 | 普通 `inspect_policy` 仍成功；需要证据的 held-out/显式 handoff 明确失败 |
| 普通 best 显式指定 weights 或 NFE | 按现有 override 优先级生效；未覆盖部分取 best，不丢失另一项参数 |
| 旧仅引用记录，目标存在 | 普通加载可取得原选择；严格评测仍执行原证据检查 |
| 旧仅引用记录，目标不存在 | 明确失败，不猜 checkpoint、不声称可自动兼容 |
| new best 快照与被引用选择不一致 | 评测拒绝；普通推理只按快照实际选择加载，不声称已核验证据 |
| 显式 handoff 与 checkpoint/EMA/NFE override 冲突 | 继续拒绝；无冲突时使用同一固定选择 |
| held-out seed 与 selection seed 重叠 | 继续拒绝；普通 best 推理不要求 seed 报告 |
| selection 发布技术失败 | 旧 best 不变；不发布不完整快照 |

用临时实验目录、最小 JSON 和标记 checkpoint 文件可验证解析与路由；标记文件不能证明 Torch 权重恢复。确需验证权重恢复时使用小型真实 checkpoint，不运行 selection rollout 或真机。

**收益/风险**：复制 `config.yaml + checkpoint + best_ckpt.json` 即可保持所选推理参数，减少普通部署对评测中间材料的依赖。风险中等，集中在共用调用者、参数覆盖和 held-out 语义；因此 writer/reader/调用者需一次改完整，单独提交。

## 5. 追加的三项

### F11：显式写入 SAT 当前 Beta 参数

**定位**：`dexmani_policy/configs/sat.yaml` 的 `agent`；默认值来自 `dexmani_policy/agents/core/sat.py::SATAgent.__init__`，传给本地 RectifiedFlow。

在现有 flow 配置旁加入：

```yaml
agent:
  # 与当前 SATAgent 默认值一致；显式保存本地时间采样配方。
  beta_s: 0.999
  beta_alpha: 1.0
  beta_beta: 1.5
```

这里只展示要加入的键，不复制整个 `agent` 节点覆盖原配置。`dexmani_policy/configs/ddp/sat.yaml` 继承主配置，不重复写第二份。保持 Python 默认值与已有旧配置加载行为；不扩充 `build_agent_contract`，不升级 strict resume 格式，不覆盖旧实验保存的 YAML，也不顺手调整 flow 训练网格或其他默认值。

**验收**：主配置及 DDP overlay 解析出的三值与修改前构造默认一致，实际传给 decoder 的值一致。沿用 config-only smoke 和配置检查即可，不为这三行新建测试框架、不跑训练。

**收益/风险**：阅读及保存配置能直接看到有效配方，当前模型行为不变。它是本地配置可读性改进，不是新发现的默认训练 bug。

### F45：padding 只解释一次，索引内核只生成索引

**定位**：`dexmani_policy/datasets/sampler.py::SequenceSampler.__init__/create_indices`。

**确认事实**：外层用原始 padding 计算最短 episode 长度，内层再 clamp padding。例：H=4、L=4、`pad_before=-1`，外层算最短长度 5 而拒绝，直接调用内层则 clamp 为 0 并生成一个窗口。合法默认窗口未由此发现错误。

**选定方案：拒绝非法输入，不做隐式纠正。**

1. 在 `SequenceSampler` 入口一次检查 sequence length 是正整数；两个 padding 是整数且满足 `0 <= pad < sequence_length`。沿用本文件现有 `Integral` 风格，允许 NumPy integer、拒绝 bool；不要把小数/字符串强转为整数。
2. 检查后统一用这些值计算 `min_required_length`、筛掉过短 episode、生成索引。过短 episode 的现有跳过/全空报错行为保持；不以改变短片段策略作为简化手段。
3. 当前 `create_indices` 在仓库内仅由 `SequenceSampler` 调用。确认这一点后将其作为内部 Numba 索引内核（建议名 `_create_indices`），移除其中 padding clamp 及重复的短 episode 策略分支；它接收已验证参数和已筛选 mask，只保留索引计算与必要一致性断言。不要新增公开 wrapper 或第二套 validator 来兼容一个未在仓库使用的低层入口。
4. 保持 Numba 内核，不把依赖 Python 类型判断的入口校验塞进 nopython 循环。除这个内部函数外，不扩大改名范围。

**验收**：H=1 的 padding 只能为 0；负值、`pad>=H`、小数、bool 明确失败；合法端点 0 和 H-1、两侧 padding、被跳过的短 episode、所有 episode 被排除的情况保持原语义。用独立窗口 oracle 验证合法输入的行序、重复边界与窗口数，不能只把旧实现复制成“期望值”。同时核对 F12 每个 key 的裁剪长度。

**收益/风险**：删除了两层对同一输入的不同解释；不承诺默认训练速度提升。低层非法参数行为有意变为明确拒绝，合法配置不变。

### F16：在现有文档中区分 RGB 主路径与 RGB-D 可选 API

**定位**：`dexmani_policy/agents/core/dp.py::DPObsEncoder.forward`；`dexmani_policy/agents/obs_encoder/rgb/__init__.py`、`base.py`、`image_processor.py`、`geometry_processor.py`；现有 `docs/项目架构.md` 的相关说明。

**实施**：在现有模块说明和架构文档的对应段落中，简短说明下面两条调用路径；不要新建一套 RGB 使用手册。

| 路径 | 输入和用途 |
|---|---|
| 当前 DP RGB 策略 | Dataset/确定性 eval 预处理 → `ImageProcessor.process_images` → backbone `forward` → `global_token` 与状态特征。已处理输入可为 uint8 或 float32，具体空间布局按现有接口。 |
| 可选 RGB-D 几何接口 | 显式传 depth/intrinsics 等 → `process_rgbd` / `backproject` / `GeometryProcessor`；供几何实验及已有模块示例使用，不是 DP.forward 自动执行的步骤。 |

链接到仓库里真实存在的示例，如 `dexmani_policy/agents/obs_encoder/rgb/resnet.py`、`dexmani_policy/agents/obs_encoder/rgb/dino.py` 的 `__main__` 示例。它们是模块内示例，不能编造不存在的 `examples/` 文件或命令。

保留所有公开方法、返回字段、导入路径与示例，不拆文件、不删除 GeometryProcessor，不将其描述为全局死代码。此项仅文档/docstring 变化；没有主路径计算开销可据此宣称被消除。

**验收**：说明与实际调用者一致，路径/链接可达，diff 不改变运行逻辑。无需下载 backbone 权重、运行 RGB-D 示例或添加行为测试。

## 6. 建议实施顺序与提交组织

1. **正确性先完成**：F10 → F07 → F01 → F44 → F08。每组用最小反例和正常路径对拍确认，再推进下一组。F44 的诊断字段兼容属于该修复的必要联动。
2. **处理局部简化和配置**：F45 → F12；随后 F38、F11。F45/F12 共用窗口 oracle，但每个补丁保持可独立审阅；F38 不与代码本格式改动混在一起。
3. **单独完成 DEP-01**：writer、普通 reader、严格评测 reader 和全部调用者一起修改，并跑表中路由矩阵。
4. **最后完成 F16 与相关使用说明**：DEP-01 改变稳定使用说明，因此同步修正 `docs/项目架构.md` 中“best 仅含引用”的表述及必要 CLI help；其他全局文档不为凑齐更新而重写。

每次提交只说明实际完成的项目、行为变化和验证结果。不要把 F11/F16 说成 bug 修复，也不要把 F12/F38 的机制收益写成已测吞吐收益。

## 7. 验收方式与完成标准

### 局部检查即可证明的内容

| 检查组 | 最小充分证据 |
|---|---|
| 数值边界 | 小型真实 Torch codebook 的非有限值拒绝及有限值对拍；Gaussian 配置/已标注 checkpoint 的拒绝；非 DQ 合法配置不受影响 |
| RGB | 小张量覆盖 dtype/layout 错误及三条合法输送路径；固定随机状态验证预处理结果 |
| Dataset | 临时小型 Zarr/内存数据；窗口 oracle、owned-memory 隔离、lazy validation、holdout/normalizer 隔离；strict resume 精确字段差异 |
| R3D | 实际 grouper 的点数边界和合法输出；不需要完整 ViT/backbone 或训练资产 |
| 导入与配置 | sklearn 隔离进程、SAT resolved config、对应 config-only smoke |
| Deployment | 临时目录中的实际 resolver/consumer、成功发布分支的定向检查；权重恢复与路径解析分开报告 |
| 文档 | diff、相对路径、参考链接、示例符号和命令核对 |

不要为单纯文档/同值 YAML 写实现镜像测试。必要回归测试应覆盖真实边界和跨层行为；若实施时仍无专用测试目录，可将这些定向检查合并到少量测试文件，例如 `tests/test_review_fixes.py`，标准库 `unittest` 已足够，无需新依赖。独立 oracle 用窗口位置的数学定义和边界重复语义生成期望值；不要要求执行者依赖本仓库以外的历史 review 临时脚本。

在仓库已有运行依赖可用时，使用现有入口做配置联动检查：

```bash
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise r3d sat maniflow multitask_dit ddp/sat ddp/dqrise
```

这个命令导入真实目标，但不等于完整运行/训练。若定向检查已证明改动，不扩展成长时间 smoke、DDP 训练、性能 sweep 或任务成功率实验。只有具体剩余风险需要时才运行对应的小型前向/恢复检查。

验收不允许用通用 `except` 吞错、放宽正常训练语义或强改数据来制造通过。缺 Torch、Numba、Zarr、权重或相应硬件时，准确记录 **NOT VERIFIED**；stub 只证明它覆盖的边界，不能代替真实库数值/后端检查。GPU 性能与任务质量没有测量时明确写“未测”，不作为本轮实施的前置门槛。

### 最终交付清单

- [ ] 11 项均有明确的代码/配置/文档变更，或有“当前 HEAD 已修复”的源码依据；未完成项说明具体原因。
- [ ] 五个必须修复项的触发反例被覆盖，正常支持路径保持既有行为。
- [ ] 默认训练配方、有效数据源行、normalizer 拟合范围、原型顺序和持久化状态没有被顺带改变。
- [ ] F44 不再为主训练提前构造验证 sampler；VQ 显式验证与 holdout 隔离保留。
- [ ] F12 不泄露共享可变缓存；F45 不改变合法窗口。
- [ ] F38 维持公开 API、state_dict/NPZ 格式；F11 仅显式化当前值；F16 仅修正文档。
- [ ] DEP-01 新 best 普通加载不依赖详细报告，同时 held-out 与显式 handoff 的现有证据要求仍成立。
- [ ] 提供逐项状态、实际验证命令及结果、兼容边界；只把实际执行通过的检查写为 PASS。
- [ ] 未运行训练消融、长评测或真机运动，未修改无关实验产物。

本任务书的事实基础是固定版本源码、配置、真实调用者与上述参考实现。这里的验收均为实施阶段要求，不是已经执行通过的测试报告。
