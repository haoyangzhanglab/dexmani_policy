# Codex CLI 任务书：让 DexMani Policy 简洁、正确、好用

日期：2026-10-26。审查基线：`45572589515b88560924632eaa20faff99cf9336`。

**本文是待执行任务书，不是已完成的修复报告。** 实施前必须读取当前工作区的 `AGENTS.md`，核对当前 HEAD 与本基线的差异。历史 review、约简复现及理论计算只提供线索；实际源码、调用链和本次运行的测试决定最终结论。不得为了符合旧报告而制造问题或修改正确代码。

## 1. 目标与范围

在保留现有九种策略和研究语义的前提下，修复有证据的错误，收敛重复实现和隐式状态，降低启动、加载及推理开销。保持项目是论文实验代码，不建设生产平台。

完成意味着：每项真实缺陷有最小修复和项目级回归；每项优化有采用或不采用的证据；有意的算法差异得到保留；外部环境缺失导致的验证阻塞如实记录。**不允许将“关闭检查”“删掉功能”“跳过测试”或“未发现新问题”解释为消除了所有问题。**

本次允许修改实现、相关测试、必要的配置和现有文档。默认不引入新的策略算法、真实机器人控制、AFRO 预训练任务、PEAfowl 多视角策略、通用插件系统或大规模实验调度器。现有公开训练/评测/部署入口应继续可用。

### 1.1 必须保留的研究不变量

- 数据时序保持 `obs[t]` 在执行 `action[t]` 之前；关节顺序、radians、绝对命令和控制维度不变。
- 普通策略从 `n_obs_steps - 1` 返回未来动作；LatentFlow 从 query 起返回完整 H 步。Policy 返回完整 `future_actions`，执行端消费前 C 步。
- 不改变已有 horizon、padding 监督、训练预算、global batch、normalizer 统计总体、默认 NFE/C；非必要不得改变 RNG 消费顺序。
- DP 的 FiLM 保留 `scale * x + bias`。不要替换为 `1 + scale`，不要补回未使用的 inpainting/local-conditioning 分支。
- ManiFlow 保留在线观测条件、EMA 动作教师、当前 consistency 目标、两个分支分别求均值后相加，以及本地非空/余数分组处理。
- optimizer、scheduler、EMA 按有效 optimizer update 推进；保留尾累计组和 DDP `no_sync` 的正确边界。
- 保留本地 frame-only PE；它是明确的架构适配，不是待修复的官方实现差异。
- LAMP/LatentFlow 的冻结 decoder 必须仍向预测 latent 反传；冻结权重不等于对整段解码使用 `no_grad`。
- 不用 `strict=False` 掩盖权重键或结构变化，不自动修写旧数据、旧码本、checkpoint 和评测事实。

数值缺陷修复或实例 RNG 隔离可能有必要改变新运行的轨迹。必须单独说明变化范围；不能声称修复前后训练逐位一致。

### 1.2 复杂度预算

优先使用现有模块和标准库。只允许为实际边界增加必要参数或少量元信息。

不要新增配置注册表、通用事件总线、插件工厂、实验谱系数据库、独立状态机、校验等级开关、缓存版本管理系统。不要为一个小 helper 建立新包。不要将不同算法强行合并成带大量模式开关的类。不要为了“全部优化”把没有实测收益的缓存、导出格式或 EMA 分支写进主路径。

必要的新增内容应限于：独立采样 Generator、少量证据摘要字段、针对真实问题的测试，以及确实消除重复的共享实现。每次引入新内容，都应删除或替代对应的旧逻辑，而不是叠加平行机制。

## 2. 参考项目：借鉴原则，不复制整个框架

| 参考与固定版本 | 借鉴 | 不照搬 |
| --- | --- | --- |
| [Diffusion Policy @5ba07ac](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f) | 显式 Generator、动作时间切片、直接的条件编码 | 本地不需要的 inpainting 分支；上游 workspace 的梯度累计/EMA 计步方式 |
| [ManiFlow Policy @ef2f116](https://github.com/geyan21/ManiFlow_Policy/tree/ef2f116f1f90163ed36e657b8c5503740bb468af) | Flow/consistency 数学目标、教师/学生职责 | 推理保存全部 ODE 中间轨迹、分别截断 batch 丢余数、在训练热路径 `.item()` 记录所有指标 |
| [AFRO @283024a](https://github.com/KolaKivy/AFRO/tree/283024a49e2c14bc030debf15383a4e7ee73836d) | 视觉预训练与下游策略分离、视觉/状态参数分组 | 把视觉变化 latent 当作手部动作 codec；在 `compute_loss()` 内更新教师 |
| [PEAfowl @ed95a37](https://github.com/Qingyu-Fan/PEAfowl/tree/ed95a3704e8060213a74f673f7d86b3d4a1a8fa4) | 相机几何、图像变换、跨视角对应显式传递 | 整套 ModelMixin/operation_order 构建体系；把先 dense `cdist` 再 top-k 称作稀疏计算 |

关键对照文件：DP 的 `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`、`model/diffusion/conditional_unet1d.py`；ManiFlow 的 `ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py`、`model/diffusion/ditx.py`；AFRO 的 `afro_workspace/policy/afro_policy.py`、`afro.py`；PEAfowl 的 `model/peafowl.py`、`model/geo_modules/geo_mv_fuser.py`。

使用本地已有参考副本或按固定提交读取必要文件即可。不把参考仓库复制进本项目，不依赖参考仓库的 main 当前状态，也不为了参考代码升级本项目依赖。

## 3. 审查事项与处理决策

编号沿用审查汇总；它们不是二十个必须增加新功能的需求。

| ID | 事项 | 本次决策 | 工作段 |
| --- | --- | --- | --- |
| R01 | VQ EMA 初始累计量与计数不一致 | 必修，修新预训练的数值缺陷 | B1 |
| R02 | Frozen Test 没有强制完整证据复核 | 必修，复用现有复核逻辑 | C1 |
| R03 | 独立源码副本误用父 Git 仓库身份 | 必修，记录实际代码身份 | C2 |
| R04 | Resume 新目录漏掉历史 milestone | 必修，保持完整候选集合且不覆盖旧 run | C3 |
| R05 | 多 runtime 共享全局 RNG | 必修，覆盖构造、加载、reset、采样、warmup | B2 |
| R06 | 训练/测试来源和产物身份的保证范围不清 | 补必要证据；未知来源明确为未验证，不能伪造独立性 | C1/C2/C4 |
| R07 | R3D 非默认 normalization 与固定 clamp 不匹配 | 限制不支持的新训练组合，不改变默认数值 | B3 |
| R08 | 推理读取完整训练状态及可信反序列化边界 | 优先 mmap/缩短驻留；明确仅加载可信训练产物 | E1 |
| R09 | 浮点有效性扫描未利用已选 episode | 消除不必要扫描，保持有效窗口和统计完全一致 | E2 |
| R10 | 零采样权重任务仍进入 normalizer | 保留既有定义，测试并写清楚；不新增加权统计模式 | D3 |
| R11 | PyTorch3D 的方法依赖不够明确 | 文档及方法级早期检查，不把原生依赖强塞入通用安装 | A/D4 |
| R12 | cfg 回写、恢复特例、配方硬约束和配置重复 | 收敛真实耦合；不引入新配置框架 | D1/D2 |
| R13 | DP/MultiTaskDiT 的 RGB+state encoder 重复 | 直接共享同一小模块，保留参数树 | D2 |
| R14 | obs_lr=0 冻结粒度过粗 | 保留全冻结含义；RGB 复用已有 tune_mode；不为未来 AFRO 增加闲置开关 | D2 |
| R15 | CLIP 初始化隐式占用默认 GPU | 明确设备，默认 CPU 构造/初始化，不抢 GPU0 | B3/D2 |
| R16 | 非主 rank 完整 EMA 副本 | 测量后决定；无明显收益则保留，避免新增 rank 专用恢复体系 | E3 |
| R17 | 每 rank 全量 Python 任务索引 | 小规模保持；大规模有收益时换紧凑表示，顺序不能漂移 | E3 |
| R18 | 主 Trainer 无统一离线 validation | 不新增训练循环；保留仿真选点。诊断通过已有 smoke/独立检查完成 | F |
| R19 | ManiFlow 条件投影可做推理期仿射合并 | 数值与端到端性能双门控；未验证不进入主路径 | E4 |
| R20 | 几何/触觉组件未完整接线 | 明确为研究组件，保护已有几何接口；不在本任务增加新策略 | D3 |

“保留”必须有理由及相应契约检查；“性能门控未通过”不是 bug 修复完成。缺少硬件/数据时，记录阻塞，不将本轮总体状态写成全通过。

## 4. 目标结构与数据流

保持现有目录主干，不做整体搬迁：

```text
输入配置 ──> Dataset / normalizer + 数据事实
        └─> Policy / optimizer / EMA + 模型事实
                         │
                 一处组装 effective_config
                         │
                       Trainer
                         │
                  checkpoint / logs
                    │          │
          selection + verifier  PolicyRuntime
                    │          │
              冻结测试       物理 future_actions
```

- Dataset 定义样本与统计，不决定机器人执行；Policy 定义算法，不写实验文件。
- Trainer 推进优化状态，不决定正式测试种子和 best 排名。
- 选点/正式测试共用同一份事实复核，不维护两套“近似相同”的验证器。
- Runtime 持有自身采样状态，不控制宿主进程的 RNG。
- `effective_config` 是一次构建的结果快照，不是供任意模块随时写入的共享总线。

## A. 开工基线与测试落点

1. 读取 `AGENTS.md`、`README.md`、`docs/项目架构.md`、`docs/仿真评测机制.md`、`docs/手部动作表示.md` 及实际调用方。检查 `git status --short`；保留用户现有改动。
2. 记录 HEAD、Python/Torch/CUDA、可用数据及预训练资源。检查当前代码是否已修复某项；已修复则补证据，不重复改动。
3. 优先使用已配置的 Conda `policy` 环境。检查 `requirements.txt` 与 `pyproject.toml` 的一致性及真实导入，不按记忆猜版本兼容性。出现依赖问题，先在隔离环境复现，禁止为跑测试全量升级依赖。
4. 建立少量 `tests/test_*.py` 回归。若仓库尚无测试框架，优先标准库 `unittest`；沿用已有 pytest 也可以，但不引入自制测试框架。轻量测试使用临时 Zarr、临时 Git 仓库、合成小 Tensor，不下载大模型。
5. 测试必须导入实际项目函数/类。不得把 AST 摘取函数、复制一份算法或 mock 整个内部模块后的结果当成项目级通过。可以在真正的外部 I/O 边界 mock 下载、仿真器和设备。
6. 基线测试允许揭示已知失败；先写能定位原因的失败回归，再修实现。测试数量和覆盖率百分比不是目标。

建议按数值/RNG、评测/恢复、数据/配置三个主题组织测试，不为每个小 helper 建一个目录。

## B. 数值与运行时正确性

### B1. 修复 VQ EMA 初始化（R01）

文件：`agents/hand_prior/vqvae.py`，DQ 预训练配置及手部表示文档。

问题是非零 `embed_avg` 与零 `cluster_size` 不一致；未命中 code 首次更新会除以接近 eps 的计数。不要靠增大 eps、截断输出或自动打开 kmeans 掩盖问题。

首选最小实现：非 kmeans 随机初始化使用一致的单位伪计数，`cluster_size=ones(K)`、`embed_avg=embed*cluster_size[:,None]`；kmeans 分支保持由真实 counts 与 centers 构造累计量。单位伪计数作为内部初始化约定，不新增可调超参数。核对零计数、死码替换和保存恢复仍自洽。

旧 checkpoint 加载必须恢复它已有的 buffer，不在 load hook 中“修正”历史统计。已训练策略的内嵌 codebook 不得替换；重新训练 VQ 的产物使用新目录，下游 Policy 需要重新训练。文档明确这是新预训练行为变化，不能宣称旧训练逐位可续。

验收：导入真实 `EMAVectorQuantizer`；低多样性首批、全 code 命中、持续未命中、kmeans、死码替换、state_dict 往返均有限；未命中 code 不出现初始范数数万倍放大；检查 encoder 的 straight-through 梯度和 quantizer buffer 更新时机。无需长训练来证明局部修复，也不得用这些测试宣称成功率改善。

### B2. 隔离实例采样状态（R05）

文件：`deployment/runtime.py`、Policy 的推理入口、`action_decoders/diffusion.py`、`flow.py`、`sampling/rtc.py` 及必要调用方。

- Runtime 持有设备匹配的 `torch.Generator`。在推理接口中增加一个明确的可选 keyword-only `generator`，贯穿所有实际随机采样点；训练噪声仍由现有训练 RNG 管理。
- 使用目标 Torch 版本支持的 `torch.randn(shape, device=..., dtype=..., generator=...)`，不要假定 `randn_like` 接受 generator。
- 包括 Diffusion、Flow、DQ-RISE、LatentFlow 和 RTC 初始噪声，以及 scheduler 中确实存在的随机分支。LAMP 无随机采样时不额外消费 RNG。
- `reset_episode()` 只重置本实例；不调用全局 Python/NumPy/Torch seed。加载/构造模型如果创建临时随机权重，也必须限定并恢复其影响，不能只修改 reset 而遗漏 `load_policy()`。
- warmup 保存并恢复本实例 Generator 状态，确保 warmup 不改变第一条正式预测。不要用每次预测临时保存整个进程 RNG 的包装器冒充并发隔离。
- 旧调用方省略 generator 的行为需明确；仿真模型采样可以使用每 episode 局部 Generator，但环境自身 RNG 的重置仍归环境执行器。不要无关改变训练 seed、数据采样 seed 和物理任务 seed。
- 这项修复不承诺同一 runtime 可被多个线程同时调用；scheduler 等可变状态仍要求串行使用。不要额外建设线程池或异步队列。

验收：真实小策略/decoder 上 A 单独预测与 A/B 交错预测一致；B 的构造、load、reset、warmup 不改变 A 或宿主 Python/NumPy/Torch 随机流；相同设备/seed 可重放，不同 seed 可产生不同噪声；CPU 必测，CUDA 有设备时验证。RTC 还要验证 prefix guidance 的 autograd 未被 `inference_mode` 或 generator 传递破坏。

### B3. 修补两个小边界（R07、R15）

R3D：当前实现对 normalized 点云固定 clamp。保持默认 `limits` 的行为，针对新训练，在方法级边界拒绝不兼容的 `identity/gaussian`，给出清晰错误；不要新增一个无人使用的 clipping 模式矩阵。历史非默认实验使用对应源码，不静默改变恢复后的数学行为。

CLIP：移除“CUDA 可用就隐式放到默认 GPU”的构造行为。默认 CPU，显式设备由调用方决定；cached task embedding 产出后随 Policy 迁移。配置检查不能下载权重或占用 GPU。默认闭集文本 checkpoint 恢复应不依赖原 CLIP 下载；未实现的开放文本离线恢复不要伪装为已支持。

验收：R3D 默认配置通过，两个不兼容新训练配置早期失败；模拟 CUDA 可用时，指定 CPU/非默认 GPU 不产生 GPU0 隐式分配；默认闭集文本恢复期间禁网仍可完成（使用本地小 fixture，不下载真实 CLIP）。

## C. 评测证据、源码身份与续训选点

### C1. 唯一正式证据复核路径（R02、R06）

文件：`evaluation/config.py`、`artifacts.py`、`selection.py`、`rollout.py`、必要的 `common/experiments.py`。

所有正式 frozen test 必须复用完整候选复核：completed 状态、计划任务/seeds、每候选事实完整性、从 episode 事实重算统计、固定排序规则、最终选择和推理配方一致。普通研究评测继续允许，不伪装为正式测试。

**注意现有依赖方向：`selection.py` 已导入 `evaluation/config.py`。不要从 config 顶层反向 import selection 而制造循环。** 优先把完整证据复核及其纯排名 helper 收敛到现有 `artifacts.py`，供配置解析、选点和独立 verifier 入口共同调用；移动而非复制实现。CLI 仍委托同一实现。

新增权重身份只使用一个标准 SHA-256 字段，记录于候选事实及选中证据；在正式使用前核对实际文件。哈希在候选冻结/实际加载边界计算，不在每个 episode 重算。它用于防止误替换，不声称是防恶意攻击的签名系统。文件名、global_step、配置相同不能代替内容身份。

旧记录没有完整身份信息时，不回填当前权重哈希后冒充历史选点证据；可进行普通评测，或明确重新选点。沿用“历史实验配对应源码”的约定，不建设多版本迁移层。

新正式测试还应核对选点时记录的代码内容身份、Torch/CUDA、参数 dtype 和数值设置；允许既有协议明确允许的 device 编号/输出位置改变。只把实际影响协议的字段作为比较条件，显示路径和 Git 备注不是第二份算法身份。

验收：缺候选结果、伪造统计、错误赢家、重复/缺 episode、seed 交叉、权重被同名替换均在仿真前失败；合法记录完成后才能发布 best；失败保留旧 best；搬迁实验目录后按相对引用工作；独立 CLI 与 pipeline 的复核结果相同；导入图无环。使用真实小 checkpoint 验证读写，不以空 `.pt` 作为模型恢复成功证据。

### C2. 记录正在执行的代码，不猜父仓库（R03、R06）

文件：`evaluation/artifacts.py`、远程同步/启动脚本中实际相关的部分。

- `code_version()` 必须核对 Git 工作树根与本项目源码根。无自身 Git 身份的 snapshot 不能继承父仓库的 commit/dirty。
- 对实际运行的研究代码计算稳定内容摘要。集合明确覆盖 Python 源码、实验配置、相关脚本、依赖声明；按相对路径排序并包含文件内容。排除数据、模型、日志、生成结果、`.git` 和摘要自身，不能沿共享数据符号链接扫描整块磁盘。
- snapshot 无 Git 时如实记录 `commit=null`，但仍有实际代码摘要。普通仓库可同时记录 commit/dirty；摘要才表明运行内容。内容未知时明确未知，不做静默猜测。
- 复用一个小哈希实现即可，不需要源码注册服务、签名清单、发布平台或每步追踪。
- 如果同步副本记录来源 commit，把它命名为来源说明，不能代替副本内容身份。远程脚本仍使用独立源码副本，不能为了方便 provenance 改回覆盖正在训练的共享目录。

验收：临时父仓库+被忽略的 snapshot 内放不同源码，不能报告为父仓库的干净副本；代码内容变化使摘要变化；仅日志变化不影响摘要；文件遍历顺序不影响摘要；无 Git 的合法源码目录也能工作。只用本地临时目录测试，不实际启动远程训练。

### C3. 保留新 run，继承有效 milestone（R04）

文件：`training/builder.py`、`workspace.py`、`common/experiments.py`、恢复和同步相关调用。

选择的方案是：**续训仍新建 run；在首次 optimizer update 前，将同一轨迹已产生的有效 milestone 归档为新 run/checkpoints 下的普通文件。** 保持现有选点路径和实验目录自包含，不引入跨 run 候选注册表，也不把旧 run 改为原地覆盖。

规则：

1. 只沿明确的 `resume_from` 来源读取，不扫描整个 experiments；候选必须与恢复 checkpoint 的训练配方匹配，且 step 不大于本次恢复 step。
2. 优先使用父 run 已归档的候选。对旧的多段续训，只有缺少应有前缀时才沿唯一父链补齐；每段 cutoff 随恢复点收紧，不能引入之后发生的另一条训练分支。检测缺失来源和环并明确报告，不能静默声称候选完整。
3. 在新目录中生成普通文件，不使用 symlink。同文件系统可用 hardlink，跨文件系统仅对 EXDEV 采用原子复制；其他错误不吞掉。沿用现有原子写入方式，严禁原位修改已继承文件。hardlink 不是独立备份，文档应说明。
4. 同一 step/里程碑且内容相同可去重；内容不同必须失败，不能按目录顺序任选。不要继承 latest 别名、普通临时保存或超出 cutoff 的 milestone。
5. 新 run 自包含后，选点不再依赖旧 run 的路径。多次续训应保留同一候选前缀。候选集合完整性与一次调用是否跑到总预算是两个概念；已有 ordinary evaluation 能力不取消。
6. 归档失败时不开始优化。只由 rank 0 写文件，其他 rank 在必要同步点等待，不增加后台复制线程。

验收：100k 计划在 50k 恢复，最终候选为 20/40/60/80/100%；从更早 checkpoint 分叉时排除未来候选；两次续训、跨文件系统、冲突内容、来源缺失、整个新 run 搬迁均覆盖。比较的是候选集合，不能把它称为 DataLoader worker RNG 的逐位恢复。

### C4. 数据身份与训练/测试独立性的诚实边界（R06）

继续复用现有 split manifest、source_id、episode/trial 身份，不再另建一份近似重复的 DatasetSpec。

存在任务/仿真配置可比的 episode seeds、trial IDs 或稳定来源身份时，做有意义的训练/selection/test 交叉核验；不能仅因整数 seed 相同就跨不同任务判定泄漏，也不能仅因 seed 不同就证明场景独立。

缺少这些信息时，结果明确记录“训练/测试来源独立性未验证”；保持 `heldout_from_selection` 只说明 selection/test 隔离。数据配方摘要不能标作数据内容 hash。没有生成端内容版本时，不在每次训练启动对完整 Zarr 做昂贵哈希，不制造无法支持的内容不可变保证。

用少量现有记录字段说明证据及保证范围即可。不要新增强制审批或多级证据状态机。任何仍需用户提供原始采集信息的项目都列为外部验证阻塞，不列为已证明无泄漏。

## D. 收敛配置、构造与重复实现

### D1. 一处组装有效配方（R12）

文件：`training/data.py`、`builder.py`、`common/config.py`、`checkpoint.py`、相关 Policy/encoder。

让数据构建返回 Dataset、normalizer 和必要的数据事实；让模型暴露实际模型事实。由 builder 在一个明确位置合成最终 `effective_config`，冻结为本次运行快照再交给 Trainer/Workspace。不要让 Dataset/decoder 通过修改传入 cfg 偷渡运行状态。可以用现有 dict/小 dataclass，不新建通用配置体系。

保存 constructor 默认值及真实 RGB architecture 的目的仍需保留。不要删除 bind_defaults 后重新依赖未来源码默认值，也不要每次 checkpoint 重新解析 YAML。

恢复只使用保存配方与完整权重。具体 RGB/文本/先验初始化知识归所属模块；fresh-only 权重加载优先统一到现有 `initialize_training()`。默认闭集恢复应无需原训练数据或外部先验文件。若迁出通用恢复特例需要一个窄的 Policy hook，它必须替代旧分支并实际减少耦合；禁止为了去掉几行 if 新建工厂/注册表/多层 adapter。不要用新构造 flag 同时控制 train/eval、设备、是否加载权重等多个不同概念。

只合并稳定重复的运行/日志默认值；最多保持简单的公共配置+方法覆盖，不把方法独有的数据与优化设置强行统一。更改配置组织后，比较 resolve 后的九种默认研究配方，除明确批准的修复外应一致。

SAT 的默认 C=8 可继续留在 YAML；如果构造函数仅因配方写死 C，而数学结构不需要，则改成共同的有效未来窗口检查。H/关节结构等真正的架构约束保留。不要误把 Dataset pad_after 改成评测 C 的函数。

### D2. 小模块复用与显式初始化（R13、R14、R15）

- 将 DP 和 MultiTaskDiT 重复的 RGB+state observation encoder 合并为一个清楚命名的小模块，放入现有 obs_encoder 目录。保留 `backbone`、`state_mlp` 等参数路径。文本仍由 MultiTaskDiT Policy 处理。
- 不把 LAMP global pooling 与 LatentFlow spatial token 两种不同输出强行统一。
- 保留 `obs_lr=0` 表示冻结整个 observation encoder；视觉-only 冻结优先复用 RGB 现有 `tune_mode=freeze`，状态 MLP 仍可训练。没有实际 Sparse3D 预训练调用方时，不增加闲置的 `vision_lr` 或多阶段优化配置。
- `requires_grad=False` 与 eval mode 分开决定；验证冻结视觉的 dropout/BatchNorm 行为，不能全局 `train()` 后悄悄开启本应冻结的统计。
- fresh 初始化的设备显式；完整 checkpoint 恢复不临时下载模型或占用未请求的 GPU。

验收：共享 encoder 前后在相同 state_dict、输入和 eval 模式下输出一致；权重键一致；raw/EMA 往返；视觉-only 冻结时 state MLP 有梯度，视觉模块无梯度；现有全冻结含义不变。

### D3. 用明确契约关闭非 bug，而不是扩功能（R10、R20）

零采样权重任务仍进入 normalizer 的现有定义保持，补一个测试和文档说明；用户要完全去掉任务时应从 dataset 列表移除并重训/重算统计。不要悄悄改变旧码本坐标。

FeatureAlignment/HandScene/SceneAggregator/触觉模块继续标为独立研究组件。清理确定无引用的过时说明，但不盲删研究代码，也不把它们接入默认九种策略。已有 projection/RGB transform 的小几何测试应保护 resize/crop 半像素约定、点序和 mask。文档说明：未来融合需要传递相机矩阵、raw-to-final image transform、有效性和一致点序；native ResNet 特征格点不能未经核对就当 uniform ViT grid。

不新增 PEAfowl 式 depth teacher、跨视角循环或 AFRO 式 latent pretrainer；这些改变研究问题，另立实验。

### D4. 方法依赖与可用性（R11）

README 和方法配置检查明确：Sparse3D 的 patch FPS 需要与 Torch/CUDA 匹配的 PyTorch3D；R3D 及其他实际触发 FPS 的路径也需按调用核对。DP3 输入点数不触发 FPS 时，不应因为其他策略需要 PyTorch3D 而无条件导入它。

缺原生依赖时，在方法构建/首次必需边界给出可执行的说明；不要每次 forward 重复检查安装情况。不要未经测试同时把多个依赖 pin 改为 latest。`requirements.txt` 和 `pyproject.toml` 中确有不一致或不可安装组合时，给出隔离环境证据后最小调整。

## E. 性能优化：先移除确定浪费，再让测量决定剩余改动

### E1. Checkpoint 加载（R08）

优先验证目标 Torch 2.4 环境中的 mmap 加载：对本仓库常规 zip checkpoint 使用按需映射，并在模型构造和搬到 GPU 前释放未选择的 raw/EMA、optimizer、scheduler 引用，避免无关状态长时间驻留。注意 mmap 不等于不反序列化 metadata，也不等于安全加载；`weights_only=True` 也不是筛选 optimizer 的开关。

完整训练 `.pt` 仍限可信本项目产物。不要对不可信文件自动回退到 `weights_only=False`，不要宣传完整训练格式可安全接收外部上传。

先测冷加载时间、peak RSS、GPU 峰值和恢复一致性。若 mmap/对象生命周期已解决实际问题，就不新增导出格式。只有部署确有需求且测量仍表明训练容器成本显著，才做一个显式离线推理导出（现有 safetensors+最小 JSON 元信息即可），不在每个 milestone 自动保存两套文件。若采用导出，需保留归一化、内嵌手部表示和保存结构的完整恢复，并有原格式/导出格式一致性测试。

### E2. 只扫描会消费的 episode（R09）

在 split 确定后，按 train/val episode 并集、Zarr chunk 连续区间做浮点有限值检查。不要只扫训练而导致后续验证漏检；LAMP 历史也不能跨被排除的 episode。uint8 RGB 沿现有轻量检查。

保留逐角色有效性：observation 只检查实际观测窗口，action 检查实际动作窗口；不要把未扫描行默认当作有效数据。normalizer 继续统计有效训练源行去重集合，LatentFlow relative command 继续按 query window/reference 统计。

验收：含 NaN/Inf、短 episode、首尾 padding、excluded episodes 的小 Zarr，优化前后 train/val indices、窗口、源行集合及 normalizer 一致；spy 实际读取区间，证明排除块没有被不必要读取。不给数据加载再加一层持久缓存和缓存失效协议。

### E3. EMA 和采样索引（R16、R17）

这两项不是默认必改项。

非主 rank 只需要 EMA decoder 的事实先通过调用链确认并测量 encoder 副本成本。只有内存/更新收益明显且能维持同样更新计数、DDP 初始化和 strict restore，才将 teacher 参数传递收窄；rank 0 仍保存完整 EMA Policy。不要通过改用 EMA encoder 条件来“利用”已有副本，这会改变算法。缺两张 GPU 的集成验证时，不把此项标为完成。

MultiTaskSampler 可将全量 Python int/list 改成紧凑数组，但必须逐索引比较相同 seed/epoch 的输出，包含 balanced/proportional/weighted、零权重、各 rank 分片和 resume cursor。若改变随机数消费或配额结果，不能以性能优化名义合入。只在实际数据规模或代表性规模上证明 CPU 时间/RSS 收益后采用；不建设共享索引服务。

### E4. ManiFlow 推理期仿射 K/V 合并（R19）

当前单请求 K/V 缓存应保留。默认路径中 `context_embedder` 与各层 K/V linear 之间只有 frame embedding 加法，可在冻结推理阶段组合：

```text
z = Wc * context + bc + frame_embedding
KV_l = Wl * z + bl
     = (Wl * Wc) * context + Wl * (bc + frame_embedding) + bl
```

只适用于中间没有时间依赖/非线性变换的路径；`pre_norm_modality=true` 不适用。K norm 仍在合并后的 K 上执行。不要预计算训练时会变化的权重，不跨观测请求复用观测 K/V，不把派生权重存入训练 state_dict。

先做真实 backbone 的 fixed-noise 整段输出对照，再测 encoder、KV prepare、decoder sampling、端到端时间和峰值显存。包含 NFE=1/4/10、代表性 B 和点数，固定 dtype/device，GPU 测试正确同步、预热并重复，分别报告冷/热开销；不能只测一个线性子图。

理论上默认 K/V 投影主项可从约 28.99G MAC 降到 7.25G MAC，但输出 K/V 缓存大小不变，也不能推出四倍整体加速。只有收益超过测量噪声、缓存失效/设备 dtype 变化处理仍简单、数值和内存无不可接受回归时才合入。否则删除原型并记录不采用的原因，不留一个默认关闭的复杂优化分支。

## F. 验证、交付与停止条件

### F1. 分层验证，而不是一句“测试通过”

| 层级 | 必做内容 | 缺环境时如何处理 |
| --- | --- | --- |
| CPU 回归 | 上述数值/RNG、证据、临时 Git、候选归档、数据和共享模块测试，导入真实实现 | 核心依赖无法导入必须记录失败/阻塞，不用假模块替代 |
| 配置/导入 | 九种现有默认策略的配置解析；实际方法依赖检查；原 CLI `--help` | 记录具体方法和缺失依赖，不能声称九种均通过 |
| 短训练/恢复 | 实际小数据、raw/EMA 保存恢复、采样一致性；新训练和 resume | 可用真实本地资源时执行，禁止无授权长训练 |
| GPU/DDP | 目标 Torch/CUDA；涉及的两个 rank 更新和恢复；RTC、compile 相关路径 | 无硬件则明确未验证，涉及 DDP 的优化不得默认启用 |
| 仿真/真机 | 固定开发 seeds 的小 rollout；不使用 frozen test 调参 | 无 dexmani_sim 则阻塞；本任务不操作真实机器人 |

“config-only 通过”不能证明前向、反向、点云算子或闭环成功。CPU 子图等价不能证明 GPU 整体速度。测试 skip 不算 pass。

### F2. 推荐命令与执行边界

从仓库根目录运行，并先核对当前入口实际参数：

```bash
git status --short
python --version
python -m pip check
python -m unittest discover -s tests -v
python dexmani_policy/smoke_test.py --config-only dp3
python dexmani_policy/smoke_test.py --data-only dp3
python dexmani_policy/smoke_test.py dp3 --max-updates 4
git diff --check
```

后两条需要相应本地数据，训练 smoke 需要其实际要求的 CUDA/资源；不能假装当前环境具备。按改动覆盖其他方法，不是只测 dp3 就代表全仓库。对改过的 shell 执行 `bash -n`；远程脚本只做本地语法/临时夹具验证，不实际 SSH 启动任务。不要修改外部 dexmani_sim/dexmani_real 来掩盖本仓库接口问题。

若已有测试框架不是 unittest，使用项目现有命令并在结果中说明。不要为了满足上述命令形式重写已有有效测试体系。

### F3. 提交顺序

建议小提交顺序：

1. A+B：实际回归、VQ 数值修复、RNG 隔离及小边界。
2. C1+C2：统一证据复核及实际代码/权重身份。
3. C3+C4：续训候选归档及数据独立性范围。
4. D：去重、明确初始化、收敛配置和依赖说明。
5. E：确定的数据/加载浪费；通过测量门控的性能优化。
6. F：最终集成、文档和结果清单。

允许按真实依赖调整顺序；不要求机械凑六个 commit。修改前后运行相应测试，避免未验证的大批量迁移。未经用户另行授权，不 push、不建 PR、不改分支保护；可在独立本地分支形成仅包含本任务的提交。保留其他人的未提交改动。

### F4. 最终交付

更新受影响的 README/现有技术文档，避免再维护一份重复的项目架构。提交必要测试；大模型、数据、日志、benchmark 原始产物不进入 Git。

新增一个简短的 `docs/CODEX_REFACTOR_RESULTS.md`，逐项覆盖 R01–R20，列出：

```text
ID | 已修复/保留正确行为/门控未采用/外部验证阻塞
源码位置 | 实际修改 | 测试命令和结果 | 行为/兼容性变化 | 未验证范围
```

性能项额外记录硬件、版本、输入规模、dtype、NFE、计时范围和前后数据；没有实测则写未测。明确哪些旧实验必须保留原源码、哪些新产物需要重新训练。

停止条件：所有必修缺陷已修并完成可执行的项目级验证；相关集成验证完成或明确阻塞；所有低优先级条目有可审查的采用/不采用理由；最终 diff 无无关更改，没有为了这次重构新增多套协议、配置来源或缓存体系。

**不要仅更新本文、写设计总结或把所有项改成“已完成”后结束。执行任务的目标是代码落地、验证和准确交付；本任务书本身不代表任何测试已运行或问题已修复。**
