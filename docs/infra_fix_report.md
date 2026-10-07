# 基础设施修复报告

日期：2026-10-04。执行依据：根目录 `CODEX_INFRA_TASKS.md`。

## 当时验收状态（2026-10-04）

截至本轮工作区验收：基础设施修复已实施，完整回归 32 项中 30 PASS、2 CUDA SKIP；13 个配置、全部 16 个 shell 的逐文件语法检查及 Python 编译通过。不继承系统包的 CPU venv 已完成安装、依赖检查和默认 DP/DP3 构造；实际 CUDA/NCCL、CUDA 扩展及模拟器 rollout 仍未验证。

本文按实施轮次保留证据。下面的命令、计数和环境限制是历史记录；后续已删除旧 RNG 列表与无版本 resume contract 的转换，`read_best_ckpt_json()` 也已移除。当前恢复边界见 [README](../README.md#训练)，不要用本文的兼容承诺或沙盒内 CUDA 不可见推断当前行为。

## 初轮基线与实际范围

- 起始 HEAD：`5d189e14b37fcb72b256f2a8cd28706d2b047263`；分支 `main`；起始工作区干净，任务书已在本地，无需 fetch/reset。
- 已读取根 `AGENTS.md` 和完整任务书；未发现适用于修改目录的更深层指令。相对审查基线 `141b0337…` 仅新增任务书，所述缺陷仍存在。
- 按 A → B → C → D → E → F 实施 F01–F19、G01/G02。保留现有 Trainer、采样分布、动作表示、训练预算、EMA/scheduler step 语义和选点排名。
- 保留已有正确行为：最终评测独立目录、严格 saved-config 推理、原子 checkpoint、正常失败与 fatal error 的区别、sync_down 首遍不保留 partial。没有重复替换这些机制。
- 没有修改 `AGENTS.md`，没有启动真机、远程训练、长训练或真实仿真，没有提交或推送。训练回归仅使用临时目录中的小型合成 CPU 模型；没有改写已有实验/数据/权重。

实现状态与环境验证分开：下表的 **IMPLEMENTED 不代表 GPU 或真实模拟器验收通过**。

## 初轮逐项处置

| 编号 | 实现状态 | 修改位置及行为 | 实际验证与边界 |
|---|---|---|---|
| F01 | IMPLEMENTED | `datasets/multi_task_dataset.py`：pickle 排除 Manager owner，保留 proxy；owner PID 限制 shutdown，重复 close 安全。 | PASS：真实双 spawn persistent workers，balanced/weighted/proportional 的 epoch 0→1→0 与父进程索引一致；num_workers=0；重复清理。完整 NCCL 训练 NOT VERIFIED。 |
| F02 | IMPLEMENTED | `utils/config.py`、所有基础 YAML、`training/run_identity.py`、workspace、两条训练入口：父进程 run ID，永久独占认领，W&B ID 包含认领身份，恢复写新目录。 | PASS：跨启动 ID 不同、双进程同目录仅一个成功、已有配置字节不变、真实 CPU 恢复不写来源目录。保留 Hydra 启动文件不在认领事务内的边界。 |
| F03 | IMPLEMENTED | `scripts/remote/train_remote.sh`：独立 session/log，移除启动 kill，日志 noclobber，复合任务名符合 stop 字符集，保留远端启动非零退出码。 | PASS：完整脚本经 ssh/rsync 替身执行两次；不 kill、名字不同、旧日志不截断，训练 exit 7 留在日志，启动 exit 17 传回。未连接真实服务器。 |
| F04 | IMPLEMENTED | `requirements.txt`、`pyproject.toml`、README、pointcloud ops：对齐实际直接依赖，Hydra 1.3.x、peft 明确声明，PyTorch3D 按使用延迟导入。 | PASS：独立 venv 复用现有软件包的 editable 安装、pip check、全部声明依赖导入、13 个配置、默认 DP/DP3 CPU 构造、CPU FPS。从零索引解析失败；全新环境安装及 CUDA 扩展 GPU 执行 NOT VERIFIED，见下文。 |
| F05 | IMPLEMENTED | `training/resume.py::build_train_loader`：在恢复游标应用前，仅拒绝实际混合大小的累积组。 | PASS：5/2/3 拒绝、4/2/3 与 5/2/2 允许、A=1/drop_last 允许；等大累积与等效大 batch 数值一致；中断恢复样本顺序不变。 |
| F06 | IMPLEMENTED | `datasets/replay_buffer.py`：共享 episode 元数据校验，先检查整数/维度/正值/递增及读取字段时间维，再读大数组；内存 root 同样检查。 | PASS：回退、重复、空、浮点、bool、越界等拒绝；替身证明非法边界未读取 data 数组；合法短 episode、padding、val split 可用。没有判定用户数据已损坏。 |
| F07 | IMPLEMENTED | ReplayBuffer/BaseDataset、`training/build_utils.py`、`training/resume.py`：加载时捕获 revision，根级 `data_identity` 快照与 contract，逐任务比较。 | PASS：实际临时 Zarr revision 进入构建快照；同值通过、改变/丢失拒绝、历史缺失警告、多任务比较；不注入 dataset kwargs。仓库未发现数据生成器，未新增转换 pipeline。revision 不等于内容 hash。 |
| F08 | IMPLEMENTED | `evaluation/protocol.py`、selector、final eval、demo、共享 loader：有效快照、展开环境参数、实际 runner 输入、最终解析参数、具体权重/step、版本及结果引用。 | PASS：真实 build_eval_runner 的保存输入覆写、table/instance 参数展开、CLI 优先级、single/sweep 快照与实参、独立输出。当时只固定函数内部读取的权重，CLI 重复读取缺口在后续轮次补齐。真实 rollout NOT VERIFIED。 |
| F09 | IMPLEMENTED | `env_runner/multi_task_sim_runner.py` 与共享协议：任务顺序、有序池 SHA256/长度、所有已派发候选/阶段的真实 task/seed 请求排除集合。 | PASS：实际 runner 映射方法；非 reference 池重排、任务重排拒绝，固定池及旧单任务通过，最终真实集合无交集。旧多任务缺池身份不能声称 held-out。 |
| F10 | IMPLEMENTED | `scripts/remote/sync_down.sh`：新增小型 JSON/YAML/result checksum pass；保留大文件 ignore-existing、metrics/latest 更新及 symlink。 | PASS：用脚本实际 rsync 参数跑本地两轮，同大小同 mtime 的 best 更新，summary/config 可同步，大 checkpoint inode 不变，latest 保持 symlink；首遍无 partial，无 delete。 |
| F11 | IMPLEMENTED | selector：保存每候选每阶段 episode_details、请求及完成状态、统计；视频按 run/候选/阶段/task 隔离；CLI 默认关闭，`--videos` 开启。 | PASS：两个候选、两个阶段路径不同，明细可重算成功数/分母；无录像调用不生成视频。真实编码/仿真录像 NOT VERIFIED。 |
| F12 | IMPLEMENTED | selector、`evaluation/protocol.py::atomic_json`、`agents/loader.py`：成功 summary 后原子发布 best；失败保留旧指针并重新抛出异常；新指针严格核对 summary/step/seed 证据。 | PASS：成功 A→全零/fatal B 保留 A→成功 C；os.replace 失败不产生半个 best；缺 summary 明确报错。SIGKILL 不保证留下失败 summary，旧指针保持原子语义。 |
| F13 | IMPLEMENTED | `training/run_identity.py`、单卡/DDP 父入口：外部 resume 以项目根解析，展开 `~`，存在性检查，目录解析 latest；保存来源路径。 | PASS：相对目录/文件、绝对目录/文件、`~`、不同 cwd、缺来源及内部 latest tag；CPU checkpoint 恢复至新目录。 |
| F14 | IMPLEMENTED | `scripts/training/train_vq_hand.py`：train/val 的 enc/vq/MSE 按样本数累计；局部 `evaluate_vq` 供真实训练与回归共同调用。 | PASS：257 样本反例，batch=256/257 都为约 0.389105，均优于 MSE=1 候选。没有运行完整 VQ 训练或改变 usage/perplexity 定义。 |
| F15 | IMPLEMENTED | `training/trainer.py::apply_gradient_step`：统一 error_if_nonfinite 裁剪；失败诊断后保留原异常，不推进状态；移除失效 fast 开关的 Trainer 参数与入口透传，历史配置仍接受并忽略。 | PASS：开启/关闭裁剪均拒绝 NaN/Inf；开启裁剪还拒绝约 1e20 的有限梯度总范数溢出，权重、optimizer、scheduler、EMA、step 不推进；正常梯度通过。旧 fast 两取值构建相同 resume contract。 |
| F16 | IMPLEMENTED | Diffusion、四处构造点、五个基础 YAML、BaseAgent、normalization、builder、loader、resume contract：严格 bool clip_sample，配置预检查和真实 scheduler 检查一致。 | PASS：sample/epsilon/v_prediction 的 ±3 oracle、默认回归、五条 Agent 路径透传、Gaussian 动作拒绝/允许矩阵、Gaussian 观测与 flow 放行、raw/EMA 恢复、missing≡true、true≠false。实际大模型 GPU 输出 NOT VERIFIED。 |
| F17 | IMPLEMENTED | `utils/random.py`、Trainer、resume、两条入口和 smoke：每 rank 显式实际 device 的 CUDA RNG Tensor；旧 list 用来源配置确定旧槽位。 | PASS：CPU 流、mock 非零设备捕获/重映射、旧单卡/DDP 槽位选择、未知来源拒绝。两项实际 CUDA/NCCL 测试已提供，本机均 SKIP/NOT VERIFIED。 |
| F18 | IMPLEMENTED | `agents/vq_hand/codebook_manager.py`：NPZ/state_dict 共同校验；转换前检查整数/permutation，补 affine 有限/非零；NPZ 局部准备后统一提交。 | PASS：合法 NPZ/state_dict roundtrip 查表、空初始化及 EMA；坏 pose/weights/shape/幂关系/permutation/整数元数据/hand range/scale/metadata 拒绝；坏 NPZ 不改原对象。 |
| F19 | IMPLEMENTED | `deployment/runtime.py`：按 config+checkpoint 递归发现，支持可变深度；相对选择限制在 experiments 内，绝对路径保留。 | PASS：普通三层与 DDP 四层发现、选择及 config-only inspect，错误路径拒绝；不加载模型或真机。 |
| G01 | IMPLEMENTED | `evaluation/protocol.py::wilson_interval` 和统计序列化：每任务实际分母及 95% Wilson；零分母 null。 | PASS：0/0、0/n、n/n、一般成功数；micro/macro 和排名未改变。区间不代表训练 seed 方差。 |
| G02 | IMPLEMENTED | `training/build_utils.py::build_normalizer`：joint action:auto 单块直接 fit、多块 fit_field_chunks。 | PASS：单/多块及常量维数值等价，patch concatenate 确认该路径不分配拼接数组。action_ee mixed 逻辑与统计范围保留；不声称未经 profile 的速度提升。 |

上述源码路径除 `scripts/` 等显式前缀外，均相对于 `dexmani_policy/`。

## 初轮实际命令与结果

### 环境与安装

系统 shell 默认没有 `python`，使用已有 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`。没有升级或替换该环境。

实际执行：

```bash
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m venv --system-site-packages /tmp/dexmani-infra-env
/tmp/dexmani-infra-env/bin/python -m pip install --no-index --no-build-isolation -e . -r requirements.txt
/tmp/dexmani-infra-env/bin/python -m pip check
```

**PASS**：editable 安装完成，`No broken requirements found`。这是隔离项目安装、复用原环境软件包的验证，**不是全新空环境安装**。

实际环境：Python 3.10.20、Torch 2.4.1+cu124、torchvision 0.19.1+cu124、Hydra 1.3.5、OmegaConf 2.3.1、diffusers 0.27.2、Zarr 2.18.3、transformers 4.48.0、huggingface-hub 0.25.2、peft 0.20.0。requirements 记录此次实际可导入的直接依赖，未用新算法或关闭 LoRA 绕过安装问题。

其他实际检查：

- **PASS**：requirements 中全部直接依赖对应模块导入；小型 Bert+LoRA 构造。
- **PASS**：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 下，`hydra.utils.instantiate(load_config(name).agent)` 构造默认 `dp`（99,245,651 参数）及 `dp3`（68,762,131 参数），未下载权重、未启动训练。
- **PASS**：实际 PyTorch3D CPU FPS，输入 `[1,8,3]`、输出 `[1,4,3]`。
- **FAIL（环境解析）**：`python -m pip install --dry-run --ignore-installed --no-cache-dir --timeout 5 --retries 0 -r requirements.txt`，当前可用索引返回 `No matching distribution found for torch==2.4.1`。因此全新环境下载/安装 **NOT VERIFIED**，不能据此宣称该组合已从公共索引完整重建。

### 定向回归

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/dexmani-infra-env/bin/python -m unittest discover -s tests -p 'test_infra_*.py' -v
```

最终结果：**27 项，25 PASS，2 SKIP（NOT VERIFIED：实际 CUDA）**；`OK (skipped=2)`，耗时约 11.4 秒。测试文件按 launch、training、evaluation、codebook、resume、CUDA 分组，均调用仓库真实实现，未引入新测试框架。

CPU 连续/恢复对照为 6 个 optimizer steps，使用 gradient accumulation=2，在第 2 步中断并恢复。权重、EMA、Adam、scheduler/LR、step、样本顺序完全一致，来源目录全部文件字节不变。

真实 Manager spawn 在初次沙箱内执行时因本地 socket 被禁止而失败；经自动审查批准在沙箱外执行后通过。评测测试中新加的可选模拟器导入/结构化配置 fixture 曾失败，已修正 fixture 后重跑整套通过，没有为通过测试放松生产校验。

### 配置与静态检查

```bash
/tmp/dexmani-infra-env/bin/python dexmani_policy/smoke_test.py --config-only \
  dp dp3 dqrise r3d multitask_dit maniflow sat \
  ddp/dp ddp/dqrise ddp/r3d ddp/multitask_dit ddp/maniflow ddp/sat
python3 -m compileall -q dexmani_policy scripts/training tests
bash -n scripts/remote/train_remote.sh scripts/remote/stop_remote.sh \
  scripts/remote/sync_down.sh scripts/eval/select_best_ckpt.sh
git diff --check
```

**PASS**：13 个实际基础/DDP 配置 compose/resolve 与 Agent import、语法编译和 diff 空白检查。上面的单条 `bash -n` 命令只检查第一个脚本，后续轮次改为逐文件检查并覆盖全部 16 个脚本。配置 smoke 不等于数据训练、完整模型前向或仿真成功率验证。

## 兼容规则与新操作

1. **安装**：README 给出 requirements → editable install → pip check 顺序；PyTorch3D 按目标 Torch/CUDA 单独安装。已有 policy 环境及后续 CPU 空环境安装均已验证；CUDA 扩展仍须在目标环境复验。
2. **恢复与目录冲突**：`+resume_from=experiments/<policy>/<task>/<run>` 或 checkpoint 文件；相对路径以项目根为准，支持 `~`。始终写新目录，记录来源；不要删除运行标记来接管旧输出。W&B 身份含运行认领 token，避免 sweep 的 `0` 等重复 basename 混写。
3. **Gaussian**：`normalization.action=gaussian agent.clip_sample=false`；旧 limits 缺开关按 true。只对已知 Diffusion 语义做这项缺省兼容，其他严格字段仍检查。旧 Gaussian+true 不静默纠正；忠实复现需旧版本，修正结果另记实验。
4. **选点与 best**：每次 selection 都有独立 ID、成功/失败 summary 及有效配置。best 是最近一次成功发布结果；失败不删旧 best，也不继续 pipeline。缺 summary/权重直接报错。旧 best 可作为历史推理记录，来源缺失会警告；旧多任务 held-out 必须重新选点。
5. **录像与统计**：selection 默认关闭录像，`--videos` 开启独立候选/阶段目录；demo 独立加载选定权重。每任务 Wilson 区间使用实际完成分母，不改变选点规则、不替代训练 seed 方差。
6. **数据身份**：新数据使用新路径、新非空 `data_revision`。该身份在加载时捕获；已知值改变/丢失拒绝恢复，历史未知身份不反填为已验证。没有 producer 代码改动，因为当前仓库没有相应转换入口。
7. **RNG**：新记录只保存该 rank 训练设备的一份 CUDA 状态，映射到当前目标；旧 list 必须有可信旧槽位证据，唯一槽位且无矛盾时可确定。信息不足不猜精确恢复，权重推理仍可用。world_size 不可借此改变；worker augmentation/prefetch、硬件及非确定性算子边界保留。
8. **同步与远程启动**：session/log 不再复用或启动时 kill；使用输出中的 stop 命令。sync_down 对小型可变结果 checksum 更新，大 checkpoint 不全量 checksum，不 delete；latest 保留符号链接。

## 初轮环境阻塞与复验

- `torch.cuda.is_available()` 为 False，device_count=0，`nvidia-smi` 无法连接驱动。**NOT VERIFIED**：真实 CUDA 映射、可见 GPU 数减少、非零设备短更新、NCCL/DDP 恢复、完整 GPU policy smoke。
- 默认 dp/dp3/dqrise/r3d 数据目录实际存在；没有据此宣称数据内容或真实训练已验收。完整数据路径 smoke 因 GPU 条件不足未执行。
- `dexmani_sim` 未安装。**NOT VERIFIED**：真实 rollout、成功率、视频编码、模拟器随机化实际效果。替身 runner 只验证参数流和协议。
- 初轮包索引未能解析 Torch 2.4.1；后续确认沙箱网络限制并完成 CPU 空环境安装，此项不再是当前安装阻塞。目标 CUDA 扩展仍未验证。

在具备对应条件的环境复验：

```bash
# 先按 README 安装/激活环境，再验证依赖及定向回归
python -m pip check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_infra_*.py' -v

# 两张可见 GPU：真实 RNG 跨设备及两步小模型 NCCL 中断/恢复
CUDA_VISIBLE_DEVICES=0,1 python -m unittest discover -s tests -p test_infra_cuda.py -v
# 改用其他物理 GPU 映射时可重复上述命令；这不验证改变 world_size。

# 有真实数据/权重/GPU时：实际配置 smoke，不改变研究超参数
python dexmani_policy/smoke_test.py dp dp3 dqrise r3d multitask_dit

# 模拟器就绪后，以一个明确选定的实验进行少量评测（未在本次执行）
python dexmani_policy/eval_best_ckpt.py --policy-name <policy> --task-name <task> \
  --exp-name <run> --ckpt-tag best --episodes 2 --no-videos
```

全新环境应先使用可用的 Torch wheel 源安装匹配的 Torch/torchvision，再执行 `python -m pip install -r requirements.txt` 和 `python -m pip install -e .`。安装成功后仍须独立验证 PyTorch3D 与目标 CUDA，而不是仅看 `pip check`。

## 对论文比较的影响

修复前后代码版本及数据身份必须区分。F14 的加权统计、F15 的非有限更新拒绝、F16 的 Gaussian 裁剪修正和数据 revision 变化可能改变训练/选点结果；已有实验产物不会自动被修正。没有改动任务书暂缓的动作表示、模态缺失目标、采样分布、训练预算、normalizer 统计范围或公共测试池机制。

有效配置、版本/dirty 标志和 episode 明细提高可追溯性，但 dirty=true 不能还原未提交源码。真实机器人外部单位、关节顺序、成功判据与控制时序仍需相应环境验证，本报告不构成真机验收。

## 后续代码与文档清理（2026-10-04）

在上述未提交修改基础上检查调用者并完成清理：

- 移除 `fast_grad_finite_check` 的两条入口透传、Trainer 构造参数和无用实例字段。历史 YAML 仍可包含该字段；resume contract 继续忽略它。直接构造 Trainer 的代码应移除该关键字。
- 移除 checkpoint 私有保存链中未使用的 `epoch` 参数，继续以 `self.current_epoch` 保存实际恢复位置；同步 CUDA 测试调用。
- 删除无调用者的 `ResumableDistributedSampler.full_num_batches`，以及 selector/runtime 的无用导入。
- 修正梯度诊断、run 目录认领、demo 输出、selection 失败保留 best、评测 seed 优先级及远程 session 的过时注释和帮助；删除帮助中的旧实验配置示例。
- 更新 README、项目架构与仿真评测文档，覆盖独占目录、数据 revision、逐块 joint normalization、Diffusion 裁剪、设备 RNG 和 demo 产物。任务书保留为执行依据；未仅因当前默认配置不引用某研究模型就删除其实现。

本轮实际验证：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/dexmani-infra-env/bin/python -m unittest discover -s tests -p 'test_infra_*.py' -v
/tmp/dexmani-infra-env/bin/python dexmani_policy/smoke_test.py --config-only \
  dp dp3 dqrise r3d multitask_dit maniflow sat \
  ddp/dp ddp/dqrise ddp/r3d ddp/multitask_dit ddp/maniflow ddp/sat
python3 -m compileall -q dexmani_policy scripts tests
for script in scripts/remote/train_remote.sh scripts/remote/stop_remote.sh scripts/eval/eval_pipeline.sh; do
  bash -n "$script" || exit
done
git diff --check
```

**PASS**：25 项回归、13 个配置、Python 编译、三个 shell 文件逐一语法检查及 diff 检查。梯度回归覆盖裁剪开/关；历史 fast 字段的两种值生成相同 contract；CPU 连续/恢复对照继续验证权重、EMA、Adam、scheduler、step 和样本序列一致。完整测试共 27 项，耗时约 11.5 秒；两项 CUDA 测试 **SKIP / NOT VERIFIED**。真实 GPU 训练、模拟器、真机与从零安装仍未验证，沿用上文复验命令。没有提交、推送或改写已有实验产物。

## 剩余缺口修复与重新验收（2026-10-04，基线 5df7c07）

### 本轮范围与根因

实际起始及交付时 HEAD 均为 `5df7c07e062331aeda50595433fbcce2414658f0`，分支 `main`；开始时工作区干净，相对本轮审查基线没有后续提交。本轮修改保留为未提交工作区改动，没有 reset、commit 或 push。已读取 AGENTS、任务书和既有报告，检查实际调用者。

**更正此前 F08 表述**：原最终评测函数内部固定了第二次读取所得的权重，但 CLI 已从第一次读取取得 EMA/NFE，因此仍可能混用两次 selection。本轮将记录与当时解析的具体路径一起传递，才闭合整条调用链。下面的新验收结果补充此前记录；尤其 F04 的 CPU 空环境安装现已完成，不再沿用“从零安装未验证”的笼统结论。

| 项目 | 实现状态及原因 | 本轮测试状态 |
|---|---|---|
| best 单次解析 | IMPLEMENTED：loader 的 `resolve_best_checkpoint()` 返回记录和已校验的具体路径；保留 `read_best_ckpt_json()` 字典接口。CLI、single/sweep、demo 传递这一对值；共享调用者 `inspect_policy()` 同样修复。 | PASS：确定性读取 A 后发布 B，A=step 20/EMA/NFE 10，B=step 40/raw/NFE 3；权重及 selection 证据仍全来自 A。CLI、dotlist、直接函数覆盖、single/sweep/demo/inspection 均覆盖。 |
| held-out 与来源 | IMPLEMENTED：已解析记录独立于路径参数保留；实际 step 校验、selection seeds 排除、task/seed 与池身份检查沿用该记录。已有快照新增 selection_id、selection_summary、selection；实际覆盖的 EMA/NFE 继续单独保存。 | PASS：具体路径加固定记录仍执行 held-out；最终种子为 A 的补集；快照与 loader/runner 实参及结果引用一致。demo 保持非 held-out。 |
| codebook 转换后数值 | IMPLEMENTED：NPZ/state_dict 共用 `_prepare_buffers()`，先检查原始 schema/结构/整数排列，再按目标 dtype/device 转换并检查有限值、非零 scale、正且有限的 range。候选通过后才改对象。构造器复用范围检查，set_hand_normalizer 复用数值检查。 | PASS：两入口的 float64 范围塌缩、scale 下溢、range 差值溢出均拒绝，buffer/维度属性/metadata 不变；double/half 目标类型检查，合法 roundtrip/解码、空初始化及 EMA 通过。 |
| 远程 UUID | IMPLEMENTED：本地 python3→python 发现及可用性检查，调用 `uuid.uuid4().hex`，不使用服务器 CONDA_PYTHON 或 `/proc` UUID。 | PASS：ssh/rsync 替身验证 dry-run、两次唯一 session、停止字符集、无 kill、日志不截断、训练 exit 7/启动 exit 17；解释器缺失及 fallback 均覆盖。未连接真实服务器。 |
| checkout 位置 | IMPLEMENTED：临时 cwd、项目相对/绝对 fixture，局部替换 Path.expanduser 的 `~` 解析；其余 resume 处理真实执行，不改 HOME 或生产语义。 | PASS：当前 checkout 的四项 launch 测试；另复制当前 Python/YAML/shell 源码到主目录外的临时 `workspace/dexmani_policy`，四项同样通过。测试不再使用硬编码 `/tmp` 或 `relative_to(Path.home())`。 |
| F04 安装 | ALREADY FIXED：真实空环境解析、安装及使用未发现依赖冲突，requirements/pyproject 无需改动；README 更新实际验证范围。 | PASS：Python 3.10.20 + Torch 2.4.1+cpu / torchvision 0.19.1+cpu，见下文。CUDA 扩展/GPU 不包含在此 PASS 中。 |

未重复修改已正确实现的 F01/F02/F05–F07/F10–F17/F19/G01/G02。完整回归继续覆盖 CPU 连续/中断恢复、非有限梯度拒绝、Gaussian/clip_sample 支持矩阵、真实多任务池身份与 task/seed 排除、失败 selection 保留成功 best、分块 normalizer 和本地实际 rsync 元数据更新。

### 实际回归与静态检查

基线先执行现有 `test_infra_*.py`：27 项，24 PASS、2 CUDA SKIP、1 Manager spawn 因沙箱 socket 权限失败。允许本地进程通信后，单独运行原 `test_spawn_epoch` PASS，没有改 owner/proxy、吞异常或伪造通过。原 13 配置及静态检查亦通过。

修复后的命令（`PY` 指已有环境；完整回归运行在允许本地进程通信的执行环境）：

```bash
PY=/home/zhanghaoyang/miniconda3/envs/policy/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -m unittest discover -s tests -p 'test_infra_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" dexmani_policy/smoke_test.py --config-only \
  $(rg --files dexmani_policy/configs -g '*.yaml' | sed 's@dexmani_policy/configs/@@;s@\.yaml$@@' | sort)
python3 -m compileall -q dexmani_policy scripts tests
while IFS= read -r script; do bash -n "$script" || exit; done \
  < <(rg --files scripts -g '*.sh')
git diff --check
```

- **PASS**：修复后已有环境完整回归 32 项中 30 项通过、2 项 CUDA SKIP，约 11.6 秒。最后的 sweep 显式列表优先级补充也在下述空环境完整回归覆盖。
- **PASS**：13 个当前实际配置：`dp`、`dp3`、`dqrise`、`r3d`、`multitask_dit`、`maniflow`、`sat`，以及 `ddp/dp`、`ddp/dqrise`、`ddp/r3d`、`ddp/multitask_dit`、`ddp/maniflow`、`ddp/sat`；不是根据旧报告猜测配置列表。
- **PASS**：Python 编译、全部 16 个 shell 文件逐个 `bash -n`、diff 空白检查。
- **PASS**：主目录外临时 checkout 中 `python -m unittest discover -s tests -p test_infra_launch.py -v`，4 项通过。复制仅涉及源码，使用临时目录与独立 PYTHONPATH，没有复制数据、实验或权重。
- 测试开发中 `~` fixture 曾失败：Python 3.10 的 pathlib accessor 缓存了 expanduser，仅 patch os.path 不生效。改为只替换 Path.expanduser 的用户目录展开后通过；未为修测试修改生产路径函数。

### F04：真实空环境安装证据

临时环境为 `/tmp/dexmani-clean-install-yde7ewg9`，使用 `venv.EnvBuilder(with_pip=True, system_site_packages=False)` 创建；`pyvenv.cfg` 明确为 `include-system-site-packages = false`，sys.path 不包含已有 policy 的 site-packages。没有升级用户全局 Python/CUDA。

初次沙箱内 pip 报 `No matching distribution found for torch==2.4.1`。单独 URL 探测返回 `Operation not permitted`；允许网络后，PyPI 和官方 cu124 Torch 索引均 HTTP 200，均存在 2.4.1，证实此前错误不能判为依赖冲突。临时 venv 初始 pip 23.0.1 的 dry-run 会下载完整 797 MB wheel；主动取消后只升级该 venv 的 pip 至 26.2.1，使用 wheel metadata 完成原 requirements 的解析。未循环重试大包下载。

随后因本机无 CUDA，使用官方 CPU wheels 完成真实安装，保留研究主版本。实际命令：

```bash
INFRA_VENV=/tmp/dexmani-clean-install-yde7ewg9
"$INFRA_VENV/bin/python" -m pip install --upgrade pip --timeout 10 --retries 0
"$INFRA_VENV/bin/python" -m pip install --dry-run --report /tmp/dexmani-clean-resolve.json \
  --timeout 10 --retries 0 -r requirements.txt
"$INFRA_VENV/bin/python" -m pip install 'torch==2.4.1+cpu' 'torchvision==0.19.1+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu --timeout 20 --retries 0
"$INFRA_VENV/bin/python" -m pip install -r requirements.txt --timeout 20 --retries 0
"$INFRA_VENV/bin/python" -m pip install -e . --timeout 20 --retries 0
"$INFRA_VENV/bin/python" -m pip check
```

**PASS**：实际安装、editable 项目安装、`No broken requirements found`、requirements 全部 25 个直接依赖导入。实际版本保留 diffusers 0.27.2、Zarr 2.18.3、Hydra 1.3.5、transformers 4.48.0、huggingface-hub 0.25.2、peft 0.20.0。导入使用 importlib 逐项读取 requirements；包名映射为 hydra、huggingface_hub、sklearn、yaml、imageio_ffmpeg，其余同包名。

在该空环境把上述回归命令的 `PY` 换成 `$INFRA_VENV/bin/python` 后：**PASS**，32 项中 30 PASS、2 CUDA SKIP，约 10.8 秒；全部 13 配置 config-only **PASS**。Manager spawn 同样在允许 socket 的环境真实执行。

已有环境与空环境均实际执行默认 Agent 构造：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$INFRA_VENV/bin/python" - <<'PY'
import gc, hydra
from dexmani_policy.smoke_test import load_config
for name in ('dp', 'dp3'):
    agent = hydra.utils.instantiate(load_config(name).agent)
    print(name, sum(p.numel() for p in agent.parameters()))
    del agent
    gc.collect()
PY
```

**PASS**：dp 99,245,651 参数、dp3 68,762,131 参数，使用现有权重缓存，默认 DINO/LoRA 未关闭或替换。沿用原 DINO preset/processor 提示，没有为消除该提示改研究配置。这里只验收构造，不声称点云 CUDA 扩展、完整前向或 rollout 通过。

临时证据日志：`/tmp/dexmani-infra-baseline-tests.log`、`/tmp/dexmani-infra-fixed-tests.log`、`/tmp/dexmani-clean-resolve-metadata.log`、`/tmp/dexmani-clean-torch.log`、`/tmp/dexmani-clean-requirements.log`、`/tmp/dexmani-clean-project.log`、`/tmp/dexmani-clean-configs.log`、`/tmp/dexmani-clean-tests.log`。这些是本机临时文件，不作为仓库运行依赖。

### 兼容边界与剩余复验

- `read_best_ckpt_json()` 仍返回原记录字典，不改变 best JSON schema。新增的已解析参数为 `(record, concrete_path)`，没有缓存、锁或新配置层。
- 直接 single/sweep 函数显式传入的 EMA/NFE 仍优先；未传 EMA、single 未传 NFE 时使用 best/config 默认值。single 解析出多个 NFE 时要求调用 sweep，避免静默选取一项。CLI 优先级和 demo 优先级不变。
- 历史 best 缺 selection 身份仍警告，快照对应字段为 null；旧单任务明确 seeds 可继续 held-out，旧多任务缺池证据仍拒绝。demo 保存来源，但不宣称 held-out。具体 checkpoint 内容被外部原地篡改不在本次路径固定的保证内，未新增大权重 hash 或锁服务。
- codebook 保留 v3 schema、合法空状态和原有效历史 checkpoint；原始浮点 permutation 仍拒绝。转换后的零 scale、塌缩/溢出范围属于无效运行时状态，不再作为兼容输入接受。
- **NOT VERIFIED**：实际 CUDA RNG、非零 GPU、同 world_size 的 GPU 顺序变化、可见设备数量减少、NCCL、CUDA 扩展。已有 CUDA 构建环境仍报告 `torch.cuda.is_available() == False`、device_count=0；CPU wheel 安装成功不能替代这些验收。
- **NOT VERIFIED**：真实模拟器加载/rollout、成功率及视频。`dexmani_sim` 未安装；本机虽有数据和一个包含 6 个 checkpoint 的 maniflow 实验，仍不具备仿真条件。没有启动长训练、大批量评测或真机，没有改写这些产物。

最短复验入口（前提满足后执行）：

```bash
# 所需依赖装好且允许本地进程通信
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_infra_*.py' -v
# 至少两张可用 GPU：实际设备随机流及同 world_size GPU 顺序变化
CUDA_VISIBLE_DEVICES=0,1 python -m unittest discover -s tests -p test_infra_cuda.py -v
# 模拟器/数据/权重就绪后，仅少量实际 rollout
python dexmani_policy/eval_best_ckpt.py --policy-name <policy> --task-name <task> \
  --exp-name <run> --ckpt-tag best --episodes 2 --no-videos
```

可见设备减少需要分别在两种可见设备集合的独立进程保存/恢复同一个 rank 流，保持 world_size，不把现有两卡测试伪称为已覆盖此场景。完整从零 CUDA 环境安装及 PyTorch3D 目标扩展仍需在 GPU 机器复验。所有研究变量、归一化统计范围和选点排名保持不变。

## 本轮清理与文档对齐（2026-10-04）

在上述未提交修复上继续清理，没有改变研究语义：

- 删除测试模块未使用的 re、shlex、compute_eval_stats 导入；替身 SSH 脚本字符串内部实际使用的导入保留。
- 删除 NPZ 导入中仅为重复 shape 检查读取的 pose 数组，复用共同结构校验；声明维度、整数排列、转换后数值检查及原子提交仍完整保留。修正源值检查误称 runtime dtype 的错误说明。
- 修正 final-eval 模块注释中的不存在 `--seed`、默认参数来源和调用顺序；简化 NFE 校验说明，补充已固定记录的参数传递说明。shell 帮助明确 episodes 来自保存配置、sweep 使用 dotlist、EMA/raw 可显式覆盖。
- 同步评测机制、项目架构和 SSH 部署文档。任务书标记为历史审查要求；报告增加当前状态，明确初轮安装限制已解除、更正早期 best 固定表述，并指出早期单条多参数 `bash -n` 实际只检查首个文件。
- 保留 `read_best_ckpt_json()` 接口、历史 best/RNG/fast 配置兼容和仍在使用的研究路径；未以“默认配置未引用”为由删除模型。

本次实际验证使用 `/tmp/dexmani-clean-install-yde7ewg9/bin/python`，设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，分别执行：

```bash
python -m unittest discover -s tests -p test_infra_codebook.py -v
python -m unittest discover -s tests -p test_infra_evaluation.py -v
python -m unittest discover -s tests -p test_infra_launch.py -v
python3 -m compileall -q dexmani_policy scripts tests
while IFS= read -r script; do bash -n "$script" || exit; done \
  < <(rg --files scripts -g '*.sh')
bash scripts/eval/eval_best_ckpt.sh --help
git diff --check
```

**PASS**：6 + 7 + 4 = 17 项定向回归、Python 编译、16 个 shell 逐文件语法、帮助输出与 diff 检查。其余完整回归和空环境安装沿用前一轮证据，本次未重复安装或运行 GPU/仿真；环境未验证边界不变。
