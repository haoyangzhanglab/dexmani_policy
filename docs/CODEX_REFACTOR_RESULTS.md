# 修复与精简结果（2026-10-26）

本轮实际修改了源码和项目级回归，未 push、未创建 PR，未修改 AGENTS.md、外部 dexmani_sim/dexmani_real、全局依赖或设备。必修缺陷已完成 CPU 验证，并经用户授权在 sandbox 外补验单卡 CUDA 与两进程 CPU/Gloo。**双 GPU/NCCL、部分默认资源与训练/测试来源独立性仍有未验证范围**；最新补验和真实性能边界见文末。

## 基线与环境

- 起点为干净的 `main`，HEAD=`56c368efa8ae5b9ad0b2786ce9670e5ca3c6aaab`；相对任务书 review 基线 `4557258` 仅新增任务书。任务书已在工作区，无需 fetch/reset。已读取 AGENTS、README、架构、评测、手部表示、数据格式及实际训练/部署调用方。
- 使用 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`：Python 3.10.20、Torch 2.4.1+cu124、torchvision 0.19.1、CUDA build 12.4、NumPy 1.26.4、diffusers 0.27.2、Zarr 2.18.3、PyTorch3D 0.7.8。CPU 为 Ryzen 9 9900X。先前 `torch.cuda.is_available()=False` 和 nvidia-smi 失败仅发生在 sandbox 内；用户授权 sandbox 外验证后，确认 CUDA 正常：单张 RTX 4090 24 GiB、driver 580.173.02，实际张量运算通过。此前的驱动不可用归因已更正。
- requirements 的 25 个 pin 与 pyproject 的 25 个范围均满足当前安装版本，未发现需要改 pin 的本项目证据。`pip check` **失败**：外部 dexmani-sim 要求 rerun-sdk >=0.23.3,<0.24，环境实际 0.22.1；未替用户升级。
- 实际数据：multi_grasp 150 episodes/30,187 frames；pick_apple_messy 150/27,365；place_milk_box 150/36,945；pour 150/42,354。点云为 `[T,1024,6]`，RGB 240×320。均有 episode_seeds 和部分仿真配置，缺少足够的稳定 source/episode/trial 与生成身份。
- 有 Uni3D、SigLIP 资源及 DINO-small/CLIP 缓存目录；未以目录存在证明完整可用。默认多任务 pick_bottle/open_box 数据及匹配 DQ/LAMP/LatentFlow 预训练产物缺失。外部 sim 可定位到本地源码；实际 rollout 状态见补验。
- 基线实际复现：VQ 首次未命中 code 放大约 80,002 倍；Runtime reset/load 改变宿主 RNG；frozen 请求接受缺候选事实的记录，而独立 verifier 拒绝；ignored snapshot 误记父 Git 身份；R3D 接受不兼容归一化；excluded episode 被扫描。RGB 重复、零权重统计、ManiFlow 分组等按当前源码核实，未照旧 review 修改正确语义。

## 验证命令与范围

下文用 T1–T5 引用实际执行的命令；统一环境为：

```bash
conda activate policy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
# T1：完整项目回归
python -m unittest discover -s tests -v
# T2：九种默认配置及实际目标导入
python dexmani_policy/smoke_test.py --config-only dp dp3 sat r3d maniflow multitask_dit dqrise lamp latent_flow
# T3：真实数据和默认 CPU DataLoader
python dexmani_policy/smoke_test.py --data-only dp3
# T4：现有入口
for script in scripts/eval/{eval_pipeline,evaluate,select_checkpoint}.sh scripts/training/{train,train_ddp}.sh; do
  bash -n "$script" && bash "$script" --help
done
# T5：依赖与差异检查
python -m pip check
git diff --check
```

- T1：**27 tests 通过，无 skip**，清理后最终运行 6.810 s。数值/RNG、证据/归档、数据/配置、九法契约和短训练分别位于 `tests/test_numerics_runtime.py`、`test_evaluation_resume.py`、`test_data_config.py`、`test_policy_contracts.py`、`test_training.py`。修复前数值主题 4 个失败、数据/去重主题 3 个失败；归档接口当时不存在。没有 AST 抽取、复制算法或 mock 内部模型。
- 九法小模型实际运行：DP、DP3、SAT、R3D、ManiFlow、MultiTaskDiT、DQ-RISE、LAMP、LatentFlow 均 CPU FP32 forward/backward、完整 future_actions、显式 Generator 重放、raw/EMA strict restore；另测 MultiTaskDiT rectified_flow。H=8（SAT=16、DQ=20、LAMP=1），NFE=2（LAMP=1），B=3，点数=16，RGB 用真实 ResNet18/32×32；R3D 保留实际 timm EVA tiny。使用真实 PyTorch3D CPU FPS。DQ 用公开 buffer 接口装入合法合成码本；LAMP/HandVAE 用实际模型做一次合成预训练 update 并保存/读取有效 artifact。它们只验证机制，不是可用研究权重。
- 短训练：临时真实 Zarr，经实际 data builder、模型/optimizer/EMA builder、Trainer、Workspace 完成 4 updates，再在新 run 恢复 2 updates；总计划 10、B=4、accum=3、7 batches/epoch，覆盖尾部单 microbatch update。scheduler/EMA 计数为 4→6，raw/EMA 固定噪声输出严格相等，归档候选为 20/40→20/40/60%。`obs_lr=0` 的全冻结含义不变。没有将此称为 worker RNG 逐位续训或九法默认训练通过。
- T2 九法通过；另在 HEAD 的 `/tmp/dexmani_refactor_baseline` 完整源码副本与当前源码分别解析默认配置，只固定运行目录/日志名称，九个 JSON SHA-256 逐项相同。未改默认预算、batch、NFE/C、padding 或 YAML。
- T3：沙箱内默认 8 workers 因本地 IPC `PermissionError` 阻塞并中止；经执行权限审核后，在允许本地 IPC 的环境原命令通过。80 train episodes、14,037 windows，检查两个默认 B=128 batch；无 holdout。测试未通过修改 worker 数绕过。T4 五个入口 help/语法检查通过；没有修改 shell。T5 diff check 通过；pip check 的外部依赖失败见上。
- `python -m compileall -q dexmani_policy tests` 通过。细分日志保留于本机 `/tmp/dexmani_{numerics_before,numerics_after,archive_after,phase_d,final,configs_final,data_unsandboxed}.log`；这些不是提交产物，也不是跨机器持久证据。

## R01–R20 逐项结果

路径均相对 `dexmani_policy/`；测试编号见上。CPU 证据为 T1–T5；T6 指文末 sandbox 外 CUDA/Gloo 补验，仍区分小模型、默认配置和真正双卡。

| ID / 状态 | 实际源码与处理 | 验证、兼容性与未验证范围 |
| --- | --- | --- |
| R01 已修复 | `agents/hand_prior/vqvae.py`：非 kmeans 单位伪计数与 embed_avg 一致；旧 buffer 加载不改写 | T1：低多样性/全命中/持续未命中、kmeans、死码、ST 梯度和往返通过。新 VQ 轨迹改变，需新目录与下游重训；未证明成功率改善 |
| R02 已修复 | `evaluation/artifacts.py` 收拢唯一完整 verifier/ranking；`config.py` frozen 入口、`selection.py` 发布 best 共用 | T1：缺事实、假统计、错误赢家、重复/缺 episode、交叉 seeds、同名换权重均拒绝；失败保留旧 best；CLI/搬迁通过；T6 普通仿真 rollout 已完成；T7 实际 selection/frozen pipeline 使用外部环境替身通过，完整真实仿真选点仍未执行 |
| R03 已修复 | `evaluation/artifacts.code_version`：核对 Git 根、按相对路径/内容哈希实际研究源码，排除资源与 symlink | T1 临时父 Git+ignored snapshot、源码变化、日志不变、搬迁/符号链接通过；实际 policy/sim 摘要可计算。commit/dirty 仅备注，不代替内容身份 |
| R04 已修复，双 GPU/NCCL 待验 | `common/experiments.archive_resume_milestones`、`training/builder.py`：首次 update 前从自包含来源 run 归档完整有效前缀，不再回溯历史父链或哈希去重 | T1：50/100 恢复、早期分叉排除未来、二次续训、搬迁、缺失/重复候选/配方与状态冲突、EXDEV 原子复制；真实 Trainer 恢复通过。hardlink 非独立备份。T6 两进程 CPU/Gloo 的 rank 0 归档同步、尾累计和恢复通过；双 GPU/NCCL 未验证 |
| R05 已修复，CPU/CUDA 验证通过 | `deployment/runtime.py`、Policy 推理入口、Diffusion/Flow/RTC、`env_runner/sim_runner.py`：实例/episode Generator；`common/checkpoint.py` 恢复构造的宿主 RNG | T1：A/B 交错、load/reset/warmup、Python/NumPy/Torch 隔离、九法重放、RTC autograd 通过。训练 RNG 不变；直接调用省略 generator 仍全局采样。新轨迹不保证旧版逐位兼容；T6 CUDA 九法与 runtime/RTC 重放通过；多线程未验证，同实例只支持串行 |
| R06 已补证据，独立性外部阻塞 | `common/experiments.file_sha256`、evaluation 三入口：候选冻结/加载时检查权重；`demonstration_dataset.py` 保存已存在的 task/seed；`training_source_evidence` 报告同任务交集 | T1 证明身份变更会拒绝；Torch/CUDA/dtype/numerics 不同拒绝，设备编号/Git 备注可变。旧记录不补 hash；恢复不回填旧 checkpoint，也不再删除当前来源事实来适配旧配方；缺字段由严格配方比较拒绝。数据没有内容 hash，训练/测试独立性仍 unverified；需要原生成端与测试场景来源证据 |
| R07 已修复 | `agents/policies/r3d.py` 新训练方法校验限制 point_cloud=limits | T1/T2：默认通过，identity/gaussian 拒绝；默认 clamp 数学不变，旧非默认实验用原源码 |
| R08 生命周期与推理 mmap 已采用 | `common/checkpoint.py` 在构造/搬设备前释放无关 payload，strict copy 后释放 state；推理/归档配方读取显式 mmap | T1/T6 九法 CPU/CUDA raw/EMA 严格往返；CPU/GPU 加载测量见下。完整 `.pt` 仍只信任项目常规 zip 产物；无新导出格式、无不可信文件 fallback。CUDA 峰值未增加，权重/固定噪声输出 hash 一致 |
| R09 已修复 | `datasets/common/replay_buffer.py` selected spans/chunks；`demonstration_dataset.finite_rows` train/val 并集；`hand_prior/lamp_dataset.py` 只装所选历史 | T1 NaN/Inf、短 episode、padding、角色源行、excluded I/O spy、LAMP、LatentFlow query/relative 统计通过。真实数据前后窗口/源行/normalizer hash 一致；收益见下，不加持久缓存 |
| R10 保留正确行为 | `datasets/multitask_dataset.iter_normalization_data` 保持所有 child 统计；补架构说明 | T1 零采样权重仍决定 normalizer max；欲完全移除须删 child 并重训，未新增加权统计模式 |
| R11 已修复说明与早期边界 | `agents/obs_encoder/pointcloud/ops.py` 清楚报原生依赖错误；Sparse3D/R3D 方法检查及构造提前导入；README 说明 PyTorch3D | T2 九法导入、T1 实际 CPU FPS 通过；DP3 不下采样仍懒加载。未升级依赖；T6 实际 CUDA FPS/九法小模型通过 |
| R12 已收敛 | `training/data.py` 返回三元事实；`training/builder.assemble_effective_config` 唯一合成/冻结快照；`common/checkpoint.prepare_restore_config` 用窄 Policy hook 替代类名分支；SAT C 使用窗口检查 | T1 输入 cfg 不回写、真实 train/resume、闭集离线恢复通过；T2+默认配置摘要一致。SAT C=3 小模型通过，默认 C=8 不变；H/关节约束保留。RGB fresh 权重加载沿用现有 constructor，不强迁无收益接口；未建设配置继承体系 |
| R13 已去重 | 新 `agents/obs_encoder/rgb_state.py` 替代 DP/MultiTaskDiT 两份相同 encoder | T1 state_dict 路径/输出一致，raw/EMA strict 恢复通过；仅内部 encoder 类名改变，没有旧别名或并行实现。LAMP global 与 LatentFlow spatial 保持独立 |
| R14 保留正确行为 | builder 的 obs_lr=0 全冻结、RGB tune_mode=freeze 视觉冻结均保留 | T1 视觉 BatchNorm 保持 eval、视觉无梯度、state MLP 有梯度；真实训练全冻结检查通过。未增加 vision_lr 或未使用的分阶段设置 |
| R15 已修复，非默认 CUDA 外部阻塞 | `agents/obs_encoder/text.py` 显式 device，默认 CPU；闭集 task embedding 随 Policy 移动 | T1 模拟 CUDA available 时 CPU 构造不分配 GPU；真实 tiny CLIP 的下载边界 fixture；闭集 checkpoint 禁外部文本加载恢复通过。T6 真实 CUDA 可用时默认 CPU 与显式 cuda:0 构造通过；仅一张 GPU，cuda:1 与开放文本离线恢复未验证 |
| R16 性能门控未采用 / 外部阻塞 | 保留 builder 的 ManiFlow 非主 rank 完整 EMA；`BasePolicy.get_training_loss_kwargs` 只传 EMA decoder，在线 encoder 条件不变 | T1 EMA encoder 无调用、教师无梯度；B=3 为 2+1、B=2 为 1+1、B=1 拒绝；loss 为两个独立 mean 相加。T6 默认 ManiFlow bf16/compile 短训练与恢复通过，但缺双 GPU，未测 EMA 精简的整体内存/update/恢复，未放入原型 |
| R17 性能门控未采用 | `datasets/common/distributed_sampler.py` 保留原 Python mapping、配额与 RNG 顺序 | 现实现 14k/100k/1M 的时间/对象内存见下；当前规模成本小，未证明替换的端到端收益，未新增数组分支。没有声称优化后 rank/cursor 等价 |
| R18 保留正确行为 | 主 Trainer 继续按有效 update 优化，通过 simulation selection 选点；不新增主循环 validation | T1 尾累计、scheduler/EMA、DP FiLM `scale*x+bias`、手部 decoder latent 梯度、完整动作窗口通过；T3 实际数据诊断。T6 DP3/DP/SAT/ManiFlow 默认 CUDA bf16/compile 通过，CPU/Gloo 覆盖尾累计；R3D 默认显存不足，但 B=4/A=32 保持 nominal global batch=128 的补验通过；双 GPU/NCCL 待验 |
| R19 性能门控未采用 | 保留 `action_decoders/flow.py` 请求内 K/V 缓存及 `backbones/ditx.py`，未合并仿射、未存派生权重 | T6 默认 ManiFlow CUDA 短训练及 NFE=10 推理/恢复通过；未完成仿射合并在 NFE=1/4/10 的完整数值/端到端比较，因此未采用，不能再归因为无 CUDA；没有闲置开关 |
| R20 保留研究边界 | projection/RGB geometry 现实现保留；架构文档已明确 FeatureAlignment/HandScene/SceneAggregator 尚未接入九法，触觉也不扩展为新策略 | T1 resize/crop 半像素变换、投影点序、前方/越界 mask 通过。相机矩阵/最终图像变换/一致点序仍为调用契约，native ResNet 网格不视为 uniform ViT grid |

## 性能测量与采用决策

使用上面 CPU/Torch 环境，FP32、OMP/MKL=1；基线以 `git archive HEAD` 解到 `/tmp/dexmani_refactor_baseline`，没有 checkout/reset 工作区。本小节为 CPU 阶段测量，各项独立 Python 进程重复三次，报告中位数与范围；GPU 峰值补验见 T6。没有仿真速度对照数据。

另用 `/tmp/dexmani_scan_equivalence.py "$source"` 对两个完整源码版本执行相同临时 Zarr（NaN observation、Inf action、excluded NaN、短 episode 与 padding）。比较所有 train/val indices、角色源行、逐样本窗口/context 和 normalizer tensor：DemonstrationDataset train/val=10/5（hash `dabd7206…f45f52a6`）、LAMP=9/7（`f8274fa2…60a71667`）、LatentFlow=3/2（`0055a347…8ae2259`），前后均完全相同；日志 `/tmp/dexmani_scan_equivalence.log`。这里未将 exclusions 的内部 valid mask 计入等价要求，因为本轮有意将未扫描行标为无效。

| 测量 | 基线 | 当前 | 数值与结论 |
| --- | --- | --- | --- |
| 实际 pick_apple_messy 默认 DP3 Dataset 构建+normalizer，不含 import/训练；80/150 episodes、27,365 frames、1024×6 点 | 2.291 s（2.258–2.304），peak RSS 669.3 MiB | 2.156 s（2.144–2.228），665.8 MiB | 14,037 windows；indices、两类源行、全部 normalizer tensor 联合 hash 均 `fe6dfe3d…eeb76fb`。约 6% 启动节省，仅是缓存状态未受控的本地读测量；主要采用理由是少读确定不用的数据且不增缓存机制 |
| 177,444,570-byte（169.2 MiB）真实 DP3/AdamW 容器，实际 restore_policy_agent EMA→CPU，UNet [128,256,512]，H=8；不含 import/采样 | 0.492 s（0.491–0.503），674.7 MiB | 0.484 s（0.480–0.499），546.4 MiB | 全部恢复 tensor hash 相同 `922c54ec…43e54e3b`；时间区间重叠，不宣称显著提速。约 128 MiB 峰值减少支持提前释放引用 |
| 同容器 read_checkpoint 并读取所选 EMA 全部 tensor，eager→mmap；不构造 Policy | eager 0.0771 s，544.3 MiB | mmap 0.0443 s，417.0 MiB | hash 相同；仅 metadata 返回耗时 0.0585→0.0185 s。每次调用文件级 POSIX_FADV_DONTNEED，属 advisory cold，不能保证真正冷盘。这是 CPU 阶段测量；后续 T6 完整 CUDA 恢复验证通过后，推理已启用 mmap |
| 原 multitask_indices，两个等权任务、seed=42、epoch=0/1/2 | 14,037：1.42 ms / 0.49 MiB；100k：10.0 ms / 3.43 MiB；1M：144.9 ms / 34.76 MiB | 未改 | 内存是 Python list+int 的 owned size，不是进程 peak RSS；未测替代实现，不当作性能收益。当前数据规模不值得引入另一表示 |

执行的测量脚本在本机 `/tmp/dexmani_scan_bench.py`、`/tmp/dexmani_load_bench.py`，日志分别为 `/tmp/dexmani_scan_bench.log`、`/tmp/dexmani_load_bench.log`，没有提交数据/模型/大日志。调用如下（source 分别取上述基线和当前仓库绝对路径；各三次）：

```bash
python /tmp/dexmani_scan_bench.py "$source"
python /tmp/dexmani_load_bench.py /home/zhanghaoyang/Desktop/dexmani_policy create
python /tmp/dexmani_load_bench.py "$source" restore
python /tmp/dexmani_load_bench.py /home/zhanghaoyang/Desktop/dexmani_policy eager
python /tmp/dexmani_load_bench.py /home/zhanghaoyang/Desktop/dexmani_policy mmap
```

## T6：sandbox 外 CUDA / Gloo / 仿真补验

用户授权后实际确认一张 RTX 4090。以下检查均在 sandbox 外运行；未改驱动、依赖、外部仓库或默认 YAML。CUDA 检查显式要求 GPU，无 CPU fallback/skip。

```bash
# 复用同一份实际 Policy/Trainer 检查，device 是测试参数。
python -m tests.cuda_checks
CUDA_VISIBLE_DEVICES='' python -m tests.ddp_checks
python dexmani_policy/smoke_test.py dp3 --max-updates 4
python dexmani_policy/smoke_test.py sat --max-updates 4
python dexmani_policy/smoke_test.py dp --max-updates 4
python dexmani_policy/smoke_test.py r3d --max-updates 4
python dexmani_policy/smoke_test.py maniflow --max-updates 4
# 仅临时诊断：默认 R3D 配置覆盖 batch_size=4、accumulation=32 后保存此 YAML。
python dexmani_policy/smoke_test.py /tmp/dexmani_r3d_accum.yaml --max-updates 4
python /tmp/dexmani_cuda_rollout.py
```

共同环境仍为 policy、OMP/MKL=1、HF_HUB_OFFLINE=1；compile 缓存限定 `/tmp/dexmani_inductor`、`/tmp/dexmani_triton`。每个默认训练命令最多 4 updates，并有限时保护。

| 范围 | 实际结果与限制 |
| --- | --- |
| CUDA 专项 | `tests.cuda_checks` 4 项通过（清理后 3.855 s，无 skip）；包含九法小模型 FP32 forward/backward、真实 CUDA FPS、raw/EMA strict restore、固定 Generator 重放、host/CUDA RNG 隔离、RTC、VQ、CLIP 默认 CPU/显式 cuda:0 和 Trainer 4→6 updates 续训。恢复比较保留 rtol=atol=0 |
| 两进程 CPU/Gloo | DP3 实际 builder/Trainer/DDP 新训练 4 updates、续训 2 updates，通过 rank 配方一致性、跨 rank 参数逐位同步、尾累计/no_sync、rank 0 milestone 归档和两个 rank RNG 保存。不是双 GPU/NCCL 验证，也未覆盖 ManiFlow 非主 rank EMA 的多 GPU 恢复 |
| DP3 默认 | B=128、bf16、compile=True，4 updates + raw/EMA 恢复通过；compiled/eager 最大差 raw=0.001606822、EMA=0.0020122528 |
| SAT 默认 | B=128、bf16、compile=True，4 updates + raw/EMA 恢复通过；上述最大差均为 0 |
| DP 默认 | B=64、实际缓存 DINO-small/LoRA、bf16、compile=True，4 updates + raw/EMA 恢复通过；最大差 raw=0.0011906028、EMA=0.00093960762 |
| ManiFlow 默认 | B=128、bf16、compile=True，4 updates + raw/EMA 恢复通过；最大差 raw=1.1920929e-6、EMA=1.7881393e-7。实测构造参数 180.25M，其中 obs_encoder 约 0.25M；保留完整 EMA，不为这部分增加 rank 专用恢复机制 |
| R3D 默认 | **失败：真实 CUDA OOM**。B=128 时进程约 20.22 GiB，另需 8 GiB，24 GiB GPU 剩余约 1.78 GiB；未调 allocator、关闭验证或修改默认配置掩盖失败 |
| R3D 补验 | 原模型/预训练 Uni3D、bf16/compile 保留；临时 B=4、accum=32，nominal global batch=128，4 updates + raw/EMA 恢复通过。最大差 raw=0.0016450882、EMA=0.001537919。微批与 epoch 尾组不同，不能称为默认 B=128 训练或逐位等价 |
| 其他默认方法 | MultiTaskDiT 默认多任务数据、DQ/LAMP/LatentFlow 匹配先验产物仍缺；九法小模型 CUDA 通过不替代这些默认配置集成 |
| 真实仿真 | `pick_apple_messy`，固定开发 seeds 420001/420002（不在既有 evaluation pool），默认每 episode 240 步、H=16/Nobs=2/C=8/NFE=10，两次均完整结束并写 `status=completed`，**0/2 成功**。模型是实际数据归一化、B=4、4 updates 的小 DP3 诊断模型，不是默认模型/训练预算的效果评估；仅验证普通评测、SAPIEN、完整 future_actions→执行前 C 步及事实落盘链路。无录像，无 frozen selection/test |

compiled/eager 差异由原 smoke 报告，未要求二者逐位相同；raw/EMA 恢复对照仍沿用既有验证，未放宽容差。此阶段 CPU 回归为 25 tests 通过（6.360 s）；后续清理后的最终结果见 T7。

R08 的完整 GPU 加载补验：仍使用 169.2 MiB 的实际 DP3+AdamW zip checkpoint，FP32、B=1、H=8、8 点、NFE=2。在没有本任务其他 GPU 作业时顺序测量原版、提前释放版、mmap 版。每版 3 个独立进程，每进程一次 advisory cold load 与两次 warm load；先初始化 CUDA，所有计时正确 synchronize。CPU RSS 是进程 peak，包含首个预测/数值摘要阶段，不能当作纯加载的独占内存。

| 指标（中位数） | 原版 | 提前释放 + eager | 提前释放 + mmap |
| --- | --- | --- | --- |
| cold load 到 CUDA | 535.6 ms | 515.4 ms | 501.5 ms |
| cold 首个预测 | 153.5 ms | 158.4 ms | 165.4 ms |
| warm load 到 CUDA | 85.7 ms | 82.5 ms | 65.9 ms |
| warm 首个预测 | 4.58 ms | 4.47 ms | 4.45 ms |
| cold CPU peak RSS | 1115.9 MiB | 1110.8 MiB | 1103.3 MiB |
| 重复加载后 CPU peak RSS | 1322.3 MiB | 1274.9 MiB | 1182.8 MiB |
| load CUDA peak allocated，cold / warm | 42.74 / 50.36 MiB | 42.74 / 50.36 MiB | 42.74 / 50.36 MiB |

27 次结果的全部 tensor hash 均为 `922c54ec…43e54e3b`，固定噪声输出 hash 均为 `2379d2f0…a85b87ec`。cold 总耗时变化较小，不声称显著冷启动加速；warm load/进程内存改善且无 GPU 内存、数值回归，故采用 `restore_policy_agent` 中的一行 `mmap=True`，不加配置开关、导出格式或恢复分支。采用后重新通过上述 CPU 25 项和 CUDA 4 项（覆盖九法）。完整训练恢复仍使用原 eager 路径。

可核查的本机日志：`/tmp/dexmani_cuda_{checks,mmap_final,dp3,sat,dp,r3d,r3d_accum,maniflow,rollout}.log`、`/tmp/dexmani_gloo_checks.log`、`/tmp/dexmani_gpu_load_bench_final.log`、`/tmp/dexmani_cpu_mmap_final.log`。加载脚本为 `/tmp/dexmani_gpu_load_bench.py`，分别传入 `/tmp/dexmani_refactor_baseline`、`/tmp/dexmani_gpu_eager`、`/tmp/dexmani_gpu_mmap` 完整源码副本。仿真事实在 `/tmp/dexmani-cuda-rollout-of2x5hks/evaluation/20261026_084915_qqye1qas/result_details.json`。上述大产物未进入 Git。

## T7：调用链清理与最终对齐

- `common/experiments.archive_resume_milestones` 删除历史父链递归、cycle 状态、每个 milestone 的 SHA-256、内容去重和已有目标合并。当前来源 run 必须提供恢复点之前完整且无歧义的候选前缀；重复 step 即使字节相同也拒绝。保留配方/文件状态核对、未来分支排除、不可变文件约束及 EXDEV 原子复制。新 run 自包含，二次续训和搬迁不依赖祖先目录。
- `training/data.py` 删除专门为旧配方隐藏 `source_evidence` 的分支。当前数据事实照常生成，旧 checkpoint 缺字段由现有严格配方比较拒绝；不迁移旧产物、不回填历史来源。真实 Trainer CPU/CUDA 回归同时检查完整配方可恢复、缺来源事实的配方被拒绝。
- `evaluation/selection.py` 的 `publish_best` 直接接收 record 路径，返回已验证 checkpoint；调用方及 CLI 对齐，`--publish-best` 不再先验一次再重复验一次。`artifacts.py` 删除与排名复核重复的计划/候选校验和由实际内容比较覆盖的摘要长度检查。保留正式评测必要的实际源码、权重身份和全部候选事实复核，不增加 hash 开关或版本适配。
- 新增实际 `run_pipeline` 回归：两候选模型严格恢复、SimRunner 执行动作、selection 排名、best 发布、frozen request 及测试结果落盘均运行项目源码，只替换外部 `SimRunner.make_env`。另补归档 state/recipe 损坏在写入前拒绝。现有 CLI 测试增加 `--publish-best`；缺前缀的旧父链不再自动修补，测试改为明确拒绝此已删除流程。新接口及删除旧回溯的回归在实现前分别复现错误/失败，修改后通过。
- 补齐源码身份与来源交集容器、Hydra 策略类的实际类型；DataLoader 平面标量参数直接用 `dict(cfg.dataloader)`，移除通用容器转换及其无用导入。未加类型兜底、抑制注解或运行时检查。
- 架构、评测文档同步当前归档与严格恢复约束；纠正录像描述：selection 和冻结测试默认都不录像，在创建协议时显式覆盖可同时开启。训练/恢复、smoke、DDP、部署、RNG/RTC、数据窗口及共享 RGB 编码器调用点已检索核对，旧 encoder 类名和旧发布签名无悬空引用。未为无实际调用需求增加新的兼容机制。

最终执行（沿用上述 policy 环境和 OMP/MKL/HF 环境变量）：

```bash
python -m unittest discover -s tests -v
python -m tests.cuda_checks                  # sandbox 外，用户已授权
CUDA_VISIBLE_DEVICES='' python -m tests.ddp_checks  # sandbox 外本地 IPC
python dexmani_policy/smoke_test.py --config-only dp dp3 sat r3d maniflow multitask_dit dqrise lamp latent_flow
python -m compileall -q dexmani_policy tests
git diff --check
```

结果：CPU 27 tests（6.810 s）、CUDA 4 tests（3.855 s）、两进程 CPU/Gloo fresh/resume/tail/archive、九种默认配置与目标导入、compileall 和最终 diff check 均通过，无 skip。最后简化 DataLoader 后复跑 CPU/CUDA/Gloo，并在 sandbox 外执行 `python dexmani_policy/smoke_test.py --data-only dp3`：真实数据、默认 8 workers、B=128 的两批均通过，14,037 windows 不变。日志为 `/tmp/dexmani_cleanup_{cpu,cuda,gloo,configs,data}.log`。此次未改 shell；先前五个入口的语法/help 验证仍有效。

另外使用现有 mypy 和 setuptools 完成局部类型与包构建检查：

```bash
python -m mypy --follow-imports=silent --ignore-missing-imports --check-untyped-defs \
  --cache-dir=/tmp/dexmani-cleanup-mypy \
  dexmani_policy/common/experiments.py dexmani_policy/evaluation/artifacts.py \
  dexmani_policy/evaluation/selection.py dexmani_policy/training/data.py
# 在 /tmp 中复制当前 pyproject.toml 和 dexmani_policy 包，不改原仓库、不安装产物：
python -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir /tmp/dexmani-cleanup-build-mqih2l9c/wheel \
  /tmp/dexmani-cleanup-build-mqih2l9c/source
```

mypy 初次报 30 项，其中 HEAD 基线在相同检查下已有 data.py 的 28 项；修正相关类型和参数转换后 **4 个模块全部通过**。此命令检查函数体但不追查依赖内部，不代表整个仓库类型检查通过。wheel 构建成功，未安装或进行分发包端到端运行。日志为 `/tmp/dexmani_cleanup_{mypy,mypy_baseline,build}.log`。本轮清理没有算法数值变化或新的性能收益主张，前述默认 R3D 显存、缺失产物、双 GPU 和真实完整 selection 的限制仍然成立。

## 尚需的最小外部验证

1. 无需修驱动，单卡 CUDA 已验证。为 R3D 默认 B=128 提供足够显存；为 DQ/LAMP/LatentFlow 和 MultiTaskDiT 提供匹配手部产物/数据后，运行各自默认 `python dexmani_policy/smoke_test.py <method> --max-updates 4`。九法小模型通过不代表所有默认配置已训练通过。
2. 两张 GPU 上做短 DDP 新训练及跨 milestone 恢复，检查 rank 0 归档、所有 rank 同步、尾累计 no_sync、optimizer/scheduler/EMA 计步；完成前不采用 EMA 精简或其他 GPU 性能原型。
3. 本轮真实仿真链路已完成两次开发 rollout。外部 rerun-sdk 版本冲突仍由环境维护者处理；要报告研究结果，需要训练充分的权重与重新生成的正式 selection 证据，不能把本轮 0/2 的短训诊断当成效果验证，也不要用 frozen test seeds 调参。
4. 若要声称训练/测试来源独立，需要生成端提供稳定 source/trial/场景版本与可比仿真设置，并与实际测试来源核对。仅有当前 episode_seeds 不足以完成此项。

本轮没有运行长训练、远程训练、真实机器人或冻结测试调参。最终审查保留必要的一份 verifier、一份 RGB encoder 和一处有效配置组装，没有 registry、缓存框架、导出格式或未使用的优化开关。
