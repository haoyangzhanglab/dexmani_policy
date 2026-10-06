# 增量整改报告

更新：2026-10-06。起始 `main` HEAD：`f55a02f6507901fcd126546789a01fc127b60996`，工作区干净；仅根 `AGENTS.md` 适用。按最新任务书第 0/15 节实施，整改验收时未 reset、未提交/推送/合并，未启动正式训练、批量仿真、远程 stop 或真机。使用项目 research-iterate、preflight-experiment 工作流。以下“实现”与“验证”独立：**19 个剩余代码 FIX 已实施，R08 补验部分完成；B10/E03 保持已修复。不能称全部生产验收通过。**

主要改动：VQ run 隔离与固定选点、usage 分块；显式配置消费和边界校验；RGB projection dtype、patch/PE 一致性、零延迟 RTC warmup；单次 checkpoint 读取与里程碑去重；固定 seed 清单、不可变 selection 交接、源码归档；远端查询状态的本地 fixture 修复。已有 streaming、共享 split/hand affine、EMA 时钟和累积边界未重写。

后续清理：用户明确要求删除过时/无用代码后，S07 从暂缓更新为已实施，删除三组无活跃调用的预留组件，并同步入口注释与使用文档。前述“起始工作区干净”指增量整改开始时；清理在已有未提交修改上继续，未覆盖先前成果。清理验证单独列在文末，不替代下述原始运行记录。

## 验证记录与环境

解释器 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`（下文 `$PY`），Torch `2.4.1+cu124`。默认 shell 没有 `python`。沙箱内 CUDA 不可见，Manager socket 报 `PermissionError`；通过允许的沙箱外执行后实际有单张 RTX 4090 24 GiB。环境未安装/升级任何依赖。CPU 夹具仍是 CPU，沙箱外执行不使它们变成 GPU 测试。

| 标签 | 命令/证据 | 本次结果与范围 |
|---|---|---|
| T | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 120 $PY -m pytest -q tests/test_review_remediation.py tests/test_streaming_dataset.py tests/test_policy_vq_alignment.py tests/test_benchmark_dataset_streaming.py tests/test_policy_windows.py tests/test_policy_rtc.py tests/test_infra_codebook.py tests/test_infra_training.py tests/test_infra_resume.py tests/test_infra_evaluation.py tests/test_infra_launch.py tests/test_infra_cuda.py` | PASS：145 passed、28 subtests passed（见 [pytest.log](review_remediation_evidence/pytest.log)）；两项双卡测试 SKIP = NOT VERIFIED。新定向用例使用真实局部实现；评测/远程 fixture 替换环境，不是实际 rollout/SSH。 |
| C | `$PY dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat maniflow r3d multitask_dit` | 七配置 PASS；仅配置/import/target 检查，[原始输出](review_remediation_evidence/config_only.log)。 |
| G | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 180 $PY -u dexmani_policy/smoke_test.py dp3 dqrise` | PASS，默认真实 `pick_apple_messy.zarr`，各一次 forward/backward、optimizer/scheduler/EMA 更新、predict、raw/EMA strict restore + predict，[原始输出](review_remediation_evidence/cuda_smoke.log)。smoke 不启用生产 compile/AMP/DDP；DQ 使用现有合成且 affine 对齐的临时码本，不是训练好的生产码本。 |
| B | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 90 $PY tests/benchmark_dataset_streaming.py robot_data/pick_apple_messy.zarr --gpu-config dexmani_policy/configs/dp3.yaml --workers 0 --batch-size 16 --batches 8 --order random` | PASS，真实 CUDA 3 warmup + 8 measured batch 均有限；[完整输出](review_remediation_evidence/gpu_streaming.log)。不含 optimizer/EMA/checkpoint/闭环。 |
| P | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 90 $PY docs/review_remediation_evidence/rgb_precision_probe.py` | PASS，已有缓存 DINOv2-small LoRA、固定合成 RGB 的三步 encoder 诊断、strict state restore；[输出](review_remediation_evidence/rgb_precision.log)。不是完整 DP/RGB 训练或冷缓存离线恢复。 |
| S | `bash -n scripts/training/train_vq_hand.sh scripts/eval/eval_pipeline.sh scripts/remote/stop_remote.sh`；`git diff --check` | PASS；语法/diff 检查，不是远程执行。 |

主回归后按新增/收紧的边界做了两组定向复验，**与主回归重叠，不相加当成独立总数**：

- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 60 $PY -m pytest -q tests/test_review_remediation.py tests/test_infra_codebook.py`：48 passed、20 subtests passed，覆盖原子导出发布、重复/近 tie prototype、非等权真实 VQ decoder 导出重建和全部码字/half-up 边界；[输出](review_remediation_evidence/vq_edges.log)。
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 45 $PY -m pytest -q tests/test_review_remediation.py::test_selection_record_cli_survives_new_best tests/test_review_remediation.py::test_explicit_protocol_and_immutable_handoff tests/test_review_remediation.py::test_config_consumption_and_child_contract tests/test_infra_evaluation.py`：11 passed、8 subtests passed；[输出](review_remediation_evidence/handoff.log)。调用真实 eval/demo CLI 解析、固定记录加载与产物保存，模型/runner 为本地 fixture。
- 最后完成所有修改 Python 文件 AST、62 个编号唯一覆盖、README/报告本地链接、shell 语法、`git diff --check` 检查：PASS。实际小型 DP3 构造后的摘要显示 `MultiStagePointNet input_channels=6/out_dim=16/hidden_channels=128/num_layers=4`、真实 observation fields 与参数统计；不是 YAML 回显。最终 HEAD 仍为起始 `f55a02f`，实现未提交。

初始检查经历的失败没有隐藏：首次整组运行在 worker 路径停滞后中止；沙箱内随机 Manager 明确 socket 权限失败。新测试初版暴露预测 key、normalizer fixture 添加顺序、action key fixture 和 mock 签名问题，随后修正并复跑。历史提交中的 81/92/6 等数量不计作本次 PASS，也不累加去重。源码修改后末轮检查见原始日志；不以提交说明替代执行证据。

## 62 项完整处置表

路径省略 `dexmani_policy/` 前缀时指包内路径，`scripts/`、`tests/`、`review_remediation_evidence/` 除外。T/C/G/B/P/S 仅引用上述本次证据；“源码检查”没有暗示运行通过。

| 编号 | 实现状态 | 代码/处置依据 | 实际验证 | 兼容边界与剩余问题 |
|---|---|---|---|---|
| B10 | ALREADY_FIXED | `scripts/training/train_vq_hand.py::load_policy_config/build_policy_dataset/prepare_policy_data` | T：本次重新运行 tests/test_policy_vq_alignment.py | 保留共享 split、有效 action 源行和 hand affine；无新 holdout；实现来自 d93eff2，不虚构历史 PASS |
| R01 | PRESERVED | `datasets/base_dataset.py::iter_normalization_data` | T：源行/验证极值检查 | 训练合格窗口去重源行统计；验证不参与 fit |
| E03 | ALREADY_FIXED | `datasets/replay_buffer.py; base_dataset.py; agents/normalization.py::fit_field_chunks` | T：本次 streaming 单遍/所有权/角色读取/进程测试；B | 保留 lazy reader、进程局部有界 cache、mixed action 单遍统计；源行索引仍 O(N) |
| E02 | PARTIAL | `train_ddp.py::ddp_worker → build_dataset_and_normalizer` | 源码确认：payload 已流式；B 仅单卡启动统计，非跨 rank 证据 | 每 rank 重复扫描/fit 尚在；仅1 GPU，不能证明 rank0 分发收益及失败传播，暂缓余项 |
| B01 | IMPLEMENTED | `scripts/training/train_vq_hand.py; training/run_identity.py; scripts/training/extract_vq_codebook.py` | T：新 run、重复 run、legacy 产物和导出冲突；旧字节保持 | 目录不能复用；导出覆盖需 --overwrite；无 VQ resume |
| B03 | IMPLEMENTED | `scripts/training/train_vq_hand.py::train/evaluate_vq` | T：有/无验证、NaN/Inf 保留 best、257 样本尾批 | 固定选点名写入 split_metadata；异常可能留下诊断 epoch checkpoint，不能当 best |
| E06 | IMPLEMENTED | `scripts/training/measure_vq_usage.py::measure` | T：chunk=1/7/4096、重复 prototype/近 tie，与原 full distance/count/quantile 公式对照 | 保留 raw 最近 prototype 与 normalized nn_l2 区别；仍物化低维 hand 和 O(N) 精确距离，无提速结论 |
| A08 | PRESERVED | `agents/vq_hand/vqvae.py; scripts/training/measure_vq_usage.py` | T：codebook/usage 回归 | learned softmax 层权重；encoder tuple 与 runtime prototype usage 分开 |
| U01 | PRESERVED | `agents/vq_hand/codebook_manager.py::extract_from_vqvae` | T：非等权导出/重建检查 | 不引入上游等权导出假设 |
| U02 | PRESERVED | `agents/core/dqrise.py::initialize_training/_require_codebook` | T/G：训练/预测/自包含恢复 | 码本在模型 state 中；预测不重复读NPZ |
| D06 | PRESERVED | `agents/vq_hand/codebook_manager.py::continuous_index_to_hand_pose` | T：码字往返及边界检查 | 保持 K−1 缩放、half-up/clamp |
| B06 | IMPLEMENTED | `agents/obs_encoder/pointcloud/registry.py; uni3d.py; agents/core/dp3.py; datasets/multi_task_dataset.py` | T：实际 idp3 层/宽度、global token、错键、child action 合同；C/G；logging.print_param_count 读取实际模块/输入字段/可训练参数 | factory 明列可用键；DP3 外层 FPS 仍在 preprocess 生效；未知键不再静默忽略；child auxiliary/shape 不一致拒绝 |
| B05 | IMPLEMENTED | `agents/action_decoders/backbone/unet1d.py` | T：H=15/16/18/20，两/三层，[12,24,48]/groups=4；G | 非法 kernel/groups/forward horizon 拒绝，不 pad/crop；旧加载 fixture 改合法 H=4 |
| B08 | IMPLEMENTED | `agents/action_decoders/time_sampler.py; consistency_flow.py` | T：K=10 B=1/2/3 拒绝，合法 B 与旧 RNG 公式一致；consistency 子批次 | 实际子批次检查，分流后提前检查；合法旧分布不变 |
| B09 | IMPLEMENTED | `datasets/augmentation.py::PointColorJitter` | T：定值 brightness 非恒等/恒等；已有颜色相关回归 | 四种颜色变换均使用定值；零宽区间不新增 RNG；旧错误非恒等固定配方需新 run |
| B13 | IMPLEMENTED | `training/build_utils.py; trainer.py; lr_scheduler.py; scripts/training/train_vq_hand.py` | T：bool/小数/NaN/Inf/0 控制量拒绝；原累积尾组检查；C | CLI/config/直接 Trainer 校验；warmup=0 仍合法 |
| A02 | IMPLEMENTED | `agents/action_decoders/backbone/attention.py::CrossAttention` | T：fused/manual、非零 bias、空/部分 mask 输出及梯度 | 全无效行在最终投影/dropout 后清零；未接 mask 的默认路径不受影响 |
| A03 | IMPLEMENTED | `agents/obs_encoder/rgb/geometry_processor.py` | T：NaN/Inf 深度、全空 patch、ratio=0、非法 intrinsics | 投影前 where，变换后清零，count>0；校验非有限/奇异标定；未将几何分支接入策略 |
| B02 | IMPLEMENTED | `agents/obs_encoder/rgb/base.py::project_features; dino.py; clip.py; siglip.py` | T：三个 encoder 的受支持 patch/CLS/pooler 局部真实 Linear，无 autocast；P：DINO LoRA | Identity 不 cast；完整 CLIP/SigLIP 预训练恢复 NOT VERIFIED；不改变 backbone/EMA dtype |
| R04 | PRESERVED | `agents/obs_encoder/rgb/base.py::set_tune_mode; review_remediation_evidence/rgb_precision_probe.py` | P：当前 DINO LoRA BF16 三步非零梯度/更新，strict restore | 未证明默认精度错误；短 encoder probe 不代表完整 policy 或长期收敛；不采用 FP32 新配方 |
| E04 | PRESERVED | `training/ema_model.py::_step_loop/_step_foreach` | P：实际 DINO 参数一步 loop/foreach 最大绝对误差0；T：EMA 恢复 | 没有整步收益对照，不启用 foreach 默认、不缓存参数/删buffer copy |
| E05 | DEFERRED | `agents/obs_encoder/rgb/image_processor.py; utils.py::to_rgb_tensor` | 源码检查：mean/std 按调用搬迁，float 每次范围检查；无端到端基准 | 默认 DP3/DQ 不走 RGB；常量 cache 候选暂缓，保留外部 float 校验 |
| A09 | PRESERVED | `agents/obs_encoder/rgb/r3m.py` | 源码检查；完整 R3M 训练 NOT VERIFIED | 既定 BN→GN 适配，已在配方披露，不回退 BN |
| S08 | IMPLEMENTED | `train.py; train_ddp.py; training/build_utils.py; trainer.py::load_for_resume` | T：提供 payload 不再读盘；连续/中断恢复逐位对照；G：保存恢复 | 每进程入口只 load 一次；恢复后清除 CPU payload 引用；生产 DDP 复验仍缺第二 GPU |
| D03 | PRESERVED | `training/ema_model.py; training/trainer.py` | T/P/G：teacher/EMA 相关检查 | teacher 保持 eval |
| D04 | PRESERVED | `training/resume.py::restore_training_state` | T：连续与恢复 EMA 权重/时钟相同 | EMA updater 时钟与模型一起恢复 |
| D05 | PRESERVED | `training/trainer.py::train/train_one_step` | T：4/4/2 尾组、连续恢复；G：一次真实更新 | optimizer/scheduler/EMA 按 optimizer step 前进 |
| D09 | PRESERVED | `agents/action_decoders/consistency_flow.py::compute_loss` | T：既有小 batch 测试与分流拒绝检查 | B=1 flow-only；其余样本守恒，不移植独立 floor 分流 |
| B12 | IMPLEMENTED | `training/trainer.py::_init_milestone_state/_check_milestone` | T：总步数 1/2/3/4/5/103，最终100pct、单 step 单文件、恢复跳过；selector 回归 | ceil 整数映射，碰撞保留最大 pct；不补写历史 run、不强凑五个候选 |
| R08 | PARTIAL | `tests/test_policy_vq_alignment.py; smoke_test.py; tests/test_infra_resume.py; tests/test_infra_cuda.py` | T：tiny VQ 优化→导出→实际 DQ 优化/预测；G：默认 DP3/DQ 单次更新/预测/raw+EMA 恢复；CPU 精确续训 | 保留原 streaming/benchmark/RTC 检查；生产 flags 已写入双卡测试，但 2 项 CUDA/DDP 测试 SKIP，仅1 GPU；单 rank 失败传播仍 NOT VERIFIED |
| E01 | DEFERRED | `training/trainer.py::train_one_step/apply_gradient_step; agents/core/dqrise.py::compute_loss` | 源码检查：micro-batch 日志与 DQ 诊断 host 标量仍在；G 非计时 profile | 没有日志同步占比证据；不删 finite/grad norm/rank 协调；暂缓日志优化 |
| B07 | IMPLEMENTED | `agents/obs_encoder/pointcloud/uni3d.py::PatchDropout/forward; r3d_obs_encoder.py` | T：真实 timm blocks 与可识别 centers，保留索引核对 PE；eval 不减 token | R3D 仅 pointsam；独立 Uni3D cls/max_pooling 保留；默认 dropout=0 路径不变；完整 R3D CUDA NOT VERIFIED |
| E09 | DEFERRED | `agents/obs_encoder/pointcloud/uni3d.py::KNNGrouper/knn_points; ops.py::knn_point` | 源码检查：cdist/topk 与 PyTorch3D 后端路径；没有同输入端到端对照 | tie/固定K/真实 R3D 整步收益尚未闭合，不替换后端 |
| E07 | DEFERRED | `agents/obs_encoder/pointcloud/uni3d.py::random_point_dropout` | 源码检查：逐云 ratio/mask，整行 first-point 替换 | 未验证新随机调用顺序的序列/恢复与净收益，保留旧 RNG |
| A01 | DEFERRED | `agents/obs_encoder/pointcloud/ops.py::farthest_point_sample` | 源码检查：零噪声/非零噪声、swap/remap/shuffle/gather 分支 | 未取得同输入 tie 与整步收益证据；不改原生 random_start_point |
| A06 | NOT_APPLICABLE | `agents/obs_encoder/pointcloud/uni3d.py::__init__/forward` | 源码检查：timm head 未调用；本轮无整理 head 的用途 | 不以未用 head 宣称默认 static_graph DDP 崩溃；保留 state keys |
| B11 | IMPLEMENTED | `deployment/runtime.py::LoadedPolicy.warmup` | T：实际 RTC CPU VJP；delay=0/3、cap=0、sync/async | 按配置后的 guidance_cap 热身，零延迟仍有 prefix；首次 CUDA 时延与真机 NOT VERIFIED |
| E08 | DEFERRED | `agents/action_decoders/rtc.py::guided_step/predict_rtc` | T：保留现有真实 RTC VJP 数值检查；源码检查 device→int 和 mask | 没有 CUDA 首次/稳态收益对照；保持 scheduler cast/sqrt、零权重排除、detach/VJP |
| R03 | IMPLEMENTED | `evaluation/protocol.py::load_seed_manifest/fixed_test_seeds; select_best_ckpt.py; eval_best_ckpt.py` | T：跨任务同整数、跨角色相交/缺失/池身份变化、训练 seed 改变后固定 test | 显式清单保存完整角色/内容与 runner 池 hash；多任务须能由现有 paired 映射表示；未编造生产 seed，仿真 NOT VERIFIED |
| R06 | IMPLEMENTED | `training/source_snapshot.py; training/workspace.py; scripts/training/train_vq_hand.py` | T：未跟踪源码进入归档、排除非源码/隐藏文件、内容修改变 hash、无Git unknown | 新 run 每次保存实际源码/依赖；归档失败直接报错；不隔离 lazy import；README 禁止运行中原位 sync；不改 resume 合同 |
| S03 | IMPLEMENTED | `select_best_ckpt.py --result-file; agents/loader.py; eval_best_ckpt.py/record_demo.py --selection-record; scripts/eval/eval_pipeline.sh` | T：另一 selector 更新 best 后旧结果不变；现有 eval/demo pinned 单次入口回归；S | 交接绑定 milestone/global_step/selection/inference/seed 清单；覆盖目标、缺证据、latest/symlink 拒绝；三阶段真实仿真 NOT VERIFIED |
| C03 | IMPLEMENTED | `本报告「当前配方与论文边界」；configs/*.yaml 与实际 Agent/runner` | DOC：源码/配置披露；C；仅 DP3/DQ 有 G，不代表论文实验 | 仅完成披露/受控实验设计，未运行论文比较或声称闭环成功率 |
| D08 | IMPLEMENTED | `本报告「当前配方与论文边界」；configs/*.yaml 与实际 Agent/runner` | DOC：源码/配置披露；C；仅 DP3/DQ 有 G，不代表论文实验 | 仅完成披露/受控实验设计，未运行论文比较或声称闭环成功率 |
| S01 | PARTIAL | `datasets/multi_task_dataset.py::_epoch_val/set_epoch` | T：deterministic 无 Manager；随机路径 spawn/persistent-worker epoch 回归 | 随机路径仍用 Manager 同步；本轮无随机多任务真实性能/精确恢复收益证据，保留 RNG/消费序列 |
| S05 | NOT_APPLICABLE | `agents/core/multi_task.py; agents/obs_encoder/text/clip.py` | 源码检查：冻结语言 encoder 与缓存仍服务现有输入 | 无固定任务发布请求，未创建 embedding 查表产物 |
| A04 | PRESERVED | `deployment/runtime.py::load_policy/LoadedPolicy` | T：现有 RTC/runtime 用例；真机 NOT VERIFIED | 仍只支持已声明 metadata/动作/观测合同；未扩大 MultiTask/真机接入 |
| B04 | NOT_APPLICABLE | `pyproject.toml; README.md editable 安装入口` | 源码检查：setuptools 未显式 configs package data | 无 wheel 分发用途，未改构建；wheel YAML 仍需以后验证 |
| R05 | DEFERRED | `agents/loader.py::restore_policy_agent; agents/obs_encoder/rgb/dino.py; text/clip.py` | P：已有缓存 offline DINO 构造/strict restore；冷缓存 NOT VERIFIED | 未提出离线冷缓存发布工作流；HF from_pretrained 仍需资源；不新增不完整导出格式 |
| S04 | NOT_APPLICABLE | `training/checkpoint.py; agents/loader.py` | G/T：完整 TrainCheckpoint 保存恢复仍可用 | 无轻量发布需求；不新建推理格式，推理/续训边界保留 |
| S02 | PRESERVED | `train_ddp.py::main/ddp_worker` | 源码检查；双卡 tests SKIP | 保留单机 mp.spawn，不迁移 torchrun |
| S07 | IMPLEMENTED | 后续用户明确要求清理后，删除无活跃调用的 `text/t5.py`、`backbone/ditx_rms.py`、`plugins/token_compressor.py` 及空 plugins 包说明 | 源码/脚本/测试/当前配置及本地 experiments YAML/JSON 未发现引用；清理复验见下节 | 先前暂缓记录由本次明确清理需求闭合；外部自定义 import/_target_ 若曾引用这些预留模块，需使用清理前源码；不提供占位兼容层，不宣称提速 |
| A05 | IMPLEMENTED | `scripts/remote/stop_remote.sh::list_sessions/ensure_server_reachable` | T：本地 ssh/tmux fixture，--all/--list × 无session/成功/命令错误/127/255；S | 先获取 tmux 状态，再本地 cut；保留真实退出码；未连接服务器执行 stop |
| C01 | IMPLEMENTED | `本报告「当前配方与论文边界」；configs/*.yaml 与实际 Agent/runner` | DOC：源码/配置披露；C；仅 DP3/DQ 有 G，不代表论文实验 | 仅完成披露/受控实验设计，未运行论文比较或声称闭环成功率 |
| C02 | IMPLEMENTED | `本报告「当前配方与论文边界」；configs/*.yaml 与实际 Agent/runner` | DOC：源码/配置披露；C；仅 DP3/DQ 有 G，不代表论文实验 | 仅完成披露/受控实验设计，未运行论文比较或声称闭环成功率 |
| C04 | IMPLEMENTED | `本报告「当前配方与论文边界」；configs/*.yaml 与实际 Agent/runner` | DOC：源码/配置披露；C；仅 DP3/DQ 有 G，不代表论文实验 | 仅完成披露/受控实验设计，未运行论文比较或声称闭环成功率 |
| D07 | IMPLEMENTED | `本报告「当前配方与论文边界」；configs/*.yaml 与实际 Agent/runner` | DOC：源码/配置披露；C；仅 DP3/DQ 有 G，不代表论文实验 | 仅完成披露/受控实验设计，未运行论文比较或声称闭环成功率 |
| R02 | DEFERRED | `agents/core/sat.py::SATObsEncoder.forward` | 源码检查：按 patch 序号跨帧拼接 | 物理点对应是新方法，未做共享锚点/匹配实验 |
| R07 | DEFERRED | `agents/action_decoders/consistency_flow.py::get_consistency_velocity` | 源码检查：absolute target_t_next 可超过1，relative 默认 | 外推目标研究暂缓；不孤立 clamp 分母不匹配的 target |
| S06 | DEFERRED | `agents/action_decoders/backbone/sat.py::SATBackbone.forward` | 源码检查：action/EJC 同步 shuffle 与 inverse；无原模型等价/收益对照 | 不以 toy 等变性推断 dropout/梯度分布等价，保留 shuffle |
| D01 | PRESERVED | `agents/obs_encoder/proprio/state_mlp.py; agents/core/sat.py` | 源码检查；C | 有序 MLP 保留固定机器人关节身份 |
| D02 | PRESERVED | `agents/core/sat.py::SATObsEncoder` | 源码检查；C | 不引入上游未接通的状态 padding mask 路径 |
| A07 | PRESERVED | `agents/action_decoders/consistency_flow.py::predict_action` | 源码检查；已有相关回归 | KV cache 限单次观测采样，不跨观测复用 |

## 当前配方与论文边界（C01/C02/C03/C04/D07/D08）

下面是当前 YAML 与源码的事实，不是历史实验配置。共同默认 `horizon=16 / n_obs_steps=2 / n_action_steps=8`，默认 100,000 optimizer updates；动作 joint 为 7+12，EEF 为 xyz+rot6d 9+12。统计遵循当前 action normalizer；历史实验须读各自保存配置，不能按本表反推。实际机器人单位与关节顺序沿用数据/runtime metadata，本次未转换。

| 方法 | 当前感知/状态与容量 | 动作/目标与默认 NFE | 默认预算与比较边界 |
|---|---|---|---|
| DP | DINOv2-small RGB LoRA、avg token、有序 state MLP；UNet [256,512,1024] | joint19；DDIM sample；10 | batch64；RGB 感知与点云方法不同，系统比较不能单独归因生成机制 |
| DP3 | XYZRGB PointNet + state；UNet [256,512,1024]；G 实际 68.76M 总参数（obs 0.25M，decoder 68.52M，显示值有四舍五入） | joint19；DDIM sample；10 | batch128；本地宽度/模态为适配，不能等同原论文完整配方 |
| DQ-RISE | iDP3 + state/多帧；UNet [256,512]；G 实际17.88M（obs0.27M，decoder17.61M） | 默认 EEF21 → TCP9+scalar index；16 个12维 hand prototype；DDIM sample；20 | batch128；不同于官方 RISE 感知，动作与容量也不同；不是仅把连续手部换成量化手部的单变量实验 |
| SAT | PointNext 96 patch + learned global token，token256；state 有序 MLP；跨帧按 patch 序号融合；8层×768，8 heads；patch attention 4层 | joint19；structured action/EJC 字段求和、shuffle；flow beta；10 | batch128；是 structured-action 思想适配，非完整官方感知/状态/训练复现；不保证跨帧物理点对应 |
| ManiFlow | PointNetDense128、continuous XYZ PE；12层×768，8 heads；本地 slot PE 机制见 backbone/consistency_ditx.py | joint19；flow beta + consistency discrete、dt uniform、relative；10 | batch128；decoder AdamW lr1e-4/wd1e-3/betas(.9,.95)，obs wd1e-6；通用/Dex 配方差异须逐项控制，不自动回退 |
| R3D | Uni3D EVA02 tiny、embed256、512 groups×32，预训练路径 `data/pretrained/uni3d`；OneWay depth4/embed256 | joint19；DDIM sample；10 | batch128；eval FPS 确定性是本地适配；本轮没有完整 R3D CUDA 质量/性能证据 |

除 ManiFlow 上述差异外，表中默认 policy AdamW lr1e-4、wd1e-6、betas(.95,.999)，obs lr1e-4/wd1e-6；policy EMA 默认启用（power .75），训练配置 BF16 autocast 默认启用。本轮 G smoke 是单步普通精度验证，不代表生产 BF16/compile 已验收。R3M 的 BN→GN 是已有适配，仍保留；当前默认 DP 是 DINO。实际参数量只报告 G 打印过的 DP3/DQ，未以理论估算冒充其它完整模型测量。

DQ VQ 与 policy 是两阶段配方：当前 VQ latent256、hidden512、**3 个 hidden Linear 层**、2组×4码、learned softmax group weights；kmeans_init=true、iters10、dead threshold2、字典 EMA decay .8；重建 L1 按手指权重，enc×3 + commitment×5；AdamW lr3e-4/betas(.95,.999)/wd1e-6、batch256、1500 epochs、warmup150。它不同于 policy optimizer/EMA；字典 EMA 与 policy EMA 不可混称。上游库内部未被训练循环调用的 optimizer 不是 policy optimizer。Legacy VQ 全量 hand 统计与目标 Policy 对齐入口不是同一配方；后者使用实际 Policy split，默认无 validation 时按 train_mse 选点。

**NFE、查询间隔、执行窗口和 ensemble 分开报告**：NFE 是每次预测的 DDIM/Euler 迭代次数；执行窗口由 n_action_steps/runtime 模式决定；查询间隔属于 runner/runtime 调度；temporal ensemble 若启用应标明发生在解码后的动作空间。DQ 先将 index 解码为 hand pose，不能把 index 平均与 pose 平均混为一谈。G 只测 predict 输出，未测真实查询时序、ensemble 闭环或控制频率。

后续论文实验（本轮未启动）：

- C01/C04：固定感知、动作维度/单位、H/obs/execution window、容量、预算、训练 seeds、selection/test 清单与 NFE，只改变 EJC 或对应待证明机制。
- C02/D07：固定 iDP3/state/动作/预算，对比 continuous hand 与 VQ hand；同时报告 held-out 重建误差、两种 usage、跳变及固定闭环清单成功率，再判断 16 个 prototype 是否不足。不自动扩大码本。
- C03/D08：系统级表明确所有上述差异；机制归因另做控制变量表，不从系统分数直接推导机制优劣。默认 DQ NFE20 与其它 NFE10 不能隐藏成同等推理预算。
- R02/R07：跨帧对应与 absolute 外推需先定义新目标，再做新实验；本轮显式 DEFER。

## 数值与性能证据、采用决定

B 的单次短测量：真实数据 21,293 rows、12,994 valid windows，batch16/workers0/random seed42，3 warmup+8 measured；启动 dataset+normalizer 1.853 s、数据等待 p50/p95/p99=25.368/26.467/26.572 ms，GPU H2D+forward+backward 平均10.232 ms，约464.85 samples/s；父进程 peak RSS1513.10 MiB、peak CUDA allocated1056.64 MiB。normalizer action_scale_sum=54.448864，完整 order hash/数据配置见 B 原始 JSON。**这是现状诊断，不是与基线比较的提速幅度，也没有测 CUDA overlap 的因果收益。** 本轮没有重测历史提交中的 GPU 对比，不复制其中未取得的数字。

P 的当前 DINO encoder（无自定义 projection，Identity）：223 个 frozen BF16 参数张量、144 个 trainable BF16 LoRA 张量，EMA 继承 dtype。3步 loss=6.91075/3.67673/2.80478；trainable 元素变化比例约0.5000/0.8905/0.8070，EMA 元素变化比例约0.5000/0.8905/0.7507；peak CUDA240.29 MiB。第一步约一半元素更新也与 LoRA 初始化有关，不能全部归因 BF16 量化；该小样本不证明长期精度无风险。相同实际参数的 loop/foreach 单次更新最大误差0，但未测完整 step 收益。**保留现有 BF16/EMA 配方与 foreach 默认值**，不写入未经验证的 FP32 新设置。B02 的非 Identity FP32 projection 已单独用真实 Linear 测试，无需依赖 autocast。

E06/S08 消除了源码可见的全量距离/tuple 中间量与重复读取，没有测得独立大模型端到端提速。其它 CHECK 的未采用理由见逐项表：缺完整数值/随机流/恢复及净收益证据时保留机制，不加永久开关占位。

## 用户入口与兼容边界

VQ 的完整命令见 [README](../README.md)。省略输出目录时自动创建 `experiments/vq_hand/<zarr_stem>/<timestamp_uuid>`；显式目录同样要原子认领。导出包含 checkpoint SHA256、split_metadata 和 normalizer SHA256/affine，推荐 `run/codebook.npz`。重跑要换 run；不会静默覆盖或重拟合旧产物。`measure_vq_usage --chunk-size 4096` 仅控制临时计算块，保留精确 p95/p99。

显式论文协议清单结构如下（**仅说明 schema 的小 fixture，不是有效生产 seed 清单**）：

```json
{"pool_id":"fixture-only","selection":{"task_a":[0,1]},"tie_break":{"task_a":[2]},"test":{"task_a":[3,4,5]}}
```

生产文件必须用实际 runner 的有效 task/seed 池预先制定；multi-task 每角色的 task→seed 列表须匹配现有参考序号映射。清单不足不会随机补选；只列70个 test 就按70个，不能宣称100个。完整 JSON、内容 SHA256 与 runner 池 SHA256 随 selection 保存，test 不受训练 seed 或是否触发 tie-break 影响。`episodes` 的 legacy 数量设置不截短显式 test 清单，实际分母仍由完成的 episode details 决定。

```bash
# 在实际任务和实验名下使用；这段不会生成/猜测生产 seeds。
PY=/home/zhanghaoyang/miniconda3/envs/policy/bin/python
EXP_NAME="已有实验名"
MANIFEST=/absolute/path/predeclared_seeds.json
HANDOFF=/absolute/path/new_selection_result.json
$PY dexmani_policy/select_best_ckpt.py --policy-name dp3 --task-name pour \
  --exp-name "$EXP_NAME" --result-file "$HANDOFF" "eval.seed_manifest=$MANIFEST"
$PY dexmani_policy/eval_best_ckpt.py --policy-name dp3 --task-name pour \
  --exp-name "$EXP_NAME" --selection-record "$HANDOFF" --no-videos
$PY dexmani_policy/record_demo.py --policy-name dp3 --task-name pour \
  --exp-name "$EXP_NAME" --selection-record "$HANDOFF"
# 或流水线（会运行真实评测/录制；本次未执行）：
SEED_MANIFEST="$MANIFEST" bash scripts/eval/eval_pipeline.sh dp3 pour "$EXP_NAME" --no-videos
```

手工 `best` 仍可读旧记录并支持既有覆盖优先级；新的 `--selection-record` 要求成功的 summary、immutable milestone、global_step 和 inference 身份，拒绝更改该交接的 raw/EMA/NFE。流水线三个进程共享自己生成的唯一结果文件，后来的 selector 更新 `best_ckpt.json` 不影响它。

兼容性说明：

- 无关数据、实验、checkpoint、视频、权重未删除/改写。B10/E03 及当前默认采样分布/动作意义/预算/NFE 保留。
- 修复前曾被忽略的配置键现在生效或拒绝：如 idp3 hidden_channels/num_layers 真正改变结构，不能把旧“写了但没生效”的 config 当成新结构 strict resume。需要旧语义的实验使用保存原代码，或在确证实际旧结构后单独处理，不能猜测迁移。
- 非法 UNet H/kernel/groups、过小 discrete_pow 子批次、不一致 child action layout 会更早报错；合法旧配置继续运行。PatchDropout>0 的旧错位行为、固定非恒等颜色变换、全无效 attention/geometry 边界的结果会改变，触发这些条件的历史研究不能假设逐位续训；新 run 应记录版本。
- B10 改过的两阶段数据/坐标口径可能要求待比较模型重训；历史分数是否受影响按保存配置/产物判断，代码修复不会自动使历史分数有效，也不无证据全盘作废。
- source.zip 只含运行源码/配置/构建依赖文件与清单，不含模型、数据或完整环境镜像；不证明外部仿真器/预训练模型已归档。无 .git 的远端 commit=unknown。归档不隔离 lazy import，运行中的源码目录不可原位 sync；hash 不成为额外 strict resume 拒绝条件。

## 剩余验证与复验入口

1. **NOT VERIFIED：生产双卡 NCCL/DDP**。当前只有1张 GPU。`tests/test_infra_cuda.py` 已使用生产 `find_unused_parameters=False, gradient_as_bucket_view=True, static_graph=True` 并设有限超时；在双卡环境运行 `OMP_NUM_THREADS=1 $PY -m pytest -q tests/test_infra_cuda.py`。这仍是 tiny DDP 保存/恢复检查，真实策略和单 rank 失败传播要在目标双卡环境补最短验证，不能将 skip 计为通过。
2. **NOT VERIFIED：完整 RGB projection 的预训练矩阵/冷缓存恢复、R3D CUDA、RTC 首次/稳态性能和真实闭环**。已完成对应局部 tensor/module 及 RTC input-VJP 回归。需要相关数据、权重、仿真服务后按目标配置执行 `$PY dexmani_policy/smoke_test.py dp` 或 `r3d`；RTC 性能需真实 LoadedPolicy/设备计时，真机必须另获明确运动授权。
3. E02/S01 余项仍有真实重复 fit/Manager 开销候选；只有目标多 rank/随机多任务性能与序列/恢复证据支持才改。其它性能候选同理，不以理论张量大小代替实测峰值。
4. 论文生产 seed 文件与仿真成功率没有生成。表中所有 DOC 已完成披露/设计；DEFER 和 USE 的处置不表示研究假设得到验证或发布功能已实现。

## 后续代码与文档清理（2026-10-06）

按用户新的清理要求，检查当前源码、脚本、测试、Hydra 配置以及本地 `experiments/` 中可见的 4 个 YAML/JSON 文件，未发现以下预留组件的活跃调用，删除其实现与自带演示：

- `agents/obs_encoder/text/t5.py`；text 包说明改为实际保留的 CLIP。
- `agents/action_decoders/backbone/ditx_rms.py`；活跃的 `consistency_ditx.py` 保留。
- `agents/obs_encoder/plugins/token_compressor.py` 及仅包含导入说明的 `plugins/__init__.py`。

删除范围共四个文件、713 行。没有更改活跃模型参数、state keys、默认配方、streaming、VQ 对齐或旧 checkpoint 读取机制，也未改写实验/权重。外部代码或仓库外保存的自定义配置若直接引用这些预留路径，须使用清理前版本；本地无调用不代表已核验全部外部使用。此次清理不作为速度或显存收益证据。

同步 README、架构、仿真评测及远程部署文档：修正全量 RAM 读取的旧图示，说明短训练里程碑合并与单次恢复读取，更新 VQ 独立 run/导出路径，区分 manifest 与 legacy seed 协议、普通 best 覆盖与固定 handoff，并补充源码归档和远端查询状态边界。评测模块 docstring 和 shell 帮助同步当前参数；任务书顶部指向本报告，其发布时要求保留供追溯。

本次清理实际复验（与前轮用例重叠，数量不累加）：

| 检查 | 命令/证据 | 结果 |
|---|---|---|
| 配置加载 | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 90 $PY dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat maniflow r3d multitask_dit ddp/dp ddp/dqrise ddp/sat ddp/maniflow ddp/r3d ddp/multitask_dit`；[日志](review_remediation_evidence/cleanup_config_only.log) | PASS：13 配置；仅加载/target 检查，不是 DDP 运行 |
| 现有定向回归 | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 90 $PY -m pytest -q tests/test_review_remediation.py tests/test_policy_vq_alignment.py tests/test_infra_evaluation.py tests/test_policy_rtc.py`；[日志](review_remediation_evidence/cleanup_pytest.log) | PASS：88 passed、8 subtests passed；局部 CPU/fixture 路径 |
| 静态与帮助 | Python AST、四个 eval shell 的 `bash -n` 与 `--help`、文档本地链接、62 个编号唯一覆盖、已删模块活跃引用搜索、`git diff --check` | PASS |

本次清理没有重跑真实 CUDA 训练/benchmark、仿真、远程操作或真机；这些路径的证据及未验证边界仍按前文记录。清理验收时 HEAD 为 `f55a02f6507901fcd126546789a01fc127b60996`，全部变更尚未提交。

随后用户明确授权将全部改动提交并 push 至 `origin/main`。报告及引用的 9 份原始日志一并纳入版本控制；上文 HEAD 与未提交说明保留为各验证阶段的历史状态，实际提交身份以 Git 记录为准。提交同步不增加训练或评测通过证据。
