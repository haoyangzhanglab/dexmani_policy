# 历史修复与验证记录

本文记录当时的实现与验证，不作为当前接口规范；文中的测试结果来自已移除的测试套件。权重 SHA-256 机制现已移除，当前行为见[仿真评测机制](仿真评测机制.md#checkpoint-selection)。

本轮基线为 `d349a878fb3fd9802e17a120aa5fab51ec9dac82`，开始时工作区干净。以下仅记录本轮实际修改与验证；未修改 AGENTS.md、默认 YAML、外部仿真仓库或依赖，未提交数据、模型和日志，未 push 或创建 PR。

## 修复与删除

- **P0：恢复配方。** `recipe_content` 直接排除 `data_recipe[].source_evidence`。采集 task/seed 信息仍供评测报告使用，split/source 身份、窗口、normalization、模型、optimizer 和采样配方仍比较；normalizer 从 checkpoint 恢复。删除“附加审计信息不同必须拒绝恢复”的测试要求。
- **P0：普通 best。** best 必须有有效 SHA-256，普通入口将它传给已有 rollout 加载前检查。同路径、同 step 换权重也在 episode 前失败；没有额外反序列化或 selection 排名。
- **P0：恢复与正式选点。** 恢复只归档现存且 step≤恢复点的有效候选，允许历史候选缺失；损坏、冲突、配方/step/百分比错误仍失败，不复制未来候选或回溯父链。正式 selection 在哈希和仿真前检查预算对应的完整集合，报告缺失 step；独立 verifier 也拒绝部分集合。
- **P1：依赖边界。** 删除 Sparse3D/R3D 配置校验中的 PyTorch3D 原生加载，保留模型构造和实际 FPS/kNN 的明确错误，不引入替代后端。VQ/LAMP 数据准备无需点云网络。
- **P1：评测测试。** 外部模拟器源码发现独立为一个小函数；测试只替换该来源及环境创建。真实 Policy、Runner、checkpoint、runtime 数值记录、ranking 和 verifier 均参与测试，未创建模拟 Python 包或降低 verifier 要求。
- **P1：重复工作。** `publish_best`、`select_best_checkpoint` 返回已验证 record 与 checkpoint 路径，pipeline 直接构造 frozen request。完整 verifier 调用由两次降为一次；独立 Frozen Test 仍完整复核，真正加载前仍检查权重。保留候选冻结、各次加载和发布时的必要哈希，不引入缓存或 skip 标志。删除已由 step 唯一性和文件名检查覆盖的重复路径校验。
- **P2：R3D。** README 提供 B=4、gradient accumulation=32 的单卡命令。默认 B=128/A=1 不变；nominal batch 相同不意味着随机过程、尾组或结果逐位一致。

## 保留的算法与实现

按用户指定的固定提交查阅相关源码：

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py)：核对 noise/scheduler/Generator、训练目标与动作时间窗口，并查阅 conditional_unet1d、Dataset、Normalizer、Workspace。保留本地 `scale*x+bias` FiLM、物理动作恢复、完整 `future_actions`，scheduler/EMA 仍按有效 optimizer update 推进。
- [ManiFlow](https://github.com/geyan21/ManiFlow_Policy/blob/ef2f116f1f90163ed36e657b8c5503740bb468af/ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py)：核对插值、velocity target、EMA endpoint consistency、时间采样与 DiTX。保留在线 encoder 条件、EMA decoder teacher、非空且不丢余数的分组、两个独立均值相加、frame-only 位置编码和单次请求 K/V。
- [R3D](https://github.com/Wushr-Lance/R3D-Policy/blob/e637c0148376ddc4b5e667fa8f8e108cb8ff7a85/R3D/r3d/policy/dp3.py)：核对 Uni3D、点云 normalization/clipping、空间条件及 diffusion backbone。保留明确的 19D joint 与 9D EEF 分组，不采用上游按动作宽度对半切片。

SAT 关节 token、DQ-RISE 码本 lookup、LAMP 历史命令与单步预测、LatentFlow 相对 arm/HandVAE 解码梯度、MultiTaskDiT 文本与采样语义均未改动，继续运行真实模型契约回归。

评测入口共用的 seed、runtime 数值记录及 selection/test 比较机制保留。Dataset 按块读取和限定 episode 扫描、推理 mmap/释放无用状态、EMA 与 MultiTaskSampler 均保留；没有新的端到端证据支持专用 rank EMA、sampler 重写或 K/V 仿射合并，因此未采用这些优化。未新增配置、持久化状态、旧格式兼容或迁移工具。

## 实际验证

环境：Conda `policy`，Python 3.10.20、Torch 2.4.1+cu124；沙箱内执行 CPU 检查，沙箱外执行 CUDA/IPC/真实仿真检查。GPU 为一张 RTX 4090，24 GiB。

| 检查 | 本轮结果与范围 |
| --- | --- |
| 完整 unittest | **36 tests 通过，无 skip**，最终 18.385 s。覆盖新格式 fresh/save/resume、改变审计 seed 后恢复、真正语义改变拒绝、raw/EMA 权重和输出、缺 milestone 续训、归档冲突/搬迁/EXDEV、完整候选与换权重拒绝。 |
| 模拟评测 | 无 `dexmani_sim` 时，五候选×两次 selection episode、一次 pipeline frozen episode、一次独立 frozen episode及一次普通 best episode全部执行。Policy/Runner/恢复/排名/验真实际运行；只模拟环境与外部源码身份。计数 wrapper 调用真实 verifier，确认 pipeline 一次、独立 frozen 一次。 |
| 配置与预训练准备 | 九法默认配置 smoke 全通过；独立进程屏蔽 PyTorch3D 后九法仍通过，真实点云构造明确拒绝缺后端。真实 VQ/LAMP Dataset 和准备函数在屏蔽后端时通过。 |
| CUDA 专项 | **4 tests 通过**，3.999 s；九法小模型 FP32 forward/backward、采样、raw/EMA strict restore、实例 RNG/RTC、VQ/文本设备、4→6 updates 续训。新增 R3D 辅助模式覆盖 28D 训练目标、19D 控制输出和加权 loss；CUDA 续训覆盖缺历史 milestone。 |
| 两进程 DDP | CPU/Gloo 真实 builder/Trainer 4→6 updates 通过，包括尾累积、参数同步、各 rank RNG、rank 0 归档和缺历史候选。未将其称为双 GPU/NCCL 验证。 |
| R3D 低显存实际训练 | 真实 pick_apple_messy 数据及本地 Uni3D 预训练权重，默认网络/bf16/compile/8 workers，B=4/A=32，**2 个 optimizer updates**，参数更新和 raw/EMA 恢复通过。PyTorch peak allocated **1.638 GiB**、reserved **2.242 GiB**，不是整卡显存占用。compiled/eager 预测最大绝对差约 0.001615，仅记录该差异；checkpoint 往返按已有 smoke 使用 eager 对照，不宣称编译输出逐位一致。 |
| 真实开发仿真 | 小型 DP3 使用真实数据训练 2 updates，普通 latest 评测在正式 seed 池外的 **420001** 运行 1 episode，动作上限 4；结果 `completed`、4 steps、success=false，无视频。仅证明实际恢复与仿真调用链连通，不是任务成功率实验。 |
| 静态检查 | `compileall`、`git diff --check` 通过；AGENTS.md 与默认 configs 无改动。 |

仍可使用的基础检查命令（激活 `policy` 环境后）：

```bash
python dexmani_policy/smoke_test.py --config-only dp dp3 sat r3d maniflow multitask_dit dqrise lamp latent_flow
python -m compileall -q dexmani_policy
git diff --check
```

R3D 实测通过现有 `load_smoke_config('r3d')` 设置 B=4/A=32 后调用 `validate_training(cfg, max_updates=2)`，临时训练产物自动清理。真实开发仿真脚本在 `/tmp/dexmani_fix_real_rollout.py`；该脚本及 `/tmp/dexmani_fix_*.log` 仅为本机检查产物，未加入仓库。

## 未验证范围

未执行长训练、双 GPU/NCCL、完整预算的真实仿真 selection/frozen test 或真实机器人操作。低显存结果只对应上述配置与本机短跑，不替代其他数据规模的测量。训练/测试场景来源独立性仍为 `unverified`，整数 seed 交集不能证明场景重叠或独立。
