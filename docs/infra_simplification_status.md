# 基础设施简化实施状态

实施日期：2026-10-25。原方案与执行要求来自 policy `84ab7a3ff8811ad4bb53559aefd44d086fdbed99`，通过 `git show` 完整读取，未为获取文档切换分支或覆盖工作区。开始时 policy HEAD=`cf3d70ee4ae7c8b56ba0f21a159364e9d8358e74`、real HEAD=`f2ec75621ede85d24fce8b189b348405926f642f`，两个工作区均干净；读取了各自 AGENTS、README 和验证约定。

## 成功标准与实际批次

| 批次 | 实际改动与落点 | 状态与边界 |
| --- | --- | --- |
| A | common/hand_representation.py 承担纯数据/手部语义；training 只保留批次与 KL 逻辑；Policy.training_data_config/effective_config 和组件实际构建值替代 builder 的内部结构及 LatentFlow 类型判断 | 已实施。训练状态、loss、EMA/DDP 主循环未改；CPU 恢复及 DP3 CUDA raw/EMA 恢复检查通过 |
| B | Policy 完整解码后唯一输出 future_actions；Runtime/SimRunner 只适配或消费；保存自动生成的 policy_metadata 并在加载时核对；迁移 context、smoke 与测试 | 已实施。N=2/H=16 得到 P=15，辅助维度不泄漏；Real float64、SIM 原 dtype、LAMP/LF context 与 sync 限制保留 |
| C | 平面 eval 配置、单次请求解析、一个 SimRunner、普通 rollout 与纯结果排名；原子 episode 前缀和单一统计；录像并入 evaluate；同步脚本先传候选事实再传指针 | 已实施。固定预算、冻结测试和均值修正是可见协议变化；真实仿真闭环未运行 |
| D | 数据 seed 独立，固定原 padding；保存实际 manifest、恢复直接复用；已知 trial 整组划分，真实稳定 source_id 支持搬迁 | 已实施。MultiTask 运行采样 seed 仍跟随训练 seed；未修改历史数据、划分或实验产物 |
| E | Real 入口解析 best/default、preflight 校验模式；RTC 参数按调用传递；Runtime 删除执行配置状态与 scheduler 访问；轻量结果必存、Raw 可选 | 已成对应用到实际 dexmani_real；6 项纯离线检查通过。设备连接前预热、单在途查询、动态授权/新鲜度/停止/失败收尾保留；真机未验证 |
| F | 审阅现有耗时与数值路径，未加入性能配置、缓存、精度或 compile 改动 | 无足够测量证据支持进一步优化，不宣称性能收益 |

本地提交位于两个仓库的 `infra/simplification` 分支，原 main 保持基线，未推送或合并：

| 仓库 | 提交 | 范围 |
| --- | --- | --- |
| policy | `cb7b5a2` | A：共享语义与实际配置归属 |
| policy | `4ae573d` | B：完整未来动作与调用接口 |
| policy | `52828c6` | C：评测结构、固定预算及冻结/统计协议 |
| policy | `268924d` | D：固定数据划分、seed 与 padding |
| real | `dbd4610` | B/E：入口解析、模式准入与按调用 RTC 参数 |
| real | `c0d4efc` | E：轻量结果必存、Raw 可选与离线检查 |

拆分后的 A/B/C/D 快照另在隔离副本逐批通过 41/41/51/53 项 CPU 检查；最终增加的 warmup 成功/失败后 RNG reset 断言也已定向通过。文档随单独提交维护。

当前使用入口集中维护在 [README](../README.md)、[项目架构](项目架构.md)、[仿真评测机制](仿真评测机制.md) 与 Real README；[方案](infra_simplification_plan.md) 保留基线证据并补充实施调整。

## 保持行为与协议变化

保持完整 H 解码、训练窗口及 padding 有效值、normalizer 统计总体、零采样权重 child 参与统计、LatentFlow 原始 reference、LAMP 实际提交/应用命令历史、完整训练状态恢复、当前 dtype/compile/autograd 路径及模式限制。恢复不承诺多 worker 增强与预取逐位重放。

核对原 Real run.py 后保留 execution.action_steps → n_action_steps 的默认规则；best 仅提供 checkpoint、权重和 NFE，不将仿真选点 C 隐式带入真机。

明确变化如下：

- 固定预算：所有候选每任务使用相同 K，当前配置 K=25、test=70，移除平局追加。test 前缀不随 K 改变。
- 统计：avg_steps 直接聚合成功且有有效步数的 episode，保留浮点；不再对任务均值等权平均或截断为整数。
- 冻结测试：显式选择记录只允许 device/output 变化；普通 best 可覆盖研究参数，但始终不是正式 held-out 证据。
- 配置/产物：移除 eval 分区、多 NFE 协调、重复 CLI 别名、独立 demo 入口和重复结果快照。一个请求写一个 result_details.json，选择记录只引用候选事实。
- 数据：普通新实验 split seed=42、LF=0；training.seed 不再改变 child split。已知 trial 不跨 train/val，无法容纳整组时不拆 trial。原 padding 普通=7、DQ=15、LAMP/LF=0。
- 真机结果：移除 record_results=False / --no-results；轻量配置与结果必存。record_raw=True 默认不变，--no-record 只关闭 Raw，设备需求仍按控制/录制并集确定。

当前产物要求自动生成的 policy_metadata 和新选择记录结构。历史实验使用保存事实与对应旧源码；未自动补齐旧 checkpoint、改写 best、重生成历史 split 或增加永久兼容层。未计算内容哈希；结构相同的原地数据修改、dirty 源码变化不构成可验证的内容身份。

## 验证矩阵

使用已存在的 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`，未升级依赖或降低正式配置。CPU 检查设置 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。

| 类别 | 结果 | 实际覆盖与限制 |
| --- | --- | --- |
| 配置 | PASS | 9 个单卡配置 + 6 个 DDP overlay 的 config-only/target 导入；不表示 DDP 运行通过 |
| 合成 CPU | PASS | 53 项 unittest：窗口/normalizer、DQ/LAMP/LF、raw/EMA/续训下一步、完整未来与 dtype、RTC 局部 autograd、固定预算/平局/全失败、精确 seeds、冻结、失败前缀与收尾、manifest 身份 |
| 真实数据 | PASS | 原配置 dp3、latent_flow 各最多 2 个 train/val batch；DP3 12,994 训练窗口，LF 11,013 训练和 1,667 验证窗口；包含原始 reference 和完整统计总体 |
| CUDA/BF16/compile | PASS（DP3） | RTX 4090、Torch 2.4.1+cu124，原配置 4 次更新、预测、raw/EMA 保存恢复；compiled/eager 最大绝对差约 0.00103/0.00121，在已有 smoke 容差内 |
| 其他方法实际 CUDA | NOT VERIFIED | 未执行；缺少 DQ Codebook、LAMP prior、LF codec 等匹配实际产物，不能用合成资产代替 |
| 多卡 DDP | NOT VERIFIED | 本次仅确认 1 张可用 GPU；未启动多进程 DDP，CPU 模拟 rank 不替代 DDP |
| 仿真闭环 | NOT VERIFIED | 未运行真实 dexmani_sim rollout；未准备匹配当前格式的已训练实验，合成环境仅验证流程与统计 |
| Real 纯离线逻辑 | PASS | 6 项 unittest，隔离副本和实际配套仓库均通过：模式、迟到/过期/撤权、bootstrap/RTC/时钟、reset 旧结果、单在途/串行关闭、两种记录组合及写入失败 |
| 真机 | NOT VERIFIED | 未获设备连接、HOME、运动、实时采集、回放或标定写入授权，均未执行 |
| 性能/闭环质量 | NOT VERIFIED | 未做受控延迟/吞吐/成功率实验，不宣称提速、质量提升或硬件安全 |

实际检查命令（python 指上述解释器）：

```bash
python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise sat r3d maniflow multitask_dit lamp latent_flow
python dexmani_policy/smoke_test.py --config-only ddp/dp ddp/dqrise ddp/maniflow ddp/multitask_dit ddp/r3d ddp/sat
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
python dexmani_policy/smoke_test.py --data-only dp3 latent_flow --max-batches 2
timeout --signal=INT 600s python dexmani_policy/smoke_test.py dp3 --max-updates 4
PYTHONPYCACHEPREFIX=/tmp/dexmani-compile-cache python -m compileall -q dexmani_policy
# dexmani_real 根目录，PYTHONPATH 指向两个实际仓库
python scripts/check_deployment.py
PYTHONPYCACHEPREFIX=/tmp/dexmani-real-compile-cache python -m compileall -q dexmani_real scripts
git diff --check
```

另实际通过 evaluate/select_checkpoint/eval_pipeline 及 Real run_policy 的 `--help`，四个相关 Shell 的 `bash -n`，以及 10 份文档的本地链接路径检查。

日志位于本机 `/tmp/dexmani-config.log`、`/tmp/dexmani-final-cpu.log`、`/tmp/dexmani-data-host.log`、`/tmp/dexmani-cuda-smoke.log`、`/tmp/dexmani-real-paired-tests.log`；临时目录不是永久实验归档。

初次真实数据 smoke 被 sandbox 的多 worker IPC 权限阻断，主动中断并清理；随后获自动审批在 sandbox 外使用原配置通过。sandbox 内 CUDA 不可见，外部探测确认 RTX 4090 并完成短 smoke，未把 sandbox 限制解释为无 GPU。Black 多进程退出受阻后停止，使用相同工具的串行 API 核对 AST 等价。

## 跨仓库配套与剩余工作

Real 改动先在 `/tmp/dexmani-infra-real` 隔离副本验证；重新核对原仓库 HEAD 与干净 diff 后，应用补丁到 `/home/zhanghaoyang/Desktop/dexmani_real`。涉及 deployment/run.py、preflight.py、session.py、episode.py、scripts/run_policy.py、新增 scripts/check_deployment.py 和 README。没有修改设备 SDK、schedule 算法、Raw 格式、物理标定或实验产物。

后续需要匹配真实手部产物的 CUDA 检查、多卡恢复验证、真实模拟器短闭环及另行授权的真机验证。性能优化以测量为依据。本次不推送、不合并，不发布只有一侧的接口迁移。
