# 论文实验 recipe 与六项修正记录

更新日期：2026-10-06。代码基线为 `b93cbc64882e62c36ee732003b84d463e0393d38`，下述“当前默认”指该基线加本轮六项修正及后续清理。任务来源：[六项任务书](../CODEX_PAPER_READINESS_TASKS.md)。这是实现与低成本验证记录，不是策略质量验收或论文结果表。

## 当前方法机制与本地适配

下表由有效 Hydra 配置及其实际消费者核对。除 SAT 外，本轮没有固定并复核其他方法的上游 revision；这些方法与原论文/上游的完整对应均为 **NOT VERIFIED**，不能将方法名解释为官方 recipe 的逐项复现。

| 方法与配置 | 当前保留的机制、观测与 encoder | 预测目标 / 损失 | 本地适配与来源边界 |
| --- | --- | --- | --- |
| [DP](../dexmani_policy/configs/dp.yaml) | RGB + joint state；DINOv2-small average patch feature + state MLP；条件 UNet `[256,512,1024]` | 100-step diffusion，`sample`/action MSE；DDIM 推理 | [DPAgent](../dexmani_policy/agents/core/dp.py) 使用 HF `facebook/dinov2-small`，LoRA rank 16 / alpha 32 / dropout .05 / RSLoRA；新 adapter FP32，冻结主干 BF16。上游精确 revision 未核实 |
| [DP3](../dexmani_policy/configs/dp3.yaml) | XYZRGB 1024 点 + state；全局 PointNet 128d + state MLP；同 DP 的 UNet | 100-step diffusion，`sample` MSE | [DP3Agent](../dexmani_policy/agents/core/dp3.py) → [PointNet registry](../dexmani_policy/agents/obs_encoder/pointcloud/registry.py)；scratch/full，本地点云颜色和数据适配；上游精确 revision 未核实 |
| [DQ-RISE](../dexmani_policy/configs/dqrise.yaml) | XYZRGB 1024 点 + state；`idp3` MultiStagePointNet 128d；UNet `[256,512]` | 100-step diffusion，`epsilon` MSE；内部预测 TCP 9 + 一个连续 code index，最后恢复 hand 12 | [DQRISEAgent](../dexmani_policy/agents/core/dqrise.py) 的控制输出为 21d；本地 VQ/PCA 排序 codebook（2 groups × 4），必须匹配 policy 数据窗口/归一化。scratch encoder，codebook 来自 `robot_data/sorted_hand_poses_${task_name}.npz`；上游精确 revision 未核实 |
| [ManiFlow](../dexmani_policy/configs/maniflow.yaml) | XYZRGB 1024 点 + state；dense PointNet 128d、8-frequency XYZ PE 投影、state 广播；两帧沿 token 维展开；12-layer / 768d ConsistencyDiTX | [ConsistencyFlowMatch](../dexmani_policy/agents/action_decoders/consistency_flow.py)：flow velocity MSE + EMA consistency MSE；batch 分配 .75/.25，两项均值直接相加 | [本地观测编码](../dexmani_policy/agents/core/maniflow.py) 为 scratch/full；flow 时间 beta，consistency 离散网格 10、dt uniform、target time relative；agent fallback NFE 10，实际 eval 为 4。上游精确 revision 未核实 |
| [SAT](../dexmani_policy/configs/sat.yaml) | XYZRGB 1024 点 + state；多尺度 PointNext、96 patches、256d token、4-layer patch attention、learned global prefix；8-layer / 768d SAT | [RectifiedFlow](../dexmani_policy/agents/action_decoders/rectified_flow.py) velocity MSE；训练时间 beta、网格参数 10；Euler 推理；训练 action-token shuffle | [本地 SAT](../dexmani_policy/agents/core/sat.py) 为 scratch/full；按 patch 槽位沿特征维融合两帧，state 广播进每个 token，三字段 EJC 投影求和。官方固定版本对照见下节；本轮只修无内部前缀分支 |
| [R3D](../dexmani_policy/configs/r3d.yaml) | XYZRGB + state；Uni3D `eva02_tiny_patch14_224`，512 groups × 32 points、PointSAM feature；256d / 4-layer OneWay backbone | 100-step diffusion，`sample` MSE | [R3DAgent](../dexmani_policy/agents/core/r3d.py) 默认 `use_aux_ee=false`，19d joint 控制；可选辅助 EEF 9d 是额外预测，控制仍取 joint 19。预训练读取 `data/pretrained/uni3d/model.safetensors`，代码下载源 `eddie-cui/r3d-weights`；选择性加载 `pc_encoder.*`，活跃路径 full tuning、无用 timm 参数冻结；上游/权重 revision 未核实 |
| [Multi-task DiT](../dexmani_policy/configs/multitask_dit.yaml) | RGB + state + task text；scratch ResNet18 + GroupNorm/full；冻结 CLIP text `openai/clip-vit-base-patch16`，缓存 task embeddings + trainable text projection；8-layer / 512d DiT | 默认 diffusion、100-step、`sample` MSE；配置中的 flow 参数在当前 diffusion 分支不生效 | [MultiTaskAgent](../dexmani_policy/agents/core/multi_task.py)；任务为 pick_bottle/open_box，文本为 “pick up the bottle” / “open the box”；不使用 RGB 通用默认 DINO 预训练。该组合为本地基线，上游精确对应未核实 |

动作和损失消费者分别为 [BaseAgent](../dexmani_policy/agents/core/base.py)、[Diffusion](../dexmani_policy/agents/action_decoders/diffusion.py) 及上表各 agent。默认 `action` 为 19 = arm joint 7 + hand 12；DQ-RISE 的 `action_ee` 为 21 = xyz 3 + rot6d 6 + hand 12。所有当前默认 `H / observation horizon / action chunk = 16 / 2 / 8`，执行片段从 `n_obs_steps - 1` 开始。字段维度描述不替代数据中的实际排列校验。

## 当前训练与评测配方

所有行都使用 100000 **optimizer updates**，候选为 20000/40000/60000/80000/100000；accumulation=1，`drop_last=true`，seed 默认 42，gradient clip=1，cosine scheduler，末端 LR ratio 的实现默认值为 .1。EMA 每个 optimizer update 更新一次：after=0、inv_gamma=1、power=.75、max=.9999，当前 `foreach` 缺省为 false。配置均启用 BF16 autocast 与 compile；这不表示参数以 BF16 存储，也不表示本轮验证了 CUDA/compile。

精度 **P**：冻结 DINO BF16，LoRA 参数/对应 AdamW `exp_avg`、`exp_avg_sq`/EMA shadow FP32；其他可训练模块保持 FP32。精度 **F**：按当前构造路径，参数及相应 AdamW 动量、EMA shadow 为 FP32；没有 LoRA，Multi-task 的冻结 text backbone 也是 FP32。F 是源码存储配方，未为所有完整模型加载权重运行验收。

| 有效配置 | per-rank × world × accumulation = global batch | AdamW LR / obs LR；weight decay / obs WD；betas；warmup | 存储 / autocast | 主评测 NFE / 权重 / 协议 |
| --- | --- | --- | --- | --- |
| dp | 64 × 1 × 1 = 64 | 1e-4 / 1e-4；1e-6 / 1e-6；.95/.999；500 | P / BF16 | 10 / EMA / S |
| dp3 | 128 × 1 × 1 = 128 | 同 dp | F / BF16 | 10 / EMA / S |
| dqrise | 128 × 1 × 1 = 128 | 3e-4 / 3e-4；1e-6 / 1e-6；.95/.999；2000 | F / BF16 | 20 / EMA / S |
| maniflow | 128 × 1 × 1 = 128 | 1e-4 / 1e-4；1e-3 / 1e-6；.9/.95；500 | F / BF16 | 4 / EMA / S |
| sat | 128 × 1 × 1 = 128 | 同 dp | F / BF16 | 10 / EMA / S |
| r3d | 128 × 1 × 1 = 128 | 同 dp | F / BF16 | 10 / EMA / S |
| multitask_dit | 64 × 1 × 1 = 64 | 同 dp | F / BF16 | 10 / EMA / M（待发布） |
| [ddp/dp](../dexmani_policy/configs/ddp/dp.yaml) | 48 × 4 × 1 = 192 | 同 dp | P / BF16 | 10 / EMA / S |
| [ddp/dqrise](../dexmani_policy/configs/ddp/dqrise.yaml) | 32 × 4 × 1 = 128 | 同 dqrise | F / BF16 | 20 / EMA / S |
| [ddp/maniflow](../dexmani_policy/configs/ddp/maniflow.yaml) | 32 × 4 × 1 = 128 | 同 maniflow | F / BF16 | 4 / EMA / S |
| [ddp/sat](../dexmani_policy/configs/ddp/sat.yaml) | 32 × 4 × 1 = 128 | 同 sat | F / BF16 | 10 / EMA / S |
| [ddp/r3d](../dexmani_policy/configs/ddp/r3d.yaml) | 32 × 4 × 1 = 128 | 同 r3d | F / BF16 | 10 / EMA / S |
| [ddp/multitask_dit](../dexmani_policy/configs/ddp/multitask_dit.yaml) | 16 × 4 × 1 = 64 | 同 multitask_dit | F / BF16 | 10 / EMA / M（待发布） |

单卡 workers=8，六个 overlay workers=4；overlay 没有统一改变其他研究配方。DP 的单卡与 DDP global batch 不相等。实际用户 overrides 和历史 resolved config 优先于本表。

### 数据、归一化与增强

单任务默认数据为 `robot_data/pick_apple_messy.zarr`，最多 80 条训练 episode、val_ratio=0、dataset seed=`training.seed`。多任务各读取 pick_bottle/open_box 的 Zarr，各最多 80 条、val_ratio=0，child dataset seed=`seed`，外层 balanced sampling seed=`training.seed`；仅覆盖 `training.seed` 不会同步改 child episode split seed。不同训练 seed 的 80 条上限不证明相同 episode；实际身份须记录 split、source rows 和 data revision。默认配置不提供可据以宣称数据版本一致的 revision。

[BaseDataset](../dexmani_policy/datasets/base_dataset.py) 先做 episode split/downsample，再按真实数据有效窗口采样；padding 为 before=1、after=7，obs_horizon=2。[MultiTaskDataset](../dexmani_policy/datasets/multi_task_dataset.py) 聚合各 child 的统计源。[normalizer 构造](../dexmani_policy/training/build_utils.py) 仅使用有效训练窗口涉及的唯一源行：joint state 为 limits，joint action 的 auto 为 limits；EEF action 的 xyz/hand 为 limits、rot6d 为 identity。DP3/DQ-RISE/ManiFlow/R3D 点云为 limits，SAT 点云为 identity。RGB 的配置归一化为 identity，实际 image processor 仍执行各 backbone 的像素预处理。

所有方法的 state Gaussian noise std=.0002、prob=1。点云方法 XYZ noise std=.002，color brightness=.125/contrast=.5/saturation=.5/hue=0，prob=1；DP3/DQ-RISE/SAT 另有 max dropout ratio=.8，ManiFlow/R3D 无此项。FPS 训练允许 random start/output shuffle，noise scale=0，eval 强制 deterministic。SAT 的 FPS 输出随机顺序与官方 attention 前独立 patch shuffle 是不同机制。

DP/Multi-task：resize 240×240、训练随机 crop 224×224、eval center crop；颜色增强 brightness=.3/contrast=.2/saturation=.2/hue=.05/grayscale=.1，prob=.25，noise/blur=0。新 DP `rgb_keep_uint8=true`，Multi-task child 未配置该字段，按 BaseDataset 缺省 false。像素 transport 与 LoRA 参数存储精度是独立变量。

### 固定协议身份与使用边界

**S** 为真实单任务清单 [pick_apple_messy.json](../dexmani_policy/configs/eval_protocols/pick_apple_messy.json)：实际池 200，selection/tie-break/test=25/5/100，partition seed=1066。来源为 simulator `DATA_DIR/eval_seeds/pick_apple_messy.txt`，读取时 simulator commit=`c3d56fd58f402fc6f67a635b9452e10eee935afe`。

- manifest canonical SHA256：`44362705d1bd5575dc766b4b98e7953e8b6ec770a357d0000b42df82c6ed4a3a`。
- 完整 paired runner pool SHA256：`5ac859dd04b29670ce5ad3bd309dc4c148e12040be5ea0469484247aa9c32f91`。
- hash 沿用 `json.dumps(..., sort_keys=True)` 的既有算法；不是文件原始字节 hash。pool hash 只涵盖完整 task→ordered physical seeds mapping，不含路径或时间。
- 已只读比较 A/B 原 manifest：三个角色的 seed 和顺序均相同；新清单有自己的 metadata/hash，没有覆盖 A/B 清单或改变其协议。

**M** 为默认 task-set `pick_bottle+open_box` 对应路径。本机两个任务均无实际 seed 文件，真实 runner 各 fallback 100，生成器实际拒绝 25+5+100；**真实清单待发布 / NOT VERIFIED**，没有提交占位 JSON。必须准备足够实际 paired seeds 后一次发布；若事先决定更小 test，应显式生成另一份协议、记录新计数，不能自动截断。若以后比较单/多任务同一任务，需核对各角色的 physical seeds 完全一致；当前 paired mapping 无法表达的配对不能宣称成立。

新 selection 必须指定有效 manifest，完整预留 tie seeds；选点后 pinned eval/demo 使用不可变记录中的内嵌内容，原路径删除或修改不影响默认 handoff。显式 `eval.seed_manifest=...` 必须与记录 canonical hash 相同，显式 null 也不能解除合同；初次选点和后续评测都核对完整池身份。demo 仍按原规则挑可视化 seeds，不计入 held-out 指标。历史无 manifest 的记录只沿 legacy 路径复现，不能贴上新协议冒充重新选点。

完整正常的全零 selection 使用原排序（成功率、成功步数、较大 global_step），发布 `selection.selection_all_zero=true` 和流程 `status=success`，然后进行完整 test；空结果、episode 缺失/重复/错误、技术异常失败并保留旧成功指针。旧记录缺此字段表示“未记录”，不推断非全零。

主结果应预先固定各方法的 NFE/EMA/协议。NFE sweep 使用同一 test 并完整披露用途，不从 test 最大成功率反选主表。多训练 seed 的波动与单次 episode 的 Wilson 区间是不同不确定性来源，不能互相替代。

## SAT 官方固定源码对照

实际读取了 [XiaohanLei/SAT](https://github.com/XiaohanLei/SAT/tree/cd7c0a8877d6090a9a85ebee0ceca961830b3654) 的固定 revision `cd7c0a8877d6090a9a85ebee0ceca961830b3654`，包括 [sat.yaml](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/config/sat.yaml)、[obs_tokenizer.py](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/vision/obs_tokenizer.py)、[policy/sat.py](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/policy/sat.py)、[model/diffusion/sat.py](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/diffusion/sat.py)。没有安装或运行官方训练栈；源码机制核对不等于官方可运行性验收。

| 核对项 | 官方固定版本的有效行为 | 本地差异 / 本轮处理 |
| --- | --- | --- |
| 配置与调用链 | `use_pc_color=false`、`pointnet_type=pointnet`；SATPolicy 构造 Obs_Tokenizer，实际选 FPSPointNetEncoderXYZ | 本地默认 XYZRGB 多尺度 PointNext，不改配置或 encoder |
| global 来源和返回 | `global_pn(x)` 的 MLP/max-pool 生成观测相关前缀；patch 加 center PE 后独立 shuffle，与 global 一起经 Transformer/final projection；返回处理后 tensor | 本地前缀为 learned parameter；保留该结构，纠正“等同 global_pn”注释；仅修复无前缀分支返回旧 patch 的 bug |
| 无 global 消融 | 发布代码没有本地 `prepend_global_in_attn` 开关；policy 训练和推理各有注释掉的 `pts_token[:,1:,:]` | 删除 attention 后输出首 token 不等于移除 attention 输入前缀；本地 False 是本地消融，不声称官方已运行该消融 |
| tokenizer 消费者 | Obs_Tokenizer 返回 `(pn_feat,state_feat)`；StateAttn 由学习 query 对标量 state 投影做 cross-attention；policy 训练/推理分别将两者从 `(BT,N,F)` 转为 `(B,N,TF)`，再沿 token 维拼接 | 本地先将 state MLP 特征广播到每个 point/global token，再沿特征维融合时间；保留融合。官方 patch 主动 shuffle，本地随机 FPS 也不提供物理跨帧对应保证 |
| joint/action | backbone 将 action `B,T,Da→B,Da,T`；robot/joint 两个 embedding 拼接，再与轨迹 embedding 拼接；训练同步 shuffle action/身份并逆排列输出 | 本地 EJC 是 embodiment/function/axis 三字段各自投影后求和；不改变参数布局、shuffle 或 checkpoint |

本地 [PointNextPatchTokenizer](../dexmani_policy/agents/obs_encoder/pointcloud/pointnext_tokenizer.py) 的三个分支：attention 关闭返回 patch/center；有内部前缀返回处理后 patch/center/attn_global；无内部前缀返回处理后 patch/center。外层 `include_global_token=true` 仍从修正后的 patch 聚合 global，满足 SATObsEncoder 三元接口。默认 SAT 有内部前缀，不受该返回值补丁影响。

## 历史实验事实（不由新默认反推）

| 可读取的本地产物 | saved config / 证据 | 本轮可下的结论 |
| --- | --- | --- |
| [RGB A/B float32 seed 42 config](../experiments/rgb_transport_ab/20261006_195505/runs/rgb_float32_s42/config.yaml) | DP DINO-small LoRA，缺 `lora_dtype`；`rgb_keep_uint8=false`，batch 64，seed 42，预算 100000 updates，EMA/NFE=10；观察时目录有 20k/40k/60k/80k milestone 文件名 | “float32”是 RGB transport；该保存配置仍构造 BF16 adapter。未加载这些 checkpoint 验证状态或重新评测，不能宣称训练已完成。A/B 固定 policy `3e74a99…` / simulator `c3d56fd…` 和配对 seed 42/43/44 见[原记录](rgb_transport_closed_loop_ab.md)，质量结论仍未完成 |
| [历史 ManiFlow config](../experiments/maniflow/pick_place_toy/2026-09-28_04-48-18_42/config.yaml) | pick_place_toy；batch 128，seed 42，预算 60000 updates；encoder pointnet_dense；agent fallback 10、eval NFE 4、EMA true；目录有 12k/24k/36k/48k/60k milestone 文件名 | 不能写成当前默认 100000 updates 或固定协议结果；保存配置没有 manifest。本轮未核验 checkpoint 内容、历史代码 SHA 或成功率，均为 NOT VERIFIED |

这些实验目录被 Git 忽略，链接仅在持有对应产物的环境可用。没有其他六种方法的真实论文结果证据，不填推测“实际运行”值。本轮未改动上述实验、A/B 队列源码、saved config、权重、选择记录或数据。

新 FP32 DP 结果需要另行重训或配对实验，旧 BF16 checkpoint 转 dtype 无法补回过去更新，strict resume 拒绝跨配方。SAT 无内部前缀的已训练消融需要重训；默认有前缀实验不因该补丁自动失效。T2/T3 通常只要求重新选点和重新评测，不要求重训。T6 修复配置准备，不表示 VQ 已训练。

## 最短使用顺序

从仓库根，在已有 `policy` 环境运行；先准备实际任务 Zarr、模型权重和 DQ-RISE 所需 policy-aligned codebook，并确认 simulator 能读取实际 seed 池。不要覆盖已发布协议；本仓库已提供的 pick_apple_messy 清单直接复用。为尚未发布的任务一次生成，例如准备足够 pick_bottle/open_box 池之后：

```bash
python scripts/eval/make_seed_manifest.py --config-name multitask_dit \
  --output dexmani_policy/configs/eval_protocols/pick_bottle+open_box.json \
  --partition-seed 1066 --selection 25 --tie-break 5 --test 100
python dexmani_policy/smoke_test.py --config-only dp ddp/dp
# 以下训练/闭环命令供用户以后主动运行，本次验收没有执行：
bash scripts/training/train.sh dp
```

训练保存配置后，用已有 CLI 重新选点，再把本次记录传给 test；先设置 `EXPERIMENT_NAME`、`SELECTION_RECORD_PATH`（不存在的路径、父目录已存在）：

```bash
python dexmani_policy/select_best_ckpt.py \
  --policy-name dp --task-name pick_apple_messy --exp-name "$EXPERIMENT_NAME" \
  --result-file "$SELECTION_RECORD_PATH" --no-videos \
  eval.seed_manifest=dexmani_policy/configs/eval_protocols/pick_apple_messy.json
python dexmani_policy/eval_best_ckpt.py \
  --policy-name dp --task-name pick_apple_messy --exp-name "$EXPERIMENT_NAME" \
  --selection-record "$SELECTION_RECORD_PATH" --no-videos
```

显式协议决定完整角色 seed，`initial_episodes` / `batch_size` 只作为请求记录，不截断 manifest；`max_episodes` 仍是 selection 加预留 tie-break 的硬上限，不足则运行前拒绝。快照另存 `effective_episode_counts`。final eval 完整运行 test，数量与请求不同时提示，并记录 `effective_episodes`。Python eval/sweep 的 `dotlist_overrides` 可传显式协议声明；普通 cfg 中的路径仅是默认来源。shell pipeline 保持 `SEED_MANIFEST` 只传 selector、后两步传 `--selection-record`；demo 无 dot-list 参数。解读结果时区分流程成功、策略成功率、训练 seed 波动及协议是否 legacy。

## 实施与实际验证记录

解释器为 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`。以下为实际运行过的低成本命令；日志写入 `/tmp`，未启动完整训练、DDP、闭环评测或机器人动作。

```bash
# 单任务真实协议发布（已完成，不应再次对相同输出执行）
PYTHONPATH=.:/home/zhanghaoyang/Desktop/dexmani_sim OMP_NUM_THREADS=1 \
/home/zhanghaoyang/miniconda3/envs/policy/bin/python scripts/eval/make_seed_manifest.py \
  --config-name dp --output dexmani_policy/configs/eval_protocols/pick_apple_messy.json

# 六项改动及相邻既有回归
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-mpl \
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m pytest -q \
  tests/test_paper_readiness.py tests/test_paper_protocol.py \
  tests/test_infra_resume.py tests/test_infra_evaluation.py \
  tests/test_review_remediation.py tests/test_policy_vq_alignment.py tests/test_infra_training.py

# 上述矩阵中 Manager IPC 被沙箱拒绝的一项，随后在沙箱外单独重跑
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-mpl \
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m pytest -q \
  tests/test_infra_training.py::TrainingInfraTests::test_spawn_epoch

# 最终配置矩阵
PYTHONPATH=. OMP_NUM_THREADS=1 /home/zhanghaoyang/miniconda3/envs/policy/bin/python \
  dexmani_policy/smoke_test.py --config-only \
  dp dp3 dqrise maniflow sat r3d multitask_dit \
  ddp/dp ddp/dqrise ddp/maniflow ddp/sat ddp/r3d ddp/multitask_dit

git diff --check
```

| 任务 | 代码/产物状态与关键位置 | 实际检查结果 | 尚未验证 / 后续 |
| --- | --- | --- | --- |
| T1 | RGB 三个构造器/shared base、DP config、resume、build/logging/trainer 完成 | `test_paper_readiness.py` 的 LoRA/storage/resume 与小更新/EMA 测试，及 `test_infra_resume.py`：PASS。真实离线 HF DINO + PEFT、真实 DP/AdamW/EMA；raw/EMA 保存恢复、旧缺字段↔backbone、跨精度拒绝后状态未变；CLIP/SigLIP 传参；loop/foreach 数值检查 | 默认 CUDA BF16 autocast/compile：NOT VERIFIED；新 DP 质量需重训 |
| T2 | protocol、selector/eval/demo、7 基础配置、manifest 生成器完成；真实 S 已发布 | `test_paper_protocol.py` 和既有 evaluation/remediation：PASS。200→201 池变化、单/多任务映射、42/43、无加赛/有加赛、显式 null/不同清单、内嵌协议、单次/sweep/model-load 次数、shell argv、demo snapshot；真实 S 生成及身份复验 PASS | 默认多任务 M 真实池仅 100，生成器实际报错：拒绝逻辑 PASS；M 发布 NOT VERIFIED；真实闭环 NOT VERIFIED |
| T3 | selector dispatch 完整性校验及全零发布完成 | 5 milestones × 25 正常失败 episodes、5-seed 加赛、单/多任务固定 test 全零、空/缺失/重复/技术异常及旧 pointer 保留：PASS；没有使用假 optimizer/attention 绕过研究机制 | 历史全零需重新选点/test；实际成功率未评测 |
| T4 | 本文及 README 入口完成 | 13 配置解析及代码消费者对表 PASS；当前默认/本地适配/历史保存配置分开 | 未固定的其他方法上游 revision、完整权重/数据质量与历史结果：NOT VERIFIED |
| T5 | PointNext 无前缀返回修复、SAT/关节注释澄清完成 | 固定 SHA 四份官方源码已读取；`test_sat_attention_branches_and_gradients` 三分支 PASS，真实 CPU PyTorch3D + Transformer（无几何替身），返回值依赖/梯度/三步更新及 SAT 外层接口成立 | 官方完整训练与本地 GPU/闭环 NOT VERIFIED；受影响无前缀消融需重训 |
| T6 | `train_vq_hand.load_policy_config` 局部修复完成 | `test_vq_policy_config_roots` 与 `test_policy_vq_alignment.py`：PASS，基础/overlay/absolute/relative/overrides、外部 YAML、缺路径、连续调用及已有宿主 Hydra 状态保护 | 未训练新 codebook、未运行 DDP |

六项实施阶段综合回归结果：沙箱内 **107 passed，1 failed，8 subtests passed**；失败项为 `test_spawn_epoch` 的 multiprocessing Manager socket 被沙箱以 `PermissionError` 拒绝，随后在沙箱外保持原测试和真实 Manager 单独重跑，**1 passed**。因此全部 108 个测试均有实际通过证据，但不是一次沙箱内全绿；没有禁用 IPC 分支或引入替身。配置矩阵 **13/13 PASS**；Python AST 语法、本文本地链接、最终 `git diff --check` 均 PASS。预检真实 VQ 入口曾复现 overlay `RecursionError`，修复后通过。新单任务清单不是 synthetic fixture；tests 中的 synthetic pool 仅验证控制流程。

GPU 查询 `nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader` 实际失败，提示无法连接 NVIDIA driver；因此未启动 GPU smoke，也未占用在途 A/B 队列 GPU。真实多任务生成尝试使用 `--config-name multitask_dit --output /tmp/paper_multitask_manifest.json`，以 `Actual paired seed pool has 100 seeds; requested 25+5+100` 失败，没有生成文件。本轮可交付实现不代表上述受阻验收已完成。


### 后续清理与复验（2026-10-06）

清理范围为上述六项实现及相关操作文档：删除 eval/sweep 重复的协议解析、绑定后的不可达 fallback、demo 重复赋值；保留旧 checkpoint 构造、strict resume 与 legacy 记录复现路径。single/sweep 共同在加载模型前解析完整 test 并校验 held-out 身份，runner/model 仍只构建一次。

复核任务书发现 T2 的 `max_episodes` 硬上限遗漏，现已补齐：selection 加全部预留 tie-break 必须容纳于上限内，即使最终无需加赛也不能忽略预留数量。清单不被请求参数截断；selection/test 快照分别记录请求与实际数量，test 数量不一致时明确提示。未改变 manifest、历史选择记录或 A/B 队列。

清理 SAT 中误称本地求和编码、时间融合为官方 paper spec 的注释及未经核实的会议/章节标注；参数和默认架构不变。README、仿真评测机制、项目架构、shell help 与任务书状态同步至当前行为；历史报告和原审查依据保留。

本轮实际命令：

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-mpl \
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m pytest -q \
  tests/test_paper_readiness.py tests/test_paper_protocol.py \
  tests/test_infra_resume.py tests/test_infra_evaluation.py \
  tests/test_review_remediation.py tests/test_policy_vq_alignment.py

PYTHONPATH=. OMP_NUM_THREADS=1 /home/zhanghaoyang/miniconda3/envs/policy/bin/python \
  dexmani_policy/smoke_test.py --config-only \
  dp dp3 dqrise maniflow sat r3d multitask_dit \
  ddp/dp ddp/dqrise ddp/maniflow ddp/sat ddp/r3d ddp/multitask_dit

bash -n scripts/eval/select_best_ckpt.sh scripts/eval/eval_best_ckpt.sh scripts/eval/eval_pipeline.sh
bash scripts/eval/select_best_ckpt.sh --help
bash scripts/eval/eval_best_ckpt.sh --help
bash scripts/eval/eval_pipeline.sh --help
git diff --check
```

结果：**102 passed、8 subtests passed**（29.80s，5 条既有 fixture/历史数据身份警告）；**13/13 配置 PASS**；shell 语法与三个 help 入口 PASS；21 个改动/新增 Python 文件 AST、5 份更新文档的本地链接及最终 `git diff --check` PASS。新增测试覆盖上限不足时模型/rollout 零调用、恰好容纳边界、有/无 tie pool、无需加赛仍预留、请求与实际数量分别保存，以及 `episodes=1` 不截断 100-seed test 的提示。日志：`/tmp/paper_cleanup_tests.log`、`/tmp/paper_cleanup_configs.log`。此前 GPU、闭环、真实多任务清单的 NOT VERIFIED 状态不变；此次清理没有产生新的训练或成功率结果。
