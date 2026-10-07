# 论文实验 recipe 与验证边界

更新日期：2026-10-07。本文保留当前方法配方、官方源码对照与使用入口。历史实施记录见文末 Git 链接；历史检查不代表当前配置已完成生产或策略质量验收。

## 当前方法机制与本地适配

下表由有效 Hydra 配置及其实际消费者核对。除 SAT 外，本轮没有固定并复核其他方法的上游 revision；这些方法与原论文/上游的完整对应均为 **NOT VERIFIED**，不能将方法名解释为官方 recipe 的逐项复现。

| 方法与配置 | 当前保留的机制、观测与 encoder | 预测目标 / 损失 | 本地适配与来源边界 |
| --- | --- | --- | --- |
| [DP](../dexmani_policy/configs/dp.yaml) | RGB + joint state；DINOv2-small average patch feature + state MLP；条件 UNet `[256,512,1024]` | 100-step diffusion，`sample`/action MSE；DDIM 推理 | [DPAgent](../dexmani_policy/agents/core/dp.py) 使用 HF `facebook/dinov2-small`，LoRA rank 16 / alpha 32 / dropout .05 / RSLoRA；新 adapter FP32，冻结主干 BF16。上游精确 revision 未核实 |
| [DP3](../dexmani_policy/configs/dp3.yaml) | XYZRGB 1024 点 + state；全局 PointNet 128d + state MLP；同 DP 的 UNet | 100-step diffusion，`sample` MSE | [DP3Agent](../dexmani_policy/agents/core/dp3.py) → [PointNet registry](../dexmani_policy/agents/obs_encoder/pointcloud/registry.py)；scratch/full，本地点云颜色和数据适配；上游精确 revision 未核实 |
| [DQ-RISE](../dexmani_policy/configs/dqrise.yaml) | XYZRGB 1024 点 + state；`idp3` MultiStagePointNet 128d；UNet `[256,512]` | 100-step diffusion，`epsilon` MSE；内部预测 TCP 9 + 一个连续 code index，最后恢复 hand 12 | [DQRISEAgent](../dexmani_policy/agents/core/dqrise.py) 的控制输出为 21d；本地 VQ/PCA 排序 codebook（2 groups × 4），必须匹配 policy 数据窗口/归一化。scratch encoder，codebook 来自 `robot_data/sorted_hand_poses_${task_name}.npz`；上游精确 revision 未核实 |
| [ManiFlow](../dexmani_policy/configs/maniflow.yaml) | XYZRGB 1024 点 + state；dense PointNet 128d、8-frequency XYZ PE 投影、state 广播；两帧沿 token 维展开；12-layer / 768d ConsistencyDiTX | [ConsistencyFlowMatch](../dexmani_policy/agents/action_decoders/consistency_flow.py)：flow velocity MSE + EMA consistency MSE；batch 分配 .75/.25，两项均值直接相加 | [本地观测编码](../dexmani_policy/agents/core/maniflow.py) 为 scratch/full；flow 时间 beta，consistency 离散网格 10、dt uniform、target time relative；agent fallback NFE 10，实际 eval 为 4。上游精确 revision 未核实 |
| [SAT](../dexmani_policy/configs/sat.yaml) | XYZRGB 1024 点 + state；多尺度 PointNext、96 patches、256d token、4-layer patch attention、learned global prefix；8-layer / 768d SAT | [RectifiedFlow](../dexmani_policy/agents/action_decoders/rectified_flow.py) velocity MSE；训练时间连续 beta，不使用离散训练网格；Euler 推理；训练 action-token shuffle | [本地 SAT](../dexmani_policy/agents/core/sat.py) 为 scratch/full；按 patch 槽位沿特征维融合两帧，state 广播进每个 token，三字段 EJC 投影求和。官方固定版本对照见下节；本轮只修无内部前缀分支 |
| [R3D](../dexmani_policy/configs/r3d.yaml) | XYZRGB + state；Uni3D `eva02_tiny_patch14_224`，512 groups × 32 points、PointSAM feature；256d / 4-layer OneWay backbone | 100-step diffusion，`sample` MSE | [R3DAgent](../dexmani_policy/agents/core/r3d.py) 默认 `use_aux_ee=false`，19d joint 控制；可选辅助 EEF 9d 是额外预测，控制仍取 joint 19。预训练读取 `data/pretrained/uni3d/model.safetensors`，代码下载源 `eddie-cui/r3d-weights`；选择性加载 `pc_encoder.*`，活跃路径 full tuning、无用 timm 参数冻结；上游/权重 revision 未核实 |
| [Multi-task DiT](../dexmani_policy/configs/multitask_dit.yaml) | RGB + state + task text；scratch ResNet18 + GroupNorm/full；首次初始化用冻结 CLIP text `openai/clip-vit-base-patch16` 生成固定任务 embedding 表，训练仅保留表与 trainable text projection；8-layer / 512d DiT | 默认 diffusion、100-step、`sample` MSE；当前配置不暴露未使用的 flow 参数 | [MultiTaskAgent](../dexmani_policy/agents/core/multi_task.py)；任务为 pick_bottle/open_box，文本为 “pick up the bottle” / “open the box”；不使用 RGB 通用默认 DINO 预训练。该组合为本地基线，上游精确对应未核实 |

动作和损失消费者分别为 [BaseAgent](../dexmani_policy/agents/core/base.py)、[Diffusion](../dexmani_policy/agents/action_decoders/diffusion.py) 及上表各 agent。默认 `action` 为 19 = arm joint 7 + hand 12；DQ-RISE 的 `action_ee` 为 21 = xyz 3 + rot6d 6 + hand 12。所有当前默认 `H / observation horizon / action chunk = 16 / 2 / 8`，执行片段从 `n_obs_steps - 1` 开始。字段维度描述不替代数据中的实际排列校验。

## 当前训练与评测配方

所有行都使用 100000 **optimizer updates**，候选为 20000/40000/60000/80000/100000；accumulation=1，`drop_last=true`，seed 默认 42，gradient clip=1，标准 cosine scheduler，不使用 `lr_min_ratio`；该参数只用于 `cosine_min_lr`。EMA 每个 optimizer update 更新一次：after=0、inv_gamma=1、power=.75、max=.9999，当前 `foreach` 缺省为 false。配置均启用 BF16 autocast 与 compile；这不表示参数以 BF16 存储，也不表示本轮验证了 CUDA/compile。

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

所有方法的 state Gaussian noise std=.0002、prob=1。点云方法 XYZ noise std=.002，color brightness=.125/contrast=.5/saturation=.5/hue=0，prob=1；DP3/DQ-RISE/SAT 另有 max dropout ratio=.8，ManiFlow/R3D 无此项。FPS 训练保留 random start，noise scale=0，eval 强制 deterministic。DP3/DQ-RISE 的全局 PointNet 配方关闭同一已选点集的 output shuffle，改变新 run 的 RNG 消耗；其他点云配方保留各自 shuffle 设置。SAT 的 FPS 输出随机顺序与官方 attention 前独立 patch shuffle 是不同机制。

DP/Multi-task：resize 240×240、训练随机 crop 224×224、eval center crop；颜色增强 brightness=.3/contrast=.2/saturation=.2/hue=.05/grayscale=.1，prob=.25，noise/blur=0。新 DP `rgb_keep_uint8=true`，Multi-task child 未配置该字段，按 BaseDataset 缺省 false。像素 transport 与 LoRA 参数存储精度是独立变量。

### 固定协议身份与使用边界

**S** 为真实单任务清单 [pick_apple_messy.json](../dexmani_policy/configs/eval_protocols/pick_apple_messy.json)：实际池 200，selection/tie-break/test=25/5/100，partition seed=1066。来源为 simulator `DATA_DIR/eval_seeds/pick_apple_messy.txt`，读取时 simulator commit=`c3d56fd58f402fc6f67a635b9452e10eee935afe`。

- manifest canonical SHA256：`44362705d1bd5575dc766b4b98e7953e8b6ec770a357d0000b42df82c6ed4a3a`。
- 完整 paired runner pool SHA256：`5ac859dd04b29670ce5ad3bd309dc4c148e12040be5ea0469484247aa9c32f91`。
- hash 沿用 `json.dumps(..., sort_keys=True)` 的既有算法；不是文件原始字节 hash。上述 pool hash 是发布时的来源快照，当前执行只依赖清单实际 task→seeds，不再校验未使用成员或全池顺序。
- 已只读比较 A/B 原 manifest：三个角色的 seed 和顺序均相同；新清单有自己的 metadata/hash，没有覆盖 A/B 清单或改变其协议。

**M** 为默认 task-set `pick_bottle+open_box` 对应路径。本机两个任务均无实际 seed 文件，真实 runner 各 fallback 100，生成器实际拒绝 25+5+100；**真实清单待发布 / NOT VERIFIED**，没有提交占位 JSON。必须准备足够实际 paired seeds 后一次发布；若事先决定更小 test，应显式生成另一份协议、记录新计数，不能自动截断。若以后比较单/多任务同一任务，需核对各角色的 physical seeds 完全一致；当前 runner 直接执行这些实际列表，各任务预算仍须相等。

新 selection 必须指定有效 manifest，完整预留 tie seeds；选点后 pinned held-out eval 使用不可变记录中的内嵌内容，原路径删除或修改不影响默认 handoff。显式 `eval.seed_manifest=...` 必须与记录 canonical hash 相同，显式 null 也不能解除合同；初次选点和后续 held-out 评测检查所用 seed 可用、角色互斥及完成列表。demo 的固定交接只锁定 checkpoint/EMA/NFE，仍验证选择记录；根据各任务池选择可视化 seeds，显式 `--seeds` 表示物理数字，不读取未使用的 test 清单，不计入 held-out 指标。历史无 manifest 的记录只沿 legacy 路径复现，不能贴上新协议冒充重新选点。

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

下表为 2026-10-06 的观察快照；2026-10-07 的 saved config / 产物复核见[历史实验计划](https://github.com/haoyangzhanglab/dexmani_policy/blob/a99ed34beb5ba7a7bc75ec8177f0caef6043031a/docs/paper_recipes.md#后续实验计划本轮未启动)。在途进程状态只代表各次查询时刻。

| 可读取的本地产物 | saved config / 证据 | 本轮可下的结论 |
| --- | --- | --- |
| [RGB A/B float32 seed 42 config](../experiments/rgb_transport_ab/20261006_195505/runs/rgb_float32_s42/config.yaml) | DP DINO-small LoRA，缺 `lora_dtype`；`rgb_keep_uint8=false`，batch 64，seed 42，预算 100000 updates，EMA/NFE=10；观察时目录有 20k/40k/60k/80k milestone 文件名 | “float32”是 RGB transport；该保存配置仍构造 BF16 adapter。未加载这些 checkpoint 验证状态或重新评测，不能宣称训练已完成。A/B 固定 policy `3e74a99…` / simulator `c3d56fd…` 和配对 seed 42/43/44 见[原记录](rgb_transport_closed_loop_ab.md)，质量结论仍未完成 |
| [历史 ManiFlow config](../experiments/maniflow/pick_place_toy/2026-09-28_04-48-18_42/config.yaml) | pick_place_toy；batch 128，seed 42，预算 60000 updates；encoder pointnet_dense；agent fallback 10、eval NFE 4、EMA true；目录有 12k/24k/36k/48k/60k milestone 文件名 | 不能写成当前默认 100000 updates 或固定协议结果；保存配置没有 manifest。本轮未核验 checkpoint 内容、历史代码 SHA 或成功率，均为 NOT VERIFIED |

这些实验目录被 Git 忽略，链接仅在持有对应产物的环境可用。没有其他六种方法的真实论文结果证据，不填推测“实际运行”值。本轮未改动上述实验、A/B 队列源码、saved config、权重、选择记录或数据。

新 FP32 DP 结果需要独立新训练或配对实验，旧 BF16 checkpoint 转 dtype 无法补回过去更新，strict resume 拒绝跨配方。SAT 只需重训修复前训练且 `use_patch_self_attn=true`、`prepend_global_in_attn=false` 的受影响消融；默认有前缀实验不因该补丁自动失效。固定协议与合法全零选择的修正通常只要求重新选点和重新评测，不要求重训。VQ Hydra 入口修复配置准备，是否重训 VQ 取决于实际数据窗口、归一化及 codebook 配方是否匹配。

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

显式协议决定完整角色 seed，已删除无效的 `initial_episodes` / `batch_size` 请求参数，旧 CLI 参数会报错；`max_episodes` 仍是 selection 加预留 tie-break 的硬上限，不足则运行前拒绝。快照另存 `effective_episode_counts`。final eval 完整运行 test，数量与请求不同时提示，并记录 `effective_episodes`。Python eval/sweep 的 `dotlist_overrides` 可传显式协议声明；普通 cfg 中的路径仅是默认来源。shell pipeline 保持 `SEED_MANIFEST` 只传 selector、final eval 传 `--selection-record`；demo 独立执行，无 dot-list 参数。解读结果时区分流程成功、策略成功率、训练 seed 波动及协议是否 legacy。

## 剩余验收与历史记录

以下是既有报告尚未闭合的验证边界，清理文档时未重新运行验收或查询实验进程。历史 CPU、单卡 smoke 和 config-only 结果不能替代生产配置验证。

- **CUDA Trainer BF16/compile：NOT VERIFIED。** 新 FP32 LoRA 配方需沿真实 Trainer 路径验证参数、AdamW 动量、EMA 精度与实际更新，以及 raw/EMA 保存恢复和跨精度拒绝。普通 smoke 不覆盖 Trainer 的 autocast/compile；有限步或小 batch 检查不能证明默认 batch、收敛或成功率。
- **多 GPU：NOT VERIFIED。** 生产 NCCL/DDP、设备 RNG/映射变化、同 world_size 恢复及单 rank 失败传播需在目标环境补验；现有入口为 `tests/test_infra_cuda.py`，tiny DDP 用例不替代真实策略验收。
- **完整模型与部署：NOT VERIFIED。** RGB 预训练矩阵（含 CLIP/SigLIP）、冷缓存恢复、完整 R3M 训练与 R3D CUDA、RTC 首次/稳态性能仍缺完整证据。CPU 几何检查不代表目标 CUDA 扩展通过；真机运动仍需用户明确授权。
- **多任务协议与数据：NOT VERIFIED。** 历史检查中 pick_bottle/open_box 只有各 100 个 fallback seeds，不足 25/5/100，且缺对应训练 Zarr。须准备各任务经过验证的实际 seed 池与数据后发布 M；不能把 fallback 整数当作已验证池或自动缩小 test。
- **策略质量与完整仿真流程：NOT VERIFIED。** selection → held-out eval → demo 的真实运行、成功率、视频及 RTC 闭环不能由 fixture 证明。RGB transport 配对实验的协议和历史状态见 [A/B 记录](rgb_transport_closed_loop_ab.md)；新 FP32 LoRA 质量需独立实验，不能复用旧 BF16 配方的结论。历史 ManiFlow 的 pick_place_toy 结果进入新协议前，仍需补齐任务 seed 池并核实数据、代码和 checkpoint 身份。
- **VQ / DQ-RISE：NOT VERIFIED。** 历史检查未找到可供复用审查的 codebook；真实 VQ 产物仍需与目标 policy 的窗口、split、source rows、action layout 和 normalizer 核对。Hydra 入口修复本身不要求全部 codebook 重训。

历史报告中的性能优化候选并非当前缺陷清单；相关实现已有后续变化，是否继续优化应重新读取调用链并以 profiling 为依据。

完整过程、当时环境与验证范围可从 Git 历史追溯：

- [基础设施修复报告](https://github.com/haoyangzhanglab/dexmani_policy/blob/a99ed34beb5ba7a7bc75ec8177f0caef6043031a/docs/infra_fix_report.md)
- [增量整改报告](https://github.com/haoyangzhanglab/dexmani_policy/blob/a99ed34beb5ba7a7bc75ec8177f0caef6043031a/docs/review_remediation_report.md)
- [论文实验实施与验收记录](https://github.com/haoyangzhanglab/dexmani_policy/blob/a99ed34beb5ba7a7bc75ec8177f0caef6043031a/docs/paper_recipes.md#实施与实际验证记录)

[原始验证证据](review_remediation_evidence/)与 [RGB transport 报告](rgb_transport_report.md)继续保留；证据仅支持各自记录的代码版本和运行范围。
