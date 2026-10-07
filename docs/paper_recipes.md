# 论文实验 recipe 与六项修正记录

更新日期：2026-10-07。当前机制说明已随简化整改更新；带日期的实施记录对应 `4c97cfbdb4ea9761a2fe03907d0a28c10c4f1a42` 及随后验收版本。历史任务来源：[六项任务书](../CODEX_PAPER_READINESS_TASKS.md)。这是实现与低成本验证记录，不是策略质量验收或论文结果表。

历史阶段状态见[验收收尾](#验收收尾2026-10-07)：测试隔离、单任务真实清单与 SAT 原生 CPU 检查通过；多任务真实清单待发布，CUDA Trainer BF16/compile 尚未验证。下文带日期的实施记录保留当时结果，不能视为当前全部验收通过。

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

下表为 2026-10-06 的观察快照；2026-10-07 的 saved config / 产物复核见[后续实验计划](#后续实验计划本轮未启动)。在途进程状态只代表各次查询时刻。

| 可读取的本地产物 | saved config / 证据 | 本轮可下的结论 |
| --- | --- | --- |
| [RGB A/B float32 seed 42 config](../experiments/rgb_transport_ab/20261006_195505/runs/rgb_float32_s42/config.yaml) | DP DINO-small LoRA，缺 `lora_dtype`；`rgb_keep_uint8=false`，batch 64，seed 42，预算 100000 updates，EMA/NFE=10；观察时目录有 20k/40k/60k/80k milestone 文件名 | “float32”是 RGB transport；该保存配置仍构造 BF16 adapter。未加载这些 checkpoint 验证状态或重新评测，不能宣称训练已完成。A/B 固定 policy `3e74a99…` / simulator `c3d56fd…` 和配对 seed 42/43/44 见[原记录](rgb_transport_closed_loop_ab.md)，质量结论仍未完成 |
| [历史 ManiFlow config](../experiments/maniflow/pick_place_toy/2026-09-28_04-48-18_42/config.yaml) | pick_place_toy；batch 128，seed 42，预算 60000 updates；encoder pointnet_dense；agent fallback 10、eval NFE 4、EMA true；目录有 12k/24k/36k/48k/60k milestone 文件名 | 不能写成当前默认 100000 updates 或固定协议结果；保存配置没有 manifest。本轮未核验 checkpoint 内容、历史代码 SHA 或成功率，均为 NOT VERIFIED |

这些实验目录被 Git 忽略，链接仅在持有对应产物的环境可用。没有其他六种方法的真实论文结果证据，不填推测“实际运行”值。本轮未改动上述实验、A/B 队列源码、saved config、权重、选择记录或数据。

新 FP32 DP 结果需要独立新训练或配对实验，旧 BF16 checkpoint 转 dtype 无法补回过去更新，strict resume 拒绝跨配方。SAT 只需重训修复前训练且 `use_patch_self_attn=true`、`prepend_global_in_attn=false` 的受影响消融；默认有前缀实验不因该补丁自动失效。T2/T3 通常只要求重新选点和重新评测，不要求重训。T6 修复配置准备，是否重训 VQ 取决于实际数据窗口、归一化及 codebook 配方是否匹配。

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

### 验收收尾（2026-10-07）

**起点与范围。** 开始时 `git rev-parse HEAD` 为审查基线 `4c97cfbdb4ea9761a2fe03907d0a28c10c4f1a42`，`git status --short --branch` 为 `## main...origin/main`，无本地修改；没有基线后的差异需要迁移。已读 AGENTS、任务书及本文历史记录。本轮只修改构造器测试隔离和本文，不改变生产模型、默认配方、已发布清单或历史产物，不 commit/push。

**实际环境。** Linux `6.17.0-40-generic x86_64`；以下命令使用 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`（Python 3.10.20），Torch `2.4.1+cu124`、torchvision `0.19.1+cu124`、PyTorch3D `0.7.8`、transformers `4.48.0`、PEFT `0.20.0`、Hydra `1.3.5`、pytest `9.1.1`。PyTorch3D 的 `_C.cpython-310-x86_64-linux-gnu.so` 可导入。沙箱中 `nvidia-smi` 无法连接 driver，`torch.cuda.is_available()` 为 False；这不是宿主没有 GPU 的证据。约 00:32 CST 的沙箱外只读查询确认唯一 GPU 为 RTX 4090（24564 MiB，使用 11737 MiB，利用率 93%），训练 PID `1445123` 使用 10422 MiB，属于在途 `rgb_uint8_s42`；队列 PID `1152964`。因此没有空闲验收 GPU，本轮没有启动 CUDA 任务，也未干预该队列。

| 验收项 | 本轮结果与证明范围 |
| --- | --- |
| 代码修复 | **PASS**：所有 `targets`/`backbones` 所在模块先导入，再进入原 `ExitStack`；退出后逐个检查全部 10 个符号的对象身份。新子进程回归先断言消费模块未导入，再执行构造器矩阵，确认 `dqrise.DP3ObsEncoder` / `multi_task.DPObsEncoder` 是真实 provider 对象而非 Mock。生产模型和 VQ 集成覆盖不变 |
| 测试污染复现与修复 | 修改前，构造器→两个 VQ 用例为 **1 passed / 2 failed**，真实 DQ-RISE 收到残留 `DummyEncoder`，报 `expected 2, got 1`；属于测试隔离缺陷。修改后，两个独立新进程分别运行正序和反序，均 **3 passed**（19.84s / 20.55s）。历史整组通过可能受提前导入影响，不能替代此次顺序验证 |
| 单任务真实协议 | **PASS**：复用已有 pick_apple_messy 清单；真实池 200，25/5/100、partition seed 1066、映射、互斥、顺序及完整池 hash 均核验；与 A/B 三个角色的 physical seeds 和顺序相同。无重抽、无覆盖 |
| 多任务真实协议 | **真实清单待发布 / NOT VERIFIED**；真实轻量 runner 只有 100 个配对参考 seeds。生成器不足池拒绝 **PASS**，没有生成占位文件，详见下文 |
| 原生几何 | **PASS**：重新执行 SAT 三分支，**3 passed, 7 deselected**（6.48s）。真实 CPU PyTorch3D + Torch Transformer；包含返回值依赖、位置/attention 梯度、三步 AdamW 更新、中心顺序及外层 global-token 接口。没有替换几何算子；本轮无 reference 几何结果混入此结论 |
| FP32 LoRA CPU 检查 | **PASS**：定向回归包含真实离线 tiny HF DINO、PEFT、DP、AdamW 和 EMA，新旧精度、raw/EMA 保存恢复、同配方恢复一致性、跨精度拒绝且状态未变、小更新和 loop/foreach EMA 检查。合成输入 batch 2、28×28、tiny DINO/缩小 UNet；不能据此证明默认 CUDA batch、compile 或策略质量 |
| CUDA BF16 / compile | **NOT VERIFIED（资源被在途实验占用）**。没有用普通 `compute_loss` smoke 代替 Trainer 路径，也未关闭 BF16/compile 来获得通过结果 |
| 必要回归与配置 | 四份相关测试 **46 passed, 4 warnings**（44.27s，含新冷启动回归）；警告为历史数据身份及 tiny 本地 DINO 路径提示。13 配置 config-only **PASS**；最终 `git diff --check` **PASS** |

本轮实际测试命令（均从仓库根执行；两个顺序是两次独立 Python 进程）：

```bash
export PYTHONPATH=.:tests
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/paper-mpl
PY=/home/zhanghaoyang/miniconda3/envs/policy/bin/python

# 修改前也执行过第一条，结果见表中反例。
"$PY" -m pytest -q \
  tests/test_infra_resume.py::ResumeInfraTests::test_all_constructors_and_config_matrix \
  tests/test_policy_vq_alignment.py::test_vq_checkpoint_export_and_actual_policy_load
"$PY" -m pytest -q \
  tests/test_policy_vq_alignment.py::test_vq_checkpoint_export_and_actual_policy_load \
  tests/test_infra_resume.py::ResumeInfraTests::test_all_constructors_and_config_matrix
"$PY" -m pytest -q tests/test_paper_readiness.py -k sat_attention_branches_and_gradients
"$PY" -m pytest -q tests/test_infra_resume.py tests/test_policy_vq_alignment.py \
  tests/test_paper_readiness.py tests/test_paper_protocol.py
"$PY" dexmani_policy/smoke_test.py --config-only \
  dp dp3 dqrise maniflow sat r3d multitask_dit \
  ddp/dp ddp/dqrise ddp/maniflow ddp/sat ddp/r3d ddp/multitask_dit
git diff --check
```

对应本机日志为 `/tmp/paper_acceptance_{before,forward,reverse,sat,regression,configs}.log`，临时日志不是随仓库分发的论文产物。历史 `/tmp/paper_t5.log` 的三分支通过仍可读取，但本轮以当前代码重新执行的原生结果为准。

#### 真实 seed 池复核

使用 `PYTHONPATH=.:/home/zhanghaoyang/Desktop/dexmani_sim` 组合 `multitask_dit` / `dp`，仅 `hydra.utils.instantiate(cfg.env_runner)`，未构造 model/dataset/env，未 reset/step。实际 simulator package 为 `/home/zhanghaoyang/Desktop/dexmani_sim/dexmani_sim`，revision `c3d56fd58f402fc6f67a635b9452e10eee935afe`，Git 工作树干净；`DATA_DIR` 为 `/home/zhanghaoyang/Desktop/dexmani_sim/mimic_data`。

- `multitask_dit` 的任务顺序仍为 `pick_bottle, open_box`；joint control、instance/table random=true、texture random=false。两者 `eval_seeds=None`，`DATA_DIR/eval_seeds/pick_bottle.txt` 和 `open_box.txt` 均不存在，各回退到 `range(100)`。按位置配对只有 **min(100,100)=100**，不能相加为 200。该完整配对池的 `runner_pool_sha256` 为 `9ae4763a777f5f12b47b8ab4a868099bc10ef376e46a3067fba0c061e177c5bf`。
- 单任务实际文件 `DATA_DIR/eval_seeds/pick_apple_messy.txt` 有 200 个 seeds，文件字节 SHA256 为 `c1dcd80c3822cbcfec610254db7fdd9898295b6725b95db8f31931f65073e2d2`。已有清单 canonical hash 仍为 `44362705d1bd5575dc766b4b98e7953e8b6ec770a357d0000b42df82c6ed4a3a`，完整 pool hash 仍为 `5ac859dd04b29670ce5ad3bd309dc4c148e12040be5ea0469484247aa9c32f91`。通过 `load_seed_manifest` 校验，并将有序池用 `random.Random(1066).shuffle` 后的前 130 项与三个角色顺序逐项比较。
- 现有 seed 文件只有 `multi_grasp.txt`、`pick_apple_messy.txt`、`place_milk_box.txt`、`pour.txt`。不能挪用其他任务文件来补足 M。需要补齐上述两个任务各至少 130 个可追溯、经过相应任务验证的唯一 physical seeds，并固定文件顺序和生成/验证 revision，让正式 runner 也从相同 `DATA_DIR` 读取。当前相对 130 的数量缺口为每任务 30，但现有 fallback 也不是已验证池，因此不能只追加 30 个整数。更小 test 必须另行预先确定研究协议。

实际拒绝命令如下，退出码 1，错误为 `Actual paired seed pool has 100 seeds; requested 25+5+100`；输出文件不存在。日志 `/tmp/paper_acceptance_multitask.log`，完整只读池核验脚本/日志为 `/tmp/paper_acceptance_pools.py` / `.log`。

```bash
PYTHONPATH=.:/home/zhanghaoyang/Desktop/dexmani_sim OMP_NUM_THREADS=1 \
/home/zhanghaoyang/miniconda3/envs/policy/bin/python scripts/eval/make_seed_manifest.py \
  --config-name multitask_dit --output /tmp/paper_acceptance_multitask_manifest.json \
  --partition-seed 1066 --selection 25 --tie-break 5 --test 100
```

补齐实际池后，使用上文“最短使用顺序”的正式输出路径一次发布；若届时已有清单，先验证映射/顺序/计数/完整 hash 后复用，不覆盖重抽。当前 `pick_bottle+open_box.json` 仍不存在。

#### SAT 官方复核与 CUDA 边界

本轮重新读取固定版本四份官方源码：[配置](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/config/sat.yaml)、[tokenizer](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/vision/obs_tokenizer.py)、[policy](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/policy/sat.py)、[diffusion backbone](https://github.com/XiaohanLei/SAT/blob/cd7c0a8877d6090a9a85ebee0ceca961830b3654/sim/sat/model/diffusion/sat.py)。有效调用仍为 XYZ FPSPointNet、输入相关 `global_pn` 前缀、patch PE/shuffle/Transformer、单独 StateAttn；policy 分别合并 point/state 时间后拼接 token；backbone 拼接 robot/joint embedding 并同步 shuffle/inverse。上文适配边界成立；本地 learned prefix、state 广播、三字段 EJC 与现有时间融合保持不变。没有迁移官方模块或声称官方训练栈已运行。

已有 [RGB CUDA smoke 日志](review_remediation_evidence/rgb_cuda_smoke.log) 来自 `3e74a99`，展示普通 smoke 的一次 optimizer/EMA update；在途 A/B 的 saved config 缺 `lora_dtype`。两者都不能作为新 FP32 LoRA 在 Trainer BF16/compile 下通过的证据。缓存 `facebook/dinov2-small` 的 `ed25f3a31f01632728cabb09d1542f84ab7b0056` snapshot 下 config/weights 存在；本轮没有加载默认权重做 CUDA 检查。

已核对真实路径：[train.py](../dexmani_policy/train.py) 构造 model/EMA、optimizer/scheduler 和 Trainer；[Trainer.train](../dexmani_policy/training/trainer.py) 在训练前调用 [compile_models](../dexmani_policy/training/build_utils.py)，编译 model/EMA 的 action backbone；`train_step` 使用 BF16 autocast，`apply_gradient_step` 执行有限梯度检查、真实 AdamW、scheduler 和 EMA。普通 `smoke_test.py` 直接调用 `compute_loss`，不经过这些 autocast/compile 开关。

下一次有空闲 GPU 时，使用此真实 Trainer 路径做 **3–5 optimizer updates** 后停止的检查：保留 `use_bfloat16=true/use_compile=true`，允许 batch 2 或合成输入但必须保存完整检查配置；断言冻结 DINO BF16 且逐参数不变、LoRA/AdamW 两个动量/EMA shadow FP32、loss 和梯度有限、adapter 确有更新，再验证 raw/EMA 保存恢复和跨精度 strict resume 拒绝。该 CUDA 检查仍待执行；不能从有限步推断默认 batch 64 能运行、收敛或成功率。

#### 后续实验计划（本轮未启动）

只读扫描本机实际 saved config（排除冻结源码副本）发现三份训练配置：A/B 的 `rgb_float32_s42`、`rgb_uint8_s42` 和历史 ManiFlow。未发现 SAT 训练配置、VQ checkpoint 或 `robot_data/*.npz` codebook。

| 项目 | 产物事实与下一步 |
| --- | --- |
| 新 FP32 LoRA DP | `robot_data/pick_apple_messy.zarr` 可读，125 episodes，RGB uint8、action/joint_state 19d；缓存 DINO 与 S 清单存在。A/B 两份配置均缺 `lora_dtype`，仍属 BF16 adapter；float32 臂状态记录 train returncode=0，并有 20k…100k 五个 milestone，uint8 臂仍训练中。这些 checkpoint 本轮未加载，不宣称质量通过。新配方必须新训练；旧 BF16 checkpoint 转 dtype 不能补回历史更新，不能当同配方 resume |
| SAT | 未找到可据以判定需要重训的本机 saved config。只重训**修复前训练且 `use_patch_self_attn=true`、`prepend_global_in_attn=false`** 的实验；需要实验目录、训练代码 SHA、saved config、数据身份和 checkpoint。默认有前缀实验不因此失效；不能从名称判断开关 |
| 固定协议 / 合法全零 | 通常复用已有 checkpoint 重新 selection，再用该次 immutable selection record 跑完整 test；禁止给旧记录补 manifest。A/B 当前由队列管理，本轮不重选或改写其 pointer。历史 ManiFlow `pick_place_toy/2026-09-28_04-48-18_42` 为 60k updates、五个 12k…60k milestone、EMA/NFE=4、无 manifest/data revision/source_manifest；其任务也无 seed 文件。需要补齐该任务真实 ≥130 池、核实数据/历史代码与 checkpoint 身份后才能发布新协议并闭环，不能使用 pick_apple_messy 清单 |
| VQ / DQ-RISE | Hydra 入口修复本身不要求全部 codebook 重训。本机没有可供复用审查的 codebook；其他环境的产物需提供 VQ saved recipe、checkpoint/export metadata、目标 policy saved config、有效 source rows/window、split seed、action layout 与 normalizer。匹配则复用，不匹配才新训。当前目标数据具有 `action_ee` 21d，但尚未将实际 policy 窗口与 VQ 元数据作匹配验收 |
| Multi-task DiT | 除 M 未发布，还缺 `robot_data/pick_bottle.zarr` 和 `robot_data/open_box.zarr`；不能据配置组合通过宣称可训练 |

新 DP 的具体单卡方案如下，**仅供资源释放、CUDA 短检查完成及数据身份固定后执行**；本轮未执行。保留当前默认 batch 64、100000 updates、候选 20k/40k/60k/80k/100k、EMA/NFE=10 和 S=25/5/100。为多训练 seed 结果可分别执行 seed 42/43/44；不同 seed 的 80-episode 子集需要单独记录 source rows，不能假定三者相同。若研究问题是“LoRA 精度的因果影响”，每个 seed 还需要独立 `backbone` 对照臂，固定同一数据、RGB transport、初始化与其他超参，不能拿正在进行的 RGB transport A/B 混作对照。

```bash
# 未来执行：输出必须是未使用的新目录；不要传 resume_from。
PY=/home/zhanghaoyang/miniconda3/envs/policy/bin/python
export PYTHONPATH=.:/home/zhanghaoyang/Desktop/dexmani_sim
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PY" dexmani_policy/train.py --config-name=dp \
  training.seed=42 agent.rgb_backbone_config.lora_dtype=float32 \
  hydra.run.dir=experiments/dp/pick_apple_messy/paper_fp32_lora_s42

# 待上述训练完成；result-file 是新 handoff 文件，不能复用旧选择记录。
"$PY" dexmani_policy/select_best_ckpt.py \
  --policy-name dp --task-name pick_apple_messy --exp-name paper_fp32_lora_s42 \
  --ema --inference-steps 10 --no-videos \
  --result-file experiments/dp/pick_apple_messy/paper_fp32_lora_s42/paper_selection_handoff.json \
  eval.seed_manifest=dexmani_policy/configs/eval_protocols/pick_apple_messy.json
"$PY" dexmani_policy/eval_best_ckpt.py \
  --policy-name dp --task-name pick_apple_messy --exp-name paper_fp32_lora_s42 \
  --selection-record experiments/dp/pick_apple_messy/paper_fp32_lora_s42/paper_selection_handoff.json \
  --no-videos
```

selector 详情/不可变记录写入该实验的 `eval_ckpt_selector/<run-id>/`，test 写入 `eval_dexsim/<unique-run>/`；handoff 固定该次选择。对 seed 43/44 同时替换 seed 和目录后再执行。配对精度对照只将 `lora_dtype=backbone` 并使用另一个新目录 `paper_backbone_lora_s42`（及对应 43/44）；不得从新/旧精度 checkpoint 跨配方恢复。正式运行前还需固定当前数据的可追溯身份；Zarr attrs 没有 revision，不能仅凭目录名保证未来数据相同。

本轮未启动完整训练、DDP、长时间闭环、采集或真机动作；未修改 A/B worktree、队列、config、manifest、checkpoint 或结果。真实多任务清单和 CUDA Trainer 验收仍是具体待办，不阻塞上述已完成的代码修复、单任务协议和原生 CPU 验收。

### 验收后的相关清理（2026-10-07）

删除构造器测试未使用的 `EMAModel` 导入，将 patch 目标解析和原对象记录合并为一次遍历，保留全部预导入、恢复断言及冷启动回归。更新 README 的 smoke 证明范围、任务书状态、本文基线与历史快照日期；评测入口注释/help 改为说明完整 manifest test 和不可变 handoff，移除容易误读为截断固定 test 的示例。生产执行逻辑、旧 checkpoint 精度兼容及 legacy 评测路径保留。

清理后实际执行：

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-mpl \
/home/zhanghaoyang/miniconda3/envs/policy/bin/python -m pytest -q \
  tests/test_infra_resume.py::ResumeInfraTests::test_all_constructors_and_config_matrix \
  tests/test_policy_vq_alignment.py::test_vq_checkpoint_export_and_actual_policy_load \
  tests/test_infra_resume.py::ResumeInfraTests::test_constructor_patches_restore_after_cold_import
bash -n scripts/eval/eval_best_ckpt.sh
bash scripts/eval/eval_best_ckpt.sh --help
git diff --check
```

结果 **4 passed**（35.93s；`/tmp/paper_followup_cleanup_tests.log`）；shell 语法/help、修改的 Python AST、文档本地文件链接与 diff 检查均通过。未增加新的训练或 CUDA 验收结论。
