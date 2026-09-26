# R3D / Point-Cloud Augmentation Fix — Codex Task

## 1. 任务目标

本任务面向 PhD thesis experimental codebase。目标不是做生产级防御，而是保证：

1. R3D baseline 的有效模型结构与官方 released implementation 一致；
2. thesis 中不同 point-cloud policies 在清晰、可解释的 R3D-derived augmentation protocol 下比较；
3. 保留各 policy 的 architecture-native sampling、tokenization 和 decoder 语义；
4. 对官方论文与官方代码发生 source drift 的地方做显式决策，不把论文文字或代码 quirks 机械覆盖到实验实现。

本任务只处理经过官方 R3D 最新论文与当前公开代码共同审查后确认的事项：

- R3D-01：R3D proprio/state latent width 由错误的 64 恢复为 256；
- AUG-01：所有 thesis 主 point-cloud configs 移除非 R3D recipe 的 hue jitter；
- AUG-02：将 R3D 的完整增强思想迁移到适合的 point-cloud baselines：
  - 所有策略共享 XYZ noise、RGB brightness/contrast/saturation、proprio noise；
  - FPS randomization 仅作用于实际存在的 FPS sampling site；
  - DP3、DQ-RISE、SAT 增加官方语义的 R3D point dropout；
  - R3D 继续使用内部官方 point dropout，避免 double-drop；
  - ManiFlow 明确不启用 point dropout，作为当前 dense-token 架构的结构性例外；
- R3D-02：R3D 对 normalization 后完整 XYZRGB tensor 做官方一致的 clamp。

不要扩大到 optimizer、training budget、EMA、normalizer、recording/Zarr、生产式 assertion 或大规模测试框架。

---

## 2. Source of Truth 与版本优先级

执行前仍应读取当前仓库实际调用链。本文审查时采用以下参考：

- DexMani Policy baseline before implementation:
  haoyangzhanglab/dexmani_policy@497c49dc816dd05ae22cd147a6ff0abd56ce9d46
- Current task-document baseline:
  haoyangzhanglab/dexmani_policy@a0849031431be0575859a9ba129c185bcb08f66b
- R3D official code:
  Wushr-Lance/R3D-Policy@e637c0148376ddc4b5e667fa8f8e108cb8ff7a85
- R3D latest paper:
  arXiv:2604.15281 v2, revised 2026-09-20
- R3D project page:
  https://r3d-policy.github.io/
- SAT official:
  XiaohanLei/SAT@cd7c0a8877d6090a9a85ebee0ceca961830b3654
- ManiFlow official:
  geyan21/ManiFlow_Policy@ef2f116f1f90163ed36e657b8c5503740bb468af
- DP3 official:
  YanjieZe/3D-Diffusion-Policy@47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97

### 2.1 优先级

遇到 R3D source drift 时使用：

1. 最新 arXiv v2：用于解释论文的科学结论、augmentation ablation、作者当前方法表述；
2. 官方 released GitHub code/config：用于确定当前可执行 baseline 的真实计算图和具体实现语义；
3. project page / README：用于概览，不覆盖前两者。

原因：project page 当前仍保留部分 v1 叙述，而 GitHub main HEAD 早于 2026-09-20 的 v2 论文修订。

---

## 3. 本轮审查确认的官方事实

### 3.1 R3D state branch

官方 DP3Encoder 虽然函数形参仍写有 state_mlp_size=(64, 64)，但有效初始化路径立即覆盖为：

    state_mlp_size = (64, pointcloud_encoder_cfg["embed_dim"])

官方 R3D config 的 embed_dim=256，因此实际 proprio/state branch 为：

    state_dim -> 64 -> 256

DexMani 当前 R3D 的 state_out_dim=64 是真实结构偏差，不是 19-DoF embodiment adaptation。

### 3.2 R3D augmentation：最新 v2 的结论

R3D v2 将以下五类 augmentation 作为系统性研究对象：

1. FPS randomization；
2. point-cloud RGB color jitter；
3. point-cloud XYZ Gaussian noise；
4. robot/proprio state Gaussian noise；
5. random point dropout。

v2 的结论是：单独每一种 augmentation 都有帮助，组合全部 augmentation 整体最好，困难任务收益尤其明显。

论文 Appendix A.2 给出的关键参数：

    FPS randomization:
        randomize the FPS input/order during training

    RGB color jitter:
        brightness = [-0.125, 0.125]
        contrast   = [0.5, 1.5]
        saturation = [0.5, 1.5]
        no hue augmentation

    XYZ noise:
        Gaussian sigma = 0.002

    robot/proprio noise:
        Gaussian sigma = 0.0002

    point dropout:
        r ~ Uniform(0, 0.8) for each training sample

官方 released code进一步给出了 fixed-length point-dropout realization：

    dropout_ratio = rand() * 0.8
    drop_mask = rand(N) <= dropout_ratio
    dropped_points = first_point

即：

- 每个 sample 独立采样 r ~ U(0, 0.8)；
- 每个 point 独立 Bernoulli drop；
- 被 drop 的完整 XYZRGB point 用该 sample 的第一个 point 替换；
- 不是固定比例 drop；
- 不是从 kept points 随机重采样。

### 3.3 R3D color recipe

官方 dataset 的 apply_color_jitter 仅包含：

- additive brightness；
- contrast around flattened RGB mean；
- BT.601 saturation；
- final RGB clip to [0, 1]。

官方 R3D config 没有 hue 参数。

因此 thesis 主 point-cloud recipe 统一 hue=0.0。

PointColorJitter 组件本身可以继续保留 hue 能力供其他 recipe 使用。

### 3.4 R3D full point-cloud clamp

官方 R3D policy 在 normalizer 之后、encoder 之前对完整 normalized point-cloud tensor执行：

    torch.clamp(point_cloud, min=-1-1e-6, max=1+1e-6)

不是只 clamp XYZ。

DexMani 只需要在 R3D path 恢复该语义；不把它重新提升成所有 policy 的全局 BaseAgent 行为。

### 3.5 FPS randomization

官方 released code包括：

- random start；
- random_noise_scale=0；
- sampled-index output shuffle；
- 真正有 subset sampling 时改变 selected points。

DexMani 继续采用：

    train: randomized FPS
    eval: deterministic FPS

这是 thesis evaluation 的有意适配；不要为了逐行复制官方 code 而恢复 stochastic eval。

不要把 preprocess_point_cloud 中 N > K 的条件改成 N >= K。

当 N == K 时 FPS 只会选回全部点，最多改变顺序，不构成有效 subset augmentation。

---

## 4. 必须显式保留的 paper–code source drift

这些是审查结论，但不是本任务要求修改的代码。

### SRC-01 — proprio tokenization

R3D v2 Sec. 4.5 用三类 token 描述 decoder：

- geometric tokens；
- independent proprioception tokens；
- action tokens。

但当前 released GitHub config 明确使用：

    cat_on_token: false

当前 released code 在该配置下会：

    point token 256
    + broadcast proprio feature 256
    = 512-d observation feature per spatial token

而不是把 proprio 作为独立 sequence token。

本任务决策：继续跟随官方当前可执行 config/code，保持 cat_on_token=false 的现有 DexMani 语义。

理由：

- 这是当前 released reproduction config 的真实有效路径；
- v2 论文发布时间晚于当前 main code；
- 直接切换到 separate proprio tokens 会改变 baseline architecture；
- 当前官方 cat_on_token=true 分支未经本任务验证，不能根据论文文字静默替换已发布 baseline。

如果 thesis 最终需要声称严格实现 v2 文本中的三类 token，应作为单独实验/任务处理，不混入本轮 fidelity fix。

### SRC-02 — BatchNorm / EMA revision

R3D v2 已修正早期结论：

- 问题并不是 BatchNorm 本身无法 scaling；
- 根因是 EMA rollout model 没有同步 BN running buffers；
- 官方最新 code 的 EMAModel 已对 BatchNorm buffers/params 做同步；
- 修复后 BN 与 LN 可达到相近表现；
- R3D 主配置仍使用 LayerNorm，主要因为 pretrained Point-SAM / transformer encoder 本身采用 LN。

本任务决策：不修改 DexMani EMA。

原因：

- 当前 R3D 主 baseline 是 LayerNorm；
- BN running-stat EMA问题在默认路径不 active；
- 如果未来复现 R3D BN/LN ablation，再单独实现 BN-buffer sync。

任务书和后续论文叙述不得继续写成“R3D 证明 BN 本身导致 scaling failure”。

---

## 5. Thesis 主实验的最终 R3D-Aug protocol

对于以下主 point-cloud policies：

- R3D
- DP3
- DQ-RISE
- SAT
- ManiFlow

统一定义公共 sensor augmentation：

    XYZ noise:
        Gaussian std = 0.002
        clip = ±0.004
        prob = 1.0

    joint/proprio noise:
        Gaussian std = 0.0002
        clip = ±0.0004
        prob = 1.0

    RGB:
        brightness = 0.125
        contrast   = 0.5
        saturation = 0.5
        hue        = 0.0
        prob       = 1.0

architecture-sensitive augmentation matrix：

| Policy | R3D-style point dropout | FPS randomization |
| --- | --- | --- |
| R3D | Yes — existing encoder-level official path | Yes — internal 1024→512 |
| DP3 | Yes — dataset-level exact R3D semantics | only when an actual FPS/downsampling site exists |
| DQ-RISE | Yes — dataset-level exact R3D semantics | only when an actual FPS/downsampling site exists |
| SAT | Yes — dataset-level exact R3D semantics | Yes — actual patch FPS |
| ManiFlow | No — intentional architecture-aware exception | only when N_raw > num_points; current 1024→1024 path is inactive |

### 5.1 为什么 ManiFlow 是例外

当前 DexMani ManiFlow 是：

    1024 points
    -> PointNetDense
    -> 1024 dense context tokens
    -> DiTX cross-attention

当前 stored point count 与 num_points 都为 1024，因此没有真实的 pre-token FPS bottleneck。

如果机械复制官方 R3D anchor-replacement dropout，最高约 80% dense tokens 可能变成同一首点 token。对 softmax cross-attention，重复 identical K/V 会改变 token multiplicity / attention measure，不等价于简单删除观测点。

因此：

    ManiFlow:
        common sensor augmentation
        + architecture-native FPS if active
        + NO R3D point dropout

这是本 thesis benchmark 的显式结构适配，不声称是 ManiFlow 官方 recipe。

### 5.2 为什么 DP3 / DQ-RISE / SAT 可以使用 R3D point dropout

- DP3：PointNet-style pointwise encoding + global max aggregation；重复点 multiplicity 不会像 dense attention 那样线性放大 softmax mass。
- DQ-RISE 默认 iDP3 / MultiStagePointNet：同样以 pointwise transforms + global max aggregation为主。
- SAT：dropout 后还有真实 FPS / local grouping；与 R3D 的 dropout -> FPS -> local patches 结构更接近。

这是将 R3D v2 的完整 augmentation lesson 用作统一 thesis training recipe，而不是声称这些 baseline 的原论文默认使用该 dropout。

论文表述应类似：

All point-cloud baselines are trained under our unified R3D-derived augmentation protocol, with architecture-aware exceptions where a transform changes token semantics.

不要写成“完全复现每个 baseline 的 native training augmentation”。

---

## 6. 必须修改

### R3D-01 — 恢复 R3D state latent width = 256

修改：

- dexmani_policy/configs/r3d.yaml
- dexmani_policy/agents/core/r3d.py
- dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py

要求：

    # r3d.yaml
    state_out_dim: 256

    # R3DAgent.__init__
    state_out_dim: int = 256

    # R3DObsEncoder.__init__
    state_out_dim: int = 256

不要修改共享 create_state_mlp 默认值，也不要改变其他 policies 的 state width。

预期有效结构：

    joint_state 19
        -> Linear(19, 64)
        -> activation
        -> Linear(64, 256)

    point feature 256 + broadcast state 256 = 512
    pc spatial PE 256 remains separate in current R3D backbone contract

注意：保持当前 released-code-equivalent cat_on_token=false 语义，不把 state 改为独立 token。

### AUG-01 — 所有主 point-cloud configs 关闭 hue

修改：

- dexmani_policy/configs/r3d.yaml
- dexmani_policy/configs/dp3.yaml
- dexmani_policy/configs/dqrise.yaml
- dexmani_policy/configs/sat.yaml

将 hue: 0.08 改为 hue: 0.0。

maniflow.yaml 当前已经 hue: 0.0，保持。

五个 config 的 common sensor values 应保持：

    augmentation_cfg:
      pc:
        coord_noise: {noise_std: 0.002, prob: 1.0}
        color: {brightness: 0.125, contrast: 0.5, saturation: 0.5, hue: 0.0, prob: 1.0}
      state:
        noise: {noise_std: 0.0002, prob: 1.0}

不要为此新增 Hydra inheritance/registry abstraction。

### AUG-02 — 恢复官方语义的 R3D point dropout，并用于 DP3 / DQ-RISE / SAT

修改：

- dexmani_policy/datasets/augmentation.py
- dexmani_policy/configs/dp3.yaml
- dexmani_policy/configs/dqrise.yaml
- dexmani_policy/configs/sat.yaml

#### 6.2.1 修改 generic PointDropout

当前 generic PointDropout：

- fixed dropout_ratio=0.3；
- 精确 drop int(N * ratio)；
- dropped positions 从 kept points 随机 resample。

这不是官方 R3D point-dropout semantics，而且当前仓库没有主 config 使用它。

改为简洁的官方语义：

    class PointDropout(Aug):
        __slots__ = ("max_dropout_ratio",)

        def __init__(self, max_dropout_ratio=0.8, prob=1.0):
            if not 0 <= max_dropout_ratio <= 1:
                raise ValueError("max_dropout_ratio must be between 0 and 1")
            super().__init__(prob=prob)
            self.max_dropout_ratio = float(max_dropout_ratio)

        def _augment(self, x):
            if self.max_dropout_ratio <= 0:
                return

            T, N = x.shape[:2]
            if N == 0:
                return

            for t in range(T):
                ratio = np.random.uniform(0.0, self.max_dropout_ratio)
                drop_mask = np.random.random(N) <= ratio
                if np.any(drop_mask):
                    anchor = x[t, 0].copy()
                    x[t, drop_mask] = anchor

语义要求：

- 每个 observation frame/sample 独立采样 ratio；
- ratio 位于 [0, 0.8]；
- per-point Bernoulli mask；
- replacement 是完整首点 XYZRGB；
- 不做 zero fill；
- 不做 resample-kept；
- 不要求精确 drop 固定数量。

BaseDataset 先切到 obs_horizon 再做 augmentation；T-frame 独立处理等价于官方 encoder 将 B*To flatten 后逐 sample dropout 的语义。

dataset-level dropout 在 normalization 前执行。由于该操作只是复制完整 point vector，channel-wise affine normalization 与 point-copy 可交换，因此无需把 DP3 / DQ-RISE / SAT 的 dropout逻辑分别塞进 encoder。

#### 6.2.2 DP3 / DQ-RISE / SAT config

在这三个 config 的 augmentation_cfg.pc 中加入：

    dropout: {max_dropout_ratio: 0.8, prob: 1.0}

保持 registry 原有顺序即可：

    coord noise
    -> color jitter
    -> optional color noise
    -> point dropout
    -> normalization later

官方 R3D 也是 sensor augmentation 后、encoder 内再 point dropout，因此该顺序语义一致。

#### 6.2.3 R3D 不添加 dataset dropout

r3d.yaml 不得添加 augmentation_cfg.pc.dropout。

R3D 已在 Uni3DPointcloudEncoder.forward 中：

    if training:
        random_point_dropout(..., max_dropout_ratio=0.8)

保留该路径，避免 double augmentation。

#### 6.2.4 ManiFlow 不添加 dropout

maniflow.yaml 保持没有 augmentation_cfg.pc.dropout。

不要为 ManiFlow 发明 resample-kept / mask-attention 等新机制；如果以后专门研究 dense-token dropout，再作为独立 ablation。

### R3D-02 — R3D 对 normalization 后完整 XYZRGB 做 clamp

修改：

- dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py

从：

    pc = pc.clone()
    pc[..., :3].clamp_(min=-1 - 1e-6, max=1 + 1e-6)

改为：

    pc = pc.clone()
    pc.clamp_(min=-1 - 1e-6, max=1 + 1e-6)

注释应准确说明：

- 输入已由 policy normalizer normalize；
- official R3D released policy clamp 的是完整 normalized point cloud；
- raw RGB 通常在 [0,1]，正常 normalized RGB 已处于 [-1,1] 时该操作是恒等；
- clamp 只保留在 R3D path，不重新放回 shared BaseAgent。

保持 clone。

### Existing smoke — 只同步现有实验断言

修改：

- dexmani_policy/agents/core/maniflow_smoke_test.py

更新 AugmentationTest.test_shared_policy_dataset_augmentation：

- 五个主 PC configs：hue == 0.0；
- dp3、dqrise、sat：
  - 存在 dataset dropout；
  - max_dropout_ratio == 0.8；
- r3d、maniflow：
  - dataset config 中没有 dropout；
- R3D internal dropout 不需要在这里复制一套 framework 验证。

现有 PointColorJitter hue-component test 保留，因为组件仍支持其它 recipe。

对 PointDropout 的修改属于算法语义变化，可在现有 AugmentationTest 内增加一个很小的定向检查，确认：

- sampled ratio不是固定 0.3；
- dropped rows使用 first-point replacement；
- XYZRGB整行一起 replacement。

不要新建 tests 目录，不要创建新的测试框架。

---

## 7. 明确保持不变

本任务不得顺带修改：

- cat_on_token=false 的当前 R3D released-code-equivalent 语义；
- SAT temporal patch matching；
- SAT use_shuffle_output / use_random_start；
- R3D / SAT / PointNext 当前真实 FPS sites；
- eval deterministic FPS；
- preprocess_point_cloud 的 N > K 条件；
- DP3 / DQ-RISE / ManiFlow 的无效 1024 -> 1024 FPS；
- ManiFlow num_points=1024 / dense-token budget；
- ManiFlow point dropout；
- optimizer / grad clipping / batch size / total train steps / scheduler；
- EMA implementation；
- BN/LN ablation；
- shared normalizer near-constant branch；
- auxiliary EE；
- recording / raw episode / Zarr schema；
- pretrained key coverage gate；
- config fail-fast production assertions；
- README / AGENTS / frozen docs；
- augmentation architecture abstraction / Hydra refactor。

也不要根据 R3D v2 的 separate proprio-token 文字直接改 architecture。

---

## 8. 推荐执行顺序

1. 读取 AGENTS.md 与当前 diff，保护用户已有改动。
2. 核对当前五个 PC configs。
3. 完成 R3D-01。
4. 完成 AUG-01。
5. 将 generic PointDropout 改成 official R3D semantics。
6. 给 DP3 / DQ-RISE / SAT 增加 dataset dropout；确认 R3D / ManiFlow 不加。
7. 完成 R3D-02。
8. 同步现有 AugmentationTest。
9. 查看最终 diff，撤销 scope 外改动。
10. 只做最小验证，不启动完整训练、DDP、rollout、视频。

---

## 9. 最小验证

遵循 AGENTS.md。

### 9.1 静态检查

    git diff --check

    conda run --no-capture-output -n policy python -m py_compile       dexmani_policy/datasets/augmentation.py       dexmani_policy/agents/core/r3d.py       dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py       dexmani_policy/agents/core/maniflow_smoke_test.py

### 9.2 Config resolve

    for cfg in r3d dp3 dqrise sat maniflow; do
      conda run --no-capture-output -n policy         python dexmani_policy/smoke_test.py --config-only "$cfg" || exit 1
    done

### 9.3 现有 augmentation 定向 smoke

    conda run --no-capture-output -n policy       python -m unittest       dexmani_policy.agents.core.maniflow_smoke_test.AugmentationTest

### 9.4 R3D model smoke

环境具备数据、权重和 GPU 时：

    conda run --no-capture-output -n policy       python dexmani_policy/smoke_test.py r3d

因环境缺失无法运行则报告 NOT VERIFIED；不要为通过 smoke 改模型。

---

## 10. 验收标准

实施后必须满足：

1. r3d.yaml：state_out_dim == 256。
2. R3DAgent / R3DObsEncoder R3D-specific default：state_out_dim == 256。
3. 五个主 PC configs common sensor recipe 一致，全部 hue == 0.0。
4. PointDropout 使用：
   - max_dropout_ratio=0.8；
   - per-sample/frame r ~ U(0, 0.8)；
   - per-point Bernoulli mask；
   - first-point full-row replacement。
5. DP3 / DQ-RISE / SAT dataset config 启用该 dropout。
6. R3D dataset config 无 dropout，继续只用现有 internal official dropout。
7. ManiFlow dataset config 无 dropout。
8. R3D normalized XYZRGB 全 tensor clamp 到 [-1-1e-6, 1+1e-6]。
9. cat_on_token / SAT temporal semantics / FPS / ManiFlow token budget 未改。
10. EMA / BN-LN / normalizer / optimizer / training protocol / recording-Zarr 未改。
11. Existing augmentation smoke 与上述 protocol 一致。
12. 最终 diff 不包含生产式 guardrails 或无关重构。

---

## 11. 论文实验解释边界

实施后，主实验应被描述为：

Unified R3D-derived augmentation benchmark for point-cloud policies.

不是：

exact native training recipe reproduction for every baseline.

需要明确：

- R3D 是最接近 official full recipe 的 baseline；
- DP3 / DQ-RISE / SAT 使用同一 R3D-derived augmentation，以减少 augmentation confound；
- ManiFlow 对 point dropout 做 architecture-aware exception，因为当前 DexMani realization 直接保留 1024 dense context tokens；
- FPS randomization 只在真正使用 FPS 采样的结构中有意义。

如果 thesis 后续需要证明 point dropout 的迁移收益，可单独做小规模 ablation：

    DP3 + common sensor aug
    vs
    DP3 + common sensor aug + R3D point dropout

以及：

    R3D full
    vs
    R3D without point dropout

这些 ablation 不属于本次 Codex implementation task。

---

## 12. 最终汇报格式

完成后简洁汇报：

1. Changed：按文件列出实际改动；
2. Protocol：列出五个 PC policies 最终 augmentation matrix；
3. Preserved intentionally：明确 cat_on_token=false、SAT FPS/shuffle、ManiFlow no-dropout、EMA/training protocol 未改；
4. Validation：所有实际命令标 PASS / FAIL / NOT VERIFIED；
5. Remaining：只有真实阻塞或新的 paper/code contradiction 才列出。

不要把已明确排除的工程化事项重新加入问题列表。
