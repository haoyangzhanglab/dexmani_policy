# R3D / Point-Cloud Augmentation Fix — Codex Task

## 1. 任务目标

在 **不扩大研究范围、不引入生产级工程化机制、不改变其他 baseline 方法语义** 的前提下，修复当前 `dexmani_policy` 中已确认会影响博士论文实验正确性或公平性的 R3D / point-cloud augmentation 问题。

本任务只解决以下三类事项：

1. **R3D 方法本体修正**：恢复官方 R3D 有效计算图中的 proprio/state latent width。
2. **论文主实验的公共 point-cloud sensor augmentation 对齐**：统一 XYZ / RGB BCS / proprio noise recipe，并移除不属于 R3D recipe 的 hue jitter。
3. **R3D fidelity 修正**：R3D 对 normalization 后的完整 XYZRGB point cloud 做官方一致的范围 clamp。

任务定位是 **PhD thesis experimental codebase**。优先保证方法语义、实验可解释性和比较公平性；不要为未来错误配置增加额外 guardrail、assertion、registry、测试框架或抽象层。

---

## 2. 当前事实与参考基线

执行前先读取当前仓库实际代码，不要仅依赖本文。本文制定方案时核对的基线为：

- DexMani Policy: `haoyangzhanglab/dexmani_policy@497c49dc816dd05ae22cd147a6ff0abd56ce9d46`
- R3D official: `Wushr-Lance/R3D-Policy@e637c0148376ddc4b5e667fa8f8e108cb8ff7a85`
- SAT official: `XiaohanLei/SAT@cd7c0a8877d6090a9a85ebee0ceca961830b3654`
- ManiFlow official: `geyan21/ManiFlow_Policy@ef2f116f1f90163ed36e657b8c5503740bb468af`
- DP3 official: `YanjieZe/3D-Diffusion-Policy@47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`

关键已确认事实：

- 官方 R3D 的有效 `DP3Encoder` 路径会把 state MLP 改为 `(64, pointcloud_encoder_cfg['embed_dim'])`；当前 R3D `embed_dim=256`，因此 state branch 应为 `state_dim -> 64 -> 256`。
- DexMani 当前 R3D config / R3D 专属构造器默认值仍为 `state_out_dim=64`，这是模型结构偏差，不是 embodiment adaptation。
- 官方 R3D color augmentation 为 brightness / contrast / saturation，不包含 hue。
- 官方 R3D 在 normalization 后对完整 point-cloud tensor 做 `[-1-1e-6, 1+1e-6]` clamp；DexMani 当前仅 clamp XYZ。
- SAT 官方方法本身将 local patch tokens 视为 unordered set，存在独立 FPS / local-token random shuffle 后再进行 temporal feature fusion；**不要把 lack of cross-frame patch correspondence 当作本任务 bug 修复**。
- ManiFlow 的 visual token 数本身是 task-dependent hyperparameter；**不要在本任务中修改当前 DexMani ManiFlow 的 1024-token setting**。
- R3D point dropout 与 FPS 属于 architecture/method-specific mechanism，不属于本任务需要强制统一的公共 sensor augmentation。

---

## 3. 最终实验协议

论文主实验中，以下 **continuous sensor-level augmentation** 对所有单任务 point-cloud policies 保持一致：

```text
XYZ noise:
    Gaussian std = 0.002
    clip = ±0.004
    prob = 1.0

joint/proprio noise:
    Gaussian std = 0.0002
    clip = ±0.0004
    prob = 1.0

point-cloud RGB:
    brightness = 0.125
    contrast = 0.5
    saturation = 0.5
    hue = 0.0
    prob = 1.0
```

适用 configs：

- `r3d.yaml`
- `dp3.yaml`
- `dqrise.yaml`
- `sat.yaml`
- `maniflow.yaml`

注意：

- 这里统一的是 **sensor-level recipe**，不是要求所有模型拥有完全相同的 sampling / tokenization / point-dropout。
- R3D 保留其官方 encoder-level random point dropout。
- DP3 / DQ-RISE / SAT / ManiFlow **不要新增 dataset-level R3D point dropout**。
- FPS 只在模型实际存在 sampling site 时生效；不要人为执行 `1024 -> 1024` FPS。
- 现有 train-random / eval-deterministic FPS 语义保持不变。

---

## 4. 必须修改

### R3D-01 — 恢复 R3D state latent width = 256

修改：

- `dexmani_policy/configs/r3d.yaml`
- `dexmani_policy/agents/core/r3d.py`
- `dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py`

要求：

1. `r3d.yaml`
   ```yaml
   state_out_dim: 256
   ```

2. `R3DAgent.__init__` 的 R3D-specific 默认值：
   ```python
   state_out_dim: int = 256
   ```

3. `R3DObsEncoder.__init__` 的 R3D-specific 默认值：
   ```python
   state_out_dim: int = 256
   ```

不要修改共享 `create_state_mlp` 的默认值，也不要改 DP3 / SAT / ManiFlow / DQ-RISE 的 state width。

预期 R3D 主实验有效结构：

```text
joint_state 19
    -> Linear(19, 64)
    -> activation
    -> Linear(64, 256)

point token 256 + state 256 = obs feature 512
512 + pc spatial PE 256 = R3D obs token storage dim 768

OneWayTransformer 对 feature / PE 按现有实现继续处理；
不要借此重构 token layout。
```

---

### AUG-01 / AUG-02 — 统一公共 sensor augmentation，关闭 hue

修改：

- `dexmani_policy/configs/r3d.yaml`
- `dexmani_policy/configs/dp3.yaml`
- `dexmani_policy/configs/dqrise.yaml`
- `dexmani_policy/configs/sat.yaml`

将：

```yaml
hue: 0.08
```

改为：

```yaml
hue: 0.0
```

`maniflow.yaml` 当前已经是 `hue: 0.0`，保持不变。

同时核对上述五个 point-cloud configs 的公共 sensor augmentation 数值均为：

```yaml
pc:
  coord_noise: {noise_std: 0.002, prob: 1.0}
  color: {brightness: 0.125, contrast: 0.5, saturation: 0.5, hue: 0.0, prob: 1.0}
state:
  noise: {noise_std: 0.0002, prob: 1.0}
```

若数值已经一致，不要为了“共享配置”进行 Hydra 抽象或重构；直接保持显式 config，方便论文实验阅读和 override。

更新与新语义冲突的局部注释，例如 R3D config 中不应继续描述 `contrast/saturation/hue` 为官方 R3D color recipe。

---

### R3D-02 — R3D 对 normalization 后完整 XYZRGB 做 clamp

修改：

- `dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py`

当前语义：

```python
pc = pc.clone()
pc[..., :3].clamp_(min=-1 - 1e-6, max=1 + 1e-6)
```

改为官方 R3D 一致的完整 normalized point cloud clamp：

```python
pc = pc.clone()
pc.clamp_(min=-1 - 1e-6, max=1 + 1e-6)
```

同时更新附近注释，准确表达：

- 此处输入已经经过 policy normalizer；
- clamp 作用于 normalized XYZRGB；
- RGB raw 值通常来自 `[0,1]`，正常 normalization 后位于 `[-1,1]` 时 clamp 是恒等操作；
- 该 clamp 只保留在 R3D encoder boundary，不要重新提升为所有 point-cloud policy 的全局行为。

保持 `clone()`，避免原地修改调用方持有的 normalized observation tensor。

---

### Existing smoke assertion — 只同步已有测试的过时实验假设

修改：

- `dexmani_policy/agents/core/maniflow_smoke_test.py`

只调整现有 `AugmentationTest.test_shared_policy_dataset_augmentation` 中关于 hue 的旧断言：

- 五个 configs 都应断言 `hue == 0.0`。
- ManiFlow 继续断言没有 dataset-level `dropout`。
- **不要新增一套 R3D fidelity test framework。**

保留 `PointColorJitter` 对 hue 功能本身的单元检查（`test_hue_and_clipping_remain_available`）；组件可以支持 hue，只是 thesis point-cloud main recipe 不使用 hue。

---

## 5. 明确禁止修改

本任务不得顺带处理以下事项：

- 不给 DP3 / DQ-RISE / SAT / ManiFlow 增加 R3D point dropout。
- 不修改 `datasets/augmentation.py::PointDropout` 的 API 或语义。
- 不做 SAT temporal patch matching / Hungarian matching / shared FPS centers。
- 不关闭 SAT 的 `use_shuffle_output` 或 `use_random_start`。
- 不改变 R3D / SAT / PointNext 的现有 FPS randomization。
- 不把 `preprocess_point_cloud` 的条件从 `N > K` 改成 `N >= K`。
- 不强制 DP3 / DQ-RISE / ManiFlow 执行无效的 `1024 -> 1024` FPS。
- 不修改 ManiFlow `num_points=1024` / dense-token budget。
- 不修改 optimizer、gradient clipping、batch size、训练步数、LR scheduler、EMA。
- 不修改 shared normalizer 的 near-constant-dimension 公式。
- 不修改 auxiliary EE 路径。
- 不新增 config fail-fast assertion、pretrained key coverage gate、schema constraint 等生产级防御机制。
- 不修改 recording / Zarr schema。
- 不新增 `tests/` 目录。
- 不重构 augmentation registry / dataset pipeline。
- 不修改 README / AGENTS / docs，除非任务执行中发现本次改动导致现有局部说明直接错误；即使如此优先只修局部代码注释。

---

## 6. 推荐执行顺序

按下面顺序执行，避免无关探索和重复运行：

1. 阅读：
   - `AGENTS.md`
   - 五个 point-cloud configs
   - `agents/core/r3d.py`
   - `agents/obs_encoder/pointcloud/r3d_obs_encoder.py`
   - `agents/core/maniflow_smoke_test.py::AugmentationTest`
2. 检查工作区现有改动，绝不覆盖用户未提交修改。
3. 完成 R3D-01。
4. 完成 AUG-01 / AUG-02。
5. 完成 R3D-02。
6. 同步已有 augmentation smoke assertion。
7. 查看最终 diff；若出现本任务之外的文件，先判断并撤销无关改动。
8. 运行最小验证，不启动训练或 rollout。

不要在实施前重新设计整个 point-cloud pipeline；本文中的方法决策已经完成。

---

## 7. 最小验证

遵循 `AGENTS.md`，只执行与本任务直接相关的低成本验证。

### 7.1 静态检查

```bash
git diff --check

conda run --no-capture-output -n policy python -m py_compile \
  dexmani_policy/agents/core/r3d.py \
  dexmani_policy/agents/obs_encoder/pointcloud/r3d_obs_encoder.py \
  dexmani_policy/agents/core/maniflow_smoke_test.py
```

### 7.2 受影响 config resolve

```bash
for cfg in r3d dp3 dqrise sat maniflow; do
  conda run --no-capture-output -n policy \
    python dexmani_policy/smoke_test.py --config-only "$cfg" || exit 1
done
```

### 7.3 现有 augmentation 定向 smoke

```bash
conda run --no-capture-output -n policy \
  python -m unittest \
  dexmani_policy.agents.core.maniflow_smoke_test.AugmentationTest
```

### 7.4 R3D model smoke

由于修改了 R3D encoder / constructor，在环境具备依赖时再运行：

```bash
conda run --no-capture-output -n policy \
  python dexmani_policy/smoke_test.py r3d
```

如果因 GPU、数据、Uni3D 权重或外部环境缺失无法执行，明确报告 **NOT VERIFIED**；不要为了让 smoke 通过而改变模型逻辑。

不要启动完整训练、DDP、仿真 rollout 或视频评测。

---

## 8. 验收标准

完成后必须同时满足：

1. `r3d.yaml` 的 `state_out_dim == 256`。
2. `R3DAgent` 与 `R3DObsEncoder` 的 R3D-specific 默认 `state_out_dim == 256`。
3. `r3d/dp3/dqrise/sat/maniflow` 五个主 point-cloud configs 的公共 sensor augmentation 数值一致，且 `hue == 0.0`。
4. R3D 仍只有官方 encoder-level point dropout；其他四个主 baseline 没有因本任务新增 dataset-level point dropout。
5. R3D encoder 对 normalization 后的完整 point cloud tensor 做 `[-1-1e-6, 1+1e-6]` clamp。
6. clamp 仍是 R3D-local 行为，没有泄漏到 shared BaseAgent / shared point-cloud preprocessing。
7. SAT FPS / patch shuffle / temporal fusion 未改。
8. ManiFlow token count / FPS behavior 未改。
9. shared normalizer、optimizer、training protocol、recording/Zarr 均未改。
10. 已有 augmentation smoke 的 hue expectation 与新主实验协议一致。
11. 所有实际执行的验证结果在最终汇报中逐项标为 PASS / FAIL / NOT VERIFIED。
12. 最终 diff 小而完整，除必要配置、R3D 专属实现和现有 smoke assertion 外不包含无关重构。

---

## 9. 最终汇报格式

实施完成后只需简洁汇报：

1. **Changed**：按文件列出实际修改。
2. **Preserved intentionally**：明确 point dropout、SAT FPS/shuffle、ManiFlow token budget、training protocol 未改。
3. **Validation**：列出实际运行命令及 PASS / FAIL / NOT VERIFIED。
4. **Remaining issues**：只有发现真实阻塞或与本文事实冲突时才列出；不要把已明确排除的工程化事项重新加入问题列表。

如果当前代码已发生变化导致本文某个前提不再成立，优先以实际调用链为准，并在修改前确认该变化是否会改变论文实验语义；不要机械套用旧行号或旧注释。
