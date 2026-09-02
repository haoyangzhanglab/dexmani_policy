# ActionFlow Stabilization v1 — 结果报告

> **日期**：2026-09-02
> **范围**：`docs/ActionFlow-Stabilization与NFE校准执行指南.md` 定义的一轮 Stabilization v1（correctness/stability 修复 + 一次 100k 训练 + 受控 midpoint-only NFE 校准）。
> **任务**：`pour`（seed=42，max_train_episodes=80，单卡 4090 D 24GB）。

---

## 1. 修改文件与原因

| 文件 | 改动 | 指南条款 |
|------|------|----------|
| `dexmani_policy/agents/core/action_flow.py` | `ActionFlowAgent.__init__` 顶部新增 `n_obs_steps=2` + `n_obs_steps-1+n_action_steps>horizon` fail-fast | §2.1 |
| `dexmani_policy/agents/action_decoders/action_flow_flowmatch.py` | ① `__init__` 补 `noise_shift_alpha>0` / `0≤ratio≤1` / `denoise_steps` 正整数 / midpoint 偶数校验；② `_resolve_nfe` 去 `int()` 静默截断；③ `setup_kv_cache` 移入 try | §2.2 / §2.3 |
| `tests/test_action_flow_contract.py`（新增） | 8 项契约回归测试（fail-fast、flow 校验、KV parity、异常清理、state_hist=38） | §2.4 |
| `dexmani_policy/smoke_test.py` | ActionFlow 参数预算注释更新 + 新增宽松 gate（16–18M / 58–60M / 74–78M） | §2.5 |
| `dexmani_policy/agents/obs_encoder/pointcloud/geoformer.py` | coordinate-wavelength / normalized-workspace RoPE 注释修正（仅注释，无行为变化） | §2.6 |

## 2. 未改变的 canonical recipe（明确声明）

horizon=16 / n_obs_steps=2 / n_action_steps=8 / state_dim=19 / action_dim 19|21 / GeoFormer 4L×576 / ActionDiT 8L×768 ctx=384 / NoiseShift α=3 ratio=0.75 / solver=midpoint / NFE=2 / batch64×grad-accum2 / lr=1e-4 wd=1e-3（obs_wd=1e-6）/ cosine+warmup500 / bf16+compile default / max_train_episodes=80 / total_train_steps=100000 —— **全部未动**。

§3 禁止项（FPS `use_shuffle_output`、point-cloud 归一化、patch count/radius、state token、multi-scale memory、NoiseShift sampler、dropout/drop-path、step-conditioning、架构扩展、EMA 等）一律未碰。

## 3. 静态 + smoke + preflight 状态

- `pytest tests/`：**76 passed / 7 skipped**（含新增 8 项 contract 测试）。
- `smoke_test.py action_flow`：**PASSED** —— 全 trainable 参数有梯度、action/action_ee state 契约、strict checkpoint roundtrip、KV parity、参数量 16,851,168 / 58,806,675 / 75,657,843 落预算。
- 600-step preflight：**PASSED** —— loss 1.395→0.025 单调有限、5 个 milestone 全存、无 NaN/Inf、~3.6 it/s 无重复 recompile。

> 环境注意：本机 `pytest` 需 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`（ROS `launch_testing` 插件缺 `lark` 会拖垮插件自动加载，与代码无关）。

## 4. 100k 实验路径

`experiments/action_flow/pour/2026-09-01_22-47_42/`

7.25h，100000/100000 步，最终 loss 0.00052（lr 余弦衰减到 0），5 个 milestone（20/40/60/80/100pct）全存，训练全程零 non-finite。

## 5. 40pct NFE screening（25 seeds，EMA，midpoint）

| NFE | 成功数 | SR |
|-----|--------|-----|
| 2 | 12/25 | 0.48 |
| 4 | 16/25 | 0.64 |
| 8 | 17/25 | 0.68 |
| 10 | 16/25 | 0.64 |

paired net wins vs NFE2：NFE4=+4(7 胜/3 负)、NFE8=+5(7/2)、NFE10=+4(5/1)。元数据（ckpt_tag=40pct、ckpt_path、n_total=25、denoise_steps、eval_seed=1066）四组全一致，25 seeds 同集。

## 6. 最终 NFE candidate N\* 与理由

`max(S₄,S₈,S₁₀) − S₂ = 5 > 1` → 不满足「NFE2 plateau」；取距 max(17) ≤1 的**最小** NFE = **4**，paired net wins +4 ≥ 2 → **N\* = 4**。

## 7. 100-seed NFE2 vs NFE\*

| | NFE2 | NFE4 |
|---|---|---|
| 100-seed SR | 61/100 = 0.61 | 65/100 = 0.65 |
| paired vs NFE2 | — | +4（15 胜 11 负） |

`SR(NFE4) − SR(NFE2) = +0.04 < 0.05` → **保留 default NFE = 2**（NFE4 仅边缘更好，未达 §9 的 5pp 阈值）。

> 注：screening 25-seed 曾给出 NFE2=0.48 vs NFE4=0.64 的假象（差距 0.16），100-seed 确认收窄到 0.04 —— 正是 §9 要防的「小样本过信」。

## 8. 100pct/25ep tail diagnostic

**18/25 = 72.0%**（NFE2、EMA、strict 加载）。对比 40pct@25（48%）→ **无 late degradation，40k→100k 反而显著提升（48%→72%）**。

## 9. Engineering / performance gate 结论

- **Engineering：PASS** —— tests / smoke / preflight / 100k 无 NaN / 5 milestone / strict EMA restore / NFE 评测全通过。
- **40pct 性能 gate：YELLOW** —— default NFE=2 的 40pct 100-seed SR = **0.61**（历史 anchor 0.68 略低）。不触发 §14 stop（仅 Red<0.60 停），但属边缘，需人工 review。

## 10. 后续研究建议（仅列，不自动执行）

1. pretrained 3D perception（Uni3D random vs pretrained）；
2. 80→125 demos 数据多样性；
3. proprio/state token 或 FK keypoint；
4. fine/global multi-scale memory；
5. isotropic/granularity-aware 几何归一化；
6. 仅当 NFE2 明显低于高 NFE 时才研究 Shortcut/Consistency Flow —— 本轮 NFE2 vs NFE4 差距仅 0.04，**不满足此前提**。

---

## 结论

Stabilization v1 全部执行完成：代码 fail-fast/校验/gate 落地，100k 训练健康无 NaN，NFE 校准判定 **default NFE 维持 2**（NFE4 边缘 +0.04 未达 5pp 阈值）。唯一关注点是 **40pct NFE2 = 0.61 落 Yellow**（略低于历史 0.68），但 100pct 达 72% 且无 late degradation，整体健康。
