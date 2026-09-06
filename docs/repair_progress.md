# Repair Status

当前没有进行中的修复 workflow。历史主修复和后续 correctness repair 均已完成；逐阶段执行日志已从仓库文档中移除，可按需从 Git 历史查阅。

## 当前契约

- `best_ckpt.json` 仅接受严格的 record version 2；不提供旧记录、迁移层或静默回退。
- deployment artifact 保持 v3，并要求显式的 `temporal_ensemble_coeff`；不提供 legacy artifact 回退。
- Real deployment export 只接受 Policy Zarr v7；point-cloud field semantics 保留已验证的
  preprocessing identity，fingertip field 必须携带已冻结的 derivation、policy ID 与 geometry SHA-256。
- 启用 EMA 的训练恢复必须同时拥有 EMA 权重和非负整数 updater step。consistency teacher 要求 `training.use_ema=true`。
- `best` 的最终评测与 demo 复用 selection record 的推理策略；非 `best` 保持配置驱动行为。评测结果目录与视频目录彼此独立。
- temporal ensemble coefficient 只能为 `null` 或有限、非负实数；selection record 的 policy RNG 模式固定为 `episode_seed`。

## 最近一次代码级验证

- `tests.test_training_regressions`: 9/9 PASS
- `tests.test_eval_regressions`: 30/30 PASS
- `tests.test_deployment_contract`: 13/13 PASS
- `tests.test_deployment_runtime`: 8/8 PASS
- `tests.test_deployment_export`: 10/10 PASS
- 完整 unit suite: 82/82 PASS
- 受影响 Python 文件 `py_compile`、eval shell `bash -n` 与 `git diff --check`: PASS

以上验证仅使用 CPU synthetic tensors、fake runner、temporary checkpoint/file 与静态检查。未执行真实训练、DDP/NCCL、simulator rollout、checkpoint selection rollout、最终评测 rollout 或 demo rollout。

## Deferred

保留既有 deferred 范围：真实 DDP/NCCL、ManiFlow GPU smoke、部署 CPU qualification、DINO/CLIP fully-offline construction、MultiTask deployment contract、normalizer/XYZ/point-ordering 改动及研究性实验。除非后续任务明确授权，不应以本修复为由扩展到这些范围。

## 后续修复

新的修复任务从 [repair_workflow.md](repair_workflow.md) 开始，并在完成后将本文件更新为简洁的当前状态；不要把已完成的历史 phase checklist 重新作为活动任务执行。
