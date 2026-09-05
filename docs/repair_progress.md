# 修复进度与阶段交接

当前状态：五个代码阶段均已完成；最终 CPU 验收通过。
下一步：如需补齐 GPU 验收，需在可提供超过默认 ManiFlow smoke 显存需求的空闲宿主 GPU
上重跑该 smoke；不修改模型或 smoke 配置来规避资源限制。

## 已知基线

- HEAD：`9e9597f4e3a06702f23cc88084b821217053a580`，与指南一致。
- 制定方案前，工作区仅原修复指南文件未跟踪。
- `policy` 环境：Python 3.10.20，PyTorch 2.4.1+cu124。
- Preflight full suite：25 tests passed，0 failed，0 skipped。
- 宿主 GPU（命令级提权）：CUDA 可用，1 × NVIDIA GeForce RTX 4090。
- 用户已明确指定 `policy` 为本轮唯一 Conda 环境。
- Phase 1 改动：`BaseAgent.forward()` 委托给 `compute_loss()`；训练调用 model forward；
  gradient accumulation 以 logical group size 缩放并处理尾组；空 loader 和 K<1 fail-fast；
  旧 divisibility warning/API 被替换。
- Phase 1 targeted：`tests.test_training_regressions` 5/5 PASS；full suite 30/30 PASS。
- Phase 1 GPU smoke：`dp3` PASS，`sat` PASS；`maniflow` default smoke batch 在 RTX 4090
  上 OOM，NOT VERIFIED。未改变模型、配置或 smoke 工具来规避该资源限制。
- Phase 2 改动：runner 复用 validation RGB preprocess；DP 与每个 MultiTask child 绑定
  各自 dataset RGB 尺寸；blender 仅接收 control action dimensions；MultiTask control
  mode 校验与 fatal EvalEpisodeError 传播已修复。
- Phase 2 targeted：`tests.test_eval_regressions` 6/6 PASS；RGB config bindings PASS。
- Phase 2 GPU smoke：`dp` PASS，`multitask_dit` PASS。
- Phase 3 改动：selector 写入严格 v2 `best_ckpt.json`（实际推理参数、实际 unique
  selection seeds 与相对 checkpoint 路径）；`best` 仅解析有效 v2 record，EMA 缺失
  fail-closed；final eval 排除 selection seeds，并记录 held-out/inference 元数据。
- Phase 3 precedence：显式 CLI > 显式 dotlist > selection record > config；非 `best`
  不读取 selection record 的推理设置。
- Phase 3 targeted：`tests.test_eval_regressions` 16/16 PASS；`git diff --check` PASS。
- Phase 4 改动：export 与 direct restore 共用 selected inference resolver；v3 artifact 的
  `temporal_ensemble_coeff` 为必填显式字段（float 或 null），缺失不回退；runtime 用
  已验证且裁切至 control dimensions 的 CPU prediction 做 overlap blending，reset/warmup
  正确隔离 episode state。
- Phase 4 targeted：`test_deployment_*.py` 28/28 PASS；全套 52/52 PASS；
  `git diff --check` PASS。
- Phase 5 改动：ManiFlow absolute consistency 的 student/teacher target 已纠正，relative
  分支不变；FlowMatch 与 MultiTaskDataset 的必要构造参数 fail-fast；PointDropout=0、
  normalizer field-view cache、ReplayBuffer root attrs 已修复。
- Phase 5 targeted：`tests.test_algorithm_regressions` 7/7 PASS；最终全套 59/59 PASS；
  `git diff --check` PASS。
- Phase 5 GPU smoke：`maniflow` 在默认 forward/backward 处 OOM（RTX 4090，实际仅余
  约 1.07 GiB，额外申请 1.50 GiB），NOT VERIFIED；未为通过该检查修改模型、配置或 smoke。
- 后续清理：移除无效的 `best.pt` / `--link-best` 生成路径、未调用的
  `resolve_best_checkpoint()` 与一次性 NFE null 特判；评测脚本统一使用 `policy`
  的 `conda run`，README、评测机制文档和执行指南均改为当前严格 v2 / held-out 语义。
- 清理验证：受影响 Python `py_compile`、5 个 eval shell 的 `bash -n`、
  `tests.test_eval_regressions` 16/16、全套 59/59 和 `git diff --check` 均 PASS。
- 用户确认本机有 GPU；sandbox 内不可见不代表宿主无 GPU。
- 本轮未执行真实 DDP/NCCL、真实 deployment artifact 导出或完整仿真评测；这些均不在
  本 workflow 的授权范围内。
- workflow 配置检查：四个 TOML 解析及三档模型/effort 映射通过；`git diff --check` 通过。
  本机 bundled catalog 支持指定组合。三个自定义 agent 的客户端加载尚未实际运行验证。
  `codex --strict-config features list` 不支持 strict-config 参数，不能用作配置验收。

## 固定决策

- 五阶段自动推进；代码失败留在当前阶段修复，环境未验证项单独记账。
- sol-high / terra-xhigh / luna-max 按 workflow 分工；同文件单写入者。
- 不新增 legacy 回退、兼容层、迁移器或临时代码；严格 best v2 record 与显式 artifact 字段。
- 原指南其他范围、不变量、算法机制与 Deferred 保持。

## 验收和 compact

- Preflight：PASS。
- Phase 1：PASS（ManiFlow GPU smoke：NOT VERIFIED，默认测试 batch OOM）。
- Phase 2：PASS。
- Phase 3：PASS。
- Phase 4：PASS。
- Phase 5：代码/CPU 验收 PASS；ManiFlow GPU smoke NOT VERIFIED。
- 最终验收：CPU/定向验收 PASS；GPU 验收 NOT VERIFIED（ManiFlow 默认 smoke OOM，真实
  多 GPU DDP/NCCL 也未在本 workflow 中执行）。
- 阶段交接摘要：Preflight 至 Phase 5 已写入本文件；原生 compact 仍不可由当前工具调用。
- 原生 compact：NOT VERIFIED（当前会话没有可调用接口）。
- 最终交付前必须保留尚未验证的 GPU/DDP 等项目，不将其合并为 PASS。
