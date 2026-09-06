# DexMani_Policy Repair Workflow

本文件定义未来有明确 scope 的修复流程，不保存某次已完成任务的 phase 清单或执行日志。当前状态见 [repair_progress.md](repair_progress.md)。

## 启动与范围

1. 先读仓库根目录的 `AGENTS.md`、本文件与 `repair_progress.md`，再核对实际调用链和当前工作区改动。
2. 用户给出的 scope、禁止项和 stop rule 优先。不要把代码修复扩展成架构、性能、数据集或实验改造。
3. 不新增 legacy 兼容分支、迁移器、旧 API 别名、临时 adapter 或静默 fallback。现有 strict v2 selection record 和 v3 deployment artifact 契约保持不变，除非用户明确变更它们。
4. 不覆盖无关的用户改动，不修改相邻仓库、实验产物、checkpoint、视频或数据集。

## 执行循环

`定位证据 → 最小 regression/contract test → 最小修复 → 定向验证 → 更新关联文档 → 记录简洁状态`

- 优先修复根因；接口改动同步检查所有直接调用方。
- 一项状态或参数只保留一个权威解析点，避免多层重复校验与隐式推断。
- 可并行的独立工作才使用协作；同一文件同一时刻只允许一个写入者。
- 如果验证受环境限制，记录为 `NOT VERIFIED`，不通过改模型、配置或测试工具来规避环境限制。

## 验证

按风险选择最接近的检查，先语法/导入级，再定向单元测试，最后在必要时运行完整 unit suite。默认使用：

```bash
conda run --no-capture-output -n policy python -m unittest discover -s tests -p 'test_*.py' -v
conda run --no-capture-output -n policy python -m py_compile <changed-python-files>
git diff --check
```

真实训练、真实 DDP/NCCL、simulator rollout、视频录制、deployment export 或机器人执行只在用户明确授权时运行。GPU/数据/显示不可用不是修改核心行为的理由。

## 收尾

- 删除本次改动链中已证实无调用、过时、临时或兼容性残留的代码和注释；不要删除仍承载当前格式或运行时职责的实现。
- 更新 README、机制文档和架构说明中受影响的命令、优先级、输出目录和失败语义。
- 将 `repair_progress.md` 压缩为当前状态、已验证项、未验证项和 deferred；历史执行细节保留在 Git 历史，而不是活动文档中。
- 达到用户的 stop rule 后停止，不额外重构或开展研究性工作。
