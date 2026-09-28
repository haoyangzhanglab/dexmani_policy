# AGENTS.md — DexMani_Policy AI 研究协作约定

本仓库是**个人机器人学习研究仓库**。AI coding 的目标是帮助研究想法快速、可靠地落地和验证，而不是把代码库建设成通用框架或企业级软件。

文档、解释和任务总结默认使用中文；代码标识符、配置键、命令、库名以及已经形成约定的技术术语保持原样，不做生硬翻译。

## 1. 工作优先级

遇到设计选择时，按以下顺序判断：

1. 研究问题是否清楚，主要实验变量是什么。
2. config、code、data 的真实行为是否正确。
3. 实现是否足够简单，便于快速修改和做对照实验。
4. 实验是否可复现、可解释。
5. 工程抽象与代码美化只做到当前研究真正需要的程度。

不要为了“更通用”或“以后可能复用”主动引入 registry、factory、plugin system、兼容层、大型基类或跨目录重构。

## 2. 事实来源

- **当前 Policy**：读取 `dexmani_policy/configs/*.yaml`，再沿 `agent._target_` 进入实际 Python 实现。
- **已有实验**：以实验目录中保存的 resolved `config.yaml` 和 checkpoint 为准，不用当前默认 config 推测历史实验。
- **运行行为**：以实际 training / evaluation / loader / runtime 代码及其校验为准。
- `README.md` 与 `docs/` 负责导航和背景说明，不替代 config/code。

不要假设 config 名、Python 文件名和 Agent 类名一一对应。

## 3. 开始编码前

处理 Policy 新增、修改或 review 时，先沿真实调用链阅读：

```text
config
→ agent._target_
→ Agent
→ observation encoder
→ backbone / action decoder
→ compute_loss
→ predict_action
```

先明确这次修改主要属于哪一层：

- observation / representation
- policy architecture
- action representation
- training objective
- inference algorithm

如果一次实验同时改变很多层，应先判断是否能拆成更小的可解释实验。

## 4. 实现原则

- **最小但完整**：优先做能端到端运行的最小修改，不留下半套新旧逻辑。
- **控制变量**：与研究问题无关的行为尽量保持不变，避免顺手改超参数、命名或邻近模块。
- **语义复用**：已有组件语义一致时直接复用；语义不同则保持新逻辑局部化，不为了形式统一强行共用。
- **少做抽象**：单个实验使用的小机制可以直接放在最自然的位置，不必为它创建新的公共层。
- **先读再改**：修改共享组件前先搜索真实调用者，确认影响范围。
- **不因非关键细节停滞**：研究想法已足够明确但局部实现细节未指定时，选择最简单合理方案并在结果中说明。

不要把单次模型宽度、学习率、batch size、NFE、训练时长或实验结论写进全局 AI 文档。

## 5. 验证策略

按成本从低到高验证，只运行能证明当前改动的检查：

1. **纯文档**：检查 diff、路径、链接和命令是否与仓库一致。
2. **Python 局部修改**：语法 / import / 最小定向检查。
3. **Config 或 Policy 修改**：

   ```bash
   python dexmani_policy/smoke_test.py --config-only <config_name>
   ```

4. **涉及 dataset / model / forward / inference 链路**：在环境允许时运行

   ```bash
   python dexmani_policy/smoke_test.py <config_name>
   ```

5. **完整训练、DDP、长时间评测、视频录制**：只有用户明确要求，或它确实是验证当前问题所必需时才启动。

只把真正执行过的检查报告为 `PASS`。受 GPU、数据、权重、仿真环境等限制的项目明确写 `NOT VERIFIED`，不要为了让检查通过而改变算法语义或降低校验标准。

## 6. 实验与产物

- 不删除或覆盖与当前任务无关的 `robot_data/`、`experiments/`、checkpoint、视频、W&B 日志和预训练权重。
- 已有实验恢复时，始终尊重该实验保存的 resolved config 与 checkpoint。
- 不自动启动真机运动；Real robot motion 必须由用户明确要求。
- 除非任务明确要求，不主动进行大规模训练、批量评测或生成大量实验产物。

## 7. 文档职责

- `README.md`：给人看的稳定入口、常用命令和仓库地图。
- `AGENTS.md`：AI coding 的研究协作原则与工作方法。
- `CLAUDE.md`：Claude Code 的加载入口，只引用共享规则。
- `docs/`：较长的背景、机制和运维说明。

普通 Policy 实验不需要同步修改这些全局文档；只有稳定入口、仓库结构、通用工作流或长期约定发生变化时再更新。

## 8. 常用入口

- `dexmani_policy/configs/`：Hydra Policy configs
- `dexmani_policy/agents/`：Agent / encoders / backbones / decoders
- `dexmani_policy/datasets/`：datasets / replay buffer / preprocessing
- `dexmani_policy/training/`：training / checkpoint / resume / EMA
- `dexmani_policy/evaluation/`：offline evaluation
- `dexmani_policy/env_runner/`：simulation runners
- `dexmani_policy/deployment/`：saved-policy / Real runtime
- `dexmani_policy/train.py`、`train_ddp.py`、`smoke_test.py`：根入口
- `scripts/training/`、`scripts/eval/`、`scripts/remote/`、`scripts/utils/`：用户工作流

## 9. 项目级 Skills

任务匹配时优先使用短 workflow skill：

- `research-iterate`：把研究假设转成最小实现，并完成定向验证和下一步实验命令。
- `preflight-experiment`：昂贵训练或评测前检查 config、模态、shape、normalization、checkpoint 与 inference 设置。

Skill 是工作流提示，不是新的工程抽象层；不要为了 skill 再建立 supporting framework。

