---
name: preflight-experiment
description: 在消耗 GPU 时间前，对 DexMani 的训练、续训或评测配置做一次简洁的高价值预检查。
---

# 实验预检查

适用于准备启动较贵的训练 / 评测，或当前命令、config、checkpoint 看起来可能存在不一致时。

## 检查内容

1. 解析目标 config 与 overrides，确认实际 `agent._target_`。
2. 核对 observation keys / modalities、history / window / horizon、action dimension 和明显的 shape contract。
3. 核对 normalization，以及 observation encoder → backbone → action decoder 的接口是否一致。
4. 核对容易静默漂移的训练 / 推理设置，尤其是 checkpoint、raw / EMA 选择和 `inference_steps`。
5. 核对 dataset / task 路径，以及请求的模态是否真实存在。
6. 对 resume / evaluation，确认实验的 resolved config 与 checkpoint 属于同一语义，并且恢复方式有效。

## 验证

适用时优先运行：

```bash
python dexmani_policy/smoke_test.py --config-only <config_name>
```

只有当完整 smoke 能实际覆盖当前风险、且所需 GPU / 数据环境可用时，才继续运行 full smoke。

预检查本身不启动正式训练或长时间评测，除非用户明确要求。

## 输出

结果保持简短，至少包含：

- `通过`：已经实际验证，可以进入下一步；
- `阻塞`：需要先修复的具体问题；
- `未验证`：因 GPU / 数据 / 权重 / 环境限制无法确认的项目；
- 建议执行的完整命令。

