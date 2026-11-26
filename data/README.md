# 预训练权重

`data/pretrained/` 存放外部视觉模型和手部先验，`robot_data/` 存放示教数据。 [示教数据格式](../docs/仿真数据集格式.md)

## 外部模型

在仓库根目录运行下载脚本。例如，InteractionFlow 默认使用的权重可用 `bash scripts/utils/download_pretrained.sh dinov2` 下载到 `data/pretrained/dinov2-small/`。

| 模型与来源 | 下载项 | 用途 |
| --- | --- | --- |
| [DINOv2-small](https://huggingface.co/facebook/dinov2-small) | `dinov2`（也可用 `sat`、`dqrise`） | InteractionFlow、SAT、DQ-RISE 和 DexLatent 的默认 RGB 主干 |
| [Uni3D](https://huggingface.co/eddie-cui/r3d-weights) | `r3d` | R3D 的点云编码器 |
| [DexLatent](https://github.com/EmptyBlueBox/DexLatent/blob/502750df34b9fa6a1b56a66481586887ec722f9c/Checkpoints/20260311_225425/checkpoint_epoch_1000.pt) | `dexlatent` | 冻结的 XHand 手部编码器与解码器，同时下载 DINOv2-small |
| [DINOv3-S](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m) | `dinov3` | 可选 RGB 主干 |
| [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) | `siglip` | 可选 RGB 主干 |

`all` 下载默认配方使用的 DINOv2-small、Uni3D 和 DexLatent 手部先验。切换 RGB 主干时，在配方中同时指定 `agent.rgb_backbone_name` 和对应的 `initialization.rgb`。DP 默认从头训练 ResNet18，配方中附有 DINOv2 的完整示例。 [DP 配方](../dexmani_policy/configs/dp.yaml)

新训练使用 `agent.rgb_backbone_config.tune_mode=freeze` 或 `lora` 时，必须提供 `initialization.rgb`，否则会冻结随机初始化的主干，训练入口将报错。此约束同样适用于显式冻结的 ResNet。若要从零训练，使用 `tune_mode=full` 和 `initialization.rgb=null`；DINOv2、SigLIP 还需显式提供 `rgb_backbone_config.architecture`。通过 `initialization.policy` 热启动或恢复 checkpoint 时，RGB 权重来自 checkpoint，无需外部 RGB 权重来源。

## 上游源码与资产条款

| 组件 | 来源与核验状态 |
| --- | --- |
| AFRO 动力学预训练 | Qiwei Liang 的实现，MIT。 [上游 LICENSE](https://github.com/KolaKivy/AFRO/blob/283024a49e2c14bc030debf15383a4e7ee73836d/LICENSE) |
| CordViP 接触与协调 | Aureleo 的实现，MIT。 [上游 LICENSE](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/LICENSE) |
| DQ-RISE VQ-VAE 与码本 | DQ-RISE 上游源码采用 CC BY-NC-SA 4.0。 [上游 LICENSE](https://github.com/rise-policy/DQ-RISE/blob/2889d27fce823288e8dd12ec6e91506b52ed086a/LICENSE) |
| DexLatent XHand 先验 | 本地改编保留 Yutong Liang 的 MIT 声明。 [本地声明](../dexmani_policy/agents/hand_prior/dexlatent.py) · [上游版本](https://github.com/EmptyBlueBox/DexLatent/tree/502750df34b9fa6a1b56a66481586887ec722f9c) |
| ODE-Policy | 上游源码采用 Apache-2.0。 [上游 LICENSE](https://github.com/NTUMARS/ODE-POLICY/blob/bcea95a299dc3fa8b2a571603a5e8b50e1c449e9/LICENSE) |
| R3D / Uni3D | Shenrui Wu 的 R3D 上游源码采用 MIT。Uni3D 发布权重的单独许可待确认。 [上游 LICENSE](https://github.com/Wushr-Lance/R3D-Policy/blob/e637c0148376ddc4b5e667fa8f8e108cb8ff7a85/LICENSE) · [权重页](https://huggingface.co/eddie-cui/r3d-weights) |
| AdapTac 力引导 | 上游包元数据写 MIT，签入的 LICENSE 写 CC BY-NC-SA 4.0，适用条款待确认。 [包元数据](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/pyproject.toml) · [上游 LICENSE](https://github.com/kingchou007/adaptac-dex/blob/60e2ef1b7b0a29f5d29f5d755e8fe6d048f69fa5/LICENSE) |
| DINOv3 权重 | 适用发布资产随附的 DINOv3 条款。 [权重 LICENSE](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/blob/3bf4720a82ec2066db88137180ff1f83a675cef0/LICENSE.md) |

## 项目内预训练产物

- AFRO 和 CordViP 从示教训练 `encoder.pt`，将观测表征迁移到下游策略。 [观测预训练](../docs/观测预训练.md)
- DQ-RISE 从手部关节目标训练 `codebook.npz`，用于将手姿索引还原为关节目标。 [手部动作表示](../docs/手部动作表示.md)

在远端使用这些权重时，按 SSH 指南同步所需文件。 [SSH 服务器训练部署](../docs/SSH服务器训练部署.md)
