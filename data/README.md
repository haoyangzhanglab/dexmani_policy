# 预训练权重

`data/pretrained/` 存放外部预训练权重，为策略提供视觉表征或手部先验。示教数据单独存放于 `robot_data/`，见[仿真数据集格式](../docs/仿真数据集格式.md)。

## 外部模型

| 模型与来源 | 用途 |
| --- | --- |
| [Uni3D（R3D 权重）](https://huggingface.co/eddie-cui/r3d-weights) | 初始化 R3D 的点云编码器 |
| [DINOv2-small](https://huggingface.co/facebook/dinov2-small) | DP 的可选 RGB 主干 |
| [DINOv3-S](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m) | SAT、DQ-RISE、DexLatent 的默认 RGB 主干 |
| [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) | RGB 策略的可选视觉主干 |
| [DexLatent](https://github.com/EmptyBlueBox/DexLatent/blob/502750df34b9fa6a1b56a66481586887ec722f9c/Checkpoints/20260311_225425/checkpoint_epoch_1000.pt) | 提供冻结的 XHand 手部编码器与解码器 |

视觉主干提供观测表征，手部先验负责动作表示；DexLatent 策略同时使用两者。

在仓库根目录执行 `bash scripts/utils/download_pretrained.sh <下载项>`。`sat`、`dqrise` 下载 DINOv3；`dexlatent` 额外下载手部先验；`r3d` 下载 Uni3D。`all` 下载这些默认所需权重；可选 DINOv2、SigLIP 分别使用 `dinov2`、`siglip` 单独下载。默认 DP 无需下载权重。

DP 默认使用随机初始化的 ResNet18＋GroupNorm（`initialization.rgb=null`、`tune_mode=full`），联合训练。图像预处理为 84×84，裁剪与编码器 `image_size` 均为 76×76，避免 ImageProcessor 再放大到 224。

可选 DINOv2 完整配置保留在 `dp.yaml` 末尾注释中。切换时整体替换 `rgb_backbone_config`，避免 ResNet 的 `model_name/norm_mode` 被传给 ViT，并同步替换 `initialization.rgb` 和图像处理尺寸。

ResNet 可通过 `initialization.rgb=DEFAULT` 使用 torchvision 预训练权重，支持 `freeze` 和 `full`。`pretrained + freeze + group_norm` 可用于消融。BN→GN 保留卷积权重与 γ、β，GroupNorm 使用当前样本的分组统计，因此与使用 BN 运行统计的表征存在差异。恢复和部署使用完整 checkpoint。

R3D 发布配方采用 `normalization.point_cloud=limits`，与 Uni3D 输入前的固定区间裁剪配套。该约束也适用于从 checkpoint 恢复。

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

以下产物由本项目训练产生，不属于外部模型权重：

| 产物 | 下游作用 |
| --- | --- |
| AFRO / CordViP 观测编码器 | 将场景动力学或接触与协调表征迁移到策略，见[观测预训练](../docs/观测预训练.md) |
| DQ-RISE 手姿码本 | 将生成的手姿索引映射为关节目标，见[手部动作表示](../docs/手部动作表示.md) |

整体关系见[项目架构](../docs/项目架构.md)，服务器上的权重同步见[SSH 服务器训练部署](../docs/SSH服务器训练部署.md)。
