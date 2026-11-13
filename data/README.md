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

可选 DINOv2 完整配置保留在 `dp.yaml` 末尾注释中。切换时整体替换 `rgb_backbone_config`，避免 ResNet 的 `model_name/norm_mode` 被传给 ViT，并同步替换 `initialization.rgb` 和图像处理尺寸。已有 DINO checkpoint 仍按保存的配方恢复，无需权重转换。

ResNet 也可设置 `initialization.rgb=DEFAULT` 使用 torchvision 预训练权重；支持 `freeze`/`full`，不支持 LoRA。`pretrained + freeze + group_norm` 可用于消融；BN→GN 保留卷积权重，但不保留 BN 统计量和 affine 参数，转换后的表征适用性由研究者判断。恢复和部署使用 checkpoint，无需再次下载。

## 项目内预训练产物

以下产物由本项目训练产生，不属于外部模型权重：

| 产物 | 下游作用 |
| --- | --- |
| AFRO / CordViP 观测编码器 | 将场景动力学或接触与协调表征迁移到策略，见[观测预训练](../docs/观测预训练.md) |
| DQ-RISE 手姿码本 | 将生成的手姿索引映射为关节目标，见[手部动作表示](../docs/手部动作表示.md) |

整体关系见[项目架构](../docs/项目架构.md)，服务器上的权重同步见[SSH 服务器训练部署](../docs/SSH服务器训练部署.md)。
