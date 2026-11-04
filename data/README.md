# data/ — 外部预训练权重

`data/pretrained/` 存放外部模型的初始化权重；示教数据位于 `robot_data/`，格式见[仿真数据集格式](../docs/仿真数据集格式.md)。

## 权重来源与用途

| 模型 | 来源 | 在项目中的用途 | 本地位置 |
| --- | --- | --- | --- |
| Uni3D | [R3D 预训练权重](https://huggingface.co/eddie-cui/r3d-weights) | R3D 的点云编码器初始化 | `pretrained/uni3d/` |
| DINOv2-small | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | DP、SAT、DQ-RISE、DexLatent 的默认 RGB 主干 | `pretrained/dinov2-small/` |
| DINOv3-S | [timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m) | 上述 RGB 策略的可选视觉主干 | `pretrained/dinov3-small/` |
| SigLIP Base patch16 / 224 | [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) | 上述 RGB 策略的可选视觉主干，仅使用视觉部分 | `pretrained/siglip-base-patch16-224/` |
| DexLatent | [官方预训练 checkpoint](https://github.com/EmptyBlueBox/DexLatent/blob/502750df34b9fa6a1b56a66481586887ec722f9c/Checkpoints/20260311_225425/checkpoint_epoch_1000.pt) | DexLatent 的冻结 XHand 手部编码器与解码器 | `pretrained/dexlatent/` |

表中路径均相对于 `data/`。DINOv2-small 是默认 RGB 来源，DINOv3-S 与 SigLIP 是可选替代；DexLatent 策略同时使用 RGB 主干与手部先验。

## 项目内预训练产物

AFRO、CordViP 的观测编码器和 DQ-RISE 的手部码本由本项目训练产生，保留在各自的预训练实验目录，不属于外部权重目录：

| 产物 | 下游作用 | 机制说明 |
| --- | --- | --- |
| AFRO / CordViP 编码器及输入归一化统计 | 初始化对应策略的观测编码器 | [观测预训练](../docs/观测预训练.md) |
| DQ-RISE 有序手姿码本及手部尺度 | 定义手部连续索引与物理手姿的映射 | [手部动作表示](../docs/手部动作表示.md) |

下载 RGB 权重不会生成 DQ-RISE 码本。

## 初始化与恢复

外部权重和项目预训练产物用于新训练的初始化。完整策略 checkpoint 保存所需模型、先验、码表及归一化统计；从中续训或推理无需重新访问这些初始化文件，续训仍需原训练数据。

整体关系见[项目架构](../docs/项目架构.md)，服务器上的权重同步见[SSH 服务器训练部署](../docs/SSH服务器训练部署.md)。
