# data/ — 外部预训练权重

## 目录结构

```
data/
  pretrained/          # 下载脚本准备的本地预训练权重
    uni3d/             # Uni3D ViT-tiny 点云编码器预训练权重
      model.safetensors  (~53MB)
    dinov2-small/      # 默认 RGB 主干，Transformers 格式
      config.json
      model.safetensors  (~84MiB)
    dinov3-small/      # 可选 RGB 主干，timm 格式
      config.json
      model.safetensors  (~82MiB)
      LICENSE.md
    siglip-base-patch16-224/  # 可选 RGB 主干，官方完整图文权重
      config.json
      preprocessor_config.json
      README.md
      model.safetensors  (~776MiB)
    dexlatent/         # 官方四手预训练 checkpoint，策略仅加载冻结 XHand 分支
      checkpoint_epoch_1000.pt  (~2.16MB)
```

## 预训练模型

`bash scripts/utils/download_pretrained.sh [方法]` 在既有 `policy` 环境中将权重下载到 `data/pretrained/`，重复运行复用已有文件。HF 文件直接使用 Hub 的本地目录下载与续传，完成后落盘；本地模型加载不依赖全局 HF 缓存，权重不纳入 Git：

| 参数 | 准备内容 |
| --- | --- |
| 无参数或 `all` | Uni3D、DINOv2-small、DexLatent |
| `dexlatent` | DINOv2-small、DexLatent |
| `sat`、`dqrise`、`dp` | DINOv2-small |
| `r3d` | Uni3D |
| `dinov3` | 可选 DINOv3-S |
| `siglip` | 可选 SigLIP Base patch16 / 224 |

DQ-RISE 的码本仍由项目原有预训练流程生成，下载 DINO 不会准备码本。自定义 RGB 型号沿用配置中的初始化来源。

自产 encoder 和码本直接引用预训练 run 中的文件，见 [观测预训练](../docs/观测预训练.md)和[手部动作表示](../docs/手部动作表示.md)，不存入本目录。

### Uni3D（R3D 策略的 3D 点云编码器）

- **来源**: [eddie-cui/r3d-weights](https://huggingface.co/eddie-cui/r3d-weights) @ HuggingFace Hub
- **用途**: R3D Agent 的 fresh 点云编码器初始化；设 `initialization.uni3d=null` 从头训练
- **下载**: 运行 `bash scripts/utils/download_pretrained.sh r3d`
- **配置路径**: `initialization.uni3d: data/pretrained/uni3d`

### DINOv2-small（SAT / DQ-RISE / DexLatent 和默认 RGB 策略）

`bash scripts/utils/download_pretrained.sh sat` 将 `facebook/dinov2-small` 的 `config.json` 和 `model.safetensors` 下载到 `data/pretrained/dinov2-small/`。DP、SAT、DQ-RISE、DexLatent 的默认 `initialization.rgb` 均指向此本地目录；`dexlatent` 和 `all` 也会准备这些文件。

文本 CLIP 组件继续保留，其来源由独立文本编码器配置，当前策略未使用。可选 SigLIP 使用下述 Google Base 型号；其他 RGB 型号通过 fresh 的 `initialization.rgb` 指定本地目录。模型构造只使用已解析的 architecture；完整策略 checkpoint 内嵌架构与参数，恢复无需这些初始文件或网络。

### DINOv3-S（可选 RGB 主干）

`bash scripts/utils/download_pretrained.sh dinov3` 将 `timm/vit_small_patch16_dinov3.lvd1689m` 的权重、配置和许可证下载到 `data/pretrained/dinov3-small/`。新训练同时设置 `agent.rgb_backbone_name=dinov3` 和 `initialization.rgb=data/pretrained/dinov3-small`。模型由 timm 的 `vit_small_patch16_dinov3` 构造，不读取 Transformers config。`all` 仍只准备默认配方所需权重。

### SigLIP Base patch16 / 224（可选 RGB 主干）

`bash scripts/utils/download_pretrained.sh siglip` 将 [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) 的配置、预处理说明、模型卡和完整 FP32 权重下载到 `data/pretrained/siglip-base-patch16-224/`。新训练设置 `agent.rgb_backbone_name=siglip` 和 `initialization.rgb=data/pretrained/siglip-base-patch16-224`。项目通过 Transformers 仅加载视觉主干，默认 `avg` 不构造 pooling head；224×224 输入输出 14×14 个 768D patch tokens，无 CLS/register。沿用项目预处理，mean/std 均为 0.5。默认配方与 `all` 的下载内容不变。

### DexLatent（冻结 XHand 手部先验）

- **固定来源**：[DexLatent checkpoint_epoch_1000.pt](https://raw.githubusercontent.com/EmptyBlueBox/DexLatent/502750df34b9fa6a1b56a66481586887ec722f9c/Checkpoints/20260311_225425/checkpoint_epoch_1000.pt)，保留官方四手原始文件。
- **下载**：`bash scripts/utils/download_pretrained.sh dexlatent`；已有文件直接复用，缺失时在目标目录所在文件系统的临时目录下载，成功后原子替换。
- **路径**：`initialization.dexlatent: data/pretrained/dexlatent/checkpoint_epoch_1000.pt`，文件不纳入 Git。
- **用途**：仅 fresh 初始化读取 XHand 参数及安装官方 URDF 手部尺度；完整策略 raw/EMA checkpoint 内嵌 E/D、action affine、训练集 latent affine 和架构，续训/推理不读取原始权重。

新训练默认 FP32，`training.use_bfloat16=true` 可选启用 BF16 autocast，续训沿用保存的配方；正式推理严格加载完整 checkpoint 后统一转为 FP32，不重新下载或替换 RGB 权重，也不修改磁盘上的 checkpoint。

远程复用 `bash scripts/remote/sync_data.sh --host <host> --local data/pretrained --remote /data/project/data/pretrained`，训练启动沿用 `--data-root /data/project`。同步这些本地目录后无需另行复制 HF 缓存；源码同步不携带 `data/` 权重。

示教与 CordViP 加工数据位于 `robot_data/`，格式见[数据集文档](../docs/仿真数据集格式.md)。仓库 `data/` 专用于外部权重；服务器 `/data/project/experiments/` 等磁盘路径不受此约定限制。
