# data/ — 项目数据目录

## 目录结构

```
data/
  pretrained/          # 手动管理的预训练模型权重
    uni3d/             # Uni3D ViT-tiny 点云编码器预训练权重
      model.safetensors  (~53MB)
```

## 预训练模型

### Uni3D（R3D 策略的 3D 点云编码器）

- **来源**: [eddie-cui/r3d-weights](https://huggingface.co/eddie-cui/r3d-weights) @ HuggingFace Hub
- **用途**: R3D Agent 的点云编码器初始化（`r3d.yaml` 中 `use_pretrained_weights: true`）
- **下载**: 运行 `bash scripts/utils/download_pretrained.sh`
- **配置路径**: `pretrained_weights_path: data/pretrained/uni3d`

### DINOv2-small（SAT / DQ-RISE 和默认 RGB 策略）

`bash scripts/utils/download_pretrained.sh` 同时将 `facebook/dinov2-small` 的 `config.json` 和 `model.safetensors` 下载到 HuggingFace Hub 缓存，默认位于 `~/.cache/huggingface/`，也遵循 `HF_HOME` 等既有环境设置。已有 Uni3D 权重时仍会准备 DINO 缓存；重复运行复用已有文件。

CLIP / SigLIP 或其他 DINO 型号由 `transformers` 按实际配置下载到同一 HF 缓存。首次离线训练前需先准备对应型号；完整策略 checkpoint 内嵌架构与参数，恢复无需这些初始文件。

## 训练数据

训练数据（Zarr 文件）存放于顶层 `robot_data/` 目录（不在此 `data/` 目录下）。
原因：训练数据体积大且常跨磁盘 symlink，独立目录便于管理。

格式: `robot_data/<task_name>.zarr`
