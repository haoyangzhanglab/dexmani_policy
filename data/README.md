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
- **下载**: 运行 `bash scripts/download_pretrained.sh`
- **配置路径**: `pretrained_weights_path: data/pretrained/uni3d`

### 其他预训练模型

DINOv2 / CLIP / SigLIP / T5 等视觉/文本骨干网络由 `transformers` 库自动从 HuggingFace Hub
下载缓存至 `~/.cache/huggingface/`，无需手动管理。

## 训练数据

训练数据（Zarr 文件）存放于顶层 `robot_data/` 目录（不在此 `data/` 目录下）。
原因：训练数据体积大且常跨磁盘 symlink，独立目录便于管理。

格式: `robot_data/sim/<task_name>.zarr`
