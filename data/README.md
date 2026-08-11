# data/ — 项目数据目录

## 目录结构

```
data/
  pretrained/          # 手动管理的预训练模型权重
    uni3d/             # Uni3D ViT-tiny 点云编码器预训练权重
      model.safetensors  (~53MB)
    galr/              # GaLR 手部自编码器预训练权重 (OPFA Stage 1)
      epoch-129.pth.tar  (~34MB)
      best.pt → epoch-129.pth.tar
```

## 预训练模型

### Uni3D（R3D 策略的 3D 点云编码器）

- **来源**: [eddie-cui/r3d-weights](https://huggingface.co/eddie-cui/r3d-weights) @ HuggingFace Hub
- **用途**: R3D Agent 的点云编码器初始化（`r3d.yaml` 中 `use_pretrained_weights: true`）
- **下载**: 运行 `bash scripts/utils/download_pretrained.sh`
- **配置路径**: `pretrained_weights_path: data/pretrained/uni3d`

### GaLR（OPFA 策略的手部自编码器）

- **来源**: [mujc2021/one-policy-fits-all](https://huggingface.co/mujc2021/one-policy-fits-all) @ HuggingFace Hub
- **用途**: OPFA 推理时将 1024-d hand latent 解码为 12-d XHand 关节角度（`opfa.yaml` 中 `galr_ckpt_path`）
- **下载**:
  ```bash
  huggingface-cli download mujc2021/one-policy-fits-all epoch-129.pth.tar --local-dir data/pretrained/galr/
  cd data/pretrained/galr/ && ln -s epoch-129.pth.tar best.pt
  ```
- **配置路径**: `galr_ckpt_path: data/pretrained/galr/best.pt`

### 其他预训练模型

DINOv2 / CLIP / SigLIP / T5 等视觉/文本骨干网络由 `transformers` 库自动从 HuggingFace Hub
下载缓存至 `~/.cache/huggingface/`，无需手动管理。

## 训练数据

训练数据（Zarr 文件）存放于顶层 `robot_data/` 目录（不在此 `data/` 目录下）。
原因：训练数据体积大且常跨磁盘 symlink，独立目录便于管理。

格式: `robot_data/sim/<task_name>.zarr`
