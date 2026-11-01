# data/ — 项目数据目录

## 目录结构

```
data/
  pretrained/          # 手动管理的预训练模型权重
    uni3d/             # Uni3D ViT-tiny 点云编码器预训练权重
      model.safetensors  (~53MB)
    dexlatent/         # 官方四手预训练 checkpoint，策略仅加载冻结 XHand 分支
      checkpoint_epoch_1000.pt  (~2.16MB)
```

## 预训练模型

### Uni3D（R3D 策略的 3D 点云编码器）

- **来源**: [eddie-cui/r3d-weights](https://huggingface.co/eddie-cui/r3d-weights) @ HuggingFace Hub
- **用途**: R3D Agent 的 fresh 点云编码器初始化；设 `initialization.uni3d=null` 从头训练
- **下载**: 运行 `bash scripts/utils/download_pretrained.sh`
- **配置路径**: `initialization.uni3d: data/pretrained/uni3d`

### DINOv2-small（SAT / DQ-RISE / DexLatent 和默认 RGB 策略）

`bash scripts/utils/download_pretrained.sh` 同时将 `facebook/dinov2-small` 的 `config.json` 和 `model.safetensors` 下载到 HuggingFace Hub 缓存，默认位于 `~/.cache/huggingface/`，也遵循 `HF_HOME` 等既有环境设置。已有 Uni3D 权重时仍会准备 DINO 缓存；重复运行复用已有文件。

文本 CLIP 组件继续保留，其来源由独立文本编码器配置。RGB 的 SigLIP 或其他 DINO 型号通过 fresh 的 `initialization.rgb` 准备；首次离线训练前需先准备对应型号。模型构造只使用已解析的 architecture；完整策略 checkpoint 内嵌架构与参数，恢复无需这些初始文件或网络。

### DexLatent（冻结 XHand 手部先验）

- **固定来源**：[DexLatent checkpoint_epoch_1000.pt](https://raw.githubusercontent.com/EmptyBlueBox/DexLatent/502750df34b9fa6a1b56a66481586887ec722f9c/Checkpoints/20260311_225425/checkpoint_epoch_1000.pt)，保留官方四手原始文件。
- **下载**：同一 `bash scripts/utils/download_pretrained.sh` 入口；已有文件直接复用，下载先写临时目录，成功后原子替换。
- **路径**：`initialization.dexlatent: data/pretrained/dexlatent/checkpoint_epoch_1000.pt`，文件不纳入 Git。
- **用途**：仅 fresh 初始化读取 XHand 参数及安装官方 URDF 手部尺度；完整策略 raw/EMA checkpoint 内嵌 E/D、action affine、训练集 latent affine 和架构，续训/推理不读取原始权重。

远程复用 `bash scripts/remote/sync_data.sh --host <host> --local data/pretrained/dexlatent --remote /data/project/data/pretrained/dexlatent`，训练启动沿用 `--data-root /data/project`。DINO HF 缓存需在目标环境提前准备；源码同步不携带 `data/` 权重。

## 训练数据

训练数据（Zarr 文件）存放于顶层 `robot_data/` 目录（不在此 `data/` 目录下）。
原因：训练数据体积大且常跨磁盘 symlink，独立目录便于管理。

格式: `robot_data/<task_name>.zarr`

训练使用的数据版本不就地修改。远程同步显式指定路径，例如 `bash scripts/remote/sync_data.sh --host <host> --local robot_data --remote /data/project/robot_data`；脚本保留目标独有文件。启动时的 `--data-root /data/project` 为源码副本提供 `robot_data/`、`data/` 和 `pretrained_models/` 链接，实验目录另外由 `--run` 指定，详见 [SSH 文档](../docs/SSH服务器训练部署.md)。
