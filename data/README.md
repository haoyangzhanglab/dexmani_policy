# data/ — 项目数据目录

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

### 项目内预训练产物

AFRO/CordViP 的 `encoder.pt` 保存在线 encoder、输入 affine 和归一化模式，identity 也随模块继承；`resume.pt` 另存完整预训练恢复状态。可从完整 `resume.pt` 独立导出 encoder，见 [观测预训练](../docs/观测预训练.md)。完整策略 checkpoint 内嵌 encoder 和最终 normalizer，恢复无需原 encoder 文件。

DQ-RISE 的 `codebook.npz` 只需 `sorted_hand_poses [K,12]`、`hand_normalizer_scale [12]`、`hand_normalizer_offset [12]`，姿态为 normalized FP32 坐标。运行时不依赖预训练分组或数据划分。默认 VQ 从 last 导出，显式 val 时从 lookup_mse best 导出；任意完整 VQ checkpoint 均可重导出，命令见 [手部动作表示](../docs/手部动作表示.md)。fresh 策略保存实际 `num_codes`，完整策略恢复无需原 NPZ。

CordViP 的 `.cordvip.zarr` 自包含 hand/object XYZ、joint_state、contact_map 和存在时的源 action，数值数组固定 FP32，episode 边界为 int64；根 attrs 只保留已有的 dt 与 hand_pointcloud_config。无 action 可用于预训练，主策略仍需 action。完整 Policy 的 normalizer 形状从实际权重恢复，不另存 shape 配置；部署事实不依赖来源标签。

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

## 训练数据

训练数据（本地 Zarr 2 目录，不支持 ZIP、远程 store 或后端插件）存放于顶层 `robot_data/` 目录（不在此 `data/` 目录下）。
原因：训练数据体积大且常跨磁盘 symlink，独立目录便于管理。

格式: `robot_data/<task_name>.zarr`

主策略和独立预训练均默认 `dataset.seed=42`、`max_train_episodes=null`、`val_episodes=0`。预留 V 条后，训练数量为 `N−V` 或 `min(M,N−V)`；同一排列依次选择 train、val、unused。150 条、上限 100、留出 20 得到 100/20/30；80 条、上限 100、留出 10 得到 70/10/0。主策略从原先最多 100 条改为全部数据属于训练配方变化，100 条预算需显式设置；resume 沿用保存的参数。新统计只使用本阶段 train 的真实源帧，训练不读取来源身份或自动按 trial 分组。各阶段数据参数独立，主策略 val 仅留出，未运行离线验证。

训练使用的数据版本不就地修改。远程同步显式指定路径，例如 `bash scripts/remote/sync_data.sh --host <host> --local robot_data --remote /data/project/robot_data`；脚本保留目标独有文件。启动时的 `--data-root /data/project` 为源码副本提供 `robot_data/`、`data/` 和 `pretrained_models/` 链接，实验目录另外由 `--run` 指定，详见 [SSH 文档](../docs/SSH服务器训练部署.md)。
