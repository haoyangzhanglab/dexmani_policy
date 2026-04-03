from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


GlobalTokenType = Literal["cls", "avg", "pooler"]
OutputType = Literal["global", "patch"]
TuneMode = Literal["freeze", "lora"]
DINOFamily = Literal["dinov2", "dinov3"]


def infer_dino_family(model_name: str) -> DINOFamily:
    name = model_name.lower()
    if "dinov3" in name:
        return "dinov3"
    if "dinov2" in name:
        return "dinov2"
    raise ValueError(f"Cannot infer DINO family from model name: {model_name}")


def get_register_token_num(dino_family: DINOFamily, config) -> int:
    if dino_family == "dinov2":
        return 0
    return int(getattr(config, "num_register_tokens", 0))


def get_lora_target_modules(dino_family: DINOFamily):
    if dino_family == "dinov2":
        return ["projection", "query", "key", "value", "dense", "fc1", "fc2"]
    return ["patch_embeddings", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def build_lora_backbone(
    backbone: nn.Module,
    dino_family: DINOFamily,
    lora_rank: int,
    lora_dropout: float,
) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    backbone.requires_grad_(False)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=get_lora_target_modules(dino_family),
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=True,
    )
    backbone = get_peft_model(backbone, config)

    for name, param in backbone.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()

    return backbone


def get_patch_grid_size(image_hw: Tuple[int, int], patch_size: int) -> Tuple[int, int]:
    image_h, image_w = image_hw
    if image_h % patch_size != 0 or image_w % patch_size != 0:
        raise ValueError(
            f"Input size {(image_h, image_w)} must be divisible by patch size {patch_size}."
        )
    return image_h // patch_size, image_w // patch_size


def reshape_patch_tokens_to_feature_map(
    patch_tokens: torch.Tensor,
    patch_grid_size: Tuple[int, int],
) -> torch.Tensor:
    batch_size, token_num, channel_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid_size
    if token_num != grid_h * grid_w:
        raise ValueError(
            f"Patch token number {token_num} does not match patch grid size {patch_grid_size}."
        )
    return patch_tokens.view(batch_size, grid_h, grid_w, channel_dim).permute(0, 3, 1, 2).contiguous()


def get_global_token(
    outputs,
    patch_tokens: torch.Tensor,
    global_token_type: GlobalTokenType,
    proj: nn.Module,
) -> torch.Tensor:
    if global_token_type == "cls":
        return proj(outputs.last_hidden_state[:, 0])
    if global_token_type == "avg":
        return patch_tokens.mean(dim=1)
    if global_token_type == "pooler":
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is None:
            return proj(outputs.last_hidden_state[:, 0])
        return proj(pooler_output)
    raise ValueError(f"Unsupported global_token_type: {global_token_type}")


def compute_patch_coords(
    image_world_coords: torch.Tensor,
    image_valid_mask: torch.Tensor,
    patch_grid_size: Tuple[int, int],
    flatten: bool = True,
) -> Dict[str, torch.Tensor]:
    if image_world_coords.ndim != 4 or image_world_coords.shape[1] != 3:
        raise ValueError("image_world_coords should have shape [B, 3, H, W].")
    if image_valid_mask.ndim != 4 or image_valid_mask.shape[1] != 1:
        raise ValueError("image_valid_mask should have shape [B, 1, H, W].")

    grid_h, grid_w = patch_grid_size
    image_h, image_w = image_world_coords.shape[-2:]
    if image_h % grid_h != 0 or image_w % grid_w != 0:
        raise ValueError(
            f"Image size {(image_h, image_w)} is not divisible by patch grid size {patch_grid_size}."
        )

    kernel_h = image_h // grid_h
    kernel_w = image_w // grid_w

    coords = image_world_coords.float()
    mask = image_valid_mask.float()

    coords_num = F.avg_pool2d(coords * mask, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
    coords_den = F.avg_pool2d(mask, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
    patch_coords = coords_num / coords_den.clamp_min(1e-6)
    patch_valid_mask = coords_den > 1e-6

    if flatten:
        patch_coords = patch_coords.flatten(2).transpose(1, 2).contiguous()
        patch_valid_mask = patch_valid_mask.flatten(2).transpose(1, 2).contiguous()

    return {
        "patch_coords": patch_coords,
        "patch_valid_mask": patch_valid_mask,
    }


class DINOEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        tune_mode: TuneMode = "freeze",
        output_type: OutputType = "patch",
        global_token_type: GlobalTokenType = "cls",
        out_dim: Optional[int] = None,
        lora_rank: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.dino_family = infer_dino_family(model_name)
        self.output_type = output_type
        self.global_token_type = global_token_type

        backbone = AutoModel.from_pretrained(model_name)
        self.patch_size = int(backbone.config.patch_size)
        self.hidden_dim = int(backbone.config.hidden_size)
        self.num_register_tokens = get_register_token_num(self.dino_family, backbone.config)
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        if tune_mode == "freeze":
            backbone.requires_grad_(False)
        elif tune_mode == "lora":
            backbone = build_lora_backbone(
                backbone=backbone,
                dino_family=self.dino_family,
                lora_rank=lora_rank,
                lora_dropout=lora_dropout,
            )
        else:
            raise ValueError(f"Unsupported tune_mode: {tune_mode}")

        self.backbone = backbone
        self.proj = nn.Identity() if self.out_dim == self.hidden_dim else nn.Linear(self.hidden_dim, self.out_dim)

    def forward(
        self,
        images: torch.Tensor,
        output_type: Optional[OutputType] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images should have shape [B, 3, H, W].")

        output_type = self.output_type if output_type is None else output_type
        patch_grid_size = get_patch_grid_size(images.shape[-2:], self.patch_size)

        outputs = self.backbone(pixel_values=images, return_dict=True)
        patch_tokens = outputs.last_hidden_state[:, 1 + self.num_register_tokens :]
        patch_tokens = self.proj(patch_tokens)

        if output_type == "global":
            return {
                "global_token": get_global_token(
                    outputs=outputs,
                    patch_tokens=patch_tokens,
                    global_token_type=self.global_token_type,
                    proj=self.proj,
                )
            }

        if output_type == "patch":
            return {
                "patch_tokens": patch_tokens,
                "feature_map": reshape_patch_tokens_to_feature_map(patch_tokens, patch_grid_size),
            }

        raise ValueError(f"Unsupported output_type: {output_type}")

    def get_patch_coords(
        self,
        image_world_coords: torch.Tensor,
        image_valid_mask: torch.Tensor,
        flatten: bool = True,
    ) -> Dict[str, torch.Tensor]:
        patch_grid_size = get_patch_grid_size(image_world_coords.shape[-2:], self.patch_size)
        return compute_patch_coords(
            image_world_coords=image_world_coords,
            image_valid_mask=image_valid_mask,
            patch_grid_size=patch_grid_size,
            flatten=flatten,
        )


DINOv2Encoder = DINOEncoder


def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov2-small"

    print(f"Loading {model_name} on {device} ...")
    encoder = DINOEncoder(
        model_name=model_name,
        tune_mode="freeze",
        output_type="patch",
        global_token_type="cls",
    ).to(device)
    encoder.eval()

    images = torch.randn(2, 3, 224, 224, device=device)
    image_world_coords = torch.randn(2, 3, 224, 224, device=device)
    image_valid_mask = torch.ones(2, 1, 224, 224, device=device)

    with torch.no_grad():
        patch_out = encoder(images, output_type="patch")
        global_out = encoder(images, output_type="global")
        coord_out = encoder.get_patch_coords(image_world_coords, image_valid_mask, flatten=True)

    print("patch_tokens:", tuple(patch_out["patch_tokens"].shape))
    print("feature_map :", tuple(patch_out["feature_map"].shape))
    print("global_token:", tuple(global_out["global_token"].shape))
    print("patch_coords:", tuple(coord_out["patch_coords"].shape))
    print("patch_valid_mask:", tuple(coord_out["patch_valid_mask"].shape))


if __name__ == "__main__":
    try:
        example()
    except Exception as error:
        print("example() failed")
        print(error)