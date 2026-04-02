import math
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        tune_mode: str = "freeze",
        global_token_type: str = "model",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.global_token_type = global_token_type

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device)
        self.patch_size = self.backbone.config.patch_size
        self.hidden_dim = self.backbone.config.hidden_size
        self.feature_dim = self.backbone.config.projection_dim

        self.setTuneMode(tune_mode, lora_rank, lora_alpha, lora_dropout)

    def setTuneMode(self, tune_mode: str, lora_rank: int, lora_alpha: int, lora_dropout: float):
        if tune_mode == "freeze":
            for p in self.backbone.parameters():
                p.requires_grad = False
            return

        if tune_mode == "last_block":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.vision_model.encoder.layers[-1].parameters():
                p.requires_grad = True
            for p in self.backbone.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            for p in self.backbone.visual_projection.parameters():
                p.requires_grad = True
            return

        if tune_mode == "lora":
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
            )
            self.backbone = get_peft_model(self.backbone, config)
            return

        raise ValueError(f"Unsupported tune_mode: {tune_mode}")

    def flattenRgb(self, rgb: torch.Tensor):
        leading_shape = rgb.shape[:-3]
        flat_rgb = rgb.reshape(-1, *rgb.shape[-3:])
        return flat_rgb, leading_shape

    def preprocessRgb(self, rgb: torch.Tensor):
        flat_rgb, leading_shape = self.flattenRgb(rgb)
        flat_rgb = flat_rgb.to(self.device, non_blocking=True)
        inputs = self.processor(images=flat_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        return pixel_values, leading_shape

    def buildFeatureMap(self, patch_tokens: torch.Tensor, height: int, width: int):
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if patch_tokens.shape[1] != grid_h * grid_w:
            grid_h = int(math.sqrt(patch_tokens.shape[1]))
            grid_w = patch_tokens.shape[1] // grid_h
        feature_map = patch_tokens.reshape(patch_tokens.shape[0], grid_h, grid_w, patch_tokens.shape[-1])
        return feature_map.permute(0, 3, 1, 2).contiguous()

    def getGlobalToken(self, outputs, patch_tokens: torch.Tensor):
        if self.global_token_type == "avg":
            return patch_tokens.mean(dim=1)
        return outputs.image_embeds

    def forward(self, rgb: torch.Tensor):
        pixel_values, leading_shape = self.preprocessRgb(rgb)
        outputs = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True, return_dict=True)

        patch_tokens = outputs.last_hidden_state[:, 1:]
        feature_map = self.buildFeatureMap(patch_tokens, pixel_values.shape[-2], pixel_values.shape[-1])
        global_token = self.getGlobalToken(outputs, patch_tokens)

        patch_tokens = patch_tokens.reshape(*leading_shape, patch_tokens.shape[1], patch_tokens.shape[2])
        feature_map = feature_map.reshape(*leading_shape, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
        global_token = global_token.reshape(*leading_shape, global_token.shape[-1])

        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
        }


@torch.no_grad()
def example():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rgb_bt = torch.randint(0, 256, (2, 3, 480, 640, 3), dtype=torch.uint8, device=device)
    rgb_btn = torch.randint(0, 256, (2, 3, 2, 480, 640, 3), dtype=torch.uint8, device=device)

    encoder = CLIPEncoder(device=device, tune_mode="freeze")
    encoder.eval()

    out_bt = encoder(rgb_bt)
    out_btn = encoder(rgb_btn)

    print("BT patch_tokens :", tuple(out_bt["patch_tokens"].shape))
    print("BT feature_map  :", tuple(out_bt["feature_map"].shape))
    print("BT global_token :", tuple(out_bt["global_token"].shape))
    print("BTN patch_tokens:", tuple(out_btn["patch_tokens"].shape))
    print("BTN feature_map :", tuple(out_btn["feature_map"].shape))
    print("BTN global_token:", tuple(out_btn["global_token"].shape))


if __name__ == "__main__":
    example()