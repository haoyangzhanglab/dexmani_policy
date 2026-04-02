import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        tune_mode: str = "freeze",
        global_token_type: str = "model",
        resize_shape=(256, 256),
        crop_shape=(224, 224),
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        self.tune_mode = tune_mode
        self.global_token_type = global_token_type
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device)

        self.patch_size = self.backbone.config.patch_size
        self.hidden_size = self.backbone.config.hidden_size
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)

        self.setTuneMode(tune_mode)

    def setTuneMode(self, tune_mode: str):
        self.tune_mode = tune_mode

        if tune_mode == "freeze":
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        if tune_mode == "last_block":
            for param in self.backbone.parameters():
                param.requires_grad = False

            last_idx = self.backbone.config.num_hidden_layers - 1
            prefixes = (
                f"encoder.layers.{last_idx}",
                f"encoder.layer.{last_idx}",
                "layernorm",
                "post_layernorm",
                "ln_f",
            )
            for name, param in self.backbone.named_parameters():
                if any(prefix in name for prefix in prefixes):
                    param.requires_grad = True
            return

        if tune_mode == "lora":
            from peft import LoraConfig, get_peft_model

            for param in self.backbone.parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules="all-linear",
            )
            self.backbone = get_peft_model(self.backbone, lora_config).to(self.device)
            return

        raise ValueError("tune_mode must be one of: freeze, last_block, lora")

    def flattenRgb(self, rgb: torch.Tensor):
        leading_shape = rgb.shape[:-3]
        rgb = rgb.reshape(-1, *rgb.shape[-3:])
        return rgb, leading_shape

    def preprocessRgb(self, rgb: torch.Tensor):
        rgb, leading_shape = self.flattenRgb(rgb)
        rgb = rgb.to(self.device, non_blocking=True)

        processor_kwargs = {
            "images": rgb,
            "return_tensors": "pt",
        }

        if self.resize_shape is not None:
            processor_kwargs["do_resize"] = True
            processor_kwargs["size"] = {"height": self.resize_shape[0], "width": self.resize_shape[1]}
        else:
            processor_kwargs["do_resize"] = False

        if self.crop_shape is not None:
            processor_kwargs["do_center_crop"] = True
            processor_kwargs["crop_size"] = {"height": self.crop_shape[0], "width": self.crop_shape[1]}
        else:
            processor_kwargs["do_center_crop"] = False

        pixel_values = self.processor(**processor_kwargs)["pixel_values"]
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        return pixel_values, leading_shape

    def forwardBackbone(self, pixel_values: torch.Tensor):
        return self.backbone(pixel_values=pixel_values, return_dict=True)

    def buildPatchTokens(self, outputs):
        return outputs.last_hidden_state[:, 1 + self.num_register_tokens :, :]

    def buildFeatureMap(self, patch_tokens: torch.Tensor, pixel_values: torch.Tensor):
        batch_size, _, hidden_size = patch_tokens.shape
        num_patches_h = pixel_values.shape[-2] // self.patch_size
        num_patches_w = pixel_values.shape[-1] // self.patch_size
        feature_map = patch_tokens.reshape(batch_size, num_patches_h, num_patches_w, hidden_size)
        feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
        return feature_map

    def getGlobalToken(self, outputs, patch_tokens: torch.Tensor):
        if self.global_token_type == "avg":
            return patch_tokens.mean(dim=1)
        return outputs.pooler_output

    def forward(self, rgb: torch.Tensor):
        pixel_values, leading_shape = self.preprocessRgb(rgb)
        outputs = self.forwardBackbone(pixel_values)
        patch_tokens = self.buildPatchTokens(outputs)
        feature_map = self.buildFeatureMap(patch_tokens, pixel_values)
        global_token = self.getGlobalToken(outputs, patch_tokens)

        patch_tokens = patch_tokens.reshape(*leading_shape, patch_tokens.shape[-2], patch_tokens.shape[-1])
        feature_map = feature_map.reshape(*leading_shape, feature_map.shape[-3], feature_map.shape[-2], feature_map.shape[-1])
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

    encoder = DINOv3Encoder(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        tune_mode="freeze",
        global_token_type="model",
        resize_shape=(256, 256),
        crop_shape=(224, 224),
        device=device,
    )
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