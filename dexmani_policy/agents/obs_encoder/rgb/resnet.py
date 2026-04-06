import torch
import torch.nn as nn
import torchvision

from dexmani_policy.agents.obs_encoder.img.utils import (
    ImageTransform, 
    IMAGENET_MEAN, 
    IMAGENET_STD,
)

FEATURE_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        weight_source: str | None = None,
        global_token_type: str = "avg",
        resize_shape=(240, 240),
        crop_shape=(224, 224),
        random_crop: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        self.weight_source = weight_source
        self.global_token_type = global_token_type
        self.feature_dim = FEATURE_DIM[model_name]

        self.image_transform = ImageTransform(
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            random_crop=random_crop,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        self.backbone = self.buildBackbone(model_name, weight_source).to(self.device)

        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.avgpool = self.backbone.avgpool

    def buildBackbone(self, model_name: str, weight_source: str | None):
        if weight_source is None:
            return getattr(torchvision.models, model_name)(weights=None)

        if weight_source == "r3m":
            # cd r3m && pip install -e . &&
            import r3m

            r3m.device = "cpu"
            model = r3m.load_r3m(model_name)
            if hasattr(model, "module"):
                model = model.module
            if hasattr(model, "convnet"):
                model = model.convnet
            return model

        raise ValueError("weight_source only supports None or 'r3m'")

    def preprocessRgb(self, rgb: torch.Tensor):
        rgb = rgb.to(self.device, non_blocking=True)
        rgb = self.image_transform(rgb)
        leading_shape = rgb.shape[:-3]
        rgb = rgb.reshape(-1, *rgb.shape[-3:])
        return rgb, leading_shape

    def encodeFeatureMap(self, rgb: torch.Tensor):
        x = self.stem(rgb)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def buildPatchTokens(self, feature_map: torch.Tensor):
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def buildGlobalToken(self, feature_map: torch.Tensor, patch_tokens: torch.Tensor):
        if self.global_token_type == "avg":
            return patch_tokens.mean(dim=1)
        return self.avgpool(feature_map).flatten(1)

    def forward(self, rgb: torch.Tensor):
        rgb, leading_shape = self.preprocessRgb(rgb)
        feature_map = self.encodeFeatureMap(rgb)
        patch_tokens = self.buildPatchTokens(feature_map)
        global_token = self.buildGlobalToken(feature_map, patch_tokens)

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

    scratch_encoder = ResNetEncoder(
        model_name="resnet18",
        weight_source=None,
        device=device,
    )
    scratch_encoder.eval()

    out_bt = scratch_encoder(rgb_bt)
    out_btn = scratch_encoder(rgb_btn)

    print("scratch BT patch_tokens :", tuple(out_bt["patch_tokens"].shape))
    print("scratch BT feature_map  :", tuple(out_bt["feature_map"].shape))
    print("scratch BT global_token :", tuple(out_bt["global_token"].shape))
    print("scratch BTN patch_tokens:", tuple(out_btn["patch_tokens"].shape))
    print("scratch BTN feature_map :", tuple(out_btn["feature_map"].shape))
    print("scratch BTN global_token:", tuple(out_btn["global_token"].shape))

    print("r3m usage: ResNetEncoder(model_name='resnet18', weight_source='r3m', device=device)")


if __name__ == "__main__":
    example()