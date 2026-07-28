import logging
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from typing import Dict, Optional

from dexmani_policy.agents.obs_encoder.rgb.base import ViTEncoder
from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
from dexmani_policy.agents.obs_encoder.rgb.types import GlobalTokenType, TuneMode

logger = logging.getLogger(__name__)

_DINO_VARIANTS = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
}

def _resolve_dino_model_name(name: str) -> str:
    """Expand short variant names to full HuggingFace model IDs.

    Short names like ``"small"`` / ``"base"`` are expanded to the
    corresponding ``facebook/dinov2-*`` ID.  Full HF IDs pass through
    unchanged, preserving backward compatibility.
    """
    resolved = _DINO_VARIANTS.get(name, name)
    if resolved != name:
        logger.info("DINO variant shorthand '%s' resolved to '%s'", name, resolved)
    return resolved

class DINO(ViTEncoder):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        tune_mode: TuneMode = "freeze",
        global_token_type: GlobalTokenType = "avg",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        model_name = _resolve_dino_model_name(model_name)
        self.model_name = model_name
        self.tune_mode = tune_mode
        self.global_token_type = global_token_type
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "sdpa"
        self.backbone = AutoModel.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)

        if not hasattr(self.backbone.config, "patch_size"):
            raise ValueError(f"{model_name} does not look like a ViT-style DINO model.")
        if not hasattr(self.backbone.config, "hidden_size"):
            raise ValueError(f"{model_name} is missing hidden_size in backbone config.")

        self.patch_size = int(self.backbone.config.patch_size)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.num_prefix_tokens = 1 + self.num_register_tokens
        self.out_dim = self.hidden_dim if out_dim is None else int(out_dim)

        logger.info(
            "DINO backbone %s: hidden_dim=%d num_register_tokens=%d num_prefix_tokens=%d",
            model_name, self.hidden_dim, self.num_register_tokens, self.num_prefix_tokens,
        )

        self.proj = nn.Identity() if self.out_dim == self.hidden_dim else nn.Linear(self.hidden_dim, self.out_dim)
        self.set_tune_mode(tune_mode)

    def _get_lora_target_modules(self) -> list[str]:
        if bool(getattr(self.backbone.config, "use_swiglu_ffn", False)):
            return ["query", "key", "value", "dense", "weights_in", "weights_out"]
        return ["query", "key", "value", "dense", "fc1", "fc2"]

    def get_global_token(self, outputs, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.global_token_type == "avg":
            return patch_tokens.mean(dim=1)

        if self.global_token_type == "cls":
            return self.proj(outputs.last_hidden_state[:, 0])

        if self.global_token_type == "pooler":
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is None:
                raise ValueError(
                    f"{self.model_name} does not provide pooler_output. "
                    "Use global_token_type='cls' or 'avg'."
                )
            return self.proj(pooler_output)

        raise ValueError(f"Unsupported global_token_type: {self.global_token_type}")

def example() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov2-small"

    image_processor = ImageProcessor.from_preset("dino")

    images = torch.randint(0, 256, (16, 2, 480, 640, 3), dtype=torch.uint8)
    depths = torch.randint(1, 2000, (16, 2, 480, 640), dtype=torch.uint16)
    intrinsics = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_to_world = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5]],
        dtype=torch.float32,
    )

    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).expand(images.shape[0], images.shape[1], -1, -1)
    camera_to_world = camera_to_world.unsqueeze(0).unsqueeze(0).expand(images.shape[0], images.shape[1], -1, -1)

    try:
        encoder = DINO(model_name=model_name, tune_mode="freeze").to(device)
        encoder.eval()

        rgbd_batch = image_processor.process_rgbd(
            images=images,
            depths=depths,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )

        rgb = rgbd_batch["image"].to(device)
        depth = rgbd_batch["depth"].to(device)
        intrinsics = rgbd_batch["intrinsics"].to(device)
        camera_to_world = None if rgbd_batch["camera_to_world"] is None else rgbd_batch["camera_to_world"].to(device)

        with torch.no_grad():
            vision_out = encoder(rgb)
            geometry_out = encoder.backproject(
                depth=depth,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                depth_scale=1000.0,
                min_depth=0.01,
                max_depth=3.0,
            )
            feature_map = encoder.patch_tokens_to_featmap(
                vision_out["patch_tokens"],
                image_hw=rgb.shape[-2:],
            )

        print("rgb             :", tuple(rgb.shape))
        print("patch_tokens    :", tuple(vision_out["patch_tokens"].shape))
        print("global_token    :", tuple(vision_out["global_token"].shape))
        print("feature_map     :", tuple(feature_map.shape))
        print("patch_coords    :", tuple(geometry_out["patch_coords"].shape))
        print("patch_valid_mask:", tuple(geometry_out["patch_valid_mask"].shape))

    except Exception as error:
        print("dino example failed.")
        print(error)

if __name__ == "__main__":
    example()
