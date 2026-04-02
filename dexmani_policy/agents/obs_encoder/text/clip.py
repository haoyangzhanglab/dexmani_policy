import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoTokenizer, CLIPTextModelWithProjection


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder for robot task instructions."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.text_backbone = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)
        self.text_backbone.eval()
        self.embed_dim = self.text_backbone.config.projection_dim

        for param in self.text_backbone.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, task_texts: List[str]) -> torch.Tensor:
        task_tokens = self.tokenizer(
            task_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        task_embeds = self.text_backbone(
            input_ids=task_tokens.input_ids,
            attention_mask=task_tokens.attention_mask,
        ).text_embeds
        return task_embeds.unsqueeze(1)


if __name__ == "__main__":
    encoder = CLIPTextEncoder()
    task_texts = [
        "pick up the red mug",
        "open the drawer and place the spoon inside",
    ]

    task_embeds = encoder(task_texts)
    print("encoder:", encoder.__class__.__name__)
    print("device:", encoder.device)
    print("embed_dim:", encoder.embed_dim)
    print("output_shape:", tuple(task_embeds.shape))
    print("requires_grad:", task_embeds.requires_grad)