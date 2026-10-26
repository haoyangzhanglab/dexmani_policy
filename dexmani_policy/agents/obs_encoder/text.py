"""CLIP text encoding for task conditioning."""

import torch
from torch import nn
from transformers import AutoTokenizer, CLIPTextModelWithProjection


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text embeddings with shape [B,1,D]."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.text_backbone = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        self.text_backbone.requires_grad_(False)
        self.text_backbone.eval()
        self.embed_dim = self.text_backbone.config.projection_dim

    @property
    def device(self) -> torch.device:
        return next(self.text_backbone.parameters()).device

    def train(self, mode: bool = True) -> "CLIPTextEncoder":
        super().train(mode)
        self.text_backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, task_texts: list[str]) -> torch.Tensor:
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
