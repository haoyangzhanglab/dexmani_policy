import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoTokenizer, T5EncoderModel


class T5TextEncoder(nn.Module):
    """Frozen T5 text encoder for robot task instructions."""

    def __init__(
        self,
        model_name: str = "google-t5/t5-small",
        max_length: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.text_backbone = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.text_backbone.eval()
        self.embed_dim = self.text_backbone.config.d_model

        for param in self.text_backbone.parameters():
            param.requires_grad = False

    @staticmethod
    def _masked_mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(token_embeds.dtype)
        pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    @torch.no_grad()
    def forward(self, task_texts: List[str]) -> torch.Tensor:
        task_tokens = self.tokenizer(
            task_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        token_embeds = self.text_backbone(
            input_ids=task_tokens.input_ids,
            attention_mask=task_tokens.attention_mask,
        ).last_hidden_state

        task_embeds = self._masked_mean_pool(token_embeds, task_tokens.attention_mask)
        return task_embeds.unsqueeze(1)


if __name__ == "__main__":
    encoder = T5TextEncoder()
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