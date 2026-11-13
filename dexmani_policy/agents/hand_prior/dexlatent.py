"""Frozen XHand branch from DexLatent (502750df), using deterministic means."""

# MIT License
# Copyright (c) 2026 Yutong Liang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path

import torch
from torch import nn

from dexmani_policy.agents.modules import build_mlp


class DexLatentHandPrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hand_encoder_backbone = build_mlp([12, 64, 128, 64], activation=nn.ReLU, activate_last=True)
        self.hand_mean_head = nn.Linear(64, 32)
        self.hand_decoder = build_mlp([32, 64, 128, 64, 12], activation=nn.ReLU)
        self.hand_decoder.append(nn.Tanh())
        self.requires_grad_(False)

    def initialize_weights(self, source: str | Path | None) -> dict[str, torch.Tensor]:
        if source is None:
            raise ValueError("Fresh DexLatent training requires initialization.dexlatent")
        payload = torch.load(Path(source).expanduser(), map_location="cpu", weights_only=True)
        prefix = "xarm7_xhand_right."
        self.load_state_dict({
            key.removeprefix(prefix): value
            for key, value in payload["autoencoders"].items()
            if key.startswith(prefix) and not key.startswith(prefix + "hand_logvar_head.")
        }, strict=True)
        lower = torch.tensor([0, -0.698, 0, -0.174, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        upper = torch.tensor([
            1.832, 1.745, 1.745, 0.174, 1.919, 1.919,
            1.919, 1.919, 1.919, 1.919, 1.919, 1.919,
        ], dtype=torch.float32)
        scale = 2 / (upper - lower)
        return {"scale": scale, "offset": -1 - lower * scale}

    @torch.no_grad()
    def encode(self, hand: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hand.device.type, enabled=False):
            return self.hand_mean_head(self.hand_encoder_backbone(hand.float().clamp(-1, 1)))

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return self.hand_decoder(latent.float())
