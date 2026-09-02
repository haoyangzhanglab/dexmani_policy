"""Regression: R3M applies official preprocessing ([0, 1] → ImageNet norm).

Guards the P0 fix — official R3M takes uint8 ``[0, 255]``, divides by 255 to
``[0, 1]``, then applies ImageNet normalization.  This is equivalent to
normalizing a ``[0, 1]`` image directly; there must be **no** ×255.  The
pre-fix code erroneously multiplied by 255 before normalizing.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from dexmani_policy.agents.obs_encoder.rgb.r3m import (
    _R3M_IMAGENET_MEAN,
    _R3M_IMAGENET_STD,
    R3M,
    _load_r3m_convnet_state_dict,
)
from dexmani_policy.agents.obs_encoder.rgb.resnet import replace_batch_norm_with_group_norm


def _normlayer():
    return T.Normalize(mean=list(_R3M_IMAGENET_MEAN), std=list(_R3M_IMAGENET_STD))


class _ReferenceConvnet(nn.Module):
    """Raw convnet mirroring R3M's internal backbone, with an explicit norm.

    Built with the same weights, BN→GN replacement and avgpool/fc strip as
    ``R3M.__init__``, so ``forward`` applies the norm before the convnet — the
    ground truth for official R3M preprocessing.
    """

    def __init__(self, model_name: str = "resnet18"):
        super().__init__()
        resnet_fn = getattr(torchvision.models, model_name)
        backbone = resnet_fn(weights=None, norm_layer=nn.BatchNorm2d)
        backbone.fc = nn.Identity()
        convnet_state = _load_r3m_convnet_state_dict(model_name)
        backbone.load_state_dict(convnet_state, strict=False)
        backbone = replace_batch_norm_with_group_norm(backbone)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.normlayer = _normlayer()
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(self.normlayer(x))
        return feature_map.mean(dim=(-2, -1))


class R3MPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            # Forces the checkpoint download/cache check once, so a missing
            # weight (no cache + no network) skips cleanly rather than failing.
            _load_r3m_convnet_state_dict("resnet18")
        except Exception as exc:  # ImportError (gdown) / OSError / network
            raise unittest.SkipTest(f"R3M weights unavailable: {exc}")

        cls.reference = _ReferenceConvnet("resnet18")
        cls.r3m = R3M(model_name="resnet18")
        cls.r3m.eval()

        # Deterministic [0, 1] input (single image, batched).
        torch.manual_seed(0)
        cls.x = torch.rand(2, 3, 224, 224)

    def test_matches_official_normalize(self):
        # R3M must equal the raw convnet fed (x - mean) / std — bit exact, since
        # both apply the identical ImageNet norm with identical op order.
        with torch.no_grad():
            got = self.r3m(self.x)["global_token"]
        ref = self.reference(self.x)
        self.assertTrue(torch.allclose(got, ref, atol=0.0, rtol=0.0))

    def test_does_not_multiply_by_255(self):
        # A reintroduced ×255 (the pre-fix bug) would shift the output far from
        # the official path — assert they differ.
        with torch.no_grad():
            got = self.r3m(self.x)["global_token"]
        ref = self.reference(self.x * 255.0)
        self.assertFalse(torch.allclose(got, ref, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
