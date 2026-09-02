"""RGB observation encoders.

Import encoders directly from their modules (e.g.
``from ...rgb.r3m import R3M``); this package does not re-export them, so a
single encoder import never eagerly pulls torchvision/transformers backbones
(DINO/CLIP/SigLIP) that are not being used.
"""
