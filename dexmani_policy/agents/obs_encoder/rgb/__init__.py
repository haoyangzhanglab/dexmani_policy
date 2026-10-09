"""RGB observation encoders.

DP uses dataset/eval RGB preprocessing, ImageProcessor.process_images, then
backbone forward/global_token and state features. Processed RGB may be uint8
or float32. Optional process_rgbd/backproject/GeometryProcessor APIs require
explicit depth/intrinsics; DP does not invoke them automatically.
See the resnet.py and dino.py __main__ examples for geometry usage.

Import encoders directly from their modules (e.g.
``from ...rgb.r3m import R3M``); this package does not re-export them, so a
single encoder import never eagerly pulls torchvision/transformers backbones
(DINO/CLIP/SigLIP) that are not being used.
"""
