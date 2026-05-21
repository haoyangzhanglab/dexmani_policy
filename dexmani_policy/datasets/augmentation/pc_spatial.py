import numpy as np
from .base import Aug


class PCSpatialAug(Aug):
    """Spatial augmentation with temporal consistency.

    Reference: RISE (IROS 2024) — Z-axis rotation + XY translation.

    Applies random rotation around Z (gravity axis), XY translation, and
    uniform scaling. Same transform parameters applied to ALL T frames.

    Input:  (T, N, C) float32 numpy, XYZ in channels [..., :3]
    Output: (T, N, C) float32 numpy, same shape
    """

    def __init__(self, rot_z=15.0, trans_xy=0.10, scale=1.02,
                 prob=1.0):
        super().__init__(prob=prob)
        self.rot_z = float(rot_z)
        self.trans_xy = float(trans_xy)
        self.scale = float(scale)

    def _augment(self, x):
        pc = x.copy()
        xyz = pc[..., :3]  # (T, N, 3)

        # sample once, apply to all frames
        angle = np.deg2rad(np.random.uniform(-self.rot_z, self.rot_z))
        dx = np.random.uniform(-self.trans_xy, self.trans_xy)
        dy = np.random.uniform(-self.trans_xy, self.trans_xy)
        s = np.random.uniform(min(1.0 / self.scale, self.scale), max(1.0 / self.scale, self.scale))

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        T, N, _ = xyz.shape

        # rotate Z + scale
        rx = xyz[..., 0] * cos_a - xyz[..., 1] * sin_a
        ry = xyz[..., 0] * sin_a + xyz[..., 1] * cos_a
        pc[..., 0] = rx * s + dx
        pc[..., 1] = ry * s + dy
        pc[..., 2] = xyz[..., 2] * s

        # rotate normals (channels 3:6) if present — same rotation, no translation/scale
        if pc.shape[-1] >= 6:
            nxyz = x[..., 3:6]
            pc[..., 3] = nxyz[..., 0] * cos_a - nxyz[..., 1] * sin_a
            pc[..., 4] = nxyz[..., 0] * sin_a + nxyz[..., 1] * cos_a

        return pc
