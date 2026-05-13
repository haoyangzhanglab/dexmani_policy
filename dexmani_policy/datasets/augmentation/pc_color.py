import numpy as np
from .base import Aug


class PCColorJitter(Aug):
    """HSV color space jitter for point cloud RGB channels.

    Reference: ManiFlow (CoRL 2025) PointCloudColorJitter, ported to numpy.
    Applies brightness → contrast → saturation → hue in sequence.
    All frames share the same jitter parameters.

    Input:  (T, N, C) float32 numpy, C >= 6, RGB in channels [..., -3:]
    Output: (T, N, C) float32 numpy, same shape
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05,
                 enabled=True, prob=1.0):
        super().__init__(enabled=enabled, prob=prob)
        self.brightness = self.make_range(brightness, center=1.0)
        self.contrast = self.make_range(contrast, center=1.0)
        self.saturation = self.make_range(saturation, center=1.0)
        self.hue = self.make_range(hue, center=0.0, symmetric=True)

    @staticmethod
    def make_range(value, center, symmetric=False):
        if symmetric:
            if isinstance(value, (int, float)):
                return [-abs(value), abs(value)]
            return [value[0], value[1]]
        if isinstance(value, (int, float)):
            value = abs(value)
            return [max(0.0, center - value), center + value]
        return [value[0], value[1]]

    def _augment(self, x):
        if x.shape[-1] < 6:
            return x

        pc = x.copy()
        rgb = pc[..., 3:6]  # (T, N, 3)
        orig_shape = rgb.shape
        rgb = rgb.reshape(-1, 3)

        rgb = self.apply_brightness(rgb)
        rgb = self.apply_contrast(rgb)
        rgb = self.apply_saturation(rgb)
        rgb = self.apply_hue(rgb)
        rgb = np.clip(rgb, 0.0, 1.0)

        pc[..., 3:6] = rgb.reshape(orig_shape)
        return pc

    # --- per-pixel ops on (P, 3) ---

    def apply_brightness(self, rgb):
        if self.brightness[0] == self.brightness[1]:
            return rgb
        factor = np.random.uniform(*self.brightness)
        return rgb * factor

    def apply_contrast(self, rgb):
        if self.contrast[0] == self.contrast[1]:
            return rgb
        factor = np.random.uniform(*self.contrast)
        return (rgb - 0.5) * factor + 0.5

    def apply_saturation(self, rgb):
        if self.saturation[0] == self.saturation[1]:
            return rgb
        factor = np.random.uniform(*self.saturation)
        gray = np.dot(rgb, [0.299, 0.587, 0.114])[:, None]
        return rgb * factor + gray * (1.0 - factor)

    def apply_hue(self, rgb):
        if self.hue[0] == self.hue[1]:
            return rgb
        shift = np.random.uniform(*self.hue)
        if abs(shift) < 1e-6:
            return rgb
        hsv = self.rgb_to_hsv(rgb)
        hsv[:, 0] = (hsv[:, 0] + shift) % 1.0
        return self.hsv_to_rgb(hsv)

    # --- numpy RGB ↔ HSV ---

    @staticmethod
    def rgb_to_hsv(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        max_v = np.max(rgb, axis=-1)
        min_v = np.min(rgb, axis=-1)
        diff = max_v - min_v
        max_idx = np.argmax(rgb, axis=-1)

        h = np.zeros_like(max_v)
        mask = diff != 0

        # R is max
        rm = mask & (max_idx == 0)
        h[rm] = ((g[rm] - b[rm]) / diff[rm]) / 6.0

        # G is max
        gm = mask & (max_idx == 1)
        h[gm] = ((b[gm] - r[gm]) / diff[gm] + 2.0) / 6.0

        # B is max
        bm = mask & (max_idx == 2)
        h[bm] = ((r[bm] - g[bm]) / diff[bm] + 4.0) / 6.0

        h = h % 1.0
        s = np.where(max_v != 0, diff / max_v, 0.0)
        v = max_v
        return np.stack([h, s, v], axis=-1)

    @staticmethod
    def hsv_to_rgb(hsv):
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        h = h * 6.0
        i = np.floor(h).astype(np.int32) % 6
        f = h - np.floor(h)

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        rgb = np.zeros_like(hsv)
        # i%6==0: (v,t,p)    i%6==1: (q,v,p)    i%6==2: (p,v,t)
        # i%6==3: (p,q,v)    i%6==4: (t,p,v)    i%6==5: (v,p,q)
        for idx, (r_src, g_src, b_src) in enumerate([
            (v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q),
        ]):
            mask = i == idx
            rgb[mask, 0] = r_src[mask]
            rgb[mask, 1] = g_src[mask]
            rgb[mask, 2] = b_src[mask]
        return rgb
