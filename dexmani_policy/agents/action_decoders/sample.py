import math
import torch
import numpy as np
from torch.distributions import Beta

def logit_normal_density(t: np.ndarray, m: float = 0.0, s: float = 1.0) -> np.ndarray:
    t = np.clip(t, 1e-10, 1 - 1e-10)
    logit_t = np.log(t / (1 - t))
    density = (1 / (s * np.sqrt(2 * np.pi))) * (1 / (t * (1 - t))) * np.exp(-(logit_t - m) ** 2 / (2 * s ** 2))
    return density

def sample_logit_normal(
    batch_size: int, m: float = 0.0, s: float = 1.0, device: str = "cuda"
) -> torch.Tensor:
    u = torch.normal(mean=m, std=s, size=(batch_size,), device=device)
    t = torch.sigmoid(u)
    return t

def sample_mode(batch_size: int, s: float = 1.29, device: str = "cuda") -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    t = 1 - u - s * (torch.cos(torch.pi / 2 * u) ** 2 - 1 + u)
    t = torch.clamp(t, 0, 1)
    return t

def sample_cosmap(batch_size: int, device: str = "cuda") -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    t = 1 - 1 / (torch.tan(torch.pi / 2 * u) + 1)
    t = torch.clamp(t, 0, 1)
    return t

def sample_beta(
    batch_size: int, s: float = 0.999, alpha: float = 1.0, beta: float = 1.5, device: str = "cuda"
) -> torch.Tensor:
    beta_dist = Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )
    raw_samples = beta_dist.sample((batch_size,))
    t = s * raw_samples
    return t

def sample_discrete_pow(
    batch_size: int, denoise_timesteps: int, device: str = "cuda"
) -> torch.Tensor:
    log2_sections = math.floor(math.log2(denoise_timesteps)) + 1
    dt_base = torch.repeat_interleave(
        torch.arange(log2_sections - 1, -1, -1, dtype=torch.long, device=device),
        batch_size // log2_sections,
    )
    remaining = batch_size - dt_base.shape[0]
    if remaining > 0:
        dt_base = torch.cat([dt_base, torch.zeros(remaining, dtype=torch.long, device=device)])

    dt_sections = 2 ** dt_base
    t = torch.rand(batch_size, device=device) * dt_sections.float()
    t = torch.floor(t).long()
    t = t.float() / dt_sections.float()
    return t

class TimeSampler:

    def __init__(
        self,
        denoise_timesteps: int,
        lognorm_m: float = 0.0,
        lognorm_s: float = 1.0,
        mode_s: float = 1.29,
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
    ) -> None:
        self.denoise_timesteps = denoise_timesteps
        self.lognorm_m = lognorm_m
        self.lognorm_s = lognorm_s
        self.mode_s = mode_s
        self.beta_s = beta_s
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

        # Pre-built dispatch table: mode name → (batch_size, device) → tensor.
        # Built once in __init__ to avoid dict + 7 lambda allocations on every
        # forward pass (called twice per compute_loss: flow + consistency).
        K = denoise_timesteps
        self._samplers = {
            "uniform":      lambda B, dev: torch.rand((B,), device=dev),
            "lognorm":      lambda B, dev: sample_logit_normal(B, m=lognorm_m, s=lognorm_s, device=dev),
            "mode":         lambda B, dev: sample_mode(B, s=mode_s, device=dev),
            "cosmap":       lambda B, dev: sample_cosmap(B, device=dev),
            "beta":         lambda B, dev: sample_beta(B, s=beta_s, alpha=beta_alpha, beta=beta_beta, device=dev),
            "discrete":     lambda B, dev: torch.randint(0, K, (B,), device=dev).float() / K,
            "discrete_pow": lambda B, dev: sample_discrete_pow(B, K, device=dev),
        }

    def sample(self, batch_size: int, mode: str, device: str) -> torch.Tensor:
        sampler = self._samplers.get(mode)
        if sampler is None:
            raise ValueError(f"Unknown sampling mode for t and delta_t in flowmatch: {mode}")
        return sampler(batch_size, device).reshape(batch_size)