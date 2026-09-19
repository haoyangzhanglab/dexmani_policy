import math

import torch
from torch.distributions import Beta


def sample_logit_normal(
    batch_size: int, m: float = 0.0, s: float = 1.0, device: str = "cuda"
) -> torch.Tensor:
    u = torch.normal(mean=m, std=s, size=(batch_size,), device=device)
    return torch.sigmoid(u)


def sample_mode(batch_size: int, s: float = 1.29, device: str = "cuda") -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    t = 1 - u - s * (torch.cos(torch.pi / 2 * u) ** 2 - 1 + u)
    return torch.clamp(t, 0, 1)


def sample_cosmap(batch_size: int, device: str = "cuda") -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    t = 1 - 1 / (torch.tan(torch.pi / 2 * u) + 1)
    return torch.clamp(t, 0, 1)


def sample_beta(
    batch_size: int,
    s: float = 0.999,
    alpha: float = 1.0,
    beta: float = 1.5,
    device: str = "cuda",
) -> torch.Tensor:
    beta_dist = Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )
    return s * beta_dist.sample((batch_size,))


def sample_discrete_pow(
    batch_size: int, num_steps: int, device: str = "cuda"
) -> torch.Tensor:
    log2_sections = math.floor(math.log2(num_steps)) + 1
    dt_base = torch.repeat_interleave(
        torch.arange(log2_sections - 1, -1, -1, dtype=torch.long, device=device),
        batch_size // log2_sections,
    )
    remaining = batch_size - dt_base.shape[0]
    if remaining > 0:
        dt_base = torch.cat(
            [dt_base, torch.zeros(remaining, dtype=torch.long, device=device)]
        )

    dt_sections = 2**dt_base
    t = torch.rand(batch_size, device=device) * dt_sections.float()
    t = torch.floor(t).long()
    return t.float() / dt_sections.float()


def shift_time_to_noise(t: torch.Tensor, alpha: float) -> torch.Tensor:
    """Bias paper-coordinate flow times toward the noise endpoint t=0.

    Implements Eq. (9) from Let It Be Simple:
        t_shifted = t / (1 + (alpha - 1) * (1 - t))

    alpha=1 is the identity map. Larger values allocate more training
    probability mass near high-noise states while keeping the standard
    rectified-flow interpolation and velocity target unchanged.
    """
    if alpha < 1.0:
        raise ValueError(f"time-shift alpha must be >= 1, got {alpha}")
    if alpha == 1.0:
        return t
    return t / (1.0 + (alpha - 1.0) * (1.0 - t))


class TimeSampler:
    def __init__(
        self,
        num_steps: int,
        lognorm_m: float = 0.0,
        lognorm_s: float = 1.0,
        mode_s: float = 1.29,
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be greater than 0")

        self.num_steps = num_steps
        self.lognorm_m = lognorm_m
        self.lognorm_s = lognorm_s
        self.mode_s = mode_s
        self.beta_s = beta_s
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

        K = num_steps
        self._samplers = {
            "uniform": lambda B, dev: torch.rand((B,), device=dev),
            "lognorm": lambda B, dev: sample_logit_normal(
                B, m=lognorm_m, s=lognorm_s, device=dev
            ),
            "mode": lambda B, dev: sample_mode(B, s=mode_s, device=dev),
            "cosmap": lambda B, dev: sample_cosmap(B, device=dev),
            "beta": lambda B, dev: sample_beta(
                B, s=beta_s, alpha=beta_alpha, beta=beta_beta, device=dev
            ),
            "discrete": lambda B, dev: torch.randint(0, K, (B,), device=dev).float() / K,
            "discrete_pow": lambda B, dev: sample_discrete_pow(B, K, device=dev),
        }

    def sample(
        self, batch_size: int, mode: str, device: str | torch.device
    ) -> torch.Tensor:
        sampler = self._samplers.get(mode)
        if sampler is None:
            raise ValueError(f"Unknown time sampling mode: {mode}")
        return sampler(batch_size, device).reshape(batch_size)
