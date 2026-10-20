"""Flow-matching time distributions and Euler integration grids."""

import math

import torch
from torch.distributions import Beta

from dexmani_policy.utils.config import positive_int


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


def sample_dyadic(batch_size: int, num_steps: int, device: str = "cuda") -> torch.Tensor:
    """Stratify the batch over dyadic time grids, including the zero-time remainder."""
    log2_sections = math.floor(math.log2(num_steps)) + 1
    if batch_size < log2_sections:
        raise ValueError(
            f"discrete_pow requires actual sub-batch >= {log2_sections}, got {batch_size}"
        )
    dt_base = torch.repeat_interleave(
        torch.arange(log2_sections - 1, -1, -1, dtype=torch.long, device=device),
        batch_size // log2_sections,
    )
    remaining = batch_size - dt_base.shape[0]
    if remaining > 0:
        dt_base = torch.cat([dt_base, torch.zeros(remaining, dtype=torch.long, device=device)])

    dt_sections = 2**dt_base
    t = torch.rand(batch_size, device=device) * dt_sections.float()
    t = torch.floor(t).long()
    return t.float() / dt_sections.float()


def shift_time_to_noise(t: torch.Tensor, alpha: float) -> torch.Tensor:
    """Bias training toward noise at t=0: https://arxiv.org/abs/2606.05737."""

    if alpha == 1.0:
        return t
    return t / (1.0 + (alpha - 1.0) * (1.0 - t))


def euler_time_grid(num_steps, reference):
    step_size = 1.0 / num_steps
    return torch.tensor(
        [step * step_size for step in range(num_steps)],
        device=reference.device,
        dtype=reference.dtype,
    )


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
        self.num_steps = positive_int(num_steps, "num_steps")
        self.lognorm_m = lognorm_m
        self.lognorm_s = lognorm_s
        self.mode_s = mode_s
        self.beta_s = beta_s
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

    def sample(self, batch_size: int, mode: str, device: str | torch.device) -> torch.Tensor:
        if mode == "uniform":
            return torch.rand(batch_size, device=device)
        if mode == "lognorm":
            return sample_logit_normal(batch_size, self.lognorm_m, self.lognorm_s, device)
        if mode == "mode":
            return sample_mode(batch_size, self.mode_s, device)
        if mode == "cosmap":
            return sample_cosmap(batch_size, device)
        if mode == "beta":
            return sample_beta(batch_size, self.beta_s, self.beta_alpha, self.beta_beta, device)
        if mode == "discrete":
            return (
                torch.randint(0, self.num_steps, (batch_size,), device=device).float()
                / self.num_steps
            )
        if mode == "discrete_pow":
            return sample_dyadic(batch_size, self.num_steps, device)
        raise ValueError(f"Unknown time sampling mode: {mode}")
