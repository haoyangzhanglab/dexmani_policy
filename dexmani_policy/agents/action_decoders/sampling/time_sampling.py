"""Flow-matching time distributions and Euler integration grids."""

import torch
from torch.distributions import Beta

from dexmani_policy.common.config import positive_int


def shift_time_to_noise(t: torch.Tensor, alpha: float) -> torch.Tensor:
    """Bias training toward noise at t=0: https://arxiv.org/abs/2606.05737."""

    if alpha == 1.0:
        return t
    return t / (1.0 + (alpha - 1.0) * (1.0 - t))


def euler_time_grid(num_steps: int, reference: torch.Tensor) -> torch.Tensor:
    step_size = 1.0 / num_steps
    return torch.tensor(
        [step * step_size for step in range(num_steps)],
        device=reference.device,
        dtype=torch.float32,
    )


class TimeSampler:
    def __init__(self, num_steps: int) -> None:
        self.num_steps = positive_int(num_steps, "num_steps")

    def sample(self, batch_size: int, mode: str, device: str | torch.device) -> torch.Tensor:
        if mode == "uniform":
            return torch.rand(batch_size, device=device)
        if mode == "beta":
            distribution = Beta(
                torch.tensor(1.0, device=device, dtype=torch.float32),
                torch.tensor(1.5, device=device, dtype=torch.float32),
                validate_args=False,
            )
            return 0.999 * distribution.sample(torch.Size((batch_size,)))
        if mode == "discrete":
            return (
                torch.randint(0, self.num_steps, (batch_size,), device=device).float()
                / self.num_steps
            )
        raise ValueError(f"Unknown time sampling mode: {mode}")
