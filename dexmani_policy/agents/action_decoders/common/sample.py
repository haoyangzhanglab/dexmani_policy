import math
import torch
import numpy as np
from torch.distributions import Beta


def logit_normal_density(t, m=0.0, s=1.0):
    t = np.clip(t, 1e-10, 1 - 1e-10)
    logit_t = np.log(t / (1 - t))
    density = (1 / (s * np.sqrt(2 * np.pi))) * (1 / (t * (1 - t))) * np.exp(-(logit_t - m) ** 2 / (2 * s ** 2))
    return density

def sample_logit_normal(batch_size, m=0.0, s=1.0, device="cuda"):
    u = torch.normal(mean=m, std=s, size=(batch_size,), device=device)
    t = torch.sigmoid(u)
    return t

def f_mode(u, s):
    return 1 - u - s * (torch.cos(torch.pi / 2 * u) ** 2 - 1 + u)

def sample_mode(batch_size, s=1.29, device="cuda"):
    u = torch.rand(batch_size, device=device)
    t = f_mode(u, s)
    t = torch.clamp(t, 0, 1)
    return t

def sample_cosmap(batch_size, device="cuda"):
    u = torch.rand(batch_size, device=device)
    t = 1 - 1 / (torch.tan(torch.pi / 2 * u) + 1)
    t = torch.clamp(t, 0, 1)
    return t

def sample_beta(batch_size, s=0.999, alpha=1.0, beta=1.5, device="cuda"):
    beta_dist = Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device),
    )
    raw_samples = beta_dist.sample((batch_size,))
    t = s * raw_samples
    return t


def sample_discrete_pow(batch_size, denoise_timesteps, device="cuda"):
    log2_sections = int(math.log2(denoise_timesteps)) + 1
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


class SampleLibrary:

    def __init__(
        self,
        denoise_timesteps,
        lognorm_m=0.0,
        lognorm_s=1.0,
        mode_s=1.29,
        beta_s=0.999,
        beta_alpha=1.0,
        beta_beta=1.5,
    ):
        self.denoise_timesteps = denoise_timesteps

        self.lognorm_m = lognorm_m
        self.lognorm_s = lognorm_s
        self.mode_s = mode_s

        self.beta_s = beta_s
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

    def sample(self, batch_size, mode, device):

        if mode == "uniform":
            sample = torch.rand((batch_size,), device=device)

        elif mode == "lognorm":
            sample = sample_logit_normal(
                batch_size, m=self.lognorm_m, s=self.lognorm_s, device=device
            )

        elif mode == "mode":
            sample = sample_mode(batch_size, s=self.mode_s, device=device)

        elif mode == "cosmap":
            sample = sample_cosmap(batch_size, device=device)

        elif mode == "beta":
            sample = sample_beta(
                batch_size,
                s=self.beta_s,
                alpha=self.beta_alpha,
                beta=self.beta_beta,
                device=device,
            )

        elif mode == "discrete":
            # Discrete time grid: {0, 1/K, ..., (K-1)/K}
            sample = torch.randint(
                low=0,
                high=self.denoise_timesteps,
                size=(batch_size,),
                device=device,
            ).float()
            sample = sample / self.denoise_timesteps

        elif mode == "discrete_pow":
            sample = sample_discrete_pow(batch_size, self.denoise_timesteps, device=device)

        else:
            raise ValueError(f"Unknown sampling mode for t and delta_t in flowmatch: {mode}")

        return sample.reshape(batch_size)