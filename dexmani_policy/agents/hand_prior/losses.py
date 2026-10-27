from torch import Tensor


def diagonal_gaussian_kl(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL to the standard normal, averaged over batch and all latent axes."""
    return -0.5 * (1 + log_var - mu.square() - log_var.exp()).mean()
