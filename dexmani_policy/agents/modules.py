import torch.nn as nn


def build_mlp(
    channels: list[int],
    *,
    activation: type[nn.Module] = nn.GELU,
    normalize: bool = True,
    activate_last: bool = False,
) -> nn.Sequential:
    """Linear layers with optional LayerNorm and activation after hidden layers."""
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(channels, channels[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(channels) - 2 or activate_last:
            if normalize:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation())
    return nn.Sequential(*layers)
