from typing import List, Optional

import torch
import torch.nn as nn



class StateMLP(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: Optional[List[int]] = None,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64]

        self.in_dim = input_channels
        self.out_dim = output_channels
        self.mlp = create_mlp(
            in_channels=input_channels,
            hidden_channels=hidden_channels,
            out_channels=output_channels,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def create_state_mlp(
    state_dim: int,
    state_out_dim: int = 64,
    **kwargs,
) -> StateMLP:
    """Create the shared StateMLP for observation encoders.

    Policy observation encoders using this helper embed the
    robot's joint state through this MLP before concatenating it with
    vision / point-cloud features.  This factory centralises the
    project-wide default so that a change to the state-encoding
    architecture (e.g. adding LayerNorm, dropout, or changing the
    hidden size) only needs to be made in one place.
    """
    return StateMLP(input_channels=state_dim, output_channels=state_out_dim, **kwargs)


def create_mlp(
    in_channels: int,
    hidden_channels: List[int],
    out_channels: Optional[int] = None,
    activation: type = nn.ReLU,
    use_norm: bool = False,
    norm_before_activation: bool = False,
):
    layers = []
    prev = in_channels
    for h in hidden_channels:
        if use_norm and norm_before_activation:
            layers.append(nn.LayerNorm(prev))
        layers.extend([nn.Linear(prev, h), activation(inplace=True)])
        if use_norm and not norm_before_activation:
            layers.append(nn.LayerNorm(h))
        prev = h
    if out_channels is not None:
        layers.append(nn.Linear(prev, out_channels))
    return nn.Sequential(*layers)
