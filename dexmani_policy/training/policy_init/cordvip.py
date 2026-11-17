from pathlib import Path

import torch
from omegaconf import open_dict

from .common import fit_normalizers, instantiate_policy


def build_fresh_policy(agent, initialization, normalization, dataset, field_dims):
    model = instantiate_policy(agent, dataset)
    pretrained_state = None
    inherited_fields: tuple[str, ...] = ()
    if initialization.encoder is not None:
        payload = torch.load(
            Path(initialization.encoder).expanduser(), map_location="cpu", weights_only=True,
        )
        model.obs_encoder.interaction_encoder.load_state_dict(payload["encoder_state"], strict=True)
        fields = ("hand_pointcloud", "object_pointcloud", "joint_state")
        with open_dict(normalization):
            for key in fields:
                normalization[key] = payload["normalization"][key]
        inherited_fields = tuple(key for key in fields if normalization[key] == "limits")
        pretrained_state = payload["normalizer_state"]
        del payload
    fit_normalizers(
        model, dataset, normalization, field_dims,
        pretrained_state=pretrained_state, inherited_fields=inherited_fields,
    )
    return model
