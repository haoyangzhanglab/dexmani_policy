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
        model.obs_encoder.pc_encoder.load_state_dict(payload["encoder_state"], strict=True)
        with open_dict(normalization):
            normalization.point_cloud = payload["normalization"]["point_cloud"]
        inherited_fields = ("point_cloud",) if normalization.point_cloud == "limits" else ()
        pretrained_state = payload["normalizer_state"]
        del payload
    fit_normalizers(
        model, dataset, normalization, field_dims,
        pretrained_state=pretrained_state, inherited_fields=inherited_fields,
    )
    return model


def log_initialization(model, encoder_source, policy_source, resume_step):
    if resume_step is not None:
        weights = f"resume checkpoint (step={resume_step})"
    elif policy_source is not None:
        weights = f"policy checkpoint: {policy_source}"
    elif encoder_source is not None:
        weights = f"encoder: {encoder_source}"
    else:
        weights = "random initialization"
    frozen = not any(p.requires_grad for p in model.obs_encoder.pc_encoder.parameters())
    print(f"[AFRO] weights={weights}, visual_frozen={frozen}")
