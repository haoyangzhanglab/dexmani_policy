import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_model(model):
    """Unwrap DDP model to get the underlying module."""
    if isinstance(model, DDP):
        return model.module
    return model


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapping."""
    return unwrap_model(model).state_dict()
