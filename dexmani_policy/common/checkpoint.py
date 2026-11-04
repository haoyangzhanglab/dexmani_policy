from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dexmani_policy.common.io_util import atomic_path

if TYPE_CHECKING:
    import torch

    from dexmani_policy.agents.policies.base import Policy


@dataclass
class Checkpoint:
    global_step: int
    model_state: dict[str, Any]
    ema_model_state: dict[str, Any] | None
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    rng_state: dict[str, Any]
    effective_config: dict[str, Any]


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save(vars(checkpoint), temporary)
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    import torch

    # Project checkpoints contain trusted training state, not just tensors.
    return Checkpoint(**torch.load(path, map_location="cpu", weights_only=False, mmap=True))


def policy_state_dict(model) -> dict:
    # Compiled backbones add _orig_mod to parameter names.
    return {name.replace("_orig_mod.", ""): value for name, value in model.state_dict().items()}


def restore_policy(checkpoint_path: str | Path, *, use_ema: bool, device: "str | torch.device") -> "Policy":
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint.effective_config
    state_dict = checkpoint.ema_model_state if use_ema else checkpoint.model_state
    # Drop training state and unselected weights before constructing the model.
    del checkpoint
    return _restore_policy(cfg, state_dict, device)


def _restore_policy(
    effective_config: dict[str, Any], state_dict: dict[str, Any] | None,
    device: "str | torch.device",
) -> "Policy":
    import hydra
    import torch

    from dexmani_policy.common.normalizer import normalizer_shapes_from_state
    from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state

    torch.set_float32_matmul_precision("highest")
    if state_dict is None:
        raise ValueError("Requested EMA weights are absent; select raw explicitly")

    feature_dims = effective_config["normalization_dims"]
    feature_shapes = normalizer_shapes_from_state(
        state_dict, feature_dims=feature_dims,
    )
    rng_state = get_rng_state()
    try:
        agent = hydra.utils.instantiate(effective_config["agent"], _convert_="all")
    finally:
        set_rng_state(rng_state)

    agent.configure_normalizer(feature_dims, feature_shapes)
    agent.load_state_dict(state_dict, strict=True, assign=True)
    agent.to(device=device, dtype=torch.float32)
    agent.eval()
    rgb_encoder = getattr(agent.obs_encoder, "rgb_encoder", None)
    if rgb_encoder is not None and rgb_encoder.tune_mode == "lora":
        rgb_encoder.backbone = rgb_encoder.backbone.merge_and_unload()
    return agent
