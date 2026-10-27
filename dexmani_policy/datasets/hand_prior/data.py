from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.split import split_episodes


def prepare_hand_source(cfg, keys):
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare hand artifacts from a fresh policy config, not resume_from")
    if cfg.dataset.action_key != "action" or cfg.agent.action_dim != 19:
        raise ValueError("Hand pretraining requires action=arm7+hand12 joint commands")
    buffer = ReplayBuffer(cfg.dataset.zarr_path, keys=keys)
    if buffer["action"].shape[1:] != (19,):
        raise ValueError("Hand pretraining requires action shape [T,19]")
    train, val, split = split_episodes(
        buffer, manifest=cfg.dataset.get("split_manifest"), seed=cfg.dataset.seed,
        val_ratio=cfg.dataset.val_ratio, max_train_episodes=cfg.dataset.max_train_episodes,
    )
    return buffer, train, val, split
