import numpy as np
from omegaconf import OmegaConf

from dexmani_policy.common.config import resolve_normalization_spec
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.split import split_episodes
from dexmani_policy.datasets.common.validation import check_finite, validate_fields
from dexmani_policy.datasets.hand_prior.metadata import describe_hand_data
from dexmani_policy.datasets.hand_prior.lamp_dataset import load_hand_history


def prepare_hand_source(cfg, expected_target, keys):
    if cfg.agent._target_ != expected_target:
        raise ValueError(f"Hand preparation requires {expected_target}")
    if cfg.get("resume_from") is not None:
        raise ValueError("Prepare hand artifacts from a fresh policy config, not resume_from")
    if cfg.dataset.split_manifest is None:
        raise ValueError("Pretraining requires an explicit shared dataset.split_manifest")
    if cfg.dataset.action_key != "action" or cfg.agent.action_dim != 19:
        raise ValueError("Hand pretraining requires action=arm7+hand12 joint commands")
    buffer = ReplayBuffer(cfg.dataset.zarr_path, keys=keys)
    if buffer["action"].shape[1:] != (19,):
        raise ValueError("Hand pretraining requires action shape [T,19]")
    train, val, split = split_episodes(buffer, manifest=cfg.dataset.split_manifest)
    hand_data = describe_hand_data(
        buffer, action_key=cfg.dataset.action_key, horizon=cfg.dataset.horizon,
        obs_horizon=cfg.dataset.obs_horizon, pad_before=cfg.dataset.pad_before,
        pad_after=cfg.dataset.pad_after, history_steps=cfg.dataset.get("history_steps"),
    )
    return buffer, train, val, split, hand_data


def prepare_policy_data(cfg):
    spec = resolve_normalization_spec(cfg)
    if spec["action"] not in {"auto", "limits"}:
        raise ValueError("DQ/VQ hand prototypes require action normalization auto or limits")
    if cfg.agent.tcp_dim != 7:
        raise ValueError("DQ/VQ requires tcp_dim=7")
    buffer, train, val, split, hand_data = prepare_hand_source(
        cfg, "dexmani_policy.agents.policies.dqrise.DQRISEPolicy", ["action"],
    )
    validate_fields(buffer, train | val, {"action": slice(7, 19)})

    def read_hand(mask):
        chunks = [values for _, values in buffer.iter_chunks("action", columns=slice(7, 19), episode_mask=mask)]
        return np.concatenate(chunks) if chunks else np.empty((0, 12), dtype=np.float32)

    train_hand, val_hand = read_hand(train), read_hand(val)
    normalizer = LinearNormalizer()
    normalizer.fit_field("hand", (train_hand[start:start + 8192] for start in range(0, len(train_hand), 8192)), mode="limits")
    metadata = dict(
        hand_data=hand_data, split_manifest=split,
        policy_config=OmegaConf.to_container(cfg, resolve=False), normalization_spec=spec,
    )
    metadata["policy_config"]["dataset"] = OmegaConf.to_container(cfg.dataset, resolve=True)
    metadata["policy_config"]["dataset"]["zarr_path"] = buffer.path
    train_hand = normalizer["hand"].normalize(train_hand).numpy()
    val_hand = normalizer["hand"].normalize(val_hand).numpy()
    return train_hand, val_hand, normalizer, metadata


def prepare_prior_data(cfg):
    if (cfg.dataset.horizon, cfg.dataset.obs_horizon, cfg.agent.n_action_steps) != (1, 1, 1):
        raise ValueError("LAMP requires H=To=Ta=1")
    if cfg.dataset.history_steps != cfg.agent.history_steps or cfg.dataset.history_steps < 1:
        raise ValueError("LAMP history_steps must match and be positive")
    spec = resolve_normalization_spec(cfg)
    if spec["action"] != "identity" or spec["joint_state"] != "gaussian":
        raise ValueError("LAMP requires identity action and gaussian joint_state")
    buffer, train, val, split, hand_data = prepare_hand_source(
        cfg, "dexmani_policy.agents.policies.lamp.LAMPPolicy", ["action", "joint_state"],
    )
    validate_fields(buffer, train | val, {"action": slice(7, 19)})
    actions, bootstrap = load_hand_history(buffer, train | val)
    starts = np.r_[0, buffer.episode_ends[:-1]]
    for episode in np.flatnonzero(train | val):
        check_finite(buffer, "joint_state", int(starts[episode]), bootstrap[episode:episode + 1], slice(7, 19))

    def rows(mask):
        if not mask.any():
            return np.empty(0, dtype=np.int64)
        return np.concatenate([np.arange(starts[i], buffer.episode_ends[i]) for i in np.flatnonzero(mask)])

    return dict(
        hand_actions=actions, bootstrap=bootstrap, episode_ends=buffer.episode_ends,
        history_steps=cfg.dataset.history_steps, train_rows=rows(train), val_rows=rows(val),
        hand_data=hand_data, split_manifest=split,
    )
