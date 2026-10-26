from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import TrainingCheckpoint, save_training_checkpoint
from dexmani_policy.common.config import load_policy_config
from dexmani_policy.training.builder import bind_defaults


def tiny_policy():
    cfg = load_policy_config('dp3', overrides=[
        'horizon=8', 'n_action_steps=3', 'training.device=cpu',
        'agent.num_points=8', 'agent.pc_out_dim=8', '+agent.state_out_dim=8',
        'agent.down_dims=[8,16]', 'agent.diffusion_step_embed_dim=16',
        'agent.kernel_size=3', 'agent.n_groups=4', 'agent.num_training_steps=10',
        'agent.num_inference_steps=2', 'eval.inference_steps=2',
    ])
    cfg.workspace.output_dir = '/tmp/unused_policy_fixture'
    OmegaConf.resolve(cfg)
    agent = hydra.utils.instantiate(cfg.agent)
    widths = {'action': 19, 'joint_state': 19, 'point_cloud': 6}
    for key, width in widths.items():
        agent.normalizer.fit_field(key, [np.tile([[-1.0], [1.0]], (1, width))], mode='limits')
    agent.set_normalization_spec(dict(cfg.normalization), widths)
    bind_defaults(cfg.agent, type(agent))
    with open_dict(cfg):
        cfg.normalization_dims = widths
        cfg.policy_metadata = agent.inference_metadata
    agent.eval()
    info = SimpleNamespace(
        observation_fields=('joint_state', 'point_cloud'), rgb_preprocessing={},
        n_obs_steps=2, pc_dim=6, future_steps=7, inference_steps=2,
        hand_history_steps=None, pointcloud_config={'num_points': 8},
    )
    obs = {
        'joint_state': np.zeros((2, 19), np.float32),
        'point_cloud': np.linspace(-.5, .5, 96, dtype=np.float32).reshape(2, 8, 6),
    }
    return agent, cfg, info, obs


def checkpoint_fixture(root, *, step=20, pct=20, total=100, resume_from=None):
    agent, cfg, info, obs = tiny_policy()
    with open_dict(cfg):
        cfg.training.loop.total_train_steps = total
        cfg.resume_from = resume_from
    config = OmegaConf.to_container(cfg, resolve=True)
    checkpoint = TrainingCheckpoint(
        epoch=0, global_step=step, next_micro_step=0,
        model_state=agent.state_dict(), ema_model_state=agent.state_dict(),
        optimizer_state={}, scheduler_state={}, resume_contract={},
        ema_updater_step=step, rng_states=[], effective_config=config,
    )
    tag = 'bounded' if pct is None else f'milestone={pct}pct'
    path = Path(root) / 'checkpoints' / f'epoch=0-step={step}-{tag}.pt'
    save_training_checkpoint(path, checkpoint)
    info.saved_config = config
    info.checkpoint_path = path
    info.weights = 'ema'
    return checkpoint, agent, info, obs
