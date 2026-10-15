import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.normalization import LinearNormalizer, normalize_tensor, validate_normalizer_state
from dexmani_policy.datasets.preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.deployment.runtime import LoadedPolicy, PolicyInfo
from dexmani_policy.env_runner.base_runner import BaseRunner
from dexmani_policy.training.checkpoint import CheckpointStore, TrainCheckpoint
from dexmani_policy.utils.config import checkpoint_source_config, resolve_input_recipe


def make_normalizer(width):
    normalizer = LinearNormalizer()
    normalizer.fit({'action': torch.arange(5 * width, dtype=torch.float32).reshape(5, width)})
    return normalizer


@pytest.mark.parametrize('width', [19, 21, 28])
def test_normalizer_roundtrip_and_restore(width):
    n = make_normalizer(width)
    x = torch.randn(2, 16, width)
    restored = LinearNormalizer()
    restored.load_state_dict(n.state_dict())
    validate_normalizer_state(restored, {'action': 'limits'}, {'action': width})
    torch.testing.assert_close(restored['action'].unnormalize(restored['action'].normalize(x)), x,
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('scale,offset', [
    (torch.ones(2), torch.zeros(2)), (torch.ones(19), torch.zeros(2)),
    (torch.ones(1, 19), torch.zeros(1, 19)), (torch.tensor(1.), torch.tensor(0.)),
    (torch.ones(5, 3), torch.zeros(5, 3)),
    (torch.zeros(19), torch.zeros(19)), (torch.full((19,), float('nan')), torch.zeros(19)),
    (torch.ones(19), torch.full((19,), float('inf'))),
])
def test_normalizer_rejects_invalid_state(scale, offset):
    n = LinearNormalizer()
    n.params_dict['action'] = torch.nn.ParameterDict({'scale': scale, 'offset': offset})
    with pytest.raises(ValueError):
        validate_normalizer_state(n, {'action': 'limits'}, {'action': 19})


def test_normalizer_cannot_repartition_time():
    with pytest.raises(ValueError, match='feature shape'):
        normalize_tensor(torch.ones(2, 16, 19), {'scale': torch.ones(2), 'offset': torch.zeros(2)})
    n = LinearNormalizer()
    n.fit({'force': torch.randn(2, 5, 120, 3)})
    validate_normalizer_state(n, {'force': 'limits', 'rgb': 'identity'}, {'force': 3})
    from dexmani_policy.agents.normalization import normalization_dims_from_config
    cfg = {'agent': {'action_dim': 19}, 'normalization': {'finger_tactile_force': 'gaussian'}}
    assert normalization_dims_from_config(cfg)['finger_tactile_force'] == 15
    for key, width in (('finger_tactile_force', 15), ('fingertip_xyz', 15), ('eef_pose', 9)):
        cfg['normalization_dims'] = {key: width}
        assert normalization_dims_from_config(cfg)[key] == width
        cfg['normalization_dims'] = {key: 3}
        with pytest.raises(ValueError, match=f'{key} feature width must be {width}'):
            normalization_dims_from_config(cfg)


@pytest.mark.parametrize('tensor', [False, True])
def test_runner_frame_commit_is_atomic(tensor):
    runner = BaseRunner(3, 1, sensor_modalities=['a', 'b', 'text'])
    array = torch.tensor if tensor else np.array
    for i in (1, 2):
        runner.update_obs({'a': array([i]), 'b': array([i]), 'text': str(i)})
    snapshot = copy.deepcopy(runner.__dict__)
    for invalid in ({'a': array([3]), 'text': '3'},
                    {'a': array([3]), 'b': array([3, 4]), 'text': '3'}):
        with pytest.raises(ValueError):
            runner.update_obs(invalid)
        assert runner._obs_cursor == snapshot['_obs_cursor']
        assert runner._obs_count == snapshot['_obs_count']
        assert runner._obs_str_buffer == snapshot['_obs_str_buffer']
        for key, value in runner._obs_buffer.items():
            np.testing.assert_array_equal(value, snapshot['_obs_buffer'][key])
    for i in (3, 4):
        runner.update_obs({'a': array([i]), 'b': array([i]), 'text': str(i)})
    np.testing.assert_array_equal(runner.get_stacked_obs()['a'][:, 0], [2, 3, 4])


class OfflineAgent(BaseAgent):
    consumed_observation_fields = ('joint_state', 'point_cloud')

    def __init__(self, key, aux=False):
        width = (19 if key == 'action' else 21) + (9 if aux else 0)
        super().__init__(torch.nn.Identity(), torch.nn.Identity(), 4, 2, 2, width)
        self.action_key, self.use_aux_ee, self.ee_dim = key, aux, 9
        self.normalizer = make_normalizer(width)
        self.set_normalization_spec({'action': 'limits', 'joint_state': 'identity', 'point_cloud': 'identity'})
        self.output_width = width
        self.eval()

    @property
    def control_action_dim(self):
        return self.action_dim - (9 if self.use_aux_ee else 0)

    def predict_action(self, obs_dict, **kwargs):
        self.seen = self.preprocess(obs_dict)
        self.kwargs = kwargs
        return {'pred_action': torch.arange(4.).reshape(1, 4, 1).expand(1, 4, self.output_width)}


def runtime_inputs(key='action', aux=False):
    agent = OfflineAgent(key, aux)
    cfg = {'action_key': key, 'agent': dict(horizon=4, n_obs_steps=2, n_action_steps=2,
            action_dim=agent.action_dim, use_aux_ee=aux),
           'dataset': {'sensor_modalities': list(agent.consumed_observation_fields)},
           'normalization': dict(agent.normalization_spec)}
    info = PolicyInfo(Path('.'), Path('unused.pt'), 'offline', 'test', agent.consumed_observation_fields,
                      2, 2, 4, 'joint' if key == 'action' else 'eef', None, None, None, 'raw', 1,
                      agent.action_dim, agent.control_action_dim, rgb_preprocessing_kwargs(cfg['dataset']),
                      dict(agent.normalization_spec), None)
    obs = {'joint_state': np.broadcast_to(np.arange(4.)[:, None], (4, 19)).copy(),
           'point_cloud': np.zeros((4, 5, 6))}
    return agent, cfg, info, obs


@pytest.mark.parametrize('key,aux', [('action', False), ('action_ee', False), ('action', True)])
def test_real_latest_history_complete_future_and_rtc(key, aux):
    agent, cfg, info, obs = runtime_inputs(key, aux)
    policy = LoadedPolicy(agent, cfg, info, device='cpu', seed=0)
    future = policy.predict(obs, rtc_prefix=np.zeros((2, agent.control_action_dim)), delay_steps=1)
    assert future.shape == (3, agent.control_action_dim)
    np.testing.assert_array_equal(future[:, 0], [1, 2, 3])
    np.testing.assert_array_equal(agent.seen['joint_state'][:, 0], [2, 3])
    assert agent.kwargs['rtc_prefix'].shape[-1] == agent.control_action_dim
    policy.predict({k: v[:2] for k, v in obs.items()})
    for bad in ({k: v[:1] for k, v in obs.items()}, dict(obs, point_cloud=obs['point_cloud'][:3])):
        with pytest.raises(ValueError, match='equal raw lengths'):
            policy.predict(bad)
    agent.output_width += 1
    with pytest.raises(ValueError, match='pred_action'):
        policy.predict(obs)


@pytest.mark.parametrize('changes', [{'action_mode': 'joint'}, {'action_mode': 'unknown'},
                                   {'n_obs_steps': 1}, {'action_dim': 19},
                                   {'observation_fields': ('joint_state',)}])
def test_real_rejects_info_mismatch(changes):
    agent, cfg, info, obs = runtime_inputs('action_ee')
    with pytest.raises(ValueError):
        LoadedPolicy(agent, cfg, replace(info, **changes), device='cpu', seed=0)


def test_real_rejects_independent_normalizer_layout():
    agent, cfg, info, _ = runtime_inputs()
    agent.normalizer.fit_field_chunks('point_cloud', [torch.randn(2, 5, 6)], mode='gaussian')
    cfg['normalization']['point_cloud'] = 'gaussian'
    agent.set_normalization_spec(cfg['normalization'], {'point_cloud': 6})
    cfg['normalization_dims'] = {'point_cloud': 2}
    info = replace(info, normalization=cfg['normalization'])
    with pytest.raises(ValueError, match='normalization feature widths'):
        LoadedPolicy(agent, cfg, info, device='cpu', seed=0)


def test_real_rgb_preprocessing_and_failed_load_cleanup(monkeypatch):
    from dexmani_policy.agents.core.dp import DPAgent
    from dexmani_policy.deployment.runtime import load_policy
    agent, cfg, info, _ = runtime_inputs()
    closed = []
    agent.close = lambda: closed.append(True)
    monkeypatch.setattr('dexmani_policy.agents.loader.restore_policy_agent', lambda *a, **kw: agent)
    with pytest.raises(ValueError):
        load_policy(cfg, replace(info, action_mode='eef'), device='cpu')
    assert closed == [True]
    args = dict(horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
                state_dim=19, rgb_backbone_name='resnet',
                rgb_backbone_config={'load_pretrained': False, 'image_size': 32},
                down_dims=[32, 64], diffusion_step_embed_dim=32)
    model = DPAgent(**args)
    model.action_key = 'action'
    model.normalizer = make_normalizer(19)
    spec = {'action': 'limits', 'joint_state': 'identity', 'rgb': 'identity'}
    model.set_normalization_spec(spec)
    cfg.update(agent=args, dataset={'sensor_modalities': ['joint_state', 'rgb']}, normalization=spec)
    info = replace(info, observation_fields=('joint_state', 'rgb'), normalization=spec)
    LoadedPolicy(model, cfg, info, device='cpu', seed=0)
    model.obs_encoder.image_processor.image_mean += .1
    with pytest.raises(ValueError, match='RGB numerical preprocessing'):
        LoadedPolicy(model, cfg, info, device='cpu', seed=0)


def checkpoint_at(tmp_path):
    config = {'training': {'lr_warmup_steps': 2, 'lr_scheduler': 'cosine', 'lr_min_ratio': .01, 'seed': 42},
              'ema': {'power': .75}, 'agent': {'action_dim': 19}, 'action_key': 'action',
              'dataset': {'zarr_path': '/original/data'},
              'workspace': {'_target_': 'dexmani_policy.training.workspace.TrainWorkspace',
                            'output_dir': '/original/run',
                            'wandb_cfg': {'_target_': 'dexmani_policy.training.workspace.WandbConfig',
                                          'project': 'test', 'group': 'test', 'name': 'test',
                                          'id': 'test', 'resume': 'allow', 'mode': 'online'}}}
    store = CheckpointStore(tmp_path / 'checkpoints')
    checkpoint = TrainCheckpoint(0, 0, 0, {'x': torch.zeros(1)}, None, {}, {}, {}, None, None, [{}], config)
    path = store.save('test.pt', checkpoint)
    OmegaConf.save(OmegaConf.create(config), tmp_path / 'config.yaml')
    return store, path, config


@pytest.mark.parametrize('key,value', [('training.lr_warmup_steps', 4), ('training.lr_scheduler', 'constant'),
                                     ('training.lr_min_ratio', .2),
                                     ('ema.power', .5), ('agent.action_dim', 21), ('action_key', 'action_ee')])
def test_checkpoint_rejects_conflicting_sidecar(tmp_path, key, value):
    store, path, cfg = checkpoint_at(tmp_path)
    bad = OmegaConf.create(cfg)
    OmegaConf.update(bad, key, value)
    OmegaConf.save(bad, tmp_path / 'config.yaml')
    with pytest.raises(ValueError, match='source recipe mismatch'):
        store.load(path)


def test_checkpoint_binding_overrides_and_legacy(tmp_path):
    store, path, cfg = checkpoint_at(tmp_path)
    assert store.load(path).effective_config == cfg
    incoming = OmegaConf.create(dict(cfg, resume_from=str(path)))
    resumed = resolve_input_recipe(incoming, overrides=['workspace.output_dir=/new/run', 'dataset.zarr_path=/new/data'])
    assert resumed.workspace.output_dir == '/new/run'
    assert resumed.dataset.zarr_path == '/new/data'
    payload = store.load_payload(path)
    payload.pop('effective_config')
    payload['_format'] = 'simple.v3'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Strict resume requires'):
        store.load(path)
    payload['effective_config'] = cfg  # Retrofitting a sidecar cannot certify v3.
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Strict resume requires'):
        store.load(path)
    with pytest.warns(UserWarning, match='unbound'):
        assert checkpoint_source_config(payload, path, strict_resume=False) == cfg


@pytest.mark.parametrize('override', [
    'workspace.wandb_cfg=null',
    'workspace.wandb_cfg.mode=disabled',
    'workspace.wandb_cfg={_target_:dexmani_policy.training.workspace.WandbConfig,'
    'project:local,group:test,name:resumed,id:resumed,resume:allow,mode:disabled}',
])
def test_resume_wandb_operational_overrides(tmp_path, monkeypatch, override):
    import hydra
    from dexmani_policy.utils.config import recipe_content

    _, path, cfg = checkpoint_at(tmp_path)
    resumed = resolve_input_recipe(OmegaConf.create({'resume_from': str(path)}), overrides=[override])
    assert recipe_content(resumed) == recipe_content(cfg)
    if override.endswith('=null'):
        assert resumed.workspace.wandb_cfg is None

        def forbidden(*args, **kwargs):
            pytest.fail('Disabled W&B must not construct a remote logger or require login')

        monkeypatch.setattr('dexmani_policy.training.workspace.WandbLogger', forbidden)
        resumed.workspace.output_dir = str(tmp_path / 'resumed')
        workspace = hydra.utils.instantiate(resumed.workspace)
        try:
            assert workspace.wandb_logger is None
            workspace.save_hydra_config(resumed)
        finally:
            workspace.close()
    else:
        wandb_cfg = hydra.utils.instantiate(resumed.workspace.wandb_cfg)
        assert wandb_cfg.mode == 'disabled'
        assert wandb_cfg.project == ('local' if 'project:local' in override else 'test')


@pytest.mark.parametrize('override', [
    'agent.action_dim=21', 'training.lr_scheduler=constant', 'ema.power=0.5',
    'optimizer.lr=0.1', 'dataset.sampling_strategy=weighted',
    'workspace={output_dir:/new/run}',
])
def test_resume_wandb_override_does_not_allow_recipe_changes(tmp_path, override):
    _, path, _ = checkpoint_at(tmp_path)
    with pytest.raises(ValueError, match='forbidden override'):
        resolve_input_recipe(OmegaConf.create({'resume_from': str(path)}),
                             overrides=['workspace.wandb_cfg=null', override])


@pytest.mark.parametrize('alpha', [float('nan'), float('inf'), -float('inf'), .99, -1.])
def test_time_shift_rejects_nonfinite_or_below_one(alpha):
    from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
    from dexmani_policy.agents.action_decoders.time_sampler import shift_time_to_noise
    with pytest.raises(ValueError, match='finite and >= 1'):
        RectifiedFlow(torch.nn.Identity(), time_shift_alpha=alpha)
    with pytest.raises(ValueError, match='finite and >= 1'):
        shift_time_to_noise(torch.tensor([0., .5, 1.]), alpha)


@pytest.mark.parametrize('alpha', [1., 1.5, 4.])
def test_time_shift_preserves_formula_and_endpoints(alpha):
    from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
    from dexmani_policy.agents.action_decoders.time_sampler import shift_time_to_noise
    decoder = RectifiedFlow(torch.nn.Identity(), time_shift_alpha=alpha)
    t = torch.tensor([0., .1, .5, .9, 1.], dtype=torch.float64)
    actual = shift_time_to_noise(t, decoder.time_shift_alpha)
    expected = torch.tensor([x / (1 + (alpha - 1) * (1 - x)) for x in t.tolist()],
                            dtype=t.dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual[0] == 0 and actual[-1] == 1
    if alpha == 1.:
        assert actual is t
