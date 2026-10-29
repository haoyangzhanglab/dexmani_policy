"""Observation transfer and bounded training checks for AFRO, CordViP and SAT."""

import subprocess
import sys
import tempfile
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import POINT_CLOUD_FIELDS, load_policy_config
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.test_tools.smoke_checks import (
    expect_value_error, validate_data_batch, validate_training,
)


def validate_pretrained_observation_transfer(config):
    import hydra
    import zarr
    from dexmani_policy.agents.policies.afro import AFROPolicy
    from dexmani_policy.training.common.data import build_dataset
    from dexmani_policy.training.trainer import PolicyTrainer
    from dexmani_policy.training.pretrain.obs.afro import AFROPretrain
    from dexmani_policy.training.pretrain.obs.cordvip import CordViPPretrain
    from dexmani_policy.training.pretrain.obs.common import save_run

    is_afro = issubclass(hydra.utils.get_class(config.agent._target_), AFROPolicy)
    method = 'afro' if is_afro else 'cordvip'
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.agent.update(dict(down_dims=[16, 32], diffusion_step_embed_dim=16, num_inference_steps=2))
    cfg.dataset.augmentation_cfg = None
    cfg.dataset.max_train_episodes = None
    cfg.dataloader.update(dict(batch_size=2, num_workers=0, pin_memory=False))
    cfg.training.update(dict(use_compile=False, use_bfloat16=False, use_ema=True, lr_warmup_steps=0))
    cfg.training.loop.update(dict(total_train_steps=10, log_interval_steps=1, gradient_accumulation_steps=1))
    cfg.optimizer.update(dict(lr=1e-4, obs_lr=1e-4))
    cfg.workspace.wandb_cfg = None
    with open_dict(cfg):
        cfg.ema = {'_target_': 'dexmani_policy.training.common.optimization.EMA'}
    fields = ('point_cloud',) if is_afro else ('hand_pointcloud', 'object_pointcloud', 'joint_state')
    spec = {key: cfg.normalization[key] for key in fields}
    pretrained = AFROPretrain(spec) if is_afro else CordViPPretrain(spec, n_obs_steps=cfg.n_obs_steps)
    encoder = pretrained.visual if is_afro else pretrained.obs_encoder
    for index, (key, field) in enumerate(pretrained.normalizer.fields.items()):
        field.scale.fill_(2.5 + index)
        field.offset.fill_(0 if spec[key] in {'rms', 'max_abs'} else -4.5 - index)
    inherited = {key: value.clone() for key, value in pretrained.normalizer.state_dict().items()}

    with tempfile.TemporaryDirectory(prefix=f'{method}-transfer-') as temporary:
        directory = Path(temporary)
        cfg.zarr_path = cfg.dataset.zarr_path = str(directory / 'data.zarr')
        frames = max(8, cfg.horizon + 2)
        ends = [frames, 2 * frames]
        cfg.dataset.split_manifest = dict(episode_ends=ends, train_indices=[0], val_indices=[1])
        root = zarr.open_group(cfg.dataset.zarr_path, mode='w')
        root.create_dataset('meta/episode_ends', data=np.array(ends))
        rng = np.random.default_rng(7)
        for key in (*cfg.dataset.sensor_modalities, 'action'):
            shape = (ends[-1], 8, cfg.agent.pc_dim) if key in POINT_CLOUD_FIELDS else (ends[-1], 19)
            root.create_dataset(f'data/{key}', data=(rng.normal(size=shape) * 3 + 12).astype(np.float32))
        export_cfg = load_policy_config(f'pretrain/obs/{method}')
        export_cfg.normalization = spec
        if not is_afro:
            export_cfg.n_obs_steps = cfg.n_obs_steps
        export_cfg.workspace.output_dir = str(directory)
        export_cfg.training.device = 'cpu'
        optimizer = torch.optim.AdamW(pretrained.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        save_run(pretrained, encoder, optimizer, scheduler, export_cfg, 0, 0)
        path = directory / 'encoder.pt'
        payload = torch.load(path, weights_only=True)
        assert set(payload) == {'encoder_state', 'normalizer_state'}
        torch.testing.assert_close(payload['normalizer_state'], inherited, rtol=0, atol=0)
        assert all(t.device.type == 'cpu' and not t.requires_grad for t in payload['encoder_state'].values())
        assert all(t.device.type == 'cpu' and t.dtype == torch.float32 and not t.requires_grad
                   for t in payload['normalizer_state'].values())
        cfg.agent.pretrained_path = str(path)
        dataset, description = build_dataset(cfg)
        normalizer = dataset.get_normalizer(cfg.normalization)
        downstream = {key: value.clone() for key, value in normalizer.state_dict().items()}
        expected = {**downstream, **inherited}
        assert all(not torch.equal(value, downstream[key]) for key, value in inherited.items()
                   if key.endswith('.scale'))
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        with torch.no_grad():
            normalized = {key: pretrained.normalizer[key].normalize(batch['obs'][key])
                          if key in pretrained.normalizer.fields else batch['obs'][key] for key in fields}
            features = encoder(normalized['point_cloud'].flatten(0, 1))['global_token'] if is_afro else None
        del optimizer, scheduler, encoder, pretrained, payload

        def check_normalizer(value):
            torch.testing.assert_close(value.state_dict(), expected, rtol=0, atol=0, check_device=False)
            assert all(t.dtype == torch.float32 for t in value.buffers())

        variants = ('frozen', 'finetune', 'scratch') if is_afro else ('finetune', 'scratch')
        fitted_fields = {}
        fit_field = LinearNormalizer.fit_field
        for variant in variants:
            cfg.agent.pretrained_path = None if variant == 'scratch' else str(path)
            if is_afro:
                cfg.agent.freeze_visual = variant == 'frozen'
            cfg.workspace.output_dir = str(directory / variant)
            dataset, description = build_dataset(cfg)
            validation = dataset.get_validation_dataset()
            fitted_fields[variant] = []

            def fit(normalizer, key, *args, **kwargs):
                fitted_fields[variant].append(key)
                return fit_field(normalizer, key, *args, **kwargs)

            with patch.object(LinearNormalizer, 'fit_field', fit), patch('torch.load', wraps=torch.load) as load, closing(
                PolicyTrainer(cfg, dataset, description)
            ) as trainer:
                assert load.call_count == (0 if variant == 'scratch' else 1)
                inherited_fields = {key.split('.')[1] for key in inherited} if variant != 'scratch' else set()
                assert set(fitted_fields[variant]) == {
                    key for key, mode in cfg.normalization.items() if mode != 'identity' and key not in inherited_fields
                }
                normalizer = dataset.get_normalizer(cfg.normalization)
                assert dataset.get_normalizer(cfg.normalization) is normalizer
                assert validation.get_normalizer(cfg.normalization) is normalizer
                expect_value_error(lambda: dataset.get_normalizer(cfg.normalization, initial_state=inherited))
                assert all(t.device.type == 'cpu' for t in normalizer.buffers())
                assert trainer.model.normalizer is not normalizer
                torch.testing.assert_close(trainer.model.state_dict(), trainer.ema_model.state_dict(), rtol=0, atol=0)
                if variant == 'scratch':
                    torch.testing.assert_close(normalizer.state_dict(), downstream, rtol=0, atol=0)
                    torch.testing.assert_close(trainer.model.normalizer.state_dict(), downstream, check_device=False)
                else:
                    for value in (normalizer, trainer.model.normalizer, trainer.ema_model.normalizer):
                        check_normalizer(value)
                    for key, value in normalizer.state_dict().items():
                        assert value.data_ptr() != trainer.model.normalizer.state_dict()[key].data_ptr()
                    processed = trainer.model.preprocess(dict_apply(batch['obs'], lambda t: t.to(trainer.device)))
                    for key in fields:
                        torch.testing.assert_close(processed[key].cpu(), normalized[key].flatten(0, 1))
                    if is_afro:
                        with torch.no_grad():
                            actual = trainer.model.obs_encoder.visual(processed['point_cloud'])['global_token']
                        torch.testing.assert_close(actual.cpu(), features)
                        assert all(p.requires_grad == (variant == 'finetune')
                                   for p in trainer.model.obs_encoder.visual.parameters())
                # Identity normalization leaves channel selection to the fixed XYZ encoder.
                xyz = {key: value.flatten(0, 1).to(trainer.device) for key, value in batch['obs'].items()}
                rgb = {key: torch.cat((value, torch.ones_like(value)), dim=-1) if key in POINT_CLOUD_FIELDS else value
                       for key, value in xyz.items()}
                encoder = trainer.model.obs_encoder.eval()
                with torch.no_grad():
                    torch.testing.assert_close(encoder(rgb), encoder(xyz), rtol=0, atol=0)
            del trainer

        if is_afro:
            cfg.agent.freeze_visual = True
            invalid = hydra.utils.instantiate(cfg.agent)
            expect_value_error(lambda: invalid.initialize_training(description))
            del invalid
        cfg.agent.pretrained_path = str(path)
        incomplete = directory / 'missing_normalizer.pt'
        torch.save({'encoder_state': torch.load(path, weights_only=True)['encoder_state']}, incomplete)
        invalid = hydra.utils.instantiate(cfg.agent, pretrained_path=str(incomplete))
        try:
            invalid.initialize_training(description)
        except KeyError as error:
            assert error.args == ('normalizer_state',)
        else:
            raise AssertionError('Encoder package silently accepted missing normalization')
        del invalid

        def check_batch(batch, dataset, run_cfg):
            validate_data_batch(batch, dataset, run_cfg)
            check_normalizer(dataset.get_normalizer(run_cfg.normalization))
            # validate_training calls this after saving, before inference restoration/resume.
            path.unlink(missing_ok=True)

        report = validate_training(
            cfg, max_updates=1, resume_updates=1, batch_check=check_batch,
            deployment_check=None if is_afro else validate_cordvip_deployment,
        )
        assert not path.exists() and len(report['training']) == 2
        return {'status': 'passed', 'variants': variants, 'fitted_fields': fitted_fields, **report}


def validate_sat_training(config):
    import zarr

    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.agent.update(dict(n_layers=1, num_points=16, num_inference_steps=2))
    cfg.agent.pc_encoder_config.update(dict(channels=48, num_heads=2, num_patches=4, group_size=4, depth=1))
    cfg.dataset.augmentation_cfg = None
    cfg.dataset.max_train_episodes = None
    cfg.dataloader.update(dict(batch_size=2, num_workers=0, pin_memory=False))
    cfg.training.update(dict(use_compile=False, use_bfloat16=False, lr_warmup_steps=0))
    cfg.training.loop.update(dict(total_train_steps=10, log_interval_steps=1, gradient_accumulation_steps=1))
    frames = max(20, cfg.horizon + 2)
    ends = [frames, 2 * frames]
    with tempfile.TemporaryDirectory(prefix='sat-contract-') as directory:
        root = zarr.open_group(directory, mode='w')
        root.create_dataset('meta/episode_ends', data=np.array(ends))
        rng = np.random.default_rng(11)
        shapes = {'point_cloud': (ends[-1], 16, cfg.agent.pc_dim),
                  'joint_state': (ends[-1], 19), 'action': (ends[-1], 19)}
        for key, shape in shapes.items():
            root.create_dataset(f'data/{key}', data=rng.normal(size=shape).astype(np.float32))
        cfg.zarr_path = cfg.dataset.zarr_path = directory
        cfg.dataset.split_manifest = dict(episode_ends=ends, train_indices=[0], val_indices=[1])
        return validate_training(cfg, max_updates=1, resume_updates=1)


def _check_warmup(runtime, observation, *, action_context=None, **warmup_kwargs):
    runtime.predict(observation, action_context=action_context)
    state = runtime.generator.get_state()
    expected = runtime.predict(observation, action_context=action_context)
    runtime.generator.set_state(state)
    timings = runtime.warmup(observation, samples=2, action_context=action_context, **warmup_kwargs)
    assert torch.equal(runtime.generator.get_state(), state)
    assert set(timings) == {'bootstrap', 'steady'}
    assert all(len(values) == 2 and all(np.isfinite(t) and t > 0 for t in values) for values in timings.values())
    np.testing.assert_array_equal(runtime.predict(observation, action_context=action_context), expected)


def validate_cordvip_deployment(experiment, dataset, device):
    from dexmani_policy.common.config import load_experiment_config
    from dexmani_policy.deployment import inspect_policy, load_policy
    from dexmani_policy.env_runner.sim_runner import SimRunner
    from dexmani_policy.evaluation.config import load_eval_config

    observation = {name: value.numpy() for name, value in dataset[0]['obs'].items()}
    observation['object_pointcloud'] = observation['object_pointcloud'][:, :5]
    data_path = Path(dataset.zarr_path)
    data_path.rename(data_path.with_name('unavailable.zarr'))
    assert 'real_runtime' not in load_experiment_config(experiment)
    subprocess.run([
        sys.executable, '-c',
        'import sys; from dexmani_policy.deployment import inspect_policy; '
        'info = inspect_policy(sys.argv[1], checkpoint="latest", weights="raw"); '
        'assert "torch" not in sys.modules; assert "hand_pointcloud" in info.observation_fields',
        str(experiment),
    ], check=True)
    info = inspect_policy(experiment, checkpoint='latest', weights='raw', inference_steps=2)
    assert inspect_policy(experiment, checkpoint=info.checkpoint_path.name).checkpoint_path == info.checkpoint_path
    request = load_eval_config(experiment, info.checkpoint_path.name, {'eval': {'seeds': [0], 'inference_steps': 2}})
    assert request.checkpoint == str(info.checkpoint_path)
    check = TestCase()
    with patch.object(LinearNormalizer, 'fit_field', side_effect=AssertionError('inference must not fit')):
        for weights in ('raw', 'ema'):
            with closing(load_policy(replace(info, weights=weights), device=str(device), seed=42)) as runtime:
                result = runtime.predict(observation)
                assert result.shape == (info.future_steps, 19) and result.dtype == np.float64
                assert np.isfinite(result).all()
                _check_warmup(runtime, observation)
                rgb = {
                    name: np.concatenate((value, np.full_like(value, .5)), axis=-1).astype(np.float64)
                    if name in POINT_CLOUD_FIELDS else value.astype(np.float64)
                    for name, value in observation.items()
                }
                histories = {
                    name: np.concatenate((np.full((index + 1, *value.shape[1:]), np.nan), value))
                    for index, (name, value) in enumerate(rgb.items())
                }
                for variant in (rgb, histories):
                    torch.testing.assert_close(runtime.prepare_observations(variant), runtime.prepare_observations(observation), rtol=0, atol=0)
                    runtime.reset_episode()
                    np.testing.assert_array_equal(runtime.predict(variant), result)
                for name in observation:
                    bad = {**observation, name: observation[name][:-1]}
                    expect_value_error(lambda: runtime.prepare_observations(bad))
                    for value in (np.nan, np.inf, 1e100):
                        bad = {**observation, name: observation[name].astype(np.float64)}
                        bad[name].flat[0] = value
                        expect_value_error(lambda: runtime.prepare_observations(bad))
                for name in POINT_CLOUD_FIELDS.intersection(observation):
                    for value in (observation[name][..., :2], observation[name][:, :0], observation[name][:, 0]):
                        expect_value_error(lambda: runtime.prepare_observations({**observation, name: value}))
                expect_value_error(lambda: runtime.agent.format_action_output(torch.full((1, info.horizon, 19), float('inf'))))
                state = runtime.generator.get_state()
                with check.assertRaises(NotImplementedError):
                    runtime.warmup(observation, samples=1, rtc_guidance_cap=1.0)
                assert torch.equal(state, runtime.generator.get_state())
                runner = SimRunner(**request.runner_options)
                for frame in range(info.n_obs_steps):
                    runner.update_obs({name: value[frame] for name, value in rgb.items()})
                direct = runner.predict_action_chunk(
                    runner.get_obs_batch(device), runtime.agent, inference_steps=2,
                    action_steps=info.n_action_steps, generator=torch.Generator(device=device).manual_seed(42),
                )
                np.testing.assert_allclose(direct, result[:info.n_action_steps], rtol=0, atol=0)
    return {'status': 'passed', 'weights': ['raw', 'ema'],
            'checks': 'Torch-free inspect, no source files, XYZ/XYZRGB, histories, warmup RNG, direct SimRunner'}


def validate_runtime_modes(device):
    """Exercise RTC and action_context using actual small DP3/LAMP checkpoints."""
    import hydra
    from dexmani_policy.agents.obs_encoder.pointcloud.ops import load_pointcloud_ops
    from dexmani_policy.common.checkpoint import Checkpoint, prepare_restore_config, save_checkpoint
    from dexmani_policy.deployment import inspect_policy, load_policy

    try:
        load_pointcloud_ops()
    except ImportError as error:
        return {'status': 'not_run', 'reason': str(error)}
    check = TestCase()
    for name in ('dp3', 'lamp'):
        cfg = load_policy_config(name)
        cfg.workspace.output_dir = '/unused'
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.agent.num_points = 8
        if name == 'dp3':
            cfg.agent.update(down_dims=[16, 32], diffusion_step_embed_dim=16, pc_out_dim=16, state_out_dim=16)
        else:
            cfg.agent.update(prior_hidden_dim=16, state_out_dim=16)
            cfg.agent.pc_encoder_config.update(channels=48, num_patches=4, group_size=4, depth=1)
            # Offline inspection must use model metadata, even without Dataset history configuration.
            del cfg.dataset.history_steps
        agent = hydra.utils.instantiate(prepare_restore_config(cfg.agent))
        dims = {'action': 19, 'joint_state': 19}
        if cfg.normalization.point_cloud != 'identity':
            dims['point_cloud'] = cfg.agent.pc_dim
        agent.set_normalization_spec(cfg.normalization, dims, action_key=cfg.action_key)
        with open_dict(cfg):
            cfg.normalization_dims = dims
            cfg.policy_metadata = agent.inference_metadata
            # Legacy deployment metadata is deliberately unusable and must be ignored.
            cfg.real_runtime = {'hand_pointcloud': None, 'depth': 'unused'}
        cfg.eval.update(use_ema=False, inference_steps=1 if name == 'lamp' else 2)
        saved = OmegaConf.to_container(cfg, resolve=True)
        with tempfile.TemporaryDirectory(prefix=f'{name}-runtime-') as directory:
            if name == 'lamp':
                from dexmani_policy.training.pretrain.hand_prior.lamp import save_prior

                prior = agent.prior.requires_grad_(True)
                optimizer = torch.optim.AdamW(prior.parameters(), lr=1e-3)
                history = torch.randn(2, prior.history_steps, 12)
                target = torch.randn(2, 12)
                loss, _, _ = prior.compute_loss(history, target, beta=.01)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    mu, _ = prior.encode_history(history)
                    mse = (prior.decode(mu) - target).square().mean().item()
                save_prior(
                    Path(directory) / 'prior.pt', prior,
                    {'action_key': 'action', 'dt': .05, 'history_steps': prior.history_steps}, cfg,
                    step=1, metrics={'evaluation': {'mse': mse}},
                    split_manifest={'episode_ends': [2], 'train_indices': [0], 'val_indices': []},
                )
                prior.freeze()
            path = Path(directory) / 'checkpoints/latest.pt'
            save_checkpoint(path, Checkpoint(
                epoch=0, global_step=0, model_state=agent.state_dict(), ema_model_state=None,
                optimizer_state={}, scheduler_state={}, ema_updater_step=None, rng_state={}, effective_config=saved,
            ))
            OmegaConf.save(cfg, Path(directory) / 'config.yaml')
            del agent
            info = inspect_policy(directory, checkpoint='latest')
            rng = np.random.default_rng(8)
            observation = {
                'joint_state': rng.normal(size=(info.n_obs_steps, 19)).astype(np.float32),
                'point_cloud': rng.normal(size=(info.n_obs_steps, 16, cfg.agent.pc_dim)).astype(np.float32),
            }
            with closing(load_policy(info, device=str(device))) as runtime:
                if name == 'dp3':
                    _check_warmup(runtime, observation, rtc_delay=1, rtc_guidance_cap=.25)
                    for value in (float('nan'), float('inf'), 1e100):
                        expect_value_error(lambda: runtime.predict(
                            observation, rtc_prefix=np.full((info.future_steps, 19), value),
                        ))
                else:
                    shape = info.action_context_shapes['hand_history']
                    context = {'hand_history': np.zeros(shape, dtype=np.float32)}
                    _check_warmup(runtime, observation, action_context=context)
                    expect_value_error(lambda: runtime.predict(observation))
                    expect_value_error(lambda: runtime.predict(observation, action_context={'hand_history': np.zeros((1, 12))}))
                    for value in (float('nan'), float('inf'), 1e100):
                        expect_value_error(lambda: runtime.predict(
                            observation, action_context={'hand_history': np.full(shape, value)},
                        ))
                    state = runtime.generator.get_state()
                    with check.assertRaises(NotImplementedError):
                        runtime.warmup(observation, samples=1, action_context=context, rtc_guidance_cap=.25)
                    assert torch.equal(state, runtime.generator.get_state())
                    runtime.info = inspect_policy(directory, checkpoint='latest', inference_steps=2)
                    expect_value_error(lambda: runtime.predict(observation, action_context=context))
    return {'status': 'passed', 'checks': 'DP3 RTC, LAMP context/NFE, explicit warmup RNG, legacy metadata ignored'}
