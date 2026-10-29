"""Observation transfer and bounded training checks for AFRO, CordViP and SAT."""

import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import POINT_CLOUD_FIELDS, load_policy_config
from dexmani_policy.common.pytorch_util import dict_apply
from dexmani_policy.test_tools.smoke_checks import (
    expect_value_error, validate_data_batch, validate_training,
)


def validate_pretrained_observation_transfer(config):
    import hydra
    import zarr
    from dexmani_policy.agents.policies.afro import AFROPolicy
    from dexmani_policy.training.common.data import build_dataset_and_normalizer
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
        dataset, normalizer, description = build_dataset_and_normalizer(cfg)
        validation = dataset.get_validation_dataset()
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
        for variant in variants:
            cfg.agent.pretrained_path = None if variant == 'scratch' else str(path)
            if is_afro:
                cfg.agent.freeze_visual = variant == 'frozen'
            cfg.workspace.output_dir = str(directory / variant)
            normalizer.load_state_dict(downstream, strict=True)
            assert validation.get_normalizer(cfg.normalization) is normalizer
            with patch('torch.load', wraps=torch.load) as load, closing(
                PolicyTrainer(cfg, dataset, normalizer, description)
            ) as trainer:
                assert load.call_count == (0 if variant == 'scratch' else 1)
                assert dataset.get_normalizer(cfg.normalization) is normalizer
                assert validation.get_normalizer(cfg.normalization) is normalizer
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

        report = validate_training(cfg, max_updates=1, resume_updates=1, batch_check=check_batch)
        assert not path.exists() and len(report['training']) == 2
        return {'status': 'passed', 'variants': variants, 'training': report['training']}


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
