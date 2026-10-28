"""Validate initialization, saved affine and self-contained raw/EMA policy restoration."""
import argparse
from contextlib import closing, ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from dexmani_policy.common.checkpoint import load_checkpoint, load_policy_state
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.smoke_test import validate_policy_restoration
from dexmani_policy.training.common.data import build_dataset_and_normalizer, close_data_loader
from dexmani_policy.training.pretrain.obs.runtime import prepare_data
from dexmani_policy.training.trainer import PolicyTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['afro', 'cordvip'], required=True)
    parser.add_argument('--fixture', type=Path, default=Path('/tmp/dexmani-obs-validation'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    method = args.method
    run = args.run or Path(f'experiments/{method}/pick_apple_messy/obs_smoke')
    checkpoint_path = (run / 'checkpoints/latest.pt').resolve()
    checkpoint = load_checkpoint(checkpoint_path, mmap=True)
    cfg = OmegaConf.load(run / 'config.yaml')
    pretrain_path = Path(cfg.agent.pretrained_path)
    artifact = torch.load(pretrain_path, map_location='cpu', weights_only=True)
    results = {'method': method, 'step': checkpoint.global_step}
    with ExitStack() as resources:
        cfg.workspace.output_dir = resources.enter_context(TemporaryDirectory(prefix=f'{method}-initialization-'))
        cfg.workspace.wandb_cfg = None
        set_seed(cfg.training.seed)
        dataset, normalizer, description = build_dataset_and_normalizer(cfg)
        trainer = resources.enter_context(closing(PolicyTrainer(cfg, dataset, normalizer, description)))
        encoder = trainer.model.obs_encoder.visual if method == 'afro' else trainer.model.obs_encoder.encoder
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value.cpu(), artifact['encoder_state'][key], rtol=0, atol=0)
        results['pretrained_loaded'] = True
        if method == 'afro':
            assert all(not p.requires_grad for p in encoder.parameters())
            assert all(p.requires_grad for p in trainer.model.obs_encoder.state.parameters())
            assert all(p.requires_grad for p in trainer.model.action_decoder.parameters())
            for key, value in artifact['encoder_state'].items():
                torch.testing.assert_close(checkpoint.model_state['obs_encoder.visual.' + key], value, rtol=0, atol=0)
            assert any(not torch.equal(value.cpu(), checkpoint.model_state['obs_encoder.state.' + key])
                       for key, value in trainer.model.obs_encoder.state.state_dict().items())
            results['only_visual_frozen'] = True
        else:
            assert all(p.requires_grad for p in encoder.parameters())
            assert trainer.ema_model is not None
            raw = trainer.model.state_dict()
            for key, value in trainer.ema_model.state_dict().items():
                torch.testing.assert_close(raw[key], value, rtol=0, atol=0)
            assert any(not torch.equal(value, checkpoint.model_state['obs_encoder.encoder.' + key])
                       for key, value in artifact['encoder_state'].items())
            results['initial_raw_ema_equal'] = True
            results['encoder_finetuned'] = True
        groups = trainer.optimizer.param_groups
        names = [name for group in groups for name in group['param_names']]
        assert len(names) == len(set(names))
        assert set(names) == {name for name, value in trainer.model.named_parameters() if value.requires_grad}
        assert all(group['weight_decay'] == 1e-6 for group in groups)
        results['optimizer_groups'] = len(groups)
        load_policy_state(trainer.model, checkpoint.model_state)
        if trainer.ema_model is not None:
            load_policy_state(trainer.ema_model, checkpoint.ema_model_state)
        batch = next(iter(trainer.train_loader))
        close_data_loader(trainer.train_loader)
        load = torch.load

        def load_checkpoint_only(path, *args, **kwargs):
            assert Path(path).resolve() == checkpoint_path, f'Unexpected external weights: {path}'
            return load(path, *args, **kwargs)

        with patch('torch.load', side_effect=load_checkpoint_only), patch(
            'zarr.open_group', side_effect=AssertionError('Inference must not open training data or labels')
        ):
            validate_policy_restoration(trainer.cfg, trainer, checkpoint_path, batch=batch)
        results['restore_without_external_weights_or_data'] = True
        results['future_actions_shape'] = [1, trainer.model.future_steps, 19]
        results['fixed_rng_predictions_match'] = True
    # Restoration must never call fit_field, for either training entry point.
    resumed = torch.load(args.fixture / f'{method}_resume/resume.pt', map_location='cpu', weights_only=False)
    pretrain = torch.load(pretrain_path.with_name('resume.pt'), map_location='cpu', weights_only=False)
    assert resumed['global_step'] == pretrain['global_step'] + 2
    assert resumed['scheduler_state']['last_epoch'] == pretrain['scheduler_state']['last_epoch'] + 2
    assert resumed['effective_config']['dataset']['split_manifest'] == pretrain['effective_config']['dataset']['split_manifest']
    for key in pretrain['model_state']:
        if key.startswith('normalizer.'):
            torch.testing.assert_close(pretrain['model_state'][key], resumed['model_state'][key], rtol=0, atol=0)
    with patch.object(LinearNormalizer, 'fit_field', side_effect=AssertionError('refit on resume')):
        pretrain_dataset, _, loader = prepare_data(OmegaConf.create(resumed['effective_config']), resumed)
        close_data_loader(loader)
        resume_cfg = OmegaConf.create(checkpoint.effective_config)
        resume_cfg.resume_from = str(checkpoint_path)
        policy_dataset, policy_normalizer, _ = build_dataset_and_normalizer(resume_cfg, resume_checkpoint=checkpoint)
    assert pretrain_dataset.split_manifest == policy_dataset.split_manifest
    for key, value in policy_normalizer.state_dict().items():
        torch.testing.assert_close(value, checkpoint.model_state['normalizer.' + key], rtol=0, atol=0)
    results['resume_reuses_split_affine_scheduler'] = True
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
