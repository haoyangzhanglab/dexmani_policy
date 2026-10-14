import copy
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from dexmani_policy.agents.loader import restore_policy_agent
from dexmani_policy.smoke_test import load_config
from dexmani_policy.training.build_utils import build_dataset_and_normalizer, build_model_and_ema, build_optimizer_and_scheduler
from dexmani_policy.training.checkpoint import CheckpointStore
from dexmani_policy.training.resume import build_train_loader, build_resume_contract, restore_training_state
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.workspace import TrainWorkspace
from dexmani_policy.utils.random import set_seed
from test_data_semantics import write_data


def equal_state(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def components(cfg, checkpoint=None):
    ds, norm = build_dataset_and_normalizer(cfg, resume_checkpoint=checkpoint, verbose=False)
    loader = build_train_loader(cfg, ds)
    model, ema, updater = build_model_and_ema(cfg, 'cpu', norm, checkpoint=checkpoint)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, verbose=False)
    workspace = TrainWorkspace(cfg.workspace.output_dir)
    trainer = Trainer('cpu', model, ema, updater, optimizer, scheduler, loader, workspace,
                      TrainLoopConfig(**dict(cfg.training.loop)), build_resume_contract(cfg, model, loader),
                      effective_config=cfg, max_grad_norm=0)
    workspace.save_hydra_config(cfg)
    return trainer


def test_sat_records_effective_patch_defaults_and_fps():
    from dexmani_policy.agents.normalization import LinearNormalizer
    import hydra
    cfg = load_config('sat')
    cfg.agent.pc_encoder_config = None
    cfg.agent.hidden_dim = 32
    cfg.agent.n_layers = 1
    cfg.agent.n_head = 4
    cfg.training.use_ema = False
    normalizer = LinearNormalizer()
    normalizer.fit({'action': torch.randn(4, 19), 'joint_state': torch.randn(4, 19)})
    model, _, _ = build_model_and_ema(cfg, 'cpu', normalizer)
    rebuilt = hydra.utils.instantiate(cfg.agent)
    assert cfg.agent.pc_encoder_config.num_patches == 96
    assert dict(cfg.agent.pc_encoder_config.fps_random_config) == dict(cfg.agent.fps_random_config)
    assert rebuilt.obs_encoder.pc_encoder.local_patch_tokenizer.fps_random_config == model.obs_encoder.pc_encoder.local_patch_tokenizer.fps_random_config
    assert [(k, tuple(v.shape)) for k,v in model.obs_encoder.state_dict().items()] == [
        (k, tuple(v.shape)) for k,v in rebuilt.obs_encoder.state_dict().items()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA scheduler placement regression')
def test_cuda_ddim_restore_is_independent_of_training_scheduler_device():
    from dexmani_policy.agents.action_decoders.diffusion import Diffusion

    class Backbone(torch.nn.Module):
        def forward(self, x, timestep, context):
            return .7 * x + context

    trained, restored = Diffusion(Backbone()), Diffusion(Backbone())
    template = torch.zeros(1, 16, 19, device='cuda')
    cond = torch.randn_like(template) * .1
    # Exercise the real training helper that mutates DDIM coefficient placement.
    trained.noise_scheduler.add_noise(template, template, torch.tensor([50], device='cuda'))
    assert trained.noise_scheduler.alphas_cumprod.is_cuda
    assert not restored.noise_scheduler.alphas_cumprod.is_cuda
    for decoder in (trained, restored):
        set_seed(42)
        output = decoder.predict_action(cond, template)
        assert decoder.noise_scheduler.alphas_cumprod.is_cuda
        assert decoder.noise_scheduler.final_alpha_cumprod.is_cuda
        if decoder is trained:
            expected = output
        else:
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


def small_dp3_config(tmp_path):
    """Bounded CPU fixture; leaves the repository's training recipe unchanged."""
    data = tmp_path / 'data'
    write_data(data, 32)
    cfg = load_config('dp3')
    cfg.dataset.zarr_path = str(data)
    cfg.dataset.augmentation_cfg = None
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.training.loop.total_train_steps = 10
    cfg.training.lr_warmup_steps = 3
    cfg.training.use_compile = False
    cfg.training.use_bfloat16 = False
    cfg.training.device = 'cpu'
    cfg.agent.down_dims = [32, 64]
    cfg.agent.diffusion_step_embed_dim = 32
    cfg.workspace.output_dir = str(tmp_path / 'continuous')
    cfg.workspace.wandb_cfg = None
    return cfg


def test_real_dp3_components_resume_multiple_updates_and_raw_ema(tmp_path):
    """Bounded CPU integration of real components, not the original CUDA smoke config."""
    cfg = small_dp3_config(tmp_path)
    set_seed(42)
    continuous = components(cfg)
    try:
        continuous.train(max_updates=1)
        path = (continuous.workspace.checkpoint_dir / 'latest.pt').resolve()
        checkpoint = CheckpointStore(path.parent).load(path)
        frozen = copy.deepcopy(checkpoint.effective_config)
        with open_dict(cfg):
            cfg.max_updates = 999  # Trainer must own a snapshot, not this reference.
        assert continuous.effective_config == frozen
        cursor = (continuous.global_step, continuous.current_epoch, continuous.next_micro_step)
        expected = []
        for _ in range(5):
            continuous.train(resume_state=cursor, max_updates=1)
            cursor = (continuous.global_step, continuous.current_epoch, continuous.next_micro_step)
            expected.append((copy.deepcopy(continuous.raw_model.state_dict()),
                             copy.deepcopy(continuous.optimizer.state_dict()),
                             copy.deepcopy(continuous.scheduler.state_dict()),
                             copy.deepcopy(continuous.ema_model.state_dict()),
                             continuous.ema_updater.optimization_step, continuous.ema_updater.decay, cursor))
        resumed_cfg = OmegaConf.create(checkpoint.effective_config)
        with open_dict(resumed_cfg):
            resumed_cfg.resume_from = str(path)
            resumed_cfg.workspace.output_dir = str(tmp_path / 'resumed')
        resumed = components(resumed_cfg, checkpoint)
        try:
            cursor = restore_training_state(checkpoint, resume_contract=resumed.resume_contract,
                      model=resumed.raw_model, ema_model=resumed.ema_model, ema_updater=resumed.ema_updater,
                      optimizer=resumed.optimizer, scheduler=resumed.scheduler, device='cpu')
            assert resumed.ema_updater.decay == checkpoint.ema_decay
            for expected_step in expected:
                resumed.train(resume_state=cursor, max_updates=1)
                cursor = (resumed.global_step, resumed.current_epoch, resumed.next_micro_step)
                actual = (resumed.raw_model.state_dict(), resumed.optimizer.state_dict(),
                          resumed.scheduler.state_dict(), resumed.ema_model.state_dict(),
                          resumed.ema_updater.optimization_step, resumed.ema_updater.decay, cursor)
                equal_state(expected_step, actual)
            latest = (resumed.workspace.checkpoint_dir / 'latest.pt').resolve()
            saved = CheckpointStore(latest.parent).load(latest)
            assert saved.effective_config['workspace']['output_dir'] == str(tmp_path / 'resumed')
            obs = {k: v.unsqueeze(0) for k, v in resumed.train_loader.dataset[0]['obs'].items()}
            for use_ema, model in [(False, resumed.raw_model), (True, resumed.ema_model)]:
                model.eval()
                loaded = restore_policy_agent(saved.effective_config, latest, use_ema=use_ema, device='cpu')
                set_seed(5)
                with torch.no_grad():
                    prediction = model.predict_action(obs, inference_steps=2)['pred_action']
                set_seed(5)
                equal_state(prediction, loaded.predict_action(obs, inference_steps=2)['pred_action'])
            # Inference only needs bound input/model recipe and tensor weights.
            portable = tmp_path / 'portable' / 'checkpoints' / 'weights.pt'
            portable.parent.mkdir(parents=True)
            payload = CheckpointStore(latest.parent).load_payload(latest)
            payload['weights'].pop('optimizer')
            payload['weights'].pop('scheduler')
            payload['state'] = {'global_step': saved.global_step}
            torch.save(payload, portable)
            assert restore_policy_agent(None, portable, use_ema=False, device='cpu').action_dim == 19
            # Poison raw and EMA independently after Dataset normalizer loading.
            from dexmani_policy.training.resume import restore_model_weights
            for attr in ['model_state', 'ema_model_state']:
                bad = copy.deepcopy(saved)
                getattr(bad, attr)['normalizer.params_dict.action.scale'] = torch.ones(2)
                getattr(bad, attr)['normalizer.params_dict.action.offset'] = torch.zeros(2)
                with pytest.raises(ValueError, match='Normalizer state'):
                    restore_model_weights(bad, resumed.raw_model, resumed.ema_model, 'cpu')
                restore_model_weights(saved, resumed.raw_model, resumed.ema_model, 'cpu')
        finally:
            resumed.workspace.close()
    finally:
        continuous.workspace.close()


def test_single_process_resume_deserializes_once_and_releases_checkpoint(tmp_path, monkeypatch):
    from hydra.core.hydra_config import HydraConfig
    from dexmani_policy import train as entry

    cfg = small_dp3_config(tmp_path)
    cfg.dataloader.pin_memory = False
    with open_dict(cfg):
        cfg.max_updates = 1
    original_build = entry.build_train_components
    original_train = Trainer.train
    original_load = CheckpointStore.load
    original_torch_load = torch.load
    reads, checkpoint_refs, cursors = [], [], []

    def counted_torch_load(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return original_torch_load(path, *args, **kwargs)

    def tracked_checkpoint_load(self, path):
        checkpoint = original_load(self, path)
        checkpoint_refs.append(weakref.ref(checkpoint))
        return checkpoint

    def cpu_components(config, **kwargs):
        # Isolate only the entry's GPU availability guard. Dataset, normalization,
        # model/EMA construction, checkpoint I/O and state restoration stay real.
        with monkeypatch.context() as guard:
            guard.setattr(torch.cuda, 'is_available', lambda: True)
            return original_build(config, **kwargs)

    def bounded_train(self, **kwargs):
        assert all(ref() is None for ref in checkpoint_refs)
        cursors.append(kwargs.get('resume_state'))
        return original_train(self, **kwargs)

    monkeypatch.setattr(torch, 'load', counted_torch_load)
    monkeypatch.setattr(CheckpointStore, 'load', tracked_checkpoint_load)
    monkeypatch.setattr(entry, 'build_train_components', cpu_components)
    monkeypatch.setattr(Trainer, 'train', bounded_train)
    entry.main.__wrapped__(cfg)
    assert reads == [] and checkpoint_refs == []  # Fresh startup loads no resume file.
    source = (Path(cfg.workspace.output_dir) / 'checkpoints' / 'latest.pt').resolve()
    incoming = OmegaConf.create({'resume_from': str(source),
                                'workspace': {'output_dir': str(tmp_path / 'resumed-entry')}})
    monkeypatch.setattr(HydraConfig, 'get', lambda: SimpleNamespace(
        overrides=SimpleNamespace(task=['max_updates=1', 'workspace.wandb_cfg=null'])))
    entry.main.__wrapped__(incoming)
    assert reads == [source]
    assert len(checkpoint_refs) == 1 and checkpoint_refs[0]() is None
    assert cursors == [None, (1, 0, 1)]
    print('single-process target torch.load count:', reads.count(source), '; fresh startup: 0')

    # Standalone component callers still load and validate when no object is supplied.
    saved_cfg = OmegaConf.load(Path(cfg.workspace.output_dir) / 'config.yaml')
    with open_dict(saved_cfg):
        saved_cfg.resume_from = str(source)
        saved_cfg.workspace.output_dir = str(tmp_path / 'standalone-components')
    reads.clear()
    comp = cpu_components(saved_cfg)
    try:
        assert reads == [source]
        assert comp.resume_checkpoint is checkpoint_refs[-1]()
        assert comp.resume_checkpoint.global_step == 1
    finally:
        comp.workspace.close()
        comp.resume_checkpoint = None


@pytest.mark.parametrize('invalid_source', ['sidecar', 'legacy'])
def test_single_process_validates_source_before_build(tmp_path, monkeypatch, invalid_source):
    from dexmani_policy import train as entry
    from test_semantic_boundaries import checkpoint_at
    store, path, cfg = checkpoint_at(tmp_path)
    if invalid_source == 'sidecar':
        cfg['training']['lr_scheduler'] = 'constant'
        OmegaConf.save(OmegaConf.create(cfg), tmp_path / 'config.yaml')
        error = 'source recipe mismatch'
    else:
        payload = store.load_payload(path)
        payload['_format'] = 'simple.v3'  # Even with effective_config and a matching sidecar.
        torch.save(payload, path)
        error = 'Strict resume requires'

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid source must be rejected before building components')

    monkeypatch.setattr(entry, 'build_train_components', forbidden)
    with pytest.raises(ValueError, match=error):
        entry.main.__wrapped__(OmegaConf.create({'resume_from': str(path)}))
