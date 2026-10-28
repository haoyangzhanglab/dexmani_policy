import importlib
import json
import random
import signal

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_policy.common.config import load_policy_config
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state
from dexmani_policy.training.common.data import build_train_loader, close_data_loader, epoch_position
from dexmani_policy.training.common.resume import resolve_input_recipe, validate_source_recipe
from dexmani_policy.training.pretrain.obs.common import evaluate


class IndexDataset(torch.utils.data.Dataset):
    split_manifest = {}

    def __init__(self, size=20):
        self.size, self.reads = size, []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.reads.append(index)
        return {'obs': {'index': torch.tensor(float(index))}}

    def get_normalizer(self, spec):
        return LinearNormalizer()

    def get_validation_dataset(self):
        return None


def loader_config(size, accumulation, shuffle, drop_last, workers=0, persistent=False):
    return OmegaConf.create({
        'training': {'seed': 42, 'loop': {'gradient_accumulation_steps': accumulation}},
        'dataloader': {'batch_size': size, 'shuffle': shuffle, 'drop_last': drop_last,
                       'num_workers': workers, 'persistent_workers': persistent},
    })


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('samples,accumulation,drop_last', [(20, 1, False), (20, 3, False), (19, 3, False), (19, 2, True)])
@pytest.mark.parametrize('workers,persistent', [(0, False), (2, False), (2, True)])
def test_segmented_sample_sequence(shuffle, samples, accumulation, drop_last, workers, persistent):
    cfg = loader_config(2, accumulation, shuffle, drop_last, workers, persistent)

    def consume(segments):
        step, sequence = 0, []
        for updates in segments:
            dataset = IndexDataset(samples)
            loader = build_train_loader(cfg, dataset)
            batches = loader.batch_sampler.batches_per_epoch
            stop = step + updates
            segment = []
            iterator = None
            try:
                while step < stop:
                    epoch, start = epoch_position(step, batches, accumulation)
                    loader.batch_sampler.set_epoch(epoch, start)
                    assert len(loader) == batches - start
                    iterator = iter(loader)
                    for index, batch in enumerate(iterator, start):
                        segment.extend(batch['obs']['index'].tolist())
                        if (index + 1) % accumulation == 0 or index + 1 == batches:
                            step += 1
                            if step == stop:
                                break
            finally:
                close_data_loader(loader, iterator)
            if workers == 0:
                assert dataset.reads == segment
            sequence.extend(segment)
        return sequence, epoch_position(step, batches, accumulation)

    continuous = consume([30])
    assert consume([6] * 5) == continuous
    if samples == 20 and accumulation == 1:
        assert continuous[1] == (3, 0)


def test_mixed_microbatches_still_rejected():
    with pytest.raises(ValueError, match='Mixed micro-batch'):
        build_train_loader(loader_config(2, 2, False, False), IndexDataset(19))


@pytest.mark.parametrize('outcome', ['complete', 'before', 'interrupt', 'exception'])
@pytest.mark.parametrize('training', [False, True])
def test_evaluate_restores_rng_and_mode(outcome, training):
    stopped = outcome == 'before'

    class Model(torch.nn.Module):
        calls = 0

        def forward(self, batch):
            nonlocal stopped
            self.calls += 1
            random.random(), np.random.rand(), torch.rand(1)
            if outcome == 'exception':
                raise RuntimeError('validation failed')
            if outcome == 'interrupt':
                stopped = True
            value = batch['obs']['index'].mean()
            return value, {'loss': value}

    model = Model().train(training)
    before = get_rng_state()
    if outcome == 'exception':
        with pytest.raises(RuntimeError, match='validation failed'):
            evaluate(model, IndexDataset(5), 2, 'cpu', stop_requested=lambda: stopped)
    else:
        result = evaluate(model, IndexDataset(5), 2, 'cpu', stop_requested=lambda: stopped)
        if outcome == 'complete':
            assert result['val/loss'].item() == 2
        else:
            assert result is None
            assert model.calls == (0 if outcome == 'before' else 1)
    assert model.training == training
    actual = (random.random(), np.random.rand(), torch.rand(3))
    set_rng_state(before)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


@pytest.mark.parametrize('method,accumulation', [('afro', 1), ('cordvip', 1), ('cordvip', 3)])
def test_pretrain_loops_resume(tmp_path, monkeypatch, method, accumulation):
    # Small models isolate loop progress; real encoder integration is exercised separately.
    module = importlib.import_module(f'dexmani_policy.training.pretrain.obs.{method}')
    consumed, teacher_epochs = [], []

    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.visual = torch.nn.Linear(1, 1)
            self.normalizer = LinearNormalizer()

        @property
        def obs_encoder(self):
            return self.visual

        def forward(self, batch):
            values = batch['obs']['index']
            consumed.extend(values.tolist())
            loss = self.visual(values[:, None]).square().mean()
            return loss, {'loss': loss.detach()}

        def update_teacher(self, epoch):
            assert self.visual.weight.grad is not None
            teacher_epochs.append(epoch)

    monkeypatch.setattr(module, 'AFROPretrain' if method == 'afro' else 'CordViPPretrain', Model)
    monkeypatch.setattr(module.hydra.utils, 'instantiate', lambda *args, **kwargs: IndexDataset())
    cfg = load_policy_config(f'pretrain/obs/{method}')
    cfg.training.device = 'cpu'
    cfg.training.loop.gradient_accumulation_steps = accumulation
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    checkpoint = None
    for segment in range(5):
        cfg.workspace.output_dir = str(tmp_path / str(segment))
        module.train(cfg, checkpoint, max_updates=6)
        checkpoint = torch.load(tmp_path / str(segment) / 'resume.pt', weights_only=False)
        assert checkpoint['global_step'] == (segment + 1) * 6
        assert checkpoint['epoch'] == epoch_position((segment + 1) * 6, 10, accumulation)[0]
        # global_step owns restoration; epoch is diagnostic.
        checkpoint['epoch'] = 0
    if method == 'afro':
        assert teacher_epochs == [0] * 10 + [1] * 10 + [2] * 10
    split_consumed = consumed[:]
    consumed.clear()
    cfg.workspace.output_dir = str(tmp_path / 'continuous')
    module.train(cfg, max_updates=30)
    assert consumed == split_consumed
    reference = torch.load(tmp_path / 'continuous/resume.pt', weights_only=False)
    torch.testing.assert_close(checkpoint['model_state'], reference['model_state'], rtol=0, atol=0)
    assert checkpoint['scheduler_state'] == reference['scheduler_state']
    for line in (tmp_path / 'continuous/metrics.jsonl').read_text().splitlines():
        metrics = json.loads(line)
        assert 0 <= metrics['train/data_wait_time'] <= metrics['train/step_time']


@pytest.mark.parametrize('method', ['afro', 'cordvip'])
def test_validation_signal_saves_completed_update(tmp_path, monkeypatch, method):
    module = importlib.import_module(f'dexmani_policy.training.pretrain.obs.{method}')

    class Model(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.visual = torch.nn.Linear(1, 1)
            self.normalizer = LinearNormalizer()

        @property
        def obs_encoder(self):
            return self.visual

        def forward(self, batch):
            if not self.training:
                signal.raise_signal(signal.SIGTERM)
            loss = self.visual(batch['obs']['index'][:, None]).square().mean()
            return loss, {'loss': loss.detach()}

        def update_teacher(self, epoch):
            pass

    dataset = IndexDataset(2)
    dataset.get_validation_dataset = lambda: IndexDataset(5)
    monkeypatch.setattr(module.hydra.utils, 'instantiate', lambda *a, **k: dataset)
    monkeypatch.setattr(module, 'AFROPretrain' if method == 'afro' else 'CordViPPretrain', Model)
    cfg = load_policy_config(f'pretrain/obs/{method}')
    cfg.training.device = 'cpu'
    cfg.training.val_every_epochs = 1
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.workspace.output_dir = str(tmp_path)
    module.train(cfg, max_updates=3)
    saved = torch.load(tmp_path / 'resume.pt', weights_only=False)
    assert (saved['global_step'], saved['epoch']) == (1, 1)
    logs = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert not any(key.startswith('val/') for row in logs for key in row)


def test_resume_operational_fields_only():
    cfg = load_policy_config('pretrain/obs/afro')
    cfg.dataset.split_manifest = {}
    OmegaConf.update(cfg, 'resume_from', 'resume.pt', force_add=True)
    overrides = ['training.save_every_epochs=2', '+training.save_every_updates=3', 'training.val_every_epochs=4']
    restored = resolve_input_recipe(cfg, overrides=overrides, saved_config=OmegaConf.to_container(cfg))
    validate_source_recipe(cfg, restored)
    for key in ('model.frame_stride', 'dataloader.batch_size', 'training.seed'):
        with pytest.raises(ValueError, match='forbidden override'):
            resolve_input_recipe(cfg, overrides=[f'{key}=9'], saved_config=OmegaConf.to_container(cfg))


@pytest.mark.parametrize('accumulation', [1, 3])
def test_public_loop_accumulation_progress(accumulation):
    from types import MethodType, SimpleNamespace
    from dexmani_policy.training.trainer import PolicyTrainer

    class Dataset(IndexDataset):
        def __getitem__(self, index):
            batch = super().__getitem__(index)
            batch['action'] = torch.tensor([float(index)])
            return batch

    def run(segments):
        torch.manual_seed(42)
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        trainer.cfg = loader_config(2, accumulation, True, False)
        trainer.train_loader = build_train_loader(trainer.cfg, Dataset())
        trainer.batches_per_epoch = 10
        trainer.gradient_accumulation_steps = accumulation
        trainer.total_train_steps = 100000
        trainer.log_interval_steps = 100
        trainer.current_epoch = trainer.global_step = 0
        trainer.model = torch.nn.Linear(1, 1)
        trainer.ema_model = trainer.ema_updater = None
        trainer.max_grad_norm = 0
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1)
        trainer.logger = SimpleNamespace(log=lambda *a, **k: None)
        trainer.last_grad_norm = trainer.last_clip_ratio = None
        consumed, saved = [], []

        def train_microbatch(self, batch, *, loss_divisor, log_metrics):
            consumed.extend(batch['obs']['index'].tolist())
            loss = self.model(batch['action']).square().mean()
            (loss / loss_divisor).backward()
            return {'loss': loss.detach()}

        trainer.train_microbatch = MethodType(train_microbatch, trainer)
        trainer.check_milestone = lambda: None
        trainer.save_checkpoint = lambda tag: saved.append((trainer.global_step, trainer.current_epoch))
        for updates in segments:
            trainer.train(max_updates=updates)
        assert saved[-1] == (30, epoch_position(30, 10, accumulation)[0])
        return consumed, trainer.model.state_dict(), trainer.scheduler.state_dict()

    full, segmented = run([30]), run([6] * 5)
    assert full[0] == segmented[0]
    torch.testing.assert_close(full[1], segmented[1], rtol=0, atol=0)
    assert full[2] == segmented[2]


@pytest.mark.parametrize('method,field', [
    ('afro', 'save_every_epochs'), ('afro', 'val_every_epochs'),
    ('cordvip', 'save_every_updates'), ('cordvip', 'val_every_epochs'),
])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_pretrain_intervals_fail_before_data_loading(monkeypatch, method, field, value):
    module = importlib.import_module(f'dexmani_policy.training.pretrain.obs.{method}')
    cfg = load_policy_config(f'pretrain/obs/{method}')
    cfg.training[field] = value
    monkeypatch.setattr(module.hydra.utils, 'instantiate', lambda *a, **k: pytest.fail('dataset opened'))
    with pytest.raises(ValueError, match=f'training.{field}'):
        module.train(cfg, max_updates=1)
