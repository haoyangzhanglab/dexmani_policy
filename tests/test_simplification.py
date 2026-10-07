"""Bounded CPU and isolated shell checks; these do not validate production assets."""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from test_infra_resume import Data, TinyPolicy
from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.training.resume import (build_train_loader, build_resume_contract,
    restore_model_weights, restore_training_state)
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.workspace import TrainWorkspace
from dexmani_policy.utils.random import set_seed


def bounded_trainer(path, *, accum=4, log_interval=2, checkpoint=None):
    cfg = OmegaConf.create({
        'agent': {'_target_': 'test_infra_resume.TinyPolicy'},
        'dataset': {}, 'data_identity': {'revision': 'fixture-v1'},
        'dataloader': {'batch_size': 2, 'shuffle': True, 'drop_last': False, 'num_workers': 0},
        'optimizer': {}, 'ema': {},
        'training': {'seed': 12, 'lr_warmup_steps': 500,
                     'loop': {'total_train_steps': 10, 'gradient_accumulation_steps': accum}},
    })
    model = TinyPolicy(clip_sample=False)
    model.normalizer.fit_field('action', torch.arange(12, dtype=torch.float32).reshape(-1, 1), mode='limits')
    loader = build_train_loader(cfg, Data())
    if checkpoint is not None:
        restore_model_weights(checkpoint, model, None, "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / 500, 1.))
    workspace = TrainWorkspace(str(path))
    workspace.save_hydra_config(cfg)
    trainer = Trainer('cpu', model, None, None, opt, sched, loader, workspace,
                      TrainLoopConfig(total_train_steps=10, gradient_accumulation_steps=accum,
                                      log_interval_steps=log_interval),
                      build_resume_contract(cfg, model, loader), batches_per_epoch=6)
    return trainer


@pytest.mark.parametrize('budget,epoch,cursor', [(1, 0, 4), (2, 1, 0), (3, 1, 4), (4, 2, 0)])
def test_bounded_cursor_warmup_and_resume(tmp_path, budget, epoch, cursor):
    set_seed(12)
    full = bounded_trainer(tmp_path / 'full')
    full.train(max_updates=4)
    set_seed(12)
    first = bounded_trainer(tmp_path / 'first')
    before = first.raw_model.action_decoder.model.weight.detach().clone()
    first.train(max_updates=budget)
    checkpoint = first.workspace.checkpoint_store.load(first.workspace.resolve_checkpoint_path('latest'))
    assert (checkpoint.global_step, checkpoint.epoch, checkpoint.next_micro_step) == (budget, epoch, cursor)
    assert first.total_train_steps == 10
    assert first.scheduler.last_epoch == budget
    after = first.raw_model.action_decoder.model.weight.detach()
    assert torch.equal(before, after) if budget == 1 else not torch.equal(before, after)
    resumed = bounded_trainer(tmp_path / 'resumed', checkpoint=checkpoint)
    state = restore_training_state(checkpoint, resume_contract=resumed.resume_contract,
        model=resumed.raw_model, ema_model=None, ema_updater=None,
        optimizer=resumed.optimizer, scheduler=resumed.scheduler, device='cpu')
    if budget < 4:
        resumed.train(resume_state=state, max_updates=4 - budget)
        assert first.raw_model.seen + resumed.raw_model.seen == full.raw_model.seen
        torch.testing.assert_close(full.raw_model.action_decoder.model.weight,
                                   resumed.raw_model.action_decoder.model.weight, rtol=0, atol=0)
        assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    else:
        resumed.workspace.close()
    # At step 2/4 a milestone already saved the exact state: no bounded duplicate.
    assert len(list((tmp_path / 'first/checkpoints').glob(f'*step={budget:08d}*.pt'))) == 1


def test_bounded_validation_and_completed_no_write(tmp_path):
    trainer = bounded_trainer(tmp_path / 'run')
    for bad in (0, -1, True, 1.5):
        with pytest.raises((ValueError, TypeError)):
            trainer.train(max_updates=bad)
    trainer.train(resume_state=(10, 5, 0), max_updates=4)
    assert not list((tmp_path / 'run').rglob('*.pt'))
    assert trainer.raw_model.seen == []


def test_optimizer_coverage_rejects_missing_duplicate_and_foreign():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW([model.weight])
    with pytest.raises(ValueError, match='NOT'):
        BaseAgent._check_params_in_optimizer(model, opt)
    opt = torch.optim.AdamW(model.parameters())
    BaseAgent._check_params_in_optimizer(model, opt)
    opt.param_groups[0]['params'].append(model.weight)
    with pytest.raises(ValueError, match='Duplicate'):
        BaseAgent._check_params_in_optimizer(model, opt)
    opt.param_groups[0]['params'][-1] = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match='outside'):
        BaseAgent._check_params_in_optimizer(model, opt)


def test_inventory_never_deletes(tmp_path):
    run = tmp_path / 'tiny experiment'
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'config.yaml').write_text('training: {loop: {total_train_steps: 1}}')
    (run / 'checkpoints/epoch=0000-step=00000000.pt').write_bytes(b'incomplete')
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    cmd = ['bash', 'scripts/utils/clean_experiments.sh', '--root', str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    for flag in ('--force', '--yes', '--older-than', '--include-active', '--toy-min-steps'):
        result = subprocess.run(cmd + [flag], capture_output=True, timeout=5)
        assert result.returncode != 0
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('force', [False, True])
def test_stop_wait_never_implies_force_or_checkpoint_success(tmp_path, force):
    # No SSH/network: every external operation is a fixture executable.
    ssh = tmp_path / 'ssh'
    ssh.write_text('#!' + sys.executable + '\n' + '''
import os,sys
from pathlib import Path
root=Path(os.environ['STOP_FIXTURE'])
command=sys.argv[-1]
with (root/'calls').open('a') as f: f.write(command+'\\n')
if 'list-sessions' in command:
    if not (root/'killed').exists(): print('dex_fixture: 1 windows')
elif 'kill-session' in command: (root/'killed').touch()
''')
    ssh.chmod(0o755)
    sleep = tmp_path / 'sleep'
    sleep.write_text('#!/bin/sh\nexit 0\n'); sleep.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'], STOP_FIXTURE=str(tmp_path))
    result = subprocess.run(['bash', 'scripts/remote/stop_remote.sh', 'dex_fixture'] +
                            (['--force'] if force else []), env=env,
                            capture_output=True, text=True, timeout=5)
    assert (result.returncode == 0) == force
    calls = (tmp_path / 'calls').read_text()
    assert ('tmux kill-session' in calls) == force
    assert calls.count('tmux send-keys') == 1
    assert 'nvidia-smi' not in calls
    if force:
        assert 'checkpoint completeness is unverified' in result.stdout


def test_log_conversion_only_at_output_and_updates_unchanged(tmp_path):
    from dexmani_policy.training.logging import to_log_scalars
    set_seed(12)
    every = bounded_trainer(tmp_path / 'every', log_interval=1)
    every.train(max_updates=4)
    set_seed(12)
    sparse = bounded_trainer(tmp_path / 'sparse', log_interval=2)
    calls = []
    def convert(values):
        assert all(not value.requires_grad for value in values.values() if torch.is_tensor(value))
        calls.append(sparse.global_step)
        return to_log_scalars(values)
    with patch('dexmani_policy.training.trainer.to_log_scalars', side_effect=convert):
        sparse.train(max_updates=4)
    assert calls == [2, 4]
    torch.testing.assert_close(every.raw_model.action_decoder.model.weight,
                               sparse.raw_model.action_decoder.model.weight, rtol=0, atol=0)
    import json
    full = [json.loads(s) for s in (tmp_path / 'every/metrics.jsonl').read_text().splitlines()]
    actual = [json.loads(s) for s in (tmp_path / 'sparse/metrics.jsonl').read_text().splitlines()]
    for expected, result in zip(full[1::2], actual):
        for key in ('train/loss', 'train/lr', 'train/grad_norm', 'train/clip_ratio', 'train/samples_per_step'):
            assert expected[key] == result[key]


def test_nonfinite_loss_fails_without_full_state_dump(tmp_path):
    trainer = bounded_trainer(tmp_path / 'run')
    def fail(batch, **kwargs):
        loss = trainer.raw_model.action_decoder.model.weight * float('nan')
        return loss, {'loss': loss}
    trainer.model.forward = fail
    with pytest.raises(RuntimeError, match='Non-finite loss'):
        trainer.train(max_updates=1)
    assert trainer.global_step == 0
    assert not list((tmp_path / 'run').rglob('*.pt'))


@pytest.mark.parametrize('failed_operation', ['save_checkpoint', 'save_latest'])
def test_interrupt_save_failure_propagates_and_preserves_latest(tmp_path, monkeypatch, failed_operation):
    import signal

    trainer = bounded_trainer(tmp_path / 'run', accum=1)
    trainer._save_checkpoint(0, 'bounded')
    latest = trainer.workspace.checkpoint_dir / 'latest.pt'
    previous = latest.resolve()
    previous_bytes = previous.read_bytes()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    original_step = trainer.train_one_step

    def stop_after_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        trainer._stop_requested = True
        return result

    def fail(*args, **kwargs):
        raise OSError('checkpoint publication failed')

    monkeypatch.setattr(trainer, 'train_one_step', stop_after_step)
    monkeypatch.setattr(trainer.workspace, failed_operation, fail)
    with pytest.raises(OSError, match='checkpoint publication failed'):
        trainer.train(max_updates=1)
    assert trainer.global_step == 1
    assert latest.resolve() == previous and previous.read_bytes() == previous_bytes
    assert trainer.workspace._closed
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())


# Frozen pre-migration output from 7a27b35's M_e with lengths [3, 7, 1], seed=42.
LEGACY_MAPS = {
    'balanced': [[10,0,1,3,4,7,2,10,10,10,8], [6,1,10,5,10,3,8,2,10,10,0],
                 [2,10,10,10,0,3,10,9,5,6,1]],
    'weighted': [[10,0,1,10,10,3,8,10,10,10,7], [3,1,10,6,10,10,10,2,10,10,5],
                 [2,10,10,10,3,6,10,10,10,5,0]],
    'proportional': [[9,0,1,3,4,7,2,5,10,6,8], [6,1,4,5,9,8,3,2,10,7,0],
                     [9,8,10,2,3,6,1,4,7,0,5]],
}


@pytest.mark.parametrize('strategy', LEGACY_MAPS)
@pytest.mark.parametrize('deterministic', [False, True])
def test_sampler_preserves_both_order_layers(strategy, deterministic):
    from test_infra_training import TinyDataset
    from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
    from dexmani_policy.datasets.resumable_sampler import ResumableDistributedSampler
    from torch.utils.data import DataLoader, DistributedSampler
    ds = MultiTaskDataset([TinyDataset(3), TinyDataset(7), TinyDataset(1)], ['a', 'b', 'c'],
                          sampling_strategy=strategy, deterministic=deterministic,
                          task_weights=[.2, .3, .5] if strategy == 'weighted' else None)
    for epoch in (0, 1):
        mapping = LEGACY_MAPS[strategy][2 if deterministic else epoch]
        for ranks in (1, 2, 3):
            for rank in range(ranks):
                for shuffle in (False, True):
                    reference = DistributedSampler(range(11), num_replicas=ranks, rank=rank,
                                                   seed=17, shuffle=shuffle)
                    reference.set_epoch(epoch)
                    expected = [mapping[i] for i in reference]
                    sampler = ResumableDistributedSampler(ds, batch_size=2, num_replicas=ranks,
                                                         rank=rank, seed=17, shuffle=shuffle)
                    for cursor in (0, 1):
                        sampler.set_epoch(epoch, cursor)
                        assert list(sampler) == expected[cursor * 2:]
                        for drop in (False, True):
                            loader = DataLoader(ds, sampler=sampler, batch_size=2, drop_last=drop)
                            actual = [(task, int(local)) for batch in loader for task, local in
                                      zip(batch['obs']['task_name'], batch['obs']['index'])]
                            indices = expected[cursor * 2:]
                            if drop:
                                indices = indices[:len(indices) // 2 * 2]
                            want = [('a', i) if i < 3 else ('b', i - 3) if i < 10 else ('c', 0)
                                    for i in indices]
                            assert actual == want


def test_checkpoint_purpose_required_fields_and_metadata(tmp_path):
    from dexmani_policy.agents.loader import restore_policy_agent
    trainer = bounded_trainer(tmp_path / 'run')
    trainer.train(max_updates=2)
    store = trainer.workspace.checkpoint_store
    path = trainer.workspace.resolve_checkpoint_path('latest')
    payload = torch.load(path, weights_only=False)
    payload['extra_note'] = 'allowed'
    payload['state']['extra_note'] = 1
    payload['weights']['extra_note'] = 2
    extra = tmp_path / 'extra.pt'
    torch.save(payload, extra)
    assert store.load(extra).global_step == 2
    # Inference does not require optimizer, RNG or an entire training contract.
    for key in ('optimizer', 'scheduler'):
        del payload['weights'][key]
    payload['state'] = {'global_step': 2}
    inference = tmp_path / 'inference.pt'
    torch.save(payload, inference)
    state, step = store.load_inference(inference, use_ema=False)
    assert step == 2
    cfg = {'agent': {'_target_': 'test_infra_resume.TinyPolicy', 'clip_sample': False,
                    'horizon': 2, 'n_obs_steps': 1, 'n_action_steps': 1},
           'action_key': 'action', 'normalization': {'action': 'limits', 'joint_state': 'identity'}}
    restored = restore_policy_agent(cfg, inference, use_ema=False, device='cpu')
    torch.testing.assert_close(restored.action_decoder.model.weight,
                               trainer.raw_model.action_decoder.model.weight, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        store.load(inference)
    with pytest.raises(ValueError, match='EMA'):
        store.load_inference(inference, use_ema=True)
    del payload['weights']['model']['action_decoder.model.weight']
    torch.save(payload, inference)
    with pytest.raises(RuntimeError):
        restore_policy_agent(cfg, inference, use_ema=False, device='cpu')


def test_ema_constructed_once_and_copied_without_aliases():
    import hydra
    from dexmani_policy.training.build_utils import build_model_and_ema
    from dexmani_policy.agents.normalization import LinearNormalizer
    cfg = OmegaConf.create({'agent': {'_target_': 'test_infra_resume.TinyPolicy', 'clip_sample': False},
        'dataset': {'sensor_modalities': ['joint_state']},
        'action_key': 'action', 'normalization': {'action': 'limits', 'joint_state': 'identity'},
        'training': {'use_ema': True},
        'ema': {'_target_': 'dexmani_policy.training.ema_model.EMAModel'}})
    normalizer = LinearNormalizer()
    normalizer.fit_field('action', torch.arange(12, dtype=torch.float32).reshape(-1, 1), mode='limits')
    original = hydra.utils.instantiate
    with patch('hydra.utils.instantiate', wraps=original) as instantiate:
        raw, ema, updater = build_model_and_ema(cfg, 'cpu', normalizer)
    assert sum(str(call.args[0].get('_target_')).endswith('TinyPolicy')
               for call in instantiate.call_args_list) == 1
    for (name, param), (ema_name, ema_param) in zip(raw.named_parameters(), ema.named_parameters()):
        assert name == ema_name
        assert param.data_ptr() != ema_param.data_ptr()
        assert param.dtype == ema_param.dtype
        torch.testing.assert_close(param, ema_param, rtol=0, atol=0)
    with torch.no_grad():
        raw.action_decoder.model.weight.add_(1)
    assert not torch.equal(raw.action_decoder.model.weight, ema.action_decoder.model.weight)


def test_saved_recipe_selected_before_build_and_override_boundary(tmp_path):
    from dexmani_policy.training.resume import resolve_training_config
    source = tmp_path / 'source'
    (source / 'checkpoints').mkdir(parents=True)
    cp = source / 'checkpoints/latest.pt'; cp.write_bytes(b'not loaded during config resolution')
    saved = OmegaConf.create({'workspace': {'output_dir': str(source), 'claim_token': 'old'},
                             'agent': {'architecture': 'saved'}, 'dataset': {'zarr_path': '/old'},
                             'training': {'seed': 12, 'use_compile': True, 'loop': {'total_train_steps': 100}},
                             'max_updates': 2, 'dataloader': {'num_workers': 4}})
    OmegaConf.save(saved, source / 'config.yaml')
    incoming = OmegaConf.create({'workspace': {'output_dir': str(tmp_path / 'new')},
                                'agent': {'architecture': 'wrong-current-default'},
                                'training': {'seed': 999}, 'resume_from': str(cp)})
    actual = resolve_training_config(incoming, overrides=['+resume_from=' + str(cp)])
    assert actual.agent.architecture == 'saved' and actual.training.seed == 12
    assert actual.training.loop.total_train_steps == 100
    assert actual.workspace.output_dir != str(source) and 'claim_token' not in actual.workspace
    assert 'max_updates' not in actual
    actual = resolve_training_config(incoming, overrides=['+max_updates=4', 'dataloader.num_workers=0',
                                                         'training.use_compile=false', 'dataset.zarr_path=/new'])
    assert actual.max_updates == 4 and not actual.training.use_compile
    assert actual.dataset.zarr_path == '/new'
    for override in ('training.seed=4', 'training.num_gpus=2', 'dataloader.batch_size=1',
                     'training.loop.total_train_steps=4', 'agent.architecture=other'):
        with pytest.raises(ValueError, match='forbidden'):
            resolve_training_config(incoming, overrides=[override])


@pytest.mark.parametrize('kind', ['dino', 'clip', 'siglip'])
def test_saved_hf_architecture_builds_without_initialization_assets(kind, monkeypatch):
    # Tiny *real* HF modules on CPU; this is not production model/asset verification.
    import transformers as hf
    from importlib import import_module
    classes = {'dino': ('DINO', hf.Dinov2Config, hf.AutoModel),
               'clip': ('CLIP', hf.CLIPVisionConfig, hf.CLIPVisionModel),
               'siglip': ('SigLIP', hf.SiglipVisionConfig, hf.SiglipVisionModel)}
    symbol, config_cls, model_cls = classes[kind]
    config = config_cls(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, image_size=16, patch_size=8)
    module = import_module('dexmani_policy.agents.obs_encoder.rgb.' + kind)
    forbidden = lambda *a, **k: (_ for _ in ()).throw(AssertionError('initialization asset access'))
    monkeypatch.setattr(model_cls, 'from_pretrained', forbidden)
    monkeypatch.setattr(hf.AutoConfig if kind == 'dino' else config_cls, 'from_pretrained', forbidden)
    encoder = getattr(module, symbol)(architecture=config.to_dict(), load_pretrained=False)
    assert next(encoder.backbone.parameters()).dtype == torch.bfloat16
    assert encoder.architecture['hidden_size'] == 32
    with torch.no_grad():
        result = encoder(torch.randn(1, 3, 16, 16, dtype=torch.bfloat16))['global_token']
    assert result.shape == (1, 32) and torch.isfinite(result).all()


def test_closed_text_restoration_has_no_online_fallback(monkeypatch):
    import hydra
    from types import SimpleNamespace
    from dexmani_policy.agents.loader import checkpoint_agent_config
    from dexmani_policy.agents.core import multi_task
    from test_infra_resume import DummyEncoder, TinyBackbone
    monkeypatch.setattr(multi_task, 'DPObsEncoder', lambda *a, **k: DummyEncoder())
    monkeypatch.setattr(multi_task, 'DiTDiffusion', lambda *a, **k: TinyBackbone())
    monkeypatch.setattr(multi_task, 'CLIPTextEncoder', lambda *a, **k: pytest.fail('online text construction'))
    model = multi_task.MultiTaskAgent(task_texts=['a', 'b', 'a'], text_embed_dim=4,
                                     rgb_backbone_name='resnet', n_emb=8)
    assert not any(k.startswith('text_encoder.') for k in model.state_dict())
    model.task_emb_table.copy_(torch.arange(8).reshape(2, 4))
    with pytest.raises(ValueError, match='Unknown'):
        model.get_text_emb(['missing'])
    torch.testing.assert_close(model.get_text_emb(['b', 'a']), model.text_proj(model.task_emb_table[[1, 0]]))
    cfg = OmegaConf.create({'agent': {
        '_target_': 'dexmani_policy.agents.core.multi_task.MultiTaskAgent',
        'task_texts': ['a', 'b', 'a'], 'rgb_backbone_name': 'resnet', 'n_emb': 8,
    }})
    state = model.state_dict()
    saved_cfg = checkpoint_agent_config(cfg, state)
    assert saved_cfg.agent.text_embed_dim == 4 and 'text_embed_dim' not in cfg.agent
    restored = hydra.utils.instantiate(saved_cfg.agent)
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored.get_text_emb(['b', 'a']), model.get_text_emb(['b', 'a']))
    old_state = dict(state, **{'text_encoder.text_backbone.weight': torch.ones(1)})
    with pytest.raises(RuntimeError, match='Unexpected key'):
        restored.load_state_dict(old_state, strict=True)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        restore_model_weights(SimpleNamespace(model_state=old_state), restored, None, 'cpu')
    with pytest.raises(ValueError, match='embedding table'):
        checkpoint_agent_config(cfg, {})


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('mask_kind', ['none', 'additive', 'boolean'])
def test_one_way_attention_matches_masked_attention_and_gradients(device, mask_kind):
    import copy
    from dexmani_policy.agents.action_decoders.backbone.one_way_transformer import Attention
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('NOT VERIFIED: requires CUDA')
    model = Attention(8, 2).to(device=device, dtype=torch.float64)
    reference = copy.deepcopy(model)
    inputs = [torch.randn(2, n, 8, device=device, dtype=torch.float64, requires_grad=True)
              for n in (3, 5, 5)]
    ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    allowed = torch.ones(3, 5, device=device, dtype=torch.bool)
    allowed[:, -2:] = False
    bias = torch.zeros(3, 5, device=device, dtype=torch.float64).masked_fill(~allowed, -torch.inf)
    mask = {'none': None, 'additive': bias, 'boolean': allowed}[mask_kind]
    output = model(*inputs, attn_mask=mask)
    q, k, v = [layer(x).reshape(2, x.shape[1], 2, 4).transpose(1, 2)
               for layer, x in zip((reference.q_proj, reference.k_proj, reference.v_proj), ref_inputs)]
    logits = q @ k.transpose(-1, -2) / 2
    if mask_kind != 'none':
        logits = logits + bias
    expected = reference.out_proj((logits.softmax(-1) @ v).transpose(1, 2).reshape(2, 3, 8))
    torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-12)
    weights = torch.randn_like(output)
    left = torch.autograd.grad((output * weights).sum(), (*inputs, *model.parameters()))
    right = torch.autograd.grad((expected * weights).sum(), (*ref_inputs, *reference.parameters()))
    for actual, wanted in zip(left, right):
        torch.testing.assert_close(actual, wanted, rtol=1e-9, atol=1e-11)


def test_one_way_attention_dropout_obeys_training_mode():
    from dexmani_policy.agents.action_decoders.backbone.one_way_transformer import Attention
    model = Attention(8, 2, attn_drop=1.)
    tokens = torch.randn(2, 3, 8)
    torch.testing.assert_close(model(tokens, tokens, tokens), model.out_proj.bias.expand(2, 3, 8))
    model.eval()
    output = model(tokens, tokens, tokens)
    assert not torch.equal(output, model.out_proj.bias.expand_as(output))
    torch.testing.assert_close(model(tokens, tokens, tokens), output, rtol=0, atol=0)


def test_decoder_rejects_unsupported_semantics_and_too_small_microbatch():
    from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow
    from dexmani_policy.agents.action_decoders.consistency_flow import ConsistencyFlowMatch
    from dexmani_policy.agents.action_decoders.diffusion import Diffusion
    for cls in (RectifiedFlow, ConsistencyFlowMatch):
        decoder = cls(torch.nn.Identity())
        with pytest.raises(ValueError, match='dim_groups'):
            decoder.compute_loss(torch.zeros(2, 1), torch.zeros(2, 2, 2), dim_groups={'joint': (0, 2)})
    decoder = ConsistencyFlowMatch(torch.nn.Identity())
    with pytest.raises(ValueError, match='micro-batch'):
        decoder.compute_loss(torch.zeros(1, 1), torch.zeros(1, 2, 2), ema_decoder=decoder)
    with pytest.raises(TypeError):
        Diffusion(torch.nn.Identity()).compute_loss(torch.zeros(2, 1), torch.zeros(2, 2, 2), misspelled_loss=True)


def test_diffusion_group_loss_has_independent_mathematical_expectation():
    from dexmani_policy.agents.action_decoders.diffusion import Diffusion
    class Zero(torch.nn.Module):
        def forward(self, x, timestep, context):
            return torch.zeros_like(x)
    decoder = Diffusion(Zero(), prediction_type='sample', aux_loss_weight=.5)
    actions = torch.tensor([[[2., 4., 6.]]])
    loss, _ = decoder.compute_loss(torch.zeros(1, 1), actions, dim_groups={'joint': (0, 1), 'eef': (1, 3)})
    assert loss.item() == 4. + .5 * ((16. + 36.) / 2.)


@pytest.mark.parametrize('kind', ['pointnet', 'multistage'])
def test_same_point_set_pool_output_and_gradient(kind):
    from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet, MultiStagePointNet
    set_seed(123)
    model = PointNet(output_channels=8) if kind == 'pointnet' else MultiStagePointNet(output_channels=8, hidden_channels=8, num_layers=2)
    cloud = torch.randn(2, 17, 3, requires_grad=True)
    permutation = torch.tensor([16, 0, 8, 7, 3, 5, 2, 9, 1, 14, 13, 4, 6, 15, 12, 11, 10])
    shuffled = cloud.detach()[:, permutation].clone().requires_grad_()
    output = model(cloud)['global_token']
    shuffled_output = model(shuffled)['global_token']
    assert output.std() > .01
    weights = torch.arange(1, 9, dtype=output.dtype)
    left = torch.autograd.grad((output * weights).sum(), (cloud, *model.parameters()))
    right = torch.autograd.grad((shuffled_output * weights).sum(), (shuffled, *model.parameters()))
    torch.testing.assert_close(output, shuffled_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(left[0][:, permutation], right[0], rtol=1e-4, atol=1e-5)
    assert any(grad.abs().sum() > 0 for grad in left[1:])
    for a, b in zip(left[1:], right[1:]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_pointnet_recipe_change_does_not_propagate_to_pointnext():
    from hydra import compose, initialize_config_dir
    config_dir = str(Path(__file__).resolve().parents[1] / 'dexmani_policy/configs')
    for name in ('dp3', 'dqrise'):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=name)
            assert cfg.agent.fps_random_config.use_shuffle_output is False
            other = compose(config_name=name, overrides=['agent.encoder_type=pointnext'])
            assert other.agent.fps_random_config.use_shuffle_output is True


def test_stop_unknown_listing_status_one_never_means_absent(tmp_path):
    ssh = tmp_path / 'ssh'
    ssh.write_text('#!' + sys.executable + '\n' + '''
import os, sys
from pathlib import Path
root = Path(os.environ['STOP_FIXTURE'])
command = sys.argv[-1]
with (root/'calls').open('a') as stream: stream.write(command + '\\n')
if 'list-sessions' in command:
    if (root/'listed').exists():
        print('tmux unexpected failure', file=sys.stderr)
        sys.exit(1)
    (root/'listed').touch()
    print('dex_fixture: 1 windows')
''')
    ssh.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'], STOP_FIXTURE=str(tmp_path))
    result = subprocess.run(['bash', 'scripts/remote/stop_remote.sh', '--force', 'dex_fixture'],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    calls = (tmp_path/'calls').read_text()
    assert 'tmux kill-session' not in calls
    assert 'Stopped.' not in result.stdout


def test_restore_rejects_optimizer_bound_before_parameter_reconstruction(tmp_path):
    source = bounded_trainer(tmp_path / 'source')
    source.train(max_updates=2)
    checkpoint = source.workspace.checkpoint_store.load(source.workspace.resolve_checkpoint_path('latest'))
    target = bounded_trainer(tmp_path / 'target')
    try:
        # Normalizer loading reconstructs its ParameterDict. This obsolete order
        # leaves the optimizer attached to parameters outside the restored model.
        restore_model_weights(checkpoint, target.raw_model, None, 'cpu')
        with pytest.raises(ValueError, match='restored model parameters'):
            restore_training_state(checkpoint, resume_contract=target.resume_contract,
                model=target.raw_model, ema_model=None, ema_updater=None,
                optimizer=target.optimizer, scheduler=target.scheduler, device='cpu')
    finally:
        target.workspace.close()
