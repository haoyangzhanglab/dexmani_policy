import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import dexmani_policy.select_best_ckpt as selector
from dexmani_policy.evaluation.protocol import resolve_selection_checkpoint, select_eval_seeds, validate_heldout


class OfflineRunner:
    task_name = 'test'
    n_obs_steps = 2
    sensor_modalities = ['joint_state']
    rgb_preprocessing = {}
    record_video = False

    def __init__(self):
        self.plans = []

    def get_seed_list(self):
        return list(range(50))

    def run(self, agent, **kwargs):
        self.plans.append(list(self.eval_seeds))
        return {'episode_details': [{'seed': seed, 'success': seed % 2 == 0, 'steps': 8}
                                    for seed in self.eval_seeds]}


def test_selection_v3_single_evidence_copy_and_v2_reading(tmp_path, monkeypatch):
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    for step, pct in [(20, 20), (40, 40)]:
        (directory / f'epoch=0001-step={step:08d}-milestone={pct}pct.pt').touch()
    runner = OfflineRunner()
    cfg = OmegaConf.create({'_exp_dir': str(tmp_path), 'training': {'seed': 42}})
    monkeypatch.setattr(selector, 'build_eval_runner', lambda config: runner)
    monkeypatch.setattr(selector, 'load_ckpt_for_inference', lambda path, *a, **kw:
                        SimpleNamespace(_checkpoint_global_step=int(path.name.split('step=')[1].split('-')[0])))
    handoff = tmp_path / 'handoff.json'
    best, results = selector.select_best_checkpoint(tmp_path, cfg, initial_episodes=5, batch_size=2,
                                                    max_episodes=10, result_file=handoff)
    info, checkpoint = resolve_selection_checkpoint(tmp_path, handoff)
    assert info['_format'] == 'selection.v3' and checkpoint == best.path
    assert runner.plans[0] == runner.plans[1]
    assert runner.plans[2] == runner.plans[3]
    assert best.pct == 40
    assert all('episode_details' not in candidate for candidate in info['all_results'])
    summary_path = tmp_path / info['selection_summary']
    summary = json.loads(summary_path.read_text())
    assert 'stages' not in summary
    assert sum(len(stage['episode_details']) for stage in info['stages']) == 14
    test_plan = select_eval_seeds(runner, 99, 100, info['selection']['task_seeds'])
    assert len(test_plan['test']) == 43
    validate_heldout(runner, info, test_plan)
    result_path = tmp_path / json.loads(handoff.read_text())['selection_result']
    legacy = copy.deepcopy(info)
    legacy['_format'] = 'selection.v2'
    for candidate in legacy['all_results']:
        candidate['episode_details'] = [d for i in candidate['stage_indices'] for d in info['stages'][i]['episode_details']]
    old_path = result_path.parent / 'legacy.json'
    old_path.write_text(json.dumps(legacy))
    assert resolve_selection_checkpoint(tmp_path, old_path)[1] == best.path
    bad = copy.deepcopy(info)
    bad['stages'][0]['episode_details'].pop()
    old_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match='do not match'):
        resolve_selection_checkpoint(tmp_path, old_path)
    assert json.loads(result_path.read_text()) == info
    print('selection result JSON bytes old/new:', len(json.dumps(legacy, indent=2)), result_path.stat().st_size)


def test_failed_selection_keeps_stage_and_previous_handoff(tmp_path, monkeypatch):
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    (directory / 'epoch=0001-step=00000020-milestone=20pct.pt').touch()
    previous = tmp_path / 'best_ckpt.json'
    previous.write_text('{"previous": true}')
    runner = OfflineRunner()
    def failure(*args, **kwargs):
        raise RuntimeError('simulator unavailable')
    runner.run = failure
    monkeypatch.setattr(selector, 'build_eval_runner', lambda cfg: runner)
    monkeypatch.setattr(selector, 'load_ckpt_for_inference', lambda *a, **kw: SimpleNamespace(_checkpoint_global_step=20))
    cfg = OmegaConf.create({'_exp_dir': str(tmp_path), 'training': {'seed': 42}})
    with pytest.raises(RuntimeError, match='simulator unavailable'):
        selector.select_best_checkpoint(tmp_path, cfg, initial_episodes=2, batch_size=0)
    assert json.loads(previous.read_text()) == {'previous': True}
    summary = json.loads(next((tmp_path / 'eval_ckpt_selector').rglob('best_ckpt_selection.json')).read_text())
    assert summary['status'] == 'failed'
    assert summary['stages'][0]['status'] == 'requested'
    assert len(summary['stages'][0]['requested_task_seeds']['test']) == 2
