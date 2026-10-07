import json
import sys
import os
import re
import subprocess
import tempfile
import types
import unittest
import pytest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from dexmani_policy.evaluation.protocol import (wilson_interval,validate_heldout,
    atomic_json,build_eval_runner,save_eval_snapshot)
from dexmani_policy.agents.loader import resolve_best_checkpoint
from dexmani_policy import select_best_ckpt as selector
from dexmani_policy import eval_best_ckpt as evaluator
from dexmani_policy import record_demo as demo
from dexmani_policy.deployment import runtime

# These runner boundary tests need no simulator. Supply only the package constant if
# the optional simulator is absent, then import the actual runner implementation.
try:
    from dexmani_policy.env_runner.multi_task_sim_runner import MultiTaskSimRunner
    from dexmani_policy.env_runner.sim_runner import SimRunner
except ModuleNotFoundError as exc:
    if exc.name != 'dexmani_sim': raise
    with patch.dict('sys.modules', {'dexmani_sim': types.SimpleNamespace(DATA_DIR=Path('/unused'))}):
        from dexmani_policy.env_runner.multi_task_sim_runner import MultiTaskSimRunner
        from dexmani_policy.env_runner.sim_runner import SimRunner


class Runner:
    task_name='a'; n_obs_steps=2; sensor_modalities=['joint_state']; rgb_preprocessing={}; record_video=False
    def __init__(self, success=True): self.success=success; self.calls=[]
    def get_seed_list(self): return list(range(20))
    def run(self,agent,**kwargs):
        self.calls.append((list(self.eval_seeds),kwargs))
        if self.success == 'error': raise RuntimeError('fatal fixture')
        return {'episode_details':[{'seed':s,'success':self.success,'steps':3} for s in self.eval_seeds]}


def experiment(tmp):
    root=Path(tmp); (root/'checkpoints').mkdir()
    for pct in [20,40]:
        (root/'checkpoints'/f'epoch=0000-step={pct:08d}-milestone={pct}pct.pt').write_bytes(b'fixture')
    cfg=OmegaConf.create({'_exp_dir':str(root),'training':{'seed':0,'device':'cpu'},'eval':{'use_ema':True,'inference_steps':10},
        'env_runner':{'env_kwargs':{'table_randomization':True}}})
    OmegaConf.save(cfg,root/'config.yaml')
    return root,cfg


def publish_best(root, step):
    record = {
        'ckpt_relpath': f'checkpoints/epoch=0000-step={step:08d}-milestone={step}pct.pt',
        'global_step': step, 'pct': step,
        'selection_id': f'selection-{step}',
        'selection_summary': f'selection-{step}.json',
        'inference': {'use_ema': step == 20, 'inference_steps': 10 if step == 20 else 3},
        'selection': {'seeds': [0, 1] if step == 20 else [2, 3],
                      'task_seeds': {'a': [0, 1] if step == 20 else [2, 3]}},
    }
    atomic_json(root / record['selection_summary'], {
        'status': 'success', 'selection_id': record['selection_id'],
        'best_checkpoint': {k: record[k] for k in ('ckpt_relpath', 'global_step', 'pct')},
        'selection': record['selection'],
    })
    atomic_json(root / 'best_ckpt.json', record)
    return record


class EvaluationInfraTests(unittest.TestCase):
    def test_wilson(self):
        self.assertIsNone(wilson_interval(0,0))
        self.assertAlmostEqual(wilson_interval(0,10)[1],.2775327998628892)
        self.assertAlmostEqual(wilson_interval(10,10)[0],1-.2775327998628892)
        self.assertAlmostEqual(sum(wilson_interval(5,10)),1)

    def test_effective_runner_inputs_and_cli_priority(self):
        from dexmani_policy.smoke_test import load_config
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=load_config('dp3'); OmegaConf.save(cfg,root/'config.yaml')
            cfg=OmegaConf.load(root/'config.yaml')
            cfg._exp_dir=str(root)
            cfg.eval.seed_manifest=None  # This test exercises legacy runner/snapshot configuration.
            cfg.env_runner.env_kwargs.table_random=True
            cfg.env_runner.env_kwargs.instance_random=True
            runner=SimRunner.__new__(SimRunner)
            runner.task_name=cfg.task_name; runner.env_kwargs=OmegaConf.to_container(cfg.env_runner.env_kwargs)
            runner.n_obs_steps=99; runner.sensor_modalities=[]; runner.rgb_preprocessing={}; runner.record_video=False
            with patch('dexmani_policy.evaluation.protocol.hydra.utils.instantiate',return_value=runner):
                actual=build_eval_runner(cfg)
            save_eval_snapshot(root/'output',cfg,actual,use_ema=False,inference_steps=7)
            snapshot=OmegaConf.load(root/'output/eval_config.yaml')
            self.assertEqual(snapshot.runner_inputs[cfg.task_name].n_obs_steps,cfg.agent.n_obs_steps)
            self.assertTrue(snapshot.runner_inputs[cfg.task_name].env_kwargs.randomize_table_height)
            self.assertTrue(snapshot.runner_inputs[cfg.task_name].env_kwargs.randomize_model_id)
            best={'inference':{'use_ema':True,'inference_steps':10}}
            with patch.object(evaluator,'resolve_best_checkpoint',return_value=(best, root/'checkpoints/a.pt')):
                _,ema,steps,_=evaluator._resolve_final_eval_request(cfg,root,'best',
                    ['eval.inference_steps=4'],cli_use_ema=False,cli_inference_steps=7)
                self.assertFalse(ema); self.assertEqual(steps,[7])
                best['inference']['inference_steps'] = 0
                _, ema, steps, _ = evaluator._resolve_final_eval_request(
                    cfg, root, 'best', [], cli_use_ema=False,
                    cli_inference_steps_list=[2, 5])
                self.assertFalse(ema); self.assertEqual(steps, [2, 5])

    def test_real_multitask_direct_plan(self):
        runner = MultiTaskSimRunner.__new__(MultiTaskSimRunner)
        a, b = Runner(), Runner()
        b.task_name = 'b'
        b.get_seed_list = lambda: [103, 102, 101, 100]
        runner.runners = {'a': a, 'b': b}
        best = {'selection': {'task_seeds': {'a': [0, 1], 'b': [100, 101]}}}
        plan = {'a': [2, 3], 'b': [102, 103]}
        validate_heldout(runner, best, plan)
        # Pool order no longer controls physical requests.
        b.get_seed_list = lambda: [100, 101, 102, 103]
        validate_heldout(runner, best, plan)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_heldout(runner, best, {'a': [2, 3], 'b': [101, 103]})
        validate_heldout(Runner(), {'selection': {'seeds': [0, 1]}}, {'a': [2, 3]})

    def test_selection_publish_failure_and_final_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cfg=experiment(tmp); runner=Runner()
            cfg.eval.seed_manifest={'pool_id':'fixture','selection':{'a':[0,1]},
                'tie_break':{'a':[2]},'test':{'a':[3,4]}}
            with patch.object(selector,'build_eval_runner',return_value=runner), patch.object(selector,'load_ckpt_for_inference',side_effect=lambda path,*a,**kw: types.SimpleNamespace(_checkpoint_global_step=int(re.search(r'step=(\d+)',path.name)[1]))):
                selector.select_best_checkpoint(root,cfg,video_save_dir=root/'videos')
                first=resolve_best_checkpoint(root)[0]; before=(root/'best_ckpt.json').read_bytes()
                summary=json.loads((root/first['selection_summary']).read_text())
                self.assertEqual(len(summary['stages']),4)
                self.assertEqual(len(set(s['video_dir'] for s in summary['stages'])),4)
                for result in summary['all_results']:
                    self.assertEqual(result['success_count'],sum(d['success'] for d in result['episode_details']))
                    self.assertEqual(result['n_episodes'],len(result['episode_details']))
                runner.success=False
                selector.select_best_checkpoint(root,cfg)
                self.assertTrue(resolve_best_checkpoint(root)[0]['selection']['selection_all_zero'])
                self.assertNotEqual(before,(root/'best_ckpt.json').read_bytes())
                before=(root/'best_ckpt.json').read_bytes()
                runner.success='error'
                with self.assertRaisesRegex(RuntimeError,'fatal'): selector.select_best_checkpoint(root,cfg)
                self.assertEqual(before,(root/'best_ckpt.json').read_bytes())
                runner.success=True
                selector.select_best_checkpoint(root,cfg)
                third=resolve_best_checkpoint(root)[0]; self.assertNotEqual(first['selection_id'],third['selection_id'])
            agent=types.SimpleNamespace(_checkpoint_global_step=40)
            with patch.object(evaluator,'build_eval_runner',return_value=runner), patch.object(evaluator,'load_ckpt_for_inference',return_value=agent):
                evaluator.evaluate_checkpoint_sweep(root,cfg,inference_steps_list=[2,5],episodes=2,use_ema=False)
                evaluator.evaluate_checkpoint_robotwin(root,cfg,inference_steps=3,episodes=2,use_ema=False)
            for snapshot in (root/'eval_dexsim').glob('*/eval_config.yaml'):
                saved=OmegaConf.load(snapshot)
                self.assertFalse(saved.request.use_ema)
                self.assertTrue(saved.effective_config.env_runner.env_kwargs.table_randomization)
                self.assertEqual(saved.runner_inputs.a.n_obs_steps,2)
            for result in (root/'eval_dexsim').rglob('result_details.json'):
                info=json.loads(result.read_text()); self.assertTrue((root/info['eval_config']).is_file())
                self.assertEqual(info['global_step'],40)
                self.assertFalse(set(info['evaluation_seeds']['a']) & set(third['selection']['task_seeds']['a']))
            # Atomic publication interruption leaves a complete old pointer.
            before=(root/'best_ckpt.json').read_bytes()
            with patch('os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError): atomic_json(root/'best_ckpt.json',{'new':True})
            self.assertEqual(before,(root/'best_ckpt.json').read_bytes())
            (root/third['selection_summary']).unlink()
            self.assertEqual(resolve_best_checkpoint(root)[0], third)
            pointer = json.loads((root/'best_ckpt.json').read_text())
            self.assertEqual(set(pointer), {'selection_result'})
            (root/pointer['selection_result']).unlink()
            with self.assertRaises(FileNotFoundError): resolve_best_checkpoint(root)[0]

    def test_best_is_pinned_across_publication(self):
        # Each entry runs real parameter resolution, concrete-path loading and
        # snapshot/result persistence; only the Agent and simulator are replaced.
        cases = [
            ('eval', [], True, [10]),
            ('eval', ['eval.inference_steps_list=[2,5]', 'eval.use_ema=false'], False, [2, 5]),
            ('eval', ['--ema', '--inference-steps', '7', 'eval.use_ema=false',
                      'eval.inference_steps=4'], True, [7]),
            ('demo', [], True, [10]),
            ('demo', ['--no-ema', '--inference-steps', '7'], False, [7]),
            ('direct', [], True, [10]),
            ('direct_override', [], False, [7]),
            ('direct_sweep', [], False, [2, 5]),
        ]
        for entry, extra, expected_ema, expected_steps in cases:
            with self.subTest(entry=entry, extra=extra), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp)
                root = project / 'experiments/dp/t/run'
                root.mkdir(parents=True)
                root, cfg = experiment(root)
                first = publish_best(root, 20)
                runner = Runner()
                module = demo if entry == 'demo' else evaluator
                def read_then_publish(directory):
                    resolved = resolve_best_checkpoint(directory)
                    publish_best(root, 40)
                    return resolved
                def load(path, use_ema, **kwargs):
                    self.assertEqual(path, root / first['ckpt_relpath'])
                    self.assertEqual(use_ema, expected_ema)
                    return types.SimpleNamespace(_checkpoint_global_step=20)
                argv = ['test', '--policy-name', 'dp', '--task-name', 't',
                        '--exp-name', 'run', '--episodes', '20']
                if entry == 'eval': argv += ['--no-videos']
                with patch.object(module, 'ROOT_DIR', project), \
                     patch.object(module, 'resolve_best_checkpoint', side_effect=read_then_publish) as read, \
                     patch.object(module, 'build_eval_runner', return_value=runner), \
                     patch.object(module, 'load_ckpt_for_inference', side_effect=load) as restore, \
                     patch.object(sys, 'argv', argv + extra):
                    if entry == 'direct':
                        evaluator.evaluate_checkpoint_robotwin(root, cfg, episodes=20)
                    elif entry == 'direct_override':
                        # A concrete path plus its record must retain best held-out semantics.
                        resolved = module.resolve_best_checkpoint(root)
                        evaluator.evaluate_checkpoint_robotwin(root, cfg, episodes=20,
                            ckpt_tag_or_path=str(resolved[1]), resolved_best=resolved,
                            use_ema=False, inference_steps=7)
                    elif entry == 'direct_sweep':
                        evaluator.evaluate_checkpoint_sweep(root, cfg, episodes=20,
                            use_ema=False, inference_steps_list=[2, 5])
                    else:
                        module.main()
                    self.assertEqual(read.call_count, 1)
                    self.assertEqual(restore.call_count, 1)
                self.assertEqual([call[1]['inference_steps'] for call in runner.calls], expected_steps)
                for seeds, _ in runner.calls:
                    if entry != 'demo':
                        self.assertEqual(set(seeds), set(range(2, 20)))
                snapshots = list(root.rglob('eval_config.yaml'))
                self.assertEqual(len(snapshots), 1)
                saved = OmegaConf.load(snapshots[0]).request
                self.assertEqual(saved.selection_id, first['selection_id'])
                self.assertEqual(saved.selection_summary, first['selection_summary'])
                self.assertEqual(OmegaConf.to_container(saved.selection), first['selection'])
                self.assertEqual(saved.checkpoint, first['ckpt_relpath'])
                self.assertEqual(saved.global_step, 20)
                self.assertEqual(saved.use_ema, expected_ema)
                self.assertEqual(list(saved.inference_steps_list), expected_steps)
                self.assertEqual(saved.heldout_from_selection, entry != 'demo')
                self.assertEqual(list(saved.task_seeds.a), runner.calls[0][0])
                if entry != 'demo': self.assertEqual(list(saved.selection_seeds_excluded.a), [0, 1])
                results = list(root.rglob('result_details.json'))
                self.assertEqual(len(results), len(expected_steps))
                for result in results:
                    info = json.loads(result.read_text())
                    self.assertEqual(info['global_step'], 20)
                    self.assertEqual(info['use_ema'], expected_ema)
                    self.assertTrue((root / info['eval_config']).is_file())

    def test_inspection_pins_best_and_historical_boundaries(self):
        from dexmani_policy.smoke_test import load_config
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = experiment(tmp)
            OmegaConf.save(load_config('dp3'), root / 'config.yaml')
            first = publish_best(root, 20)
            def read_then_publish(directory):
                resolved = resolve_best_checkpoint(directory)
                publish_best(root, 40)
                return resolved
            with patch.object(runtime, 'resolve_best_checkpoint', side_effect=read_then_publish) as read:
                info = runtime.inspect_policy(root)
                self.assertEqual(read.call_count, 1)
            self.assertEqual(info.checkpoint_path, root / first['ckpt_relpath'])
            self.assertEqual((info.weights, info.inference_steps), ('ema', 10))
            old = {k: v for k, v in first.items() if k not in ('selection_id', 'selection_summary')}
            atomic_json(root / 'best_ckpt.json', old)
            with self.assertWarnsRegex(UserWarning, 'Historical best'):
                resolved = resolve_best_checkpoint(root)
            self.assertEqual(resolved[0], old)
            with patch.object(evaluator, 'build_eval_runner', return_value=Runner()), \
                 patch.object(evaluator, 'load_ckpt_for_inference', return_value=types.SimpleNamespace(_checkpoint_global_step=20)):
                evaluator.evaluate_checkpoint_robotwin(root, cfg, resolved_best=resolved, episodes=2)
            multi = MultiTaskSimRunner.__new__(MultiTaskSimRunner)
            multi.runners = {'a': None, 'b': None}
            with self.assertRaisesRegex(ValueError, 'evidence'):
                validate_heldout(multi, old, {'a': [2,3], 'b': [102,103]})

    def test_actual_rsync_options_two_rounds(self):
        script=Path('scripts/remote/sync_down.sh').read_text()
        arrays=re.findall(r'PASS[123]_OPTS=\(.*?\n\)',script,re.S)
        self.assertEqual(len(arrays),3)
        self.assertNotIn('--partial',arrays[0])
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'src'; dst=Path(tmp)/'dst'; src.mkdir(); dst.mkdir()
            (src/'checkpoints').mkdir(); (src/'checkpoints/old.pt').write_bytes(b'old weights')
            (src/'checkpoints/latest.pt').symlink_to('old.pt')
            (src/'best_ckpt.json').write_text('"old.pt"'); os.utime(src/'best_ckpt.json',(10,10))
            commands='WANDB_EXCLUDE=(); DRY_RUN=""\n'+'\n'.join(arrays)+ '\n'+'\n'.join(f'rsync "${{PASS{i}_OPTS[@]}}" "$SRC/" "$DST/"' for i in (1,2,3))
            env=dict(os.environ,SRC=str(src),DST=str(dst))
            subprocess.run(['bash','-c',commands],env=env,check=True,capture_output=True)
            inode=(dst/'checkpoints/old.pt').stat().st_ino
            (src/'checkpoints/new.pt').write_bytes(b'new weights')
            (src/'checkpoints/latest.pt').unlink(); (src/'checkpoints/latest.pt').symlink_to('new.pt')
            (src/'best_ckpt.json').write_text('"new.pt"'); os.utime(src/'best_ckpt.json',(10,10))
            (src/'eval_config.yaml').write_text('nfe: 5\n')
            subprocess.run(['bash','-c',commands],env=env,check=True,capture_output=True)
            self.assertEqual((dst/'best_ckpt.json').read_text(),'"new.pt"')
            self.assertEqual((dst/'checkpoints/old.pt').stat().st_ino,inode)
            self.assertEqual(os.readlink(dst/'checkpoints/latest.pt'),'new.pt')
            self.assertTrue((dst/'eval_config.yaml').exists())


if __name__=='__main__': unittest.main()


def local_sync_script(tmp_path):
    """Run the real script/rsync with local trees; no SSH transport is possible."""
    source, target = tmp_path/'remote', tmp_path/'local'
    source.mkdir(); target.mkdir()
    text = Path('scripts/remote/sync_down.sh').read_text()
    text = text.replace('REMOTE_EXP="$SERVER:/data_ssd/ZHY/experiments"', f'REMOTE_EXP="{source}"')
    text = text.replace('LOCAL_EXP="$PROJECT_ROOT/experiments"', f'LOCAL_EXP="{target}"')
    assert '$SERVER:/data_ssd' not in text
    script = tmp_path/'sync_down.sh'; script.write_text(text)
    return source, target, script


@pytest.mark.parametrize('depth', ['', 'dp/task', 'dp/task/run'])
def test_sync_immutable_and_mutable_boundaries(tmp_path, depth):
    source, target, script = local_sync_script(tmp_path)
    src, dst = source/'dp/task/run', target/'dp/task/run'
    for root in (src, dst):
        (root/'checkpoints').mkdir(parents=True)
        (root/'eval_ckpt_selector/selection').mkdir(parents=True)
        (root/'config.yaml').write_text('recipe: same\n')
    immutable = ['checkpoints/milestone.pt', 'source.zip', 'eval_config.yaml',
                 'eval_ckpt_selector/selection/selection_result.json', 'result_details.json', '_result.txt']
    mutable = ['best_ckpt.json', 'eval_ckpt_selector/selection/best_ckpt_selection.json']
    for name in immutable + mutable:
        (src/name).write_bytes(b'NEW!'); (dst/name).write_bytes(b'OLD!')
        os.utime(src/name, (10,10)); os.utime(dst/name, (10,10))
    (src/'checkpoints/new.pt').write_bytes(b'complete checkpoint')
    (src/'checkpoints/latest.pt').symlink_to('new.pt')
    (dst/'checkpoints/latest.pt').symlink_to('milestone.pt')
    (src/'metrics.jsonl').write_text('new metrics\n'); (dst/'metrics.jsonl').write_text('old\n')
    (src/'vqvae_hand_best.pt').write_bytes(b'new best weights'); (dst/'vqvae_hand_best.pt').write_bytes(b'old')
    (src/'vqvae_hand_best.pt.tmp').write_bytes(b'old writer partial')
    (src/'wandb').mkdir(); (src/'wandb/best_ckpt.json').write_text('private')
    (src/'.dexmani-publish-inflight.tmp').write_bytes(b'incomplete')
    (src/'.dexmani-publish-export.tmp.npz').write_bytes(b'incomplete codebook')
    (src/'.dexmani-publish-directory.tmp').mkdir()
    (src/'.dexmani-publish-directory.tmp/legitimate').write_text('keep directory')
    result = subprocess.run(['bash', str(script), *([depth] if depth else [])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in immutable: assert (dst/name).read_bytes() == b'OLD!', name
    for name in mutable: assert (dst/name).read_bytes() == b'NEW!', name
    assert (dst/'checkpoints/latest.pt').read_bytes() == b'complete checkpoint'
    assert (dst/'metrics.jsonl').read_text() == 'new metrics\n'
    assert (dst/'vqvae_hand_best.pt').read_bytes() == b'new best weights'
    assert not (dst/'vqvae_hand_best.pt.tmp').exists()
    assert not (dst/'wandb').exists()
    assert not (dst/'.dexmani-publish-inflight.tmp').exists()
    assert not (dst/'.dexmani-publish-export.tmp.npz').exists()
    assert (dst/'.dexmani-publish-directory.tmp/legitimate').exists()


@pytest.mark.parametrize('identity', ['config.yaml', 'source_manifest.json'])
def test_sync_rejects_existing_run_identity_conflict(tmp_path, identity):
    source, target, script = local_sync_script(tmp_path)
    src, dst = source/'dp/task/run', target/'dp/task/run'
    for root in (src, dst): root.mkdir(parents=True)
    (src/identity).write_text('NEW!'); (dst/identity).write_text('OLD!')
    os.utime(src/identity,(10,10)); os.utime(dst/identity,(10,10))
    (src/'best_ckpt.json').write_text('new alias')
    (dst/'best_ckpt.json').write_text('old alias')
    result = subprocess.run(['bash',str(script)],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'conflict' in result.stderr.lower()
    assert (dst/identity).read_text() == 'OLD!'
    assert (dst/'best_ckpt.json').read_text() == 'old alias'


@pytest.mark.parametrize('rc,listing', [(255,''), (7,''), (0,''), (0,'b\na\n')])
def test_sync_list_preserves_errors(tmp_path, rc, listing):
    binary = tmp_path/'ssh'
    binary.write_text('#!/bin/sh\nprintf "%s" "$LISTING"\nif [ "$RC" != 0 ]; then echo lookup-failed >&2; fi\nexit "$RC"\n')
    binary.chmod(0o755)
    env = dict(os.environ,PATH=str(tmp_path)+':'+os.environ['PATH'],RC=str(rc),LISTING=listing)
    result = subprocess.run(['bash','scripts/remote/sync_down.sh','--list'],env=env,capture_output=True,text=True)
    assert result.returncode == rc
    if rc: assert 'lookup-failed' in result.stderr
    elif listing: assert result.stdout.endswith('a\nb\n')
    else: assert 'No experiments found' in result.stdout


@pytest.mark.parametrize('rc', [23,24])
def test_failed_download_never_updates_references(tmp_path, rc):
    source,target,script = local_sync_script(tmp_path)
    binary = tmp_path/'rsync'
    binary.write_text('#!/bin/sh\ncase " $* " in *--ignore-existing*) exit '+str(rc)+';; esac\n'
                      'case " $* " in *--out-format*) exit 0;; esac\n'
                      'echo unexpected-mutable-pass >&2\nexit 99\n')
    binary.chmod(0o755)
    (target/'best_ckpt.json').write_text('old complete pointer')
    result = subprocess.run(['bash',str(script)],capture_output=True,text=True,
        env=dict(os.environ,PATH=str(tmp_path)+':'+os.environ['PATH']))
    assert result.returncode == rc
    assert 'unexpected-mutable-pass' not in result.stderr
    assert (target/'best_ckpt.json').read_text() == 'old complete pointer'
    assert not list(target.glob('.dexmani-publish-*'))


def test_atomic_publication_and_sync_during_write(tmp_path):
    from dexmani_policy.utils.atomic import atomic_path
    source, target, script = local_sync_script(tmp_path)
    final = source/'source.zip'
    with atomic_path(final, overwrite=False) as temporary:
        temporary.write_bytes(b'first half')
        assert not final.exists()
        subprocess.run(['bash', str(script)], check=True, capture_output=True)
        assert not (target/final.name).exists()
        assert not list(target.glob('.dexmani-publish-*'))
        temporary.write_bytes(b'complete archive')
    subprocess.run(['bash', str(script)], check=True, capture_output=True)
    assert (target/final.name).read_bytes() == b'complete archive'
    with pytest.raises(FileExistsError):
        with atomic_path(final, overwrite=False) as temporary:
            temporary.write_bytes(b'cannot replace')
    with pytest.raises(RuntimeError):
        with atomic_path(final) as temporary:
            temporary.write_bytes(b'failed write')
            assert final.read_bytes() == b'complete archive'
            raise RuntimeError('interrupted')
    assert final.read_bytes() == b'complete archive'
    assert not list(source.glob('.dexmani-publish-*'))


def test_atomic_writers_preserve_files_on_errors(tmp_path, monkeypatch):
    from dexmani_policy.training.source_snapshot import save_source_snapshot
    from dexmani_policy.training.workspace import TrainWorkspace
    (tmp_path/'dexmani_policy').mkdir()
    (tmp_path/'dexmani_policy/a.py').write_text('valid source')
    save_source_snapshot(tmp_path, root=tmp_path)
    archive = (tmp_path/'source.zip').read_bytes()
    with pytest.raises(FileExistsError): save_source_snapshot(tmp_path, root=tmp_path)
    assert (tmp_path/'source.zip').read_bytes() == archive
    workspace = TrainWorkspace.__new__(TrainWorkspace)
    workspace.output_dir, workspace.wandb_logger = tmp_path, None
    workspace.save_hydra_config(OmegaConf.create({'value': 'complete'}))
    previous = (tmp_path/'config.yaml').read_bytes()
    def fail_save(cfg, path, **kwargs):
        Path(path).write_text('partial')
        raise OSError('interrupted config')
    monkeypatch.setattr(OmegaConf, 'save', fail_save)
    with pytest.raises(OSError): workspace.save_hydra_config(OmegaConf.create({'value': 'new'}))
    assert (tmp_path/'config.yaml').read_bytes() == previous
    evaluator._write_result(tmp_path/'_result.txt', 'complete\n')
    with pytest.raises(FileExistsError): evaluator._write_result(tmp_path/'_result.txt', 'new\n')
    assert (tmp_path/'_result.txt').read_text() == 'complete\n'
    assert not list(tmp_path.glob('.dexmani-publish-*'))


def test_demo_does_not_read_unused_test_manifest(tmp_path,monkeypatch):
    root = tmp_path/'experiments/dp/task/run'; root.mkdir(parents=True)
    root,cfg = experiment(root)
    cfg.eval.seed_manifest = str(tmp_path/'missing-protocol.json')
    OmegaConf.save(cfg,root/'config.yaml')
    runner = Runner()
    monkeypatch.setattr(demo,'ROOT_DIR',tmp_path)
    monkeypatch.setattr(demo,'build_eval_runner',lambda cfg: runner)
    monkeypatch.setattr(demo,'load_ckpt_for_inference',lambda *a,**kw: types.SimpleNamespace(_checkpoint_global_step=20))
    monkeypatch.setattr(sys,'argv',['demo','--policy-name','dp','--task-name','task',
        '--exp-name','run','--ckpt-tag','20pct','--seeds','2','3'])
    demo.main()
    assert runner.calls[0][0] == [2,3]
    snapshot = OmegaConf.load(next(root.rglob('eval_config.yaml')))
    assert snapshot.seed_manifest is None
    assert snapshot.request.heldout_from_selection is False
    assert snapshot.effective_config.eval.seed_manifest == cfg.eval.seed_manifest
    with pytest.raises(FileNotFoundError):
        with patch.object(evaluator,'build_eval_runner',return_value=Runner()):
            evaluator.evaluate_checkpoint_robotwin(root,cfg,ckpt_tag_or_path='20pct',episodes=2)


@pytest.mark.parametrize('seeds', [[], ['99'], ['-1'], ['2','2']])
def test_demo_rejects_unavailable_or_invalid_seeds(tmp_path,monkeypatch,seeds):
    root = tmp_path/'experiments/dp/task/run'; root.mkdir(parents=True)
    experiment(root)
    runner = Runner()
    monkeypatch.setattr(demo,'ROOT_DIR',tmp_path)
    monkeypatch.setattr(demo,'build_eval_runner',lambda cfg: runner)
    monkeypatch.setattr(demo,'load_ckpt_for_inference',lambda *a,**kw: types.SimpleNamespace(_checkpoint_global_step=20))
    monkeypatch.setattr(sys,'argv',['demo','--policy-name','dp','--task-name','task',
        '--exp-name','run','--ckpt-tag','20pct','--seeds',*seeds])
    with pytest.raises(ValueError): demo.main()
    assert not runner.calls


def test_multitask_direct_consumer_validates_episode_fields():
    runner = MultiTaskSimRunner.__new__(MultiTaskSimRunner)
    child = Runner(); child.task_text = 'a'
    child.run = lambda *a,**kw: {'success_rate':0.,'avg_steps':None,
        'episode_details':[{'seed':1,'success':'False','steps':None}]}
    runner.runners = {'a':child}
    with pytest.raises(RuntimeError,match='failed for tasks'):
        runner.run(None,task_seeds={'a':[1]},inference_steps=2)


@pytest.mark.parametrize('kind', ['runtime', 'unexpected', 'interrupt', 'episode'])
def test_multitask_errors_never_become_task_outcomes(kind):
    from dexmani_policy.env_runner.base_runner import EvalEpisodeError

    error = {
        'runtime': RuntimeError('fixture failure'),
        'unexpected': KeyError('fixture failure'),
        'interrupt': KeyboardInterrupt(),
        'episode': EvalEpisodeError('model', 1, 'fixture failure'),
    }[kind]
    def fail(*args, **kwargs):
        raise error

    runner = MultiTaskSimRunner.__new__(MultiTaskSimRunner)
    runner.runners = {'a': types.SimpleNamespace(task_text='a', run=fail)}
    with patch.object(runner, 'print_summary') as summary:
        if kind in ('interrupt', 'episode'):
            with pytest.raises(type(error)) as raised:
                runner.run(None, task_seeds={'a': [1]}, inference_steps=2)
            assert raised.value is error
            summary.assert_not_called()
        else:
            with pytest.raises(RuntimeError, match='Refusing to report'):
                runner.run(None, task_seeds={'a': [1]}, inference_steps=2)
            record = summary.call_args.args[0]['a']
            assert record['success_rate'] is None
            assert record['episode_details'] == []
            assert record['error_type'] == type(error).__name__
            assert record['error_category'] == ('runtime_error' if kind == 'runtime' else 'KeyError')


def test_snapshot_never_rebinds_or_reopens_protocol(tmp_path,monkeypatch):
    from dexmani_policy.evaluation import protocol
    root,cfg = experiment(tmp_path)
    cfg.eval.seed_manifest = '/missing/unused.json'
    monkeypatch.setattr(protocol,'bind_seed_manifest',lambda *a,**kw: pytest.fail('snapshot rebound protocol'))
    monkeypatch.setattr(protocol,'load_seed_manifest',lambda *a,**kw: pytest.fail('snapshot reopened manifest'))
    resolved = {'manifest':{'pool_id':'already-validated'},'roles':{'test':{'a':[2]}},'sha256':'evidence'}
    save_eval_snapshot(root/'snapshot',cfg,Runner(),protocol=resolved,task_seeds={'a':[2]})
    assert OmegaConf.to_container(OmegaConf.load(root/'snapshot/eval_config.yaml').seed_manifest) == resolved


@pytest.mark.parametrize('interrupt', [False,True])
def test_source_writer_sync_before_publication(tmp_path,monkeypatch,interrupt):
    import zipfile
    from dexmani_policy.training.source_snapshot import save_source_snapshot
    source,target,script = local_sync_script(tmp_path)
    code = tmp_path/'code/dexmani_policy'; code.mkdir(parents=True)
    (code/'sample.py').write_text('complete source')
    original = os.link
    published = []
    def publish(temporary, destination):
        if destination.name == 'source.zip':
            assert not destination.exists()
            with zipfile.ZipFile(temporary) as archive:
                assert archive.read('dexmani_policy/sample.py') == b'complete source'
            subprocess.run(['bash',str(script)],check=True,capture_output=True)
            assert not (target/'source.zip').exists()
            if interrupt: raise OSError('publication interrupted')
        original(temporary,destination)
        published.append(destination.name)
    monkeypatch.setattr(os,'link',publish)
    if interrupt:
        with pytest.raises(OSError): save_source_snapshot(source,root=code.parent)
        assert not (source/'source.zip').exists()
    else:
        save_source_snapshot(source,root=code.parent)
        assert published == ['source.zip','source_manifest.json']
        subprocess.run(['bash',str(script)],check=True,capture_output=True)
        assert (target/'source.zip').read_bytes() == (source/'source.zip').read_bytes()
    assert not list(source.glob('.dexmani-publish-*'))


def test_checkpoint_failed_write_cleans_only_its_temporary(tmp_path,monkeypatch):
    import torch
    from dexmani_policy.training.checkpoint import CheckpointStore, TrainCheckpoint
    checkpoint = TrainCheckpoint(epoch=0,global_step=0,next_micro_step=0,model_state={},
        ema_model_state=None,optimizer_state={},scheduler_state={},resume_contract={},
        ema_updater_step=None,ema_decay=None,rng_states=[])
    final = tmp_path/'milestone.pt'; final.write_bytes(b'old complete weight')
    unrelated = tmp_path/'.dexmani-publish-other.tmp'; unrelated.write_bytes(b'other writer')
    def fail(payload,path):
        Path(path).write_bytes(b'partial weight')
        assert final.read_bytes() == b'old complete weight'
        raise OSError('serialization interrupted')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError): CheckpointStore(tmp_path).save('milestone.pt',checkpoint)
    assert final.read_bytes() == b'old complete weight'
    assert list(tmp_path.glob('.dexmani-publish-*')) == [unrelated]


def test_saved_rgb_recipe_reaches_simulation_and_real(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from dexmani_policy.env_runner.base_runner import BaseRunner
    from dexmani_policy.datasets.preprocessing import preprocess_validation_rgb
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.deployment.runtime import LoadedPolicy
    cfg = OmegaConf.create(OmegaConf.to_container(load_config('dp'), resolve=True))
    cfg.dataset.rgb_preprocess_size = [7, 9]
    cfg.dataset.rgb_random_crop_size = [5, 6]
    cfg._exp_dir = str(tmp_path)
    OmegaConf.save(cfg, tmp_path/'config.yaml')
    raw = np.random.default_rng(4).integers(0, 256, (2, 4, 8, 3), dtype=np.uint8)
    expected = preprocess_validation_rgb(raw, resize_hw=(7, 9), center_crop_hw=(5, 6),
                                        keep_uint8=False).mul(255).round().clamp(0, 255).to(torch.uint8)
    runner = BaseRunner.__new__(BaseRunner)
    runner.task_name = cfg.task_name
    monkeypatch.setattr('dexmani_policy.evaluation.protocol.hydra.utils.instantiate', lambda *a: runner)
    runner = build_eval_runner(cfg)
    runner.get_stacked_obs = lambda: {'rgb': raw.copy()}
    torch.testing.assert_close(runner.get_obs_batch('cpu')['rgb'], expected[None], rtol=0, atol=0)
    class Agent:
        consumed_observation_fields = ("rgb",)
        def predict_action(self, obs, **kwargs):
            torch.testing.assert_close(obs['rgb'], expected[None], rtol=0, atol=0)
            return {'pred_action': torch.zeros(1, 16, 19)}
    info = types.SimpleNamespace(observation_fields=('rgb',), action_mode='joint',
                                 n_obs_steps=2, horizon=16, inference_steps=10)
    policy = LoadedPolicy(Agent(), OmegaConf.to_container(cfg, resolve=True), info, device='cpu', seed=0)
    assert policy.predict({'rgb': raw}).shape == (15, 19)


@pytest.mark.parametrize('fault', ['external_link', 'internal_link', 'broken_link', 'directory'])
def test_milestone_paths_fail_before_selection_and_preserve_best(tmp_path, monkeypatch, fault):
    from dexmani_policy.evaluation.protocol import discover_milestone_checkpoints, resolve_checkpoint_path

    root, cfg = experiment(tmp_path)
    valid = root / 'checkpoints/epoch=0000-step=00000020-milestone=20pct.pt'
    invalid = root / 'checkpoints/epoch=0000-step=00000040-milestone=40pct.pt'
    assert resolve_checkpoint_path(root, '20pct')[0] == valid
    (root / 'checkpoints/latest.pt').symlink_to(valid.name)
    assert resolve_checkpoint_path(root, 'latest')[0] == valid
    invalid.unlink()
    if fault == 'directory':
        invalid.mkdir()
    elif fault == 'internal_link':
        invalid.symlink_to(valid.name)
    else:
        target = tmp_path / 'other_experiment/checkpoints/model.pt'
        if fault == 'external_link':
            target.parent.mkdir(parents=True)
            target.write_bytes(b'other experiment')
        invalid.symlink_to(target)
    previous = b'{"previous": "selection"}'
    (root / 'best_ckpt.json').write_bytes(previous)
    monkeypatch.setattr(selector, 'build_eval_runner', lambda cfg: pytest.fail('invalid candidate reached runner'))
    for operation in (lambda: discover_milestone_checkpoints(root),
                      lambda: resolve_checkpoint_path(root, '40pct'),
                      lambda: selector.select_best_checkpoint(root, cfg)):
        with pytest.raises(ValueError):
            operation()
    assert (root / 'best_ckpt.json').read_bytes() == previous


@pytest.fixture
def episode_runner():
    import numpy as np
    import torch
    from dexmani_policy.env_runner.base_runner import BaseRunner

    class Env:
        video_fps = 15
        def __init__(self, runner):
            self.runner = runner
            self.closed = 0
        def reset(self, seed, options=None):
            self.action_cnt = 0
            return {'joint_state': np.zeros(1, dtype=np.float32)}, {}
        def step(self, action):
            if self.runner.episode_error is not None:
                raise self.runner.episode_error
            self.action_cnt += 1
            return {'joint_state': np.zeros(1, dtype=np.float32)}, 0, True, False, {
                'success': True, 'success_condition': True}
        def get_video(self):
            return np.zeros((1, 2, 2, 3), dtype=np.uint8)
        def close(self):
            self.closed += 1

    class EpisodeRunner(BaseRunner):
        def __init__(self):
            super().__init__(1, 1, sensor_modalities=['joint_state'], clear_cache_freq=2)
            self.envs = []
            self.eval_seeds = [7]
            self.episode_error = None
        def make_env(self):
            env = Env(self)
            self.envs.append(env)
            return env
        def get_seed_list(self):
            return self.eval_seeds

    agent = torch.nn.Linear(1, 1)
    agent.predict_action = lambda **kwargs: {'control_action': torch.zeros(1, 1, 1)}
    return EpisodeRunner(), agent


@pytest.mark.parametrize('fault', ['mkdir', 'encode'])
@pytest.mark.parametrize('episode_fails', [False, True])
def test_video_failure_preserves_episode_outcome(episode_runner, tmp_path, monkeypatch, fault, episode_fails):
    from dexmani_policy.env_runner.base_runner import EvalEpisodeError

    runner, agent = episode_runner
    output = tmp_path / 'videos'
    if fault == 'mkdir':
        output.write_bytes(b'not a directory')

    def encode(*args):
        assert fault == 'encode', 'encoding must not start when mkdir fails'
        raise OSError('encoding failed')

    monkeypatch.setattr(runner, '_encode_video', encode)
    if episode_fails:
        runner.episode_error = ValueError('original model failure')
        with pytest.raises(EvalEpisodeError, match='original model failure') as error:
            runner.run(agent, video_save_dir=output)
        assert error.value.__cause__ is runner.episode_error
        assert error.value.seed == 7 and error.value.category == 'value_error'
    else:
        result = runner.run(agent, video_save_dir=output)
        assert result['success_rate'] == 1.0 and result['episodes_collected'] == 1
        assert result['episode_details'][0]['steps'] == 1
        assert result['videos'] == []
    assert all(env.closed == 1 for env in runner.envs)


@pytest.mark.parametrize('episodes', [1, 2, 3, 4, 5])
def test_environment_refresh_only_between_episodes(episode_runner, episodes):
    runner, agent = episode_runner
    runner.eval_seeds = list(range(episodes))
    result = runner.run(agent, eval_episodes=episodes)
    assert result['episodes_collected'] == episodes and result['success_rate'] == 1.0
    assert [d['seed'] for d in result['episode_details']] == list(range(episodes))
    assert len(runner.envs) == (episodes + 1) // 2
    assert all(env.closed == 1 for env in runner.envs)
