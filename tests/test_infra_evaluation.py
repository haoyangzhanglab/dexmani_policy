import json
import sys
import os
import re
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
from dexmani_policy.evaluation.protocol import (wilson_interval,validate_heldout,
    mapped_task_seeds,atomic_json,build_eval_runner,save_eval_snapshot)
from dexmani_policy.agents.loader import read_best_ckpt_json, resolve_best_checkpoint
from dexmani_policy import select_best_ckpt as selector
from dexmani_policy import eval_best_ckpt as evaluator
from dexmani_policy import record_demo as demo
from dexmani_policy.deployment import runtime

# The mapping methods need no simulator. Supply only the package constant if
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

    def test_real_multitask_mapping(self):
        runner=MultiTaskSimRunner.__new__(MultiTaskSimRunner)
        runner.runners={'a':None,'b':None}; runner._task_seed_pools={'a':[0,1,2,3],'b':[100,101,102,103]}
        best={'selection':{'seeds':[0,1],'seed_protocol':runner.seed_protocol(),'task_seeds':mapped_task_seeds(runner,[0,1])}}
        validate_heldout(runner,best,[2,3])
        runner._task_seed_pools['b']=[102,103,100,101]
        with self.assertRaisesRegex(ValueError,'identity'): validate_heldout(runner,best,[2,3])
        runner._task_seed_pools['b']=[100,101,102,103]; runner.runners={'b':None,'a':None}
        with self.assertRaises(ValueError): validate_heldout(runner,best,[2,3])
        validate_heldout(Runner(),{'selection':{'seeds':[0,1]}},[2,3])

    def test_selection_publish_failure_and_final_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cfg=experiment(tmp); runner=Runner()
            cfg.eval.seed_manifest={'pool_id':'fixture','selection':{'a':[0,1]},
                'tie_break':{'a':[2]},'test':{'a':[3,4]}}
            with patch.object(selector,'build_eval_runner',return_value=runner), patch.object(selector,'load_ckpt_for_inference',side_effect=lambda path,*a,**kw: types.SimpleNamespace(_checkpoint_global_step=int(re.search(r'step=(\d+)',path.name)[1]))):
                selector.select_best_checkpoint(root,cfg,initial_episodes=2,batch_size=1,video_save_dir=root/'videos')
                first=read_best_ckpt_json(root); before=(root/'best_ckpt.json').read_bytes()
                summary=json.loads((root/first['selection_summary']).read_text())
                self.assertEqual(len(summary['stages']),4)
                self.assertEqual(len(set(s['video_dir'] for s in summary['stages'])),4)
                for result in summary['all_results']:
                    self.assertEqual(result['success_count'],sum(d['success'] for d in result['episode_details']))
                    self.assertEqual(result['n_episodes'],len(result['episode_details']))
                runner.success=False
                selector.select_best_checkpoint(root,cfg,initial_episodes=2,batch_size=1)
                self.assertTrue(read_best_ckpt_json(root)['selection']['selection_all_zero'])
                self.assertNotEqual(before,(root/'best_ckpt.json').read_bytes())
                before=(root/'best_ckpt.json').read_bytes()
                runner.success='error'
                with self.assertRaisesRegex(RuntimeError,'fatal'): selector.select_best_checkpoint(root,cfg,initial_episodes=2)
                self.assertEqual(before,(root/'best_ckpt.json').read_bytes())
                runner.success=True
                selector.select_best_checkpoint(root,cfg,initial_episodes=2,batch_size=1)
                third=read_best_ckpt_json(root); self.assertNotEqual(first['selection_id'],third['selection_id'])
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
                self.assertFalse(set(info['evaluation_seeds']) & set(third['selection']['seeds']))
            # Atomic publication interruption leaves a complete old pointer.
            before=(root/'best_ckpt.json').read_bytes()
            with patch('os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError): atomic_json(root/'best_ckpt.json',{'new':True})
            self.assertEqual(before,(root/'best_ckpt.json').read_bytes())
            (root/third['selection_summary']).unlink()
            with self.assertRaises(FileNotFoundError): read_best_ckpt_json(root)

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
                if entry != 'demo': self.assertEqual(list(saved.selection_seeds_excluded), [0, 1])
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
            multi._task_seed_pools = {'a': [0,1,2,3], 'b': [100,101,102,103]}
            with self.assertRaisesRegex(ValueError, 'identity'):
                validate_heldout(multi, old, [2,3])

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
        def predict_action(self, obs, **kwargs):
            torch.testing.assert_close(obs['rgb'], expected[None], rtol=0, atol=0)
            return {'pred_action': torch.zeros(1, 16, 19)}
    info = types.SimpleNamespace(observation_fields=('rgb',), action_mode='joint',
                                 n_obs_steps=2, horizon=16, inference_steps=10)
    policy = LoadedPolicy(Agent(), OmegaConf.to_container(cfg, resolve=True), info, device='cpu', seed=0)
    assert policy.predict({'rgb': raw}).shape == (15, 19)
