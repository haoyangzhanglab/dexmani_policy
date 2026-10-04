import json
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
    mapped_task_seeds,compute_eval_stats,atomic_json,build_eval_runner,save_eval_snapshot)
from dexmani_policy.agents.loader import read_best_ckpt_json
from dexmani_policy import select_best_ckpt as selector
from dexmani_policy import eval_best_ckpt as evaluator

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
            with patch.object(evaluator,'read_best_ckpt_json',return_value=best):
                _,ema,steps,_=evaluator._resolve_final_eval_request(cfg,root,'best',
                    ['eval.inference_steps=4'],cli_use_ema=False,cli_inference_steps=7)
                self.assertFalse(ema); self.assertEqual(steps,[7])

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
                with self.assertRaises(RuntimeError): selector.select_best_checkpoint(root,cfg,initial_episodes=2,batch_size=1)
                self.assertEqual(before,(root/'best_ckpt.json').read_bytes())
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
