"""Protocol control-flow tests; rollout fixtures are not simulator validation."""
import copy
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from test_infra_evaluation import Runner, experiment
from dexmani_policy import select_best_ckpt as selector, eval_best_ckpt as evaluator
from dexmani_policy.agents.loader import resolve_best_checkpoint
from dexmani_policy.evaluation.protocol import load_seed_manifest, fixed_test_seeds, bind_seed_manifest
from scripts.eval.make_seed_manifest import make_manifest


class Pool(Runner):
    def __init__(self, multi=False, size=200):
        super().__init__(False)
        self.multi, self.size = multi, size
    def get_seed_list(self): return list(range(self.size))
    def map_eval_seeds(self,seeds):
        result={'a':list(seeds)}
        if self.multi: result['b']=[s+1000 for s in seeds]
        return result
    def run(self,agent,**kwargs):
        self.calls.append((list(self.eval_seeds),kwargs))
        return {'episode_details':[dict(task_name=t,seed=s,success=False,steps=None)
                for t,seeds in self.map_eval_seeds(self.eval_seeds).items() for s in seeds]}


def test_manifest_generation_identity_and_validation(tmp_path):
    for multi in (False,True):
        runner=Pool(multi)
        m=make_manifest(runner,pool_id='synthetic-fixture')
        assert m==make_manifest(runner,pool_id='synthetic-fixture')
        path=tmp_path/'manifest.json'; path.write_text(json.dumps(m))
        p=load_seed_manifest(m,runner)
        assert p==load_seed_manifest(path,runner)==load_seed_manifest(OmegaConf.create(m),runner)
        assert [len(p['roles'][r]) for r in ['selection','tie_break','test']]==[25,5,100]
        assert len(set(sum(p['roles'].values(),[])))==130
        runner.size=201
        with pytest.raises(ValueError,match='pool hash'): load_seed_manifest(m,runner)
        runner.size=129
        with pytest.raises(ValueError,match='129'): make_manifest(runner,pool_id='short')
        runner.size=200
        cases=[]
        for role in ('selection','test'):
            bad=copy.deepcopy(m); bad[role]['a']=[]; cases.append(bad)
        for value in ([999], [True], [1,1]):
            bad=copy.deepcopy(m); bad['selection']['a']=value; cases.append(bad)
        bad=copy.deepcopy(m); bad['test']['a'][0]=bad['selection']['a'][0]; cases.append(bad)
        bad=copy.deepcopy(m); del bad['selection']['a']; cases.append(bad)
        bad=copy.deepcopy(m); del bad['test']; cases.append(bad)
        for bad in cases:
            with pytest.raises(ValueError): load_seed_manifest(bad,runner)


@pytest.mark.parametrize('multi',[False,True])
@pytest.mark.parametrize('tie_count',[0,5])
def test_zero_selection_fixed_test_and_pinning(tmp_path,monkeypatch,capsys,multi,tie_count):
    root,cfg=experiment(tmp_path)
    for p in (root/'checkpoints').iterdir(): p.unlink()
    for pct in (20,40,60,80,100):
        (root/'checkpoints'/f'epoch=0000-step={pct:08d}-milestone={pct}pct.pt').write_bytes(b'rollout fixture')
    pool=Pool(multi)
    manifest=make_manifest(pool,pool_id='synthetic-fixture',tie_break=tie_count)
    source=root/'seeds.json'; source.write_text(json.dumps(manifest))
    cfg.eval.seed_manifest=str(source)
    monkeypatch.setattr(selector,'build_eval_runner',lambda cfg:Pool(multi))
    def load(path,*a,**kw):
        return SimpleNamespace(_checkpoint_global_step=int(re.search(r'step=(\d+)',path.name)[1]))
    monkeypatch.setattr(selector,'load_ckpt_for_inference',load)
    original_record=None
    for train_seed in (42,43):
        cfg.training.seed=train_seed
        for model_name in ('dp','sat'):
            cfg.policy_name=model_name
            handoff=root/f'{model_name}-{train_seed}.json'
            selector.select_best_checkpoint(root,cfg,result_file=handoff)
            info,path=resolve_best_checkpoint(root,handoff)
            assert info['global_step']==100 and info['selection']['selection_all_zero'] is True
            summary=json.loads((root/info['selection_summary']).read_text())
            assert summary['status']=='success'
            assert all(len(c['episode_details'])==(25+tie_count)*(2 if multi else 1) for c in summary['all_results'])
            assert fixed_test_seeds(cfg,Pool(multi),info)==load_seed_manifest(manifest,pool)['roles']['test']
            if original_record is None: original_record=(info,path)
    source.unlink()  # Saved config now points at a nonexistent file.
    build=Mock(side_effect=lambda cfg:Pool(multi)); model_load=Mock(side_effect=load)
    monkeypatch.setattr(evaluator,'build_eval_runner',build)
    monkeypatch.setattr(evaluator,'load_ckpt_for_inference',model_load)
    result=evaluator.evaluate_checkpoint_robotwin(root,cfg,resolved_best=original_record, episodes=1,
                                                  result_save_dir=root/"final-eval")
    assert "manifest fixes 100 test seeds" in capsys.readouterr().out
    snapshot=OmegaConf.load(root/"final-eval/eval_config.yaml")
    assert snapshot.request.episodes==1 and snapshot.request.effective_episodes==100
    assert result[0]==0 and result[3]==100*(2 if multi else 1)
    assert build.call_count==model_load.call_count==1
    evaluator.evaluate_checkpoint_sweep(root,cfg,resolved_best=original_record,inference_steps_list=[2,4])
    assert build.call_count==model_load.call_count==2
    # Explicit null cannot unpin; reject before model loading, including Python API.
    with pytest.raises(ValueError,match='differs'):
        evaluator.evaluate_checkpoint_robotwin(root,cfg,resolved_best=original_record,
                                               dotlist_overrides=['eval.seed_manifest=null'])
    assert model_load.call_count==2
    changed=copy.deepcopy(manifest); changed['pool_id']='different'
    source.write_text(json.dumps(changed))
    assert fixed_test_seeds(cfg,Pool(multi),original_record[0])==load_seed_manifest(manifest,Pool(multi))['roles']['test']
    with pytest.raises(ValueError,match='differs'):
        evaluator._resolve_final_eval_request(cfg,root,'best',[f'eval.seed_manifest={source}'],resolved_best=original_record)
    # An unchanged explicit declaration is accepted and stays bound on repeated resolution.
    source.write_text(json.dumps(manifest))
    effective,*_=evaluator._resolve_final_eval_request(cfg,root,'best',[f'eval.seed_manifest={source}'],resolved_best=original_record)
    source.unlink()
    evaluator._resolve_final_eval_request(effective,root,'best',[],resolved_best=original_record)
    pool_changed=Pool(multi,size=201)
    monkeypatch.setattr(evaluator,'build_eval_runner',lambda cfg:pool_changed)
    with pytest.raises(ValueError,match='pool hash'):
        evaluator.evaluate_checkpoint_robotwin(root,cfg,resolved_best=original_record)
    assert model_load.call_count==2


@pytest.mark.parametrize('fault',['empty','missing','duplicate','wrong','no_success','no_steps','error','failed_tasks'])
def test_selection_episode_failures_preserve_pointer(tmp_path,monkeypatch,fault):
    root,cfg=experiment(tmp_path); runner=Pool()
    cfg.eval.seed_manifest=make_manifest(runner,pool_id='synthetic-fixture')
    monkeypatch.setattr(selector,'build_eval_runner',lambda cfg:runner)
    def result(cfg,r,mc,seeds,*a,**kw):
        details=[dict(seed=s,success=False,steps=None) for s in seeds]
        if fault=='empty': details=[]
        if fault=='missing': details.pop()
        if fault=='duplicate': details[-1]=details[0]
        if fault=='wrong': details[0]['seed']=999
        if fault=='no_success': del details[0]['success']
        if fault=='no_steps': del details[0]['steps']
        if fault=='error': details[0]['error']='technical failure'
        return dict(episode_details=details,failed_tasks=['a'] if fault=='failed_tasks' else [])
    monkeypatch.setattr(selector,'evaluate_checkpoint',result)
    pointer=root/'best_ckpt.json'; pointer.write_text('previous evidence')
    handoff=root/'handoff.json'
    with pytest.raises(RuntimeError): selector.select_best_checkpoint(root,cfg,result_file=handoff)
    assert pointer.read_text()=='previous evidence' and not handoff.exists()
    assert json.loads(next(root.rglob('best_ckpt_selection.json')).read_text())['status']=='failed'


def test_selector_requires_valid_protocol_before_model(tmp_path,monkeypatch):
    root,cfg=experiment(tmp_path)
    runner=Pool(); monkeypatch.setattr(selector,'build_eval_runner',lambda cfg:runner)
    load=Mock(); monkeypatch.setattr(selector,'load_ckpt_for_inference',load)
    for source in [None,str(tmp_path/'missing.json'),{}]:
        cfg.eval.seed_manifest=source
        with pytest.raises((ValueError,FileNotFoundError)): selector.select_best_checkpoint(root,cfg)
    cfg.eval.seed_manifest=make_manifest(runner,pool_id='synthetic-fixture')
    runner.size=201
    with pytest.raises(ValueError,match='pool hash'): selector.select_best_checkpoint(root,cfg)
    assert load.call_count==0
    with pytest.raises(ValueError,match='same manifest'):
        bind_seed_manifest(cfg,{'selection':{'seeds':[0]}})


@pytest.mark.parametrize('tie_count', [0, 5])
def test_selection_hard_cap_and_effective_counts(tmp_path, monkeypatch, tie_count):
    root, cfg = experiment(tmp_path)
    runner = Pool()
    cfg.eval.seed_manifest = make_manifest(runner, pool_id='synthetic-fixture', tie_break=tie_count)
    monkeypatch.setattr(selector, 'build_eval_runner', lambda cfg: runner)
    model_load = Mock()
    monkeypatch.setattr(selector, 'load_ckpt_for_inference', model_load)
    cap = 25 + tie_count
    with pytest.raises(ValueError, match='max_episodes'):
        selector.select_best_checkpoint(root, cfg, max_episodes=cap - 1)
    assert model_load.call_count == 0 and runner.calls == []
    # A unique winner does not use tie seeds, but they must still fit the cap.
    def rollout(cfg, r, mc, seeds, *args, **kwargs):
        return {'episode_details': [dict(seed=s, success=mc.global_step == 40, steps=3)
                                    for s in seeds]}
    monkeypatch.setattr(selector, 'evaluate_checkpoint', rollout)
    selector.select_best_checkpoint(root, cfg, initial_episodes=1, batch_size=0, max_episodes=cap)
    info, _ = resolve_best_checkpoint(root)
    assert not info['selection']['tie_break_used']
    snapshot = OmegaConf.load((root / info['selection_summary']).parent / 'eval_config.yaml')
    assert snapshot.request.initial_episodes == 1 and snapshot.request.batch_size == 0
    assert snapshot.request.max_episodes == cap
    assert dict(snapshot.request.effective_episode_counts) == {
        'selection': 25, 'tie_break': tie_count, 'test': 100}


@pytest.mark.parametrize('tie_success',[False,True])
def test_ranking_and_unused_tie_reservation(tmp_path,monkeypatch,tie_success):
    root,cfg=experiment(tmp_path); runner=Pool()
    manifest=make_manifest(runner,pool_id='synthetic-fixture')
    cfg.eval.seed_manifest=manifest
    monkeypatch.setattr(selector,'build_eval_runner',lambda cfg:runner)
    def rollout(cfg,r,mc,seeds,*a,**kw):
        tie=seeds==manifest['tie_break']['a']
        success=(tie and mc.global_step==20) if tie_success else mc.global_step==40
        return {'episode_details':[dict(seed=s,success=success,steps=3 if success else None) for s in seeds]}
    monkeypatch.setattr(selector,'evaluate_checkpoint',rollout)
    selector.select_best_checkpoint(root,cfg)
    info,_=resolve_best_checkpoint(root)
    assert info['global_step']==(20 if tie_success else 40)
    assert info['selection']['tie_break_used']==tie_success
    assert info['selection']['selection_all_zero'] is False
    assert not set(manifest['tie_break']['a']) & set(fixed_test_seeds(cfg,runner,info))


def test_manifest_cli_and_shell_handoff(tmp_path,monkeypatch):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from scripts.eval import make_seed_manifest as generator
    # Actual compose/CLI with an explicit lightweight pool; no environment runs.
    # Use the existing simulator package constant fixture only for provenance.
    import types
    monkeypatch.setitem(sys.modules,'dexmani_sim',types.SimpleNamespace(DATA_DIR=tmp_path,PACKAGE_DIR=tmp_path))
    monkeypatch.setattr(generator.hydra.utils,'instantiate',lambda cfg:Pool())
    output=tmp_path/'fixed.json'
    monkeypatch.setattr(sys,'argv',['make','--config-name','dp','--output',str(output),'task_name=a'])
    # Pool fixture supplies the leaf attributes inspected by the generator.
    monkeypatch.setattr(generator,'iter_leaf_env_runners',lambda runner:[SimpleNamespace(task_name='a',eval_seeds=None,get_seed_list=runner.get_seed_list)])
    generator.main()
    assert load_seed_manifest(output,Pool())['roles']['test']
    before=output.read_bytes()
    with pytest.raises(FileExistsError): generator.main()
    assert output.read_bytes()==before
    # Capture all shell argv in a temporary repository layout; never invoke stages.
    shell=tmp_path/'scripts/eval/eval_pipeline.sh'; shell.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/eval/eval_pipeline.sh',shell)
    exp=tmp_path/'experiments/dp/task/run'; exp.mkdir(parents=True); (exp/'config.yaml').write_text('{}')
    binary=tmp_path/'bin'; binary.mkdir()
    conda=binary/'conda'
    conda.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CAPTURE_ARGS"\n')
    conda.chmod(0o755)
    captured=tmp_path/'args.txt'
    subprocess.run(['bash',str(shell),'dp','task','run','--no-videos'],check=True,capture_output=True,
                   env=dict(os.environ,PATH=f'{binary}:'+os.environ['PATH'],SEED_MANIFEST=str(output),CAPTURE_ARGS=str(captured)))
    lines=captured.read_text().splitlines()
    assert len(lines)==3 and f'eval.seed_manifest={output}' in lines[0]
    assert all('eval.seed_manifest=' not in line for line in lines[1:])
    handoff=next(arg.split('=',1)[1] for arg in lines[0].split() if arg.startswith('--result-file='))
    assert all(f'--selection-record={handoff}' in line for line in lines[1:])
