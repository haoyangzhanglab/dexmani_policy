"""Targeted regressions for the incremental review fixes; all fixtures are local."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from test_policy_vq_alignment import policy_config
from scripts.training import train_vq_hand as vq


@pytest.fixture(autouse=True)
def bounded_threads():
    torch.set_num_threads(1)


def tiny_vq_args(tmp_path):
    defaults = vq._load_yaml_config('dexmani_policy/configs/dqrise.yaml')
    defaults.update(output_dir=str(tmp_path / 'run'), zarr_path='unused', action_key='action', tcp_dim=7,
                    num_epochs=2, batch_size=8, num_workers=0, latent_dim=4,
                    hidden_dim=8, num_layers=1, num_groups=1, codebook_size=2,
                    kmeans_init=True, kmeans_iters=2, threshold_ema_dead_code=0,
                    warmup_steps=0, save_epochs=1, codebook_report_epochs=99, device='cpu')
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize('validation', [True, False])
def test_vq_run_and_fixed_selection(policy_config, tmp_path, validation):
    args = tiny_vq_args(tmp_path)
    if not validation:
        policy_config.dataset.val_ratio = 0
    vq.train(args, policy_cfg=policy_config)
    best = Path(args.output_dir) / 'vqvae_hand_best.pt'
    before = best.read_bytes()
    ckpt = torch.load(best, weights_only=False)
    key = 'val_mse' if validation else 'train_mse'
    assert ckpt['split_metadata']['selection_metric'] == key
    assert np.isfinite(ckpt['metrics'][key])
    with pytest.raises(FileExistsError):
        vq.train(args, policy_cfg=policy_config)
    assert best.read_bytes() == before
    from scripts.training.extract_vq_codebook import extract_codebook
    output = Path(args.output_dir) / 'codebook.npz'
    manager = extract_codebook(best, output, device='cpu')
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        extract_codebook(best, output, device='cpu')
    assert output.read_bytes() == before
    from scripts.training.measure_vq_usage import measure
    results = [measure(str(best), policy_config.dataset.zarr_path,
                       codebook_path=str(output), chunk_size=k) for k in (1, 7, 4096)]
    for result in results[1:]:
        assert result == results[0]
    # Match the pre-change full formulas, including raw-space runtime labels.
    data, _, normalizer, _ = vq.prepare_policy_data(policy_config)
    hand = torch.from_numpy(data)
    continuous = manager.hand_pose_to_continuous_index(hand)
    ids = torch.floor(((continuous.flatten()+1)*.5*(manager.num_codes-1)).clamp(0, manager.num_codes-1)+.5).long()
    assert torch.bincount(ids, minlength=manager.num_codes).tolist() == results[0]['nn_counts']
    distances = (hand[:, None] - manager._from_raw(manager.sorted_hand_poses)[None]).square().sum(-1).min(-1).values.sqrt()
    assert results[0]['nn_l2_p99'] == pytest.approx(float(torch.quantile(distances, .99)))
    for gap in (0., .01):
        center=manager._to_raw(hand[len(hand)//2])
        manager.sorted_hand_poses[0]=center-gap
        manager.sorted_hand_poses[1]=center+gap
        manager.save(output)  # controlled fixture: duplicate or near-tied prototypes
        runtime=manager.hand_pose_to_continuous_index(hand)
        expected_ids=torch.floor(((runtime.flatten()+1)*.5*(manager.num_codes-1)).clamp(0,manager.num_codes-1)+.5).long()
        chunked=[measure(str(best),policy_config.dataset.zarr_path,
                         codebook_path=str(output),chunk_size=k) for k in (1,7,4096)]
        assert chunked[0] == chunked[1] == chunked[2]
        assert chunked[0]['nn_counts'] == torch.bincount(expected_ids,minlength=manager.num_codes).tolist()


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_validation_preserves_best(policy_config, tmp_path, monkeypatch, bad):
    args = tiny_vq_args(tmp_path)
    calls = []
    first_bytes = []
    def evaluate(*a):
        calls.append(1)
        if len(calls) == 2:
            first_bytes.append((Path(args.output_dir)/'vqvae_hand_best.pt').read_bytes())
        return dict(enc=1., vq=1., mse=1. if len(calls) == 1 else bad)
    monkeypatch.setattr(vq, 'evaluate_vq', evaluate)
    with pytest.raises(FloatingPointError, match='val_mse'):
        vq.train(args, policy_cfg=policy_config)
    assert (Path(args.output_dir)/'vqvae_hand_best.pt').read_bytes() == first_bytes[0]


def test_old_vq_directory_is_not_claimable(tmp_path):
    from dexmani_policy.training.run_identity import claim_run
    old = tmp_path/'vqvae_hand_best.pt'; old.write_bytes(b'old')
    with pytest.raises(FileExistsError): claim_run(tmp_path)
    assert old.read_bytes() == b'old'
    assert not (tmp_path/'.training_run.json').exists()


@pytest.mark.parametrize('dims,groups', [([12,24,48],4), ([16,32],8)])
@pytest.mark.parametrize('horizon', [15,16,18,20])
def test_unet_shapes(dims, groups, horizon):
    from dexmani_policy.agents.action_decoders.backbone.unet1d import ConditionalUnet1D
    model = ConditionalUnet1D(3, 8, diffusion_step_embed_dim=16, down_dims=dims, n_groups=groups)
    x = torch.randn(2, horizon, 3)
    if horizon % (2**(len(dims)-1)):
        with pytest.raises(ValueError, match='horizon'): model(x, 1, torch.randn(2,8))
    else:
        out = model(x, 1, torch.randn(2,8))
        assert out.shape == x.shape
        out.square().mean().backward()


def test_config_consumption_and_child_contract(policy_config):
    from dexmani_policy.agents.obs_encoder.pointcloud.registry import build_pc_global_encoder, build_pc_patch_tokenizer
    from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DPointcloudEncoder
    from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
    model = build_pc_global_encoder('idp3', 6, {'hidden_channels':12, 'num_layers':2})
    assert model.input_projection.out_channels == 12 and len(model.point_layers) == 2
    tokenizer = build_pc_patch_tokenizer('pointnext_tokenizer', 6, {'include_global_token':False})
    assert not tokenizer.include_global_token
    assert not hasattr(tokenizer, 'global_token_projection')
    from dexmani_policy.agents.core.sat import SATObsEncoder
    with pytest.raises(ValueError, match='global token'):
        SATObsEncoder('pointnext_tokenizer', 6, 19, 16, 2,
                      pc_encoder_config={'include_global_token':False})
    for key in ('typo', 'fps_random_config'):
        with pytest.raises(ValueError, match=key): build_pc_global_encoder('idp3', 6, {key: {}})
    with pytest.raises(ValueError, match='typo'): build_pc_patch_tokenizer('pointnet_dense', 6, {'typo':1})
    with pytest.raises(ValueError, match='norm'): Uni3DPointcloudEncoder(norm='bn')
    ds = vq.build_policy_dataset(policy_config)
    with pytest.raises(ValueError, match='action_key'):
        MultiTaskDataset([ds], ['one'], deterministic=True, action_key='action_ee')


def test_discrete_pow_subbatch_and_old_rng():
    from dexmani_policy.agents.action_decoders.time_sampler import sample_discrete_pow
    from dexmani_policy.agents.action_decoders.consistency_flow import ConsistencyFlowMatch
    for b in (1,2,3):
        with pytest.raises(ValueError, match='sub-batch'): sample_discrete_pow(b,10,'cpu')
    for b in (4,7,10):
        torch.manual_seed(1)
        actual = sample_discrete_pow(b,10,'cpu')
        torch.manual_seed(1)
        exponent = torch.repeat_interleave(torch.arange(3,-1,-1), b//4)
        exponent = torch.cat([exponent, torch.zeros(b-len(exponent), dtype=torch.long)])
        sections = 2**exponent
        expected = torch.floor(torch.rand(b)*sections.float()).long().float()/sections.float()
        torch.testing.assert_close(actual, expected, rtol=0,atol=0)
    decoder = ConsistencyFlowMatch(nn.Identity(), t_sample_mode_for_consistency='discrete_pow')
    with pytest.raises(ValueError, match='sub-batch'):
        decoder.compute_loss(torch.zeros(8,2), torch.zeros(8,4,2), ema_decoder=decoder)


@pytest.mark.parametrize('fused', [True,False])
def test_mask_after_bias_and_gradients(fused):
    from dexmani_policy.agents.action_decoders.backbone.attention import CrossAttention
    model = CrossAttention(8,2); model.fused_attn=fused
    model.proj.bias.data.fill_(3)
    x=torch.randn(2,3,8,requires_grad=True); c=torch.randn(2,4,8,requires_grad=True)
    out=model(x,c,torch.tensor([[0,0,0,0],[1,0,1,0]]))
    assert torch.equal(out[0],torch.zeros_like(out[0])) and torch.isfinite(out).all()
    out.sum().backward()
    assert c.grad[0].count_nonzero() == 0 and x.grad[0].count_nonzero() == 0


def test_geometry_invalid_depth_and_empty_patch():
    from dexmani_policy.agents.obs_encoder.rgb.geometry_processor import GeometryProcessor
    g=GeometryProcessor()
    depth=torch.tensor([[[[float('nan'), float('inf')],[0.,1000.]]]])
    result=g.backproject_depth(depth, torch.eye(3), torch.eye(4))
    assert torch.isfinite(result['coords']).all()
    assert result['valid_mask'].sum()==1
    pooled=g.pool_patch_coordinates(result['coords'], result['valid_mask'], patch_size=1, min_valid_ratio=0)
    assert pooled['patch_valid_mask'].sum()==1
    k=torch.eye(3); k[0,0]=float('nan')
    with pytest.raises(ValueError, match='intrinsics'): g.backproject_depth(depth,k)


@pytest.mark.parametrize('name,modes', [('dino',['avg','cls','pooler']),('clip',['avg','cls','pooler']),('siglip',['avg','pooler'])])
def test_vit_projection_all_branches_without_backbone_download(name,modes):
    import importlib
    cls=getattr(importlib.import_module('dexmani_policy.agents.obs_encoder.rgb.'+name), {'dino':'DINO','clip':'CLIP','siglip':'SigLIP'}[name])
    from dexmani_policy.agents.obs_encoder.rgb.base import ViTEncoder
    model=cls.__new__(cls); ViTEncoder.__init__(model)
    model.proj=nn.Linear(4,3); model.model_name='fixture'
    outputs=SimpleNamespace(last_hidden_state=torch.randn(2,3,4,dtype=torch.bfloat16),pooler_output=torch.randn(2,4,dtype=torch.bfloat16))
    patches=model.project_features(outputs.last_hidden_state)
    for mode in modes:
        model.global_token_type=mode
        value=model.get_global_token(outputs,patches)
        assert value.dtype == torch.float32
        value.sum().backward()
    model.proj=nn.Identity()
    assert model.project_features(outputs.last_hidden_state) is outputs.last_hidden_state


def test_patch_dropout_keeps_centers_and_tokens_together():
    from dexmani_policy.agents.obs_encoder.pointcloud.uni3d import Uni3DPointcloudEncoder
    model=Uni3DPointcloudEncoder(embed_dim=8,num_group=8,group_size=2,patch_dropout=.5).train()
    centers=torch.linspace(-.9,.9,24).reshape(1,8,3)
    class Patches(nn.Module):
        def forward(self,*a): return {'embeddings':torch.zeros(1,8,512),'centers':centers}
    model.patch_embed=Patches()
    seen={}
    def keep_hook(module,args,output): seen['keep']=output[1]
    model.patch_dropout.register_forward_hook(keep_hook)
    tokens, pe=model(torch.rand(1,16,6))
    assert tokens.shape[1] == 4
    torch.testing.assert_close(pe, model.pe_layer(centers[:,seen['keep'][0]]))
    model.eval()
    tokens, pe=model(torch.rand(1,16,6),inference_mode=True)
    assert tokens.shape[1] == 8


def test_fixed_colors_and_integer_controls():
    from dexmani_policy.datasets.augmentation import PointColorJitter
    from dexmani_policy.training.trainer import TrainLoopConfig
    x=np.array([[[0.,0.,0.,.2,.4,.6]]],dtype='f4'); before=x.copy()
    PointColorJitter(brightness=[.1,.1],contrast=[1,1],saturation=[1,1],hue=[0,0])._augment(x)
    np.testing.assert_allclose(x[...,3:], before[...,3:]+.1)
    PointColorJitter(brightness=[0,0],contrast=[1,1],saturation=[1,1],hue=[0,0])._augment(x)
    np.testing.assert_allclose(x[...,3:], before[...,3:]+.1)
    for key in ('total_train_steps','log_interval_steps','gradient_accumulation_steps'):
        for bad in (True,1.5,float('nan'),float('inf'),0):
            with pytest.raises(ValueError): TrainLoopConfig(**{key:bad})


def test_colliding_milestones_and_resume():
    from dexmani_policy.training.trainer import Trainer
    for total in (1,2,3,4,5,103):
        t=Trainer.__new__(Trainer); t.total_train_steps=total; t.global_step=0
        t._passed_milestones=t._init_milestone_state(); saved=[]
        t._save_milestone_checkpoint=lambda step,ratio: saved.append((step,ratio))
        for step in range(1,total+1): t._check_milestone(step)
        assert saved[-1] == (total,1.) and len({step for step,_ in saved})==len(saved)
        t.global_step=total; t._passed_milestones=t._init_milestone_state()
        t._check_milestone(total)
        assert saved.count((total,1.)) == 1


def test_source_snapshot_content_identity(tmp_path):
    from dexmani_policy.training.source_snapshot import save_source_snapshot
    root=tmp_path/'source'; (root/'dexmani_policy').mkdir(parents=True)
    (root/'dexmani_policy/untracked.py').write_text('x=1')
    (root/'dexmani_policy/secret.pt').write_bytes(b'weights')
    (root/'.env').write_text('secret')
    a=tmp_path/'a'; b=tmp_path/'b'; a.mkdir(); b.mkdir()
    first=save_source_snapshot(a,root)
    assert set(first['files'])=={'dexmani_policy/untracked.py'}
    (root/'dexmani_policy/untracked.py').write_text('x=2')
    second=save_source_snapshot(b,root)
    assert first['source_sha256'] != second['source_sha256']
    assert first['commit']=='unknown'


def test_explicit_protocol_and_immutable_handoff(tmp_path, monkeypatch):
    from test_infra_evaluation import Runner, experiment
    from dexmani_policy import select_best_ckpt as selector
    from dexmani_policy.evaluation.protocol import load_seed_manifest, resolve_test_protocol
    from dexmani_policy.agents.loader import resolve_best_checkpoint
    root,cfg=experiment(tmp_path)
    runner=Runner()
    manifest={'pool_id':'fixture','selection':{'a':[0,1]},'tie_break':{'a':[2]},'test':{'a':[3,4,5]}}
    source=tmp_path/'seeds.json'; source.write_text(json.dumps(manifest))
    cfg.eval.seed_manifest=str(source)
    monkeypatch.setattr(selector,'build_eval_runner',lambda cfg:runner)
    monkeypatch.setattr(selector,'load_ckpt_for_inference',lambda path,*a,**kw:
        SimpleNamespace(_checkpoint_global_step=int(path.name.split('step=')[1].split('-')[0])))
    # The fake runner still passes through the public episode-result boundary.
    result=tmp_path/'handoff.json'
    selector.select_best_checkpoint(root,cfg,result_file=result)
    info,path=resolve_best_checkpoint(root,result)
    before=result.read_bytes()
    cfg.training.seed=999
    selector.select_best_checkpoint(root,cfg)
    assert resolve_best_checkpoint(root,result)[0]['selection_id']==info['selection_id']
    assert result.read_bytes()==before
    assert resolve_test_protocol(cfg,runner,info)['roles']['test']=={'a':[3,4,5]}
    bad=copy.deepcopy(manifest); bad['test']['a']=[2]
    with pytest.raises(ValueError,match='overlap'): load_seed_manifest(bad,runner)
    bad=copy.deepcopy(manifest); bad['test']['a']=[999]
    with pytest.raises(ValueError,match='unavailable'): load_seed_manifest(bad,runner)


@pytest.mark.parametrize('option', ['--all','--list'])
@pytest.mark.parametrize('mode,expected', [('empty',0),('okay',0),('tmux_error',2),('missing',127),('ssh_error',255)])
def test_remote_list_preserves_exit_status_local_only(tmp_path, option, mode, expected):
    import os
    import subprocess
    ssh=tmp_path/'ssh'
    ssh.write_text('#!/bin/bash\nif [[ "$FIXTURE_MODE" == ssh_error ]]; then exit 255; fi\ncommand="${@: -1}"\nexec bash -c "$command"\n')
    tmux=tmp_path/'tmux'
    tmux.write_text('''#!/bin/bash
case "$FIXTURE_MODE" in
empty) echo 'no server running on /tmp/fixture' >&2; exit 1;;
okay) echo 'unrelated: 1 windows'; exit 0;;
tmux_error) echo 'server failed' >&2; exit 2;;
missing) echo 'tmux: command not found' >&2; exit 127;;
esac
exit 9
''')
    ssh.chmod(0o755); tmux.chmod(0o755)
    result=subprocess.run(['bash','scripts/remote/stop_remote.sh',option],
                          env=dict(os.environ,PATH=str(tmp_path)+':'+os.environ['PATH'],FIXTURE_MODE=mode),
                          capture_output=True,text=True,timeout=5)
    assert result.returncode==expected, result.stderr


def test_resume_normalizer_uses_supplied_payload(policy_config, tmp_path, monkeypatch):
    import zarr
    from dexmani_policy.training.build_utils import build_dataset_and_normalizer
    from dexmani_policy.training.checkpoint import CheckpointStore
    # This fixture exercises simulation dataset metadata, not real robot deployment.
    root=zarr.open_group(policy_config.dataset.zarr_path,mode='a'); root.attrs['domain']='sim'
    _, normalizer=build_dataset_and_normalizer(policy_config)
    payload=SimpleNamespace(model_state={'normalizer.'+k:v for k,v in normalizer.state_dict().items()})
    policy_config.resume_from=str(tmp_path/'checkpoint.pt')
    monkeypatch.setattr(CheckpointStore,'load',lambda *a:pytest.fail('second full checkpoint read'))
    _, restored=build_dataset_and_normalizer(policy_config,resume_checkpoint=payload)
    for key,value in normalizer.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value)


def test_manifest_task_identity_and_pool_change():
    from dexmani_policy.evaluation.protocol import load_seed_manifest, resolve_test_protocol
    from test_infra_evaluation import Runner
    a, b = Runner(), Runner()
    a.get_seed_list = lambda: [0, 1, 2, 3]
    b.task_name = 'b'; b.get_seed_list = lambda: [1, 2, 3, 4]
    runner = SimpleNamespace(runners={'a': a, 'b': b})
    manifest={'pool_id':'fixture','selection':{'a':[0],'b':[1]},
              'tie_break':{'a':[1],'b':[2]},'test':{'a':[2,3],'b':[3,4]}}
    protocol=load_seed_manifest(manifest,runner)
    assert protocol['roles']['test']==manifest['test']
    cfg=OmegaConf.create({'eval':{}})
    best={'selection':{'seed_manifest':protocol}}
    assert resolve_test_protocol(cfg,runner,best)['roles']['test']==manifest['test']
    a.get_seed_list=lambda:[4,3,2,1,0]
    assert resolve_test_protocol(cfg,runner,best)['roles']['test']==manifest['test']
    a.get_seed_list=lambda:[0,1,2]
    with pytest.raises(ValueError,match='unavailable'): resolve_test_protocol(cfg,runner,best)


def test_vq_validation_uses_sample_count_for_tail():
    from torch.utils.data import DataLoader,TensorDataset
    class Metrics(nn.Module):
        def forward(self,x):
            value=x.mean(); return value,value,None,value
    x=torch.zeros(257,1); x[-1]=257
    result=vq.evaluate_vq(Metrics(),DataLoader(TensorDataset(x),batch_size=256),'cpu')
    assert result['mse']==pytest.approx(1.)


@pytest.mark.parametrize(('entry', 'with_manifest'), [
    ('eval', False), ('demo', False), ('demo', True),
])
@pytest.mark.parametrize('source_change', ['missing', 'modified'])
def test_selection_record_cli_survives_new_best(tmp_path, monkeypatch, entry, with_manifest, source_change):
    import sys
    from test_infra_evaluation import Runner, experiment, publish_best
    from dexmani_policy import eval_best_ckpt as evaluator, record_demo as demo
    from dexmani_policy.evaluation.protocol import load_seed_manifest

    class SeedOverrideRunner(Runner):
        # Match SimRunner: recording overrides replace the visible seed pool.
        def get_seed_list(self):
            seeds = getattr(self, 'eval_seeds', None)
            return seeds if seeds is not None else super().get_seed_list()

    runner = SeedOverrideRunner()
    root=tmp_path/'experiments/dp/task/run'; root.mkdir(parents=True)
    root,cfg=experiment(root)
    first=publish_best(root,20)
    if with_manifest:
        manifest = {'pool_id': 'demo-fixture', 'selection': {'a': [0, 1]},
                    'tie_break': {'a': [2]}, 'test': {'a': [3, 4, 5]}}
        source = root/'seeds.json'
        source.write_text(json.dumps(manifest))
        cfg.eval.seed_manifest = str(source)
        OmegaConf.save(cfg, root/'config.yaml')
        protocol = load_seed_manifest(source, runner)
        first['selection']['seed_manifest'] = protocol
    summary=root/first['selection_summary']
    payload=json.loads(summary.read_text()); payload['inference']=first['inference']
    payload['selection'] = first['selection']
    summary.write_text(json.dumps(payload))
    handoff=root/'handoff.json'; handoff.write_text(json.dumps(first))
    publish_best(root,40)  # A different successful selector publishes between stages.
    if with_manifest:
        if source_change == 'missing':
            source.unlink()
        else:
            source.write_text('{}')  # Demo uses its own seeds; old protocol remains provenance.
    module=demo if entry=='demo' else evaluator
    def load(path,use_ema,**kw):
        assert path==root/first['ckpt_relpath'] and use_ema is True
        return SimpleNamespace(_checkpoint_global_step=20)
    monkeypatch.setattr(module,'ROOT_DIR',tmp_path)
    monkeypatch.setattr(module,'build_eval_runner',lambda *a:runner)
    monkeypatch.setattr(module,'load_ckpt_for_inference',load)
    args=['test','--policy-name','dp','--task-name','task','--exp-name','run',
          '--selection-record',str(handoff),'--episodes','2']
    if entry=='eval': args+=['--no-videos']
    if with_manifest:
        selected_seeds = [4, 1]  # Demo may include selection seeds; preserve order.
        args += ['--seeds', *map(str, selected_seeds)]
    monkeypatch.setattr(sys,'argv',args)
    module.main()
    assert runner.calls[0][1]['inference_steps']==10
    snapshot=OmegaConf.load(next(root.rglob('eval_config.yaml')))
    assert snapshot.request.selection_id==first['selection_id']
    assert snapshot.request.global_step == 20
    assert snapshot.request.use_ema is True
    assert snapshot.request.checkpoint == first['ckpt_relpath']
    if with_manifest:
        assert snapshot.seed_manifest is None
        assert OmegaConf.to_container(snapshot.request.selection.seed_manifest) == protocol
        assert snapshot.request.task_seeds.a == selected_seeds
        assert snapshot.request.heldout_from_selection is False
        assert snapshot.request.inference_steps_list == [10]
        assert runner.get_seed_list() == selected_seeds
        assert runner.calls == [(selected_seeds, {
            'inference_steps': 10, 'eval_episodes': 2,
            'video_save_dir': next(root.rglob('eval_config.yaml')).parent,
        })]
        result = json.loads(next(root.rglob('result_details.json')).read_text())
        assert [d['seed'] for d in result['episode_details']] == selected_seeds
        assert result['heldout_from_selection'] is False
        assert (result['checkpoint'], result['global_step'], result['use_ema'],
                result['inference_steps']) == (first['ckpt_relpath'], 20, True, 10)


def test_learned_weight_export_reconstruction_and_rounding(tmp_path):
    import itertools
    from dexmani_policy.agents.vq_hand import VQVAEHand,CodebookManager
    model=VQVAEHand(3,[1.]*3,latent_dim=4,hidden_dim=8,num_groups=2,
                    codebook_size=2,num_layers=1,kmeans_init=True,kmeans_iters=2)
    model(torch.randn(8,3).clamp(-1,1))
    with torch.no_grad(): model.vq_layer.layer_weights.copy_(torch.tensor([-1.,2.]))
    model.eval()
    manager=CodebookManager.extract_from_vqvae(model)
    manager.set_hand_normalizer(torch.ones(3),torch.zeros(3))
    manager.reindex_by_pca(model)
    expected=[]
    for combination in itertools.product(range(2),repeat=2):
        latent=sum(manager.layer_weights[g]*model.codebooks[g,i] for g,i in enumerate(combination))
        expected.append(model.decode_from_latent(latent[None]).clamp(-1,1).squeeze(0))
    expected=torch.stack(expected)[manager.pca_permutation]
    torch.testing.assert_close(manager._from_raw(manager.sorted_hand_poses),expected,rtol=1e-5,atol=1e-6)
    path=tmp_path/'weighted.npz'; manager.save(path)
    restored=CodebookManager(3,2,2); restored.load(path)
    indices=torch.linspace(-1,1,4)
    poses,ids=restored.continuous_index_to_hand_pose(indices)
    assert ids.tolist()==[0,1,2,3]
    torch.testing.assert_close(poses,expected,rtol=1e-5,atol=1e-6)
    _,ids=restored.continuous_index_to_hand_pose(torch.tensor([-2.,0.,2.]))
    assert ids.tolist()==[0,2,3]  # midpoint half-up and endpoint clamps


def test_rgb_processor_transport_range_and_cache(monkeypatch):
    import pickle
    from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
    from dexmani_policy.agents.obs_encoder.rgb.utils import to_rgb_tensor
    processor = ImageProcessor(image_size=None)
    rgb = torch.arange(2*3*5*7, dtype=torch.uint8).reshape(2, 3, 5, 7)
    unit = rgb.float()/255
    torch.testing.assert_close(processor.process_images(rgb)['image'],
                               processor.process_images(unit)['image'], rtol=0, atol=0)
    for value in (-.01, 1.01):
        with pytest.raises(ValueError, match='expected to be in'):
            processor.process_images(torch.full((1, 3, 5, 7), value))
    def forbidden(*args, **kwargs):
        raise AssertionError('trusted RGB must not reduce or synchronize')
    with monkeypatch.context() as m:
        for method in ('amin', 'amax', 'item'):
            m.setattr(torch.Tensor, method, forbidden)
        processor.process_images(unit, validate_float_range=False)
        processor.process_images(rgb)  # integer bypasses float checks even by default
        unchecked = to_rgb_tensor(unit + 2, validate_float_range=False)
    torch.testing.assert_close(unchecked, unit+2)
    assert unchecked.dtype == torch.float32 and unchecked.is_contiguous()
    for dtype in (torch.float32, torch.float64):
        x = unit.to(dtype)
        expected = (x - processor.image_mean.to(dtype).view(1, 3, 1, 1)) / processor.image_std.to(dtype).view(1, 3, 1, 1)
        torch.testing.assert_close(processor.normalize(x), expected, rtol=0, atol=0)
        cached = processor._normalization_cache[('cpu', None, dtype)]
        processor.normalize(x.clone())
        assert processor._normalization_cache[('cpu', None, dtype)] is cached
        assert all(t.numel() == 3 and not t.requires_grad for t in cached)
    processor.normalize(torch.empty(2, 3, 5, 7, device='meta'))
    assert processor._normalization_cache[('meta', None, torch.float32)][0].device.type == 'meta'
    restored = pickle.loads(pickle.dumps(processor))
    assert restored._normalization_cache == {}
    torch.testing.assert_close(restored.process_images(rgb)['image'], processor.process_images(rgb)['image'])


def test_dp_encoder_uses_trusted_float_boundary(monkeypatch):
    from dexmani_policy.agents.core import dp
    from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
    class Backbone(nn.Module):
        out_dim = 3
        def forward(self, x):
            assert x.dtype == torch.float32
            return {'global_token': x.mean((-2, -1))}
    processor = ImageProcessor(image_size=None)
    monkeypatch.setattr(dp, 'build_backbone', lambda *a, **kw: (Backbone(), processor))
    encoder = dp.DPObsEncoder('fixture', state_dim=2, n_obs_steps=2)
    rgb = torch.randint(0, 256, (4, 3, 5, 7), dtype=torch.uint8)
    state = torch.randn(4, 2)
    expected = encoder({'rgb': rgb, 'joint_state': state})[0]
    def forbidden(*a, **kw): raise AssertionError('unexpected range reduction')
    monkeypatch.setattr(torch.Tensor, 'amin', forbidden)
    monkeypatch.setattr(torch.Tensor, 'amax', forbidden)
    actual = encoder({'rgb': rgb.float()/255, 'joint_state': state})[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rgb_transport_config_and_resume_contract(tmp_path):
    from dexmani_policy.smoke_test import load_config
    from dexmani_policy.training.build_utils import validate_config
    from dexmani_policy.training.resume import build_resume_contract, validate_resume_contract
    from test_infra_resume import TinyPolicy, Data
    from torch.utils.data import DataLoader
    cfg = load_config('dp')
    assert cfg.dataset.rgb_keep_uint8 is True
    validate_config(cfg)
    for mode in ('limits', 'gaussian'):
        cfg.normalization.rgb = mode
        with pytest.raises(ValueError, match='rgb_keep_uint8 requires normalization.rgb=identity'):
            validate_config(cfg)
    cfg.normalization.rgb = 'identity'
    model, loader = TinyPolicy(), DataLoader(Data(), batch_size=2)
    current = build_resume_contract(cfg, model, loader)
    for old_recipe in (False, None):
        old_cfg = copy.deepcopy(cfg)
        if old_recipe is None:
            del old_cfg.dataset.rgb_keep_uint8
        else:
            old_cfg.dataset.rgb_keep_uint8 = old_recipe
        saved = build_resume_contract(old_cfg, model, loader)
        validate_resume_contract(saved, saved)
        # Recipe identity is now owned by the saved config, before construction.
        from dexmani_policy.training.resume import resolve_training_config
        run = tmp_path / str(old_recipe)
        (run / 'checkpoints').mkdir(parents=True)
        checkpoint = run / 'checkpoints/latest.pt'
        checkpoint.write_bytes(b'config-only fixture')
        OmegaConf.save(old_cfg, run / 'config.yaml')
        incoming = copy.deepcopy(cfg)
        from omegaconf import open_dict
        with open_dict(incoming):
            incoming.resume_from = str(checkpoint)
        restored = resolve_training_config(incoming, overrides=[])
        assert restored.dataset.get('rgb_keep_uint8') == old_recipe
        with pytest.raises(ValueError, match='dataset.rgb_keep_uint8'):
            resolve_training_config(incoming, overrides=['dataset.rgb_keep_uint8=true'])
    cfg = OmegaConf.create(OmegaConf.to_container(load_config('multitask_dit'), resolve=True))
    cfg.dataset.datasets[0].rgb_keep_uint8 = True
    with pytest.raises(ValueError, match='consistent rgb_keep_uint8'):
        validate_config(cfg)
    for child in cfg.dataset.datasets:
        child.rgb_keep_uint8 = True
    validate_config(cfg)
    cfg.normalization.rgb = 'limits'
    with pytest.raises(ValueError, match='rgb_keep_uint8 requires normalization.rgb=identity'):
        validate_config(cfg)
