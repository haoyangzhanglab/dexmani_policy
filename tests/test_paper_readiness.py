"""Small real-mechanism checks for the paper-readiness changes (CPU/offline)."""
import copy
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_policy.training.ema_model import EMAModel
from dexmani_policy.training.resume import resolve_training_config, restore_training_state
from dexmani_policy.training.checkpoint import CheckpointStore, TrainCheckpoint
from dexmani_policy.utils.random import get_rng_state, set_rng_state


@pytest.fixture
def tiny_dino(tmp_path):
    from transformers import Dinov2Config, Dinov2Model
    cfg = Dinov2Config(image_size=28, patch_size=14, hidden_size=32,
                      num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)
    path = tmp_path / 'dino'
    Dinov2Model(cfg).save_pretrained(path)
    return str(path)


def adapter_params(model):
    return {n: p for n, p in model.named_parameters() if 'lora_' in n}


@pytest.mark.parametrize('precision', [None, 'backbone', 'float32'])
def test_lora_storage_training_resume(tiny_dino, tmp_path, precision):
    from dexmani_policy.agents.loader import restore_policy_agent
    torch.set_num_threads(1)
    rgb = dict(model_name=tiny_dino, tune_mode='lora', image_size=[28, 28])
    if precision is not None:
        rgb['lora_dtype'] = precision
    cfg = OmegaConf.create(dict(action_key='action', normalization={'action':'limits', 'joint_state':'identity'},
        agent=dict(_target_='dexmani_policy.agents.core.dp.DPAgent', horizon=4, n_obs_steps=1,
        n_action_steps=2, action_dim=2, state_dim=2, state_out_dim=8, rgb_backbone_name='dino',
        rgb_backbone_config=rgb, down_dims=[16,32], diffusion_step_embed_dim=16,
        num_training_steps=10, num_inference_steps=2, clip_sample=False)))
    cfg.training = {'use_ema':True, 'use_bfloat16':False, 'lr_scheduler':'cosine',
                    'lr_warmup_steps':0, 'loop':{'total_train_steps':10}}
    cfg.ema = {'_target_':'dexmani_policy.training.ema_model.EMAModel'}
    cfg.optimizer = {'lr':1e-3,'weight_decay':1e-6}
    def build(config=cfg, checkpoint=None):
        from dexmani_policy.training.build_utils import build_model_and_ema, build_optimizer_and_scheduler
        from dexmani_policy.agents.normalization import LinearNormalizer
        normalizer = LinearNormalizer()
        normalizer.fit_field('action', torch.tensor([[-1.,-1.],[1.,1.]]), mode='limits')
        model, ema, update = build_model_and_ema(config, 'cpu', normalizer,
            checkpoint=checkpoint)
        opt, sch = build_optimizer_and_scheduler(config, model, 4)
        return model, ema, update, opt, sch
    model, ema, update, opt, sch = build()
    expected = torch.float32 if precision == 'float32' else torch.bfloat16
    assert adapter_params(model) and {p.dtype for p in adapter_params(model).values()} == {expected}
    assert {p.dtype for p in adapter_params(ema).values()} == {expected}
    frozen = {n: p.detach().clone() for n,p in model.named_parameters() if 'backbone' in n and not p.requires_grad}
    assert frozen and {p.dtype for p in frozen.values()} == {torch.bfloat16}
    batch = {'obs':{'rgb':torch.rand(2,1,3,28,28), 'joint_state':torch.randn(2,1,2)},
             'action':torch.randn(2,4,2)}
    def step(parts):
        m,e,u,o,s = parts
        loss, _ = m.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter_params(m).values())
        o.step(); s.step(); o.zero_grad(set_to_none=True); u.step(m)
    parts = model,ema,update,opt,sch
    for _ in range(3): step(parts)
    for p in adapter_params(model).values():
        assert opt.state[p]['exp_avg'].dtype == expected
        assert opt.state[p]['exp_avg_sq'].dtype == expected
    for n,p in model.named_parameters():
        if n in frozen: assert torch.equal(p, frozen[n])
    contract = {'facts_format':1, 'data_identity':{'revision':'tiny'},
                'world_size':1, 'batches_per_epoch':4,
                'training':{'loop':{'gradient_accumulation_steps':1}}}
    checkpoint = TrainCheckpoint(0,3,3,model.state_dict(),ema.state_dict(),opt.state_dict(),
        sch.state_dict(),contract,update.optimization_step,update.decay,[get_rng_state('cpu')])
    store = CheckpointStore(tmp_path/'checkpoints')
    path = store.save('latest.pt',checkpoint)
    checkpoint = store.load(path)
    # This is a tiny local HF fixture, not a production cache. New snapshots own structure.
    import shutil
    shutil.rmtree(tiny_dino)
    for use_ema in (False,True):
        restored = restore_policy_agent(OmegaConf.to_container(cfg),path,use_ema=use_ema,device='cpu')
        wanted = ema if use_ema else model
        for n,p in restored.named_parameters():
            assert torch.equal(p,dict(wanted.named_parameters())[n])
        assert {p.dtype for p in adapter_params(restored).values()} == {expected}
    target_cfg = copy.deepcopy(cfg)
    if precision is None: target_cfg.agent.rgb_backbone_config.lora_dtype = 'backbone'
    resumed = build(target_cfg, checkpoint=checkpoint)
    target_contract = copy.deepcopy(contract)
    m,e,u,o,s = resumed
    restore_training_state(checkpoint,resume_contract=target_contract,model=m,ema_model=e,
        ema_updater=u,optimizer=o,scheduler=s,device='cpu')
    state = get_rng_state('cpu')
    step(parts); set_rng_state(state,'cpu'); step(resumed)
    for left,right in ((model,m),(ema,e)):
        for a,b in zip(left.parameters(),right.parameters()): assert torch.equal(a,b)
    bad = copy.deepcopy(target_contract)
    bad['world_size'] = 2
    before = copy.deepcopy((m.state_dict(),e.state_dict(),o.state_dict(),s.state_dict()))
    with pytest.raises(ValueError,match='world_size'):
        restore_training_state(checkpoint,resume_contract=bad,model=m,ema_model=e,
            ema_updater=u,optimizer=o,scheduler=s,device='cpu')
    assert_tree_equal(before,(m.state_dict(),e.state_dict(),o.state_dict(),s.state_dict()))


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys() == b.keys()
        for k in a: assert_tree_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else: assert a==b


def test_lora_variants_and_contract(tiny_dino,tmp_path):
    from transformers import CLIPVisionConfig, CLIPVisionModel, SiglipVisionConfig, SiglipVisionModel
    from dexmani_policy.agents.obs_encoder.rgb.dino import DINO
    from dexmani_policy.agents.obs_encoder.rgb.clip import CLIP
    from dexmani_policy.agents.obs_encoder.rgb.siglip import SigLIP
    with pytest.raises(ValueError,match='lora_dtype'):
        DINO(tiny_dino,tune_mode='lora',lora_dtype='float16')
    for config,model,encoder in [(CLIPVisionConfig,CLIPVisionModel,CLIP),(SiglipVisionConfig,SiglipVisionModel,SigLIP)]:
        path=tmp_path/encoder.__name__
        model(config(image_size=28,patch_size=14,hidden_size=32,intermediate_size=64,
                     num_hidden_layers=1,num_attention_heads=4)).save_pretrained(path)
        for dtype in ['backbone','float32']:
            enc=encoder(str(path),tune_mode='lora',lora_dtype=dtype)
            assert {p.dtype for p in adapter_params(enc).values()} == {torch.float32 if dtype=='float32' else torch.bfloat16}
    for name in ['dino', 'clip', 'siglip']:
        root = tmp_path / ('saved-' + name)
        (root / 'checkpoints').mkdir(parents=True)
        checkpoint = root / 'checkpoints/latest.pt'
        checkpoint.write_bytes(b'config-only fixture')
        saved = OmegaConf.create({'agent': {'rgb_backbone_name': name,
            'rgb_backbone_config': {'tune_mode': 'lora'}},
            'workspace': {'output_dir': str(root)}})
        OmegaConf.save(saved, root / 'config.yaml')
        incoming = copy.deepcopy(saved)
        incoming.resume_from = str(checkpoint)
        incoming.workspace.output_dir = str(tmp_path / 'new-run')
        restored = resolve_training_config(incoming, overrides=[])
        assert 'lora_dtype' not in restored.agent.rgb_backbone_config
        with pytest.raises(ValueError, match='lora_dtype'):
            resolve_training_config(incoming, overrides=['agent.rgb_backbone_config.lora_dtype=float32'])



@pytest.mark.parametrize('foreach',[False,True])
def test_fp32_small_updates_and_late_ema(foreach):
    for dtype in [torch.float32,torch.bfloat16]:
        model=torch.nn.Linear(1,1,bias=False,dtype=dtype)
        with torch.no_grad(): model.weight.fill_(1)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0)
        for _ in range(10):
            model.weight.grad=torch.ones_like(model.weight); opt.step()
        assert (model.weight.item()<1) == (dtype==torch.float32)
        shadow=copy.deepcopy(model)
        with torch.no_grad(): shadow.weight.fill_(1); model.weight.fill_(0.9)
        ema=EMAModel(shadow,foreach=foreach,min_value=.9999,max_value=.9999)
        ema.optimization_step=1000000
        for _ in range(10): ema.step(model)
        assert (shadow.weight.item()<1) == (dtype==torch.float32)


@pytest.mark.parametrize(('attention','prefix'), [(False,False),(True,True),(True,False)])
def test_sat_attention_branches_and_gradients(attention,prefix):
    from dexmani_policy.agents.obs_encoder.pointcloud.pointnext_tokenizer import PointNextPatchTokenizer
    from dexmani_policy.agents.obs_encoder.pointcloud.ops import farthest_point_sample
    from dexmani_policy.agents.core.sat import SATObsEncoder
    torch.set_num_threads(1); torch.manual_seed(12)
    config=dict(stem_channels=16,token_channels=16,num_patches=4,patch_radii=(2.,),
                patch_neighbors=(4,),fps_random_config={'use_random':False},
                use_patch_self_attn=attention,prepend_global_in_attn=prefix,
                patch_attn_layers=1,patch_attn_heads=4,patch_attn_dropout=0.)
    tokenizer=PointNextPatchTokenizer(input_channels=3,**config)
    pc=torch.randn(4,16,3)
    core=tokenizer.local_patch_tokenizer
    feature=tokenizer.geometry_stem(pc)
    core_out=core(pc,feature)
    assert len(core_out)==(3 if attention and prefix else 2)
    torch.testing.assert_close(core_out[1],farthest_point_sample(pc,4,use_random=False)[0],rtol=0,atol=0)
    # Capture real Transformer inputs/outputs; no operator is replaced.
    captured={}
    hooks=[core.token_projection.register_forward_hook(lambda m,a,o:captured.update(raw=o))]
    if attention:
        hooks.append(core.patch_transformer.register_forward_hook(lambda m,a,o:captured.update(attention=o)))
    patches,centers,global_token=tokenizer(pc,return_global_token=True)
    for hook in hooks: hook.remove()
    assert patches.shape==(4,4,16) and centers.shape==(4,4,3) and global_token.shape==(4,1,16)
    expected=captured['attention'][:,1:] if attention and prefix else captured.get('attention',captured['raw'])
    torch.testing.assert_close(patches,expected,rtol=0,atol=0)
    if attention and prefix:
        torch.testing.assert_close(global_token,captured['attention'][:,:1],rtol=0,atol=0)
    if attention:
        assert not torch.allclose(patches,captured['raw'])
        opt=torch.optim.AdamW(tokenizer.parameters(),lr=1e-3)
        names=['center_pos_embed.0.weight','patch_transformer.layers.0.self_attn.in_proj_weight']
        before={n:dict(core.named_parameters())[n].detach().clone() for n in names}
        target=torch.randn_like(patches)
        for _ in range(3):
            out=tokenizer(pc,return_global_token=True)[0]
            loss=(out-target).square().mean(); loss.backward()
            for n in names:
                grad=dict(core.named_parameters())[n].grad
                assert grad is not None and grad.abs().sum()>0
            opt.step(); opt.zero_grad(set_to_none=True)
        for n in names: assert not torch.equal(before[n],dict(core.named_parameters())[n])
        # Perturb each mechanism independently and observe the returned tokens.
        tokenizer.eval()
        for n in names:
            p=dict(core.named_parameters())[n]
            old=p.detach().clone(); original=tokenizer(pc)[0].detach()
            with torch.no_grad(): p.add_(torch.randn_like(p)*.1)
            assert not torch.allclose(original,tokenizer(pc)[0])
            with torch.no_grad(): p.copy_(old)
    encoder=SATObsEncoder('pointnext_tokenizer',3,2,16,2,state_out_dim=8,pc_encoder_config=config)
    out,_=encoder({'point_cloud':pc,'joint_state':torch.randn(4,2)})
    assert out.shape==(2,5,48)


def test_vq_policy_config_roots(tmp_path):
    from scripts.training.train_vq_hand import load_policy_config, _project_root
    from hydra.core.global_hydra import GlobalHydra
    root=_project_root/'dexmani_policy/configs'
    for name in ['dqrise','ddp/dqrise']:
        for absolute in [False,True]:
            for overrides in [[],['seed=43','dataloader.batch_size=7']]:
                path=root/f'{name}.yaml' if absolute else Path('dexmani_policy/configs')/f'{name}.yaml'
                actual=load_policy_config(path,overrides)
                with hydra.initialize_config_dir(config_dir=str(root),version_base=None):
                    expected=hydra.compose(config_name=name,overrides=overrides)
                for cfg in (actual,expected): cfg.workspace.output_dir='/tmp/paper-vq-config'
                for key in ['agent','dataset','action_key','normalization','horizon','n_obs_steps',
                            'n_action_steps','seed','dataloader','training','policy_name']:
                    a=OmegaConf.to_container(actual,resolve=True)[key]
                    b=OmegaConf.to_container(expected,resolve=True)[key]
                    assert a==b
                if name=='ddp/dqrise' and not overrides:
                    assert actual.policy_name=='ddp/dqrise'
                    assert actual.dataloader.batch_size==32 and actual.training.num_gpus==4
                assert not GlobalHydra.instance().is_initialized()
    external=tmp_path/'standalone.yaml'; external.write_text('value: 1\n')
    assert load_policy_config(external,['value=2']).value==2
    assert load_policy_config(external).value==1
    with pytest.raises(FileNotFoundError): load_policy_config(tmp_path/'missing.yaml')
    with hydra.initialize_config_dir(config_dir=str(root),version_base=None):
        with pytest.raises(ValueError,match='already initialized'): load_policy_config(external)
        assert GlobalHydra.instance().is_initialized()
