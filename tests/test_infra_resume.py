import copy
import importlib
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import Dataset

from dexmani_policy.agents.core.base import BaseAgent
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.normalization import LinearNormalizer
from dexmani_policy.agents.loader import restore_policy_agent
from dexmani_policy.training.resume import (validate_resume_contract,validate_data_identity,
    build_train_loader,build_resume_contract,restore_training_state)
from dexmani_policy.training.build_utils import build_model_and_ema,capture_data_identity,validate_config
from dexmani_policy.training.trainer import Trainer,TrainLoopConfig
from dexmani_policy.training.workspace import TrainWorkspace,WandbConfig
from dexmani_policy.utils.random import get_rng_state,set_rng_state,set_seed
from dexmani_policy.smoke_test import load_config


class DummyEncoder(nn.Module):
    out_dim=4; embed_dim=4; num_obs_tokens=1; obs_token_dim=4; pc_pe_dim=3
    def forward(self,*args,**kwargs): return torch.zeros(1,4)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.tensor(.1))
    def forward(self,x,timestep,context): return x*self.weight


class TinyPolicy(BaseAgent):
    def __init__(self,clip_sample=True,horizon=2,n_obs_steps=1,n_action_steps=1,action_dim=1):
        super().__init__(DummyEncoder(),Diffusion(TinyBackbone(),clip_sample=clip_sample),horizon,n_obs_steps,n_action_steps,action_dim)
        self.action_key='action'; self.seen=[]
    def forward(self,batch,**kwargs):
        self.seen.extend(batch['obs']['joint_state'].flatten().tolist())
        x=batch['action']; loss=((self.action_decoder.model.weight*x+torch.randn_like(x)*.1)-1).square().mean()
        return loss,{'loss':loss.detach()}


class Data(Dataset):
    data_revision='fixture-v1'
    def __len__(self): return 12
    def __getitem__(self,i):
        return {'obs':{'joint_state':torch.tensor([[float(i)]])},'action':torch.full((2,1),float(i+1))}


class NullLogger:
    def __init__(self,*a,**kw): pass
    def log(self,*a,**kw): pass
    def log_config(self,*a,**kw): pass
    def close(self): pass


class ResumeInfraTests(unittest.TestCase):
    def test_scheduler_oracles(self):
        class Oracle(nn.Module):
            def forward(self,x,timestep,context):
                x0=torch.ones_like(x)*3
                x0[...,0]=-3
                alpha=decoder.noise_scheduler.alphas_cumprod[timestep]
                noise=(x-alpha.sqrt()*x0)/(1-alpha).sqrt()
                if prediction=='sample': return x0
                if prediction=='epsilon': return noise
                return alpha.sqrt()*noise-(1-alpha).sqrt()*x0
        for prediction in ('sample','epsilon','v_prediction'):
            for clip in (True,False):
                decoder=Diffusion(Oracle(),num_training_steps=10,num_inference_steps=5,prediction_type=prediction,clip_sample=clip)
                actual=decoder.predict_action(torch.zeros(1,1),torch.zeros(1,2,2))
                wanted=torch.tensor([[[-1.,1.],[-1.,1.]]])*(1 if clip else 3)
                torch.testing.assert_close(actual,wanted,rtol=1e-4,atol=1e-4)
        with self.assertRaises(TypeError): Diffusion(Oracle(),clip_sample='false')
        a=Diffusion(TinyBackbone()); b=Diffusion(TinyBackbone(),clip_sample=True)
        set_seed(3); x=a.predict_action(torch.zeros(1,1),torch.zeros(1,2,2))
        set_seed(3); y=b.predict_action(torch.zeros(1,1),torch.zeros(1,2,2))
        torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_all_constructors_and_config_matrix(self):
        targets=['dexmani_policy.agents.core.dp.DPObsEncoder','dexmani_policy.agents.core.dp3.DP3ObsEncoder',
                 'dexmani_policy.agents.core.dqrise.DP3ObsEncoder','dexmani_policy.agents.core.r3d.R3DObsEncoder',
                 'dexmani_policy.agents.core.multi_task.DPObsEncoder','dexmani_policy.agents.core.multi_task.CLIPTextEncoder']
        backbones=['dexmani_policy.agents.core.base.ConditionalUnet1D','dexmani_policy.agents.core.dqrise.ConditionalUnet1D',
                   'dexmani_policy.agents.core.r3d.OneWayTransformerBackbone','dexmani_policy.agents.core.multi_task.DiTDiffusion']
        # Import consumers before patching their providers: a cold import inside
        # patch would otherwise retain the provider's mock as its original value.
        originals = {}
        for target in targets + backbones:
            module_name, symbol = target.rsplit('.', 1)
            module = importlib.import_module(module_name)
            originals[target] = (module, symbol, getattr(module, symbol))
        with ExitStack() as stack:
            for target in targets: stack.enter_context(patch(target,side_effect=lambda *a,**k:DummyEncoder()))
            for target in backbones: stack.enter_context(patch(target,side_effect=lambda *a,**k:TinyBackbone()))
            for name in ('dp','dp3','dqrise','r3d','multitask_dit'):
                cfg=load_config(name)
                for clip in (False,True):
                    cfg.agent.clip_sample=clip
                    agent=hydra.utils.instantiate(cfg.agent)
                    self.assertEqual(agent.action_decoder.noise_scheduler.config.clip_sample,clip)
                    agent.set_normalization_spec({'action':'limits','joint_state':'gaussian'})
                    if clip:
                        with self.assertRaises(ValueError): agent.set_normalization_spec({'action':'gaussian'})
                    else: agent.set_normalization_spec({'action':'gaussian'})
                cfg.normalization.action='gaussian'; cfg.agent.clip_sample=True
                with self.assertRaisesRegex(ValueError,'Gaussian action'): validate_config(cfg)
                cfg.agent.clip_sample=False; validate_config(cfg)
            cfg=load_config('multitask_dit'); cfg.agent.action_decoder_type='rectified_flow'
            cfg.normalization.action='gaussian'; validate_config(cfg)
            agent=hydra.utils.instantiate(cfg.agent); agent.set_normalization_spec({'action':'gaussian'})
        for target, (module, symbol, original) in originals.items():
            self.assertIs(getattr(module, symbol), original, target)

    def test_constructor_patches_restore_after_cold_import(self):
        # A fresh interpreter prevents test collection/order from warming imports
        # and hiding pollution of module-level ``from ... import ...`` aliases.
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys
sys.path.insert(0, 'tests')
from test_infra_resume import ResumeInfraTests
assert 'dexmani_policy.agents.core.dqrise' not in sys.modules
assert 'dexmani_policy.agents.core.multi_task' not in sys.modules
ResumeInfraTests('test_all_constructors_and_config_matrix').test_all_constructors_and_config_matrix()
from dexmani_policy.agents.core import dp, dp3, dqrise, multi_task
from unittest.mock import Mock
assert dqrise.DP3ObsEncoder is dp3.DP3ObsEncoder
assert multi_task.DPObsEncoder is dp.DPObsEncoder
assert not isinstance(dqrise.DP3ObsEncoder, Mock)
assert not isinstance(multi_task.DPObsEncoder, Mock)
'''], cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_resume_facts_are_compared_without_conversion_or_mutation(self):
        from torch.utils.data import DataLoader
        cfg = load_config('dp3')
        with open_dict(cfg):
            cfg.data_identity = {'revision': 'fixture-v1'}
        current = build_resume_contract(cfg, TinyPolicy(), DataLoader(Data(), batch_size=2))
        saved = copy.deepcopy(current)
        validate_resume_contract(saved, current)
        self.assertEqual(saved, current)
        changed = copy.deepcopy(current)
        changed['loader']['batch_size'] += 1
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            validate_resume_contract(saved, changed)
        self.assertEqual(saved, current)
        for version in (None, 0, 2):
            unsupported = dict(saved, facts_format=version)
            if version is None:
                del unsupported['facts_format']
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'facts format'):
                validate_resume_contract(unsupported, current)
        with self.assertRaisesRegex(ValueError, 'facts format'):
            validate_resume_contract(saved, dict(current, facts_format=2))

    def test_rng_device_and_revision(self):
        state=get_rng_state('cpu')
        expected=(random.random(),np.random.rand(),torch.rand(3))
        set_rng_state(state,'cpu')
        self.assertEqual(random.random(),expected[0]); self.assertEqual(np.random.rand(),expected[1])
        torch.testing.assert_close(torch.rand(3),expected[2],rtol=0,atol=0)
        with patch('torch.cuda.get_rng_state',return_value=torch.ones(3,dtype=torch.uint8)) as capture:
            new=get_rng_state('cuda:1'); capture.assert_called_once_with(torch.device('cuda:1'))
        with patch('torch.cuda.set_rng_state') as setter:
            set_rng_state(new,'cuda:0'); self.assertEqual(setter.call_args.kwargs['device'],torch.device('cuda:0'))
            legacy=dict(state,torch_cuda=[torch.zeros(3,dtype=torch.uint8),torch.ones(3,dtype=torch.uint8)])
            setter.reset_mock()
            with self.assertRaisesRegex(ValueError,'per-rank tensor'): set_rng_state(legacy,'cuda:0')
            setter.assert_not_called()
        known={'revision':'v1'}; validate_data_identity(known,known)
        for current in ({'revision':'v2'},{'revision':None},None):
            with self.assertRaises(ValueError): validate_data_identity(known,current)
        with self.assertWarnsRegex(UserWarning,'身份未验证'): validate_data_identity(None,known)
        with self.assertRaises(ValueError): validate_data_identity({'tasks':{'a':known}},{'tasks':{'a':{'revision':'v2'}}})
        self.assertEqual(capture_data_identity(Data()),{'revision':'fixture-v1'})

    def test_continuous_vs_resume_and_shared_loader(self):
        cfg=OmegaConf.create({'agent':{'_target_':'test_infra_resume.TinyPolicy','clip_sample':False,
             'horizon':2,'n_obs_steps':1,'n_action_steps':1,'action_dim':1},
             'dataset':{'sensor_modalities':['joint_state']},'action_key':'action',
             'normalization':{'action':'limits','joint_state':'identity'},
             'data_identity':capture_data_identity(Data()),
             'dataloader':{'batch_size':2,'shuffle':True,'drop_last':False,'num_workers':0},
             'training':{'seed':123,'use_ema':True,'loop':{'total_train_steps':6,'gradient_accumulation_steps':2}},
             'optimizer':{},'ema':{'_target_':'dexmani_policy.training.ema_model.EMAModel'}})
        def build(path, checkpoint=None):
            norm=LinearNormalizer(); norm.fit_field('action',np.arange(12,dtype='float32').reshape(-1,1),mode='limits')
            model,ema,updater=build_model_and_ema(cfg,'cpu',norm,
                checkpoint=checkpoint)
            loader=build_train_loader(cfg,Data()); opt=torch.optim.Adam(model.parameters(),lr=.01)
            scheduler=torch.optim.lr_scheduler.StepLR(opt,1,.95)
            contract = build_resume_contract(cfg, model, loader)
            for fast in (False, True):
                historical = copy.deepcopy(cfg)
                historical.training.fast_grad_finite_check = fast
                self.assertEqual(contract, build_resume_contract(historical, model, loader))
            ws=TrainWorkspace(str(path),WandbConfig('test','test','test','test','allow','disabled'))
            ws.save_hydra_config(cfg)
            return Trainer('cpu',model,ema,updater,opt,scheduler,loader,ws,
                TrainLoopConfig(total_train_steps=6,gradient_accumulation_steps=2),contract)
        with tempfile.TemporaryDirectory() as tmp, patch('dexmani_policy.training.workspace.WandbLogger',NullLogger):
            root=Path(tmp)
            set_seed(123); full=build(root/'full'); full.train()
            set_seed(123); first=build(root/'first')
            original=first.apply_gradient_step
            def interrupt():
                original()
                if first.global_step==1: first._stop_requested=True
            first.apply_gradient_step=interrupt; first.train()
            source=(root/'first/checkpoints/latest.pt').resolve()
            old_bytes={str(p):p.read_bytes() for p in (root/'first').rglob('*') if p.is_file()}
            checkpoint = first.workspace.checkpoint_store.load(source)
            resumed=build(root/'resumed', checkpoint=checkpoint)
            with patch.object(resumed.workspace.checkpoint_store, 'load', side_effect=AssertionError('second read')):
                state=restore_training_state(checkpoint, resume_contract=resumed.resume_contract,
                    model=resumed.raw_model, ema_model=resumed.ema_model, ema_updater=resumed.ema_updater,
                    optimizer=resumed.optimizer, scheduler=resumed.scheduler, device='cpu')
            del checkpoint
            resumed.train(resume_state=state)
            self.assertEqual(first.raw_model.seen+resumed.raw_model.seen,full.raw_model.seen)
            for left,right in [(full.raw_model.state_dict(),resumed.raw_model.state_dict()),(full.ema_model.state_dict(),resumed.ema_model.state_dict())]:
                for k,v in left.items(): torch.testing.assert_close(v,right[k],rtol=0,atol=0)
            self.assertEqual(full.global_step,resumed.global_step); self.assertEqual(full.scheduler.state_dict(),resumed.scheduler.state_dict())
            self.assertEqual(full.ema_updater.optimization_step,resumed.ema_updater.optimization_step)
            for key,values in full.optimizer.state_dict()['state'].items():
                for name,value in values.items(): torch.testing.assert_close(value,resumed.optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
            self.assertEqual(old_bytes,{str(p):p.read_bytes() for p in (root/'first').rglob('*') if p.is_file()})
            for ema in (False,True):
                restored=restore_policy_agent(OmegaConf.to_container(cfg),source,use_ema=ema,device='cpu')
                self.assertFalse(restored.action_decoder.noise_scheduler.config.clip_sample)
                expected=first.ema_model if ema else first.raw_model
                for k,v in expected.state_dict().items(): torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
