"""Optional, tiny actual-CUDA checks. No data, weights, W&B, or simulator."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,TensorDataset

from dexmani_policy.training.checkpoint import CheckpointStore
from dexmani_policy.training.ema_model import EMAModel
from dexmani_policy.training.resume import restore_training_state
from dexmani_policy.training.trainer import Trainer,TrainLoopConfig
from dexmani_policy.training.workspace import TrainWorkspace,WandbConfig
from dexmani_policy.utils.random import get_rng_state,set_rng_state


class NullLogger:
    def __init__(self,*a,**kw): pass
    def log(self,*a,**kw): pass
    def close(self): pass


def ddp_worker(rank,root,phase,ids):
    root=Path(root); device=torch.device(f'cuda:{ids[rank]}')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',rank=rank,world_size=2,init_method=(root/f'init_{phase}').as_uri())
    try:
        torch.manual_seed(17)
        model=torch.nn.Linear(2,1).to(device); ema=copy.deepcopy(model); updater=EMAModel(ema)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
        contract={'batches_per_epoch':2,'world_size':2,'training':{'loop':{'gradient_accumulation_steps':1}}}
        wrapped=DistributedDataParallel(model,device_ids=[device.index])
        ws=None
        if rank==0:
            with patch('dexmani_policy.training.workspace.WandbLogger',NullLogger):
                ws=TrainWorkspace(str(root/phase),WandbConfig('test','test','test','test','allow','disabled'))
        t=Trainer(device,wrapped,ema,updater,optimizer,scheduler,
                  DataLoader(TensorDataset(torch.zeros(2,1))),ws,TrainLoopConfig(total_train_steps=2),
                  contract,distributed=True,is_main_process=rank==0)
        def update():
            x=torch.randn(3,2,device=device)
            loss=wrapped(x).square().mean(); loss.backward(); t.apply_gradient_step()
        if phase=='source':
            update(); t.global_step=1; t.next_micro_step=1
            t._save_checkpoint(1,'interrupt')
            update()
            torch.save({'model':model.state_dict(),'ema':ema.state_dict()},root/f'expected_{rank}.pt')
        else:
            store=CheckpointStore(root/'source/checkpoints')
            checkpoint=store.load(store.resolve_path('latest'))
            restore_training_state(checkpoint,resume_contract=contract,model=model,ema_model=ema,
                ema_updater=updater,optimizer=optimizer,scheduler=scheduler,device=device,rank=rank)
            update()
            expected=torch.load(root/f'expected_{rank}.pt',map_location=device,weights_only=True)
            for name,values in [('model',model.state_dict()),('ema',ema.state_dict())]:
                for key,tensor in values.items():
                    torch.testing.assert_close(tensor,expected[name][key],rtol=1e-6,atol=1e-7)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, 'NOT VERIFIED: requires two CUDA GPUs')
class ActualCudaTests(unittest.TestCase):
    def test_rng_remapping_and_nonzero_device(self):
        for source,target in ((0,1),(1,0),(1,1)):
            torch.cuda.manual_seed_all(31)
            state=get_rng_state(f'cuda:{source}')
            expected=torch.rand(20,device=f'cuda:{source}').cpu()
            set_rng_state(state,f'cuda:{target}')
            actual=torch.rand(20,device=f'cuda:{target}').cpu()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            legacy=dict(state,torch_cuda=torch.cuda.get_rng_state_all())
            expected=torch.rand(20,device=f'cuda:{source}').cpu()
            set_rng_state(legacy,f'cuda:{target}',source_config={'training':{'device':f'cuda:{source}'}})
            torch.testing.assert_close(torch.rand(20,device=f'cuda:{target}').cpu(),expected,rtol=0,atol=0)

    def test_short_ddp_resume_gpu_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker,args=(tmp,'source',[0,1]),nprocs=2,join=True)
            mp.spawn(ddp_worker,args=(tmp,'resumed',[1,0]),nprocs=2,join=True)


if __name__=='__main__': unittest.main()
