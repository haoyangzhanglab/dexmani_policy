import copy
import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from dexmani_policy.datasets.multi_task_dataset import MultiTaskDataset
from dexmani_policy.datasets.replay_buffer import ReplayBuffer
from dexmani_policy.training.run_identity import claim_run, resolve_resume_source
from dexmani_policy.training.resume import build_train_loader
from dexmani_policy.training.trainer import Trainer, TrainLoopConfig
from dexmani_policy.training.ema_model import EMAModel
from dexmani_policy.training.build_utils import build_normalizer


class TinyDataset(Dataset):
    def __init__(self, n=5): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return {'obs': {'index': i}, 'action': torch.tensor([float(i)])}


def claim_child(root, queue):
    try:
        claim_run(root)
        queue.put(True)
    except FileExistsError:
        queue.put(False)


def tiny_trainer():
    model = torch.nn.Linear(2, 1)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    return Trainer('cpu', model, ema, EMAModel(ema), optimizer,
                   torch.optim.lr_scheduler.StepLR(optimizer, 1, .9),
                   DataLoader(TinyDataset(), batch_size=1), None,
                   TrainLoopConfig(total_train_steps=3), {}, max_grad_norm=1.)


class TrainingInfraTests(unittest.TestCase):
    def test_spawn_epoch(self):
        for strategy in ('balanced', 'weighted', 'proportional'):
            ds = MultiTaskDataset([TinyDataset(3), TinyDataset(7)], ['a', 'b'],
                                  sampling_strategy=strategy,
                                  task_weights=[1., 3.] if strategy == 'weighted' else None)
            try:
                for workers in (0, 2):
                    options = dict(num_workers=workers, batch_size=None)
                    if workers: options.update(multiprocessing_context='spawn', persistent_workers=True)
                    loader = DataLoader(ds, **options)
                    try:
                        for epoch in (0, 1, 0):
                            ds.set_epoch(epoch)
                            expected = [(ds[i]['obs']['task_name'], ds[i]['obs']['index']) for i in range(len(ds))]
                            actual = [(x['obs']['task_name'], x['obs']['index']) for x in loader]
                            self.assertEqual(expected, actual)
                    finally:
                        if workers and loader._iterator: loader._iterator._shutdown_workers()
                self.assertIsNone(ds.__getstate__()['_manager'])
            finally:
                ds.close(); ds.close()

    def test_claim_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = mp.get_context('spawn'); q = ctx.Queue()
            ps = [ctx.Process(target=claim_child, args=(tmp,q)) for _ in range(2)]
            for p in ps: p.start()
            outcomes = [q.get(timeout=30) for p in ps]
            for p in ps: p.join(30); self.assertEqual(p.exitcode, 0)
            self.assertEqual(sorted(outcomes), [False, True]); q.close()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'config.yaml').write_bytes(b'old')
            with self.assertRaises(FileExistsError): claim_run(root)
            self.assertEqual((root/'config.yaml').read_bytes(), b'old')
            (root/'checkpoints').mkdir(); ckpt=root/'checkpoints/latest.pt'; ckpt.write_bytes(b'old')
            self.assertEqual(resolve_resume_source(root), str(ckpt.resolve()))
            self.assertEqual(resolve_resume_source(ckpt), str(ckpt.resolve()))
            with self.assertRaises(FileNotFoundError): resolve_resume_source(root/'missing')

    def test_accumulation(self):
        for n,b,a,drop,reject in [(5,2,3,False,True),(4,2,3,False,False),
                                  (5,2,2,False,False),(5,2,1,False,False),(5,2,3,True,False)]:
            cfg=OmegaConf.create({'dataloader':{'batch_size':b,'drop_last':drop,'num_workers':0},
                                  'training':{'seed':0,'loop':{'gradient_accumulation_steps':a}}})
            if reject:
                with self.assertRaisesRegex(ValueError,'Mixed micro-batch'): build_train_loader(cfg,TinyDataset(n))
            else: self.assertGreater(len(build_train_loader(cfg,TinyDataset(n))),0)

    def test_equal_accumulation_matches_large_batch(self):
        torch.manual_seed(1)
        full=tiny_trainer(); small=tiny_trainer()
        small.raw_model.load_state_dict(full.raw_model.state_dict())
        small.ema_model.load_state_dict(full.ema_model.state_dict())
        for trainer in (full,small):
            original=trainer.raw_model.forward
            def forward(batch, linear=original):
                loss=(linear(batch['action'])-1).square().mean()
                return loss, {'loss':loss.detach()}
            trainer.raw_model.forward=forward
            trainer.raw_model.get_training_loss_kwargs=lambda ema: {}
        x=torch.arange(8,dtype=torch.float32).reshape(4,2)/8
        full.train_one_step({'action':x})
        small.train_one_step({'action':x[:2]},loss_divisor=2,is_accumulation_boundary=False)
        small.train_one_step({'action':x[2:]},loss_divisor=2)
        for k,v in full.raw_model.state_dict().items():
            torch.testing.assert_close(v,small.raw_model.state_dict()[k],rtol=1e-6,atol=1e-7)

    def test_zarr_metadata_before_data_and_revision_snapshot(self):
        import zarr
        from dexmani_policy.training.build_utils import build_dataset_and_normalizer
        from dexmani_policy.training.resume import validate_data_identity
        with tempfile.TemporaryDirectory() as tmp:
            root=zarr.open(tmp,mode='w')
            root.create_dataset('data/action',data=np.arange(8,dtype='float32').reshape(8,1))
            root.create_dataset('data/joint_state',data=np.zeros((8,1),dtype='float32'))
            root.create_dataset('meta/episode_ends',data=np.array([4,2,8]))
            original=zarr.Array.__getitem__
            def read(array,key):
                if array.path.startswith('data/'):
                    raise AssertionError('data materialized before episode validation')
                return original(array,key)
            with patch.object(zarr.Array,'__getitem__',read):
                with self.assertRaises(ValueError): ReplayBuffer.copy_from_path(tmp)
            root['meta/episode_ends'][:]=[1,4,8]
            root.attrs['data_revision']='synthetic-v1'
            cfg=OmegaConf.create({'dataset':{'_target_':'dexmani_policy.datasets.base_dataset.BaseDataset',
                'zarr_path':tmp,'sensor_modalities':['joint_state'],'horizon':2,'pad_before':1,'pad_after':1,
                'val_ratio':.34},'action_key':'action','normalization':{'action':'auto','joint_state':'identity'}})
            ds,norm=build_dataset_and_normalizer(cfg)
            self.assertEqual(cfg.data_identity.revision,'synthetic-v1')
            self.assertEqual(ds.data_revision,'synthetic-v1')
            self.assertNotIn('data_identity',cfg.dataset)
            self.assertEqual(ds[0]['action'].shape,(2,1))
            self.assertIsNotNone(ds.get_validation_dataset())
            root.attrs['data_revision']='synthetic-v2'
            newer=ReplayBuffer.copy_from_path(tmp)
            with self.assertRaises(ValueError): validate_data_identity(dict(cfg.data_identity),{'revision':newer.data_revision})
            root.attrs['data_revision']=''
            with self.assertRaises(ValueError): ReplayBuffer.copy_from_path(tmp)

    def test_bad_episode_boundaries(self):
        for ends in ([10,5,20],[10,10,20],[],[0,20],[20.], [True], [[20]], [21]):
            with self.assertRaises(ValueError): ReplayBuffer({'data':{'x':np.zeros((20,2))},'meta':{'episode_ends':np.asarray(ends)}})
        ReplayBuffer({'data':{'x':np.zeros((20,2))},'meta':{'episode_ends':np.asarray([1,20],dtype=np.uint64)}})

    def test_gradient_failure_does_not_advance(self):
        for max_norm in (0., 1.):
            values = (float('nan'), float('inf')) + ((1e20,) if max_norm > 0 else ())
            for value in values:
                t=tiny_trainer(); t.max_grad_norm=max_norm
                before=copy.deepcopy(t.raw_model.state_dict()); epoch=t.scheduler.last_epoch
                for p in t.raw_model.parameters(): p.grad=torch.full_like(p,value)
                with self.assertRaises(RuntimeError): t.apply_gradient_step()
                self.assertEqual(t.global_step,0); self.assertEqual(t.scheduler.last_epoch,epoch)
                self.assertEqual(t.ema_updater.optimization_step,0); self.assertFalse(t.optimizer.state)
                for key,v in before.items(): torch.testing.assert_close(v,t.raw_model.state_dict()[key],rtol=0,atol=0)
        t=tiny_trainer()
        for p in t.raw_model.parameters(): p.grad=torch.ones_like(p)
        t.apply_gradient_step(); self.assertEqual(t.ema_updater.optimization_step,1)

    def test_chunked_joint_normalizer(self):
        values=np.random.default_rng(0).normal(size=(31,4)).astype('float32'); values[:,0]=2
        class Chunks:
            def __init__(self,chunks): self.chunks=chunks
            def iter_normalization_data(self,key): return iter(self.chunks)
        reference=build_normalizer(Chunks([values]),{'action':'auto'},'action')
        with patch('dexmani_policy.training.build_utils.np.concatenate',side_effect=AssertionError('full copy')):
            split=build_normalizer(Chunks([values[:13],values[13:]]),{'action':'auto'},'action')
        for k,v in reference.state_dict().items(): torch.testing.assert_close(v,split.state_dict()[k],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(reference['action'].normalize(values),split['action'].normalize(values),rtol=1e-5,atol=1e-6)


if __name__ == '__main__': unittest.main()
