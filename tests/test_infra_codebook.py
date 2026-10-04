import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from dexmani_policy.agents.vq_hand.codebook_manager import CodebookManager
from scripts.training.train_vq_hand import evaluate_vq


def manager():
    m=CodebookManager(3,2,2)
    m.sorted_hand_poses=torch.arange(12,dtype=torch.float32).reshape(4,3)
    m.layer_weights=torch.tensor([1.,2.]); m.pca_permutation=torch.arange(4)
    m.set_hand_normalizer(np.ones(3),np.zeros(3))
    return m


class CodebookInfraTests(unittest.TestCase):
    def test_sample_mean_ranking(self):
        class ErrorModel(torch.nn.Module):
            def forward(self,batch):
                error=batch.square().mean()
                return error,error,None,error
        errors=torch.zeros(257,1); errors[-1]=10
        for size in (256,257):
            metric=evaluate_vq(ErrorModel(),DataLoader(TensorDataset(errors),batch_size=size),'cpu')['mse']
            self.assertAlmostEqual(metric,100/257,places=6)
            self.assertLess(metric,1.)

    def test_roundtrip_and_empty_ema(self):
        m=manager()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'m.npz'; m.save(path)
            n=CodebookManager(3,2,2); n.load(path)
            restored=CodebookManager(3); restored.load_state_dict(n.state_dict())
            for x in (-1.,0.,1.):
                q=torch.tensor([[x]])
                torch.testing.assert_close(m.continuous_index_to_hand_pose(q),restored.continuous_index_to_hand_pose(q))
        empty=CodebookManager(3); empty.load_state_dict(copy.deepcopy(empty.state_dict()))
        from dexmani_policy.training.ema_model import EMAModel
        ema=EMAModel(copy.deepcopy(m)); ema.step(m)

    def test_invalid_npz_atomic(self):
        m=manager()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'m.npz'; m.save(path)
            with np.load(path) as data: good={k:data[k].copy() for k in data.files}
            changes=[('sorted_hand_poses',np.full((4,3),np.nan)),('sorted_hand_poses',np.zeros((3,3))),
                ('layer_weights',np.array([np.inf,1.])),('pca_permutation',np.array([0,1,1,3])),
                ('pca_permutation',np.arange(4,dtype=float)),('hand_dim',np.array(3.2)),
                ('hand_min',np.array(70000.)),('hand_normalizer_scale',np.zeros(3)),
                ('hand_normalizer_scale',np.full(3,np.nan)),('hand_normalizer_offset',np.zeros(2)),
                ('metadata_json',np.array('[]')),('format_version',np.array(3.))]
            before=copy.deepcopy(m.state_dict()); metadata=copy.deepcopy(m.artifact_metadata)
            for key,value in changes:
                bad=dict(good); bad[key]=value; np.savez(path,**bad)
                with self.subTest(key=key), self.assertRaises((ValueError,RuntimeError)): m.load(path)
                for name,tensor in before.items(): torch.testing.assert_close(tensor,m.state_dict()[name],rtol=0,atol=0)
                self.assertEqual(m.artifact_metadata,metadata)

    def test_state_dict_preconversion_validation(self):
        m=manager()
        for key,value in [('pca_permutation',torch.arange(4).float()),
                          ('hand_normalizer_scale',torch.zeros(3)),
                          ('sorted_hand_poses',torch.zeros(3,3))]:
            state=copy.deepcopy(m.state_dict()); state[key]=value
            with self.assertRaises(RuntimeError): m.load_state_dict(state)


if __name__=='__main__': unittest.main()
