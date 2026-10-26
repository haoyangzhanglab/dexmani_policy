from contextlib import closing
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from dexmani_policy.common.checkpoint import load_training_checkpoint, restore_policy_agent
from dexmani_policy.training.builder import build_training_components, build_trainer
from dexmani_policy.training.data import build_dataset_and_normalizer
from dexmani_policy.training.resume import restore_training_state
from tests.helpers import tiny_policy
from tests.test_data_config import zarr_fixture


class TrainingTests(unittest.TestCase):
    def test_cpu_training_resume_raw_ema_and_effective_updates(self):
        self.check_training("cpu")

    def check_training(self, device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, group, _ = zarr_fixture(root)
            group.create_dataset('data/point_cloud', data=np.random.default_rng(4).normal(
                size=(26, 8, 6)).astype(np.float32), chunks=(4, 8, 6))
            _, cfg, _, obs = tiny_policy()
            with open_dict(cfg):
                cfg.training.device = device
                cfg.zarr_path = str(path)
                cfg.dataset.zarr_path = str(path)
                cfg.dataset.augmentation_cfg = None
                cfg.dataloader.update(batch_size=4, num_workers=0, pin_memory=False,
                                      persistent_workers=False, prefetch_factor=None)
                cfg.training.update(use_compile=False, use_bfloat16=False, lr_warmup_steps=0)
                cfg.training.loop.update(total_train_steps=10, gradient_accumulation_steps=3)
                cfg.optimizer.obs_lr = 0
                cfg.workspace.output_dir = str(root / 'first')
                cfg.workspace.wandb_cfg = None
            original = OmegaConf.to_container(cfg, resolve=True)
            dataset, normalizer, facts = build_dataset_and_normalizer(cfg, verbose=False)
            components = build_training_components(cfg, torch.device(device), dataset, normalizer,
                                                   data_description=facts)
            self.assertEqual(original, OmegaConf.to_container(cfg, resolve=True))
            self.assertTrue(OmegaConf.is_readonly(components.effective_config))
            self.assertTrue(all(not p.requires_grad for p in components.model.obs_encoder.parameters()))
            # The seven batches include a final one-microbatch update.
            self.assertNotEqual(len(components.train_loader) % 3, 0)
            with closing(components):
                components.workspace.save_hydra_config(components.effective_config)
                self.assertTrue((root / 'first/split_manifest_0.json').is_file())
                trainer = build_trainer(components)
                trainer.train(max_updates=4)
                self.assertEqual(components.ema_updater.optimization_step, 4)
                self.assertEqual(components.scheduler.last_epoch, 4)
                saved = load_training_checkpoint(root / 'first/checkpoints/latest.pt')
                self.assertEqual(saved.global_step, 4)
                tensors = {key: torch.from_numpy(value).unsqueeze(0).to(device) for key, value in obs.items()}
                for use_ema, expected in ((False, components.model), (True, components.ema_model)):
                    restored = restore_policy_agent(saved.effective_config,
                        root / 'first/checkpoints/latest.pt', use_ema=use_ema, device=device)
                    for key, value in expected.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                    expected.eval()
                    with torch.no_grad():
                        a = expected.predict_action(tensors, generator=torch.Generator(device=device).manual_seed(7))
                        b = restored.predict_action(tensors, generator=torch.Generator(device=device).manual_seed(7))
                    torch.testing.assert_close(a['future_actions'], b['future_actions'], rtol=0, atol=0)
            resume_cfg = OmegaConf.create(saved.effective_config)
            resume_cfg.workspace.output_dir = str(root / 'second')
            resume_cfg.resume_from = str((root / 'first/checkpoints/latest.pt').resolve())
            ds, norm, facts = build_dataset_and_normalizer(resume_cfg, resume_checkpoint=saved, verbose=False)
            resumed = build_training_components(resume_cfg, torch.device(device), ds, norm,
                                                data_description=facts, checkpoint=saved)
            with closing(resumed):
                inherited = list((root / 'second/checkpoints').glob('*milestone*'))
                self.assertEqual(len(inherited), 2)
                incomplete_config = copy.deepcopy(saved.effective_config)
                del incomplete_config['data_recipe'][0]['source_evidence']
                incomplete = replace(saved, effective_config=incomplete_config)
                _, _, current_facts = build_dataset_and_normalizer(
                    resume_cfg, resume_checkpoint=incomplete, verbose=False)
                self.assertEqual(current_facts, facts)
                with self.assertRaisesRegex(ValueError, 'source_evidence'):
                    restore_training_state(incomplete, resumed, effective_config=resumed.effective_config)
                state = restore_training_state(saved, resumed, effective_config=resumed.effective_config)
                build_trainer(resumed).train(resume_state=state, max_updates=2)
                self.assertEqual(resumed.ema_updater.optimization_step, 6)
                self.assertEqual(resumed.scheduler.last_epoch, 6)
                self.assertEqual(load_training_checkpoint(root / 'second/checkpoints/latest.pt').global_step, 6)
                self.assertEqual(len(list((root / 'second/checkpoints').glob('*milestone*'))), 3)


if __name__ == '__main__':
    unittest.main()
