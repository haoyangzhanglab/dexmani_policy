import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from dexmani_policy.common.config import PROJECT_ROOT, load_policy_config


class MethodIdentityTests(unittest.TestCase):
    def test_ddp_inherits_method_and_preserves_recipe(self):
        directory = PROJECT_ROOT / "dexmani_policy/configs"
        batches = {"dp": 48, "dqrise": 32, "maniflow": 32,
                   "multitask_dit": 16, "r3d": 32, "sat": 32}
        overlays = sorted((directory / "ddp").glob("*.yaml"))
        self.assertEqual({path.stem for path in overlays}, set(batches))
        for path in overlays:
            with self.subTest(config=path.stem):
                self.assertNotIn("policy_name", OmegaConf.load(path))
                base = load_policy_config(path.stem)
                actual = load_policy_config(f"ddp/{path.stem}")
                expected = OmegaConf.merge(OmegaConf.to_container(base), {
                    "training": {"num_gpus": 4, "gpu_ids": None},
                    "dataloader": {"batch_size": batches[path.stem], "num_workers": 4},
                })
                # Hydra owns this value only during a launch.
                actual.workspace.output_dir = expected.workspace.output_dir = "/tmp/output"
                self.assertEqual(
                    OmegaConf.to_container(actual, resolve=True),
                    OmegaConf.to_container(expected, resolve=True),
                )
                self.assertEqual(actual.policy_name, base.policy_name)
                self.assertEqual(actual.workspace.wandb_cfg.group, base.policy_name)
                with hydra.initialize_config_dir(config_dir=str(directory), version_base=None):
                    cfg = hydra.compose(config_name=f"ddp/{path.stem}", return_hydra_config=True)
                    for output in (cfg.hydra.run.dir, cfg.hydra.sweep.dir):
                        self.assertEqual(Path(output).parts[:3],
                                         ("experiments", base.policy_name, cfg.task_name))

    def test_training_scripts_pass_config_selector_to_hydra(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conda = root / "conda"
            conda.write_text(f"#!{sys.executable}\n" + '''import os, sys
assert sys.argv[1:6] == ["run", "--no-capture-output", "-n", "policy", "python"]
os.execv(sys.executable, [sys.executable, *sys.argv[6:]])
''')
            conda.chmod(0o755)
            configs = [("train.sh", "dp")]
            configs.extend(("train_ddp.sh", f"ddp/{path.stem}") for path in sorted(
                (PROJECT_ROOT / "dexmani_policy/configs/ddp").glob("*.yaml")
            ))
            for script, selector in configs:
                with self.subTest(script=script, selector=selector):
                    result = subprocess.run(
                        ["bash", str(PROJECT_ROOT / "scripts/training" / script), selector,
                         "task_name=offline_task", "training.seed=7",
                         f"workspace.output_dir={root / 'output'}", "--cfg", "job", "--resolve"],
                        cwd=root, text=True, capture_output=True, timeout=30,
                        env={**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                             "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"},
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    cfg = OmegaConf.create(result.stdout)
                    expected = load_policy_config(selector)
                    self.assertEqual(cfg.policy_name, expected.policy_name)
                    self.assertEqual(cfg.task_name, "offline_task")
                    self.assertEqual(cfg.training.seed, 7)
                    self.assertEqual(cfg.training.get("num_gpus", 1),
                                     expected.training.get("num_gpus", 1))
                    self.assertEqual(cfg.dataloader.batch_size, expected.dataloader.batch_size)
                    self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
