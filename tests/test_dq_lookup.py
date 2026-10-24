import argparse
import copy
import itertools
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook
from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.common.normalization import AffineNormalizer, FieldNormalizer
from dexmani_policy.training.pretrain.hand_prior import vqvae as training
from dexmani_policy.training.pretrain.hand_prior.export_codebook import (
    build_hand_prototypes,
    build_runtime_codebook,
    deployment_diagnostics,
    export_codebook,
)


class DQLookupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = dict(
            hand_dim=12, latent_dim=3, hidden_dim=8, num_groups=2,
            codebook_size=3, num_layers=1, act_scale=1.7, loss_weight=[1.0] * 12,
        )
        self.model = HandVQVAE(**self.config)
        with torch.no_grad():
            self.model.quantizer.layer_weights.copy_(torch.tensor([-0.7, 1.2]))
            self.model.decoder.head.bias[0] = 2
        self.rows = torch.randn(11, 12) * 0.4
        self.normalizer = AffineNormalizer()
        self.normalizer["hand"] = FieldNormalizer.from_parameters(
            torch.linspace(0.5, 2, 12), torch.linspace(-0.2, 0.3, 12)
        )

    def loader(self, rows=None, batch_size=4):
        return DataLoader(
            TensorDataset(self.rows if rows is None else rows), batch_size=batch_size,
            generator=torch.Generator().manual_seed(7),
        )

    def checkpoint(self):
        return dict(
            epoch=3, model_config=self.config, model_state_dict=self.model.state_dict(),
            normalizer_state_dict=self.normalizer.state_dict(),
            split_metadata={
                "normalization_spec": {"action": "limits"},
                "hand_data": {"val_episodes": [1]},
                "selection_metric": "val_lookup_mse", "selection_split": "validation",
            },
            metrics={"val_lookup_mse": 0.125},
        )

    def test_prototypes_and_export_roundtrip(self):
        poses, decoded = build_hand_prototypes(self.model.eval())
        weights = self.model.quantizer.layer_weights.softmax(0)
        latents = torch.stack([
            sum(weights[g] * self.model.codebooks[g, index] for g, index in enumerate(pair))
            for pair in itertools.product(range(3), repeat=2)
        ])
        torch.testing.assert_close(decoded, self.model.decoder(latents) * 1.7)
        torch.testing.assert_close(poses, decoded.clamp(-1, 1))
        self.assertTrue((decoded.abs() > 1).any())
        score = training.evaluate_vq(self.model, self.loader(), "cpu")["lookup_mse"]
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            checkpoint = Path(directory) / "vq.pt"
            artifact = Path(directory) / "codebook.npz"
            payload = self.checkpoint()
            payload["metrics"]["val_lookup_mse"] = score
            torch.save(payload, checkpoint)
            export_codebook(checkpoint, artifact)
            codebook = HandStateCodebook(12, 2, 3)
            codebook.load(artifact)
            self.assertEqual(codebook.artifact_metadata["selection"]["value"], score)
            continuous, _ = codebook.encode(self.rows)
            restored, _ = codebook.decode(continuous.squeeze(-1))
            expected = (restored - self.rows).double().square().mean().item()
            self.assertAlmostEqual(score, expected, places=7)
            report = deployment_diagnostics(codebook, {"held_out": self.rows.numpy()})
            self.assertAlmostEqual(report["splits"]["held_out"]["lookup_mse"], score, places=7)
            payload = torch.load(checkpoint, weights_only=False)
            reloaded = HandVQVAE.from_checkpoint(payload)
            self.assertEqual(
                score, training.evaluate_vq(reloaded, self.loader(), "cpu")["lookup_mse"]
            )

    def test_partition_permutation_and_model_state(self):
        state = copy.deepcopy(self.model.state_dict())
        poses, decoded = build_hand_prototypes(self.model)
        rng = torch.get_rng_state().clone()
        score = training.evaluate_vq(self.model, self.loader(), "cpu")["lookup_mse"]
        self.assertTrue(self.model.training)
        torch.testing.assert_close(self.model.state_dict(), state, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with patch.object(training, "build_hand_prototypes", return_value=(poses.flip(0), decoded)):
            for size in (1, 3, 20):
                result = training.evaluate_vq(
                    self.model, self.loader(batch_size=size), "cpu", chunk_size=2,
                    reconstruction=False,
                )
                self.assertEqual(result, {"lookup_mse": score})
        self.model.eval()
        with self.assertRaises(ValueError):
            training.evaluate_vq(self.model, self.loader(self.rows[:0]), "cpu")
        self.assertFalse(self.model.training)
        with self.assertRaises(FloatingPointError):
            training.evaluate_vq(
                self.model, self.loader(torch.full_like(self.rows, float("nan"))),
                "cpu", reconstruction=False,
            )
        torch.testing.assert_close(self.model.state_dict(), state, rtol=0, atol=0)

    def test_selection_metadata(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for split, metric in (
                ("validation", "val_lookup_mse"), ("train_no_holdout", "train_lookup_mse")
            ):
                checkpoint = self.checkpoint()
                checkpoint["split_metadata"].update(selection_split=split, selection_metric=metric)
                checkpoint["metrics"] = {metric: 0.125}
                result = build_runtime_codebook(checkpoint, "vq.pt", device="cpu")
                selection = result.artifact_metadata["selection"]
                self.assertEqual(selection["metric"], metric)
                self.assertEqual(selection["value"], 0.125)
                self.assertEqual(selection["split"], split)
            del checkpoint["split_metadata"]["selection_split"]
            with self.assertRaisesRegex(KeyError, "selection_split"):
                build_runtime_codebook(checkpoint, "vq.pt", device="cpu")

    def test_training_selection_and_no_holdout_tail(self):
        for holdout in (False, True):
            with self.subTest(holdout=holdout), tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(
                    **self.config, output_dir=directory, zarr_path="synthetic.zarr",
                    device="cpu", seed=19, batch_size=4, num_workers=0, num_epochs=2,
                    save_epochs=2, codebook_report_epochs=100, vq_decay=0.8,
                    threshold_ema_dead_code=0, kmeans_init=False, kmeans_iters=2,
                    lr=1e-3, betas=(0.9, 0.999), weight_decay=0, warmup_steps=0,
                    enc_loss_weight=1.0, vq_loss_weight=1.0, max_grad_norm=1.0,
                )
                val = self.rows[:5].numpy() if holdout else np.empty((0, 12), np.float32)
                metadata = self.checkpoint()["split_metadata"]
                metadata["hand_data"]["val_episodes"] = [1] if holdout else []
                data = (self.rows.numpy(), val, self.normalizer, metadata)
                evaluate = training.evaluate_vq
                scores = []

                def record_evaluation(model, loader, device, **kwargs):
                    self.assertFalse(loader.drop_last)
                    self.assertEqual(len(loader.dataset), 5 if holdout else 11)
                    self.assertEqual(kwargs["reconstruction"], holdout)
                    result = evaluate(model, loader, device, **kwargs)
                    scores.append(result["lookup_mse"])
                    if holdout:
                        # Force opposing rankings to test selection independently of optimization.
                        result.update(lookup_mse=float(len(scores)), mse=3.0 - len(scores))
                    return result

                with (
                    patch.object(training, "prepare_policy_data", return_value=data),
                    patch.object(training, "evaluate_vq", side_effect=record_evaluation),
                    warnings.catch_warnings(),
                ):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    training.train(args, policy_cfg=None)
                best = torch.load(Path(directory) / "vqvae_hand_best.pt", weights_only=False)
                metric = "val_lookup_mse" if holdout else "train_lookup_mse"
                self.assertEqual(best["split_metadata"]["selection_metric"], metric)
                self.assertEqual(
                    best["split_metadata"]["selection_split"],
                    "validation" if holdout else "train_no_holdout",
                )
                self.assertEqual(best["epoch"], 1 if holdout else int(np.argmin(scores)) + 1)
                if not holdout:
                    model = HandVQVAE.from_checkpoint(best)
                    actual = evaluate(model, self.loader(), "cpu", reconstruction=False)
                    self.assertEqual(best["metrics"][metric], actual["lookup_mse"])


if __name__ == "__main__":
    unittest.main()
