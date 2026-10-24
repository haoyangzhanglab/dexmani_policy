import copy
import inspect
import unittest

import torch
from torch import nn

from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
from dexmani_policy.agents.policies.dqrise import DQRISEPolicy
from dexmani_policy.agents.policies.sat import SATPolicy
from dexmani_policy.common.config import load_policy_config
from dexmani_policy.common.normalization import AffineNormalizer, FieldNormalizer
from dexmani_policy.env_runner.base_runner import BaseRunner
from dexmani_policy.training.builder import build_optimizer
from dexmani_policy.training.pretrain.hand_prior.export_codebook import build_hand_prototypes


class DQNetworkTests(unittest.TestCase):
    def test_default_vq_depth_math_and_ema(self):
        cfg = load_policy_config("dqrise")
        kwargs = {k: cfg.vq_vae[k] for k in inspect.signature(HandVQVAE).parameters if k in cfg.vq_vae}
        model = HandVQVAE(hand_dim=12, **kwargs)
        for mlp in (model.encoder, model.decoder):
            self.assertEqual(sum(isinstance(m, nn.Linear) for m in mlp.trunk), 6)
            self.assertEqual(sum(isinstance(m, nn.ReLU) for m in mlp.trunk), 6)
            for m in mlp.trunk:
                if isinstance(m, nn.Linear):
                    self.assertEqual(m.out_features, 512)
                    torch.testing.assert_close(m.bias, torch.zeros_like(m.bias))
        self.assertEqual(model.encoder.head.out_features, 256)
        hand = torch.randn(4, 12)
        state = copy.deepcopy(model.state_dict())
        encoded = model.encoder(hand)
        residual, weighted, commits = encoded, 0, []
        with torch.no_grad():
            model.quantizer.layer_weights.copy_(torch.tensor([-1., 2.]))
        weights = model.quantizer.layer_weights.softmax(0)
        for weight, layer in zip(weights, model.quantizer.layers):
            self.assertTrue(layer.is_initialized)
            self.assertEqual(layer.decay, 0.8)
            self.assertEqual(layer.threshold_ema_dead_code, 0)
            distances = (residual[:, None] - layer.embed).square().sum(-1)
            q = layer.embed[distances.argmin(-1)].clone()
            commits.append((residual - q.detach()).square().mean())
            weighted = weighted + weight * (residual + (q - residual).detach())
            residual = residual - q.detach()
        expected_l1 = ((model.decoder(weighted) - hand).abs() * model.loss_weight).mean()
        expected_commit = sum(commits)
        expected_loss = 3 * expected_l1 + 5 * expected_commit
        expected_grad = torch.autograd.grad(expected_loss, model.encoder.head.weight, retain_graph=True)[0]
        l1, commitment, indices, _ = model(hand)
        torch.testing.assert_close(l1, expected_l1)
        torch.testing.assert_close(commitment, expected_commit)
        (3 * l1 + 5 * commitment).backward()
        torch.testing.assert_close(model.encoder.head.weight.grad, expected_grad)
        self.assertEqual(indices.shape, (4, 2))
        self.assertFalse(torch.equal(model.quantizer.layers[0].embed, state["quantizer.layers.0.embed"]))
        model.eval()
        with torch.no_grad():
            latent, codes, _ = model.quantizer(model.encoder(hand))
            table, decoded = build_hand_prototypes(model)
            row = codes[:, 0] * 4 + codes[:, 1]
            torch.testing.assert_close(model.decode(latent), decoded[row])
            torch.testing.assert_close(model.decode(latent).clamp(-1, 1), table[row])
        fixed = copy.deepcopy(model.state_dict())
        model(hand)
        torch.testing.assert_close(model.state_dict(), fixed, rtol=0, atol=0)

    def test_default_diffusion_observation_loss_sampling_and_runner(self):
        torch.manual_seed(19)
        model = DQRISEPolicy(
            horizon=20, n_obs_steps=1, n_action_steps=16, action_dim=19,
            num_points=8, state_out_dim=8,
            pc_encoder_config=dict(channels=24, num_patches=2, group_size=4, depth=1, num_heads=4),
            fps_random_config={"use_random": False},
        )
        normalizer = AffineNormalizer()
        scale, offset = torch.linspace(0.5, 2, 19), torch.linspace(-1, 1, 19)
        normalizer["action"] = FieldNormalizer.from_parameters(scale, offset)
        normalizer["joint_state"] = FieldNormalizer.from_parameters(torch.ones(19), torch.zeros(19))
        model.set_normalization_spec({"action": "auto", "joint_state": "limits", "point_cloud": "identity"},
                                     {"action": 19, "joint_state": 19})
        model.load_normalizer_from_dataset(normalizer)
        model.hand_codebook.set_buffers(dict(
            sorted_hand_poses=torch.linspace(-1, 1, 16)[:, None].expand(16, 12).clone(),
            pca_permutation=torch.arange(16), layer_weights=torch.tensor([0.2, 0.8]),
            hand_normalizer_scale=scale[7:], hand_normalizer_offset=offset[7:],
        ))
        backbone = model.action_decoder.model
        self.assertEqual(backbone.diffusion_step_encoder.net[1].in_features, 256)
        self.assertEqual(backbone.down_modules[0][0].blocks[0].block[0].out_channels, 256)
        self.assertEqual(backbone.down_modules[1][0].blocks[0].block[0].out_channels, 512)
        self.assertEqual(backbone.down_modules[0][0].blocks[0].block[0].kernel_size, (5,))
        self.assertEqual(backbone.down_modules[0][0].blocks[0].block[1].num_groups, 8)
        noise = model.action_decoder.noise_scheduler
        self.assertEqual(noise.config.num_train_timesteps, 100)
        expected_betas = [min(1 - math_cosine((i + 1) / 100) / math_cosine(i / 100), 0.999) for i in range(100)]
        torch.testing.assert_close(noise.betas, torch.tensor(expected_betas))
        self.assertEqual(noise.config.prediction_type, "epsilon")
        obs = {"point_cloud": torch.randn(2, 1, 8, 6), "joint_state": torch.randn(2, 1, 19)}
        batch = dict(obs=obs, action=torch.randn(2, 20, 19))
        loss, _ = model.compute_loss(batch)
        self.assertTrue(loss.isfinite())
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters() if p.requires_grad))
        optimizer = build_optimizer(model, lr=3e-4, weight_decay=1e-6)
        self.assertTrue(all(g["weight_decay"] == 1e-6 for g in optimizer.param_groups))
        model.eval()
        obs = {k: v[:1] for k, v in obs.items()}
        rng = torch.get_rng_state()
        result = model.predict_action(obs)
        self.assertEqual(result["future_actions"].shape, (1, 20, 19))
        self.assertEqual(set(result), {"future_actions", "pred_code_index", "pred_code_continuous"})
        self.assertEqual(len(noise.timesteps), 20)
        runner = BaseRunner(1, ["joint_state", "point_cloud"])
        torch.set_rng_state(rng)
        executed = runner.predict_action_chunk(obs, model, action_steps=16)
        torch.testing.assert_close(torch.from_numpy(executed), result["future_actions"][0, :16])
        torch.testing.assert_close(torch.from_numpy(executed[0]), result["future_actions"][0, 0])
        sat = SATPolicy(horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19,
                        num_points=8, state_out_dim=8, n_layers=1,
                        pc_encoder_config=dict(channels=24, num_patches=2, group_size=4, depth=1, num_heads=4),
                        fps_random_config={"use_random": False})
        with torch.no_grad():
            encoded = sat.obs_encoder({"joint_state": obs["joint_state"].repeat(1, 2, 1).flatten(0, 1),
                                      "point_cloud": obs["point_cloud"].repeat(1, 2, 1, 1).flatten(0, 1)})
        self.assertEqual(encoded.shape, (1, 256))


def math_cosine(t):
    import math
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


if __name__ == "__main__":
    unittest.main()
