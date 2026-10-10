"""Small CPU checks for the single-time backbone and unchanged flow decoder."""

import unittest

import torch

from dexmani_policy.agents.action_decoders.backbone.ditx import DiTX
from dexmani_policy.agents.action_decoders.rectified_flow import RectifiedFlow


def small_ditx():
    # Keep the production default of eight layers while reducing test width.
    return DiTX(
        horizon=4,
        action_dim=7,
        n_obs_steps=2,
        obs_token_dim=24,
        timestep_embed_dim=16,
        hidden_dim=32,
        n_head=4,
        mlp_ratio=2.0,
        p_drop_attn=0.0,
    )


class StandardDiTXTest(unittest.TestCase):
    def test_flow_training_and_inference(self):
        torch.manual_seed(7)
        model = small_ditx()
        self.assertEqual(len(model.ditx_blocks), 8)
        self.assertFalse(any("target_t" in name for name, _ in model.named_parameters()))
        flow = RectifiedFlow(model, num_inference_steps=2, time_shift_alpha=2.0)
        context = torch.randn(2, 10, 24, requires_grad=True)
        actions = torch.randn(2, 4, 7)
        optimizer = torch.optim.AdamW(model.get_optim_groups(), lr=1e-3)

        # Zero-initialized output and AdaLN gates open in successive updates.
        # Three real loss updates verify that observation gradients reach the
        # encoder after this intended initialization, without modifying gates.
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            context.grad = None
            loss, metrics = flow.compute_loss(context, actions)
            self.assertTrue(torch.isfinite(loss).item())
            self.assertIn("loss_action", metrics)
            loss.backward()
            self.assertTrue(
                all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            )
            optimizer.step()
        self.assertIsNotNone(context.grad)
        self.assertGreater(context.grad.abs().sum().item(), 0.0)

        model.eval()
        predicted = flow.predict_action(context.detach(), torch.zeros_like(actions))
        self.assertEqual(predicted.shape, actions.shape)
        self.assertTrue(torch.isfinite(predicted).all().item())
        for timestep in (0.25, torch.tensor(0.25), torch.tensor([0.25, 0.75])):
            velocity = model(actions, timestep, context.detach())
            self.assertEqual(velocity.shape, actions.shape)

    def test_optimizer_covers_each_trainable_parameter_once(self):
        model = small_ditx()
        groups = model.get_optim_groups(weight_decay=0.1)
        assigned = [id(p) for group in groups for p in group["params"]]
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), trainable)
        no_decay = {id(p) for group in groups if group["weight_decay"] == 0 for p in group["params"]}
        self.assertIn(id(model.input_pos_embed), no_decay)
        self.assertIn(id(model.context_frame_pos_embed), no_decay)
        self.assertIn(id(model.final_layer.norm_final.weight), no_decay)

    def test_context_uses_frame_major_temporal_positions(self):
        model = small_ditx()
        with torch.no_grad():
            model.context_embedder.weight.zero_()
            model.context_embedder.bias.zero_()
            model.context_frame_pos_embed[:, 0].fill_(1.0)
            model.context_frame_pos_embed[:, 1].fill_(2.0)
        embedded = model._embed_context(torch.zeros(2, 10, 24))
        self.assertTrue(torch.equal(embedded[:, :5], torch.ones(2, 5, 32)))
        self.assertTrue(torch.equal(embedded[:, 5:], torch.full((2, 5, 32), 2.0)))

    def test_rejects_incomplete_frames_and_wrong_action_horizon(self):
        model = small_ditx()
        with self.assertRaisesRegex(ValueError, "divisible"):
            model(torch.zeros(2, 4, 7), 0.5, torch.zeros(2, 9, 24))
        with self.assertRaisesRegex(ValueError, "horizon"):
            model(torch.zeros(2, 3, 7), 0.5, torch.zeros(2, 10, 24))


if __name__ == "__main__":
    unittest.main()
