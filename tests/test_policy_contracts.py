import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import CLIPTextConfig, CLIPTextModelWithProjection

from dexmani_policy.agents.hand_prior.hand_vae import HandVAE
from dexmani_policy.agents.hand_prior.motion_prior import HandMotionPrior
from dexmani_policy.agents.obs_encoder.text import CLIPTextEncoder
from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalResidualBlock1D
from dexmani_policy.common.checkpoint import TrainingCheckpoint, restore_policy_agent, save_training_checkpoint
from dexmani_policy.common.hand_representation import HAND_ACTION_SEMANTICS
from dexmani_policy.common.runtime import dict_apply
from dexmani_policy.training.builder import bind_defaults


def hand_artifacts(root):
    # Synthetic, one-update artifacts exercise the real artifact readers. They
    # establish mechanics only, not a useful hand prior or a research result.
    prior = HandMotionPrior(history_steps=3, hidden_dim=16)
    optim = torch.optim.Adam(prior.parameters(), lr=1e-3)
    prior.compute_loss(torch.randn(2, 3, 12), torch.randn(2, 12), .01)[0].backward()
    optim.step()
    prior.ready.fill_(True)
    torch.save(dict(model_config=prior.model_config, model_state_dict=prior.state_dict(),
        coordinate_space=prior.coordinate_space, metadata=dict(
            hand_joint_order=prior.hand_joint_order, hand_data={}, split_manifest={})), root / 'prior.pt')
    codec = HandVAE(horizon=8, latent_dim=2, hidden_dim=16, n_layers=1)
    optim = torch.optim.Adam(codec.parameters(), lr=1e-3)
    actions, q0 = torch.randn(2, 8, 12), torch.zeros(2, 12)
    mu, _ = codec.encode(actions, q0)
    codec.reconstruction_loss(codec.decode(mu, q0), actions).backward()
    optim.step()
    codec.ready.fill_(True)
    metadata = dict(model_config=codec.model_config, hand_data=dict(contract={
        **HAND_ACTION_SEMANTICS, 'horizon': 8, 'dt': .0625}),
        loss_reduction=dict(kl='mean_over_batch_time_latent'))
    torch.save(dict(model_config=codec.model_config, state_dict=codec.state_dict(), metadata=metadata),
               root / 'codec.pt')


def small_agent_config(name, root):
    classes = dict(dp='DPPolicy', dp3='DP3Policy', sat='SATPolicy', r3d='R3DPolicy',
                   maniflow='ManiFlowPolicy', multitask_dit='MultiTaskDiTPolicy',
                   dqrise='DQRISEPolicy', lamp='LAMPPolicy', latent_flow='LatentFlowPolicy')
    args = dict(_target_=f'dexmani_policy.agents.policies.{name}.{classes[name]}',
                horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19,
                state_dim=19, state_out_dim=8, num_inference_steps=2)
    unet = dict(down_dims=[8, 16], diffusion_step_embed_dim=16, kernel_size=3,
                n_groups=4, num_training_steps=10)
    sparse = dict(channels=48, num_heads=4, num_patches=4, group_size=4, depth=1)
    if name in ('dp', 'dp3', 'dqrise'):
        args.update(unet)
    if name in ('sat', 'dqrise', 'lamp', 'latent_flow'):
        args.update(pc_dim=6, num_points=16, pc_encoder_config=sparse)
    if name in ('dp', 'multitask_dit'):
        args.update(rgb_backbone_name='resnet', rgb_backbone_config=dict(model_name='resnet18',
            weights=None, load_pretrained=False, tune_mode='freeze',
            norm_mode='batch_norm', image_size=[32, 32]))
    if name == 'dp3':
        args.update(encoder_type='dp3', pc_dim=6, pc_out_dim=8, num_points=16)
    elif name == 'sat':
        args.update(horizon=16, n_layers=1)  # Also exercise valid C != default 8.
    elif name == 'r3d':
        args.update(pc_encoder_config=dict(embed_dim=32, num_group=4, group_size=4,
            use_pretrained_weights=False), timestep_embed_dim=16, embedding_dim=32,
            depth=1, num_heads=4, mlp_dim=64, num_training_steps=10)
    elif name == 'maniflow':
        args.update(pc_dim=6, num_points=16, pc_encoder_config=dict(out_channels=8),
            timestep_embed_dim=16, target_t_embed_dim=16, n_layers=1, hidden_dim=32,
            n_head=4, p_drop_attn=0.)
    elif name == 'multitask_dit':
        args.update(task_texts=['fixture'], text_embed_dim=4, n_emb=16, n_layers=1,
                    num_heads=4, attn_drop=0., proj_drop=0., num_training_steps=10,
                    prediction_type='epsilon')
    elif name == 'dqrise':
        args.update(horizon=20, n_obs_steps=1, n_action_steps=16)
    elif name == 'lamp':
        args.update(horizon=1, n_obs_steps=1, n_action_steps=1, num_inference_steps=1,
            prior_path=str(root / 'prior.pt'), history_steps=3, prior_hidden_dim=16,
            history_noise_std=.01, arm_state_noise_std=0., hand_loss_weight=1., drift_reg_weight=.1)
    elif name == 'latent_flow':
        args.update(codec_path=str(root / 'codec.pt'), latent_dim=2, codec_hidden_dim=16,
            codec_n_layers=1, n_layers=1, hidden_dim=32, n_head=4, p_drop_attn=0.)
    return OmegaConf.create(args)


class PolicyContractsTests(unittest.TestCase):
    def test_dp_film_zero_scale_removes_feature_path(self):
        block = ConditionalResidualBlock1D(4, 4, 8, n_groups=2)
        with torch.no_grad():
            block.cond_encoder[-1].weight.zero_()
            block.cond_encoder[-1].bias.zero_()
            block.cond_encoder[-1].bias[4:] = 1
        inputs = []
        handle = block.blocks[1].register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        block(torch.randn(2, 4, 8), torch.randn(2, 8))
        handle.remove()
        torch.testing.assert_close(inputs[0], torch.ones(2, 4, 8), rtol=0, atol=0)

    def test_nine_real_cpu_policies_backward_sampling_and_strict_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hand_artifacts(root)
            for name in ('dp', 'dp3', 'sat', 'r3d', 'maniflow', 'multitask_dit', 'dqrise', 'lamp', 'latent_flow'):
                with self.subTest(policy=name):
                    self.check_policy(name, root)

    def check_policy(self, name, root, device="cpu"):
        config = small_agent_config(name, root)
        agent = hydra.utils.instantiate(config)
        if name in ('lamp', 'latent_flow'):
            agent.initialize_training()
        if name == 'dqrise':
            agent.hand_codebook.set_buffers(dict(sorted_hand_poses=torch.linspace(-1, 1, 192).reshape(16, 12),
                pca_permutation=torch.arange(16), layer_weights=torch.ones(2),
                hand_normalizer_scale=torch.ones(12), hand_normalizer_offset=torch.zeros(12)))
        spec = dict(action='limits', joint_state='limits')
        widths = dict(action=19, joint_state=19)
        if name == 'lamp':
            spec.update(action='identity', joint_state='gaussian')
        if 'point_cloud' in agent.consumed_observation_fields:
            spec['point_cloud'] = 'identity' if name in ('sat', 'dqrise', 'lamp', 'latent_flow') else 'limits'
            widths['point_cloud'] = 6
        for key, mode in spec.items():
            if mode != 'identity':
                agent.normalizer.fit_field(key, [np.tile([[-1.0], [1.0]], (1, widths[key]))], mode=mode)
        agent.set_normalization_spec(spec, widths)
        agent.validate_normalizer()
        agent.to(device)
        batch_size, nobs = 3, agent.n_obs_steps
        obs = dict(joint_state=torch.randn(batch_size, nobs, 19) * .1)
        if 'point_cloud' in agent.consumed_observation_fields:
            obs['point_cloud'] = torch.randn(batch_size, nobs, 16, 6) * .1
        else:
            obs['rgb'] = torch.zeros(batch_size, nobs, 32, 32, 3, dtype=torch.uint8)
        if name == 'multitask_dit':
            obs['task_text'] = ['fixture'] * batch_size
        context = {}
        if name == 'lamp':
            context['hand_history'] = torch.zeros(batch_size, 3, 12)
        elif name == 'latent_flow':
            context['reference_state'] = obs['joint_state'][:, -1].clone()
        batch = dict(obs=obs, action=torch.randn(batch_size, agent.horizon, 19) * .1,
                     action_context=context)
        batch = dict_apply(batch, lambda value: value.to(device))
        obs, context = batch['obs'], batch['action_context']
        teacher = copy.deepcopy(agent).requires_grad_(False).eval() if name == 'maniflow' else None
        teacher_observations = []
        if teacher is not None:
            handle = teacher.obs_encoder.register_forward_hook(
                lambda module, args, result: teacher_observations.append(result))
        agent.train()
        loss, metrics = agent.compute_loss(batch, **agent.get_training_loss_kwargs(teacher))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for p in agent.parameters()))
        if teacher is not None:
            self.assertEqual((metrics['flow_batch_size'], metrics['consistency_batch_size']), (2, 1))
            torch.testing.assert_close(loss, metrics['loss_flow'] + metrics['loss_consistency'], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertEqual(teacher_observations, [])
            handle.remove()
            for size in (2, 1):
                small = dict(obs={k: v[:size] for k, v in obs.items()}, action=batch['action'][:size])
                if size == 1:
                    with self.assertRaises(ValueError):
                        agent.compute_loss(small, **agent.get_training_loss_kwargs(teacher))
                else:
                    _, grouped = agent.compute_loss(small, **agent.get_training_loss_kwargs(teacher))
                    self.assertEqual((grouped['flow_batch_size'], grouped['consistency_batch_size']), (1, 1))
        if name in ('lamp', 'latent_flow'):
            decoder = agent.prior if name == 'lamp' else agent.codec
            self.assertTrue(all(p.grad is None for p in decoder.parameters()))
            z = torch.randn((2, 2) if name == 'lamp' else (2, 8, 2), requires_grad=True, device=device)
            result = decoder.decode(z) if name == 'lamp' else decoder.decode(z, torch.zeros(2, 12, device=device))
            result.square().mean().backward()
            self.assertGreater(z.grad.abs().sum(), 0)
        agent.eval()
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(device).clone() if str(device).startswith("cuda") else None
        kwargs = dict(action_context=context) if context else {}
        with torch.no_grad():
            first = agent.predict_action(obs, generator=torch.Generator(device=device).manual_seed(21), **kwargs)
            other = agent.predict_action(obs, generator=torch.Generator(device=device).manual_seed(22), **kwargs)
            repeat = agent.predict_action(obs, generator=torch.Generator(device=device).manual_seed(21), **kwargs)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        if cuda_before is not None:
            self.assertTrue(torch.equal(cuda_before, torch.cuda.get_rng_state(device)))
        torch.testing.assert_close(first['future_actions'], repeat['future_actions'], rtol=0, atol=0)
        expected = agent.horizon if name == 'latent_flow' else agent.horizon - nobs + 1
        self.assertEqual(first['future_actions'].shape, (batch_size, expected, 19))
        if name != 'lamp':
            self.assertFalse(torch.equal(first['future_actions'], other['future_actions']))
        bind_defaults(config, type(agent))
        config.update(agent.effective_config)
        saved = dict(agent=OmegaConf.to_container(config, resolve=True), policy_metadata=agent.inference_metadata,
            dataset=dict(sensor_modalities=list(obs), rgb_preprocess_size=None), normalization=spec,
            normalization_dims=widths, action_key='action')
        checkpoint = TrainingCheckpoint(0, 1, 0, agent.state_dict(), agent.state_dict(), {}, {}, {}, 1, [], saved)
        path = save_training_checkpoint(root / f'{name}.pt', checkpoint)
        # Complete restoration must use embedded artifacts/text even without their files.
        for key in ('prior_path', 'codec_path'):
            if key in saved['agent']:
                saved['agent'][key] = '/missing/external/artifact.pt'
        with patch('transformers.CLIPTextModelWithProjection.from_pretrained',
                   side_effect=AssertionError('Unexpected external text load')):
            for ema in (False, True):
                restored = restore_policy_agent(saved, path, use_ema=ema, device=device)
                with torch.no_grad():
                    actual = restored.predict_action(obs, generator=torch.Generator(device=device).manual_seed(21), **kwargs)
                torch.testing.assert_close(first['future_actions'], actual['future_actions'], rtol=0, atol=0)

        if name == 'multitask_dit':
            flow_config = copy.deepcopy(config)
            flow_config.action_decoder_type = 'rectified_flow'
            flow_config.flow_num_inference_steps = 2
            flow = hydra.utils.instantiate(flow_config)
            flow.load_normalizer(agent.normalizer)
            flow.set_normalization_spec(spec, widths)
            flow.to(device)
            flow.compute_loss(batch)[0].backward()
            flow.eval()
            one = flow.predict_action(obs, generator=torch.Generator(device=device).manual_seed(8))
            two = flow.predict_action(obs, generator=torch.Generator(device=device).manual_seed(8))
            torch.testing.assert_close(one['future_actions'], two['future_actions'], rtol=0, atol=0)

    def test_clip_default_device_is_cpu_when_cuda_is_available(self):
        model = CLIPTextModelWithProjection(CLIPTextConfig(hidden_size=8, intermediate_size=16,
                    num_hidden_layers=1, num_attention_heads=2, projection_dim=4, vocab_size=16))
        # Downloads and device availability are the external boundaries.
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
             patch('transformers.CLIPTextModelWithProjection.from_pretrained', return_value=model), \
             patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(CLIPTextEncoder().device, torch.device('cpu'))
            self.assertEqual(CLIPTextEncoder(device='cpu').device, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
