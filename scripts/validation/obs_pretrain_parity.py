"""Compare active local computations against definitions at the pinned upstream SHAs."""
import argparse
import ast
import copy
import json
import logging
import math
import numpy as np
from pathlib import Path
import re
import subprocess
import typing

import einops
import torch
from einops.layers.torch import Rearrange
from omegaconf import OmegaConf
from timm.models.vision_transformer import Attention, Mlp
from torch import nn
from torch.nn import functional as F

from dexmani_policy.agents.obs_encoder.pointcloud.pointnet import PointNet
from dexmani_policy.agents.obs_encoder.pretrained import AFRO_SHA, CORDVIP_SHA
from dexmani_policy.common.pytorch_util import dict_apply


class IdentityNormalizer(nn.Module):
    def normalize(self, values):
        return values

    def __getitem__(self, key):
        return self


def definitions(path, names, namespace, replacements=()):
    source = path.read_text()
    for old, new in replacements:
        assert old in source, (path, old)
        source = source.replace(old, new)
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names), (path, names)
    exec(compile(ast.Module(body=[*nodes], type_ignores=[]), str(path), 'exec'), namespace)


def reference_namespace():
    return dict(torch=torch, nn=nn, F=F, math=math, copy=copy, Dict=typing.Dict,
                Union=typing.Union, List=typing.List, Type=typing.Type, einops=einops,
                Rearrange=Rearrange, Attention=Attention, Mlp=Mlp, dict_apply=dict_apply,
                cprint=lambda *a, **k: None, print_params=lambda *a: None,
                LinearNormalizer=IdentityNormalizer, logger=logging.getLogger('reference'),
                DDPMScheduler=object)


def check(left, right):
    torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)
    return float((left - right).abs().max()) if left.numel() else 0.0


def transfer(local, reference, rename=lambda name: name):
    state = reference.state_dict()
    local.load_state_dict({name: state[rename(name)] for name in local.state_dict()}, strict=True)


def gradients(local, reference, rename=lambda name: name):
    other = dict(reference.named_parameters())
    error = 0.0
    for name, parameter in local.named_parameters():
        expected = other[rename(name)].grad
        if parameter.grad is None or expected is None:
            assert parameter.grad is None and expected is None, name
        else:
            error = max(error, check(parameter.grad, expected))
    return error


def pointnet_reference(root, namespace, cordvip=False):
    prefix = 'train_policy/policy/CordViP/CordViP/CordViP' if cordvip else 'afro_workspace'
    definitions(root / prefix / 'model/vision/pointnet_extractor.py', ['PointNetEncoderXYZ'], namespace)
    return namespace['PointNetEncoderXYZ']


def pointnet_parity(root, device, dim=64, cordvip=False):
    namespace = reference_namespace()
    cls = pointnet_reference(root, namespace, cordvip)
    ref = cls(in_channels=3, out_channels=dim, use_layernorm=True, final_norm='layernorm').to(device)
    local = PointNet(3, dim, pooling='max').to(device)
    rename = lambda name: name.replace('output_projection.', 'final_projection.')
    transfer(local, ref, rename)
    errors = {}
    for tied in (False, True):
        x = torch.randn(2, 1024, 3, device=device)
        if tied:
            x[:] = x[:, :1].clone()
        a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
        out, expected = local(a)['global_token'], ref(b)
        probe = torch.randn_like(out)
        (out * probe).sum().backward()
        (expected * probe).sum().backward()
        errors[str(tied)] = dict(output=check(out, expected), input_grad=check(a.grad, b.grad),
                                 parameter_grad=gradients(local, ref, rename))
        local.zero_grad(set_to_none=True)
        ref.zero_grad(set_to_none=True)
    return errors


def afro_parity(root, device):
    from dexmani_policy.training.pretrain.obs.afro import AFROPretrain, vicreg_loss
    namespace = reference_namespace()
    definitions(root / 'afro_workspace/model/transformer/DiT1D.py',
                ['modulate', 'TimestepEmbedder', 'DiTBlock', 'DiTFeaturePredictor'], namespace)
    pointnet_reference(root, namespace)
    definitions(root / 'afro_workspace/model/vision/pointnet_extractor.py', ['VisEncoder', 'DiTFDMDecoder'], namespace)
    definitions(root / 'afro_workspace/policy/afro.py', ['AFRO'], namespace)
    local = AFROPretrain().to(device)
    assert not torch.equal(local.visual.mlp[0].weight, local.teacher.mlp[0].weight)
    ref = namespace['AFRO'](
        shape_meta=OmegaConf.create({'action': {'shape': [19]}, 'obs': {'point_cloud': {'shape': [1024, 3]}}}),
        noise_scheduler=copy.deepcopy(local.noise_scheduler), horizon=33, encoder_output_dim=64,
        pointcloud_encoder_cfg=OmegaConf.create(dict(in_channels=3, out_channels=64,
                                                    use_layernorm=True, final_norm='layernorm')),
    ).to(device)

    def rename(name):
        return name.replace('visual.', 'vis_encoder.extractor.').replace('teacher.', 'ema_vis_encoder.extractor.').replace(
            'output_projection.', 'final_projection.').replace('idm.', 'idm_mlp.').replace('fdm.', 'fdm_vis_decoder.fdm_decoder.')

    # Check both AdaLN-Zero initialization and an active conditioning gradient path.
    records = {}
    for active in (False, True):
        if active:
            with torch.no_grad():
                for block in ref.fdm_vis_decoder.fdm_decoder.blocks:
                    block.adaLN_modulation[-1].weight.normal_(std=0.01)
        transfer(local, ref, rename)
        a = torch.randn(2, 33, 1024, 3, device=device, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        torch.manual_seed(791)
        loss, metrics = local({'obs': {'point_cloud': a}})
        torch.manual_seed(791)
        expected, reference_metrics = ref.compute_loss({'obs': {'point_cloud': b}}, epoch=37)
        loss.backward()
        expected.backward()
        term_names = {'forward_invariance': 'inv_loss', 'forward_variance': 'var_loss',
                      'forward_covariance': 'cov_loss', 'reverse_invariance': 'reverse_mse',
                      'reverse_variance': 'reverse_var_loss', 'reverse_covariance': 'reverse_cov_loss'}
        term_error = max(check(metrics[k], metrics[k].new_tensor(reference_metrics[v])) for k, v in term_names.items())
        local.update_teacher(37)
        teacher_error = max(check(value, ref.state_dict()[rename('teacher.' + key)])
                            for key, value in local.teacher.state_dict().items())
        records[str(active)] = dict(loss=check(loss, expected), terms=term_error,
                                    input_grad=check(a.grad, b.grad), parameter_grad=gradients(local, ref, rename),
                                    teacher=teacher_error)
        local.zero_grad(set_to_none=True)
        ref.zero_grad(set_to_none=True)
    for n in (1, 7):
        a, b = torch.randn(n, 64, device=device), torch.randn(n, 64, device=device)
        actual = vicreg_loss(a, b)
        expected = ref._vicreg_loss(a, b, 25, 25, 1)
        records[f'vicreg_n{n}'] = max(check(x, y) for x, y in zip(actual, expected))
    return records


def unet_parity(root, device):
    from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D
    namespace = reference_namespace()
    prefix = root / 'afro_workspace/model/diffusion'
    definitions(prefix / 'positional_embedding.py', ['SinusoidalPosEmb'], namespace)
    definitions(prefix / 'conv1d_components.py', ['Downsample1d', 'Upsample1d', 'Conv1dBlock'], namespace)
    definitions(prefix / 'conditional_unet1d.py', ['ConditionalResidualBlock1D', 'ConditionalUnet1D'], namespace)
    # The same active FiLM branch is tested at both native horizons/conditioning widths.
    records = {}
    for horizon, context, time_dim in ((16, 256, 128), (12, 8704, 64)):
        local = ConditionalUNet1D(19, context, time_dim, (512, 1024, 2048), 5, 8, True).to(device)
        ref = namespace['ConditionalUnet1D'](19, global_cond_dim=context, diffusion_step_embed_dim=time_dim,
                                           down_dims=[512, 1024, 2048], kernel_size=5, n_groups=8).to(device)
        def rename(name):
            name = name.replace('diffusion_step_encoder.net.', 'diffusion_step_encoder.')
            return re.sub(r'((?:down|up)_modules\.\d+\.2)\.(weight|bias)$', r'\1.conv.\2', name)
        transfer(local, ref, rename)
        a = torch.randn(1, horizon, 19, device=device, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        c = torch.randn(1, context, device=device, requires_grad=True)
        d = c.detach().clone().requires_grad_()
        t = torch.tensor([37], device=device)
        out, expected = local(a, t, c), ref(b, t, global_cond=d)
        probe = torch.randn_like(out)
        (out * probe).mean().backward()
        (expected * probe).mean().backward()
        records[str(horizon)] = dict(output=check(out, expected), input_grad=check(a.grad, b.grad),
                                     context_grad=check(c.grad, d.grad), parameter_grad=gradients(local, ref, rename))
        del local, ref, out, expected, a, b, c, d
    return records


def contact_parity(root):
    import open3d as o3d
    from dexmani_policy.training.pretrain.obs.contact_map import compute_contact_map, estimate_normals
    namespace = dict(np=np, o3d=o3d)
    definitions(root / 'train_policy/script/contact_map.py',
                ['estimate_normals', 'compute_aligned_distance', 'compute_contact_value', 'compute_contact_map'], namespace)
    rng = np.random.default_rng(42)
    objects = rng.normal(size=(1024, 3)) * 0.03
    hand = rng.normal(size=(1024, 3)) * 0.03
    normals = estimate_normals(objects)
    np.testing.assert_allclose(normals, namespace['estimate_normals'](objects), rtol=0, atol=0)
    actual = compute_contact_map(objects, normals, hand)
    expected = namespace['compute_contact_map'](objects, normals, hand, gamma=1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    objects, normals = np.zeros((1, 3)), np.array([[1., 0., 0.]])
    hand = np.array([[0., 0.1, 0.], [0.15, 0., 0.]])
    distance = namespace['compute_aligned_distance'](objects[0], normals[0], hand)
    assert 0.15 <= distance < 0.151
    special = compute_contact_map(objects, normals, hand)
    np.testing.assert_allclose(special, namespace['compute_contact_map'](objects, normals, hand, gamma=1), atol=1e-12)
    return dict(max_abs=float(np.max(np.abs(actual - expected))), normals_max_abs=0.0,
                weighted_nearest_differs=True)


def cordvip_parity(root, device, *, native):
    from dexmani_policy.agents.obs_encoder.interaction.cordvip_encoder import CordViPEncoder
    from dexmani_policy.agents.obs_encoder.interaction.cordvip_transformer import Transformer
    from dexmani_policy.common.normalizer import LinearNormalizer
    from dexmani_policy.training.pretrain.obs.cordvip import CordViPPretrain, create_fc
    prefix = root / 'train_policy/policy/CordViP/CordViP/CordViP'
    namespace = reference_namespace()
    namespace.update(BasePolicy=nn.Module, reduce=einops.reduce)
    pointnet_reference(root, namespace, True)
    transformer_names = ['Transformer', 'clones', 'attention', 'LayerNorm', 'EncoderDecoder',
                         'Decoder', 'DecoderLayer', 'SublayerConnection', 'Encoder', 'EncoderLayer',
                         'MultiHeadedAttention', 'PositionwiseFeedForward']
    definitions(prefix / 'model/vision/transformer.py', transformer_names, namespace)
    encoder_changes = []
    head_changes = [('return loss', 'return loss, (loss_contact_map, loss_arm, loss_hand)')]
    if native:
        encoder_changes = [('nn.Linear(16,robot_embedding_dim)', 'nn.Linear(12,robot_embedding_dim)'),
                           ('nn.Linear(6,robot_embedding_dim)', 'nn.Linear(7,robot_embedding_dim)'),
                           ('joint_state[:, :, :16]', 'joint_state[:, :, 7:19]'),
                           ('joint_state[:, :, 16:]', 'joint_state[:, :, :7]')]
        head_changes += [('output_dim=6,', 'output_dim=7,'), ('output_dim=16,', 'output_dim=12,'),
                         ('joint_state_at_t4[:, :16]', 'joint_state_at_t4[:, 7:19]'),
                         ('joint_state_at_t4[:, 16:]', 'joint_state_at_t4[:, :7]')]
    definitions(prefix / 'model/vision/encoder.py', ['PointROEncoder'], namespace, encoder_changes)
    definitions(prefix / 'model/vision/create_fc.py', ['create_fc'], namespace)
    definitions(prefix / 'policy/CordViP_pretrain.py', ['CordViP_Pretrain'], namespace, head_changes)
    ref = namespace['CordViP_Pretrain'](
        shape_meta=OmegaConf.create({'obs': {}}), noise_scheduler=None, horizon=12, n_obs_steps=4,
        n_action_steps=6, encoder_emb_dim=1024,
        pointcloud_encoder_cfg=OmegaConf.create(dict(in_channels=3, out_channels=1024,
                                                    use_layernorm=True, final_norm='layernorm')),
    ).to(device)
    if native:
        local = CordViPPretrain().to(device)
    else:
        # Only the test shim restores the original hand16-before-arm6 layout.
        local_namespace: dict[str, typing.Any] = dict(
            nn=nn, torch=torch, F=F, PointNet=PointNet, Transformer=Transformer,
            LinearNormalizer=LinearNormalizer, create_fc=create_fc,
        )
        package = Path(__file__).resolve().parents[2] / 'dexmani_policy'
        definitions(package / 'agents/obs_encoder/interaction/cordvip_encoder.py', ['CordViPEncoder'], local_namespace,
                    [('nn.Linear(12, 64)', 'nn.Linear(16, 64)'), ('nn.Linear(7, 64)', 'nn.Linear(6, 64)'),
                     ('joint_state[:, :, 7:19]', 'joint_state[:, :, :16]'),
                     ('joint_state[:, :, :7]', 'joint_state[:, :, 16:]')])
        definitions(package / 'training/pretrain/obs/cordvip.py', ['CordViPPretrain'], local_namespace,
                    [('2112, 7,', '2112, 6,'), ('2112, 12,', '2112, 16,'),
                     ('target[:, :7]', 'target[:, 16:]'), ('target[:, 7:19]', 'target[:, :16]')])
        local = local_namespace['CordViPPretrain']().to(device)
    rename = lambda name: name.replace('output_projection.', 'final_projection.')
    transfer(local, ref, rename)
    joint_dim = 19 if native else 22
    inputs = {'hand_pointcloud': torch.randn(2, 4, 1024, 3, device=device, requires_grad=True),
              'object_pointcloud': torch.randn(2, 4, 1024, 3, device=device, requires_grad=True),
              'joint_state': torch.randn(2, 4, joint_dim, device=device, requires_grad=True)}
    reference_inputs = {key.replace('pointcloud', 'point_cloud'): value.detach().clone().requires_grad_()
                        for key, value in inputs.items()}
    targets = torch.rand(2, 4, 1024, device=device)
    local_outputs, reference_outputs = [], []
    local_handle = local.obs_encoder.register_forward_hook(lambda module, args, output: local_outputs.extend(output))
    ref_handle = ref.obs_encoder.register_forward_hook(lambda module, args, output: reference_outputs.extend(output))
    loss, metrics = local({'obs': inputs, 'contact_map': targets})
    expected, terms = ref.compute_loss_hand_and_arm({'obs': reference_inputs, 'contact_map': targets,
                                                    'joint_action': torch.zeros(2, 12, joint_dim, device=device)})
    local_handle.remove()
    ref_handle.remove()
    loss.backward()
    expected.backward()
    records = dict(output=max(check(a, b) for a, b in zip(local_outputs, reference_outputs)),
                   loss=check(loss, expected), terms=max(check(metrics[k], term) for k, term in zip(['contact', 'arm', 'hand'], terms)),
                   input_grad=max(check(value.grad, reference_inputs[key.replace('pointcloud', 'point_cloud')].grad)
                                  for key, value in inputs.items()), parameter_grad=gradients(local, ref, rename))
    with torch.no_grad():
        original = [value.detach() for value in local.obs_encoder(inputs)]
        changed = {key: value.detach().clone() for key, value in inputs.items()}
        for value in changed.values():
            value[1] += 17
        isolated = local.obs_encoder(changed)
        records['batch_isolation'] = max(check(a[0], b[0]) for a, b in zip(original, isolated))
        changed = {key: value.detach().clone() for key, value in inputs.items()}
        changed['hand_pointcloud'][:, -1] += 4
        temporal = local.obs_encoder(changed)
        assert (original[0][:, 0] - temporal[0][:, 0]).abs().max() > 1e-5
        records['attention_over_time'] = True
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--afro-root', type=Path, default=Path('/tmp/dexmani-afro-ref'))
    parser.add_argument('--cordvip-root', type=Path, default=Path('/tmp/dexmani-cordvip-ref'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--method', choices=['afro', 'cordvip', 'all'], default='all')
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    results = dict(dtype='float32', eager=True, rtol=3e-4, atol=3e-5, device=args.device)
    if args.method in ('afro', 'all'):
        assert subprocess.check_output(['git', '-C', str(args.afro_root), 'rev-parse', 'HEAD'], text=True).strip() == AFRO_SHA
        results['pointnet64'] = pointnet_parity(args.afro_root, args.device)
        results['afro'] = afro_parity(args.afro_root, args.device)
        results['unet'] = unet_parity(args.afro_root, args.device)
    if args.method in ('cordvip', 'all'):
        assert subprocess.check_output(['git', '-C', str(args.cordvip_root), 'rev-parse', 'HEAD'], text=True).strip() == CORDVIP_SHA
        results['pointnet1024'] = pointnet_parity(args.cordvip_root, args.device, 1024, True)
        results['contact'] = contact_parity(args.cordvip_root)
        results['cordvip22'] = cordvip_parity(args.cordvip_root, args.device, native=False)
        results['cordvip19'] = cordvip_parity(args.cordvip_root, args.device, native=True)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
