import pytest
import torch
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "grest_under_test",
    ROOT / "dexmani_policy/agents/obs_encoder/interaction/scene_context.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Tokenizer = MODULE.GeometricRegionRelationTokenizer
RegionRelation = MODULE.RegionRelation
coverage_seeds = MODULE.coverage_seeds


@pytest.fixture(scope="module", autouse=True)
def single_thread_for_small_cpu_tests():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make(**kwargs):
    defaults = dict(token_channels=24, num_scene_tokens=6, route_dim=12,
                    relation_dim=16, relation_heads=4)
    defaults.update(kwargs)
    return Tokenizer(**defaults)


def data(b=2, n=37, d=24):
    g = torch.Generator().manual_seed(37)
    return torch.randn(b, n, d, generator=g), torch.randn(b, n, 3, generator=g) * .10


@pytest.mark.parametrize('k', [1, 4, 8, 12, 16, 24, 32])
def test_real_interface(k):
    m = Tokenizer(num_scene_tokens=k)
    out = m(torch.randn(2,192,192), torch.randn(2,192,3)*.1)
    assert out['scene_token'].shape == (2,k,192)
    assert out['scene_center'].shape == (2,k,3)
    assert out['scene_mask'].all()
    assert m.out_shape == (k,192)
    assert m.out_dim == 192
    assert torch.isfinite(out['scene_token']).all()


def test_assignment_and_support():
    z,p=data(); out=make()(z,p,True)
    a=out['assignment']; w=out['pool_weight']
    torch.testing.assert_close(a.sum(1),torch.ones(2,37))
    torch.testing.assert_close(w.sum(-1),torch.ones(2,6))
    assert (out['scene_mass'] >= 1-1e-6).all()
    assert (a.masked_select(~out['routing_allowed']) == 0).all()
    for b in range(2):
        seeds=out['anchor_indices'][b]
        assert seeds.unique().numel()==6
        torch.testing.assert_close(a[b,:,seeds],torch.eye(6))


def test_same_weights_for_geometry_features():
    z,p=data(); out=make()(z,p,True); w=out['pool_weight']
    torch.testing.assert_close(out['pooled_token'],w@z)
    torch.testing.assert_close(out['scene_center'],w@p)
    torch.testing.assert_close((out['scene_mass'][...,None]*out['pooled_token']).sum(1),z.sum(1),rtol=1e-5,atol=1e-5)
    delta=p[:,None]-out['scene_center'][:,:,None]
    expected=torch.einsum('bkn,bkni,bknj->bkij',w,delta,delta)
    torch.testing.assert_close(out['scene_covariance'],expected)
    assert (torch.linalg.eigvalsh(expected) >= -1e-7).all()


def test_covariance_units_under_translation():
    m=make(); z,p=data(); out=m(z,p,True); shift=torch.tensor([1.,-2.,3.])
    other=m(z,p+shift,True)
    torch.testing.assert_close(out['assignment'],other['assignment'],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(out['scene_center']+shift,other['scene_center'],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(out['scene_covariance'],other['scene_covariance'],atol=1e-6,rtol=2e-5)
    # The final token has absolute position features, so no translation invariance is claimed.


def test_masked_nan_padding_equivalence():
    m=make();z,p=data(); ref=m(z,p,True)
    zp=torch.cat((z,torch.full((2,7,24),float('nan'))),1)
    pp=torch.cat((p,torch.full((2,7,3),float('inf'))),1)
    mask=torch.cat((torch.ones(2,37,dtype=torch.bool),torch.zeros(2,7,dtype=torch.bool)),1)
    out=m(zp,pp,True,patch_mask=mask)
    torch.testing.assert_close(ref['scene_token'],out['scene_token'],atol=2e-6,rtol=2e-5)
    assert (out['assignment'][...,-7:] == 0).all()


def test_permutation_without_geometric_ties():
    m=make().eval(); z,p=data();perm=torch.randperm(37)
    ref=m(z,p,True);out=m(z[:,perm],p[:,perm],True)
    torch.testing.assert_close(ref['scene_token'],out['scene_token'],rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(ref['assignment'][:,:,perm],out['assignment'],rtol=2e-5,atol=2e-6)


def test_coincident_geometry_finite_unique_seeds():
    z,p=data();p.zero_();out=make()(z,p,True)
    assert torch.isfinite(out['scene_token']).all()
    assert out['scene_mass'].min()>=1-1e-6
    for row in out['anchor_indices']:
        assert row.unique().numel()==6


def test_k_equals_n():
    z,p=data(n=6);out=make()(z,p,True)
    assert torch.isfinite(out['scene_token']).all()
    torch.testing.assert_close(out['scene_mass'],torch.ones(2,6))
    torch.testing.assert_close(out['scene_covariance'],torch.zeros(2,6,3,3))


def test_routing_and_relation_gradients():
    m=make();z,p=data();z.requires_grad_();p.requires_grad_()
    out=m(z,p);loss=(out['scene_token']*torch.randn_like(out['scene_token'])).sum()
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert torch.isfinite(p.grad).all()
    for param in [m.route_projection.weight,m.relation.qkv.weight,m.relation.edge[-1].weight]:
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum()>0


def test_bfloat16_forward_backward():
    m=make();z,p=data();z.requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=m(z,p,True)
        loss=(out['scene_token'].float()*torch.randn_like(out['scene_token'].float())).mean()
    loss.backward()
    assert torch.isfinite(out['scene_token']).all()
    assert torch.isfinite(z.grad).all()
    assert out['scene_center'].dtype==torch.float32
    assert out['assignment'].dtype==torch.float32


def test_relation_equivariance():
    module=RegionRelation(24,16,4,.1);z,p=data(n=6);perm=torch.tensor([4,1,0,5,2,3])
    ref,att=module(z,p);out,oa=module(z[:,perm],p[:,perm])
    torch.testing.assert_close(out,ref[:,perm],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(oa,att[:,:,perm][:,:,:,perm])


def test_edge_value_has_geometry_signal_with_constant_node_values():
    # Bias-only attention cannot alter a convex sum of equal values. This module
    # also carries edge values, and can therefore transmit different relative geometry.
    torch.manual_seed(4);module=RegionRelation(24,16,4,.1)
    z=torch.zeros(1,3,24);p=torch.tensor([[[0.,0.,0.],[.1,0.,0.],[0.,.3,0.]]])
    out,_=module(z,p)
    assert not torch.allclose(out[:,0],out[:,1])


def test_relation_ablation():
    z,p=data();m=make(use_relation=False);out=m(z,p,True)
    assert m.relation is None
    assert 'relation_attention' not in out
    assert out['scene_token'].shape==(2,6,24)


@pytest.mark.parametrize('bad', ['empty','short','center','integer','nan','mask_type','too_few_valid'])
def test_bad_inputs(bad):
    m=make();z,p=data();mask=None
    if bad=='empty':z,p=z[:0],p[:0]
    elif bad=='short':z,p=z[:,:5],p[:,:5]
    elif bad=='center':p=p[:,:,:2]
    elif bad=='integer':z=z.long()
    elif bad=='nan':z[0,0,0]=float('nan')
    elif bad=='mask_type':mask=torch.ones(2,37)
    elif bad=='too_few_valid':mask=torch.zeros(2,37,dtype=torch.bool);mask[:,:5]=True
    with pytest.raises((ValueError,TypeError)):
        m(z,p,patch_mask=mask)


@pytest.mark.parametrize('kwargs',[{'num_scene_tokens':0},{'route_dim':True},{'geometry_scale':0.},
                                    {'relation_dim':15},{'routing_steps':0}])
def test_bad_constructor(kwargs):
    with pytest.raises(ValueError):make(**kwargs)


def test_coverage_padding_not_at_end():
    z,p=data();mask=torch.ones(2,37,dtype=torch.bool);mask[:,1::3]=False
    p[:,1::3]=float('nan');z[:,1::3]=float('inf');out=make()(z,p,True,patch_mask=mask)
    assert mask.gather(1,out['anchor_indices']).all()
    assert torch.isfinite(out['scene_token']).all()


def test_compile_capture_eager_backend_only():
    # This checks graph capture, NOT Inductor, CUDA kernels, deployment speed or
    # whole-repository torch.compile integration.
    torch._dynamo.reset()
    m=make(validate_inputs=False).eval();z,p=data(b=1,n=17)
    with torch.no_grad():
        ref=m(z,p)['scene_token']
        compiled=torch.compile(m,backend='eager',fullgraph=True)
        out=compiled(z,p)['scene_token']
    torch.testing.assert_close(ref,out)


@pytest.mark.parametrize("k", [1, 4, 8, 12, 16, 24])
def test_budget_independent_parameter_layout(k):
    model = Tokenizer(num_scene_tokens=k)
    reference = Tokenizer(num_scene_tokens=12)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )
    # Shape compatibility is NOT evidence of cross-budget policy performance.


def test_default_is_twelve_not_hardcoded_sixteen():
    model = Tokenizer()
    assert model.out_shape == (12, 192)


def test_cauchy_prior_avoids_forced_nearest_assignment_in_sparse_scene():
    # Identical content isolates the locality prior. Each seed keeps ownership.
    model = make(num_scene_tokens=2, routing_steps=1, nearest_slots=2)
    z = torch.ones(1, 3, 24)
    p = torch.tensor([[[0., 0., 0.], [2., 0., 0.], [.8, 0., 0.]]])
    out = model(z, p, True)
    distance2 = (out['routing_anchors'][:, :, None] - p[:, None]).square().sum(-1)
    expected = (-torch.log1p(distance2 / model.geometry_scale**2)).softmax(1)
    torch.testing.assert_close(out['assignment'][:, :, 2], expected[:, :, 2])
    gaussian = (-.5 * distance2 / model.geometry_scale**2).softmax(1)
    assert out['assignment'][0, :, 2].min() > .25
    assert gaussian[0, :, 2].min() < 1e-12
    # This tests numerical behavior only; neither prior is proved optimal.


@pytest.mark.parametrize('scale', [.02, .1, .5, 2.])
def test_router_gradient_across_metric_extents(scale):
    torch.manual_seed(7)
    model=make(); z,p=data(); p=p/.1*scale
    out=model(z,p)['scene_token']
    (out * torch.randn_like(out)).sum().backward()
    grad=model.route_projection.weight.grad
    assert torch.isfinite(grad).all() and grad.norm() > 1e-8


def test_diagnostics_do_not_change_output():
    model=make().eval(); z,p=data()
    torch.testing.assert_close(model(z,p)['scene_token'], model(z,p,True)['scene_token'])


@pytest.mark.parametrize('kwargs', [
    {'validate_inputs':'false'}, {'use_relation':'false'},
    {'relation_dim':4, 'relation_heads':4},
])
def test_strict_boolean_and_nondegenerate_head_config(kwargs):
    with pytest.raises((TypeError,ValueError)):
        make(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'channels':0}, {'heads':0}, {'attention_dim':15}, {'scale':0.},
])
def test_relation_standalone_validation(kwargs):
    args=dict(channels=24, attention_dim=16, heads=4, scale=.1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        RegionRelation(**args)


def test_bfloat16_module_weights():
    model=make().to(torch.bfloat16); z,p=data(); z=z.to(torch.bfloat16)
    out=model(z,p)['scene_token']
    (out.float()*torch.randn_like(out.float())).sum().backward()
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    assert torch.isfinite(model.route_projection.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device not available')
def test_cuda_autocast_backward():
    model=make().cuda(); z,p=data(); z,p=z.cuda().requires_grad_(),p.cuda()
    with torch.autocast('cuda', dtype=torch.float16):
        out=model(z,p)['scene_token']
        loss=(out.float()*torch.randn_like(out.float())).mean()
    loss.backward()
    assert torch.isfinite(out).all() and torch.isfinite(z.grad).all()
