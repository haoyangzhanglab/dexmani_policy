"""Run production encoder/agent class bodies with explicit dependency doubles.

These tests validate wiring, token budgets and modality isolation, not GAPT,
hand retrieval, DiTX, a training run, or closed-loop robot behavior.
"""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "dexmani_policy/agents/obs_encoder/interaction/scene_context.py"
spec = importlib.util.spec_from_file_location("grest_wiring_module", SCENE)
scene_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene_module)
Tokenizer = scene_module.GeometricRegionRelationTokenizer


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def load_class(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class GeometryDouble(nn.Module):
    def __init__(self, input_channels, num_patches, **kwargs):
        super().__init__()
        self.out_dim, self.num_patches = 32, num_patches
        self.project = nn.Linear(input_channels, self.out_dim)

    def forward(self, pointcloud, return_geometry=False):
        n = self.num_patches
        self.last_output = {
            "tokens": self.project(pointcloud[:, :n]),
            "xyz": pointcloud[:, :n, :3],
            "mask": torch.ones(pointcloud.shape[0], n, dtype=torch.bool),
        }
        return self.last_output


class TactileDouble(nn.Module):
    out_dim = 3

    def forward(self, force, valid):
        return force * valid[..., None]


class EvidenceDouble(nn.Module):
    def __init__(self, input_dim, channels):
        super().__init__()
        self.project = nn.Linear(input_dim, channels)

    def forward(self, frames, force, valid):
        fingers = self.project(frames) * valid[..., None]
        return torch.cat((torch.zeros_like(fingers[:, :1]), fingers), dim=1)


class SpatialDouble(nn.Module):
    def __init__(self, token_dim, **kwargs):
        super().__init__()
        self.project = nn.Linear(3, token_dim)

    def forward(self, xyz):
        return self.project(xyz)


class GateDouble(nn.Module):
    def __init__(self, channels):
        super().__init__()

    def forward(self, hand, evidence, scene, force, valid, enabled):
        self.last_scene = scene
        gain = torch.ones_like(hand[..., :1])
        return {"tactile_gain": gain, "geometry_gain": gain}


class RelationDouble(nn.Module):
    def __init__(self, token_channels):
        super().__init__()

    def forward(self, pointcloud, patches, anchor_xyz, hand_tokens,
                query_context=None, return_intermediate=False):
        self.last_patches = patches
        self.last_query = query_context
        return {
            "hand_query": hand_tokens,
            "relation_update": patches["tokens"].mean(1, keepdim=True) + query_context,
        }


class CoordinationDouble(nn.Identity):
    def __init__(self, *args):
        super().__init__()


def encoder_class():
    return load_class(
        "dexmani_policy/agents/obs_encoder/interaction/encoder.py", "InteractionObsEncoder",
        dict(torch=torch, nn=nn, GeometryAwarePointTransformer=GeometryDouble,
             GeometricRegionRelationTokenizer=Tokenizer,
             XHandForceEncoder=TactileDouble, XHandTaxelEncoder=TactileDouble,
             FingerEvidenceEncoder=EvidenceDouble, FingerModalityGate=GateDouble,
             HandSpatialEncoder=SpatialDouble, HandSceneRelationEncoder=RelationDouble,
             InteractionSelfAttentionBlock=CoordinationDouble,
             create_state_mlp=lambda source, target: nn.Linear(source, target)),
    )


def make_encoder(k=12, mode="query", **kwargs):
    return encoder_class()(
        n_obs_steps=2, token_dim=24, num_patches=32, num_scene_tokens=k,
        tactile_fusion=mode, **kwargs,
    )


def observations():
    g = torch.Generator().manual_seed(12)
    return {
        "point_cloud": torch.randn(4, 40, 6, generator=g) * .1,
        "joint_state": torch.randn(4, 19, generator=g),
        "fingertip_xyz": torch.randn(4, 5, 3, generator=g) * .1,
        "eef_pose": torch.randn(4, 9, generator=g) * .1,
        "finger_tactile_force": torch.randn(4, 5, 3, generator=g),
    }


@pytest.mark.parametrize("k", [4, 8, 12, 24])
@pytest.mark.parametrize("mode", ["query", "late", "none"])
def test_production_encoder_wiring(k, mode):
    model = make_encoder(k, mode)
    cond, aux = model(observations(), return_intermediate=True)
    assert cond.shape == (2, 2 * (k + 7), 24)
    assert aux["scene"]["assignment"].shape == (4, k, 32)
    assert model.relation.last_patches is model.geometry.last_output
    assert model.relation.last_patches["tokens"].shape == (4, 32, 24)
    torch.testing.assert_close(model.modality_gate.last_scene,
                               aux["scene"]["scene_token"].mean(1))
    (cond * torch.randn_like(cond)).mean().backward()
    assert torch.isfinite(model.scene.route_projection.weight.grad).all()
    assert torch.isfinite(model.geometry_projection[0].weight.grad).all()


@pytest.mark.parametrize("mode", ["query", "late", "none"])
def test_touch_never_conditions_scene_routing(mode):
    model = make_encoder(mode=mode).eval()
    obs = observations()
    _, first = model(obs, return_intermediate=True)
    obs["finger_tactile_force"] = obs["finger_tactile_force"] + 7.0
    _, second = model(obs, return_intermediate=True)
    torch.testing.assert_close(first["scene"]["scene_token"], second["scene"]["scene_token"])
    torch.testing.assert_close(first["scene"]["assignment"], second["scene"]["assignment"])
    if mode == "none":
        torch.testing.assert_close(first["frame_tokens"], second["frame_tokens"])


def test_no_diagnostics_and_relation_ablation():
    model = make_encoder(scene_tokenizer={"use_relation": False, "route_dim": 8})
    cond, aux = model(observations())
    assert aux == {} and cond.shape == (2, 38, 24)
    assert model.scene.relation is None and model.scene.route_dim == 8


@pytest.mark.parametrize("options", [{"num_scene_tokens": 3}, {"token_channels": 48}])
def test_no_duplicate_budget_or_width(options):
    with pytest.raises(ValueError, match="not scene_tokenizer"):
        make_encoder(scene_tokenizer=options)


@pytest.mark.parametrize("k", [0, -1, True, 33])
def test_encoder_budget_validation(k):
    with pytest.raises(ValueError):
        make_encoder(k)


def test_unknown_option_fails():
    with pytest.raises(TypeError):
        make_encoder(scene_tokenizer={"unknown_option": 1})


class BaseDouble(nn.Module):
    def __init__(self, obs_encoder, action_decoder, **kwargs):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder


class BackboneDouble(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs


class FlowDouble(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model, self.options = model, kwargs


def agent_class():
    return load_class(
        "dexmani_policy/agents/core/interaction_flow.py", "InteractionFlowAgent",
        dict(torch=torch, BaseAgent=BaseDouble, InteractionObsEncoder=encoder_class(),
             DiTX=BackboneDouble, RectifiedFlow=FlowDouble),
    )


def test_agent_passes_budget_and_tokenizer_config():
    agent = agent_class()(horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19,
                          token_dim=24, num_patches=32, num_scene_tokens=8,
                          scene_tokenizer={"route_dim": 8, "use_relation": False})
    assert agent.obs_encoder.scene.out_shape == (8, 24)
    assert agent.obs_encoder.scene.route_dim == 8
    assert agent.obs_encoder.scene.relation is None
    assert agent.action_decoder.options["num_inference_steps"] == 10
    assert agent.action_decoder.model.options["n_layers"] == 8


def test_config_defaults_and_ddp_inheritance():
    import inspect
    cfg = yaml.safe_load((ROOT / "dexmani_policy/configs/interaction_flow.yaml").read_text())
    options = cfg["agent"]
    assert options["num_scene_tokens"] == 12
    model = Tokenizer(token_channels=options["token_dim"],
                      num_scene_tokens=options["num_scene_tokens"],
                      **options["scene_tokenizer"])
    assert model.out_shape == (12, 192)
    assert set(options) - {"_target_"} <= set(inspect.signature(agent_class()).parameters)
    assert cfg["normalization"]["point_cloud"] == "identity"
    assert cfg["normalization"]["fingertip_xyz"] == "identity"
    assert cfg["normalization"]["eef_pose"] == "identity"
    assert cfg["training"]["seed"] == "${seed}"
    assert "grest${agent.num_scene_tokens}" in cfg["hydra"]["run"]["dir"]
    ddp = yaml.safe_load((ROOT / "dexmani_policy/configs/ddp/interaction_flow.yaml").read_text())
    assert ddp["defaults"] == ["/interaction_flow", "_self_"]
    assert "agent" not in ddp
