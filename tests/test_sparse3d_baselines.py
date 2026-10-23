"""Baseline contracts using real CPU Sparse3D ops; no learned codebook required."""

import ast
import copy
import inspect
import subprocess
import sys
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_policy.agents.action_decoders.backbones.sat import SAT, XHandJointEmbedding
from dexmani_policy.agents.obs_encoder.pointcloud.global_sparse3d import (
    Sparse3DGlobalObservationEncoder,
)
from dexmani_policy.common.config import load_policy_config
from dexmani_policy.training.validation import validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("prefix", ["", "ddp/"])
def test_matching_recipes(prefix):
    sat, dq = (load_policy_config(prefix + name) for name in ("sat", "dqrise"))
    for cfg in (sat, dq):
        validate_config(cfg)
    fields = [
        "dataset",
        "normalization",
        "env_runner",
        "seed",
        "horizon",
        "n_obs_steps",
        "n_action_steps",
        "action_key",
        "action_dim",
        "dataloader",
        "training.loop",
        "training.seed",
        "eval.seed",
        "eval.select_best",
        "eval.offline",
        "eval.demo",
        "eval.video",
        "eval.use_ema",
        "agent.pc_dim",
        "agent.state_dim",
        "agent.state_out_dim",
        "agent.num_points",
        "agent.pc_encoder_config",
        "agent.fps_random_config",
        "agent.modality_dropout_probs",
    ]
    for field in fields:
        left, right = (OmegaConf.select(cfg, field) for cfg in (sat, dq))
        assert left == right, field
    for cfg in (sat, dq):
        assert (
            cfg.dataloader.batch_size
            * cfg.training.get("num_gpus", 1)
            * cfg.training.loop.gradient_accumulation_steps
            == 128
        )
        assert (
            cfg.agent.pc_encoder_config.channels // cfg.agent.pc_encoder_config.num_heads % 6 == 0
        )
    assert dq.codebook_path is None


@pytest.mark.parametrize("name", ["sat", "dqrise"])
@pytest.mark.parametrize(
    "override",
    [
        "normalization.point_cloud=limits",
        "normalization.joint_state=gaussian",
        "action_key=action_ee",
        "agent.pc_dim=3",
        "n_obs_steps=1",
        "env_runner.env_kwargs.control_mode=ee",
    ],
)
def test_reject_wrong_baseline_recipe(name, override):
    with pytest.raises(ValueError):
        validate_config(load_policy_config(name, [override]))


@pytest.mark.parametrize("name", ["sat", "dqrise"])
def test_omitted_encoder_config_uses_global_defaults(name):
    cfg = load_policy_config(name, ["agent.num_points=64"])
    del cfg.agent.pc_encoder_config
    validate_config(cfg)
    model = hydra.utils.instantiate(cfg.agent)
    assert model.obs_encoder.pc_encoder.effective_config["num_patches"] == 64
    assert model.obs_encoder.pc_encoder.effective_config["depth"] == 4


def test_dqrise_constructor_defaults_match_recipe():
    cfg = load_policy_config("dqrise")
    policy = hydra.utils.get_class(cfg.agent._target_)
    defaults = inspect.signature(policy).parameters
    for name in ("diffusion_step_embed_dim", "down_dims", "num_inference_steps", "prediction_type"):
        configured = cfg.agent[name]
        if name == "down_dims":
            configured = tuple(configured)
        assert defaults[name].default == configured, name


def test_import_boundaries():
    for filename, forbidden in (
        ("agents/policies/sat.py", "dqrise"),
        ("agents/policies/dqrise.py", "sat"),
        ("agents/obs_encoder/pointcloud/global_sparse3d.py", "policies"),
    ):
        tree = ast.parse((ROOT / "dexmani_policy" / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""] + [alias.name for alias in node.names]
            else:
                continue
            assert all(forbidden not in name.split(".") for name in imports), filename
    tree = ast.parse((ROOT / "dexmani_policy/agents/policies/__init__.py").read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


@pytest.mark.parametrize(
    "name,other,cls", [("sat", "dqrise", "SATPolicy"), ("dqrise", "sat", "DQRISEPolicy")]
)
def test_independent_import_and_construction(name, other, cls):
    script = f"""
import importlib, importlib.abc, sys
class BlockOther(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("dexmani_policy.agents.policies.{other}", "dexmani_policy.agents.action_decoders.backbones.{other}"):
            raise AssertionError("Forbidden dependency: " + fullname)
sys.meta_path.insert(0, BlockOther())
from dexmani_policy.common.config import load_policy_config
import hydra
from dexmani_policy.agents.obs_encoder.pointcloud.global_sparse3d import Sparse3DGlobalObservationEncoder
cfg = load_policy_config("{name}")
model = hydra.utils.instantiate(cfg.agent)
assert type(model).__name__ == "{cls}"
assert type(model.obs_encoder) is Sparse3DGlobalObservationEncoder
assert "dexmani_policy.agents.policies.{other}" not in sys.modules
print("{name} parameters", sum(p.numel() for p in model.parameters()))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


@pytest.mark.parametrize("batch", [1, 3])
def test_global_encoder_shapes_order_geometry_and_gradients(batch):
    encoder = Sparse3DGlobalObservationEncoder().eval()
    pc = torch.rand(batch * 2, 128, 6)
    pc[..., :3] += torch.tensor([0.2, -0.4, 0.05])
    state = torch.randn(batch * 2, 19)
    obs = {"point_cloud": pc, "joint_state": state}
    captured = {}

    def capture_pc(module, args):
        captured["pc"] = args[0].detach().clone()

    handle = encoder.pc_encoder.register_forward_pre_hook(capture_pc)
    output = encoder(obs)
    handle.remove()
    assert output.shape == (batch, 256)
    torch.testing.assert_close(captured["pc"], pc, rtol=0, atol=0)
    with torch.no_grad():
        repeated = encoder(obs)
        per_frame = torch.cat(
            (encoder.pc_encoder(pc)["tokens"].mean(1), encoder.state_mlp(state)), -1
        )
        expected = encoder.temporal_projection(per_frame.reshape(batch, 2, 448).flatten(1))
    torch.testing.assert_close(output, repeated, rtol=0, atol=0)
    torch.testing.assert_close(output, expected)
    reversed_obs = {
        key: value.unflatten(0, (batch, 2)).flip(1).flatten(0, 1) for key, value in obs.items()
    }
    assert not torch.allclose(output, encoder(reversed_obs))
    output.square().mean().backward()
    for module in (encoder.pc_encoder, encoder.state_mlp, encoder.temporal_projection):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    assert encoder.warmup_num_points == 1024
    assert encoder.pc_encoder.effective_config["depth"] == 4
    assert encoder.pc_encoder.effective_config["num_patches"] == 64
    encoder.train()
    assert encoder.pc_encoder.training and encoder.state_mlp.training
    with pytest.raises(ValueError, match="aligned"):
        encoder({"point_cloud": pc, "joint_state": state[:-1]})


def test_pooling_is_invariant_to_each_frames_patch_order():
    encoder = Sparse3DGlobalObservationEncoder().eval()
    obs = {"point_cloud": torch.rand(4, 128, 6), "joint_state": torch.randn(4, 19)}
    with torch.no_grad():
        expected = encoder(obs)

        def permute_patches(module, args, result):
            result["tokens"] = torch.stack(
                [tokens[torch.randperm(len(tokens))] for tokens in result["tokens"]]
            )
            return result

        handle = encoder.pc_encoder.register_forward_hook(permute_patches)
        actual = encoder(obs)
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_ejc_mapping_initialization_and_restore():
    backbone = SAT(16, 19)
    ejc = backbone.joint_embedding
    expected = {
        "joint_ids": list(range(19)),
        "group_ids": [0] * 7 + [1] * 3 + [2] * 3 + [3] * 2 + [4] * 2 + [5] * 2,
        "role_ids": [0] * 7 + [1, 2, 3] + [1, 2, 3] + [2, 3] * 3,
        "axis_ids": [0] * 7 + [4, 2, 2] + [3, 1, 1] + [1, 1] * 3,
    }
    for (name, values), limit in zip(expected.items(), (19, 6, 4, 5)):
        actual = getattr(ejc, name)
        assert actual.tolist() == values and actual.shape == (19,)
        assert int(actual.min()) >= 0 and int(actual.max()) < limit
    assert ejc.joint_ids.unique().numel() == 19
    assert ejc().shape == (19, 256)
    assert sum(p.numel() for p in ejc.parameters()) == 8704
    restored = XHandJointEmbedding()
    restored.load_state_dict(ejc.state_dict())
    torch.testing.assert_close(restored(), ejc(), rtol=0, atol=0)
    for key in expected:
        broken = copy.deepcopy(ejc.state_dict())
        broken[key][0] = 1
        with pytest.raises(RuntimeError, match="canonical"):
            restored.load_state_dict(broken)
    trajectory = backbone.x_embedder(torch.randn(64, 19, 16))
    structure = ejc()
    rms = [
        value.detach().square().mean().sqrt().item()
        for value in (trajectory, structure, trajectory + structure)
    ]
    print("RMS trajectory / structure / sum:", rms)
    assert all(0 < value < float("inf") for value in rms)
    # Guard against order-of-magnitude regressions, not an optimal research ratio.
    assert max(rms[:2]) / min(rms[:2]) < 10
    assert all(
        abs(e.weight.std().item() - 0.005) < 0.0005
        for e in (ejc.joint, ejc.group, ejc.role, ejc.axis)
    )
    with pytest.raises(RuntimeError):
        backbone.load_state_dict({"joint_codebook.joint_function": torch.arange(19)})


def test_sat_fixed_observation_and_attention_information_flow():
    backbone = SAT(16, 19).eval()
    with torch.no_grad():
        for block in backbone.blocks:
            block.adaln_modulation[-1].weight.normal_(std=0.02)
            block.adaln_modulation[-1].bias[512:768].fill_(1)
        backbone.final_layer.linear.weight.normal_(std=0.02)
    action = torch.randn(2, 19, 16, requires_grad=True)
    obs = torch.randn(2, 1, 256, requires_grad=True)
    saved_obs = obs.detach().clone()
    seen = []

    def capture(module, args):
        seen.append(args[1].detach().clone())

    handles = [block.register_forward_pre_hook(capture) for block in backbone.blocks]
    velocity = backbone(action, torch.tensor([0.2, 0.7]), obs)
    assert velocity.shape == action.shape
    velocity[:, 0].square().mean().backward()
    assert obs.grad.abs().sum() > 0
    assert action.grad[:, 1:].abs().sum() > 0
    assert torch.isfinite(action.grad).all()
    assert len(seen) == 8
    for value in seen:
        torch.testing.assert_close(value, saved_obs, rtol=0, atol=0)
    torch.testing.assert_close(obs, saved_obs, rtol=0, atol=0)
    for handle in handles:
        handle.remove()
    attn = backbone.blocks[0].attn
    tokens = torch.randn(2, 20, 256)
    mask = tokens.new_zeros(20, 20)
    mask[0, 1:] = float("-inf")
    changed = tokens.clone()
    changed[:, 1:] = torch.randn_like(changed[:, 1:])
    torch.testing.assert_close(
        attn(tokens, mask)[:, :1], attn(changed, mask)[:, :1], rtol=0, atol=0
    )
    assert not torch.allclose(
        backbone(action.detach(), 0.1, obs.detach()), backbone(action.detach(), 0.9, obs.detach())
    )
