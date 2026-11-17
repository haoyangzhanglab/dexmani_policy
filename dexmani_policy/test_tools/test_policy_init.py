import copy
import importlib
from contextlib import closing

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.deployment import load_policy
from dexmani_policy.train import _prepare_policy, build_trainer
from dexmani_policy.training.config import load_config, validate_config
from dexmani_policy.training.resume import prepare_resume


class SmallRGBEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rgb_encoder = SmallRGBWeights()
        self.out_dim = 8
        self.n_obs_steps = kwargs["n_obs_steps"]

    def forward(self, obs):
        features = self.rgb_encoder(obs["joint_state"][:, :2]).repeat(1, 4)
        return features.unflatten(0, (-1, self.n_obs_steps))


class SmallRGBWeights(nn.Linear):
    def __init__(self):
        super().__init__(2, 2)

    def initialize_weights(self, source):
        if source is not None:
            self.load_state_dict(torch.load(source, weights_only=True))


@pytest.mark.parametrize("method", ["dqrise", "dexlatent"])
def test_hand_affine_and_latent_order(tmp_path, make_demo, monkeypatch, method):
    module = importlib.import_module(f"dexmani_policy.agents.policies.{method}")
    monkeypatch.setattr(module, "RGBPointObservationEncoder", SmallRGBEncoder)
    dataset = DemonstrationDataset(make_demo(), sensor_modalities=["joint_state"], horizon=4, obs_horizon=1)
    cfg = load_config(method, [
        "training.device=cpu", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.hidden_dim=16", "agent.n_layers=1", "agent.n_head=2",
    ])
    cfg.agent.action_dim = 19
    rgb = SmallRGBWeights()
    with torch.no_grad():
        rgb.weight.fill_(0.125)
    rgb_path = tmp_path / "rgb.pt"
    torch.save(rgb.state_dict(), rgb_path)
    cfg.initialization.rgb = str(rgb_path)
    if method == "dqrise":
        path = tmp_path / "codebook.npz"
        poses = np.linspace(-1, 1, 36, dtype=np.float32).reshape(3, 12)
        scale, offset = torch.full((12,), 2.0), torch.full((12,), -0.5)
        np.savez(path, sorted_hand_poses=poses, hand_normalizer_scale=scale.numpy(), hand_normalizer_offset=offset.numpy())
        cfg.initialization.codebook = str(path)
    else:
        from dexmani_policy.agents.hand_prior.dexlatent import DexLatentHandPrior

        prior = DexLatentHandPrior()
        path = tmp_path / "prior.pt"
        torch.save({"autoencoders": {"xarm7_xhand_right." + key: value for key, value in prior.state_dict().items()}}, path)
        cfg.initialization.dexlatent = str(path)
        affine = prior.initialize_weights(path)
        scale, offset = affine["scale"], affine["offset"]
    model = _prepare_policy(cfg, dataset, {"action": 19, "joint_state": 19})
    torch.testing.assert_close(model.obs_encoder.rgb_encoder.state_dict(), rgb.state_dict())
    torch.testing.assert_close(model.normalizer["action"].scale[7:], scale)
    torch.testing.assert_close(model.normalizer["action"].offset[7:], offset)
    if method == "dqrise":
        assert cfg.agent.num_codes == model.hand_codebook.num_codes == 3
        torch.testing.assert_close(model.hand_codebook.sorted_hand_poses, torch.from_numpy(poses))
    else:
        reference = LinearNormalizer()
        chunks = [torch.from_numpy(chunk) for chunk in dataset.iter_normalization_data("action")]
        latents = [prior.encode(chunk[:, 7:] * scale + offset) for chunk in chunks]
        reference.fit_field("hand_latent", latents)
        torch.testing.assert_close(model.normalizer["hand_latent"].state_dict(), reference["hand_latent"].state_dict())
        normalized = model._encode_actions(torch.cat(chunks))
        assert normalized.shape[-1] == 39
    before = copy.deepcopy(model.state_dict())
    before_config = copy.deepcopy(cfg.agent)
    sat_module = importlib.import_module("dexmani_policy.agents.policies.sat")
    monkeypatch.setattr(sat_module, "RGBPointObservationEncoder", SmallRGBEncoder)
    sat_cfg = load_config("sat", [
        "training.device=cpu", "agent.action_dim=19", "agent.hidden_dim=16",
        "agent.n_layers=1", "agent.n_head=2", "initialization.rgb=null",
    ])
    sat = _prepare_policy(sat_cfg, dataset, {"action": 19, "joint_state": 19})
    with torch.no_grad():
        sat.normalizer["action"].scale.fill_(9)
        sat.obs_encoder.rgb_encoder.weight.zero_()
    sat_cfg.agent.rgb_backbone_config.tune_mode = "full"
    torch.testing.assert_close(model.state_dict(), before)
    assert cfg.agent == before_config


@pytest.mark.parametrize("method", ["afro", "cordvip"])
@pytest.mark.parametrize("mode", ["limits", "identity"])
def test_encoder_normalization_inherited_before_configuration(tmp_path, make_demo, method, mode):
    import zarr

    path = make_demo()
    store = zarr.open_group(str(path), mode="a")
    for key in ("hand_pointcloud", "object_pointcloud"):
        store.create_dataset("data/" + key, data=store["data/point_cloud"][:], chunks=(4, 8, 6))
    fields = ["point_cloud"] if method == "afro" else ["hand_pointcloud", "object_pointcloud", "joint_state"]
    modalities = list(dict.fromkeys([*fields, "joint_state"]))
    dataset = DemonstrationDataset(path, sensor_modalities=modalities, pc_dim=3, horizon=4, obs_horizon=1)
    cfg = load_config(method, [
        "training.device=cpu", "horizon=4", "n_obs_steps=1", "n_action_steps=4", "agent.down_dims=[16,32]",
    ])
    cfg.agent.action_dim = 19
    reference = hydra.utils.instantiate(cfg.agent, _convert_="all")
    encoder = reference.obs_encoder.pc_encoder if method == "afro" else reference.obs_encoder.interaction_encoder
    dims = {"action": 19, **{key: 19 if key == "joint_state" else 3 for key in modalities}}
    inherited = LinearNormalizer({key: (dims[key],) for key in fields if mode == "limits"})
    for field in inherited.fields.values():
        field.scale.fill_(2.5)
        field.offset.fill_(0.75)
    source = tmp_path / "encoder.pt"
    torch.save({"encoder_state": encoder.state_dict(), "normalizer_state": inherited.state_dict(),
                "normalization": dict.fromkeys(fields, mode)}, source)
    cfg.initialization.encoder = str(source)
    model = _prepare_policy(cfg, dataset, dims)
    actual_encoder = model.obs_encoder.pc_encoder if method == "afro" else model.obs_encoder.interaction_encoder
    torch.testing.assert_close(actual_encoder.state_dict(), encoder.state_dict())
    for key in fields:
        assert cfg.normalization[key] == mode
        if mode == "limits":
            torch.testing.assert_close(model.normalizer[key].state_dict(), inherited[key].state_dict())
        else:
            assert key not in model.normalizer.fields
    expected = LinearNormalizer()
    expected.fit_field("action", dataset.iter_normalization_data("action"))
    torch.testing.assert_close(model.normalizer["action"].state_dict(), expected["action"].state_dict())
    if method == "afro":
        expected.fit_field("joint_state", dataset.iter_normalization_data("joint_state"))
        torch.testing.assert_close(model.normalizer["joint_state"].state_dict(), expected["joint_state"].state_dict())


def test_resume_and_warm_start_state_boundaries(tmp_path, small_dp3_config, monkeypatch):
    directory = tmp_path / "source"
    with closing(build_trainer(small_dp3_config(directory))) as trainer:
        trainer.train(max_updates=1)
    saved = load_checkpoint(directory / "checkpoints/latest.pt")
    for override in ("action_key=action_ee", "agent._target_=dexmani_policy.agents.policies.dp.DP"):
        invalid = small_dp3_config(tmp_path / "invalid", f"initialization.policy={directory / 'checkpoints/latest.pt'}", override)
        with pytest.raises(ValueError, match="same policy and action semantics"):
            build_trainer(invalid)

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint restoration must not fit normalization")

    monkeypatch.setattr(LinearNormalizer, "fit_field", forbidden)
    monkeypatch.setattr("dexmani_policy.train._build_fresh_policy", forbidden)
    monkeypatch.setattr("dexmani_policy.training.policy_init.rgb.resolve_rgb_architecture", forbidden)
    with closing(prepare_resume(directory)) as resumed:
        torch.testing.assert_close(resumed.model.state_dict(), saved.model_state)
        torch.testing.assert_close(resumed.ema_model.state_dict(), saved.ema_model_state)
        torch.testing.assert_close(resumed.optimizer.state_dict()["state"], saved.optimizer_state["state"])
        assert resumed.optimizer.state_dict()["param_groups"] == saved.optimizer_state["param_groups"]
        assert resumed.scheduler.state_dict() == saved.scheduler_state
        assert resumed.global_step == 1
        actual_parameters = {id(p) for p in resumed.model.parameters() if p.requires_grad}
        assert {id(p) for group in resumed.optimizer.param_groups for p in group["params"]} == actual_parameters
        assert {id(p) for p in resumed.ema_updater.parameters} == actual_parameters
        observation = {key: value.numpy() for key, value in resumed.train_loader.dataset[0]["obs"].items()}
        for weights, model in (("raw", resumed.model), ("ema", resumed.ema_model)):
            with closing(load_policy(directory, weights=weights, device="cpu", seed=7, inference_steps=2)) as runtime:
                model.eval()
                torch.testing.assert_close(runtime.agent.state_dict(), model.state_dict())
                assert runtime.info.control_action_dim == 19
                assert runtime.info.action_mode == "joint"
                assert runtime.info.data_processing == saved.effective_config["data_processing"]
                expected = model.predict_action(
                    {key: torch.from_numpy(value).unsqueeze(0) for key, value in observation.items()},
                    inference_steps=2, generator=torch.Generator().manual_seed(7),
                )["future_actions"][0].numpy()
                np.testing.assert_allclose(runtime.predict(observation), expected)
    warm_cfg = small_dp3_config(tmp_path / "warm", f"initialization.policy={directory / 'checkpoints/latest.pt'}")
    with closing(build_trainer(warm_cfg)) as warm:
        torch.testing.assert_close(warm.model.state_dict(), saved.model_state)
        assert warm.global_step == 0
        assert warm.scheduler.last_epoch == 0
        assert warm.optimizer.state_dict()["state"] == {}
        assert warm.logger.output_dir == tmp_path / "warm"
        torch.testing.assert_close(warm.ema_model.state_dict(), warm.model.state_dict())


@pytest.mark.parametrize("kind", ["dino", "siglip"])
def test_rgb_resolution_and_external_weights(tmp_path, monkeypatch, kind):
    from transformers import AutoConfig, AutoModel, Dinov2Config, SiglipVisionConfig, SiglipVisionModel
    from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder
    from dexmani_policy.training.policy_init.rgb import inherit_rgb_architecture, resolve_rgb_architecture

    config_class = Dinov2Config if kind == "dino" else SiglipVisionConfig
    architecture = config_class(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, image_size=8, patch_size=4)
    if kind == "siglip":
        architecture.vision_use_head = False
    source = AutoModel.from_config(architecture) if kind == "dino" else SiglipVisionModel(architecture)
    directory = tmp_path / "rgb"
    source.save_pretrained(directory)
    agent = OmegaConf.create({"rgb_backbone_name": kind, "rgb_backbone_config": {"tune_mode": "full"}})
    resolve_rgb_architecture(agent, str(directory))
    assert agent.rgb_backbone_config.architecture.hidden_size == 16
    encoder = ViTEncoder(kind, tune_mode="full", architecture=OmegaConf.to_container(agent.rgb_backbone_config.architecture))
    encoder.initialize_weights(str(directory))
    torch.testing.assert_close(encoder.backbone.state_dict(), source.state_dict())
    result = encoder(torch.randn(2, 3, 8, 8))
    result["patch_tokens"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.backbone.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("resolved or inherited architecture must not read a source")

    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(SiglipVisionConfig, "from_pretrained", forbidden)
    resolve_rgb_architecture(agent, "missing-source")
    warm = OmegaConf.create({"rgb_backbone_name": kind, "rgb_backbone_config": {}})
    inherit_rgb_architecture(warm, OmegaConf.to_container(agent))
    assert warm.rgb_backbone_config.architecture == agent.rgb_backbone_config.architecture
    warm.rgb_backbone_name = "resnet"
    with pytest.raises(ValueError, match="RGB backbone"):
        inherit_rgb_architecture(warm, OmegaConf.to_container(agent))


def test_method_owned_warm_checks():
    from dexmani_policy.training.policy_init.adaptac import prepare_warm_start as tactile_warm
    from dexmani_policy.training.policy_init.dqrise import prepare_warm_start as dq_warm

    agent = OmegaConf.create({"rgb_backbone_name": "dino", "rgb_backbone_config": {}})
    saved = {"rgb_backbone_name": "dino", "rgb_backbone_config": {"architecture": {"hidden_size": 16}}, "num_codes": 5}
    dq_warm(agent, saved)
    assert agent.num_codes == 5
    assert agent.rgb_backbone_config.architecture.hidden_size == 16
    agent.num_codes = 7
    dq_warm(agent, saved)
    assert agent.num_codes == 7
    tactile = OmegaConf.create({"tactile_encoder": {"_target_": "force"}})
    tactile_warm(tactile, {"tactile_encoder": {"_target_": "force"}})
    with pytest.raises(ValueError, match="Tactile encoder"):
        tactile_warm(tactile, {"tactile_encoder": {"_target_": "taxel"}})


@pytest.mark.parametrize("method,fields", [
    ("dp", {"rgb"}), ("sat", {"rgb", "point_cloud", "camera_intrinsic", "camera_extrinsic"}),
    ("dqrise", {"rgb", "point_cloud", "camera_intrinsic", "camera_extrinsic"}),
    ("dexlatent", {"rgb", "point_cloud", "camera_intrinsic", "camera_extrinsic"}),
    ("adaptac", {"point_cloud"}),
])
def test_identity_fields_owned_by_policy(method, fields):
    from dexmani_policy.training.config import validate_normalization

    cfg = load_config(method)
    cls = hydra.utils.get_class(cfg.agent._target_)
    for field in fields:
        with pytest.raises(ValueError, match="identity"):
            validate_normalization(cls, {field: "limits"}, {field: 3})
        validate_normalization(cls, {field: "identity"}, {field: 3})
        validate_normalization(cls, {field: "limits"}, {})


@pytest.mark.parametrize("method", [
    "dp", "dp3", "r3d", "maniflow", "sat", "dqrise", "dexlatent", "afro", "cordvip", "adaptac", "ode",
])
def test_recipe_composition(method):
    cfg = load_config(method)
    assert cfg.agent.action_dim is None
    assert cfg.agent.horizon == cfg.dataset.horizon == cfg.horizon
    assert cfg.dataset.obs_horizon == cfg.agent.n_obs_steps == cfg.n_obs_steps
    validate_config(cfg)
    hydra.utils.get_class(cfg.dataset._target_)
    if method == "r3d":
        assert cfg.dataset.use_aux_ee == cfg.agent.use_aux_ee == cfg.use_aux_ee


def test_fixed_ode_layout():
    from dexmani_policy.agents.policies.ode import ODEPolicy

    with pytest.raises(ValueError, match="19"):
        ODEPolicy(horizon=4, n_obs_steps=1, n_action_steps=4, action_dim=20)


@pytest.mark.parametrize("tactile", ["force", "taxel"])
def test_adaptac_tactile_recipe(tactile):
    cfg = load_config("adaptac", [f"tactile={tactile}"])
    validate_config(cfg)
    encoder = hydra.utils.instantiate(cfg.agent.tactile_encoder)
    assert encoder.input_key in cfg.dataset.sensor_modalities
    assert cfg.normalization[encoder.input_key] == "identity"
