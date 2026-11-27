import copy
import inspect
from contextlib import closing
from unittest.mock import Mock, patch

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, default_collate

from dexmani_policy.agents.action_decoders.backbones.unet1d import ConditionalUNet1D
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.obs_encoder import rgb_point
from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.deployment import load_policy, prepare_observations
from dexmani_policy.training import trainer as policy_training
from dexmani_policy.training.config import assemble_policy_config, load_config, load_policy_config, validate_config
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer


class TinyObservationEncoder(nn.Module):
    def __init__(self, state_dim, n_obs_steps):
        super().__init__()
        self.linear = nn.Linear(state_dim, 8)
        self.n_obs_steps = n_obs_steps

    def forward(self, obs):
        return self.linear(obs["joint_state"]).unflatten(0, (-1, self.n_obs_steps)).flatten(1)


class TinyPolicy(Policy):
    uses_ddim = True
    supports_rtc = True

    def __init__(self, horizon, n_obs_steps, n_action_steps, action_dim, state_dim=19,
                 num_training_steps=100, num_inference_steps=2, center_rotvec=False):
        encoder = TinyObservationEncoder(state_dim, n_obs_steps)
        backbone = ConditionalUNet1D(
            action_dim, 8 * n_obs_steps, 16, (16, 32), horizon=horizon,
        )
        decoder = Diffusion(backbone, num_training_steps, num_inference_steps)
        super().__init__(encoder, decoder, horizon, n_obs_steps, n_action_steps, action_dim,
                         center_rotvec=center_rotvec)


def tiny_policy_config(path, output, *overrides):
    cfg = load_config("common")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, {
        "policy_name": "tiny", "task_name": "test", "zarr_path": str(path),
        "horizon": 4, "n_obs_steps": 1, "n_action_steps": 4,
        "agent": {"_target_": "tests.helpers.TinyPolicy", "action_dim": None,
                  "num_training_steps": 100, "num_inference_steps": 2},
        "dataset": {"_target_": "dexmani_policy.datasets.demonstration.DemonstrationDataset",
                    "sensor_modalities": ["joint_state"], "augmentation_cfg": None},
        "normalization": {"joint_state": "limits"},
        "dataloader": {"batch_size": 2, "num_workers": 0, "drop_last": False},
        "optimizer": {"lr": 1e-4, "weight_decay": 1e-6, "betas": [0.95, 0.999]},
        "training": {"device": "cpu", "use_compile": False, "lr_warmup_steps": 0,
                     "loop": {"total_train_steps": 3, "log_interval_steps": 1}},
        "eval": {"inference_steps": "${agent.num_inference_steps}"},
        "workspace": {"output_dir": str(output), "wandb_cfg": None},
    }, OmegaConf.from_dotlist([value.lstrip("+") for value in overrides]))
    return assemble_policy_config(cfg)


class MeanDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def compute_loss(self, condition, target, *, log_metrics=True):
        loss = (self.weight - target).square().mean()
        return loss, {"loss": loss.detach()} if log_metrics else {}


def training_config(directory, batch_size, accumulation, updates):
    return OmegaConf.create({
        "training": {
            "device": "cpu", "seed": 42, "use_ema": True,
            "max_grad_norm": 0.1, "use_bfloat16": False,
            "lr_scheduler": "constant", "lr_warmup_steps": 0,
            "save_every_updates": updates,
            "loop": {"total_train_steps": updates, "log_interval_steps": 1,
                     "gradient_accumulation_steps": accumulation},
        },
        "optimizer": {"lr": 0.1, "weight_decay": 0},
        "ema": {"_target_": "dexmani_policy.training.optimization.EMA"},
        "dataloader": {"batch_size": batch_size, "shuffle": False,
                       "drop_last": False, "num_workers": 0},
        "workspace": {"output_dir": str(directory), "wandb_enabled": False},
        "dataset": {}, "normalization": {},
        "data_processing": None,
    })


class SmallRGBWeights(nn.Linear):
    out_dim = 2

    def __init__(self):
        super().__init__(2, 2)
        self.effective_config = {}

    def initialize_weights(self, source):
        self.load_state_dict(torch.load(source, weights_only=True))


def build_small_rgb(name, config):
    return SmallRGBWeights(), ImageProcessor.from_preset(name, config)


def tiny_hf_rgb_config(kind):
    from transformers import Dinov2Config, SiglipVisionConfig

    config_class = Dinov2Config if kind == "dino" else SiglipVisionConfig
    architecture = config_class(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, image_size=8, patch_size=4)
    if kind == "siglip":
        architecture.vision_use_head = False
    return architecture


def save_tiny_hf_rgb(directory, kind):
    from transformers import AutoModel, SiglipVisionModel

    architecture = tiny_hf_rgb_config(kind)
    source = AutoModel.from_config(architecture) if kind == "dino" else SiglipVisionModel(architecture)
    source.save_pretrained(directory)
    return source


def assert_loader_matches_items(dataset, monkeypatch):
    indices = [*reversed(range(len(dataset))), 0, 0]
    np.random.seed(17)
    expected = default_collate([dataset[index] for index in indices])
    sample_batch = Mock(wraps=dataset.sampler.sample_batch)
    monkeypatch.setattr(dataset.sampler, "sample_batch", sample_batch)
    np.random.seed(17)
    actual, = DataLoader(dataset, batch_sampler=[tuple(indices)])
    sample_batch.assert_called_once_with(indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dataset.__getitems__([]) == []
    first, second = dataset.__getitems__([0, 0])
    key = dataset.sensor_modalities[0]
    original = second["obs"][key].clone()
    first["obs"][key].fill_(-1)
    torch.testing.assert_close(second["obs"][key], original, rtol=0, atol=0)
    return actual


def check_recipe_composition(method):
    cfg = load_policy_config(method)
    assert cfg.agent.action_dim is None
    assert cfg.agent.horizon == cfg.dataset.horizon == cfg.horizon
    assert cfg.dataset.obs_horizon == cfg.agent.n_obs_steps == cfg.n_obs_steps
    validate_config(cfg)
    hydra.utils.get_class(cfg.dataset._target_)
    return cfg


def check_flow_recipe_defaults(method, cfg):
    assert cfg.training.use_bfloat16 is False
    assert cfg.agent.num_inference_steps == cfg.eval.inference_steps == 10
    assert cfg.agent.qk_norm is True
    defaults = inspect.signature(hydra.utils.get_class(cfg.agent._target_)).parameters
    assert defaults["num_inference_steps"].default == 10
    assert defaults["qk_norm"].default is True
    overridden = load_policy_config(method, ["agent.num_inference_steps=5", "training.use_bfloat16=true"])
    assert overridden.eval.inference_steps == 5
    assert overridden.training.use_bfloat16 is True


def check_observation_identity_constraints(method, fields):
    from dexmani_policy.common.normalizer import normalization_feature_shapes

    cfg = load_policy_config(method)
    cls = hydra.utils.get_class(cfg.agent._target_)
    options = {"required_identity_fields": cls.required_identity_fields}
    for field in fields:
        for mode in ("limits", "mean_std"):
            with pytest.raises(ValueError, match="identity"):
                normalization_feature_shapes({field: 3}, {field: mode}, **options)
        assert normalization_feature_shapes({field: 3}, {field: "identity"}, **options) == {}


def check_context_cache(monkeypatch, backbone, decoder_class, time_kwargs):
    with torch.no_grad():
        # Open zero-initialized gates/heads without saturating the time MLP.
        for module in backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    actions = torch.randn(2, 4, 19, requires_grad=True)
    cond = torch.randn(2, 3, 8, requires_grad=True)

    def gradients(cache):
        output = backbone(actions, 0.4, context=cond, context_kv=cache, **time_kwargs)
        return output, torch.autograd.grad(output.square().mean(), (actions, cond, *backbone.parameters()),
                                          allow_unused=True)

    torch.testing.assert_close(gradients(backbone.prepare_context_kv(cond)), gradients(None))
    decoder = decoder_class(backbone, num_inference_steps=3).eval()

    def sample(context):
        return decoder.predict_action(context, actions.shape, generator=torch.Generator().manual_seed(3))

    first = sample(cond)
    changed = cond + torch.randn_like(cond)
    second = sample(changed)
    assert (second - first).abs().max() > 1e-6
    with monkeypatch.context() as patch:
        patch.setattr(backbone, "prepare_context_kv", lambda context: None)
        torch.testing.assert_close(sample(changed), second)


def check_rgb_flow_lifecycle(
    tmp_path, make_demo, monkeypatch, method, options, tune_mode,
):
    from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder

    directory, source = tmp_path / "fresh", tmp_path / "rgb"
    pretrained = save_tiny_hf_rgb(source, "dino")
    path = make_demo(auxiliary=False)
    data = zarr.open_group(str(path), mode="r+")["data"]
    n_frames = len(data["action"])
    data["rgb"] = np.full((n_frames, 8, 8, 3), 128, dtype=np.uint8)
    data["camera_intrinsic"] = np.tile(np.eye(3, dtype=np.float32).ravel(), (n_frames, 1))
    data["camera_extrinsic"] = np.tile(np.eye(4, dtype=np.float32)[:3].ravel(), (n_frames, 1))
    cfg = load_policy_config(method, [
        f"zarr_path='{path}'", "task_name=test", f"workspace.output_dir='{directory}'",
        "workspace.wandb_cfg=null", "training.device=cpu", "training.use_compile=false",
        "dataloader.num_workers=0", "dataloader.batch_size=2",
        "horizon=4", "n_action_steps=3", f"initialization.rgb='{source}'",
        *options,
        f"agent.rgb_backbone_config.tune_mode={tune_mode}",
        "agent.rgb_backbone_config.image_size=[8,8]", "agent.num_inference_steps=2",
        "agent.pc_encoder_config.num_patches=4", "agent.pc_encoder_config.group_size=4",
        "agent.n_layers=1", "agent.hidden_dim=16", "agent.n_head=2",
    ])

    with (
        patch.object(ViTEncoder, "initialize_weights", autospec=True,
                     side_effect=ViTEncoder.initialize_weights) as initialize_rgb,
        closing(build_trainer(cfg)) as fresh,
    ):
        assert isinstance(fresh.model.obs_encoder, rgb_point.RGBPointObservationEncoder)
        assert "joint_state" in fresh.model.normalizer.fields
        initialize_rgb.assert_called_once_with(fresh.model.obs_encoder.rgb_encoder, str(source))
        for model in (fresh.model, fresh.ema_model):
            backbone = model.obs_encoder.rgb_encoder.backbone
            if tune_mode == "lora":
                state = backbone.get_base_model().state_dict()
                state = {key.replace(".base_layer.", "."): value for key, value in state.items() if ".lora_" not in key}
            else:
                state = backbone.state_dict()
            torch.testing.assert_close(state, pretrained.state_dict())
        prior, = fresh.cfg.prior_data
        assert prior == {"source": None, "artifact": str(source), "budget_verified": False}
        assert fresh.cfg.agent.rgb_backbone_config.architecture.hidden_size == 16
        capture_context = Mock(return_value=None)
        with fresh.model.action_decoder.model.context_embedder.register_forward_pre_hook(capture_context):
            fresh.train(max_updates=1)
        assert capture_context.call_args.args[1][0].shape == (2, 10, 192)

        def forbidden(*args, **kwargs):
            pytest.fail("Checkpoint restoration must use the saved RGB architecture")

        monkeypatch.setattr(policy_training.initialization, "resolve_rgb_architecture", forbidden)
        latest = directory / "checkpoints/latest.pt"
        saved = load_checkpoint(latest)
        with closing(load_policy(latest, weights="raw", device="cpu")) as runtime:
            if tune_mode == "lora":
                deployed = copy.deepcopy(fresh.model)
                rgb = deployed.obs_encoder.rgb_encoder
                rgb.backbone = rgb.backbone.merge_and_unload()
                torch.testing.assert_close(runtime.agent.state_dict(), deployed.state_dict())
            else:
                torch.testing.assert_close(runtime.agent.state_dict(), saved.model_state)
            assert runtime.info.observation_fields == tuple(cfg.dataset.sensor_modalities)
            obs = {key: data[key][:cfg.n_obs_steps] for key in cfg.dataset.sensor_modalities}
            with runtime.agent.action_decoder.model.context_embedder.register_forward_pre_hook(capture_context):
                actions = runtime.predict(obs)
            assert capture_context.call_args.args[1][0].shape == (1, 10, 192)
            assert actions.shape == (3, 19) and np.isfinite(actions).all()
            batched = prepare_observations(
                obs, normalization_dims=runtime.agent.normalization_dims, n_obs_steps=cfg.n_obs_steps,
                rgb_preprocessing=runtime.info.rgb_preprocessing, device="cpu",
            )
            with pytest.raises(ValueError, match=r"requires \[B,T\]"):
                runtime.agent.predict_action({key: tensor.transpose(0, 1) for key, tensor in batched.items()})
        warm_cfg = copy.deepcopy(cfg)
        warm_cfg.initialization.policy = str(latest)
        warm_cfg.initialization.rgb = None
        warm_cfg.workspace.output_dir = str(tmp_path / "warm")
        with closing(build_trainer(warm_cfg)) as warm:
            saved_rgb = saved.effective_config["agent"]["rgb_backbone_config"]
            assert warm.cfg.agent.rgb_backbone_name == "dino"
            assert warm.cfg.agent.rgb_backbone_config.architecture == saved_rgb["architecture"]
            torch.testing.assert_close(warm.model.state_dict(), saved.model_state)
            torch.testing.assert_close(warm.ema_model.state_dict(), saved.model_state)
        with closing(prepare_resume(directory)) as resumed:
            torch.testing.assert_close(resumed.model.state_dict(), saved.model_state)
            torch.testing.assert_close(resumed.ema_model.state_dict(), saved.ema_model_state)
            assert resumed.global_step == saved.global_step
            resumed.train(max_updates=1)
            assert resumed.global_step == saved.global_step + 1
        assert initialize_rgb.call_count == 1
    return fresh.model, cfg
