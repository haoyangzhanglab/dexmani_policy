import copy
import json
import math
import random
import shutil
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, default_collate

from dexmani_policy.agents.action_decoders.backbones import ditx, sat_action_io
from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.diffusion import Diffusion, ForceDiffusion
from dexmani_policy.agents.action_decoders.flow import ConsistencyFlow, RectifiedFlow
from dexmani_policy.agents.action_decoders.ode import ODE
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, load_codebook
from dexmani_policy.agents.obs_encoder import rgb_point
from dexmani_policy.agents.obs_encoder.interaction import (
    FingerTactileEncoder,
    HandObjectInteractionEncoder,
    HandKinematicEncoder,
    GeoAwareRelationAttention,
    TactileFusion,
)
from dexmani_policy.agents.obs_encoder.pointcloud import point_transformer
from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor
from dexmani_policy.agents.obs_encoder.rgb.resnet import convert_group_norm
from dexmani_policy.agents.policies import dp, r3d
from dexmani_policy.agents.policies.adaptac import AdapTac
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.common import config as common_config, io_util
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from dexmani_policy.common.inference import predict_actions, prepare_observations, resolve_inference_settings, restore_policy
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state, set_seed
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain.afro import AFROPretrainDataset
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.datasets.preprocess.contact_map import batched_contact_maps, compute_contact_map
from dexmani_policy.datasets.preprocess.contact_map import generate as generate_contacts
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.datasets.variants.ode import ODEDataset
from dexmani_policy.datasets.variants.r3d import R3DDataset
from dexmani_policy.datasets.variants.tactile import AdapTacDataset
from dexmani_policy.deployment import load_policy
from dexmani_policy.env_runner.dexjoco_runner import DexJoCoRunner
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation import rollout, selection
from dexmani_policy.training import resume, trainer as policy_training
from dexmani_policy.training.config import load_config, validate_config
from dexmani_policy.training.data import (
    build_dataset, build_train_loader, fit_action_normalizer,
    record_prior_data, record_training_data, restore_data_records, use_policy_split,
)
from dexmani_policy.training.initialization import build_policy
from dexmani_policy.training.optimization import clip_gradients
from dexmani_policy.training.pretrain.hand_prior.vqvae import train as train_vq
from dexmani_policy.training.pretrain.obs import common as obs_training
from dexmani_policy.training.pretrain.obs.afro import AFRO
from dexmani_policy.training.pretrain.obs.cordvip import CordViP
from dexmani_policy.training.resume import prepare_resume
from dexmani_policy.training.trainer import build_trainer, run_trainer


# Shared fixtures


@pytest.fixture(scope="session", autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def make_demo(tmp_path):
    def create(name="demo", *, auxiliary=True, action_dim=19, episode_ends=(12,)):
        path = tmp_path / f"{name}.zarr"
        store = zarr.open_group(str(path), mode="w")
        store.attrs.update(task_name="test", action_space="joint", dt=0.05)
        rng = np.random.default_rng(1)
        fields = {
            "action": rng.normal(size=(12, action_dim)).astype(np.float32),
            "joint_state": rng.normal(size=(12, 19)).astype(np.float32),
            "point_cloud": rng.normal(size=(12, 8, 6)).astype(np.float32),
        }
        if auxiliary:
            fields["action_ee"] = rng.normal(size=(12, 21)).astype(np.float32)
        for key, value in fields.items():
            store.create_dataset(f"data/{key}", data=value, chunks=(4, *value.shape[1:]))
        store.create_dataset("meta/episode_ends", data=np.array(episode_ends, dtype=np.int64))
        return path
    return create


@pytest.fixture
def small_dp3_config(make_demo):
    path = make_demo()

    def create(output, *overrides):
        return load_config("dp3", [
            f"zarr_path={path}", "task_name=test", f"workspace.output_dir='{output}'",
            "workspace.wandb_cfg=null", "training.device=cpu", "training.use_compile=false",
            "training.loop.total_train_steps=3", "training.lr_warmup_steps=0",
            "training.loop.log_interval_steps=1", "dataloader.num_workers=0",
            "dataloader.batch_size=2", "dataloader.drop_last=false",
            "dataset.augmentation_cfg=null", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
            "agent.down_dims=[16,32]", "agent.num_points=8", *overrides,
        ])
    return create


@pytest.fixture(autouse=True)
def preserve_process_state():
    rng = get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    precision = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        torch.set_float32_matmul_precision(precision)
        set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@pytest.fixture
def small_dp3_checkpoint(tmp_path, small_dp3_config):
    directory = tmp_path / "dp3_run"
    with closing(build_trainer(small_dp3_config(directory))) as trainer:
        trainer.train(max_updates=1)
    return directory / "checkpoints/latest.pt"


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf])
def test_inherited_hand_affine_validates_full_zarr_action(make_demo, bad):
    path = make_demo()
    actions = zarr.open_group(str(path), mode="r+")["data/action"]
    if bad is not None:
        actions[6, 8] = bad
    dataset = DemonstrationDataset(path, sensor_modalities=["joint_state"], obs_horizon=1)
    normalizer = LinearNormalizer({"action": (19,)})
    hand = {"scale": torch.linspace(0.5, 1.5, 12), "offset": torch.linspace(-0.2, 0.3, 12)}
    if bad is not None:
        with pytest.raises(ValueError, match=r"field=action, episode=0, source=0, row=6, channel=\(8,\)") as error:
            fit_action_normalizer(dataset, normalizer, hand_affine=hand)
        assert str(path) in str(error.value)
        return
    fit_action_normalizer(dataset, normalizer, hand_affine=hand)
    field = normalizer["action"]
    torch.testing.assert_close(field.scale[7:], hand["scale"])
    torch.testing.assert_close(field.offset[7:], hand["offset"])
    raw = torch.from_numpy(actions[:])
    coded = field.normalize(raw)
    prior = HandStateCodebook(hand_dim=12, num_codes=len(raw))
    prior.sorted_hand_poses.copy_(raw[:, 7:] * hand["scale"] + hand["offset"])
    reconstructed = torch.cat((coded[:, :7], prior.decode(prior.encode(coded[:, 7:]).squeeze(-1))), -1)
    torch.testing.assert_close(field.unnormalize(reconstructed), raw)


@pytest.mark.parametrize("mode", ["force", "taxel"])
def test_adaptac_padded_force_invariance(monkeypatch, mode):
    # CPU reference grouping keeps this loss test independent of the native point-cloud extension.
    def group(points, num_patches, group_size, *args):
        centers = points[:, :num_patches, :3]
        return centers, torch.cdist(centers, points[..., :3]).topk(group_size, largest=False).indices

    monkeypatch.setattr(point_transformer, "sample_point_patches", group)
    cfg = load_config("adaptac", [
        f"tactile={mode}", "agent.action_dim=19", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.feature_dim=32", "agent.state_out_dim=16", "agent.num_points=16",
        "agent.pc_encoder_config.channels=48", "agent.pc_encoder_config.num_patches=4",
        "agent.pc_encoder_config.group_size=4", "agent.pc_encoder_config.num_heads=4",
        "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=32",
        "agent.force_predictor_config.depth=1", "agent.force_predictor_config.embed_dim=32",
    ])
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    shape = common_config.TACTILE_FEATURE_SHAPES[model.force_key]
    model.configure_normalizer(
        {"action": 19, "point_cloud": 6, "joint_state": 19, model.force_key: math.prod(shape)},
        {"action": (19,), "joint_state": (19,)},
    )
    batch = {"obs": {"point_cloud": torch.randn(2, 1, 16, 6), "joint_state": torch.randn(2, 1, 19),
                     model.force_key: torch.randn(2, 1, *shape)}, "action": torch.randn(2, 4, 19)}
    force = torch.randn(2, 4, *shape)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    conditions = []
    action_loss = model.action_decoder.compute_loss

    def capture(condition, *args, **kwargs):
        conditions.append(condition.detach())
        return action_loss(condition, *args, **kwargs)

    monkeypatch.setattr(model.action_decoder, "compute_loss", capture)

    def run(target, mask):
        torch.manual_seed(11)
        loss, metrics = model.compute_loss(dict(batch, future_force=target, future_force_valid=mask))
        return conditions[-1], metrics, torch.autograd.grad(loss, params, allow_unused=True)

    expected = run(force, valid)
    for invalid in (0.0, -1e9, float("nan"), float("inf"), -float("inf")):
        target = force.clone()
        target[~valid] = invalid
        torch.testing.assert_close(run(target, valid), expected, rtol=0, atol=0)

    model.force_loss_weight = 0
    _, metrics, grads = run(force, valid)
    assert metrics["loss_force_weighted"] == 0
    torch.testing.assert_close(metrics["loss"], metrics["loss_action"])
    force_parameters = {id(parameter) for parameter in model.force_predictor.parameters()}
    assert any(grad is not None and grad.abs().max() > 0
               for parameter, grad in zip(params, grads) if id(parameter) in force_parameters)

    model.force_loss_weight = 1
    all_valid = torch.ones_like(valid)
    actual = run(force, all_valid)

    def original(condition, target, mask):
        prediction, target = model.force_predictor._training_prediction(condition, target.float())
        error = (prediction.float() - target).square() * mask[..., None]
        return error.sum() / (target.shape[-1] * mask.float().sum()), prediction.float()

    monkeypatch.setattr(model.force_predictor, "compute_loss_and_prediction", original)
    torch.testing.assert_close(run(force, all_valid), actual, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("steps", [10, 20])
def test_ddim_actual_time_grid(steps):
    visited = []

    class Denoiser(nn.Module):
        def forward(self, x, timestep, context):
            visited.append(timestep.item())
            return x * 0.5

    cond, shape = torch.zeros(1, 1, 2), (1, 4, 2)
    decoder = Diffusion(Denoiser(), num_inference_steps=steps)
    result = decoder.predict_action(cond, shape, generator=torch.Generator().manual_seed(9))
    stride = 100 // steps
    assert visited == list(range(100 - stride, -1, -stride))
    assert result.isfinite().all()
    with pytest.raises(ValueError, match="t=0"):
        Diffusion(Denoiser(), num_inference_steps=1)
    with pytest.raises(ValueError, match="t=0"):
        decoder.predict_action(cond, shape, inference_steps=1)
    with pytest.raises(ValueError, match="t=0"):
        decoder.predict_action(cond, shape, inference_steps=1, rtc_prefix=torch.zeros(1, 2, 2), rtc_guidance_cap=1)


def test_ddim_config_and_checkpoint_overrides(small_dp3_checkpoint):
    for recipe, overrides in (
        ("dp3", ["agent.num_inference_steps=1", "agent.state_dim=19"]),
        ("adaptac", ["agent.force_num_inference_steps=1"]),
    ):
        cfg = load_config(recipe, ["agent.action_dim=19", *overrides])
        with pytest.raises(hydra.errors.InstantiationException) as error:
            hydra.utils.instantiate(cfg.agent)
        assert isinstance(error.value.__cause__, ValueError)
        assert "leading DDIM" in str(error.value.__cause__)
    cfg = load_config("dp3", ["eval.inference_steps=1"])
    with pytest.raises(ValueError, match="leading DDIM"):
        resolve_inference_settings(cfg)
    with pytest.raises(ValueError, match="leading DDIM"):
        load_policy(small_dp3_checkpoint, inference_steps=1, device="cpu")
    with pytest.raises(ValueError, match="leading DDIM"):
        rollout.evaluate_policy(small_dp3_checkpoint, inference_steps=1, device="cpu")
    for name in ("ode", "maniflow"):
        cfg = load_config(name, ["eval.inference_steps=1"])
        assert resolve_inference_settings(cfg)[1] == 1


def test_ddim_rtc_matches_two_step_reference():
    class Denoiser(nn.Module):
        def forward(self, x, timestep, context):
            return (0.3 + 0.001 * timestep) * x

    decoder = Diffusion(Denoiser(), num_inference_steps=2, clip_sample=False)
    prefix, shape = torch.full((1, 1, 1), -0.4), (1, 2, 1)
    with torch.inference_mode():
        actual = decoder.predict_action(
            torch.zeros(1, 1, 1), shape, generator=torch.Generator().manual_seed(18),
            rtc_prefix=prefix, history_offset=1, rtc_guidance_cap=0.8,
        )
    reference = torch.randn(shape, generator=torch.Generator().manual_seed(18))
    alpha_bar = decoder.noise_scheduler.alphas_cumprod
    soft_weight = 0.5 * math.expm1(0.5) / math.expm1(1)
    for current, previous in [(50, 0), (0, -50)]:
        a, a_prev = float(alpha_bar[current]), float(alpha_bar[previous]) if previous >= 0 else 1.0
        scale = 0.3 + 0.001 * current
        clean = scale * reference
        residual = prefix - clean[:, 1:]
        predicted_noise = (reference - math.sqrt(a) * clean) / math.sqrt(1 - a)
        reference = math.sqrt(a_prev) * clean + math.sqrt(1 - a_prev) * predicted_noise
        strength = (math.sqrt(a_prev * (1 - a)) - math.sqrt(a * (1 - a_prev))) * min(
            0.8, 1 / math.sqrt(a * (1 - a)),
        )
        reference[:, 1:] += scale * soft_weight * strength * residual
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("action_dim", [22, 44])
def test_centered_rotvec_action_coordinates(action_dim):
    actions = torch.linspace(0, 1, 3 * action_dim).view(3, action_dim)
    arms = actions.unflatten(-1, (action_dim // 22, 22))
    arms[..., 3] = torch.tensor([math.pi - 0.1, 0.3 - math.pi, 0.15 - math.pi])[:, None]
    arms[..., 4:6] = 0
    expected = torch.tensor([-1.0, 1.0, 0.25])[:, None].expand(-1, action_dim // 22)

    class Decoder(nn.Module):
        def compute_loss(self, condition, target, **kwargs):
            torch.testing.assert_close(target[0].unflatten(-1, (-1, 22))[..., 3], expected)
            return target.sum(), {}

        def predict_action(self, condition, action_shape, *, rtc_prefix, **kwargs):
            assert rtc_prefix.shape == (1, 3, action_dim)
            torch.testing.assert_close(rtc_prefix[0].unflatten(-1, (-1, 22))[..., 3], expected)
            return rtc_prefix

    dataset = SimpleNamespace(
        action_key="action_rotvec", iter_normalization_data=lambda key: iter((actions[:1], actions[1:])),
    )
    policy = Policy(nn.Identity(), Decoder(), 3, 1, 3, action_dim, center_rotvec=True)
    policy.supports_rtc = True
    fit_action_normalizer(dataset, policy.normalizer, center_rotvec=True)
    policy.compute_loss({"obs": {}, "action": actions[None]})
    output = policy._predict_from_condition(torch.zeros(1, 1), rtc_prefix=actions[None])
    torch.testing.assert_close(output["future_actions"][0], actions)


@pytest.mark.parametrize("pc_dim", [3, 6])
def test_r3d_limits_config_and_restoration(tmp_path, make_demo, monkeypatch, pc_dim):
    class SmallUni3D(nn.Module):
        def __init__(self, embed_dim, **kwargs):
            super().__init__()
            self.out_dim = embed_dim
            self.project = nn.Linear(3, embed_dim)

        def initialize_weights(self, source):
            assert source is None

        def forward(self, cloud):
            projected = self.project(cloud[..., :3])
            return projected, torch.zeros_like(projected)

    monkeypatch.setattr(r3d, "Uni3DEncoder", SmallUni3D)
    data = make_demo()
    directory = tmp_path / "r3d"
    cfg = load_config("r3d", [
        f"zarr_path={data}", "task_name=test", f"workspace.output_dir={directory}",
        "workspace.wandb_cfg=null", "initialization.uni3d=null", "dataset.augmentation_cfg=null",
        f"+dataset.pc_dim={pc_dim}",
        "training.device=cpu", "training.use_compile=false", "training.loop.total_train_steps=3",
        "dataloader.num_workers=0", "dataloader.batch_size=2", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.pc_encoder_config.embed_dim=16", "agent.embedding_dim=16", "agent.num_heads=2", "agent.depth=1",
        "agent.state_out_dim=8", "agent.mlp_dim=32", "agent.timestep_embed_dim=8",
    ])
    unsupported = copy.deepcopy(cfg)
    unsupported.normalization.point_cloud = "mean_std"
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        build_trainer(unsupported)
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    checkpoint = load_checkpoint(latest)
    assert checkpoint.effective_config["normalization"]["point_cloud"] == "limits"

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not rescan training arrays")

    with monkeypatch.context() as patch:
        patch.setattr(EpisodeStore, "iter_chunks", forbidden)
        with closing(prepare_resume(directory)) as trainer:
            assert trainer.global_step == 1
            torch.testing.assert_close(trainer.model.normalizer["point_cloud"].scale,
                                       checkpoint.model_state["normalizer.fields.point_cloud.scale"])
    cfg.workspace.output_dir = str(tmp_path / "warm")
    cfg.initialization.policy = str(latest)
    warm_data = make_demo("warm")
    cfg.zarr_path = cfg.dataset.zarr_path = str(warm_data)
    cloud = zarr.open_group(str(warm_data), mode="r+")["data/point_cloud"]
    if pc_dim == 3:
        cloud[6, 2, 5] = np.inf
    with closing(build_trainer(cfg)) as trainer:
        assert trainer.global_step == 0
    for weights in ("raw", "ema"):
        assert load_policy(latest, weights=weights, device="cpu").agent.normalization_dims["point_cloud"] == pc_dim
    cfg.workspace.output_dir = str(tmp_path / "warm_bad")
    column = pc_dim - 1
    value = cloud[6, 2, column]
    cloud[6, 2, column] = np.inf
    with pytest.raises(ValueError, match=rf"field=point_cloud,.*row=6, channel=\(2, {column}\)"):
        build_trainer(cfg)
    assert load_checkpoint(latest).global_step == 1
    assert not Path(cfg.workspace.output_dir).exists()
    cloud[6, 2, column] = value
    checkpoint.effective_config["normalization"]["point_cloud"] = "mean_std"
    bad = save_checkpoint(directory / "checkpoints/bad.pt", checkpoint)
    cfg.initialization.policy = str(bad)
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        build_trainer(cfg)
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        build_trainer(OmegaConf.create(checkpoint.effective_config), checkpoint=checkpoint)
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        load_policy(bad, device="cpu")


def test_bn_to_gn_copies_learned_affines():
    batch_norm = nn.BatchNorm2d(4).double()
    with torch.no_grad():
        batch_norm.weight.copy_(torch.tensor([2., 0.5, 1.5, -1.]))
        batch_norm.bias.fill_(0.4)
        batch_norm.running_mean.fill_(5)
    model = convert_group_norm(nn.Sequential(batch_norm))
    assert isinstance(model[0], nn.GroupNorm)
    torch.testing.assert_close(model[0].weight, batch_norm.weight)
    torch.testing.assert_close(model[0].bias, batch_norm.bias)
    assert set(model[0].state_dict()) == {"weight", "bias"}


def test_ode_reads_real_held_joint_targets_from_action(make_demo):
    path = make_demo(episode_ends=(4, 12))
    root = zarr.open_group(str(path), mode="a")
    root.attrs.update(
        format="dexmani.real.canonical", action_semantics="held_accepted_joint_target",
        source_episodes=["capture", "capture"], source_row_ranges=[[0, 4], [4, 12]],
    )
    action = root["data/action"]
    joint_state = root["data/joint_state"]
    action[1] = action[0]
    dataset = ODEDataset(path, sensor_modalities=["joint_state"], obs_horizon=1)

    np.testing.assert_array_equal(dataset[0]["prev_action"], joint_state[0])
    np.testing.assert_array_equal(dataset[1]["action"][0], action[0])
    np.testing.assert_array_equal(dataset[2]["prev_action"], action[1])
    np.testing.assert_array_equal(dataset[4]["prev_action"], joint_state[4])


def test_afro_default_frames_use_only_complete_windows(segments):
    path, fields = segments
    cfg = load_config("pretrain/obs/afro")
    assert (cfg.dataset.pad_before, cfg.dataset.pad_after, cfg.dataset.frame_stride) == (0, 0, 2)
    dataset = hydra.utils.instantiate(cfg.dataset, zarr_path=str(path), horizon=5)
    assert len(dataset) == 1
    np.testing.assert_array_equal(dataset[0]["obs"]["point_cloud"], fields["point_cloud"][[3, 5, 7], :, :3])
    with pytest.raises(ValueError, match="No AFROPretrainDataset training windows"):
        AFROPretrainDataset(path, frame_stride=2, horizon=6)
    loader_cfg = OmegaConf.create({"dataloader": {"num_workers": 0, "batch_size": 2, "drop_last": True}})
    with pytest.raises(ValueError, match="Training DataLoader is empty"):
        build_train_loader(loader_cfg, dataset)


def test_same_source_pretraining_carries_real_group_budget(tmp_path, make_demo):
    path = make_demo()
    data = zarr.open_group(str(path), mode="a")
    data["meta/episode_ends"].resize(6)
    data["meta/episode_ends"][:] = [2, 4, 6, 8, 10, 12]
    sources = ["a", "a", "b", "c", "d", "e"]
    data.attrs.update(format="dexmani.real.canonical", source_episodes=sources)
    policy = load_config("dqrise", [
        f"zarr_path={path}", "task_name=test", "dataset.seed=9",
        "dataset.val_episodes=1", "dataset.max_train_episodes=2",
        f"workspace.output_dir={tmp_path / 'policy'}",
    ])
    policy_path = tmp_path / "policy.yaml"
    # Hydra --cfg job prints the recipe before runtime assembly fills dataset.zarr_path.
    policy_recipe = OmegaConf.to_container(policy, resolve=True)
    del policy_recipe["dataset"]["zarr_path"]
    OmegaConf.save(OmegaConf.create(policy_recipe), policy_path)
    kwargs = {"seed": 9, "val_episodes": 1, "max_train_episodes": 2}
    dataset = DemonstrationDataset(path, sensor_modalities=["joint_state"], obs_horizon=1, **kwargs)
    chosen = set(np.asarray(sources)[dataset.train_mask])
    held_out = set(np.asarray(sources)[dataset.val_mask])
    assert len(chosen) == 2 and len(held_out) == 1 and chosen.isdisjoint(held_out)
    assert dataset.train_mask[0] == dataset.train_mask[1]

    run = tmp_path / "vq"
    cfg = load_config("pretrain/hand_prior/vqvae", [
        f"workspace.output_dir={run}", "vq_vae.device=cpu", "vq_vae.batch_size=2",
        "vq_vae.codebook_size=2", "vq_vae.num_groups=2", "vq_vae.num_epochs=1",
    ])
    use_policy_split(cfg, policy_path)
    train_vq(cfg)
    _, affine, record = load_codebook(run / "codebook.npz")
    assert record["source"] == str(path)
    assert set(record["train_source_episodes"]) == chosen
    assert set(record["val_source_episodes"]) == held_out
    rows = np.repeat(dataset.train_mask, 2)
    hand = torch.from_numpy(data["data/action"][:][rows, 7:])
    torch.testing.assert_close(affine["scale"], 2 / (hand.amax(0) - hand.amin(0)))
    record_prior_data(policy, dataset, "codebook.npz", record)
    assert policy.prior_data[0].budget_verified
    assert "training_data" not in policy
    outside_budget = set(sources) - chosen - held_out
    for group, key in [(next(iter(outside_budget)), "train_source_episodes"),
                       (next(iter(held_out)), "train_source_episodes"),
                       (next(iter(outside_budget)), "val_source_episodes")]:
        with pytest.raises(ValueError, match="same-source prior exceeds"):
            record_prior_data(policy, dataset, "leaky.pt", dict(record, **{key: [group]}))
    copied = tmp_path / "copy" / path.name
    shutil.copytree(path, copied)
    record_prior_data(policy, dataset, "external.pt", dict(record, source=str(copied)))
    record_prior_data(policy, dataset, "unknown.pt", {"source": None})
    assert not policy.prior_data[-2].budget_verified
    assert policy.prior_data[-1] == {"source": None, "artifact": "unknown.pt", "budget_verified": False}


@pytest.mark.parametrize("changed", ["episode_ends", "episode_groups", "episode_seeds", "source_row_ranges"])
def test_pretraining_checks_episode_layout(tmp_path, make_demo, changed):
    source = make_demo(episode_ends=(4, 8, 12))
    raw = zarr.open_group(str(source), mode="a")
    if changed in ("episode_groups", "source_row_ranges"):
        raw.attrs.update(
            format="dexmani.real.canonical", source_episodes=["a", "b", "c"],
            source_row_ranges=[[0, 4], [0, 4], [0, 4]],
        )
    else:
        raw.create_dataset("meta/episode_seeds", data=[32, 33, 34])
    derived = tmp_path / "derived.zarr"
    shutil.copytree(source, derived)
    pretraining = zarr.open_group(str(derived), mode="a")
    pretraining.attrs["source_zarr_path"] = str(source)

    train = np.ones(3, dtype=bool)

    def record_source():
        cfg = OmegaConf.create({"policy_zarr_path": str(source), "task_name": "test", "dataset": {}})
        record_training_data(cfg, EpisodeStore(derived, keys=()), train)
        return cfg

    recorded = record_source()
    saved = OmegaConf.to_container(recorded, resolve=True)
    if changed == "episode_ends":
        pretraining["meta/episode_ends"][:] = [3, 8, 12]
    elif changed == "episode_groups":
        pretraining.attrs["source_episodes"] = ["b", "a", "c"]
    elif changed == "episode_seeds":
        pretraining["meta/episode_seeds"][:] = [33, 32, 34]
    else:
        pretraining.attrs["source_row_ranges"] = [[1, 5], [0, 4], [0, 4]]
    with pytest.raises(ValueError, match="share the policy episode layout"):
        record_source()
    with pytest.raises(ValueError, match="Resume episode layout mismatch"):
        restore_data_records(recorded, EpisodeStore(derived, keys=()), train, saved=saved)


class ConstantVelocity(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(value))

    def forward(self, x, timestep, context, **kwargs):
        self.last_input = x.detach()
        self.last_time = timestep
        self.last_target_time = kwargs.get("target_t")
        return x * 0.0 + self.velocity

    def prepare_context_kv(self, context):
        return None


def test_flow_direction_and_rtc_against_reference():
    cond, action = torch.zeros(1, 1, 2), torch.full((1, 2, 1), 0.7)
    decoder = RectifiedFlow(ConstantVelocity(0.25)).eval()
    torch.manual_seed(31)
    noise, t = torch.randn_like(action), torch.rand(1)
    torch.manual_seed(31)
    loss, _ = decoder.compute_loss(cond, action)
    torch.testing.assert_close(decoder.model.last_input, torch.lerp(noise, action, t))
    torch.testing.assert_close(loss, (0.25 - (action - noise)).square().mean())

    def generator():
        return torch.Generator().manual_seed(31)

    plain = decoder.predict_action(cond, action.shape, inference_steps=1, generator=generator())
    torch.testing.assert_close(plain, noise + 0.25)
    prefix = torch.full((1, 1, 1), 0.9)
    guided = decoder.predict_action(
        cond, action.shape, inference_steps=2, generator=generator(),
        rtc_prefix=prefix, history_offset=1, rtc_guidance_cap=0.2,
    )
    expected = noise.clone()
    weight = 0.5 * math.expm1(0.5) / math.expm1(1)
    for t in (0., 0.5):
        residual = prefix - (expected[:, 1:] + (1 - t) * 0.25)
        expected += 0.125
        expected[:, 1:] += 0.1 * weight * residual
    torch.testing.assert_close(guided, expected)


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_consistency_teacher_target_matches_reference(monkeypatch, mode):
    decoder = ConsistencyFlow(ConstantVelocity(0.2), target_t_sample_mode=mode)
    teacher = ConstantVelocity(-0.4)
    t, dt = torch.tensor([0.2, 0.9]), torch.tensor([0.4, 0.3])
    times = iter((t, dt))
    monkeypatch.setattr(decoder.time_sampler, "sample", lambda *args, **kwargs: next(times))
    target = torch.tensor([[[0.8]], [[-0.3]]])
    torch.manual_seed(8)
    noise = torch.randn_like(target)
    torch.manual_seed(8)
    result = decoder.sample_consistency_targets(target, torch.zeros(2, 1, 2), teacher)
    start = torch.lerp(noise, target, t[:, None, None])
    next_t = torch.minimum(t + dt, torch.ones_like(t))
    next_action = torch.lerp(noise, target, next_t[:, None, None])
    final = next_action - 0.4 * (1 - next_t[:, None, None])
    torch.testing.assert_close(teacher.last_input, next_action)
    torch.testing.assert_close(teacher.last_time, next_t)
    torch.testing.assert_close(teacher.last_target_time, dt if mode == "relative" else next_t + dt)
    torch.testing.assert_close(result["target_t"], dt if mode == "relative" else next_t)
    torch.testing.assert_close(result["vt_target"], (final - start) / (1 - t[:, None, None]))
    assert not result["vt_target"].requires_grad


@pytest.mark.parametrize("solver", ["euler", "rk4"])
def test_ode_closed_form_and_detached_side_head(monkeypatch, solver):
    decoder = ODE(4, 2, 3, 4, cond_dim=8, hidden_dim=8, num_layers=1,
                  solver=solver, vel_scale=0.2, perturb_prob=0, side_head=2)
    state, initial, condition = torch.randn(2, 3), torch.randn(2, 2), torch.randn(2, 4)
    with monkeypatch.context() as patch:
        patch.setattr(decoder.model, "forward", lambda action, *args: torch.full_like(action, 0.75))
        actual = decoder.integrate(initial, state, None, substeps=3)
        expected = initial[:, None] + 0.15 * torch.arange(1, 5).view(1, 4, 1)
        torch.testing.assert_close(actual, expected)
    reference = copy.deepcopy(decoder)
    reference.num_components = 0
    target = torch.randn(2, 4, 2)
    loss, _ = decoder.compute_loss(condition, initial, state, target, perturb_std=0)
    expected, _ = reference.compute_loss(condition, initial, state, target, perturb_std=0)
    main = (*decoder.model.parameters(), *decoder.condition_encoder.parameters())
    ref = (*reference.model.parameters(), *reference.condition_encoder.parameters())
    grads = torch.autograd.grad(loss, (*main, *decoder.side_head.parameters()))
    torch.testing.assert_close(grads[:len(main)], torch.autograd.grad(expected, ref))
    assert any(grad.abs().sum() > 0 for grad in grads[len(main):])


@pytest.mark.parametrize("backbone_class", [DiT, ditx.DiTX])
def test_context_cache_outputs_gradients_and_new_observation(monkeypatch, backbone_class):
    action_io = sat_action_io.SATActionIO(4, 16)
    layout = {"horizon": 4, "action_dim": 19} if backbone_class is ditx.DiTX else {}
    backbone = backbone_class(
        n_obs_steps=1, obs_token_dim=8, hidden_dim=16, n_head=2, n_layers=2,
        timestep_embed_dim=8, p_drop_attn=0, action_io=action_io, **layout,
    ).eval()
    with torch.no_grad():
        for parameter in backbone.parameters():
            parameter.normal_(mean=0.2, std=0.15)
    actions = torch.randn(2, 4, 19, requires_grad=True)
    cond = torch.randn(2, 3, 8, requires_grad=True)

    def gradients(cache):
        output = backbone(actions, 0.4, cond, context_kv=cache)
        return output, torch.autograd.grad(output.square().mean(), (actions, cond, *backbone.parameters()),
                                          allow_unused=True)

    torch.testing.assert_close(gradients(backbone.prepare_context_kv(cond)), gradients(None))
    decoder = RectifiedFlow(backbone, num_inference_steps=3).eval()

    def sample(context):
        return decoder.predict_action(context, actions.shape, generator=torch.Generator().manual_seed(3))

    first = sample(cond)
    changed = cond + torch.randn_like(cond)
    second = sample(changed)
    assert (second - first).abs().max() > 1e-6
    with monkeypatch.context() as patch:
        patch.setattr(backbone, "prepare_context_kv", lambda context: None)
        torch.testing.assert_close(sample(changed), second)


# Data sampling


@pytest.fixture
def segments(tmp_path):
    path = tmp_path / "segments.zarr"
    group = zarr.open_group(str(path), mode="w")
    rows = np.arange(12, dtype=np.float32)
    fields = {
        "action": np.broadcast_to(rows[:, None] + 100, (12, 19)).copy(),
        "joint_state": np.broadcast_to(rows[:, None] + 200, (12, 19)).copy(),
        "point_cloud": np.arange(12 * 2 * 6, dtype=np.float64).reshape(12, 2, 6),
        "finger_tactile_force": np.broadcast_to(rows[:, None, None], (12, 5, 3)).copy(),
        "rgb": np.arange(12 * 2 * 3 * 3, dtype=np.uint8).reshape(12, 2, 3, 3),
    }
    for key, value in fields.items():
        group.create_dataset(f"data/{key}", data=value, chunks=(2, *value.shape[1:]))
    group.create_dataset("meta/episode_ends", data=np.array([1, 3, 8, 12]))
    return path, fields


@pytest.mark.parametrize("horizon,before,after", [(1, 0, 0), (4, 0, 0), (4, 3, 3), (6, 2, 5)])
@pytest.mark.parametrize("obs_horizon", [1, 2])
def test_window_boundaries_and_short_observations(segments, horizon, before, after, obs_horizon):
    path, fields = segments
    obs_horizon = min(obs_horizon, horizon)
    store = EpisodeStore(path, keys=("action", "point_cloud", "rgb"))
    mask = np.array([True, False, True, True])
    sampler = SequenceSampler(
        store, horizon, before, after, episode_mask=mask,
        fields={"action": None, "point_cloud": slice(0, 3), "rgb": None},
        field_horizons={"point_cloud": obs_horizon, "rgb": obs_horizon},
    )
    expected_rows = []
    for start, end, selected in zip([0, 1, 3, 8], [1, 3, 8, 12], mask):
        if selected:
            for offset in range(-before, end - start - horizon + after + 1):
                expected_rows.append([min(max(start + offset + t, start), end - 1) for t in range(horizon)])
    assert len(sampler) == len(expected_rows)
    for index, rows in enumerate(expected_rows):
        sample = sampler.sample_sequence(index)
        np.testing.assert_array_equal(sampler.sample_source_rows(index), rows)
        np.testing.assert_array_equal(sample["action"], fields["action"][rows])
        np.testing.assert_array_equal(sample["point_cloud"], fields["point_cloud"][rows[:obs_horizon], :, :3])
        np.testing.assert_array_equal(sample["rgb"], fields["rgb"][rows[:obs_horizon]])
        assert sample["point_cloud"].dtype == np.float32
        assert sample["rgb"].dtype == np.uint8
        sample["action"][:] = -1
        np.testing.assert_array_equal(sampler.sample_sequence(index)["action"], fields["action"][rows])

    indices = [*reversed(range(len(expected_rows))), 0, len(expected_rows) - 1]
    for index, sample in zip(indices, sampler.sample_batch(indices), strict=True):
        for key, value in sampler.sample_sequence(index).items():
            np.testing.assert_array_equal(sample[key], value)
        sample["action"][:] = -1


def test_real_sources_split_together_and_normalize_train_only(segments):
    path, fields = segments
    group = zarr.open_group(str(path), mode="a")
    group.attrs.update({
        "format": "dexmani.real.canonical",
        "source_episodes": ["a", "b", "a", "c"],
        "source_row_ranges": [[0, 1], [0, 2], [3, 8], [0, 4]],
    })
    dataset = DemonstrationDataset(
        path, sensor_modalities=["joint_state"], obs_horizon=1, horizon=3,
        pad_before=1, pad_after=2, val_episodes=1, max_train_episodes=1,
    )
    train, val = dataset.train_mask, dataset.val_mask
    assert train[0] == train[2] and val[0] == val[2]
    assert not (train & val).any()
    assert dataset.episode_store.count_groups(train) == dataset.episode_store.count_groups(val) == 1
    selected_rows = [row for start, end, selected in zip([0, 1, 3, 8], [1, 3, 8, 12], train)
                     if selected for row in range(start, end)]
    expected = fields["action"][selected_rows]
    chunks = list(dataset.iter_normalization_data("action"))
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    normalizer = LinearNormalizer()
    normalizer.fit_field("action", chunks)
    np.testing.assert_allclose(normalizer["action"].normalize(expected).amin(0), -1, atol=1e-6)
    np.testing.assert_allclose(normalizer["action"].normalize(expected).amax(0), 1, atol=1e-6)
    states = fields["joint_state"][selected_rows]
    normalizer.fit_field("joint_state", dataset.iter_normalization_data("joint_state"), mode="mean_std")
    np.testing.assert_allclose(normalizer["joint_state"].normalize(states),
                               (states - states.mean(0)) / states.std(0), atol=2e-5)
    for index in range(len(dataset)):
        rows = dataset.sampler.sample_source_rows(index)
        episodes = np.searchsorted([1, 3, 8, 12], rows, side="right")
        assert np.all(episodes == episodes[0]) and train[episodes[0]]


@pytest.mark.parametrize("key,feature_shape", [("point_cloud", (3,)), ("fingertip_xyz", (5, 3))])
def test_streaming_mean_std(key, feature_shape):
    values = np.random.default_rng(5).normal(size=(11, 5, 3))
    values[..., 0] *= 1e-6
    values[..., 1] = 5
    points = values.reshape(-1, *feature_shape)
    mean, std = points.mean(axis=0), points.std(axis=0)
    divisor = np.where(std < 1e-4, 1, std)
    normalizer = LinearNormalizer()
    normalizer.fit_field(key, np.split(values, [1, 1, 4]), mode="mean_std")
    field = normalizer[key]
    np.testing.assert_allclose(field.scale, 1 / divisor, rtol=1e-6)
    np.testing.assert_allclose(field.offset, -mean / divisor, rtol=1e-6)
    np.testing.assert_allclose(field.normalize(values), (values - mean) / divisor, atol=1e-6)


@pytest.mark.parametrize("width", [23, 46])
def test_proprio_frame_shape(width):
    def prepare(frame_shape):
        return prepare_observations(
            {"proprio": np.zeros((2, *frame_shape))}, normalization_dims={"proprio": width},
            n_obs_steps=2, rgb_preprocessing={}, device="cpu",
        )["proprio"]

    assert prepare((width,)).shape == (1, 2, width)
    with pytest.raises(ValueError, match=rf"proprio.*{width}"):
        prepare((1,))
    with pytest.raises(ValueError, match="proprio"):
        prepare((1, width))


def test_custom_observation_uses_checkpoint_channel_width():
    with pytest.raises(ValueError, match="custom_state.*width 7"):
        prepare_observations(
            {"custom_state": np.ones((2, 1))}, normalization_dims={"custom_state": 7},
            n_obs_steps=2, rgb_preprocessing={}, device="cpu",
        )


def test_mean_std_deployment_matches_training_model(tmp_path, small_dp3_config):
    cfg = small_dp3_config(tmp_path / "run", "normalization.joint_state=mean_std")
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
        policy = trainer.ema_model.eval()
        dataset = trainer.train_loader.dataset
        obs = {k: v.numpy() for k, v in dataset[0]["obs"].items()}
        batch = prepare_observations(obs, normalization_dims=policy.normalization_dims,
                                     n_obs_steps=1, rgb_preprocessing={}, device="cpu")
        expected = predict_actions(policy, batch, inference_steps=2, generator=torch.Generator().manual_seed(23))
        runtime = load_policy(tmp_path / "run", device="cpu", weights="ema", inference_steps=2, seed=23)
        with closing(runtime):
            torch.testing.assert_close(runtime.agent.normalizer.state_dict(), policy.normalizer.state_dict(),
                                       rtol=0, atol=0)
            np.testing.assert_array_equal(runtime.predict(obs), expected)


def test_ode_tactile_and_afro_temporal_targets(segments, monkeypatch):
    path, fields = segments
    group = zarr.open_group(str(path), mode="a")
    group["meta/episode_ends"][:] = [3, 6, 9, 12]
    options = dict(horizon=4, obs_horizon=2, pad_before=1, pad_after=3)
    ode = ODEDataset(path, sensor_modalities=["joint_state"], **options)
    tactile = AdapTacDataset(path, sensor_modalities=["finger_tactile_force"], **options)
    afro = AFROPretrainDataset(path, frame_stride=2, horizon=4, pad_before=1, pad_after=3)
    assert len(ode) == len(afro) == 16
    assert len(tactile) == 8
    for episode in range(4):
        start = episode * 3
        for offset, query in enumerate([start, start + 1, start + 2, start + 2]):
            expected = fields["joint_state"][query] if query == start else fields["action"][query - 1]
            np.testing.assert_array_equal(ode[episode * 4 + offset]["prev_action"], expected)
        for offset in range(2):
            sample = tactile[episode * 2 + offset]
            expected = [start + 1, start + 2, 0] if offset == 0 else [start + 2, 0, 0]
            np.testing.assert_array_equal(sample["future_force"][:, 0, 0], expected)
            np.testing.assert_array_equal(sample["future_force_valid"], [True, offset == 0, False])
        for offset, rows in enumerate([[start, start + 1], [start, start + 2], [start + 1, start + 2], [start + 2] * 2]):
            np.testing.assert_array_equal(afro[episode * 4 + offset]["obs"]["point_cloud"], fields["point_cloud"][rows, :, :3])

    read = ode.episode_store.read
    action_reads = []

    def record_read(key, index, **kwargs):
        if key == "action":
            action_reads.append(index)
        return read(key, index, **kwargs)

    monkeypatch.setattr(ode.episode_store, "read", record_read)
    np.testing.assert_array_equal(ode[1]["prev_action"], fields["action"][0])
    assert action_reads == [slice(0, 3)]
    single = ODEDataset(path, sensor_modalities=["joint_state"], obs_horizon=1)
    np.testing.assert_array_equal(single[0]["prev_action"], fields["joint_state"][0])
    np.testing.assert_array_equal(single[1]["prev_action"], fields["action"][0])


# R3D dataset and loss


class CustomR3DDataset(R3DDataset):
    pass


@pytest.mark.parametrize("auxiliary,action_key,width", [
    (False, "action", 19), (True, "action", 28), (False, "action_ee", 21),
])
def test_r3d_sampling_and_normalization(make_demo, auxiliary, action_key, width):
    path = make_demo(auxiliary=auxiliary or action_key == "action_ee")
    dataset = R3DDataset(
        path, use_aux_ee=auxiliary, action_key=action_key,
        sensor_modalities=["joint_state"], horizon=2, obs_horizon=1,
    )
    values = dataset.episode_store.read(action_key, slice(None))
    if auxiliary:
        values = np.concatenate((values, dataset.episode_store.read("action_ee", slice(None))[:, :9]), -1)
        assert dataset.consumed_fields["action_ee"] == slice(0, 9)
    elif action_key == "action":
        assert "action_ee" not in dataset.episode_store.keys()
        assert "action_ee" not in dataset.consumed_fields
    assert dataset.action_dim == width
    np.testing.assert_array_equal(dataset[0]["action"], values[:2])
    np.testing.assert_array_equal(np.concatenate(list(dataset.iter_normalization_data("action"))), values)
    normalizer = LinearNormalizer({"action": (width,)})
    fit_action_normalizer(dataset, normalizer)
    scale = 2 / (values.max(0) - values.min(0))
    offset = -1 - scale * values.min(0)
    if action_key == "action_ee":
        scale[3:9], offset[3:9] = 1, 0
    torch.testing.assert_close(normalizer["action"].scale, torch.from_numpy(scale))
    torch.testing.assert_close(normalizer["action"].offset, torch.from_numpy(offset))


@pytest.mark.parametrize("action_key", ["action_ee", "action_rotvec"])
def test_auxiliary_requires_joint_action(make_demo, action_key):
    with pytest.raises(ValueError, match="joint action"):
        R3DDataset(make_demo(), use_aux_ee=True, action_key=action_key,
                   sensor_modalities=["joint_state"], horizon=2, obs_horizon=1)


@pytest.mark.parametrize("auxiliary,width", [(False, 19), (True, 28)])
def test_r3d_explicit_dimensions_and_custom_target(make_demo, auxiliary, width):
    target = f"{__name__}.CustomR3DDataset"
    cfg = load_config("r3d", [
        f"zarr_path={make_demo()}", "task_name=test", "dataset.augmentation_cfg=null",
        f"++dataset._target_={target}", "++agent.action_dim=20",
        f"use_aux_ee={str(auxiliary).lower()}",
    ])
    assert cfg.dataset._target_ == target
    assert cfg.agent.action_dim == 20
    with pytest.raises(ValueError, match="action width"):
        build_dataset(cfg)
    cfg.agent.action_dim = None
    dataset, description = build_dataset(cfg)
    assert isinstance(dataset, CustomR3DDataset)
    assert cfg.agent.action_dim == description["field_dims"]["action"] == width


class SmallObservationEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(19, 32)
        self.out_dim = 32
        self.pc_pe_dim = 16

    def forward(self, obs):
        return self.projection(obs["joint_state"]).unsqueeze(1)


class ConstantPrediction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.values = nn.Parameter(torch.linspace(-0.5, 0.5, dim))

    def forward(self, x, timestep, context):
        return self.values.expand_as(x) + context.mean((1, 2))[:, None, None]


@pytest.mark.parametrize("auxiliary,weight", [(False, 1), (True, 0), (True, 0.7)])
@pytest.mark.parametrize("log_metrics", [False, True])
def test_r3d_group_loss_and_gradients(monkeypatch, auxiliary, weight, log_metrics):
    monkeypatch.setattr(r3d, "R3DEncoder", SmallObservationEncoder)
    dim = 28 if auxiliary else 19
    model = r3d.R3D(
        horizon=4, n_obs_steps=1, n_action_steps=4, action_dim=dim, state_dim=19,
        use_aux_ee=auxiliary, aux_loss_weight=weight, embedding_dim=16, depth=1,
        num_heads=2, mlp_dim=32, timestep_embed_dim=8, num_inference_steps=2,
    )
    model.action_decoder.model = ConstantPrediction(dim)
    model.configure_normalizer({"action": dim}, {"action": (dim,)})
    model.normalizer["action"].scale.copy_(torch.linspace(1, 2, dim))
    model.normalizer["action"].offset.fill_(-0.25)
    actions = torch.randn(2, 4, dim)
    batch = {"obs": {"joint_state": torch.randn(2, 1, 19)}, "action": actions}
    token = object()
    compute_loss = model.action_decoder.compute_loss

    def with_extra_kwarg(*args, probe, **kwargs):
        assert probe is token
        return compute_loss(*args, **kwargs)

    monkeypatch.setattr(model.action_decoder, "compute_loss", with_extra_kwarg)
    loss, metrics = model.compute_loss(batch, log_metrics=log_metrics, probe=token)
    condition = model._encode_obs(batch["obs"])
    prediction = model.action_decoder.model(actions, 0, condition)
    error = (prediction - model._encode_actions(actions)).square()
    joint = error[..., :19].mean()
    expected = joint + weight * error[..., 19:].mean() if auxiliary else error.mean()
    torch.testing.assert_close(loss, expected)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(loss, parameters)
    expected_gradients = torch.autograd.grad(expected, parameters)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    if auxiliary and weight == 0:
        assert torch.count_nonzero(actual_gradients[-1][19:]) == 0
    if log_metrics:
        torch.testing.assert_close(metrics["loss"], expected.detach())
        if auxiliary:
            torch.testing.assert_close(metrics["loss_joint"], joint.detach())
            torch.testing.assert_close(metrics["loss_ee"], error[..., 19:].mean().detach())
        else:
            assert set(metrics) == {"loss"}
    else:
        assert metrics == {}
    assert model.predict_action(batch["obs"])["future_actions"].shape == (2, 4, 19)


# Hand-centric interaction encoder


def test_hand_token_construction():
    encoder = HandKinematicEncoder()
    fingertips_wrist = torch.arange(1, 31, dtype=torch.float32).reshape(2, 5, 3) / 256
    wrist_xyz_1 = torch.tensor([[0.125, 0.25, 0.375], [-0.25, 0.125, 0.5]])
    wrist_xyz_2 = torch.tensor([[-0.125, 0.375, 0.25], [0.25, -0.125, 0.375]])
    rotation_1 = torch.eye(3).expand(2, -1, -1)
    rotation_2 = torch.tensor([
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    ])
    capture_wrist, capture_fingers = Mock(return_value=None), Mock(return_value=None)
    finger_token_sets = []
    for wrist_xyz_base, rotation in ((wrist_xyz_1, rotation_1), (wrist_xyz_2, rotation_2)):
        rotation_6d = torch.cat((rotation[:, :, 0], rotation[:, :, 1]), dim=-1)
        eef_pose = torch.cat((wrist_xyz_base, rotation_6d), dim=-1).requires_grad_()
        fingertips_base = (fingertips_wrist @ rotation.transpose(-1, -2) + wrist_xyz_base[:, None]).requires_grad_()
        with (
            encoder.wrist_mlp.register_forward_pre_hook(capture_wrist),
            encoder.finger_mlp.register_forward_pre_hook(capture_fingers),
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            hand_tokens, hand_landmarks_base = encoder(eef_pose, fingertips_base)

        assert hand_tokens.shape == (2, 6, 192)
        assert hand_landmarks_base.shape == (2, 6, 3)
        expected_landmarks = torch.cat((wrist_xyz_base[:, None], fingertips_base), dim=1)
        assert torch.equal(hand_landmarks_base, expected_landmarks)
        _, (wrist_input,) = capture_wrist.call_args.args
        _, (finger_inputs,) = capture_fingers.call_args.args
        expected_wrist_input = torch.cat((wrist_xyz_base / 0.25, rotation_6d), dim=-1)
        torch.testing.assert_close(wrist_input, expected_wrist_input)
        torch.testing.assert_close(finger_inputs, fingertips_wrist / 0.10)
        finger_token_sets.append(hand_tokens[:, 1:])

        pose_grad, fingertips_grad = torch.autograd.grad(finger_inputs.sum(), (eef_pose, fingertips_base))
        assert pose_grad.isfinite().all() and torch.count_nonzero(pose_grad[:, 3:]) > 0
        assert fingertips_grad.isfinite().all() and torch.count_nonzero(fingertips_grad) > 0

    torch.testing.assert_close(finger_token_sets[0], finger_token_sets[1])

    with pytest.raises(ValueError, match="fingertip_xyz"):
        encoder(eef_pose, fingertips_base[:, :0])
    with pytest.raises(RuntimeError, match="split_sizes"):
        encoder(torch.cat((eef_pose, eef_pose[:, :1]), dim=-1), fingertips_base)


def test_multi_finger_tactile_encoding():
    encoder = FingerTactileEncoder()
    force = torch.randn(2, 5, 3)
    features = encoder(force)
    assert features["local_tokens"].shape == (2, 5, 64)
    assert features["global_token"].shape == (2, 64)
    assert torch.equal(features["global_token"], features["local_tokens"].mean(dim=1))
    torch.testing.assert_close(encoder(force.double()), features)

    zero_features = encoder(torch.zeros_like(force))
    assert torch.count_nonzero(zero_features["local_tokens"]) == 0
    assert torch.count_nonzero(zero_features["global_token"]) == 0


def test_tactile_fusion_zero_force_invariance():
    encoder, fusion = FingerTactileEncoder(), TactileFusion()
    hand_tokens = torch.randn(2, 6, 192)
    force = torch.zeros(2, 5, 3)
    assert torch.equal(fusion(hand_tokens, encoder(force)), hand_tokens)

    force[:, 2, 0] = 1
    features = encoder(force)
    residual = fusion(torch.zeros_like(hand_tokens), features)
    assert torch.equal(residual[:, [1, 2, 4, 5]], residual[:, :1].expand(-1, 4, -1))
    assert torch.count_nonzero(residual[:, 0]) > 0
    assert torch.count_nonzero(residual[:, 3] - residual[:, 0]) > 0

    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        fusion(hand_tokens, {**features, "global_token": features["global_token"][:1]})
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        fusion(hand_tokens, {**features, "local_tokens": features["local_tokens"][:, :1]})

    fixed_features = {key: torch.ones_like(value) for key, value in features.items()}
    hand_states = torch.zeros_like(hand_tokens)
    hand_states[1, 0, 0] = 4
    hand_states[1, 3, 0] = 2
    with torch.no_grad():
        for gate in (fusion.local_gate, fusion.contact_gate):
            torch.testing.assert_close(gate.bias, torch.full_like(gate.bias, -1.5))
            gate.weight.zero_()
            gate.weight[:, [0, 64]] = 1

    capture_local, capture_contact = Mock(return_value=None), Mock(return_value=None)
    with (
        fusion.local_gate.register_forward_hook(capture_local),
        fusion.contact_gate.register_forward_hook(capture_contact),
    ):
        fused = fusion(hand_states, fixed_features)

    _, (local_input,), local_logits = capture_local.call_args.args
    _, (contact_input,), contact_logits = capture_contact.call_args.args
    torch.testing.assert_close(local_input, torch.cat((fixed_features["local_tokens"], hand_states[:, 1:]), dim=-1))
    torch.testing.assert_close(contact_input, torch.cat((fixed_features["global_token"], hand_states.mean(dim=1)), dim=-1))
    expected_local_logits = torch.full((2, 5, 1), -0.5)
    expected_local_logits[1, 2] = 1.5
    expected_contact_logits = torch.tensor([[-0.5], [0.5]])
    torch.testing.assert_close(local_logits, expected_local_logits)
    torch.testing.assert_close(contact_logits, expected_contact_logits)
    expected = hand_states.clone()
    expected[:, 1:] += (
        expected_contact_logits.sigmoid().unsqueeze(1)
        * expected_local_logits.sigmoid()
        * fusion.local_projection(fixed_features["local_tokens"])
    )
    expected += (expected_contact_logits.sigmoid() * fusion.global_projection(fixed_features["global_token"]))[:, None]
    torch.testing.assert_close(fused, expected)

    optimizer = torch.optim.SGD([*encoder.parameters(), *fusion.parameters()], lr=0.1)
    fusion(hand_tokens, features).square().mean().backward()
    optimizer.step()
    assert torch.equal(fusion(hand_tokens, encoder(torch.zeros_like(force))), hand_tokens)


def test_tactile_contact_gate_scales_local_update():
    fusion = TactileFusion()
    local_tokens = torch.ones(2, 5, 64)
    features = {"local_tokens": local_tokens, "global_token": local_tokens.mean(dim=1)}
    hand_tokens = torch.zeros(2, 6, 192)
    hand_tokens[1, 0, 0] = 6
    with torch.no_grad():
        fusion.local_gate.weight.zero_()
        fusion.local_gate.bias.zero_()
        fusion.contact_gate.weight.zero_()
        fusion.contact_gate.bias.fill_(-3)
        fusion.contact_gate.weight[:, 64] = 6

    output = fusion(hand_tokens, features)
    residual = output - hand_tokens
    # Subtracting the wrist residual isolates the finger-local path.
    contact_weight = torch.tensor([[-3.0], [3.0]]).sigmoid()
    expected_local = contact_weight.unsqueeze(1) * 0.5 * fusion.local_projection(local_tokens)
    torch.testing.assert_close(residual[:, 1:] - residual[:, :1], expected_local)


def test_hand_scene_attention_geometry_bias():
    attention = GeoAwareRelationAttention()
    hand_tokens = torch.randn(2, 6, 192)
    scene_tokens = torch.randn(2, 128, 192)
    hand_landmarks_base = torch.randn(2, 6, 3, dtype=torch.bfloat16)
    scene_points_base = torch.randn(2, 128, 3, dtype=torch.bfloat16)
    assert torch.count_nonzero(attention.geometry_bias[-1].weight) == 0
    assert torch.count_nonzero(attention.geometry_bias[-1].bias) == 0
    with torch.no_grad():
        attention.geometry_bias[-1].weight[:, :4].copy_(torch.eye(4))

    capture_geometry = Mock(return_value=None)
    with (
        attention.geometry_bias.register_forward_hook(capture_geometry),
        patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa,
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        tokens = attention(hand_tokens, scene_tokens, hand_landmarks_base, scene_points_base)

    assert tokens.shape == (2, 6, 192)
    _, (geometry_input,), geometry_output = capture_geometry.call_args.args
    assert geometry_input.shape == geometry_output.shape == (2, 6, 128, 4)
    assert geometry_input.dtype == geometry_output.dtype == torch.float32
    relative_xyz = scene_points_base.float()[:, None] - hand_landmarks_base.float()[:, :, None]
    expected_geometry = torch.cat(
        (
            relative_xyz / 0.10,
            torch.linalg.vector_norm(relative_xyz, dim=-1, keepdim=True) / 0.10,
        ),
        dim=-1,
    )
    torch.testing.assert_close(geometry_input, expected_geometry)
    query, key, value = sdpa.call_args.args[:3]
    bias = sdpa.call_args.kwargs["attn_mask"]
    assert query.shape == (2, 4, 6, 48)
    assert key.shape == value.shape == (2, 4, 128, 48)
    assert bias.shape == (2, 4, 6, 128)
    assert bias.dtype == query.dtype == torch.bfloat16
    assert torch.count_nonzero(bias) > 0
    torch.testing.assert_close(bias, geometry_output.permute(0, 3, 1, 2).to(query.dtype))

    with pytest.raises(ValueError, match="batch"):
        attention(hand_tokens, scene_tokens[:1], hand_landmarks_base, scene_points_base[:1])
    with pytest.raises(ValueError, match="batch"):
        attention(hand_tokens, scene_tokens, hand_landmarks_base[:, :1], scene_points_base)


def make_hand_scene_observation(point_channels=3):
    point_cloud = torch.rand(2, 160, point_channels)
    point_cloud[..., :3] = 0.20 * point_cloud[..., :3] - 0.1
    point_cloud[..., 2] -= 0.2
    camera_extrinsic = torch.eye(4)[:3].repeat(2, 1, 1)
    camera_extrinsic[0, 2, 3] = 1
    return {
        "point_cloud": point_cloud,
        "rgb": torch.randint(256, (2, 3, 8, 8), dtype=torch.uint8),
        "camera_intrinsic": torch.tensor([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]).expand(2, -1, -1),
        "camera_extrinsic": camera_extrinsic,
        "eef_pose": torch.tensor([[0.1, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).expand(2, -1),
        "fingertip_xyz": torch.rand(2, 5, 3) * 0.20,
        "finger_tactile_force": torch.randn(2, 5, 3),
    }


@pytest.mark.parametrize("scene_config,point_channels,samples_per_patch", [
    pytest.param({}, 3, 1, id="center-xyz"),
    pytest.param({"rgb_patch_sampling": "group_mean", "pc_dim": 6}, 6, 32, id="group-mean-xyzrgb"),
])
def test_hand_scene_encoder(scene_config, point_channels, samples_per_patch):
    encoder = HandObjectInteractionEncoder(
        rgb_backbone_name="dino",
        rgb_backbone_config={"architecture": tiny_hf_rgb_config("dino").to_dict(), "image_size": 8},
        **scene_config,
    ).eval()

    obs = make_hand_scene_observation(point_channels)
    point_cloud = obs["point_cloud"]
    camera_extrinsic = obs["camera_extrinsic"]
    tokens = encoder(obs)
    assert tokens.shape == (2, 6, 192)
    result = encoder.forward_with_aux({
        **obs,
        "camera_intrinsic": obs["camera_intrinsic"].flatten(1),
        "camera_extrinsic": camera_extrinsic.flatten(1),
    })
    assert {key: tuple(value.shape) for key, value in result.items()} == {
        "tokens": (2, 6, 192),
        "scene_tokens": (2, 128, 192),
        "scene_points_base": (2, 128, 3),
        "hand_landmarks_base": (2, 6, 3),
        "local_tokens": (2, 5, 64),
        "global_token": (2, 64),
        "rgb_projection_valid": (2, 128, samples_per_patch),
    }
    torch.testing.assert_close(result["tokens"], tokens)
    expected_landmarks = torch.cat((obs["eef_pose"][:, None, :3], obs["fingertip_xyz"]), dim=1)
    assert torch.equal(result["hand_landmarks_base"], expected_landmarks)
    patch_center_in_cloud = (result["scene_points_base"][:, :, None] == point_cloud[:, None, :, :3]).all(dim=-1)
    assert patch_center_in_cloud.any(dim=-1).all()
    assert all(result[key].requires_grad for key in ("tokens", "scene_tokens", "local_tokens", "global_token"))
    valid = result["rgb_projection_valid"]
    assert valid.dtype == torch.bool
    assert valid[0].all() and not valid[1].any()
    with pytest.raises(ValueError, match="extrinsics"):
        encoder({**obs, "camera_extrinsic": camera_extrinsic.transpose(-1, -2)})

    mixed_points = point_cloud.clone()
    mixed_points[0, :, :3].zero_()
    with pytest.raises(ValueError, match="empty frame"):
        encoder({**obs, "point_cloud": mixed_points})


def test_hand_scene_encoder_backward():
    torch.manual_seed(0)
    encoder = HandObjectInteractionEncoder(
        rgb_backbone_name="dino",
        rgb_backbone_config={"architecture": tiny_hf_rgb_config("dino").to_dict(), "image_size": 8},
    )
    obs = make_hand_scene_observation()
    tokens = encoder(obs)
    assert tokens.shape == (2, 6, 192)
    probe = torch.randn_like(tokens)
    (tokens * probe).mean().backward()

    for name in (
        "patch_encoder.geometry_embedding.point_mlp.0.weight",
        "rgb_fusion.semantic_projection.weight",
        "spatial_encoder.blocks.0.qkv.weight",
        "scene_projection.weight",
        "hand_encoder.wrist_mlp.0.weight",
        "hand_encoder.finger_mlp.0.weight",
        "hand_encoder.slot_embeddings",
        "tactile_encoder.force_encoder.0.weight",
        "tactile_fusion.local_projection.weight",
        "tactile_fusion.global_projection.weight",
        "tactile_fusion.local_gate.weight",
        "tactile_fusion.local_gate.bias",
        "tactile_fusion.contact_gate.weight",
        "tactile_fusion.contact_gate.bias",
        "geometric_relation.hand_attention.in_proj_weight",
        "geometric_relation.query_projection.weight",
        "geometric_relation.scene_kv.weight",
        "geometric_relation.geometry_bias.2.weight",
        "geometric_relation.ffn.0.weight",
    ):
        grad = encoder.get_parameter(name).grad
        assert grad is not None, name
        assert grad.isfinite().all(), name
        assert grad.count_nonzero() > 0, name

    geometry_bias = encoder.geometric_relation.geometry_bias
    # The zero-initialized final layer blocks gradients to the first layer until an update.
    optimizer = torch.optim.SGD(geometry_bias.parameters(), lr=0.1)
    optimizer.step()
    encoder.zero_grad(set_to_none=True)
    (encoder(obs) * probe).mean().backward()

    distance_grad = geometry_bias[0].weight.grad[:, 3]
    assert distance_grad.isfinite().all()
    assert distance_grad.count_nonzero() > 0


# Policy initialization


class SmallRGBWeights(nn.Linear):
    out_dim = 2

    def __init__(self):
        super().__init__(2, 2)
        self.effective_config = {}

    def initialize_weights(self, source):
        if source is not None:
            self.load_state_dict(torch.load(source, weights_only=True))


def build_small_rgb(name, config):
    return SmallRGBWeights(), ImageProcessor.from_preset(name, config)


@pytest.mark.parametrize("method,module", [("dp", dp), ("sat", rgb_point)])
def test_rgb_initialization_in_dp_and_sat(tmp_path, make_demo, monkeypatch, method, module):
    monkeypatch.setattr(module, "build_rgb_encoder", build_small_rgb)
    dataset = DemonstrationDataset(make_demo(), sensor_modalities=["joint_state"], horizon=4, obs_horizon=1)
    source = tmp_path / "rgb.pt"
    rgb = SmallRGBWeights()
    torch.save(rgb.state_dict(), source)
    model_config = ["agent.down_dims=[16,32]"] if method == "dp" else [
        "agent.hidden_dim=16", "agent.n_layers=1", "agent.n_head=2", "agent.pc_encoder_config.channels=24",
    ]
    cfg = load_config(method, [
        "training.device=cpu", "agent.action_dim=19", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        f"initialization.rgb={source}", *model_config,
    ])
    with patch.object(SmallRGBWeights, "initialize_weights", autospec=True,
                      side_effect=SmallRGBWeights.initialize_weights) as initialize:
        model = build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
    assert isinstance(model.obs_encoder, dp.DPEncoder if method == "dp" else rgb_point.RGBPointObservationEncoder)
    initialize.assert_called_once_with(model.obs_encoder.rgb_encoder, str(source))
    torch.testing.assert_close(model.obs_encoder.rgb_encoder.state_dict(), rgb.state_dict())


def test_non_rgb_training_ignores_rgb_source(tmp_path, small_dp3_config):
    cfg = small_dp3_config(tmp_path / "non_rgb", "+initialization.rgb=unused")
    with closing(build_trainer(cfg)) as trainer:
        assert not hasattr(trainer.model.obs_encoder, "rgb_encoder")
        assert "prior_data" not in trainer.cfg


@pytest.mark.parametrize("method", ["dqrise", "dexlatent"])
def test_hand_affine_and_latent_order(tmp_path, make_demo, monkeypatch, method):
    def encode_joint_state(self, obs):
        features = self.rgb_encoder(obs["joint_state"][:, :2]).repeat(1, self.out_dim // 2)
        return features.unflatten(0, (-1, self.n_obs_steps))

    monkeypatch.setattr(rgb_point, "build_rgb_encoder", build_small_rgb)
    monkeypatch.setattr(rgb_point.RGBPointObservationEncoder, "forward", encode_joint_state)
    data_path = make_demo()
    dataset = DemonstrationDataset(data_path, sensor_modalities=["joint_state"], horizon=4, obs_horizon=1)
    directory = tmp_path / "run"
    cfg = load_config(method, [
        f"zarr_path={data_path}", "task_name=test", f"workspace.output_dir={directory}",
        "workspace.wandb_cfg=null", "dataset.sensor_modalities=[joint_state]", "dataset.augmentation_cfg=null",
        "training.device=cpu", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.hidden_dim=16", "agent.n_layers=1", "agent.n_head=2", "agent.pc_encoder_config.channels=24",
        "training.use_compile=false", "training.loop.total_train_steps=3",
        "dataloader.num_workers=0", "dataloader.batch_size=2",
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
        training_data = json.dumps({"source": str(make_demo("codebook_source")), "train_source_episodes": [0]})
        np.savez(
            path, sorted_hand_poses=poses, hand_normalizer_scale=scale.numpy(), hand_normalizer_offset=offset.numpy(),
            training_data=training_data,
        )
        cfg.initialization.codebook = str(path)
        invalid_cfg = copy.deepcopy(cfg)
        invalid_path = tmp_path / "invalid_codebook.npz"
        np.savez(
            invalid_path, sorted_hand_poses=poses, hand_normalizer_scale=np.zeros(12, dtype=np.float32),
            hand_normalizer_offset=offset.numpy(), training_data=training_data,
        )
        invalid_cfg.initialization.codebook = str(invalid_path)
        with pytest.raises(ValueError, match="Invalid affine parameters"):
            build_policy(invalid_cfg, dataset, {"action": 19, "joint_state": 19})
    else:
        from dexmani_policy.agents.hand_prior.dexlatent import DexLatentHandPrior

        prior = DexLatentHandPrior()
        path = tmp_path / "prior.pt"
        torch.save({"autoencoders": {"xarm7_xhand_right." + key: value for key, value in prior.state_dict().items()}}, path)
        cfg.initialization.dexlatent = str(path)
        affine = prior.initialize_weights(path)
        scale, offset = affine["scale"], affine["offset"]
    with patch.object(SmallRGBWeights, "initialize_weights", autospec=True,
                      side_effect=SmallRGBWeights.initialize_weights) as initialize:
        model = build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
    assert isinstance(model.obs_encoder, rgb_point.RGBPointObservationEncoder)
    initialize.assert_called_once_with(model.obs_encoder.rgb_encoder, str(rgb_path))
    torch.testing.assert_close(model.obs_encoder.rgb_encoder.state_dict(), rgb.state_dict())
    torch.testing.assert_close(model.normalizer["action"].scale[7:], scale)
    torch.testing.assert_close(model.normalizer["action"].offset[7:], offset)
    if method == "dqrise":
        assert cfg.agent.num_codes == model.hand_codebook.num_codes == 3
        torch.testing.assert_close(model.hand_codebook.sorted_hand_poses, torch.from_numpy(poses))
        assert cfg.prior_data[0].artifact == str(path)
        assert not cfg.prior_data[0].budget_verified
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
    sat_cfg = load_config("sat", [
        "training.device=cpu", "agent.action_dim=19", "agent.hidden_dim=16",
        "agent.n_layers=1", "agent.n_head=2", "initialization.rgb=null", "agent.pc_encoder_config.channels=24",
    ])
    sat = build_policy(sat_cfg, dataset, {"action": 19, "joint_state": 19})
    with torch.no_grad():
        sat.normalizer["action"].scale.fill_(9)
        sat.obs_encoder.rgb_encoder.weight.zero_()
    sat_cfg.agent.rgb_backbone_config.tune_mode = "full"
    torch.testing.assert_close(model.state_dict(), before)
    assert cfg.agent == before_config

    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    saved = load_checkpoint(latest)
    assert saved.effective_config["prior_data"] == OmegaConf.to_container(cfg.prior_data)
    cfg.initialization.policy = str(latest)
    cfg.workspace.output_dir = str(tmp_path / "warm")
    for restore in (lambda: prepare_resume(directory), lambda: build_trainer(cfg)):
        with closing(restore()) as trainer:
            torch.testing.assert_close(trainer.model.state_dict(), saved.model_state)
    cfg.workspace.output_dir = str(tmp_path / "warm_bad")
    warm_data = make_demo("warm_bad")
    cfg.zarr_path = cfg.dataset.zarr_path = str(warm_data)
    zarr.open_group(str(warm_data), mode="r+")["data/action"][6, 8] = np.inf
    with pytest.raises(ValueError, match=r"field=action,.*row=6, channel=\(8,\)"):
        build_trainer(cfg)
    assert load_checkpoint(latest).global_step == 1
    assert not Path(cfg.workspace.output_dir).exists()


@pytest.mark.parametrize("method", ["afro", "cordvip"])
@pytest.mark.parametrize("mode", ["limits", "identity", "mean_std"])
def test_encoder_normalization_inherited_before_configuration(tmp_path, make_demo, method, mode):
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
    inherited = LinearNormalizer({key: (dims[key],) for key in fields if mode != "identity"})
    for field in inherited.fields.values():
        field.scale.fill_(2.5)
        field.offset.fill_(0.75)
    source = tmp_path / "encoder.pt"
    data_source = str(make_demo("independent")) if mode == "identity" else dataset.episode_store.source_path
    payload = {
        "encoder_state": encoder.state_dict(),
        "normalizer_state": inherited.state_dict(),
        "normalization": dict.fromkeys(fields, mode),
        "training_data": {"source": data_source, "train_source_episodes": [0]},
    }
    torch.save(payload, source)
    cfg.initialization.encoder = str(source)
    model = build_policy(cfg, dataset, dims)
    assert cfg.prior_data[0].budget_verified == (mode != "identity")
    actual_encoder = model.obs_encoder.pc_encoder if method == "afro" else model.obs_encoder.interaction_encoder
    torch.testing.assert_close(actual_encoder.state_dict(), encoder.state_dict())
    for key in fields:
        assert cfg.normalization[key] == mode
        if mode != "identity":
            torch.testing.assert_close(model.normalizer[key].state_dict(), inherited[key].state_dict())
        else:
            assert key not in model.normalizer.fields
    expected = LinearNormalizer()
    expected.fit_field("action", dataset.iter_normalization_data("action"))
    torch.testing.assert_close(model.normalizer["action"].state_dict(), expected["action"].state_dict())
    if method == "afro":
        expected.fit_field("joint_state", dataset.iter_normalization_data("joint_state"))
        torch.testing.assert_close(model.normalizer["joint_state"].state_dict(), expected["joint_state"].state_dict())


def test_resume_and_warm_start_state_boundaries(tmp_path, small_dp3_config, make_demo, monkeypatch):
    directory = tmp_path / "source"
    data = make_demo("source", episode_ends=(4, 8, 12))
    first_source, resumed_source = "gpu:/snapshots/first", "gpu:/snapshots/resumed"
    monkeypatch.setenv("DEXMANI_POLICY_REMOTE_SOURCE", first_source)
    with closing(build_trainer(small_dp3_config(directory, f"zarr_path={data}"))) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    first_checkpoint = latest.resolve()
    saved = load_checkpoint(first_checkpoint)
    assert saved.effective_config["remote_source"] == first_source
    for override in ("action_key=action_ee", "agent._target_=dexmani_policy.agents.policies.dp.DP"):
        invalid = small_dp3_config(tmp_path / "invalid", f"initialization.policy={first_checkpoint}", override)
        with pytest.raises(ValueError, match="same policy and action semantics"):
            build_trainer(invalid)

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint restoration must not fit normalization")

    monkeypatch.setattr(LinearNormalizer, "fit_field", forbidden)
    monkeypatch.setattr("dexmani_policy.training.initialization.resolve_rgb_architecture", forbidden)
    monkeypatch.setenv("DEXMANI_POLICY_REMOTE_SOURCE", resumed_source)
    with closing(prepare_resume(directory)) as resumed:
        torch.testing.assert_close(resumed.model.state_dict(), saved.model_state)
        torch.testing.assert_close(resumed.ema_model.state_dict(), saved.ema_model_state)
        optimizer = resumed.optimizer.state_dict()
        torch.testing.assert_close(optimizer["state"], saved.optimizer_state["state"])
        assert optimizer["param_groups"] == saved.optimizer_state["param_groups"]
        assert resumed.scheduler.state_dict() == saved.scheduler_state
        assert resumed.global_step == 1
        actual_parameters = {id(p) for p in resumed.model.parameters() if p.requires_grad}
        assert {id(p) for group in resumed.optimizer.param_groups for p in group["params"]} == actual_parameters
        assert {id(p) for p in resumed.ema_updater.parameters} == actual_parameters
        resumed.train(max_updates=1)
    continued = load_checkpoint(latest)
    assert continued.global_step == continued.scheduler_state["last_epoch"] == 2
    for group, base_lr in zip(continued.optimizer_state["param_groups"], saved.scheduler_state["base_lrs"]):
        assert group["lr"] == pytest.approx(0.25 * base_lr)
    assert continued.effective_config["remote_source"] == resumed_source
    assert continued.effective_config["training_data"] == saved.effective_config["training_data"]
    assert "prior_data" not in continued.effective_config
    assert OmegaConf.load(directory / "config.yaml").remote_source == first_source

    new_data = tmp_path / "copied.zarr"
    shutil.copytree(data, new_data)
    warm_cfg = small_dp3_config(
        tmp_path / "warm", f"initialization.policy={first_checkpoint}",
        f"zarr_path={data}", "dataset.max_train_episodes=1",
    )
    with pytest.raises(ValueError, match="same-source prior exceeds"):
        build_trainer(warm_cfg)
    warm_cfg.zarr_path = warm_cfg.dataset.zarr_path = str(new_data)
    with closing(build_trainer(warm_cfg)) as warm:
        torch.testing.assert_close(warm.model.state_dict(), saved.model_state)
        assert warm.global_step == 0
        assert warm.scheduler.last_epoch == 0
        assert warm.optimizer.state_dict()["state"] == {}
        assert warm.logger.output_dir == tmp_path / "warm"
        assert warm.cfg.training_data.source == str(new_data)
        assert len(warm.cfg.training_data.train_source_episodes) == 1
        prior, = warm.cfg.prior_data
        assert prior.source == saved.effective_config["training_data"]["source"]
        assert prior.train_source_episodes == [0, 1, 2]
        assert not prior.budget_verified
        torch.testing.assert_close(warm.ema_model.state_dict(), warm.model.state_dict())
        warm.train(max_updates=1)

    warm_checkpoint = tmp_path / "warm/checkpoints/latest.pt"
    warm_saved = load_checkpoint(warm_checkpoint)
    repeated_cfg = OmegaConf.create(warm_saved.effective_config)
    repeated_cfg.workspace.output_dir = str(tmp_path / "warm_again")
    repeated_cfg.initialization.policy = str(warm_checkpoint)
    with closing(build_trainer(repeated_cfg)) as repeated:
        assert repeated.cfg.training_data == warm_saved.effective_config["training_data"]
        prior, = repeated.cfg.prior_data
        assert prior == {
            **warm_saved.effective_config["training_data"],
            "artifact": str(warm_checkpoint.resolve()), "budget_verified": False,
        }

    resume_cfg = OmegaConf.create(warm_saved.effective_config)
    alias = tmp_path / "alias.zarr"
    alias.symlink_to(new_data)
    resume_cfg.dataset.zarr_path = str(alias)
    del resume_cfg.training_data
    resume_cfg.prior_data[0].budget_verified = True
    resume_cfg.control_dt_s = 0.1
    with closing(build_trainer(resume_cfg, checkpoint=warm_saved)) as resumed:
        assert resumed.cfg.zarr_path == resumed.cfg.dataset.zarr_path == str(new_data)
        assert resumed.cfg.control_dt_s == warm_saved.effective_config["control_dt_s"]
        assert resumed.cfg.training_data == warm_saved.effective_config["training_data"]
        assert resumed.cfg.prior_data == warm_saved.effective_config["prior_data"]


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


def test_interflow_rgb_initialization_and_checkpoint_lifecycle(tmp_path, make_demo, monkeypatch):
    from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder

    directory, source = tmp_path / "fresh", tmp_path / "rgb"
    pretrained = save_tiny_hf_rgb(source, "dino")
    path = make_demo()
    data = zarr.open_group(str(path), mode="r+")["data"]
    observations = make_hand_scene_observation()
    observations["rgb"] = observations["rgb"].permute(0, 2, 3, 1)
    for key in ("camera_intrinsic", "camera_extrinsic"):
        observations[key] = observations[key].flatten(1)
    frame_indices = torch.arange(len(data["action"])) % len(observations["rgb"])
    for key, frames in observations.items():
        data.create_dataset(key, data=frames[frame_indices].numpy(), overwrite=True)
    cfg = load_config("interflow", [
        f"zarr_path={path}", "task_name=test", f"workspace.output_dir={directory}",
        "workspace.wandb_cfg=null", "training.device=cpu", "training.use_compile=false",
        "dataloader.num_workers=0", "dataloader.batch_size=1", f"initialization.rgb={source}",
        "agent.rgb_backbone_name=dino", "agent.rgb_backbone_config.tune_mode=full",
        "agent.rgb_backbone_config.image_size=[8,8]", "agent.n_layers=1", "agent.hidden_dim=32",
    ])

    with (
        patch.object(ViTEncoder, "initialize_weights", autospec=True,
                     side_effect=ViTEncoder.initialize_weights) as initialize_rgb,
        closing(build_trainer(cfg)) as fresh,
    ):
        agent = fresh.model
        encoder = agent.obs_encoder
        rgb = encoder.rgb_encoder
        initialize_rgb.assert_called_once_with(rgb, str(source))
        assert [name for name, module in encoder.named_modules(remove_duplicate=False) if module is rgb] == [
            "interaction_encoder.rgb_encoder",
        ]
        for model in (fresh.model, fresh.ema_model):
            torch.testing.assert_close(model.obs_encoder.rgb_encoder.backbone.state_dict(), pretrained.state_dict())
        prior, = fresh.cfg.prior_data
        assert prior == {"source": None, "artifact": str(source), "budget_verified": False}
        assert fresh.cfg.agent.rgb_backbone_config.architecture.hidden_size == 16

        batch = default_collate([fresh.train_loader.dataset[1]])
        obs = batch["obs"]
        agent.eval()
        with torch.inference_mode():
            condition = agent._encode_obs(obs)
            assert condition.shape == (1, 14, 192)
            flat = agent._preprocess_obs(obs)
            hand = encoder.interaction_encoder(flat) + encoder.token_type_embedding[:, :1]
            state = encoder.state_encoder(flat["joint_state"]).unsqueeze(1) + encoder.token_type_embedding[:, 1:]
            frames = encoder.token_norm(torch.cat((hand, state), dim=1))
            torch.testing.assert_close(condition[0, :7], frames[0])
            torch.testing.assert_close(condition[0, 7:], frames[1])
            result = agent.predict_action(obs, generator=torch.Generator().manual_seed(13))["future_actions"]
        assert result.shape == (1, 15, 19) and result.isfinite().all()
        runner = SimRunner("test", agent.n_obs_steps, list(obs), {}, fresh.cfg.control_dt_s)
        chunk = runner.predict_action_chunk(
            obs, agent, action_steps=agent.n_action_steps, generator=torch.Generator().manual_seed(13),
        )
        assert chunk.shape == (8, 19)
        np.testing.assert_allclose(chunk, result[0, :8].numpy())

        agent.train()
        assert isinstance(agent.action_decoder, RectifiedFlow) and not agent.requires_ema_for_loss
        loss, metrics = agent(batch)
        assert loss.isfinite() and set(metrics) == {"loss"}
        loss.backward()

        with torch.no_grad():
            next(rgb.parameters()).fill_(2)
            next(fresh.ema_model.obs_encoder.rgb_encoder.parameters()).fill_(3)
        fresh.save_checkpoint()

        def forbidden(*args, **kwargs):
            pytest.fail("Checkpoint restoration must use the saved RGB architecture")

        monkeypatch.setattr(policy_training.initialization, "resolve_rgb_architecture", forbidden)
        latest = directory / "checkpoints/latest.pt"
        saved = load_checkpoint(latest)
        assert saved.effective_config["agent"]["geometry_scales"] == fresh.model.effective_config["geometry_scales"]
        with closing(prepare_resume(directory)) as resumed:
            torch.testing.assert_close(resumed.model.state_dict(), saved.model_state)
            torch.testing.assert_close(resumed.ema_model.state_dict(), saved.ema_model_state)
        with closing(load_policy(latest, weights="raw", device="cpu")) as runtime:
            torch.testing.assert_close(runtime.agent.state_dict(), saved.model_state)
        warm_cfg = copy.deepcopy(cfg)
        warm_cfg.initialization.policy = str(latest)
        warm_cfg.workspace.output_dir = str(tmp_path / "warm")
        with open_dict(warm_cfg.agent):
            del warm_cfg.agent.rgb_backbone_name
        with closing(build_trainer(warm_cfg)) as warm:
            saved_rgb = saved.effective_config["agent"]["rgb_backbone_config"]
            assert warm.cfg.agent.rgb_backbone_name == "dino"
            assert warm.cfg.agent.rgb_backbone_config.architecture == saved_rgb["architecture"]
            torch.testing.assert_close(warm.model.state_dict(), saved.model_state)
            torch.testing.assert_close(warm.ema_model.state_dict(), saved.model_state)
        assert initialize_rgb.call_count == 1


@pytest.mark.parametrize("kind,preprocessing", [("dino", {"image_size": 8}), ("siglip", {})])
def test_rgb_resolution_and_external_weights(tmp_path, monkeypatch, kind, preprocessing):
    from transformers import AutoConfig, SiglipVisionConfig
    from dexmani_policy.training.initialization import inherit_rgb_architecture, resolve_rgb_architecture

    directory = tmp_path / "rgb"
    source = save_tiny_hf_rgb(directory, kind)
    agent = OmegaConf.create({
        "rgb_backbone_name": kind, "rgb_backbone_config": {"tune_mode": "full", **preprocessing},
    })
    resolve_rgb_architecture(agent, str(directory))
    assert agent.rgb_backbone_config.architecture.hidden_size == 16
    encoder = HandObjectInteractionEncoder(**OmegaConf.to_container(agent))
    config = copy.deepcopy(encoder.effective_config)
    assert config["geometry_scales"] == {
        "wrist_position_base": 0.25,
        "finger_position_wrist": 0.10,
        "hand_scene_relative_position_base": 0.10,
    }
    encoder = HandObjectInteractionEncoder(**config)
    mismatched_scales = {**config["geometry_scales"], "wrist_position_base": 0.50}
    with pytest.raises(ValueError, match="geometry_scales"):
        HandObjectInteractionEncoder(**{**config, "geometry_scales": mismatched_scales})
    initial_weights = copy.deepcopy(encoder.rgb_encoder.state_dict())
    encoder.rgb_encoder.initialize_weights(None)
    torch.testing.assert_close(encoder.rgb_encoder.state_dict(), initial_weights, rtol=0, atol=0)
    encoder.rgb_encoder.initialize_weights(str(directory))
    torch.testing.assert_close(encoder.rgb_encoder.backbone.state_dict(), source.state_dict())
    larger_config = copy.deepcopy(config)
    larger_config["rgb_backbone_config"]["architecture"]["num_hidden_layers"] += 1
    larger_encoder = HandObjectInteractionEncoder(**larger_config)
    with pytest.raises(ValueError, match="missing configured parameters"):
        larger_encoder.rgb_encoder.initialize_weights(str(directory))
    assert config == encoder.effective_config
    assert config["rgb_backbone_config"]["architecture"]["hidden_size"] == 16
    assert config["rgb_backbone_config"]["image_size"] == (8, 8)
    images = encoder.image_processor.process_images(torch.rand(2, 3, 16, 16))
    result = encoder.rgb_encoder(images)
    result["patch_tokens"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.rgb_encoder.backbone.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("resolved or inherited architecture must not read a source")

    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(SiglipVisionConfig, "from_pretrained", forbidden)
    resolve_rgb_architecture(agent, "missing-source")
    warm = OmegaConf.create({"rgb_backbone_name": kind, "rgb_backbone_config": {}})
    saved_agent = OmegaConf.to_container(agent)
    inherit_rgb_architecture(warm, saved_agent)
    assert warm.rgb_backbone_config.architecture == agent.rgb_backbone_config.architecture
    warm.rgb_backbone_config.architecture.hidden_size = 32
    inherit_rgb_architecture(warm, saved_agent)
    assert warm.rgb_backbone_config.architecture.hidden_size == 32
    warm.rgb_backbone_name = "resnet"
    with pytest.raises(ValueError, match="RGB backbone"):
        inherit_rgb_architecture(warm, saved_agent)


def test_dinov3_timm_architecture_and_local_weights(tmp_path, monkeypatch):
    import timm
    from safetensors.torch import save_file
    from transformers import AutoConfig, SiglipVisionConfig

    def forbidden(*args, **kwargs):
        pytest.fail("DINOv3 uses a timm architecture")

    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(SiglipVisionConfig, "from_pretrained", forbidden)
    agent = OmegaConf.create({"rgb_backbone_name": "dinov3"})
    policy_training.initialization.resolve_rgb_architecture(agent, str(tmp_path))
    assert "rgb_backbone_config" not in agent

    overrides = {"img_size": 32, "embed_dim": 24, "depth": 1, "num_heads": 3}
    architecture = {"model_name": "vit_small_patch16_dinov3", **overrides}
    source = timm.create_model(**architecture, pretrained=False)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    with patch.object(timm, "create_model", wraps=timm.create_model) as create_model:
        encoder = HandObjectInteractionEncoder(rgb_backbone_config={
            "architecture": overrides, "image_size": 32, "tune_mode": "full",
        })
    create_model.assert_called_once_with(**architecture, pretrained=False)
    config = copy.deepcopy(encoder.effective_config)
    encoder.rgb_encoder.initialize_weights(str(tmp_path))
    torch.testing.assert_close(encoder.rgb_encoder.backbone.state_dict(), source.state_dict())
    assert config == encoder.effective_config
    assert config["rgb_backbone_config"]["architecture"] == architecture
    assert config["rgb_backbone_config"]["image_size"] == (32, 32)


def test_method_owned_warm_checks():
    from dexmani_policy.training.initialization import prepare_warm_start

    agent = OmegaConf.create({"rgb_backbone_name": "dino", "rgb_backbone_config": {}})
    saved = {"rgb_backbone_name": "dino", "rgb_backbone_config": {"architecture": {"hidden_size": 16}}, "num_codes": 5}
    prepare_warm_start("dqrise", agent, saved)
    assert agent.num_codes == 5
    assert agent.rgb_backbone_config.architecture.hidden_size == 16
    agent.num_codes = 7
    prepare_warm_start("dqrise", agent, saved)
    assert agent.num_codes == 7
    tactile = OmegaConf.create({"tactile_encoder": {"_target_": "force"}})
    prepare_warm_start("adaptac", tactile, {"tactile_encoder": {"_target_": "force"}})
    with pytest.raises(ValueError, match="Tactile encoder"):
        prepare_warm_start("adaptac", tactile, {"tactile_encoder": {"_target_": "taxel"}})


@pytest.mark.parametrize("method,fields", [
    ("dp", {"rgb", "finger_tactile_force", "taxel_tactile_force"}),
    ("sat", {"point_cloud", "camera_intrinsic", "camera_extrinsic"}),
    ("interflow", {"point_cloud", "camera_intrinsic", "camera_extrinsic", "eef_pose", "fingertip_xyz"}),
    ("dqrise", {"point_cloud", "camera_intrinsic", "camera_extrinsic"}),
    ("dexlatent", {"point_cloud", "camera_intrinsic", "camera_extrinsic"}),
    ("adaptac", {"point_cloud"}),
])
def test_observation_identity_constraints(method, fields):
    from dexmani_policy.common.normalizer import normalization_feature_shapes

    cfg = load_config(method)
    cls = hydra.utils.get_class(cfg.agent._target_)
    options = {"required_identity_fields": cls.required_identity_fields}
    for field in fields:
        for mode in ("limits", "mean_std"):
            with pytest.raises(ValueError, match="identity"):
                normalization_feature_shapes({field: 3}, {field: mode}, **options)
        assert normalization_feature_shapes({field: 3}, {field: "identity"}, **options) == {}


@pytest.mark.parametrize("method", [
    "dp", "dp3", "r3d", "maniflow", "interflow", "sat", "dqrise", "dexlatent", "afro", "cordvip", "adaptac", "ode",
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


# SAT readout


def test_sat_readout_formula_with_nonzero_output():
    torch.manual_seed(3)
    io = sat_action_io.SATActionIO(4, 16)
    readout = io.final_layer
    with torch.no_grad():
        readout.output.weight.normal_()
        readout.output.bias.normal_()
    tokens = torch.randn(2, 19, 16, requires_grad=True)
    reference = tokens.detach().clone().requires_grad_(True)
    actual = io.decode_velocity(tokens)
    normalized = reference * (reference.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * readout.norm.weight
    hidden = F.gelu(F.linear(normalized, readout.hidden.weight, readout.hidden.bias), approximate="tanh")
    expected = F.linear(hidden, readout.output.weight, readout.output.bias).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    parameters = tuple(io.final_layer.parameters())
    actual_grads = torch.autograd.grad(actual.square().mean(), (tokens, *parameters))
    expected_grads = torch.autograd.grad(expected.square().mean(), (reference, *parameters))
    torch.testing.assert_close(actual_grads, expected_grads)


@pytest.mark.parametrize("backbone_class", [DiT, ditx.DiTX])
def test_sat_backbones(backbone_class):
    io = sat_action_io.SATActionIO(4, 16)
    layout = {"horizon": 4, "action_dim": 19} if backbone_class is ditx.DiTX else {}
    backbone = backbone_class(
        n_obs_steps=1, obs_token_dim=8, hidden_dim=16, n_head=2, n_layers=2,
        timestep_embed_dim=8, p_drop_attn=0, action_io=io, **layout,
    )
    actions = torch.randn(2, 4, 19, requires_grad=True)
    tokens = io.encode(actions)
    assert tokens.shape == (2, 19, 16)
    context = torch.randn(2, 3, 8)
    output = backbone(actions, torch.tensor([0.2, 0.7]), context)
    assert output.shape == actions.shape
    assert torch.count_nonzero(output) == 0
    (output - 1).square().mean().backward()
    head = io.final_layer.output
    assert head.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(head.bias) == 0
    with torch.no_grad():
        head.weight.normal_()
    backbone.zero_grad()
    backbone(actions, 0.5, context).square().mean().backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0


# Gradient accumulation


class Samples(Dataset):
    sensor_modalities = ("target",)

    def __init__(self, targets, episode_store):
        self.targets = targets
        self.obs_horizon = 1
        self.episode_store = episode_store
        self.train_mask, self.val_mask = np.array([True]), np.array([False])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        value = self.targets[index].reshape(1, 1)
        return {"obs": {"target": value}, "action": value}

    def get_validation_dataset(self):
        return None

    def training_summary(self):
        return "deterministic scalar regression"


class MeanDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def compute_loss(self, condition, target, *, log_metrics=True):
        loss = (self.weight - target).square().mean()
        return loss, {"loss": loss.detach()} if log_metrics else {}


class ObservationModel(nn.Module):
    def __init__(self, normalization, n_obs_steps, contact_loss_weight=0):
        super().__init__()
        self.encoder = MeanDecoder()
        self.normalizer = LinearNormalizer()

    def forward(self, batch):
        return self.encoder.compute_loss(None, batch["obs"]["target"])


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
        "dataset": {}, "normalization": {}, "model": {"contact_loss_weight": 0},
        "data_processing": None,
    })


@pytest.mark.parametrize("count,batch_size,accumulation", [
    (4, 2, 2), (5, 4, 2), (9, 2, 3), (9, 4, 2),
    (1, 128, 4), (5, 2, 1), (129, 128, 2),
])
@pytest.mark.parametrize("loop", ["policy", "observation"])
def test_sample_weighted_updates(tmp_path, monkeypatch, make_demo, count, batch_size, accumulation, loop):
    targets = torch.arange(count).float() / count
    if count == 129:
        targets.zero_()
        targets[-1] = 1
    updates = math.ceil(math.ceil(count / batch_size) / accumulation)
    cfg = training_config(tmp_path, batch_size, accumulation, updates)
    dataset = Samples(targets, EpisodeStore(make_demo(), keys=()))
    actual_gradients, optimizers, schedulers = [], [], []

    def capture_clip(parameters, finite_loss, max_grad_norm=0, *, step):
        parameters = list(parameters)
        actual_gradients.append(parameters[0].grad.clone())
        return clip_gradients(parameters, finite_loss, max_grad_norm, step=step)

    def sgd(parameters, **kwargs):
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        optimizers.append(optimizer)
        return optimizer

    def scheduler(optimizer, *args, **kwargs):
        result = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        schedulers.append(result)
        return result

    if loop == "policy":
        model = Policy(nn.Identity(), MeanDecoder(), 1, 1, 1, 1)
        model.configure_normalizer({"action": 1}, {"action": (1,)})
        monkeypatch.setattr(policy_training, "build_optimizer", lambda model, **kw: sgd(model.parameters()))
        monkeypatch.setattr(policy_training, "build_scheduler", scheduler)
        monkeypatch.setattr(policy_training, "clip_gradients", capture_clip)
        trainer = policy_training.PolicyTrainer(cfg, model, dataset)
        ema_steps = []
        ema_step = trainer.ema_updater.step

        def update_ema(step):
            ema_steps.append(step)
            ema_step(step)

        monkeypatch.setattr(trainer.ema_updater, "step", update_ema)
        try:
            trainer.train()
            weight = model.action_decoder.weight.detach()
            assert trainer.global_step == updates
            assert ema_steps == list(range(updates))
        finally:
            trainer.close()
    else:
        monkeypatch.setattr(obs_training, "load_run", lambda *args: (cfg, None, None))
        monkeypatch.setattr(obs_training.hydra.utils, "instantiate", lambda *args, **kwargs: dataset)
        monkeypatch.setattr(obs_training.torch.optim, "AdamW", sgd)
        monkeypatch.setattr(obs_training, "build_scheduler", scheduler)
        monkeypatch.setattr(obs_training, "clip_gradients", capture_clip)
        obs_training.train("cordvip", ObservationModel)
        weight = optimizers[0].param_groups[0]["params"][0].detach()

    reference = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([reference], lr=0.1)
    expected_losses, expected_gradients, expected_norms = [], [], []
    for group in targets.split(batch_size * accumulation):
        optimizer.zero_grad()
        loss = (reference - group).square().mean()
        loss.backward()
        expected_losses.append(loss.detach())
        expected_gradients.append(reference.grad.clone())
        expected_norms.append(torch.nn.utils.clip_grad_norm_([reference], 0.1 if loop == "policy" else float("inf")))
        optimizer.step()
    torch.testing.assert_close(actual_gradients, expected_gradients)
    torch.testing.assert_close(weight, reference.detach())
    assert schedulers[0].last_epoch == updates
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == list(range(1, updates + 1))
    for record, loss, norm in zip(records, expected_losses, expected_norms):
        assert record["train/loss"] == pytest.approx(float(loss))
        assert record["train/lr"] == 0.1
        if loop == "policy":
            assert record["train/grad_norm"] == pytest.approx(float(norm))
    if count == 129:
        assert records[0]["train/loss"] == pytest.approx(1 / 129)
        assert float(actual_gradients[0]) == pytest.approx(-2 / 129)


@pytest.mark.parametrize("force_weight", [0.0, 0.7])
def test_adaptac_accumulation_matches_combined_force_and_action_means(tmp_path, monkeypatch, force_weight):
    class FixedForce(ForceDiffusion):
        def _training_prediction(self, condition, target):
            return self.model.weight.expand_as(target), target

    class ActionDecoder(MeanDecoder):
        def compute_loss(self, condition, target, *, log_metrics):
            loss = (self.weight + condition - target).square().mean()
            return loss, {}

    class SmallAdapTac(AdapTac):
        def __init__(self):
            Policy.__init__(self, nn.Identity(), ActionDecoder(), 4, 1, 4, 1)
            Policy.configure_normalizer(self, {"action": 1}, {"action": (1,)})
            self.force_predictor = FixedForce(MeanDecoder())
            self.force_fusion = nn.Identity()
            self.force_loss_weight = force_weight

        def _encode_force_inputs(self, obs):
            return None, None, None

        def _action_condition(self, features, current_force, prediction):
            return prediction

    dataset = [
        {"obs": {}, "action": torch.full((4, 1), action),
         "future_force": torch.full((4, 1), force), "future_force_valid": torch.arange(4) < valid}
        for action, force, valid in [(0.5, 0.0, 1), (0.7, 0.0, 1), (0.9, 1.0, 4)]
    ]
    model = SmallAdapTac()
    expected_loss, expected_metrics = model.compute_loss(default_collate(dataset))
    expected_gradients = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    updates = []

    def capture_clip(parameters, finite_loss, max_grad_norm=0, *, step):
        updates.append([p.grad.clone() for p in model.parameters()])
        return clip_gradients(parameters, finite_loss, max_grad_norm, step=step)

    monkeypatch.setattr(policy_training, "clip_gradients", capture_clip)
    cfg = training_config(tmp_path, batch_size=2, accumulation=2, updates=1)
    with closing(policy_training.PolicyTrainer(cfg, model, dataset)) as trainer:
        trainer.train()
    assert len(updates) == 1
    torch.testing.assert_close(updates[0], list(expected_gradients))
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    for key, value in expected_metrics.items():
        assert record[f"train/{key}"] == pytest.approx(float(value))


def test_nonfinite_gradient_rejected():
    parameter = nn.Parameter(torch.zeros(()))
    parameter.grad = torch.tensor(float("inf"))
    with pytest.raises(FloatingPointError):
        clip_gradients([parameter], torch.tensor(True), step=0)
    parameter.grad.zero_()
    with pytest.raises(FloatingPointError):
        clip_gradients([parameter], torch.tensor(False), step=0)


def test_afro_pretraining_still_requires_single_batch(tmp_path, monkeypatch):
    cfg = load_config("pretrain/obs/afro", [
        f"workspace.output_dir={tmp_path}", "training.loop.gradient_accumulation_steps=2",
    ])
    monkeypatch.setattr(obs_training, "load_run", lambda *args: (cfg, None, None))
    with pytest.raises(ValueError, match="VICReg requires accumulation=1"):
        obs_training.train("afro", ObservationModel)


# Atomic file publishing


def test_atomic_publish_does_not_unlink_after_replace(tmp_path, monkeypatch):
    target = tmp_path / "result"
    target.write_text("old")
    unlink = Path.unlink
    calls = []

    def track_unlink(path, *args, **kwargs):
        calls.append(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", track_unlink)
    with io_util.atomic_path(target) as temporary:
        assert temporary.parent == target.parent
        temporary.write_text("new")
    assert target.read_text() == "new"
    assert not temporary.exists()
    assert calls == []


@pytest.mark.parametrize("source", ["writer", "replace"])
def test_failed_atomic_publish_keeps_previous_file(tmp_path, monkeypatch, source):
    target = tmp_path / "result"
    target.write_text("old")

    def fail(*args):
        raise OSError("publish failed")

    if source == "replace":
        monkeypatch.setattr(io_util.os, "replace", fail)
    with pytest.raises(OSError):
        with io_util.atomic_path(target) as temporary:
            temporary.write_text("new")
            if source == "writer":
                fail()
    assert target.read_text() == "old"
    assert not temporary.exists()


def test_keyboard_interrupt_cleans_up(tmp_path):
    target = tmp_path / "result"
    target.write_text("old")
    interrupt = KeyboardInterrupt("stop writer")
    with pytest.raises(KeyboardInterrupt) as caught:
        with io_util.atomic_path(target) as temporary:
            raise interrupt
    assert caught.value is interrupt
    assert not temporary.exists()
    assert target.read_text() == "old"


def test_checkpoint_and_latest_symlink_publish(tmp_path, monkeypatch):
    latest = tmp_path / "latest.pt"
    first = save_checkpoint(tmp_path / "step_1.pt", Checkpoint(1, {}, {}, {}, {}))
    latest.symlink_to(first.name)
    second = save_checkpoint(tmp_path / "step_2.pt", Checkpoint(2, {}, {}, {}, {}))
    unlink = Path.unlink
    calls = []

    def track_unlink(path, *args, **kwargs):
        calls.append(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", track_unlink)
    with io_util.atomic_path(latest) as temporary:
        temporary.unlink()
        temporary.symlink_to(second.name)
    assert calls == [temporary]
    assert latest.is_symlink() and latest.readlink() == Path(second.name)
    assert load_checkpoint(latest).global_step == 2
    assert first.exists() and second.exists()
    assert not temporary.exists() and not temporary.is_symlink()


# Training artifacts


@pytest.fixture(params=["~/run", "relative/run", "absolute"])
def output_path(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    original = Path.expanduser

    def expand(path):
        if path.parts and path.parts[0] == "~":
            return tmp_path / "mapped_home" / Path(*path.parts[1:])
        return original(path)

    monkeypatch.setattr(Path, "expanduser", expand)
    value = str(tmp_path / "absolute") if request.param == "absolute" else request.param
    yield value, Path(value).expanduser().resolve()
    assert not (tmp_path / "~").exists()


def test_policy_artifact_paths(output_path, small_dp3_config):
    value, expected = output_path
    cfg = small_dp3_config(value)
    with closing(build_trainer(cfg)) as trainer:
        assert trainer.logger.output_dir == expected
        assert trainer.cfg.workspace.output_dir == str(expected)
        trainer.train(max_updates=1)
    saved_cfg = OmegaConf.load(expected / "config.yaml")
    saved = load_checkpoint(expected / "checkpoints/latest.pt")
    assert saved_cfg.workspace.output_dir == saved.effective_config["workspace"]["output_dir"] == str(expected)
    original_config = (expected / "config.yaml").read_bytes()
    original_metrics = (expected / "metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build_trainer(cfg)
    assert (expected / "config.yaml").read_bytes() == original_config
    assert (expected / "metrics.jsonl").read_bytes() == original_metrics
    with closing(prepare_resume(expected)) as resumed:
        assert resumed.logger.output_dir == expected
        assert resumed.global_step == 1


def test_vq_artifact_paths(output_path, make_demo):
    value, expected = output_path
    data = make_demo()
    cfg = load_config("pretrain/hand_prior/vqvae", [
        f"zarr_path={data.name}", f"workspace.output_dir='{value}'",
        "vq_vae.device=cpu", "vq_vae.num_epochs=1", "vq_vae.save_epochs=1",
        "vq_vae.batch_size=12", "vq_vae.warmup_steps=0", "vq_vae.hidden_dim=16",
        "vq_vae.latent_dim=8", "vq_vae.num_layers=1",
    ])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    actions = zarr.open_group(str(data), mode="r+")["data/action"]
    valid = actions[0, 7]
    actions[0, 7] = np.inf
    with pytest.raises(ValueError, match="Nonfinite data"):
        train_vq(cfg)
    assert not expected.exists()
    actions[0, 7] = valid
    train_vq(cfg)
    assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    assert torch.get_float32_matmul_precision() == "medium"
    assert cfg.workspace.output_dir == str(expected)
    saved_cfg = OmegaConf.load(expected / "config.yaml")
    assert saved_cfg.workspace.output_dir == str(expected)
    assert saved_cfg.zarr_path == saved_cfg.dataset.zarr_path == str(data)
    for name in ("metrics.jsonl", "codebook.npz", "checkpoints/last.pt"):
        assert (expected / name).exists()
    with pytest.raises(FileExistsError):
        train_vq(cfg)


@pytest.fixture
def contact_data(make_demo):
    path = make_demo(auxiliary=False, episode_ends=(4, 8, 12))
    store = zarr.open_group(str(path), mode="a")
    points = store["data/point_cloud"][:, :, :3]
    store.create_dataset("data/hand_pointcloud", data=points)
    template = np.random.default_rng(45).normal(size=(1024, 3)).astype("<f4")
    world_points = template[None] + np.arange(12, dtype=np.float32)[:, None, None] * 0.1
    store.create_dataset("data/object_pointcloud", data=world_points)
    store.create_dataset("data/contact_map", data=np.zeros((12, 1024), dtype=np.float32))
    return path


def test_observation_validation_reports_nonfinite_contact_metric(contact_data):
    dataset = CordViPPretrainDataset(contact_data, contact_loss_weight=1, val_episodes=1, obs_horizon=1)
    validation = dataset.get_validation_dataset()
    row = validation.sampler.sample_source_rows(0)[0]
    zarr.open_group(str(contact_data), mode="a")["data/contact_map"][row, 0] = np.nan
    model = CordViP(dict.fromkeys(dataset.sensor_modalities, "identity"), n_obs_steps=1)

    with pytest.raises(ValueError, match="Non-finite observation pretraining validation metrics: .*val/contact"):
        obs_training.evaluate(model, validation, 2, torch.device("cpu"), stop=SimpleNamespace(requested=False))


def test_afro_validation_reuses_noise_and_preserves_training_rng():
    model = AFRO({"point_cloud": "identity"})
    generator = torch.Generator().manual_seed(17)
    dataset = [{"obs": {"point_cloud": torch.rand(4, 16, 3, generator=generator)}} for _ in range(4)]
    metrics = []
    for seed in (42, 112):
        state = torch.Generator().manual_seed(seed).get_state()
        torch.set_rng_state(state)
        metrics.append(obs_training.evaluate(
            model, dataset, 2, torch.device("cpu"), stop=SimpleNamespace(requested=False),
        ))
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        assert model.training
    assert metrics[0] == metrics[1]


def test_observation_artifacts(output_path, contact_data, tmp_path, monkeypatch):
    value, directory = output_path
    source = contact_data
    raw = zarr.open_group(str(source), mode="a")
    groups = ["a", "b", "c"]
    raw.attrs.update(format="dexmani.real.canonical", source_episodes=groups)
    derived = source.parent / "cordvip" / source.name
    derived.parent.mkdir()
    shutil.copytree(source, derived)
    del raw["data/contact_map"]
    store = zarr.open_group(str(derived), mode="a")
    store.attrs["source_zarr_path"] = str(source)
    policy = tmp_path / "policy.yaml"
    OmegaConf.save(OmegaConf.create(dict(
        task_name="test", zarr_path=str(source),
        dataset=dict(seed=42, val_episodes=1, max_train_episodes=1),
    )), policy)
    args = [
        "--output-dir", value, "--policy-config", str(policy),
        "--device", "cpu", "--no-wandb", "--max-updates", "1",
        "normalization.joint_state=mean_std", "n_obs_steps=1", "dataloader.num_workers=0", "dataloader.batch_size=2",
        "training.loop.total_train_steps=3", "training.loop.gradient_accumulation_steps=2",
    ]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    obs_training.train("cordvip", CordViP, args)
    assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    assert torch.get_float32_matmul_precision() == "medium"
    saved = load_checkpoint(directory / "checkpoints/latest.pt")
    assert saved.global_step == 1
    assert saved.effective_config["dataset"]["zarr_path"] == str(derived)
    betas = saved.optimizer_state["param_groups"][0]["betas"]
    assert isinstance(betas, tuple) and betas == (0.95, 0.999)
    assert saved.effective_config["workspace"]["output_dir"] == str(directory)
    assert OmegaConf.load(directory / "config.yaml").workspace.output_dir == str(directory)
    with pytest.raises(FileExistsError):
        obs_training.train("cordvip", CordViP, args)
    assert any(key.startswith("contact_head.") for key in saved.model_state)
    payload = torch.load(directory / "encoder.pt", weights_only=True)
    assert payload["training_data"] == saved.effective_config["training_data"]
    assert "episode_layout" not in payload
    assert saved.effective_config["episode_layout"]["episode_groups"] == groups
    assert payload["training_data"]["source"] == str(source)
    train_group, = payload["training_data"]["train_source_episodes"]
    start = 4 * groups.index(train_group)
    states = store["data/joint_state"][start:start + 4]
    np.testing.assert_allclose(payload["normalizer_state"]["fields.joint_state.scale"],
                               1 / states.std(axis=0, dtype=np.float64), rtol=1e-6)
    for name in ("encoder", "normalizer"):
        expected = {key.removeprefix(name + "."): value for key, value in saved.model_state.items()
                    if key.startswith(name + ".")}
        torch.testing.assert_close(payload[name + "_state"], expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not use fresh fitting or the fresh output guard")

    monkeypatch.setattr(obs_training, "fit_observation_normalizer", forbidden)
    monkeypatch.setattr(obs_training, "check_fresh_output", forbidden)
    resume_args = ["--resume-from", value, "--max-updates", "1"]
    obs_training.train("cordvip", CordViP, resume_args)
    resumed = load_checkpoint(directory / "checkpoints/latest.pt")
    assert resumed.global_step == 2
    assert resumed.scheduler_state["last_epoch"] == 2
    assert resumed.effective_config["training_data"] == saved.effective_config["training_data"]
    for key in payload["normalizer_state"]:
        torch.testing.assert_close(resumed.model_state["normalizer." + key], payload["normalizer_state"][key])

    artifacts = [directory / name for name in ("config.yaml", "metrics.jsonl", "encoder.pt", "checkpoints/latest.pt")]
    before = [path.read_bytes() for path in artifacts]
    with pytest.raises(OSError):
        obs_training.train("cordvip", CordViP, ["--resume-from", str(artifacts[-1]), "--max-updates", "1"])
    with pytest.raises(SystemExit) as rejected:
        obs_training.train("cordvip", CordViP, [*resume_args, "--zarr-path", str(source)])
    assert rejected.value.code == 2

    val_group, = payload["training_data"]["val_source_episodes"]
    store.attrs["source_episodes"] = ["changed" if group == val_group else group for group in groups]
    with pytest.raises(ValueError, match="Resume validation source groups mismatch"):
        obs_training.train("cordvip", CordViP, resume_args)
    store.attrs["source_episodes"] = groups
    store.attrs["source_zarr_path"] = str(derived)
    with pytest.raises(ValueError, match="Resume data source mismatch"):
        obs_training.train("cordvip", CordViP, resume_args)
    assert [path.read_bytes() for path in artifacts] == before


def test_contact_requires_aligned_points(contact_data):
    store = zarr.open_group(str(contact_data), mode="a")
    del store["data/contact_map"]
    store.create_dataset("data/contact_map", data=np.zeros((12, 1024), dtype=np.int64))
    with pytest.raises(ValueError, match="contact_map must be floating point"):
        CordViPPretrainDataset(contact_data, contact_loss_weight=1)

    del store["data/contact_map"]
    store.create_dataset("data/contact_map", data=np.zeros((12, 63), dtype=np.float32))
    with pytest.raises(ValueError, match="Contact labels must match the object point slots"):
        CordViPPretrainDataset(contact_data, contact_loss_weight=1)

    points = store["data/object_pointcloud"][:, :63]
    hand = store["data/hand_pointcloud"][:]
    points[1] = points[0, ::-1]
    hand[1] = hand[0]
    del store["data/object_pointcloud"]
    store.create_dataset("data/object_pointcloud", data=points)
    store["data/hand_pointcloud"][:] = hand
    output = contact_data.parent / "cordvip" / contact_data.name
    generate_contacts(contact_data)
    labelled = zarr.open_group(str(output), mode="r")
    labels = labelled["data/contact_map"][:]
    assert labels.shape == (12, 63)
    np.testing.assert_array_equal(labelled["data/object_pointcloud"][:], points)
    np.testing.assert_allclose(labels[1], labels[0, ::-1], rtol=1e-6)

    dataset = CordViPPretrainDataset(output, contact_loss_weight=1, obs_horizon=1)
    sample = dataset[1]
    torch.testing.assert_close(sample["contact_map"], torch.from_numpy(labels[1:2]))
    model = CordViP(dict.fromkeys(dataset.sensor_modalities, "identity"), n_obs_steps=1)
    batch = {
        "obs": {key: value.unsqueeze(0) for key, value in sample["obs"].items()},
        "contact_map": sample["contact_map"].unsqueeze(0),
    }
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_contact_matches_dense_formula():
    rng = np.random.default_rng(32)
    objects = rng.normal(size=(23, 3))
    objects[-1] = [20, 0, 0]
    normals = rng.normal(size=objects.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    hand = rng.normal(size=(19, 3))
    hand[0] = objects[0]
    differences = hand[None] - objects[:, None]
    distance = np.linalg.norm(differences, axis=-1)
    alignment = np.abs(np.einsum("ijk,ik->ij", differences, normals)) / (distance + 1e-8)
    minimum = (distance * np.exp(1 - alignment)).min(axis=1)
    expected = 2 / (1 + np.exp(10 * minimum))
    contacts = compute_contact_map(objects, normals, hand, point_chunk_size=7)
    np.testing.assert_allclose(contacts, expected, atol=1e-15)
    np.testing.assert_allclose(contacts[-1], expected[-1], rtol=1e-12, atol=0)
    batched = batched_contact_maps(
        np.stack((objects, objects[::-1])),
        np.stack((normals, normals[::-1])),
        np.stack((hand, hand)),
        "cpu",
    )
    np.testing.assert_array_equal(batched, np.stack((contacts, contacts[::-1])).astype(np.float32))


def test_contact_reads_metadata_and_points_from_same_resolved_source(contact_data, tmp_path, monkeypatch):
    source = zarr.open_group(str(contact_data), mode="a")
    source.create_dataset("meta/episode_seeds", data=[32, 33, 34])
    alternate = tmp_path / "alternate.zarr"
    shutil.copytree(contact_data, alternate)
    other = zarr.open_group(str(alternate), mode="a")
    other["meta/episode_seeds"][:] = [42, 43, 44]
    other["meta/episode_ends"][:] = [3, 8, 12]
    alias = tmp_path / "latest.zarr"
    alias.symlink_to(contact_data, target_is_directory=True)
    init = EpisodeStore.__init__

    def move_alias(store, *args, **kwargs):
        init(store, *args, **kwargs)
        alias.unlink()
        alias.symlink_to(alternate, target_is_directory=True)

    monkeypatch.setattr(EpisodeStore, "__init__", move_alias)
    output = tmp_path / "prepared.zarr"
    generate_contacts(alias, output)
    labelled = zarr.open_group(str(output), mode="r")
    assert labelled.attrs["source_zarr_path"] == str(contact_data)
    np.testing.assert_array_equal(labelled["meta/episode_seeds"][:], [32, 33, 34])
    np.testing.assert_array_equal(labelled["meta/episode_ends"][:], [4, 8, 12])
    np.testing.assert_array_equal(labelled["data/object_pointcloud"][:], source["data/object_pointcloud"][:])


def test_disabled_contact_allows_other_point_counts(contact_data):
    store = zarr.open_group(str(contact_data), mode="a")
    del store["data/contact_map"]
    del store["data/object_pointcloud"]
    store.create_dataset("data/object_pointcloud", data=store["data/hand_pointcloud"][:])
    dataset = CordViPPretrainDataset(contact_data, contact_loss_weight=0)
    assert "contact_map" not in dataset.episode_store.keys()
    model = CordViP(
        dict.fromkeys(dataset.sensor_modalities, "identity"), contact_loss_weight=0,
        arm_loss_weight=0,
    )
    assert not hasattr(model, "contact_head")
    assert not hasattr(model, "arm_head")
    batch = {"obs": {key: value.unsqueeze(0) for key, value in dataset[0]["obs"].items()}}
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "hand"}


@pytest.mark.parametrize("mode,benchmark", [(None, False), ("default", True)])
def test_training_options_and_resume(tmp_path, small_dp3_config, monkeypatch, mode, benchmark):
    overrides = [] if mode is None else [f"training.compile_mode={mode}"]
    cfg = small_dp3_config(
        tmp_path / "run", "training.use_compile=true",
        f"training.cudnn_benchmark={str(benchmark).lower()}", *overrides,
    )
    expected = "reduce-overhead" if mode is None else mode
    compiled = []
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = not benchmark
    torch.set_float32_matmul_precision("medium")
    with closing(build_trainer(cfg)) as trainer:
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark == benchmark
        assert torch.get_float32_matmul_precision() == "medium"
        monkeypatch.setattr(trainer.model, "compile_backbone", lambda *, mode: compiled.append(mode))
        trainer.compile_models()
        trainer.train(max_updates=1)
        assert torch.get_float32_matmul_precision() == "highest"
    with closing(prepare_resume(tmp_path / "run")) as resumed:
        assert resumed.cfg.training.compile_mode == expected
        assert resumed.cfg.training.cudnn_benchmark == benchmark
        monkeypatch.setattr(resumed.model, "compile_backbone", lambda *, mode: compiled.append(mode))
        resumed.compile_models()
    assert compiled == [expected, expected]


def test_training_failure_closes_trainer():
    closed = []

    def fail():
        raise ValueError("training failed")

    trainer = SimpleNamespace(
        compile_models=fail,
        close=lambda: closed.append(True),
    )
    with pytest.raises(ValueError, match="training failed"):
        run_trainer(trainer)
    assert closed == [True]


# Resume paths


def test_resume_uses_saved_zarr_and_groups(tmp_path, small_dp3_checkpoint, make_demo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = load_checkpoint(small_dp3_checkpoint)
    with pytest.raises(SystemExit) as rejected:
        resume.main(["--run", "dp3_run", "--zarr-path", "other.zarr"])
    assert rejected.value.code == 2

    cfg = OmegaConf.create(saved.effective_config)
    other_data = make_demo("other")
    cfg.dataset.zarr_path = str(other_data)
    with pytest.raises(ValueError, match="Resume Zarr path mismatch"):
        build_trainer(cfg, checkpoint=saved)
    other_store = zarr.open_group(str(other_data), mode="a")
    other_store.attrs["source_zarr_path"] = saved.effective_config["training_data"]["source"]
    with pytest.raises(ValueError, match="Resume Zarr path mismatch"):
        build_trainer(cfg, checkpoint=saved)

    ends = zarr.open_group(saved.effective_config["dataset"]["zarr_path"], mode="a")["meta/episode_ends"]
    ends.resize(2)
    ends[:] = [6, 12]
    with pytest.raises(ValueError, match="Resume train source groups mismatch"):
        resume.main(["--run", "dp3_run", "--max-updates", "1"])
    ends.resize(1)
    ends[:] = [12]
    assert load_checkpoint(small_dp3_checkpoint).global_step == 1

    layout = saved.effective_config["episode_layout"]
    saved.effective_config["episode_layout"] = {**layout, "episode_ends": [11]}
    save_checkpoint(small_dp3_checkpoint, saved)
    with pytest.raises(ValueError, match="Resume episode layout mismatch"):
        resume.main(["--run", "dp3_run", "--max-updates", "1"])
    assert load_checkpoint(small_dp3_checkpoint).global_step == 1

    saved.effective_config["episode_layout"] = layout
    save_checkpoint(small_dp3_checkpoint, saved)
    resume.main(["--run", "dp3_run", "--max-updates", "1"])
    assert Path.cwd() == tmp_path
    checkpoint = load_checkpoint(small_dp3_checkpoint)
    assert checkpoint.global_step == 2
    assert checkpoint.effective_config["workspace"]["output_dir"] == str(tmp_path / "dp3_run")
    assert checkpoint.effective_config["dataset"]["zarr_path"] == str(tmp_path / "demo.zarr")
    assert checkpoint.effective_config["training_data"] == saved.effective_config["training_data"]
    assert checkpoint.effective_config["episode_layout"] == layout


# Policy inference


def backend_settings():
    return (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.get_float32_matmul_precision(),
    )


def test_set_seed_only_seeds_rng(monkeypatch):
    calls = []
    for module, name, label in (
        (random, "seed", "python"), (np.random, "seed", "numpy"), (torch, "manual_seed", "torch"),
    ):
        original = getattr(module, name)

        def record(seed, original=original, label=label):
            calls.append(label)
            return original(seed)

        monkeypatch.setattr(module, name, record)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    samples = []
    for _ in range(2):
        calls.clear()
        set_seed(73)
        assert backend_settings() == (True, False, "high")
        assert calls == ["python", "numpy", "torch"]
        samples.append((random.random(), np.random.random(4), torch.rand(4)))
    assert samples[0][0] == samples[1][0]
    np.testing.assert_array_equal(samples[0][1], samples[1][1])
    torch.testing.assert_close(samples[0][2], samples[1][2], rtol=0, atol=0)


@pytest.mark.parametrize("api,precision", [("restore", "high"), ("deployment", "medium")])
@pytest.mark.parametrize("case", ["raw", "ema", "missing_ema", "strict_error"])
def test_restoration_inherits_host_settings(small_dp3_checkpoint, tmp_path, api, case, precision):
    checkpoint = load_checkpoint(small_dp3_checkpoint)
    # Make weight selection and FP32 conversion observable.
    for state in (checkpoint.model_state, checkpoint.ema_model_state):
        for key, value in state.items():
            if value.is_floating_point():
                state[key] = value.double()
    parameter = next(key for key in checkpoint.ema_model_state if not key.startswith("normalizer."))
    checkpoint.ema_model_state[parameter] = checkpoint.ema_model_state[parameter] + 0.125
    if case == "missing_ema":
        checkpoint.ema_model_state = None
    elif case == "strict_error":
        checkpoint.model_state.pop(parameter)
    source = save_checkpoint(tmp_path / "restore.pt", checkpoint)
    weights = "ema" if case in {"ema", "missing_ema"} else "raw"
    state = checkpoint.ema_model_state if weights == "ema" else checkpoint.model_state
    cfg = checkpoint.effective_config
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision(precision)
    rng = get_rng_state()

    def restore():
        if api == "restore":
            return restore_policy(cfg, state, "cpu")
        return load_policy(source, weights=weights, device="cpu", seed=73, inference_steps=2)

    if case in {"missing_ema", "strict_error"}:
        error, message = (
            (ValueError, "EMA weights are absent") if case == "missing_ema"
            else (RuntimeError, "Missing key")
        )
        with pytest.raises(error, match=message):
            restore()
    else:
        restored = restore()
    assert backend_settings() == (True, False, precision)
    after = get_rng_state()
    assert rng["python"] == after["python"]
    assert rng["numpy"][0] == after["numpy"][0] and rng["numpy"][2:] == after["numpy"][2:]
    np.testing.assert_array_equal(rng["numpy"][1], after["numpy"][1])
    torch.testing.assert_close(rng["torch"], after["torch"])
    if case in {"missing_ema", "strict_error"}:
        return

    agent = restored if api == "restore" else restored.agent
    assert all(not module.training for module in agent.modules())
    expected_state = {key: value.float() if value.is_floating_point() else value for key, value in state.items()}
    torch.testing.assert_close(agent.state_dict(), expected_state, rtol=0, atol=0)
    assert all(value.dtype == torch.float32 for value in agent.state_dict().values() if value.is_floating_point())
    if api == "restore":
        return

    dataset = hydra.utils.instantiate(cfg["dataset"], _convert_="all")
    observation = {key: value.numpy() for key, value in dataset[0]["obs"].items()}
    tensors = prepare_observations(
        observation, normalization_dims=agent.normalization_dims,
        n_obs_steps=agent.n_obs_steps, rgb_preprocessing={}, device="cpu",
    )
    expected = predict_actions(agent, tensors, inference_steps=2, generator=torch.Generator().manual_seed(73))
    assert restored.info.weights == weights
    assert restored.info.control_action_dim == 19
    assert restored.info.action_mode == "joint"
    assert restored.info.data_processing == cfg["data_processing"]
    generator_state = restored.generator.get_state()
    restored.warmup(observation)
    torch.testing.assert_close(restored.generator.get_state(), generator_state)
    actual = restored.predict(observation)
    restored.reset_episode()
    np.testing.assert_array_equal(restored.predict(observation), actual)
    restored.close()
    np.testing.assert_array_equal(actual, expected)
    assert backend_settings() == (True, False, precision)


# Simulation evaluation


class SmallPolicy(torch.nn.Module):
    n_obs_steps = 3
    n_action_steps = 2
    future_steps = 3
    control_action_dim = 19
    normalization_dims = {"joint_state": 19, "action": 19}
    requires_prev_action = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def predict_action(self, obs, *, generator, prev_action=None, **kwargs):
        self.calls.append((obs["joint_state"].clone(), prev_action, generator.initial_seed()))
        return {"future_actions": torch.tensor([10., 20., 30.])[None, :, None].expand(1, 3, 19)}


class SmallEnv:
    dt = 0.05

    def __init__(self):
        self.state = np.zeros(19, dtype=np.float32)
        self.close_count = 0
        self.step_error = False

    def reset(self, *, seed):
        self.action_cnt = 0
        self.state.fill(0)
        return {"joint_state": self.state}, {}

    def step(self, action):
        if self.step_error:
            raise ValueError("step failed")
        self.action_cnt += 1
        self.state.fill(self.action_cnt)
        done = self.action_cnt == 3
        return {"joint_state": self.state}, 0, done, False, {
            "success_condition": self.action_cnt >= 2,
            "success": done,
            "executed_joint_action": action + 1,
        }

    def get_video(self):
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    def close(self):
        self.close_count += 1


@pytest.fixture
def environments(monkeypatch):
    created = []

    def make_env(self, *, record_video):
        env = SmallEnv()
        created.append(env)
        return env

    monkeypatch.setattr(SimRunner, "make_env", make_env)
    return created


def make_runner(**kwargs):
    return SimRunner("test", 3, ["joint_state"], {}, 0.05, **kwargs)


def test_observation_history_owns_snapshots():
    runner = make_runner()
    value = np.zeros(19, dtype=np.float32)
    for step, expected in enumerate(([0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3])):
        value.fill(step)
        runner.update_obs({"joint_state": value, "unused": np.zeros(1)})
        value.fill(-1)
        history = runner.get_stacked_obs()
        assert set(history) == {"joint_state"}
        np.testing.assert_array_equal(history["joint_state"][:, 0], expected)
        history["joint_state"][:] = -2
        np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"][:, 0], expected)
    runner.reset()
    value.fill(9)
    runner.update_obs({"joint_state": value})
    np.testing.assert_array_equal(runner.get_stacked_obs()["joint_state"][:, 0], [9, 9, 9])


def test_rollout_uses_executed_action_and_stops_mid_chunk(environments):
    agent = SmallPolicy()
    results = list(make_runner(env_rebuild_interval=1).run(
        agent, seeds=[7, 8], inference_steps=2, action_steps=2,
    ))
    assert [(r["seed"], r["success"], r["steps"], r["total_steps"]) for r in results] == [
        (7, True, 2, 3), (8, True, 2, 3),
    ]
    assert [env.close_count for env in environments] == [1, 1]
    for offset, seed in ((0, 7), (2, 8)):
        assert agent.calls[offset][1] is None
        torch.testing.assert_close(agent.calls[offset + 1][1], torch.full((1, 19), 21.))
        np.testing.assert_array_equal(agent.calls[offset + 1][0][0, :, 0], [0, 1, 2])
        assert agent.calls[offset][2] == agent.calls[offset + 1][2] == seed


def test_stopping_rollout_iteration_closes_env(environments):
    results = make_runner().run(SmallPolicy(), seeds=[7], inference_steps=2, action_steps=2)
    assert next(results)["success"]
    results.close()
    assert environments[0].close_count == 1


def test_fast_rollout_preserves_history_and_stops_in_skipped_steps(monkeypatch):
    monkeypatch.setitem(sys.modules, "dexmani_sim", SimpleNamespace(TACTILE_OBSERVATION_KEYS=("finger_tactile_force",)))

    class CountingEnv(SmallEnv):
        def __init__(self):
            super().__init__()
            self.captured_steps = []

        def step(self, action):
            self.captured_steps.append(self.action_cnt + 1)
            return super().step(action)

        def step_physics(self, action, *, collect_tactile):
            assert not collect_tactile
            _, reward, done, truncated, info = super().step(action)
            return None, reward, done, truncated, info

    calls = []
    for skip in (False, True):
        env = CountingEnv()
        runner = SimRunner("test", 1, ["joint_state"], {}, 0.05, skip_intermediate_observations=skip)
        monkeypatch.setattr(runner, "make_env", lambda *, record_video: env)
        agent = SmallPolicy()
        result, = runner.run(agent, seeds=[7], inference_steps=2, action_steps=2)
        assert result["success"] and result["steps"] == 2 and result["total_steps"] == 3
        assert env.captured_steps == ([2] if skip else [1, 2, 3])
        assert env.close_count == 1
        calls.append(agent.calls)
    torch.testing.assert_close(calls[0], calls[1], rtol=0, atol=0)


def test_step_error_closes_env(environments):
    results = make_runner().run(SmallPolicy(), seeds=[7, 8], inference_steps=2, action_steps=2)
    next(results)
    environments[0].step_error = True
    with pytest.raises(ValueError, match="step failed"):
        next(results)
    assert environments[0].close_count == 1


def test_control_period_mismatch_closes_env(monkeypatch):
    created = []

    class WrongPeriodEnv(SmallEnv):
        def __init__(self, **kwargs):
            assert kwargs["observation_keys"] == ("joint_state",)
            super().__init__()
            created.append(self)
            self.dt = 0.1

    monkeypatch.setitem(sys.modules, "dexmani_sim", SimpleNamespace(resolve_env_class=lambda task: WrongPeriodEnv))
    with pytest.raises(ValueError, match="Control period mismatch"):
        make_runner().make_env(record_video=False)
    assert created[0].close_count == 1


@pytest.mark.parametrize("change", [None, "num_points", "size"])
def test_dexjoco_point_cloud_observation_processing(monkeypatch, change):
    processing = {"proprio": {"layout": "wxyz"}, "rgb": {"size": [224, 224]}, "point_cloud": {"num_points": 4096}}
    actual = copy.deepcopy(processing)
    if change == "num_points":
        actual["point_cloud"]["num_points"] = 1024
    elif change == "size":
        actual["rgb"]["size"] = [128, 128]
    env = SmallEnv()
    env.data_processing = actual
    monkeypatch.setitem(sys.modules, "dexjoco.policy_env", SimpleNamespace(DexJoCoEnv=lambda *args, **kwargs: env))
    runner = DexJoCoRunner("test", 1, ["proprio", "point_cloud"], {}, env.dt, data_processing=processing)
    if change is None:
        assert runner.make_env(record_video=False) is env
        env.close()
    else:
        with pytest.raises(ValueError, match=f"DexJoCo observation processing mismatch.*{change}"):
            runner.make_env(record_video=False)
    assert env.close_count == 1


@pytest.fixture
def checkpoints(tmp_path, monkeypatch, environments):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")

    def restore(*args):
        assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
        assert torch.get_float32_matmul_precision() == "highest"
        return SmallPolicy()

    monkeypatch.setattr(rollout, "restore_policy", restore)
    monkeypatch.setattr(SimRunner, "load_seed_pool", staticmethod(lambda task: list(range(100, 110))))
    cfg = {
        "eval": {"use_ema": False, "inference_steps": 2},
        "env_runner": {"_target_": "dexmani_policy.env_runner.sim_runner.SimRunner", "task_name": "test", "env_kwargs": {}},
        "control_dt_s": 0.05, "normalization_dims": SmallPolicy.normalization_dims, "dataset": {}, "action_key": "action",
    }
    return [save_checkpoint(tmp_path / f"{step}.pt", Checkpoint(step, {}, {}, {}, cfg)) for step in (1, 2)]


@pytest.mark.parametrize("skip_intermediate_observations", [False, True])
@pytest.mark.parametrize("default_kwargs_index", [0, 1])
@pytest.mark.parametrize("omit_env_kwargs", [False, True])
def test_selection_and_independent_test(
    checkpoints, tmp_path, skip_intermediate_observations, default_kwargs_index, omit_env_kwargs,
):
    checkpoint_path = checkpoints[default_kwargs_index]
    checkpoint = load_checkpoint(checkpoint_path)
    env_runner = checkpoint.effective_config["env_runner"]
    if omit_env_kwargs:
        del env_runner["env_kwargs"]
    else:
        env_runner["env_kwargs"] = None
    save_checkpoint(checkpoint_path, checkpoint)
    path = selection.run_pipeline(
        checkpoints, selection_episodes=2, test_episodes=3, seed=5, device="cpu", output_dir=tmp_path / "eval",
        skip_intermediate_observations=skip_intermediate_observations,
    )
    record = json.loads(path.read_text())
    assert record["global_step"] == 2
    assert record["skip_intermediate_observations"] is skip_intermediate_observations
    assert set(record["selection_seeds"]).isdisjoint(record["test_seeds"])
    for candidate in record["candidates"]:
        result = json.loads((path.parent / candidate["result"]).read_text())
        assert result["seeds"] == record["selection_seeds"]
        assert result["skip_intermediate_observations"] is skip_intermediate_observations
        assert result["completed"] and result["statistics"]["n_success"] == 2
    result = json.loads((path.parent / record["test_result"]).read_text())
    assert result["seeds"] == record["test_seeds"] and result["completed"]
    assert result["skip_intermediate_observations"] is skip_intermediate_observations
    assert (result["weights"], result["inference_steps"], result["action_steps"]) == ("raw", 2, 2)
    candidates = [
        {"global_step": step, "statistics": {"success_rate": rate, "avg_steps": steps}}
        for rate, steps, step in [(0, None, 100), (0.5, 1, 100), (1, 5, 100), (1, 2, 1), (1, 2, 2)]
    ]
    assert sorted(candidates, key=selection.candidate_score) == candidates


def test_selection_rejects_different_env_kwargs(checkpoints, tmp_path):
    checkpoint = load_checkpoint(checkpoints[1])
    checkpoint.effective_config["env_runner"]["env_kwargs"] = {"randomize": True}
    save_checkpoint(checkpoints[1], checkpoint)
    output = tmp_path / "different_env"
    with pytest.raises(ValueError, match="Candidates must use the same env_kwargs"):
        selection.run_pipeline(checkpoints, device="cpu", output_dir=output)
    assert not output.exists()


def test_fixed_checkpoint_uses_only_test_seeds(checkpoints, tmp_path):
    output = tmp_path / "fixed_eval"
    with pytest.raises(ValueError, match="at least two distinct checkpoints"):
        selection.run_pipeline(checkpoints[:1], output_dir=output)
    _, path = rollout.evaluate_policy(checkpoints[0], episodes=3, seed=5, device="cpu", output_dir=output)
    assert path == output / "result_details.json"
    result = json.loads(path.read_text())
    test_seeds, _ = selection.split_evaluation_seeds(range(100, 110), 5, 3, 2)
    assert [row["seed"] for row in result["episodes"]] == test_seeds
    assert result["control_dt_s"] == SmallEnv.dt
    assert result["completed"] and not (output / "candidates").exists()


@pytest.mark.parametrize("runner,action_key", [(SimRunner, "action_rotvec"), (DexJoCoRunner, "action")])
def test_eval_rejects_unsupported_action_before_loading_environment(checkpoints, tmp_path, runner, action_key):
    checkpoint = load_checkpoint(checkpoints[0])
    checkpoint.effective_config["env_runner"]["_target_"] = f"{runner.__module__}.{runner.__name__}"
    checkpoint.effective_config["action_key"] = "action_rotvec" if runner is DexJoCoRunner else "action"
    save_checkpoint(checkpoints[0], checkpoint)
    checkpoint.effective_config["action_key"] = action_key
    source = save_checkpoint(tmp_path / "wrong_action.pt", checkpoint)
    output = tmp_path / "eval"
    error = f"{runner.__name__} does not support {action_key}"
    with pytest.raises(ValueError, match=error):
        rollout.evaluate_policy(source, device="cpu", output_dir=output)
    with pytest.raises(ValueError, match=error):
        selection.run_pipeline([checkpoints[0], source], device="cpu", output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["video", "write"])
def test_evaluation_failure_closes_env_and_keeps_published_episodes(checkpoints, environments, tmp_path, monkeypatch, failure):
    output = tmp_path / "failed_eval"
    if failure == "video":
        def fail_video(frames, fps, path):
            assert fps == 1 / SmallEnv.dt
            if path.stem == "episode_8":
                raise OSError("video failed")
            return str(path)

        monkeypatch.setattr("dexmani_policy.env_runner.video.save_episode_video", fail_video)
    else:
        write_json = rollout.write_json

        def fail_write(path, record):
            if len(record["episodes"]) == 2:
                raise OSError("write failed")
            write_json(path, record)

        monkeypatch.setattr(rollout, "write_json", fail_write)
    with pytest.raises(OSError, match=f"{failure} failed"):
        rollout.evaluate_policy(checkpoints[0], seeds=[7, 8], device="cpu", video=failure == "video", output_dir=output)
    assert environments[0].close_count == 1
    record = json.loads((output / "result_details.json").read_text())
    assert not record["completed"]
    assert record["statistics"]["n_success"] == 1
    assert [row["seed"] for row in record["episodes"]] == [7]
    expected_video = str(Path("videos/test/episode_7.mp4")) if failure == "video" else None
    assert record["episodes"][0]["video"] == expected_video
