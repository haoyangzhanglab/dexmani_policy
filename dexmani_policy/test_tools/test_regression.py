import copy
import hashlib
import importlib
import json
import logging
import math
import random
import shutil
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dexmani_policy.agents.action_decoders.backbones import ditx, sat_action_io
from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.diffusion import Diffusion
from dexmani_policy.agents.action_decoders.flow import ConsistencyFlow, RectifiedFlow
from dexmani_policy.agents.action_decoders.ode import ODE
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, load_codebook
from dexmani_policy.agents.obs_encoder.pointcloud import point_transformer
from dexmani_policy.agents.obs_encoder.rgb.resnet import convert_group_norm
from dexmani_policy.agents.policies import r3d
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.common import config as common_config, io_util
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from dexmani_policy.common.inference import predict_actions, prepare_observations, resolve_inference_settings, restore_policy
from dexmani_policy.common.io_util import close_resources
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.common.pytorch_util import get_rng_state, set_rng_state, set_seed
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.pretrain.afro import AFROPretrainDataset
from dexmani_policy.datasets.pretrain.cordvip import CordViPPretrainDataset
from dexmani_policy.datasets.preprocess.contact_map import generate as generate_contacts
from dexmani_policy.datasets.sampler import SequenceSampler
from dexmani_policy.datasets.variants.ode import ODEDataset
from dexmani_policy.datasets.variants.r3d import R3DDataset
from dexmani_policy.datasets.variants.tactile import AdapTacDataset
from dexmani_policy.deployment import load_policy
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation import rollout, selection
from dexmani_policy.training import resume, trainer as policy_training
from dexmani_policy.training.config import load_config, validate_config
from dexmani_policy.training.data import (
    build_dataset, build_train_loader, fit_action_normalizer, inherit_training_data, use_policy_split,
)
from dexmani_policy.training.initialization import build_policy
from dexmani_policy.training.optimization import clip_gradients
from dexmani_policy.training.pretrain.hand_prior.vqvae import train as train_vq
from dexmani_policy.training.pretrain.obs import common as obs_training
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
    def create(name="demo", *, auxiliary=True, action_dim=19):
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
        store.create_dataset("meta/episode_ends", data=np.array([12], dtype=np.int64))
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
    for recipe, override in (("dp3", "agent.num_inference_steps=1"), ("dp3", "eval.inference_steps=1"),
                             ("adaptac", "agent.force_num_inference_steps=1")):
        with pytest.raises(ValueError, match="leading DDIM"):
            load_config(recipe, [override])
    with pytest.raises(ValueError, match="leading DDIM"):
        load_policy(small_dp3_checkpoint, inference_steps=1, device="cpu")
    with pytest.raises(ValueError, match="leading DDIM"):
        rollout.evaluate_policy(small_dp3_checkpoint, inference_steps=1, device="cpu")
    with pytest.raises(ValueError, match="leading DDIM"):
        selection.run_pipeline([small_dp3_checkpoint], inference_steps=1, device="cpu")
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


def test_r3d_limits_config_and_restoration(tmp_path, make_demo, monkeypatch):
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
        "training.device=cpu", "training.use_compile=false", "training.loop.total_train_steps=3",
        "dataloader.num_workers=0", "dataloader.batch_size=2", "horizon=4", "n_obs_steps=1", "n_action_steps=4",
        "agent.pc_encoder_config.embed_dim=16", "agent.embedding_dim=16", "agent.num_heads=2", "agent.depth=1",
        "agent.state_out_dim=8", "agent.mlp_dim=32", "agent.timestep_embed_dim=8",
    ])
    with pytest.raises(ValueError, match=r"normalization.point_cloud must use limits"):
        load_config("r3d", ["normalization.point_cloud=mean_std"])
    with closing(build_trainer(cfg)) as trainer:
        trainer.train(max_updates=1)
    latest = directory / "checkpoints/latest.pt"
    checkpoint = load_checkpoint(latest)
    assert checkpoint.effective_config["normalization"]["point_cloud"] == "limits"
    with closing(prepare_resume(directory)) as trainer:
        assert trainer.global_step == 1
        torch.testing.assert_close(trainer.model.normalizer["point_cloud"].scale,
                                   checkpoint.model_state["normalizer.fields.point_cloud.scale"])
    cfg.workspace.output_dir = str(tmp_path / "warm")
    cfg.initialization.policy = str(latest)
    with closing(build_trainer(cfg)) as trainer:
        assert trainer.global_step == 0
    for weights in ("raw", "ema"):
        assert load_policy(latest, weights=weights, device="cpu").agent.normalization_dims["point_cloud"] == 6
    cfg.normalization.point_cloud = "mean_std"
    with pytest.raises(ValueError, match="point_cloud must use limits"):
        build_trainer(cfg)
    cfg.normalization.point_cloud = "limits"
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


def test_ode_refuses_real_attempt_only_store(make_demo):
    path = make_demo()
    zarr.open_group(str(path), mode="a").attrs.update(
        format="dexmani.real.canonical", source_episodes=["capture"], source_row_ranges=[[0, 12]],
    )
    with pytest.raises(ValueError, match="Real ODE requires action_ode"):
        ODEDataset(path, sensor_modalities=["joint_state"], obs_horizon=1)


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
    data.attrs.update(
        format="dexmani.real.canonical", source_episodes=sources,
        source_row_ranges=[[0, 2], [3, 5], [0, 2], [0, 2], [0, 2], [0, 2]],
    )
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
    _, affine, records = load_codebook(run / "codebook.npz")
    record, = records
    assert record["source"] == str(path)
    assert set(record["train_source_episodes"]) == chosen
    assert set(record["val_source_episodes"]) == held_out
    rows = np.repeat(dataset.train_mask, 2)
    hand = torch.from_numpy(data["data/action"][:][rows, 7:])
    torch.testing.assert_close(affine["scale"], 2 / (hand.amax(0) - hand.amin(0)))
    inherit_training_data(policy, dataset, records, "codebook.npz")
    outside_budget = set(sources) - chosen - held_out
    for group, key in [(next(iter(outside_budget)), "train_source_episodes"),
                       (next(iter(held_out)), "train_source_episodes"),
                       (next(iter(outside_budget)), "val_source_episodes")]:
        with pytest.raises(ValueError, match="same-source prior exceeds"):
            inherit_training_data(policy, dataset, [dict(record, **{key: [group]})], "leaky.pt")
    inherit_training_data(policy, dataset, [dict(record, source="/external.zarr")], "external.pt")
    inherit_training_data(policy, dataset, [], "unknown.pt")
    assert policy.training_data[-1] == {"source": None, "artifact": "unknown.pt"}


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
    backbone = backbone_class(
        horizon=4, action_dim=19, n_obs_steps=1, obs_token_dim=8, hidden_dim=16,
        n_head=2, n_layers=2, timestep_embed_dim=8, p_drop_attn=0, action_io=action_io,
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
    store = EpisodeStore(path)
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


def test_ode_tactile_and_afro_temporal_targets(segments):
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
            expected = [start + 1, start + 2, start + 2] if offset == 0 else [start + 2] * 3
            np.testing.assert_array_equal(sample["future_force"][:, 0, 0], expected)
            np.testing.assert_array_equal(sample["future_force_valid"], [True, offset == 0, False])
        for offset, rows in enumerate([[start, start + 1], [start, start + 2], [start + 1, start + 2], [start + 2] * 2]):
            np.testing.assert_array_equal(afro[episode * 4 + offset]["obs"]["point_cloud"], fields["point_cloud"][rows, :, :3])


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


# Policy initialization


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
        np.savez(
            path, sorted_hand_poses=poses, hand_normalizer_scale=scale.numpy(), hand_normalizer_offset=offset.numpy(),
            training_data=json.dumps([{"source": dataset.episode_store.source_path, "train_source_episodes": [0]}]),
        )
        cfg.initialization.codebook = str(path)
    else:
        from dexmani_policy.agents.hand_prior.dexlatent import DexLatentHandPrior

        prior = DexLatentHandPrior()
        path = tmp_path / "prior.pt"
        torch.save({"autoencoders": {"xarm7_xhand_right." + key: value for key, value in prior.state_dict().items()}}, path)
        cfg.initialization.dexlatent = str(path)
        affine = prior.initialize_weights(path)
        scale, offset = affine["scale"], affine["offset"]
    model = build_policy(cfg, dataset, {"action": 19, "joint_state": 19})
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
    sat = build_policy(sat_cfg, dataset, {"action": 19, "joint_state": 19})
    with torch.no_grad():
        sat.normalizer["action"].scale.fill_(9)
        sat.obs_encoder.rgb_encoder.weight.zero_()
    sat_cfg.agent.rgb_backbone_config.tune_mode = "full"
    torch.testing.assert_close(model.state_dict(), before)
    assert cfg.agent == before_config


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
    torch.save({
        "encoder_state": encoder.state_dict(),
        "normalizer_state": inherited.state_dict(),
        "normalization": dict.fromkeys(fields, mode),
        "training_data": [{"source": dataset.episode_store.source_path, "train_source_episodes": [0]}],
    }, source)
    cfg.initialization.encoder = str(source)
    model = build_policy(cfg, dataset, dims)
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


def test_resume_and_warm_start_state_boundaries(tmp_path, small_dp3_config, monkeypatch):
    directory = tmp_path / "source"
    first_source, resumed_source = "gpu:/snapshots/first", "gpu:/snapshots/resumed"
    monkeypatch.setenv("DEXMANI_POLICY_REMOTE_SOURCE", first_source)
    with closing(build_trainer(small_dp3_config(directory))) as trainer:
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
    assert OmegaConf.load(directory / "config.yaml").remote_source == first_source

    warm_cfg = small_dp3_config(tmp_path / "warm", f"initialization.policy={first_checkpoint}")
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
    from dexmani_policy.training.initialization import inherit_rgb_architecture, resolve_rgb_architecture

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
    backbone = backbone_class(
        horizon=4, action_dim=19, n_obs_steps=1, obs_token_dim=8,
        hidden_dim=16, n_head=2, n_layers=2, timestep_embed_dim=8,
        p_drop_attn=0, action_io=io,
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


def test_dit_branch_does_not_construct_ditx_output(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SAT must own its readout")

    monkeypatch.setattr(ditx, "DiTXOutput", forbidden)
    module = importlib.reload(sat_action_io)
    io = module.SATActionIO(4, 16)
    backbone = DiT(
        horizon=4, action_dim=19, n_obs_steps=1, obs_token_dim=8,
        hidden_dim=16, n_head=2, n_layers=1, action_io=io,
    )
    assert backbone(torch.randn(2, 4, 19), 0.5, torch.randn(2, 3, 8)).shape == (2, 4, 19)


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
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_publish_failure_preserves_exception_and_traceback(tmp_path, monkeypatch, caplog, source, cleanup_fails):
    target = tmp_path / "result"
    target.write_text("old")
    primary = ValueError("writer failed") if source == "writer" else OSError("replace failed")
    secondary = PermissionError("cleanup failed")
    original_traceback = None
    calls = []
    unlink = Path.unlink

    def fail(*args):
        nonlocal original_traceback
        try:
            raise primary
        except BaseException:
            original_traceback = primary.__traceback__
            raise

    def cleanup(path, *args, **kwargs):
        if path == temporary:
            calls.append(path)
            if cleanup_fails:
                raise secondary
        return unlink(path, *args, **kwargs)

    if source == "replace":
        monkeypatch.setattr(io_util.os, "replace", fail)
    monkeypatch.setattr(Path, "unlink", cleanup)
    with caplog.at_level(logging.WARNING, logger=io_util.__name__):
        with pytest.raises(type(primary)) as caught:
            with io_util.atomic_path(target) as temporary:
                temporary.write_text("new")
                if source == "writer":
                    fail()
    assert caught.value is primary
    traceback = caught.value.__traceback__
    while traceback is not None and traceback is not original_traceback:
        traceback = traceback.tb_next
    assert traceback is original_traceback
    assert target.read_text() == "old"
    assert calls == [temporary]
    assert temporary.exists() == cleanup_fails
    if cleanup_fails:
        record, = [record for record in caplog.records if record.name == io_util.__name__]
        assert str(temporary) in record.getMessage()
        assert record.exc_info[1] is secondary
        assert "PermissionError: cleanup failed" in caplog.text
    else:
        assert not caplog.records


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
    cfg = load_config("pretrain/hand_prior/vqvae", [
        f"zarr_path={make_demo()}", f"workspace.output_dir='{value}'",
        "vq_vae.device=cpu", "vq_vae.num_epochs=1", "vq_vae.save_epochs=1",
        "vq_vae.batch_size=12", "vq_vae.warmup_steps=0", "vq_vae.hidden_dim=16",
        "vq_vae.latent_dim=8", "vq_vae.num_layers=1",
    ])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    train_vq(cfg)
    assert not torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    assert torch.get_float32_matmul_precision() == "medium"
    assert cfg.workspace.output_dir == str(expected)
    assert OmegaConf.load(expected / "config.yaml").workspace.output_dir == str(expected)
    for name in ("metrics.jsonl", "codebook.npz", "checkpoints/last.pt"):
        assert (expected / name).exists()
    with pytest.raises(FileExistsError):
        train_vq(cfg)


@pytest.fixture
def contact_data(make_demo):
    path = make_demo(auxiliary=False)
    store = zarr.open_group(str(path), mode="a")
    points = store["data/point_cloud"][:, :, :3]
    store.create_dataset("data/hand_pointcloud", data=points)
    template = np.random.default_rng(45).normal(size=(1024, 3)).astype("<f4")
    world_points = template[None] + np.arange(12, dtype=np.float32)[:, None, None] * 0.1
    store.create_dataset("data/object_pointcloud", data=world_points)
    store.create_dataset("data/contact_map", data=np.zeros((12, 1024), dtype=np.float32))
    offsets = np.array([0, len(template)], dtype="<i8")
    payload = b"".join(
        np.array(values.shape, dtype="<i8").tobytes() + values.tobytes()
        for values in (template, offsets)
    )
    store.attrs["data_processing"] = {"object_pointcloud": {"geometry_sha256": hashlib.sha256(payload).hexdigest()}}
    return path


def test_observation_artifacts(output_path, contact_data, tmp_path, monkeypatch):
    value, directory = output_path
    source = tmp_path / "source.zarr"
    shutil.copytree(contact_data, source)
    zarr.open_group(str(contact_data), mode="a").attrs["source_zarr_path"] = str(source)
    policy = tmp_path / "policy.yaml"
    OmegaConf.save(OmegaConf.create(dict(
        task_name="test", zarr_path=str(source),
        dataset=dict(seed=42, val_episodes=0, max_train_episodes=1),
    )), policy)
    args = [
        "--output-dir", value, "--policy-config", str(policy), "--zarr-path", str(contact_data),
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
    assert saved.effective_config["workspace"]["output_dir"] == str(directory)
    assert OmegaConf.load(directory / "config.yaml").workspace.output_dir == str(directory)
    with pytest.raises(FileExistsError):
        obs_training.train("cordvip", CordViP, args)
    assert any(key.startswith("contact_head.") for key in saved.model_state)
    payload = torch.load(directory / "encoder.pt", weights_only=True)
    assert payload["training_data"] == saved.effective_config["training_data"]
    assert payload["training_data"][0]["source"] == str(source)
    states = zarr.open_group(str(contact_data), mode="r")["data/joint_state"][:]
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
    obs_training.train("cordvip", CordViP, ["--resume-from", value, "--max-updates", "1"])
    resumed = load_checkpoint(directory / "checkpoints/latest.pt")
    assert resumed.global_step == 2
    assert resumed.scheduler_state["last_epoch"] == 2
    for key in payload["normalizer_state"]:
        torch.testing.assert_close(resumed.model_state["normalizer." + key], payload["normalizer_state"][key])

    artifacts = [directory / name for name in ("config.yaml", "metrics.jsonl", "encoder.pt", "checkpoints/latest.pt")]
    before = [path.read_bytes() for path in artifacts]
    other_data = tmp_path / "other.zarr"
    shutil.copytree(contact_data, other_data)
    del zarr.open_group(str(other_data), mode="a").attrs["source_zarr_path"]
    with pytest.raises(ValueError, match="share the policy source and episode split"):
        obs_training.train("cordvip", CordViP, [
            "--resume-from", value, "--max-updates", "1", "--zarr-path", str(other_data),
        ])
    assert [path.read_bytes() for path in artifacts] == before
    store = zarr.open_group(str(contact_data), mode="a")
    store.attrs["data_processing"] = {"object_pointcloud": {"geometry_sha256": "f" * 64}}
    with pytest.raises(ValueError, match="same object geometry"):
        obs_training.train("cordvip", CordViP, ["--resume-from", value, "--max-updates", "1"])
    assert [path.read_bytes() for path in artifacts] == before


def test_contact_requires_geometry_identity(contact_data, tmp_path):
    store = zarr.open_group(str(contact_data), mode="a")
    del store["data/contact_map"]
    store.create_dataset("data/contact_map", data=np.zeros((12, 1023), dtype=np.float32))
    with pytest.raises(ValueError, match="Contact labels must match the 1024 object point slots"):
        CordViPPretrainDataset(contact_data, contact_loss_weight=1)
    store.attrs["data_processing"] = {"object_pointcloud": {"geometry_sha256": None}}
    with pytest.raises(ValueError, match="geometry_sha256"):
        CordViPPretrainDataset(contact_data, contact_loss_weight=1)
    output = tmp_path / "missing.zarr"
    with pytest.raises(ValueError, match="geometry_sha256"):
        generate_contacts(contact_data, output)
    assert not output.exists()


def test_disabled_contact_does_not_require_labels_or_geometry(contact_data):
    store = zarr.open_group(str(contact_data), mode="a")
    del store["data/contact_map"]
    del store["data/object_pointcloud"]
    store.create_dataset("data/object_pointcloud", data=store["data/hand_pointcloud"][:])
    del store.attrs["data_processing"]
    dataset = CordViPPretrainDataset(contact_data, contact_loss_weight=0)
    assert "contact_map" not in dataset.episode_store.keys()
    model = CordViP(dict.fromkeys(dataset.sensor_modalities, "identity"), contact_loss_weight=0)
    assert not hasattr(model, "contact_head")
    batch = {"obs": {key: value.unsqueeze(0) for key, value in dataset[0]["obs"].items()}}
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "arm", "hand"}


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


def test_training_failure_closes_all_resources(capsys):
    closed = []

    def fail():
        raise ValueError("training failed")

    def fail_close():
        raise RuntimeError("close failed")

    trainer = SimpleNamespace(
        compile_models=fail,
        close=lambda: close_resources(fail_close, lambda: closed.append(True)),
    )
    with pytest.raises(ValueError, match="training failed"):
        run_trainer(trainer)
    assert closed == [True]
    assert "close failed" in capsys.readouterr().err


# Resume paths


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "tilde", "saved_zarr"])
def test_cli_and_api_choose_caller_files(tmp_path, small_dp3_config, make_demo, monkeypatch, path_kind):
    caller, project = tmp_path / "caller", tmp_path / "project"
    for directory, marker, step in ((caller, "caller", 1), (project, "project", 2)):
        directory.mkdir()
        shutil.copytree(make_demo(marker), directory / "data.zarr")
        zarr.open_group(str(directory / "data.zarr"), mode="a").attrs["marker"] = marker
        cfg = small_dp3_config(directory / "run", f"zarr_path={directory / 'data.zarr'}")
        save_checkpoint(directory / "run/checkpoints/latest.pt", Checkpoint(
            step, {"marker": marker}, {}, {}, OmegaConf.to_container(cfg, resolve=True),
        ))
    monkeypatch.setattr(common_config, "PROJECT_ROOT", project)
    monkeypatch.chdir(caller)
    if path_kind == "tilde":
        expanduser = Path.expanduser

        def controlled_home(path):
            return caller.joinpath(*path.parts[1:]) if path.parts[:1] == ("~",) else expanduser(path)

        monkeypatch.setattr(Path, "expanduser", controlled_home)
    run_arg = str(caller / "run") if path_kind == "absolute" else "~/run" if path_kind == "tilde" else "run"
    zarr_arg = str(caller / "data.zarr") if path_kind == "absolute" else "~/data.zarr" if path_kind == "tilde" else "data.zarr"
    if path_kind == "saved_zarr":
        zarr_arg = None

    constructed = []

    def inspect_build(cfg, *, checkpoint):
        dataset, _ = build_dataset(cfg)
        result = SimpleNamespace(
            directory=cfg.workspace.output_dir, data_path=dataset.episode_store.path,
            data_marker=dataset.episode_store.attrs["marker"],
            checkpoint_marker=checkpoint.model_state["marker"], step=checkpoint.global_step,
        )
        constructed.append(result)
        return result

    monkeypatch.setattr(policy_training, "build_trainer", inspect_build)
    monkeypatch.setattr(policy_training, "run_trainer", lambda result, **kwargs: result.step)
    api = resume.prepare_resume(run_arg, zarr_path=zarr_arg)
    assert Path.cwd() == caller
    prepare_resume = resume.prepare_resume

    def inspect_prepare(run, *, zarr_path, **kwargs):
        assert Path.cwd() == project
        assert run == str(caller / "run")
        assert zarr_path == (None if path_kind == "saved_zarr" else str(caller / "data.zarr"))
        return prepare_resume(run, zarr_path=zarr_path, **kwargs)

    monkeypatch.setattr(resume, "prepare_resume", inspect_prepare)
    args = ["--run", run_arg]
    if zarr_arg is not None:
        args += ["--zarr-path", zarr_arg]
    resume.main(args)
    cli = constructed[-1]
    assert cli == api
    assert cli.checkpoint_marker == cli.data_marker == "caller"
    assert cli.step == 1 and cli.directory == str(caller / "run")
    assert cli.data_path == str(caller / "data.zarr")
    assert Path.cwd() == project


@pytest.mark.parametrize("missing", ["run", "zarr"])
def test_missing_paths_fail_at_existing_io_boundary(tmp_path, small_dp3_checkpoint, monkeypatch, missing):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(common_config, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    run_arg = "missing_run" if missing == "run" else "dp3_run"
    zarr_arg = "missing.zarr" if missing == "zarr" else "demo.zarr"
    prepare_resume = resume.prepare_resume
    entered = []

    def inspect_prepare(run, *, zarr_path, **kwargs):
        entered.append(True)
        assert Path.cwd() == project
        assert run == str(tmp_path / run_arg)
        assert zarr_path == str(tmp_path / zarr_arg)
        return prepare_resume(run, zarr_path=zarr_path, **kwargs)

    monkeypatch.setattr(resume, "prepare_resume", inspect_prepare)
    error_type = FileNotFoundError if missing == "run" else hydra.errors.InstantiationException
    with pytest.raises(error_type) as caught:
        resume.main(["--run", run_arg, "--zarr-path", zarr_arg])
    assert entered == [True]
    if missing == "run":
        assert Path(caught.value.filename) == tmp_path / "missing_run/checkpoints/latest.pt"
    else:
        assert isinstance(caught.value.__cause__, zarr.errors.GroupNotFoundError)


def test_cli_really_restores_and_continues_current_dp3(tmp_path, small_dp3_checkpoint, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(common_config, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    with closing(resume.prepare_resume("dp3_run", zarr_path="demo.zarr")) as restored:
        assert Path.cwd() == tmp_path
        assert restored.global_step == 1
        assert restored.logger.output_dir == tmp_path / "dp3_run"
        assert restored.train_loader.dataset.episode_store.path == str(tmp_path / "demo.zarr")
    run_trainer = policy_training.run_trainer
    steps = []

    def inspect_run(restored, **kwargs):
        assert Path.cwd() == project
        assert restored.global_step == 1
        assert restored.logger.output_dir == tmp_path / "dp3_run"
        assert restored.cfg.dataset.zarr_path == str(tmp_path / "demo.zarr")
        steps.append(run_trainer(restored, **kwargs))

    monkeypatch.setattr(policy_training, "run_trainer", inspect_run)
    resume.main(["--run", "dp3_run", "--zarr-path", "demo.zarr", "--max-updates", "1"])
    assert steps == [2]
    checkpoint = load_checkpoint(small_dp3_checkpoint)
    assert checkpoint.global_step == 2
    assert checkpoint.effective_config["workspace"]["output_dir"] == str(tmp_path / "dp3_run")
    assert checkpoint.effective_config["dataset"]["zarr_path"] == str(tmp_path / "demo.zarr")


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
    video_fps = 20

    def __init__(self):
        self.state = np.zeros(19, dtype=np.float32)
        self.close_count = 0
        self.close_error = False
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
        if self.close_error:
            raise RuntimeError("close failed")


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


@pytest.mark.parametrize("early_close", [False, True])
def test_normal_shutdown_propagates_close_failure(environments, early_close):
    results = make_runner().run(SmallPolicy(), seeds=[7], inference_steps=2, action_steps=2)
    assert next(results)["success"]
    environments[0].close_error = True
    with pytest.raises(RuntimeError, match="close failed"):
        results.close() if early_close else next(results)
    assert environments[0].close_count == 1


def test_step_error_preserved_during_cleanup(environments, capsys):
    results = make_runner().run(SmallPolicy(), seeds=[7, 8], inference_steps=2, action_steps=2)
    next(results)
    environments[0].step_error = environments[0].close_error = True
    with pytest.raises(ValueError, match="step failed"):
        next(results)
    assert environments[0].close_count == 1
    assert "close failed" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["init", "dt"])
def test_partial_environment_cleanup(monkeypatch, capsys, failure):
    created = []

    class FailingEnv(SmallEnv):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(self)
            self.close_error = True
            if failure == "init":
                raise ValueError("init failed")
            self.dt = 0.1

    monkeypatch.setattr(SimRunner, "_environment_class", lambda self: FailingEnv)
    with pytest.raises(ValueError, match="init failed|Control period mismatch"):
        make_runner().make_env(record_video=False)
    assert created[0].close_count == 1
    assert "close failed" in capsys.readouterr().err


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
        "control_dt_s": 0.05, "normalization_dims": SmallPolicy.normalization_dims, "dataset": {},
    }
    return [save_checkpoint(tmp_path / f"{step}.pt", Checkpoint(step, {}, {}, {}, cfg)) for step in (1, 2)]


def test_selection_and_independent_test(checkpoints, tmp_path):
    path = selection.run_pipeline(
        checkpoints, selection_episodes=2, test_episodes=3, seed=5, device="cpu", output_dir=tmp_path / "eval",
    )
    record = json.loads(path.read_text())
    assert record["global_step"] == 2
    assert set(record["selection_seeds"]).isdisjoint(record["test_seeds"])
    for candidate in record["candidates"]:
        result = json.loads((path.parent / candidate["result"]).read_text())
        assert result["seeds"] == record["selection_seeds"]
        assert result["completed"] and result["statistics"]["n_success"] == 2
    result = json.loads((path.parent / record["test_result"]).read_text())
    assert result["seeds"] == record["test_seeds"] and result["completed"]
    assert (result["weights"], result["inference_steps"], result["action_steps"]) == ("raw", 2, 2)
    candidates = [
        {"global_step": step, "statistics": {"success_rate": rate, "avg_steps": steps}}
        for rate, steps, step in [(0, None, 100), (0.5, 1, 100), (1, 5, 100), (1, 2, 1), (1, 2, 2)]
    ]
    assert sorted(candidates, key=selection.candidate_score) == candidates


@pytest.mark.parametrize("failure", ["video", "write"])
def test_evaluation_failure_closes_env_and_preserves_primary(checkpoints, environments, tmp_path, monkeypatch, capsys, failure):
    output = tmp_path / "failed_eval"
    if failure == "video":
        def fail_video(*args):
            environments[0].close_error = True
            raise OSError("video failed")

        monkeypatch.setattr("dexmani_policy.env_runner.video.save_episode_video", fail_video)
    else:
        write_json = rollout.write_json

        def fail_write(path, record):
            if record["episodes"]:
                environments[0].close_error = True
                raise OSError("write failed")
            write_json(path, record)

        monkeypatch.setattr(rollout, "write_json", fail_write)
    with pytest.raises(OSError, match=f"{failure} failed"):
        rollout.evaluate_policy(checkpoints[0], seeds=[7], device="cpu", video=failure == "video", output_dir=output)
    assert environments[0].close_count == 1
    assert "close failed" in capsys.readouterr().err
    record = json.loads((output / "result_details.json").read_text())
    assert not record["completed"]
    if failure == "video":
        assert record["statistics"]["n_success"] == 1
        assert record["episodes"][0]["video"] is None
