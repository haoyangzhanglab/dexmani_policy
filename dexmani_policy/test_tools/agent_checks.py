"""Numerical, sampling and synthetic policy checks: python -m dexmani_policy.test_tools.agent_checks."""

import argparse
import copy
import io
import json
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import zarr

from dexmani_policy.agents.action_decoders.backbones.dit import DiT
from dexmani_policy.agents.action_decoders.backbones.ditx import ConsistencyDiTX, CoupledDiTX, DiTX
from dexmani_policy.agents.action_decoders.backbones.one_way_transformer import OneWayTransformer
from dexmani_policy.agents.action_decoders.sampling.time_sampling import TimeSampler, euler_time_grid
from dexmani_policy.agents.hand_prior.action_io import SATActionIO
from dexmani_policy.agents.hand_prior.codebook import HandStateCodebook, load_codebook
from dexmani_policy.agents.hand_prior.vqvae import EMAVectorQuantizer, HandVQVAE
from dexmani_policy.agents.obs_encoder.interaction.future_tactile import (
    ForceDiffusion, ForceGuidedFusion, ForceTransformer, VisuoTactileAttention,
)
from dexmani_policy.agents.obs_encoder.interaction.hand_interaction import (
    HandHistoryEncoder, HandSceneFusion, SceneTokenizer, _sample_region_seeds,
)
from dexmani_policy.agents.obs_encoder.rgb.vit import ViTEncoder
from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandForceEncoder, XHandPatchEncoder
from dexmani_policy.agents.policies.base import Policy
from dexmani_policy.agents.policies.dp import DP
from dexmani_policy.agents.policies.dp3 import DP3
from dexmani_policy.agents.policies.dqrise import DQRISE
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.config import validate_observation_fields
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.deployment import inspect_policy, load_policy
from dexmani_policy.test_tools.dataset_checks import make_source
from dexmani_policy.train import run_training
from dexmani_policy.training.common.config import assemble_effective_config, load_policy_config, prepare_training, validate_config
from dexmani_policy.training.common.data import build_dataset
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.common.optimization import build_optimizer
from dexmani_policy.training.common.resume import prepare_resume
from dexmani_policy.training.pretrain.hand_prior.codebook import export_codebook
from dexmani_policy.training.pretrain.hand_prior.vqvae import evaluate_vq
from dexmani_policy.training.pretrain.hand_prior import vqvae as vq_training


def check_training_input_filter(device):
    from dexmani_policy.agents.policies.adaptac import AdapTac
    from dexmani_policy.training.trainer import PolicyTrainer

    class UnusedObservation:
        def to(self, *args, **kwargs):
            raise AssertionError("Unused observation reached device transfer")

    class RecordingPolicy(torch.nn.Module):
        consumed_observation_fields = AdapTac.consumed_observation_fields

        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.ones(()))
            self.obs_encoder = SimpleNamespace(consumed_observation_fields=("taxel_tactile_force",))

        def get_training_loss_kwargs(self, ema):
            return {}

        def forward(self, batch, **kwargs):
            assert set(batch["obs"]) == {"taxel_tactile_force", "finger_tactile_force"}
            assert set(batch) == {"obs", "action", "future_force", "future_force_valid"}
            for value in (*batch["obs"].values(), batch["action"], batch["future_force"], batch["future_force_valid"]):
                assert value.device == self.gain.device
            return self.gain * batch["action"].sum(), {}

    trainer = object.__new__(PolicyTrainer)
    trainer.model = RecordingPolicy().to(device)
    trainer.device, trainer.ema_model, trainer.use_bfloat16 = device, None, False
    batch: dict[str, Any] = dict(obs={"taxel_tactile_force": torch.ones(1), "finger_tactile_force": torch.ones(1),
                      "unused": UnusedObservation()}, action=torch.ones(1),
                 future_force=torch.ones(1), future_force_valid=torch.ones(1, dtype=torch.bool))
    trainer.train_microbatch(batch)
    assert "unused" in batch["obs"] and trainer.model.gain.grad is not None


def check_evaluation(_device):
    from dexmani_policy.common.io_util import write_json
    from dexmani_policy.deployment import inspect_policy
    from dexmani_policy.env_runner.sim_runner import SimRunner
    from dexmani_policy.evaluation import rollout, selection
    from dexmani_policy.evaluation.seeds import split_evaluation_seeds

    pool = list(range(10))
    test, selected = split_evaluation_seeds(pool, 1024, 3, 2)
    assert (test, selected) == split_evaluation_seeds(list(reversed(pool)), 1024, 3, 2)
    assert test == split_evaluation_seeds(pool, 1024, 3)[0]
    assert set(test).isdisjoint(selected)

    config = dict(
        policy_name="dp3", task_name="test", action_key="action",
        eval=dict(use_ema=True, inference_steps=2),
        agent=dict(n_action_steps=2, horizon=4, n_obs_steps=1, action_dim=19),
        normalization_dims=dict(joint_state=19, point_cloud=3, action=19), dataset={}, deployment_metadata={},
        policy_metadata=dict(action_start_index=0, control_action_dim=19, supported_execution_modes=["sync"]),
        env_runner=dict(task_name="test", env_kwargs=dict(frame_skip=6), env_rebuild_interval=1),
    )
    with tempfile.TemporaryDirectory(prefix="dexmani-evaluation-check-") as temporary:
        root = Path(temporary)
        checkpoints = [save_checkpoint(root / folder / "model.pt", Checkpoint(
            step, {}, {}, {}, {}, {}, config)) for step, folder in ((1, "external-a"), (2, "external-b"))]
        run = root / "publication"
        (run / "checkpoints").mkdir(parents=True)
        link = run / "checkpoints/link.pt"
        link.symlink_to(checkpoints[1])
        for explicit in (checkpoints[1], link, "link.pt"):
            assert inspect_policy(run, checkpoint=explicit).checkpoint_path == checkpoints[1]
        old_best = (json.dumps(dict(
            ckpt_relpath="../external-a/model.pt",
            inference=dict(use_ema=True, inference_steps=2, action_steps=2),
        )) + "\n").encode()
        best = run / "best_ckpt.json"
        for mode in ("success", "test_failure", "save_failure", "candidate_failure"):
            best.write_bytes(old_best)
            directory = root / f"results-{mode}"
            events, calls = [], []

            def evaluate(checkpoint, **options):
                is_test = options["output_dir"].name == "test"
                events.append("test" if is_test else "candidate")
                assert best.read_bytes() == old_best
                assert not (directory / "selection_result.json").exists()
                assert options["weights"] == "ema" and options["inference_steps"] == options["action_steps"] == 2
                if mode == "candidate_failure" or (mode == "test_failure" and is_test):
                    raise RuntimeError("injected evaluation failure")
                calls.append(options)
                if is_test:
                    assert Path(checkpoint).resolve() == checkpoints[1]
                result = dict(global_step=2 if Path(checkpoint).resolve() == checkpoints[1] else 1,
                              statistics=dict(success_rate=0.0 if is_test else 0.8, avg_steps=10))
                output = rollout.create_run_directory(None, options["output_dir"])
                path = output / "result_details.json"
                write_json(path, result)
                return result, path

            def publish(path, record, **options):
                events.append(path.name)
                if path.name == "selection_result.json" and mode == "save_failure":
                    raise OSError("injected save failure")
                if path == best:
                    assert (directory / "selection_result.json").is_file()
                write_json(path, record, **options)

            with patch.object(selection, "evaluate_policy", side_effect=evaluate), \
                 patch.object(selection, "load_seed_pool", return_value=pool) as seed_pool, \
                 patch.object(selection, "write_json", side_effect=publish):
                try:
                    path = selection.run_pipeline(checkpoints, run_dir=run, output_dir=directory,
                                                  selection_episodes=2, test_episodes=3, device="cpu")
                    seed_pool.assert_called_once_with("test")
                except (RuntimeError, OSError) as error:
                    assert "injected" in str(error)
                    assert mode != "success"
                    assert best.read_bytes() == old_best
                    assert inspect_policy(run).checkpoint_path == checkpoints[0]
                    assert not (directory / "selection_result.json").exists()
                    continue
            assert mode == "success"
            assert events == ["candidate", "candidate", "test", "selection_result.json", "best_ckpt.json"]
            assert calls[0]["seeds"] == calls[1]["seeds"]
            assert set(calls[0]["seeds"]).isdisjoint(calls[2]["seeds"])
            record = json.loads(path.read_text())
            assert (path.parent / record["ckpt_relpath"]).resolve() == checkpoints[1]
            assert (path.parent / record["test_result"]).is_file()
            for candidate, checkpoint in zip(record["candidates"], checkpoints):
                assert (path.parent / candidate["checkpoint"]).resolve() == checkpoint
                assert (path.parent / candidate["result"]).is_file()
            assert set(json.loads(best.read_text())) == {"ckpt_relpath", "inference"}
            assert inspect_policy(run).checkpoint_path == checkpoints[1]

        @dataclass
        class Randomization:
            randomize_model_id: bool = True
            table_height_range: tuple[float, float] = (-0.01, 0.02)

        @dataclass
        class Stability:
            stable_check_steps: int = 240

        class Environment:
            def __init__(self, frame_skip=15, action_cnt_limit=3, success_hold_steps=2):
                self.frame_skip = frame_skip
                self.action_cnt_limit = action_cnt_limit
                self.success_hold_steps = success_hold_steps
                self.randomization, self.stability = Randomization(), Stability()
                self.closed, self.dt_reads = False, 0

            @property
            def dt(self):
                self.dt_reads += 1
                return self.frame_skip / 240

            def observation(self):
                return dict(joint_state=np.full(19, self.action_cnt, dtype=np.float32), point_cloud=object())

            def reset(self, *, seed):
                self.action_cnt = 0
                return self.observation(), {}

            def step(self, action):
                assert self.action_cnt < self.action_cnt_limit and not self.closed
                self.action_cnt += 1
                if fail and self.action_cnt == 2:
                    raise RuntimeError("injected rollout failure")
                done = self.action_cnt == self.action_cnt_limit
                return self.observation(), 0, done, False, dict(
                    success=done, success_condition=self.action_cnt >= 2)

            def close(self):
                self.closed = True

        agent = SimpleNamespace(parameters=torch.nn.Linear(1, 1).parameters,
                                n_obs_steps=3, n_action_steps=1, future_steps=2,
                                consumed_observation_fields=("joint_state",),
                                normalization_dims=config["normalization_dims"])

        for fail in (False, True):
            directory = root / f"rollout-{fail}"
            environments, runners = [], []

            def make_env(runner, *, record_video):
                assert not record_video
                assert runner.n_obs_steps == agent.n_obs_steps
                assert runner.sensor_modalities == list(agent.consumed_observation_fields)
                if environments:
                    assert environments[-1].closed
                else:
                    assert not (directory / "result_details.json").exists()
                    assert runner.environment == {}
                env = Environment(**runner.env_kwargs)
                environments.append(env)
                runners.append(runner)
                return env

            def predict(obs_dict, *, inference_steps, generator):
                assert inference_steps == 2 and set(obs_dict) == {"joint_state"}
                assert obs_dict["joint_state"].shape == (1, 3, 19)
                expected = torch.tensor(((0, 0, 0), (0, 0, 1), (0, 1, 2))[environments[-1].action_cnt])
                torch.testing.assert_close(obs_dict["joint_state"][0, :, 0], expected.float())
                assert set(runners[-1].obs_buffer) == {"joint_state"}
                return {"future_actions": torch.zeros(1, agent.future_steps, 19)}

            agent.predict_action = predict
            with patch.object(rollout, "_restore_policy", return_value=agent), \
                 patch.object(rollout, "load_seed_pool", side_effect=AssertionError("Explicit seeds read pool")), \
                 patch.object(SimRunner, "make_env", autospec=True, side_effect=make_env), \
                 patch.object(rollout, "write_json", wraps=write_json) as save:
                try:
                    record, path = rollout.evaluate_policy(checkpoints[0], seeds=[8, 3], episodes=9, device="cpu",
                                                           output_dir=directory, video=False)
                except RuntimeError as error:
                    assert "injected" in str(error)
                    assert fail and not (directory / "result_details.json").exists()
                    assert save.call_count == 0
                else:
                    assert not fail and path.is_file()
                    assert save.call_count == 1
                    assert json.loads(json.dumps(record)) == json.loads(path.read_text())
                    assert record["global_step"] == 1 and record["action_steps"] == agent.n_action_steps
                    assert record["runner_options"]["n_obs_steps"] == agent.n_obs_steps
                    assert record["runner_options"]["sensor_modalities"] == list(agent.consumed_observation_fields)
                    assert record["runner_options"]["env_kwargs"] == {"frame_skip": 6}
                    assert record["seeds"] == [row["seed"] for row in record["episodes"]] == [8, 3]
                    assert record["statistics"]["n_episodes"] == len(record["episodes"]) == 2
                    assert all(row["steps"] == 2 and row["total_steps"] == 3 for row in record["episodes"])
                    assert record["environment"] == dict(
                        control_dt_s=6 / 240, frame_skip=6, action_cnt_limit=3, success_hold_steps=2,
                        randomization=dict(randomize_model_id=True, table_height_range=(-0.01, 0.02)),
                        stability=dict(stable_check_steps=240))
                    assert [env.dt_reads for env in environments] == [1, 0]
                    runner = runners[0]
                    runner.env_kwargs = {"frame_skip": 12}
                    list(runner.run(agent, seeds=[8], inference_steps=2, action_steps=agent.n_action_steps))
                    assert runner.environment["control_dt_s"] == 12 / 240
                    assert record["environment"]["control_dt_s"] == 6 / 240
            assert all(env.closed for env in environments)

        with patch.object(rollout, "_restore_policy", return_value=agent):
            try:
                rollout.evaluate_policy(checkpoints[0], seeds=[1], action_steps=3, device="cpu",
                                        output_dir=root / "too-many-actions", video=False)
            except ValueError as error:
                assert "future_steps (2)" in str(error)
            else:
                raise AssertionError("Evaluation ignored the agent's future window")


def check_time(device):
    grid = euler_time_grid(1024, torch.empty((), device=device, dtype=torch.bfloat16))
    assert grid.dtype == torch.float32 and grid.unique().numel() == 1024
    assert grid[0] == 0 and grid[-1] < 1 and (grid[1:] > grid[:-1]).all()
    sampler = TimeSampler(10)
    rng = torch.random.get_rng_state() if device.type == "cpu" else torch.cuda.get_rng_state(device)
    expected = 0.999 * torch.distributions.Beta(
        torch.tensor(1.0, device=device), torch.tensor(1.5, device=device),
    ).sample(torch.Size((128,)))
    if device.type == "cpu":
        torch.random.set_rng_state(rng)
    else:
        torch.cuda.set_rng_state(rng, device)
    assert torch.equal(expected, sampler.sample(128, "beta", device))
    for value in (0, -1, float("nan"), float("inf"), 1e-99, 1e99):
        try:
            TimeSampler(10, beta_alpha=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid FP32 beta concentration: {value}")


def check_precision(device):
    models = (
        (DiTX(16, 19, 2, 256, hidden_dim=32, n_head=4, n_layers=1, p_drop_attn=0.0,
              qkv_bias=True, qk_norm=True, action_io=SATActionIO(16, 32)),
         "ditx_blocks", (2, 16, 19), (2, 194, 256)),
        (DiT(4, 3, 16, n_emb=32, num_heads=4, n_layers=1), "blocks", (2, 4, 3), (2, 16)),
        (ConsistencyDiTX(4, 3, 2, 16, hidden_dim=32, n_head=4, n_layers=1),
         "ditx_blocks", (2, 4, 3), (2, 4, 16)),
    )
    for model, blocks, action_shape, context_shape in models:
        model.to(device)
        seen = []
        hook = getattr(model, blocks)[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].dtype))
        kwargs = dict(x=torch.randn(action_shape, device=device), timestep=torch.rand(2, device=device),
                      context=torch.randn(context_shape, device=device))
        if isinstance(model, ConsistencyDiTX):
            kwargs["target_t"] = torch.rand(2, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = model(**kwargs)
        hook.remove()
        assert seen == [torch.bfloat16] and torch.isfinite(output).all()


def check_decoder_precision(device):
    from dexmani_policy.agents.action_decoders.diffusion import Diffusion
    from dexmani_policy.agents.action_decoders.flow import ConsistencyFlow, RectifiedFlow

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.1, device=device))

        def prepare_context_kv(self, context):
            return None

        def forward(self, x, timestep, context, **kwargs):
            self.input = x.detach()
            return x * self.gain.to(x.dtype)

    cond = torch.zeros(2, 4, device=device, dtype=torch.bfloat16)
    shape = (2, 4, 3)
    for kind in ("sample", "epsilon", "v_prediction"):
        model = Head()
        decoder = Diffusion(model, prediction_type=kind)
        for dtype in (torch.float32, torch.bfloat16):
            actions = torch.zeros(shape, device=device, dtype=dtype)
            noise = torch.ones_like(actions)
            t = torch.zeros(2, device=device, dtype=torch.long)
            with patch("torch.randn_like", return_value=noise), patch("torch.randint", return_value=t):
                loss, _ = decoder.compute_loss(cond, actions)
            a = decoder.noise_scheduler.alphas_cumprod[0].float()
            expected = ((1 - a).sqrt() * noise).to(dtype)
            torch.testing.assert_close(model.input, expected, rtol=0, atol=0)
            assert model.input.abs().min() > 0
            target = actions if kind == "sample" else noise if kind == "epsilon" else (a.sqrt() * noise).to(dtype)
            torch.testing.assert_close(loss, F.mse_loss((expected * model.gain.to(dtype)).float(), target.float()))
            loss.backward()
            assert model.gain.grad is not None and torch.isfinite(model.gain.grad)

    for sampling_decoder in (Diffusion(Head()), RectifiedFlow(Head()), ConsistencyFlow(Head())):
        sampling_decoder.eval()
        def sample(**kwargs):
            return sampling_decoder.predict_action(cond, shape, 4,
                generator=torch.Generator(device=device).manual_seed(10), **kwargs)
        with torch.inference_mode():
            ordinary = sample()
            assert ordinary.dtype == torch.float32
            torch.testing.assert_close(ordinary, sample(dtype=torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(ordinary, sample(rtc_prefix=ordinary[:, :2], rtc_guidance_cap=0), rtol=0, atol=0)
            assert sample(dtype=torch.bfloat16).dtype == torch.bfloat16
            guided = sample(rtc_prefix=ordinary[:, :2], rtc_guidance_cap=2)
            assert guided.dtype == torch.float32 and torch.isfinite(guided).all()


def check_tactile(device):
    encoder = XHandPatchEncoder().to(device)
    for mode in ("random", "zero", "ties"):
        x = torch.randn(2, 5, 120, 3, device=device)
        if mode == "zero":
            x.zero_()
        elif mode == "ties":
            x = x.sign()
        x.requires_grad_()
        mask = (encoder.region_ids[:, None] == torch.arange(5, device=device)[None, :, None])[None]
        magnitude = torch.linalg.vector_norm(x, dim=-1)
        weights = magnitude / (magnitude + encoder.activity_scale)
        weights = weights[:, :, None] * mask
        mass = weights.sum(-1)
        mean = torch.einsum("bfpt,bftc->bfpc", weights, x) / torch.where(mass > 0, mass, 1)[..., None]
        expected = torch.cat((
            mean, (x.abs()[:, :, None] * mask[..., None]).amax(-2),
            (mass / mask.sum(-1))[..., None],
            (magnitude[:, :, None] * mask).amax(-1)[..., None],
        ), -1)
        actual = encoder._patch_descriptors(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        probe = torch.randn_like(actual)
        reference_grad, = torch.autograd.grad((expected * probe).mean(), x)
        new_grad, = torch.autograd.grad((actual * probe).mean(), x)
        torch.testing.assert_close(reference_grad, new_grad, rtol=1e-5, atol=1e-6)
    with patch.object(XHandPatchEncoder, "__init__", autospec=True,
                      side_effect=XHandPatchEncoder.__init__) as constructor:
        policy = hydra.utils.instantiate({
            "_target_": "dexmani_policy.agents.policies.adaptac.AdapTac",
            "horizon": 4, "n_obs_steps": 1, "n_action_steps": 2, "action_dim": 19,
            "tactile_source": "real", "tactile_encoder": encoder.effective_config,
            "pc_dim": 3, "num_points": 16, "feature_dim": 16, "state_out_dim": 8,
            "pc_encoder_config": dict(channels=24, num_patches=4, group_size=4, depth=1, num_heads=4),
            "down_dims": [16, 32], "diffusion_step_embed_dim": 32,
            "force_predictor_config": dict(depth=1, num_heads=4, embed_dim=16),
            "fusion_num_heads": 4,
        }, _convert_="all")
    assert type(constructor.call_args.kwargs["region_ids"]) is list
    restored = policy.obs_encoder.tactile_encoder.to(device)
    assert isinstance(restored, XHandPatchEncoder)
    restored.load_state_dict(encoder.state_dict(), strict=True)

    from dexmani_policy.env_runner.sim_runner import SimRunner
    from dexmani_policy.agents.policies.interact_dex import InterActDex

    runner = SimRunner("unused", 1, list(policy.consumed_observation_fields), {})
    with patch.object(runner, "make_env", side_effect=AssertionError("Created environment for real tactile")):
        try:
            next(runner.run(policy, seeds=[1], inference_steps=2, action_steps=2))
        except ValueError as error:
            assert "sim-source" in str(error)
        else:
            raise AssertionError("SimRunner accepted real tactile")
    force = torch.tensor([1., 2., 3.], device=device).expand(1, 1, 5, 3)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
    pose = torch.tensor([0., 0., 0., 0., 1., 0., -1., 0., 0.], device=device).reshape(1, 1, 9)
    batch = dict(obs=dict(finger_tactile_force=torch.zeros_like(force), eef_pose=pose),
                 future_force=force, future_force_valid=torch.ones(1, 1, dtype=torch.bool, device=device))
    for source in ("real", "sim"):
        owner = SimpleNamespace(tactile_source=source,
                                normalizer={"finger_tactile_force": SimpleNamespace(scale=2.)})
        target = (force @ rotation if source == "sim" else force) * 2
        loss, _ = InterActDex._compute_future_force_loss(cast(Any, owner), target, batch)
        torch.testing.assert_close(loss, torch.zeros((), device=device))


def check_interaction(device):
    scene = SceneTokenizer(24).to(device)
    for patch_count in (12, 24):
        points = torch.zeros(2, patch_count, 3, device=device)
        seeds = _sample_region_seeds(points, 12)
        assert all(row.unique().numel() == 12 for row in seeds)
        tokens = torch.randn(2, patch_count, 24, device=device, requires_grad=True)
        for bf16 in (False, True):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                output = scene(tokens, points)
            assert output.shape == (2, 12, 24) and torch.isfinite(output).all()
            grad, = torch.autograd.grad((output * torch.randn_like(output)).sum(), tokens)
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0

    model = HandSceneFusion(24, XHandForceEncoder()).to(device).eval()
    points = torch.randn(2, 64, 3, device=device) * 0.03
    anchors = torch.randn(2, 6, 3, device=device) * 0.02
    patches = dict(tokens=torch.randn(2, 12, 24, device=device), xyz=points[:, :12])
    hand = torch.randn(2, 6, 192, device=device)
    rotation = torch.eye(3, device=device).expand(2, -1, -1)
    context = torch.randn(2, 192, device=device)
    tactile = torch.randn(2, 5, 3, device=device)
    kwargs = dict(rotation_world_from_eef=rotation, query_context=context, tactile=tactile)
    output = model(points, patches, anchors, hand, **kwargs)
    changed = tactile.clone()
    changed[:, 2] += 2
    finger_output = model(points, patches, anchors, hand, **{**kwargs, "tactile": changed})
    torch.testing.assert_close(output[:, [1, 2, 4, 5]], finger_output[:, [1, 2, 4, 5]])
    assert (output[:, [0, 3]] - finger_output[:, [0, 3]]).abs().sum() > 0

    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
    shift = torch.tensor([0.2, -0.1, 0.3], device=device)
    rotated = model(
        points @ rotation.T + shift,
        {**patches, "xyz": patches["xyz"] @ rotation.T + shift},
        anchors @ rotation.T + shift, hand,
        **{**kwargs, "rotation_world_from_eef": rotation.expand(2, -1, -1)},
    )
    torch.testing.assert_close(output, rotated, rtol=1e-5, atol=1e-6)

    geometry_inputs = []
    hook = model.geometry_projection.register_forward_pre_hook(
        lambda _, args: geometry_inputs.append(args[0].detach())
    )
    for bf16 in (False, True):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            empty = model(points + 10, patches, anchors, hand, **kwargs)
        assert empty.dtype == torch.float32 and torch.isfinite(empty).all()
        assert torch.count_nonzero(geometry_inputs[-1][..., :32]) == 0
        (empty * torch.randn_like(empty)).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    hook.remove()

    history = HandHistoryEncoder(4).to(device).eval()
    tokens = torch.randn(2, 4, 6, 192, device=device)
    changed = tokens.clone()
    changed[:, 2:] += torch.randn_like(changed[:, 2:])
    for bf16 in (False, True):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            original, perturbed = history(tokens), history(changed)
        torch.testing.assert_close(original[:, :2], perturbed[:, :2], rtol=0, atol=0)


def check_force_interaction(device):
    interaction = VisuoTactileAttention(24, 64, 5).to(device).eval()
    visual = torch.randn(2, 12, 24, device=device, requires_grad=True)
    tactile = torch.randn(2, 5, 64, device=device, requires_grad=True)
    v, t = interaction(visual, tactile)
    vg, = torch.autograd.grad((t * torch.randn_like(t)).mean(), visual, retain_graph=True)
    tg, = torch.autograd.grad((v * torch.randn_like(v)).mean(), tactile)
    assert vg.abs().sum() > 0 and tg.abs().sum() > 0

    predictor = ForceDiffusion(ForceTransformer(4, 32, 15, depth=1, embed_dim=32, dropout=0)).to(device)
    fusion = ForceGuidedFusion(16, 16, 75, hidden_dim=32, num_heads=4, dropout=0).to(device)
    noisy = torch.randn(2, 4, 15, device=device)
    changed = noisy.clone()
    changed[:, 2:] += 10
    condition = torch.randn(2, 1, 32, device=device)
    torch.testing.assert_close(
        predictor.model(noisy, 3, condition)[:, :2],
        predictor.model(changed, 3, condition)[:, :2], rtol=0, atol=0,
    )
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]], device=device)
    force = torch.zeros(2, 4, 15, device=device)
    for bf16 in (False, True):
        predictor.zero_grad(set_to_none=True)
        fusion.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            loss, prediction = predictor.compute_loss_and_prediction(
                torch.randn(2, 1, 32, device=device), force, valid,
            )
            guide = torch.cat((torch.zeros(2, 1, 15, device=device), prediction), 1).flatten(1)
            output = fusion(torch.randn(2, 16, device=device), torch.randn(2, 16, device=device), guide)
        torch.testing.assert_close(loss, prediction[valid].square().mean())
        target_grad, = torch.autograd.grad(loss, prediction, retain_graph=True)
        assert torch.count_nonzero(target_grad[~valid]) == 0
        assert target_grad[valid].abs().sum() > 0
        (output * torch.randn_like(output)).sum().backward()
        assert predictor.model.head.weight.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())


def check_token_decoders(device):
    coupled = CoupledDiTX(
        horizon=8, action_dim=19, n_obs_steps=2, obs_token_dim=16,
        hidden_dim=32, n_head=4, n_layers=2, p_drop_attn=0, interaction_horizon=3,
    ).to(device).eval()
    # Zero-initialized gates and readout would hide a broken attention mask.
    for block in coupled.ditx_blocks:
        torch.nn.init.normal_(block.adaln_modulation[-1].weight, std=0.1)
    torch.nn.init.normal_(coupled.final_layer.ffn_final.fc2.weight, std=0.1)
    actions = torch.randn(2, 8, 19, device=device)
    context = torch.randn(2, 18, 16, device=device)
    timestep = torch.rand(2, device=device)
    outside = actions.clone()
    outside[:, [0, 4, 5, 6, 7]] += 3
    changed_body = context.clone()
    changed_body.unflatten(1, (2, 9))[:, -1, :6] += 2
    auxiliary = OneWayTransformer(
        horizon=8, action_dim=28, n_obs_steps=2, obs_token_dim=48, pc_pe_dim=32,
        embedding_dim=32, depth=2, num_heads=4, mlp_dim=64,
        use_aux_ee=True, joint_dim=19, ee_dim=9,
    ).to(device).eval()
    joint_ee = torch.randn(2, 8, 28, device=device)
    changed_ee = joint_ee.clone()
    changed_ee[..., 19:] += 3
    observations = torch.randn(2, 8, 48, device=device)
    for bf16 in (False, True):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            velocity, features = coupled(actions, timestep, context, return_features=True)
            perturbed, perturbed_features = coupled(outside, timestep, context, return_features=True)
            cached_kv = coupled.prepare_context_kv(context)
            cached = coupled(actions, timestep, context, context_kv=cached_kv, return_features=True)
            dynamic = coupled(actions, timestep, changed_body, context_kv=cached_kv, return_features=True)
            prediction = auxiliary(joint_ee, timestep, observations)
            perturbed_ee = auxiliary(changed_ee, timestep, observations)
        torch.testing.assert_close(velocity[:, 1:4], perturbed[:, 1:4], rtol=0, atol=0)
        torch.testing.assert_close(features[:, 8:], perturbed_features[:, 8:], rtol=0, atol=0)
        assert (velocity[:, 4:] - perturbed[:, 4:]).abs().sum() > 0
        torch.testing.assert_close(cached, (velocity, features), rtol=0, atol=0)
        assert (dynamic[1][:, 8:] - features[:, 8:]).abs().sum() > 0
        torch.testing.assert_close(prediction[..., :19], perturbed_ee[..., :19], rtol=0, atol=0)
        assert (prediction[..., 19:] - perturbed_ee[..., 19:]).abs().sum() > 0


def check_policy_roundtrips(device):
    for method, auxiliary in (("dp", False), ("afro", False), ("maniflow", False),
                              ("dexlatent", False), ("cordvip", False), ("adaptac", False), ("interact_dex", False),
                              ("sat", False), ("r3d", False), ("r3d", True)):
        overrides = ["horizon=8", "n_action_steps=2", "agent.num_inference_steps=2"]
        if method not in ("r3d", "cordvip", "afro", "dp"):
            overrides += ["agent.pc_dim=3"]
        if method in ("interact_dex", "sat", "dexlatent", "maniflow"):
            overrides += ["agent.n_layers=1", "agent.hidden_dim=32", "agent.n_head=4",
                          "agent.timestep_embed_dim=32"]
        elif method == "r3d":
            overrides += [f"use_aux_ee={str(auxiliary).lower()}", "agent.embedding_dim=32",
                          "agent.depth=1", "agent.num_heads=4", "agent.mlp_dim=64",
                          "agent.timestep_embed_dim=32", "agent.num_training_steps=10",
                          "agent.pc_encoder_config.embed_dim=32", "agent.pc_encoder_config.num_group=4",
                          "agent.pc_encoder_config.group_size=8", "agent.pc_encoder_config.drop_path_rate=0",
                          "agent.fps_random_config.use_random=false"]
        else:
            overrides += ["agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=32",
                          "agent.num_training_steps=10"]
        if method not in ("cordvip", "r3d", "afro", "dp", "maniflow"):
            overrides += ["agent.num_points=32", "agent.pc_encoder_config.channels=24",
                          "agent.pc_encoder_config.num_patches=12", "agent.pc_encoder_config.group_size=8",
                          "agent.pc_encoder_config.depth=1", "agent.pc_encoder_config.num_heads=4",
                          "agent.fps_random_config.use_random=false"]
        if method == "adaptac":
            overrides += ["agent.feature_dim=32", "agent.state_out_dim=8", "agent.force_num_inference_steps=2",
                          "agent.force_predictor_config.depth=1", "agent.force_predictor_config.embed_dim=32"]
        if method in ("sat", "dexlatent"):
            overrides += ["agent.rgb_backbone_config.image_size=[28,28]",
                          "+agent.rgb_backbone_config.architecture={model_type:dinov2,hidden_size:24,"
                          "intermediate_size:48,num_hidden_layers:1,num_attention_heads:4,image_size:28,patch_size:14}"]
        if method == "dp":
            overrides += ["agent.rgb_backbone_name=resnet"]
        if method == "maniflow":
            overrides += ["agent.num_points=32",
                          "agent.pc_encoder_config.out_channels=32", "agent.state_out_dim=8",
                          "agent.fps_random_config.use_random=false"]
        cfg = load_policy_config(method, overrides)
        if method == "dp":
            cfg.agent.rgb_backbone_config = {}
        agent_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.agent, resolve=True))
        policy = hydra.utils.instantiate(agent_cfg, _convert_="all")
        validate_observation_fields(policy, cfg.dataset.sensor_modalities)
        shapes = dict(joint_state=(19,), point_cloud=(32, 3), hand_pointcloud=(32, 3),
                      object_pointcloud=(32, 3), finger_tactile_force=(5, 3),
                      eef_pose=(9,), fingertip_xyz=(5, 3), rgb=(3, 28, 28),
                      camera_intrinsic=(9,), camera_extrinsic=(12,))
        if method == "r3d":
            shapes["point_cloud"] = (32, 6)
        obs = {key: torch.randn(2, cfg.n_obs_steps, *shapes[key], device=device)
               for key in cfg.dataset.sensor_modalities}
        if "eef_pose" in obs:
            obs["eef_pose"][..., 3:] = torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
        if method in ("sat", "dexlatent"):
            obs["rgb"] = torch.randint(256, obs["rgb"].shape, device=device, dtype=torch.uint8)
            obs["point_cloud"][..., 2] = obs["point_cloud"][..., 2].abs() + 1
            intrinsic = torch.tensor([[20., 0., 13.5], [0., 20., 13.5], [0., 0., 1.]], device=device)
            obs["camera_intrinsic"] = intrinsic.flatten().expand(2, cfg.n_obs_steps, -1)
            obs["camera_extrinsic"] = torch.eye(4, device=device)[:3].flatten().expand(2, cfg.n_obs_steps, -1)
        dims = {key: (3 if key == "rgb" else value.shape[-1]) for key, value in obs.items()}
        dims["action"] = cfg.action_dim
        normalizer_shapes = {key: shape for key, shape in normalization_feature_shapes(dims).items()
                             if cfg.normalization[key] != "identity"}
        if method == "dexlatent":
            normalizer_shapes["hand_latent"] = (32,)
        policy.configure_normalizer(dims, normalizer_shapes)
        policy.to(device)
        optimizer = build_optimizer(policy, **cfg.optimizer)
        batch = dict(obs=obs, action=torch.randn(2, cfg.horizon, cfg.action_dim, device=device))
        if method in ("adaptac", "interact_dex"):
            count = 1 if method == "interact_dex" else policy.future_steps
            valid = torch.ones(2, count, dtype=torch.bool, device=device)
            valid[1, -1] = False
            batch.update(future_force=torch.randn(2, count, 5, 3, device=device), future_force_valid=valid)
        ema = copy.deepcopy(policy).eval().requires_grad_(False) if method == "maniflow" else None
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss, metrics = policy.compute_loss(batch, **policy.get_training_loss_kwargs(ema))
            if policy.loss_groups is not None:
                expected = sum(metrics[f"loss_{name}"] * weight
                               for name, (_, _, weight) in policy.loss_groups.items())
                torch.testing.assert_close(metrics["loss_action"], expected)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
            optimizer.step()
        policy.eval()
        saved_agent = {**agent_cfg, **policy.effective_config}
        with tempfile.TemporaryDirectory(prefix="dexmani-policy-check-") as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, Checkpoint(
                2, policy.state_dict(), None, optimizer.state_dict(), {}, get_rng_state(device),
                dict(agent=saved_agent, normalization_dims=dims),
            ))
            # Deployment restores all parameters in FP32, including the frozen RGB encoder.
            policy.float()
            with torch.no_grad():
                expected = policy.predict_action(obs, generator=torch.Generator(device=device).manual_seed(11))
            assert expected["future_actions"].shape == (2, policy.future_steps, policy.control_action_dim)
            restored = restore_policy(path, use_ema=False, device=device)
            torch.testing.assert_close(restored.state_dict(), policy.state_dict(), rtol=0, atol=0)
            with torch.no_grad():
                actual = restored.predict_action(obs, generator=torch.Generator(device=device).manual_seed(11))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if "rtc" in restored.supported_execution_modes:
                restored.zero_grad(set_to_none=True)
                prefix = expected["future_actions"][:, :3]
                with torch.inference_mode():
                    unguided = restored.predict_action(
                        obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=0,
                        generator=torch.Generator(device=device).manual_seed(11),
                    )
                    guided = restored.predict_action(
                        obs, rtc_prefix=prefix, delay_steps=1, rtc_guidance_cap=2,
                        generator=torch.Generator(device=device).manual_seed(11),
                    )["future_actions"]
                torch.testing.assert_close(unguided, expected, rtol=0, atol=0)
                assert guided.shape == expected["future_actions"].shape
                assert torch.isfinite(guided).all() and not guided.requires_grad
                assert all(p.grad is None for p in restored.parameters())
        print(f"PASS policy roundtrip: {method}, auxiliary={auxiliary} ({device})", flush=True)


def check_output_boundaries(device):
    from types import SimpleNamespace
    from dexmani_policy.deployment.runtime import PolicyRuntime
    from dexmani_policy.env_runner.sim_runner import SimRunner

    class OutputPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), device=device))
            self.output = torch.zeros(1, 3, 19, device=device)

        def predict_action(self, *args, **kwargs):
            return {"future_actions": self.output}

    agent = OutputPolicy()
    runtime = PolicyRuntime(agent, SimpleNamespace(inference_steps=2), seed=0)
    runner = SimRunner("test", 2, [], {})
    with patch.object(runtime, "prepare_observations", return_value={}):
        assert runtime.predict({}).shape == (3, 19)
        assert runner.predict_action_chunk({}, agent, action_steps=1).shape == (1, 19)
        for value in (float("nan"), float("inf")):
            agent.output[0, -1, 0] = value
            for predict in (lambda: runtime.predict({}),
                            lambda: runner.predict_action_chunk({}, agent, action_steps=1)):
                try:
                    predict()
                except ValueError as error:
                    assert "Nonfinite physical" in str(error)
                else:
                    raise AssertionError("Nonfinite future actions escaped the CPU boundary")


def check_sim_observations(device):
    from dexmani_policy.common.config import POINT_CLOUD_FIELDS
    from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
    from dexmani_policy.deployment.runtime import PolicyRuntime
    from dexmani_policy.env_runner.sim_runner import SimRunner

    fields = sorted(POINT_CLOUD_FIELDS)
    runner = SimRunner("unused", 2, fields, {})
    dims = dict.fromkeys(fields, 3)
    agent = Policy(torch.nn.Linear(1, 1), torch.nn.Identity(), 2, 2, 1, 19).to(device)
    agent.normalization_dims = dims
    info = SimpleNamespace(observation_fields=fields, n_obs_steps=2, pc_dim=3)
    runtime = PolicyRuntime(agent, info, seed=0)
    observations = []
    tensor = torch.as_tensor
    for index in range(3):
        observation = {key: np.arange(8 * 6, dtype=np.float32).reshape(8, 6) + index * 100
                       for key in fields}
        observations.append(observation)
        runner.update_obs(observation)
        with patch("dexmani_policy.env_runner.sim_runner.torch.as_tensor", wraps=tensor) as transfer:
            actual = runner.get_obs_batch(device, normalization_dims=dims)
        assert len(transfer.call_args_list) == len(fields)
        for call in transfer.call_args_list:
            assert call.args[0].shape == (2, 8, 3)
            assert call.args[0].flags.c_contiguous
        for key in fields:
            history = np.stack([observations[max(0, index - 1)][key], observations[index][key]])
            old_device_slice = tensor(history, device=device)[..., :3].unsqueeze(0)
            torch.testing.assert_close(actual[key], old_device_slice, rtol=0, atol=0)
        external = runtime.prepare_observations(runner.get_stacked_obs())
        torch.testing.assert_close(external, actual, rtol=0, atol=0)
    dims["object_pointcloud"] = 6
    history = {key: np.stack([obs[key] for obs in observations]).astype(np.float64) for key in fields}
    external = runtime.prepare_observations(history)
    torch.testing.assert_close(external, runner.get_obs_batch(device, normalization_dims=dims), rtol=0, atol=0)
    for key, channels in dims.items():
        assert external[key].shape == (1, 2, 8, channels)
        assert external[key].dtype == torch.float32 and external[key].is_contiguous()
        try:
            runtime.prepare_observations({**history, key: history[key][..., :channels - 1]})
        except ValueError as error:
            assert key in str(error) and "fewer channels" in str(error)
        else:
            raise AssertionError(f"Runtime accepted too few {key} channels")
    invalid = history["object_pointcloud"].copy()
    invalid[-1, 0, -1] = np.finfo(np.float64).max
    try:
        runtime.prepare_observations({**history, "object_pointcloud": invalid})
    except ValueError as error:
        assert "Nonfinite" in str(error)
    else:
        raise AssertionError("Nonfinite point cloud escaped FP32 conversion")
    runner.reset()
    assert runner.get_obs_batch(device, normalization_dims={}) == {}
    kwargs = rgb_preprocessing_kwargs({"rgb_preprocess_size": (10, 12), "rgb_color_aug": {"present": True}})
    runner = SimRunner("unused", 2, ["rgb"], kwargs)
    rgb = np.random.default_rng(3).integers(256, size=(7, 9, 3), dtype=np.uint8)
    runner.update_obs({"rgb": rgb})
    simulated = runner.get_obs_batch(device, normalization_dims={"rgb": 3})["rgb"]
    info = SimpleNamespace(observation_fields=("rgb",), n_obs_steps=2, rgb_preprocessing=kwargs)
    runtime = PolicyRuntime(torch.nn.Linear(1, 1).to(device), info, seed=0)
    external = runtime.prepare_observations({"rgb": np.stack((rgb, rgb))})["rgb"]
    assert simulated.dtype == external.dtype == torch.uint8
    assert simulated.is_contiguous() and external.is_contiguous()
    torch.testing.assert_close(simulated, external, rtol=0, atol=0)


def check_image_geometry(device):
    from dexmani_policy.agents.obs_encoder.pointcloud.projection import project_points
    from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor

    height, width = 6, 8
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((x / (width - 1), y / (height - 1), (x + y) / (height + width - 2)))[None]
    processor = ImageProcessor((10, 14), image_mean=(0, 0, 0), image_std=(1, 1, 1)).to(device)
    resized, transform = processor.process_images(image, return_transform=True)
    pixels = torch.tensor([[[2.2, 1.7], [4.1, 3.2]]], device=device)
    points = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    projection = project_points(points, torch.eye(3, device=device)[None],
                                torch.eye(4, device=device)[None, :3], transform, (resized.shape[-2], resized.shape[-1]))
    sampled = F.grid_sample(resized, projection.grid_xy[:, :, None], align_corners=False)[:, :, :, 0].transpose(1, 2)
    expected = torch.stack((pixels[..., 0] / (width - 1), pixels[..., 1] / (height - 1),
                            pixels.sum(-1) / (height + width - 2)), -1)
    assert (projection.confidence == 1).all()
    torch.testing.assert_close(sampled, expected, rtol=2e-6, atol=2e-6)
    quantized = image.mul(255).round().to(torch.uint8)
    torch.testing.assert_close(processor.process_images(quantized),
                               processor.process_images(quantized.float() / 255), rtol=0, atol=0)


def check_ddim_clipping(device):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from dexmani_policy.common.normalizer import FieldNormalizer

    limits = FieldNormalizer(torch.tensor([0.5], device=device), torch.tensor([-2.], device=device))
    physical = torch.tensor([[[2.], [3.], [5.], [6.]]], device=device)
    bounded = limits.normalize(physical)
    outside = torch.tensor([[[-2.], [-0.5], [0.5], [2.]]], device=device)
    for clip in (True, False):
        scheduler = DDIMScheduler(num_train_timesteps=10, prediction_type="sample", clip_sample=clip)
        scheduler.set_timesteps(2, device=device)
        def clean_sample(value):
            return scheduler.step(value, scheduler.timesteps[0], torch.zeros_like(value)).pred_original_sample
        torch.testing.assert_close(limits.unnormalize(clean_sample(bounded)), physical, rtol=0, atol=0)
        expected = outside.clamp(-1, 1) if clip else outside
        torch.testing.assert_close(clean_sample(outside), expected, rtol=0, atol=0)


def check_rgb(device):
    architecture = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, image_size=32, patch_size=16)
    for mode in ("freeze", "full", "lora"):
        model = ViTEncoder("siglip", tune_mode=mode, out_dim=16,
                           architecture=architecture).to(device)
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert not hasattr(model.backbone.vision_model, "head")
        assert model.effective_config["architecture"]["vision_use_head"] is False
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        rgb = torch.randn(2, 3, 32, 32, device=device)
        model.train()
        for amp in (False, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model(rgb)["global_token"].float().square().mean()
            loss.backward()
            assert model.proj.weight.grad is not None and model.proj.weight.grad.abs().sum() > 0
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if mode == "freeze":
                assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())
            optimizer.step()
        restored = ViTEncoder("siglip", **model.effective_config)
        snapshot = {name: value.clone() for name, value in model.state_dict().items()}
        restored.load_state_dict(snapshot, strict=True, assign=True)
        restored.to(device)
        restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-3)
        restored_optimizer.load_state_dict(optimizer.state_dict())
        assert list(dict(model.named_parameters())) == list(dict(restored.named_parameters()))
        with torch.no_grad():
            torch.testing.assert_close(model.eval()(rgb)["global_token"],
                                       restored.eval()(rgb)["global_token"], rtol=0, atol=0)
    pooler = ViTEncoder("siglip", global_token_type="pooler", architecture=architecture).to(device)
    assert hasattr(pooler.backbone.vision_model, "head")
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        assert torch.isfinite(pooler(rgb)["global_token"]).all()
    assert "attn_implementation" not in architecture

    from transformers import Dinov2Config, Dinov2Model, SiglipVisionConfig, SiglipVisionModel
    from dexmani_policy.training.common.config import load_policy_config, prepare_training

    for kind, config_class, model_class in (
        ("dino", Dinov2Config, Dinov2Model), ("siglip", SiglipVisionConfig, SiglipVisionModel),
    ):
        for mode in ("freeze", "full", "lora"):
            config = config_class(**architecture)
            if kind == "siglip":
                config.vision_use_head = False
            source_model = model_class(config)
            with tempfile.TemporaryDirectory(prefix="dexmani-rgb-source-") as directory:
                source_model.save_pretrained(directory)
                cfg = load_policy_config("dp", [
                    "horizon=8", "n_action_steps=3", "agent.down_dims=[16,32]",
                    "agent.diffusion_step_embed_dim=32",
                ])
                cfg.initialization.rgb = directory
                cfg.agent.rgb_backbone_name = kind
                cfg.agent.rgb_backbone_config = {"tune_mode": mode, "out_dim": 16, "image_size": [32, 32]}
                prepare_training(cfg)
                plain = cast(dict[str, Any], OmegaConf.to_container(cfg.agent, resolve=True))
                plain.pop("_target_")
                rng = torch.get_rng_state()
                reference = DP(**plain)
                initialized_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                with patch.object(ViTEncoder, "__init__", autospec=True,
                                  side_effect=ViTEncoder.__init__) as constructor:
                    policy = hydra.utils.instantiate(cfg.agent, _convert_="all")
                assert type(constructor.call_args.kwargs["architecture"]) is dict
                assert torch.equal(torch.get_rng_state(), initialized_rng)
                expected_state, actual_state = reference.state_dict(), policy.state_dict()
                assert list(actual_state) == list(expected_state)
                for key in expected_state:
                    torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
                model = policy.obs_encoder.rgb_encoder
                model.initialize_weights(directory)
                assert all(p.dtype == torch.float32 for p in model.parameters())
                base = model.backbone.get_base_model() if mode == "lora" else model.backbone
                source_state = source_model.state_dict()
                for key, value in base.state_dict().items():
                    if ".lora_" not in key:
                        torch.testing.assert_close(value, source_state[key.replace(".base_layer.", ".")], rtol=0, atol=0)
            # The source directory is gone: structure and parameters alone must restore offline.
            restored = ViTEncoder(model.kind, **model.effective_config)
            restored.load_state_dict(model.state_dict(), strict=True)
            model.to(device).eval()
            restored.to(device).eval()
            expected = model(rgb)["global_token"]
            actual = restored(rgb)["global_token"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected.float().square().mean().backward()
            actual.float().square().mean().backward()
            for (name, parameter), (other_name, other) in zip(model.named_parameters(), restored.named_parameters()):
                assert name == other_name
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


def check_rgb_deployment(device):
    from peft import PeftModel
    from dexmani_policy.common.checkpoint import load_checkpoint

    for kind in ("dino", "siglip", "dinov3"):
        architecture: dict[str, Any] = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, image_size=32, patch_size=16)
        if kind == "dino":
            architecture["model_type"] = "dinov2"
        elif kind == "dinov3":
            architecture = dict(embed_dim=32, depth=1, num_heads=4)
        kwargs: dict[str, Any] = dict(horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19, state_dim=19,
                      rgb_backbone_name=kind, state_out_dim=8, down_dims=[16, 32],
                      diffusion_step_embed_dim=32, num_training_steps=10, num_inference_steps=2,
                      rgb_backbone_config=dict(tune_mode="lora", architecture=architecture,
                                               out_dim=16, image_size=[32, 32]))
        policy = DP(**kwargs)
        dims = dict(joint_state=19, rgb=3, action=19)
        shapes = dict(joint_state=(19,), action=(19,))
        policy.configure_normalizer(dims, shapes)
        policy.to(device)
        obs = dict(joint_state=torch.randn(2, 2, 19, device=device),
                   rgb=torch.rand(2, 2, 3, 32, 32, device=device))
        batch: dict[str, Any] = dict(obs=obs, action=torch.randn(2, 8, 19, device=device))
        optimizer = build_optimizer(policy, lr=1e-3, weight_decay=0)
        for amp in (False, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, _ = policy.compute_loss(batch)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
            optimizer.step()
        assert any(p.detach().abs().sum() > 0 for n, p in policy.named_parameters() if ".lora_B." in n)
        ema = copy.deepcopy(policy)
        with torch.no_grad():
            for name, param in ema.named_parameters():
                if ".lora_B." in name:
                    param.mul_(0.8)
        config = dict(agent={"_target_": "dexmani_policy.agents.policies.dp.DP", **kwargs},
                      normalization_dims=dims)
        with tempfile.TemporaryDirectory(prefix="dexmani-lora-restore-") as directory:
            raw, averaged = copy.deepcopy(policy), copy.deepcopy(ema)
            for model in (raw, averaged):
                model.eval()
            path = save_checkpoint(Path(directory) / "model.pt", Checkpoint(
                2, raw.state_dict(), averaged.state_dict(), {}, {}, {}, config,
            ))
            for use_ema, reference in ((False, raw), (True, averaged)):
                reference.float()
                restored = restore_policy(path, use_ema=use_ema, device=device)
                assert not isinstance(restored.obs_encoder.rgb_encoder.backbone, PeftModel)
                assert not any("lora_" in name for name in restored.state_dict())
                assert all(p.dtype == torch.float32 for p in restored.parameters())
                with torch.no_grad():
                    torch.testing.assert_close(restored._encode_obs(obs), reference._encode_obs(obs),
                                               atol=2e-6, rtol=2e-6)
                for rtc in (False, True):
                    options: dict[str, Any] = dict(rtc_prefix=batch["action"][:, :3], delay_steps=1,
                                   rtc_guidance_cap=2) if rtc else {}
                    # Isolate FP32 merge error from TF32 convolution rounding.
                    with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
                        expected = reference.predict_action(obs, **options,
                            generator=torch.Generator(device=device).manual_seed(31))
                        actual = restored.predict_action(obs, **options,
                            generator=torch.Generator(device=device).manual_seed(31))
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5,
                        msg=lambda message: f"{kind}, EMA={use_ema}, RTC={rtc}: {message}")
            saved = load_checkpoint(path)
            assert any("lora_" in name for name in saved.model_state)
            resumed = DP(**kwargs)
            resumed.configure_normalizer(dims, shapes)
            resumed.load_state_dict(saved.model_state, strict=True, assign=True)
            assert isinstance(resumed.obs_encoder.rgb_encoder.backbone, PeftModel)
            assert all(p.dtype == torch.float32 for p in resumed.parameters())


def check_dinov3(device):
    from safetensors.torch import save_file
    from dexmani_policy.agents.obs_encoder.spatial import SpatialObservationEncoder
    from dexmani_policy.training.common.config import prepare_training

    architecture = dict(embed_dim=32, depth=1, num_heads=4)
    source = ViTEncoder("dinov3", architecture=architecture).backbone.state_dict()
    rgb = torch.randn(2, 3, 32, 48, device=device)
    for mode in ("freeze", "full", "lora"):
        model = ViTEncoder("dinov3", tune_mode=mode, architecture=architecture, out_dim=16)
        with tempfile.TemporaryDirectory(prefix="dexmani-dinov3-source-") as directory:
            save_file(source, str(Path(directory) / "model.safetensors"))
            model.initialize_weights(directory)
        base = model.backbone.get_base_model() if mode == "lora" else model.backbone
        for name, value in base.state_dict().items():
            if ".lora_" not in name:
                torch.testing.assert_close(value, source[name.replace(".base_layer.", ".")], rtol=0, atol=0)
        assert model.num_prefix_tokens == 5 and model.patch_size == 16
        assert all(p.dtype == torch.float32 for p in model.parameters())
        model.to(device).eval()
        with torch.no_grad():
            hidden = model.backbone.forward_features(rgb)
            output = model(rgb)
            assert output["patch_tokens"].shape == (2, 6, 16)
            torch.testing.assert_close(output["patch_tokens"], model.proj(hidden[:, 5:]))
            torch.testing.assert_close(output["global_token"], output["patch_tokens"].mean(1))
            model.global_token_type = "cls"
            torch.testing.assert_close(model(rgb)["global_token"], model.proj(hidden[:, 0]))
            model.global_token_type = "avg"
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        for amp in (False, True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model(rgb)["global_token"].float().square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            if mode == "freeze":
                assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())
            else:
                gradients = [p.grad for p in model.backbone.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            if mode == "lora":
                qkv = model.backbone.get_base_model().blocks[0].attn.qkv
                assert qkv.lora_B["default"].weight.grad.abs().sum() > 0
            optimizer.step()
        restored = ViTEncoder("dinov3", **model.effective_config).to(device)
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored.eval()(rgb), model.eval()(rgb), rtol=0, atol=0)

    for method in ("dp", "sat", "dqrise", "dexlatent"):
        cfg = load_policy_config(method, ["agent.rgb_backbone_name=dinov3"])
        with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("unexpected HF config")):
            prepare_training(cfg)
        assert cfg.agent.rgb_backbone_name == "dinov3"

    for kind in ("dino", "siglip", "dinov3"):
        arch: dict[str, Any] = architecture if kind == "dinov3" else dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, image_size=32, patch_size=16,
        )
        if kind == "dino":
            arch = {"model_type": "dinov2", **arch}
        spatial_encoder = SpatialObservationEncoder(
            num_points=32, n_obs_steps=2, rgb_backbone_name=kind,
            rgb_backbone_config=dict(architecture=arch, image_size=[32, 32]),
            pc_encoder_config=dict(channels=24, num_patches=12, group_size=8, depth=1, num_heads=4),
            fps_random_config=dict(use_random=False, use_random_start=False),
        ).to(device).eval()
        intrinsic = torch.tensor([[20., 0., 15.5], [0., 20., 15.5], [0., 0., 1.]], device=device)
        points = torch.rand(2, 32, 3, device=device) * 0.2
        points[..., 2] += 1
        obs = dict(point_cloud=points, joint_state=torch.randn(2, 19, device=device),
                   rgb=torch.randint(256, (2, 3, 32, 32), dtype=torch.uint8, device=device),
                   camera_intrinsic=intrinsic.expand(2, -1, -1),
                   camera_extrinsic=torch.eye(4, device=device)[:3].expand(2, -1, -1))
        output = spatial_encoder(obs)
        assert output.shape == (1, 26, 256) and torch.isfinite(output).all()
        output.square().mean().backward()
        gradient = spatial_encoder.alignment.proj.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0
        restored_spatial = SpatialObservationEncoder(num_points=32, n_obs_steps=2, **spatial_encoder.effective_config).to(device).eval()
        restored_spatial.load_state_dict(spatial_encoder.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored_spatial(obs), output, rtol=0, atol=0)


def check_vq(device):
    quantizer = EMAVectorQuantizer(12, 4).to(device)
    for _ in range(4):
        x = torch.randn(32, 12, device=device, requires_grad=True)
        state = {k: v.clone() for k, v in quantizer.state_dict().items()}
        expected_indices = (x.detach()[:, None] - state["embed"]).square().sum(-1).argmin(-1)
        assignment = F.one_hot(expected_indices, 4).float()
        counts = state["cluster_size"].lerp(assignment.sum(0), 1 - quantizer.decay)
        sums = state["embed_avg"].lerp(assignment.T @ x.detach(), 1 - quantizer.decay)
        smoothed = (counts + quantizer.eps) / (counts.sum() + 4 * quantizer.eps) * counts.sum()
        output, indices, loss = quantizer(x)
        assert torch.equal(indices, expected_indices)
        torch.testing.assert_close(quantizer.cluster_size, counts)
        torch.testing.assert_close(quantizer.embed_avg, sums, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(quantizer.embed, sums / smoothed[:, None], rtol=1e-5, atol=1e-6)
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        assert torch.isfinite(loss)

    quantizer.eval()
    with torch.no_grad():
        quantizer.embed.fill_(3)
        quantizer.embed[0].fill_(1)
        quantizer.embed[0, 0] += 1e-4
        quantizer.embed[1].fill_(1)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        assert quantizer(torch.ones(1, 12, device=device))[1].item() == 1

    initialized = EMAVectorQuantizer(3, 4, kmeans_init=True, kmeans_iters=2).to(device)
    initialized(torch.zeros(8, 3, device=device))
    assert initialized.is_initialized and initialized.cluster_size.sum() > 0
    restored = EMAVectorQuantizer(3, 4, kmeans_init=True).to(device)
    restored.load_state_dict(initialized.state_dict(), strict=True)
    assert restored.is_initialized
    restored.eval()(torch.zeros(8, 3, device=device))

    dead = EMAVectorQuantizer(3, 4, threshold_ema_dead_code=2).to(device)
    dead.update_codebook(torch.ones(8, 3, device=device), torch.zeros(8, device=device, dtype=torch.long))
    assert (dead.cluster_size >= 2).all()
    torch.testing.assert_close(dead.embed_avg[1:], dead.embed[1:] * 2)

    vae = HandVQVAE(12, [1] * 12, latent_dim=8, hidden_dim=16, num_layers=2).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3, fused=device.type == "cuda")
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, commitment, _ = vae(torch.randn(16, 12, device=device))
        (reconstruction + commitment).backward()
        assert vae.quantizer.layer_weights.grad is not None and vae.quantizer.layer_weights.grad.abs().sum() > 0
        optimizer.step()
    loaded = HandVQVAE.from_checkpoint({
        "model_config": vae.model_config,
        "model_state_dict": {name: value.cpu() for name, value in vae.state_dict().items()},
    }, device=device)
    assert all(torch.isfinite(p).all() for p in loaded.parameters())


def check_codebook(device):
    for size in (256, 512):
        table = HandStateCodebook(12, num_codes=size).to(device)
        table.set_prototypes(torch.randn(size, 12, device=device))
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            _, index = table.decode(torch.tensor([-2, -1, 1, 2], device=device, dtype=dtype))
            assert index.tolist() == [0, 0, size - 1, size - 1]
        for invalid in (float("nan"), float("inf"), -float("inf")):
            try:
                table.decode(torch.tensor([invalid], device=device))
            except ValueError:
                pass
            else:
                raise AssertionError("Nonfinite code index was accepted")
        with torch.no_grad():
            table.sorted_hand_poses.fill_(3)
            table.sorted_hand_poses[0].fill_(1)
            table.sorted_hand_poses[0, 0] += 1e-4
            table.sorted_hand_poses[1].fill_(1)
        for shape in ((12,), (2, 12), (2, 3, 12)):
            continuous, indices = table.encode(torch.ones(shape, device=device))
            assert indices.shape == shape[:-1] and (indices == 1).all()
            assert continuous.shape == (*shape[:-1], 1)
        restored = HandStateCodebook(12, num_codes=size).to(device)
        restored.load_state_dict(table.state_dict(), strict=True)
        assert set(restored.state_dict()) == {"sorted_hand_poses"}
    single = HandStateCodebook(12, num_codes=1).to(device)
    single.set_prototypes(torch.ones(1, 12, device=device))
    continuous, indices = single.encode(torch.ones(2, 12, device=device))
    assert (continuous == -1).all() and (indices == 0).all()
    assert (single.decode(torch.tensor([-3., 0., 3.], device=device))[1] == 0).all()
    odd = HandStateCodebook(12, num_codes=3).to(device)
    odd.set_prototypes(torch.arange(3, device=device)[:, None].expand(-1, 12))
    assert odd.decode(torch.tensor([-0.5, 0., 0.5], device=device))[1].tolist() == [1, 1, 2]


def check_vq_training(device):
    with tempfile.TemporaryDirectory(prefix="dexmani-vq-training-check-") as directory:
        source = Path(directory) / "commands.zarr"
        ends = np.cumsum([7, 9, 11, 13])
        action = np.arange(int(ends[-1]) * 19, dtype=np.float32).reshape(-1, 19) / 100
        group = zarr.open_group(str(source), mode="w")
        group.create_dataset("meta/episode_ends", data=ends)
        group.create_dataset("data/action", data=action, chunks=(5, 19))
        for val_episodes in (0, 1):
            output = Path(directory) / f"val{val_episodes}"
            cfg = load_policy_config("pretrain/hand_prior/vqvae", [
                f"dataset.zarr_path={source}", "dataset.max_train_episodes=2",
                f"dataset.val_episodes={val_episodes}", f"vq_vae.output_dir={output}",
                "vq_vae.num_epochs=2", "vq_vae.batch_size=4", "vq_vae.save_epochs=1",
                "vq_vae.latent_dim=8", "vq_vae.hidden_dim=16", "vq_vae.num_layers=1",
                "vq_vae.num_groups=1", "vq_vae.codebook_size=3", "vq_vae.warmup_steps=0",
                f"vq_vae.device={device}",
            ])
            with patch.object(vq_training, "load_hand_array", wraps=vq_training.load_hand_array) as read:
                _, val, _ = vq_training.prepare_hand_data(cfg)
                assert read.call_count == 1 + val_episodes
                assert (val is None) == (val_episodes == 0)
            evaluation = ({"wraps": evaluate_vq} if val_episodes else
                          {"side_effect": AssertionError("V=0 must not evaluate training data")})
            with patch.object(vq_training, "evaluate_vq", **evaluation) as evaluate, \
                 patch.object(vq_training, "DataLoader", wraps=DataLoader) as loaders, \
                 patch.object(vq_training, "export_codebook", wraps=export_codebook) as export:
                vq_training.train(cfg)
                assert evaluate.call_count == 2 * val_episodes
                assert loaders.call_count == 1 + val_episodes
                selected = "best" if val_episodes else "last"
                assert export.call_args.args[0] == output / f"vqvae_hand_{selected}.pt"
            assert (output / "vqvae_hand_last.pt").exists()
            assert not (output / "vqvae_hand_epoch=0002.pt").exists()
            assert (output / "vqvae_hand_best.pt").exists() == bool(val_episodes)
            periodic = torch.load(output / "vqvae_hand_epoch=0001.pt", weights_only=True)
            assert "split_metadata" not in periodic
            assert ("val_lookup_mse" in periodic["metrics"]) == bool(val_episodes)
            # Export needs neither selection information nor the training epoch.
            for key in ("metrics", "epoch"):
                del periodic[key]
            minimal = output / "minimal.pt"
            torch.save(periodic, minimal)
            table = export_codebook(minimal, output / "reexported.npz")
            assert table.num_codes == 3


def check_codebook_pipeline(device):
    model = HandVQVAE(12, [1] * 12, latent_dim=8, hidden_dim=16, num_layers=1,
                      num_groups=1, codebook_size=3).to(device)
    rows = torch.randn(19, 12).clamp(-1, 1)
    loader = DataLoader(TensorDataset(rows), batch_size=7)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = evaluate_vq(model, loader, device)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with tempfile.TemporaryDirectory(prefix="dexmani-codebook-check-") as directory:
        vq_checkpoint, artifact = Path(directory) / "vq.pt", Path(directory) / "codebook.npz"
        hand_scale = torch.linspace(0.5, 2., 12)
        hand_offset = torch.linspace(-0.2, 0.3, 12)
        torch.save(dict(
            model_config=model.model_config, model_state_dict=model.state_dict(),
            normalizer_state_dict={"fields.hand.scale": hand_scale, "fields.hand.offset": hand_offset},
        ), vq_checkpoint)
        table = export_codebook(vq_checkpoint, artifact)
        with np.load(artifact) as payload:
            assert set(payload.files) == {"sorted_hand_poses", "hand_normalizer_scale", "hand_normalizer_offset"}
        loaded_table, affine = load_codebook(artifact)
        torch.testing.assert_close(affine["scale"], hand_scale)
        torch.testing.assert_close(affine["offset"], hand_offset)
        for name, value in table.state_dict().items():
            torch.testing.assert_close(value, loaded_table.state_dict()[name], rtol=0, atol=0)
        lookup_mse = (rows[:, None] - table.sorted_hand_poses).square().sum(-1).amin(-1).mean() / model.hand_dim
        torch.testing.assert_close(torch.tensor(metrics["lookup_mse"]),
                                   lookup_mse, rtol=1e-5, atol=1e-7)
        policy_config: dict[str, Any] = dict(
            horizon=4, n_obs_steps=2, n_action_steps=2, action_dim=19,
            num_codes=loaded_table.num_codes,
            n_layers=1, hidden_dim=32, n_head=4,
            num_inference_steps=2, num_points=32,
            pc_encoder_config=dict(channels=48, num_patches=8, group_size=4, depth=2, num_heads=4),
            fps_random_config=dict(use_random=False),
            # Tiny random DINO only checks the codebook/policy pipeline.
            rgb_backbone_config=dict(
                tune_mode="freeze",
                image_size=(28, 28), architecture=dict(
                    model_type="dinov2", hidden_size=24, intermediate_size=48,
                    num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14,
                ),
            ),
        )
        policy = DQRISE(**policy_config)
        spec = dict(joint_state="limits", point_cloud="identity", rgb="identity",
                    camera_intrinsic="identity", camera_extrinsic="identity", action="limits")
        dims = dict(joint_state=19, point_cloud=3, rgb=3,
                    camera_intrinsic=9, camera_extrinsic=12, action=19)
        policy.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
        description = dict(
            hand_coordinates=dict(hand_joint_order=["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"],
                       hand_units="radians"),
            action_key="action",
        )
        policy.initialize_training(description, loaded_table)
        policy.normalizer["action"].scale[-12:].copy_(affine["scale"])
        policy.normalizer["action"].offset[-12:].copy_(affine["offset"])
        policy.to(device)
        xyz = torch.rand(2, 2, 32, 3, device=device) * 0.2 - 0.1
        xyz[..., 2] += 1
        intrinsics = torch.tensor([[40., 0., 15.5], [0., 40., 11.5], [0., 0., 1.]], device=device)
        extrinsics = torch.eye(4, device=device)[:3]
        obs = dict(
            joint_state=torch.randn(2, 2, 19, device=device), point_cloud=xyz,
            rgb=torch.randint(256, (2, 2, 24, 32, 3), dtype=torch.uint8, device=device).permute(0, 1, 4, 2, 3).contiguous(),
            camera_intrinsic=intrinsics.flatten().expand(2, 2, -1).clone(),
            camera_extrinsic=extrinsics.flatten().expand(2, 2, -1).clone(),
        )
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = policy.compute_loss(dict(obs=obs, action=torch.rand(2, 4, 19, device=device)))
        loss.backward()
        assert torch.isfinite(loss)
        prediction = policy.eval().predict_action(obs)["future_actions"]
        assert prediction.shape == (2, 3, 19) and torch.isfinite(prediction).all()
        try:
            policy._predict_from_condition(
                torch.zeros(2, 256, device=device), rtc_prefix=prediction[:, :1],
            )
        except NotImplementedError:
            pass
        else:
            raise AssertionError("DQ-RISE accepted an unsupported RTC prefix")

        ema = copy.deepcopy(policy)
        with torch.no_grad():
            next(ema.parameters()).add_(0.01)
        saved = Path(directory) / "policy.pt"
        effective = dict(
            agent={"_target_": "dexmani_policy.agents.policies.dqrise.DQRISE",
                   **policy_config, **policy.effective_config},
            normalization_dims=dims,
        )
        checkpoint = Checkpoint(
            global_step=1, model_state=policy.state_dict(), ema_model_state=ema.state_dict(),
            optimizer_state={}, scheduler_state={}, rng_state={}, effective_config=effective,
        )
        save_checkpoint(saved, checkpoint)
        artifact.unlink()
        for use_ema, expected in ((False, policy), (True, ema)):
            before_rng = get_rng_state(device)
            restored = restore_policy(saved, use_ema=use_ema, device=device)
            after_rng = get_rng_state(device)
            assert before_rng["python"] == after_rng["python"]
            assert before_rng["numpy"][0] == after_rng["numpy"][0]
            assert (before_rng["numpy"][1] == after_rng["numpy"][1]).all()
            assert before_rng["numpy"][2:] == after_rng["numpy"][2:]
            torch.testing.assert_close(before_rng["torch"], after_rng["torch"], rtol=0, atol=0)
            if device.type == "cuda":
                torch.testing.assert_close(before_rng["torch_cuda"], after_rng["torch_cuda"], rtol=0, atol=0)
            expected.float()
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            expected_prediction = expected.predict_action(
                obs, generator=torch.Generator(device=device).manual_seed(1024),
            )["future_actions"]
            actual_prediction = restored.predict_action(
                obs, generator=torch.Generator(device=device).manual_seed(1024),
            )["future_actions"]
            torch.testing.assert_close(actual_prediction, expected_prediction, rtol=0, atol=0)
        assert checkpoint.ema_model_state is not None
        key = next(iter(checkpoint.model_state))
        del checkpoint.model_state[key]
        del checkpoint.ema_model_state[key]
        invalid = Path(directory) / "incomplete.pt"
        save_checkpoint(invalid, checkpoint)
        for use_ema in (False, True):
            try:
                restore_policy(invalid, use_ema=use_ema, device=device)
            except RuntimeError as error:
                assert "Missing key(s)" in str(error)
            else:
                raise AssertionError("Inference accepted an incomplete policy state")


def check_policies(device):
    boundary = Policy(torch.nn.Identity(), torch.nn.Identity(), 4, 2, 3, 19)
    assert boundary.action_start_index == 1 and boundary.future_steps == 3
    assert Policy(torch.nn.Identity(), torch.nn.Identity(), 1, 1, 1, 19).future_steps == 1
    for horizon, obs_steps, action_steps in ((4, 2, 4), (4, 5, 1), (0, 1, 1), (4, 0, 1), (4, 2, 0)):
        try:
            Policy(torch.nn.Identity(), torch.nn.Identity(), horizon, obs_steps, action_steps, 19)
        except ValueError:
            pass
        else:
            raise AssertionError("Policy accepted an invalid action window")
    kwargs: dict[str, Any] = dict(horizon=8, n_obs_steps=2, n_action_steps=3, action_dim=19, encoder_type="dp3",
                  pc_dim=3, pc_out_dim=16, state_dim=19, num_points=64, state_out_dim=8,
                  down_dims=(32, 64), diffusion_step_embed_dim=32, n_groups=8,
                  num_training_steps=10, num_inference_steps=2)
    policy = DP3(**kwargs)
    spec = dict(joint_state="limits", point_cloud="identity", action="limits")
    dims = dict(joint_state=19, point_cloud=3, action=19)
    policy.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
    policy.to(device)
    with torch.no_grad():
        policy.normalizer["action"].scale.copy_(torch.linspace(0.5, 1.5, 19, device=device))
        policy.normalizer["action"].offset.copy_(torch.linspace(-0.2, 0.2, 19, device=device))
    obs = dict(joint_state=torch.randn(2, 2, 19, device=device),
               point_cloud=torch.rand(2, 2, 64, 3, device=device))
    optimizer = build_optimizer(policy, lr=1e-4, weight_decay=1e-6)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = policy.compute_loss(dict(obs=obs, action=torch.randn(2, 8, 19, device=device)))
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
    policy.eval()
    with torch.inference_mode():
        ordinary = policy.predict_action(
            obs, generator=torch.Generator(device=device).manual_seed(17),
        )["future_actions"]
        zero = policy.predict_action(
            obs, rtc_prefix=ordinary[:, :3], rtc_guidance_cap=0,
            generator=torch.Generator(device=device).manual_seed(17),
        )["future_actions"]
        guided = policy.predict_action(obs, rtc_prefix=ordinary[:, :3], delay_steps=1,
                                       rtc_guidance_cap=2)["future_actions"]
    assert ordinary.shape == zero.shape == guided.shape == (2, 7, 19)
    torch.testing.assert_close(ordinary, zero, rtol=0, atol=0)
    assert torch.isfinite(guided).all()
    storage = io.BytesIO()
    torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict()), storage)
    storage.seek(0)
    saved = torch.load(storage, map_location=device, weights_only=True)
    restored = DP3(**kwargs)
    restored.configure_normalizer(dims, {key: (dims[key],) for key, mode in spec.items() if mode != "identity"})
    restored.load_state_dict(saved["model"], strict=True, assign=True)
    restored.to(device)
    new_optimizer = build_optimizer(restored, lr=1e-4, weight_decay=1e-6)
    assert [g["param_names"] for g in optimizer.param_groups] == [g["param_names"] for g in new_optimizer.param_groups]
    new_optimizer.load_state_dict(saved["optimizer"])
    actions = torch.randn(2, 8, 19, device=device)
    for model, opt in ((policy, optimizer), (restored, new_optimizer)):
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(29)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            resumed_loss, _ = model.compute_loss(dict(obs=obs, action=actions))
        resumed_loss.backward()
        opt.step()
    for value, replay in zip(policy.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(value, replay, rtol=0, atol=0)


def check_resume_logging(root):
    for index, history in enumerate((b'{"step": 99}\n', b'broken\n{"step": 99}\n', b'broken', b'')):
        directory = root / f"logging-{index}"
        directory.mkdir()
        path = directory / "metrics.jsonl"
        path.write_bytes(history)
        logger = ExperimentLogger(directory, resume=True, wandb_enabled=False)
        try:
            logger.log({"loss": 1.}, step=2)
            logger.log({"loss": 0.5}, step=2)
        finally:
            logger.close()
        boundary = b"\n" if history and not history.endswith(b"\n") else b""
        prefix = history + boundary
        assert path.read_bytes().startswith(prefix)
        records = [json.loads(line) for line in path.read_bytes()[len(prefix):].splitlines()]
        assert [record["step"] for record in records] == [2, 2]
        assert list(directory.iterdir()) == [path]
    for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
        directory = root / f"fresh-{name}"
        directory.mkdir()
        (directory / name).touch()
        try:
            ExperimentLogger(directory, wandb_enabled=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Fresh training overwrote an existing experiment")


def check_normalizer_restore(root, device):
    from dexmani_policy.env_runner.sim_runner import SimRunner
    from dexmani_policy.evaluation import rollout
    from dexmani_policy.evaluation.rollout import evaluate_policy
    from dexmani_policy.evaluation.seeds import split_evaluation_seeds

    cfg = load_policy_config("dp3", ["agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16",
                                     "agent.pc_dim=3"])
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    dims = dict(joint_state=19, point_cloud=3, action=19)
    model.configure_normalizer(dims, dict(joint_state=(19,), action=(19,), hand_latent=(4,)))
    raw = model.state_dict()
    raw["normalizer.fields.action.scale"].fill_(2)
    averaged = {key: value.clone() for key, value in raw.items()}
    averaged["normalizer.fields.action.scale"].fill_(3)
    for part in ("scale", "offset"):
        averaged[f"normalizer.fields.hand_latent.{part}"] = torch.ones(5)
    config: dict[str, Any] = dict(agent=OmegaConf.to_container(cfg.agent, resolve=True), normalization_dims=dims,
                  eval=dict(use_ema=True, inference_steps=2), dataset={},
                  policy_metadata=dict(action_start_index=model.action_start_index),
                  env_runner=dict(task_name="synthetic"))
    seed_pool = list(range(10))
    checkpoint = Checkpoint(7, raw, averaged, {"momentum": torch.ones(1)},
                            {"step": torch.ones(1)}, {"torch": torch.ones(1)}, config)
    path = root / "normalizer.pt"
    save_checkpoint(path, checkpoint)
    torch_load, instantiate = torch.load, hydra.utils.instantiate
    obs = dict(joint_state=torch.randn(1, model.n_obs_steps, 19, device=device),
               point_cloud=torch.rand(1, model.n_obs_steps, 8, 3, device=device))
    for use_ema, state in ((False, raw), (True, averaged)):
        unused_refs: list[weakref.ReferenceType[torch.Tensor]] = []

        def read_checkpoint(*args, **kwargs):
            payload = torch_load(*args, **kwargs)
            unselected = payload["model_state" if use_ema else "ema_model_state"]
            unused_refs[:] = [weakref.ref(next(iter(unselected.values())))] + [
                weakref.ref(next(iter(payload[key].values())))
                for key in ("optimizer_state", "scheduler_state", "rng_state")
            ]
            return payload

        def construct(*args, **kwargs):
            assert unused_refs and all(ref() is None for ref in unused_refs)
            return instantiate(*args, **kwargs)

        def episodes(runner, agent, *, seeds, inference_steps, action_steps, video_save_dir):
            assert runner.task_name == "synthetic" and video_save_dir is None
            assert seeds == split_evaluation_seeds(seed_pool, 1024, 2)[0]
            assert inference_steps == 2 and action_steps == model.n_action_steps
            assert not agent.training
            assert all(p.dtype == torch.float32 and p.device == obs["joint_state"].device for p in agent.parameters())
            assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(1024).get_state())
            assert agent.normalizer["hand_latent"].scale.shape == restored.normalizer["hand_latent"].scale.shape
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
            expected = restored.predict_action(obs, inference_steps=inference_steps,
                generator=torch.Generator(device=device).manual_seed(17))["future_actions"]
            actual = agent.predict_action(obs, inference_steps=inference_steps,
                generator=torch.Generator(device=device).manual_seed(17))["future_actions"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for seed in seeds:
                yield dict(success=True, steps=2, total_steps=3, seed=seed)

        with patch("torch.load", side_effect=read_checkpoint) as load, \
             patch("hydra.utils.instantiate", side_effect=construct), \
             patch.object(rollout, "load_seed_pool", return_value=seed_pool) as load_seeds, \
             patch.object(SimRunner, "run", autospec=True, side_effect=episodes):
            restored = restore_policy(path, use_ema=use_ema, device=device)
            assert load.call_count == 1
            load.reset_mock()
            record, result_path = evaluate_policy(path, weights=None if use_ema else "raw",
                episodes=2, device=device, video=False, output_dir=root / f"evaluation-{use_ema}")
            assert load.call_count == 1
            load_seeds.assert_called_once_with("synthetic")
            assert record["weights"] == ("ema" if use_ema else "raw") and record["global_step"] == 7
            assert json.loads(result_path.read_text())["seeds"] == record["seeds"]
            assert record["runtime"]["parameter_dtypes"] == ["torch.float32"]
        assert "point_cloud" not in restored.normalizer.fields
        assert restored.normalization_dims["point_cloud"] == 3
        assert tuple(restored.normalizer["hand_latent"].scale.shape) == ((5,) if use_ema else (4,))
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value.to(device), rtol=0, atol=0)
    for bad_key in ("scale", "offset", "shape"):
        state = dict(raw)
        if bad_key == "shape":
            state["normalizer.fields.action.offset"] = torch.zeros(18)
        else:
            del state[f"normalizer.fields.action.{bad_key}"]
        invalid = save_checkpoint(root / f"invalid-normalizer-{bad_key}.pt",
                                  Checkpoint(7, state, None, {}, {}, {}, config))
        try:
            restore_policy(invalid, use_ema=False, device=device)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Strict loading accepted invalid affine {bad_key}")
    checkpoint.ema_model_state = None
    save_checkpoint(path, checkpoint)
    try:
        restore_policy(path, use_ema=True, device=device)
    except ValueError as error:
        assert "EMA" in str(error)
    else:
        raise AssertionError("Missing EMA silently selected raw weights")
    try:
        evaluate_policy(path, seeds=[1], device=device, video=False,
                        output_dir=root / "evaluation-missing-ema")
    except ValueError as error:
        assert "EMA" in str(error)
    else:
        raise AssertionError("Evaluation silently selected raw weights for missing EMA")
    assert not (root / "evaluation-missing-ema/result_details.json").exists()


def check_deployment_facts(root):
    path = root / "deployment.zarr"
    source = make_source(path, [3])
    source.create_dataset("data/depth", data=np.ones((3, 4, 5), dtype=np.uint16))
    source.create_dataset("data/fingertip_xyz", data=np.zeros((3, 5, 3), dtype=np.float32))
    cfg = load_policy_config("dp3", [f"zarr_path={path}", "horizon=2", "n_obs_steps=1", "n_action_steps=1",
        "dataset.sensor_modalities=[joint_state,point_cloud,hand_pointcloud,fingertip_xyz,depth]"])
    cfg.workspace.output_dir = str(root / "deployment-run")
    _, description = build_dataset(cfg)
    assert "action_start_index" not in description
    assert description["deployment_metadata"] == {"depth_config": {"shape": [4, 5], "depth_scale_m": None}}
    source.attrs.update(dt=0.02, pointcloud_config={"num_points": 8},
                        hand_pointcloud_config={"num_points": 8},
                        fingertip_link_names=["a", "b", "c", "d", "e"], depth_scale_m_per_unit=0.001,
                        hand_units="radians", hand_joint_order=["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"],
                        unused_collector_note={"large": "irrelevant"})
    _, description = build_dataset(cfg)
    assert "format" not in source.attrs
    assert "attrs" not in description and "unused_collector_note" not in description
    assert description["hand_coordinates"] == {
        "hand_units": "radians", "hand_joint_order": ["thumb:3", "index:3", "middle:2", "ring:2", "pinky:2"],
    }
    assert description["deployment_metadata"] == {
        "control_dt_s": 0.02, "pointcloud_config": {"num_points": 8},
        "hand_pointcloud_config": {"num_points": 8}, "fingertip_link_names": ["a", "b", "c", "d", "e"],
        "depth_config": {"shape": [4, 5], "depth_scale_m": 0.001},
    }
    model = SimpleNamespace(normalization_dims={"joint_state": 19, "action": 19},
                            consumed_observation_fields=("joint_state",),
                            inference_metadata={"action_start_index": 0, "control_action_dim": 19},
                            effective_config={})
    effective = assemble_effective_config(cfg, description, model)
    assert effective.deployment_metadata == {"control_dt_s": 0.02}
    assert not OmegaConf.is_readonly(effective)
    cfg.agent.action_dim = 3
    assert effective.agent.action_dim == 19


def check_inherited_policy(root, method, device=torch.device("cpu")):
    from dexmani_policy.agents.policies.afro import AFROEncoder
    from dexmani_policy.agents.policies.cordvip import CordViPEncoder

    path = root / f"{method}.zarr"
    source = make_source(path, [6, 12, 18])
    encoder = AFROEncoder(2).pc_encoder if method == "afro" else CordViPEncoder(2).interaction_encoder
    fields = ("point_cloud",) if method == "afro" else ("hand_pointcloud", "object_pointcloud", "joint_state")
    state = {} if method == "afro" else {
        "fields.hand_pointcloud.scale": torch.full((3,), 1.5),
        "fields.hand_pointcloud.offset": torch.full((3,), 2.),
        "fields.joint_state.scale": torch.full((19,), 2.),
        "fields.joint_state.offset": torch.full((19,), 3.),
    }
    modes = {key: "identity" for key in fields}
    if method == "cordvip":
        modes.update(hand_pointcloud="gaussian", joint_state="limits")
    artifact = root / f"{method}-encoder.pt"
    torch.save(dict(encoder_state=encoder.state_dict(), normalizer_state=state, normalization=modes), artifact)
    overrides = [
        f"zarr_path={path}", f"initialization.encoder={artifact}", f"workspace.output_dir={root / method}",
        "workspace.wandb_cfg=null", f"training.device={device}", "training.use_compile=false",
        "training.lr_warmup_steps=0", "training.loop.total_train_steps=8", "training.loop.log_interval_steps=1",
        "horizon=4", "n_obs_steps=2", "n_action_steps=2", "agent.down_dims=[16,32]",
        "agent.diffusion_step_embed_dim=16", "agent.num_inference_steps=2",
        "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
    ]
    if method == "cordvip":
        overrides += ["normalization.hand_pointcloud=identity"]
    else:
        overrides += ["~normalization.point_cloud"]
    extra = "object_pointcloud" if method == "afro" else "point_cloud"
    sensors = ("point_cloud", "joint_state", extra) if method == "afro" else (*fields, extra)
    overrides += [f"dataset.sensor_modalities=[{','.join(sensors)}]"]
    cfg = prepare_training(load_policy_config(method, overrides))
    if method == "afro":
        incomplete = load_policy_config(method, [*overrides, "~normalization.action"])
        with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Fit before required config")):
            try:
                run_training(incomplete, max_updates=1)
            except ValueError as error:
                assert "Missing normalization modes" in str(error) and "action" in str(error)
            else:
                raise AssertionError("Missing action normalization accepted")
    original = DemonstrationDataset.iter_normalization_data

    def only_new_fields(dataset, key):
        assert key not in (*fields, extra), f"Fitted inherited or unused field {key}"
        yield from original(dataset, key)

    with patch.object(DemonstrationDataset, "iter_normalization_data", only_new_fields):
        assert run_training(cfg, max_updates=2) == 2
    checkpoint_path = root / method / "checkpoints/latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    assert {key: checkpoint.effective_config["normalization"][key] for key in fields} == modes
    assert extra not in checkpoint.effective_config["normalization_dims"]
    for key, value in state.items():
        torch.testing.assert_close(checkpoint.model_state[f"normalizer.{key}"], value, rtol=0, atol=0)
    artifact.unlink()
    info = inspect_policy(root / method, checkpoint="latest", weights="ema" if cfg.training.use_ema else "raw")
    assert extra not in info.observation_fields
    runtime = load_policy(info, device=device)
    restored = runtime.agent
    observation = {key: source[f"data/{key}"][:2] for key in info.observation_fields}
    assert runtime.predict(observation).shape == (3, 19)
    validate_observation_fields(restored, [*restored.consumed_observation_fields, extra])
    assert restored._preprocess_obs({extra: torch.empty(1, 2, 8, 6, device=device)}) == {}
    resumed_cfg, checkpoint = prepare_resume(root / method, device=str(device))
    with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")):
        assert run_training(resumed_cfg, checkpoint=checkpoint, max_updates=1) == 3


def check_codebook_training(root, device=torch.device("cpu")):
    path = root / "dqrise.zarr"
    source = make_source(path, [6, 12, 18])
    rng = np.random.default_rng(9)
    points = rng.uniform(-0.1, 0.1, (18, 32, 3)).astype(np.float32)
    points[..., 2] += 1
    source.create_dataset("data/point_cloud", data=points, overwrite=True)
    source.create_dataset("data/rgb", data=rng.integers(256, size=(18, 28, 28, 3), dtype=np.uint8))
    intrinsic = np.array([[20, 0, 13.5], [0, 20, 13.5], [0, 0, 1]], dtype=np.float32)
    source.create_dataset("data/camera_intrinsic", data=np.tile(intrinsic.reshape(1, 9), (18, 1)))
    source.create_dataset("data/camera_extrinsic", data=np.tile(np.eye(4, dtype=np.float32)[:3].reshape(1, 12), (18, 1)))
    artifact = root / "codebook.npz"
    scale = np.linspace(1, 2, 12, dtype=np.float32)
    offset = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    np.savez(artifact, sorted_hand_poses=rng.uniform(-1, 1, (3, 12)).astype(np.float32),
             hand_normalizer_scale=scale, hand_normalizer_offset=offset)
    cfg = load_policy_config("dqrise", [
        f"zarr_path={path}", f"initialization.codebook={artifact}", "initialization.rgb=null",
        f"workspace.output_dir={root / 'dqrise'}", "workspace.wandb_cfg=null",
        "horizon=4", "n_action_steps=2", "dataloader.batch_size=2", "dataloader.num_workers=0",
        "dataset.max_train_episodes=100",
        "dataloader.pin_memory=false", f"training.device={device}", "training.use_compile=false",
        "training.lr_warmup_steps=0", "training.loop.total_train_steps=8", "training.loop.log_interval_steps=1",
    ])
    with open_dict(cfg.agent):
        cfg.agent.update(dict(
            num_points=32, n_layers=1, hidden_dim=32, n_head=4, num_inference_steps=2,
            pc_encoder_config=dict(channels=48, num_patches=8, group_size=4, depth=2, num_heads=4),
            fps_random_config=dict(use_random=False, use_random_start=False),
            rgb_backbone_config=dict(tune_mode="freeze", image_size=[28, 28], architecture=dict(
                model_type="dinov2", hidden_size=24, intermediate_size=48,
                num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14,
            )),
        ))
    prepare_training(cfg)
    with patch("dexmani_policy.agents.hand_prior.codebook.load_codebook", wraps=load_codebook) as load:
        assert run_training(cfg, max_updates=2) == 2
        assert load.call_count == 1
    checkpoint_path = root / "dqrise/checkpoints/latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint.effective_config["agent"]["num_codes"] == 3
    torch.testing.assert_close(checkpoint.model_state["normalizer.fields.action.scale"][-12:], torch.from_numpy(scale))
    torch.testing.assert_close(checkpoint.model_state["normalizer.fields.action.offset"][-12:], torch.from_numpy(offset))
    artifact.unlink()
    info = inspect_policy(root / "dqrise", checkpoint="latest", weights="raw")
    runtime = load_policy(info, device=device)
    observation = {key: source[f"data/{key}"][:2] for key in info.observation_fields}
    assert runtime.predict(observation).shape == (3, 19)
    runtime.close()
    resumed_cfg, checkpoint = prepare_resume(root / "dqrise", device=str(device))
    assert resumed_cfg.dataset.max_train_episodes == 100
    with patch("dexmani_policy.agents.hand_prior.codebook.load_codebook", side_effect=AssertionError("Reload on resume")), \
         patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")):
        assert run_training(resumed_cfg, checkpoint=checkpoint, max_updates=1) == 3


def check_configs(_device):
    for method in ("dp", "dp3", "r3d", "afro", "cordvip", "maniflow", "sat", "dqrise", "dexlatent", "adaptac", "interact_dex"):
        cfg = load_policy_config(method)
        validate_config(cfg)
        assert cfg.dataset.max_train_episodes is None
    validate_config(load_policy_config("dp3", ["normalization.action=gaussian", "agent.clip_sample=false",
                                               "normalization.point_cloud=identity"]))


def check_training_lifecycle(device):
    with tempfile.TemporaryDirectory(prefix="dexmani-training-checks-") as directory:
        root = Path(directory)
        check_resume_logging(root)
        check_normalizer_restore(root, device)
        check_deployment_facts(root)
        for method in ("afro", "cordvip"):
            check_inherited_policy(root, method, device)
            print(f"PASS {method} inherited affine/identity and resume without source ({device})", flush=True)
        check_codebook_training(root, device)
        print(f"PASS DQ-RISE fresh/save/inference/resume and saved budget ({device})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    for check in (check_configs, check_training_input_filter, check_evaluation, check_time, check_precision, check_decoder_precision, check_tactile, check_interaction,
                  check_force_interaction, check_token_decoders, check_policy_roundtrips, check_rgb,
                  check_rgb_deployment, check_dinov3, check_vq,
                  check_codebook, check_vq_training, check_codebook_pipeline, check_policies, check_output_boundaries,
                  check_sim_observations, check_image_geometry, check_ddim_clipping, check_training_lifecycle):
        check(device)
        print(f"PASS {check.__name__} ({device})", flush=True)


if __name__ == "__main__":
    main()
