"""Run training checks: python -m dexmani_policy.test_tools.training_checks."""

import argparse
import copy
import io
import json
import os
import shutil
import signal
import tempfile
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict
import torch

from dexmani_policy.agents.hand_prior.codebook import load_codebook
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, restore_policy, save_checkpoint
from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.common.pytorch_util import get_rng_state
from dexmani_policy.datasets.demonstration import DemonstrationDataset
from dexmani_policy.deployment import inspect_policy, load_policy
from dexmani_policy.test_tools.dataset_checks import make_source
from dexmani_policy.train import build_trainer, run_trainer, run_training
from dexmani_policy.training.common.config import assemble_effective_config, load_policy_config, validate_config, validate_normalization
from dexmani_policy.training.common.data import build_dataset
from dexmani_policy.training.common.logging import ExperimentLogger
from dexmani_policy.training.common.optimization import build_optimizer
from dexmani_policy.training.common.resume import prepare_resume


def check_evaluation(_device):
    from dexmani_policy.common.io_util import write_json
    from dexmani_policy.deployment import metadata
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
        normalization_dims=dict(joint_state=19, action=19), dataset={}, deployment_metadata={},
        policy_metadata=dict(action_start_index=0, control_action_dim=19, supported_execution_modes=["sync"]),
        env_runner=dict(task_name="test", env_kwargs=dict(frame_skip=6, max_stability_retries=0),
                        env_rebuild_interval=1),
    )
    with tempfile.TemporaryDirectory(prefix="dexmani-evaluation-check-") as temporary:
        root = Path(temporary)
        config["workspace"] = dict(output_dir=str(root / "original-run"))
        checkpoints = [save_checkpoint(root / folder / "model.pt", Checkpoint(
            step, {}, {}, {}, {}, {}, config)) for step, folder in ((1, "external-a"), (2, "external-b"))]
        run = root / "run"
        (run / "checkpoints").mkdir(parents=True)
        link = run / "checkpoints/link.pt"
        link.symlink_to(checkpoints[1])
        for explicit in (checkpoints[1], link, "link.pt"):
            assert inspect_policy(run, checkpoint=explicit).checkpoint_path == checkpoints[1]
        with patch.object(metadata, "PROJECT_ROOT", root):
            assert inspect_policy("run", checkpoint="link.pt").checkpoint_path == checkpoints[1]
        directory = root / "selection"
        calls = []

        def evaluate(checkpoint, **options):
            calls.append(options)
            assert options["weights"] == "ema"
            assert options["inference_steps"] == options["action_steps"] == 2
            result = dict(global_step=load_checkpoint(checkpoint).global_step,
                          statistics=dict(success_rate=0.8, avg_steps=10))
            output = rollout.create_run_directory(None, options["output_dir"])
            path = output / "result_details.json"
            write_json(path, result)
            return result, path

        with patch.object(selection, "evaluate_policy", side_effect=evaluate), \
             patch.object(selection, "load_seed_pool", return_value=pool):
            path = selection.run_pipeline(checkpoints, output_dir=directory,
                                           selection_episodes=2, test_episodes=3, device="cpu")
        assert calls[0]["seeds"] == calls[1]["seeds"]
        assert set(calls[0]["seeds"]).isdisjoint(calls[2]["seeds"])
        record = json.loads(path.read_text())
        assert (path.parent / record["ckpt_relpath"]).resolve() == checkpoints[1]
        assert (path.parent / record["test_result"]).is_file()
        assert record["inference"] == dict(use_ema=True, inference_steps=2, action_steps=2)
        internal = [save_checkpoint(run / "checkpoints" / f"step_{i:08d}.pt",
                                    load_checkpoint(checkpoint))
                    for i, checkpoint in enumerate(checkpoints, 1)]
        with patch.object(selection, "evaluate_policy", side_effect=evaluate), \
             patch.object(selection, "load_seed_pool", return_value=pool):
            default_result = selection.run_pipeline(internal, selection_episodes=2, test_episodes=3, device="cpu")
        assert default_result.parent.parent == run / "eval"
        moved = root / "moved"
        shutil.copytree(run, moved, symlinks=True)
        moved_result = moved / default_result.relative_to(run)
        moved_record = json.loads(moved_result.read_text())
        moved_winner = (moved_result.parent / moved_record["ckpt_relpath"]).resolve()
        assert moved_winner == moved / "checkpoints/step_00000002.pt"
        assert load_checkpoint(moved_winner).global_step == moved_record["global_step"] == 2
        assert (moved_result.parent / moved_record["test_result"]).is_file()
        assert not (run / "checkpoints/best.pt").exists()
        # The same saved file has the same defaults via latest or an explicit path.
        (run / "checkpoints/latest.pt").symlink_to(checkpoints[1])
        latest = inspect_policy(run)
        explicit_info = inspect_policy(run, checkpoint=checkpoints[1])
        assert latest == explicit_info
        @dataclass
        class Randomization:
            randomize_model_id: bool = True
            table_height_range: tuple[float, float] = (-0.01, 0.02)

        @dataclass
        class Stability:
            stable_check_steps: int = 240
            # A distinct fake default makes a dropped checkpoint zero observable.
            max_stability_retries: int = 3

        class Environment:
            def __init__(self, frame_skip=15, action_cnt_limit=3, success_hold_steps=2,
                         max_stability_retries=3):
                self.frame_skip = frame_skip
                self.action_cnt_limit = action_cnt_limit
                self.success_hold_steps = success_hold_steps
                self.randomization = Randomization()
                self.stability = Stability(max_stability_retries=max_stability_retries)
                self.closed = False

            @property
            def dt(self):
                return self.frame_skip / 240

            def observation(self):
                return dict(joint_state=np.full(19, self.action_cnt, dtype=np.float32), point_cloud=object())

            def reset(self, *, seed):
                self.action_cnt = 0
                if fail == "reset":
                    raise RuntimeError("injected reset failure")
                return self.observation(), {}

            def step(self, action):
                assert self.action_cnt < self.action_cnt_limit and not self.closed
                self.action_cnt += 1
                if fail == "step" and self.action_cnt == 2:
                    raise RuntimeError("injected rollout failure")
                done = self.action_cnt == self.action_cnt_limit
                return self.observation(), 0, done, False, dict(
                    success=done, success_condition=self.action_cnt >= 2)

            def close(self):
                self.closed = True

        agent = SimpleNamespace(parameters=torch.nn.Linear(1, 1).parameters,
                                n_obs_steps=3, n_action_steps=1, future_steps=2,
                                normalization_dims=config["normalization_dims"])

        for fail in (None, "reset", "step"):
            directory = root / f"rollout-{fail}"
            environments: list[Environment] = []
            runners: list[SimRunner] = []

            def make_env(runner, *, record_video):
                assert not record_video
                assert runner.n_obs_steps == agent.n_obs_steps
                assert runner.sensor_modalities == ["joint_state"]
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
                 patch.object(SimRunner, "make_env", autospec=True, side_effect=make_env):
                try:
                    record, path = rollout.evaluate_policy(checkpoints[0], seeds=[8, 3], episodes=9, device="cpu",
                                                           output_dir=directory, video=False)
                except RuntimeError as error:
                    assert "injected" in str(error)
                    assert fail is not None and not (directory / "result_details.json").exists()
                    assert len(environments) == 1
                else:
                    assert fail is None and path.is_file()
                    assert json.loads(json.dumps(record)) == json.loads(path.read_text())
                    assert "force_inference_steps" not in record
                    assert record["global_step"] == 1 and record["action_steps"] == agent.n_action_steps
                    assert record["runner_options"]["n_obs_steps"] == agent.n_obs_steps
                    assert record["runner_options"]["sensor_modalities"] == ["joint_state"]
                    assert record["runner_options"]["env_kwargs"] == {
                        "frame_skip": 6, "max_stability_retries": 0}
                    assert record["seeds"] == [row["seed"] for row in record["episodes"]] == [8, 3]
                    assert record["statistics"]["n_episodes"] == len(record["episodes"]) == 2
                    assert not Path(record["checkpoint"]).is_absolute()
                    assert (path.parent / record["checkpoint"]).resolve() == checkpoints[0]
                    assert all(row["steps"] == 2 and row["total_steps"] == 3 for row in record["episodes"])
                    assert record["environment"] == dict(
                        control_dt_s=6 / 240, frame_skip=6, action_cnt_limit=3, success_hold_steps=2,
                        randomization=dict(randomize_model_id=True, table_height_range=(-0.01, 0.02)),
                        stability=dict(stable_check_steps=240, max_stability_retries=0))
                    moved_checkpoint = moved / internal[0].relative_to(run)
                    _, relocated_result = rollout.evaluate_policy(moved_checkpoint, seeds=[8], device="cpu", video=False)
                    assert relocated_result.parent.parent == moved / "eval"
                    checkpoint_bytes = moved_checkpoint.read_bytes()
                    relocated_run = root / "relocated-run"
                    shutil.move(moved, relocated_run)
                    assert not moved.exists()
                    relocated_result = relocated_run / relocated_result.relative_to(moved)
                    relocated_record = json.loads(relocated_result.read_text())
                    assert not Path(relocated_record["checkpoint"]).is_absolute()
                    referenced_checkpoint = (relocated_result.parent / relocated_record["checkpoint"]).resolve()
                    assert referenced_checkpoint == relocated_run / moved_checkpoint.relative_to(moved)
                    assert referenced_checkpoint.read_bytes() == checkpoint_bytes
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


def check_sim_construction(_device):
    from dexmani_policy.env_runner.sim_runner import SimRunner

    for construction_error, cleanup_error in (
        (None, None),
        (RuntimeError("injected construction failure"), None),
        (RuntimeError("injected construction failure"), OSError("injected cleanup failure")),
        (KeyboardInterrupt(), None),
        (KeyboardInterrupt(), OSError("injected cleanup failure")),
    ):
        environments = []

        class Environment:
            def __init__(self, *, record_video, **kwargs):
                assert record_video is False
                assert kwargs == {"render_mode": "rgb_array", "max_stability_retries": 0}
                self.resource = object()
                self.close_calls = 0
                environments.append(self)
                if construction_error is not None:
                    raise construction_error

            def close(self):
                self.close_calls += 1
                self.resource = None
                if cleanup_error is not None:
                    raise cleanup_error

        def resolve_env_class(task_name):
            assert task_name == "test"
            return Environment

        runner = SimRunner("test", 1, [], {}, env_kwargs={"max_stability_retries": 0})
        errors = io.StringIO()
        with patch.dict("sys.modules", {"dexmani_sim": SimpleNamespace(resolve_env_class=resolve_env_class)}), \
             redirect_stderr(errors):
            if construction_error is None:
                env = runner.make_env(record_video=False)
                assert env.resource is not None and env.close_calls == 0
                env.close()
            else:
                try:
                    list(runner.run(SimpleNamespace(),
                                    seeds=[8, 3], inference_steps=2, action_steps=1))
                except BaseException as error:
                    assert error is construction_error
                else:
                    raise AssertionError("Construction failure did not propagate")
        assert len(environments) == 1
        assert environments[0].resource is None and environments[0].close_calls == 1
        if cleanup_error is not None:
            assert "OSError: injected cleanup failure" in errors.getvalue()
        else:
            assert not errors.getvalue()


def check_flow_recipes(device):
    for method, flow_dim in (("sat", 19), ("dqrise", 8), ("dexlatent", 39)):
        defaults = load_policy_config(method)
        assert (defaults.n_obs_steps, defaults.horizon, defaults.n_action_steps) == (2, 16, 8)
        assert (defaults.agent.n_layers, defaults.agent.hidden_dim, defaults.dataloader.batch_size) == (6, 768, 128)
        assert defaults.training.loop.total_train_steps == 100000
        overrides = [
            "workspace.output_dir=/tmp/flow-recipe-check", "training.use_compile=false",
            "agent.n_layers=1", "agent.hidden_dim=32", "agent.n_head=4", "agent.timestep_embed_dim=32",
            "agent.p_drop_attn=0", "agent.num_points=16", "agent.pc_encoder_config.channels=24",
            "agent.pc_encoder_config.num_patches=4", "agent.pc_encoder_config.group_size=4",
            "agent.pc_encoder_config.depth=1", "agent.pc_encoder_config.num_heads=4",
            "agent.fps_random_config.use_random=false", "agent.rgb_backbone_config.image_size=[28,28]",
            "+agent.rgb_backbone_config.architecture={model_type:dinov2,hidden_size:24,"
            "intermediate_size:48,num_hidden_layers:1,num_attention_heads:4,image_size:28,patch_size:14}",
        ]
        if method == "dqrise":
            overrides += ["+agent.num_codes=3"]
        cfg = load_policy_config(method, overrides)
        policy = hydra.utils.instantiate(cfg.agent, _convert_="all")
        assert cfg.agent.num_inference_steps == cfg.eval.inference_steps == policy.action_decoder.num_inference_steps == 5
        assert policy.action_decoder.t_sample_mode == "uniform"
        assert policy.action_decoder.time_shift_alpha == policy.effective_config["time_shift_alpha"] == 3.
        assert policy.action_decoder.time_sampler.num_steps == 4
        assert policy.obs_encoder.effective_config.items() <= policy.effective_config.items()
        if method == "dqrise":
            policy.hand_codebook.sorted_hand_poses.copy_(torch.linspace(-1, 1, 36).reshape(3, 12))
            assert policy.effective_config["num_codes"] == 3
        dims = dict(joint_state=19, point_cloud=3, rgb=3, camera_intrinsic=9, camera_extrinsic=12, action=19)
        shapes = {key: shape for key, shape in normalization_feature_shapes(dims).items()
                  if cfg.normalization[key] != "identity"}
        if method == "dexlatent":
            shapes["hand_latent"] = (32,)
        policy.configure_normalizer(dims, shapes)
        policy.to(device)
        obs = dict(joint_state=torch.randn(2, 2, 19, device=device),
                   point_cloud=torch.randn(2, 2, 16, 3, device=device) * .05,
                   rgb=torch.randint(256, (2, 2, 3, 28, 28), dtype=torch.uint8, device=device),
                   camera_intrinsic=torch.tensor([20., 0., 13.5, 0., 20., 13.5, 0., 0., 1.], device=device).expand(2, 2, 9),
                   camera_extrinsic=torch.eye(4, device=device)[:3].flatten().expand(2, 2, 12))
        obs["point_cloud"][..., 2] += 1
        optimizer = build_optimizer(policy, **cfg.optimizer)
        grouped = set()
        decay_sides = set()
        for group in optimizer.param_groups:
            assert group["lr"] == 1e-4 and tuple(group["betas"]) == (.95, .999)
            for name, param in zip(group["param_names"], group["params"]):
                assert param.requires_grad and id(param) not in grouped
                grouped.add(id(param))
                side = "obs" if name.startswith("obs_encoder.") else "action"
                assert side == "obs" or name.startswith("action_decoder.")
                owner, field = name.rsplit(".", 1)
                module = policy.get_submodule(owner)
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d,
                                       torch.nn.MultiheadAttention)) and field.endswith("weight"):
                    assert group["weight_decay"] == (1e-6 if side == "obs" else 1e-3)
                    decay_sides.add(side)
                else:
                    assert group["weight_decay"] == 0, (name, type(module), group["weight_decay"])
        assert decay_sides == {"obs", "action"}
        assert grouped == {id(p) for p in policy.parameters() if p.requires_grad}
        assert any(not p.requires_grad for p in policy.obs_encoder.rgb_encoder.parameters())
        batch = dict(obs=obs, action=torch.randn(2, 16, 19, device=device))
        with patch.object(policy.action_decoder, "compute_loss", wraps=policy.action_decoder.compute_loss) as flow:
            loss, _ = policy.compute_loss(batch)
        assert flow.call_args.args[1].shape == (2, 16, flow_dim)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.action_decoder.parameters())
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        policy.eval()
        forward = policy.action_decoder.model.forward
        for steps in (None, 7):
            reference = None
            for alpha in (1., 3.):
                policy.action_decoder.time_shift_alpha = alpha
                evaluations = []

                def velocity(**kwargs):
                    result = forward(**kwargs)
                    evaluations.append((kwargs["x"].clone(), kwargs["timestep"].clone(), result.clone()))
                    return result

                with patch.object(policy.action_decoder.model, "forward", side_effect=velocity):
                    prediction = policy.predict_action(obs, inference_steps=steps,
                                                       generator=torch.Generator(device=device).manual_seed(19))
                count = steps or 5
                assert len(evaluations) == count
                for index, (state, time, value) in enumerate(evaluations):
                    torch.testing.assert_close(time, torch.full_like(time, index / count), rtol=0, atol=0)
                    if index + 1 < count:
                        torch.testing.assert_close(evaluations[index + 1][0], state + value / count)
                assert prediction["future_actions"].shape == (2, 15, 19)
                assert prediction["future_actions"][:, :policy.n_action_steps].shape == (2, 8, 19)
                assert torch.isfinite(prediction["future_actions"]).all()
                if reference is not None:
                    torch.testing.assert_close(prediction, reference, rtol=0, atol=0)
                reference = prediction
        effective = assemble_effective_config(cfg, {"deployment_metadata": {}}, policy)
        assert effective.agent.time_shift_alpha == 3.
        with tempfile.TemporaryDirectory(prefix="dexmani-flow-recipe-") as temporary:
            for legacy in (False, True):
                saved_cfg = OmegaConf.to_container(effective, resolve=True)
                state = copy.deepcopy(optimizer.state_dict())
                if legacy:
                    saved_cfg["agent"].pop("time_shift_alpha")
                    saved_cfg["agent"]["num_inference_steps"] = saved_cfg["eval"]["inference_steps"] = 10
                    saved_cfg["optimizer"]["weight_decay"] = 1e-6
                    for group in state["param_groups"]:
                        if group["weight_decay"]:
                            group["weight_decay"] = 1e-6
                path = save_checkpoint(Path(temporary) / f"{legacy}.pt", Checkpoint(
                    1, policy.state_dict(), None, state, {}, {}, saved_cfg))
                restored = restore_policy(path, use_ema=False, device=device)
                assert restored.action_decoder.time_shift_alpha == (1. if legacy else 3.)
                assert restored.action_decoder.num_inference_steps == (10 if legacy else 5)
                torch.testing.assert_close(restored.state_dict(), policy.state_dict(), rtol=0, atol=0)
                restored_optimizer = build_optimizer(restored, **cfg.optimizer)
                restored_optimizer.load_state_dict(load_checkpoint(path).optimizer_state)
                restored_state = restored_optimizer.state_dict()
                assert restored_state["param_groups"] == state["param_groups"]
                torch.testing.assert_close(restored_state["state"], state["state"], rtol=0, atol=0)
                expected = policy.predict_action(obs, inference_steps=10 if legacy else None,
                                                 generator=torch.Generator(device=device).manual_seed(19))
                actual = restored.predict_action(obs, generator=torch.Generator(device=device).manual_seed(19))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        override = load_policy_config(method, ["agent.time_shift_alpha=2.0", "agent.num_inference_steps=3"])
        assert override.agent.time_shift_alpha == 2. and override.eval.inference_steps == 3
        print(f"PASS {method}: LBS, 5-step Euler, optimizer groups, loss/backward and checkpoint restore", flush=True)


def check_policy_roundtrips(device):
    for method, auxiliary in (("dp", False), ("afro", False), ("maniflow", False),
                              ("dexlatent", False), ("cordvip", False), ("adaptac", False), ("interact_dex", False),
                              ("sat", False), ("r3d", False), ("r3d", True)):
        overrides = [] if method == "cordvip" else ["horizon=8", "n_action_steps=2", "agent.num_inference_steps=2"]
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
            count = len(policy.future_force_horizons) if method == "interact_dex" else policy.future_steps
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
                dict(agent=saved_agent, normalization_dims=dims,
                     normalization=OmegaConf.to_container(cfg.normalization)),
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


def check_resume_logging(root):
    complete = b'{"step": 98}\n{"step": 99}\n'
    partial = b'{"loss": "' + b'x' * 9000
    for index, (history, prefix) in enumerate((
        (b'', b''), (complete, complete), (complete + partial, complete), (partial, b''),
    )):
        directory = root / f"logging-{index}"
        directory.mkdir()
        path = directory / "metrics.jsonl"
        path.write_bytes(history)
        logger = ExperimentLogger(directory, resume=True, wandb_enabled=False)
        try:
            assert path.read_bytes() == prefix
            logger.log({"loss": 1.}, step=2)
            logger.log({"loss": 0.5}, step=2)
        finally:
            logger.close()
        assert path.read_bytes().startswith(prefix)
        records = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert [record["step"] for record in records] == ([98, 99] if prefix else []) + [2, 2]
        assert list(directory.iterdir()) == [path]
    for name in ("config.yaml", "metrics.jsonl", "checkpoints"):
        directory = root / f"fresh-{name}"
        directory.mkdir()
        (directory / name).touch()
        try:
            run_training(load_policy_config("dp3", [
                f"workspace.output_dir={directory}", f"zarr_path={root / 'missing.zarr'}",
            ]), max_updates=1)
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
    model.configure_normalizer(dims, dict(joint_state=(19,), action=(19,)))
    raw = model.state_dict()
    raw["normalizer.fields.action.scale"].fill_(2)
    averaged = {key: value.clone() for key, value in raw.items()}
    averaged["normalizer.fields.action.scale"].fill_(3)
    config: dict[str, Any] = dict(agent=OmegaConf.to_container(cfg.agent, resolve=True), normalization_dims=dims,
                  normalization=dict(joint_state="limits", point_cloud="identity", action="limits"),
                  eval=dict(use_ema=True, inference_steps=2), dataset={},
                  policy_metadata=dict(action_start_index=model.action_start_index),
                  env_runner=dict(task_name="synthetic"))
    seed_pool = list(range(10))
    checkpoint = Checkpoint(7, raw, averaged, {"momentum": torch.ones(1)},
                            {"step": torch.ones(1)}, {"torch": torch.ones(1)}, config)
    path = root / "normalizer.pt"
    save_checkpoint(path, checkpoint)
    obs = dict(joint_state=torch.randn(1, model.n_obs_steps, 19, device=device),
               point_cloud=torch.rand(1, model.n_obs_steps, 8, 3, device=device))
    for use_ema, state in ((False, raw), (True, averaged)):
        def episodes(runner, agent, *, seeds, inference_steps, action_steps, video_save_dir):
            assert runner.task_name == "synthetic" and video_save_dir is None
            assert seeds == split_evaluation_seeds(seed_pool, 1024, 2)[0]
            assert inference_steps == 2 and action_steps == model.n_action_steps
            assert not agent.training
            assert all(p.dtype == torch.float32 and p.device == obs["joint_state"].device for p in agent.parameters())
            assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(1024).get_state())
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
            expected = restored.predict_action(obs, inference_steps=inference_steps,
                generator=torch.Generator(device=device).manual_seed(17))["future_actions"]
            actual = agent.predict_action(obs, inference_steps=inference_steps,
                generator=torch.Generator(device=device).manual_seed(17))["future_actions"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for seed in seeds:
                yield dict(success=True, steps=2, total_steps=3, seed=seed)

        with patch.object(rollout, "load_seed_pool", return_value=seed_pool) as load_seeds, \
             patch.object(SimRunner, "run", autospec=True, side_effect=episodes):
            restored = restore_policy(path, use_ema=use_ema, device=device)
            record, result_path = evaluate_policy(path, weights=None if use_ema else "raw",
                episodes=2, device=device, video=False, output_dir=root / f"evaluation-{use_ema}")
            load_seeds.assert_called_once_with("synthetic")
            assert record["weights"] == ("ema" if use_ema else "raw") and record["global_step"] == 7
            assert json.loads(result_path.read_text())["seeds"] == record["seeds"]
            assert record["runtime"]["parameter_dtypes"] == ["torch.float32"]
        assert "point_cloud" not in restored.normalizer.fields
        assert restored.normalization_dims["point_cloud"] == 3
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value.to(device), rtol=0, atol=0)
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
                            inference_metadata={"action_start_index": 0, "control_action_dim": 19},
                            effective_config={})
    effective = assemble_effective_config(cfg, description, model)
    assert effective.deployment_metadata == description["deployment_metadata"]
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
    cfg = load_policy_config(method, overrides)
    original = DemonstrationDataset.iter_normalization_data

    def only_new_fields(dataset, key):
        assert key not in fields, f"Fitted inherited field {key}"
        yield from original(dataset, key)

    with patch.object(DemonstrationDataset, "iter_normalization_data", only_new_fields):
        assert run_training(cfg, max_updates=2) == 2
    checkpoint_path = root / method / "checkpoints/latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    assert {key: checkpoint.effective_config["normalization"][key] for key in fields} == modes
    for key, value in state.items():
        torch.testing.assert_close(checkpoint.model_state[f"normalizer.{key}"], value, rtol=0, atol=0)
    artifact.unlink()
    info = inspect_policy(root / method, checkpoint="latest", weights="ema" if cfg.training.use_ema else "raw")
    runtime = load_policy(info, device=device)
    restored = runtime.agent
    observation = {key: source[f"data/{key}"][:2] for key in info.observation_fields}
    assert runtime.predict(observation).shape == (3, 19)
    with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")):
        trainer = prepare_resume(root / method, device=str(device))
        assert trainer is not None
        assert run_trainer(trainer, max_updates=1) == 3


def check_codebook_training(root, device=torch.device("cpu")):
    from dexmani_policy.agents.hand_prior.vqvae import HandVQVAE
    from dexmani_policy.training.pretrain.hand_prior.codebook import export_codebook

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
    vq = HandVQVAE(12, [1] * 12, latent_dim=8, hidden_dim=16, num_layers=1,
                   num_groups=1, codebook_size=3)
    vq_checkpoint = root / "vq.pt"
    torch.save(dict(model_config=vq.model_config, model_state_dict=vq.state_dict(),
                    normalizer_state_dict={"fields.hand.scale": torch.from_numpy(scale),
                                           "fields.hand.offset": torch.from_numpy(offset)}), vq_checkpoint)
    assert export_codebook(vq_checkpoint, artifact) == 3
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
    with patch("dexmani_policy.agents.hand_prior.codebook.load_codebook", side_effect=AssertionError("Reload on resume")), \
         patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")):
        trainer = prepare_resume(root / "dqrise", device=str(device))
        assert trainer is not None
        assert trainer.cfg.dataset.max_train_episodes == 100
        assert run_trainer(trainer, max_updates=1) == 3


def check_configs(_device):
    from dexmani_policy.training.common.config import LAUNCH_ID, CONFIG_ROOT, register_resolvers

    for method in ("afro", "cordvip", "vqvae"):
        config = f"pretrain/hand_prior/{method}" if method == "vqvae" else f"pretrain/obs/{method}"
        cfg = load_policy_config(config, ["task_name=collection", "zarr_path=/tmp/explicit.zarr"])
        expected = f"experiments/pretrain/{method}/collection/{LAUNCH_ID}_seed42"
        assert cfg.workspace.output_dir == expected
        OmegaConf.resolve(cfg)
        assert cfg.workspace.output_dir == expected
        assert load_policy_config(config, ["task_name=collection"]).workspace.output_dir == expected
        assert load_policy_config(config, ["workspace.output_dir=/tmp/custom"]).workspace.output_dir == "/tmp/custom"
    register_resolvers()
    with hydra.initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        cfg = hydra.compose(config_name="dp3", return_hydra_config=True)
    assert cfg.hydra.run.dir == f"experiments/train/dp3/pick_apple_messy/{LAUNCH_ID}_seed42"
    assert cfg.hydra.sweep.dir == cfg.hydra.run.dir
    assert OmegaConf.to_container(cfg.hydra.sweep)["subdir"] == "${hydra.job.num}"
    methods = ("dp", "dp3", "r3d", "afro", "cordvip", "maniflow", "sat", "dqrise",
               "dexlatent", "adaptac", "interact_dex")
    cases: list[tuple[str, list[str]]] = [(method, []) for method in methods]
    cases += [("r3d", [f"normalization.point_cloud={mode}"])
              for mode in ("identity", "gaussian", "rms", "max_abs")]
    cases += [("adaptac", [f"tactile={kind}"]) for kind in ("force", "taxel", "patch")]
    for method, overrides in cases:
        cfg = load_policy_config(method, overrides)
        validate_config(cfg)
        validate_normalization(cfg)
        assert cfg.dataset.max_train_episodes is None
        if method == "adaptac":
            assert (cfg.n_obs_steps, cfg.horizon, cfg.n_action_steps) == (2, 16, 8)
            assert (cfg.agent.n_obs_steps, cfg.agent.horizon, cfg.agent.n_action_steps, cfg.agent.action_dim) == (2, 16, 8, 19)
            assert (cfg.dataset.obs_horizon, cfg.dataset.horizon, cfg.dataset.pad_before, cfg.dataset.pad_after) == (2, 16, 1, 7)
            assert (cfg.agent.num_inference_steps, cfg.agent.force_num_inference_steps) == (20, 20)
            assert (cfg.dataloader.batch_size, cfg.training.loop.gradient_accumulation_steps,
                    cfg.training.loop.total_train_steps) == (128, 1, 100000)
        if method in ("adaptac", "interact_dex"):
            encoder = hydra.utils.instantiate(cfg.agent.tactile_encoder)
            assert cfg.normalization[encoder.input_key] == "limits"
            assert encoder.input_key in cfg.dataset.sensor_modalities
            assert (encoder.num_tokens, encoder.out_dim) == (5, 64)
            if method == "interact_dex":
                assert cfg.agent.future_force_horizons == cfg.dataset.future_force_horizons == [1, 2, 4]
                no_force = load_policy_config(method, [*overrides, "agent.future_force_loss_weight=0"])
                assert "future_force_horizons" not in no_force.dataset
                assert no_force.agent.future_force_horizons == [1, 2, 4]
        disabled = load_policy_config(method, [*overrides, "training.use_ema=false"])
        assert not disabled.eval.use_ema
    validate_normalization(load_policy_config("dp3", ["normalization.action=gaussian",
                                                       "agent.clip_sample=false"]))
    for mode in ("limits", "gaussian", "rms", "max_abs"):
        cfg = load_policy_config("adaptac", [f"normalization.point_cloud={mode}"])
        try:
            validate_normalization(cfg)
        except ValueError as error:
            assert all(part in str(error) for part in ("point_cloud", "identity", "physical geometry"))
        else:
            raise AssertionError(f"AdapTac accepted non-metric point_cloud normalization: {mode}")


def small_training_config(source, output, device, *, bfloat16=False):
    return load_policy_config("dp3", [
        f"zarr_path={source}", f"workspace.output_dir={output}", "workspace.wandb_cfg=null",
        f"training.device={device}", "training.use_compile=false",
        f"training.use_bfloat16={str(bfloat16).lower()}", "training.lr_warmup_steps=0",
        "training.loop.total_train_steps=8", "training.loop.log_interval_steps=100",
        "training.loop.gradient_accumulation_steps=2", "horizon=4", "n_obs_steps=2", "n_action_steps=2",
        "agent.down_dims=[16,32]", "agent.diffusion_step_embed_dim=16", "agent.num_points=8",
        "agent.num_inference_steps=2", "dataset.augmentation_cfg=null", "dataloader.batch_size=2",
        "dataloader.num_workers=0", "dataloader.pin_memory=false",
    ])


def check_training_resume(device):
    from dexmani_policy.training.common.data import epoch_position

    with tempfile.TemporaryDirectory(prefix="dexmani-resume-check-") as temporary:
        root = Path(temporary)
        source = make_source(root / "source.zarr", [4, 9, 14])
        for bfloat16 in (False, True):
            used_lrs = []

            def record_lr(optimizer, args, kwargs):
                used_lrs.append(optimizer.param_groups[0]["lr"])

            source.attrs.update(dt=0.02, pointcloud_config={"num_points": 8})
            full, split = root / f"full-{bfloat16}", root / f"split-{bfloat16}"
            cfg = small_training_config(root / "source.zarr", full, device, bfloat16=bfloat16)
            cfg.training.lr_warmup_steps = 2
            cfg.training.loop.log_interval_steps = 1
            with open_dict(cfg):
                cfg.max_updates = 6
            trainer = build_trainer(cfg)
            trainer.optimizer.register_step_pre_hook(record_lr)
            assert run_trainer(trainer, max_updates=6) == 6
            cfg = small_training_config(root / "source.zarr", split, device, bfloat16=bfloat16)
            cfg.training.lr_warmup_steps = 2
            cfg.training.loop.log_interval_steps = 1
            trainer = build_trainer(cfg)
            trainer.optimizer.register_step_pre_hook(record_lr)
            optimizer_step = trainer.optimizer_step

            def interrupt_at_milestone():
                optimizer_step()
                if trainer.global_step + 1 == 4:
                    trainer.signal_handler(signal.SIGINT, None)

            with patch.object(trainer, "optimizer_step", side_effect=interrupt_at_milestone), \
                 patch("dexmani_policy.training.trainer.save_checkpoint", wraps=save_checkpoint) as save:
                assert run_trainer(trainer, max_updates=4) == 4
            assert trainer.stop_requested
            assert save.call_count == 2
            try:
                build_trainer(cfg)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Fresh policy overwrote an existing run")
            assert sorted(p.name for p in (split / "checkpoints").glob("step_*.pt")) == [
                "step_00000002.pt", "step_00000004.pt"]
            original = load_checkpoint(split / "checkpoints/latest.pt")
            original_split = split
            split = root / f"moved-{bfloat16}"
            shutil.copytree(original_split, split, symlinks=True)
            old_metrics = (original_split / "metrics.jsonl").read_bytes()
            latest = split / "checkpoints/latest.pt"
            assert latest.is_symlink() and not Path(os.readlink(latest)).is_absolute()
            assert latest.resolve().name == "step_00000004.pt"
            with (split / "metrics.jsonl").open("ab") as stream:
                stream.write(b'{"step":')
            source.attrs.update(dt=0.1, pointcloud_config={"num_points": 123})
            with patch("dexmani_policy.training.trainer.fit_normalizer", side_effect=AssertionError("Refit on resume")):
                trainer = prepare_resume(split, device=str(device))
                assert trainer is not None
            trainer.optimizer.register_step_pre_hook(record_lr)
            assert trainer.cfg.deployment_metadata == original.effective_config["deployment_metadata"]
            assert all(p.dtype == torch.float32 for p in trainer.model.parameters())
            assert all(p.dtype == torch.float32 for p in trainer.ema_model.parameters())
            assert trainer.ema_updater.optimization_step == 4
            position = epoch_position(6, trainer.batches_per_epoch, trainer.gradient_accumulation_steps)
            assert position[0] > 0, "Fixture must cross an epoch"
            assert trainer.batches_per_epoch % trainer.gradient_accumulation_steps, "Fixture must include a tail group"
            # Every train() call must discard gradients left by earlier work.
            parameter = trainer.trainable_parameters[0]
            parameter.grad = torch.ones_like(parameter)
            assert run_trainer(trainer, max_updates=2) == 6
            assert Path(trainer.cfg.workspace.output_dir) == split
            assert (original_split / "metrics.jsonl").read_bytes() == old_metrics
            assert not (original_split / "checkpoints/step_00000006.pt").exists()
            continuous = load_checkpoint(full / "checkpoints/latest.pt")
            resumed = load_checkpoint(split / "checkpoints/latest.pt")
            for key in ("model_state", "ema_model_state", "scheduler_state"):
                torch.testing.assert_close(getattr(continuous, key), getattr(resumed, key), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(continuous.optimizer_state["state"], resumed.optimizer_state["state"], rtol=1e-6, atol=1e-7)
            assert continuous.optimizer_state["param_groups"] == resumed.optimizer_state["param_groups"]
            assert continuous.global_step == resumed.global_step == 6
            assert "max_updates" not in continuous.effective_config
            assert resumed.effective_config["deployment_metadata"] == original.effective_config["deployment_metadata"]
            for key in ("torch", "torch_cuda"):
                torch.testing.assert_close(continuous.rng_state[key], resumed.rng_state[key], rtol=0, atol=0)
            assert used_lrs[:6] == used_lrs[6:] and len(set(used_lrs[:6])) > 1
            for directory, lrs in ((full, used_lrs[:6]), (split, used_lrs[6:])):
                records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
                assert [row["step"] for row in records] == list(range(1, 7))
                assert [row["train/lr"] for row in records] == lrs
                assert [row["train/epoch"] for row in records] == [
                    epoch_position(step - 1, trainer.batches_per_epoch,
                                   trainer.gradient_accumulation_steps)[0]
                    for step in range(1, 7)
                ]
                assert all(np.isfinite(row["train/loss"]) for row in records)

        # Finish the actual saved budget, then resume without the original data.
        trainer = prepare_resume(split, device=str(device))
        assert trainer is not None
        assert run_trainer(trainer) == 8
        for weights in ("raw", "ema"):
            info = inspect_policy(split, weights=weights)
            runtime = load_policy(info, device=device)
            observation = {key: source[f"data/{key}"][:2] for key in info.observation_fields}
            prediction = runtime.predict(observation)
            assert prediction.shape == (3, 19) and np.isfinite(prediction).all()
            runtime.close()
        (root / "source.zarr").rename(root / "unavailable.zarr")
        artifacts = {path.relative_to(split): (path.stat().st_mtime_ns, path.read_bytes())
                     for path in split.rglob("*") if path.is_file()}
        from dexmani_policy.training.common.resume import main as resume_main
        with patch("dexmani_policy.train.build_trainer", side_effect=AssertionError("Built completed trainer")):
            assert prepare_resume(split, zarr_path=str(root / "missing.zarr"), use_compile=True, wandb=True) is None
            assert resume_main(["--run", str(split), "--zarr-path", str(root / "missing.zarr"),
                                "--compile", "--wandb"]) is None
        assert artifacts == {path.relative_to(split): (path.stat().st_mtime_ns, path.read_bytes())
                             for path in split.rglob("*") if path.is_file()}


def check_nonfinite_group(device):
    with tempfile.TemporaryDirectory(prefix="dexmani-loss-check-") as temporary:
        root = Path(temporary)
        make_source(root / "source.zarr", [6, 12])
        cfg = small_training_config(root / "source.zarr", root / "run", device)
        cfg.training.max_grad_norm = 0
        trainer = build_trainer(cfg)
        before = copy.deepcopy(trainer.model.state_dict())
        averaged = copy.deepcopy(trainer.ema_model.state_dict())
        scheduler = copy.deepcopy(trainer.scheduler.state_dict())
        forward = trainer.model.forward
        calls = 0

        def early_bad_loss(*args, **kwargs):
            nonlocal calls
            loss, metrics = forward(*args, **kwargs)
            calls += 1
            return (loss * float("nan") if calls == 1 else loss), metrics

        with patch.object(trainer.model, "forward", side_effect=early_bad_loss):
            try:
                run_trainer(trainer, max_updates=1)
            except RuntimeError as error:
                assert "Non-finite loss" in str(error)
            else:
                raise AssertionError("Early non-finite micro-batch entered optimizer")
        assert calls == 2 and trainer.global_step == trainer.ema_updater.optimization_step == 0
        assert not trainer.optimizer.state
        torch.testing.assert_close(before, trainer.model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(averaged, trainer.ema_model.state_dict(), rtol=0, atol=0)
        assert scheduler == trainer.scheduler.state_dict()
        assert not list((root / "run").rglob("*.pt"))


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


def check_interact_representation(device):
    cfg = load_policy_config("interact_dex", [
        "agent.pc_dim=3", "agent.num_points=32", "agent.pc_encoder_config.channels=24",
        "agent.pc_encoder_config.num_patches=12", "agent.pc_encoder_config.group_size=8",
        "agent.pc_encoder_config.depth=1", "agent.pc_encoder_config.num_heads=4",
        "agent.n_layers=1", "agent.hidden_dim=32", "agent.n_head=4",
        "agent.fps_random_config.use_random=false", "agent.num_inference_steps=2",
    ])
    model = hydra.utils.instantiate(cfg.agent, _convert_="all").to(device).eval()
    dims = dict(action=19, joint_state=19, point_cloud=3, eef_pose=9, fingertip_xyz=3, finger_tactile_force=3)
    model.configure_normalizer(dims, dict(action=(19,), joint_state=(19,), finger_tactile_force=(5, 3)))
    model.to(device)
    model.normalizer["action"].scale.fill_(2)
    model.normalizer["action"].offset.fill_(7)
    model.normalizer["joint_state"].offset.fill_(-9)
    obs = dict(joint_state=torch.randn(2, 4, 19, device=device),
               point_cloud=torch.randn(2, 4, 32, 3, device=device) * .03,
               eef_pose=torch.zeros(2, 4, 9, device=device),
               fingertip_xyz=torch.randn(2, 4, 5, 3, device=device) * .02,
               finger_tactile_force=torch.randn(2, 4, 5, 3, device=device))
    obs["eef_pose"][..., 3] = 1
    obs["eef_pose"][..., 7] = 1
    batch = dict(obs=obs, action=torch.randn(2, 16, 19, device=device),
                 future_force=torch.randn(2, 3, 5, 3, device=device),
                 future_force_valid=torch.tensor([[True, True, False], [True, False, False]], device=device))
    condition = model._encode_obs(obs)
    assert condition.shape == (2, 24, 192)
    assert sum(p.numel() for p in model.future_force_head.parameters()) == 35331
    assert torch.count_nonzero(model._predict_future_force(condition, batch)) == 0
    with torch.no_grad():
        model.future_force_head[-1].weight.normal_(std=.1)
    seen = []
    hook = model.future_force_head.register_forward_pre_hook(lambda m, a: seen.append(a[0]))
    prediction = model._predict_future_force(condition, batch)
    hook.remove()
    assert prediction.shape == (2, 3, 5, 3)
    expected = (batch["action"][:, 3:7] - obs["joint_state"][:, -1, None]) * 2
    torch.testing.assert_close(seen[0][:, 2, 0, 192:268], expected.flatten(1))
    torch.testing.assert_close(seen[0][0, :, 0, -4:], torch.tensor([[1,0,0,0],[1,1,0,0],[1,1,1,1]], device=device).float())
    for index, h in enumerate((1, 2, 4)):
        changed = {**batch, "action": batch["action"].clone()}
        changed["action"][:, 3+h:] += 10
        torch.testing.assert_close(model._predict_future_force(condition, changed)[:, index], prediction[:, index])
        changed["action"][:, 3] += 1
        assert not torch.allclose(model._predict_future_force(condition, changed)[:, index], prediction[:, index])
    changed_labels = {**batch, "future_force": torch.full_like(batch["future_force"], float("nan")),
                      "future_force_valid": torch.zeros_like(batch["future_force_valid"])}
    torch.testing.assert_close(model._predict_future_force(condition, changed_labels), prediction)
    zero, _ = model._compute_future_force_loss(prediction, changed_labels)
    zero_grad, = torch.autograd.grad(zero, prediction, retain_graph=True)
    assert zero == 0 and torch.count_nonzero(zero_grad) == 0
    loss, _ = model._compute_future_force_loss(prediction, batch)
    loss.backward()
    assert model.obs_encoder.hand_history.temporal_ffn[-1].weight.grad.abs().sum() > 0
    with patch.object(model, "_encode_obs", wraps=model._encode_obs) as encode:
        model.compute_loss(batch)
        assert encode.call_count == 1
    with patch.object(model.future_force_head, "forward", side_effect=AssertionError("online future head")):
        assert model.predict_action(obs)["future_actions"].shape == (2, 13, 19)
    # Unequal horizon counts must not weight the densely supervised horizon more heavily.
    probe = torch.zeros(2, 3, 5, 3, device=device, requires_grad=True)
    with torch.no_grad():
        probe[:, 1] = 2
    target_batch = {**batch, "future_force": obs["finger_tactile_force"][:, -1:].expand(-1, 3, -1, -1)}
    value, _ = model._compute_future_force_loss(probe, target_batch)
    torch.testing.assert_close(value, value.new_tensor(.75))
    optimizer = build_optimizer(model, **cfg.optimizer)
    names = [name for group in optimizer.param_groups for name in group["param_names"]]
    assert len(names) == len(set(names))
    for group in optimizer.param_groups:
        if any(name.endswith("null_logits") for name in group["param_names"]):
            assert group["weight_decay"] == 0 and group["lr"] == cfg.optimizer.obs_lr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(4)
    for check in (check_interact_representation, check_evaluation, check_sim_construction, check_flow_recipes, check_policy_roundtrips, check_configs, check_training_resume, check_nonfinite_group, check_training_lifecycle,):
        check(args.device)
        print(f"PASS {check.__name__} ({args.device})", flush=True)


if __name__ == "__main__":
    main()
