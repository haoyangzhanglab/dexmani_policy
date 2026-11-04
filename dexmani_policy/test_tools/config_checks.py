"""CPU configuration regressions: python -m pytest dexmani_policy/test_tools/config_checks.py."""

import copy
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint
from dexmani_policy.evaluation import selection
from dexmani_policy.evaluation.seeds import split_evaluation_seeds
from dexmani_policy.training.common.config import load_policy_config


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    checkpoints = [tmp_path / f"candidate-{index}.pt" for index in range(3)]
    for path in checkpoints:
        path.touch()
    configs = [dict(
        env_runner=dict(task_name="test", env_kwargs={}),
        eval=dict(use_ema=True, inference_steps=10),
        agent=dict(n_action_steps=4, n_obs_steps=2),
    ) for _ in checkpoints]
    references = []

    def load(path):
        # Preflight must release each checkpoint before reading the next one.
        assert all(reference() is None for reference in references)
        checkpoint = Checkpoint(1, {}, {}, {}, {}, {}, configs[checkpoints.index(path)])
        references.append(weakref.ref(checkpoint))
        return checkpoint

    scores = [(0.8, 10, 1), (0.8, 10, 2), (0.8, 10, 3)]

    def evaluate(path, **options):
        assert load_checkpoint.call_count == len(checkpoints)
        assert all(reference() is None for reference in references)
        success_rate, avg_steps, step = scores[checkpoints.index(Path(path).resolve())]
        return dict(statistics=dict(success_rate=success_rate, avg_steps=avg_steps),
                    global_step=step), options["output_dir"] / "result_details.json"

    load_checkpoint = Mock(side_effect=load)
    seed_pool = Mock(return_value=list(range(20)))
    rollout = Mock(side_effect=evaluate)
    monkeypatch.setattr(selection, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(selection, "load_seed_pool", seed_pool)
    monkeypatch.setattr(selection, "evaluate_policy", rollout)
    return SimpleNamespace(checkpoints=checkpoints, configs=configs, scores=scores,
                           load=load_checkpoint, seeds=seed_pool, rollout=rollout,
                           output=tmp_path / "output")


@pytest.mark.parametrize("before,after,field,values", [
    ({"task_name": "test"}, {"task_name": "other"}, "task_name", ("'test'", "'other'")),
    ({"env_rebuild_interval": 25}, {"env_rebuild_interval": 1}, "env_rebuild_interval", ("25", "1")),
    ({}, {"env_rebuild_interval": None}, "env_rebuild_interval", ("25", "None")),
    *[({"env_kwargs": {key: left}}, {"env_kwargs": {key: right}}, f"env_kwargs.{key}",
       (repr(left), repr(right))) for key, left, right in (
        ("randomize_model_id", True, False),
        ("randomize_table_height", True, False),
        ("control_mode", "joint", "ee"),
        ("frame_skip", 6, 12),
        ("table_height_range", [-0.01, 0.02], [-0.02, 0.02]),
        ("custom_options", {"enabled": True}, {"enabled": False}),
        ("custom_options", None, {}),
    )],
    *[({}, {"env_kwargs": {key: value}}, f"env_kwargs.{key}", ("<unspecified>", repr(value)))
      for key, value in (("randomize_model_id", True), ("control_mode", "joint"),
                         ("max_stability_retries", 0), ("custom_options", None))],
    ({}, {"env_kwargs": {"render_mode": None}}, "env_kwargs.render_mode", ("'rgb_array'", "None")),
    ({"env_kwargs": {"control_mode": "joint"}}, {"env_kwargs": {}},
     "env_kwargs.control_mode", ("'joint'", "<unspecified>")),
])
def test_selection_rejects_environment_mismatch(pipeline, monkeypatch, before, after, field, values):
    for cfg in pipeline.configs:
        cfg["env_runner"].update(copy.deepcopy(before))
    pipeline.configs[-1]["env_runner"].update(after)
    original = copy.deepcopy(pipeline.configs)
    create_directory = Mock(wraps=selection.create_run_directory)
    monkeypatch.setattr(selection, "create_run_directory", create_directory)

    with pytest.raises(ValueError) as error:
        selection.run_pipeline(pipeline.checkpoints, output_dir=pipeline.output, device="cpu")

    message = str(error.value)
    for part in (str(pipeline.checkpoints[0]), str(pipeline.checkpoints[-1]),
                 f"env_runner.{field}", *values):
        assert part in message
    pipeline.seeds.assert_not_called()
    create_directory.assert_not_called()
    pipeline.rollout.assert_not_called()
    assert not pipeline.output.exists()
    assert pipeline.configs == original


@pytest.mark.parametrize("runners", [
    [{}, {}, {}],
    [{}, {"env_kwargs": None}, {"env_kwargs": {}}],
    [{"env_kwargs": {}}, {"env_kwargs": {"render_mode": "rgb_array"}}, {"env_rebuild_interval": 25}],
    [{"env_kwargs": {"control_mode": "joint", "custom": {"a": 1, "b": 2}}},
     {"env_kwargs": {"custom": {"b": 2, "a": 1}, "control_mode": "joint"}},
     {"env_kwargs": {"render_mode": "rgb_array", "control_mode": "joint", "custom": {"b": 2, "a": 1}}}],
])
@pytest.mark.parametrize("override", [False, True])
def test_selection_equivalent_environments_and_method_differences(pipeline, runners, override):
    for index, (cfg, runner) in enumerate(zip(pipeline.configs, runners)):
        cfg["env_runner"] = {"task_name": "test", **runner}
        cfg["policy_name"] = f"method-{index}"
        cfg["agent"].update(_target_=f"method.{index}", n_obs_steps=index + 1,
                            n_action_steps=index + 4)
        cfg["training"] = dict(seed=index, learning_rate=0.001 * (index + 1))
        cfg["dataset"] = dict(sensor_modalities=[f"input-{index}"])
        cfg["eval"] = dict(use_ema=index == 0, inference_steps=10 + index)
        cfg["env_runner"].update(n_obs_steps=index + 1, sensor_modalities=[f"input-{index}"],
                                 rgb_preprocessing={"crop_shape": [index + 1, index + 1]})
    original = copy.deepcopy(pipeline.configs)
    options = dict(weights="raw", inference_steps=7, action_steps=2) if override else {}
    result_path = selection.run_pipeline(
        pipeline.checkpoints, output_dir=pipeline.output, selection_episodes=2, test_episodes=3,
        seed=12, device="cpu", video=True, **options,
    )
    record = json.loads(result_path.read_text())
    test, selected = split_evaluation_seeds(list(range(20)), 12, 3, 2)
    assert record["selection_seeds"] == selected
    assert record["test_seeds"] == test
    assert set(test).isdisjoint(selected)
    pipeline.seeds.assert_called_once_with("test")
    assert [call.args[0] for call in pipeline.load.call_args_list] == pipeline.checkpoints
    assert len(pipeline.rollout.call_args_list) == 4
    expected = options or dict(weights="ema", inference_steps=10, action_steps=4)
    for index, call in enumerate(pipeline.rollout.call_args_list):
        assert Path(call.args[0]).resolve() == pipeline.checkpoints[min(index, 2)]
        assert call.kwargs == dict(
            **expected, seeds=test if index == 3 else selected, seed=12, device="cpu",
            video=index == 3,
            output_dir=pipeline.output / ("test" if index == 3 else f"candidates/{index}"),
        )
    assert (result_path.parent / record["ckpt_relpath"]).resolve() == pipeline.checkpoints[2]
    assert record["global_step"] == 3
    assert len(record["candidates"]) == 3
    assert record["test_result"] == "test/result_details.json"
    assert record["inference"] == dict(use_ema=expected["weights"] == "ema",
                                      inference_steps=expected["inference_steps"],
                                      action_steps=expected["action_steps"])
    assert pipeline.configs == original


@pytest.mark.parametrize("scores,winner", [
    ([(0.9, 20, 1), (0.8, 5, 3), (0.7, 1, 9)], 0),
    ([(0.8, 10, 1), (0.8, 5, 2), (0.8, None, 9)], 1),
    ([(0.8, 10, 1), (0.8, 10, 3), (0.8, 10, 2)], 1),
    ([(0.8, 10, 1), (0.8, 10, 1), (0.8, 10, 1)], 0),
])
def test_selection_scoring_and_ties(pipeline, scores, winner):
    pipeline.scores[:] = scores
    path = selection.run_pipeline(pipeline.checkpoints, output_dir=pipeline.output,
                                  selection_episodes=2, test_episodes=3, device="cpu")
    record = json.loads(path.read_text())
    assert (path.parent / record["ckpt_relpath"]).resolve() == pipeline.checkpoints[winner]
    assert Path(pipeline.rollout.call_args.args[0]).resolve() == pipeline.checkpoints[winner]


@pytest.mark.parametrize("method", ["afro", "cordvip"])
@pytest.mark.parametrize("disabled", [(), ("randomize_model_id",), ("randomize_table_height",),
                                      ("randomize_model_id", "randomize_table_height")])
def test_randomization_inheritance(method, disabled):
    cfg = load_policy_config(method, [f"env_runner.env_kwargs.{key}=false" for key in disabled])
    assert dict(cfg.env_runner.env_kwargs) == dict(
        randomize_model_id="randomize_model_id" not in disabled,
        randomize_table_height="randomize_table_height" not in disabled,
        control_mode="joint",
    )


@pytest.mark.parametrize("overrides", [[], ["tactile=force"], ["agent.future_force_loss_weight=0"],
                                       ["future_force_horizons=[1,3]"]])
def test_interact_force_config(overrides):
    cfg = load_policy_config("interact_dex", overrides)
    assert cfg.agent.tactile_encoder._target_.endswith(".XHandForceEncoder")
    assert list(cfg.dataset.sensor_modalities) == [
        "joint_state", "point_cloud", "eef_pose", "fingertip_xyz", "finger_tactile_force",
    ]
    assert cfg.normalization.eef_pose == cfg.normalization.fingertip_xyz == "identity"
    assert cfg.agent.future_force_horizons == cfg.future_force_horizons
    if cfg.agent.future_force_loss_weight == 0:
        assert cfg.dataset._target_.endswith(".DemonstrationDataset")
        assert "future_force_horizons" not in cfg.dataset
    else:
        assert cfg.dataset._target_.endswith(".InterActDexDataset")
        assert cfg.dataset.future_force_horizons == cfg.future_force_horizons


@pytest.mark.parametrize("overrides", [
    ["tactile=taxel"], ["tactile=patch"],
    ["agent.tactile_encoder._target_=dexmani_policy.agents.obs_encoder.xhand_tactile.XHandTaxelEncoder"],
    ["agent.tactile_encoder._target_=dexmani_policy.agents.obs_encoder.xhand_tactile.XHandPatchEncoder"],
])
def test_interact_dense_rejected_during_assembly(overrides):
    with pytest.raises(ValueError, match="InterAct-Dex only supports aggregate force; use tactile=force") as error:
        load_policy_config("interact_dex", overrides)
    assert "AdapTac supports tactile=force, tactile=taxel and tactile=patch" in str(error.value)


@pytest.mark.parametrize("preset,encoder", [
    ("force", "XHandForceEncoder"), ("taxel", "XHandTaxelEncoder"), ("patch", "XHandPatchEncoder"),
])
def test_adaptac_tactile_configs(preset, encoder):
    cfg = load_policy_config("adaptac", [f"tactile={preset}"])
    assert cfg.agent.tactile_encoder._target_.endswith(f".{encoder}")
    from dexmani_policy.training.common.config import validate_config
    key = "finger_tactile_force" if preset == "force" else "taxel_tactile_force"
    assert set(cfg.dataset.sensor_modalities) == {"joint_state", "point_cloud", key}
    assert set(cfg.normalization) == {"joint_state", "point_cloud", "action", key}
    validate_config(cfg)


@pytest.mark.parametrize("overrides", [
    ["dataset.sensor_modalities=[joint_state,point_cloud]", "~normalization.finger_tactile_force"],
    ["dataset.sensor_modalities=[joint_state,point_cloud,finger_tactile_force,taxel_tactile_force]",
     "+normalization.taxel_tactile_force=limits"],
    ["agent.tactile_encoder._target_=dexmani_policy.agents.obs_encoder.xhand_tactile.XHandTaxelEncoder"],
    ["+normalization.taxel_tactile_force=limits"],
    ["~normalization.finger_tactile_force"],
])
def test_adaptac_single_field_validation(overrides):
    from dexmani_policy.training.common.config import validate_config
    cfg = load_policy_config("adaptac", overrides)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("fields", [[], ["finger_tactile_force", "taxel_tactile_force"]])
def test_adaptac_dataset_rejects_ambiguous_fields_before_open(fields):
    from dexmani_policy.datasets.variants.tactile import AdapTacDataset
    with pytest.raises(ValueError, match="exactly one tactile field"):
        AdapTacDataset("missing.zarr", sensor_modalities=fields)


@pytest.mark.parametrize("overrides,expected", [
    ([], (3, 16, 8, 128, 2, 7)),
    (["n_obs_steps=2", "horizon=8", "n_action_steps=3", "dataloader.batch_size=4"],
     (2, 8, 3, 4, 1, 2)),
])
def test_cordvip_window_config(overrides, expected):
    cfg = load_policy_config("cordvip", overrides)
    n, horizon, actions, batch, before, after = expected
    assert (cfg.agent.n_obs_steps, cfg.agent.horizon, cfg.agent.n_action_steps) == (n, horizon, actions)
    assert (cfg.dataset.obs_horizon, cfg.dataset.horizon) == (n, horizon)
    assert (cfg.dataset.pad_before, cfg.dataset.pad_after) == (before, after)
    assert cfg.dataloader.batch_size == batch and cfg.agent.action_dim == 19


def test_interact_mixed_fields_rejected():
    from dexmani_policy.training.common.config import validate_config
    from dexmani_policy.datasets.variants.tactile import InterActDexDataset

    cfg = load_policy_config("interact_dex", [
        "dataset.sensor_modalities=[joint_state,point_cloud,eef_pose,fingertip_xyz,"
        "finger_tactile_force,taxel_tactile_force]",
        "+normalization.taxel_tactile_force=limits",
    ])
    with pytest.raises(ValueError, match="one tactile field"):
        validate_config(cfg)
    with pytest.raises(ValueError, match="only tactile field"):
        InterActDexDataset("missing.zarr", sensor_modalities=cfg.dataset.sensor_modalities,
                           future_force_horizons=[1])


@pytest.mark.parametrize("benchmark", [None, False, True])
@pytest.mark.parametrize("resume_benchmark", [None, False, True])
def test_cudnn_benchmark_fresh_and_resume(tmp_path, benchmark, resume_benchmark):
    import torch
    from omegaconf import open_dict
    from dexmani_policy.test_tools.dataset_checks import make_source
    from dexmani_policy.test_tools.training_checks import small_training_config
    from dexmani_policy.train import build_trainer, run_trainer
    from dexmani_policy.training.common.resume import prepare_resume

    source, output = tmp_path / "data.zarr", tmp_path / "run"
    make_source(source, [6, 12, 18])
    cfg = small_training_config(source, output, "cpu")
    assert cfg.training.cudnn_benchmark is True
    with open_dict(cfg.training):
        if benchmark is None:
            del cfg.training.cudnn_benchmark
        else:
            cfg.training.cudnn_benchmark = benchmark
    expected = True if benchmark is None else benchmark
    with torch.backends.cudnn.flags():
        torch.backends.cudnn.benchmark = not expected
        trainer = build_trainer(cfg)
        assert torch.backends.cudnn.benchmark == expected
        assert run_trainer(trainer, max_updates=1) == 1
        expected = expected if resume_benchmark is None else resume_benchmark
        torch.backends.cudnn.benchmark = not expected
        resumed = prepare_resume(output, device="cpu", cudnn_benchmark=resume_benchmark)
        assert torch.backends.cudnn.benchmark == expected
        assert resumed.cfg.training.get("cudnn_benchmark", True) == expected
        assert run_trainer(resumed, max_updates=1) == 2
        saved = load_checkpoint(output / "checkpoints/latest.pt")
        assert saved.effective_config["training"].get("cudnn_benchmark", True) == expected


@pytest.mark.parametrize("values", [[], [0], [-1], [True], [1.5], [2, 1], [1, 1]])
def test_interact_horizons_invalid(values):
    from dexmani_policy.common.config import validate_future_force_horizons
    with pytest.raises(ValueError):
        validate_future_force_horizons(values)


def test_interact_legacy_horizon_rejected():
    with pytest.raises(ValueError, match="obsolete"):
        load_policy_config("interact_dex", ["+interaction_horizon=4"])


def test_interact_horizon_exceeds_action_window():
    from dexmani_policy.agents.policies.interact_dex import InterActDex
    from dexmani_policy.agents.obs_encoder.xhand_tactile import XHandForceEncoder

    with pytest.raises(ValueError, match="available future actions"):
        InterActDex(16, 4, 4, 19, XHandForceEncoder(), future_force_horizons=[14])


@pytest.mark.parametrize("queries", [12, 6])
def test_interact_scene_config(queries):
    import hydra
    from dexmani_policy.training.common.config import assemble_effective_config
    cfg = load_policy_config("interact_dex", [
        f"agent.scene_num_queries={queries}", "workspace.output_dir=/tmp/scene-config-check",
        "agent.hidden_dim=32", "agent.n_layers=1", "agent.n_head=4",
    ])
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    # The model's actual quantity must override a stale input recipe.
    cfg.agent.scene_num_queries = 6 if queries == 12 else 12
    effective = assemble_effective_config(cfg, {"deployment_metadata": {}}, model)
    assert effective.agent.scene_num_queries == queries
    restored = hydra.utils.instantiate(effective.agent, _convert_="all")
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.obs_encoder.scene_tokenizer.num_queries == queries
    assert not {"context_tokens", "scene_tokenizer"} & set(effective.agent)


@pytest.mark.parametrize("mode,weighting,alpha", [
    ("encoder", "soft_flow", 1), ("action_prefix", "soft_flow", 2),
    ("unknown", "none", 1), ("action_prefix", "unknown", 1),
])
def test_interact_prefix_invalid(mode, weighting, alpha):
    import hydra
    cfg = load_policy_config("interact_dex", [
        f"agent.future_force_mode={mode}", f"agent.future_force_time_weighting={weighting}",
        f"agent.time_shift_alpha={alpha}", "agent.hidden_dim=32", "agent.n_layers=1", "agent.n_head=4",
    ])
    with pytest.raises(hydra.errors.InstantiationException, match="future_force|soft_flow"):
        hydra.utils.instantiate(cfg.agent, _convert_="all")


def test_interact_default_force_mode():
    import hydra
    cfg = load_policy_config("interact_dex", ["agent.hidden_dim=32", "agent.n_layers=1", "agent.n_head=4"])
    model = hydra.utils.instantiate(cfg.agent, _convert_="all")
    del cfg.agent.future_force_mode
    del cfg.agent.future_force_time_weighting
    defaults = hydra.utils.instantiate(cfg.agent, _convert_="all")
    defaults.load_state_dict(model.state_dict(), strict=True)
    assert defaults.future_force_mode == "encoder" and defaults.future_force_time_weighting == "none"
