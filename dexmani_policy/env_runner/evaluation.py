"""Saved evaluation inputs, seed plans and rollout orchestration."""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.datasets.common.rgb_preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.env_runner.common.metrics import (
    collect_episode_details,
    compute_eval_stats,
    mean_success_steps,
    validate_episode_details,
)
from dexmani_policy.env_runner.common.seeds import (
    SEED_PLAN_VERSION,
    plan_selection_seeds,
    plan_size,
    resolve_eval_seed,
    selection_seed_counts,
    validate_eval_seed,
    validate_seed_pool,
    validate_task_seeds,
)
from dexmani_policy.training.common.checkpoint import (
    resolve_best_checkpoint,
    resolve_checkpoint_path,
    resolve_selection_record,
    restore_policy_agent,
)
from dexmani_policy.utils.config import (
    positive_int,
    resolve_action_steps,
    validate_action_key_consistency,
)
from dexmani_policy.utils.runtime import atomic_path, set_seed


class EvalSeedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        previous = getattr(namespace, self.dest, None)
        if previous is not None and previous != values:
            parser.error("Conflicting --seed values")
        setattr(namespace, self.dest, values)


def parse_eval_overrides(overrides: list[str]):
    """Only evaluation/inference controls may override checkpoint evaluation."""
    section_params = {
        "select_best": {
            "initial_episodes",
            "batch_size",
            "max_episodes",
            "inference_steps",
            "action_steps",
            "use_ema",
        },
        "offline": {
            "episodes",
            "inference_steps",
            "inference_steps_list",
            "action_steps",
            "use_ema",
        },
        "demo": {
            "episodes",
            "inference_steps",
            "inference_steps_list",
            "action_steps",
            "use_ema",
            "viewer_resolution",
        },
        "video": {"enabled", "fps"},
    }
    eval_keys = {"eval.seed"}
    for section, names in section_params.items():
        eval_keys.update(f"eval.{name}" for name in names)
        eval_keys.update(f"eval.{section}.{name}" for name in names)
    seed = None
    for override in overrides:
        key, separator, _ = override.partition("=")
        if key == "eval.seed":
            value = OmegaConf.select(OmegaConf.from_dotlist([override]), key)
            validate_eval_seed(value)
            if seed is not None and seed != value:
                raise ValueError(f"Conflicting {key} overrides")
            seed = value
        if not separator or not (
            key in eval_keys or key.startswith("env_runner.") or key == "training.device"
        ):
            raise ValueError(f"Unsupported evaluation override: {key}")
    cfg = OmegaConf.from_dotlist(overrides)
    # Inspect mapping/list overrides too, including whole multitask task_configs.
    pending = [("env_runner", OmegaConf.to_container(cfg).get("env_runner", {}))]
    while pending:
        path, node = pending.pop()
        if isinstance(node, dict):
            entries = node.items()
        elif isinstance(node, list):
            entries = enumerate(node)
        else:
            continue
        for name, value in entries:
            key = f"{path}.{name}"
            if name in {
                "n_obs_steps",
                "sensor_modalities",
                "task_text",
                "rgb_preprocess_size",
                "rgb_random_crop_size",
                "rgb_keep_uint8",
                "rgb_preprocessing",
                "rgb_color_aug",
            }:
                raise ValueError(f"Model inputs come from saved dataset config: {key}")
            pending.append((key, value))
    shared = cfg.get("eval") or {}
    # None is unspecified; keep saved defaults intact when this override is merged.
    for source in (shared, *(shared.get(section) or {} for section in section_params)):
        for name in list(source):
            if source[name] is None:
                del source[name]
    # Explicit shared values must beat saved section defaults.
    for section, names in section_params.items():
        for name in names:
            if name in shared and name not in (shared.get(section) or {}):
                OmegaConf.update(cfg, f"eval.{section}.{name}", shared[name])
    return cfg


def validate_inference_steps(inference_steps_list) -> None:
    """Require distinct positive NFEs so each output directory has one result."""
    if not inference_steps_list:
        raise ValueError("inference_steps_list must be non-empty")
    for nfe in inference_steps_list:
        positive_int(nfe, "inference_steps")
    if len(set(inference_steps_list)) != len(inference_steps_list):
        raise ValueError("Inference steps must be distinct to preserve per-value results")


def add_evaluation_arguments(parser) -> None:
    parser.add_argument(
        "--policy-name", required=True, help="Experiment policy path, e.g. ddp/sat."
    )
    parser.add_argument(
        "--task-name", required=True, help="Task directory in the saved experiment."
    )
    parser.add_argument(
        "--exp-name", required=True, help="Run directory under experiments/policy/task."
    )
    parser.add_argument(
        "--seed",
        type=int,
        action=EvalSeedAction,
        help=(
            "Evaluation seed: CLI > eval.seed dotlist > saved eval.seed > 1024. "
            "Final eval with a selection record reuses its seed and rejects conflicts."
        ),
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        help="DDIM/Euler steps; defaults come from the saved config or selection record.",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        help="Physical actions consumed per prediction; defaults to selection or saved model window.",
    )
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument(
        "--ema", dest="use_ema", action="store_true", default=None, help="Use EMA weights."
    )
    weights.add_argument(
        "--no-ema", dest="use_ema", action="store_false", help="Use raw model weights."
    )


def build_eval_runner(cfg):
    """Build the evaluation environment with the saved model input recipe."""
    validate_action_key_consistency({"action_key": cfg.action_key, "env_runner": cfg.env_runner})
    runner_config = OmegaConf.to_container(cfg.env_runner, resolve=True)
    # Old experiment YAMLs duplicated Dataset RGB sizes in the runner config.
    for node in (runner_config, *runner_config.get("task_configs", ())):
        node.pop("rgb_preprocess_size", None)
        node.pop("rgb_random_crop_size", None)
    env_runner = hydra.utils.instantiate(runner_config)
    dataset = cfg.dataset
    tasks = dataset.get("task_names", ())
    children = dict(zip(tasks, dataset.get("datasets", ()), strict=True))
    texts = dataset.get("task_texts")
    texts = dict(zip(tasks, tasks if texts is None else texts, strict=True))
    fps = get_eval_param(cfg, "fps", "video")
    for runner in iter_leaf_env_runners(env_runner):
        inputs = children[runner.task_name] if children else dataset
        runner.n_obs_steps = cfg.agent.n_obs_steps
        runner.sensor_modalities = list(inputs["sensor_modalities"])
        runner.rgb_preprocessing = rgb_preprocessing_kwargs(inputs)
        if fps is not None:
            runner.env_video_fps = fps
        if children:
            runner.task_text = texts[runner.task_name]
    return env_runner


def iter_leaf_env_runners(env_runner):
    runners = getattr(env_runner, "runners", None)
    if isinstance(runners, dict):
        return tuple(runners.values())
    return (env_runner,)


def get_eval_param(
    cfg, param: str, section: str | None = None, *, override: Any = None, default: Any = None
) -> Any:
    """Resolve explicit value → section → shared eval → default; None is unspecified."""
    if override is not None:
        return override
    eval_cfg = cfg.get("eval") or {}
    section_cfg = (eval_cfg.get(section) or {}) if section else {}
    for source in (section_cfg, eval_cfg):
        val = source.get(param)
        if val is not None:
            return val
    return default


def task_seed_pools(runner):
    """Physical seeds in task execution order."""
    return {leaf.task_name: list(leaf.get_seed_list()) for leaf in iter_leaf_env_runners(runner)}


def run_eval_plan(runner, agent, plan, *, inference_steps, action_steps=None, video_save_dir=None):
    kwargs = dict(
        inference_steps=inference_steps,
        action_steps=action_steps,
        video_save_dir=video_save_dir,
    )
    if hasattr(runner, "runners"):
        return runner.run(agent, task_seeds=plan, **kwargs)
    if set(plan) != {runner.task_name}:
        raise ValueError("Task seed plan does not match the runner task")
    return runner.run(agent, seeds=plan[runner.task_name], **kwargs)


def artifact_reference(path, exp_dir):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(exp_dir).resolve()))
    except ValueError:
        return str(path)


def code_version(directory):
    import subprocess

    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()

        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}


def evaluation_snapshot(cfg, runner, **request):
    """Capture the actual environment, inputs and request once per run."""
    import importlib.util
    import importlib.metadata
    import sys

    exp_dir = Path(cfg.experiment_dir)
    sim = {"commit": "unknown", "dirty": "unknown", "version": "unknown"}
    try:
        spec = importlib.util.find_spec("dexmani_sim")
        if spec is not None and spec.origin:
            sim.update(code_version(Path(spec.origin).parent))
        sim["version"] = importlib.metadata.version("dexmani_sim")
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        pass
    inputs = {
        leaf.task_name: {
            "n_obs_steps": leaf.n_obs_steps,
            "sensor_modalities": list(leaf.sensor_modalities),
            "task_text": getattr(leaf, "task_text", None),
            "rgb_preprocessing": leaf.rgb_preprocessing,
            "record_video": leaf.record_video,
            "render_mode": getattr(leaf, "render_mode", None),
            "viewer_resolution": getattr(leaf, "viewer_resolution", None),
            "env_video_fps": getattr(leaf, "env_video_fps", None),
            "env_kwargs": (
                leaf.expand_env_kwargs(leaf.env_kwargs)
                if hasattr(leaf, "expand_env_kwargs")
                else getattr(leaf, "env_kwargs", {})
            ),
        }
        for leaf in iter_leaf_env_runners(runner)
    }
    snapshot = {
        "model_config_source": artifact_reference(exp_dir / "config.yaml", exp_dir),
        "effective_config": OmegaConf.to_container(cfg, resolve=True),
        "runner_inputs": inputs,
        "request": request,
        "argv": list(sys.argv),
        "policy_code": code_version(Path(__file__).resolve().parents[2]),
        "simulator_code": sim,
    }
    return OmegaConf.to_container(OmegaConf.create(snapshot), resolve=True)


def save_eval_snapshot(directory, cfg, runner, **request):
    snapshot = evaluation_snapshot(cfg, runner, **request)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with atomic_path(directory / "eval_config.yaml", overwrite=False) as temporary:
        temporary.write_text(OmegaConf.to_yaml(OmegaConf.create(snapshot), resolve=True))
    return artifact_reference(directory / "eval_config.yaml", cfg.experiment_dir)


def atomic_json(path, record, *, overwrite=True):
    import json

    with atomic_path(path, overwrite=overwrite) as temporary:
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)


def create_run_directory(root, directory=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if directory is None:
        return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {directory}")
    return directory


def fixed_test_request(record):
    """Resolve the saved protocol and test budget before building a runner."""
    evaluation = record.get("evaluation")
    request = evaluation.get("request") if isinstance(evaluation, dict) else None
    if not isinstance(request, dict) or request.get("seed_plan_version") != SEED_PLAN_VERSION:
        raise ValueError("Selection record requires fixed_test_v1; rerun checkpoint selection")
    if not {"shuffle_seed", "test_task_seeds"} <= request.keys():
        raise ValueError(
            "Selection record is missing its fixed test plan; rerun checkpoint selection"
        )
    validate_eval_seed(request["shuffle_seed"])
    saved_cfg = evaluation.get("effective_config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("Fixed test selection requires its saved evaluation config")
    episodes = positive_int(get_eval_param(saved_cfg, "episodes", "offline"), "saved test episodes")
    return request, episodes


def selection_test_seeds(record, pools, episodes):
    """Validate the frozen test/reserved plans and every candidate's stage evidence."""
    request = record["evaluation"]["request"]
    stages = record.get("stages")
    if (
        not isinstance(stages, list)
        or not stages
        or any(not isinstance(stage, dict) for stage in stages)
        or not any(stage.get("checkpoint") == record["ckpt_relpath"] for stage in stages)
    ):
        raise ValueError("Selected checkpoint has no evaluation evidence")
    required = {
        "initial_episodes",
        "batch_size",
        "max_episodes",
        "initial_task_seeds",
        "possible_tie_task_seeds",
    }
    if not required <= request.keys():
        raise ValueError("Reserved selection is missing its seed plan or budgets")
    test = request["test_task_seeds"]
    initial = request["initial_task_seeds"]
    extra = request["possible_tie_task_seeds"]
    if validate_task_seeds(test, pools) != episodes:
        raise ValueError("Fixed test seed counts disagree with saved eval.offline.episodes")
    initial_count, extra_count = selection_seed_counts(
        request["initial_episodes"], request["batch_size"], request["max_episodes"]
    )
    if (
        validate_task_seeds(initial, pools) != initial_count
        or validate_task_seeds(extra, pools, allow_empty=True) != extra_count
    ):
        raise ValueError("Reserved selection seed counts disagree with its budgets")
    reserved = {}
    for task in initial:
        if set(initial[task]) & set(extra[task]):
            raise ValueError(f"{task}: initial and possible tie seeds overlap")
        reserved[task] = initial[task] + extra[task]
        pool = set(validate_seed_pool(task, pools[task]))
        if not set(reserved[task]) <= pool:
            raise ValueError(f"{task}: reserved selection seeds are outside the task seed pool")
        if not set(test[task]) <= pool:
            raise ValueError(f"{task}: fixed test seeds are outside the task seed pool")
        if set(test[task]) & set(reserved[task]):
            raise ValueError(f"{task}: fixed test and reserved selection seeds overlap")
    for stage in stages:
        if stage.get("status") != "completed":
            raise ValueError("Selection contains an incomplete stage")
        plan = stage.get("requested_task_seeds")
        validate_task_seeds(plan, pools)
        validate_episode_details(stage.get("episode_details"), plan)
        for task, seeds in plan.items():
            if not set(seeds) <= set(reserved[task]):
                raise ValueError(f"{task}: selection stage seeds are outside the reserved plan")
    return test


def resolve_eval_request(
    cfg,
    exp_dir,
    checkpoint="best",
    overrides=(),
    *,
    section="offline",
    use_ema=None,
    inference_steps_list=None,
    action_steps=None,
    selection_record=None,
    eval_seed=None,
    episodes=None,
):
    """CLI > explicit dotlist > selected checkpoint > saved defaults."""
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    override_cfg = parse_eval_overrides(list(overrides))
    merged = OmegaConf.merge(cfg, override_cfg)
    merged.experiment_dir = str(exp_dir)
    resolved = (
        resolve_selection_record(exp_dir, selection_record)
        if selection_record is not None
        else resolve_best_checkpoint(exp_dir) if checkpoint == "best" else None
    )
    requested_episodes = get_eval_param(override_cfg, "episodes", section, override=episodes)
    if resolved is not None and section == "offline":
        request, count = fixed_test_request(resolved[0])
        explicit_seed = eval_seed is not None or "seed" in (override_cfg.get("eval") or {})
        if explicit_seed and resolve_eval_seed(merged, eval_seed) != request["shuffle_seed"]:
            raise ValueError(
                "Explicit evaluation seed conflicts with selection record shuffle_seed"
            )
        seed = request["shuffle_seed"]
        if requested_episodes is not None and requested_episodes != count:
            raise ValueError("Explicit episodes conflicts with selection record test count")
    else:
        seed = resolve_eval_seed(merged, eval_seed)
        if section == "offline":
            count = get_eval_param(
                merged, "episodes", section, override=requested_episodes, default=100
            )
    if section == "offline":
        count = positive_int(
            count if requested_episodes is None else requested_episodes, "episodes"
        )
        OmegaConf.update(merged, "eval.offline.episodes", count)
    OmegaConf.update(merged, "eval.seed", seed)
    selected = resolved[0]["inference"] if resolved is not None else {}
    weights = get_eval_param(override_cfg, "use_ema", section, override=use_ema)
    if weights is None:
        weights = selected.get("use_ema", get_eval_param(cfg, "use_ema", section, default=True))

    steps = get_eval_param(
        override_cfg, "inference_steps_list", section, override=inference_steps_list
    )
    step_value = get_eval_param(override_cfg, "inference_steps", section)
    if steps is None and step_value is not None:
        steps = [step_value]
    if steps is None:
        steps = (
            [selected["inference_steps"]]
            if selected
            else get_eval_param(cfg, "inference_steps_list", section)
        )
    if steps is None:
        steps = [get_eval_param(cfg, "inference_steps", section, default=10)]
    steps = list(steps)

    if type(weights) is not bool:
        raise ValueError("use_ema must be boolean")
    validate_inference_steps(steps)
    action_steps = get_eval_param(override_cfg, "action_steps", section, override=action_steps)
    if action_steps is None:
        action_steps = selected.get("action_steps", get_eval_param(cfg, "action_steps", section))
    action_steps = resolve_action_steps(merged.agent, action_steps)
    OmegaConf.update(merged, f"eval.{section}.action_steps", action_steps)
    if selection_record is not None:
        if checkpoint != "best" and resolve_checkpoint_path(exp_dir, checkpoint) != resolved[1]:
            raise ValueError("Checkpoint override differs from selection record")
        if (
            weights != selected["use_ema"]
            or steps != [selected["inference_steps"]]
            or action_steps != selected["action_steps"]
        ):
            raise ValueError("Selection record cannot override raw/EMA, NFE or action_steps")
    path = resolved[1] if resolved else resolve_checkpoint_path(exp_dir, checkpoint)
    return merged, weights, steps, resolved[0] if resolved else None, path


def evaluate_policy(
    cfg,
    runner,
    checkpoint,
    plan,
    *,
    use_ema,
    inference_steps_list,
    output_root,
    action_steps=None,
    output_dir=None,
    record_video=False,
    selection=None,
    heldout=False,
    seed_plan_version=None,
):
    """Load once and evaluate the same physical episodes at each requested NFE."""
    seed = cfg.eval.seed
    set_seed(seed)
    for leaf in iter_leaf_env_runners(runner):
        leaf.record_video = record_video
    agent = restore_policy_agent(cfg, checkpoint, use_ema=use_ema, device=cfg.training.device)
    action_steps = resolve_action_steps(agent, action_steps)
    if selection is not None and agent.checkpoint_step != selection["global_step"]:
        raise ValueError("Selection global_step disagrees with checkpoint state")
    directory = create_run_directory(output_root, output_dir)
    selection_inference = selection["inference"] if selection is not None else None
    reference = save_eval_snapshot(
        directory,
        cfg,
        runner,
        checkpoint=artifact_reference(checkpoint, cfg.experiment_dir),
        global_step=agent.checkpoint_step,
        use_ema=use_ema,
        inference_steps_list=inference_steps_list,
        action_steps=action_steps,
        selection_inference=selection_inference,
        effective_episodes=plan_size(plan),
        shuffle_seed=seed,
        policy_seed_mode="episode_seed",
        seed_plan_version=seed_plan_version,
        task_seeds=plan,
        selection_id=selection["selection_id"] if selection is not None else None,
        selection_result=selection["selection_result"] if selection is not None else None,
        heldout_from_selection=heldout,
    )
    results = []
    for nfe in inference_steps_list:
        output = directory / f"inference_steps{nfe}" if len(inference_steps_list) > 1 else directory
        output.mkdir(parents=True, exist_ok=True)
        result = run_eval_plan(
            runner,
            agent,
            plan,
            inference_steps=nfe,
            action_steps=action_steps,
            video_save_dir=output if record_video else None,
        )
        details = collect_episode_details(result)
        stats = compute_eval_stats(result)
        inference = dict(use_ema=use_ema, inference_steps=nfe, action_steps=action_steps)
        info = {
            "checkpoint": artifact_reference(checkpoint, cfg.experiment_dir),
            "global_step": agent.checkpoint_step,
            "eval_config": reference,
            **inference,
            "selection_inference": selection_inference,
            "matches_selection_inference": (
                all(selection_inference[key] == value for key, value in inference.items())
                if selection_inference is not None
                else None
            ),
            "evaluation_seeds": plan,
            "seed_plan_version": seed_plan_version,
            "heldout_from_selection": heldout,
            "episode_details": details,
            "statistics": stats,
            "success_rate": stats["micro_success_rate"],
            "avg_steps": mean_success_steps(details),
        }
        atomic_json(output / "result_details.json", info, overwrite=False)
        with atomic_path(output / "result.txt", overwrite=False) as temporary:
            temporary.write_text(f"{info['success_rate']}\n")
        results.append(info)
        cprint(
            f"NFE={nfe}: {stats['n_success']}/{stats['n_valid_episodes']} "
            f"({info['success_rate']:.1%}), avg_steps={info['avg_steps']}",
            "cyan",
        )
    if len(results) > 1:
        atomic_json(
            directory / "eval_summary.json",
            [
                {
                    key: result[key]
                    for key in (
                        "inference_steps",
                        "action_steps",
                        "matches_selection_inference",
                        "success_rate",
                        "avg_steps",
                        "statistics",
                    )
                }
                for result in results
            ],
            overwrite=False,
        )
    cprint(f"Results saved to: {directory}", "cyan")
    return results


def evaluate_checkpoint(
    exp_dir,
    cfg,
    *,
    checkpoint="best",
    episodes=None,
    inference_steps_list=None,
    action_steps=None,
    use_ema=None,
    overrides=(),
    selection_record=None,
    output_dir=None,
    record_video=False,
    eval_seed=None,
):
    cfg, use_ema, steps, selection, path = resolve_eval_request(
        cfg,
        exp_dir,
        checkpoint,
        overrides,
        use_ema=use_ema,
        inference_steps_list=inference_steps_list,
        action_steps=action_steps,
        selection_record=selection_record,
        eval_seed=eval_seed,
        episodes=episodes,
    )
    runner = build_eval_runner(cfg)
    pools = task_seed_pools(runner)
    if selection is not None:
        plan = selection_test_seeds(selection, pools, cfg.eval.offline.episodes)
    else:
        plan, _, _ = plan_selection_seeds(
            pools,
            cfg.eval.seed,
            episodes=cfg.eval.offline.episodes,
            initial_episodes=get_eval_param(cfg, "initial_episodes", "select_best", default=25),
            batch_size=get_eval_param(cfg, "batch_size", "select_best", default=5),
            max_episodes=get_eval_param(cfg, "max_episodes", "select_best", default=100),
        )
    if record_video is None:
        record_video = get_eval_param(cfg, "enabled", "video", default=True)
    return evaluate_policy(
        cfg,
        runner,
        path,
        plan,
        use_ema=use_ema,
        inference_steps_list=steps,
        action_steps=cfg.eval.offline.action_steps,
        output_root=Path(exp_dir) / "eval_dexsim",
        output_dir=output_dir,
        record_video=record_video,
        selection=selection,
        heldout=selection is not None,
        seed_plan_version=SEED_PLAN_VERSION,
    )
