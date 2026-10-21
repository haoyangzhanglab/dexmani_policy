"""Saved evaluation inputs, seed plans and rollout orchestration."""

from __future__ import annotations

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
    plan_size,
    select_eval_seeds,
    validate_task_seeds,
)
from dexmani_policy.training.common.checkpoint import (
    resolve_best_checkpoint,
    resolve_checkpoint_path,
    resolve_selection_record,
    restore_policy_agent,
)
from dexmani_policy.utils.config import positive_int, validate_action_key_consistency
from dexmani_policy.utils.runtime import atomic_path, set_seed


def resolve_eval_seed(cfg, cli_seed: int | None = None) -> int:
    if cli_seed is not None:
        return cli_seed
    training_seed = cfg.training.get("seed", 0) if hasattr(cfg, "training") else 0
    return training_seed + 1024


def parse_eval_overrides(overrides: list[str]):
    """Only evaluation/inference controls may override checkpoint evaluation."""
    for override in overrides:
        key, separator, _ = override.partition("=")
        if key == "eval.seed_manifest":
            raise ValueError(
                "Unsupported eval.seed_manifest; use evaluation seed and episode counts"
            )
        if key.split(".")[-1] in {"denoise_steps", "denoise_timesteps_list"}:
            raise ValueError("Use inference_steps or inference_steps_list for evaluation")
        if not separator or not (
            key.startswith(("eval.", "env_runner.")) or key in {"training.device", "training.seed"}
        ):
            raise ValueError(f"Evaluation override cannot redefine saved inference config: {key}")
        if any(
            part in key.split(".")
            for part in (
                "n_obs_steps",
                "sensor_modalities",
                "task_text",
                "rgb_preprocess_size",
                "rgb_random_crop_size",
                "rgb_keep_uint8",
            )
        ):
            raise ValueError(f"Model inputs come from saved dataset config: {key}")
    return OmegaConf.from_dotlist(overrides)


def validate_inference_steps(inference_steps_list) -> None:
    """Require distinct positive NFEs so each output directory has one result."""
    if not inference_steps_list:
        raise ValueError("inference_steps_list must be non-empty")
    for nfe in inference_steps_list:
        positive_int(nfe, "inference_steps")
    if len(set(inference_steps_list)) != len(inference_steps_list):
        raise ValueError("Inference steps must be distinct to preserve per-value results")


def add_evaluation_arguments(parser) -> None:
    parser.add_argument("--policy-name", required=True, help="Experiment policy path, e.g. ddp/sat.")
    parser.add_argument("--task-name", required=True, help="Task directory in the saved experiment.")
    parser.add_argument("--exp-name", required=True, help="Run directory under experiments/policy/task.")
    parser.add_argument(
        "--inference-steps",
        type=int,
        help="DDIM/Euler steps; defaults come from the saved config or selection record.",
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
    env_runner = hydra.utils.instantiate(cfg.env_runner)
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


def get_eval_param(cfg, param: str, section: str | None = None, *, default: Any = None) -> Any:
    """Resolve section → shared eval → default; only None means unspecified."""
    eval_cfg = cfg.get("eval") or {}
    section_cfg = (eval_cfg.get(section) or {}) if section else {}
    for source in (section_cfg, eval_cfg):
        val = source.get(param)
        if val is not None:
            return val
    return default


def task_seed_pools(runner):
    """Physical seeds in task execution order; no reference-task indirection."""
    return {leaf.task_name: list(leaf.get_seed_list()) for leaf in iter_leaf_env_runners(runner)}


def run_eval_plan(runner, agent, plan, *, inference_steps, video_save_dir=None):
    kwargs = dict(
        inference_steps=inference_steps,
        video_save_dir=video_save_dir,
    )
    if hasattr(runner, "runners"):
        return runner.run(agent, task_seeds=plan, **kwargs)
    validate_task_seeds(plan, [runner.task_name])
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


def selection_task_seeds(record):
    """Derive exclusion seeds from completed stages, including losing candidates."""
    stages = record["stages"]
    if not stages or record["ckpt_relpath"] not in {stage["checkpoint"] for stage in stages}:
        raise ValueError("Selected checkpoint has no evaluation evidence")
    selected = {}
    tasks = stages[0]["requested_task_seeds"]
    for stage in stages:
        if stage["status"] != "completed":
            raise ValueError("Selection contains an incomplete stage")
        plan = stage["requested_task_seeds"]
        validate_task_seeds(plan, tasks)
        validate_episode_details(stage["episode_details"], plan)
        for task, seeds in plan.items():
            selected[task] = list(dict.fromkeys(selected.get(task, []) + seeds))
    return selected


def explicit_eval_value(cfg, section, name):
    missing = object()
    for key in (f"eval.{section}.{name}", f"eval.{name}"):
        value = OmegaConf.select(cfg, key, default=missing)
        if value is not missing:
            return True, value
    return False, None


def resolve_eval_request(
    cfg,
    exp_dir,
    checkpoint="best",
    overrides=(),
    *,
    section="offline",
    use_ema=None,
    inference_steps_list=None,
    selection_record=None,
):
    """CLI > explicit dotlist > selected checkpoint > saved defaults."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    override_cfg = parse_eval_overrides(list(overrides))
    merged = OmegaConf.merge(cfg, override_cfg)
    merged.experiment_dir = str(exp_dir)
    resolved = (
        resolve_selection_record(exp_dir, selection_record)
        if selection_record is not None
        else resolve_best_checkpoint(exp_dir) if checkpoint == "best" else None
    )
    present, value = explicit_eval_value(override_cfg, section, "use_ema")
    if use_ema is not None:
        weights = use_ema
    elif present:
        weights = value
    elif resolved is not None:
        weights = resolved[0]["inference"]["use_ema"]
    else:
        weights = get_eval_param(cfg, "use_ema", section, default=True)

    list_present, list_value = explicit_eval_value(override_cfg, section, "inference_steps_list")
    step_present, step_value = explicit_eval_value(override_cfg, section, "inference_steps")
    if inference_steps_list is not None:
        steps = list(inference_steps_list)
    elif list_present and list_value is not None:
        steps = list(list_value)
    elif step_present:
        steps = [step_value]
    elif resolved is not None:
        steps = [resolved[0]["inference"]["inference_steps"]]
    else:
        saved_steps = get_eval_param(cfg, "inference_steps_list", section)
        steps = (
            list(saved_steps)
            if saved_steps is not None
            else [get_eval_param(cfg, "inference_steps", section, default=10)]
        )

    if type(weights) is not bool:
        raise ValueError("use_ema must be boolean")
    validate_inference_steps(steps)
    if selection_record is not None:
        if checkpoint != "best" and resolve_checkpoint_path(exp_dir, checkpoint) != resolved[1]:
            raise ValueError("Checkpoint override differs from selection record")
        inference = resolved[0]["inference"]
        if weights != inference["use_ema"] or steps != [inference["inference_steps"]]:
            raise ValueError("Selection record cannot override raw/EMA or NFE")
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
    output_dir=None,
    record_video=False,
    selection=None,
    heldout=False,
):
    """Load once and evaluate the same physical episodes at each requested NFE."""
    validate_inference_steps(inference_steps_list)
    seed = resolve_eval_seed(cfg)
    set_seed(seed)
    for leaf in iter_leaf_env_runners(runner):
        leaf.record_video = record_video
    agent = restore_policy_agent(cfg, checkpoint, use_ema=use_ema, device=cfg.training.device)
    if selection is not None and agent.checkpoint_step != selection["global_step"]:
        raise ValueError("Selection global_step disagrees with checkpoint state")
    directory = create_run_directory(output_root, output_dir)
    reference = save_eval_snapshot(
        directory,
        cfg,
        runner,
        checkpoint=artifact_reference(checkpoint, cfg.experiment_dir),
        global_step=agent.checkpoint_step,
        use_ema=use_ema,
        inference_steps_list=inference_steps_list,
        effective_episodes=plan_size(plan),
        shuffle_seed=seed,
        policy_seed_mode="episode_seed",
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
            video_save_dir=output if record_video else None,
        )
        details = collect_episode_details(result)
        stats = compute_eval_stats(result)
        info = {
            "checkpoint": artifact_reference(checkpoint, cfg.experiment_dir),
            "global_step": agent.checkpoint_step,
            "eval_config": reference,
            "use_ema": use_ema,
            "inference_steps": nfe,
            "evaluation_seeds": plan,
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
                    for key in ("inference_steps", "success_rate", "avg_steps", "statistics")
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
    use_ema=None,
    overrides=(),
    selection_record=None,
    output_dir=None,
    record_video=False,
):
    cfg, use_ema, steps, selection, path = resolve_eval_request(
        cfg,
        exp_dir,
        checkpoint,
        overrides,
        use_ema=use_ema,
        inference_steps_list=inference_steps_list,
        selection_record=selection_record,
    )
    excluded = selection_task_seeds(selection) if selection is not None else None
    runner = build_eval_runner(cfg)
    episodes = (
        episodes
        if episodes is not None
        else get_eval_param(cfg, "episodes", "offline", default=100)
    )
    plan = select_eval_seeds(task_seed_pools(runner), resolve_eval_seed(cfg), episodes, excluded)
    if record_video is None:
        record_video = get_eval_param(cfg, "enabled", "video", default=True)
    return evaluate_policy(
        cfg,
        runner,
        path,
        plan,
        use_ema=use_ema,
        inference_steps_list=steps,
        output_root=Path(exp_dir) / "eval_dexsim",
        output_dir=output_dir,
        record_video=record_video,
        selection=selection,
        heldout=selection is not None,
    )
