import os
from pathlib import Path

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import write_json
from dexmani_policy.evaluation.rollout import create_run_directory, evaluate_policy
from dexmani_policy.evaluation.seeds import load_seed_pool, split_evaluation_seeds


def candidate_score(candidate):
    stats = candidate["statistics"]
    steps = stats["avg_steps"]
    return (stats["success_rate"], -(steps if steps is not None else float("inf")), candidate["global_step"])


def _environment_conditions(config):
    runner = config["env_runner"]
    # Match SimRunner's defaults; all other kwargs pass through to dexmani_sim.
    kwargs = {"render_mode": "rgb_array", **(runner.get("env_kwargs") or {})}
    return {
        "env_runner.task_name": runner["task_name"],
        "env_runner.env_rebuild_interval": runner.get("env_rebuild_interval", 25),
        **{f"env_runner.env_kwargs.{key}": value for key, value in kwargs.items()},
    }


def run_pipeline(checkpoints, *, weights=None, inference_steps=None, action_steps=None,
                 selection_episodes=25, test_episodes=100, seed=1024, device="cuda:0",
                 video=True, output_dir=None):
    checkpoints = [Path(path).expanduser().resolve(strict=True) for path in checkpoints]
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Candidates must be a nonempty list of distinct checkpoint files")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint must be a file: {checkpoint}")
    root = checkpoints[0].parent.parent
    loaded = load_checkpoint(checkpoints[0])
    cfg = loaded.effective_config
    del loaded
    baseline = _environment_conditions(cfg)
    for checkpoint in checkpoints[1:]:
        loaded = load_checkpoint(checkpoint)
        conditions = _environment_conditions(loaded.effective_config)
        del loaded
        differences = []
        for field in sorted(baseline.keys() | conditions.keys()):
            if field in baseline and field in conditions and baseline[field] == conditions[field]:
                continue
            expected = repr(baseline[field]) if field in baseline else "<unspecified>"
            actual = repr(conditions[field]) if field in conditions else "<unspecified>"
            differences.append(f"{field}: {expected} != {actual}")
        if differences:
            raise ValueError(
                f"Environment conditions differ between baseline checkpoint {checkpoints[0]} "
                f"and candidate checkpoint {checkpoint}: {'; '.join(differences)}"
            )
    test, selection = split_evaluation_seeds(load_seed_pool(cfg["env_runner"]["task_name"]), seed, test_episodes,
                                             positive_int(selection_episodes, "selection_episodes"))
    directory = create_run_directory(root / "eval", output_dir)
    path = directory / "selection_result.json"
    record = dict(selection_seeds=selection, test_seeds=test, seed=seed,
                  overrides=dict(weights=weights, inference_steps=inference_steps, action_steps=action_steps),
                  device=str(device), test_video=video, candidates=[])
    for index, checkpoint in enumerate(checkpoints):
        result, result_path = evaluate_policy(
            checkpoint, weights=weights, inference_steps=inference_steps,
            action_steps=action_steps, seeds=selection, seed=seed, device=device, video=False,
            output_dir=directory / "candidates" / str(index))
        record["candidates"].append(dict(
            checkpoint=os.path.relpath(checkpoint, directory),
            result=os.path.relpath(result_path, directory),
            statistics=result["statistics"], global_step=result["global_step"],
            weights=result["weights"], inference_steps=result["inference_steps"], action_steps=result["action_steps"],
        ))
    winner = max(record["candidates"], key=candidate_score)
    checkpoint = directory / winner["checkpoint"]
    _, result_path = evaluate_policy(
        checkpoint, weights=winner["weights"], inference_steps=winner["inference_steps"], action_steps=winner["action_steps"],
        seeds=test, seed=seed, device=device, video=video, output_dir=directory / "test")
    record.update(ckpt_relpath=winner["checkpoint"], global_step=winner["global_step"],
                  test_result=os.path.relpath(result_path, directory))
    write_json(path, record)
    return path
