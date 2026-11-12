import os
from pathlib import Path

import numpy as np

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import write_json
from dexmani_policy.evaluation.rollout import create_run_directory, evaluate_policy
from dexmani_policy.evaluation.seeds import load_seed_pool, split_evaluation_seeds


def candidate_score(candidate):
    stats = candidate["statistics"]
    steps = stats["avg_steps"]
    return (stats["success_rate"], -(steps if steps is not None else float("inf")), candidate["global_step"])


def run_pipeline(checkpoints, *, weights=None, inference_steps=None, action_steps=None,
                 selection_episodes=25, test_episodes=100, seed=1024, device="cuda:0",
                 video=False, output_dir=None):
    checkpoints = [Path(path).expanduser().resolve() for path in checkpoints]
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Candidates must be a nonempty list of distinct checkpoint files")
    root = checkpoints[0].parent.parent
    loaded = load_checkpoint(checkpoints[0])
    cfg = loaded.effective_config
    reference_task_name = cfg["env_runner"]["task_name"]
    reference_env_kwargs = cfg["env_runner"]["env_kwargs"]
    reference_control_dt_s = cfg["control_dt_s"]
    del loaded
    for checkpoint in checkpoints[1:]:
        loaded = load_checkpoint(checkpoint)
        cfg = loaded.effective_config
        if cfg["env_runner"]["task_name"] != reference_task_name:
            raise ValueError("Candidates must use the same task_name")
        if cfg["env_runner"]["env_kwargs"] != reference_env_kwargs:
            raise ValueError("Candidates must use the same env_kwargs")
        control_dt_s = cfg["control_dt_s"]
        if not np.isclose(control_dt_s, reference_control_dt_s, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Candidates must use the same control_dt_s: "
                f"reference={reference_control_dt_s}, candidate={control_dt_s} ({checkpoint})"
            )
        del loaded
    test, selection = split_evaluation_seeds(load_seed_pool(reference_task_name), seed, test_episodes,
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
    record.update(ckpt_relpath=winner["checkpoint"], global_step=winner["global_step"])
    write_json(path, record)
    checkpoint = directory / winner["checkpoint"]
    _, result_path = evaluate_policy(
        checkpoint, weights=winner["weights"], inference_steps=winner["inference_steps"], action_steps=winner["action_steps"],
        seeds=test, seed=seed, device=device, video=video, output_dir=directory / "test")
    record["test_result"] = os.path.relpath(result_path, directory)
    write_json(path, record)
    return path
