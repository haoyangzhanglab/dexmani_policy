import os
from pathlib import Path

import numpy as np

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.io_util import write_json
from dexmani_policy.deployment import select_inference_weights
from dexmani_policy.evaluation.rollout import create_run_directory, evaluate_policy, resolve_rollout_options
from dexmani_policy.evaluation.seeds import split_evaluation_seeds


def candidate_score(candidate):
    stats = candidate["statistics"]
    steps = stats["avg_steps"]
    return (stats["success_rate"], -(steps if steps is not None else float("inf")), candidate["global_step"])


def run_pipeline(checkpoints, *, weights=None, inference_steps=None, action_steps=None,
                 selection_episodes=25, test_episodes=100, seed=1024, device="cuda:0",
                 video=False, output_dir=None, skip_intermediate_observations=False):
    checkpoints = [Path(path).expanduser().resolve() for path in checkpoints]
    if len(checkpoints) < 2 or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Selection requires at least two distinct checkpoints. Use evaluate for a single checkpoint.")
    root = checkpoints[0].parent.parent

    def inspect_candidate(checkpoint):
        loaded = load_checkpoint(checkpoint)
        cfg = loaded.effective_config
        try:
            select_inference_weights(loaded, weights=weights, inference_steps=inference_steps)
            runner_class, _ = resolve_rollout_options(
                cfg, action_steps=action_steps,
                skip_intermediate_observations=skip_intermediate_observations, video=video,
            )
        except ValueError as error:
            raise ValueError(f"{checkpoint}: {error}") from error
        return cfg, runner_class

    reference, runner_class = inspect_candidate(checkpoints[0])
    reference_env = reference["env_runner"]
    reference_control_dt_s = reference["control_dt_s"]
    for checkpoint in checkpoints[1:]:
        cfg, _ = inspect_candidate(checkpoint)
        env = cfg["env_runner"]
        if env["_target_"] != reference_env["_target_"]:
            raise ValueError(f"{checkpoint}: Candidates must use the same environment runner")
        if env["task_name"] != reference_env["task_name"]:
            raise ValueError(f"{checkpoint}: Candidates must use the same task_name")
        if (env.get("env_kwargs") or {}) != (reference_env.get("env_kwargs") or {}):
            raise ValueError(f"{checkpoint}: Candidates must use the same env_kwargs")
        control_dt_s = cfg["control_dt_s"]
        if not np.isclose(control_dt_s, reference_control_dt_s, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Candidates must use the same control_dt_s: "
                f"reference={reference_control_dt_s}, candidate={control_dt_s} ({checkpoint})"
            )
    seed_pool = runner_class.load_seed_pool(reference_env["task_name"])
    test, selection = split_evaluation_seeds(
        seed_pool, seed, test_episodes, positive_int(selection_episodes, "selection_episodes"),
    )
    directory = create_run_directory(root / "eval", output_dir)
    path = directory / "selection_result.json"
    record = dict(selection_seeds=selection, test_seeds=test, seed=seed,
                  overrides=dict(weights=weights, inference_steps=inference_steps, action_steps=action_steps),
                  device=str(device), test_video=video, candidates=[],
                  skip_intermediate_observations=skip_intermediate_observations)
    for index, checkpoint in enumerate(checkpoints):
        result, result_path = evaluate_policy(
            checkpoint, weights=weights, inference_steps=inference_steps,
            action_steps=action_steps, seeds=selection, seed=seed, device=device, video=False,
            skip_intermediate_observations=skip_intermediate_observations,
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
        seeds=test, seed=seed, device=device, video=video, output_dir=directory / "test",
        skip_intermediate_observations=skip_intermediate_observations,
    )
    record["test_result"] = os.path.relpath(result_path, directory)
    write_json(path, record)
    return path
