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


def run_pipeline(checkpoints, *, weights=None, inference_steps=None, action_steps=None,
                 selection_episodes=25, test_episodes=100, seed=1024, device="cuda:0",
                 video=True, output_dir=None, run_dir=None):
    checkpoints = [Path(path).expanduser().resolve(strict=True) for path in checkpoints]
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Candidates must be a nonempty list of distinct checkpoint files")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint must be a file: {checkpoint}")
    root = checkpoints[0].parent.parent if run_dir is None else Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_checkpoint(checkpoints[0]).effective_config
    weights = ("ema" if cfg["eval"]["use_ema"] else "raw") if weights is None else weights
    if weights not in {"raw", "ema"}:
        raise ValueError("weights must be raw or ema")
    inference_steps = cfg["eval"]["inference_steps"] if inference_steps is None else inference_steps
    action_steps = cfg["agent"]["n_action_steps"] if action_steps is None else action_steps
    positive_int(inference_steps, "inference_steps")
    positive_int(action_steps, "action_steps")
    test, selection = split_evaluation_seeds(load_seed_pool(cfg["env_runner"]["task_name"]), seed, test_episodes,
                                             positive_int(selection_episodes, "selection_episodes"))
    directory = create_run_directory(root / "eval_ckpt_selector", output_dir)
    path = directory / "selection_result.json"
    record = dict(selection_seeds=selection, test_seeds=test, seed=seed,
                  inference=dict(use_ema=weights == "ema", inference_steps=inference_steps, action_steps=action_steps),
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
        ))
    winner = max(record["candidates"], key=candidate_score)
    checkpoint = directory / winner["checkpoint"]
    _, result_path = evaluate_policy(
        checkpoint, weights=weights, inference_steps=inference_steps, action_steps=action_steps,
        seeds=test, seed=seed, device=device, video=video, output_dir=directory / "test")
    record.update(ckpt_relpath=winner["checkpoint"], global_step=winner["global_step"],
                  test_result=os.path.relpath(result_path, directory))
    write_json(path, record, overwrite=False)
    write_json(root / "best_ckpt.json", dict(
        ckpt_relpath=os.path.relpath(checkpoint, root), inference=record["inference"],
    ))
    return path
