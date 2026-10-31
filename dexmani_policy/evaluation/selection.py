import fcntl
import traceback
from pathlib import Path

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.evaluation.artifacts import create_run_directory, write_failure, write_json
from dexmani_policy.evaluation.rollout import evaluate_policy, load_seed_pool
from dexmani_policy.evaluation.seeds import split_evaluation_seeds


def candidate_score(candidate):
    stats = candidate["statistics"]
    steps = stats["avg_steps"]
    return (stats["success_rate"], -(steps if steps is not None else float("inf")), candidate["global_step"])


def select_checkpoint(checkpoints, *, weights=None, inference_steps=None, action_steps=None,
                      selection_episodes=25, test_episodes=100, seed=1024, device="cuda:0",
                      video=True, output_dir=None):
    checkpoints = [Path(path).expanduser().resolve(strict=True) for path in checkpoints]
    if not checkpoints or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Candidates must be a nonempty list of distinct checkpoint files")
    root = checkpoints[0].parent.parent
    if any(path.parent != root / "checkpoints" for path in checkpoints):
        raise ValueError("Select explicit checkpoints from one training run")
    cfg = load_checkpoint(checkpoints[0], mmap=True).effective_config
    weights = ("ema" if cfg["eval"]["use_ema"] else "raw") if weights is None else weights
    if weights not in {"raw", "ema"}:
        raise ValueError("weights must be raw or ema")
    inference_steps = cfg["eval"]["inference_steps"] if inference_steps is None else inference_steps
    action_steps = cfg["agent"]["n_action_steps"] if action_steps is None else action_steps
    positive_int(inference_steps, "inference_steps")
    positive_int(action_steps, "action_steps")
    test, selection = split_evaluation_seeds(load_seed_pool(cfg), seed, test_episodes,
                                             positive_int(selection_episodes, "selection_episodes"))
    if output_dir is not None and not Path(output_dir).expanduser().resolve().is_relative_to(root):
        raise ValueError("Until external best consumers migrate, selection output must remain inside the run")
    directory = create_run_directory(root / "eval_ckpt_selector", output_dir)
    path = directory / "selection_result.json"
    record = dict(status="running", selection_result=str(path.relative_to(root)),
                  selection_seeds=selection, test_seeds=test, seed=seed,
                  inference=dict(use_ema=weights == "ema", inference_steps=inference_steps, action_steps=action_steps),
                  device=device, test_video=video, candidates=[])
    for checkpoint in checkpoints:
        step = load_checkpoint(checkpoint, mmap=True).global_step
        record["candidates"].append(dict(checkpoint=str(checkpoint.relative_to(root)), global_step=step))
    write_json(path, record, overwrite=False)
    try:
        for index, candidate in enumerate(record["candidates"]):
            result, result_path = evaluate_policy(
                root / candidate["checkpoint"], weights=weights, inference_steps=inference_steps,
                action_steps=action_steps, seeds=selection, seed=seed, device=device, video=False,
                output_dir=directory / "candidates" / str(index))
            candidate.update(result=str(result_path.relative_to(root)), statistics=result["statistics"])
        winner = max(record["candidates"], key=candidate_score)
        record.update(status="completed", ckpt_relpath=winner["checkpoint"], global_step=winner["global_step"])
        write_json(path, record)
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        write_failure(path, record)
        raise
    return record, root / winner["checkpoint"]


def run_pipeline(checkpoints, **options):
    if not checkpoints:
        raise ValueError("Candidates must be nonempty")
    root = Path(checkpoints[0]).expanduser().resolve(strict=True).parent.parent
    # Preserve the existing best publication lock until actual external consumers migrate.
    with (root / ".pipeline.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another eval pipeline is running for {root}") from error
        record, checkpoint = select_checkpoint(checkpoints, **options)
        path = root / record["selection_result"]
        try:
            write_json(root / "best_ckpt.json", {key: record[key] for key in
                       ("ckpt_relpath", "global_step", "inference", "selection_result")})
            inference = record["inference"]
            _, result_path = evaluate_policy(
                checkpoint, weights="ema" if inference["use_ema"] else "raw",
                inference_steps=inference["inference_steps"], action_steps=inference["action_steps"],
                seeds=record["test_seeds"], seed=record["seed"], device=record["device"],
                video=record["test_video"], output_dir=path.parent / "test")
            record["test_result"] = str(result_path.relative_to(root))
            write_json(path, record)
        except BaseException:
            record.update(status="failed", error=traceback.format_exc())
            write_failure(path, record)
            raise
        return path
