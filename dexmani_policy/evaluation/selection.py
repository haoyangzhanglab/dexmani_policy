"""Evaluate milestone checkpoints with a fixed budget and publish the best."""

import fcntl
import traceback
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.experiments import discover_milestone_checkpoints, milestone_steps
from dexmani_policy.evaluation.artifacts import (
    create_run_directory,
    write_failure,
    write_json,
)
from dexmani_policy.evaluation.config import (
    build_eval_config,
    frozen_test_config,
    load_seed_pool,
    merge_eval_config,
)
from dexmani_policy.evaluation.seeds import split_evaluation_seeds


def rank_selection_results(results):
    def rank(result):
        stats = result["statistics"]
        avg = stats["avg_steps"]
        return (
            stats["success_rate"],
            -(avg if avg is not None else float("inf")),
            result["global_step"],
        )
    return max(results, key=rank)


def select_best_checkpoint(exp_dir, cfg, *, overrides=None, output_dir=None) -> tuple[dict, Path]:
    from dexmani_policy.evaluation.rollout import evaluate_policy

    root = Path(exp_dir).expanduser().resolve()
    cfg = merge_eval_config(
        cfg, OmegaConf.merge({"eval": {"video": {"enabled": False}}}, overrides or {})
    )
    if cfg["eval"].get("seeds") is not None:
        raise ValueError("Selection uses its predeclared test/selection split, not eval.seeds")
    test, selection = split_evaluation_seeds(
        load_seed_pool(cfg),
        cfg["eval"]["seed"],
        cfg["eval"]["episodes"],
        positive_int(cfg["eval"]["selection_episodes"], "eval.selection_episodes"),
    )

    milestones = discover_milestone_checkpoints(root)
    expected = milestone_steps(cfg["training"]["loop"]["total_train_steps"])
    actual = {m.global_step: m.pct for m in milestones}
    if actual != expected or len(milestones) != len(expected):
        raise ValueError(
            f"Selection requires milestones {expected}; found {actual}; "
            f"missing steps: {sorted(expected.keys() - actual.keys())}"
        )
    request = build_eval_config(
        cfg,
        milestones[0].path,
        selection,
        source={"phase": "selection", "heldout_from_selection": False},
    )

    output_root = root / "eval_ckpt_selector"
    if output_dir is not None and not Path(output_dir).expanduser().resolve().is_relative_to(
        output_root
    ):
        raise ValueError("Selection output must remain inside experiment/eval_ckpt_selector")
    directory = create_run_directory(output_root, output_dir)
    path = directory / "selection_result.json"
    test_request = replace(
        request, seeds=test, source={"phase": "test", "heldout_from_selection": True}
    )
    request = replace(
        request, config={**cfg, "eval": {**cfg["eval"], "video": {"enabled": False}}}
    )
    record = {
        "status": "running",
        "selection_result": str(path.relative_to(root)),
        "selection_seeds": selection,
        "test_request": vars(test_request).copy(),
        "candidates": [
            {"checkpoint": str(m.path.relative_to(root)), "global_step": m.global_step}
            for m in milestones
        ],
        "candidate_results": [],
    }
    write_json(path, record, overwrite=False)

    results = []
    try:
        for milestone in milestones:
            result, result_path = evaluate_policy(
                replace(
                    request,
                    checkpoint=str(milestone.path),
                    source={**request.source, "global_step": milestone.global_step},
                ),
                output_root=directory / "candidates",
            )
            record["candidate_results"].append(str(result_path.relative_to(root)))
            results.append({
                "checkpoint": str(milestone.path.relative_to(root)),
                **{key: result[key] for key in ("statistics", "global_step")},
            })

        candidate = rank_selection_results(results)
        checkpoint = root / candidate["checkpoint"]
        record["test_request"]["checkpoint"] = str(checkpoint)
        record.update(
            status="completed",
            ckpt_relpath=candidate["checkpoint"],
            global_step=candidate["global_step"],
            inference={
                key: getattr(request, key) for key in ("use_ema", "inference_steps", "action_steps")
            },
        )
        write_json(path, record)
        write_json(
            root / "best_ckpt.json",
            {key: record[key] for key in ("ckpt_relpath", "global_step", "inference", "selection_result")},
        )
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        write_failure(path, record)
        raise

    return record, checkpoint


def run_pipeline(exp_dir, cfg, *, overrides=None) -> Path:
    from dexmani_policy.evaluation.rollout import evaluate_policy

    root = Path(exp_dir).expanduser().resolve()
    # Keep the inode so shell and direct Python invocations share the same lock.
    with (root / ".pipeline.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another eval pipeline is running for {root}") from error
        record, checkpoint = select_best_checkpoint(root, cfg, overrides=overrides)
        request = frozen_test_config(record, checkpoint)
        _, result_path = evaluate_policy(request, output_root=root / "eval_dexsim")
        return result_path
