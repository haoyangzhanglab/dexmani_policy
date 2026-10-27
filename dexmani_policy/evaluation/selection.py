"""Evaluate milestone checkpoints with a fixed budget and publish the best."""

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
    frozen_test_request,
    make_request,
    prepare_evaluation_config,
    task_seed_pools,
)
from dexmani_policy.evaluation.seeds import split_evaluation_seeds


def rank_selection_results(results):
    ranks = []
    for result in results:
        stats = result["statistics"]
        avg = stats["avg_steps"]
        ranks.append(
            (
                stats["micro_success_rate"],
                -(avg if avg is not None else float("inf")),
                result["global_step"],
            )
        )
    return max(range(len(results)), key=ranks.__getitem__)


def select_best_checkpoint(exp_dir, cfg, *, overrides=None, output_dir=None) -> tuple[dict, Path]:
    from dexmani_policy.evaluation.rollout import evaluate_policy

    root = Path(exp_dir).expanduser().resolve()
    cfg = prepare_evaluation_config(
        cfg, OmegaConf.merge({"eval": {"video": {"enabled": False}}}, overrides or {})
    )
    if cfg["eval"].get("task_seeds") is not None:
        raise ValueError("Selection uses its predeclared test/selection split, not eval.task_seeds")
    test, selection = split_evaluation_seeds(
        task_seed_pools(cfg),
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
    request = make_request(
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
        request, task_seeds=test, source={"phase": "test", "heldout_from_selection": True}
    )
    record = {
        "status": "running",
        "experiment_dir": str(root),
        "selection_result": str(path.relative_to(root)),
        "selection_task_seeds": selection,
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
            results.append(result)

        best = rank_selection_results(results)
        candidate = record["candidates"][best]
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

    record, checkpoint = select_best_checkpoint(exp_dir, cfg, overrides=overrides)
    request = frozen_test_request(record, checkpoint)
    _, result_path = evaluate_policy(request, output_root=Path(exp_dir) / "eval_dexsim")
    return result_path
