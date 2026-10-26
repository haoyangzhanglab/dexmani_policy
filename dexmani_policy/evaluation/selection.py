"""Equal-budget candidate evaluation followed by pure evidence-based ranking."""

import json
import traceback
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.experiments import discover_milestone_checkpoints, file_sha256
from dexmani_policy.env_runner.common.seeds import split_evaluation_seeds
from dexmani_policy.evaluation.artifacts import (
    artifact_reference,
    create_run_directory,
    rank_selection_results,
    verify_selection_record,
    training_source_evidence,
    write_failure,
    write_json,
)
from dexmani_policy.evaluation.config import (
    make_request,
    prepare_evaluation_config,
    resolve_tasks,
    task_seed_pools,
)


def select_best_checkpoint(exp_dir, cfg, *, overrides=None, output_dir=None):
    from dexmani_policy.evaluation.rollout import evaluate_policy

    root = Path(exp_dir).resolve()
    cfg = prepare_evaluation_config(
        cfg, OmegaConf.merge({"eval": {"video": {"enabled": False}}}, overrides or {})
    )
    if cfg["eval"].get("task_seeds") is not None:
        raise ValueError("Selection uses its predeclared test/selection split, not eval.task_seeds")
    tasks = resolve_tasks(cfg)
    pools = task_seed_pools(tasks)
    test, selection = split_evaluation_seeds(
        pools,
        cfg["eval"]["seed"],
        positive_int(cfg["eval"]["episodes"], "eval.episodes"),
        positive_int(cfg["eval"]["selection_episodes"], "eval.selection_episodes"),
    )
    milestones = discover_milestone_checkpoints(root)
    request = make_request(
        cfg,
        milestones[0].path,
        tasks,
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
    record = {
        "status": "running",
        "experiment_dir": str(root),
        "seed_pools": pools,
        "selection_task_seeds": selection,
        "test_request": replace(
            request, task_seeds=test, source={"phase": "test", "heldout_from_selection": True}
        ).to_dict(),
        "candidates": [
            {"checkpoint": artifact_reference(m.path, root), "global_step": m.global_step,
             "checkpoint_sha256": file_sha256(m.path)}
            for m in milestones
        ],
        "candidate_results": [],
        "training_source_evidence": training_source_evidence(cfg, {"selection": selection, "test": test}),
    }
    write_json(path, record, overwrite=False)
    results = []
    try:
        for milestone, candidate in zip(milestones, record["candidates"], strict=True):
            result_path = evaluate_policy(
                replace(
                    request,
                    checkpoint=str(milestone.path),
                    source={**request.source, "global_step": milestone.global_step,
                            "checkpoint_sha256": candidate["checkpoint_sha256"]},
                ),
                output_root=directory / "candidates",
            )
            record["candidate_results"].append(artifact_reference(result_path, root))
            results.append(json.loads(result_path.read_text()))
            write_json(path, record)
        best = rank_selection_results(record, results)
        candidate = record["candidates"][best]
        record["test_request"]["checkpoint"] = str(root / candidate["checkpoint"])
        record.update(
            status="completed",
            ckpt_relpath=candidate["checkpoint"],
            global_step=candidate["global_step"],
            checkpoint_sha256=candidate["checkpoint_sha256"],
            runtime=results[best]["runtime"],
            inference={
                key: getattr(request, key) for key in ("use_ema", "inference_steps", "action_steps")
            },
        )
        write_json(path, record)
        publish_best(root, path)
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        write_failure(path, record)
        raise
    return path


def publish_best(root, record_path):
    record, checkpoint = verify_selection_record(root, record_path)
    write_json(
        Path(root) / "best_ckpt.json",
        {key: record[key] for key in ("ckpt_relpath", "global_step", "checkpoint_sha256", "inference", "selection_result")},
    )
    return checkpoint


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify saved selection evidence without running evaluation")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--selection-record", required=True)
    parser.add_argument("--publish-best", action="store_true", help="Publish a direct best pointer after verification")
    args = parser.parse_args()
    if args.publish_best:
        checkpoint = publish_best(args.experiment_dir, args.selection_record)
    else:
        _, checkpoint = verify_selection_record(args.experiment_dir, args.selection_record)
    print(f"Selection verified: {checkpoint}")


if __name__ == "__main__":
    main()
