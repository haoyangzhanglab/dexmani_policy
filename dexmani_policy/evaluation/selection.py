"""Equal-budget candidate evaluation followed by pure evidence-based ranking."""

import json
from dataclasses import replace
from pathlib import Path
import traceback

from omegaconf import OmegaConf

from dexmani_policy.common.config import positive_int
from dexmani_policy.common.experiments import discover_milestone_checkpoints, resolve_record_path
from dexmani_policy.env_runner.common.seeds import split_evaluation_seeds
from dexmani_policy.evaluation.artifacts import (
    artifact_reference, create_run_directory, write_json, resolve_selection_record, validate_selection_plan,
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
        "selection_id": directory.name,
        "seed_pools": pools,
        "selection_task_seeds": selection,
        "test_request": replace(
            request, task_seeds=test, source={"phase": "test", "heldout_from_selection": True}
        ).to_dict(),
        "candidates": [
            {"checkpoint": artifact_reference(m.path, root), "global_step": m.global_step}
            for m in milestones
        ],
        "candidate_results": [],
    }
    write_json(path, record, overwrite=False)
    results = []
    try:
        for milestone in milestones:
            result_path = evaluate_policy(
                replace(
                    request,
                    checkpoint=str(milestone.path),
                    source={**request.source, "global_step": milestone.global_step},
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
            inference={
                key: getattr(request, key) for key in ("use_ema", "inference_steps", "action_steps")
            },
        )
        write_json(path, record)
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        write_json(path, record)
        raise
    publish_best(root, {**record, "selection_result": artifact_reference(path, root)})
    return path


def rank_selection_results(record, results):
    """Validate complete candidate evidence, then rank without file or environment I/O."""
    from dexmani_policy.env_runner.common.metrics import (
        compute_evaluation_statistics,
        validate_episodes,
    )

    candidates = record["candidates"]
    if not candidates or len(results) != len(candidates):
        raise ValueError("Selection requires complete results for every declared candidate")
    if len({c["checkpoint"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate selection candidate")
    validate_selection_plan(record)
    protocol = record["test_request"]
    selection = record["selection_task_seeds"]
    ranks = []
    for candidate, result in zip(candidates, results, strict=True):
        if result["status"] != "completed" or result["global_step"] != candidate["global_step"]:
            raise ValueError("Candidate is incomplete or global_step differs")
        request = result["request"]
        expected_checkpoint = str(Path(record["experiment_dir"]) / candidate["checkpoint"])
        if request["checkpoint"] != expected_checkpoint or request["task_seeds"] != selection:
            raise ValueError("Candidate checkpoint or fixed selection plan differs")
        for key in ("config", "tasks", "use_ema", "inference_steps", "action_steps", "numerics"):
            if request[key] != protocol[key]:
                raise ValueError(f"Candidate research conditions differ: {key}")
        for key in ("policy_code", "simulator_code", "torch", "cuda", "parameter_dtypes"):
            if result["runtime"][key] != results[0]["runtime"][key]:
                raise ValueError(f"Candidate runtime conditions differ: {key}")
        validate_episodes(result["episodes"], selection)
        stats = compute_evaluation_statistics(result["episodes"])
        if result["statistics"] != stats:
            raise ValueError("Candidate statistics disagree with episode facts")
        avg = stats["avg_steps"]
        ranks.append(
            (
                stats["micro_success_rate"],
                -(avg if avg is not None else float("inf")),
                candidate["global_step"],
            )
        )
    return max(range(len(candidates)), key=ranks.__getitem__)


def publish_best(root, record):
    write_json(Path(root) / "best_ckpt.json", {
        key: record[key] for key in ("ckpt_relpath", "global_step", "inference", "selection_result")
    })


def verify_selection_record(experiment_dir, record_path):
    root = Path(experiment_dir).resolve()
    record, checkpoint = resolve_selection_record(root, record_path)
    results = [json.loads(resolve_record_path(root, ref).read_text()) for ref in record["candidate_results"]]
    for candidate in record["candidates"]:
        resolve_record_path(root, candidate["checkpoint"], checkpoint=True)
    best = record["candidates"][rank_selection_results(record, results)]
    if (record["ckpt_relpath"], record["global_step"]) != (best["checkpoint"], best["global_step"]):
        raise ValueError("Selected checkpoint disagrees with complete candidate ranking")
    return record, checkpoint


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify saved selection evidence without running evaluation")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--selection-record", required=True)
    parser.add_argument("--publish-best", action="store_true", help="Publish a direct best pointer after verification")
    args = parser.parse_args()
    record, checkpoint = verify_selection_record(args.experiment_dir, args.selection_record)
    if args.publish_best:
        publish_best(args.experiment_dir, record)
    print(f"Selection verified: {checkpoint}")


if __name__ == "__main__":
    main()
