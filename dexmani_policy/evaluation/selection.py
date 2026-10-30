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
from dexmani_policy.evaluation.protocol import (
    build_eval_protocol,
    frozen_test_protocol,
    load_seed_pool,
    merge_eval_config,
)
from dexmani_policy.evaluation.seeds import split_evaluation_seeds


def candidate_score(candidate):
    stats = candidate["statistics"]
    steps = stats["avg_steps"]
    return (
        stats["success_rate"],
        -(steps if steps is not None else float("inf")),
        candidate["global_step"],
    )


def select_checkpoint(exp_dir, cfg, *, overrides=None, output_dir=None) -> tuple[dict, Path]:
    from dexmani_policy.evaluation.rollout import evaluate_policy

    root = Path(exp_dir).expanduser().resolve()
    cfg = merge_eval_config(
        cfg, OmegaConf.merge({"eval": {"video": {"enabled": False}}}, overrides or {})
    )
    evaluation = cfg["eval"]
    if evaluation.get("seeds") is not None:
        raise ValueError("Selection uses its predeclared test/selection split, not eval.seeds")
    test, selection = split_evaluation_seeds(
        load_seed_pool(cfg),
        evaluation["seed"],
        evaluation["episodes"],
        positive_int(evaluation["selection_episodes"], "eval.selection_episodes"),
    )

    milestones = discover_milestone_checkpoints(root)
    expected = milestone_steps(cfg["training"]["loop"]["total_train_steps"])
    actual = {m.global_step: m.pct for m in milestones}
    if actual != expected or len(milestones) != len(expected):
        raise ValueError(
            f"Selection requires milestones {expected}; found {actual}; "
            f"missing steps: {sorted(expected.keys() - actual.keys())}"
        )
    protocol = build_eval_protocol(
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
    test_protocol = replace(
        protocol, seeds=test, source={"phase": "test", "heldout_from_selection": True}
    )
    protocol = replace(
        protocol, config={**cfg, "eval": {**evaluation, "video": {"enabled": False}}}
    )
    record = {
        "status": "running",
        "selection_result": str(path.relative_to(root)),
        "selection_seeds": selection,
        "test_request": vars(test_protocol).copy(),
        "candidates": [
            {"checkpoint": str(m.path.relative_to(root)), "global_step": m.global_step}
            for m in milestones
        ],
    }
    write_json(path, record, overwrite=False)

    try:
        for candidate in record["candidates"]:
            result, result_path = evaluate_policy(
                replace(
                    protocol,
                    checkpoint=str(root / candidate["checkpoint"]),
                    source={**protocol.source, "global_step": candidate["global_step"]},
                ),
                output_root=directory / "candidates",
            )
            candidate.update(
                result=str(result_path.relative_to(root)), statistics=result["statistics"]
            )

        candidate = max(record["candidates"], key=candidate_score)
        checkpoint = root / candidate["checkpoint"]
        record["test_request"]["checkpoint"] = str(checkpoint)
        record.update(
            status="completed",
            ckpt_relpath=candidate["checkpoint"],
            global_step=candidate["global_step"],
            inference={
                key: getattr(protocol, key)
                for key in ("use_ema", "inference_steps", "action_steps")
            },
        )
        write_json(path, record)
        write_json(
            root / "best_ckpt.json",
            {
                key: record[key]
                for key in ("ckpt_relpath", "global_step", "inference", "selection_result")
            },
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
        record, checkpoint = select_checkpoint(root, cfg, overrides=overrides)
        protocol = frozen_test_protocol(record, checkpoint)
        _, result_path = evaluate_policy(protocol, output_root=root / "eval_dexsim")
        return result_path
