"""Fixed initial checkpoint evaluation followed by one optional exact-tie batch."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf
from termcolor import cprint

from dexmani_policy.env_runner.common.metrics import collect_episode_details, mean_success_steps
from dexmani_policy.env_runner.common.seeds import (
    plan_selection_seeds,
    plan_size,
    resolve_eval_seed,
)
from dexmani_policy.env_runner.evaluation import (
    artifact_reference,
    atomic_json,
    build_eval_runner,
    create_run_directory,
    evaluation_snapshot,
    get_eval_param,
    iter_leaf_env_runners,
    run_eval_plan,
    task_seed_pools,
    validate_inference_steps,
)
from dexmani_policy.training.common.checkpoint import (
    MilestoneCheckpoint,
    discover_milestone_checkpoints,
    restore_policy_agent,
)
from dexmani_policy.utils.config import resolve_action_steps
from dexmani_policy.utils.runtime import set_seed


@dataclass
class CheckpointEvaluation:
    checkpoint: MilestoneCheckpoint
    episodes: list[dict] = field(default_factory=list)

    @property
    def success_rate(self):
        return sum(d["success"] for d in self.episodes) / len(self.episodes)

    @property
    def avg_steps(self):
        return mean_success_steps(self.episodes)

    @property
    def rank(self):
        avg_steps = self.avg_steps
        return (
            self.success_rate,
            -(avg_steps if avg_steps is not None else float("inf")),
            self.checkpoint.global_step,
        )


def select_best_checkpoint(
    exp_dir,
    cfg,
    *,
    initial_episodes=None,
    batch_size=None,
    max_episodes=None,
    inference_steps=None,
    action_steps=None,
    use_ema=None,
    eval_seed=None,
    record_video=False,
    output_dir=None,
):
    initial_episodes = get_eval_param(
        cfg, "initial_episodes", "select_best", override=initial_episodes, default=25
    )
    batch_size = get_eval_param(cfg, "batch_size", "select_best", override=batch_size, default=5)
    max_episodes = get_eval_param(
        cfg, "max_episodes", "select_best", override=max_episodes, default=100
    )
    inference_steps = get_eval_param(
        cfg, "inference_steps", "select_best", override=inference_steps, default=10
    )
    use_ema = get_eval_param(cfg, "use_ema", "select_best", override=use_ema, default=True)
    episodes = get_eval_param(cfg, "episodes", "offline", default=100)
    validate_inference_steps([inference_steps])
    if type(use_ema) is not bool:
        raise ValueError("use_ema must be boolean")
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(cfg), resolve=True))
    action_steps = resolve_action_steps(
        cfg.agent, get_eval_param(cfg, "action_steps", "select_best", override=action_steps)
    )
    OmegaConf.update(cfg, "eval.select_best.action_steps", action_steps)
    exp_dir = Path(exp_dir).resolve()
    root = exp_dir / "eval_ckpt_selector"
    if output_dir is not None and not Path(output_dir).expanduser().resolve().is_relative_to(root):
        raise ValueError("Selection output must be inside experiment/eval_ckpt_selector")
    run_dir = create_run_directory(root, output_dir)
    result_path = run_dir / "selection_result.json"
    record = {
        "selection_id": run_dir.name,
        "status": "running",
        "inference": {
            "use_ema": use_ema,
            "inference_steps": inference_steps,
            "action_steps": action_steps,
            "policy_seed_mode": "episode_seed",
        },
        "stages": [],
    }
    try:
        cfg.experiment_dir = str(exp_dir)
        seed = resolve_eval_seed(cfg, eval_seed)
        OmegaConf.update(cfg, "eval.seed", seed)
        OmegaConf.update(cfg, "eval.offline.episodes", episodes)
        set_seed(seed)
        milestones = discover_milestone_checkpoints(exp_dir)
        runner = build_eval_runner(cfg)
        for leaf in iter_leaf_env_runners(runner):
            leaf.record_video = record_video
        test, initial, extra = plan_selection_seeds(
            task_seed_pools(runner),
            seed,
            episodes=episodes,
            initial_episodes=initial_episodes,
            batch_size=batch_size,
            max_episodes=max_episodes,
        )
        record["evaluation"] = evaluation_snapshot(
            cfg,
            runner,
            action_steps=action_steps,
            shuffle_seed=seed,
            test_task_seeds=test,
            initial_episodes=initial_episodes,
            batch_size=batch_size,
            max_episodes=max_episodes,
            initial_task_seeds=initial,
            possible_tie_task_seeds=extra,
            checkpoints=[artifact_reference(m.path, exp_dir) for m in milestones],
        )
        atomic_json(result_path, record)

        def evaluate(candidate, plan, phase):
            checkpoint = candidate.checkpoint
            video = run_dir / "videos" / checkpoint.path.stem / phase if record_video else None
            stage = {
                "checkpoint": artifact_reference(checkpoint.path, exp_dir),
                "phase": phase,
                "requested_task_seeds": plan,
                "status": "running",
            }
            record["stages"].append(stage)
            atomic_json(result_path, record)
            agent = restore_policy_agent(
                cfg, checkpoint.path, use_ema=use_ema, device=cfg.training.device
            )
            if agent.checkpoint_step != checkpoint.global_step:
                raise ValueError("Checkpoint filename/global_step disagrees with saved state")
            result = run_eval_plan(
                runner,
                agent,
                plan,
                inference_steps=inference_steps,
                action_steps=action_steps,
                video_save_dir=video,
            )
            details = collect_episode_details(result)
            stage.update(status="completed", episode_details=details)
            candidate.episodes.extend(details)
            atomic_json(result_path, record)

        candidates = [CheckpointEvaluation(checkpoint) for checkpoint in milestones]
        for candidate in candidates:
            evaluate(candidate, initial, "initial")
        best_rate = max(candidate.success_rate for candidate in candidates)
        tied = [candidate for candidate in candidates if candidate.success_rate == best_rate]
        if len(tied) > 1 and plan_size(extra):
            for candidate in tied:
                evaluate(candidate, extra, "tie_break")
        best = max(tied, key=lambda candidate: candidate.rank)
        record.update(
            status="success",
            ckpt_relpath=artifact_reference(best.checkpoint.path, exp_dir),
            global_step=best.checkpoint.global_step,
            pct=best.checkpoint.pct,
        )
        atomic_json(result_path, record)
        atomic_json(
            exp_dir / "best_ckpt.json",
            {
                "selection_result": artifact_reference(result_path, exp_dir),
            },
        )
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            atomic_json(result_path, record)
        except OSError as save_error:
            print(f"Could not save failed selection: {save_error}")
        raise

    for candidate in candidates:
        cprint(
            f"{candidate.checkpoint.label}: {candidate.success_rate:.1%}, "
            f"episodes={len(candidate.episodes)}, avg_steps={candidate.avg_steps}",
            "cyan",
        )
    cprint(f"Selected {best.checkpoint.label}; record: {result_path}", "green")
    return result_path
