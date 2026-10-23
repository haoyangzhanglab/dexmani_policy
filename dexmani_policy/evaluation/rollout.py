"""Checkpoint evaluation on explicit physical seed plans."""

from pathlib import Path

import hydra
from termcolor import cprint

from dexmani_policy.common.checkpoint import restore_policy_agent
from dexmani_policy.common.config import resolve_action_steps, validate_action_key_consistency
from dexmani_policy.common.rgb_preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.common.runtime import atomic_path, set_seed
from dexmani_policy.env_runner.common.metrics import (
    collect_episode_details,
    compute_evaluation_statistics,
    mean_success_steps,
)
from dexmani_policy.env_runner.common.seeds import episodes_per_task, plan_selection_seeds
from dexmani_policy.evaluation.artifacts import (
    artifact_reference,
    create_run_directory,
    save_evaluation_snapshot,
    validate_selection_evidence,
    write_json,
)
from dexmani_policy.evaluation.config import (
    EvaluationRequest,
    resolve_evaluation_parameter,
    resolve_evaluation_request,
)


def build_evaluation_runner(cfg):
    """Build the evaluation environment with the saved model input recipe."""
    validate_action_key_consistency({"action_key": cfg.action_key, "env_runner": cfg.env_runner})
    env_runner = hydra.utils.instantiate(cfg.env_runner)
    dataset = cfg.dataset
    tasks = dataset.get("task_names", ())
    children = dict(zip(tasks, dataset.get("datasets", ()), strict=True))
    texts = dataset.get("task_texts")
    texts = dict(zip(tasks, tasks if texts is None else texts, strict=True))
    fps = resolve_evaluation_parameter(cfg, "fps", "video")
    for runner in env_runner.task_runners.values():
        inputs = children[runner.task_name] if children else dataset
        runner.n_obs_steps = cfg.agent.n_obs_steps
        runner.sensor_modalities = list(inputs["sensor_modalities"])
        runner.rgb_preprocessing = rgb_preprocessing_kwargs(inputs)
        if fps is not None:
            runner.env_video_fps = fps
        if children:
            runner.task_text = texts[runner.task_name]
    return env_runner


def task_seed_pools(runner):
    return {task: list(child.get_seed_list()) for task, child in runner.task_runners.items()}


def evaluate_policy(
    request: EvaluationRequest,
    runner,
    plan,
    *,
    output_root,
    output_dir=None,
    record_video=False,
):
    """Load once and evaluate the same physical episodes at each requested NFE."""
    cfg = request.config
    selection = request.selection
    heldout = request.section == "offline" and selection is not None
    seed = cfg.eval.seed
    set_seed(seed)
    for leaf in runner.task_runners.values():
        leaf.record_video = record_video
    agent = restore_policy_agent(
        cfg, request.checkpoint, use_ema=request.use_ema, device=cfg.training.device
    )
    action_steps = resolve_action_steps(agent, request.action_steps)
    if selection is not None and agent.checkpoint_step != selection["global_step"]:
        raise ValueError("Selection global_step disagrees with checkpoint state")
    directory = create_run_directory(output_root, output_dir)
    checkpoint_reference = artifact_reference(request.checkpoint, cfg.experiment_dir)
    selection_inference = selection["inference"] if selection is not None else None
    reference = save_evaluation_snapshot(
        directory,
        cfg,
        runner.task_runners.values(),
        checkpoint=checkpoint_reference,
        global_step=agent.checkpoint_step,
        use_ema=request.use_ema,
        inference_steps_list=request.inference_steps,
        action_steps=action_steps,
        selection_inference=selection_inference,
        effective_episodes=episodes_per_task(plan),
        shuffle_seed=seed,
        policy_seed_mode="episode_seed",
        task_seeds=plan,
        selection_id=selection["selection_id"] if selection is not None else None,
        selection_result=selection["selection_result"] if selection is not None else None,
        heldout_from_selection=heldout,
    )
    results = []
    for nfe in request.inference_steps:
        output = (
            directory / f"inference_steps{nfe}" if len(request.inference_steps) > 1 else directory
        )
        output.mkdir(parents=True, exist_ok=True)
        result = runner.run(
            agent,
            task_seeds=plan,
            inference_steps=nfe,
            action_steps=action_steps,
            video_save_dir=output if record_video else None,
        )
        details = collect_episode_details(result)
        stats = compute_evaluation_statistics(result)
        inference = dict(use_ema=request.use_ema, inference_steps=nfe, action_steps=action_steps)
        info = {
            "checkpoint": checkpoint_reference,
            "global_step": agent.checkpoint_step,
            "eval_config": reference,
            **inference,
            "selection_inference": selection_inference,
            "matches_selection_inference": (
                all(selection_inference[key] == value for key, value in inference.items())
                if selection_inference is not None
                else None
            ),
            "evaluation_seeds": plan,
            "heldout_from_selection": heldout,
            "episode_details": details,
            "statistics": stats,
            "success_rate": stats["micro_success_rate"],
            "avg_steps": mean_success_steps(details),
        }
        write_json(output / "result_details.json", info, overwrite=False)
        with atomic_path(output / "result.txt", overwrite=False) as temporary:
            temporary.write_text(f"{info['success_rate']}\n")
        results.append(info)
        cprint(
            f"NFE={nfe}: {stats['n_success']}/{stats['n_valid_episodes']} "
            f"({info['success_rate']:.1%}), avg_steps={info['avg_steps']}",
            "cyan",
        )
    if len(results) > 1:
        write_json(
            directory / "eval_summary.json",
            [
                {
                    key: result[key]
                    for key in (
                        "inference_steps",
                        "action_steps",
                        "matches_selection_inference",
                        "success_rate",
                        "avg_steps",
                        "statistics",
                    )
                }
                for result in results
            ],
            overwrite=False,
        )
    cprint(f"Results saved to: {directory}", "cyan")
    return results


def evaluate_checkpoint(
    exp_dir,
    cfg,
    *,
    checkpoint="best",
    episodes=None,
    inference_steps_list=None,
    action_steps=None,
    use_ema=None,
    overrides=None,
    selection_record=None,
    output_dir=None,
    record_video=False,
    eval_seed=None,
):
    request = resolve_evaluation_request(
        cfg,
        exp_dir,
        checkpoint,
        overrides,
        use_ema=use_ema,
        inference_steps_list=inference_steps_list,
        action_steps=action_steps,
        selection_record=selection_record,
        eval_seed=eval_seed,
        episodes=episodes,
    )
    cfg = request.config
    runner = build_evaluation_runner(cfg)
    pools = task_seed_pools(runner)
    if request.selection is not None:
        plan = validate_selection_evidence(request.selection, pools, cfg.eval.offline.episodes)
    else:
        plan, _, _ = plan_selection_seeds(
            pools,
            cfg.eval.seed,
            episodes=cfg.eval.offline.episodes,
            initial_episodes=resolve_evaluation_parameter(
                cfg, "initial_episodes", "select_best", default=25
            ),
            batch_size=resolve_evaluation_parameter(cfg, "batch_size", "select_best", default=5),
            max_episodes=resolve_evaluation_parameter(
                cfg, "max_episodes", "select_best", default=100
            ),
        )
    if record_video is None:
        record_video = resolve_evaluation_parameter(cfg, "enabled", "video", default=True)
    return evaluate_policy(
        request,
        runner,
        plan,
        output_root=Path(exp_dir) / "eval_dexsim",
        output_dir=output_dir,
        record_video=record_video,
    )
