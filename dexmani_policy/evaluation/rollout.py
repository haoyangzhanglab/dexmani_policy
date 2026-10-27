"""Run a resolved evaluation request and save its results."""

import os
import traceback
from contextlib import closing
from pathlib import Path

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.deployment.runtime import restore_policy_agent
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation.artifacts import (
    create_run_directory,
    runtime_provenance,
    write_failure,
    write_json,
)
from dexmani_policy.evaluation.config import EvaluationRequest
from dexmani_policy.evaluation.metrics import compute_evaluation_statistics, validate_episodes


def evaluate_policy(request: EvaluationRequest, *, output_root, output_dir=None) -> tuple[dict, Path]:
    directory = create_run_directory(output_root, output_dir)
    path = directory / "result_details.json"
    record: dict = {"request": vars(request).copy(), "status": "running", "episodes": [], "statistics": None}
    write_json(path, record, overwrite=False)

    try:
        for task in request.tasks:
            render_mode = (task.get("env_kwargs") or {}).get("render_mode")
            if render_mode == "human" and not os.environ.get("DISPLAY"):
                raise RuntimeError("Viewer requires an X11/XWayland DISPLAY")

        set_seed(request.config["eval"]["seed"])
        record["runtime"] = runtime_provenance()
        agent = restore_policy_agent(
            request.config, request.checkpoint, use_ema=request.use_ema, device=request.device
        )
        if "global_step" in request.source and request.source["global_step"] != agent.checkpoint_step:
            raise ValueError("Selected global_step differs from checkpoint state")
        record["global_step"] = agent.checkpoint_step
        record["runtime"]["parameter_dtypes"] = sorted({str(p.dtype) for p in agent.parameters()})
        record["runtime"]["device"] = str(next(agent.parameters()).device)
        if str(request.device).startswith("cuda"):
            import torch

            record["runtime"]["device_name"] = torch.cuda.get_device_name(request.device)

        for options in request.tasks:
            runner = SimRunner(**options)
            with closing(
                runner.run(
                    agent,
                    seeds=request.task_seeds[runner.task_name],
                    inference_steps=request.inference_steps,
                    action_steps=request.action_steps,
                    video_save_dir=(
                        directory / "videos" / runner.task_name
                        if request.config["eval"]["video"]["enabled"] else None
                    ),
                )
            ) as episodes:
                record["episodes"].extend(episodes)

        validate_episodes(record["episodes"], request.task_seeds)
        record["statistics"] = compute_evaluation_statistics(record["episodes"])
        record["status"] = "completed"
        write_json(path, record)
    except BaseException:
        record.update(status="failed", error=traceback.format_exc(), statistics=None)
        write_failure(path, record)
        raise

    stats = record["statistics"]
    print(
        f"{stats['n_success']}/{stats['n_valid_episodes']}; micro={stats['micro_success_rate']:.1%}, "
        f"macro={stats['macro_success_rate']:.1%}, avg_steps={stats['avg_steps']}; {path}"
    )
    return record, path
