"""Execute a resolved request; no selection logic or configuration precedence."""

import sys
import traceback
from contextlib import closing

from dexmani_policy.common.checkpoint import restore_policy_agent
from dexmani_policy.common.runtime import set_seed
from dexmani_policy.common.experiments import file_sha256
from dexmani_policy.env_runner.common.metrics import (
    compute_evaluation_statistics,
    validate_episodes,
)
from dexmani_policy.env_runner.sim_runner import SimRunner
from dexmani_policy.evaluation.artifacts import (
    create_run_directory,
    runtime_provenance,
    validate_runtime,
    training_source_evidence,
    write_failure,
    write_json,
)


def evaluate_policy(request, *, output_root, output_dir=None):
    directory = create_run_directory(output_root, output_dir)
    path = directory / "result_details.json"
    record = {"request": request.to_dict(), "status": "running", "episodes": [], "statistics": None}
    record['training_source_evidence'] = training_source_evidence(
        request.config, {request.source['phase']: request.task_seeds}
    )
    write_json(path, record, overwrite=False)
    agent = None
    try:
        try:
            for task in request.tasks:
                if task["record_video"]:
                    from imageio_ffmpeg import get_ffmpeg_exe

                    get_ffmpeg_exe()
                if task["render_mode"] == "human":
                    import os

                    if not os.environ.get("DISPLAY"):
                        raise RuntimeError("Viewer requires an X11/XWayland DISPLAY")
            # A fresh load is deterministic; each episode resets its own sampling stream.
            set_seed(request.numerics["startup_seed"])
            record["runtime"] = runtime_provenance()
            record["checkpoint_sha256"] = file_sha256(request.checkpoint)
            expected_hash = request.source.get("checkpoint_sha256")
            if expected_hash is not None and expected_hash != record["checkpoint_sha256"]:
                raise ValueError("Checkpoint content changed since selection")
            agent = restore_policy_agent(
                request.config, request.checkpoint, use_ema=request.use_ema, device=request.device
            )
            if (
                "global_step" in request.source
                and request.source["global_step"] != agent.checkpoint_step
            ):
                raise ValueError("Selected global_step differs from checkpoint state")
            record["global_step"] = agent.checkpoint_step
            record["runtime"]["parameter_dtypes"] = sorted({str(p.dtype) for p in agent.parameters()})
            record["runtime"]["device"] = str(next(agent.parameters()).device)
            if str(request.device).startswith("cuda"):
                import torch

                record["runtime"]["device_name"] = torch.cuda.get_device_name(request.device)
            if request.source.get("heldout_from_selection"):
                validate_runtime(request.source["expected_runtime"], record["runtime"])
            write_json(path, record)
            for options in request.tasks:
                options = dict(options)
                rgb_preprocessing = options.pop("rgb_preprocessing")
                runner = SimRunner(**options)
                runner.rgb_preprocessing = rgb_preprocessing
                with closing(
                    runner.run(
                        agent,
                        seeds=request.task_seeds[runner.task_name],
                        inference_steps=request.inference_steps,
                        action_steps=request.action_steps,
                        video_save_dir=(
                            directory / "videos" / runner.task_name if runner.record_video else None
                        ),
                    )
                ) as episodes:
                    for episode in episodes:
                        record["episodes"].append(episode)
                        write_json(path, record)
            validate_episodes(record["episodes"], request.task_seeds)
        finally:
            primary = sys.exc_info()[1]
            if agent is not None:
                try:
                    agent.to("cpu")
                except BaseException:
                    record["cleanup_error"] = traceback.format_exc()
                    if primary is None:
                        raise
                    traceback.print_exc()
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
    return path
