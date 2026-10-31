import os
import tempfile
import traceback
from contextlib import closing
from datetime import datetime
from pathlib import Path

import torch

from dexmani_policy.common.checkpoint import load_checkpoint, restore_policy
from dexmani_policy.common.config import positive_int, require_tactile_source
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
from dexmani_policy.common.io_util import write_failure, write_json
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.evaluation.metrics import summarize_episodes
from dexmani_policy.evaluation.seeds import validate_seed_pool, validate_evaluation_seed


def create_run_directory(root, directory=None):
    if directory is None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=root))
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {directory}")
    return directory


def load_seed_pool(cfg):
    task = cfg["env_runner"]
    if task.get("eval_seeds") is not None:
        return validate_seed_pool(task["eval_seeds"])
    from dexmani_sim import DATA_DIR
    return [int(value) for value in (DATA_DIR / "eval_seeds" / f"{task['task_name']}.txt").read_text().split()]


def evaluate_policy(checkpoint, *, weights=None, inference_steps=None, action_steps=None,
                    seeds, seed=1024, device="cuda:0", video=True, output_dir=None):
    """Run one checkpoint with explicit evaluation settings and one result directory."""
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    cfg = load_checkpoint(checkpoint, mmap=True).effective_config
    weights = ("ema" if cfg["eval"]["use_ema"] else "raw") if weights is None else weights
    if weights not in {"raw", "ema"}:
        raise ValueError("weights must be raw or ema")
    inference_steps = positive_int(cfg["eval"]["inference_steps"] if inference_steps is None
                                   else inference_steps, "inference_steps")
    action_steps = positive_int(cfg["agent"]["n_action_steps"] if action_steps is None
                                else action_steps, "action_steps")
    future_steps = cfg["agent"]["horizon"] - cfg["policy_metadata"]["action_start_index"]
    if action_steps > future_steps:
        raise ValueError(f"action_steps must not exceed future_steps ({future_steps})")
    seeds = validate_seed_pool(list(seeds))
    if not seeds:
        raise ValueError("Evaluation seeds must be nonempty")
    validate_evaluation_seed(seed)
    if require_tactile_source(cfg["dataset"]["sensor_modalities"], cfg.get("tactile_source")) == "real":
        raise ValueError("Simulation evaluation requires sim-source tactile observations")
    options = dict(cfg["env_runner"])
    options.pop("eval_seeds", None)
    options.update(n_obs_steps=cfg["agent"]["n_obs_steps"],
                   sensor_modalities=cfg["dataset"]["sensor_modalities"],
                   rgb_preprocessing=rgb_preprocessing_kwargs(cfg["dataset"]))
    directory = create_run_directory(checkpoint.parent.parent / "eval_dexsim", output_dir)
    path = directory / "result_details.json"
    record = dict(status="running", checkpoint=str(checkpoint), weights=weights,
                  inference_steps=inference_steps, action_steps=action_steps, seeds=seeds,
                  seed=seed, device=str(device), video=video, runner_options=options,
                  episodes=[], statistics=None)
    write_json(path, record, overwrite=False)
    try:
        render_mode = (options.get("env_kwargs") or {}).get("render_mode")
        if render_mode == "human" and not os.environ.get("DISPLAY"):
            raise RuntimeError("Viewer requires an X11/XWayland DISPLAY")
        set_seed(seed)
        record["runtime"] = {"torch": str(torch.__version__), "cuda": torch.version.cuda}
        agent = restore_policy(checkpoint, use_ema=weights == "ema", device=device)
        record["global_step"] = agent.checkpoint_step
        record["runtime"]["parameter_dtypes"] = sorted({str(p.dtype) for p in agent.parameters()})
        from dexmani_policy.env_runner.sim_runner import SimRunner
        runner = SimRunner(**options)
        with closing(runner.run(agent, seeds=seeds, inference_steps=inference_steps,
                                action_steps=action_steps,
                                video_save_dir=directory / "videos" / runner.task_name if video else None)) as episodes:
            record["episodes"].extend(episodes)
        record["statistics"] = summarize_episodes(record["episodes"])
        record["status"] = "completed"
        write_json(path, record)
    except BaseException:
        record.update(status="failed", error=traceback.format_exc(), statistics=None)
        write_failure(path, record)
        raise
    print(f"{record['statistics']}; {path}")
    return record, path
