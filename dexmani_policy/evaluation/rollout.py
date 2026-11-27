import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from hydra.utils import get_class

from dexmani_policy.common.checkpoint import load_checkpoint
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
from dexmani_policy.common.io_util import cleanup_on_exit, write_json
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.deployment import restore_policy, select_inference_weights
from dexmani_policy.env_runner.dexjoco_runner import DexJoCoRunner
from dexmani_policy.env_runner.sim_runner import SimRunner, resolve_action_steps
from dexmani_policy.evaluation.metrics import summarize_episodes
from dexmani_policy.evaluation.seeds import (
    split_evaluation_seeds, validate_seed_pool, validate_evaluation_seed,
)


def resolve_rollout_options(cfg, *, action_steps, skip_intermediate_observations, video):
    runner_class = get_class(cfg["env_runner"]["_target_"])
    action_key = cfg["action_key"]
    if runner_class is SimRunner and action_key == "action_rotvec":
        raise ValueError("The built-in SimRunner does not support action_rotvec; use a compatible custom runner")
    if issubclass(runner_class, DexJoCoRunner) and action_key != "action_rotvec":
        raise ValueError(f"{runner_class.__name__} does not support {action_key}; use action_rotvec")
    if skip_intermediate_observations and (runner_class is not SimRunner or video):
        raise ValueError("--skip-intermediate-observations requires the built-in SimRunner and video=False")
    agent = cfg["agent"]
    steps = resolve_action_steps(
        action_steps, n_action_steps=agent["n_action_steps"], future_steps=agent["horizon"] - agent["n_obs_steps"] + 1,
    )
    return runner_class, steps


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


def evaluate_policy(checkpoint, *, weights=None, inference_steps=None, action_steps=None,
                    seeds=None, episodes=100, seed=1024, device="cuda:0", video=False,
                    output_dir=None, skip_intermediate_observations=False):
    """Run one checkpoint with explicit or deterministically sampled evaluation seeds."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    loaded = load_checkpoint(checkpoint)
    cfg = loaded.effective_config
    weights, inference_steps, state_dict = select_inference_weights(
        loaded, weights=weights, inference_steps=inference_steps,
    )
    global_step = loaded.global_step
    del loaded
    runner_class, action_steps = resolve_rollout_options(
        cfg, action_steps=action_steps, skip_intermediate_observations=skip_intermediate_observations, video=video,
    )
    if seeds is None:
        seeds, _ = split_evaluation_seeds(runner_class.load_seed_pool(cfg["env_runner"]["task_name"]), seed, episodes)
    else:
        seeds = validate_seed_pool(list(seeds))
        if not seeds:
            raise ValueError("Evaluation seeds must be nonempty")
        validate_evaluation_seed(seed)
    directory = create_run_directory(checkpoint.parent.parent / "eval", output_dir)
    path = directory / "result_details.json"
    options = {key: value for key, value in cfg["env_runner"].items() if key != "_target_"}
    render_mode = (options.get("env_kwargs") or {}).get("render_mode")
    if render_mode == "human" and not os.environ.get("DISPLAY"):
        raise RuntimeError("Viewer requires an X11/XWayland DISPLAY")
    set_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("highest")
    agent = restore_policy(cfg, state_dict, device)
    del state_dict
    fields = [key for key in agent.normalization_dims if key != "action"]
    options.update(
        control_dt_s=cfg["control_dt_s"],
        n_obs_steps=agent.n_obs_steps,
        sensor_modalities=fields,
        rgb_preprocessing=rgb_preprocessing_kwargs(cfg["dataset"]) if "rgb" in fields else {},
    )
    if issubclass(runner_class, DexJoCoRunner):
        options["data_processing"] = cfg["data_processing"]
    elif runner_class is SimRunner:
        options["skip_intermediate_observations"] = skip_intermediate_observations
    runner = runner_class(**options)
    record = dict(checkpoint=os.path.relpath(checkpoint, path.parent), global_step=global_step, weights=weights,
                  control_dt_s=cfg["control_dt_s"],
                  inference_steps=inference_steps, action_steps=action_steps, seeds=seeds,
                  seed=seed, device=str(device), video=video, skip_intermediate_observations=skip_intermediate_observations,
                  episodes=[], statistics=None, completed=False)
    if hasattr(agent, "force_predictor"):
        record["force_inference_steps"] = agent.force_predictor.num_inference_steps
    write_json(path, record)
    video_directory = directory / "videos" / runner.task_name if video else None
    results = runner.run(agent, seeds=seeds, inference_steps=inference_steps,
                         action_steps=action_steps, video_save_dir=video_directory)
    with cleanup_on_exit(results.close):
        for episode in results:
            video_path = episode.get("video")
            if video_path is not None:
                episode["video"] = os.path.relpath(video_path, directory)
            record["episodes"].append(episode)
            record["statistics"] = summarize_episodes(record["episodes"])
            write_json(path, record)
    record["completed"] = True
    write_json(path, record)
    print(f"{record['statistics']}; {path}")
    return record, path
