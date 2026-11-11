import os
import tempfile
from contextlib import closing
from datetime import datetime
from pathlib import Path

from dexmani_policy.common.checkpoint import restore_policy, load_checkpoint
from dexmani_policy.common.config import positive_int
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
from dexmani_policy.common.inference import resolve_inference_settings
from dexmani_policy.common.io_util import write_json
from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.evaluation.metrics import summarize_episodes
from dexmani_policy.evaluation.seeds import (
    load_seed_pool, split_evaluation_seeds, validate_seed_pool, validate_evaluation_seed,
)


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
                    seeds=None, episodes=100, seed=1024, device="cuda:0", video=False, output_dir=None):
    """Run one checkpoint with explicit or deterministically sampled evaluation seeds."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    loaded = load_checkpoint(checkpoint)
    cfg = loaded.effective_config
    weights, inference_steps = resolve_inference_settings(cfg, weights=weights, inference_steps=inference_steps)
    state_dict = loaded.ema_model_state if weights == "ema" else loaded.model_state
    global_step = loaded.global_step
    del loaded
    if seeds is None:
        seeds, _ = split_evaluation_seeds(load_seed_pool(cfg["env_runner"]["task_name"]), seed, episodes)
    else:
        seeds = validate_seed_pool(list(seeds))
        if not seeds:
            raise ValueError("Evaluation seeds must be nonempty")
        validate_evaluation_seed(seed)
    options = dict(cfg["env_runner"])
    directory = create_run_directory(checkpoint.parent.parent / "eval", output_dir)
    path = directory / "result_details.json"
    render_mode = (options.get("env_kwargs") or {}).get("render_mode")
    if render_mode == "human" and not os.environ.get("DISPLAY"):
        raise RuntimeError("Viewer requires an X11/XWayland DISPLAY")
    set_seed(seed)
    agent = restore_policy(cfg, state_dict, device)
    del state_dict
    action_steps = positive_int(agent.n_action_steps if action_steps is None else action_steps, "action_steps")
    if action_steps > agent.future_steps:
        raise ValueError(f"action_steps must not exceed future_steps ({agent.future_steps})")
    options.update(n_obs_steps=agent.n_obs_steps,
                   sensor_modalities=[key for key in cfg["normalization_dims"] if key != "action"],
                   rgb_preprocessing=rgb_preprocessing_kwargs(cfg["dataset"]))
    from dexmani_policy.env_runner.sim_runner import SimRunner
    runner = SimRunner(**options)
    with closing(runner.run(agent, seeds=seeds, inference_steps=inference_steps,
                            action_steps=action_steps,
                            video_save_dir=directory / "videos" / runner.task_name if video else None)) as results:
        episode_results = list(results)
    record = dict(checkpoint=os.path.relpath(checkpoint, path.parent), global_step=global_step, weights=weights,
                  inference_steps=inference_steps, action_steps=action_steps, seeds=seeds,
                  seed=seed, device=str(device), video=video,
                  episodes=episode_results, statistics=summarize_episodes(episode_results))
    if hasattr(agent, "force_predictor"):
        record["force_inference_steps"] = agent.force_predictor.num_inference_steps
    write_json(path, record)
    print(f"{record['statistics']}; {path}")
    return record, path
