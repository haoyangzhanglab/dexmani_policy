import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.config import (
    load_experiment_config,
    positive_int,
    validate_action_mode,
)
from dexmani_policy.common.experiments import (
    resolve_milestone_path,
    resolve_policy_checkpoint,
)
from dexmani_policy.common.image_util import rgb_preprocessing_kwargs
from dexmani_policy.evaluation.seeds import (
    split_evaluation_seeds,
    validate_evaluation_seed,
    validate_seed_pool,
)


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "overrides", nargs="*", help="Canonical eval.*, env_runner.* and training.device dotlist"
    )
    return parser


@dataclass(frozen=True)
class EvalConfig:
    config: dict
    checkpoint: str
    use_ema: bool
    inference_steps: int
    action_steps: int
    device: str
    runner_options: dict
    seeds: list[int]
    source: dict


def parse_overrides(overrides):
    allowed = {
        "eval.seed",
        "eval.episodes",
        "eval.selection_episodes",
        "eval.seeds",
        "eval.inference_steps",
        "eval.action_steps",
        "eval.use_ema",
        "eval.video.enabled",
        "training.device",
    }
    for override in overrides:
        key, separator, _ = override.partition("=")
        if not separator or (key not in allowed and not key.startswith("env_runner.")):
            raise ValueError(f"Unsupported evaluation override: {key}")
    return OmegaConf.from_dotlist(overrides)


def load_seed_pool(cfg):
    task = cfg["env_runner"]
    name = task["task_name"]
    seeds = task.get("eval_seeds")
    if seeds is None:
        from dexmani_sim import DATA_DIR

        path = DATA_DIR / "eval_seeds" / f"{name}.txt"
        seeds = [int(value) for value in path.read_text().split()]
    return seeds


def merge_eval_config(cfg, overrides=None):
    overrides = overrides or {}
    if set(overrides.get("env_runner", {})) & {
        "n_obs_steps",
        "sensor_modalities",
        "rgb_preprocessing",
    }:
        raise ValueError("Model inputs must come from saved dataset recipes")
    return OmegaConf.to_container(OmegaConf.merge(cfg, overrides), resolve=True)


def build_eval_config(cfg, checkpoint, seeds, *, source) -> EvalConfig:
    validate_action_mode(cfg)
    options = dict(cfg["env_runner"])
    options.pop("eval_seeds", None)
    options.update(
        n_obs_steps=cfg["agent"]["n_obs_steps"],
        sensor_modalities=cfg["dataset"]["sensor_modalities"],
        rgb_preprocessing=rgb_preprocessing_kwargs(cfg["dataset"]),
    )

    evaluation = cfg["eval"]
    for name, value in (
        ("use_ema", evaluation["use_ema"]),
        ("video.enabled", evaluation["video"]["enabled"]),
    ):
        if type(value) is not bool:
            raise ValueError(f"eval.{name} must be boolean")
    steps = positive_int(evaluation["inference_steps"], "eval.inference_steps")
    action_steps = evaluation.get("action_steps")
    if action_steps is None:
        action_steps = cfg["agent"]["n_action_steps"]
    positive_int(action_steps, "eval.action_steps")
    future_steps = cfg["agent"]["horizon"] - cfg["policy_metadata"]["action_start_index"]
    if action_steps > future_steps:
        raise ValueError(f"eval.action_steps must be <= future_steps ({future_steps})")
    return EvalConfig(
        config=cfg,
        checkpoint=str(checkpoint),
        use_ema=evaluation["use_ema"],
        inference_steps=steps,
        action_steps=action_steps,
        device=cfg["training"]["device"],
        runner_options=options,
        seeds=seeds,
        source=source,
    )


def frozen_test_config(record, checkpoint, *, device=None) -> EvalConfig:
    request = EvalConfig(**record["test_request"])
    return replace(
        request,
        checkpoint=str(checkpoint),
        device=request.device if device is None else device,
        source={
            "phase": "test",
            "selection_result": record["selection_result"],
            "heldout_from_selection": True,
            "global_step": record["global_step"],
        },
    )


def load_eval_config(
    exp_dir, checkpoint="best", overrides=None, *, selection_record=None
) -> EvalConfig:
    root = Path(exp_dir).expanduser().resolve()
    overrides = overrides or {}
    if selection_record is not None:
        if set(overrides) - {"training"} or set(overrides.get("training", {})) - {"device"}:
            raise ValueError("Frozen test only accepts training.device overrides")
        path = Path(selection_record).expanduser().resolve()
        reference = str(path.relative_to(root))
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            raise ValueError("Selection has not completed successfully")
        record["selection_result"] = reference
        return frozen_test_config(
            record, resolve_milestone_path(root, record["ckpt_relpath"]),
            device=overrides.get("training", {}).get("device"),
        )

    cfg = load_experiment_config(root)
    source = {"phase": "ordinary", "heldout_from_selection": False}
    checkpoint = str(checkpoint)
    path, defaults, selection_source = resolve_policy_checkpoint(
        root, checkpoint, cfg, relative_to=root / "checkpoints",
    )
    if not path.is_relative_to(root / "checkpoints"):
        raise ValueError("Checkpoint must be inside the selected experiment/checkpoints")
    if checkpoint == "best":
        inference = {key: defaults[key] for key in ("use_ema", "inference_steps", "action_steps")}
        overrides = OmegaConf.merge({"eval": inference}, overrides)
        source.update(selection_source)
    cfg = merge_eval_config(cfg, overrides)
    evaluation = cfg["eval"]
    seeds = evaluation.get("seeds")
    if seeds is None:
        seeds, _ = split_evaluation_seeds(
            load_seed_pool(cfg), evaluation["seed"], evaluation["episodes"]
        )
    else:
        validate_seed_pool(seeds)
        if not seeds:
            raise ValueError("eval.seeds must be nonempty")
        validate_evaluation_seed(evaluation["seed"])
        pool = cfg["env_runner"].get("eval_seeds")
        if pool is not None and set(seeds) - set(validate_seed_pool(pool)):
            raise ValueError("Requested seed is outside the declared task pool")
    return build_eval_config(cfg, path, seeds, source=source)
