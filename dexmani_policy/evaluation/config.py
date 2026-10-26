"""Resolve evaluation inputs once; execution receives concrete tasks and seeds."""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.config import (
    load_experiment_config,
    positive_int,
    validate_action_key_consistency,
)
from dexmani_policy.common.experiments import (
    resolve_checkpoint_path,
    resolve_milestone_path,
)
from dexmani_policy.common.rgb_preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.env_runner.common.seeds import (
    split_evaluation_seeds,
    validate_evaluation_seed,
    validate_seed_pool,
    validate_task_seeds,
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
class EvaluationRequest:
    config: dict
    checkpoint: str
    use_ema: bool
    inference_steps: int
    action_steps: int
    device: str
    tasks: list[dict]
    task_seeds: dict[str, list[int]]
    source: dict


def parse_evaluation_overrides(overrides):
    allowed = {
        "eval.seed",
        "eval.episodes",
        "eval.selection_episodes",
        "eval.task_seeds",
        "eval.inference_steps",
        "eval.action_steps",
        "eval.use_ema",
        "eval.video.enabled",
        "eval.video.fps",
        "training.device",
    }
    for override in overrides:
        key, separator, _ = override.partition("=")
        if not separator or (key not in allowed and not key.startswith("env_runner.")):
            raise ValueError(f"Unsupported evaluation override: {key}")
    return OmegaConf.from_dotlist(overrides)


def task_seed_pools(cfg):
    task = cfg["env_runner"]
    name = task["task_name"]
    seeds = task.get("eval_seeds")
    if seeds is None:
        from dexmani_sim import DATA_DIR

        path = DATA_DIR / "eval_seeds" / f"{name}.txt"
        seeds = [int(value) for value in path.read_text().split()]
    return {name: seeds}


def prepare_evaluation_config(cfg, overrides=None):
    overrides = overrides or {}
    if set(overrides.get("env_runner", {})) & {
        "n_obs_steps",
        "sensor_modalities",
        "rgb_preprocessing",
    }:
        raise ValueError("Model inputs must come from saved dataset recipes")
    return OmegaConf.to_container(OmegaConf.merge(cfg, overrides), resolve=True)


def make_request(cfg, checkpoint, seeds, *, source) -> EvaluationRequest:
    from dexmani_policy.env_runner.sim_runner import SimRunner

    validate_action_key_consistency(cfg)
    options = dict(cfg["env_runner"])
    options.pop("eval_seeds", None)
    options.update(
        n_obs_steps=cfg["agent"]["n_obs_steps"],
        sensor_modalities=cfg["dataset"]["sensor_modalities"],
        rgb_preprocessing=rgb_preprocessing_kwargs(cfg["dataset"]),
        env_kwargs=SimRunner.expand_env_kwargs(options.get("env_kwargs", {})),
        record_video=cfg["eval"]["video"]["enabled"],
    )
    if type(options["record_video"]) is not bool:
        raise ValueError("eval.video.enabled must be boolean")
    if "fps" in cfg["eval"]["video"]:
        options["env_video_fps"] = cfg["eval"]["video"]["fps"]

    use_ema = cfg["eval"]["use_ema"]
    if type(use_ema) is not bool:
        raise ValueError("eval.use_ema must be boolean")
    steps = positive_int(cfg["eval"]["inference_steps"], "eval.inference_steps")
    action_steps = cfg["eval"].get("action_steps")
    action_steps = positive_int(
        cfg["agent"]["n_action_steps"] if action_steps is None else action_steps,
        "eval.action_steps",
    )
    future_steps = cfg["agent"]["horizon"] - cfg["policy_metadata"]["action_start_index"]
    if action_steps > future_steps:
        raise ValueError(f"eval.action_steps must be <= future_steps ({future_steps})")
    return EvaluationRequest(
        config=cfg,
        checkpoint=str(checkpoint),
        use_ema=use_ema,
        inference_steps=steps,
        action_steps=action_steps,
        device=cfg["training"]["device"],
        tasks=[options],
        task_seeds=seeds,
        source=source,
    )


def frozen_test_request(record, checkpoint, *, device=None) -> EvaluationRequest:
    request = EvaluationRequest(**record["test_request"])
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


def resolve_evaluation_request(
    exp_dir, checkpoint="best", overrides=None, *, selection_record=None
) -> EvaluationRequest:
    root = Path(exp_dir).expanduser().resolve()
    if selection_record is not None:
        override = overrides or {}
        if set(override) - {"training"} or set(override.get("training", {})) - {"device"}:
            raise ValueError(
                "Frozen test permits only device/output changes; use ordinary evaluation for research overrides"
            )
        path = Path(selection_record).expanduser().resolve()
        reference = str(path.relative_to(root))
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            raise ValueError("Selection has not completed successfully")
        record["selection_result"] = reference
        return frozen_test_request(
            record, resolve_milestone_path(root, record["ckpt_relpath"]),
            device=override.get("training", {}).get("device"),
        )

    cfg = load_experiment_config(root)
    source = {"phase": "ordinary", "heldout_from_selection": False}
    if checkpoint == "best":
        selected = json.loads((root / "best_ckpt.json").read_text())
        path = resolve_milestone_path(root, selected["ckpt_relpath"])
        inference = {
            key: selected["inference"][key] for key in ("use_ema", "inference_steps", "action_steps")
        }
        overrides = OmegaConf.merge({"eval": inference}, overrides or {})
        source["selection_result"] = selected["selection_result"]
        source["global_step"] = selected["global_step"]
    else:
        path = resolve_checkpoint_path(root, checkpoint)
    cfg = prepare_evaluation_config(cfg, overrides)
    seeds = cfg["eval"].get("task_seeds")
    if seeds is None:
        seeds, _ = split_evaluation_seeds(
            task_seed_pools(cfg), cfg["eval"]["seed"], cfg["eval"]["episodes"]
        )
    else:
        task = cfg["env_runner"]["task_name"]
        validate_task_seeds(seeds, [task])
        validate_evaluation_seed(cfg["eval"]["seed"])
        pool = cfg["env_runner"].get("eval_seeds")
        if pool is not None and set(seeds[task]) - set(validate_seed_pool(task, pool)):
            raise ValueError("Requested seed is outside the declared task pool")
    return make_request(cfg, path, seeds, source=source)
