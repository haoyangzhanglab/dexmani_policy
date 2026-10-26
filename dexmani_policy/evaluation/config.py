"""Resolve evaluation inputs once; execution receives concrete tasks and seeds."""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from omegaconf import OmegaConf

from dexmani_policy.common.config import positive_int, validate_action_key_consistency
from dexmani_policy.common.experiments import (
    resolve_best_checkpoint,
    resolve_checkpoint_path,
)
from dexmani_policy.common.rgb_preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.env_runner.common.seeds import (
    split_evaluation_seeds,
    validate_evaluation_seed,
    validate_seed_pool,
    validate_task_seeds,
)
from dexmani_policy.evaluation.artifacts import verify_selection_record


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-dir")
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
    numerics: dict
    source: dict

    def to_dict(self):
        return asdict(self)


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


def resolve_tasks(cfg):
    validate_action_key_consistency(cfg)
    runner = dict(cfg["env_runner"])
    task_configs = runner.pop("task_configs", None)
    configs = [runner] if task_configs is None else [{**runner, **task} for task in task_configs]
    dataset = cfg["dataset"]
    names = dataset.get("task_names", ())
    children = dict(zip(names, dataset.get("datasets", ()), strict=True))
    texts = dataset.get("task_texts")
    texts = dict(zip(names, names if texts is None else texts, strict=True))
    tasks = []
    from dexmani_policy.env_runner.sim_runner import SimRunner

    for options in configs:
        name = options["task_name"]
        inputs = children[name] if children else dataset
        options.update(
            n_obs_steps=cfg["agent"]["n_obs_steps"],
            sensor_modalities=list(inputs["sensor_modalities"]),
            task_text=texts[name] if children else options.get("task_text"),
            rgb_preprocessing=rgb_preprocessing_kwargs(inputs),
        )
        options["env_kwargs"] = SimRunner.expand_env_kwargs(options.get("env_kwargs", {}))
        options["record_video"] = cfg["eval"]["video"]["enabled"]
        if type(options["record_video"]) is not bool:
            raise ValueError("eval.video.enabled must be boolean")
        options.setdefault("render_mode", "rgb_array")
        options.setdefault("clear_cache_freq", 25)
        options.setdefault("viewer_resolution", None)
        options["env_video_fps"] = cfg["eval"]["video"].get("fps", options.get("env_video_fps"))
        import inspect

        inspect.signature(SimRunner).bind(
            **{
                key: value
                for key, value in options.items()
                if key not in {"rgb_preprocessing", "eval_seeds"}
            }
        )
        tasks.append(options)
    if not tasks or len({task["task_name"] for task in tasks}) != len(tasks):
        raise ValueError("Evaluation requires distinct tasks")
    return tasks


def task_seed_pools(tasks):
    pools = {}
    for task in tasks:
        name = task["task_name"]
        seeds = task.get("eval_seeds")
        if seeds is None:
            from dexmani_sim import DATA_DIR

            path = DATA_DIR / "eval_seeds" / f"{name}.txt"
            seeds = [int(value) for value in path.read_text().split()]
        pools[name] = list(seeds)
    return pools


def prepare_evaluation_config(cfg, overrides=None):
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    merged = OmegaConf.to_container(OmegaConf.merge(cfg, overrides or {}), resolve=True)
    # Saved model inputs, including child language and RGB recipes, are immutable.
    if overrides is not None:
        pending = [OmegaConf.to_container(overrides).get("env_runner", {})]
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            elif isinstance(node, dict):
                if set(node) & {
                    "_target_",
                    "n_obs_steps",
                    "sensor_modalities",
                    "task_text",
                    "rgb_preprocessing",
                    "rgb_preprocess_size",
                    "rgb_random_crop_size",
                    "rgb_keep_uint8",
                    "rgb_color_aug",
                }:
                    raise ValueError("Model inputs must come from saved dataset recipes")
                pending.extend(node.values())
    return merged


def make_request(cfg, checkpoint, tasks, seeds, *, source):
    """Bind inference settings to tasks and seed plans already checked by the resolver."""
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
        checkpoint=str(Path(checkpoint).resolve()),
        use_ema=use_ema,
        inference_steps=steps,
        action_steps=action_steps,
        device=cfg["training"]["device"],
        tasks=[
            {key: value for key, value in task.items() if key != "eval_seeds"} for task in tasks
        ],
        task_seeds=seeds,
        numerics={
            "parameter_dtype": "saved_checkpoint",
            "autocast": False,
            "compile": False,
            "grad_mode": "no_grad",
            "startup_seed": cfg["eval"]["seed"],
            "policy_seed_mode": "episode_seed",
        },
        source=source,
    )


def frozen_test_request(exp_dir, record, checkpoint, *, device=None):
    """Build a held-out request from the caller's verified selection result."""
    request = EvaluationRequest(**record["test_request"])
    return replace(
        request,
        checkpoint=str(checkpoint),
        device=request.device if device is None else device,
        source={
            "phase": "test",
            "selection_result": str(Path(exp_dir).resolve() / record["selection_result"]),
            "heldout_from_selection": True,
            "global_step": record["global_step"],
            "expected_runtime": record["runtime"],
        },
    )


def resolve_evaluation_request(
    cfg, exp_dir, checkpoint="best", overrides=None, *, selection_record=None
):
    if selection_record is not None:
        override = OmegaConf.to_container(overrides) if overrides is not None else {}
        if set(override) - {"training"} or set(override.get("training", {})) - {"device"}:
            raise ValueError(
                "Frozen test permits only device/output changes; use ordinary evaluation for research overrides"
            )
        if checkpoint != "best":
            raise ValueError("Frozen test checkpoint comes exclusively from selection_record")
        record, path = verify_selection_record(exp_dir, selection_record)
        return frozen_test_request(
            exp_dir, record, path, device=override.get("training", {}).get("device"),
        )
    source = {"phase": "ordinary", "heldout_from_selection": False}
    if checkpoint == "best":
        selected, path = resolve_best_checkpoint(exp_dir)
        cfg = OmegaConf.merge(cfg, {"eval": selected["inference"]})
        source["selection_result"] = selected["selection_result"]
        source["global_step"] = selected["global_step"]
    else:
        path = resolve_checkpoint_path(exp_dir, checkpoint)
    cfg = prepare_evaluation_config(cfg, overrides)
    tasks = resolve_tasks(cfg)
    seeds = cfg["eval"].get("task_seeds")
    if seeds is None:
        seeds, _ = split_evaluation_seeds(
            task_seed_pools(tasks), cfg["eval"]["seed"], cfg["eval"]["episodes"]
        )
    else:
        validate_task_seeds(seeds, [task["task_name"] for task in tasks])
        validate_evaluation_seed(cfg["eval"]["seed"])
        for task in tasks:
            pool = task.get("eval_seeds")
            if pool is not None and set(seeds.get(task["task_name"], ())) - set(
                validate_seed_pool(task["task_name"], pool)
            ):
                raise ValueError("Requested seed is outside the declared task pool")
    return make_request(cfg, path, tasks, seeds, source=source)
