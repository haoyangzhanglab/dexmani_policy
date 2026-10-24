"""Evaluation arguments and saved-configuration precedence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dexmani_policy.common.config import positive_int, resolve_action_steps
from dexmani_policy.common.experiments import (
    resolve_best_checkpoint,
    resolve_checkpoint_path,
    resolve_selection_record,
)
from dexmani_policy.env_runner.common.seeds import resolve_evaluation_seed, validate_evaluation_seed

INFERENCE_FIELDS = {"inference_steps", "inference_steps_list"}
EVALUATION_FIELDS = {
    "select_best": {
        "initial_episodes",
        "batch_size",
        "max_episodes",
        "inference_steps",
        "action_steps",
        "use_ema",
    },
    "offline": {"episodes", "action_steps", "use_ema"} | INFERENCE_FIELDS,
    "demo": {"episodes", "action_steps", "use_ema", "viewer_resolution"} | INFERENCE_FIELDS,
    "video": {"enabled", "fps"},
}
EVALUATION_KEYS = {"eval.seed"} | {
    key
    for section, names in EVALUATION_FIELDS.items()
    for name in names
    for key in (f"eval.{name}", f"eval.{section}.{name}")
}
MODEL_INPUT_FIELDS = {
    "n_obs_steps",
    "sensor_modalities",
    "task_text",
    "rgb_preprocess_size",
    "rgb_random_crop_size",
    "rgb_keep_uint8",
    "rgb_preprocessing",
    "rgb_color_aug",
}


@dataclass
class EvaluationRequest:
    config: DictConfig
    checkpoint: Path
    selection: dict | None
    section: str
    use_ema: bool
    inference_steps: list[int]
    action_steps: int


class EvaluationSeedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        previous = getattr(namespace, self.dest, None)
        if previous is not None and previous != values:
            parser.error("Conflicting --seed values")
        setattr(namespace, self.dest, values)


def parse_evaluation_overrides(overrides: list[str]):
    """Only evaluation/inference controls may override checkpoint evaluation."""
    seed = None
    for override in overrides:
        key, separator, _ = override.partition("=")
        if key == "eval.seed":
            value = OmegaConf.select(OmegaConf.from_dotlist([override]), key)
            validate_evaluation_seed(value)
            if seed is not None and seed != value:
                raise ValueError(f"Conflicting {key} overrides")
            seed = value
        if not separator or not (
            key in EVALUATION_KEYS or key.startswith("env_runner.") or key == "training.device"
        ):
            raise ValueError(f"Unsupported evaluation override: {key}")

    cfg = OmegaConf.from_dotlist(overrides)
    # Inspect mapping/list overrides too, including whole multitask task_configs.
    pending = [("env_runner", OmegaConf.to_container(cfg).get("env_runner", {}))]
    while pending:
        path, node = pending.pop()
        if isinstance(node, dict):
            entries = node.items()
        elif isinstance(node, list):
            entries = enumerate(node)
        else:
            continue
        for name, value in entries:
            key = f"{path}.{name}"
            if name in MODEL_INPUT_FIELDS:
                raise ValueError(f"Model inputs come from saved dataset config: {key}")
            pending.append((key, value))

    shared = cfg.get("eval") or {}
    # None is unspecified; keep saved defaults intact when this override is merged.
    for source in (shared, *(shared.get(section) or {} for section in EVALUATION_FIELDS)):
        for name in list(source):
            if source[name] is None:
                del source[name]

    # Explicit shared values must beat saved section defaults.
    for section, names in EVALUATION_FIELDS.items():
        explicit = set(shared.get(section) or {})
        for name in names - explicit:
            # A section NFE overrides both shared spellings.
            if name in INFERENCE_FIELDS and explicit & INFERENCE_FIELDS:
                continue
            if name in shared:
                OmegaConf.update(cfg, f"eval.{section}.{name}", shared[name])
    return cfg


def resolve_inference_steps(*sources, override=None) -> list[int]:
    """Choose the source before the NFE spelling; section scalars beat shared lists."""
    steps = override
    for source in sources:
        if steps is not None:
            break
        steps = source.get("inference_steps_list")
        if steps is None and source.get("inference_steps") is not None:
            steps = [source["inference_steps"]]

    if steps is None:
        raise ValueError("Evaluation requires inference_steps or inference_steps_list")
    steps = list(steps)
    if not steps:
        raise ValueError("inference_steps_list must be non-empty")
    for nfe in steps:
        positive_int(nfe, "inference_steps")
    if len(set(steps)) != len(steps):
        raise ValueError("Inference steps must be distinct to preserve per-value results")
    return steps


def build_evaluation_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--policy-name", required=True, help="Experiment policy path, e.g. ddp/sat."
    )
    parser.add_argument(
        "--task-name", required=True, help="Task directory in the saved experiment."
    )
    parser.add_argument(
        "--exp-name", required=True, help="Run directory under experiments/policy/task."
    )
    parser.add_argument(
        "--seed",
        type=int,
        action=EvaluationSeedAction,
        help=(
            "Evaluation seed: CLI > eval.seed dotlist > saved eval.seed. "
            "Final eval with a selection record reuses its seed and rejects conflicts."
        ),
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        help="DDIM/Euler steps; defaults come from the saved config or selection record.",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        help="Physical actions consumed per prediction; defaults to selection or saved model window.",
    )
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument(
        "--ema", dest="use_ema", action="store_true", default=None, help="Use EMA weights."
    )
    weights.add_argument(
        "--no-ema", dest="use_ema", action="store_false", help="Use raw model weights."
    )
    parser.add_argument("overrides", nargs="*", help="Evaluation/environment dotlist overrides.")
    return parser


def resolve_evaluation_parameter(
    cfg, param: str, section: str | None = None, *, override: Any = None, default: Any = None
) -> Any:
    """Resolve explicit value → section → shared eval → default; None is unspecified."""
    if override is not None:
        return override
    eval_cfg = cfg.get("eval") or {}
    section_cfg = (eval_cfg.get(section) or {}) if section else {}
    for source in (section_cfg, eval_cfg):
        val = source.get(param)
        if val is not None:
            return val
    return default


def resolve_test_request(record):
    """Resolve the saved protocol and test budget before building a runner."""
    evaluation = record["evaluation"]
    request = evaluation["request"]
    validate_evaluation_seed(request["shuffle_seed"])
    saved_cfg = evaluation["effective_config"]
    episodes = positive_int(saved_cfg["eval"]["offline"]["episodes"], "saved test episodes")
    return request, episodes


def resolve_evaluation_request(
    cfg,
    exp_dir,
    checkpoint="best",
    overrides: DictConfig | None = None,
    *,
    section="offline",
    use_ema=None,
    inference_steps_list=None,
    action_steps=None,
    selection_record=None,
    eval_seed=None,
    episodes=None,
):
    """CLI > explicit dotlist > selected checkpoint > saved defaults."""
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    override_cfg = overrides if overrides is not None else OmegaConf.create({})
    merged = OmegaConf.merge(cfg, override_cfg)
    merged.experiment_dir = str(exp_dir)
    selection = None
    checkpoint_path = None
    if selection_record is not None:
        selection, checkpoint_path = resolve_selection_record(exp_dir, selection_record)
    elif checkpoint == "best":
        selection, checkpoint_path = resolve_best_checkpoint(exp_dir)

    requested_episodes = resolve_evaluation_parameter(
        override_cfg, "episodes", section, override=episodes
    )
    if selection is not None and section == "offline":
        request, count = resolve_test_request(selection)
        explicit_seed = eval_seed is not None or "seed" in (override_cfg.get("eval") or {})
        if explicit_seed and resolve_evaluation_seed(merged, eval_seed) != request["shuffle_seed"]:
            raise ValueError(
                "Explicit evaluation seed conflicts with selection record shuffle_seed"
            )
        seed = request["shuffle_seed"]
        if requested_episodes is not None:
            positive_int(requested_episodes, "episodes")
            if requested_episodes != count:
                raise ValueError("Explicit episodes conflicts with selection record test count")
    else:
        seed = resolve_evaluation_seed(merged, eval_seed)
        if section == "offline":
            count = positive_int(
                resolve_evaluation_parameter(
                    merged, "episodes", section, override=requested_episodes
                ),
                "episodes",
            )
    if section == "offline":
        OmegaConf.update(merged, "eval.offline.episodes", count)
    OmegaConf.update(merged, "eval.seed", seed)

    selected = selection["inference"] if selection is not None else {}
    weights = resolve_evaluation_parameter(override_cfg, "use_ema", section, override=use_ema)
    if weights is None:
        weights = selected.get(
            "use_ema", resolve_evaluation_parameter(cfg, "use_ema", section)
        )

    if type(weights) is not bool:
        raise ValueError("use_ema must be boolean")

    explicit_eval = override_cfg.get("eval") or {}
    saved_eval = cfg.get("eval") or {}
    steps = resolve_inference_steps(
        explicit_eval.get(section) or {},
        explicit_eval,
        selected,
        saved_eval.get(section) or {},
        saved_eval,
        override=inference_steps_list,
    )
    action_steps = resolve_evaluation_parameter(
        override_cfg, "action_steps", section, override=action_steps
    )
    if action_steps is None:
        action_steps = selected.get(
            "action_steps", resolve_evaluation_parameter(cfg, "action_steps", section)
        )
    action_steps = resolve_action_steps(merged.agent, action_steps)
    OmegaConf.update(merged, f"eval.{section}.action_steps", action_steps)

    if selection_record is not None:
        if checkpoint != "best" and resolve_checkpoint_path(exp_dir, checkpoint) != checkpoint_path:
            raise ValueError("Checkpoint override differs from selection record")
        if (
            weights != selected["use_ema"]
            or steps != [selected["inference_steps"]]
            or action_steps != selected["action_steps"]
        ):
            raise ValueError("Selection record cannot override raw/EMA, NFE or action_steps")
    if checkpoint_path is None:
        checkpoint_path = resolve_checkpoint_path(exp_dir, checkpoint)
    return EvaluationRequest(
        config=merged,
        checkpoint=checkpoint_path,
        selection=selection,
        section=section,
        use_ema=weights,
        inference_steps=steps,
        action_steps=action_steps,
    )
