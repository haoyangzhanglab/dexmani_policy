"""Resolve remote training inputs without importing models or opening Zarr arrays.

Run from the same project root, Python environment and with the same config name
and Hydra overrides as training. Relative paths follow the project's existing
persistent-data symlinks; absolute dataset overrides remain literal paths.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dexmani_policy.common.config import register_resolvers


_TASK_COMPONENT = re.compile(r"[a-zA-Z0-9_-]+(?:\+[a-zA-Z0-9_-]+)*")


def resolve_dataset_paths(config_name: str, overrides: list[str]) -> list[Path]:
    if (
        not re.fullmatch(r"[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*", config_name)
        or any(part in {".", ".."} for part in config_name.split("/"))
    ):
        raise ValueError(f"Invalid relative config name: {config_name!r}")
    register_resolvers()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
        if not isinstance(cfg.task_name, str) or not _TASK_COMPONENT.fullmatch(cfg.task_name):
            raise ValueError(f"Invalid resolved task identity: {cfg.task_name!r}")
        dataset = OmegaConf.to_container(cfg.dataset, resolve=True)

    def collect(node):
        if not isinstance(node, dict):
            raise ValueError("Dataset config must be a mapping")
        if "datasets" in node:
            children = node["datasets"]
            if not isinstance(children, list) or not children or "zarr_path" in node:
                raise ValueError("MultiTask dataset must contain non-empty child datasets")
            for child in children:
                yield from collect(child)
        else:
            path = node.get("zarr_path")
            if (
                not isinstance(path, str) or not path
                or not re.fullmatch(r"[a-zA-Z0-9_./+~-]+", path)
                or ".." in path.split("/")
            ):
                raise ValueError(f"Invalid dataset.zarr_path: {path!r}")
            yield Path(path).expanduser().resolve()

    return list(dict.fromkeys(collect(dataset)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--check", action="store_true", help="Fail if any resolved Zarr directory is missing.")
    parser.add_argument("overrides", nargs="*", help="The same Hydra overrides passed to training.")
    args = parser.parse_args()
    paths = resolve_dataset_paths(args.config_name, args.overrides)
    print(json.dumps({"dataset_paths": [str(path) for path in paths]}, indent=2))
    if args.check:
        missing = [str(path) for path in paths if not path.is_dir()]
        if missing:
            raise SystemExit("Missing dataset directories:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
