import argparse
import json
from pathlib import Path

from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.datasets.episode_store import EpisodeStore
from dexmani_policy.datasets.split import split_episodes


def main():
    parser = argparse.ArgumentParser(
        description="Create a shared episode split, keeping known trials together"
    )
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-episodes", type=int)
    args = parser.parse_args()
    buffer = EpisodeStore(args.zarr_path, keys=[])
    _, _, manifest = split_episodes(
        buffer,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_episodes=args.max_train_episodes,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output, overwrite=False) as temporary:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Split: train={len(manifest['train_indices'])}, validation={len(manifest['val_indices'])}; {output}"
    )


if __name__ == "__main__":
    main()
