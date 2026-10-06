"""Publish one fixed paper protocol from the configured runner's actual seed pool.

Run from the repository root. This constructs only runners; no environment,
model, dataset, reset or step is created. Existing output files are never replaced.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra

from dexmani_policy.evaluation.protocol import (
    code_version, iter_leaf_env_runners, load_seed_manifest, task_seed_pools,
)
from dexmani_policy.utils.config import register_resolvers


def make_manifest(runner, *, pool_id, partition_seed=1066, selection=25, tie_break=5,
                  test=100, pool_sources=None, simulator_revision=None):
    for name, count in (("selection", selection), ("tie_break", tie_break), ("test", test)):
        if type(count) is not int or count < (0 if name == "tie_break" else 1):
            raise ValueError(f"Invalid {name} count: {count!r}")
    if type(partition_seed) is not int:
        raise ValueError("partition_seed must be an integer")
    pools = task_seed_pools(runner)
    count = min(len(pool) for pool in pools.values())
    for seeds in pools.values():
        if any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("Actual runner seed pool contains invalid or duplicate seeds")
    if count < selection + tie_break + test:
        raise ValueError(f"Actual common seed pool has {count} seeds; requested "
                         f"{selection}+{tie_break}+{test}. Supply a sufficient real pool.")
    # Same permutation as the previous reference list, expanded once at creation.
    indices = list(range(count))
    random.Random(partition_seed).shuffle(indices)
    def plan(start, stop):
        return {task: [pool[i] for i in indices[start:stop]] for task, pool in pools.items()}
    manifest = {"pool_id": pool_id, "partition_seed": partition_seed,
                "pool_sources": pool_sources or {}, "simulator_revision": simulator_revision,
                "selection": plan(0, selection),
                "tie_break": plan(selection, selection+tie_break),
                "test": plan(selection+tie_break, selection+tie_break+test)}
    load_seed_manifest(manifest, runner)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition-seed', type=int, default=1066)
    parser.add_argument('--selection', type=int, default=25)
    parser.add_argument('--tie-break', type=int, default=5)
    parser.add_argument('--test', type=int, default=100)
    parser.add_argument('overrides', nargs='*', help='Hydra dot-list overrides')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Manifest already exists: {args.output}")
    register_resolvers()
    with hydra.initialize_config_dir(config_dir=str(ROOT/'dexmani_policy/configs'), version_base=None):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
        runner = hydra.utils.instantiate(cfg.env_runner)
    from dexmani_sim import DATA_DIR, PACKAGE_DIR
    sources = {}
    for leaf in iter_leaf_env_runners(runner):
        seed_file = DATA_DIR/'eval_seeds'/f'{leaf.task_name}.txt'
        if leaf.eval_seeds is not None:
            source = 'env_runner.eval_seeds'
        elif seed_file.is_file():
            source = f'dexmani_sim.DATA_DIR/eval_seeds/{leaf.task_name}.txt'
        else:
            source = 'SimRunner fallback range(100)'
        sources[leaf.task_name] = {'source': source, 'count': len(leaf.get_seed_list())}
    manifest = make_manifest(runner, pool_id=f'{cfg.task_name}:paper-v1',
        partition_seed=args.partition_seed, selection=args.selection, tie_break=args.tie_break,
        test=args.test, pool_sources=sources, simulator_revision=code_version(PACKAGE_DIR)['commit'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(f"Published {args.output}: task pools={ {t: len(s) for t, s in task_seed_pools(runner).items()} }; "
          f"selection/tie_break/test={args.selection}/{args.tie_break}/{args.test}")


if __name__ == '__main__':
    main()
