"""Copy three fixed-template real episodes into temporary observation-only smoke sources."""
import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from dexmani_policy.datasets.common.geometry import observation_geometry
from dexmani_policy.datasets.common.replay_buffer import ReplayBuffer
from dexmani_policy.datasets.common.split import validate_episode_metadata, validate_split_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='robot_data/pick_apple_messy.zarr')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = zarr.open_group(args.source, mode='r')
    indices = [0, 2, 4]
    ends = np.asarray(source['meta/episode_ends'])
    starts = np.r_[0, ends[:-1]]
    lengths = [int(ends[i] - starts[i]) for i in indices]
    new_ends = np.cumsum(lengths)
    recipes = observation_geometry(ReplayBuffer(args.source, keys=['hand_pointcloud', 'object_pointcloud']))
    assert all(recipes[i] == recipes[indices[0]] for i in indices)
    attrs = dict(source.attrs)
    ids, trials = validate_episode_metadata(attrs, len(ends))
    ids = [ids[i] if ids is not None else f'original_episode_{i}' for i in indices]
    trials = None if trials is None else {key: trials[key] for key in ids}
    attrs.update(source_id=f'obs-smoke:{Path(args.source).resolve()}:0,2,4',
                 episode_ids=ids, trial_ids=trials,
                 fixture_source={'path': str(Path(args.source).resolve()), 'episode_indices': indices})
    fields = {
        'afro': ['point_cloud'],
        'cordvip': ['hand_pointcloud', 'object_pointcloud', 'joint_state'],
        'policy': ['point_cloud', 'hand_pointcloud', 'object_pointcloud', 'joint_state', 'action'],
    }
    for name, keys in fields.items():
        target = zarr.open_group(str(args.output / f'{name}.zarr'), mode='w')
        target.attrs.update(attrs)
        data, meta = target.create_group('data'), target.create_group('meta')
        for key in keys:
            values = np.concatenate([source['data'][key][starts[i]:ends[i]] for i in indices])
            data.create_dataset(key, data=values, chunks=(min(16, len(values)), *values.shape[1:]))
        meta.create_dataset('episode_ends', data=new_ends)
        geometry = np.asarray([json.dumps(recipes[i], sort_keys=True) for i in indices])
        meta.create_dataset('observation_geometry_json', data=geometry)
    manifest = dict(source_id=attrs['source_id'], episode_ids=ids, trial_ids=trials,
                    episode_ends=new_ends.tolist(), train_indices=[0, 1], val_indices=[2])
    validate_split_manifest(manifest, ReplayBuffer(args.output / 'policy.zarr', keys=[]))
    (args.output / 'split.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'original_episode_indices': indices,
                      'lengths': lengths, 'train': [0, 1], 'val': [2]}, indent=2))


if __name__ == '__main__':
    main()
