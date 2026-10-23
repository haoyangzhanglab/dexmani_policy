import warnings

import numpy as np
import torch

from dexmani_policy.common.rgb_preprocessing import rgb_preprocessing_kwargs
from dexmani_policy.datasets.demonstration_dataset import DemonstrationDataset


def observation_layout(dataset, field):
    shape = tuple(dataset.replay_buffer[field].shape[1:])
    preprocessing = None
    if field == "rgb":
        preprocessing = rgb_preprocessing_kwargs(vars(dataset))
        if dataset.rgb_preprocess_size is not None:
            spatial_shape = dataset.rgb_random_crop_size or dataset.rgb_preprocess_size
            shape = (3, *spatial_shape)
    elif field == "point_cloud":
        shape = (
            dataset.random_keep_points or shape[0],
            dataset.pc_dim or shape[-1],
        )
    return (dataset.obs_horizon, *shape), preprocessing


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: list[DemonstrationDataset],
        task_names: list[str],
        sampling_strategy: str = "balanced",
        task_weights: list[float] | None = None,
        seed: int = 42,
        deterministic: bool = False,
        task_texts: list[str] | None = None,
        augmentation_cfg=None,
        action_key: str = "action",
    ):
        if len(datasets) != len(task_names):
            raise ValueError("datasets and task_names must have the same length")
        if not datasets:
            raise ValueError("MultiTaskDataset requires at least one child")
        if task_texts is not None and len(task_texts) != len(datasets):
            raise ValueError("task_texts must have the same length as datasets")
        if sampling_strategy not in {"proportional", "balanced", "weighted"}:
            raise ValueError("sampling_strategy must be 'proportional', 'balanced', or 'weighted'")

        if sampling_strategy == "weighted":
            weights = np.asarray(task_weights, dtype=float)
            if (
                weights.shape != (len(datasets),)
                or not np.isfinite(weights).all()
                or (weights < 0).any()
                or weights.sum() <= 0
            ):
                raise ValueError(
                    "task_weights must be finite nonnegative scalars with a positive sum"
                )

        if len(set(task_names)) != len(task_names):
            raise ValueError("Task names must be unique")

        if augmentation_cfg is not None:
            for dataset in datasets:
                dataset.augmentation_cfg = augmentation_cfg
                dataset.build_augmentors()

        reference = datasets[0]
        fields = set(reference.sensor_modalities)
        layouts = {field: observation_layout(reference, field) for field in fields}
        action_shape = tuple(reference.replay_buffer[reference.action_key].shape[1:])
        for name, child in zip(task_names, datasets):
            if child.action_key != action_key:
                raise ValueError(
                    f"Dataset {name} action_key={child.action_key!r} "
                    f"does not match shared action_key={action_key!r}"
                )
            if (
                tuple(child.replay_buffer[action_key].shape[1:]) != action_shape
                or child.use_aux_ee != reference.use_aux_ee
                or child.horizon != reference.horizon
            ):
                raise ValueError(
                    "Child datasets must share the same action shape and auxiliary layout"
                )
            if set(child.sensor_modalities) != fields:
                raise ValueError("Child datasets must share the complete observation modality set")
            for field in fields:
                if observation_layout(child, field) != layouts[field]:
                    raise ValueError(f"Child observation layout or preprocessing mismatch: {field}")

        self.datasets = datasets
        self.task_names = task_names
        self.action_key = action_key

        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.task_weights = task_weights
        self.task_texts = task_texts if task_texts is not None else task_names

        self.task_lengths = [len(d) for d in datasets]
        self.cumsum_lengths = np.cumsum([0] + self.task_lengths)
        self.total_length = sum(self.task_lengths)

        self.deterministic = deterministic

        if sampling_strategy == "proportional":
            self.sample_probs = np.array(self.task_lengths) / self.total_length
        elif sampling_strategy == "balanced":
            self.sample_probs = np.full(len(datasets), 1 / len(datasets))
        elif sampling_strategy == "weighted":
            self.sample_probs = weights / weights.sum()

        if sampling_strategy != "weighted" and task_weights is not None:
            warnings.warn(
                f"task_weights={task_weights} is ignored because "
                f"sampling_strategy='{sampling_strategy}'. "
                f"Set sampling_strategy='weighted' to use task weights."
            )

    def iter_normalization_data(self, key: str):
        """Stream all child statistics, including tasks with zero sampling weight."""
        for dataset in self.datasets:
            yield from dataset.iter_normalization_data(key)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total_length
        if not 0 <= idx < self.total_length:
            raise IndexError(idx)
        task_idx = int(np.searchsorted(self.cumsum_lengths, idx, side="right") - 1)
        local_idx = int(idx - self.cumsum_lengths[task_idx])

        sample = self.datasets[task_idx][local_idx]

        sample["obs"]["task_text"] = self.task_texts[task_idx]
        sample["obs"]["task_name"] = self.task_names[task_idx]

        return sample

    def get_validation_dataset(self):
        children = [
            (index, child)
            for index, dataset in enumerate(self.datasets)
            if (child := dataset.get_validation_dataset()) is not None
        ]
        if not children:
            return None
        indices, val_datasets = zip(*children)

        return MultiTaskDataset(
            datasets=list(val_datasets),
            task_names=[self.task_names[i] for i in indices],
            sampling_strategy="proportional",
            seed=self.seed,
            deterministic=True,
            task_texts=[self.task_texts[i] for i in indices],
            action_key=self.action_key,
        )
