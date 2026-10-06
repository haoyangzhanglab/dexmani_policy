import warnings

import numpy as np
import torch


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: list,
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
        if task_texts is not None and len(task_texts) != len(datasets):
            raise ValueError("task_texts must have the same length as datasets")
        if sampling_strategy not in ["proportional", "balanced", "weighted"]:
            raise ValueError(
                "sampling_strategy must be 'proportional', 'balanced', or 'weighted'"
            )

        if sampling_strategy == "weighted":
            if task_weights is None or len(task_weights) != len(datasets):
                raise ValueError(
                    "weighted sampling requires task_weights for every dataset"
                )
            weights = np.asarray(task_weights, dtype=float)
            if not np.isfinite(weights).all():
                raise ValueError("task_weights must all be finite")
            if (weights < 0).any():
                raise ValueError("task_weights must all be nonnegative")
            if weights.sum() <= 0:
                raise ValueError("task_weights must sum to a positive value")

        if len(set(task_names)) != len(task_names):
            raise ValueError(
                f"Duplicate task names detected: {task_names}. "
                f"Task names must be unique — duplicates cause silent data misalignment "
                f"in get_validation_dataset() when task_weights are used."
            )

        for i, dataset in enumerate(datasets):
            child_key = getattr(dataset, "action_key", None)
            if child_key != action_key:
                raise ValueError(
                    f"Dataset {i} ({task_names[i]}) action_key={child_key!r} "
                    f"does not match shared action_key={action_key!r}"
                )
            if len(dataset) == 0:
                raise ValueError(
                    f"Dataset {i} (task '{task_names[i]}') is empty. "
                    f"All datasets must contain at least one sample."
                )

        layouts = set()
        for dataset in datasets:
            if hasattr(dataset, "replay_buffer"):
                layouts.add((tuple(dataset.replay_buffer[action_key].shape[1:]),
                             bool(getattr(dataset, "use_aux_ee", False))))
        if len(layouts) > 1:
            raise ValueError("Child datasets must share the same action shape and auxiliary layout")
        rgb_transport = {
            bool(getattr(dataset, "rgb_keep_uint8", False))
            for dataset in datasets
            if "rgb" in getattr(dataset, "sensor_modalities", ())
        }
        if len(rgb_transport) > 1:
            raise ValueError("RGB child datasets must have consistent rgb_keep_uint8")
        self.datasets = datasets
        self.task_names = task_names
        self.action_key = action_key

        if augmentation_cfg is not None:
            for dataset in self.datasets:
                if hasattr(dataset, "_build_augmentors"):
                    dataset.augmentation_cfg = augmentation_cfg
                    dataset._build_augmentors()
        self.num_tasks = len(datasets)
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
            self.sample_probs = np.ones(self.num_tasks) / self.num_tasks
        elif sampling_strategy == "weighted":
            self.sample_probs = weights / weights.sum()

        if sampling_strategy != "weighted" and task_weights is not None:
            warnings.warn(
                f"task_weights={task_weights} is ignored because "
                f"sampling_strategy='{sampling_strategy}'. "
                f"Set sampling_strategy='weighted' to use task weights."
            )

    def iter_normalization_data(self, key: str):
        """Yield the normalization statistics source for every child dataset.

        Shared-normalization aggregation is performed by the caller
        (``build_normalizer``) via streaming; this dataset owns no normalizer.
        """
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

        # inject task identity into obs (ManiFlow convention)
        sample["obs"]["task_text"] = self.task_texts[task_idx]
        sample["obs"]["task_name"] = self.task_names[task_idx]

        return sample

    def get_validation_dataset(self):
        val_datasets = [d.get_validation_dataset() for d in self.datasets]
        valid_triples = [
            (d, name, text)
            for d, name, text in zip(val_datasets, self.task_names, self.task_texts)
            if d is not None
        ]
        if not valid_triples:
            return None
        val_ds, val_names, val_texts = zip(*valid_triples)

        if self.task_weights is not None:
            val_weights = [
                self.task_weights[self.task_names.index(name)] for name in val_names
            ]
        else:
            val_weights = None

        return MultiTaskDataset(
            datasets=list(val_ds),
            task_names=list(val_names),
            sampling_strategy=self.sampling_strategy,
            task_weights=val_weights,
            seed=self.seed,
            deterministic=True,
            task_texts=list(val_texts),
            action_key=self.action_key,
        )
