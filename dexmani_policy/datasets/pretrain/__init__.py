import copy

import torch
from torch.utils.data import Dataset

from dexmani_policy.datasets.split import format_split_summary, split_episodes


class ObservationPretrainDataset(Dataset):
    def __init__(self, episode_store, *, normalization_fields, seed, val_episodes,
                 max_train_episodes):
        self.episode_store = episode_store
        self.normalization_fields = normalization_fields
        self.train_mask, self.val_mask = split_episodes(
            episode_store.episode_groups, seed=seed,
            val_episodes=val_episodes, max_train_episodes=max_train_episodes,
        )
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError(f"No {type(self).__name__} training windows")

    def __len__(self):
        return len(self.sampler)

    def sample_to_data(self, sample):
        return {"obs": {key: torch.as_tensor(sample[key]) for key in self.sensor_modalities}}

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        return [self.sample_to_data(sample) for sample in self.sampler.sample_batch(list(indices))]

    def training_summary(self):
        store = self.episode_store
        return (
            f"{store.path}: {format_split_summary(store, self.train_mask, self.val_mask)}; "
            f"train windows={len(self):,}"
        )

    def build_sampler(self, mask):
        raise NotImplementedError

    def iter_normalization_data(self, key):
        for _, values in self.episode_store.iter_chunks(
            key, columns=self.normalization_fields[key], episode_mask=self.train_mask,
        ):
            yield values

    def get_validation_dataset(self):
        if not self.val_mask.any():
            return None
        result = copy.copy(self)
        result.sampler = self.build_sampler(self.val_mask)
        if not len(result):
            raise ValueError(f"No {type(self).__name__} validation windows in selected val episodes")
        return result
