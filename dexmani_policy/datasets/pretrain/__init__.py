import copy

from torch.utils.data import Dataset

from dexmani_policy.datasets.split import split_episodes


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

    def training_summary(self):
        store = self.episode_store
        unused = ~(self.train_mask | self.val_mask)
        return (
            f"{store.path}: source groups={store.count_groups():,}, segments={store.n_episodes:,}; "
            f"train groups={store.count_groups(self.train_mask):,}/segments={int(self.train_mask.sum()):,}, "
            f"val groups={store.count_groups(self.val_mask):,}/segments={int(self.val_mask.sum()):,}, "
            f"unused groups={store.count_groups(unused):,}/segments={int(unused.sum()):,}, "
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
