import copy

from torch.utils.data import Dataset

from dexmani_policy.datasets.split import split_episodes


class ObservationPretrainDataset(Dataset):
    def __init__(self, episode_store, *, normalization_fields, seed, val_episodes,
                 max_train_episodes):
        self.episode_store = episode_store
        self.normalization_fields = normalization_fields
        self.train_mask, self.val_mask = split_episodes(
            episode_store.n_episodes, seed=seed,
            val_episodes=val_episodes, max_train_episodes=max_train_episodes,
        )
        self.sampler = self.build_sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError(f"No {type(self).__name__} training windows")
        self.training = True

    def __len__(self):
        return len(self.sampler)

    def training_summary(self):
        return (
            f"{self.episode_store.path}: total episodes={self.episode_store.n_episodes:,}, "
            f"train={int(self.train_mask.sum()):,}, val={int(self.val_mask.sum()):,}, "
            f"unused={int((~(self.train_mask | self.val_mask)).sum()):,}, "
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
        if not self.training:
            return self
        if not self.val_mask.any():
            return None
        result = copy.copy(self)
        result.training = False
        result.sampler = self.build_sampler(self.val_mask)
        if not len(result):
            raise ValueError(f"No {type(self).__name__} validation windows in selected val episodes")
        return result
