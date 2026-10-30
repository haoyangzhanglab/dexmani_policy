import numpy as np

from dexmani_policy.datasets.demonstration import DemonstrationDataset

AUX_EE_DIM = 9


class R3DDataset(DemonstrationDataset):
    def __init__(self, zarr_path, **kwargs):
        if kwargs.get("action_key", "action") != "action":
            raise ValueError("R3D auxiliary targets require joint action")
        super().__init__(zarr_path, **kwargs)
        shape = self.episode_store["action_ee"].shape
        if len(shape) != 2 or shape[-1] < AUX_EE_DIM:
            raise ValueError("R3D auxiliary action_ee requires at least 9 columns")

    @property
    def data_keys(self):
        return super().data_keys + ["action_ee"]

    @property
    def consumed_fields(self):
        return {**super().consumed_fields, "action_ee": slice(0, AUX_EE_DIM)}

    @property
    def action_dim(self):
        return super().action_dim + AUX_EE_DIM

    def sample_to_data(self, sample):
        data = super().sample_to_data(sample)
        data["action"] = np.concatenate((data["action"], sample["action_ee"]), axis=-1)
        return data

    def iter_normalization_data(self, key):
        if key != "action":
            yield from super().iter_normalization_data(key)
            return
        if self.training_dataset is not None:
            raise ValueError("Validation must reuse the training normalizer")
        for start, values in self.episode_store.iter_chunks(self.action_key, episode_mask=self.train_mask):
            auxiliary = self.episode_store.read(
                "action_ee", slice(start, start + len(values)), columns=self.consumed_fields["action_ee"],
            )
            yield np.concatenate((values, auxiliary), axis=-1)
