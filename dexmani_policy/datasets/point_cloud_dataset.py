from typing import ClassVar

from dexmani_policy.datasets.base_dataset import BaseDataset


class PointCloudDataset(BaseDataset):
    DEFAULT_MODALITIES: ClassVar[list[str]] = ["joint_state", "point_cloud"]
