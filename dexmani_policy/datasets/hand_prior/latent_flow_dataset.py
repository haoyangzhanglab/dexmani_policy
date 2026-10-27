from dexmani_policy.common.config import positive_int
from dexmani_policy.datasets.common.robot_dataset import RobotDataset
from dexmani_policy.datasets.hand_prior.metadata import describe_hand_chunks


class LatentFlowDataset(RobotDataset):
    """Historical observations paired with full future commands at the query state."""

    future_action_window = True
    action_normalization_population = "relative_commands_over_train_windows"

    @property
    def action_offset(self):
        return self.obs_horizon - 1

    def __init__(self, zarr_path, *, sensor_modalities, data_semantics, source_id=None, **kwargs):
        if kwargs.get("action_key", "action") != "action" or kwargs.get("use_aux_ee", False):
            raise ValueError("LatentFlow requires joint action=7+12 without auxiliary targets")
        if "joint_state" not in sensor_modalities:
            raise ValueError("LatentFlow reference requires joint_state observations")
        horizon = positive_int(kwargs.get("horizon", 32), "horizon")
        obs_horizon = positive_int(kwargs.get("obs_horizon", 2), "obs_horizon")
        if horizon < 3:
            raise ValueError("LatentFlow horizon must be at least 3")
        if kwargs.get("pad_before", obs_horizon - 1) != obs_horizon - 1 or kwargs.get("pad_after", 0) != 0:
            raise ValueError("LatentFlow pads only observation history, never future actions")
        kwargs.update(horizon=horizon, obs_horizon=obs_horizon, pad_before=obs_horizon - 1, pad_after=0)
        super().__init__(zarr_path, sensor_modalities=sensor_modalities, **kwargs)
        self.hand_data = describe_hand_chunks(self, data_semantics, source_id)
        self.commands = self.replay_buffer.read("action", slice(None))
        self.states = self.replay_buffer.read("joint_state", slice(None))
        self.commands.flags.writeable = self.states.flags.writeable = False

    def describe_hand_data(self):
        validation = self.get_validation_dataset()
        return dict(
            self.hand_data,
            train_windows=len(self),
            val_windows=0 if validation is None else len(validation),
            observation_rows=len(self.sampler.observation_source_rows),
            action_rows=len(self.sampler.action_source_rows),
        )

    def sample_to_data(self, sample):
        data = super().sample_to_data(sample)
        data["action_context"] = {"reference_state": sample["joint_state"][-1].copy()}
        return data

    def iter_normalization_data(self, key):
        if key != "action":
            yield from super().iter_normalization_data(key)
            return
        if self.is_validation:
            raise ValueError("Validation must reuse the training normalizer")
        for start in range(0, len(self), 1024):
            rows = self.sampler.source_rows(self.sampler.indices[start : start + 1024])
            future = rows[:, self.action_offset :]
            commands = self.commands[future]
            reference = self.states[future[:, 0]]
            yield (commands - reference[:, None]).reshape(-1, 19)
