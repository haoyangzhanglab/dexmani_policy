from dexmani_policy.env_runner.sim_runner import SimRunner


def observation_processing(data_processing, sensor_modalities):
    fields = set(sensor_modalities)
    if "point_cloud" in fields:
        fields.add("rgb")  # The render size also determines the sampled point cloud.
    data_processing = data_processing or {}
    missing = fields - data_processing.keys()
    if missing:
        raise ValueError(f"Missing DexJoCo observation processing metadata for {sorted(missing)}")
    return {key: data_processing[key] for key in sorted(fields)}


class DexJoCoRunner(SimRunner):
    def __init__(self, *args, data_processing, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_processing = observation_processing(data_processing, self.sensor_modalities)

    @staticmethod
    def load_seed_pool(task_name):
        return list(range(10000, 20000))

    def _make_env(self, *, record_video: bool):
        from dexjoco.policy_env import DexJoCoEnv

        return DexJoCoEnv(
            self.task_name, sensor_modalities=self.sensor_modalities,
            action_format="rotvec",
            record_video=record_video, **self.env_kwargs,
        )

    def _validate_env(self, env):
        processing = env.data_processing
        changes = {
            key: (expected, processing.get(key)) for key, expected in self.data_processing.items()
            if processing.get(key) != expected
        }
        if changes:
            raise ValueError(f"DexJoCo observation processing mismatch (checkpoint, environment): {changes}")
        super()._validate_env(env)
