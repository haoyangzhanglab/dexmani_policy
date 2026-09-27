"""Snapshot the numerical runtime recipe from the actual training store."""

import json
import math

import numpy as np
import zarr


def extract_real_runtime(dataset, cfg):
    path = getattr(dataset, "zarr_path", None)
    if path is None:
        return None
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    if attrs.get("schema_name") != "dexmani-real-policy-zarr":
        return None
    if attrs.get("domain") != "real" or attrs.get("schema_version") != 15:
        raise ValueError("Expected canonical Real Policy Zarr v15")
    if attrs.get("task_name") != cfg.task_name:
        raise ValueError("Real Zarr task_name differs from the training task")
    dt = attrs.get("dt")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Real Zarr dt must be finite and positive")
    recipe = {"control_dt_s": float(dt)}
    if "point_cloud" in dataset.sensor_modalities:
        cloud = json.loads(attrs["pointcloud_config_json"])
        array = root["data/point_cloud"]
        if (
            not isinstance(cloud, dict)
            or type(cloud.get("num_points")) is not int
            or cloud["num_points"] <= 0
            or type(cloud.get("remove_table")) is not bool
            or len(array.shape) != 3
            or array.shape[1:] != (cloud["num_points"], 6)
            or np.dtype(array.dtype) != np.dtype("float32")
        ):
            raise ValueError(
                "Real point-cloud recipe disagrees with the stored XYZRGB array"
            )
        encoder = cfg.agent.get("pc_encoder_config", {})
        for count in (cfg.agent.get("num_points"), encoder.get("num_points")):
            if count is not None and count != cloud["num_points"]:
                raise ValueError(
                    "Agent point count disagrees with the actual training cloud"
                )
        recipe["pointcloud"] = cloud
    return recipe
