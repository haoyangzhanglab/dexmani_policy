import json


def observation_geometry(buffer):
    records = buffer.group["meta"]["observation_geometry_json"][:]
    if len(records) != buffer.n_episodes:
        raise ValueError("observation_geometry_json must describe every episode")
    recipes = [json.loads(record) for record in records]
    for recipe in recipes:
        if recipe["frame"] != "world" or recipe["units"] != "m":
            raise ValueError("Contact geometry requires world-frame metres")
        if recipe["object_sampling"] != "surface_area_v1":
            raise ValueError("Contact labels require the native surface_area_v1 template point order")
        parts = [recipe["hand"], *recipe["objects"]]
        for part in parts:
            if not part["template"] or len(part["part_names"]) != len(part["point_counts"]):
                raise ValueError("Geometry must identify templates, parts and point counts")
        if sum(recipe["hand"]["point_counts"]) != buffer["hand_pointcloud"].shape[1]:
            raise ValueError("Hand template layout differs from stored point slots")
        if sum(sum(obj["point_counts"]) for obj in recipe["objects"]) != buffer["object_pointcloud"].shape[1]:
            raise ValueError("Object template layout differs from stored point slots")
        roles = [obj["role"] for obj in recipe["objects"]]
        if len(roles) != len(set(roles)):
            raise ValueError("Object geometry roles must be unique")
    return recipes


def contact_source_metadata(buffer):
    return {
        "source_id": buffer.attrs.get("source_id"),
        "episode_ids": buffer.attrs.get("episode_ids"),
        "trial_ids": buffer.attrs.get("trial_ids"),
        "episode_ends": buffer.episode_ends.tolist(),
        "rows": int(buffer.episode_ends[-1]),
        "observation_geometry": observation_geometry(buffer),
    }
