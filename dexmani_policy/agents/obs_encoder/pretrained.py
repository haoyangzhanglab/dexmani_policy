from pathlib import Path

import torch

AFRO_SHA = "283024a49e2c14bc030debf15383a4e7ee73836d"
CORDVIP_SHA = "ad6d441abe7b0480cf79b0311db266965c1313b6"
AFRO_ENCODER_CONFIG = {"pc_dim": 3, "num_points": 1024, "feature_dim": 64, "pooling": "max"}
CORDVIP_ENCODER_CONFIG = {
    "pc_dim": 3, "num_points": 1024, "feature_dim": 1024, "state_feature_dim": 64,
    "joint_layout": {"arm": [0, 7], "hand": [7, 19]}, "pooling": "max",
    "observation_steps": 4,
}


def load_observation_encoder(path, *, method, encoder_config, data_description):
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if payload["method"] != method or payload["encoder_config"] != encoder_config:
        raise ValueError(f"Pretrained encoder method/structure mismatch: {path}")
    source = payload["source"]
    current = data_description["split_manifest"]
    same_source = source["zarr_path"] == data_description["zarr_path"] or (
        current["source_id"] is not None and current["source_id"] == source["source_id"]
    )
    if same_source and payload["split_manifest"] != current:
        raise ValueError("Same-source pretraining and policy training must share the saved split")
    if not same_source:
        print("[Pretraining] Separate source; cross-source held-out overlap is not automatically proven")
    if method == "cordvip" and payload["observation_geometry"] != data_description["observation_geometry"]:
        raise ValueError("Pretrained CordViP geometry/point-slot recipe differs from the policy data")
    return payload["encoder_state"]
