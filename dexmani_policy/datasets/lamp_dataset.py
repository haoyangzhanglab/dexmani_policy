"""Point-cloud windows with causal hand commands and frozen LAMP targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dexmani_policy.agents.hand_prior.lamp import file_sha256
from dexmani_policy.datasets.pc_dataset import PCDataset


OBS_ALIGNMENT = "obs[t]_before_action[t]"


def _array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


class LampDataset(PCDataset):
    """Keep N observation frames separate from K past hand command targets.

    Missing commands before episode start use the initial measured hand pose.
    This is an explicit bootstrap convention, never the first expert action.
    Absolute latent targets are indexed by source frame, including padding.
    """

    def __init__(
        self,
        zarr_path,
        *,
        history_steps=8,
        latent_path=None,
        prior_path=None,
        require_latents=True,
        obs_alignment=None,
        **kwargs,
    ):
        if type(history_steps) is not int or history_steps < 1:
            raise ValueError("history_steps must be a positive integer")
        if type(require_latents) is not bool:
            raise ValueError("require_latents must be bool")
        if kwargs.get("action_key", "action") != "action" or kwargs.get("use_aux_ee", False):
            raise ValueError("LampDataset currently requires joint action=7+12 without auxiliary targets")
        modalities = kwargs.get("sensor_modalities") or self.DEFAULT_MODALITIES
        if "joint_state" not in modalities:
            raise ValueError("LAMP bootstrap requires joint_state observations")
        self.history_steps = history_steps
        self.latent_path = str(Path(latent_path).expanduser().resolve()) if latent_path else None
        self.prior_path = str(Path(prior_path).expanduser().resolve()) if prior_path else None
        self.require_latents = require_latents
        self._alignment_assertion = obs_alignment
        self._lamp_ready = False
        self.hand_latents = None
        self.latent_scale = None
        self.prior_sha256 = None
        super().__init__(zarr_path, **kwargs)
        self._lamp_train_source_rows = self.sampler.action_source_rows.copy()
        if require_latents:
            self._load_latents()

    def _prepare_lamp_source(self):
        if self._lamp_ready:
            return
        buffer = self.replay_buffer
        attrs = buffer.attrs
        asserted = self._alignment_assertion
        if asserted == "obs_before_action":
            asserted = OBS_ALIGNMENT
        if asserted is not None and asserted != OBS_ALIGNMENT:
            raise ValueError(f"Unsupported asserted obs_alignment: {asserted!r}")
        stored = attrs.get("obs_alignment")
        if stored is not None and stored != OBS_ALIGNMENT:
            raise ValueError(f"Unsupported stored obs_alignment: {stored!r}")
        if stored is None and asserted is None:
            raise ValueError(
                "LAMP requires obs_alignment='obs[t]_before_action[t]' in data attributes "
                "or an explicit obs_alignment='obs_before_action' assertion"
            )
        dt = attrs.get("dt")
        if isinstance(dt, bool) or not isinstance(dt, (float, int)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("LAMP data requires a finite positive dt attribute")
        if buffer["action"].shape[1:] != (19,) or buffer["joint_state"].shape[1:] != (19,):
            raise ValueError("LAMP requires action and joint_state with shape [T,19]")
        self.episode_ends = np.asarray(buffer.episode_ends, dtype=np.int64)
        self._episode_starts = np.r_[0, self.episode_ends[:-1]]
        self.hand_actions = np.asarray(
            buffer.read("action", slice(None), columns=slice(7, 19)), dtype=np.float32
        )
        self.hand_state_bootstrap = np.concatenate([
            buffer.read("joint_state", slice(int(start), int(start) + 1), columns=slice(7, 19))
            for start in self._episode_starts
        ]).astype(np.float32)
        finite_hand = np.isfinite(self.hand_actions).all(axis=-1)
        self.history_valid = np.empty(len(self.hand_actions), dtype=bool)
        for episode, (start, end) in enumerate(zip(self._episode_starts, self.episode_ends)):
            local_rows = np.arange(end - start)
            invalid_prefix = np.r_[0, np.cumsum(~finite_hand[start:end])]
            valid = invalid_prefix[local_rows] == invalid_prefix[np.maximum(0, local_rows - self.history_steps)]
            valid[:self.history_steps] &= np.isfinite(self.hand_state_bootstrap[episode]).all()
            self.history_valid[start:end] = valid
        self.lamp_source_metadata = {
            "history_steps": self.history_steps,
            "history": "strictly_past_hand_command_targets",
            "bootstrap": "episode_initial_measured_hand_pose",
            "obs_alignment": OBS_ALIGNMENT,
            "alignment_source": "data_attribute" if stored is not None else "explicit_assertion",
            "dt": float(dt),
            "action_key": "action",
            "hand_slice": [7, 19],
            "data_revision": self.data_revision,
            "num_frames": len(self.hand_actions),
            "hand_actions_sha256": _array_sha256(self.hand_actions),
            "bootstrap_sha256": _array_sha256(self.hand_state_bootstrap),
            "episode_ends_sha256": _array_sha256(self.episode_ends),
        }
        canonical = json.dumps(self.lamp_source_metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.lamp_source_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        self.data_recipe["lamp_history"] = {
            "source_fingerprint": self.lamp_source_fingerprint,
            "history_steps": self.history_steps,
            "bootstrap": "episode_initial_measured_hand_pose",
        }
        self._lamp_ready = True

    def _filter_sampler(self, sampler):
        # Called inside BaseDataset.__init__, before normalizer source rows exist.
        self._prepare_lamp_source()
        sampler.filter_valid(
            self._obs_valid, self._action_valid & self.history_valid, self.obs_horizon
        )

    def history_for_rows(self, rows):
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or (rows < 0).any() or (rows >= len(self.hand_actions)).any():
            raise ValueError("History rows must be one-dimensional source-frame indices")
        rows = rows.astype(np.int64, copy=False)
        episode = np.searchsorted(self.episode_ends, rows, side="right")
        starts = self._episode_starts[episode]
        indices = rows[:, None] + np.arange(-self.history_steps, 0)
        past = indices >= starts[:, None]
        safe_indices = np.maximum(indices, starts[:, None])
        return np.where(
            past[..., None], self.hand_actions[safe_indices], self.hand_state_bootstrap[episode, None]
        ).astype(np.float32, copy=False)

    def history_for_frame(self, row):
        return self.history_for_rows(np.asarray([row]))[0]

    @property
    def lamp_metadata(self):
        validation = self.get_validation_dataset()
        return {
            "source_fingerprint": self.lamp_source_fingerprint,
            "source_metadata": dict(self.lamp_source_metadata),
            "train_mask": self.train_mask.tolist(),
            "val_mask": self.val_mask.tolist(),
            "train_source_rows": self._lamp_train_source_rows.tolist(),
            "val_source_rows": [] if validation is None else validation.sampler.action_source_rows.tolist(),
        }

    def _load_latents(self):
        if self.latent_path is None or self.prior_path is None:
            raise ValueError("LAMP training requires latent_path and prior_path")
        prior = torch.load(self.prior_path, map_location="cpu", weights_only=True)
        cache = torch.load(self.latent_path, map_location="cpu", weights_only=True)
        if not isinstance(prior, dict) or prior.get("_format") != "lamp_prior.v1":
            raise ValueError("Expected lamp_prior.v1 artifact")
        if not isinstance(cache, dict) or cache.get("_format") != "lamp_latents.v1":
            raise ValueError("Expected lamp_latents.v1 artifact")
        expected = self.lamp_metadata
        for label, artifact in (("prior", prior), ("latent cache", cache)):
            metadata = artifact.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid {label} metadata")
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"LAMP {label} metadata mismatch: {key}")
        self.prior_sha256 = file_sha256(self.prior_path)
        if cache.get("prior_sha256") != self.prior_sha256:
            raise ValueError("LAMP latent cache prior_sha256 mismatch")
        latent_dim = prior.get("model_config", {}).get("latent_dim")
        if type(latent_dim) is not int or latent_dim < 1:
            raise ValueError("LAMP prior requires a positive latent_dim")
        latents, valid = cache.get("latents"), cache.get("valid")
        errors, scale = cache.get("reconstruction_error"), cache.get("latent_scale")
        n = len(self.hand_actions)
        if not torch.is_tensor(latents) or latents.dtype != torch.float32 or latents.shape != (n, latent_dim):
            raise ValueError("LAMP latents must be float32 [source_frames, latent_dim]")
        if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != (n,):
            raise ValueError("LAMP latent valid must be bool [source_frames]")
        if not torch.is_tensor(errors) or errors.dtype != torch.float32 or errors.shape != (n,):
            raise ValueError("LAMP reconstruction_error must be float32 [source_frames]")
        if (not torch.is_tensor(scale) or scale.dtype != torch.float32 or scale.shape != (latent_dim,)
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError("LAMP latent_scale must be positive finite float32 [latent_dim]")
        frame_valid = torch.from_numpy(self.history_valid & np.isfinite(self.hand_actions).all(axis=-1))
        if (valid & ~frame_valid).any():
            raise ValueError("LAMP cache marks invalid hand/history source frames valid")
        required = torch.tensor(sorted(set(expected["train_source_rows"] + expected["val_source_rows"])), dtype=torch.long)
        if not valid[required].all():
            raise ValueError("LAMP cache lacks valid labels for qualified train/validation source rows")
        if not torch.isfinite(latents[valid]).all() or not torch.isfinite(errors[valid]).all() or (errors[valid] < 0).any():
            raise ValueError("LAMP valid latent labels and reconstruction errors must be finite")
        self.hand_latents = latents
        self.latent_scale = scale
        self.data_recipe["lamp_latents"] = {
            "sha256": file_sha256(self.latent_path), "prior_sha256": self.prior_sha256
        }
        # lamp_metadata may have constructed the shallow validation copy above.
        if self._validation_dataset is not None:
            self._validation_dataset.hand_latents = latents
            self._validation_dataset.latent_scale = scale
            self._validation_dataset.prior_sha256 = self.prior_sha256
            self._validation_dataset.data_recipe["lamp_latents"] = dict(self.data_recipe["lamp_latents"])

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        data = super().__getitem__(idx)
        rows = self.sampler.source_rows(self.sampler.indices[idx:idx + 1])[0]
        query = rows[self.obs_horizon - 1]
        data["action_context"] = {
            "hand_history": torch.from_numpy(self.history_for_frame(query))
        }
        if self.hand_latents is not None:
            data["hand_latent_target"] = self.hand_latents[torch.from_numpy(rows)]
        return data
