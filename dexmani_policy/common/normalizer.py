import logging
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import zarr

from dexmani_policy.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)


def dfs_add(dest: dict, keys: list[str], value: torch.Tensor):
    if len(keys) == 1:
        dest[keys[0]] = value
        return
    if keys[0] not in dest:
        dest[keys[0]] = nn.ParameterDict()
    dfs_add(dest[keys[0]], keys[1:], value)


def load_param_dict(state_dict: dict, prefix: str) -> nn.ParameterDict:
    out_dict = nn.ParameterDict()
    for key, value in state_dict.items():
        value: torch.Tensor
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            assert suffix.startswith("."), f"prefix '{prefix}' missing trailing dot in key '{key}'"
            param_keys = suffix.split(".")[1:]
            if param_keys:
                dfs_add(out_dict, param_keys, value.clone())
    return out_dict


class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict
        self._field_views: dict = {}

    @property
    def device(self):
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            raise RuntimeError("Normalizer has no parameters; call fit() first")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        old_keys = set(self.params_dict.state_dict().keys())
        self.params_dict = load_param_dict(state_dict, prefix + "params_dict")
        self.params_dict.requires_grad_(False)
        self._field_views.clear()

        # Only report missing/unexpected keys on reload (not first load from empty).
        # On first load old_keys is empty and ALL state-dict keys would appear
        # as "unexpected", which would break strict=True callers like
        # load_normalizer_from_dataset.
        if len(old_keys) > 0:
            state_prefix = prefix + "params_dict."
            state_keys = {k[len(state_prefix) :] for k in state_dict if k.startswith(state_prefix)}
            for k in sorted(old_keys - state_keys):
                missing_keys.append(state_prefix + k)
            for k in sorted(state_keys - old_keys):
                unexpected_keys.append(state_prefix + k)


def _params_from_stats(
    input_min,
    input_max,
    input_mean,
    input_std,
    *,
    mode,
    output_max,
    output_min,
    range_eps,
    fit_offset,
    label=None,
):
    """Build scale/offset from aggregated statistics (shared by fit_params / fit_field_chunks).

    This is the single source of truth for the affine semantics so that the
    streaming chunk fit and the one-shot fit produce identical results.
    """
    assert mode in ["limits", "gaussian"]

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = -input_mean[ignore_dim]  # zero-center without scaling
        else:
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)

    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    n_ignored = ignore_dim.sum().item()
    if n_ignored > 0:
        prefix = f"{label}: " if label else ""
        idx_list = ignore_dim.nonzero(as_tuple=True)[0].tolist()
        logger.info(
            "%s%d/%d dims near-constant (range < %.0e, %s) — kept with identity scale. Indices: %s",
            prefix,
            n_ignored,
            ignore_dim.shape[0],
            range_eps,
            mode,
            idx_list,
        )

    this_params = nn.ParameterDict()
    this_params["scale"] = scale
    this_params["offset"] = offset
    for p in this_params.parameters():
        p.requires_grad_(False)

    input_stats = {
        "min": input_min,
        "max": input_max,
        "mean": input_mean,
        "std": input_std,
    }
    return this_params, input_stats


def fit_params(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
    label=None,
):
    assert mode in ["limits", "gaussian"] and last_n_dims >= 0 and output_max > output_min

    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    return _params_from_stats(
        input_min,
        input_max,
        input_mean,
        input_std,
        mode=mode,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
        fit_offset=fit_offset,
        label=label,
    )


def normalize_tensor(x, params, forward=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    # Avoid redundant .to(device) when scale/offset are already on x's device
    if scale.device != x.device:
        scale = scale.to(device=x.device)
    if offset.device != x.device:
        offset = offset.to(device=x.device)
    x = x.to(dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    """Linear normalizer for a single data field (joint_state or action).

    Fits scale/offset parameters from data and applies affine transformation
    to map data into ``[output_min, output_max]`` (default ``[-1, 1]``).

    Supports two modes:

    - ``"limits"``: min-max normalization. Low-variance dimensions
      (range < ``range_eps``) are zero-centered without scaling to avoid
      amplifying noise.
    - ``"gaussian"``: z-score normalization (mean=0, std=1).

    Implements ``DictOfTensorMixin`` for easy serialization with
    ``torch.save``/``torch.load``.
    """

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        self.params_dict, self.input_stats = fit_params(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]] = None,
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
            }
        )
        obj = cls(params_dict)
        obj.params_dict.requires_grad_(False)
        if input_stats_dict is not None:
            obj.input_stats = dict_apply(input_stats_dict, to_tensor)
        return obj

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return normalize_tensor(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return normalize_tensor(x, self.params_dict, forward=False)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


class LinearNormalizer(DictOfTensorMixin):
    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        if isinstance(data, dict):
            self.input_stats = {}
            for key, value in data.items():
                params, stats = fit_params(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                    label=key,
                )
                self.params_dict[key] = params
                self.input_stats[key] = stats
        else:
            self.params_dict["_default"], self.input_stats = fit_params(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )
        self._field_views.clear()

    @torch.no_grad()
    def fit_field(
        self,
        key: str,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        """Fit a single field from one (full) array using the existing one-shot math."""
        params, stats = fit_params(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
            label=key,
        )
        if not hasattr(self, "input_stats"):
            self.input_stats = {}
        self.params_dict[key] = params
        self.input_stats[key] = stats
        self._field_views.clear()

    @torch.no_grad()
    def fit_field_chunks(
        self,
        key: str,
        chunks,
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        """Fit a single field from multiple arrays via streaming Welford merge.

        Never concatenates the (potentially huge) observation arrays.  ``last_n_dims``
        must be consistent across chunks; the resulting scale/offset are identical to
        ``fit_field`` on the concatenated array up to floating-point accumulation error.
        """
        assert mode in ["limits", "gaussian"]

        # float64 is used ONLY for the streaming accumulator; the stored
        # scale/offset/input_stats must match the one-shot fit_params dtype
        # (float32 by default).  dtype=None collapses to the float32 default so a
        # no-op `.to(None)` can never leak float64 into the persisted params.
        if dtype is None:
            dtype = torch.float32

        def _as_float64_array(arr):
            if isinstance(arr, zarr.Array):
                arr = arr[:]
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr)
            return arr.to(torch.float64)

        chunks = list(chunks)
        if not chunks:
            raise ValueError("fit_field_chunks requires at least one chunk")

        first = _as_float64_array(chunks[0])
        dim = int(np.prod(first.shape[-last_n_dims:])) if last_n_dims > 0 else 1

        count = 0
        mean = torch.zeros(dim, dtype=torch.float64)
        m2 = torch.zeros(dim, dtype=torch.float64)
        running_min = None
        running_max = None

        for chunk in chunks:
            arr = _as_float64_array(chunk)
            c_dim = int(np.prod(arr.shape[-last_n_dims:])) if last_n_dims > 0 else 1
            if c_dim != dim:
                raise ValueError(
                    f"fit_field_chunks: inconsistent last-dim for '{key}' "
                    f"(expected {dim}, got {c_dim})"
                )
            c = arr.reshape(-1, dim)
            n = c.shape[0]
            if n == 0:
                continue
            c_min = c.min(dim=0).values
            c_max = c.max(dim=0).values
            running_min = c_min if running_min is None else torch.minimum(running_min, c_min)
            running_max = c_max if running_max is None else torch.maximum(running_max, c_max)

            # Welford batch merge.
            delta = c - mean
            new_count = count + n
            mean = mean + delta.sum(dim=0) / new_count
            delta2 = c - mean
            m2 = m2 + (delta * delta2).sum(dim=0)
            count = new_count

        if running_min is None:
            raise ValueError(f"fit_field_chunks: no non-empty chunks for '{key}'")

        if mode == "gaussian" and count < 2:
            raise ValueError(
                f"fit_field_chunks: gaussian mode requires at least 2 samples for "
                f"'{key}', got {count}"
            )

        input_min = running_min.to(dtype)
        input_max = running_max.to(dtype)
        input_mean = mean.to(dtype)
        variance = m2 / max(count - 1, 1)
        input_std = variance.sqrt().to(dtype)

        params, stats = _params_from_stats(
            input_min,
            input_max,
            input_mean,
            input_std,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
            label=key,
        )
        if not hasattr(self, "input_stats"):
            self.input_stats = {}
        self.params_dict[key] = params
        self.input_stats[key] = stats
        self._field_views.clear()

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        if key not in self._field_views:
            obj = SingleFieldLinearNormalizer(self.params_dict[key])
            if hasattr(self, "input_stats") and key in self.input_stats:
                obj.input_stats = self.input_stats[key]
            self._field_views[key] = obj
        return self._field_views[key]

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict
        self._field_views.pop(key, None)

    def is_fitted(self, required_keys=None):
        if len(self.params_dict) == 0:
            return False
        if required_keys is not None:
            return all(k in self.params_dict for k in required_keys)
        return True

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                if key not in self.params_dict:
                    result[key] = value
                    continue
                params = self.params_dict[key]
                result[key] = normalize_tensor(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return normalize_tensor(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)


def build_mixed_action_normalizer(action_data, ee_dim=9):
    """Build a mixed normalizer for eef_hand actions: xyz(3) + rot6d(ee_dim-3) + hand(rest).

    xyz and hand → limits [-1, 1] (min-max); rot6d → identity (scale=1, offset=0).
    """
    assert action_data.shape[1] > ee_dim, (
        f"action_ee dim ({action_data.shape[1]}) must be > ee_dim ({ee_dim})"
    )
    tmp = LinearNormalizer()
    tmp.fit(
        data={
            "xyz": action_data[:, :3],
            "hand": action_data[:, ee_dim:],
        },
        last_n_dims=1,
        mode="limits",
    )

    xyz_scale = tmp["xyz"].params_dict["scale"]
    xyz_offset = tmp["xyz"].params_dict["offset"]
    xyz_stats = tmp["xyz"].input_stats
    hand_scale = tmp["hand"].params_dict["scale"]
    hand_offset = tmp["hand"].params_dict["offset"]
    hand_stats = tmp["hand"].input_stats

    rot6d_dim = ee_dim - 3
    scale = torch.cat([xyz_scale, torch.ones(rot6d_dim), hand_scale])
    offset = torch.cat([xyz_offset, torch.zeros(rot6d_dim), hand_offset])
    stats = {}
    for k, fill in [("min", -1.0), ("max", 1.0), ("mean", 0.0), ("std", 1.0)]:
        stats[k] = torch.cat(
            [
                xyz_stats[k],
                torch.full((rot6d_dim,), fill, dtype=torch.float32),
                hand_stats[k],
            ]
        )

    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stats)


ALLOWED_NORMALIZATION_MODES = frozenset({"identity", "limits", "gaussian", "auto"})
NON_NUMERIC_OBSERVATION_FIELDS = frozenset({"task_text", "task_name"})


def validate_normalization_spec(
    spec: dict,
    *,
    observation_fields=None,
) -> dict:
    """Validate one feature-level normalization spec; return a canonical plain mapping.

    This is the single shared grammar for ``normalization: {field: mode}`` used by
    training config validation, the versioned checkpoint/deployment contract
    parser (``parse_normalization_contract``), and checkpoint-owned evaluation
    restore. All paths validate normalization modes with the same grammar.

    Rules:
    - ``spec`` must be a plain mapping of non-empty string keys to string modes;
    - mode must be one of ``identity | limits | gaussian | auto``;
    - ``auto`` is only allowed for ``action``;
    - ``joint_state`` and ``action`` must be explicitly declared;
    - ``action`` must not be ``identity`` (``BaseAgent`` always calls
      ``normalizer['action'].normalize/unnormalize``, which requires fitted params);
    - ``rgb``, if present, must be ``identity`` (replay-buffer statistics are HWC
      but the runtime tensor is CHW, so the generic affine normalizer's last-dim
      feature assumption is wrong for RGB);
    - non-numeric observation fields (``task_text``/``task_name``) must not appear.

    If ``observation_fields`` (the numeric observation field names, excluding
    ``action``) is given, this additionally enforces exact coverage:
    ``set(spec) == set(observation_fields) | {"action"}`` — rejecting both a
    missing numeric field and an extra/nonexistent one.
    """
    if type(spec) is not dict:
        raise ValueError("normalization spec must be a plain mapping of field -> mode")

    result: dict = {}
    for key, mode in spec.items():
        if type(key) is not str or not key:
            raise ValueError(f"normalization keys must be non-empty strings, got {key!r}")
        if key in NON_NUMERIC_OBSERVATION_FIELDS:
            raise ValueError(
                f"normalization must not declare non-numeric field {key!r} "
                "(task_text/task_name are not statistically normalized)"
            )
        if type(mode) is not str:
            raise ValueError(f"normalization.{key} mode must be a string, got {mode!r}")
        if mode not in ALLOWED_NORMALIZATION_MODES:
            raise ValueError(
                f"normalization.{key} has invalid mode {mode!r}; "
                f"expected one of {sorted(ALLOWED_NORMALIZATION_MODES)}"
            )
        if mode == "auto" and key != "action":
            raise ValueError(
                f"normalization mode 'auto' is only allowed for 'action', got '{key}'"
            )
        result[key] = mode

    for required in ("joint_state", "action"):
        if required not in result:
            raise ValueError(
                f"normalization must explicitly declare '{required}' (e.g. {required}: limits)"
            )
    if result["action"] == "identity":
        raise ValueError(
            "normalization.action must not be 'identity': BaseAgent always calls "
            "normalizer['action'].normalize/unnormalize, which requires fitted params. "
            "Use 'auto', 'limits', or 'gaussian'."
        )
    if "rgb" in result and result["rgb"] != "identity":
        raise ValueError(
            f"normalization.rgb must be 'identity' (got {result['rgb']!r}): replay-buffer "
            "RGB statistics are HWC but the runtime tensor is CHW, so the generic affine "
            "normalizer's last-dim feature assumption is wrong for RGB."
        )

    if observation_fields is not None:
        expected = set(observation_fields) | {"action"}
        actual = set(result)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(
                "normalization fields must exactly match numeric observation fields + "
                f"action: missing={sorted(missing)}, extra={sorted(extra)}"
            )

    return result


def validate_normalizer_state(normalizer: "LinearNormalizer", normalization_spec: dict) -> None:
    """Validate fitted normalizer params against the semantic normalization spec.

    - ``identity`` fields must have no params entry;
    - ``limits`` / ``gaussian`` fields (and the resolved ``action`` field) must have a
      finite, non-degenerate (scale != 0) ``scale``/``offset`` entry;
    - no unexpected fields may be present.

    Used by training build, eval restore and deployment restore as the single validator.
    """
    params = getattr(normalizer, "params_dict", None)
    if params is None:
        raise ValueError("normalizer has no params_dict")

    actual = set(params.keys())
    expected = {k for k, mode in normalization_spec.items() if mode != "identity"}
    if actual != expected:
        raise ValueError(
            "Normalizer state does not match normalization spec: "
            f"spec fields={sorted(expected)}, params fields={sorted(actual)}. "
            f"Identity fields must have no params entry."
        )

    for key in expected:
        entry = params[key]
        if "scale" not in entry or "offset" not in entry:
            raise ValueError(
                f"Normalizer state for '{key}' is incomplete (missing scale/offset)"
            )
        scale = entry["scale"]
        offset = entry["offset"]
        if (
            not torch.is_tensor(scale)
            or not torch.is_tensor(offset)
            or scale.numel() == 0
            or offset.numel() == 0
            or not bool(torch.isfinite(scale).all())
            or not bool(torch.isfinite(offset).all())
            or bool(torch.any(scale == 0))
        ):
            raise ValueError(
                f"Normalizer state for '{key}' is invalid "
                "(empty, non-finite, or zero scale)"
            )
