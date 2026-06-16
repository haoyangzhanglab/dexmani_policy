import zarr
import torch
import numpy as np
import torch.nn as nn
from typing import Union, Dict

from dexmani_policy.common.pytorch_util import dict_apply


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
            param_keys = key[len(prefix):].split('.')[1:]
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        old_keys = set(self.params_dict.state_dict().keys())
        self.params_dict = load_param_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)

        if len(old_keys) > 0:
            state_prefix = prefix + 'params_dict.'
            state_keys = {
                k[len(state_prefix):] for k in state_dict
                if k.startswith(state_prefix)
            }
            for k in sorted(old_keys - state_keys):
                missing_keys.append(state_prefix + k)
            for k in sorted(state_keys - old_keys):
                unexpected_keys.append(state_prefix + k)



def fit_params(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode='limits',
    output_max=1.,
    output_min=-1.,
    range_eps=1e-4,
    fit_offset=True,
    label=None,
):
    assert mode in ['limits', 'gaussian'] and last_n_dims >= 0 and output_max > output_min

    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)


    if mode == 'limits':
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

    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    n_ignored = ignore_dim.sum().item()
    if n_ignored > 0:
        prefix = f"[Normalizer] {label}: " if label else "[Normalizer]: "
        print(
            f"{prefix}{n_ignored}/{ignore_dim.shape[0]} dims ignored "
            f"(range_eps={range_eps}, mode={mode}). "
            f"Ignored indices: {ignore_dim.nonzero(as_tuple=True)[0].tolist()}"
        )

    this_params = nn.ParameterDict()
    this_params['scale'] = scale
    this_params['offset'] = offset
    for p in this_params.parameters():
        p.requires_grad_(False)

    input_stats = {
        'min': input_min,
        'max': input_max,
        'mean': input_mean,
        'std': input_std,
    }
    return this_params, input_stats


def normalize_tensor(x, params, forward=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale'].to(device=x.device)
    offset = params['offset'].to(device=x.device)
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
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True
    ):
        self.params_dict, self.input_stats = fit_params(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset
        )
    
    @classmethod
    def create_fit_params(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
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

        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
        })
        obj = cls(params_dict)
        if input_stats_dict is not None:
            obj.input_stats = dict_apply(input_stats_dict, to_tensor)
        return obj

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
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
        mode='limits',
        output_max=1.,
        output_min=-1.,
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
            self.params_dict['_default'], self.input_stats = fit_params(
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
    def fit_obs_action(cls, joint_state, action, action_key, mode='limits'):
        """Factory: fit a ``LinearNormalizer`` from joint_state and action arrays.

        Handles the ``action_ee`` special case where the normalizer is fitted on
        joint_state only and the action normalizer is built from fixed ranges.
        """
        normalizer = cls()
        if action_key == 'action_ee':
            normalizer.fit(data={'joint_state': joint_state}, last_n_dims=1, mode=mode)
            normalizer['action'] = build_mixed_action_normalizer(action)
        else:
            normalizer.fit(data={'joint_state': joint_state, 'action': action},
                           last_n_dims=1, mode=mode)
        return normalizer

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        if key not in self._field_views:
            obj = SingleFieldLinearNormalizer(self.params_dict[key])
            if hasattr(self, 'input_stats') and key in self.input_stats:
                obj.input_stats = self.input_stats[key]
            self._field_views[key] = obj
        return self._field_views[key]

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def is_fitted(self, required_keys=None):
        if len(self.params_dict) == 0:
            return False
        if required_keys is not None:
            return all(k in self.params_dict for k in required_keys)
        return True

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                if key not in self.params_dict:
                    result[key] = value
                    continue
                params = self.params_dict[key]
                result[key] = normalize_tensor(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return normalize_tensor(x, params, forward=forward)


    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)


    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)




def build_mixed_action_normalizer(action_data, ee_dim=9):
    """Build a mixed normalizer for eef_hand actions: xyz(3) + rot6d(ee_dim-3) + hand(rest).

    xyz and hand → limits [-1, 1] (min-max); rot6d → identity (scale=1, offset=0).
    """
    assert action_data.shape[1] > ee_dim, \
        f"action_ee dim ({action_data.shape[1]}) must be > ee_dim ({ee_dim})"
    tmp = LinearNormalizer()
    tmp.fit(data={
        'xyz':  action_data[:, :3],
        'hand': action_data[:, ee_dim:],
    }, last_n_dims=1, mode='limits')

    xyz_scale  = tmp['xyz'].params_dict['scale']
    xyz_offset = tmp['xyz'].params_dict['offset']
    xyz_stats  = tmp['xyz'].input_stats
    hand_scale  = tmp['hand'].params_dict['scale']
    hand_offset = tmp['hand'].params_dict['offset']
    hand_stats  = tmp['hand'].input_stats

    rot6d_dim = ee_dim - 3
    scale  = torch.cat([xyz_scale, torch.ones(rot6d_dim), hand_scale])
    offset = torch.cat([xyz_offset, torch.zeros(rot6d_dim), hand_offset])
    stats = {}
    for k, fill in [('min', -1.), ('max', 1.), ('mean', 0.), ('std', 1.)]:
        stats[k] = torch.cat([
            xyz_stats[k],
            torch.full((rot6d_dim,), fill, dtype=torch.float32),
            hand_stats[k],
        ])

    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stats)