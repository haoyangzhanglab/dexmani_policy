import pytest
import torch

from dexmani_policy.common.checkpoint_io import (
    build_agent_contract,
    make_normalization_contract,
)
from dexmani_policy.common.normalizer import (
    LinearNormalizer,
    build_mixed_action_normalizer,
    validate_normalizer_state,
)


class _Model:
    def __init__(self):
        self.n_obs_steps = 2
        self.n_action_steps = 2
        self.action_dim = 19
        self.horizon = 4
        self.action_key = "action"
        self.tcp_dim = None
        self.hand_dim = None
        self.control_action_dim = 19
        self.use_aux_ee = False
        self.normalization_spec = {
            "joint_state": "limits",
            "action": "auto",
            "point_cloud": "identity",
        }


def test_make_normalization_contract():
    assert make_normalization_contract({"a": "limits"}) == {
        "version": 1,
        "fields": {"a": "limits"},
    }


def test_agent_contract_carries_normalization():
    contract = build_agent_contract(_Model())
    assert contract["normalization"]["version"] == 1
    assert contract["normalization"]["fields"] == _Model().normalization_spec


def test_validate_normalizer_state_ok():
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    validate_normalizer_state(norm, spec)  # should not raise


def test_validate_normalizer_state_missing_action_fails():
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    with pytest.raises(ValueError):
        validate_normalizer_state(norm, spec)


def test_validate_normalizer_state_identity_registered_fails():
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.fit_field("point_cloud", torch.randn(100, 6), mode="limits")
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    with pytest.raises(ValueError):
        validate_normalizer_state(norm, spec)


def test_validate_normalizer_state_mixed_action_ee_legal():
    # rot6d identity segment (scale=1, offset=0) must be accepted, not falsely rejected.
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm["action"] = build_mixed_action_normalizer(torch.randn(100, 21))
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    scale = norm.params_dict["action"]["scale"]
    offset = norm.params_dict["action"]["offset"]
    assert scale.shape == (21,) and offset.shape == (21,)
    assert (scale[3:9] == 1).all() and (offset[3:9] == 0).all()
    validate_normalizer_state(norm, spec)  # must not raise


def test_validate_normalizer_state_zero_scale_rejected():
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.params_dict["action"]["scale"][0] = 0.0
    spec = {"joint_state": "limits", "action": "auto"}
    with pytest.raises(ValueError):
        validate_normalizer_state(norm, spec)


def test_validate_normalizer_state_zero_offset_legal():
    # offset is only required to be finite; zero offset is legal.
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.params_dict["action"]["offset"][0] = 0.0
    spec = {"joint_state": "limits", "action": "auto"}
    validate_normalizer_state(norm, spec)  # must not raise


def test_validate_normalizer_state_unexpected_field_rejected():
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.fit_field("rgb", torch.randn(100, 6), mode="limits")  # not in spec at all
    spec = {"joint_state": "limits", "action": "auto"}
    with pytest.raises(ValueError):
        validate_normalizer_state(norm, spec)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_validate_normalizer_state_non_finite_scale_rejected(bad):
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.params_dict["action"]["scale"][0] = bad
    spec = {"joint_state": "limits", "action": "auto"}
    with pytest.raises(ValueError):
        validate_normalizer_state(norm, spec)


def test_validate_normalizer_state_empty_scale_rejected():
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.params_dict["action"] = torch.nn.ParameterDict(
        {"scale": torch.zeros(0), "offset": torch.zeros(0)}
    )
    spec = {"joint_state": "limits", "action": "auto"}
    with pytest.raises(ValueError):
        validate_normalizer_state(norm, spec)


def test_validate_normalizer_state_missing_offset_rejected():
    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.params_dict["action"] = torch.nn.ParameterDict({"scale": torch.ones(19)})
    spec = {"joint_state": "limits", "action": "auto"}
    with pytest.raises(ValueError):
        validate_normalizer_state(norm, spec)
