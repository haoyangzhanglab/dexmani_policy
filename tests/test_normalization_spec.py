"""Shared normalization semantic-spec grammar and versioned contract parser.

Covers the contract-hardening patch: a single ``validate_normalization_spec``
used by both training (``resolve_normalization_spec``) and the versioned
contract parser (``parse_normalization_contract``), plus the PointNext
metric-geometry compatibility rule and joint_state-identity end-to-end support.
"""

import numpy as np
import pytest
import torch

from dexmani_policy.common.checkpoint_io import (
    make_normalization_contract,
    parse_normalization_contract,
)
from dexmani_policy.common.normalizer import validate_normalization_spec
from dexmani_policy.training.build_utils import (
    _validate_normalization_config,
    build_normalizer,
    resolve_normalization_spec,
)


# ---------------------------------------------------------------------------
# Spec grammar
# ---------------------------------------------------------------------------


def test_action_identity_rejected():
    with pytest.raises(ValueError, match="action"):
        validate_normalization_spec({"joint_state": "limits", "action": "identity"})


@pytest.mark.parametrize("mode", ["auto", "limits", "gaussian"])
def test_action_legal_modes(mode):
    spec = validate_normalization_spec({"joint_state": "limits", "action": mode})
    assert spec == {"joint_state": "limits", "action": mode}


def test_auto_on_non_action_rejected():
    with pytest.raises(ValueError, match="auto"):
        validate_normalization_spec(
            {"joint_state": "limits", "action": "auto", "point_cloud": "auto"}
        )


def test_rgb_identity_ok():
    spec = validate_normalization_spec(
        {"joint_state": "limits", "action": "auto", "rgb": "identity"}
    )
    assert spec["rgb"] == "identity"


@pytest.mark.parametrize("mode", ["limits", "gaussian", "auto"])
def test_rgb_non_identity_rejected(mode):
    with pytest.raises(ValueError, match="rgb"):
        validate_normalization_spec(
            {"joint_state": "limits", "action": "auto", "rgb": mode}
        )


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="banana"):
        validate_normalization_spec(
            {"joint_state": "limits", "action": "auto", "point_cloud": "banana"}
        )


def test_non_numeric_field_rejected():
    with pytest.raises(ValueError, match="task_text"):
        validate_normalization_spec(
            {"joint_state": "limits", "action": "auto", "task_text": "identity"}
        )


def test_missing_joint_state_rejected():
    with pytest.raises(ValueError, match="joint_state"):
        validate_normalization_spec({"action": "auto"})


def test_missing_action_rejected():
    with pytest.raises(ValueError, match="action"):
        validate_normalization_spec({"joint_state": "limits"})


def test_non_mapping_rejected():
    with pytest.raises(ValueError):
        validate_normalization_spec(["joint_state", "action"])


def test_non_string_key_rejected():
    with pytest.raises(ValueError, match="key"):
        validate_normalization_spec({1: "limits", "action": "auto"})


def test_non_string_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        validate_normalization_spec({"joint_state": "limits", "action": 1})


def test_coverage_missing_field_rejected():
    with pytest.raises(ValueError, match="missing"):
        validate_normalization_spec(
            {"joint_state": "limits", "action": "auto"},
            observation_fields={"joint_state", "rgb"},
        )


def test_coverage_extra_field_rejected():
    with pytest.raises(ValueError, match="extra"):
        validate_normalization_spec(
            {
                "joint_state": "limits",
                "action": "auto",
                "rgb": "identity",
                "point_cloud": "identity",
            },
            observation_fields={"joint_state", "rgb"},
        )


def test_coverage_exact_ok():
    spec = validate_normalization_spec(
        {"joint_state": "limits", "action": "auto", "rgb": "identity"},
        observation_fields={"joint_state", "rgb"},
    )
    assert spec == {"joint_state": "limits", "action": "auto", "rgb": "identity"}


# ---------------------------------------------------------------------------
# Versioned contract parser
# ---------------------------------------------------------------------------


def test_parse_roundtrip():
    spec = {"joint_state": "limits", "action": "auto", "rgb": "identity"}
    assert parse_normalization_contract(make_normalization_contract(spec)) == spec


def test_parse_wrong_version_rejected():
    with pytest.raises(ValueError, match="version"):
        parse_normalization_contract(
            {"version": 99, "fields": {"joint_state": "limits", "action": "auto"}}
        )


def test_parse_missing_version_rejected():
    with pytest.raises(ValueError, match="version"):
        parse_normalization_contract(
            {"fields": {"joint_state": "limits", "action": "auto"}}
        )


def test_parse_extra_top_level_key_rejected():
    with pytest.raises(ValueError, match="version"):
        parse_normalization_contract(
            {
                "version": 1,
                "fields": {"joint_state": "limits", "action": "auto"},
                "extra": 1,
            }
        )


def test_parse_banana_mode_rejected():
    with pytest.raises(ValueError, match="banana"):
        parse_normalization_contract(
            {
                "version": 1,
                "fields": {
                    "joint_state": "limits",
                    "action": "auto",
                    "point_cloud": "banana",
                },
            }
        )


# ---------------------------------------------------------------------------
# resolve_normalization_spec delegates to the shared validator
# ---------------------------------------------------------------------------


def test_resolve_delegates_to_shared_validator():
    cfg = {"normalization": {"joint_state": "limits", "action": "auto"}}
    assert resolve_normalization_spec(cfg) == {"joint_state": "limits", "action": "auto"}


def test_resolve_rejects_action_identity():
    cfg = {"normalization": {"joint_state": "limits", "action": "identity"}}
    with pytest.raises(ValueError, match="action"):
        resolve_normalization_spec(cfg)


def test_resolve_rejects_missing_normalization():
    with pytest.raises(ValueError, match="normalization"):
        resolve_normalization_spec({})


# ---------------------------------------------------------------------------
# PointNext metric-geometry compatibility
# ---------------------------------------------------------------------------


def _cfg(encoder_type, point_cloud_mode):
    return {
        "normalization": {
            "joint_state": "limits",
            "action": "auto",
            "point_cloud": point_cloud_mode,
        },
        "dataset": {"sensor_modalities": ["joint_state", "point_cloud"]},
        "agent": {"encoder_type": encoder_type},
    }


def test_pointnext_encoder_limits_rejected():
    with pytest.raises(ValueError, match="pointnext"):
        _validate_normalization_config(_cfg("pointnext", "limits"))


def test_pointnext_tokenizer_encoder_limits_rejected():
    with pytest.raises(ValueError, match="pointnext_tokenizer"):
        _validate_normalization_config(_cfg("pointnext_tokenizer", "limits"))


def test_pointnext_encoder_identity_ok():
    _validate_normalization_config(_cfg("pointnext", "identity"))  # must not raise


def test_non_metric_encoder_limits_ok():
    _validate_normalization_config(_cfg("dp3", "limits"))  # must not raise


# ---------------------------------------------------------------------------
# joint_state: identity — end-to-end training support
# ---------------------------------------------------------------------------


class _ActionOnlyDataset:
    """Minimal dataset stub: only 'action' has normalization data to iterate."""

    def iter_normalization_data(self, key):
        assert key == "action"
        yield np.random.default_rng(0).standard_normal((100, 19)).astype(np.float32)


def test_joint_state_identity_builder_registers_no_params():
    spec = {"joint_state": "identity", "action": "auto"}
    norm = build_normalizer(_ActionOnlyDataset(), spec, "action")
    assert "joint_state" not in norm.params_dict
    assert "action" in norm.params_dict


def test_joint_state_identity_passthrough():
    spec = {"joint_state": "identity", "action": "auto"}
    norm = build_normalizer(_ActionOnlyDataset(), spec, "action")
    js = torch.randn(2, 5, 19)
    out = norm.normalize({"joint_state": js, "action": torch.randn(2, 5, 19)})
    assert out["joint_state"] is js
