"""Synthetic tests that the semantic normalization contract survives the deployment
boundary — no real robot data required.

Covers: inference-config construction -> deployment contract serialization/parsing ->
restore-side spec extraction/validation.  Only proves that ``{version: 1, fields: ...}``
is not dropped or rejected by any strict schema; it does NOT fake full Real Zarr parity.
"""

import pytest
import torch

from dexmani_policy.common.checkpoint_io import make_normalization_contract
from dexmani_policy.common.normalizer import LinearNormalizer, validate_normalizer_state
from dexmani_policy.deployment import export as exporter
from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    deployment_contract,
    parse_deployment_contract,
)
from dexmani_policy.deployment.restore import (
    DeploymentRestoreError,
    _extract_normalization_spec,
)


def _train():
    return {
        "action_key": "action",
        "action_dim": 19,
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "use_aux_ee": False,
    }


def _selected():
    return exporter._SelectedInferenceSettings(use_ema=False, denoise_steps=4)


def test_build_inference_config_embeds_normalization_contract():
    cfg_plain = {
        "task_name": "toy",
        "normalization": {"joint_state": "limits", "action": "auto", "point_cloud": "identity"},
    }
    agent_config = {"_target_": "dexmani_policy.agents.core.base.BaseAgent"}
    inference = exporter._build_inference_config(
        cfg_plain, agent_config, _train(), _selected()
    )
    assert inference["normalization"] == {
        "version": 1,
        "fields": {"joint_state": "limits", "action": "auto", "point_cloud": "identity"},
    }


def test_build_inference_config_requires_normalization():
    cfg_plain = {"task_name": "toy"}  # no top-level normalization
    with pytest.raises(exporter.InvalidExperimentError):
        exporter._build_inference_config(
            cfg_plain, {"_target_": "x"}, _train(), _selected()
        )


def _payload(spec):
    inference = {
        "task_name": "toy",
        "action_key": "action",
        "action_dim": 19,
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "use_aux_ee": False,
        "normalization": make_normalization_contract(spec),
        "agent": {"_target_": "dexmani_policy.agents.core.base.BaseAgent"},
        "eval": {"use_ema": False, "denoise_steps": 4},
    }
    data_contract = {
        "dt": 0.1,
        "requires_hand": True,
        "observation_fields": {
            "joint_state": {
                "shape": [19],
                "dtype": "float32",
                "semantics": {"representation": "joint_position"},
            },
        },
    }
    return {
        "_format": DEPLOYMENT_FORMAT,
        "contract": {
            "inference_config": inference,
            "data_contract": data_contract,
            "producer": {},
        },
        "weights": {"probe": torch.ones(1)},
    }


def test_normalization_contract_survives_strict_schema():
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    payload = _payload(spec)

    # The strict exporter payload validation and contract parser must neither
    # drop nor reject the extra 'normalization' key inside inference_config.
    exporter._validate_payload(payload)
    parse_deployment_contract(payload)  # must not raise

    contract = deployment_contract(payload)
    assert contract["inference_config"]["normalization"] == {
        "version": 1,
        "fields": spec,
    }


def test_restore_side_extraction_round_trips_to_valid_spec():
    # Mirrors restore_deployment_agent: extract fields, attach, then validate.
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    fields = _extract_normalization_spec(
        {"normalization": make_normalization_contract(spec)}
    )
    assert fields == spec  # version wrapper is stripped, plain {field: mode} remains

    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")

    agent = type("_Agent", (), {})()
    agent.normalizer = norm
    agent.normalization_spec = fields
    validate_normalizer_state(agent.normalizer, agent.normalization_spec)  # no raise


def test_restore_side_extraction_rejects_non_matching_normalizer():
    # If the contract declares identity for point_cloud but params carry it, the
    # shared validator must reject — proving extraction is wired to real validation.
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    fields = _extract_normalization_spec(
        {"normalization": make_normalization_contract(spec)}
    )

    norm = LinearNormalizer()
    norm.fit_field("joint_state", torch.randn(100, 19), mode="limits")
    norm.fit_field("action", torch.randn(100, 19), mode="limits")
    norm.fit_field("point_cloud", torch.randn(100, 6), mode="limits")

    agent = type("_Agent", (), {})()
    agent.normalizer = norm
    agent.normalization_spec = fields
    with pytest.raises(ValueError):
        validate_normalizer_state(agent.normalizer, agent.normalization_spec)


def test_extract_normalization_spec_enforces_version():
    spec = {"joint_state": "limits", "action": "auto"}
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(
            {"normalization": {"version": 99, "fields": spec}}
        )
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec({"normalization": {"fields": spec}})  # no version


def test_extract_normalization_spec_rejects_missing_contract():
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec({})  # no normalization key
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec({"normalization": {"version": 1}})  # no fields
