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

_OBS_FIELDS = {"joint_state", "point_cloud"}


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
    cfg_plain = {"task_name": "toy"}
    agent_config = {"_target_": "dexmani_policy.agents.core.base.BaseAgent"}
    contract = make_normalization_contract(
        {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    )
    inference = exporter._build_inference_config(
        cfg_plain, agent_config, _train(), _selected(), contract
    )
    assert inference["normalization"] == contract


# ---------------------------------------------------------------------------
# Checkpoint/config normalization reconciliation (P1-1)
# ---------------------------------------------------------------------------


def _fake_checkpoint(saved_normalization):
    """A minimal stand-in for TrainCheckpoint carrying only what reconciliation reads."""
    resume_contract = {
        "agent": {
            "n_obs_steps": 2,
            "normalization": saved_normalization,
        }
    }
    return type("_FakeCheckpoint", (), {"resume_contract": resume_contract})()


def test_reconcile_normalization_contract_matching_passes():
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "limits"}
    checkpoint = _fake_checkpoint(make_normalization_contract(spec))
    cfg_plain = {"normalization": dict(spec)}
    result = exporter._reconcile_normalization_contract(
        checkpoint, cfg_plain, ["joint_state", "point_cloud"]
    )
    assert result == make_normalization_contract(spec)


def test_reconcile_normalization_contract_mismatched_mode_rejected():
    # Checkpoint weights were fitted under point_cloud=limits; config has since
    # drifted to gaussian. Both share the same scale/offset key hierarchy, so
    # only a semantic contract comparison — not a state_dict shape/key check —
    # can catch this.
    saved_spec = {"joint_state": "limits", "action": "auto", "point_cloud": "limits"}
    current_spec = {"joint_state": "limits", "action": "auto", "point_cloud": "gaussian"}
    checkpoint = _fake_checkpoint(make_normalization_contract(saved_spec))
    cfg_plain = {"normalization": current_spec}
    with pytest.raises(exporter.InvalidCheckpointError):
        exporter._reconcile_normalization_contract(
            checkpoint, cfg_plain, ["joint_state", "point_cloud"]
        )


def test_reconcile_normalization_contract_version_mismatch_rejected():
    spec = {"joint_state": "limits", "action": "auto"}
    checkpoint = _fake_checkpoint({"version": 99, "fields": spec})
    cfg_plain = {"normalization": dict(spec)}
    with pytest.raises(exporter.InvalidCheckpointError):
        exporter._reconcile_normalization_contract(checkpoint, cfg_plain, ["joint_state"])


def test_reconcile_normalization_contract_missing_saved_contract_rejected():
    checkpoint = _fake_checkpoint(None)
    cfg_plain = {"normalization": {"joint_state": "limits", "action": "auto"}}
    with pytest.raises(exporter.InvalidCheckpointError):
        exporter._reconcile_normalization_contract(checkpoint, cfg_plain, ["joint_state"])


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


# ---------------------------------------------------------------------------
# Strict deployment contract parser (P2-6) — every violation below must be
# rejected via the shared validate_normalization_spec/parse_normalization_contract,
# not just the loose "version exists, fields is dict" check that used to exist.
# ---------------------------------------------------------------------------


def _inference(normalization):
    return {"normalization": normalization}


def test_extract_normalization_spec_rejects_extra_top_level_key():
    spec = {"joint_state": "limits", "action": "auto"}
    normalization = {**make_normalization_contract(spec), "extra": 1}
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(_inference(normalization))


def test_extract_normalization_spec_rejects_banana_mode():
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "banana"}
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(_inference(make_normalization_contract(spec)))


def test_extract_normalization_spec_rejects_auto_on_point_cloud():
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "auto"}
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(_inference(make_normalization_contract(spec)))


def test_extract_normalization_spec_rejects_action_identity():
    spec = {"joint_state": "limits", "action": "identity"}
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(_inference(make_normalization_contract(spec)))


def test_extract_normalization_spec_rejects_rgb_limits():
    spec = {"joint_state": "limits", "action": "auto", "rgb": "limits"}
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(_inference(make_normalization_contract(spec)))


def test_extract_normalization_spec_rejects_extra_nonexistent_identity_field():
    spec = {
        "joint_state": "limits",
        "action": "auto",
        "point_cloud": "identity",
        "phantom": "identity",
    }
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(
            _inference(make_normalization_contract(spec)),
            observation_fields={"joint_state", "point_cloud"},
        )


def test_extract_normalization_spec_rejects_missing_observation_field():
    spec = {"joint_state": "limits", "action": "auto"}
    with pytest.raises(DeploymentRestoreError):
        _extract_normalization_spec(
            _inference(make_normalization_contract(spec)),
            observation_fields={"joint_state", "point_cloud"},
        )


def test_extract_normalization_spec_exact_coverage_ok():
    spec = {"joint_state": "limits", "action": "auto", "point_cloud": "identity"}
    fields = _extract_normalization_spec(
        _inference(make_normalization_contract(spec)),
        observation_fields={"joint_state", "point_cloud"},
    )
    assert fields == spec


# ---------------------------------------------------------------------------
# Export-side spec-driven normalizer state validation (P1/P2-4) — no more
# hard-coded "joint_state/action always require params" or "rgb always
# forbidden"; every requirement is decided by the normalization_spec.
# ---------------------------------------------------------------------------


def _obs_fields():
    return {
        "joint_state": {"shape": [19], "dtype": "float32", "semantics": {}},
        "point_cloud": {"shape": [1024, 6], "dtype": "float32", "semantics": {}},
    }


def _state_dict(fields):
    state = {}
    for key, dim in fields.items():
        state[f"normalizer.params_dict.{key}.scale"] = torch.ones(dim)
        state[f"normalizer.params_dict.{key}.offset"] = torch.zeros(dim)
    return state


def test_export_validate_normalizer_state_joint_state_identity_accepted():
    spec = {"joint_state": "identity", "action": "auto"}
    state = _state_dict({"action": 19})  # no joint_state params
    exporter._validate_normalizer_state(state, _obs_fields(), 19, spec)  # no raise


def test_export_validate_normalizer_state_joint_state_identity_with_params_rejected():
    spec = {"joint_state": "identity", "action": "auto"}
    state = _state_dict({"action": 19, "joint_state": 19})
    with pytest.raises(exporter.InvalidCheckpointError):
        exporter._validate_normalizer_state(state, _obs_fields(), 19, spec)


def test_export_validate_normalizer_state_required_field_missing_rejected():
    spec = {"joint_state": "limits", "action": "auto"}
    state = _state_dict({"action": 19})  # missing required joint_state params
    with pytest.raises(exporter.InvalidCheckpointError):
        exporter._validate_normalizer_state(state, _obs_fields(), 19, spec)


def test_export_validate_normalizer_state_point_cloud_identity_no_params():
    spec = {
        "joint_state": "limits",
        "action": "auto",
        "point_cloud": "identity",
    }
    state = _state_dict({"action": 19, "joint_state": 19})
    exporter._validate_normalizer_state(state, _obs_fields(), 19, spec)  # no raise
