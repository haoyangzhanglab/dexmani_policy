"""Deployment integrity regression suite.

Covers the research-correctness invariants of the checkpoint-owned deployment
contract: the selected checkpoint owns architecture, action/window,
normalization, dataset constructor config and the frozen training data
semantic snapshot (``deployment_data_semantics``); a Zarr — default or
``--zarr-path`` relocation — must match that snapshot exactly; the artifact
owns selected weights plus an immutable observation/action contract; exported
restores predict identically to direct checkpoint restores; and no
researcher-facing path can publish an unverified artifact.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import (
    ACTION_EE_COMPONENTS,
    CHECKPOINT_N_HEAD,
    CONTROL_DT,
    DRIFTED_N_HEAD,
    NORMALIZATION_FIELDS,
    NUM_POINTS,
    TASK_NAME,
    add_rgb,
    agent_config,
    agent_contract,
    dataset_config,
    write_checkpoint,
    write_config,
    write_zarr,
)
from dexmani_policy.deployment import export as exporter
from dexmani_policy.deployment.contract import (
    DeploymentContractError,
    parse_deployment_contract,
    validate_agent_targets,
)
from dexmani_policy.deployment.export import (
    ArtifactPublicationError,
    ArtifactVerificationError,
    InvalidCheckpointError,
    UnsupportedPolicyError,
    export_deployment_artifact,
)
from dexmani_policy.deployment.restore import DeploymentRestoreError


def _export(experiment: Path, zarr_path: Path, **kwargs):
    return export_deployment_artifact(
        experiment, checkpoint_selector="latest.pt", zarr_path=zarr_path, **kwargs
    )


def _contract(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["contract"]


# ---------------------------------------------------------------------------
# 1. checkpoint owns the constructor
# ---------------------------------------------------------------------------


def test_export_uses_checkpoint_agent_config_not_current_config(experiment, zarr_path):
    """A post-training ``config.yaml.agent`` edit must not change an old artifact."""
    write_config(experiment, agent_cfg=agent_config(n_head=DRIFTED_N_HEAD))

    receipt = _export(experiment, zarr_path)

    exported_agent = _contract(receipt.checkpoint_path)["inference_config"]["agent"]
    assert exported_agent["n_head"] == CHECKPOINT_N_HEAD
    assert exported_agent["n_head"] != DRIFTED_N_HEAD


def test_export_ignores_current_config_agent_removal(experiment, zarr_path):
    """Deployment must not read the current config's agent section at all."""
    write_config(experiment, agent_cfg={"_target_": "not.a.real.Agent"})

    receipt = _export(experiment, zarr_path)

    agent = _contract(receipt.checkpoint_path)["inference_config"]["agent"]
    assert agent["_target_"] == agent_config()["_target_"]


# ---------------------------------------------------------------------------
# 2. checkpoint owns dataset / preprocessing semantics
# ---------------------------------------------------------------------------


def test_export_uses_checkpoint_dataset_modalities(experiment, zarr_path):
    """Current-config modality drift must not change the artifact contract."""
    write_config(
        experiment,
        dataset_cfg=dataset_config(sensor_modalities=["joint_state", "eef_pose"]),
    )

    receipt = _export(experiment, zarr_path)

    fields = _contract(receipt.checkpoint_path)["data_contract"]["observation_fields"]
    assert sorted(fields) == ["joint_state", "point_cloud"]


def test_export_uses_checkpoint_dataset_zarr_path(tmp_path, experiment):
    """The default dataset location comes from checkpoint-saved dataset semantics."""
    relocated = tmp_path / "checkpoint_owned.zarr"
    write_zarr(relocated)
    write_checkpoint(
        experiment,
        filename="relocated.pt",
        dataset_cfg=dataset_config(zarr_path=str(relocated)),
    )
    # The current config still points at a path that does not exist.
    write_config(experiment, dataset_cfg=dataset_config(zarr_path="robot_data/gone.zarr"))

    receipt = export_deployment_artifact(experiment, checkpoint_selector="relocated.pt")

    assert receipt.checkpoint_path.is_file()


def test_zarr_override_must_still_match_experiment_task_identity(tmp_path, experiment):
    """``--zarr-path`` relocates data only; it cannot re-label task identity."""
    foreign = tmp_path / "foreign.zarr"
    write_zarr(foreign, task_name="a_different_task")

    with pytest.raises(exporter.InvalidZarrError, match="task_name"):
        _export(experiment, foreign)


def test_rgb_preprocessing_reads_checkpoint_dataset(experiment, zarr_path):
    """RGB preprocessing is derived from checkpoint semantics, not current config."""
    add_rgb(zarr_path)
    checkpoint_dataset = dataset_config(
        _target_="dexmani_policy.datasets.rgb_dataset.RGBDataset",
        sensor_modalities=["joint_state", "rgb"],
        rgb_preprocess_size=[240, 240],
        rgb_random_crop_size=[224, 224],
    )
    checkpoint_path, _ = write_checkpoint(
        experiment,
        filename="rgb.pt",
        agent_cfg={**agent_config(), "rgb_backbone_name": "resnet"},
        weights_from=agent_config(),
        dataset_cfg=checkpoint_dataset,
        contract=agent_contract(
            normalization_fields={
                "joint_state": "limits",
                "action": "auto",
                "rgb": "identity",
            }
        ),
        zarr_file=zarr_path,
    )
    # The current config now declares completely different RGB preprocessing.
    write_config(
        experiment,
        dataset_cfg=dataset_config(
            _target_="dexmani_policy.datasets.rgb_dataset.RGBDataset",
            sensor_modalities=["joint_state", "rgb"],
            rgb_preprocess_size=[128, 128],
            rgb_random_crop_size=[112, 112],
        ),
    )

    source = exporter._parse_checkpoint_deployment_source(
        exporter._load_training_checkpoint(checkpoint_path)
    )
    preprocessing = exporter._rgb_preprocessing(source)

    assert preprocessing["resize_hw"] == [240, 240]
    assert preprocessing["center_crop_hw"] == [224, 224]


def test_rgb_checkpoint_without_validation_preprocessing_fails(experiment, zarr_path):
    """A checkpoint whose own RGB dataset omits preprocessing must fail fast."""
    add_rgb(zarr_path)
    checkpoint_path, _ = write_checkpoint(
        experiment,
        filename="rgb.pt",
        agent_cfg={**agent_config(), "rgb_backbone_name": "resnet"},
        weights_from=agent_config(),
        dataset_cfg=dataset_config(
            _target_="dexmani_policy.datasets.rgb_dataset.RGBDataset",
            sensor_modalities=["joint_state", "rgb"],
        ),
        contract=agent_contract(
            normalization_fields={
                "joint_state": "limits",
                "action": "auto",
                "rgb": "identity",
            }
        ),
        zarr_file=zarr_path,
    )

    source = exporter._parse_checkpoint_deployment_source(
        exporter._load_training_checkpoint(checkpoint_path)
    )
    with pytest.raises(InvalidCheckpointError, match="preprocessing"):
        exporter._rgb_preprocessing(source)


# ---------------------------------------------------------------------------
# 2b. checkpoint freezes the actual training data semantics
# ---------------------------------------------------------------------------


def _snapshot(experiment: Path, checkpoint_name: str = "latest.pt") -> dict:
    checkpoint = exporter._load_training_checkpoint(
        experiment / "checkpoints" / checkpoint_name
    )
    return checkpoint.resume_contract["deployment_data_semantics"]


def test_checkpoint_freezes_actual_training_zarr_semantics(experiment, zarr_path):
    """The snapshot must capture the real Zarr's timing/preprocessing/action facts."""
    semantics = _snapshot(experiment)

    assert semantics["task_name"] == TASK_NAME
    assert semantics["dt"] == CONTROL_DT
    assert semantics["obs_alignment"] == "obs[t]_before_action[t]"
    assert semantics["observation_alignment"] == "control_step_latest_causal"
    assert semantics["state_alignment"] == "control_step"
    assert semantics["action_semantics"] == "teleop_published_joint_target"
    assert semantics["action_ee_frame"] == "xarm_base"
    assert semantics["action_ee_components"] == ACTION_EE_COMPONENTS
    joint = semantics["observation_fields"]["joint_state"]
    assert joint["shape"] == [19]
    assert joint["dtype"] == "float32"
    points = semantics["observation_fields"]["point_cloud"]
    assert points["shape"] == [NUM_POINTS, 6]
    assert points["dtype"] == "float32"
    assert points["semantics"]["policy_id"]
    assert points["semantics"]["sampling"]
    assert points["semantics"]["transform"]
    assert json.loads(points["semantics"]["processing_config_json"])["pointcloud"]
    assert json.loads(points["semantics"]["table_plane_abcd_json"]) == [
        0.0,
        0.0,
        1.0,
        -0.02,
    ]


def test_snapshot_is_owned_by_the_checkpoint_not_the_current_config(
    experiment, zarr_path
):
    """Post-training config drift cannot rewrite the frozen snapshot."""
    before = _snapshot(experiment)
    write_config(
        experiment,
        dataset_cfg=dataset_config(
            zarr_path="robot_data/gone.zarr",
            sensor_modalities=["joint_state"],
        ),
    )

    assert _snapshot(experiment) == before

    # Export still uses the checkpoint-owned snapshot, not the drifted config.
    receipt = _export(experiment, zarr_path)
    data = _contract(receipt.checkpoint_path)["data_contract"]
    assert data["dt"] == CONTROL_DT
    assert sorted(data["observation_fields"]) == ["joint_state", "point_cloud"]
    assert data["action_ee_frame"] == "xarm_base"
    assert data["action_ee_components"] == ACTION_EE_COMPONENTS
    assert data["requires_hand"] is True


def test_legacy_checkpoint_still_loads_but_cannot_export(experiment, zarr_path):
    """A pre-snapshot checkpoint loads for analysis; Real export refuses it."""
    path, _ = write_checkpoint(
        experiment,
        filename="legacy.pt",
        zarr_file=zarr_path,
        deployment_data_semantics="absent",
    )

    checkpoint = exporter._load_training_checkpoint(path)
    assert "deployment_data_semantics" not in checkpoint.resume_contract

    with pytest.raises(
        InvalidCheckpointError, match="predates deployment_data_semantics"
    ):
        export_deployment_artifact(
            experiment, checkpoint_selector="legacy.pt", zarr_path=zarr_path
        )
    assert not list((experiment / "checkpoints").glob("*deployment*.pt"))


def test_none_semantics_checkpoint_fails_as_unsupported(experiment, zarr_path):
    """A checkpoint trained without a single Real Zarr is unsupported, not legacy."""
    write_checkpoint(experiment, filename="sim.pt", deployment_data_semantics=None)

    with pytest.raises(UnsupportedPolicyError, match="dynamic/multi-task"):
        export_deployment_artifact(
            experiment, checkpoint_selector="sim.pt", zarr_path=zarr_path
        )


def test_real_policy_contract_requires_action_ee_components(tmp_path):
    """The shared extractor itself gates the canonical EE action layout.

    A Real Zarr that does not declare ``action_ee_components`` — or declares a
    different layout — fails at the training-snapshot stage, before any
    checkpoint exists, not only later at the export equality check.
    """
    from dexmani_policy.datasets.real_policy_contract import (
        RealPolicyContractError,
        build_real_policy_data_semantics,
    )

    kwargs = dict(
        task_name=TASK_NAME,
        observation_fields=["joint_state"],
        agent_config=agent_config(),
        action_key="action",
    )
    missing = write_zarr(
        tmp_path / "missing.zarr", drop_attrs=["action_ee_components"]
    )
    with pytest.raises(RealPolicyContractError, match="action_ee_components"):
        build_real_policy_data_semantics(missing, **kwargs)

    wrong = write_zarr(
        tmp_path / "wrong.zarr",
        attrs_override={"action_ee_components": "eef_position_m(3)+eef_rot6d(6)"},
    )
    with pytest.raises(
        RealPolicyContractError, match="action_ee_components is invalid"
    ):
        build_real_policy_data_semantics(wrong, **kwargs)

    canonical = write_zarr(tmp_path / "canonical.zarr")
    semantics = build_real_policy_data_semantics(canonical, **kwargs)
    assert semantics["action_ee_components"] == ACTION_EE_COMPONENTS


# ---------------------------------------------------------------------------
# 2c. --zarr-path is semantic-equivalent relocation only
# ---------------------------------------------------------------------------


def test_semantic_equivalent_relocation_passes(tmp_path, experiment, zarr_path):
    """A different physical path with identical semantics is a valid relocation."""
    relocated = write_zarr(tmp_path / "relocated.zarr")

    receipt = _export(experiment, relocated)

    data = _contract(receipt.checkpoint_path)["data_contract"]
    assert data["dt"] == CONTROL_DT
    assert data["task_name"] == TASK_NAME


def test_schema_version_difference_is_informational(tmp_path, experiment, zarr_path):
    """A producer version bump alone is not drift; the artifact records the actual."""
    relocated = write_zarr(
        tmp_path / "reversioned.zarr", attrs_override={"schema_version": 14}
    )

    receipt = _export(experiment, relocated)

    data = _contract(receipt.checkpoint_path)["data_contract"]
    assert data["schema_version"] == 14
    assert data["dt"] == CONTROL_DT


def test_export_fails_on_dt_drift(tmp_path, experiment, zarr_path):
    """The training control rate is frozen; a re-timed store cannot replace it."""
    drifted = write_zarr(tmp_path / "drifted.zarr", dt=CONTROL_DT * 0.625)

    with pytest.raises(exporter.InvalidZarrError, match="dt"):
        _export(experiment, drifted)


def test_export_fails_on_point_cloud_processing_drift(tmp_path, experiment, zarr_path):
    """Point-cloud preprocessing is physics-facing: any config drift fails export."""
    drifted = write_zarr(
        tmp_path / "drifted.zarr",
        attrs_override={
            "processing_config_json": json.dumps(
                {"pointcloud": {"depth_max_m": 2.0}, "table_plane_abcd": None}
            )
        },
    )

    with pytest.raises(exporter.InvalidZarrError, match="processing_config_json"):
        _export(experiment, drifted)


def test_export_fails_on_table_plane_drift(tmp_path, experiment, zarr_path):
    """A moved table plane changes what the crop keeps, so it is drift."""
    drifted = write_zarr(
        tmp_path / "drifted.zarr",
        attrs_override={
            "point_cloud_table_plane_abcd_json": json.dumps(
                [0.0, 0.0, 1.0, -0.05], separators=(",", ":")
            )
        },
    )

    with pytest.raises(exporter.InvalidZarrError, match="table_plane_abcd_json"):
        _export(experiment, drifted)


def test_export_fails_on_point_cloud_transform_drift(tmp_path, experiment, zarr_path):
    drifted = write_zarr(
        tmp_path / "drifted.zarr",
        attrs_override={"point_cloud_transform": "depth_to_color_deprojection"},
    )

    with pytest.raises(exporter.InvalidZarrError, match="point-cloud semantics"):
        _export(experiment, drifted)


def test_export_fails_on_action_ee_frame_drift(tmp_path, experiment, zarr_path):
    drifted = write_zarr(
        tmp_path / "drifted.zarr", attrs_override={"action_ee_frame": "color_optical"}
    )

    with pytest.raises(exporter.InvalidZarrError, match="semantics"):
        _export(experiment, drifted)


def test_export_fails_on_action_ee_components_drift(tmp_path, experiment, zarr_path):
    """A store that stops declaring the EE action layout is not the same data."""
    drifted = write_zarr(tmp_path / "drifted.zarr", drop_attrs=["action_ee_components"])

    with pytest.raises(exporter.InvalidZarrError, match="action_ee_components"):
        _export(experiment, drifted)


def test_export_fails_on_raw_point_cloud_shape_drift(tmp_path, experiment, zarr_path):
    drifted = write_zarr(tmp_path / "drifted.zarr", point_count=2 * NUM_POINTS)

    with pytest.raises(exporter.InvalidZarrError, match="point count"):
        _export(experiment, drifted)


def test_export_fails_on_raw_dtype_drift(tmp_path, experiment, zarr_path):
    drifted = write_zarr(tmp_path / "drifted.zarr", state_dtype=np.float64)

    with pytest.raises(exporter.InvalidZarrError, match="shape/dtype"):
        _export(experiment, drifted)


# ---------------------------------------------------------------------------
# 3. normalization is owned by the checkpoint contract
# ---------------------------------------------------------------------------


def test_normalization_round_trips_from_checkpoint_contract(experiment, zarr_path):
    """The artifact's normalization is the checkpoint's own contract, verbatim."""
    drifted = {
        "joint_state": "gaussian",
        "action": "limits",
        "point_cloud": "gaussian",
    }
    write_config(experiment, normalization=drifted)

    receipt = _export(experiment, zarr_path)

    normalization = _contract(receipt.checkpoint_path)["inference_config"][
        "normalization"
    ]
    assert normalization["fields"] == NORMALIZATION_FIELDS


def test_checkpoint_normalization_field_mismatch_fails_fast(experiment, zarr_path):
    """A checkpoint whose normalization does not cover its fields is rejected."""
    write_checkpoint(
        experiment,
        filename="bad_norm.pt",
        contract=agent_contract(
            normalization_fields={
                "joint_state": "limits",
                "action": "auto",
            }
        ),
        zarr_file=zarr_path,
    )

    with pytest.raises(InvalidCheckpointError, match="normalization"):
        export_deployment_artifact(
            experiment, checkpoint_selector="bad_norm.pt", zarr_path=zarr_path
        )


def test_checkpoint_normalization_illegal_mode_fails_fast(experiment, zarr_path):
    """An illegal mode survives neither training nor deployment."""
    write_checkpoint(
        experiment,
        filename="banana.pt",
        contract=agent_contract(
            normalization_fields={
                "joint_state": "limits",
                "action": "banana",
                "point_cloud": "limits",
            }
        ),
        zarr_file=zarr_path,
    )

    with pytest.raises(InvalidCheckpointError, match="normalization"):
        export_deployment_artifact(
            experiment, checkpoint_selector="banana.pt", zarr_path=zarr_path
        )


def test_artifact_normalization_is_parsed_by_shared_contract(experiment, zarr_path):
    """A tampered artifact normalization is rejected by the shared parser."""
    receipt = _export(experiment, zarr_path)
    payload = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=True)
    payload["contract"]["inference_config"]["normalization"]["fields"]["action"] = (
        "identity"
    )

    with pytest.raises(DeploymentContractError, match="normalization"):
        parse_deployment_contract(payload)


# ---------------------------------------------------------------------------
# 4. nested _target_ validation happens at the contract boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile_target",
    [
        "os.system",
        "builtins.eval",
        "subprocess.Popen",
        "dexmani_policy.datasets.pc_dataset.PCDataset",
    ],
)
def test_hostile_nested_target_rejected_by_shared_validator(hostile_target):
    with pytest.raises(DeploymentContractError, match="deployment target"):
        validate_agent_targets(
            {
                "_target_": "dexmani_policy.agents.core.maniflow.ManiFlowAgent",
                "pc_encoder_config": {"_target_": hostile_target},
            }
        )


def test_hostile_target_fails_at_payload_validation():
    """Payload validation rejects a hostile nested _target_ before any instantiate."""
    payload = {
        "_format": "dexmani.deployment",
        "contract": {
            "inference_config": {
                "task_name": TASK_NAME,
                "action_key": "action",
                "action_dim": 19,
                "horizon": 4,
                "n_obs_steps": 2,
                "n_action_steps": 2,
                "use_aux_ee": False,
                "normalization": {
                    "version": 1,
                    "fields": dict(NORMALIZATION_FIELDS),
                },
                "agent": {
                    "_target_": "dexmani_policy.agents.core.maniflow.ManiFlowAgent",
                    "pc_encoder_config": {"_target_": "os.system"},
                },
                "eval": {"denoise_steps": 2},
            },
            "data_contract": {
                "dt": 0.1,
                "requires_hand": True,
                "observation_fields": {
                    "joint_state": {
                        "shape": [19],
                        "dtype": "float32",
                        "semantics": {},
                    }
                },
            },
            "producer": {},
        },
        "weights": {"probe": torch.ones(1)},
    }

    with pytest.raises(ArtifactVerificationError, match="metadata"):
        exporter._validate_payload(payload)


def test_hostile_target_fails_at_inspect_boundary(experiment, zarr_path):
    """``inspect_experiment`` rejects a hostile artifact with no model constructed."""
    from dexmani_policy.deployment.runtime import inspect_experiment

    receipt = _export(experiment, zarr_path)
    payload = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=True)
    payload["contract"]["inference_config"]["agent"]["pc_encoder_config"] = {
        "_target_": "os.system"
    }
    torch.save(payload, receipt.checkpoint_path)

    with pytest.raises(DeploymentRestoreError, match="deployment target"):
        inspect_experiment(experiment)


def test_export_rejects_checkpoint_with_hostile_agent_target(experiment, zarr_path):
    """The exporter's own sanitize step applies the shared target grammar."""
    checkpoint_path, _ = write_checkpoint(
        experiment, filename="hostile.pt", zarr_file=zarr_path
    )
    store = exporter.CheckpointStore(checkpoint_path.parent)
    checkpoint = store.load(checkpoint_path)
    checkpoint.resume_contract["agent_config"]["pc_encoder_config"] = {
        "_target_": "os.system"
    }
    store.save("hostile.pt", checkpoint)

    with pytest.raises(UnsupportedPolicyError, match="deployment target"):
        export_deployment_artifact(
            experiment, checkpoint_selector="hostile.pt", zarr_path=zarr_path
        )


# ---------------------------------------------------------------------------
# 5. selected state only
# ---------------------------------------------------------------------------


def test_raw_selected_does_not_require_ema(experiment, zarr_path):
    """Selecting raw weights must not care that EMA is absent."""
    write_checkpoint(
        experiment, filename="raw_only.pt", with_ema=False, zarr_file=zarr_path
    )
    write_config(experiment, use_ema=False)

    receipt = export_deployment_artifact(
        experiment, checkpoint_selector="raw_only.pt", zarr_path=zarr_path
    )

    producer = _contract(receipt.checkpoint_path)["producer"]
    assert producer["selected_weights"] == "model"
    assert producer["source_checkpoint"] == "raw_only.pt"


def test_ema_selected_without_ema_weights_fails(experiment, zarr_path):
    """Selecting EMA must fail loudly when the checkpoint has none."""
    write_checkpoint(
        experiment, filename="raw_only.pt", with_ema=False, zarr_file=zarr_path
    )
    write_config(experiment, use_ema=True)

    with pytest.raises(InvalidCheckpointError, match="ema_model"):
        export_deployment_artifact(
            experiment, checkpoint_selector="raw_only.pt", zarr_path=zarr_path
        )
    assert not list((experiment / "checkpoints").glob("*deployment*.pt"))


def test_selected_weights_match_the_stored_state(experiment, zarr_path):
    """``producer.selected_weights`` must name the state actually serialized."""
    write_checkpoint(
        experiment, filename="both.pt", with_ema=True, zarr_file=zarr_path
    )
    write_config(experiment, use_ema=True)

    receipt = export_deployment_artifact(
        experiment, checkpoint_selector="both.pt", zarr_path=zarr_path
    )
    payload = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = exporter._load_training_checkpoint(
        experiment / "checkpoints" / "both.pt"
    )

    assert payload["contract"]["producer"]["selected_weights"] == "ema_model"
    assert payload["contract"]["inference_config"]["eval"]["denoise_steps"] > 0
    for key, tensor in payload["weights"].items():
        assert torch.equal(tensor, checkpoint.ema_model_state[key])


def test_unselected_state_is_never_processed(experiment, zarr_path):
    """A corrupt unselected state cannot break an export that does not use it."""
    write_checkpoint(
        experiment, filename="corrupt_ema.pt", with_ema=True, zarr_file=zarr_path
    )
    checkpoint_path = experiment / "checkpoints" / "corrupt_ema.pt"
    store = exporter.CheckpointStore(checkpoint_path.parent)
    checkpoint = store.load(checkpoint_path)
    # EMA is broken (non-canonical keys); raw is selected, so it must not matter.
    checkpoint.ema_model_state = {"module._orig_mod.broken": torch.ones(1)}
    store.save("corrupt_ema.pt", checkpoint)
    write_config(experiment, use_ema=False)

    receipt = export_deployment_artifact(
        experiment, checkpoint_selector="corrupt_ema.pt", zarr_path=zarr_path
    )

    assert _contract(receipt.checkpoint_path)["producer"]["selected_weights"] == "model"


# ---------------------------------------------------------------------------
# 7. public export always verifies and always publishes
# ---------------------------------------------------------------------------


def test_public_export_has_no_verify_switch(monkeypatch):
    """No public API or CLI flag can publish an unverified artifact."""
    signature = inspect.signature(export_deployment_artifact)
    assert "verify" not in signature.parameters
    assert set(signature.parameters) == {
        "experiment_dir",
        "checkpoint_selector",
        "output_path",
        "zarr_path",
    }
    monkeypatch.setattr("sys.argv", ["export", "/tmp/experiment"])
    parsed = exporter._parse_args()
    assert not hasattr(parsed, "verify")
    source = Path(exporter.__file__).read_text()
    assert "--no-verify" not in source
    assert '"--verify"' not in source
    assert "verify: bool" not in source


def test_public_export_publishes_selector(experiment, zarr_path):
    """A successful public export always leaves deployment_latest.pt published."""
    receipt = _export(experiment, zarr_path)

    selector = experiment / "checkpoints" / "deployment_latest.pt"
    assert selector.is_symlink()
    assert selector.resolve() == receipt.checkpoint_path.resolve()


def test_export_failure_is_reported_before_publishing(experiment, zarr_path, monkeypatch):
    """A strict-restore failure must not publish and must leave no candidate."""
    write_checkpoint(experiment, filename="ok.pt", zarr_file=zarr_path)
    selector = experiment / "checkpoints" / "deployment_latest.pt"

    def _fail(_payload):
        raise ArtifactVerificationError("synthetic strict restore failure")

    monkeypatch.setattr(exporter, "_verify_exported_model", _fail)

    with pytest.raises(ArtifactVerificationError):
        export_deployment_artifact(
            experiment, checkpoint_selector="ok.pt", zarr_path=zarr_path
        )

    assert not selector.exists()
    assert not list((experiment / "checkpoints").glob("ok-deployment.pt"))


def test_verification_failure_keeps_previous_selector_and_allows_retry(
    experiment, zarr_path, monkeypatch
):
    """Failure must leave the old selector intact and the same command retryable."""
    write_checkpoint(experiment, filename="first.pt", zarr_file=zarr_path)
    write_checkpoint(experiment, filename="second.pt", zarr_file=zarr_path)
    first = export_deployment_artifact(
        experiment, checkpoint_selector="first.pt", zarr_path=zarr_path
    )
    selector = first.selector_path
    previous_target = selector.resolve()

    real_verify = exporter._verify_exported_model
    calls = {"count": 0}

    def _fail_once(payload):
        calls["count"] += 1
        raise ArtifactVerificationError("synthetic strict restore failure")

    monkeypatch.setattr(exporter, "_verify_exported_model", _fail_once)
    with pytest.raises(ArtifactVerificationError):
        export_deployment_artifact(
            experiment, checkpoint_selector="second.pt", zarr_path=zarr_path
        )

    # Selector unchanged, candidate gone, identical command now succeeds.
    assert selector.resolve() == previous_target
    assert not (experiment / "checkpoints" / "second-deployment.pt").exists()
    monkeypatch.setattr(exporter, "_verify_exported_model", real_verify)
    retried = export_deployment_artifact(
        experiment, checkpoint_selector="second.pt", zarr_path=zarr_path
    )
    assert selector.resolve() == retried.checkpoint_path.resolve()


def test_selector_publication_is_not_package_public_api():
    """Selector publication is an internal step of ``export_deployment_artifact``.

    The swap helper performs no reload/restore/prediction, so the package
    facade must not expose it as a researcher-facing publication API.
    """
    import dexmani_policy.deployment as deployment

    assert "publish_deployment_selector" not in deployment.__all__
    assert "publish_deployment_selector" not in deployment._EXPORT_NAMES
    assert not hasattr(deployment, "publish_deployment_selector")


# ---------------------------------------------------------------------------
# 9. cleanup failure is explicit
# ---------------------------------------------------------------------------


def test_cleanup_failure_is_not_silently_swallowed(tmp_path):
    """A failing candidate cleanup must raise, not pass silently."""
    candidate = tmp_path / "candidate.pt"
    candidate.mkdir()  # a directory cannot be unlinked as a file

    with pytest.raises(ArtifactPublicationError, match="manually"):
        exporter.cleanup_candidate_artifact(candidate)


def test_cleanup_removes_candidate(tmp_path):
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"x")

    exporter.cleanup_candidate_artifact(candidate)

    assert not candidate.exists()


# ---------------------------------------------------------------------------
# 6. DQ-RISE persistent-codebook state validation
# ---------------------------------------------------------------------------


DQ_RISE_TRAIN = {
    "action_dim": 21,
    "tcp_dim": 9,
    "hand_dim": 12,
    "action_key": "action_ee",
}
DQ_RISE_AGENT_CONFIG = {
    "_target_": "dexmani_policy.agents.core.dqrise.DQRISEAgent",
    "codebook_path": "robot_data/sorted_hand_poses.npz",
    "codebook_num_groups": 2,
    "codebook_size": 4,
    "tcp_dim": 9,
}


def _dqrise_selected_state(*, hand_scale: float = 1.0) -> dict:
    """A real DQ-RISE state dict with a complete persistent runtime codebook."""
    from dexmani_policy.agents.core.dqrise import DQRISEAgent

    agent = DQRISEAgent(
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        action_dim=21,
        tcp_dim=9,
        codebook_num_groups=2,
        codebook_size=4,
        pc_dim=6,
        state_dim=19,
        num_points=1024,
        pc_out_dim=16,
        state_out_dim=8,
        down_dims=(16,),
        n_groups=8,
    )
    manager = agent.codebook_manager
    manager.sorted_hand_poses = torch.randn(16, 12)
    manager.pca_permutation = torch.arange(16, dtype=torch.long)
    manager.layer_weights = torch.ones(2)
    manager.hand_normalizer_scale = torch.full((12,), hand_scale)
    manager.hand_normalizer_offset = torch.zeros(12)
    manager.hand_min = torch.tensor(0.0)
    manager.hand_max = torch.tensor(65535.0)
    state = dict(agent.state_dict())
    # The policy action normalizer's hand tail must agree with the codebook's.
    state["normalizer.params_dict.action.scale"] = torch.cat(
        [torch.ones(9), torch.full((12,), hand_scale)]
    )
    state["normalizer.params_dict.action.offset"] = torch.zeros(21)
    return state


def test_dqrise_selected_state_accepts_complete_persistent_codebook():
    """A complete persistent codebook sanitizes to a codebook-free constructor."""
    from dexmani_policy.deployment.export import _sanitize_agent_config

    sanitized = _sanitize_agent_config(
        dict(DQ_RISE_AGENT_CONFIG), _dqrise_selected_state(), DQ_RISE_TRAIN
    )

    assert sanitized["codebook_path"] is None


@pytest.mark.parametrize(
    "missing",
    [
        "codebook_manager.sorted_hand_poses",
        "codebook_manager.pca_permutation",
        "codebook_manager.layer_weights",
        "codebook_manager.hand_normalizer_scale",
        "codebook_manager.hand_normalizer_offset",
        "codebook_manager.hand_min",
        "codebook_manager.hand_max",
        "normalizer.params_dict.action.scale",
        "normalizer.params_dict.action.offset",
    ],
)
def test_dqrise_selected_state_requires_every_persistent_tensor(missing):
    """Each persistent runtime codebook/normalizer tensor is mandatory."""
    from dexmani_policy.deployment.export import _sanitize_agent_config

    incomplete = {
        k: v for k, v in _dqrise_selected_state().items() if k != missing
    }

    with pytest.raises(UnsupportedPolicyError, match="codebook|normalizer"):
        _sanitize_agent_config(dict(DQ_RISE_AGENT_CONFIG), incomplete, DQ_RISE_TRAIN)


def test_dqrise_codebook_hand_normalizer_must_match_policy_normalizer():
    """The codebook hand normalizer must agree with the policy action normalizer."""
    from dexmani_policy.deployment.export import _sanitize_agent_config

    state = _dqrise_selected_state()
    # Action scale's hand tail deliberately disagrees with the codebook's.
    state["normalizer.params_dict.action.scale"] = torch.cat(
        [torch.ones(9), torch.full((12,), 2.0)]
    )

    with pytest.raises(InvalidCheckpointError, match="hand normalizer"):
        _sanitize_agent_config(dict(DQ_RISE_AGENT_CONFIG), state, DQ_RISE_TRAIN)


def test_dqrise_deployment_constructor_matches_validated_state():
    """Sanitizing must disable only the codebook path, keeping parity inputs equal."""
    from dexmani_policy.deployment.export import _sanitize_agent_config

    state = _dqrise_selected_state()
    sanitized = _sanitize_agent_config(
        dict(DQ_RISE_AGENT_CONFIG), state, DQ_RISE_TRAIN
    )

    assert set(sanitized) == set(DQ_RISE_AGENT_CONFIG)
    differences = {
        key
        for key in sanitized
        if sanitized[key] != DQ_RISE_AGENT_CONFIG[key]
    }
    assert differences == {"codebook_path"}


def test_sanitize_disables_constructor_pretrained_loading():
    """Export must never re-run constructor-time pretrained weight loading."""
    from dexmani_policy.deployment.export import _sanitize_agent_config

    cfg = agent_config()
    cfg["pc_encoder_config"]["use_pretrained_weights"] = True
    cfg["pc_encoder_config"]["pretrained_path"] = "robot_data/pretrained/pc.pt"

    sanitized = _sanitize_agent_config(cfg, {"probe": torch.ones(1)}, {})

    assert sanitized["pc_encoder_config"]["use_pretrained_weights"] is False


# ---------------------------------------------------------------------------
# 11. runtime compatibility regression
# ---------------------------------------------------------------------------


def test_loaded_policy_predict_contract_is_unchanged(experiment, zarr_path):
    """``LoadedPolicy.predict`` still returns [n_action_steps, control_action_dim]."""
    from dexmani_policy.deployment.runtime import load_experiment

    _export(experiment, zarr_path)
    policy = load_experiment(experiment, device="cpu")
    try:
        spec = policy.spec
        observation = {
            field.name: np.zeros(
                (spec.n_obs_steps, *field.shape), dtype=np.dtype(field.dtype)
            )
            for field in spec.observation_fields
        }
        control = policy.predict(observation)

        assert control.shape == (spec.n_action_steps, spec.control_action_dim)
        assert control.dtype == np.float64
        assert np.isfinite(control).all()
    finally:
        policy.close()


def test_loaded_policy_reports_artifact_identity(experiment, zarr_path):
    """The runtime still exposes the artifact's own contract."""
    from dexmani_policy.deployment.runtime import inspect_experiment, load_experiment

    receipt = _export(experiment, zarr_path)
    info = inspect_experiment(experiment)
    policy = load_experiment(experiment, device="cpu")
    try:
        assert info.task_name == TASK_NAME
        assert info.checkpoint_name == receipt.checkpoint_path.name
        assert info.checkpoint_name.endswith("-deployment.pt")
        assert policy.spec.action_dim == 19
        assert policy.spec.control_action_dim == 19
        assert policy.spec.default_inference_steps == 2
    finally:
        policy.close()

    # inspect resolves the selector to the immutable filename; a Real session
    # pins exactly that resolved name for its own load.
    pinned = load_experiment(experiment, device="cpu", artifact=info.checkpoint_name)
    pinned.close()


# ---------------------------------------------------------------------------
# targeted parity: direct checkpoint restore == exported artifact restore
# ---------------------------------------------------------------------------


def _direct_prediction_snapshot(
    agent_config: dict, action_key: str, payload: dict, state: dict
):
    """One seeded prediction from a directly restored constructor plus state.

    The DeploymentSpec comes from the artifact payload: it is checkpoint-owned
    metadata, which is exactly the invariant under test — the direct restore
    and the artifact restore must agree when both run the same contract.
    """
    import hydra
    from omegaconf import OmegaConf

    from dexmani_policy.deployment.restore import (
        RestoredDeployment,
        deployment_spec,
        prediction_snapshot,
    )

    agent = hydra.utils.instantiate(OmegaConf.create(agent_config))
    agent.load_state_dict(state, strict=True)
    agent.action_key = action_key
    agent.eval()
    spec = deployment_spec(payload)
    return prediction_snapshot(RestoredDeployment(agent=agent, spec=spec), seed=0)


def test_exported_restore_prediction_matches_direct_checkpoint_restore(
    experiment, zarr_path
):
    """Research parity: the exported artifact predicts exactly like the checkpoint."""
    from dexmani_policy.deployment.restore import (
        assert_prediction_parity,
        prediction_snapshot,
        restore_deployment_agent,
    )

    receipt = _export(experiment, zarr_path)
    payload = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=True)
    artifact = prediction_snapshot(restore_deployment_agent(payload), seed=0)

    checkpoint = exporter._load_training_checkpoint(
        experiment / "checkpoints" / "latest.pt"
    )
    resume = checkpoint.resume_contract
    direct = _direct_prediction_snapshot(
        resume["agent_config"],
        resume["agent"]["action_key"],
        payload,
        checkpoint.model_state,
    )

    assert_prediction_parity(artifact, direct)


def test_rgb_artifact_round_trips_through_frozen_contract(tmp_path, experiment):
    """A full RGB export/restore keeps nested constructor data intact."""
    from dexmani_policy.deployment.runtime import load_experiment

    zarr_file = tmp_path / "rgb_task.zarr"
    write_zarr(zarr_file)
    add_rgb(zarr_file)

    rgb_agent = {
        "_target_": "dexmani_policy.agents.core.dp.DPAgent",
        "horizon": 4,
        "n_obs_steps": 2,
        "n_action_steps": 2,
        "action_dim": 19,
        "rgb_backbone_name": "resnet",
        # A nested mapping, which must survive the frozen-contract round trip.
        "rgb_backbone_config": {"image_size": [32, 32], "interpolation": "bilinear"},
        "state_dim": 19,
        "state_out_dim": 8,
        "down_dims": [16],
        "diffusion_step_embed_dim": 8,
        "n_groups": 8,
        "kernel_size": 3,
        "num_training_steps": 10,
        "num_inference_steps": 2,
        "prediction_type": "sample",
    }
    rgb_dataset = dataset_config(
        _target_="dexmani_policy.datasets.rgb_dataset.RGBDataset",
        sensor_modalities=["joint_state", "rgb"],
        rgb_preprocess_size=[48, 48],
        rgb_random_crop_size=[32, 32],
        zarr_path=str(zarr_file),
    )
    contract = agent_contract(
        normalization_fields={
            "joint_state": "limits",
            "action": "auto",
            "rgb": "identity",
        }
    )
    write_config(
        experiment,
        agent_cfg=rgb_agent,
        dataset_cfg=rgb_dataset,
        normalization=dict(contract["normalization"]["fields"]),
        use_ema=False,
    )
    write_checkpoint(
        experiment, agent_cfg=rgb_agent, dataset_cfg=rgb_dataset, contract=contract
    )

    receipt = export_deployment_artifact(
        experiment, checkpoint_selector="latest.pt", zarr_path=zarr_file
    )
    exported = _contract(receipt.checkpoint_path)["inference_config"]["agent"]
    assert exported["rgb_backbone_config"] == {
        "image_size": [32, 32],
        "interpolation": "bilinear",
    }

    policy = load_experiment(experiment, device="cpu")
    try:
        spec = policy.spec
        observation = {
            field.name: np.zeros(
                (spec.n_obs_steps, *field.shape), dtype=np.dtype(field.dtype)
            )
            for field in spec.observation_fields
        }
        observation["rgb"] = np.full((spec.n_obs_steps, 48, 48, 3), 7, dtype=np.uint8)
        control = policy.predict(observation)

        assert control.shape == (spec.n_action_steps, spec.control_action_dim)
        assert spec.rgb_preprocessing.resize_hw == (48, 48)
        assert spec.rgb_preprocessing.center_crop_hw == (32, 32)
        assert spec.rgb_preprocessing.processor_image_size_hw == (32, 32)
    finally:
        policy.close()


DQ_RISE_FULL_AGENT_CONFIG = {
    "_target_": "dexmani_policy.agents.core.dqrise.DQRISEAgent",
    "horizon": 4,
    "n_obs_steps": 2,
    "n_action_steps": 2,
    "action_dim": 21,
    "tcp_dim": 9,
    "codebook_path": "robot_data/sorted_hand_poses.npz",
    "codebook_num_groups": 2,
    "codebook_size": 4,
    "encoder_type": "idp3",
    "pc_dim": 6,
    "pc_out_dim": 16,
    "state_dim": 19,
    "num_points": 1024,
    "state_out_dim": 8,
    "down_dims": [16],
    "diffusion_step_embed_dim": 8,
    "n_groups": 8,
    "kernel_size": 3,
    "num_training_steps": 10,
    "num_inference_steps": 2,
    "prediction_type": "sample",
}


def test_dqrise_full_export_round_trip(tmp_path):
    """A DQ-RISE checkpoint exports, verifies and restores with its codebook inside."""
    from conftest import build_agent
    from dexmani_policy.deployment.runtime import load_experiment

    experiment_dir = tmp_path / "experiments" / "dqrise" / TASK_NAME / "run"
    (experiment_dir / "checkpoints").mkdir(parents=True)
    zarr_file = tmp_path / f"{TASK_NAME}.zarr"
    write_zarr(zarr_file)

    # Build with the codebook path disabled so no .npz is needed; the persistent
    # codebook tensors live in the state dict, which is what deployment uses.
    agent = build_agent(
        {**DQ_RISE_FULL_AGENT_CONFIG, "codebook_path": None},
        ["joint_state", "point_cloud"],
    )
    manager = agent.codebook_manager
    hand_scale = torch.full((12,), 1.5)
    manager.sorted_hand_poses = torch.randn(16, 12)
    manager.pca_permutation = torch.arange(16, dtype=torch.long)
    manager.layer_weights = torch.ones(2)
    manager.hand_normalizer_scale = hand_scale
    manager.hand_normalizer_offset = torch.zeros(12)
    manager.hand_min = torch.tensor(0.0)
    manager.hand_max = torch.tensor(65535.0)
    agent.normalizer.params_dict["action"] = torch.nn.ParameterDict(
        {
            "scale": torch.cat([torch.ones(9), hand_scale]),
            "offset": torch.zeros(21),
        }
    )
    state = {key: value.detach().clone() for key, value in agent.state_dict().items()}

    dataset_cfg = dataset_config(action_key="action_ee", zarr_path=str(zarr_file))
    contract = agent_contract(
        action_key="action_ee",
        action_dim=21,
        control_action_dim=21,
        tcp_dim=9,
        hand_dim=12,
    )
    write_config(
        experiment_dir,
        agent_cfg=DQ_RISE_FULL_AGENT_CONFIG,
        dataset_cfg=dataset_cfg,
        normalization=dict(contract["normalization"]["fields"]),
        use_ema=False,
    )
    write_checkpoint(
        experiment_dir,
        agent_cfg=DQ_RISE_FULL_AGENT_CONFIG,
        dataset_cfg=dataset_cfg,
        contract=contract,
        state_dict=state,
    )

    receipt = export_deployment_artifact(
        experiment_dir, checkpoint_selector="latest.pt", zarr_path=zarr_file
    )

    payload = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=True)
    exported_agent = payload["contract"]["inference_config"]["agent"]
    # The deployment constructor carries no codebook path...
    assert exported_agent["codebook_path"] is None
    # ...while the weights stay in the artifact.
    assert "codebook_manager.sorted_hand_poses" in payload["weights"]
    assert payload["weights"]["codebook_manager.sorted_hand_poses"].shape == (16, 12)

    policy = load_experiment(experiment_dir, device="cpu")
    try:
        spec = policy.spec
        observation = {
            field.name: np.zeros(
                (spec.n_obs_steps, *field.shape), dtype=np.dtype(field.dtype)
            )
            for field in spec.observation_fields
        }
        control = policy.predict(observation)

        assert spec.action_dim == 21
        assert spec.control_action_dim == 21
        assert control.shape == (spec.n_action_steps, spec.control_action_dim)
        assert np.isfinite(control).all()
    finally:
        policy.close()

    # Targeted parity: the sanitized codebook-free constructor plus the
    # persistent codebook state predicts exactly like the artifact restore.
    from dexmani_policy.deployment.restore import (
        assert_prediction_parity,
        prediction_snapshot,
        restore_deployment_agent,
    )

    sanitized_cfg = exporter._sanitize_agent_config(
        dict(DQ_RISE_FULL_AGENT_CONFIG), state, dict(DQ_RISE_TRAIN)
    )
    direct = _direct_prediction_snapshot(sanitized_cfg, "action_ee", payload, state)
    artifact = prediction_snapshot(restore_deployment_agent(payload), seed=0)
    assert_prediction_parity(artifact, direct)


# ---------------------------------------------------------------------------
# checkpoint-internal validation (deleting reconciliation != deleting validation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "overrides", "message"),
    [
        (
            "window_exceeds_horizon",
            {"n_obs_steps": 4, "n_action_steps": 8, "horizon": 4},
            "window exceeds horizon",
        ),
        (
            "action_dim_vs_action_key",
            {"action_dim": 19, "action_key": "action_ee", "control_action_dim": 21},
            "action_dim does not match action_key",
        ),
        (
            "control_dim_mismatch",
            {"action_dim": 19, "control_action_dim": 21},
            "control_action_dim",
        ),
        (
            "tcp_plus_hand",
            {"tcp_dim": 9, "hand_dim": 5},
            "tcp_dim \\+ hand_dim",
        ),
        (
            "tcp_without_hand",
            {"tcp_dim": 9, "hand_dim": None},
            "both be set or both be null",
        ),
        (
            "aux_ee_layout",
            {"use_aux_ee": True, "action_dim": 19},
            "joint19_ee9",
        ),
        ("non_positive_horizon", {"horizon": -1}, "positive integer"),
        ("unsupported_action_key", {"action_key": "banana"}, "action_key is unsupported"),
    ],
)
def test_checkpoint_internal_contract_is_strictly_parsed(
    experiment, zarr_path, label, overrides, message
):
    """A checkpoint whose own action/window contract is inconsistent must fail."""
    # An invalid action_key cannot seed a real snapshot; the export must fail
    # on the checkpoint's own contract before the snapshot gate is reached.
    semantics = (
        "auto"
        if overrides.get("action_key", "action") in {"action", "action_ee"}
        else None
    )
    path, _ = write_checkpoint(
        experiment,
        filename="bad.pt",
        contract=agent_contract(**overrides),
        zarr_file=zarr_path,
        deployment_data_semantics=semantics,
    )

    with pytest.raises(InvalidCheckpointError, match=message):
        exporter._parse_checkpoint_deployment_source(
            exporter._load_training_checkpoint(path)
        )


def test_checkpoint_missing_normalization_contract_fails(experiment, zarr_path):
    """A checkpoint without a versioned normalization contract cannot deploy."""
    contract = agent_contract()
    del contract["normalization"]
    path, _ = write_checkpoint(
        experiment, filename="no_norm.pt", contract=contract, zarr_file=zarr_path
    )

    with pytest.raises(InvalidCheckpointError, match="normalization"):
        exporter._parse_checkpoint_deployment_source(
            exporter._load_training_checkpoint(path)
        )


def test_checkpoint_missing_required_action_field_fails(experiment, zarr_path):
    """Every action/window field is mandatory in the saved contract."""
    contract = agent_contract()
    del contract["control_action_dim"]
    path, _ = write_checkpoint(
        experiment, filename="short.pt", contract=contract, zarr_file=zarr_path
    )

    with pytest.raises(InvalidCheckpointError, match="control_action_dim"):
        exporter._parse_checkpoint_deployment_source(
            exporter._load_training_checkpoint(path)
        )


def test_checkpoint_missing_dataset_contract_fails(experiment, zarr_path):
    """A checkpoint with no saved dataset semantics cannot deploy."""
    path, _ = write_checkpoint(experiment, filename="no_ds.pt", zarr_file=zarr_path)
    store = exporter.CheckpointStore(path.parent)
    checkpoint = store.load(path)
    del checkpoint.resume_contract["dataset"]
    store.save("no_ds.pt", checkpoint)

    with pytest.raises(InvalidCheckpointError, match="resume_contract.dataset"):
        exporter._parse_checkpoint_deployment_source(
            exporter._load_training_checkpoint(path)
        )


def test_artifact_records_producer_provenance(experiment, zarr_path):
    """producer records the source checkpoint, selected weights and best-effort commit."""
    receipt = _export(experiment, zarr_path)

    producer = _contract(receipt.checkpoint_path)["producer"]
    assert producer["source_checkpoint"] == "latest.pt"
    assert producer["selected_weights"] == "model"
    commit = producer.get("source_commit")
    if commit is not None:
        assert len(commit) == 40


def test_source_commit_is_optional(tmp_path, monkeypatch):
    """An unavailable git revision must not block an export."""
    assert exporter._source_commit(tmp_path) is None


# ---------------------------------------------------------------------------
# publication: atomic selector, explicit refusal, retryable failures
# ---------------------------------------------------------------------------


def test_publish_refuses_to_replace_a_regular_file_selector(experiment, zarr_path):
    """A non-symlink ``deployment_latest.pt`` must not be clobbered or leaked."""
    first = export_deployment_artifact(
        experiment, checkpoint_selector="latest.pt", zarr_path=zarr_path
    )
    selector = first.selector_path
    # Materialize the selector as a regular file, which publish refuses to replace.
    selector.unlink()
    selector.write_bytes(b"not a symlink")
    write_checkpoint(experiment, filename="second.pt", zarr_file=zarr_path)

    with pytest.raises(ArtifactPublicationError, match="non-symlink"):
        export_deployment_artifact(
            experiment, checkpoint_selector="second.pt", zarr_path=zarr_path
        )

    # No leaked candidate: removing the bad selector is enough to retry cleanly.
    assert not (experiment / "checkpoints" / "second-deployment.pt").exists()
    selector.unlink()
    retried = export_deployment_artifact(
        experiment, checkpoint_selector="second.pt", zarr_path=zarr_path
    )
    assert retried.selector_path.is_symlink()
    assert retried.selector_path.resolve() == retried.checkpoint_path.resolve()


# ---------------------------------------------------------------------------
# target allowlist container coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "container",
    [
        lambda hostile: [hostile],
        lambda hostile: (hostile,),
        lambda hostile: {"nested": (hostile,)},
    ],
    ids=["list", "tuple", "dict-of-tuple"],
)
def test_allowlist_walks_every_persisted_container(container):
    """Every container a persisted artifact can carry must be walked."""
    hostile = {"_target_": "os.system"}

    with pytest.raises(DeploymentContractError, match="deployment target"):
        validate_agent_targets(
            {
                "_target_": "dexmani_policy.agents.core.maniflow.ManiFlowAgent",
                "pc_encoder_config": container(hostile),
            }
        )


def test_allowlist_walks_mapping_subclasses():
    """A Mapping subclass (kept by weights_only unpickling) must be walked too."""
    from collections import OrderedDict

    with pytest.raises(DeploymentContractError, match="deployment target"):
        validate_agent_targets(
            {
                "_target_": "dexmani_policy.agents.core.maniflow.ManiFlowAgent",
                "pc_encoder_config": OrderedDict({"_target_": "os.system"}),
            }
        )


def test_tuple_wrapped_target_is_rejected_at_inspect(experiment, zarr_path):
    """A tuple-wrapped hostile target must fail at inspect, not at instantiate."""
    from dexmani_policy.deployment.runtime import inspect_experiment

    receipt = _export(experiment, zarr_path)
    payload = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=True)
    payload["contract"]["inference_config"]["agent"]["_hostile"] = (
        {"_target_": "tempfile.mkdtemp"},
    )
    torch.save(payload, receipt.checkpoint_path)

    with pytest.raises(DeploymentRestoreError, match="deployment target"):
        inspect_experiment(experiment)


def test_source_commit_is_omitted_for_a_nested_repository(tmp_path):
    """A package tree inside another git repo must not record the outer HEAD."""
    import subprocess

    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(
        ["git", "-C", str(outer), "commit", "-q", "--allow-empty", "-m", "x"],
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@e",
            "HOME": str(tmp_path),
        },
    )

    # inner is inside a repo but is not a repo root: provenance must be omitted
    # rather than silently naming the unrelated outer commit.
    assert exporter._source_commit(inner) is None
    # The genuine repository root still reports its own revision.
    assert exporter._source_commit(outer) is not None
