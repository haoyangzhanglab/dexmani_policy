"""Evidence-backed first-phase Policy deployment qualification matrix.

This is descriptive test ownership, not an Agent registry or construction
factory.  A policy moves to ``qualified`` only when the referenced evidence
has actually run; conditional entries state the checkpoint requirements that
must be rechecked for every exported experiment.
"""

from __future__ import annotations

from typing import Final

QUALIFICATION_MATRIX: Final[dict[str, dict[str, object]]] = {
    "dp3": {
        "status": "conditional",
        "evidence": [
            "synthetic_direct_export_exact_parity",
            "current_real_tiny_restore",
        ],
        "condition": "real checkpoint parity and clean-tree Real restore must pass",
    },
    "action_flow": {
        "status": "conditional",
        "evidence": ["config_dimension_contract"],
        "condition": (
            "actual-agent parity requires pytorch3d sample_farthest_points, "
            "knn_points, and ball_query"
        ),
    },
    "dqrise": {
        "status": "conditional",
        "evidence": ["synthetic_asset_removed_exact_parity", "network_forbidden"],
        "condition": (
            "selected checkpoint must contain the complete seven-buffer runtime "
            "codebook and a matching policy/codebook hand normalizer"
        ),
    },
    "r3d": {
        "status": "conditional",
        "evidence": [
            "synthetic_no_pretrained_exact_parity",
            "synthetic_aux_ee_full_and_control_parity",
            "network_forbidden",
        ],
        "condition": "strict restore must pass with use_pretrained_weights=false",
    },
    "maniflow": {
        "status": "deferred",
        "evidence": [],
        "condition": "no PR-3 direct/export parity evidence ran",
    },
    "sat": {
        "status": "deferred",
        "evidence": [],
        "condition": "no PR-3 direct/export parity evidence ran",
    },
    "dp_rgb": {
        "status": "deferred",
        "evidence": [],
        "condition": "RGB preprocessing deployment contract is not defined",
    },
    "moe_dp_rgb": {
        "status": "deferred",
        "evidence": [],
        "condition": "RGB preprocessing and external initialization are deferred",
    },
    "multitask_dit": {
        "status": "deferred",
        "evidence": [],
        "condition": "dynamic task-text deployment contract is not defined",
    },
}
