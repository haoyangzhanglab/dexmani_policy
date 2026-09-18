#!/usr/bin/env python
"""One-time repair for legacy dexmani_real Policy Zarr datasets that predate
the mandatory ``action_ee_components`` metadata.

Current ``dexmani_real`` producers already write the attr; this script exists
only for historical Real Zarr stores whose ``data/action_ee`` arrays already
follow the canonical 21-D layout::

    eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)

It patches exactly one explicitly named Zarr and only ever writes
``root.attrs["action_ee_components"]`` — array data is never read, recomputed
or modified.  There is deliberately no --force, no recursion and no
migration registry: a conflicting existing value is a hard error.

Usage::

    python scripts/data/patch_legacy_action_ee_components.py <zarr_path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import zarr

ACTION_EE_COMPONENTS = (
    "eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)"
)

_SCHEMA_NAME = "dexmani-real-policy-zarr"
_DOMAIN = "real"
_ACTION_SEMANTICS = "teleop_published_joint_target"
_ACTION_EE_FRAME = "xarm_base"


def patch(path: Path) -> bool:
    """Repair one Zarr's ``action_ee_components`` attr.

    Returns True if the attr was written, False if it was already canonical
    (no modification).  Raises ValueError for every refusal; nothing is
    written unless all checks pass.
    """
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)

    # C. Real Policy data only — never touch Sim Zarr.
    if attrs.get("schema_name") != _SCHEMA_NAME or attrs.get("domain") != _DOMAIN:
        raise ValueError(
            f"refusing to patch a non-Real Policy Zarr: {path} "
            f"(schema_name={attrs.get('schema_name')!r}, domain={attrs.get('domain')!r})"
        )

    # A. data/action_ee must exist and be an array.
    if "data/action_ee" not in root or not isinstance(root["data/action_ee"], zarr.Array):
        raise ValueError(f"{path}: required array data/action_ee is missing")

    # B. Defensive shape gate: 2-D [T>0, 21].  Metadata only — no data read.
    action_ee = root["data/action_ee"]
    if (
        action_ee.ndim != 2
        or action_ee.shape[1] != 21
        or action_ee.shape[0] <= 0
    ):
        raise ValueError(
            f"{path}: data/action_ee must be 2-D with shape [T>0, 21], "
            f"got {tuple(action_ee.shape)}"
        )

    # D. Existing action semantics must already match the Real contract.
    if attrs.get("action_semantics") != _ACTION_SEMANTICS:
        raise ValueError(
            f"{path}: action_semantics must be {_ACTION_SEMANTICS!r}, "
            f"got {attrs.get('action_semantics')!r}"
        )
    if attrs.get("action_ee_frame") != _ACTION_EE_FRAME:
        raise ValueError(
            f"{path}: action_ee_frame must be {_ACTION_EE_FRAME!r}, "
            f"got {attrs.get('action_ee_frame')!r}"
        )

    if "action_ee_components" in attrs:
        existing = attrs["action_ee_components"]
        if existing == ACTION_EE_COMPONENTS:
            return False  # already canonical — idempotent no-op
        # Present but different (including an explicit JSON null) is a
        # conflict — never overwrite, never guess.
        raise ValueError(
            "refusing to overwrite conflicting action_ee_components:\n"
            f"existing={existing}\n"
            f"expected={ACTION_EE_COMPONENTS}"
        )

    zarr.open_group(str(path), mode="a").attrs["action_ee_components"] = (
        ACTION_EE_COMPONENTS
    )

    # Post-write self-verification: re-read and require exact equality.
    written = zarr.open_group(str(path), mode="r").attrs.get("action_ee_components")
    if written != ACTION_EE_COMPONENTS:
        raise ValueError(
            f"{path}: post-write verification failed, attr reads back as {written!r}"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-time repair for legacy dexmani_real Policy Zarr datasets that "
            "predate the mandatory action_ee_components metadata."
        )
    )
    parser.add_argument(
        "path", type=Path, help="the single legacy Real Policy Zarr to repair"
    )
    args = parser.parse_args()

    try:
        written = patch(args.path)
    except (ValueError, KeyError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if written:
        print(f"patched: {args.path}")
        print(f"action_ee_components={ACTION_EE_COMPONENTS}")
    else:
        print(f"already canonical: {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
