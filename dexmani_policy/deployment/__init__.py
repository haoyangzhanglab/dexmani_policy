"""Policy-native deployment artifact export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dexmani_policy.deployment.export import (
        ExportReceipt,
        export_deployment_artifact,
        publish_deployment_selector,
    )
    from dexmani_policy.deployment.runtime import (
        ExperimentInfo,
        LoadedPolicy,
        PolicySpec,
        inspect_experiment,
        list_experiments,
        load_experiment,
        resolve_experiment,
    )

__all__ = [
    "ExperimentInfo",
    "ExportReceipt",
    "LoadedPolicy",
    "PolicySpec",
    "export_deployment_artifact",
    "inspect_experiment",
    "list_experiments",
    "load_experiment",
    "publish_deployment_selector",
    "resolve_experiment",
]

_EXPORT_NAMES = {
    "ExportReceipt",
    "export_deployment_artifact",
    "publish_deployment_selector",
}
_RUNTIME_NAMES = {
    "ExperimentInfo",
    "LoadedPolicy",
    "PolicySpec",
    "inspect_experiment",
    "list_experiments",
    "load_experiment",
    "resolve_experiment",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_NAMES:
        from dexmani_policy.deployment import export

        return getattr(export, name)
    if name in _RUNTIME_NAMES:
        from dexmani_policy.deployment import runtime

        return getattr(runtime, name)
    raise AttributeError(name)
