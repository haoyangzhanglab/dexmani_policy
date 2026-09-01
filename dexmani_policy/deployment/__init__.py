"""Policy-native deployment artifact export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dexmani_policy.deployment.export import (
        ExportReceipt,
        export_deployment_artifact,
    )

__all__ = ["ExportReceipt", "export_deployment_artifact"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from dexmani_policy.deployment import export

        return getattr(export, name)
    raise AttributeError(name)
