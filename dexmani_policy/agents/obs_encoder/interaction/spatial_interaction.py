"""Backward-compatible import for the renamed hand-object interaction encoder."""

from .hand_object_interaction import HandObjectInteractionEncoder, HandCentricInteractionEncoder

__all__ = ["HandObjectInteractionEncoder", "HandCentricInteractionEncoder"]
