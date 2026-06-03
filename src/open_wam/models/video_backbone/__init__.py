"""Shared video-transformer backbone boundary."""

from importlib import import_module
from typing import TYPE_CHECKING

from .config import (
    LingbotCompatibleVideoBackboneConfig,
    SharedVideoTransformerConfig,
    normalize_backbone_implementation,
    resolve_stage_attention_mode,
)

if TYPE_CHECKING:
    from .contracts import BackboneOutput, CacheState, ChunkMetadata, ConditioningState, TokenGridMetadata
    from .lingbot_compatible import LingbotCompatibleVideoBackbone, SharedVideoTransformerBackbone

__all__ = [
    "BackboneOutput",
    "CacheState",
    "ChunkMetadata",
    "ConditioningState",
    "LingbotCompatibleVideoBackbone",
    "LingbotCompatibleVideoBackboneConfig",
    "SharedVideoTransformerBackbone",
    "SharedVideoTransformerConfig",
    "TokenGridMetadata",
    "normalize_backbone_implementation",
    "resolve_stage_attention_mode",
]


def __getattr__(name: str):
    if name in {"LingbotCompatibleVideoBackbone", "SharedVideoTransformerBackbone"}:
        from .lingbot_compatible import LingbotCompatibleVideoBackbone, SharedVideoTransformerBackbone

        if name == "SharedVideoTransformerBackbone":
            return SharedVideoTransformerBackbone
        return LingbotCompatibleVideoBackbone
    if name in {"BackboneOutput", "CacheState", "ChunkMetadata", "ConditioningState", "TokenGridMetadata"}:
        module = import_module(f"{__name__}.contracts")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
