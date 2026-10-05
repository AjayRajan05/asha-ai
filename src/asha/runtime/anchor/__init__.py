"""ASHA ANCHOR: mission-aware agent governance."""

from .runtime import AnchorRuntime, anchor, current_anchor_runtime
from .adapters import anchor_any, anchor_crewai, anchor_generic

__all__ = [
    "anchor",
    "AnchorRuntime",
    "current_anchor_runtime",
    "anchor_any",
    "anchor_crewai",
    "anchor_generic",
]

_optional_exports = (
    "anchor_asha_agent",
    "anchor_graph",
    "anchor_langchain",
    "anchor_mcp",
    "anchor_autogen",
    "anchor_llamaindex",
)

for _name in _optional_exports:
    try:
        _mod = __import__(f"{__name__}.adapters", fromlist=[_name])
        globals()[_name] = getattr(_mod, _name)
        __all__.append(_name)
    except (ImportError, AttributeError):
        pass
