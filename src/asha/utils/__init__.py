# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""ASHA utilities - lazy exports to avoid import cycles."""

__all__ = [
    "process",
    "optimize",
    "sanitize",
]


from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("process", "optimize", "sanitize"):
        from . import dropin as _dropin

        return getattr(_dropin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
