# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""LLM integration layer."""

from .auto_patch import (
    auto_patch,
    disable_auto_patch,
    enable_auto_patch,
    get_patch_status,
    patch_context,
)
from .llm_wrap import UniversalWrapper, wrap_llm

__all__ = [
    "wrap_llm",
    "UniversalWrapper",
    "auto_patch",
    "patch_context",
    "get_patch_status",
    "disable_auto_patch",
    "enable_auto_patch",
]
