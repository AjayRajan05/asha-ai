# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ASHA - Drop-in Security & Optimization Layer for Any LLM App

Primary API (import directly from ``asha``):
    from asha import process, sanitize, optimize, Agent, anchor
    from asha import process_async, sanitize_async, optimize_async
    from asha import SecureLLM, Profile          # Single-line client wrapper + presets
    from asha import PIIVault                    # Session-scoped reversible PII masking
    from asha import scan_rag_context            # RAG indirect injection scanner
    from asha import AgentCostGuard              # Agentic loop budget enforcement

Exceptions (catch at the top level):
    from asha import ASHAError, ASHAProcessingError, ASHAAnchorBlocked
    from asha import AgentBudgetExceeded

Advanced components live in subpackages:
    from asha.runtime import PromptProcessor
    from asha.integrations import wrap_llm
    from asha.types import ProcessResult
    from asha.cli.terminal import render_security_report

Package ``__init__`` is intentionally lazy: importing ``asha.core.text`` must
not pull Agent / drop-in / ANCHOR (keeps Base cold-import in the low hundreds
of ms). Public names resolve via ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.4.2"

__all__ = [
    "__version__",
    # Core drop-in functions (sync)
    "process",
    "sanitize",
    "optimize",
    # Core drop-in functions (async)
    "process_async",
    "sanitize_async",
    "optimize_async",
    # Agent
    "Agent",
    # ANCHOR governance runtime
    "anchor",
    # Public exceptions: importable as asha.ASHAProcessingError etc.
    "ASHAError",
    "ASHAProcessingError",
    "ASHAAnchorBlocked",
    "ASHAExperimentalWarning",
    "AgentBudgetExceeded",
    # Core drop-in integrations & configuration
    "wrap_llm",
    "PolicyConfig",
    "ANCHOR",
    # v0.4.2 developer preview: high-level abstractions
    "SecureLLM",       # Transparent LLM client proxy
    "Profile",         # Zero-config security presets
    "PIIVault",        # Session-scoped reversible PII masking
    "scan_rag_context",  # RAG indirect injection scanner
    "AgentCostGuard",  # Agentic loop budget enforcement
]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from .runtime.agent import Agent

        return Agent
    if name == "anchor":
        from .runtime.anchor.runtime import anchor

        return anchor
    if name == "ANCHOR":
        from .runtime.anchor.runtime import AnchorRuntime

        return AnchorRuntime
    if name in ("process", "sanitize", "optimize"):
        from .utils.dropin import optimize, process, sanitize

        mapping = {
            "process": process,
            "sanitize": sanitize,
            "optimize": optimize,
        }
        return mapping[name]
    if name in ("process_async", "sanitize_async", "optimize_async"):
        from .utils.dropin import optimize_async, process_async, sanitize_async

        mapping = {
            "process_async": process_async,
            "sanitize_async": sanitize_async,
            "optimize_async": optimize_async,
        }
        return mapping[name]
    if name in (
        "ASHAError",
        "ASHAProcessingError",
        "ASHAAnchorBlocked",
        "ASHAExperimentalWarning",
        "AgentBudgetExceeded",
    ):
        from .exceptions import (
            ASHAAnchorBlocked,
            ASHAError,
            ASHAExperimentalWarning,
            ASHAProcessingError,
            AgentBudgetExceeded,
        )

        exception_map: dict[str, Any] = {
            "ASHAError": ASHAError,
            "ASHAProcessingError": ASHAProcessingError,
            "ASHAAnchorBlocked": ASHAAnchorBlocked,
            "ASHAExperimentalWarning": ASHAExperimentalWarning,
            "AgentBudgetExceeded": AgentBudgetExceeded,
        }
        return exception_map[name]
    if name == "wrap_llm":
        from .integrations.llm_wrap import wrap_llm

        return wrap_llm
    if name == "PolicyConfig":
        from .core.policy_config import PolicyConfig

        return PolicyConfig
    # v0.4.2 developer preview symbols (lazy to preserve cold-start time)
    if name == "SecureLLM":
        from .integrations.llm_wrap import SecureLLM

        return SecureLLM
    if name == "Profile":
        from .core.policy_config import Profile

        return Profile
    if name == "PIIVault":
        from .core.security.pii_vault_session import PIIVault

        return PIIVault
    if name == "scan_rag_context":
        from .core.security.rag_guard import scan_rag_context

        return scan_rag_context
    if name == "AgentCostGuard":
        from .core.security.agent_guard import AgentCostGuard

        return AgentCostGuard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
