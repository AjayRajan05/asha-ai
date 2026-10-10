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

"""Suite-wide fixtures (no forced lite/ML-off defaults: those hid real failures)."""

from __future__ import annotations

import pytest

@pytest.fixture(scope="session", autouse=True)
def preload_minilm_once() -> None:
    """
    Load MiniLM once before any test can pollute torch/numpy.
    Some CLI/transformers background threads cause ACCESS_VIOLATION on Windows
    if torch/numpy is initialized off the main thread first.
    """
    try:
        from asha.core.ml.embeddings import get_encoder
        get_encoder().ensure_loaded()
    except Exception:
        pass

    try:
        import numpy  # noqa: F401
    except ImportError:
        pass

    try:
        import docx  # noqa: F401
    except ImportError:
        pass

# Modules that need hardened models, optional frameworks, or heavy ML deps.
# Publish CI selects ``-m "not slow"``; default local runs still include these
# unless the caller passes ``-m "not slow"``.
_SLOW_PATH_FRAGMENTS = (
    "test_injection_ensemble.py",
    "test_memory_guard_hardening.py",
    "test_memory_guard_lite.py",
    "test_optimizer_hardening.py",
    "test_chain_guard_hardening.py",
    "test_auto_upgrade_policy.py",
    "test_core_ml.py",
    "test_mission_intent_hardening.py",
    "test_safety_classifier_rules.py",
    "test_security_bypass.py",
    "test_threat_scoring.py",
    "test_local_advisor.py",
    "test_alignment_hardening.py",
    "test_pii_hardening.py",
    "runtime/anchor/test_anchor_adapters.py",
    "runtime/anchor/test_anchor_adapters_golden.py",
    "runtime/anchor/test_anchor_extensions.py",
    "runtime/anchor/test_anchor_gaps.py",
    "runtime/anchor/test_anchor_governance.py",
    "runtime/anchor/test_anchor_p1.py",
    "runtime/anchor/test_anchor_crewai_integration.py",
    "runtime/anchor/test_anchor_langchain_integration.py",
    "runtime/anchor/test_anchor_langgraph_integration.py",
    "runtime/anchor/test_anchor_llamaindex_integration.py",
    "runtime/anchor/test_anchor_mcp_integration.py",
)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    slow = pytest.mark.slow
    for item in items:
        path = item.nodeid.replace("\\", "/")
        if any(fragment in path for fragment in _SLOW_PATH_FRAGMENTS):
            item.add_marker(slow)
