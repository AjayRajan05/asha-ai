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

"""Fallback warning text must name ``asha[hardened]`` (not vague 'transformers')."""

from __future__ import annotations

import warnings

import pytest

from asha.core.security.injection_detector import (
    HARDENED_INJECTION_FALLBACK_MSG,
    resolve_injection_mode,
    warn_hardened_injection_unavailable,
)


def test_hardened_injection_fallback_msg_names_extra_and_deps() -> None:
    assert "asha[hardened]" in HARDENED_INJECTION_FALLBACK_MSG
    assert "sentence-transformers" in HARDENED_INJECTION_FALLBACK_MSG
    assert "scikit-learn" in HARDENED_INJECTION_FALLBACK_MSG
    assert "lite path" in HARDENED_INJECTION_FALLBACK_MSG
    # Must not point users at bare `pip install transformers`.
    assert "pip install transformers" not in HARDENED_INJECTION_FALLBACK_MSG


def test_warn_hardened_injection_unavailable_emits_asha_hardened() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msg = warn_hardened_injection_unavailable(force=True)
    assert "asha[hardened]" in msg
    assert any("asha[hardened]" in str(w.message) for w in caught)


def test_resolve_ensemble_without_hardened_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asha.core.security.injection_detector as inj

    monkeypatch.setattr(inj, "hardened_injection_available", lambda: False)
    monkeypatch.setattr(inj, "_hardened_injection_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mode = resolve_injection_mode("ensemble")
    assert mode == "ensemble"  # explicit still wins
    assert any("asha[hardened]" in str(w.message) for w in caught)


def test_safety_classifier_fallback_print_mentions_hardened() -> None:
    """Source must keep the install hint; must not revive the old transformers-only line."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "asha"
        / "core"
        / "safety_classifier.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "asha[hardened]" in text
    assert "Install with: pip install asha[hardened]" in text
    # Exact legacy runtime string must not remain as a live print/warn argument.
    assert (
        'print(\n        "Warning: Transformers not available, using rule-based safety detection only"'
        not in text
    )
    assert (
        '"Warning: Transformers not available, using rule-based safety detection only"'
        not in text
    )
