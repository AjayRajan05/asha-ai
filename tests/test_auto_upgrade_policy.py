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

"""Uniform auto-upgrade policy: hardened artifacts+deps → upgrade; explicit wins."""

from __future__ import annotations

from pathlib import Path

import pytest

from asha.core.security.injection_detector import (
    get_injection_detector,
    hardened_injection_available,
    resolve_injection_mode,
)


def test_resolve_injection_auto_upgrades_when_hardened_available(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ASHA_INJECTION_MODE", raising=False)
    if hardened_injection_available():
        assert resolve_injection_mode() == "ensemble"
        get_injection_detector(reset=True)
        assert get_injection_detector().mode == "ensemble"
    else:
        # Base / broken-torch env: must stay on lite, not silently claim ensemble.
        assert resolve_injection_mode() == "lite"


def test_explicit_lite_beats_auto_upgrade(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ASHA_INJECTION_MODE", raising=False)
    assert resolve_injection_mode("lite") == "lite"
    monkeypatch.setenv("ASHA_INJECTION_MODE", "lite")
    assert resolve_injection_mode() == "lite"
    get_injection_detector(mode="lite", reset=True)
    assert get_injection_detector(mode="lite").mode == "lite"


def test_mission_intent_prefer_joblib_when_present():
    """Mission/intent already auto-upgrade via joblib-first load order."""
    from asha.core._ir.intent_classifier import IntentClassifier
    from asha.runtime.anchor.domain_classifier import DomainClassifier

    mission_joblib = Path.cwd() / "models" / "mission" / "domain_clf.joblib"
    intent_joblib = Path.cwd() / "models" / "intent" / "intent_clf.joblib"
    assert mission_joblib.is_file(), f"missing checked-in artifact: {mission_joblib}"
    assert intent_joblib.is_file(), f"missing checked-in artifact: {intent_joblib}"

    domain = DomainClassifier()
    domain.predict("Analyze campus marketplace transaction trends")
    assert domain._backend == "hardened"

    intent = IntentClassifier()
    intent.predict("summarize the quarterly report")
    assert intent._backend == "hardened"
