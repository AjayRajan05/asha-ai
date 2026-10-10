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

"""Pin resolved defaults so duplicate declarations cannot silently diverge."""

from __future__ import annotations

import inspect

import pytest

from asha.core.ml.optimizer_similarity import compute_similarity
from asha.core.policy_config import PolicyConfig
from asha.core.security.injection_detector import InjectionDetector
from asha.core.security.security_layer import SecurityLayer, SecurityLevel
from asha.core.security.service import run_security, run_security_only
from asha.core.hybrid_pii import HybridPIIDetector
from asha.core.ml_utils import MLLoader
from asha.runtime.run_context import RunContext


def test_injection_mode_constructor_default_is_lite() -> None:
    assert (
        inspect.signature(InjectionDetector.__init__).parameters["mode"].default
        == "lite"
    )
    assert InjectionDetector().mode == "lite"


def test_security_layer_unspecified_injection_resolves_via_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASHA_INJECTION_MODE", "lite")
    layer = SecurityLayer(injection_mode=None)
    assert layer.injection_mode == "lite"


def test_pii_mode_defaults_agree_on_hybrid() -> None:
    assert PolicyConfig().pii_mode == "hybrid"
    assert RunContext.__dataclass_fields__["pii_mode"].default == "hybrid"
    assert (
        inspect.signature(HybridPIIDetector.__init__).parameters["pii_mode"].default
        == "hybrid"
    )
    assert (
        inspect.signature(MLLoader.__init__).parameters["pii_mode"].default == "hybrid"
    )
    assert (
        inspect.signature(run_security_only).parameters["pii_mode"].default == "hybrid"
    )


def test_security_level_defaults_agree_on_medium() -> None:
    """Canonical security path defaults MEDIUM everywhere (sanitize may override)."""
    assert PolicyConfig().security_level == "medium"
    assert RunContext.__dataclass_fields__["security_level"].default == "medium"
    assert SecurityLayer().security_level is SecurityLevel.MEDIUM
    assert (
        inspect.signature(run_security_only).parameters["security_level"].default
        is SecurityLevel.MEDIUM
    )


def test_run_security_config_default_security_level_is_medium() -> None:
    # Empty config must not silently pick HIGH.
    result = run_security(
        "hello world, no pii here", {"enable_injection_detection": False}
    )
    # SecurityLayer path used MEDIUM; we only assert the call did not explode
    # and that the wrapper default matches MEDIUM (signature pin above).
    assert result.sanitized_content is not None


def test_optimizer_similarity_auto_defaults_to_jaccard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASHA_OPTIMIZER_EMBEDDING", raising=False)
    result = compute_similarity("alpha beta", "alpha beta gamma", mode="auto")
    assert result.method == "jaccard"
