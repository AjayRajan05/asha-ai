from __future__ import annotations
# --- Merged from test_fail_closed.py ---
"""Security fail-closed and safety mode tests."""

import pytest

from asha import process, sanitize
from asha.core.safety import SafetyMode
from asha.exceptions import ASHAProcessingError
from asha.runtime.processor import PromptProcessor
from asha.types.results import ProcessResult, SanitizeResult
from asha.utils.dropin_privacy import (
    SECURITY_FAIL_CLOSED_PLACEHOLDER,
    privacy_fallback,
)


def test_privacy_fallback_balanced_masks_pii():
    out = privacy_fallback("secret@company.com", True, safety=SafetyMode.BALANCED)
    assert "secret@company.com" not in out


def test_privacy_fallback_strict_placeholder_on_security_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("security failed")

    import asha.utils.dropin_privacy as dp

    monkeypatch.setattr(dp, "run_security_only", boom)
    out = privacy_fallback("secret@company.com", True, safety=SafetyMode.STRICT)
    assert out == SECURITY_FAIL_CLOSED_PLACEHOLDER


def test_sanitize_strict_raises_on_security_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("security failed")

    import asha.core.engines as engines

    monkeypatch.setattr(engines, "run_security_only", boom)
    with pytest.raises(ASHAProcessingError):
        sanitize("secret@company.com", mode="strict")


def test_sanitize_balanced_degrades_on_security_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("security failed")

    import asha.core.engines as engines

    monkeypatch.setattr(engines, "run_security_only", boom)
    result = sanitize("secret@company.com", mode="balanced")
    assert isinstance(result, SanitizeResult)
    assert result.degraded is True
    assert result.output == "secret@company.com"


def test_process_strict_raises_on_processor_failure(monkeypatch):
    def engines_boom(*args, **kwargs):
        raise RuntimeError("processor failed")

    import asha.runtime.processor as proc_mod

    monkeypatch.setattr(proc_mod.PromptProcessor, "_run_engines", engines_boom)

    with pytest.raises(ASHAProcessingError):
        process("secret@company.com", mode="strict")


def test_process_balanced_falls_back_on_processor_failure(monkeypatch):
    def engines_boom(*args, **kwargs):
        raise RuntimeError("processor failed")

    import asha.runtime.processor as proc_mod

    monkeypatch.setattr(proc_mod.PromptProcessor, "_run_engines", engines_boom)

    result = process("secret@company.com", mode="balanced")
    assert isinstance(result, ProcessResult)
    assert result.degraded is True
    assert "secret@company.com" not in result.output


# --- Consolidated from test_policy_modes.py ---
"""Tests for policy mode passthrough and stage guards."""

from asha import process
from asha.types.results import ProcessResult

from conftest import output_of


def test_mode_off_returns_byte_identical():
    prompt = "What is 2 + 2?"
    result = process(prompt, mode="off")
    assert output_of(result) == prompt


def test_mode_off_with_pii_unchanged():
    prompt = "Email me at john@example.com"
    result = process(prompt, mode="off")
    assert output_of(result) == prompt


def test_mode_strict_masks_pii():
    prompt = "Contact john@example.com"
    result = process(prompt, mode="strict")
    assert "john@example.com" not in output_of(result)


def test_mode_lite_processes_without_crash():
    prompt = "Hello, please summarize this document about quarterly revenue."
    result = process(prompt, mode="lite")
    assert isinstance(result, ProcessResult)
    out = output_of(result).lower()
    assert len(out) > 0
    assert (
        "summarize" in out
        or "document" in out
        or "quarterly" in out
        or "revenue" in out
    )


def test_privacy_false_skips_masking_in_balanced():
    prompt = "Reach me at secret@company.com for details"
    result = process(prompt, mode="off")
    assert isinstance(result, ProcessResult)
    security = result.security
    assert security is not None
    assert security.to_dict()["pii_masked"] == 0
    assert "secret@company.com" in result.original


# --- Consolidated from tests/test_default_consistency.py ---
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


# --- Consolidated from test_safety_consistency.py ---
"""Safety semantics consistency across process, sanitize, and integrations."""

import pytest

from asha import process, sanitize
from asha.core.safety import SafetyMode, safety_mode_from_policy_mode
from asha.exceptions import ASHAProcessingError
from asha.integrations.llm_wrap import (
    _handle_wrap_processing_error,
    _process_prompt_for_wrap,
)


def test_policy_mode_maps_to_safety_mode():
    assert safety_mode_from_policy_mode("strict") == SafetyMode.STRICT
    assert safety_mode_from_policy_mode("balanced") == SafetyMode.BALANCED
    assert safety_mode_from_policy_mode("lite") == SafetyMode.BALANCED
    assert safety_mode_from_policy_mode("off") == SafetyMode.OFF


def test_wrap_uses_mode_for_processing(monkeypatch):
    captured = {}

    def fake_process(prompt, **kwargs):
        captured.update(kwargs)
        from asha.types.results import ProcessResult, SecurityInfo

        return ProcessResult(
            output="masked",
            original=prompt,
            degraded=False,
            degraded_reason=None,
            privacy_applied=True,
            security=SecurityInfo(True, [], [], {}),
            metrics=None,
        )

    import asha.utils.dropin as dropin_mod

    monkeypatch.setattr(dropin_mod, "process", fake_process)
    _process_prompt_for_wrap("secret@company.com", mode="lite")
    assert captured.get("mode") == "lite"


def test_strict_process_raises_on_security_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("security failed")

    import asha.runtime.processor as proc_mod

    monkeypatch.setattr(proc_mod.PromptProcessor, "_run_engines", boom)
    with pytest.raises(ASHAProcessingError):
        process("x@y.com", mode="strict")


def test_balanced_sanitize_keeps_prompt_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("security failed")

    import asha.core.engines as engines

    monkeypatch.setattr(engines, "run_security_only", boom)
    result = sanitize("keep@me.com", mode="balanced")
    assert result.output == "keep@me.com"
    assert result.degraded is True


def test_wrap_mode_off_does_not_fail_closed_on_error():
    with pytest.raises(RuntimeError):
        _handle_wrap_processing_error("off", RuntimeError("transport"))


def test_wrap_mode_balanced_fail_closed_on_transport_error():
    with pytest.raises(ASHAProcessingError):
        _handle_wrap_processing_error("balanced", RuntimeError("transport"))


def test_auto_patch_accepts_mode_parameter():
    from asha.integrations.auto_patch import auto_patch
    import inspect

    sig = inspect.signature(auto_patch)
    assert "mode" in sig.parameters
    assert sig.parameters["mode"].default == "strict"

