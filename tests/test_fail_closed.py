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
