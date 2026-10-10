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
