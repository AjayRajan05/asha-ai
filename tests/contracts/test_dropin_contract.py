"""Contract tests for process() / sanitize() signatures and result fields."""

from __future__ import annotations

import inspect

from asha import process, sanitize
from asha.types.results import MetricsInfo, ProcessResult, SanitizeResult, SecurityInfo


def test_process_signature_kwargs_frozen() -> None:
    sig = inspect.signature(process)
    params = list(sig.parameters)
    assert params[0] == "prompt"
    assert params[1] == "mode"
    for name in (
        "policy",
        "token_budget",
        "trace",
        "debug",
        "max_retries",
        "timeout_seconds",
        "verbose",
        "log_level",
        "log_output",
        "log_file",
        "include_legacy_detail",
        "filename",
    ):
        assert name in sig.parameters


def test_sanitize_signature_kwargs_frozen() -> None:
    sig = inspect.signature(sanitize)
    assert list(sig.parameters)[0] == "prompt"
    for name in ("mode", "policy", "max_retries", "timeout_seconds", "filename"):
        assert name in sig.parameters


def test_process_result_fields_frozen() -> None:
    result = process("Contact alice@example.com for analysis", mode="balanced")
    assert isinstance(result, ProcessResult)
    for name in (
        "output",
        "original",
        "degraded",
        "degraded_reason",
        "privacy_applied",
        "security",
        "metrics",
        "trace",
        "diff",
        "debug",
        "legacy_detail",
    ):
        assert hasattr(result, name)
    assert str(result) == result.output
    assert isinstance(result.security, SecurityInfo) or result.security is None
    assert isinstance(result.metrics, MetricsInfo) or result.metrics is None
    if result.security is not None:
        for name in (
            "safe",
            "pii_detected",
            "threats",
            "masked_entities",
            "masking_map",
            "threat_level",
            "security_score",
        ):
            assert hasattr(result.security, name)
    if result.metrics is not None:
        for name in (
            "tokens_saved",
            "token_reduction_pct",
            "processing_time_ms",
        ):
            assert hasattr(result.metrics, name)


def test_sanitize_result_fields_frozen() -> None:
    result = sanitize("Email bob@corp.com")
    assert isinstance(result, SanitizeResult)
    for name in (
        "output",
        "original",
        "safe",
        "degraded",
        "degraded_reason",
        "security",
    ):
        assert hasattr(result, name)
    assert "bob@corp.com" not in result.output
    assert isinstance(result.security, SecurityInfo)
