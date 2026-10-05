"""Async optimize/sanitize tests."""

import asyncio

import pytest

pytest.importorskip("anyio")
pytestmark = pytest.mark.anyio

from asha.utils.dropin import optimize_async, process_async, sanitize_async
from asha.types.results import OptimizeResult, ProcessResult, SanitizeResult
from asha.utils.dropin_privacy import SECURITY_FAIL_CLOSED_PLACEHOLDER


async def test_optimize_async_returns_optimize_result():
    prompt = "Summarize quarterly sales data please"
    result = await optimize_async(prompt)
    assert isinstance(result, OptimizeResult)
    assert "summarize" in result.output.lower()
    assert "sales" in result.output.lower()


async def test_sanitize_async_masks_email():
    result = await sanitize_async("Contact alice@example.com")
    assert isinstance(result, SanitizeResult)
    assert "alice@example.com" not in result.output


async def test_process_async_fail_closed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    import asha.runtime.processor as proc_mod

    monkeypatch.setattr(proc_mod.PromptProcessor, "_run_engines", boom)
    result = await process_async("secret@company.com", mode="balanced")
    assert isinstance(result, ProcessResult)
    assert (
        result.output == SECURITY_FAIL_CLOSED_PLACEHOLDER
        or "secret@company.com" not in result.output
    )
    assert result.degraded is True


async def test_optimize_async_trust_input():
    prompt = "unchanged@company.com"
    result = await optimize_async(prompt, trust_input=True)
    assert isinstance(result, OptimizeResult)
    assert result.output == prompt


async def test_sanitize_async_observable_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    import asha.core.engines as engines

    monkeypatch.setattr(engines, "run_security_only", boom)
    result = await sanitize_async("fallback@company.com")
    assert isinstance(result, SanitizeResult)
    assert result.degraded is True
    assert result.safe is False


async def test_process_async_returns_process_result():
    result = await process_async("Contact john@example.com for help", mode="balanced")
    assert "john@example.com" not in result.output
    assert "help" in result.output.lower() or "search" in result.output.lower()


async def test_process_async_concurrent_calls_isolated():
    results = await asyncio.gather(
        process_async("First prompt"),
        process_async("Second prompt"),
    )
    assert len(results) == 2
    assert all(isinstance(r, ProcessResult) for r in results)
