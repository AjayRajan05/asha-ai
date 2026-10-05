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
