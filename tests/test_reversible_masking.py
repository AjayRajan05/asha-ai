"""Tests for opt-in reversible PII masking."""

import pytest

from asha import sanitize
from asha.core.policy_config import PolicyConfig
from asha.types.results import ProcessResult, SanitizeResult
from asha.utils.dropin import process
from asha.utils.unmask import unmask

from conftest import output_of


def test_sanitize_reversible_round_trip():
    prompt = "Email me at alice@example.com"
    result = sanitize(prompt, policy=PolicyConfig(reversible=True))
    assert isinstance(result, SanitizeResult)
    assert result.security.masking_map
    masked = result.output
    assert "alice@example.com" not in masked
    restored = unmask(masked, result.security.masking_map)
    assert "alice@example.com" in restored


def test_process_reversible_returns_map():
    prompt = "Contact bob@test.org about order 123"
    result = process(prompt, policy=PolicyConfig(reversible=True))
    assert isinstance(result, ProcessResult)
    masking_map = result.security.masking_map if result.security else None
    assert masking_map
    assert "bob@test.org" in masking_map.values()
    assert "bob@test.org" not in result.output


def test_default_not_reversible():
    prompt = "Email carol@example.com"
    result = sanitize(prompt, policy=PolicyConfig(reversible=False))
    assert isinstance(result, SanitizeResult)
    assert result.security.masking_map is None


def test_masking_map_not_in_trace_json(capsys):
    """Vault contents must not appear in console trace output."""
    prompt = "SSN 123-45-6789 email d@e.com"
    result = process(
        prompt, policy=PolicyConfig(reversible=True), trace=True, log_output="json"
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "123-45-6789" not in combined
    assert "d@e.com" not in combined
