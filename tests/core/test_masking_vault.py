# --- Merged from test_masking_vault.py ---
"""Tests for security/masking_vault.py - brings coverage from 0% to ~100%."""

import pytest

from asha.core.security.masking_vault import (
    MaskingVault,
    get_active_vault,
    set_active_vault,
)


# ---------------------------------------------------------------------------
# MaskingVault unit tests
# ---------------------------------------------------------------------------


def test_register_and_lookup():
    vault = MaskingVault()
    vault.register("[EMAIL_abc]", "john@example.com")
    assert vault.lookup("[EMAIL_abc]") == "john@example.com"


def test_lookup_missing_returns_none():
    vault = MaskingVault()
    assert vault.lookup("[MISSING]") is None


def test_register_ignores_empty_token():
    vault = MaskingVault()
    vault.register("", "value")  # empty token - should not crash or register
    vault.register("[TOKEN]", "")  # empty original - should not register
    assert vault.lookup("") is None
    assert vault.lookup("[TOKEN]") is None


def test_to_dict_returns_copy():
    vault = MaskingVault()
    vault.register("[T1]", "val1")
    vault.register("[T2]", "val2")
    d = vault.to_dict()
    assert d == {"[T1]": "val1", "[T2]": "val2"}
    # Mutating the returned dict must not affect the vault
    d["[T3]"] = "val3"
    assert vault.lookup("[T3]") is None


def test_clear_removes_all_entries():
    vault = MaskingVault()
    vault.register("[A]", "alpha")
    vault.register("[B]", "beta")
    vault.clear()
    assert vault.lookup("[A]") is None
    assert vault.lookup("[B]") is None
    assert len(vault) == 0


def test_len_reflects_registered_tokens():
    vault = MaskingVault()
    assert len(vault) == 0
    vault.register("[X]", "x-value")
    assert len(vault) == 1
    vault.register("[Y]", "y-value")
    assert len(vault) == 2


def test_register_overwrites_existing_token():
    vault = MaskingVault()
    vault.register("[T]", "original")
    vault.register("[T]", "updated")
    assert vault.lookup("[T]") == "updated"


def test_multiple_tokens_for_same_original():
    """Two different mask tokens can map to the same original."""
    vault = MaskingVault()
    vault.register("[T1]", "user@example.com")
    vault.register("[T2]", "user@example.com")
    assert vault.lookup("[T1]") == "user@example.com"
    assert vault.lookup("[T2]") == "user@example.com"


# ---------------------------------------------------------------------------
# Module-level vault helpers
# ---------------------------------------------------------------------------


def test_set_and_get_active_vault():
    original = get_active_vault()
    vault = MaskingVault()
    set_active_vault(vault)
    assert get_active_vault() is vault
    # Restore
    set_active_vault(original)


def test_set_active_vault_none():
    set_active_vault(None)
    assert get_active_vault() is None


def test_active_vault_initially_none_or_vault():
    """Active vault is either None or a MaskingVault - never an unexpected type."""
    v = get_active_vault()
    assert v is None or isinstance(v, MaskingVault)


def test_vault_collision_safe_same_entity_same_token():
    """Verify that looking up the same registered token returns the exact original value."""
    vault = MaskingVault()
    t1 = "[EMAIL_abc123]"
    vault.register(t1, "user@example.com")
    first_lookup = vault.lookup(t1)
    second_lookup = vault.lookup(t1)
    assert first_lookup == "user@example.com"
    assert first_lookup == second_lookup
    assert vault.lookup("[UNKNOWN_TOKEN]") is None


def test_vault_collision_safe_different_entities_distinct_tokens():
    """Verify that different entities maintain separate tokens and mappings in the vault."""
    vault = MaskingVault()
    t1 = "[EMAIL_aaa111]"
    t2 = "[EMAIL_bbb222]"
    vault.register(t1, "alice@example.com")
    vault.register(t2, "bob@example.com")
    assert vault.lookup(t1) != vault.lookup(t2)
    assert len(vault) == 2


# --- Consolidated from test_reversible_masking.py ---
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

