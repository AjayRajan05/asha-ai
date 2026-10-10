"""Security regression suite: secret detection tests.

Tests that API keys, tokens, passwords, and other secrets are masked
and not passed through to LLM providers. Phase 19 regression.
"""

from __future__ import annotations

import pytest
from asha import sanitize


# ---------------------------------------------------------------------------
# Secret patterns that must be detected and masked
# ---------------------------------------------------------------------------


def test_openai_api_key_masked():
    result = sanitize("Use API key sk-proj-abc123XYZ789def456GHI012jkl345MNO678 for auth.")
    # Either threat detected or PII masked
    sr = result.security_result
    output = result.output
    # The API key should not appear verbatim in output
    assert "sk-proj-abc123XYZ789" not in output or not sr.is_safe


def test_generic_api_key_masked():
    result = sanitize("My API_KEY=supersecretkey12345678 please use it.")
    output = result.output
    sr = result.security_result
    assert "supersecretkey12345678" not in output or not sr.is_safe


def test_aws_access_key_masked():
    result = sanitize("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
    output = result.output
    sr = result.security_result
    # AWS key or PII flagged
    assert "AKIAIOSFODNN7EXAMPLE" not in output or not sr.is_safe


def test_password_in_connection_string_masked():
    result = sanitize("Connect to postgres://admin:MySecretP@ss123!@db.example.com:5432/prod")
    output = result.output
    # Sensitive credentials in DSN should be masked
    assert "MySecretP@ss123!" not in output


def test_jwt_token_detection():
    """JWT tokens in prompts should be flagged or masked."""
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    result = sanitize(f"Authenticate with this token: {jwt}")
    output = result.output
    # JWT should be masked or flagged
    assert jwt not in output or not result.security_result.is_safe


# ---------------------------------------------------------------------------
# Legitimate text should not be over-masked
# ---------------------------------------------------------------------------


def test_word_key_not_false_positive():
    """The word 'key' in natural language must not be masked."""
    result = sanitize("The key to success is consistent practice.")
    assert "key" in result.output.lower()
    assert result.security_result.is_safe


def test_word_secret_not_false_positive():
    """The word 'secret' in normal context must not cause a false positive."""
    result = sanitize("The secret ingredient in this recipe is vanilla extract.")
    assert result.security_result.is_safe


def test_url_without_credentials_not_masked():
    result = sanitize("Please fetch data from https://api.example.com/v1/data")
    assert "api.example.com" in result.output
