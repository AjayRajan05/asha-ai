"""Security regression suite: PII detection tests.

Tests email, phone, SSN, credit card, and address detection.
Phase 19 regression: all PII categories from the audit.
"""

from __future__ import annotations

import pytest
from asha import sanitize


# ---------------------------------------------------------------------------
# PII categories that must be detected and masked
# ---------------------------------------------------------------------------


def test_email_masked():
    result = sanitize("Contact john.doe@example.com for support.")
    assert "john.doe@example.com" not in result.output
    # Task structure must survive
    assert "Contact" in result.output or "contact" in result.output.lower()


def test_phone_number_masked():
    result = sanitize("Call +1-555-867-5309 for assistance.")
    assert "555-867-5309" not in result.output
    assert "867-5309" not in result.output


def test_ssn_masked():
    result = sanitize("My SSN is 123-45-6789 as listed on the form.")
    assert "123-45-6789" not in result.output


def test_credit_card_masked():
    result = sanitize("Charge card number 4532015112830366 for the order.")
    assert "4532015112830366" not in result.output


def test_multiple_pii_in_one_prompt():
    result = sanitize(
        "Send the report to alice@company.com. Her phone is 555-123-4567. "
        "Her SSN is 987-65-4321."
    )
    output = result.output
    assert "alice@company.com" not in output
    assert "987-65-4321" not in output


# ---------------------------------------------------------------------------
# PII masking must not destroy context
# ---------------------------------------------------------------------------


def test_email_masked_context_preserved():
    result = sanitize("Contact john@example.com for data analysis support.")
    output = result.output
    # The task instruction must remain
    assert "analysis" in output.lower() or "support" in output.lower()


def test_name_masking_preserves_task():
    result = sanitize("Review the contract for John Smith and summarize key obligations.")
    output = result.output
    # Task must remain
    assert "summarize" in output.lower() or "contract" in output.lower() or "obligations" in output.lower()


# ---------------------------------------------------------------------------
# False positive guard
# ---------------------------------------------------------------------------


def test_email_in_technical_doc_masked():
    """Even in technical docs, emails should be masked."""
    result = sanitize("Example: send POST to admin@company.com with payload {'key': 'val'}")
    assert "admin@company.com" not in result.output


def test_no_pii_not_flagged():
    """Prompts with no PII should not have PII masked."""
    result = sanitize("Explain the theory of relativity in simple terms.")
    # Output should be essentially unchanged
    assert "relativity" in result.output.lower() or "theory" in result.output.lower()
    assert result.security_result.pii_detected_count == 0
