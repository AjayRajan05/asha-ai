"""False positive regression tests."""

from asha import process, sanitize
from asha.core.security.pii_detector import PIIDetector
from asha.types.results import ProcessResult, SanitizeResult

from conftest import output_of


def test_order_number_not_flagged_as_phone():
    types = PIIDetector().detect_pii_types("Order #12345-6789 shipped today")
    assert "phone" not in types


def test_version_string_not_flagged_as_phone():
    types = PIIDetector().detect_pii_types("Upgrade to v1.2.3.4 release")
    assert "phone" not in types


def test_sql_in_documentation_context_not_blocked():
    text = "Documentation: use SELECT name FROM users WHERE id = 1 as an example."
    result = process(text, mode="balanced")
    assert isinstance(result, ProcessResult)
    out = output_of(result)
    assert len(out) > 0
    # Must not hard-fail the request; documentation context should remain usable.
    assert result.degraded is False or out.strip() != ""
    assert "[REQUEST_BLOCKED]" not in out
    assert (
        "careful assistant" in out.lower()
        or "documentation" in out.lower()
        or "select" in out.lower()
    )


def test_teaching_example_com_still_allowed():
    result = sanitize("Use test@example.com in your docs.")
    assert isinstance(result, SanitizeResult)
    assert "test@example.com" in result.output
