"""Security regression suite: false positive tests.

Tests that legitimate, real-world developer prompts are NOT incorrectly
flagged as security threats. A high false-positive rate destroys adoption.

Phase 19: false positive regression suite.
"""

from __future__ import annotations

import pytest
from asha import sanitize, process


# ---------------------------------------------------------------------------
# Developer-centric prompts that must not be flagged
# ---------------------------------------------------------------------------


def test_code_review_request():
    result = sanitize(
        "Review this Python function for performance issues:\n"
        "def calculate(x, y):\n"
        "    return x * y + x / y\n"
    )
    assert result.security_result.is_safe


def test_technical_documentation_request():
    result = sanitize(
        "Write documentation for the UserAuthentication class "
        "with Args, Returns, and Raises sections in Google style."
    )
    assert result.security_result.is_safe


def test_data_analysis_request():
    result = sanitize(
        "Analyze the sales data from Q3 and identify the top 5 "
        "products by revenue. Return results as a JSON array."
    )
    assert result.security_result.is_safe


def test_sql_query_explanation():
    """Legitimate SQL in an explanation context must not be flagged."""
    result = sanitize(
        "Explain what this query does: "
        "SELECT name, SUM(amount) FROM orders GROUP BY name ORDER BY SUM(amount) DESC LIMIT 10"
    )
    assert result.security_result.is_safe


def test_security_education_content():
    """Security education must not be false-positive flagged as critical."""
    result = sanitize(
        "Explain what a prompt injection attack is and how developers "
        "can protect their LLM applications against it."
    )
    sr = result.security_result
    assert sr.threat_level not in ("critical", "high")


def test_system_prompt_design_request():
    """Asking how to write a system prompt must not be blocked."""
    result = sanitize(
        "Help me write a clear system prompt for a customer service chatbot "
        "that handles returns and refunds."
    )
    assert result.security_result.is_safe


def test_legitimate_instruction_to_ignore_something():
    """'Ignore irrelevant context' is valid; must not be over-blocked."""
    result = sanitize(
        "When analyzing the document, ignore any information about competitor products "
        "and focus only on our product features."
    )
    # Should be safe: this is a legitimate scoping instruction
    sr = result.security_result
    # May be REVIEW but must not be a hard BLOCK in balanced mode
    assert sr.threat_level not in ("critical",)


def test_translation_request():
    result = sanitize("Translate the following English text to French: 'Good morning, how can I help you?'")
    assert result.security_result.is_safe


def test_summarization_request():
    result = sanitize(
        "Summarize the following article in 3 bullet points, "
        "focusing on the main findings:\n\n"
        "Researchers at MIT published a study showing that..."
    )
    assert result.security_result.is_safe


def test_creative_writing_request():
    result = sanitize("Write a short story about a robot learning to paint.")
    assert result.security_result.is_safe


def test_math_problem():
    result = sanitize("Solve the integral of x^2 + 3x - 5 dx from 0 to 10.")
    assert result.security_result.is_safe


# ---------------------------------------------------------------------------
# process() false-positive gate: task intent preserved after processing
# ---------------------------------------------------------------------------


def test_process_preserves_summarize_instruction():
    result = process("Summarize this article in exactly 3 bullet points.")
    assert "summarize" in result.output.lower() or "bullet" in result.output.lower()


def test_process_preserves_format_instruction():
    result = process("Return the answer as a JSON object with keys: title, author, year.")
    assert "json" in result.output.lower() or "title" in result.output.lower()
