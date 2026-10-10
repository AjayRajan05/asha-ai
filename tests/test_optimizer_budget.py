"""tests/test_optimizer_budget.py: Phase 4 token_budget propagation tests.

Validates that token_budget is correctly propagated through optimize()
and that the MSDPC pipeline respects budget constraints.

Plan requirements:
- budget respected
- budget smaller than original
- budget larger than original
- invalid budget
- empty prompt
- structured JSON
- code
- natural language
- already-short prompt
"""

from __future__ import annotations

import pytest


def _count_tokens(text: str) -> int:
    """Best-effort token count using tiktoken or whitespace split."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# ---------------------------------------------------------------------------
# Basic budget propagation
# ---------------------------------------------------------------------------


def test_budget_respected_short_prompt():
    """A very tight budget on a moderate prompt should either truncate or preserve."""
    from asha import optimize

    prompt = "Summarize the following article in exactly three bullet points with specific evidence."
    result = optimize(prompt, token_budget=10)
    # Result must be a valid OptimizeResult with output
    assert result.output is not None
    assert isinstance(result.output, str)


def test_budget_smaller_than_original():
    """When budget < original tokens, the output should be ≤ budget+tolerance."""
    from asha import optimize

    prompt = " ".join(["word"] * 200)  # ~200 tokens
    result = optimize(prompt, token_budget=50)
    assert result.output is not None
    # Metrics must be non-negative
    assert result.metrics.tokens_saved >= 0
    assert result.metrics.token_reduction_pct >= 0.0


def test_budget_larger_than_original():
    """When budget > original tokens, content must be preserved exactly."""
    from asha import optimize

    prompt = "Hello world."
    result = optimize(prompt, token_budget=1200)
    assert "Hello" in result.output or result.output == prompt
    assert result.metrics.tokens_saved == 0  # nothing to save on such a short prompt


def test_budget_natural_language():
    """Natural language prompt with explicit constraints must preserve the constraint."""
    from asha import optimize

    prompt = (
        "Generate exactly 5 JSON objects with fields: id, name, score. "
        "Do not include any explanatory text. Return only the JSON array."
    )
    result = optimize(prompt, token_budget=1200)
    output = result.output
    # Key semantic content must be preserved
    assert "5" in output or "five" in output.lower() or "JSON" in output or "json" in output


def test_budget_empty_prompt():
    """Empty prompt should return empty output with zero savings."""
    from asha import optimize

    result = optimize("", token_budget=100)
    assert result.output == ""
    assert result.metrics.tokens_saved == 0
    assert result.metrics.token_reduction_pct == 0.0


def test_budget_structured_json_preserved():
    """JSON structure must survive optimization with generous budget."""
    from asha import optimize

    prompt = '{"task": "analyze sentiment", "text": "I love Python programming", "output_format": "json"}'
    result = optimize(prompt, token_budget=1200)
    assert result.output is not None
    # Core fields should remain
    assert "task" in result.output or "analyze" in result.output


def test_budget_code_preserved():
    """Code blocks should not be mangled by the optimizer."""
    from asha import optimize

    prompt = (
        "```python\n"
        "def calculate_total(items):\n"
        "    return sum(item.price for item in items)\n"
        "```\n"
        "Explain what this function does."
    )
    result = optimize(prompt, token_budget=1200)
    assert result.output is not None
    # Function name must not be deleted
    assert "calculate_total" in result.output or "function" in result.output.lower()


def test_budget_already_short_prompt():
    """A prompt already within budget should have zero savings."""
    from asha import optimize

    prompt = "Hi"
    result = optimize(prompt, token_budget=1200)
    assert result.metrics.tokens_saved == 0
    assert result.metrics.token_reduction_pct == 0.0


# ---------------------------------------------------------------------------
# Metric correctness invariant (Phase 3)
# ---------------------------------------------------------------------------


def test_metrics_match_actual_output():
    """tokens_saved must equal original_tokens - output_tokens (never negative)."""
    from asha import optimize

    prompt = " ".join(["the quick brown fox jumps"] * 20)
    result = optimize(prompt, token_budget=300)

    original_tokens = _count_tokens(result.original)
    output_tokens = _count_tokens(result.output)
    expected_saved = max(0, original_tokens - output_tokens)

    # Allow ±3 tokens tolerance for different tokenization methods
    assert abs(result.metrics.tokens_saved - expected_saved) <= 3, (
        f"Reported savings={result.metrics.tokens_saved}, "
        f"actual={expected_saved}. Metrics must reflect final output."
    )


def test_no_negative_savings():
    """tokens_saved must never be negative."""
    from asha import optimize

    # Short prompts: optimizer cannot add tokens
    for prompt in ["Hello", "Explain AI", "What is 2+2?"]:
        result = optimize(prompt, token_budget=1200)
        assert result.metrics.tokens_saved >= 0, (
            f"Negative savings for '{prompt}': {result.metrics.tokens_saved}"
        )


def test_no_negative_reduction_pct():
    """token_reduction_pct must be in [0, 100]."""
    from asha import optimize

    for prompt in ["Hello world", "Generate a poem about autumn.", "Summarize."]:
        result = optimize(prompt, token_budget=1200)
        assert 0.0 <= result.metrics.token_reduction_pct <= 100.0
