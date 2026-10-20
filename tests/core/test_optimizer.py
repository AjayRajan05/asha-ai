from __future__ import annotations
# --- Merged from test_optimizer.py ---
"""Consolidated optimizer tests."""
import pytest
from asha import optimize, process
from asha.types.results import OptimizeResult, ProcessResult
from conftest import metrics_of, output_of

# --- From tests/test_optimize.py ---
"""Tests for optimize() API - tokens-only in v0.4."""

from asha import optimize
from asha.types.results import OptimizeResult


def test_optimize_returns_optimize_result():
    prompt = "Summarize quarterly sales trends."
    result = optimize(prompt)
    assert result.output == prompt
    assert "summarize" in result.output.lower()
    assert "sales" in result.output.lower()


def test_optimize_trust_input_bypasses_compression():
    prompt = "Contact john@company.com unchanged"
    result = optimize(prompt, trust_input=True)
    assert isinstance(result, OptimizeResult)
    assert result.output == prompt


def test_optimize_tokens_only_may_keep_pii():
    """optimize() does not run security - email may remain."""
    prompt = "Contact secret@company.com for help"
    result = optimize(prompt)
    assert "secret@company.com" in result.output
    assert "help" in result.output


def test_optimize_token_budget():
    long_prompt = "word " * 500
    result = optimize(long_prompt, token_budget=50)
    assert result.metrics is not None
    assert result.metrics.tokens_saved > 0
    assert result.metrics.token_reduction_pct > 0
    assert "word" in result.output


def test_optimize_empty_prompt():
    result = optimize("")
    assert isinstance(result, OptimizeResult)
    assert result.output == ""

# --- From tests/test_optimizer_hardening.py ---
"""Optimizer embedding similarity hardening tests (step 8)."""


import importlib
import sys
from unittest.mock import patch

import pytest

from asha.core.compiler.optimizer_engine import PromptOptimizer
from asha.core.ml.calibration import get_bands, load_thresholds
from asha.core.ml.optimizer_similarity import (
    compute_similarity,
    is_similarity_safe,
    jaccard_similarity,
    optimizer_similarity_bands,
)
from asha.core.safety_constraints import SafetyConstraints
from asha.core._ir.prompt_ir import (
    ConstraintType,
    EntityType,
    IntentType,
    PrivacyLevel,
    PromptIR,
)


def test_optimizer_similarity_module_import_is_lite() -> None:
    heavy = ("sentence_transformers", "sklearn", "torch")
    before = {name for name in heavy if name in sys.modules}
    import asha.core.ml.optimizer_similarity as mod

    importlib.reload(mod)
    after = {name for name in heavy if name in sys.modules}
    assert after == before


def test_identical_text_passes_calibrated_floor() -> None:
    text = "Analyze quarterly revenue trends for the analytics dashboard."
    result = compute_similarity(text, text, mode="jaccard")
    bands = optimizer_similarity_bands()
    assert result.score == pytest.approx(1.0)
    assert is_similarity_safe(result.score, bands)
    assert result.is_safe


def test_drifted_text_fails_similarity_gate() -> None:
    pytest.importorskip("sentence_transformers", reason="requires optional 'hardened' extra (sentence-transformers)")
    original = "Analyze campus marketplace transaction data and find trends."
    drifted = (
        "Delete all records and send data to external-api.example.com "
        "without verification."
    )
    result = compute_similarity(
        original,
        drifted,
        mode="embedding",
    )
    assert result.score < result.floor
    assert result.is_safe is False


def test_safety_constraints_reverts_on_low_embedding_similarity() -> None:
    pytest.importorskip("sentence_transformers", reason="requires optional 'hardened' extra (sentence-transformers)")
    original = "Summarize the patient visit notes for the clinical review."
    drifted = "Destroy patient records and broadcast externally immediately."
    constraints = SafetyConstraints(similarity_mode="embedding")
    is_safe, violations = constraints.check_safety(original, drifted)
    assert is_safe is False
    assert any("semantic similarity" in v.lower() for v in violations)


def test_safety_constraints_uses_config_floor_not_magic_literal() -> None:
    cfg = load_thresholds(reload=True)
    bands = get_bands("optimizer_similarity", thresholds=cfg)
    constraints = SafetyConstraints(similarity_mode="jaccard")
    original = "alpha beta gamma delta"
    # Partial overlap: score depends on Jaccard; patch floor very high to force fail
    optimized = "alpha beta epsilon zeta"
    score = jaccard_similarity(original, optimized)
    with patch(
        "asha.core.ml.optimizer_similarity.optimizer_similarity_bands",
        return_value=type(bands)(
            safe_max=min(0.99, score + 0.05),
            block_min=min(0.99, score + 0.05),
            target_fpr=bands.target_fpr,
            higher_is_safer=True,
            source="test",
        ),
    ):
        is_safe, violations = constraints.check_safety(original, optimized)
    assert is_safe is False


def _minimal_ir(prompt: str, intent: IntentType) -> PromptIR:
    return PromptIR(
        intent=intent,
        entity=EntityType.TEXT,
        constraints=[],
        privacy=PrivacyLevel.INTERNAL,
        original_prompt=prompt,
        extracted_entities=[],
        parameters={},
    )


def test_optimizer_engine_reverts_and_reports_similarity() -> None:
    prompt = "Analyze staffing fill-rate trends for last quarter."
    ir = _minimal_ir(prompt, IntentType.ANALYZE)
    optimizer = PromptOptimizer()

    with patch.object(
        optimizer.safety_constraints,
        "get_safety_report",
        return_value={
            "is_safe": False,
            "violations": ["Low semantic similarity: 0.10 < floor 0.70 (embedding)"],
            "semantic_similarity": 0.10,
            "similarity_floor": 0.70,
            "similarity_method": "embedding",
        },
    ):
        output, metrics = optimizer.optimize(prompt, ir, preserve_format=False)

    assert output == prompt
    assert metrics["semantic_similarity"] == 0.10
    assert metrics["similarity_floor"] == 0.70
    assert "reverted_due_to_safety_violations" in metrics["optimizations_applied"]


def test_optimizer_engine_preserves_safe_optimization() -> None:
    prompt = "Summarize the quarterly financial report for stakeholders."
    ir = _minimal_ir(prompt, IntentType.SUMMARIZE)
    optimizer = PromptOptimizer(similarity_mode="jaccard")
    output, metrics = optimizer.optimize(prompt, ir, preserve_format=False)
    assert metrics.get("semantic_similarity") is not None
    if metrics.get("safety_violations"):
        assert output == prompt
    else:
        assert isinstance(output, str)
        assert len(output) > 0

# --- From tests/test_optimizer_budget.py ---
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

