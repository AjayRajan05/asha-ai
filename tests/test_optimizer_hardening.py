"""Optimizer embedding similarity hardening tests (step 8)."""

from __future__ import annotations

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
