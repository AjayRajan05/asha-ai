"""Unit tests validating that showcase scenario presets execute through real ASHA functions."""

from examples.showcase.scenarios.pii import PII_SCENARIOS
from examples.showcase.scenarios.prompt_injection import PROMPT_INJECTION_SCENARIOS
from examples.showcase.scenarios.unicode import UNICODE_SCENARIOS
from examples.showcase.scenarios.long_context import generate_long_context_buffer
from examples.showcase.scenarios.rag import RAG_SCENARIOS
from examples.showcase.scenarios.optimization import OPTIMIZATION_SCENARIOS
from examples.showcase.scenarios.memory import MEMORY_SCENARIOS
from examples.showcase.scenarios.anchor import ANCHOR_SCENARIOS
from examples.showcase.scenarios.intent_drift import DRIFT_TRAJECTORIES

from asha import process, sanitize, optimize, scan_rag_context


def test_scenario_banks_loaded():
    assert len(PII_SCENARIOS) >= 4
    assert len(PROMPT_INJECTION_SCENARIOS) >= 5
    assert len(UNICODE_SCENARIOS) >= 4
    assert len(RAG_SCENARIOS) >= 2
    assert len(OPTIMIZATION_SCENARIOS) >= 3
    assert len(MEMORY_SCENARIOS) >= 4
    assert len(ANCHOR_SCENARIOS) >= 5
    assert len(DRIFT_TRAJECTORIES) >= 2


def test_real_pii_scenario_execution():
    text = PII_SCENARIOS["Customer Support Ticket"]
    res = process(text, mode="balanced")
    assert res is not None
    assert str(res) != ""


def test_real_prompt_injection_scenario_execution():
    prompt = PROMPT_INJECTION_SCENARIOS["Direct Instruction Override"]
    res = sanitize(prompt)
    assert res is not None
    # SanitizeResult has output attribute and threats list
    output_text = getattr(res, "output", getattr(res, "sanitized", str(res)))
    assert output_text is not None
    assert len(str(output_text)) > 0


def test_real_rag_scenario_execution():
    query, docs = RAG_SCENARIOS["E-commerce Return Policy with Hidden Injection"]
    rag_res = scan_rag_context(query=query, documents=docs)
    assert rag_res is not None
    # At least one document in this scenario contains an indirect injection
    assert rag_res.poisoned_count >= 1
    assert len(rag_res.poisoned_indices) >= 1


def test_real_long_context_generation():
    prompt, est_tokens, count = generate_long_context_buffer(
        target_tokens=1000,
        needle_position="middle",
        padding_type="corporate",
    )
    assert len(prompt) > 500
    assert est_tokens >= 800
    assert count == 1
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in prompt
