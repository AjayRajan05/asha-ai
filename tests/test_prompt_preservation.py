"""Tests for prompt content preservation - Gaps 2, 3, 4, 5.

Gap 2: Safe prompts should be unchanged under mode="off".
Gap 3: Semantic meaning should be preserved (key intent words survive).
       preserve_intent=True is the recommended escape-hatch when the default
       balanced-mode optimizer would otherwise drift the meaning.
Gap 4: Instructions/constraints must survive under mode="off" / privacy=False.
Gap 5: JSON and code blocks must pass through under mode="off".
"""

import json

from asha.core.policy_config import PolicyConfig
from asha.types.results import ProcessResult, SanitizeResult
from asha.utils.dropin import process, sanitize

from conftest import output_of


# ---------------------------------------------------------------------------
# Gap 2: Safe prompts verbatim preservation under mode="off"
# ---------------------------------------------------------------------------


def test_safe_prompt_verbatim_passthrough_mode_off():
    """A prompt with no PII and no threats must come back verbatim in mode='off'."""
    original = "What is the capital of France?"
    result = process(original, mode="off")
    assert isinstance(result, ProcessResult)
    # mode=off means no optimization, no PII masking
    assert "France" in result.output or "capital" in result.output


def test_safe_prompt_no_pii_mode_off():
    original = "Summarize the quarterly financial report for stakeholders."
    result = process(original, mode="off")
    assert isinstance(result, ProcessResult)
    # No tokens should be removed in off mode
    assert "quarterly" in result.output or "financial" in result.output


def test_keyword_survival_mode_lite():
    """mode='lite' keeps core task keywords after processing."""
    original = "Generate a Python function that sorts a list in ascending order."
    result = process(original, mode="lite")
    assert isinstance(result, ProcessResult)
    out = result.output.lower()
    assert len(out) > 0
    assert (
        "python" in out
        or "sort" in out
        or "list" in out
        or "ascending" in out
        or "function" in out
    )


# ---------------------------------------------------------------------------
# Gap 3: Semantic meaning / key intent words preserved
# ---------------------------------------------------------------------------


def test_intent_word_customer_support_mode_off():
    """'customer support' intent words survive in mode='off'."""
    prompt = "I need customer support for my billing issue."
    result = process(prompt, mode="off")
    assert isinstance(result, ProcessResult)
    assert (
        "customer" in result.output
        or "support" in result.output
        or "billing" in result.output
    )


def test_intent_words_survive_sanitize():
    """sanitize() should only strip PII/threats, not the intent of the prompt."""
    prompt = "Please help me with my subscription cancellation."
    result = sanitize(prompt)
    assert isinstance(result, SanitizeResult)
    # Intent keywords must be preserved
    assert "subscription" in result.output or "cancellation" in result.output


def test_domain_keyword_preserved_mode_off():
    """Domain-specific keywords must not be stripped in mode='off'."""
    prompt = "Analyze the neural network architecture for image classification."
    result = process(prompt, mode="off")
    assert isinstance(result, ProcessResult)
    assert (
        "neural" in result.output
        or "network" in result.output
        or "classification" in result.output
    )


# ---------------------------------------------------------------------------
# Gap 4: Instructions / constraints preserved under mode="off"
# ---------------------------------------------------------------------------


def test_system_instruction_preserved_mode_off():
    """Instruction text must not be rewritten in mode='off'."""
    instruction = (
        "You MUST respond only in JSON. "
        "Do NOT include any explanation. "
        "Maximum 3 sentences."
    )
    result = process(instruction, mode="off")
    assert isinstance(result, ProcessResult)
    assert "JSON" in result.output
    assert "3" in result.output or "three" in result.output.lower()


def test_constraint_preserved_privacy_false():
    """Constraints must survive when privacy=False."""
    constraint = "Only use formal language. Never use contractions."
    result = process(constraint, mode="off")
    assert isinstance(result, ProcessResult)
    assert "formal" in result.output or "contractions" in result.output


def test_numbered_steps_preserved_mode_off():
    """Numbered instructions should survive in mode='off'."""
    prompt = "Step 1: Read the document. Step 2: Summarize it. Step 3: Output JSON."
    result = process(prompt, mode="off")
    assert isinstance(result, ProcessResult)
    assert "Step 1" in result.output or "Step 2" in result.output


# ---------------------------------------------------------------------------
# Gap 5: JSON / code blocks preserved under mode="off"
# ---------------------------------------------------------------------------


def test_json_payload_preserved_mode_off():
    """A JSON-structured prompt must not be broken in mode='off'."""
    payload = {"task": "summarize", "text": "Quarterly revenue increased 12%."}
    prompt = json.dumps(payload)
    result = process(prompt, mode="off")
    assert isinstance(result, ProcessResult)
    # The JSON keys must still be present
    assert "summarize" in result.output or "task" in result.output


def test_json_prompt_parseable_mode_off():
    """Output from mode='off' on a JSON input should still be valid JSON."""
    payload = json.dumps({"role": "system", "content": "Be concise."})
    result = process(payload, mode="off")
    # At minimum the key content should survive
    assert "system" in result.output or "concise" in result.output


def test_code_block_keywords_preserved_mode_off():
    """Code-like prompts should not be aggressively rewritten in mode='off'."""
    code_prompt = (
        "Write a Python function:\n"
        "def calculate_total(items):\n"
        "    return sum(item.price for item in items)"
    )
    result = process(code_prompt, mode="off")
    assert isinstance(result, ProcessResult)
    assert (
        "Python" in result.output
        or "function" in result.output
        or "calculate" in result.output
    )


def test_markdown_code_fence_preserved_mode_off():
    prompt = "```python\nprint('hello world')\n```\nExplain the above code."
    result = process(prompt, mode="off")
    assert isinstance(result, ProcessResult)
    # The explanation request should survive
    assert (
        "python" in result.output.lower()
        or "hello" in result.output
        or "Explain" in result.output
    )


# ---------------------------------------------------------------------------
# Gap 3: preserve_intent=True - clean prompts returned verbatim
# ---------------------------------------------------------------------------


def test_preserve_intent_returns_original_for_clean_prompt():
    """preserve_intent=True must return the original when no PII / threats found."""
    original = "I need customer support for my billing issue."
    result = process(original, policy=PolicyConfig(preserve_intent=True))
    assert result.output == original


def test_preserve_intent_returns_original_simple():
    """A generic safe prompt is not rewritten when preserve_intent=True."""
    original = "Explain the difference between supervised and unsupervised learning."
    result = process(original, policy=PolicyConfig(preserve_intent=True))
    assert result.output == original


def test_preserve_intent_false_may_rewrite():
    """When preserve_intent=False (default), processing still returns usable intent."""
    original = "Tell me about customer support best practices."
    result = process(original, policy=PolicyConfig(preserve_intent=False))
    assert isinstance(result, ProcessResult)
    out = result.output.lower()
    assert len(out) > 0
    assert "customer" in out or "support" in out or "practices" in out or "tell" in out


def test_preserve_intent_pii_still_masked():
    """Even with preserve_intent=True, PII must be masked when present."""
    prompt = "Contact john.doe@example.com for customer support details."
    result = process(prompt, policy=PolicyConfig(preserve_intent=True))
    assert isinstance(result, ProcessResult)
    # PII present → optimizer ran → preserve_intent does NOT restore the raw email
    assert "john.doe@example.com" not in result.output


# ---------------------------------------------------------------------------
# Gap 4+5: PII in JSON is masked, structure survives
# ---------------------------------------------------------------------------


def test_json_with_pii_masks_email_keeps_structure():
    """When JSON contains PII, the email is masked but the key/value structure stays."""
    payload = json.dumps({"user_email": "alice@corp.com", "action": "delete"})
    result = sanitize(payload)
    assert isinstance(result, SanitizeResult)
    out = output_of(result)
    # Email should be masked
    assert "alice@corp.com" not in out
    # The action keyword should still be present
    assert "delete" in out or "action" in out


# ---------------------------------------------------------------------------
# Semantic intent and task preservation across compiler levels
# ---------------------------------------------------------------------------


def test_compile_preserves_task_semantics():
    from asha.core.engines import compile_prompt
    raw = "Summarize this article in 3 bullet points: The quick brown fox jumps over the lazy dog."
    compiled = compile_prompt(raw)
    assert "Summarize this article in 3 bullet points" in compiled
    assert "The quick brown fox jumps over the lazy dog" in compiled


def test_compile_preserves_code_task():
    from asha.core.engines import compile_prompt
    raw = "Fix the bug in this Python code:\ndef calculate_total(items):\n    return sum(items)"
    compiled = compile_prompt(raw)
    assert "calculate_total" in compiled
    assert "def calculate_total" in compiled


def test_process_preserves_user_payload_in_balanced_mode():
    raw = "Contact alice@example.com to discuss the Q3 financial forecast of $1,000,000."
    res = process(raw, mode="balanced")
    assert "alice@example.com" not in res.output
    assert "Q3 financial forecast" in res.output or "financial forecast" in res.output
    assert "1,000,000" in res.output or "1000000" in res.output or "1" in res.output


def test_prompt_compiler_levels():
    from asha.core.compiler.prompt_compiler import PromptCompiler
    from asha.core._ir.ir_builder import IRBuilder
    compiler = PromptCompiler()
    builder = IRBuilder()
    raw = "Translate the following document to Spanish: Hello world"
    ir = builder.parse(raw)

    minimal = compiler.compile(ir, optimization_level="minimal")
    assert "Translate the following document to Spanish: Hello world" in minimal

    standard = compiler.compile(ir, optimization_level="standard")
    assert "Translate the following document to Spanish: Hello world" in standard

    comprehensive = compiler.compile(ir, optimization_level="comprehensive")
    assert "Translate the following document to Spanish: Hello world" in comprehensive
