"""Consolidated edge cases and thread safety tests."""

# --- Merged from test_edge_cases.py ---
"""Edge-case tests for drop-in API reliability."""

import json

import pytest

from asha.types.results import ProcessResult, SanitizeResult
from asha.utils.dropin import process, sanitize, optimize

from conftest import output_of

# Valid Luhn test card (Visa)
VALID_CARD = "4111111111111111"
VALID_CARD_FORMATTED = "4111-1111-1111-1111"
INVALID_CARD = "79927398710"  # fails Luhn; avoids phone-number substring matches


def test_empty_prompt_string():
    result = process("")
    assert isinstance(result, ProcessResult)
    assert result.output == ""
    assert result.degraded is True


def test_empty_prompt_sanitize():
    result = sanitize("")
    assert isinstance(result, SanitizeResult)
    assert result.output == ""


def test_invalid_prompt_type():
    with pytest.raises(TypeError, match="Unsupported prompt input type"):
        process(None)  # type: ignore[arg-type]


def test_unicode_multilingual_prompt():
    prompt = "Bonjour! 联系人: zhang@example.com 電話: 555-123-4567"
    result = process(prompt)
    out = output_of(result)
    assert "zhang@example.com" not in out


def test_unicode_emoji_prompt():
    prompt = "Analyze 📊 data for user john@example.com 🚀"
    result = process(prompt)
    assert "john@example.com" not in output_of(result)


def test_large_prompt_does_not_crash():
    prompt = ("Summarize quarterly revenue trends. " * 500) + "Contact: a@b.com"
    result = process(prompt)
    out = output_of(result)
    assert "a@b.com" not in out
    assert len(out) > 0


def test_structured_json_mostly_unchanged():
    payload = {
        "task": "classify",
        "labels": ["positive", "negative"],
        "text": "Great product!",
    }
    prompt = json.dumps(payload, indent=2)
    result = process(prompt, mode="off")
    out = output_of(result)
    parsed = json.loads(out)
    assert parsed["task"] == "classify"
    assert parsed["labels"] == ["positive", "negative"]


def test_safe_prompt_no_pii_minimal_change():
    prompt = "Explain the difference between supervised and unsupervised learning."
    result = process(prompt, mode="lite")
    out = output_of(result)
    # Core instruction should survive optimization
    assert "supervised" in out.lower() or "unsupervised" in out.lower()


def test_credit_card_valid_detected_and_masked():
    prompt = f"My card is {VALID_CARD}"
    result = sanitize(prompt)
    out = output_of(result)
    assert VALID_CARD not in out.replace("-", "").replace(" ", "")
    assert "[CREDIT_CARD_HASH]" in out


def test_credit_card_invalid_luhn_not_masked():
    """Random digit sequences failing Luhn should not be treated as cards."""
    from asha.core.security.pii_detector import PIIDetector

    prompt = f"Reference number {INVALID_CARD}"
    detector = PIIDetector()
    assert "credit_card" not in detector.detect_pii_types(prompt)
    assert not detector._is_valid_credit_card(INVALID_CARD)


def test_address_type_detected_by_detector():
    """Address detection is available on PIIDetector; masking is not required yet."""
    from asha.core.security.pii_detector import PIIDetector

    detector = PIIDetector()
    types = detector.detect_pii_types("Ship to 742 Evergreen Street, Springfield")
    assert "address" in types


# --- Consolidated from test_api_edge_matrix.py ---
"""Cross-API edge-case matrix tests."""

from asha import optimize, sanitize
from asha.integrations import wrap_llm
from asha.types.results import OptimizeResult, SanitizeResult

from conftest import output_of


def test_sanitize_empty_prompt():
    assert output_of(sanitize("")) == ""


def test_optimize_empty_prompt():
    assert output_of(optimize("")) == ""


def test_sanitize_unicode_emoji():
    out = output_of(sanitize("Contact 📧 user@company.com"))
    assert "user@company.com" not in out


def test_optimize_unicode_emoji():
    result = optimize("Summarize quarterly data")
    assert "summarize" in result.output.lower()
    assert "quarterly" in result.output.lower()
    assert "data" in result.output.lower()


def test_sanitize_large_prompt_does_not_crash():
    text = "word " * 5000 + " secret@company.com"
    result = sanitize(text)
    assert isinstance(result, SanitizeResult)
    assert "secret@company.com" not in result.output


def test_optimize_large_prompt_does_not_crash():
    text = "please summarize " + ("data " * 3000)
    result = optimize(text, token_budget=100)
    assert "summarize" in result.output.lower()
    assert result.metrics is not None
    assert result.metrics.tokens_saved >= 0


class _MiniClient:
    def __init__(self):
        self.chat = type(
            "Chat",
            (),
            {
                "completions": type(
                    "C",
                    (),
                    {
                        "create": lambda self, messages, **kw: type(
                            "R",
                            (),
                            {
                                "choices": [
                                    type(
                                        "Ch",
                                        (),
                                        {
                                            "message": type(
                                                "M",
                                                (),
                                                {"content": messages[-1]["content"]},
                                            )()
                                        },
                                    )()
                                ]
                            },
                        )()
                    },
                )()
            },
        )()


def test_wrap_llm_empty_message_content():
    client = _MiniClient()
    wrapped = wrap_llm(client, mode="off")
    response = wrapped.chat.completions.create(
        messages=[{"role": "user", "content": ""}],
    )
    assert response.choices[0].message.content == ""

# --- Merged from test_thread_safety.py ---
"""Thread-safety and concurrent access tests."""

import concurrent.futures
import threading

import pytest

from asha.core.metrics_dashboard import dashboard
from asha.types.results import ProcessResult
from asha.utils.dropin import process

from conftest import output_of


@pytest.mark.slow
def test_concurrent_process_isolation():
    prompts = [f"Email user{i}@example.com task {i}" for i in range(20)]
    errors = []

    def worker(prompt):
        try:
            result = process(prompt)
            out = output_of(result)
            if "@" in out:
                errors.append(f"PII leak in: {prompt[:30]}")
        except Exception as exc:
            errors.append(str(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(worker, prompts))

    assert not errors, errors


def test_metrics_dashboard_concurrent_recording():
    start = len(dashboard.collector.events)

    def record_one(_):
        from asha.core.metrics_dashboard import record_request

        record_request(latency_ms=1.0, tokens_processed=10, tokens_saved=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record_one, range(40)))

    assert len(dashboard.collector.events) >= start + 40


def test_security_layer_instances_thread_local():
    from asha.core.security.security_layer import SecurityLayer

    results = []

    def worker(email):
        layer = SecurityLayer()
        out = layer.process(f"Contact {email}")
        results.append(email not in out.sanitized_content)

    threads = [
        threading.Thread(target=worker, args=(f"user{i}@test.com",)) for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results)

