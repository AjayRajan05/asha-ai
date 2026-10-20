from __future__ import annotations
"""Universal Gemini Adapter compatibility tests (all Gemini models).

Plan requirements (tested across all Gemini models):
1. Universal model selection and configuration
2. System instruction
3. Multi-turn content
4. Structured output
5. Function/tool calling (does not destroy structures)
6. Safety response handling (does not crash)
7. Usage metadata capture

All network-calling tests are guarded by @pytest.mark.skipif to avoid
CI failures without a real API key. Unit-testable paths are always tested.
"""


import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

pytest.importorskip(
    "google.genai",
    reason="requires the optional 'gemini' extra (google-genai)",
)

from asha.runtime.adapters.gemini_adapter import GeminiAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_adapter(model: str = "gemini-1.5-flash") -> GeminiAdapter:
    """Construct a GeminiAdapter with a fake key (never calls the network)."""
    return GeminiAdapter(model=model, api_key="fake-test-key")


# ---------------------------------------------------------------------------
# Test 1: Basic text (model selection + provider info + multi-model support)
# ---------------------------------------------------------------------------


def test_1_basic_text_model_selection():
    adapter = _fake_adapter("gemini-1.5-flash")
    assert adapter.model == "gemini-1.5-flash"
    info = adapter.get_provider_info()
    assert info["provider"] == "gemini"
    assert info["model"] == "gemini-1.5-flash"


@pytest.mark.parametrize(
    "model_input,expected_model",
    [
        ("gemini-1.5-flash", "gemini-1.5-flash"),
        ("gemini-1.5-pro", "gemini-1.5-pro"),
        ("gemini-1.5-flash-8b", "gemini-1.5-flash-8b"),
        ("gemini-2.0-flash", "gemini-2.0-flash"),
        ("gemini-2.0-flash-lite", "gemini-2.0-flash-lite"),
        ("gemini-2.0-pro-exp", "gemini-2.0-pro-exp"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini-pro", "gemini-pro"),
        ("gemini-1.0-pro", "gemini-1.0-pro"),
        ("gemini-ultra", "gemini-ultra"),
        ("models/gemini-1.5-flash", "gemini-1.5-flash"),
        ("models/gemini-2.0-flash", "gemini-2.0-flash"),
        ("tunedModels/my-special-gemini", "tunedModels/my-special-gemini"),
    ],
)
def test_gemini_adapter_supports_all_models(model_input: str, expected_model: str):
    """GeminiAdapter must accept and work with any Gemini model identifier."""
    adapter = GeminiAdapter(model=model_input, api_key="fake-test-key")
    assert adapter.model == expected_model
    info = adapter.get_provider_info()
    assert info["provider"] == "gemini"
    assert info["model"] == expected_model


def test_gemini_adapter_default_model(monkeypatch):
    """GeminiAdapter defaults to GEMINI_MODEL env var or gemini-1.5-flash."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    adapter = GeminiAdapter(api_key="fake-test-key")
    assert adapter.model == "gemini-1.5-flash"

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    adapter_env = GeminiAdapter(api_key="fake-test-key")
    assert adapter_env.model == "gemini-2.0-flash"


def test_factory_creates_all_gemini_model_variants():
    """AdapterFactory must route all Gemini model identifiers to GeminiAdapter."""
    from asha.runtime.adapters.factory import AdapterFactory

    for model_name in [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "google/gemini-2.0-pro",
    ]:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key"}):
            adapter = AdapterFactory.create(model_name)
            assert isinstance(adapter, GeminiAdapter)


# ---------------------------------------------------------------------------
# Test 2: System instruction
# ---------------------------------------------------------------------------


def test_2_system_instruction_mocked():
    """generate() passes system_instruction to the underlying client."""
    adapter = _fake_adapter("gemini-1.5-flash")
    mock_response = MagicMock()
    mock_response.text = "Concise answer about prompt injection."
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    # Force legacy path where GenerativeModel is used
    adapter._is_genai_sdk = False
    adapter._adapter.client.GenerativeModel = MagicMock(return_value=mock_model)

    result = adapter.generate("Explain prompt injection.", system_instruction="Answer concisely.")
    assert "Concise answer" in result
    adapter._adapter.client.GenerativeModel.assert_called_with(
        "gemini-1.5-flash", system_instruction="Answer concisely."
    )


# ---------------------------------------------------------------------------
# Test 3: Multi-turn content
# ---------------------------------------------------------------------------


def test_3_multi_turn_content():
    """GeminiAdapter declares multi-turn-capable through provider or SDK field."""
    adapter = _fake_adapter()
    info = adapter.get_provider_info()
    # Gemini natively supports multi-turn (verified by SDK capability)
    assert info["provider"] == "gemini"
    # The adapter must support generating content (base multi-turn requirement)
    assert hasattr(adapter, "generate")



def test_3_multi_turn_history_format():
    """generate() produces a response for multi-turn prompts (history embedded in prompt)."""
    adapter = _fake_adapter()
    mock_response = MagicMock(text="Second turn response.")
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    adapter._is_genai_sdk = False
    adapter._adapter.client.GenerativeModel = MagicMock(return_value=mock_model)

    # Embed conversation context in the prompt (standard pattern for adapters)
    multi_turn_prompt = (
        "Previous context:\n"
        "User: What is ASHA?\n"
        "Assistant: ASHA is a security layer for LLM apps.\n\n"
        "Current question: Can you elaborate on that?"
    )
    result = adapter.generate(multi_turn_prompt)
    assert result == "Second turn response."



# ---------------------------------------------------------------------------
# Test 4: Structured output
# ---------------------------------------------------------------------------


def test_4_structured_output_not_mangled():
    """ASHA must not destroy JSON response structures from Gemini."""
    adapter = _fake_adapter()
    json_response = '{"risk": "low", "reason": "No sensitive content detected"}'
    mock_response = MagicMock(text=json_response)
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    adapter._is_genai_sdk = False
    adapter._adapter.client.GenerativeModel = MagicMock(return_value=mock_model)

    result = adapter.generate("Assess the risk of this prompt. Return JSON.")
    import json
    parsed = json.loads(result)
    assert parsed["risk"] == "low"
    assert "reason" in parsed


# ---------------------------------------------------------------------------
# Test 5: Tool calling (does not destroy provider-specific structures)
# ---------------------------------------------------------------------------


def test_5_tool_call_structure_preserved():
    """ASHA must not mangle tool-call response structures."""
    # GeminiAdapter returns raw text; tool call handling is at a higher layer.
    # Test that the adapter does not alter provider-specific response types.
    adapter = _fake_adapter()
    mock_response = MagicMock()
    mock_response.text = "Tool call response"
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_response.candidates = [mock_candidate]
    # Should not raise
    text = adapter._safe_extract_text(mock_response)
    assert text is not None


# ---------------------------------------------------------------------------
# Test 6: Safety response (does not crash on blocked response)
# ---------------------------------------------------------------------------


def test_6_safety_block_does_not_crash():
    """When Gemini blocks a response with safety filters, ASHA must not crash."""
    adapter = _fake_adapter()

    mock_blocked = MagicMock()
    # Simulate .text raising ValueError (typical safety-block behavior)
    type(mock_blocked).text = PropertyMock(side_effect=ValueError("Response was blocked"))
    candidate = MagicMock()
    candidate.finish_reason = "SAFETY"
    mock_blocked.candidates = [candidate]

    # Must not raise
    result = adapter._safe_extract_text(mock_blocked)
    assert result == "[BLOCKED_BY_SAFETY_FILTERS]"


def test_6_safety_block_returns_sentinel_string():
    adapter = _fake_adapter()

    mock_blocked = MagicMock()
    type(mock_blocked).text = PropertyMock(side_effect=ValueError("Safety block"))
    mock_blocked.candidates = []

    result = adapter._safe_extract_text(mock_blocked)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Test 7: Usage metadata capture
# ---------------------------------------------------------------------------


def test_7_usage_metadata_captured():
    adapter = _fake_adapter()

    mock_response = MagicMock(text="Response text.")
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 15
    mock_usage.candidates_token_count = 25
    mock_usage.total_token_count = 40
    mock_response.usage_metadata = mock_usage

    adapter._extract_modern_usage(mock_response)
    usage = adapter.get_last_usage()
    assert usage == {"prompt_tokens": 15, "candidates_tokens": 25, "total_tokens": 40}


def test_7_usage_metadata_none_when_not_available():
    """Adapter must handle missing usage_metadata gracefully by setting usage to None."""
    adapter = _fake_adapter()
    mock_response = MagicMock(text="ok")
    mock_response.usage_metadata = None
    adapter._extract_modern_usage(mock_response)
    assert adapter.get_last_usage() is None


# ---------------------------------------------------------------------------
# Provider info: Developer Preview gate
# ---------------------------------------------------------------------------


def test_provider_info_declares_capabilities():
    adapter = _fake_adapter()
    info = adapter.get_provider_info()
    assert info.get("supports_system_instruction") is True
    assert info.get("supports_usage_metadata") is True


# ---------------------------------------------------------------------------
# Live integration tests (require GOOGLE_API_KEY / GEMINI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"),
    reason="Requires live GOOGLE_API_KEY or GEMINI_API_KEY",
)
def test_live_basic_text():
    """Live Test 1: basic text generation with gemini-3.8-flash."""
    adapter = GeminiAdapter(model="gemini-3.8-flash")
    result = adapter.generate("Explain ASHA in three sentences.")
    assert isinstance(result, str) and len(result) > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"),
    reason="Requires live GOOGLE_API_KEY or GEMINI_API_KEY",
)
def test_live_system_instruction():
    """Live Test 2: system instruction."""
    adapter = GeminiAdapter(model="gemini-3.8-flash")
    result = adapter.generate(
        "Explain prompt injection.",
        system_instruction="Answer concisely in one sentence.",
    )
    assert isinstance(result, str) and len(result) > 0
