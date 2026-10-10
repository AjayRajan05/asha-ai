"""Mock-based adapter tests - no network required."""

from unittest.mock import MagicMock, patch

import pytest

from asha.runtime.adapters.factory import AdapterFactory


def test_mock_adapter_generate():
    adapter = AdapterFactory.create("mock")
    result = adapter.generate("Hello world")
    assert isinstance(result, str)
    assert "Hello world" in result or result.strip() != ""
    assert adapter.__class__.__name__ == "MockAdapter"


@pytest.mark.parametrize(
    "model_name,adapter_class,env,required_pkg",
    [
        ("gpt-4", "OpenAIAdapter", {"OPENAI_API_KEY": "test-key"}, "openai"),
        (
            "claude-3-sonnet",
            "ClaudeAdapter",
            {"ANTHROPIC_API_KEY": "test-key"},
            "anthropic",
        ),
        (
            "gemini-pro",
            "GeminiAdapter",
            {"GOOGLE_API_KEY": "test-key"},
            "google.generativeai",
        ),
        ("llama2", "OllamaAdapter", {}, None),
        ("grok-beta", "GrokAdapter", {"GROK_API_KEY": "test-key"}, "openai"),
    ],
)
def test_adapter_init_no_network(model_name, adapter_class, env, required_pkg):
    if required_pkg:
        pytest.importorskip(required_pkg)
    with patch.dict("os.environ", env, clear=False):
        adapter = AdapterFactory.create(model_name)
        assert adapter.__class__.__name__ == adapter_class


def test_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        AdapterFactory.create("unknown-model-xyz")


def test_openai_generate_mocked():
    pytest.importorskip("openai")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        adapter = AdapterFactory.create("gpt-4")
        mock_msg = MagicMock()
        mock_msg.content = "OpenAI mock response"
        mock_choice = MagicMock(message=mock_msg)
        mock_response = MagicMock(choices=[mock_choice])
        adapter._adapter.client.chat.completions.create = MagicMock(
            return_value=mock_response
        )
        assert adapter.generate("hello") == "OpenAI mock response"
        adapter._adapter.client.chat.completions.create.assert_called_once()


def test_anthropic_generate_mocked():
    pytest.importorskip("anthropic")
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        adapter = AdapterFactory.create("claude-3-sonnet")
        mock_block = MagicMock(text="Claude mock response")
        mock_response = MagicMock(content=[mock_block])
        adapter.client.messages.create = MagicMock(return_value=mock_response)
        assert adapter.generate("hello") == "Claude mock response"


def test_gemini_generate_mocked():
    pytest.importorskip("google.generativeai")
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=False):
        adapter = AdapterFactory.create("gemini-pro")
        mock_response = MagicMock(text="Gemini mock response")
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        adapter._adapter.client.GenerativeModel = MagicMock(return_value=mock_model)
        assert adapter.generate("hello") == "Gemini mock response"


def test_grok_generate_mocked():
    pytest.importorskip("openai")
    with patch.dict("os.environ", {"GROK_API_KEY": "test-key"}, clear=False):
        adapter = AdapterFactory.create("grok-beta")
        mock_msg = MagicMock()
        mock_msg.content = "Grok mock response"
        mock_choice = MagicMock(message=mock_msg)
        mock_response = MagicMock(choices=[mock_choice])
        adapter.client.chat.completions.create = MagicMock(return_value=mock_response)
        assert adapter.generate("hello") == "Grok mock response"


def test_llamacpp_adapter_endpoint():
    pytest.importorskip("openai")
    with patch.dict("os.environ", {"LLAMA_CPP_BASE_URL": "http://localhost:8080/v1"}, clear=False):
        adapter = AdapterFactory.create("llama.cpp-mistral-7b")
        assert adapter.__class__.__name__ == "OpenAIAdapter"
        assert adapter.base_url == "http://localhost:8080/v1"
