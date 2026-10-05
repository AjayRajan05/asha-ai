"""Unit tests for safe provider status detection and secret protection."""

import os
from examples.showcase.runtime.session import get_provider_statuses


def test_provider_statuses_structure():
    statuses = get_provider_statuses()
    assert len(statuses) >= 6

    provider_names = [s["provider"] for s in statuses]
    assert "Google Gemini" in provider_names
    assert "OpenAI" in provider_names
    assert "Anthropic" in provider_names
    assert "Ollama" in provider_names
    assert "llama.cpp (OpenAI Compatible)" in provider_names
    assert "Local Demo Provider" in provider_names


def test_no_secret_leakage_in_provider_status(monkeypatch):
    # Set simulated fake keys in env
    fake_gemini_key = "AIzaSySecretFakeGeminiKey999"
    fake_openai_key = "sk-proj-RealSecretLookingOpenAIKey888"

    monkeypatch.setenv("GOOGLE_API_KEY", fake_gemini_key)
    monkeypatch.setenv("OPENAI_API_KEY", fake_openai_key)

    statuses = get_provider_statuses()

    gemini_status = next(s for s in statuses if s["provider"] == "Google Gemini")
    openai_status = next(s for s in statuses if s["provider"] == "OpenAI")

    assert gemini_status["configured"] is True
    assert openai_status["configured"] is True

    # Critical check: The secret key must NEVER appear in the dictionary or string representation
    full_str = str(statuses)
    assert fake_gemini_key not in full_str
    assert fake_openai_key not in full_str
    assert "sk-" not in full_str


def test_local_demo_provider_always_configured():
    statuses = get_provider_statuses()
    local_status = next(s for s in statuses if s["provider"] == "Local Demo Provider")
    assert local_status["configured"] is True
    assert "Offline" in local_status["details"]
