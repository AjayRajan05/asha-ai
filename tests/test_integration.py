"""Optional live Gemini integration tests.

Run explicitly when an API key is set:

    GEMINI_API_KEY=... pytest tests/test_integration.py -m integration
"""

import os

import pytest

from asha.types import ProcessResult

pytestmark = pytest.mark.integration

_GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@pytest.fixture(scope="module")
def gemini_key():
    if not _GEMINI_KEY:
        pytest.skip("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    return _GEMINI_KEY


def test_dropin_process_live(gemini_key):
    from asha import process

    result = process(
        "Summarize machine learning in one sentence. Email: test@example.com",
        mode="balanced",
    )
    assert isinstance(result, ProcessResult)
    assert "test@example.com" not in result.output


def test_gemini_adapter_generate(gemini_key):
    from asha.runtime.adapters.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(model="gemini-1.5-flash")
    response = adapter.generate("Reply with the word OK only.")
    assert isinstance(response, str)
    assert "ok" in response.lower()


def test_agent_live_run(gemini_key):
    from asha import Agent

    agent = Agent(model="gemini-1.5-flash", privacy=True)
    response = agent.run("What is 2+2? Reply with the number only.")
    assert isinstance(response, str)
    assert any(ch.isdigit() for ch in response)
