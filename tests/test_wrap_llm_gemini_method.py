"""wrap_llm must detect Gemini GenerativeModel.generate_content."""

from __future__ import annotations

from asha.integrations import llm_wrap
from asha.integrations.llm_wrap import _find_generation_method, wrap_llm


class _FakeGemini:
    def generate_content(self, prompt: str, **_kwargs):
        return {"text": f"echo:{prompt}"}


def test_find_generation_method_prefers_generate_content() -> None:
    assert _find_generation_method(_FakeGemini()) == "generate_content"


def test_prepare_args_for_llm_masks_positional_string(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_wrap,
        "_process_prompt_for_wrap",
        lambda prompt, mode, token_budget=1200: f"MASKED:{prompt}",
    )
    args = llm_wrap._prepare_args_for_llm(
        ("secret@example.com hello",), mode="balanced"
    )
    assert args[0] == "MASKED:secret@example.com hello"


def test_wrap_llm_gemini_style_positional_prompt() -> None:
    client = _FakeGemini()
    wrapped = wrap_llm(client, mode="off")
    out = wrapped.generate_content("Contact priya@acme.example please")
    assert "text" in out
    assert "priya@acme.example" in out["text"] or "echo:" in out["text"]
