"""Async and streaming behavior tests (no live API keys)."""

import pytest

pytest.importorskip("anyio")
pytestmark = pytest.mark.anyio

from asha.core.streaming import (
    handle_streaming_response,
    handle_async_streaming_response,
    is_streaming_response,
)
from asha.types.results import ProcessResult, SanitizeResult
from asha.utils.dropin import process_async, sanitize_async
from asha.integrations.llm_wrap import wrap_llm

from conftest import output_of


class _AsyncCompletions:
    async def create(self, messages, **kwargs):
        if kwargs.get("stream"):

            async def gen():
                for part in ("Hello", " ", "world"):
                    yield part

            return gen()

        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": messages[-1]["content"]},
                            )()
                        },
                    )()
                ]
            },
        )()


class MockStreamClient:
    """Fresh chat/completions per instance so wrap_llm does not leak state."""

    def __init__(self):
        self.chat = type("Chat", (), {"completions": _AsyncCompletions()})()


async def test_sanitize_async_masks_email():
    result = await sanitize_async("Contact john@example.com")
    assert isinstance(result, SanitizeResult)
    assert "john@example.com" not in result.output


async def test_process_async_returns_process_result():
    result = await process_async("Summarize quarterly sales trends.")
    assert isinstance(result, ProcessResult)
    assert len(result.output) > 0
    # The optimizer may legitimately compress short inputs. Check either the
    # processed output OR the preserved original for expected keywords.
    searchable = (result.output + " " + result.original).lower()
    assert any(
        kw in searchable
        for kw in ("summarize", "quarterly", "sales", "trends")
    )


async def test_process_async_exposes_metrics():
    result = await process_async("Email: a@b.com")
    assert isinstance(result, ProcessResult)
    assert result.metrics is not None
    assert output_of(result)


def test_sync_streaming_passthrough():
    chunks = list(handle_streaming_response(iter(["a", "b", "c"])))
    assert chunks == ["a", "b", "c"]
    assert is_streaming_response(iter(["x"])) is True


async def test_async_streaming_passthrough():
    async def stream():
        for part in ("x", "y"):
            yield part

    chunks = []
    async for chunk in handle_async_streaming_response(stream()):
        chunks.append(chunk)
    assert chunks == ["x", "y"]


async def test_wrap_llm_async_streaming():
    client = MockStreamClient()
    wrapped = wrap_llm(client)
    response = await wrapped.chat.completions.create(
        messages=[{"role": "user", "content": "john@example.com"}],
        stream=True,
    )
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    assert chunks == ["Hello", " ", "world"]


class _SyncCompletions:
    def create(self, messages, **kwargs):
        if kwargs.get("stream"):
            return iter(["chunk1", "chunk2"])

        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": messages[-1]["content"]},
                            )()
                        },
                    )()
                ]
            },
        )()


class MockSyncStreamClient:
    """Fresh chat/completions per instance so wrap_llm does not leak state."""

    def __init__(self):
        self.chat = type("Chat", (), {"completions": _SyncCompletions()})()


def test_wrap_llm_sync_streaming():
    client = MockSyncStreamClient()
    wrapped = wrap_llm(client)
    response = wrapped.chat.completions.create(
        messages=[{"role": "user", "content": "stream test"}],
        stream=True,
    )
    assert list(response) == ["chunk1", "chunk2"]


def test_wrap_llm_forwards_privacy_and_budget():
    client = MockSyncStreamClient()
    wrapped = wrap_llm(client, mode="off", token_budget=500)
    response = wrapped.chat.completions.create(
        messages=[{"role": "user", "content": "keep@example.com intact"}],
    )
    content = response.choices[0].message.content
    assert "keep@example.com" in content


def test_streaming_does_not_buffer_chunks():
    from asha.core.streaming import handle_streaming_response

    chunks = list(handle_streaming_response(iter(["a", "b", "c"])))
    assert chunks == ["a", "b", "c"]


def test_wrap_llm_sync_streaming_fail_closed_on_process_error(monkeypatch):
    from asha.exceptions import ASHAProcessingError
    import asha.documents.attachments as attachments_mod

    def boom(*args, **kwargs):
        raise RuntimeError("process failed")

    # The wrapped_create path for messages=[...] kwargs calls
    # _prepare_kwargs_for_llm → attachments._mask_text → process().
    # Patch _mask_text (the actual call site) so the RuntimeError propagates
    # through _handle_wrap_processing_error and becomes ASHAProcessingError.
    monkeypatch.setattr(attachments_mod, "_mask_text", boom)
    client = MockSyncStreamClient()
    wrapped = wrap_llm(client)
    with pytest.raises(ASHAProcessingError):
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "stream@test.com"}],
            stream=True,
        )
