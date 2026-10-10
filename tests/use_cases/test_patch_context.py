# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-world use-case tests: Scoped patch_context manager."""

from __future__ import annotations

from typing import Any, Dict, List
import pytest

from asha import Profile, SecureLLM
from asha.integrations import patch_context
from asha.integrations.auto_patch import get_patch_status


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.recorded_messages: List[Dict[str, str]] = []

    def create(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        self.recorded_messages = messages
        user_msg = messages[-1]["content"] if messages else ""
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": f"Echo: {user_msg}"})()},
                    )()
                ]
            },
        )()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions()})()


def test_patch_context_client_scoped_mode():
    """patch_context(client=...) yields a SecureLLM proxy without global mutations."""
    raw_client = _FakeClient()
    original_create = raw_client.chat.completions.create

    with patch_context(client=raw_client, profile=Profile.BALANCED()) as secured:
        assert isinstance(secured, SecureLLM)
        resp = secured.chat.completions.create(
            messages=[{"role": "user", "content": "Contact ceo@acme.org for details"}]
        )
        assert resp is not None
        # Underlying client only saw masked email
        sent_content = raw_client.chat.completions.recorded_messages[-1]["content"]
        assert "ceo@acme.org" not in sent_content

    # Original client method was never modified in-place
    assert raw_client.chat.completions.create == original_create


def test_patch_context_provider_scoped_clean_restoration(monkeypatch):
    """patch_context(providers=...) strictly unpatches modules upon block exit."""
    # Ensure starting from clean unpatched state
    status_before = get_patch_status()
    initial_patch_count = status_before["total_patches_applied"]

    with patch_context(providers=["openai"], mode="balanced"):
        status_inside = get_patch_status()
        # Inside the block, patching was enabled
        assert status_inside["enabled"] is True

    # Outside the block, all patches were restored
    status_after = get_patch_status()
    assert status_after["total_patches_applied"] == initial_patch_count


def test_patch_context_reverts_even_when_exception_occurs():
    """patch_context guarantees unpatching even if an error is raised inside the context."""
    status_before = get_patch_status()
    initial_patch_count = status_before["total_patches_applied"]

    with pytest.raises(RuntimeError, match="Intentional test crash"):
        with patch_context(providers=["openai"], mode="strict"):
            assert get_patch_status()["enabled"] is True
            raise RuntimeError("Intentional test crash")

    # In finally block, state was restored
    status_after = get_patch_status()
    assert status_after["total_patches_applied"] == initial_patch_count
