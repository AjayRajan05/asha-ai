from __future__ import annotations
"""Phase 3: telemetry opt-in and sandbox isolation modes."""


from pathlib import Path

import pytest

from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.sandbox import SandboxManager
from asha.runtime.anchor.telemetry import AnchorTelemetry


def test_telemetry_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY", raising=False)
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    tel = AnchorTelemetry()
    assert tel.enabled is False
    assert not (tmp_path / "anchor_telemetry.jsonl").exists()
    # emit is a no-op
    tel._emit({"event_type": "noop"})
    assert not (tmp_path / "anchor_telemetry.jsonl").exists()


def test_telemetry_enabled_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "t.jsonl"
    monkeypatch.setenv("ASHA_ANCHOR_TELEMETRY", "1")
    monkeypatch.setenv("ASHA_ANCHOR_TELEMETRY_PATH", str(log))
    tel = AnchorTelemetry()
    assert tel.enabled is True
    tel._emit({"event_type": "test"})
    assert log.is_file()
    assert "test" in log.read_text(encoding="utf-8")


def test_runtime_default_telemetry_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY", raising=False)
    monkeypatch.delenv("ASHA_ANCHOR_TELEMETRY_PATH", raising=False)
    rt = AnchorRuntime(interactive=False)
    assert rt.telemetry.enabled is False


def test_sandbox_rejects_docker_mode() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        SandboxManager(mode="docker")


def test_sandbox_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown sandbox"):
        SandboxManager(mode="quantum")


def test_sandbox_auto_and_hard_accepted() -> None:
    assert SandboxManager(mode="auto").mode == "auto"
    assert SandboxManager(mode="hard").mode == "hard"
    assert SandboxManager(mode="subprocess").mode == "subprocess"


# --- Consolidated from tests/runtime/anchor/test_anchor_extensions.py ---
"""Extension adapter tests: AutoGen GroupChat, LlamaIndex workflows."""


import pytest

from asha.runtime.anchor.adapters.autogen import anchor_autogen, is_autogen_group_chat
from asha.runtime.anchor.adapters.llamaindex import (
    anchor_llamaindex,
    is_llamaindex_agent,
)
from asha.runtime.anchor.runtime import AnchorRuntime


class _AutoGenGroupChat:
    __module__ = "autogen.agentchat.groupchat"

    def __init__(self) -> None:
        self.agents = [_AutoGenAgent()]
        self.max_round = 5

    def initiate_chat(self, *args, **kwargs):
        return "group done"


class _AutoGenAgent:
    __module__ = "autogen.agentchat.conversable_agent"

    def __init__(self) -> None:
        self.function_map = {"load_trend_data": lambda: "loaded"}

    def register_for_llm(self, name: str, description: str = "") -> None:
        return None

    def generate_reply(self, messages=None, **kwargs):
        return {"content": "loaded"}


class _LlamaWorkflow:
    __module__ = "llama_index.core.workflow"

    def __init__(self) -> None:
        class _T:
            name = "load_trend_data"

            def _run(self, **kwargs):
                return "loaded"

        self.tools = [_T()]

    def run(self, input: str):
        return type("R", (), {"response": self.tools[0]._run()})()


def test_autogen_group_chat_wrapping() -> None:
    group = _AutoGenGroupChat()
    assert is_autogen_group_chat(group)
    wrapped = anchor_autogen(group, interactive=False)
    result = wrapped.initiate_chat(message="Analyze locally.")
    assert result == "group done"
    assert hasattr(wrapped.agents[0], "_anchor_runtime")


def test_llamaindex_workflow_run() -> None:
    pytest.importorskip("llama_index.core", reason="requires llama-index-core")
    workflow = _LlamaWorkflow()
    assert is_llamaindex_agent(workflow)
    wrapped = anchor_llamaindex(workflow, interactive=False)
    result = wrapped.run("Analyze trends locally.")
    assert "loaded" in str(getattr(result, "response", result))
