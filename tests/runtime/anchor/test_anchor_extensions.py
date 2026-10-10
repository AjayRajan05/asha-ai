"""Extension adapter tests: AutoGen GroupChat, LlamaIndex workflows."""

from __future__ import annotations

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
    workflow = _LlamaWorkflow()
    assert is_llamaindex_agent(workflow)
    wrapped = anchor_llamaindex(workflow, interactive=False)
    result = wrapped.run("Analyze trends locally.")
    assert "loaded" in str(getattr(result, "response", result))
