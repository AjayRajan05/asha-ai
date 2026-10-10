"""Production-grade adapter tests for all ANCHOR framework integrations."""

from __future__ import annotations

import pytest

from asha.runtime.anchor.adapters.autogen import anchor_autogen, is_autogen_agent
from asha.runtime.anchor.adapters.generic import anchor_generic
from asha.runtime.anchor.adapters.langchain import anchor_langchain, is_langchain_agent
from asha.runtime.anchor.adapters.llamaindex import (
    anchor_llamaindex,
    is_llamaindex_agent,
)
from asha.runtime.anchor.adapters.mcp import anchor_mcp
from asha.runtime.anchor.adapters.registry import anchor_any
from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.tool_bridge import AnchoredToolDelegate, wrap_tool


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _StubTool:
    name = "load_trend_data"
    description = "Load trend data from local file."

    def __init__(self) -> None:
        self.calls = 0

    def _run(self, **kwargs) -> str:
        self.calls += 1
        return "data loaded from trends.csv"


class _DangerTool:
    name = "send_email"
    description = "Send email externally."

    def _run(self, **kwargs) -> str:
        return "Email sent to alice@example.com"


# ---------------------------------------------------------------------------
# Generic adapter
# ---------------------------------------------------------------------------


def test_generic_adapter_governs_tools_and_memory() -> None:
    runtime = AnchorRuntime(interactive=False)

    class Agent:
        def __init__(self) -> None:
            self.tools = [_StubTool(), _DangerTool()]

        def run(self, prompt: str) -> str:
            return self.tools[0]._run()

    wrapped = anchor_generic(Agent(), runtime=runtime)
    result = wrapped.run("Analyze trends locally. Do not send externally.")
    assert result == "data loaded from trends.csv"
    assert runtime.state.baseline_mission is not None
    assert runtime.state.baseline_mission.local_only is True
    assert "load_trend_data" in runtime.state.tools_used


def test_generic_adapter_blocks_dangerous_tool() -> None:
    runtime = AnchorRuntime(interactive=False)

    class Agent:
        tools = [_DangerTool()]

        def run(self, prompt: str) -> str:
            return self.tools[0]._run()

    wrapped = anchor_generic(Agent(), runtime=runtime)
    result = wrapped.run("Analyze locally. Do not send externally.")
    assert "denied" in result.lower() or "blocked" in result.lower()


def test_generic_adapter_invoke_path() -> None:
    runtime = AnchorRuntime(interactive=False)

    class Agent:
        tools = [_StubTool()]

        def invoke(self, input: dict) -> str:
            return self.tools[0]._run()

    wrapped = anchor_generic(Agent(), runtime=runtime)
    assert wrapped.invoke({"input": "load locally"}) == "data loaded from trends.csv"


# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------


class _LangChainStubTool:
    name = "load_trend_data"
    description = "Load data"

    def invoke(self, input: dict) -> str:
        return "loaded"

    def run(self, text: str) -> str:
        return "loaded"


class _LangChainExecutor:
    __module__ = "langchain.agents.agent"

    def __init__(self) -> None:
        self.tools = [_LangChainStubTool(), _DangerTool()]
        self.agent = type("Agent", (), {"tools": self.tools})()

    def invoke(self, input: dict, config=None) -> dict:
        self.tools[0].invoke(input)
        return {"result": "ok", "output": "trend summary"}

    def run(self, input: str) -> str:
        return "loaded"


def test_langchain_detection_and_wrapping() -> None:
    pytest.importorskip("langchain_core", reason="requires langchain-core")
    executor = _LangChainExecutor()
    assert is_langchain_agent(executor)
    runtime = AnchorRuntime(interactive=False)
    wrapped = anchor_langchain(executor, runtime=runtime)
    result = wrapped.invoke({"input": "Analyze trends locally."})
    assert result["result"] == "ok"
    assert runtime.state.baseline_mission is not None


def test_langchain_blocks_dangerous_tool() -> None:
    runtime = AnchorRuntime(interactive=False)
    runtime.initialize_mission(
        "Analyze locally. Do not send externally.",
        context={"available_tools": ["send_email"], "local_only": True},
    )
    wrapped_tool = wrap_tool(_DangerTool(), runtime)
    result = wrapped_tool._run()
    assert "denied" in result.lower() or "blocked" in result.lower()


# ---------------------------------------------------------------------------
# MCP adapter
# ---------------------------------------------------------------------------


class _MCPServer:
    def list_tools(self):
        return [{"name": "load_trend_data"}, {"name": "send_email"}]

    def call_tool(self, name: str, arguments=None):
        return f"Tool result: {name}"


def test_mcp_adapter_session_and_governance() -> None:
    runtime = AnchorRuntime(interactive=False)
    server = anchor_mcp(_MCPServer(), runtime=runtime)
    server.initialize_session("Analyze trends locally. Do not send externally.")

    result = server.call_tool("load_trend_data", {"path": "data/trends.csv"})
    assert "load_trend_data" in result
    assert "load_trend_data" in runtime.state.tools_used


def test_mcp_adapter_blocks_dangerous_tool() -> None:
    runtime = AnchorRuntime(interactive=False)
    server = anchor_mcp(
        _MCPServer(),
        runtime=runtime,
        mission="Analyze locally. Do not send externally.",
    )
    result = server.call_tool("send_email", {"to": "x@y.com"})
    assert "denied" in result.lower() or "blocked" in result.lower()


def test_mcp_proxy_does_not_mutate_inner() -> None:
    inner = _MCPServer()
    server = anchor_mcp(inner, runtime=AnchorRuntime(interactive=False))
    assert isinstance(server, type(server))  # proxy returned
    assert not hasattr(inner, "_anchor_runtime")
    assert server._inner is inner


# ---------------------------------------------------------------------------
# AutoGen adapter
# ---------------------------------------------------------------------------


class _AutoGenAgent:
    __module__ = "autogen.agentchat.conversable_agent"

    def __init__(self) -> None:
        self.function_map = {
            "load_trend_data": _StubTool()._run,
            "send_email": _DangerTool()._run,
        }
        self._register_calls = 0

    def register_for_llm(self, name: str, description: str = "") -> None:
        self._register_calls += 1

    def register_for_execution(self, name: str, func) -> None:
        self.function_map[name] = func

    def generate_reply(self, messages=None, **kwargs) -> dict:
        result = self.function_map["load_trend_data"]()
        return {"content": result}


def test_autogen_detection_and_wrapping() -> None:
    agent = _AutoGenAgent()
    assert is_autogen_agent(agent)
    runtime = AnchorRuntime(interactive=False)
    wrapped = anchor_autogen(agent, runtime=runtime)
    result = wrapped.generate_reply(
        messages=[{"role": "user", "content": "Analyze locally."}]
    )
    assert "data loaded" in str(result)


def test_autogen_blocks_dangerous_function() -> None:
    runtime = AnchorRuntime(interactive=False)
    agent = _AutoGenAgent()
    wrapped = anchor_autogen(agent, runtime=runtime)
    wrapped.generate_reply(messages=[{"role": "user", "content": "Analyze locally."}])
    dangerous = wrapped.function_map["send_email"]
    result = dangerous()
    assert "denied" in result.lower() or "blocked" in result.lower()


# ---------------------------------------------------------------------------
# LlamaIndex adapter
# ---------------------------------------------------------------------------


class _LlamaIndexAgent:
    __module__ = "llama_index.core.agent.react"

    def __init__(self) -> None:
        self.tools = [_StubTool()]

    def query(self, query_str: str) -> object:
        result = self.tools[0]._run()
        return type("Response", (), {"response": result})()


def test_llamaindex_detection_and_wrapping() -> None:
    pytest.importorskip("llama_index.core", reason="requires llama-index-core")
    agent = _LlamaIndexAgent()
    assert is_llamaindex_agent(agent)
    runtime = AnchorRuntime(interactive=False)
    wrapped = anchor_llamaindex(agent, runtime=runtime)
    result = wrapped.query("Analyze trends locally.")
    assert "data loaded" in str(result)


# ---------------------------------------------------------------------------
# Tool delegate
# ---------------------------------------------------------------------------


def test_anchored_tool_delegate_preserves_metadata() -> None:
    runtime = AnchorRuntime(interactive=False)
    runtime.initialize_mission(
        "Load data locally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )
    inner = _StubTool()
    delegate = AnchoredToolDelegate(inner, runtime)
    assert delegate.name == "load_trend_data"
    assert delegate.description == inner.description
    assert delegate._run() == "data loaded from trends.csv"
    assert inner.calls == 1


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_anchor_any_dispatches_langchain() -> None:
    pytest.importorskip("langchain_core", reason="requires langchain-core")
    wrapped = anchor_any(_LangChainExecutor(), interactive=False)
    result = wrapped.invoke({"input": "Analyze trends locally."})
    assert result["result"] == "ok"
    assert hasattr(wrapped, "invoke")


def test_anchor_any_dispatches_mcp() -> None:
    wrapped = anchor_any(_MCPServer(), interactive=False)
    wrapped.initialize_session("Analyze locally. Do not send externally.")
    result = wrapped.call_tool("send_email", {})
    assert "denied" in result.lower() or "blocked" in result.lower()


def test_anchor_any_dispatches_autogen() -> None:
    wrapped = anchor_any(_AutoGenAgent(), interactive=False)
    result = wrapped.generate_reply(
        messages=[{"role": "user", "content": "Analyze locally."}]
    )
    assert "data loaded" in str(result)
    assert hasattr(wrapped, "generate_reply")


def test_anchor_any_dispatches_llamaindex() -> None:
    pytest.importorskip("llama_index.core", reason="requires llama-index-core")
    wrapped = anchor_any(_LlamaIndexAgent(), interactive=False)
    result = wrapped.query("Analyze trends locally.")
    assert "data loaded" in str(result)
    assert hasattr(wrapped, "query")
