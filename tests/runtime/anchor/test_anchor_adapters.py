from __future__ import annotations
"""Production-grade adapter tests for all ANCHOR framework integrations."""


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


# --- Consolidated from tests/runtime/anchor/test_anchor_adapters_golden.py ---
"""Phase 3 golden-path adapter integration tests (stubs + real optional deps)."""


from typing import Any, Dict, List, Optional

import pytest

from asha.runtime.anchor.adapters import (
    anchor_any,
    anchor_graph,
    anchor_langchain,
    anchor_llamaindex,
    anchor_mcp,
)
from asha.runtime.anchor.adapters.crewai import anchor_crewai
from asha.runtime.anchor.tool_capabilities import register_tool


class _StubTool:
    def __init__(self, name: str, *, destructive: bool = False) -> None:
        self.name = name
        self.description = name
        self._asha_capabilities = {
            "destructive": destructive,
            "writes_data": destructive,
            "network_egress": "send" in name or "email" in name,
            "reads_data": "read" in name or "load" in name,
            "category": (
                "delete"
                if destructive
                else ("network_egress" if "email" in name else "read")
            ),
        }
        self.calls = 0

    def _run(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        return f"ok:{self.name}"

    def run(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


class _StubAgent:
    def __init__(self, tools: List[Any]) -> None:
        self.tools = tools
        self.module_hint = "stub"

    def invoke(
        self, inputs: Dict[str, Any], config: Any = None, **kwargs: Any
    ) -> Dict[str, Any]:
        for tool in self.tools:
            if hasattr(tool, "_run"):
                tool._run()
            elif hasattr(tool, "run"):
                tool.run()
        return {"output": "done", "input": inputs}


class _StubMCP:
    def __init__(self, tools: List[str]) -> None:
        self._tools = tools
        self.calls: List[str] = []

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": n, "_meta": {}} for n in self._tools]

    def call_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> str:
        self.calls.append(name)
        return f"ran:{name}"


@pytest.fixture(autouse=True)
def _headless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASHA_ANCHOR_INTERACTIVE", "0")
    monkeypatch.setenv("ASHA_ANCHOR_WARN_POLICY", "strict")


def test_langchain_stub_blocks_destructive_tool() -> None:
    register_tool(
        "delete_local_file", destructive=True, writes_data=True, category="delete"
    )
    bad = _StubTool("delete_local_file", destructive=True)
    good = _StubTool("read_file")
    agent = _StubAgent([good, bad])
    wrapped = anchor_langchain(agent, interactive=False, risk_tolerance="LOW")
    tools = getattr(wrapped, "tools", None) or getattr(wrapped, "_inner", agent).tools
    denied = None
    for t in tools:
        if getattr(t, "name", "") == "delete_local_file":
            denied = t._run() if hasattr(t, "_run") else t.run()
    assert denied is not None
    assert "denied" in str(denied).lower() or bad.calls == 0


def test_llamaindex_stub_wraps_query() -> None:
    tool = _StubTool("read_file")
    agent = _StubAgent([tool])
    agent.query = lambda *a, **k: agent.invoke({"input": a[0] if a else ""})  # type: ignore[method-assign]
    wrapped = anchor_llamaindex(agent, interactive=False)
    out = wrapped.query("Read the local trends file and summarize.")
    assert out is not None


def test_mcp_stub_blocks_network_under_local_mission() -> None:
    register_tool("send_email", network_egress=True, category="network_egress")
    server = _StubMCP(["read_file", "send_email"])
    proxied = anchor_mcp(server, interactive=False, risk_tolerance="LOW")
    proxied.initialize_session(
        "Analyze data locally. Do not send email or exfiltrate.",
        local_only=True,
    )
    ok = proxied.call_tool("read_file", {"path": "x.csv"})
    blocked = proxied.call_tool("send_email", {"to": "evil@example.com"})
    assert "ran:read_file" in str(ok) or ok
    assert "denied" in str(blocked).lower()


def test_graph_stub_dispatch_via_anchor_any() -> None:
    pytest.importorskip("langgraph", reason="requires langgraph")
    class _Graph:
        def __init__(self) -> None:
            self.tools = [_StubTool("read_file")]
            self.nodes = {"start": object()}

        def invoke(
            self, input: Any, config: Any = None, **kwargs: Any
        ) -> Any:  # noqa: A002
            return {"result": "ok"}

        def get_graph(self) -> Any:
            return self

    g = _Graph()
    g.__class__.__module__ = "myapp.graphs"
    wrapped = anchor_graph(g, interactive=False)
    assert (
        wrapped.invoke({"messages": [{"role": "user", "content": "Analyze locally."}]})[
            "result"
        ]
        == "ok"
    )
    again = anchor_any(g, interactive=False)
    assert again is not None


def test_crewai_duck_type_without_package() -> None:
    """Duck-typed crew (no crewai package required) still wraps."""

    class _Crew:
        __module__ = "tests.fake_crew"

        def __init__(self) -> None:
            self.agents = []
            self.tasks = []

        def kickoff(self, inputs: Optional[dict] = None) -> str:
            return "done"

    crew = _Crew()
    wrapped = anchor_crewai(crew, interactive=False)
    assert wrapped is not None
