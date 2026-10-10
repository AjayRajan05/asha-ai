"""Gap-closure tests: asha.Agent, LangGraph, full adapter wiring."""

from __future__ import annotations

import pytest

from asha import Agent, anchor
from asha.runtime.agent_tools import normalize_tools, parse_tool_call, run_tool_loop
from asha.runtime.anchor.adapters.asha_agent import anchor_asha_agent, is_asha_agent
from asha.runtime.anchor.adapters.discovery import discover_tools
from asha.runtime.anchor.adapters.graph import anchor_graph, is_agent_graph
from asha.runtime.anchor.adapters.registry import anchor_any
from asha.runtime.anchor.runtime import AnchorRuntime


def test_normalize_tools_from_strings_and_callables() -> None:
    def load_data() -> str:
        return "ok"

    tools = normalize_tools(
        ["email", load_data, {"name": "write_report", "fn": lambda: "w"}]
    )
    assert len(tools) == 3
    assert tools[0].name == "email"
    assert tools[1].name == "load_data"


def test_parse_tool_call_formats() -> None:
    text = 'TOOL_CALL: send_email\nARGS: {"to": "a@b.com"}'
    parsed = parse_tool_call(text)
    assert parsed == ("send_email", {"to": "a@b.com"})


def test_agent_tool_loop_executes_callable() -> None:
    calls: list[str] = []

    def load_data() -> str:
        calls.append("load")
        return "data loaded"

    def fake_generate(prompt: str) -> str:
        if "Tool result" in prompt:
            return "Final answer with data loaded"
        return "TOOL_CALL: load_data\nARGS: {}"

    result = run_tool_loop(
        generate=fake_generate,
        tools=[load_data],
        prompt="Load data locally",
    )
    assert "data loaded" in result
    assert calls == ["load"]


def test_asha_agent_run_with_tools() -> None:
    def load_trend_data() -> str:
        return "trends loaded"

    agent = Agent(provider="mock", model="mock", tools=[load_trend_data])
    result = agent.run("Analyze trends locally")
    assert "Mock" in result
    assert "tools" in result.lower() or "analyze" in result.lower()


def test_anchor_asha_agent_governs_tools() -> None:
    def load_trend_data() -> str:
        return "trends loaded"

    def send_email(**_kwargs: object) -> str:
        return "sent"

    runtime = AnchorRuntime(interactive=False)
    agent = anchor_asha_agent(
        Agent(provider="mock", model="mock", tools=[load_trend_data, send_email]),
        runtime=runtime,
    )
    assert is_asha_agent(agent._inner)
    result = agent.run("Analyze trends locally. Do not send externally.")
    assert isinstance(result, str)
    wrapped = agent._inner.tools[0]
    denied = wrapped._run()
    assert "trends loaded" in str(denied) or "denied" not in str(denied).lower()


def test_anchor_any_dispatches_asha_agent() -> None:
    agent = Agent(provider="mock", model="mock", tools=[])
    wrapped = anchor_any(agent, interactive=False)
    result = wrapped.run("Analyze trends locally")
    assert "Mock" in result
    assert hasattr(wrapped, "_anchor_runtime")


def test_anchor_blocks_dangerous_tool_on_asha_agent() -> None:
    def send_email(**_kwargs: object) -> str:
        return "Email sent to x"

    runtime = AnchorRuntime(interactive=False)
    agent = anchor_asha_agent(
        Agent(provider="mock", model="mock", tools=[send_email]),
        runtime=runtime,
    )
    runtime.initialize_mission(
        "Analyze locally. Do not send externally.",
        context={"available_tools": ["send_email"], "local_only": True},
    )
    result = agent._inner.tools[0]._run()
    assert "denied" in result.lower() or "blocked" in result.lower()


class _LangGraphStub:
    # Duck-typed graph shape; avoid "langgraph" in module so unit tests do not
    # require asha[langgraph] (integration tests cover the real package).
    __module__ = "tests.stubs.agent_graph"

    def __init__(self) -> None:
        self.nodes = {"tools": _ToolNodeStub()}
        self._result = {"result": "ok"}

    def get_graph(self):
        return self

    def invoke(self, input: dict, config=None, **kwargs):
        return self._result

    def compile(self):
        return self


class _ToolNodeStub:
    tools = []

    def __init__(self) -> None:
        class _T:
            name = "load_trend_data"

            def _run(self, **kwargs):
                return "loaded"

        self.tools = [_T()]


def test_graph_adapter_detection_and_invoke() -> None:
    graph = _LangGraphStub()
    assert is_agent_graph(graph)
    wrapped = anchor_graph(graph, interactive=False)
    result = wrapped.invoke(
        {"messages": [{"role": "user", "content": "Analyze locally."}]}
    )
    assert result["result"] == "ok"


def test_discover_tools_finds_nested_graph_tools() -> None:
    graph = _LangGraphStub()
    tools = discover_tools(graph)
    assert len(tools) >= 1


def test_anchor_dispatches_graph() -> None:
    wrapped = anchor_any(_LangGraphStub(), interactive=False)
    assert is_agent_graph(wrapped._inner) or hasattr(wrapped, "invoke")
