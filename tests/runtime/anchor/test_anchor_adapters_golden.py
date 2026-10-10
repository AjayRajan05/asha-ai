"""Phase 3 golden-path adapter integration tests (stubs + real optional deps)."""

from __future__ import annotations

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
