from __future__ import annotations
"""Consolidated ANCHOR framework integrations tests."""
import pytest

# --- From tests/runtime/anchor/test_anchor_crewai_integration.py ---
"""CrewAI golden-path integration (skips if asha[crewai] not installed)."""


from typing import Any, Optional

import pytest

pytest.importorskip("crewai")

from asha.runtime.anchor.adapters.crewai import anchor_crewai


class _MiniCrew:
    __module__ = "crewai.crew"

    def __init__(self) -> None:
        self.agents: list[Any] = []
        self.tasks: list[Any] = []
        self.tools: list[Any] = []

    def kickoff(self, inputs: Optional[dict] = None) -> str:
        return f"done:{inputs}"


def test_crewai_package_importable() -> None:
    import crewai

    assert crewai is not None


def test_anchor_crewai_wraps_when_package_present() -> None:
    wrapped = anchor_crewai(_MiniCrew(), interactive=False, risk_tolerance="LOW")
    assert wrapped is not None
    assert hasattr(wrapped, "kickoff") or hasattr(wrapped, "_inner")

# --- From tests/runtime/anchor/test_anchor_langchain_integration.py ---
"""LangChain golden-path integration (skips if asha[langchain] not installed)."""


from typing import Any, Dict

import pytest

pytest.importorskip("langchain_core")

from asha.runtime.anchor.adapters.langchain import anchor_langchain
from asha.runtime.anchor.tool_capabilities import register_tool


class _StubTool:
    name = "read_file"
    description = "Read a local file"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return "ok"


class _StubAgent:
    __module__ = "langchain.agents.agent"

    def __init__(self) -> None:
        self.tools = [_StubTool()]

    def invoke(
        self, inputs: Dict[str, Any], config: Any = None, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"result": "ok", "input": inputs}


def test_langchain_core_importable() -> None:
    import langchain_core

    assert langchain_core is not None


def test_anchor_langchain_wraps_when_package_present() -> None:
    register_tool("read_file", reads_data=True, category="read")
    wrapped = anchor_langchain(_StubAgent(), interactive=False)
    out = wrapped.invoke({"input": "Read trends.csv locally and summarize."})
    assert out is not None

# --- From tests/runtime/anchor/test_anchor_langgraph_integration.py ---
"""LangGraph golden-path integration (skips if asha[langgraph] not installed)."""


from typing import Any

import pytest

pytest.importorskip("langgraph")

from asha.runtime.anchor.adapters.graph import anchor_graph


class _StubGraph:
    __module__ = "langgraph.graph.state"

    def __init__(self) -> None:
        self.nodes = {"start": object()}
        self.tools: list[Any] = []

    def invoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:  # noqa: A002
        return {"result": "ok"}

    def get_graph(self) -> Any:
        return self


def test_langgraph_importable() -> None:
    import langgraph

    assert langgraph is not None


def test_anchor_graph_wraps_when_package_present() -> None:
    wrapped = anchor_graph(_StubGraph(), interactive=False)
    result = wrapped.invoke(
        {"messages": [{"role": "user", "content": "Analyze locally."}]}
    )
    assert result["result"] == "ok"

# --- From tests/runtime/anchor/test_anchor_llamaindex_integration.py ---
"""LlamaIndex integration (skips if asha[llamaindex] not installed)."""


from typing import Any

import pytest

pytest.importorskip("llama_index")

from asha.runtime.anchor.adapters.llamaindex import anchor_llamaindex


class _StubQueryEngine:
    __module__ = "llama_index.core.query_engine"

    def __init__(self) -> None:
        self.tools: list[Any] = []

    def query(self, *args: Any, **kwargs: Any) -> str:
        return f"answer:{args[0] if args else ''}"


def test_llama_index_importable() -> None:
    import llama_index.core

    assert llama_index.core is not None


def test_anchor_llamaindex_wraps_when_package_present() -> None:
    wrapped = anchor_llamaindex(_StubQueryEngine(), interactive=False)
    out = wrapped.query("Summarize the local index.")
    assert out is not None

# --- From tests/runtime/anchor/test_anchor_mcp_integration.py ---
"""MCP integration (skips if asha[mcp] not installed)."""


from typing import Any, Dict, List

import pytest

pytest.importorskip("mcp")

from asha.runtime.anchor.adapters.mcp import anchor_mcp
from asha.runtime.anchor.tool_capabilities import register_tool


class _StubMCP:
    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "read_file", "_meta": {}},
            {
                "name": "send_email",
                "_meta": {
                    "asha_capabilities": {
                        "network_egress": True,
                        "category": "network_egress",
                    }
                },
            },
        ]

    def call_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> str:
        return f"ran:{name}"


def test_mcp_package_importable() -> None:
    import mcp

    assert mcp is not None


def test_anchor_mcp_blocks_egress_when_package_present() -> None:
    register_tool("send_email", network_egress=True, category="network_egress")
    proxied = anchor_mcp(_StubMCP(), interactive=False, risk_tolerance="LOW")
    proxied.initialize_session(
        "Analyze data locally. Do not send email.",
        local_only=True,
    )
    blocked = proxied.call_tool("send_email", {"to": "x@y.com"})
    assert "denied" in str(blocked).lower()
