"""LangGraph golden-path integration (skips if asha[langgraph] not installed)."""

from __future__ import annotations

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
