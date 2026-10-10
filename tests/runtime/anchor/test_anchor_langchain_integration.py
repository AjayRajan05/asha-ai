"""LangChain golden-path integration (skips if asha[langchain] not installed)."""

from __future__ import annotations

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
