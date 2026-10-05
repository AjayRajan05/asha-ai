"""LangGraph / agent-graph adapter for ASHA ANCHOR."""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from ..runtime import AnchorRuntime
from .base import (
    bind_runtime,
    extract_prompt_from_input,
    finalize_session,
    finalize_step_output,
    initialize_mission,
    refresh_mission_phase,
    safe_setattr,
    tool_name,
    wrap_tool_list,
)
from .discovery import discover_tools
from .langchain import _AnchorLangChainCallback, _wrap_langchain_tool


def is_agent_graph(target: Any) -> bool:
    module = type(target).__module__ or ""
    if "langgraph" in module:
        return True
    if callable(getattr(target, "invoke", None)) and callable(
        getattr(target, "get_graph", None)
    ):
        return True
    if callable(getattr(target, "invoke", None)) and hasattr(target, "nodes"):
        return True
    return False


def _wrap_graph_nodes(target: Any, runtime: AnchorRuntime) -> None:
    """Wrap tools on LangGraph ToolNode instances when accessible."""
    nodes = getattr(target, "nodes", None)
    if not isinstance(nodes, dict):
        builder = getattr(target, "builder", None)
        nodes = getattr(builder, "nodes", None) if builder is not None else None
    if not isinstance(nodes, dict):
        return

    for node in nodes.values():
        node_tools = getattr(node, "tools", None)
        if not node_tools:
            continue
        wrapped = wrap_tool_list(list(node_tools), runtime, _wrap_langchain_tool)
        safe_setattr(node, "tools", wrapped)


class _AnchoredGraphProxy:
    """Wrap LangGraph compiled graphs and StateGraph workflows."""

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._wrap_all_tools()

    def _collect_tools(self) -> List[Any]:
        return discover_tools(self._inner)

    def _wrap_all_tools(self) -> None:
        tools = self._collect_tools()
        if tools:
            wrapped = wrap_tool_list(tools, self._runtime, _wrap_langchain_tool)
            if hasattr(self._inner, "tools"):
                safe_setattr(self._inner, "tools", wrapped)
        _wrap_graph_nodes(self._inner, self._runtime)

    def _begin_step(self, input_value: Any, *, phase: bool = False) -> None:
        prompt = extract_prompt_from_input(input_value)
        tools = self._collect_tools()
        if phase and self._runtime.state.baseline_mission is not None:
            refresh_mission_phase(self._runtime, prompt, tools)
        else:
            initialize_mission(self._runtime, prompt, tools)

    def _inject_callbacks(self, config: Any) -> Any:
        handler = _AnchorLangChainCallback(self._runtime)
        if config is None:
            return {"callbacks": [handler]}
        if isinstance(config, dict):
            callbacks = list(config.get("callbacks", []) or [])
            callbacks.append(handler)
            return {**config, "callbacks": callbacks}
        return config

    def _finish(self, result: Any) -> Any:
        required = [tool_name(t) for t in self._collect_tools()]
        return finalize_step_output(self._runtime, result, required_tools=required)

    def invoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:  # noqa: A002
        self._begin_step(input)
        config = self._inject_callbacks(config)
        try:
            with bind_runtime(self._runtime):
                result = (
                    self._inner.invoke(input, config, **kwargs)
                    if config is not None
                    else self._inner.invoke(input, **kwargs)
                )
            return self._finish(result)
        finally:
            finalize_session(self._runtime)

    async def ainvoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:  # noqa: A002
        self._begin_step(input)
        config = self._inject_callbacks(config)
        try:
            with bind_runtime(self._runtime):
                result = (
                    await self._inner.ainvoke(input, config, **kwargs)
                    if config is not None
                    else await self._inner.ainvoke(input, **kwargs)
                )
            return self._finish(result)
        finally:
            finalize_session(self._runtime)

    def stream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Iterator[Any]:  # noqa: A002
        self._begin_step(input)
        config = self._inject_callbacks(config)
        try:
            with bind_runtime(self._runtime):
                chunks = list(
                    self._inner.stream(input, config, **kwargs)
                    if config is not None
                    else self._inner.stream(input, **kwargs)
                )
            if chunks:
                self._finish(chunks[-1])
            return iter(chunks)
        finally:
            finalize_session(self._runtime)

    async def astream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:  # noqa: A002
        self._begin_step(input)
        config = self._inject_callbacks(config)
        try:
            with bind_runtime(self._runtime):
                if not callable(getattr(self._inner, "astream", None)):
                    for chunk in self.stream(input, config=config, **kwargs):
                        yield chunk
                    return
                agen = (
                    self._inner.astream(input, config, **kwargs)
                    if config is not None
                    else self._inner.astream(input, **kwargs)
                )
                last = None
                async for chunk in agen:
                    last = chunk
                    yield chunk
                if last is not None:
                    self._finish(last)
        finally:
            finalize_session(self._runtime)

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        compiled = self._inner.compile(*args, **kwargs)
        return anchor_graph(compiled, runtime=self._runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def anchor_graph(
    target: Any,
    *,
    runtime: Optional[AnchorRuntime] = None,
    isolation: str = "auto",
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Any:
    """
    Wrap a LangGraph StateGraph or compiled graph with ANCHOR governance.

    Example::

        from langgraph.graph import StateGraph
        from asha.runtime.anchor.adapters import anchor_graph

        graph = StateGraph(...)
        graph = anchor_graph(graph, interactive=False)
        app = graph.compile()
        app.invoke({"messages": [...]})

    Install: ``pip install asha[langgraph]``
    """
    module = type(target).__module__ or ""
    if "langgraph" in module:
        from .deps import require_module

        require_module("langgraph", adapter="anchor_graph")
    if not is_agent_graph(target):
        raise TypeError(
            f"anchor_graph() expects a LangGraph graph or compiled workflow, got {type(target).__name__}."
        )
    runtime = runtime or AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
    if isinstance(target, _AnchoredGraphProxy):
        return target
    return _AnchoredGraphProxy(target, runtime)
