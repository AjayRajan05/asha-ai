"""LlamaIndex adapter for ASHA ANCHOR: production parity with CrewAI."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from ..runtime import AnchorRuntime
from ..tool_bridge import wrap_tool
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


def is_llamaindex_agent(target: Any) -> bool:
    module = type(target).__module__ or ""
    if "llama_index" in module:
        return True
    workflow_methods = (
        "query",
        "chat",
        "run",
        "achat",
        "aquery",
        "stream_events",
        "run_step",
    )
    if any(callable(getattr(target, method, None)) for method in workflow_methods):
        return True
    return bool(discover_tools(target))


class _AnchoredLlamaIndexProxy:
    """Wrap LlamaIndex agents and workflows with full ANCHOR governance."""

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._wrap_tools()

    def _collect_tools(self) -> List[Any]:
        return discover_tools(self._inner)

    def _wrap_tools(self) -> None:
        tools = self._collect_tools()
        if not tools:
            return
        wrapped = wrap_tool_list(tools, self._runtime, wrap_tool)
        if hasattr(self._inner, "tools"):
            safe_setattr(self._inner, "tools", wrapped)
        for attr in ("agent_worker", "agent", "workflow"):
            child = getattr(self._inner, attr, None)
            if child is not None and hasattr(child, "tools"):
                safe_setattr(child, "tools", wrapped)

    def _bind_mission(self, prompt: str, *, phase: bool = False) -> None:
        tools = self._collect_tools()
        if phase and self._runtime.state.baseline_mission is not None:
            refresh_mission_phase(self._runtime, prompt, tools)
        else:
            initialize_mission(self._runtime, prompt, tools)

    def _execute_and_finalize(
        self, prompt: str, runner: Any, *args: Any, **kwargs: Any
    ) -> Any:
        self._bind_mission(
            prompt, phase=self._runtime.state.baseline_mission is not None
        )
        try:
            with bind_runtime(self._runtime):
                result = runner(*args, **kwargs)
            output = getattr(result, "response", result)
            return finalize_step_output(
                self._runtime,
                output,
                required_tools=[tool_name(t) for t in self._collect_tools()],
                raise_on_block=False,
            )
        finally:
            finalize_session(self._runtime)

    def query(self, *args: Any, **kwargs: Any) -> Any:
        prompt = str(kwargs.get("str_or_query_bundle") or (args[0] if args else ""))
        return self._execute_and_finalize(prompt, self._inner.query, *args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        prompt = str(kwargs.get("message") or (args[0] if args else ""))
        return self._execute_and_finalize(prompt, self._inner.chat, *args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prompt = extract_prompt_from_input(
            kwargs.get("input") or (args[0] if args else "")
        )
        return self._execute_and_finalize(prompt, self._inner.run, *args, **kwargs)

    def run_step(self, *args: Any, **kwargs: Any) -> Any:
        prompt = extract_prompt_from_input(
            kwargs.get("input") or (args[0] if args else "")
        )
        return self._execute_and_finalize(prompt, self._inner.run_step, *args, **kwargs)

    def stream_events(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        prompt = extract_prompt_from_input(
            kwargs.get("input") or (args[0] if args else "")
        )
        self._bind_mission(prompt)
        with bind_runtime(self._runtime):
            events = list(self._inner.stream_events(*args, **kwargs))
        if events:
            finalize_step_output(self._runtime, events[-1], raise_on_block=False)
        finalize_session(self._runtime)
        return iter(events)

    async def aquery(self, *args: Any, **kwargs: Any) -> Any:
        prompt = str(kwargs.get("str_or_query_bundle") or (args[0] if args else ""))
        self._bind_mission(
            prompt, phase=self._runtime.state.baseline_mission is not None
        )
        try:
            with bind_runtime(self._runtime):
                result = await self._inner.aquery(*args, **kwargs)
            output = getattr(result, "response", result)
            return finalize_step_output(self._runtime, output, raise_on_block=False)
        finally:
            finalize_session(self._runtime)

    async def achat(self, *args: Any, **kwargs: Any) -> Any:
        prompt = str(kwargs.get("message") or (args[0] if args else ""))
        self._bind_mission(
            prompt, phase=self._runtime.state.baseline_mission is not None
        )
        try:
            with bind_runtime(self._runtime):
                result = await self._inner.achat(*args, **kwargs)
            output = getattr(result, "response", result)
            return finalize_step_output(self._runtime, output, raise_on_block=False)
        finally:
            finalize_session(self._runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def anchor_llamaindex(
    target: Any,
    *,
    runtime: Optional[AnchorRuntime] = None,
    isolation: str = "auto",
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Any:
    """Wrap a LlamaIndex agent or workflow with ANCHOR governance.

    Install: ``pip install asha[llamaindex]``
    """
    module = type(target).__module__ or ""
    if "llama_index" in module:
        from .deps import require_module

        require_module("llama_index", adapter="anchor_llamaindex")
    if not is_llamaindex_agent(target):
        raise TypeError(
            f"anchor_llamaindex() expects a LlamaIndex agent/workflow, got {type(target).__name__}."
        )
    runtime = runtime or AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
    if isinstance(target, _AnchoredLlamaIndexProxy):
        return target
    return _AnchoredLlamaIndexProxy(target, runtime)
