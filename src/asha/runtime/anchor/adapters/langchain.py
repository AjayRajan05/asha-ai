"""LangChain adapter for ASHA ANCHOR: production parity with CrewAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence

from ..llm_guard import evaluate_llm_tool_calls
from ..runtime import AnchorRuntime
from ..tool_bridge import wrap_tool
from .base import (
    bind_runtime,
    extract_prompt_from_input,
    finalize_session,
    finalize_step_output,
    initialize_mission,
    is_wrapped,
    refresh_mission_phase,
    safe_setattr,
    tool_name,
    wrap_tool_list,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler as _LCCallbackHandler
    from langchain_core.tools import BaseTool as _LangChainBaseTool
else:
    try:
        from langchain_core.callbacks import BaseCallbackHandler as _LCCallbackHandler
        from langchain_core.tools import BaseTool as _LangChainBaseTool
    except ImportError:  # pragma: no cover - optional dependency
        _LCCallbackHandler = object
        _LangChainBaseTool = object


from .discovery import discover_tools


def is_langchain_agent(target: Any) -> bool:
    module = type(target).__module__ or ""
    if "langgraph" in module:
        return False
    if "langchain" in module:
        return True
    agent = getattr(target, "agent", None)
    if agent is not None and "langchain" in (type(agent).__module__ or ""):
        return True
    return callable(getattr(target, "invoke", None)) and (
        hasattr(target, "tools")
        or hasattr(target, "agent")
        or "Runnable" in type(target).__name__
    )


def _wrap_langchain_tool(tool: Any, runtime: AnchorRuntime) -> Any:
    if is_wrapped(tool):
        return tool
    if _LangChainBaseTool is not object and isinstance(tool, _LangChainBaseTool):
        return _create_langchain_tool_subclass(tool, runtime)
    return wrap_tool(tool, runtime)


def _create_langchain_tool_subclass(inner: Any, runtime: AnchorRuntime) -> Any:
    """Create a LangChain BaseTool subclass that delegates to ANCHOR."""
    delegate = wrap_tool(inner, runtime)
    tool_name_str = delegate.name
    tool_desc = delegate.description
    tool_schema = delegate.args_schema

    class _AnchoredLC(_LangChainBaseTool):
        name: str = tool_name_str
        description: str = tool_desc
        args_schema: Any = tool_schema

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            return delegate._run(*args, **kwargs)

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:
            return self._run(*args, **kwargs)

    return _AnchoredLC()


class _AnchorLangChainCallback(_LCCallbackHandler):
    """Callback handler that validates LLM-proposed and runtime tool calls through ANCHOR."""

    def __init__(self, runtime: AnchorRuntime) -> None:
        self._runtime = runtime
        if _LCCallbackHandler is not object:
            super().__init__()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = str(serialized.get("name", "") or "")
        if not tool_name:
            return
        allowed = self._runtime.evaluate_action_request(
            "tool_call",
            {"tool_name": tool_name, "arguments": input_str},
            raise_on_block=False,
            record=False,
        )
        if not allowed:
            raise RuntimeError(
                f"ANCHOR blocked tool '{tool_name}' before LangChain execution."
            )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        generations = getattr(response, "generations", None)
        if not generations:
            return
        for gen_list in generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                if message is not None:
                    evaluate_llm_tool_calls(
                        message,
                        self._runtime,
                        raise_on_block=False,
                        record=False,
                    )


class _AnchoredLangChainProxy:
    """Wrap LangChain AgentExecutor / Runnable agents with full ANCHOR governance."""

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._inputs: Dict[str, Any] = {}
        self._wrap_tools()

    def _collect_tools(self) -> List[Any]:
        return discover_tools(self._inner)

    def _wrap_tools(self) -> None:
        tools = self._collect_tools()
        if not tools:
            return
        wrapped = wrap_tool_list(tools, self._runtime, _wrap_langchain_tool)
        if hasattr(self._inner, "tools"):
            safe_setattr(self._inner, "tools", wrapped)
        agent = getattr(self._inner, "agent", None)
        if agent is not None and hasattr(agent, "tools"):
            safe_setattr(agent, "tools", wrapped)

    def _mission_prompt(self, input_value: Any) -> str:
        base = extract_prompt_from_input(input_value)
        if self._inputs:
            return f"{' '.join(f'{k}: {v}' for k, v in self._inputs.items())} {base}"
        return base

    def _begin_step(self, input_value: Any, *, phase: bool = False) -> None:
        prompt = self._mission_prompt(input_value)
        tools = self._collect_tools()
        if phase and self._runtime.state.baseline_mission is not None:
            refresh_mission_phase(self._runtime, prompt, tools)
        else:
            initialize_mission(self._runtime, prompt, tools)

    def _end_step(self, result: Any) -> Any:
        required = [tool_name(t) for t in self._collect_tools()]
        return finalize_step_output(self._runtime, result, required_tools=required)

    def _inject_callbacks(self, config: Any) -> Any:
        handler = _AnchorLangChainCallback(self._runtime)
        if config is None:
            return {"callbacks": [handler]}
        if isinstance(config, dict):
            callbacks = list(config.get("callbacks", []) or [])
            callbacks.append(handler)
            return {**config, "callbacks": callbacks}
        return config

    def invoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:  # noqa: A002
        self._begin_step(input)
        config = self._inject_callbacks(config)
        try:
            with bind_runtime(self._runtime):
                result = (
                    self._inner.invoke(input, config, **kwargs)
                    if config
                    else self._inner.invoke(input, **kwargs)
                )
            return self._end_step(result)
        finally:
            finalize_session(self._runtime)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prompt = kwargs.get("input") or (args[0] if args else "")
        self._begin_step(prompt)
        try:
            with bind_runtime(self._runtime):
                result = self._inner.run(*args, **kwargs)
            return self._end_step(result)
        finally:
            finalize_session(self._runtime)

    async def ainvoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:  # noqa: A002
        self._begin_step(input)
        config = self._inject_callbacks(config)
        with bind_runtime(self._runtime):
            result = (
                await self._inner.ainvoke(input, config, **kwargs)
                if config
                else await self._inner.ainvoke(input, **kwargs)
            )
        return self._end_step(result)

    def stream(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Iterator[Any]:  # noqa: A002
        self._begin_step(input)
        config = self._inject_callbacks(config)
        with bind_runtime(self._runtime):
            chunks = list(
                self._inner.stream(input, config, **kwargs)
                if config
                else self._inner.stream(input, **kwargs)
            )
        if chunks:
            self._end_step(chunks[-1])
        return iter(chunks)

    def batch(
        self, inputs: Sequence[Any], config: Any = None, **kwargs: Any
    ) -> List[Any]:
        results = []
        for item in inputs:
            results.append(self.invoke(item, config=config, **kwargs))
        return results

    def with_inputs(self, **inputs: Any) -> _AnchoredLangChainProxy:
        self._inputs = dict(inputs)
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def anchor_langchain(
    target: Any,
    *,
    runtime: Optional[AnchorRuntime] = None,
    isolation: str = "auto",
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Any:
    """
    Wrap a LangChain AgentExecutor or Runnable with ANCHOR mission governance.

    Requires ``pip install asha[langchain]`` when using real LangChain objects.

    Example::

        from langchain.agents import create_react_agent, AgentExecutor
        from asha.runtime.anchor.adapters import anchor_langchain

        executor = AgentExecutor(agent=agent, tools=tools)
        executor = anchor_langchain(executor, risk_tolerance="LOW")
        result = executor.invoke({"input": "Analyze trends locally."})
    """
    module = type(target).__module__ or ""
    if "langchain" in module:
        from .deps import require_module

        require_module("langchain_core", adapter="anchor_langchain")
    runtime = runtime or AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
    if isinstance(target, _AnchoredLangChainProxy):
        return target
    if not is_langchain_agent(target) and not hasattr(target, "invoke"):
        raise TypeError(
            f"anchor_langchain() expects a LangChain Runnable or agent, got {type(target).__name__}."
        )
    return _AnchoredLangChainProxy(target, runtime)
