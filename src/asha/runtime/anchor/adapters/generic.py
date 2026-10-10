"""Generic duck-typed ANCHOR adapter: production parity with CrewAI."""

from __future__ import annotations

from typing import Any, List, Optional

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


class AnchoredAgentProxy:
    """Wrap any agent-like object exposing run/invoke and optional tools."""

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._wrap_tools()

    def _collect_tools(self) -> List[Any]:
        return list(getattr(self._inner, "tools", None) or [])

    def _wrap_tools(self) -> None:
        tools = self._collect_tools()
        if not tools:
            return
        wrapped = wrap_tool_list(tools, self._runtime, wrap_tool)
        safe_setattr(self._inner, "tools", wrapped)

    def _bind_mission(self, prompt: str, *, phase: bool = False) -> None:
        tools = self._collect_tools()
        if phase and self._runtime.state.baseline_mission is not None:
            refresh_mission_phase(self._runtime, prompt, tools)
        else:
            initialize_mission(self._runtime, prompt, tools)

    def _execute(self, prompt: str, runner: Any, *args: Any, **kwargs: Any) -> Any:
        phase = self._runtime.state.baseline_mission is not None
        self._bind_mission(prompt, phase=phase)
        try:
            with bind_runtime(self._runtime):
                result = runner(*args, **kwargs)
            return finalize_step_output(
                self._runtime,
                result,
                required_tools=[tool_name(t) for t in self._collect_tools()],
                raise_on_block=False,
            )
        finally:
            finalize_session(self._runtime)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prompt = extract_prompt_from_input(
            kwargs.get("prompt") or (args[0] if args else "")
        )
        if not hasattr(self._inner, "run"):
            raise AttributeError("Anchored agent has no run() method.")
        return self._execute(prompt, self._inner.run, *args, **kwargs)

    def invoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:  # noqa: A002
        prompt = extract_prompt_from_input(input)
        if not hasattr(self._inner, "invoke"):
            raise AttributeError("Anchored agent has no invoke() method.")
        if config is not None:
            return self._execute(prompt, self._inner.invoke, input, config, **kwargs)
        return self._execute(prompt, self._inner.invoke, input, **kwargs)

    async def ainvoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> Any:  # noqa: A002
        prompt = extract_prompt_from_input(input)
        phase = self._runtime.state.baseline_mission is not None
        self._bind_mission(prompt, phase=phase)
        with bind_runtime(self._runtime):
            if config is not None:
                result = await self._inner.ainvoke(input, config, **kwargs)
            else:
                result = await self._inner.ainvoke(input, **kwargs)
        return finalize_step_output(self._runtime, result, raise_on_block=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def anchor_generic(
    target: Any,
    *,
    runtime: Optional[AnchorRuntime] = None,
    isolation: str = "auto",
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Any:
    """
    Wrap any duck-typed agent with ANCHOR mission governance.

    Example::

        class MyAgent:
            tools = [load_data, write_report]
            def run(self, prompt): ...

        agent = anchor_generic(MyAgent(), risk_tolerance="LOW")
        agent.run("Analyze trends locally.")
    """
    runtime = runtime or AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
    if isinstance(target, AnchoredAgentProxy):
        return target
    if not any(callable(getattr(target, m, None)) for m in ("run", "invoke")):
        raise TypeError(
            f"anchor_generic() expects an agent with run() or invoke(), got {type(target).__name__}."
        )
    return AnchoredAgentProxy(target, runtime)
