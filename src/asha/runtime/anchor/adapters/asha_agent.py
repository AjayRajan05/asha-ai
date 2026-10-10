"""ASHA Agent adapter for ANCHOR: full tool-loop governance."""

from __future__ import annotations

from typing import Any, Optional

from ...agent import Agent
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
)
from ...agent_tools import normalize_tools


def is_asha_agent(target: Any) -> bool:
    if isinstance(target, Agent):
        return True
    return (
        type(target).__name__ == "Agent"
        and hasattr(target, "processor")
        and hasattr(target, "adapter")
        and callable(getattr(target, "run", None))
    )


class _AnchoredASHAAgentProxy:
    """Wrap asha.Agent with mission governance over the tool loop."""

    def __init__(self, inner: Agent, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._wrap_tools()

    def _wrap_tools(self) -> None:
        tools = list(getattr(self._inner, "tools", None) or [])
        if not tools:
            return
        normalized = normalize_tools(tools)
        wrapped = [wrap_tool(t, self._runtime) for t in normalized]
        safe_setattr(self._inner, "tools", wrapped)

    def _tool_names(self) -> list[str]:
        return [tool_name(t) for t in getattr(self._inner, "tools", None) or []]

    def _bind_mission(self, prompt: str, *, phase: bool = False) -> None:
        names = self._tool_names()
        if phase and self._runtime.state.baseline_mission is not None:
            refresh_mission_phase(self._runtime, prompt, names)
        else:
            initialize_mission(self._runtime, prompt, names)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        prompt = extract_prompt_from_input(
            kwargs.get("prompt") or (args[0] if args else "")
        )
        phase = self._runtime.state.baseline_mission is not None
        self._bind_mission(prompt, phase=phase)
        try:
            with bind_runtime(self._runtime):
                result = self._inner.run(*args, **kwargs)
            return finalize_step_output(
                self._runtime,
                result,
                required_tools=self._tool_names(),
                raise_on_block=False,
            )
        finally:
            finalize_session(self._runtime)

    def invoke(self, input: Any, **kwargs: Any) -> Any:  # noqa: A002
        if isinstance(input, dict) and "prompt" in input:
            return self.run(input["prompt"], **{**kwargs, **input})
        return self.run(str(input), **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def anchor_asha_agent(
    target: Any,
    *,
    runtime: Optional[AnchorRuntime] = None,
    isolation: str = "auto",
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Any:
    """
    Wrap an asha.Agent with ANCHOR mission governance over its tool loop.

    Example::

        from asha import Agent, anchor

        agent = Agent(provider="mock", model="mock", tools=[load_data, send_email])
        agent = anchor(agent, risk_tolerance="LOW")
        agent.run("Analyze trends locally.")
    """
    if not is_asha_agent(target):
        raise TypeError(
            f"anchor_asha_agent() expects asha.Agent, got {type(target).__name__}."
        )
    runtime = runtime or AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
    if isinstance(target, _AnchoredASHAAgentProxy):
        return target
    return _AnchoredASHAAgentProxy(target, runtime)
