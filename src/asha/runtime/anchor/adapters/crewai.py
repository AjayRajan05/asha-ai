"""CrewAI adapter for ASHA ANCHOR runtime governance."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from pydantic import PrivateAttr

try:
    from crewai.tools import BaseTool as CrewAIBaseTool
except ImportError:  # pragma: no cover - optional dependency
    CrewAIBaseTool = object

from ..llm_guard import evaluate_llm_tool_calls
from ..runtime import AnchorRuntime, current_anchor_runtime
from .base import (
    bind_runtime,
    block_message,
    finalize_session,
    finalize_step_output,
    mission_context,
    resolve_templates,
    safe_setattr,
    tool_name as _tool_name,
    initialize_mission,
    refresh_mission_phase,
)


def _collect_tools_from_agents(agents: List[Any]) -> List[Any]:
    tools: List[Any] = []
    for agent in agents:
        inner = getattr(agent, "_agent", agent)
        tools.extend(getattr(inner, "tools", None) or [])
    return tools


def _raise_on_block(runtime: AnchorRuntime) -> bool:
    """Tool wrappers return denial text; never raise into the agent loop."""
    return False


def _is_crew(target: Any) -> bool:
    return (
        callable(getattr(target, "kickoff", None))
        and hasattr(target, "agents")
        and hasattr(target, "tasks")
    )


def _is_agent(target: Any) -> bool:
    return callable(getattr(target, "execute_task", None))


class AnchoredTool(CrewAIBaseTool):
    """CrewAI-compatible tool wrapper that enforces ANCHOR governance."""

    _inner: Any = PrivateAttr()
    _runtime: AnchorRuntime = PrivateAttr()

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        init_kwargs = {
            "name": getattr(inner, "name", "tool"),
            "description": getattr(inner, "description", ""),
            "env_vars": getattr(inner, "env_vars", []),
            "result_as_answer": getattr(inner, "result_as_answer", False),
        }
        if getattr(inner, "args_schema", None) is not None:
            init_kwargs["args_schema"] = inner.args_schema
        if getattr(inner, "max_usage_count", None) is not None:
            init_kwargs["max_usage_count"] = inner.max_usage_count
        if CrewAIBaseTool is object:
            for key, value in init_kwargs.items():
                setattr(self, key, value)
        else:
            super().__init__(**init_kwargs)
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        tool_name = _tool_name(self)
        allowed = self._runtime.evaluate_action_request(
            "tool_call",
            {
                "tool_name": tool_name,
                "arguments": str({"args": args, "kwargs": kwargs}),
            },
            raise_on_block=False,
        )
        if not allowed:
            return block_message(self._runtime, tool_name)

        def _runner(*a: Any, **k: Any) -> Any:
            result = self._inner._run(*a, **k)
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return result

        return self._runtime.sandbox.execute(
            _runner, tool_name=tool_name, args=args, kwargs=kwargs
        )


class _AnchoredLLMProxy:
    """Proxy around CrewAI LLM that binds ANCHOR context and guards tool calls."""

    def __init__(self, llm: Any, runtime: AnchorRuntime) -> None:
        self._llm = llm
        self._runtime = runtime
        self._anchor_runtime = runtime

    def call(self, *args: Any, **kwargs: Any) -> Any:
        token = current_anchor_runtime.set(self._runtime)
        original_handle = getattr(self._llm, "_handle_tool_call", None)

        def guarded_handle_tool_call(
            tool_calls: List[Any],
            available_functions: Optional[Dict[str, Any]] = None,
            from_task: Any = None,
            from_agent: Any = None,
        ) -> Any:
            allowed_calls: List[Any] = []
            for tool_call in tool_calls or []:
                function = getattr(tool_call, "function", None)
                if function is None:
                    continue
                tool_name = str(getattr(function, "name", "") or "")
                allowed = self._runtime.evaluate_action_request(
                    "tool_call",
                    {
                        "tool_name": tool_name,
                        "arguments": str(getattr(function, "arguments", "") or ""),
                    },
                    raise_on_block=_raise_on_block(self._runtime),
                    record=False,
                )
                if allowed:
                    allowed_calls.append(tool_call)
            if original_handle is None:
                return None
            if not allowed_calls:
                return []
            return original_handle(
                allowed_calls,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

        try:
            try:
                import litellm

                original_completion = litellm.completion

                def guarded_completion(*cargs: Any, **ckwargs: Any) -> Any:
                    response = original_completion(*cargs, **ckwargs)
                    evaluate_llm_tool_calls(
                        response,
                        self._runtime,
                        raise_on_block=_raise_on_block(self._runtime),
                        record=False,
                    )
                    return response

                litellm.completion = guarded_completion
            except ImportError:
                original_completion = None

            if original_handle is not None:
                self._llm._handle_tool_call = guarded_handle_tool_call
            return self._llm.call(*args, **kwargs)
        finally:
            try:
                import litellm

                if (
                    "original_completion" in locals()
                    and original_completion is not None
                ):
                    litellm.completion = original_completion
            except ImportError:
                pass
            if original_handle is not None:
                self._llm._handle_tool_call = original_handle
            current_anchor_runtime.reset(token)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


class _AnchoredTaskProxy:
    """Proxy that refreshes the mission contract before each task executes."""

    def __init__(
        self,
        task: Any,
        runtime: AnchorRuntime,
        inputs_getter: Callable[[], Dict[str, Any]],
    ) -> None:
        self._task = task
        self._runtime = runtime
        self._inputs_getter = inputs_getter

    def execute_sync(
        self,
        agent: Any,
        context: Any = None,
        tools: Optional[List[Any]] = None,
    ) -> Any:
        inputs = self._inputs_getter()
        description = resolve_templates(
            getattr(self._task, "description", "") or "", inputs
        )
        agent_tools = tools or getattr(agent, "tools", None) or []

        self._runtime.refresh_mission_phase(
            description,
            context=mission_context(self._runtime, agent_tools),
        )

        with bind_runtime(self._runtime):
            result = self._task.execute_sync(agent, context=context, tools=tools)

        task_tools = tools or getattr(agent, "tools", None) or []
        required = [_tool_name(t) for t in task_tools if _tool_name(t)]
        return finalize_step_output(
            self._runtime,
            result,
            required_tools=required,
            raise_on_block=True,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task, name)


class _AnchoredAgentProxy:
    """Proxy around a CrewAI agent with anchored tools and LLM."""

    def __init__(self, agent: Any, runtime: AnchorRuntime) -> None:
        self._agent = agent
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._wrap_llm()
        self._wrap_tools()

    def _wrap_llm(self) -> None:
        llm = getattr(self._agent, "llm", None)
        if llm is not None and not isinstance(llm, _AnchoredLLMProxy):
            safe_setattr(self._agent, "llm", _AnchoredLLMProxy(llm, self._runtime))

    def _wrap_tools(self) -> None:
        tools = getattr(self._agent, "tools", None) or []
        safe_setattr(
            self._agent,
            "tools",
            [
                t if isinstance(t, AnchoredTool) else AnchoredTool(t, self._runtime)
                for t in tools
            ],
        )

    def execute_task(self, task: Any, context: Any = None, **kwargs: Any) -> Any:
        description = getattr(task, "description", "") or str(task)
        agent_tools = getattr(self._agent, "tools", None) or []
        if self._runtime.state.mission is None:
            initialize_mission(self._runtime, description, agent_tools)
        else:
            refresh_mission_phase(self._runtime, description, agent_tools)

        with bind_runtime(self._runtime):
            return self._agent.execute_task(task, context=context, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


class _AnchoredCrewProxy:
    """Proxy around a CrewAI Crew that initializes ANCHOR on kickoff."""

    def __init__(self, crew: Any, runtime: AnchorRuntime) -> None:
        self._crew = crew
        self._anchor_runtime = runtime
        self._inputs: Dict[str, Any] = {}
        self._wrap_components()

    def _inputs_getter(self) -> Dict[str, Any]:
        return self._inputs

    def _wrap_components(self) -> None:
        crew_llm = getattr(self._crew, "llm", None)
        if crew_llm is not None and not isinstance(crew_llm, _AnchoredLLMProxy):
            safe_setattr(
                self._crew, "llm", _AnchoredLLMProxy(crew_llm, self._anchor_runtime)
            )

        wrapped_agents = [
            (
                a
                if isinstance(a, _AnchoredAgentProxy)
                else _AnchoredAgentProxy(a, self._anchor_runtime)
            )
            for a in getattr(self._crew, "agents", None) or []
        ]
        safe_setattr(self._crew, "agents", wrapped_agents)

        wrapped_tasks = [
            (
                t
                if isinstance(t, _AnchoredTaskProxy)
                else _AnchoredTaskProxy(t, self._anchor_runtime, self._inputs_getter)
            )
            for t in getattr(self._crew, "tasks", None) or []
        ]
        safe_setattr(self._crew, "tasks", wrapped_tasks)

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        self._inputs = dict(inputs or {})
        prompt = _mission_prompt_from_crew(self._crew, self._inputs)
        available_tools: List[str] = []
        for agent in getattr(self._crew, "agents", None) or []:
            for tool in getattr(agent, "tools", None) or []:
                available_tools.append(_tool_name(tool))

        self._anchor_runtime.initialize_mission(
            prompt,
            context={
                **mission_context(
                    self._anchor_runtime,
                    _collect_tools_from_agents(
                        getattr(self._crew, "agents", None) or []
                    ),
                ),
                "available_tools": list(dict.fromkeys(available_tools)),
            },
        )

        token = current_anchor_runtime.set(self._anchor_runtime)
        try:
            return self._crew.kickoff(inputs=inputs, **kwargs)
        finally:
            current_anchor_runtime.reset(token)
            finalize_session(self._anchor_runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._crew, name)


def _mission_prompt_from_crew(crew: Any, inputs: Optional[Dict[str, Any]]) -> str:
    """Build a mission prompt from crew tasks and kickoff inputs."""
    parts: List[str] = []
    if inputs:
        parts.append(" ".join(f"{k}: {v}" for k, v in inputs.items()))

    tasks = getattr(crew, "tasks", None) or []
    for task in tasks:
        inner = getattr(task, "_task", task)
        description = getattr(inner, "description", "")
        if description:
            parts.append(resolve_templates(description, inputs or {}))

    if not parts:
        return "Execute the assigned CrewAI workflow."
    return " ".join(parts)


def _wrap_agent(agent: Any, runtime: AnchorRuntime) -> _AnchoredAgentProxy:
    return _AnchoredAgentProxy(agent, runtime)


def _wrap_crew(crew: Any, runtime: AnchorRuntime) -> _AnchoredCrewProxy:
    return _AnchoredCrewProxy(crew, runtime)


def anchor_crewai(
    target: Any,
    *,
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
    isolation: str = "auto",
) -> Any:
    """
    Wrap a CrewAI Agent or Crew with ANCHOR mission governance.

    When attached to a terminal, ANCHOR prompts for human approval on blocked or
    high-risk actions automatically. Set interactive=False or ASHA_ANCHOR_INTERACTIVE=0
    for headless auto-deny.

    Documented usage (see README / docs/anchor.md):

        from asha.runtime.anchor.adapters import anchor_crewai
        from crewai import Agent, Crew, Task

        researcher = Agent(role="...", goal="...", tools=[...])
        crew = Crew(agents=[researcher], tasks=[...])
        crew = anchor_crewai(crew, risk_tolerance="LOW", interactive=False)
        crew.kickoff(inputs={"topic": "AI trends"})

    Install: ``pip install asha[crewai]``
    """
    module = type(target).__module__ or ""
    if "crewai" in module:
        from .deps import require_module

        require_module("crewai", adapter="anchor_crewai")
    runtime = AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )

    if _is_crew(target):
        return _wrap_crew(target, runtime)
    if _is_agent(target):
        return _wrap_agent(target, runtime)

    raise TypeError(
        f"anchor_crewai() expects a CrewAI Agent or Crew, got {type(target).__name__}."
    )
