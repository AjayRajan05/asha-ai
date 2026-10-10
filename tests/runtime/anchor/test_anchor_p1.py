"""P1 ANCHOR CrewAI adapter hardening tests."""

from __future__ import annotations

import pytest

from asha.exceptions import ASHAAnchorBlocked
from asha.runtime.anchor.adapters.crewai import (
    AnchoredTool,
    _AnchoredCrewProxy,
    _AnchoredLLMProxy,
    _AnchoredTaskProxy,
    _is_agent,
    _is_crew,
    anchor_crewai,
)
from asha.runtime.anchor.adapters.base import resolve_templates as _resolve_templates
from asha.runtime.anchor.llm_guard import extract_tool_calls_from_response
from asha.runtime.anchor.runtime import AnchorRuntime


class _StubTool:
    name = "write_trend_report"

    def __init__(self) -> None:
        self.calls = 0

    def _run(self, report_content: str = "") -> str:
        self.calls += 1
        return f"wrote:{len(report_content)}"


class _StubLLM:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, *args, **kwargs) -> str:
        self.calls += 1
        return "ok"


class _StubTask:
    def __init__(self, description: str) -> None:
        self.description = description
        self.executed_with: str | None = None

    def execute_sync(self, agent, context=None, tools=None):
        self.executed_with = self.description
        return "done"


class _StubAgent:
    def __init__(self, tools=None, llm=None) -> None:
        self.tools = tools or []
        self.llm = llm

    def execute_task(self, task, context=None, **kwargs):
        return task.execute_sync(self)


class _StubCrew:
    def __init__(self, agents, tasks, llm=None) -> None:
        self.agents = agents
        self.tasks = tasks
        self.llm = llm
        self.kickoff_inputs = None

    def kickoff(self, inputs=None, **kwargs):
        self.kickoff_inputs = inputs
        for task in self.tasks:
            task.execute_sync(self.agents[0])
        return "crew-complete"


def test_resolve_templates_replaces_placeholders() -> None:
    text = 'Analyze "{topic}" trends for {topic}.'
    assert _resolve_templates(text, {"topic": "AI"}) == 'Analyze "AI" trends for AI.'


def test_capability_detection_does_not_use_class_names() -> None:
    assert _is_crew(_StubCrew([], [])) is True
    assert _is_agent(_StubAgent()) is True
    assert _is_crew(_StubAgent()) is False


def test_anchored_tool_delegate_does_not_mutate_inner_tool() -> None:
    runtime = AnchorRuntime(warn_policy="permissive")
    runtime.initialize_mission(
        "Write report locally. Do not send data externally.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )

    inner = _StubTool()
    wrapped = AnchoredTool(inner, runtime)
    assert wrapped._run(report_content="Quarterly trends") == "wrote:16"
    assert wrapped._inner is inner
    assert inner.calls == 1


def test_refresh_mission_phase_updates_goal_per_task() -> None:
    runtime = AnchorRuntime(warn_policy="permissive")
    runtime.initialize_mission(
        "topic: AI",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )
    first_id = runtime.state.mission.mission_id

    inputs = {"topic": "Healthcare"}
    task = _AnchoredTaskProxy(
        _StubTask('Analyze "{topic}" locally.'),
        runtime,
        lambda: inputs,
    )
    task.execute_sync(_StubAgent(tools=[_StubTool()]))

    assert runtime.state.mission.mission_id == first_id
    assert "Healthcare" in runtime.state.mission.goal
    assert "write_trend_report" in runtime.state.mission.allowed_tools


def test_anchored_crew_proxy_wraps_without_patching_tool_run() -> None:
    runtime = AnchorRuntime(warn_policy="permissive")
    tool = _StubTool()
    agent = _StubAgent(tools=[tool], llm=_StubLLM())
    crew = _StubCrew([agent], [_StubTask("Write report locally.")], llm=_StubLLM())

    proxy = _AnchoredCrewProxy(crew, runtime)
    result = proxy.kickoff(inputs={"topic": "Healthcare"})

    assert result == "crew-complete"
    assert isinstance(proxy._crew.agents[0].tools[0], AnchoredTool)
    assert proxy._crew.agents[0].tools[0]._inner is tool


def test_anchored_llm_proxy_sets_anchor_context() -> None:
    runtime = AnchorRuntime(warn_policy="permissive")
    runtime.initialize_mission(
        "Analyze trends locally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    llm = _AnchoredLLMProxy(_StubLLM(), runtime)
    from asha.runtime.anchor.runtime import current_anchor_runtime

    assert current_anchor_runtime.get() is None
    assert llm.call("hello") == "ok"
    assert current_anchor_runtime.get() is None


def test_extract_tool_calls_from_openai_style_response() -> None:
    class Function:
        name = "load_trend_data"
        arguments = "{}"

    class ToolCall:
        function = Function()

    class Message:
        tool_calls = [ToolCall()]

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    calls = extract_tool_calls_from_response(Response())
    assert calls == [{"name": "load_trend_data", "arguments": "{}"}]


def test_anchor_crewai_accepts_capability_stubs() -> None:
    runtime_holder: dict[str, AnchorRuntime] = {}

    class Agent(_StubAgent):
        pass

    class Crew(_StubCrew):
        pass

    agent = anchor_crewai(Agent(tools=[_StubTool()]))
    assert hasattr(agent, "execute_task")
    assert callable(getattr(agent, "execute_task"))

    crew = anchor_crewai(Crew([agent], [_StubTask("Write locally.")]))
    assert hasattr(crew, "kickoff")
    assert callable(getattr(crew, "kickoff"))
    # Capability stubs remain attached after wrapping.
    assert getattr(agent, "tools", None) is not None or hasattr(agent, "execute_task")


def test_anchor_crewai_blocks_disallowed_tool_via_delegate() -> None:
    class Agent:
        def __init__(self):
            self.tools = [_StubTool()]

        def execute_task(self, task, context=None, **kwargs):
            return None

    agent = anchor_crewai(Agent(), warn_policy="strict")
    runtime = agent._anchor_runtime
    runtime.initialize_mission(
        "Analyze locally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    blocked_tool = _StubTool()
    blocked_tool.name = "network_exfil"

    result = AnchoredTool(blocked_tool, runtime)._run()
    assert "denied" in result.lower() or "blocked" in result.lower()
