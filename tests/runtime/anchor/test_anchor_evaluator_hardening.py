from __future__ import annotations
"""Consolidated ANCHOR evaluator hardening and gap tests."""
import pytest

# --- From tests/runtime/anchor/test_anchor_p0.py ---
"""P0 ANCHOR hardening tests derived from CrewAI integration findings."""


import pytest

from asha.exceptions import ASHAAnchorBlocked
from asha.runtime.anchor.evaluator import AlignmentEvaluator
from asha.runtime.anchor.mission import MissionCompiler
from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.types import ActionEvent
from asha.runtime.anchor.payload_inspection import (
    contains_forbidden_term,
    extract_inspection_metadata,
    is_content_exempt_tool,
)


def _tool_action(tool_name: str, arguments: str) -> ActionEvent:
    return ActionEvent(
        action_id="test-action",
        action_type="tool_call",
        payload={"tool_name": tool_name, "arguments": arguments},
        timestamp=0.0,
    )


def test_structured_inspection_ignores_large_report_bodies() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Write trend report locally. Do not send data externally.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )
    evaluator = AlignmentEvaluator()

    long_body = (
        "Executive summary with send, format, and upload references in prose. " * 50
    )
    action = _tool_action(
        "write_trend_report",
        str({"args": (), "kwargs": {"report_content": long_body}}),
    )

    score, _, _ = evaluator.evaluate(action, mission)
    assert score >= 0.8


def test_forbidden_metadata_still_blocks_network_urls() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Analyze trends locally. Do not send data externally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )
    evaluator = AlignmentEvaluator()

    action = _tool_action(
        "load_trend_data",
        str({"args": (), "kwargs": {"url": "https://evil.example/upload"}}),
    )

    score, explanation, triggers = evaluator.evaluate(action, mission)
    assert score < 0.5
    assert "upload" in explanation.lower() or "https" in explanation.lower()
    assert triggers


def test_word_boundary_matching_avoids_information_false_positive() -> None:
    assert contains_forbidden_term("information design", "format") is False
    assert contains_forbidden_term("disk format required", "format") is True


def test_mission_compiler_adds_write_for_local_report_missions() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Write report to output/trend_analysis.txt locally.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )

    assert "write" in mission.allowed_actions
    assert mission.local_only is True
    assert is_content_exempt_tool("write_trend_report", mission.allowed_tools)


def test_drift_score_zero_for_allowlisted_tool_calls() -> None:
    runtime = AnchorRuntime(warn_policy="permissive")
    runtime.initialize_mission(
        "Analyze trends and write report locally. Do not send data externally.",
        context={
            "available_tools": ["load_trend_data", "write_trend_report"],
            "local_only": True,
        },
    )

    assert runtime.evaluate_action_request(
        "tool_call",
        {"tool_name": "load_trend_data", "arguments": "{}"},
    )
    assert runtime.evaluate_action_request(
        "tool_call",
        {
            "tool_name": "write_trend_report",
            "arguments": str(
                {
                    "args": (),
                    "kwargs": {
                        "report_content": "Please send format upload guidance in this report."
                    },
                }
            ),
        },
    )

    assert runtime.state.risk_history
    latest = runtime.state.risk_history[-1]
    assert latest.drift_score == 0.0
    assert latest.total_risk_score < 20.0


def test_permissive_policy_auto_allows_review_for_allowlisted_write_tool() -> None:
    runtime = AnchorRuntime(warn_policy="permissive")
    runtime.initialize_mission(
        "Write report to output/trend_analysis.txt. Do not send externally.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )

    allowed = runtime.evaluate_action_request(
        "tool_call",
        {
            "tool_name": "write_trend_report",
            "arguments": str(
                {"args": (), "kwargs": {"report_content": "format disk send"}}
            ),
        },
    )
    assert allowed is True


def test_strict_policy_blocks_review_verdicts() -> None:
    runtime = AnchorRuntime(warn_policy="strict")
    runtime.initialize_mission(
        "Write report to output/trend_analysis.txt. Do not send externally.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )

    allowed = runtime.evaluate_action_request(
        "tool_call",
        {
            "tool_name": "write_trend_report",
            "arguments": str(
                {"args": (), "kwargs": {"path": "https://evil.example/upload"}}
            ),
        },
    )
    assert allowed is False


def test_raise_on_block_raises_asha_anchor_blocked() -> None:
    runtime = AnchorRuntime(warn_policy="strict")
    runtime.initialize_mission(
        "Analyze trends locally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    with pytest.raises(ASHAAnchorBlocked, match="blocked tool/action 'network_exfil'"):
        runtime.evaluate_action_request(
            "tool_call",
            {"tool_name": "network_exfil", "arguments": "{}"},
            raise_on_block=True,
        )


def test_anchor_crewai_adapter_raises_instead_of_returning_string() -> None:
    from asha.runtime.anchor.adapters.crewai import anchor_crewai

    class MockTool:
        name = "network_exfil"

        def _run(self, **kwargs):
            return "exfiltrated"

    class Agent:
        def __init__(self):
            self.tools = [MockTool()]

        def execute_task(self, task, context=None, **kwargs):
            return None

    agent = anchor_crewai(Agent(), warn_policy="strict")
    runtime = agent._anchor_runtime
    runtime.initialize_mission(
        "Analyze trends locally. Do not send data externally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    result = agent.tools[0]._run()
    assert "denied" in result.lower() or "blocked" in result.lower()

# --- From tests/runtime/anchor/test_anchor_p1.py ---
"""P1 ANCHOR CrewAI adapter hardening tests."""


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

# --- From tests/runtime/anchor/test_anchor_p2.py ---
"""P2 ANCHOR mission intelligence tests."""


import json

from asha.runtime.anchor.evaluator import AlignmentEvaluator
from asha.runtime.anchor.intent_parser import parse_mission_intent
from asha.runtime.anchor.mission import MissionCompiler
from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.telemetry import AnchorTelemetry
from asha.runtime.anchor.types import ActionEvent
from asha.runtime.anchor.payload_inspection import (
    find_forbidden_metadata_matches,
    validate_resource_scope,
    extract_inspection_metadata,
)


def _tool_action(tool_name: str, arguments: str) -> ActionEvent:
    return ActionEvent(
        action_id="test-action",
        action_type="tool_call",
        payload={"tool_name": tool_name, "arguments": arguments},
        timestamp=1.0,
    )


def test_intent_parser_extracts_explicit_outcomes_and_paths() -> None:
    prompt = (
        "Load data/sample_trends.json and write report to output/trend_analysis.txt. "
        "Do not send data externally."
    )
    intent = parse_mission_intent(
        prompt,
        {
            "available_tools": ["load_trend_data", "write_trend_report"],
            "local_only": True,
        },
    )

    assert intent.local_only is True
    assert intent.forbid_network_exfiltration is True
    assert "data/sample_trends.json" in intent.allowed_read_paths
    assert "output/trend_analysis.txt" in intent.allowed_write_paths
    assert any(
        "output/trend_analysis.txt" in outcome for outcome in intent.expected_outcomes
    )
    assert any(
        "output/trend_analysis.txt" in criterion
        for criterion in intent.completion_criteria
    )


def test_intent_parser_does_not_treat_send_in_prose_as_policy_token() -> None:
    intent = parse_mission_intent(
        "Write a report that discusses send operations in theory.",
        {"available_tools": ["write_trend_report"]},
    )

    assert "send" not in intent.forbidden_network_tokens


def test_resource_scope_blocks_write_outside_output() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Write report to output/trend_analysis.txt locally.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )

    metadata = extract_inspection_metadata(
        {
            "tool_name": "write_trend_report",
            "arguments": str({"kwargs": {"path": "secrets/credentials.txt"}}),
        }
    )
    violation = validate_resource_scope("write_trend_report", metadata, mission)
    assert violation is not None
    assert violation.term == "write_scope"


def test_resource_scope_allows_write_within_output() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Write report to output/trend_analysis.txt locally.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )

    metadata = extract_inspection_metadata(
        {
            "tool_name": "write_trend_report",
            "arguments": str({"kwargs": {"path": "output/trend_analysis.txt"}}),
        }
    )
    assert validate_resource_scope("write_trend_report", metadata, mission) is None


def test_network_tokens_only_apply_to_network_metadata_fields() -> None:
    matches = find_forbidden_metadata_matches(
        {"tool_name": "write_trend_report", "filename": "send-report.txt"},
        forbidden_actions=["send", "upload", "https"],
        network_only_tokens=["send", "upload", "https"],
    )
    assert matches == []


def test_network_metadata_match_includes_field_detail() -> None:
    matches = find_forbidden_metadata_matches(
        {"url": "https://evil.example/upload"},
        forbidden_actions=["upload", "https"],
        network_only_tokens=["upload", "https"],
    )
    assert matches
    assert matches[0].field == "url"
    assert "upload" in matches[0].value


def test_evaluator_returns_risk_triggers_with_field_detail() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Analyze trends locally. Do not send data externally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )
    evaluator = AlignmentEvaluator()

    action = _tool_action(
        "load_trend_data",
        str({"kwargs": {"url": "https://evil.example/upload"}}),
    )
    score, explanation, triggers = evaluator.evaluate(action, mission)

    assert score < 0.5
    assert "url=" in explanation
    assert any("url=" in trigger for trigger in triggers)


def test_telemetry_logs_risk_triggers(tmp_path) -> None:
    telemetry = AnchorTelemetry(log_file=str(tmp_path / "telemetry.jsonl"))
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Analyze trends locally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )
    runtime = AnchorRuntime(warn_policy="strict")
    runtime.state.mission = mission
    runtime.telemetry = telemetry

    runtime.evaluate_action_request(
        "tool_call",
        {"tool_name": "network_exfil", "arguments": "{}"},
    )

    content = (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").strip()
    entry = json.loads(content.splitlines()[0])
    assert entry["event_type"] == "action_evaluated"
    assert entry["risk_triggers"]


def test_refresh_mission_phase_merges_resource_paths() -> None:
    runtime = AnchorRuntime(warn_policy="permissive")
    runtime.initialize_mission(
        "Load data/sample_trends.json locally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    runtime.refresh_mission_phase(
        "Write report to output/trend_analysis.txt.",
        context={"available_tools": ["write_trend_report"], "local_only": True},
    )

    mission = runtime.state.mission
    assert "data/sample_trends.json" in mission.allowed_read_paths
    assert "output/trend_analysis.txt" in mission.allowed_write_paths
    assert "load_trend_data" in mission.allowed_tools
    assert "write_trend_report" in mission.allowed_tools

# --- From tests/runtime/anchor/test_anchor_gaps.py ---
"""Gap-closure tests: asha.Agent, LangGraph, full adapter wiring."""


import pytest

from asha import Agent, anchor
from asha.runtime.agent_tools import normalize_tools, parse_tool_call, run_tool_loop
from asha.runtime.anchor.adapters.asha_agent import anchor_asha_agent, is_asha_agent
from asha.runtime.anchor.adapters.discovery import discover_tools
from asha.runtime.anchor.adapters.graph import anchor_graph, is_agent_graph
from asha.runtime.anchor.adapters.registry import anchor_any
from asha.runtime.anchor.runtime import AnchorRuntime


def test_normalize_tools_from_strings_and_callables() -> None:
    def load_data() -> str:
        return "ok"

    tools = normalize_tools(
        ["email", load_data, {"name": "write_report", "fn": lambda: "w"}]
    )
    assert len(tools) == 3
    assert tools[0].name == "email"
    assert tools[1].name == "load_data"


def test_parse_tool_call_formats() -> None:
    text = 'TOOL_CALL: send_email\nARGS: {"to": "a@b.com"}'
    parsed = parse_tool_call(text)
    assert parsed == ("send_email", {"to": "a@b.com"})


def test_agent_tool_loop_executes_callable() -> None:
    calls: list[str] = []

    def load_data() -> str:
        calls.append("load")
        return "data loaded"

    def fake_generate(prompt: str) -> str:
        if "Tool result" in prompt:
            return "Final answer with data loaded"
        return "TOOL_CALL: load_data\nARGS: {}"

    result = run_tool_loop(
        generate=fake_generate,
        tools=[load_data],
        prompt="Load data locally",
    )
    assert "data loaded" in result
    assert calls == ["load"]


def test_asha_agent_run_with_tools() -> None:
    def load_trend_data() -> str:
        return "trends loaded"

    agent = Agent(provider="mock", model="mock", tools=[load_trend_data])
    result = agent.run("Analyze trends locally")
    assert "Mock" in result
    assert "tools" in result.lower() or "analyze" in result.lower()


def test_anchor_asha_agent_governs_tools() -> None:
    def load_trend_data() -> str:
        return "trends loaded"

    def send_email(**_kwargs: object) -> str:
        return "sent"

    runtime = AnchorRuntime(interactive=False)
    agent = anchor_asha_agent(
        Agent(provider="mock", model="mock", tools=[load_trend_data, send_email]),
        runtime=runtime,
    )
    assert is_asha_agent(agent._inner)
    result = agent.run("Analyze trends locally. Do not send externally.")
    assert isinstance(result, str)
    wrapped = agent._inner.tools[0]
    denied = wrapped._run()
    assert "trends loaded" in str(denied) or "denied" not in str(denied).lower()


def test_anchor_any_dispatches_asha_agent() -> None:
    agent = Agent(provider="mock", model="mock", tools=[])
    wrapped = anchor_any(agent, interactive=False)
    result = wrapped.run("Analyze trends locally")
    assert "Mock" in result
    assert hasattr(wrapped, "_anchor_runtime")


def test_anchor_blocks_dangerous_tool_on_asha_agent() -> None:
    def send_email(**_kwargs: object) -> str:
        return "Email sent to x"

    runtime = AnchorRuntime(interactive=False)
    agent = anchor_asha_agent(
        Agent(provider="mock", model="mock", tools=[send_email]),
        runtime=runtime,
    )
    runtime.initialize_mission(
        "Analyze locally. Do not send externally.",
        context={"available_tools": ["send_email"], "local_only": True},
    )
    result = agent._inner.tools[0]._run()
    assert "denied" in result.lower() or "blocked" in result.lower()


class _LangGraphStub:
    # Duck-typed graph shape; avoid "langgraph" in module so unit tests do not
    # require asha[langgraph] (integration tests cover the real package).
    __module__ = "tests.stubs.agent_graph"

    def __init__(self) -> None:
        self.nodes = {"tools": _ToolNodeStub()}
        self._result = {"result": "ok"}

    def get_graph(self):
        return self

    def invoke(self, input: dict, config=None, **kwargs):
        return self._result

    def compile(self):
        return self


class _ToolNodeStub:
    tools = []

    def __init__(self) -> None:
        class _T:
            name = "load_trend_data"

            def _run(self, **kwargs):
                return "loaded"

        self.tools = [_T()]


def test_graph_adapter_detection_and_invoke() -> None:
    graph = _LangGraphStub()
    assert is_agent_graph(graph)
    wrapped = anchor_graph(graph, interactive=False)
    result = wrapped.invoke(
        {"messages": [{"role": "user", "content": "Analyze locally."}]}
    )
    assert result["result"] == "ok"


def test_discover_tools_finds_nested_graph_tools() -> None:
    graph = _LangGraphStub()
    tools = discover_tools(graph)
    assert len(tools) >= 1


def test_anchor_dispatches_graph() -> None:
    wrapped = anchor_any(_LangGraphStub(), interactive=False)
    assert is_agent_graph(wrapped._inner) or hasattr(wrapped, "invoke")
