"""P0 ANCHOR hardening tests derived from CrewAI integration findings."""

from __future__ import annotations

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
