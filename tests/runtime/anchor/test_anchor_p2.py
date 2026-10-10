"""P2 ANCHOR mission intelligence tests."""

from __future__ import annotations

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
