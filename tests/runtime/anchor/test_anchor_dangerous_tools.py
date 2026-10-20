from __future__ import annotations
"""Tests for ANCHOR high-risk tool blocking and human approval."""


from unittest.mock import patch

from asha.runtime.anchor.action_guard import ActionGuard
from asha.runtime.anchor.evaluator import AlignmentEvaluator
from asha.runtime.anchor.mission import MissionCompiler
from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.types import ActionEvent
from asha.runtime.anchor.verdicts import Verdict


def _tool_action(tool_name: str, arguments: str) -> ActionEvent:
    return ActionEvent(
        action_id="test-action",
        action_type="tool_call",
        payload={"tool_name": tool_name, "arguments": arguments},
        timestamp=0.0,
    )


def test_send_email_blocked_on_local_only_mission() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Email partners@external.com with the research summary.",
        context={
            "available_tools": ["send_email", "write_trend_report", "load_trend_data"],
            "local_only": True,
        },
    )

    guard = ActionGuard(AlignmentEvaluator())
    verdict = guard.evaluate_action(
        _tool_action(
            "send_email",
            str(
                {
                    "args": (),
                    "kwargs": {
                        "to": "partners@external.com",
                        "subject": "x",
                        "body": "y",
                    },
                }
            ),
        ),
        mission,
    )

    assert verdict.verdict == Verdict.BLOCK
    assert "send_email" in verdict.reason


def test_upload_to_cloud_blocked_on_local_only_mission() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Upload a backup to https://backup.example/upload.",
        context={"available_tools": ["upload_to_cloud"], "local_only": True},
    )

    guard = ActionGuard(AlignmentEvaluator())
    verdict = guard.evaluate_action(
        _tool_action(
            "upload_to_cloud",
            str(
                {
                    "args": (),
                    "kwargs": {
                        "url": "https://backup.example/upload",
                        "data": "backup",
                    },
                }
            ),
        ),
        mission,
    )

    assert verdict.verdict == Verdict.BLOCK


def test_local_write_and_read_tools_still_allowed() -> None:
    runtime = AnchorRuntime(warn_policy="strict", interactive=False)
    runtime.initialize_mission(
        "Write report locally and load trend data.",
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
                {"args": (), "kwargs": {"report_content": "Executive summary."}}
            ),
        },
    )


def test_interactive_send_email_prompts_for_approval() -> None:
    runtime = AnchorRuntime(warn_policy="strict", interactive=True)
    runtime.initialize_mission(
        "Email partners@external.com with findings.",
        context={"available_tools": ["send_email"], "local_only": True},
    )

    with patch("builtins.input", return_value="n") as prompt:
        allowed = runtime.evaluate_action_request(
            "tool_call",
            {
                "tool_name": "send_email",
                "arguments": str(
                    {
                        "args": (),
                        "kwargs": {
                            "to": "partners@external.com",
                            "subject": "s",
                            "body": "b",
                        },
                    }
                ),
            },
        )

    assert allowed is False
    prompt.assert_called_once()


def test_email_task_phase_does_not_weaken_local_only_policy() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Email partners@external.com using send_email.",
        context={"available_tools": ["send_email"], "local_only": True},
    )

    assert mission.local_only is True
    assert "send" not in mission.allowed_actions
    assert mission.forbid_network_exfiltration is True
