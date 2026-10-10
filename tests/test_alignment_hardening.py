# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Alignment evaluator banding + hard-override tests (step 6)."""

from __future__ import annotations

import time

from asha.core.ml.calibration import load_thresholds
from asha.runtime.anchor.action_guard import ActionGuard
from asha.runtime.anchor.alignment_bands import (
    has_hard_block_trigger,
    score_to_verdict,
)
from asha.runtime.anchor.contracts import MissionContract
from asha.runtime.anchor.evaluator import AlignmentEvaluator
from asha.runtime.anchor.types import ActionEvent
from asha.runtime.anchor.verdicts import Verdict


def _contract(**overrides) -> MissionContract:
    base = dict(
        mission_id="t",
        goal="t",
        intent_summary="t",
        allowed_actions=["read", "write"],
        forbidden_actions=[],
        allowed_tools=["read_file", "write_file", "load_trend_data"],
        allowed_domains=[],
        allowed_memory_scopes=["session"],
        required_resources=[],
        expected_outcomes=[],
        completion_criteria=[],
        risk_tolerance="MEDIUM",
        created_at=time.time(),
        local_only=True,
        forbid_network_exfiltration=True,
        allowed_read_paths=["data/"],
        allowed_write_paths=["output/"],
    )
    base.update(overrides)
    return MissionContract(**base)  # type: ignore[arg-type]


def _tool(name: str, **payload) -> ActionEvent:
    body = {"tool_name": name, "arguments": "{}"}
    body.update(payload)
    return ActionEvent(
        action_id="a",
        action_type="tool_call",
        payload=body,
        timestamp=0.0,
    )


def test_score_bands_from_config_not_literals():
    load_thresholds(reload=True)
    assert score_to_verdict(0.95) is Verdict.ALLOW
    assert score_to_verdict(0.10) is Verdict.BLOCK
    mid = score_to_verdict(0.55)
    assert mid in (Verdict.WARN, Verdict.REVIEW, Verdict.ALLOW)


def test_hard_block_trigger_detection():
    assert has_hard_block_trigger(["Resource scope violation: path outside"])
    assert has_hard_block_trigger([], "Tool 'x' is not in allowed tools list.")
    assert not has_hard_block_trigger(["mild metadata hint"])


def test_high_risk_under_local_only_forces_block():
    guard = ActionGuard(AlignmentEvaluator())
    verdict = guard.evaluate_action(_tool("send_email"), _contract(local_only=True))
    assert verdict.verdict is Verdict.BLOCK
    assert verdict.risk_score >= 0.99


def test_allowed_local_read_allows():
    guard = ActionGuard(AlignmentEvaluator())
    verdict = guard.evaluate_action(
        _tool("read_file", path="data/log.txt"),
        _contract(allowed_tools=["read_file"], local_only=True),
    )
    assert verdict.verdict is Verdict.ALLOW


def test_resource_scope_hard_override():
    guard = ActionGuard(AlignmentEvaluator())
    # Path outside allowed_read_paths
    action = ActionEvent(
        action_id="a",
        action_type="tool_call",
        payload={
            "tool_name": "read_file",
            "arguments": '{"path": "/etc/passwd"}',
        },
        timestamp=0.0,
    )
    verdict = guard.evaluate_action(
        action,
        _contract(allowed_tools=["read_file"], allowed_read_paths=["data/"]),
    )
    assert verdict.verdict is Verdict.BLOCK
    assert (
        any("scope" in t.lower() for t in verdict.risk_triggers)
        or "scope" in verdict.reason.lower()
    )


def test_forbidden_action_hard_override():
    guard = ActionGuard(AlignmentEvaluator())
    action = ActionEvent(
        action_id="a",
        action_type="shell",
        payload={},
        timestamp=0.0,
    )
    verdict = guard.evaluate_action(action, _contract(forbidden_actions=["shell"]))
    assert verdict.verdict is Verdict.BLOCK


def test_hard_override_cannot_be_outvoted_by_perfect_band():
    """Override replaces verdict outright; medium/perfect scores cannot rescue."""
    from unittest.mock import patch

    guard = ActionGuard(AlignmentEvaluator())
    action = ActionEvent(
        action_id="a",
        action_type="shell",
        payload={},
        timestamp=0.0,
    )
    # Even if banding would map the calibrated score to ALLOW, forbidden wins.
    with patch(
        "asha.runtime.anchor.action_guard.score_to_verdict",
        return_value=Verdict.ALLOW,
    ):
        with patch(
            "asha.runtime.anchor.action_guard.calibrate_score",
            return_value=1.0,
        ):
            verdict = guard.evaluate_action(
                action, _contract(forbidden_actions=["shell"])
            )
    assert verdict.verdict is Verdict.BLOCK
    assert verdict.risk_score >= 0.99


def test_unknown_tool_not_allowlisted_blocks():
    guard = ActionGuard(AlignmentEvaluator())
    verdict = guard.evaluate_action(_tool("unknown_tool"), _contract())
    assert verdict.verdict is Verdict.BLOCK


def test_medium_penalties_cannot_rescue_hard_block():
    """Even if soft score math were higher, hard triggers force BLOCK."""
    evaluator = AlignmentEvaluator()
    score, expl, triggers = evaluator.evaluate(
        _tool("send_email"),
        _contract(local_only=True, allowed_tools=["send_email"]),
    )
    # send_email under local_only is hard-blocked even if allowlisted name-wise
    # when not mission-local: score must be 0.
    assert score == 0.0
    assert has_hard_block_trigger(triggers, expl)
