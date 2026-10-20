from __future__ import annotations
"""Governance integration tests: mission session, chain/plan guards, sandbox, adapters."""


import pytest

from asha.runtime.anchor.adapters.generic import anchor_generic
from asha.runtime.anchor.adapters.mcp import anchor_mcp, is_mcp_server
from asha.runtime.anchor.adapters.registry import anchor_any
from asha.runtime.anchor.chain_guard import ChainGuard
from asha.runtime.anchor.mission import MissionCompiler
from asha.runtime.anchor.mission_session import (
    MissionSession,
    merge_mission_with_baseline,
)
from asha.runtime.anchor.plan_guard import evaluate_plan_output
from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.sandbox import SandboxManager
from asha.runtime.anchor.sandbox.hooks import SandboxViolation, enforcement_hooks
from asha.runtime.anchor.sandbox.policy import SandboxPolicy
from asha.runtime.anchor.tool_bridge import guarded_tool_call, wrap_callable_tool
from asha.runtime.anchor.types import ActionEvent


def _tool_action(tool_name: str) -> ActionEvent:
    return ActionEvent(
        action_id="a",
        action_type="tool_call",
        payload={"tool_name": tool_name, "arguments": "{}"},
        timestamp=0.0,
    )


def test_mission_session_preserves_baseline_local_only() -> None:
    session = MissionSession(MissionCompiler(default_risk_tolerance="LOW"))
    baseline = session.initialize(
        "Analyze trends locally. Do not send data externally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )
    assert baseline.local_only is True

    merged = session.refresh_phase(
        "Email stakeholders the summary.",
        context={"available_tools": ["send_email"], "local_only": False},
    )
    assert merged.local_only is True
    assert "send_email" in merged.allowed_tools


def test_merge_mission_keeps_baseline_write_paths() -> None:
    compiler = MissionCompiler()
    baseline = compiler.compile(
        "Write report to output/report.txt locally.",
        context={"local_only": True, "allowed_write_paths": ["output/"]},
    )
    phase = compiler.compile(
        "Upload backup to cloud.",
        context={"local_only": False, "allowed_write_paths": ["/tmp/"]},
    )
    merged = merge_mission_with_baseline(baseline, phase)
    assert merged.local_only is True
    assert "/tmp/" not in merged.allowed_write_paths
    assert any(str(path).startswith("output") for path in merged.allowed_write_paths)


def test_chain_guard_detects_read_to_exfil_sequence() -> None:
    compiler = MissionCompiler()
    mission = compiler.compile(
        "Load data and analyze locally.",
        context={
            "available_tools": ["load_trend_data", "send_email"],
            "local_only": True,
        },
    )
    guard = ChainGuard()
    history = [_tool_action("load_trend_data"), _tool_action("send_email")]
    verdict = guard.evaluate_chain(history, mission)
    assert verdict.verdict.value in ("BLOCK", "REVIEW")


def test_plan_guard_blocks_narrated_success_without_receipt() -> None:
    verdict = evaluate_plan_output(
        "The email was sent successfully to all stakeholders.",
        required_tool_markers=["send_email"],
        tools_used=[],
    )
    assert verdict.verdict.value == "BLOCK"


def test_plan_guard_allows_output_with_tool_receipt() -> None:
    verdict = evaluate_plan_output(
        "Email sent to alice@example.com with subject Weekly trends.",
        required_tool_markers=["send_email"],
        tools_used=["send_email"],
    )
    assert verdict.verdict.value == "ALLOW"


def test_sandbox_blocks_writes_outside_allowed_paths() -> None:
    policy = SandboxPolicy(allow_file_write=True, allowed_write_paths=("output/",))
    with enforcement_hooks(policy):
        with pytest.raises(SandboxViolation):
            open("/etc/passwd", "w")  # noqa: SIM115


def test_sandbox_manager_applies_mission_policy() -> None:
    runtime = AnchorRuntime(isolation="auto", interactive=False)
    runtime.initialize_mission(
        "Write report locally only.",
        context={"local_only": True, "allowed_write_paths": ["output/"]},
    )
    assert runtime.sandbox.policy.allow_network is False


def test_guarded_tool_call_blocks_high_risk_tool() -> None:
    runtime = AnchorRuntime(interactive=False)
    runtime.initialize_mission(
        "Analyze trends locally. Do not send externally.",
        context={
            "available_tools": ["load_trend_data", "send_email"],
            "local_only": True,
        },
    )

    def send_email(**_kwargs: object) -> str:
        return "should not run"

    result = guarded_tool_call(runtime, "send_email", send_email, to="a@b.com")
    assert "denied" in result.lower() or "blocked" in result.lower()


def test_generic_adapter_wraps_callable_tools() -> None:
    runtime = AnchorRuntime(interactive=False)

    def load_data() -> str:
        return "ok"

    class Agent:
        tools = [load_data]

        def run(self, prompt: str) -> str:
            return self.tools[0]()

    wrapped = anchor_generic(Agent(), runtime=runtime)
    runtime.initialize_mission(
        "Load data locally.",
        context={"available_tools": ["load_data"], "local_only": True},
    )
    assert wrapped.run("load") == "ok"


def test_mcp_adapter_intercepts_call_tool() -> None:
    class MCPServer:
        def list_tools(self):
            return [{"name": "send_email"}]

        def call_tool(self, name: str, arguments=None):
            return f"called:{name}"

    server = MCPServer()
    assert is_mcp_server(server)
    runtime = AnchorRuntime(interactive=False)
    wrapped = anchor_mcp(server, runtime=runtime)
    result = wrapped.call_tool("send_email", {"to": "x@y.com"})
    assert "denied" in result.lower() or "blocked" in result.lower()


def test_anchor_any_falls_back_to_generic() -> None:
    class SimpleAgent:
        tools = []

        def run(self, prompt: str) -> str:
            return prompt

    wrapped = anchor_any(SimpleAgent(), interactive=False)
    assert wrapped.run("hello") == "hello"


def test_wrap_callable_tool_records_execution() -> None:
    runtime = AnchorRuntime(interactive=False)
    runtime.initialize_mission(
        "Load trend data.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    def load_trend_data() -> str:
        return "data loaded"

    wrapped = wrap_callable_tool(load_trend_data, runtime)
    assert wrapped() == "data loaded"
    assert "load_trend_data" in runtime.state.tools_used


# --- Consolidated from tests/runtime/anchor/test_alignment_hardening.py ---
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
