"""Governance integration tests: mission session, chain/plan guards, sandbox, adapters."""

from __future__ import annotations

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
