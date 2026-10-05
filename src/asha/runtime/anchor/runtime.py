"""ANCHOR central runtime controller."""

from __future__ import annotations

import contextvars
import time
from typing import Any, Dict, Optional

from ...exceptions import ASHAAnchorBlocked
from .action_guard import ActionGuard
from .approval import ApprovalEngine
from .chain_guard import ChainGuard
from .evaluator import AlignmentEvaluator
from .memory_guard import MemoryGuard
from .mission import MissionCompiler
from .mission_session import MissionSession
from .plan_guard import evaluate_plan_output, tools_used_from_history
from .sandbox import SandboxManager
from .settings import resolve_interactive, resolve_warn_policy
from .telemetry import AnchorTelemetry
from .types import ActionEvent, AnchorState, RiskSummary
from .verdicts import RiskSeverity, Verdict

current_anchor_runtime: contextvars.ContextVar[Optional["AnchorRuntime"]] = (
    contextvars.ContextVar(
        "current_anchor_runtime",
        default=None,
    )
)


class AnchorRuntime:
    """Central runtime controller for the Anchor subsystem."""

    def __init__(
        self,
        risk_tolerance: str = "LOW",
        warn_policy: Optional[str] = None,
        interactive: Optional[bool] = None,
        isolation: str = "auto",
        *,
        mission: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        allowed_actions: Optional[list[str]] = None,
        forbidden_actions: Optional[list[str]] = None,
        local_only: bool = False,
        enable_telemetry: Optional[bool] = None,
        telemetry_path: Optional[str] = None,
        **extra_mission_kwargs: Any,
    ):
        resolved_interactive = resolve_interactive(interactive)
        resolved_warn_policy = resolve_warn_policy(
            warn_policy, interactive=resolved_interactive
        )
        self.state = AnchorState()
        self.mission_compiler = MissionCompiler(default_risk_tolerance=risk_tolerance)
        self.mission_session = MissionSession(self.mission_compiler)
        self.evaluator = AlignmentEvaluator()
        self.action_guard = ActionGuard(self.evaluator)
        self.memory_guard = MemoryGuard()
        self.chain_guard = ChainGuard()
        self.approval_engine = ApprovalEngine(
            warn_policy=resolved_warn_policy,
            interactive=resolved_interactive,
        )
        # Opt-in only: never create anchor_telemetry.jsonl by default.
        self.telemetry = AnchorTelemetry(
            log_file=telemetry_path,
            enabled=enable_telemetry,
        )
        self.sandbox = SandboxManager(mode=isolation)

        if mission is not None:
            ctx = {
                "available_tools": allowed_tools or [],
                "allowed_tools": allowed_tools or [],
                "allowed_actions": allowed_actions or [],
                "forbidden_actions": forbidden_actions or [],
                "local_only": local_only,
                "risk_tolerance": risk_tolerance,
                **extra_mission_kwargs,
            }
            self.initialize_mission(mission, ctx)

    def initialize_mission(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create the session baseline mission contract."""
        mission = self.mission_session.initialize(prompt, context)
        self.state.baseline_mission = mission
        self.state.mission = mission
        self._apply_sandbox_from_mission(mission)

    def refresh_mission_phase(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Refresh phase goal without weakening baseline mission constraints."""
        effective_context: Dict[str, Any] = dict(context or {})
        if self.state.baseline_mission is not None:
            baseline = self.state.baseline_mission
            merged_tools = list(baseline.allowed_tools)
            for tool in effective_context.get("available_tools", []):
                if tool not in merged_tools:
                    merged_tools.append(tool)
            effective_context["available_tools"] = merged_tools
            effective_context.setdefault("risk_tolerance", baseline.risk_tolerance)
            effective_context.setdefault("local_only", baseline.local_only)
            effective_context.setdefault(
                "allowed_read_paths", list(baseline.allowed_read_paths)
            )
            effective_context.setdefault(
                "allowed_write_paths", list(baseline.allowed_write_paths)
            )

        mission = self.mission_session.refresh_phase(prompt, effective_context)
        self.state.mission = mission
        self._apply_sandbox_from_mission(mission)

    def _apply_sandbox_from_mission(self, mission: Any) -> None:
        self.sandbox.apply_mission(
            local_only=mission.local_only,
            allowed_write_paths=list(mission.allowed_write_paths),
        )

    def _is_off_task_action(self, action: ActionEvent) -> bool:
        mission = self.state.mission
        if mission is None or not mission.allowed_tools:
            return False
        if action.action_type == "tool_call":
            tool_name = str(action.payload.get("tool_name", ""))
            return tool_name not in mission.allowed_tools
        return action.action_type not in mission.allowed_tools

    def _record_tool_use(self, payload: Dict[str, Any]) -> None:
        tool_name = str(payload.get("tool_name", "") or "")
        if tool_name and tool_name not in self.state.tools_used:
            self.state.tools_used.append(tool_name)

    def _update_risk_summary(self) -> None:
        if not self.state.mission:
            return

        avg_alignment = 1.0
        if self.state.alignment_history:
            avg_alignment = sum(self.state.alignment_history) / len(
                self.state.alignment_history
            )

        mission_drift = 1.0 - avg_alignment
        intent_drift = 0.0
        if self.state.actions_history:
            off_task_actions = sum(
                1
                for action in self.state.actions_history
                if self._is_off_task_action(action)
            )
            intent_drift = min(
                1.0, off_task_actions / max(len(self.state.actions_history), 1)
            )

        drift_score = (mission_drift + intent_drift) / 2.0

        memory_integrity_score = 1.0
        if self.state.memory_history:
            high_risk_writes = sum(
                1
                for event in self.state.memory_history
                if event.operation in ["write", "update"] and event.scope != "session"
            )
            total_writes = sum(
                1
                for event in self.state.memory_history
                if event.operation in ["write", "update"]
            )
            if total_writes > 0:
                memory_integrity_score = max(0.0, 1.0 - (high_risk_writes * 0.2))

        total_risk = (drift_score * 0.6) + ((1.0 - memory_integrity_score) * 0.4)
        total_risk_scaled = total_risk * 100.0

        if total_risk_scaled > 80:
            severity = RiskSeverity.CRITICAL
        elif total_risk_scaled > 50:
            severity = RiskSeverity.HIGH
        elif total_risk_scaled > 20:
            severity = RiskSeverity.MEDIUM
        else:
            severity = RiskSeverity.LOW

        summary = RiskSummary(
            alignment_score=avg_alignment,
            memory_integrity_score=memory_integrity_score,
            drift_score=drift_score,
            total_risk_score=total_risk_scaled,
            severity=severity,
            explanation=f"Cumulative risk calculated. Drift: {drift_score:.2f}",
        )
        self.state.risk_history.append(summary)
        self.telemetry.log_risk_summary(summary, time.time())

    def evaluate_action_request(
        self,
        action_type: str,
        payload: Dict[str, Any],
        *,
        raise_on_block: bool = False,
        record: bool = True,
    ) -> bool:
        if not self.state.mission:
            if raise_on_block:
                raise ASHAAnchorBlocked(
                    "ANCHOR blocked action: mission contract is not initialized."
                )
            return False

        action = self.action_guard.normalize_action(action_type, payload)
        if record:
            self.state.actions_history.append(action)

        action_verdict = self.action_guard.evaluate_action(action, self.state.mission)
        self.telemetry.log_action_evaluated(action, action_verdict)
        self.state.alignment_history.append(1.0 - action_verdict.risk_score)

        approval_record = self.approval_engine.process_verdict(
            action_verdict.verdict,
            action_verdict.reason,
            action.action_id,
            action=action,
            mission=self.state.mission,
        )
        self.state.approval_history.append(approval_record)
        self.telemetry.log_approval(
            approval_record.record_id,
            approval_record.target_id,
            approval_record.verdict.verdict.value,
            approval_record.verdict.reason,
            approval_record.timestamp,
        )

        if approval_record.verdict.verdict == Verdict.BLOCK:
            self._update_risk_summary()
            if raise_on_block:
                tool_name = payload.get("tool_name", action_type)
                raise ASHAAnchorBlocked(
                    f"ANCHOR blocked tool/action '{tool_name}': {approval_record.verdict.reason}"
                )
            return False

        chain_verdict = self.chain_guard.evaluate_chain(
            self.state.actions_history, self.state.mission
        )
        chain_event = self.chain_guard.normalize_chain(
            list(self.state.actions_history),
            pattern=(
                chain_verdict.reason if chain_verdict.verdict != Verdict.ALLOW else None
            ),
        )
        self.telemetry.log_chain_evaluated(chain_event, chain_verdict)
        if chain_verdict.verdict in [Verdict.BLOCK, Verdict.REVIEW]:
            chain_approval = self.approval_engine.process_verdict(
                chain_verdict.verdict,
                chain_verdict.reason,
                action.action_id,
                action=action,
                mission=self.state.mission,
            )
            self.state.approval_history.append(chain_approval)
            self.telemetry.log_approval(
                chain_approval.record_id,
                chain_approval.target_id,
                chain_approval.verdict.verdict.value,
                chain_approval.verdict.reason,
                chain_approval.timestamp,
            )
            if chain_approval.verdict.verdict == Verdict.BLOCK:
                self._update_risk_summary()
                if raise_on_block:
                    tool_name = payload.get("tool_name", action_type)
                    raise ASHAAnchorBlocked(
                        f"ANCHOR blocked tool/action '{tool_name}': {chain_approval.verdict.reason}"
                    )
                return False

        if action_type == "tool_call":
            self._record_tool_use(payload)

        self._update_risk_summary()
        return True

    def evaluate_memory_request(self, operation: str, content: str, scope: str) -> bool:
        if not self.state.mission:
            return False

        event = self.memory_guard.normalize_memory_event(operation, content, scope)
        self.state.memory_history.append(event)

        memory_verdict = self.memory_guard.evaluate_memory(
            event, self.state.mission, content
        )
        self.telemetry.log_memory_evaluated(event, memory_verdict)

        approval_record = self.approval_engine.process_verdict(
            memory_verdict.verdict,
            memory_verdict.reason,
            event.event_id,
            action=None,
            mission=self.state.mission,
        )
        self.state.approval_history.append(approval_record)
        self.telemetry.log_approval(
            approval_record.record_id,
            approval_record.target_id,
            approval_record.verdict.verdict.value,
            approval_record.verdict.reason,
            approval_record.timestamp,
        )

        self._update_risk_summary()
        return approval_record.verdict.verdict != Verdict.BLOCK

    def evaluate_plan_request(
        self,
        output: str,
        *,
        required_tools: Optional[list[str]] = None,
    ) -> bool:
        """Evaluate agent output through plan guard and approval engine. Returns True if allowed."""
        tools_used = self.state.tools_used or tools_used_from_history(
            (event.action_type, event.payload) for event in self.state.actions_history
        )
        plan_verdict = evaluate_plan_output(
            output,
            required_tool_markers=required_tools or [],
            tools_used=tools_used,
        )
        target_id = f"plan-{int(time.time())}"
        self.telemetry.log_plan_evaluated(
            plan_verdict, target_id=target_id, timestamp=time.time()
        )

        if plan_verdict.verdict == Verdict.ALLOW:
            return True

        approval = self.approval_engine.process_verdict(
            plan_verdict.verdict,
            plan_verdict.reason,
            target_id=target_id,
            mission=self.state.mission,
        )
        self.state.approval_history.append(approval)
        self.telemetry.log_approval(
            approval.record_id,
            approval.target_id,
            approval.verdict.verdict.value,
            approval.verdict.reason,
            approval.timestamp,
        )
        self._update_risk_summary()
        return approval.verdict.verdict != Verdict.BLOCK

    def govern_step_output(
        self,
        output: Any,
        *,
        required_tools: Optional[list[str]] = None,
        raise_on_block: bool = True,
    ) -> Any:
        """Apply memory and plan governance after an agent step. Returns output if allowed."""
        output_text = str(output or "")

        memory_ok = self.evaluate_memory_request("write", output_text[:2000], "session")
        if not memory_ok:
            reason = (
                self.state.approval_history[-1].verdict.reason
                if self.state.approval_history
                else "Memory guard blocked output."
            )
            if raise_on_block:
                raise RuntimeError(f"ANCHOR memory guard blocked output: {reason}")
            return output

        plan_ok = self.evaluate_plan_request(
            output_text, required_tools=required_tools or []
        )
        if not plan_ok:
            reason = (
                self.state.approval_history[-1].verdict.reason
                if self.state.approval_history
                else "Plan guard blocked output."
            )
            if raise_on_block:
                raise RuntimeError(f"ANCHOR plan guard blocked output: {reason}")

        return output

    def finalize_session(self) -> Optional[RiskSummary]:
        """Finalize the ANCHOR session, compute final risk, and emit telemetry."""
        if not self.state.mission:
            return None
        self._update_risk_summary()
        if not self.state.risk_history:
            return None
        summary = self.state.risk_history[-1]
        self.telemetry.log_session_finalized(summary, self.state, time.time())
        return summary


def anchor(
    target: Any,
    *,
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
    isolation: str = "auto",
) -> Any:
    """
    One-line ANCHOR adoption for any supported agent or framework.

    Examples:
        agent = anchor(Agent(...))
        crew = anchor(crewai_crew)
        executor = anchor(langchain_agent_executor)
    """
    from .adapters.registry import anchor_any

    return anchor_any(
        target,
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
