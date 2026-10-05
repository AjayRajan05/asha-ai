import time
import uuid
from typing import Optional
from .verdicts import ApprovalVerdict, Verdict
from .types import ApprovalRecord, ActionEvent
from .contracts import MissionContract
from .payload_inspection import is_write_tool, is_high_risk_tool, is_mission_local_tool

_DANGEROUS_TOOL_HINTS = (
    "send",
    "email",
    "upload",
    "delete",
    "network",
    "http",
    "exfil",
)


class ApprovalEngine:
    """
    Human approval engine.
    Handles escalation to humans when guards return REVIEW.
    Records decisions and ensures no silent auto-approvals.
    """

    def __init__(self, warn_policy: str = "strict", interactive: bool = False):
        self.warn_policy = warn_policy
        self.interactive = interactive

    def _prompt_human(self, reason: str, context: str = "") -> str:
        print("\n" + "=" * 60)
        print("ANCHOR HUMAN APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Reason: {reason}")
        if context:
            print(f"Context: {context}")
        print("=" * 60)
        return input("Allow this action? [y/N]: ").strip().lower()

    def _auto_allow_review(
        self,
        action: Optional[ActionEvent],
        mission: Optional[MissionContract],
    ) -> bool:
        """Permissive headless policy for allowlisted local mission tools."""
        if self.warn_policy != "permissive" or action is None or mission is None:
            return False

        if action.action_type != "tool_call" or not mission.local_only:
            return False

        tool_name = str(action.payload.get("tool_name", ""))
        if any(hint in tool_name.lower() for hint in _DANGEROUS_TOOL_HINTS):
            return False
        if tool_name not in mission.allowed_tools:
            return False

        if is_write_tool(tool_name) and "write" in mission.allowed_actions:
            return True

        return tool_name in mission.allowed_tools

    def request_approval(
        self, target_id: str, reason: str, context: str = ""
    ) -> ApprovalRecord:
        """
        Escalates an action or event for human review.
        This function represents a blocking call until a decision is made.
        """
        if self.interactive:
            decision = self._prompt_human(reason, context)
        else:
            decision = "n"

        if decision == "y":
            verdict_type = Verdict.ALLOW
            reviewer_reason = "Human explicitly approved."
        else:
            verdict_type = Verdict.BLOCK
            reviewer_reason = "Human rejected or timeout occurred."

        verdict = ApprovalVerdict(
            verdict=verdict_type, reason=reviewer_reason, reviewer="human_operator"
        )

        return ApprovalRecord(
            record_id=str(uuid.uuid4()),
            target_id=target_id,
            verdict=verdict,
            timestamp=time.time(),
        )

    def process_verdict(
        self,
        verdict_enum: Verdict,
        reason: str,
        target_id: str,
        *,
        action: Optional[ActionEvent] = None,
        mission: Optional[MissionContract] = None,
    ) -> ApprovalRecord:
        """
        Processes standard verdicts or triggers a human review if the verdict is REVIEW.
        """
        if verdict_enum == Verdict.REVIEW:
            if self._auto_allow_review(action, mission):
                verdict = ApprovalVerdict(
                    verdict=Verdict.ALLOW,
                    reason=f"[PERMISSIVE AUTO-ALLOW] {reason}",
                    reviewer="system_auto",
                )
                return ApprovalRecord(
                    record_id=str(uuid.uuid4()),
                    target_id=target_id,
                    verdict=verdict,
                    timestamp=time.time(),
                )
            # Headless: never pretend a human rejected. Permissive mode allows
            # non-tool reviews (e.g. session memory mid-band); tool reviews that
            # failed auto-allow stay blocked. Strict mode blocks pending review.
            if not self.interactive:
                if self.warn_policy == "permissive" and action is None:
                    verdict = ApprovalVerdict(
                        verdict=Verdict.ALLOW,
                        reason=f"[PERMISSIVE HEADLESS] {reason}",
                        reviewer="system_auto",
                    )
                else:
                    verdict = ApprovalVerdict(
                        verdict=Verdict.BLOCK,
                        reason=(
                            f"[HEADLESS] Review required; no human operator available. "
                            f"{reason}"
                        ),
                        reviewer="system_auto",
                    )
                return ApprovalRecord(
                    record_id=str(uuid.uuid4()),
                    target_id=target_id,
                    verdict=verdict,
                    timestamp=time.time(),
                )
            return self.request_approval(target_id, reason)

        if verdict_enum == Verdict.BLOCK and self.interactive:
            action_context = ""
            if action is not None:
                tool_name = action.payload.get("tool_name")
                if tool_name:
                    action_context = f"tool={tool_name}"
            return self.request_approval(target_id, reason, context=action_context)

        if verdict_enum == Verdict.WARN and action is not None:
            tool_name = str(action.payload.get("tool_name", ""))
            if (
                tool_name
                and mission is not None
                and (mission.local_only or mission.forbid_network_exfiltration)
                and is_high_risk_tool(tool_name)
                and not is_mission_local_tool(tool_name, mission)
            ):
                return self.request_approval(
                    target_id,
                    f"[HIGH-RISK WARNING] {reason}",
                    context=f"tool={tool_name}",
                )

        if verdict_enum == Verdict.WARN:
            reason = f"[ACTIONABLE WARNING] {reason}"

        verdict = ApprovalVerdict(
            verdict=verdict_enum, reason=reason, reviewer="system_auto"
        )

        return ApprovalRecord(
            record_id=str(uuid.uuid4()),
            target_id=target_id,
            verdict=verdict,
            timestamp=time.time(),
        )
