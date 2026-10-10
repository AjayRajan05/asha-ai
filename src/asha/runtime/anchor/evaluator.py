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

"""Mission-alignment scoring for ANCHOR actions.

Hard-block signals (forbidden action, resource-scope violation, high-risk
tool under local-only) force score → 0.0 so no combination of medium-confidence
penalties can average up into a false ALLOW. Soft multiplicative penalties
remain for non-hard policy mismatches.
"""

from __future__ import annotations

from typing import List, Tuple

from .contracts import MissionContract
from .types import ActionEvent
from .payload_inspection import (
    extract_inspection_metadata,
    find_forbidden_metadata_matches,
    format_forbidden_matches,
    is_high_risk_tool,
    is_mission_local_tool,
    tool_name_policy_violations,
    validate_resource_scope,
)
from .tool_capabilities import get_tool_capabilities


class AlignmentEvaluator:
    """Evaluate actions against the Mission Contract → alignment score in [0, 1]."""

    def evaluate(
        self, action: ActionEvent, contract: MissionContract
    ) -> Tuple[float, str, List[str]]:
        score = 1.0
        explanations: List[str] = []
        triggers: List[str] = []
        hard_block = False

        action_type = action.action_type
        payload = action.payload if isinstance(action.payload, dict) else {}
        tool_name = str(
            payload.get("tool_name")
            or payload.get("tool")
            or payload.get("name")
            or payload.get("function")
            or ""
        )
        if not tool_name and action_type not in frozenset(
            {"lifecycle", "start", "stop", "think", "thought", "message", "plan", "state_update", "finish"}
        ):
            tool_name = action_type
            action_type = "tool_call"

        def _hard(msg: str) -> None:
            nonlocal score, hard_block
            hard_block = True
            score = 0.0
            explanations.append(msg)
            triggers.append(msg)

        # --- Hard overrides (independent signals; cannot be averaged away) ---

        # Forbidden action type
        if action_type in contract.forbidden_actions:
            _hard(f"Action '{action_type}' is explicitly forbidden.")

        # High-risk / network-egress under local-only (name helpers + capabilities)
        if action_type == "tool_call" and tool_name and not hard_block:
            caps = get_tool_capabilities(tool_name)
            local_restricted = (
                contract.local_only or contract.forbid_network_exfiltration
            )
            high_risk = (
                is_high_risk_tool(tool_name) or caps.network_egress or caps.destructive
            )
            if (
                local_restricted
                and high_risk
                and not is_mission_local_tool(tool_name, contract)
            ):
                _hard(
                    f"Tool '{tool_name}' performs external or destructive side-effects "
                    "and is blocked under the current mission scope."
                )

            name_violations = tool_name_policy_violations(
                tool_name, contract.forbidden_actions
            )
            if name_violations and not is_mission_local_tool(tool_name, contract):
                detail = ", ".join(name_violations)
                _hard(
                    f"Tool '{tool_name}' matches forbidden capability token(s): {detail}."
                )

            if tool_name in contract.forbidden_actions:
                _hard(f"Tool '{tool_name}' is explicitly forbidden.")

        # Allowed-tools allowlist miss → hard block for tool_call
        if (
            not hard_block
            and contract.allowed_tools
            and action_type == "tool_call"
            and tool_name
            and tool_name not in contract.allowed_tools
        ):
            _hard(f"Tool '{tool_name}' is not in allowed tools list.")

        metadata = extract_inspection_metadata(action.payload)

        # Resource-scope violation → hard block
        if not hard_block and action_type == "tool_call" and tool_name:
            scope_violation = validate_resource_scope(tool_name, metadata, contract)
            if scope_violation is not None:
                _hard(
                    f"Resource scope violation: {scope_violation.term} on "
                    f"{scope_violation.field}='{scope_violation.value}'"
                )

        if hard_block:
            explanation = " | ".join(explanations)
            return 0.0, explanation, triggers

        # --- Soft signals (multiplicative; cannot alone force ALLOW over hard) ---

        if contract.allowed_tools and action_type not in contract.allowed_tools:
            if action_type != "tool_call":
                score *= 0.8
                explanations.append(
                    f"Action '{action_type}' is not recognized as an allowed tool."
                )

        if action_type == "tool_call":
            forbidden_matches = find_forbidden_metadata_matches(
                metadata,
                contract.forbidden_actions,
                network_only_tokens=(
                    contract.forbidden_network_tokens
                    if contract.forbid_network_exfiltration
                    else None
                ),
            )
        else:
            forbidden_matches = find_forbidden_metadata_matches(
                {"payload": str(action.payload)},
                contract.forbidden_actions,
            )

        if forbidden_matches:
            # Soft penalty only when not already a hard capability match.
            score *= 0.3 ** len(forbidden_matches)
            details = format_forbidden_matches(forbidden_matches)
            explanations.append("Metadata policy violations: " + "; ".join(details))
            triggers.extend(details)

        if any(
            keyword in action_type for keyword in ["write", "update", "delete", "send"]
        ):
            if contract.risk_tolerance == "LOW":
                score *= 0.7
                explanations.append(
                    "State-mutating action encountered under LOW risk tolerance."
                )
        elif (
            action_type == "tool_call"
            and tool_name
            and "write" in tool_name.lower()
            and tool_name in contract.allowed_tools
            and "write" in contract.allowed_actions
        ):
            pass
        elif (
            action_type == "tool_call"
            and tool_name
            and any(
                keyword in tool_name.lower()
                for keyword in ["write", "update", "delete", "send"]
            )
            and contract.risk_tolerance == "LOW"
        ):
            score *= 0.7
            explanations.append(
                "State-mutating tool encountered under LOW risk tolerance."
            )

        explanation = (
            " | ".join(explanations)
            if explanations
            else "Action aligns with mission parameters."
        )
        return max(0.0, min(1.0, score)), explanation, triggers
