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

"""Action enforcement gate: calibrated bands + hard-override ensemble."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict

from .alignment_bands import (
    calibrate_score,
    has_hard_block_trigger,
    score_to_verdict,
)
from .contracts import MissionContract
from .evaluator import AlignmentEvaluator
from .payload_inspection import is_high_risk_tool, is_mission_local_tool
from .tool_capabilities import get_tool_capabilities
from .types import ActionEvent
from .verdicts import ActionVerdict, Verdict

_DESTRUCTIVE_COMMAND_PATTERNS = (
    r"\brm\s+-(?:rf|fr|r)\b",
    r"\bdrop\s+table\b",
    r"\bdrop\s+database\b",
    r"\btruncate\s+table\b",
    r"\bformat\s+[a-z]:\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
)

LIFECYCLE_ACTIONS = frozenset(
    {
        "lifecycle",
        "start",
        "stop",
        "think",
        "thought",
        "message",
        "plan",
        "state_update",
        "finish",
    }
)


class ActionGuard:
    """Intercept, normalize, and evaluate actions against the Mission Contract."""

    def __init__(self, evaluator: AlignmentEvaluator) -> None:
        self.evaluator = evaluator
        self._calibrator = None
        self._calibrator_loaded = False

    def _get_calibrator(self) -> Any:
        if not self._calibrator_loaded:
            from .alignment_bands import load_isotonic_calibrator

            self._calibrator = load_isotonic_calibrator()
            self._calibrator_loaded = True
        return self._calibrator

    def normalize_action(
        self, action_type: str, payload: Dict[str, Any]
    ) -> ActionEvent:
        norm_payload = dict(payload) if isinstance(payload, dict) else {}
        if action_type not in LIFECYCLE_ACTIONS and action_type != "tool_call":
            if "tool_name" not in norm_payload:
                candidate = (
                    norm_payload.get("tool")
                    or norm_payload.get("name")
                    or norm_payload.get("function")
                    or action_type
                )
                norm_payload["tool_name"] = str(candidate)
            action_type = "tool_call"

        return ActionEvent(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            payload=norm_payload,
            timestamp=time.time(),
        )

    def _resolve_tool_name(self, action: ActionEvent) -> str:
        if action.payload and isinstance(action.payload, dict):
            for key in ("tool_name", "tool", "name", "function"):
                val = action.payload.get(key)
                if val:
                    return str(val)
        if action.action_type not in LIFECYCLE_ACTIONS and action.action_type != "tool_call":
            return action.action_type
        return ""

    def evaluate_action(
        self, action: ActionEvent, contract: MissionContract
    ) -> ActionVerdict:
        raw_score, explanation, triggers = self.evaluator.evaluate(action, contract)
        score = calibrate_score(raw_score, self._get_calibrator())
        verdict = score_to_verdict(score)
        risk_score = 1.0 - score

        # Ensemble hard overrides: independent signals force BLOCK even if
        # calibrated banding would allow a medium aggregate score through.
        if has_hard_block_trigger(triggers, explanation):
            verdict = Verdict.BLOCK
            risk_score = 1.0

        tool_name = self._resolve_tool_name(action)

        if action.action_type in contract.forbidden_actions:
            verdict = Verdict.BLOCK
            risk_score = 1.0
            explanation = f"Action '{action.action_type}' explicitly forbidden by mission contract."

        if tool_name and tool_name in contract.forbidden_actions:
            verdict = Verdict.BLOCK
            risk_score = 1.0
            explanation = f"Tool '{tool_name}' explicitly forbidden by mission contract."

        # Check payload for destructive command patterns
        if action.payload and isinstance(action.payload, dict):
            payload_text = json.dumps(action.payload)
            for pat in _DESTRUCTIVE_COMMAND_PATTERNS:
                if re.search(pat, payload_text, re.IGNORECASE):
                    verdict = Verdict.BLOCK
                    risk_score = 1.0
                    explanation = "Destructive command pattern detected in action payload."
                    triggers.append("destructive_command_pattern")
                    break

        if tool_name:
            if contract.local_only or contract.forbid_network_exfiltration:
                caps = get_tool_capabilities(tool_name)
                if (
                    is_high_risk_tool(tool_name)
                    or caps.network_egress
                    or caps.destructive
                ) and not is_mission_local_tool(tool_name, contract):
                    verdict = Verdict.BLOCK
                    risk_score = 1.0
                    explanation = (
                        f"High-risk tool '{tool_name}' requires human approval "
                        "before execution."
                    )

            # Resource-scope trigger already in evaluator; belt-and-suspenders:
            if any("resource scope violation" in t.lower() for t in triggers):
                verdict = Verdict.BLOCK
                risk_score = 1.0

        return ActionVerdict(
            verdict=verdict,
            reason=explanation,
            risk_score=risk_score,
            risk_triggers=tuple(triggers),
        )
