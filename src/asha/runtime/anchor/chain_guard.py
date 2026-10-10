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

"""Chain analysis over tool *categories* (Markov rarity), not name prefixes.

Capability metadata is declared via ``tool_capabilities.register_tool``.
Legacy name-prefix patterns remain only as a secondary corroborating signal.
"""

from __future__ import annotations

import uuid
import time
from typing import Any, List, Optional

from .verdicts import ChainVerdict, Verdict
from .types import ActionEvent, ChainEvent
from .contracts import MissionContract
from .payload_inspection import is_high_risk_tool, is_read_tool
from .tool_capabilities import categorize_tool, get_tool_capabilities
from .transition_model import score_category_sequence

from asha.core.ml.calibration import Verdict as MLVerdict


class ChainGuard:
    """Chain analysis engine for multi-step action sequences."""

    def __init__(self, *, use_markov: bool = True) -> None:
        self.use_markov = use_markov
        self._last_score: Optional[dict[str, Any]] = None

    def normalize_chain(
        self, actions: List[ActionEvent], pattern: Optional[str] = None
    ) -> ChainEvent:
        return ChainEvent(
            chain_id=str(uuid.uuid4()),
            actions=actions,
            pattern_detected=pattern,
            timestamp=time.time(),
        )

    @staticmethod
    def _tool_names(history: List[ActionEvent]) -> List[str]:
        names: List[str] = []
        for action in history:
            if action.action_type != "tool_call":
                continue
            name = str(action.payload.get("tool_name", "") or "")
            if name:
                names.append(name)
        return names

    @staticmethod
    def _categories(tool_names: List[str]) -> List[str]:
        return [categorize_tool(name) for name in tool_names]

    def evaluate_chain(
        self, history: List[ActionEvent], contract: MissionContract
    ) -> ChainVerdict:
        if len(history) < 2:
            return ChainVerdict(
                Verdict.ALLOW, "Insufficient history for chain analysis.", 0.0
            )

        tool_names = self._tool_names(history)
        categories = self._categories(tool_names)

        # --- Primary: Markov rarity over capability categories ---
        if self.use_markov and len(categories) >= 2:
            score = score_category_sequence(categories)
            self._last_score = score
            verdict_ml = score["verdict"]
            rarest = score.get("rarest")
            min_p = float(score["min_transition_probability"])

            if verdict_ml is MLVerdict.BLOCK:
                reason = "Anomalous tool-category transition"
                if rarest:
                    reason = (
                        f"Anomalous transition {rarest[0]}→{rarest[1]} "
                        f"(p={rarest[2]:.4f})"
                    )
                v = Verdict.BLOCK
                risk = 1.0
                if contract.risk_tolerance in ("HIGH", "CRITICAL"):
                    v = Verdict.REVIEW
                    risk = 0.85
                return ChainVerdict(v, reason, risk)

            if verdict_ml is MLVerdict.REVIEW:
                reason = "Unusual tool-category sequence: review required"
                if rarest:
                    reason = (
                        f"Unusual transition {rarest[0]}→{rarest[1]} "
                        f"(p={min_p:.4f}): review required"
                    )
                return ChainVerdict(Verdict.REVIEW, reason, max(0.6, 1.0 - min_p))

        # --- Capability hard rules (explicit flags, not name prefixes) ---
        if len(tool_names) >= 2:
            caps_seq = [get_tool_capabilities(n) for n in tool_names]
            earlier_read = any(c.reads_data for c in caps_seq[:-1])
            last = caps_seq[-1]
            if earlier_read and last.network_egress:
                if contract.local_only or contract.forbid_network_exfiltration:
                    return ChainVerdict(
                        Verdict.BLOCK,
                        "Data-read capability followed by network-egress tool.",
                        1.0,
                    )
            if any(c.writes_data for c in caps_seq[:-1]) and last.destructive:
                return ChainVerdict(
                    Verdict.BLOCK,
                    "Write capability followed by destructive tool.",
                    1.0,
                )

        # Local-only + high-risk (legacy helper, still useful)
        if contract.local_only and len(tool_names) >= 2:
            if any(
                is_read_tool(name) for name in tool_names[:-1]
            ) and is_high_risk_tool(tool_names[-1]):
                return ChainVerdict(
                    Verdict.BLOCK,
                    "Local data access followed by high-risk external tool.",
                    1.0,
                )

        return ChainVerdict(Verdict.ALLOW, "No malicious chains detected.", 0.0)
