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

"""Structured telemetry engine for Anchor Runtime (opt-in)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

from .types import ActionEvent, ChainEvent, MemoryEvent, RiskSummary, AnchorState
from .verdicts import ActionVerdict, ChainVerdict, MemoryVerdict
from .plan_guard import PlanVerdict


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _clear_logger_handlers(logger: logging.Logger) -> None:
    """Drop cached handlers (logging keeps loggers alive after GC; ids can be reused)."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


class AnchorTelemetry:
    """
    Structured telemetry for Anchor Runtime.

    **Opt-in:** by default no file is created. Enable with
    ``ASHA_ANCHOR_TELEMETRY=1``, ``telemetry_path=...``, or ``enabled=True``.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        *,
        enabled: Optional[bool] = None,
    ):
        import pathlib

        env_path = os.environ.get("ASHA_ANCHOR_TELEMETRY_PATH", "").strip()
        if enabled is None:
            enabled = (
                _env_truthy("ASHA_ANCHOR_TELEMETRY") or bool(log_file) or bool(env_path)
            )
        self.enabled = bool(enabled)

        if log_file or env_path:
            resolved_path = log_file or env_path
        else:
            # Default to an absolute, user-scoped path to avoid cwd-relative
            # writes that can leak events across tenants on shared servers.
            resolved_path = str(
                pathlib.Path.home() / ".cache" / "asha" / "telemetry" / "anchor_telemetry.jsonl"
            )
            if self.enabled:
                logging.getLogger(__name__).warning(
                    "ASHA ANCHOR telemetry is enabled. Writing governance events to %s. "
                    "Set ASHA_ANCHOR_TELEMETRY_PATH to override or "
                    "unset ASHA_ANCHOR_TELEMETRY to disable.",
                    resolved_path,
                )

        self.log_file = resolved_path

        # Unique name per instance: never reuse id(self); logging caches loggers and
        # object ids can collide after GC, leaving stale FileHandlers attached.
        self.logger = logging.getLogger(f"AnchorTelemetry.{uuid.uuid4().hex}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        _clear_logger_handlers(self.logger)

        if self.enabled:
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def _emit(self, log_entry: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.logger.info(json.dumps(log_entry))
        for handler in self.logger.handlers:
            handler.flush()

    def _redact_payload(self, payload: Dict[str, Any]) -> Dict[str, str]:
        return {k: "[REDACTED]" for k in payload.keys()}

    def log_action_evaluated(self, action: ActionEvent, verdict: ActionVerdict) -> None:
        self._emit(
            {
                "event_type": "action_evaluated",
                "timestamp": action.timestamp,
                "action_id": action.action_id,
                "action_type": action.action_type,
                "payload_keys": self._redact_payload(action.payload),
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "risk_score": verdict.risk_score,
                "risk_triggers": list(verdict.risk_triggers),
            }
        )

    def log_memory_evaluated(self, event: MemoryEvent, verdict: MemoryVerdict) -> None:
        self._emit(
            {
                "event_type": "memory_evaluated",
                "timestamp": event.timestamp,
                "event_id": event.event_id,
                "operation": event.operation,
                "scope": event.scope,
                "content": "[REDACTED_FOR_TELEMETRY]",
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "risk_score": verdict.risk_score,
            }
        )

    def log_chain_evaluated(
        self, chain_event: ChainEvent, verdict: ChainVerdict
    ) -> None:
        self._emit(
            {
                "event_type": "chain_evaluated",
                "timestamp": chain_event.timestamp,
                "chain_id": chain_event.chain_id,
                "pattern_detected": chain_event.pattern_detected,
                "action_count": len(chain_event.actions),
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "risk_score": verdict.risk_score,
            }
        )

    def log_plan_evaluated(
        self, verdict: PlanVerdict, *, target_id: str, timestamp: float
    ) -> None:
        self._emit(
            {
                "event_type": "plan_evaluated",
                "timestamp": timestamp,
                "target_id": target_id,
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "risk_score": verdict.risk_score,
            }
        )

    def log_approval(
        self,
        record_id: str,
        target_id: str,
        verdict: str,
        reason: str,
        timestamp: float,
    ) -> None:
        self._emit(
            {
                "event_type": "approval_recorded",
                "timestamp": timestamp,
                "record_id": record_id,
                "target_id": target_id,
                "verdict": verdict,
                "reason": reason,
            }
        )

    def log_risk_summary(self, summary: RiskSummary, timestamp: float) -> None:
        self._emit(
            {
                "event_type": "risk_summary",
                "timestamp": timestamp,
                "alignment_score": summary.alignment_score,
                "memory_integrity_score": summary.memory_integrity_score,
                "drift_score": summary.drift_score,
                "total_risk_score": summary.total_risk_score,
                "severity": summary.severity.value,
                "explanation": summary.explanation,
            }
        )

    def log_session_finalized(
        self, summary: RiskSummary, state: AnchorState, timestamp: float
    ) -> None:
        self._emit(
            {
                "event_type": "session_finalized",
                "timestamp": timestamp,
                "mission_id": state.mission.mission_id if state.mission else None,
                "tools_used": list(state.tools_used),
                "action_count": len(state.actions_history),
                "memory_events": len(state.memory_history),
                "total_risk_score": summary.total_risk_score,
                "severity": summary.severity.value,
            }
        )
