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

"""Trace and event recording abstraction for ASHA Control Center."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Standardized lifecycle events for ASHA execution."""

    INPUT_RECEIVED = "INPUT_RECEIVED"
    PII_SCAN = "PII_SCAN"
    SANITIZATION_STARTED = "SANITIZATION_STARTED"
    SANITIZATION_COMPLETED = "SANITIZATION_COMPLETED"
    COMPILATION_STARTED = "COMPILATION_STARTED"
    COMPILATION_COMPLETED = "COMPILATION_COMPLETED"
    OPTIMIZATION_STARTED = "OPTIMIZATION_STARTED"
    OPTIMIZATION_CANDIDATE = "OPTIMIZATION_CANDIDATE"
    OPTIMIZATION_ACCEPTED = "OPTIMIZATION_ACCEPTED"
    OPTIMIZATION_REJECTED = "OPTIMIZATION_REJECTED"
    POLICY_CHECK = "POLICY_CHECK"
    ACTION_NORMALIZED = "ACTION_NORMALIZED"
    ACTION_ALLOWED = "ACTION_ALLOWED"
    ACTION_BLOCKED = "ACTION_BLOCKED"
    PROVIDER_REQUEST = "PROVIDER_REQUEST"
    PROVIDER_RESPONSE = "PROVIDER_RESPONSE"
    RESPONSE_VALIDATION = "RESPONSE_VALIDATION"
    RESULT_CREATED = "RESULT_CREATED"
    REQUEST_COMPLETE = "REQUEST_COMPLETE"
    ERROR = "ERROR"


def _safe_mask_secrets(data: Any) -> Any:
    """Recursively redact secrets and credentials from data dictionaries."""
    if isinstance(data, dict):
        safe_dict = {}
        for k, v in data.items():
            key_lower = str(k).lower()
            if any(secret_term in key_lower for secret_term in ("key", "secret", "password", "token", "credential", "auth")):
                if isinstance(v, str) and len(v) > 0:
                    safe_dict[k] = "[REDACTED_SECRET]"
                else:
                    safe_dict[k] = "[REDACTED_SECRET]"
            else:
                safe_dict[k] = _safe_mask_secrets(v)
        return safe_dict
    elif isinstance(data, (list, tuple)):
        return [_safe_mask_secrets(item) for item in data]
    return data


@dataclass
class TraceEvent:
    """Individual event in an ASHA execution trace."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    status: str = "SUCCESS"  # SUCCESS, WARNING, BLOCKED, ERROR, INFO
    details: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    token_count: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.details

    @metadata.setter
    def metadata(self, val: Dict[str, Any]) -> None:
        self.details = val

    def to_dict(self) -> Dict[str, Any]:
        ev_val = self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type)
        return {
            "event_type": ev_val,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "details": _safe_mask_secrets(self.details),
            "metadata": _safe_mask_secrets(self.details),
            "reason": self.reason,
            "token_count": self.token_count,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


@dataclass
class Trace:
    """Full execution trace for an ASHA pipeline execution."""

    trace_id: str = field(default_factory=lambda: f"TR-{uuid.uuid4().hex[:8].upper()}")
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    events: List[TraceEvent] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    input_text: str = ""
    output_text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_saved: int = 0
    pii_count: int = 0
    blocked_count: int = 0
    asha_latency_ms: float = 0.0
    provider_latency_ms: float = 0.0
    status: str = "PENDING"  # SUCCESS, BLOCKED, ERROR

    @property
    def total_latency_ms(self) -> float:
        if self.end_time and self.start_time:
            return max(0.0, (self.end_time - self.start_time) * 1000.0)
        total_ev_duration = sum(ev.duration_ms for ev in self.events if ev.duration_ms > 0)
        if total_ev_duration > 0:
            return total_ev_duration
        return self.asha_latency_ms + self.provider_latency_ms

    @property
    def duration_ms(self) -> float:
        return self.total_latency_ms

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.start_time).isoformat()

    def add_event(
        self,
        event_type: EventType,
        status: str = "SUCCESS",
        duration_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        token_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_type=event_type,
            timestamp=time.time(),
            duration_ms=duration_ms,
            status=status,
            details=metadata if metadata is not None else (details or {}),
            reason=reason,
            token_count=token_count,
            provider=self.provider,
            model=self.model,
            error=error,
        )
        self.events.append(event)

        # Update trace aggregate status
        if status == "BLOCKED":
            self.status = "BLOCKED"
            self.blocked_count += 1
        elif status == "ERROR":
            self.status = "ERROR"
        elif self.status == "PENDING" and status in ("SUCCESS", "INFO"):
            self.status = "SUCCESS"

        return event

    def record_event(
        self,
        event_type: EventType,
        status: str = "SUCCESS",
        duration_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        token_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> TraceEvent:
        return self.add_event(
            event_type=event_type,
            status=status,
            duration_ms=duration_ms,
            details=details,
            metadata=metadata,
            reason=reason,
            token_count=token_count,
            error=error,
        )

    def complete(self, status: str = "SUCCESS") -> None:
        self.end_time = time.time()
        self.status = status
        self.add_event(EventType.REQUEST_COMPLETE, status=status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "metrics": {
                "total_latency_ms": round(self.total_latency_ms, 2),
                "asha_latency_ms": round(self.asha_latency_ms, 2),
                "provider_latency_ms": round(self.provider_latency_ms, 2),
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tokens_saved": self.tokens_saved,
                "pii_count": self.pii_count,
                "blocked_count": self.blocked_count,
            },
            "events": [event.to_dict() for event in self.events],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
