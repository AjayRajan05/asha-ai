# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Typed dataclass results for process(), sanitize(), and optimize()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ThreatLevelStr(str):
    """Case-insensitive string representation of threat levels."""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.lower() == other.lower()
        if hasattr(other, "value"):
            return self.lower() == str(other.value).lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.lower())


@dataclass(frozen=True)
class SecurityInfo:
    """Security analysis attached to a processing result."""

    safe: bool
    pii_detected: List[str]
    threats: List[Any]
    masked_entities: Dict[str, Any]
    masking_map: Optional[Dict[str, str]] = None
    threat_level: str = "LOW"
    security_score: float = 0.0

    @property
    def is_safe(self) -> bool:
        """Alias for safe (API consistency)."""
        return self.safe

    @property
    def pii_detected_count(self) -> int:
        """Total count of PII entities detected."""
        return len(self.pii_detected)

    @property
    def threats_detected(self) -> int:
        """Total count of threats detected."""
        return len(self.threats)

    @property
    def pii_masked(self) -> int:
        """Total count of PII entities masked."""
        return len(self.masked_entities)

    @classmethod
    def from_summary(
        cls, summary: Optional[Dict[str, Any]]
    ) -> Optional["SecurityInfo"]:
        if not summary:
            return None
        threats = summary.get("threats") or []
        raw_level = summary.get("threat_level", "LOW")
        return cls(
            safe=bool(summary.get("is_safe", True)),
            pii_detected=list(summary.get("pii_detected") or []),
            threats=list(threats),
            masked_entities=dict(summary.get("masked_entities") or {}),
            masking_map=summary.get("masking_map"),
            threat_level=ThreatLevelStr(str(raw_level)),
            security_score=float(summary.get("security_score", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "is_safe": self.safe,
            "pii_detected": self.pii_detected,
            "threats": self.threats,
            "masked_entities": self.masked_entities,
            "threat_level": str(self.threat_level),
            "security_score": self.security_score,
            "threats_detected": len(self.threats),
            "pii_masked": len(self.masked_entities),
        }
        if self.masking_map:
            out["masking_map"] = self.masking_map
        return out


@dataclass(frozen=True)
class MetricsInfo:
    """Token and performance metrics."""

    tokens_saved: int
    token_reduction_pct: float
    processing_time_ms: float
    cost_reduction: Optional[str] = None
    pii_detected: List[str] = field(default_factory=list)
    risk_level: str = "low"
    threats_blocked: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "tokens_saved": self.tokens_saved,
            "token_reduction_percentage": self.token_reduction_pct,
            "processing_time_ms": self.processing_time_ms,
            "cost_reduction": self.cost_reduction or "0%",
            "pii_detected": self.pii_detected,
            "risk_level": self.risk_level,
            "threats_blocked": self.threats_blocked,
        }
        base.update(self.extra)
        return base


@dataclass(frozen=True)
class ProcessResult:
    """Result of process()."""

    output: str
    original: str
    degraded: bool
    degraded_reason: Optional[str]
    privacy_applied: bool
    security: Optional[SecurityInfo]
    metrics: Optional[MetricsInfo]
    trace: Optional[Dict[str, Any]] = None
    diff: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    legacy_detail: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.output

    @property
    def is_safe(self) -> bool:
        """Convenience property indicating whether prompt passed security checks."""
        return self.security.safe if self.security else True

    @property
    def security_result(self) -> Optional[SecurityInfo]:
        """Convenience alias for security attribute."""
        return self.security

    def to_dict(self) -> Dict[str, Any]:
        """Legacy dict shape for return_dict=True shim."""
        out: Dict[str, Any] = {
            "optimized": self.output,
            "original": self.original,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "privacy_applied": self.privacy_applied,
            "token_reduction": (
                self.metrics.token_reduction_pct if self.metrics else 0
            ),
        }
        if self.security:
            out["security_result"] = self.security.to_dict()
        if self.metrics:
            out["metrics"] = self.metrics.to_dict()
        if self.trace:
            out["trace"] = self.trace
        if self.diff:
            out["diff"] = self.diff
        if self.debug:
            out["debug"] = self.debug
        if self.security and self.security.masking_map:
            out["masking_map"] = self.security.masking_map
        return out


@dataclass(frozen=True)
class SanitizeResult:
    """Result of sanitize()."""

    output: str
    original: str
    safe: bool
    degraded: bool
    degraded_reason: Optional[str]
    security: SecurityInfo

    def __str__(self) -> str:
        return self.output

    @property
    def is_safe(self) -> bool:
        """Alias for safe (API consistency)."""
        return self.safe

    @property
    def security_result(self) -> SecurityInfo:
        """Convenience alias for security attribute."""
        return self.security

    def to_dict(self) -> Dict[str, Any]:
        """Legacy dict shape for return_details=True shim."""
        return {
            "sanitized": self.output,
            "original": self.original,
            "is_safe": self.safe,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "pii_detected": self.security.pii_detected,
            "threats": self.security.threats,
            "masked_entities": self.security.masked_entities,
            **(
                {"masking_map": self.security.masking_map}
                if self.security.masking_map
                else {}
            ),
        }


@dataclass(frozen=True)
class AgentResult:
    """Result of Agent.run(trace=True)."""

    output: str
    original: str
    response: Any
    degraded: bool
    degraded_reason: Optional[str]
    privacy_applied: bool
    security: Optional[SecurityInfo]
    metrics: Optional[MetricsInfo]
    trace: Optional[Dict[str, Any]] = None
    adapter_info: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None

    def __str__(self) -> str:
        return str(self.response)

    @property
    def process_result(self) -> ProcessResult:
        """Compatibility accessor - prefer top-level fields on AgentResult."""
        return ProcessResult(
            output=self.output,
            original=self.original,
            degraded=self.degraded,
            degraded_reason=self.degraded_reason,
            privacy_applied=self.privacy_applied,
            security=self.security,
            metrics=self.metrics,
            trace=self.trace,
        )

    @classmethod
    def from_process(
        cls,
        proc: "ProcessResult",
        response: Any,
        *,
        adapter_info: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        extra_trace: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        return cls(
            output=proc.output,
            original=proc.original,
            response=response,
            degraded=proc.degraded,
            degraded_reason=proc.degraded_reason,
            privacy_applied=proc.privacy_applied,
            security=proc.security,
            metrics=proc.metrics,
            trace=extra_trace or proc.trace,
            adapter_info=adapter_info,
            mode=mode,
        )


@dataclass(frozen=True)
class OptimizeResult:
    """Result of optimize() - token compression only."""

    output: str
    original: str
    degraded: bool
    degraded_reason: Optional[str]
    metrics: Optional[MetricsInfo]

    def __str__(self) -> str:
        return self.output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimized": self.output,
            "original": self.original,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "token_reduction": (
                self.metrics.token_reduction_pct if self.metrics else 0
            ),
            "metrics": self.metrics.to_dict() if self.metrics else {},
        }
