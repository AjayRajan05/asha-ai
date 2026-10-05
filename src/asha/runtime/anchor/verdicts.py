from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ActionVerdict:
    verdict: Verdict
    reason: str
    risk_score: float
    risk_triggers: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryVerdict:
    verdict: Verdict
    reason: str
    risk_score: float


@dataclass(frozen=True)
class ChainVerdict:
    verdict: Verdict
    reason: str
    risk_score: float


@dataclass(frozen=True)
class ApprovalVerdict:
    verdict: Verdict
    reason: str
    reviewer: Optional[str] = None
