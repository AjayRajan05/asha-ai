from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .verdicts import ApprovalVerdict, RiskSeverity
from .contracts import MissionContract


@dataclass(frozen=True)
class ActionEvent:
    action_id: str
    action_type: str  # tool_call, function_call, shell_command, network_request, database_write, file_write
    payload: Dict[str, Any]
    timestamp: float


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    operation: str  # read, write, update, delete
    content_summary: str
    scope: str
    timestamp: float


@dataclass(frozen=True)
class ChainEvent:
    chain_id: str
    actions: List[ActionEvent]
    pattern_detected: Optional[str]
    timestamp: float


@dataclass(frozen=True)
class RiskSummary:
    alignment_score: float
    memory_integrity_score: float
    drift_score: float
    total_risk_score: float  # 0 to 100
    severity: RiskSeverity
    explanation: str


@dataclass(frozen=True)
class ApprovalRecord:
    record_id: str
    target_id: str  # id of the action or memory event
    verdict: ApprovalVerdict
    timestamp: float


@dataclass
class AnchorState:
    mission: Optional[MissionContract] = None
    baseline_mission: Optional[MissionContract] = None
    actions_history: List[ActionEvent] = field(default_factory=list)
    memory_history: List[MemoryEvent] = field(default_factory=list)
    risk_history: List[RiskSummary] = field(default_factory=list)
    approval_history: List[ApprovalRecord] = field(default_factory=list)
    alignment_history: List[float] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
