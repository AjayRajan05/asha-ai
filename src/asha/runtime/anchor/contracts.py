from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class MissionContract:
    """
    The immutable Mission Contract.
    This is the source of truth for all runtime decisions.
    It must not be mutated after creation.
    """

    mission_id: str
    goal: str
    intent_summary: str
    allowed_actions: List[str]
    forbidden_actions: List[str]
    allowed_tools: List[str]
    allowed_domains: List[str]
    allowed_memory_scopes: List[str]
    required_resources: List[str]
    expected_outcomes: List[str]
    completion_criteria: List[str]
    risk_tolerance: str
    created_at: float
    local_only: bool = False
    forbid_network_exfiltration: bool = False
    low_confidence: bool = False
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)
    forbidden_network_tokens: List[str] = field(default_factory=list)
