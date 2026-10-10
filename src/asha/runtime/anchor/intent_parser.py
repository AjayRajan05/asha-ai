"""Explicit mission intent parsing for ANCHOR."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .payload_inspection import is_write_tool

_PATH_PATTERN = re.compile(
    r"(?P<path>(?:output|data)/[\w./_-]+)",
    re.IGNORECASE,
)

_LOCAL_ONLY_PHRASES = (
    "do not send",
    "not send data",
    "locally",
    "local file",
    "local data",
    "externally",
    "without network",
)

_WRITE_PHRASES = (
    "write",
    "save",
    "persist",
    "output/",
    "to a text file",
    "to file",
)

_NETWORK_FORBIDDEN_TOKENS = (
    "network_request",
    "upload",
    "http",
    "https",
    "ftp",
    "smtp",
)


@dataclass(frozen=True)
class ParsedMissionIntent:
    """Structured intents extracted from a natural-language mission prompt."""

    local_only: bool
    forbid_network_exfiltration: bool
    write_requested: bool
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)
    forbidden_network_tokens: List[str] = field(default_factory=list)


def _normalize_path(path: str) -> str:
    cleaned = path.replace("\\", "/").strip().lower()
    return cleaned.rstrip(".,;:!?")


def _extract_paths(prompt: str) -> List[str]:
    return list(
        dict.fromkeys(_normalize_path(match) for match in _PATH_PATTERN.findall(prompt))
    )


def parse_mission_intent(
    prompt: str, context: Dict[str, Any] | None = None
) -> ParsedMissionIntent:
    """
    Parse explicit mission intents from prompt text.

    Distinguishes policy constraints (e.g. forbid network exfiltration) from
    incidental words that may appear in report prose.
    """
    if context is None:
        context = {}

    prompt_lower = prompt.lower()
    allowed_tools = list(context.get("available_tools", []))

    local_only = bool(
        context.get("local_only")
        or any(phrase in prompt_lower for phrase in _LOCAL_ONLY_PHRASES)
    )
    forbid_network_exfiltration = local_only or any(
        phrase in prompt_lower
        for phrase in ("do not send", "not send data", "externally")
    )

    write_requested = any(phrase in prompt_lower for phrase in _WRITE_PHRASES) or any(
        is_write_tool(tool) for tool in allowed_tools
    )

    extracted_paths = _extract_paths(prompt)
    context_read_paths = [
        _normalize_path(p) for p in context.get("allowed_read_paths", [])
    ]
    context_write_paths = [
        _normalize_path(p) for p in context.get("allowed_write_paths", [])
    ]

    read_paths = list(
        dict.fromkeys(
            [p for p in extracted_paths if p.startswith("data/")] + context_read_paths
        )
    )
    write_paths = list(
        dict.fromkeys(
            [p for p in extracted_paths if p.startswith("output/")]
            + context_write_paths
        )
    )

    if write_requested and not write_paths:
        write_paths = ["output/"]

    if not read_paths and any(
        token in prompt_lower for token in ("load", "read", "dataset", "local data")
    ):
        read_paths = ["data/"]

    expected_outcomes: List[str] = list(context.get("expected_outcomes", []))
    completion_criteria: List[str] = list(
        context.get("completion_criteria", ["Goal explicitly satisfied"])
    )

    for path in write_paths:
        if path.endswith(".txt") or path.endswith(".json") or path.endswith(".md"):
            outcome = f"File written to {path}"
            if outcome not in expected_outcomes:
                expected_outcomes.append(outcome)
            criterion = f"Confirm {path} exists with mission deliverable"
            if criterion not in completion_criteria:
                completion_criteria.append(criterion)

    if (
        write_requested
        and "Deliverable written to local storage" not in expected_outcomes
    ):
        expected_outcomes.append("Deliverable written to local storage")

    forbidden_network_tokens = list(_NETWORK_FORBIDDEN_TOKENS)
    if forbid_network_exfiltration:
        forbidden_network_tokens.extend(["send network request", "network_request"])

    return ParsedMissionIntent(
        local_only=local_only,
        forbid_network_exfiltration=forbid_network_exfiltration,
        write_requested=write_requested,
        allowed_read_paths=read_paths,
        allowed_write_paths=write_paths,
        expected_outcomes=expected_outcomes,
        completion_criteria=completion_criteria,
        forbidden_network_tokens=list(dict.fromkeys(forbidden_network_tokens)),
    )
