"""Structured metadata extraction for ANCHOR action payload inspection."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import MissionContract

METADATA_KEYS = frozenset(
    {
        "tool_name",
        "path",
        "file_path",
        "filepath",
        "url",
        "to",
        "email",
        "host",
        "endpoint",
        "destination",
        "filename",
        "output_path",
    }
)

NETWORK_METADATA_KEYS = frozenset(
    {"url", "host", "endpoint", "destination", "to", "email"}
)

_WRITE_TOOL_HINTS = ("write", "save", "persist", "export")
_READ_TOOL_HINTS = ("read", "load", "fetch", "get")

_HIGH_RISK_TOOL_NAMES = frozenset(
    {
        "send_email",
        "upload_to_cloud",
        "send_network_request",
        "delete_local_file",
        "network_exfil",
    }
)

_RISKY_TOOL_SUBSTRINGS = (
    "send_",
    "upload_",
    "delete_",
    "network_",
    "exfil",
    "_email",
    "smtp",
    "http_request",
    "bash",
    "shell",
    "exec",
    "terminal",
    "command",
)


def is_high_risk_tool(tool_name: str) -> bool:
    """Tools that perform external side-effects and require human review."""
    lowered = tool_name.lower()
    if lowered in _HIGH_RISK_TOOL_NAMES:
        return True
    return any(fragment in lowered for fragment in _RISKY_TOOL_SUBSTRINGS)


def is_mission_local_tool(tool_name: str, contract: MissionContract) -> bool:
    """Allowlisted local read/write tools that stay within mission scope.

    Network-egress and destructive tools are never "local" even if allowlisted;
    they still require hard-block / approval under local_only missions.
    """
    if tool_name not in contract.allowed_tools:
        return False
    from .tool_capabilities import get_tool_capabilities

    caps = get_tool_capabilities(tool_name)
    if caps.network_egress or caps.destructive or is_high_risk_tool(tool_name):
        return False
    if is_read_tool(tool_name):
        return True
    if is_write_tool(tool_name) and "write" in contract.allowed_actions:
        return True
    return False


def tool_name_policy_violations(
    tool_name: str,
    forbidden_actions: Iterable[str],
) -> List[str]:
    """Match forbidden capability tokens against tool names (substring, not prose)."""
    lowered = tool_name.lower()
    violations: List[str] = []
    for forbidden in forbidden_actions:
        token = str(forbidden).lower()
        if token and token in lowered:
            violations.append(str(forbidden))
    return violations


@dataclass(frozen=True)
class ForbiddenMatch:
    """A forbidden policy match with field-level detail for telemetry."""

    term: str
    field: str
    value: str


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Best-effort parse of tool argument blobs from adapters."""
    if isinstance(arguments, dict):
        return dict(arguments)

    if not isinstance(arguments, str):
        return {}

    text = arguments.strip()
    if not text:
        return {}

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    return {}


def flatten_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Normalize CrewAI-style {'args': ..., 'kwargs': ...} argument envelopes."""
    parsed = parse_tool_arguments(arguments)
    if not parsed:
        return {}

    flattened: Dict[str, Any] = {}
    kwargs = parsed.get("kwargs")
    if isinstance(kwargs, dict):
        flattened.update(kwargs)

    args = parsed.get("args")
    if isinstance(args, dict):
        flattened.update(args)
    elif isinstance(args, (list, tuple)):
        for index, value in enumerate(args):
            flattened[f"arg_{index}"] = value

    for key, value in parsed.items():
        if key not in {"args", "kwargs"}:
            flattened.setdefault(key, value)

    return flattened


def extract_inspection_metadata(payload: Dict[str, Any]) -> Dict[str, str]:
    """Collect only security-relevant metadata fields from a tool payload."""
    metadata: Dict[str, str] = {}

    tool_name = payload.get("tool_name")
    if tool_name is not None:
        metadata["tool_name"] = str(tool_name)

    flattened = flatten_tool_arguments(payload.get("arguments", ""))
    for key, value in flattened.items():
        key_lower = str(key).lower()
        if key_lower in METADATA_KEYS or key_lower.endswith("_path"):
            metadata[key_lower] = str(value)

    return metadata


def metadata_inspection_text(metadata: Dict[str, str]) -> str:
    return " ".join(metadata.values()).lower()


def is_write_tool(tool_name: str) -> bool:
    lowered = tool_name.lower()
    return any(hint in lowered for hint in _WRITE_TOOL_HINTS)


def is_read_tool(tool_name: str) -> bool:
    lowered = tool_name.lower()
    return any(hint in lowered for hint in _READ_TOOL_HINTS)


def is_content_exempt_tool(tool_name: str, allowed_tools: Iterable[str]) -> bool:
    """Allowlisted write tools may carry large benign document bodies."""
    if tool_name not in allowed_tools:
        return False
    return is_write_tool(tool_name)


def contains_forbidden_term(text: str, forbidden: str) -> bool:
    """Match forbidden tokens on word boundaries to reduce substring false positives."""
    if not text or not forbidden:
        return False
    pattern = rf"\b{re.escape(forbidden.lower())}\b"
    return re.search(pattern, text) is not None


def _path_is_within_scope(path: str, allowed_prefixes: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/").strip().lower()
    if not allowed_prefixes:
        return True
    return any(
        normalized == prefix.rstrip("/")
        or normalized.startswith(prefix if prefix.endswith("/") else f"{prefix}/")
        for prefix in allowed_prefixes
    )


def validate_resource_scope(
    tool_name: str,
    metadata: Dict[str, str],
    contract: MissionContract,
) -> Optional[ForbiddenMatch]:
    """Validate read/write paths against mission resource scopes."""
    path_fields = [
        key
        for key in metadata
        if key.endswith("path")
        or key in {"path", "filepath", "file_path", "filename", "output_path"}
    ]

    for field in path_fields:
        value = metadata.get(field, "")
        if not value:
            continue

        if is_write_tool(tool_name) and contract.allowed_write_paths:
            if not _path_is_within_scope(value, contract.allowed_write_paths):
                return ForbiddenMatch(
                    term="write_scope",
                    field=field,
                    value=value,
                )

        if is_read_tool(tool_name) and contract.allowed_read_paths:
            if not _path_is_within_scope(value, contract.allowed_read_paths):
                return ForbiddenMatch(
                    term="read_scope",
                    field=field,
                    value=value,
                )

    return None


def find_forbidden_metadata_matches(
    metadata: Dict[str, str],
    forbidden_actions: Iterable[str],
    *,
    network_only_tokens: Optional[Iterable[str]] = None,
) -> List[ForbiddenMatch]:
    """Return forbidden matches with field-level detail."""
    matches: List[ForbiddenMatch] = []
    network_tokens = {token.lower() for token in (network_only_tokens or [])}

    for field, value in metadata.items():
        if field == "tool_name":
            continue

        value_lower = value.lower()
        tokens_to_check = forbidden_actions
        if field in NETWORK_METADATA_KEYS and network_tokens:
            tokens_to_check = network_tokens
        elif network_tokens and field not in NETWORK_METADATA_KEYS:
            tokens_to_check = [
                t for t in forbidden_actions if t.lower() not in network_tokens
            ]

        for forbidden in tokens_to_check:
            if contains_forbidden_term(value_lower, forbidden):
                matches.append(
                    ForbiddenMatch(term=str(forbidden), field=field, value=value)
                )

    return matches


def format_forbidden_matches(matches: List[ForbiddenMatch]) -> List[str]:
    return [f"{match.term} in {match.field}='{match.value}'" for match in matches]


# Explicit capability registration API (preferred over name-prefix inference).
from .tool_capabilities import (  # noqa: E402
    ToolCapabilities,
    categorize_tool,
    get_tool_capabilities,
    register_tool,
    register_tool_capabilities,
)

__all__ = [
    "METADATA_KEYS",
    "NETWORK_METADATA_KEYS",
    "ForbiddenMatch",
    "ToolCapabilities",
    "categorize_tool",
    "contains_forbidden_term",
    "format_forbidden_matches",
    "get_tool_capabilities",
    "is_content_exempt_tool",
    "is_high_risk_tool",
    "is_mission_local_tool",
    "is_read_tool",
    "is_write_tool",
    "parse_tool_arguments",
    "register_tool",
    "register_tool_capabilities",
    "tool_name_policy_violations",
    "validate_resource_scope",
]
