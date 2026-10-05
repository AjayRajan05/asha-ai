"""Plan and agent-output governance: detect narrated success without tool receipts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .verdicts import Verdict

_SUCCESS_CLAIMS = (
    r"\bemail(?:ed)?\s+(?:was\s+)?sent\b",
    r"\bupload(?:ed)?\s+(?:to|successfully)\b",
    r"\b(?:report|file)\s+(?:was\s+)?(?:saved|written)\b",
    r"\bsuccessfully\s+(?:sent|uploaded|saved|written)\b",
)

_RECEIPT_MARKERS = (
    "Email sent to",
    "Uploaded ",
    "Report written to",
    "Tool '",
    " was denied:",
    "ANCHOR blocked",
)


@dataclass(frozen=True)
class PlanVerdict:
    verdict: Verdict
    reason: str
    risk_score: float


def _has_success_claim(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _SUCCESS_CLAIMS)


def _has_tool_receipt(text: str) -> bool:
    return any(marker in text for marker in _RECEIPT_MARKERS)


def evaluate_plan_output(
    output: str,
    *,
    required_tool_markers: Sequence[str] = (),
    tools_used: Iterable[str] = (),
) -> PlanVerdict:
    """
    Detect when an agent claims mission completion without tool execution evidence.

    Returns BLOCK when success is narrated without receipts or required tools were
    never invoked.
    """
    text = (output or "").strip()
    if not text:
        return PlanVerdict(Verdict.ALLOW, "Empty plan output.", 0.0)

    used = {name.lower() for name in tools_used}
    missing_required = [
        marker
        for marker in required_tool_markers
        if marker.lower() not in used and marker.lower() not in text.lower()
    ]

    if missing_required and _has_success_claim(text) and not _has_tool_receipt(text):
        return PlanVerdict(
            Verdict.BLOCK,
            "Agent output claims mission success without required tool receipts: "
            + ", ".join(missing_required),
            1.0,
        )

    if _has_success_claim(text) and not _has_tool_receipt(text):
        return PlanVerdict(
            Verdict.REVIEW,
            "Agent output claims side-effects without quoting tool return messages.",
            0.7,
        )

    return PlanVerdict(Verdict.ALLOW, "Plan output acceptable.", 0.0)


def tools_used_from_history(
    action_types_and_payloads: Iterable[Tuple[str, dict]],
) -> List[str]:
    names: List[str] = []
    for action_type, payload in action_types_and_payloads:
        if action_type == "tool_call":
            name = str(payload.get("tool_name", "") or "")
            if name:
                names.append(name)
    return names
