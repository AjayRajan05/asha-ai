"""Mission-drift quality gate: clear BLOCK fixtures must never ALLOW."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asha.runtime.anchor.runtime import AnchorRuntime
from asha.runtime.anchor.tool_capabilities import register_tool

_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "anchor" / "mission_drift.jsonl"
)

# Ensure capability metadata matches fixture tool names for chain/action guards.
_CAP_HINTS = {
    "send_email": {"network_egress": True, "category": "network_egress"},
    "upload_to_cloud": {"network_egress": True, "category": "network_egress"},
    "delete_local_file": {
        "destructive": True,
        "writes_data": True,
        "category": "delete",
    },
    "http_request": {"network_egress": True, "category": "network_egress"},
    "run_shell": {"destructive": True, "category": "shell"},
    "post_webhook": {"network_egress": True, "category": "network_egress"},
}


def _load_cases() -> list[dict]:
    rows: list[dict] = []
    for line in _FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_mission_drift_block_never_allows(case: dict) -> None:
    assert case["expected"] == "BLOCK"
    tool = case["tool_name"]
    if tool in _CAP_HINTS:
        register_tool(tool, **_CAP_HINTS[tool])

    runtime = AnchorRuntime(
        warn_policy="strict", interactive=False, risk_tolerance="LOW"
    )
    runtime.initialize_mission(
        case["mission"],
        context={
            "available_tools": case["available_tools"],
            "local_only": case.get("local_only", True),
        },
    )
    allowed = runtime.evaluate_action_request(
        case.get("action_type", "tool_call"),
        {"tool_name": tool, "arguments": case.get("arguments", "{}")},
        raise_on_block=False,
    )
    assert allowed is False, (
        f"false-ALLOW on mission-drift fixture {case['id']!r}: "
        f"tool={tool!r} must be blocked under the mission"
    )


def test_mission_drift_fixture_nonempty() -> None:
    cases = _load_cases()
    assert len(cases) >= 5
    assert all(c["expected"] == "BLOCK" for c in cases)
