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

"""Explicit tool capability metadata for ANCHOR guards.

Tools should declare capabilities at registration time so ChainGuard can
score category transitions without inferring intent from function-name
prefixes. Unregistered tools fall back to conservative name heuristics.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ToolCapabilities:
    """Structured capability flags declared at tool registration."""

    reads_data: bool = False
    writes_data: bool = False
    network_egress: bool = False
    destructive: bool = False
    # Primary Markov category (read/write/analyze/notify/verify/payment/delete/network_egress)
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# Categories aligned with asha_training_data_generators chain_guard model.
KNOWN_CATEGORIES = (
    "read",
    "write",
    "analyze",
    "notify",
    "verify",
    "payment",
    "delete",
    "network_egress",
)

_REGISTRY: Dict[str, ToolCapabilities] = {}
_LOCK = threading.Lock()

# Seed well-known ASHA tools so tests/runtime work before explicit registration.
_DEFAULT_CAPS: Dict[str, ToolCapabilities] = {
    "load_trend_data": ToolCapabilities(reads_data=True, category="read"),
    "read_file": ToolCapabilities(reads_data=True, category="read"),
    "fetch_url": ToolCapabilities(
        reads_data=True, network_egress=True, category="read"
    ),
    "write_trend_report": ToolCapabilities(writes_data=True, category="write"),
    "write_file": ToolCapabilities(writes_data=True, category="write"),
    "analyze_trends": ToolCapabilities(category="analyze"),
    "send_email": ToolCapabilities(network_egress=True, category="network_egress"),
    "send_bulk_email": ToolCapabilities(network_egress=True, category="network_egress"),
    "upload_to_cloud": ToolCapabilities(
        network_egress=True, writes_data=True, category="network_egress"
    ),
    "send_network_request": ToolCapabilities(
        network_egress=True, category="network_egress"
    ),
    "network_exfil": ToolCapabilities(network_egress=True, category="network_egress"),
    "network_upload": ToolCapabilities(network_egress=True, category="network_egress"),
    "delete_local_file": ToolCapabilities(
        destructive=True, writes_data=True, category="delete"
    ),
    "delete_file": ToolCapabilities(
        destructive=True, writes_data=True, category="delete"
    ),
    "delete_patient_record": ToolCapabilities(
        destructive=True, writes_data=True, category="delete"
    ),
    "make_payment": ToolCapabilities(category="payment"),
    "approve_payment": ToolCapabilities(category="payment"),
    "verify_identity": ToolCapabilities(category="verify"),
    "notify_user": ToolCapabilities(category="notify"),
    "read_file": ToolCapabilities(reads_data=True, category="read"),
    "write_file": ToolCapabilities(writes_data=True, category="write"),
    "bash_command": ToolCapabilities(writes_data=True, destructive=True, category="write"),
    "bash": ToolCapabilities(writes_data=True, destructive=True, category="write"),
    "execute_command": ToolCapabilities(writes_data=True, destructive=True, category="write"),
    "run_command": ToolCapabilities(writes_data=True, destructive=True, category="write"),
}


def register_tool_capabilities(
    tool_name: str,
    capabilities: ToolCapabilities,
) -> None:
    """Register (or overwrite) capability metadata for ``tool_name``."""
    name = str(tool_name).strip()
    if not name:
        raise ValueError("tool_name must be non-empty")
    with _LOCK:
        _REGISTRY[name] = capabilities


def register_tool(
    tool_name: str,
    *,
    reads_data: bool = False,
    writes_data: bool = False,
    network_egress: bool = False,
    destructive: bool = False,
    category: Optional[str] = None,
) -> ToolCapabilities:
    """Convenience wrapper around :func:`register_tool_capabilities`."""
    caps = ToolCapabilities(
        reads_data=reads_data,
        writes_data=writes_data,
        network_egress=network_egress,
        destructive=destructive,
        category=category,
    )
    register_tool_capabilities(tool_name, caps)
    return caps


def get_tool_capabilities(tool_name: str) -> ToolCapabilities:
    """Return registered caps, defaults, or heuristic inference."""
    name = str(tool_name or "").strip()
    with _LOCK:
        if name in _REGISTRY:
            return _REGISTRY[name]
    if name in _DEFAULT_CAPS:
        return _DEFAULT_CAPS[name]
    return infer_capabilities_from_name(name)


def clear_tool_capabilities() -> None:
    """Clear runtime registrations (defaults remain). For tests."""
    with _LOCK:
        _REGISTRY.clear()


def list_registered_tools() -> List[str]:
    with _LOCK:
        return sorted(set(_REGISTRY) | set(_DEFAULT_CAPS))


def categorize_tool(tool_name: str) -> str:
    """Map a tool to a Markov category using capabilities first."""
    caps = get_tool_capabilities(tool_name)
    if caps.category and caps.category in KNOWN_CATEGORIES:
        return caps.category
    if caps.network_egress:
        return "network_egress"
    if caps.destructive:
        return "delete"
    if caps.reads_data and not caps.writes_data:
        return "read"
    if caps.writes_data:
        return "write"
    return infer_category_from_name(tool_name)


def infer_capabilities_from_name(tool_name: str) -> ToolCapabilities:
    """Conservative name-based fallback when no explicit registration exists."""
    lowered = tool_name.lower()
    reads = any(h in lowered for h in ("read", "load", "fetch", "get"))
    writes = any(h in lowered for h in ("write", "save", "persist", "export"))
    egress = any(
        h in lowered
        for h in ("send_", "upload_", "network_", "exfil", "email", "smtp", "http")
    )
    destructive = any(h in lowered for h in ("delete_", "remove_", "destroy"))
    return ToolCapabilities(
        reads_data=reads,
        writes_data=writes,
        network_egress=egress,
        destructive=destructive,
        category=infer_category_from_name(tool_name),
    )


def infer_category_from_name(tool_name: str) -> str:
    lowered = tool_name.lower()
    if any(
        h in lowered
        for h in ("send_", "upload_", "network_", "exfil", "email", "call_api", "http")
    ):
        return "network_egress"
    if any(h in lowered for h in ("delete_", "remove_", "destroy")):
        return "delete"
    if "payment" in lowered or "pay_" in lowered:
        return "payment"
    if any(h in lowered for h in ("verify", "approve", "run_tests", "test")):
        return "verify"
    if "notify" in lowered or "alert" in lowered:
        return "notify"
    if any(
        h in lowered
        for h in ("read", "load", "fetch", "get", "search", "db_query", "query")
    ):
        return "read"
    if any(h in lowered for h in ("write", "save", "persist", "export", "edit")):
        return "write"
    if any(h in lowered for h in ("analy", "summar", "summarize")):
        return "analyze"
    return "analyze"


def capabilities_from_iterable(
    tools: Iterable[str],
) -> Dict[str, ToolCapabilities]:
    return {name: get_tool_capabilities(name) for name in tools}
