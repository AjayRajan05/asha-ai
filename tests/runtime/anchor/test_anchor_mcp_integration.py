"""MCP integration (skips if asha[mcp] not installed)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytest.importorskip("mcp")

from asha.runtime.anchor.adapters.mcp import anchor_mcp
from asha.runtime.anchor.tool_capabilities import register_tool


class _StubMCP:
    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "read_file", "_meta": {}},
            {
                "name": "send_email",
                "_meta": {
                    "asha_capabilities": {
                        "network_egress": True,
                        "category": "network_egress",
                    }
                },
            },
        ]

    def call_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> str:
        return f"ran:{name}"


def test_mcp_package_importable() -> None:
    import mcp

    assert mcp is not None


def test_anchor_mcp_blocks_egress_when_package_present() -> None:
    register_tool("send_email", network_egress=True, category="network_egress")
    proxied = anchor_mcp(_StubMCP(), interactive=False, risk_tolerance="LOW")
    proxied.initialize_session(
        "Analyze data locally. Do not send email.",
        local_only=True,
    )
    blocked = proxied.call_tool("send_email", {"to": "x@y.com"})
    assert "denied" in str(blocked).lower()
