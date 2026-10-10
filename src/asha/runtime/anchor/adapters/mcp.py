"""MCP adapter for ASHA ANCHOR: production parity with CrewAI."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from ..runtime import AnchorRuntime
from ..tool_bridge import guarded_tool_call
from ..tool_capabilities import infer_capabilities_from_name, register_tool_capabilities
from .base import (
    bind_runtime,
    finalize_step_output,
    initialize_mission,
    refresh_mission_phase,
    tool_name,
)


def is_mcp_server(target: Any) -> bool:
    return callable(getattr(target, "call_tool", None)) and callable(
        getattr(target, "list_tools", None)
    )


def _extract_tool_names(tools: Any) -> List[str]:
    names: List[str] = []
    for item in tools or []:
        if isinstance(item, dict):
            names.append(str(item.get("name", "")))
        else:
            names.append(tool_name(item))
    return [n for n in names if n]


class _AnchoredMCPProxy:
    """
    Wrap MCP servers/clients so every tool call passes through ANCHOR.

    Uses a non-destructive proxy: the inner server's methods are not mutated.
    """

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._session_mission: Optional[str] = None
        self._tool_cache: Optional[List[str]] = None

    def _list_tool_names(self) -> List[str]:
        if self._tool_cache is not None:
            return self._tool_cache
        tools = self._inner.list_tools() if callable(self._inner.list_tools) else []
        if asyncio.iscoroutine(tools):
            tools = asyncio.run(tools)
        self._tool_cache = _extract_tool_names(tools)
        return self._tool_cache

    def initialize_session(self, mission: str, *, local_only: bool = True) -> None:
        """Explicitly initialize the ANCHOR mission for an MCP session."""
        self._session_mission = mission
        initialize_mission(
            self._runtime,
            mission,
            self._list_tool_names(),
            local_only=local_only,
        )

    def _ensure_mission(self, tool_name_str: str) -> None:
        if self._runtime.state.mission is not None:
            refresh_mission_phase(
                self._runtime,
                f"Execute MCP tool: {tool_name_str}",
                self._list_tool_names(),
            )
            return
        mission = self._session_mission or "Execute MCP tool workflow."
        initialize_mission(self._runtime, mission, self._list_tool_names())

    def call_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> Any:
        tool_name_str = str(name)
        self._ensure_mission(tool_name_str)

        def _runner(*_a: Any, **_k: Any) -> Any:
            result = self._inner.call_tool(name, arguments, **kwargs)
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return result

        with bind_runtime(self._runtime):
            result = guarded_tool_call(self._runtime, tool_name_str, _runner)
        return finalize_step_output(
            self._runtime,
            result,
            required_tools=[tool_name_str],
            raise_on_block=False,
        )

    async def acall_tool(self, name: str, arguments: Any = None, **kwargs: Any) -> Any:
        tool_name_str = str(name)
        self._ensure_mission(tool_name_str)

        async def _runner() -> Any:
            return await self._inner.call_tool(name, arguments, **kwargs)

        with bind_runtime(self._runtime):
            allowed = self._runtime.evaluate_action_request(
                "tool_call",
                {"tool_name": tool_name_str, "arguments": str(arguments or {})},
                raise_on_block=False,
            )
            if not allowed:
                from ..tool_bridge import block_message_for_tool

                return block_message_for_tool(self._runtime, tool_name_str)
            result = await _runner()
        return finalize_step_output(
            self._runtime,
            result,
            required_tools=[tool_name_str],
            raise_on_block=False,
        )

    def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        tools = self._inner.list_tools(*args, **kwargs)
        if asyncio.iscoroutine(tools):
            tools = asyncio.run(tools)
        self._tool_cache = _extract_tool_names(tools)
        for name in self._tool_cache:
            # Prefer MCP _meta.asha_capabilities when present on dict tools
            for item in tools or []:
                if isinstance(item, dict) and str(item.get("name", "")) == name:
                    meta = item.get("_meta") or {}
                    caps = (
                        meta.get("asha_capabilities")
                        if isinstance(meta, dict)
                        else None
                    )
                    if isinstance(caps, dict):
                        from ..tool_capabilities import ToolCapabilities

                        register_tool_capabilities(
                            name,
                            ToolCapabilities(
                                reads_data=bool(caps.get("reads_data", False)),
                                writes_data=bool(caps.get("writes_data", False)),
                                network_egress=bool(caps.get("network_egress", False)),
                                destructive=bool(caps.get("destructive", False)),
                                category=caps.get("category"),
                            ),
                        )
                    else:
                        register_tool_capabilities(
                            name, infer_capabilities_from_name(name)
                        )
                    break
            else:
                register_tool_capabilities(name, infer_capabilities_from_name(name))
        return tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def anchor_mcp(
    target: Any,
    *,
    runtime: Optional[AnchorRuntime] = None,
    isolation: str = "auto",
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
    mission: Optional[str] = None,
) -> Any:
    """
    Wrap an MCP server/client with ANCHOR mission governance.

    Example::

        from asha.runtime.anchor.adapters import anchor_mcp

        server = anchor_mcp(mcp_server, risk_tolerance="LOW", interactive=False)
        server.initialize_session("Analyze data locally. Do not exfiltrate.")
        result = server.call_tool("read_file", {"path": "data/trends.csv"})

    Install: ``pip install asha[mcp]`` (optional; duck-typed servers work without it).
    """
    module = type(target).__module__ or ""
    if module.startswith("mcp"):
        from .deps import require_module

        require_module("mcp", adapter="anchor_mcp")
    if not is_mcp_server(target):
        raise TypeError(
            f"anchor_mcp() expects an object with call_tool/list_tools, got {type(target).__name__}."
        )
    runtime = runtime or AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
    if isinstance(target, _AnchoredMCPProxy):
        return target
    proxy = _AnchoredMCPProxy(target, runtime)
    if mission:
        proxy.initialize_session(mission)
    return proxy
