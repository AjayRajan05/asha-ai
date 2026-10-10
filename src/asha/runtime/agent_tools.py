"""Tool normalization and ReAct loop for ASHA Agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

_TOOL_CALL_RE = re.compile(
    r"TOOL_CALL:\s*([A-Za-z_][\w]*)\s*(?:\n|\r\n?)ARGS:\s*(\{.*?\})",
    re.DOTALL | re.IGNORECASE,
)
_JSON_TOOL_RE = re.compile(
    r'\{\s*"tool(?:_name)?"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL | re.IGNORECASE,
)

ToolInput = Union[str, Callable[..., Any], Dict[str, Any], Any]


@dataclass
class AgentTool:
    """Normalized runnable tool for the ASHA Agent loop."""

    name: str
    runner: Callable[..., Any]
    description: str = ""
    args_schema: Any = None

    def run(self, **kwargs: Any) -> Any:
        return self.runner(**kwargs)

    def _run(self, **kwargs: Any) -> Any:
        return self.run(**kwargs)


def normalize_tools(tools: Sequence[ToolInput]) -> List[AgentTool]:
    """Convert arbitrary tool definitions into runnable AgentTool objects."""
    normalized: List[AgentTool] = []
    for item in tools or []:
        tool = _normalize_one(item)
        if tool is not None:
            normalized.append(tool)
    return normalized


def _normalize_one(item: ToolInput) -> Optional[AgentTool]:
    if isinstance(item, AgentTool):
        return item
    if isinstance(item, str):
        name = item

        def _stub_runner(**_kwargs: Any) -> str:
            return f"Tool '{name}' has no implementation. Pass a callable tool."

        return AgentTool(name=name, description=f"Tool: {name}", runner=_stub_runner)
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("tool_name") or "")
        fn = (
            item.get("fn")
            or item.get("func")
            or item.get("callable")
            or item.get("run")
        )
        if name and callable(fn):
            return AgentTool(
                name=name,
                description=str(item.get("description", f"Tool: {name}")),
                runner=fn,
                args_schema=item.get("args_schema"),
            )
        if name:

            def _stub(**_kwargs: Any) -> str:
                return f"Tool '{name}' has no implementation."

            return AgentTool(
                name=name,
                description=str(item.get("description", f"Tool: {name}")),
                runner=_stub,
            )
        return None
    if callable(item):
        name = str(
            getattr(item, "name", None)
            or getattr(item, "__name__", type(item).__name__)
        )
        return AgentTool(
            name=name,
            description=str(getattr(item, "description", "") or f"Tool: {name}"),
            runner=item,
            args_schema=getattr(item, "args_schema", None),
        )
    if hasattr(item, "_run") or hasattr(item, "run") or hasattr(item, "invoke"):
        name = getattr(item, "name", None) or type(item).__name__

        def _runner(**kwargs: Any) -> Any:
            if hasattr(item, "_run"):
                return item._run(**kwargs)
            if hasattr(item, "run"):
                return item.run(**kwargs)
            return item.invoke(kwargs or {})

        return AgentTool(
            name=str(name),
            description=str(getattr(item, "description", "") or f"Tool: {name}"),
            runner=_runner,
            args_schema=getattr(item, "args_schema", None),
        )
    return None


def build_tool_system_prompt(tools: Sequence[AgentTool]) -> str:
    lines = [
        "You have access to the following tools. To call a tool, respond with:",
        "TOOL_CALL: <tool_name>",
        'ARGS: {"key": "value"}',
        "",
        "Available tools:",
    ]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description or 'No description'}")
    lines.append("")
    lines.append(
        "When the task is complete, respond with the final answer only (no TOOL_CALL)."
    )
    return "\n".join(lines)


def parse_tool_call(text: str) -> Optional[tuple[str, Dict[str, Any]]]:
    """Parse a tool call from LLM text output."""
    match = _TOOL_CALL_RE.search(text or "")
    if match:
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        return name, args if isinstance(args, dict) else {}

    json_match = _JSON_TOOL_RE.search(text or "")
    if json_match:
        name = json_match.group(1)
        try:
            args = json.loads(json_match.group(2))
        except json.JSONDecodeError:
            args = {}
        return name, args if isinstance(args, dict) else {}
    return None


def run_tool_loop(
    *,
    generate: Callable[[str], str],
    tools: Sequence[ToolInput],
    prompt: str,
    max_iterations: int = 8,
    tool_executor: Optional[Callable[[AgentTool, Dict[str, Any]], Any]] = None,
) -> str:
    """
    Run a ReAct-style tool loop: LLM proposes tools, tools execute, results feed back.
    """
    agent_tools = normalize_tools(tools)
    if not agent_tools:
        return generate(prompt)

    tools_by_name = {t.name: t for t in agent_tools}
    system = build_tool_system_prompt(agent_tools)
    conversation = f"{system}\n\nUser task:\n{prompt}"

    def _default_executor(tool: AgentTool, args: Dict[str, Any]) -> Any:
        return tool.run(**args)

    executor = tool_executor or _default_executor

    for _ in range(max_iterations):
        response = str(generate(conversation) or "")
        parsed = parse_tool_call(response)
        if not parsed:
            return response

        tool_name, args = parsed
        tool = tools_by_name.get(tool_name)
        if tool is None:
            conversation += (
                f"\n\nAssistant:\n{response}\n\n"
                f"System: Unknown tool '{tool_name}'. Use one of: {', '.join(tools_by_name)}."
            )
            continue

        try:
            result = executor(tool, args)
        except Exception as exc:  # pragma: no cover - surfaced to LLM
            result = f"Tool error: {exc}"

        conversation += (
            f"\n\nAssistant:\n{response}\n\n" f"Tool result ({tool_name}):\n{result}"
        )

    return generate(conversation)
