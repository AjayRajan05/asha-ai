"""Shared production utilities for all ANCHOR framework adapters."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional

from ..runtime import AnchorRuntime, current_anchor_runtime


def tool_name(tool: Any) -> str:
    """Resolve a stable tool name from any framework tool object."""
    if isinstance(tool, str):
        return tool
    if hasattr(tool, "name"):
        name = getattr(tool, "name", None)
        if name:
            return str(name)
    metadata = getattr(tool, "metadata", None)
    if metadata is not None:
        meta_name = getattr(metadata, "name", None)
        if meta_name:
            return str(meta_name)
    inner = getattr(tool, "_inner", None)
    if inner is not None:
        return tool_name(inner)
    return getattr(tool, "__name__", type(tool).__name__)


def safe_setattr(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except (ValueError, AttributeError, TypeError):
        object.__setattr__(obj, name, value)


def resolve_templates(text: str, inputs: Dict[str, Any]) -> str:
    resolved = text
    for key, value in inputs.items():
        resolved = resolved.replace(f"{{{key}}}", str(value))
    return resolved


def is_wrapped(tool: Any) -> bool:
    return bool(
        getattr(tool, "_anchor_wrapped", False)
        or getattr(tool, "_anchor_runtime", None)
    )


def block_message(runtime: AnchorRuntime, tool_name: str) -> str:
    if runtime.state.approval_history:
        reason = runtime.state.approval_history[-1].verdict.reason
    else:
        reason = "Action blocked by mission contract."
    return f"Tool '{tool_name}' was denied: {reason}"


def mission_context(
    runtime: AnchorRuntime,
    tools: Iterable[Any],
    *,
    local_only: bool = True,
) -> Dict[str, Any]:
    return {
        "available_tools": list(dict.fromkeys(tool_name(t) for t in tools)),
        "risk_tolerance": runtime.mission_compiler.default_risk_tolerance,
        "local_only": local_only,
    }


def initialize_mission(
    runtime: AnchorRuntime,
    prompt: str,
    tools: Iterable[Any],
    *,
    local_only: bool = True,
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    ctx = mission_context(runtime, tools, local_only=local_only)
    if extra_context:
        ctx.update(extra_context)
    runtime.initialize_mission(prompt, context=ctx)


def refresh_mission_phase(
    runtime: AnchorRuntime,
    prompt: str,
    tools: Iterable[Any],
    *,
    local_only: bool = True,
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    ctx = mission_context(runtime, tools, local_only=local_only)
    if extra_context:
        ctx.update(extra_context)
    runtime.refresh_mission_phase(prompt, context=ctx)


def finalize_step_output(
    runtime: AnchorRuntime,
    output: Any,
    *,
    required_tools: Optional[List[str]] = None,
    raise_on_block: bool = True,
) -> Any:
    """Apply memory and plan governance after an agent step completes."""
    return runtime.govern_step_output(
        output,
        required_tools=required_tools,
        raise_on_block=raise_on_block,
    )


def finalize_session(runtime: AnchorRuntime) -> None:
    """Emit final session telemetry and risk summary."""
    runtime.finalize_session()


@contextmanager
def bind_runtime(runtime: AnchorRuntime) -> Generator[None, None, None]:
    token = current_anchor_runtime.set(runtime)
    try:
        yield
    finally:
        current_anchor_runtime.reset(token)


@contextmanager
def anchor_session(runtime: AnchorRuntime) -> Generator[AnchorRuntime, None, None]:
    """Bind runtime context and finalize session on exit."""
    with bind_runtime(runtime):
        try:
            yield runtime
        finally:
            finalize_session(runtime)


def run_async_if_needed(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def wrap_tool_list(
    tools: List[Any],
    runtime: AnchorRuntime,
    wrapper_factory: Callable[[Any, AnchorRuntime], Any],
) -> List[Any]:
    wrapped: List[Any] = []
    for tool in tools:
        if is_wrapped(tool):
            wrapped.append(tool)
        else:
            wrapped.append(wrapper_factory(tool, runtime))
    return wrapped


def extract_prompt_from_input(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, dict):
        for key in ("input", "query", "prompt", "messages", "question"):
            if key in input_value:
                val = input_value[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, list) and val:
                    last = val[-1]
                    if isinstance(last, dict):
                        return str(last.get("content", last))
                    return str(getattr(last, "content", last) or last)
        return str(input_value)
    return str(input_value)
