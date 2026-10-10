"""AutoGen adapter for ASHA ANCHOR: production parity with CrewAI."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..runtime import AnchorRuntime
from ..tool_bridge import wrap_tool
from .base import (
    bind_runtime,
    extract_prompt_from_input,
    finalize_session,
    finalize_step_output,
    initialize_mission,
    is_wrapped,
    refresh_mission_phase,
    safe_setattr,
    tool_name,
)
from .discovery import discover_tools


def is_autogen_agent(target: Any) -> bool:
    module = type(target).__module__ or ""
    if "autogen" in module or "pyautogen" in module:
        return True
    if hasattr(target, "register_for_llm") and hasattr(target, "generate_reply"):
        return True
    return hasattr(target, "agents") and hasattr(target, "initiate_chat")


def is_autogen_group_chat(target: Any) -> bool:
    module = type(target).__module__ or ""
    if "groupchat" in module.lower() or "group_chat" in type(target).__name__.lower():
        return True
    return hasattr(target, "agents") and hasattr(target, "max_round")


def _message_content(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(getattr(last, "content", last) or last)


class _AnchoredAutoGenAgentProxy:
    """Wrap AutoGen ConversableAgent with full ANCHOR governance."""

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._wrap_function_map()
        self._patch_register_hooks()

    def _function_map(self) -> Dict[str, Callable[..., Any]]:
        fn_map = getattr(self._inner, "function_map", None)
        return dict(fn_map) if isinstance(fn_map, dict) else {}

    def _wrap_function_map(self) -> None:
        fn_map = self._function_map()
        if not fn_map:
            return
        wrapped: Dict[str, Callable[..., Any]] = {}
        for name, fn in fn_map.items():
            if is_wrapped(fn):
                wrapped[name] = fn
            else:
                wrapped[name] = wrap_tool(fn, self._runtime, name=name)
        safe_setattr(self._inner, "function_map", wrapped)

    def _patch_register_hooks(self) -> None:
        original_register = getattr(self._inner, "register_for_execution", None)
        if not callable(original_register) or getattr(
            original_register, "_anchor_patched", False
        ):
            return

        runtime = self._runtime

        def guarded_register(name: str, func: Callable[..., Any]) -> Any:
            return original_register(name, wrap_tool(func, runtime, name=name))

        guarded_register._anchor_patched = True  # type: ignore[attr-defined]
        safe_setattr(self._inner, "register_for_execution", guarded_register)

    def _tool_names(self) -> List[str]:
        names = list(self._function_map().keys())
        if names:
            return names
        return [tool_name(t) for t in discover_tools(self._inner)]

    def _bind_mission(self, prompt: str, *, phase: bool = False) -> None:
        names = self._tool_names()
        if phase and self._runtime.state.baseline_mission is not None:
            refresh_mission_phase(self._runtime, prompt, names)
        else:
            initialize_mission(self._runtime, prompt, names)

    def generate_reply(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or (args[0] if args else [])
        prompt = _message_content(messages) or "AutoGen agent workflow."
        self._bind_mission(
            prompt, phase=self._runtime.state.baseline_mission is not None
        )
        try:
            with bind_runtime(self._runtime):
                result = self._inner.generate_reply(*args, **kwargs)
            output = (
                result.get("content", result) if isinstance(result, dict) else result
            )
            return finalize_step_output(
                self._runtime,
                output,
                required_tools=self._tool_names(),
                raise_on_block=False,
            )
        finally:
            finalize_session(self._runtime)

    def initiate_chat(self, recipient: Any, message: str = "", **kwargs: Any) -> Any:
        if not is_wrapped(recipient):
            recipient = anchor_autogen(recipient, runtime=self._runtime)
        self._bind_mission(
            message or extract_prompt_from_input(kwargs.get("message", ""))
        )
        try:
            with bind_runtime(self._runtime):
                return self._inner.initiate_chat(recipient, message=message, **kwargs)
        finally:
            finalize_session(self._runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AnchoredAutoGenGroupProxy:
    """Wrap AutoGen GroupChat / GroupChatManager: governs all member agents."""

    def __init__(self, inner: Any, runtime: AnchorRuntime) -> None:
        self._inner = inner
        self._runtime = runtime
        self._anchor_runtime = runtime
        self._wrap_group_agents()

    def _wrap_group_agents(self) -> None:
        agents = list(getattr(self._inner, "agents", None) or [])
        wrapped = []
        for agent in agents:
            if is_wrapped(agent):
                wrapped.append(agent)
            else:
                wrapped.append(anchor_autogen(agent, runtime=self._runtime))
        if wrapped and hasattr(self._inner, "agents"):
            safe_setattr(self._inner, "agents", wrapped)

    def initiate_chat(self, *args: Any, **kwargs: Any) -> Any:
        message = kwargs.get("message") or (args[1] if len(args) > 1 else "")
        names = [tool_name(a) for a in discover_tools(self._inner)]
        initialize_mission(
            self._runtime, str(message or "AutoGen group workflow."), names
        )
        try:
            with bind_runtime(self._runtime):
                return self._inner.initiate_chat(*args, **kwargs)
        finally:
            finalize_session(self._runtime)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def anchor_autogen(
    target: Any,
    *,
    runtime: Optional[AnchorRuntime] = None,
    isolation: str = "auto",
    risk_tolerance: str = "LOW",
    warn_policy: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> Any:
    """Wrap AutoGen ConversableAgent or GroupChat with ANCHOR governance."""
    if not is_autogen_agent(target) and not is_autogen_group_chat(target):
        raise TypeError(
            f"anchor_autogen() expects an AutoGen agent or group chat, got {type(target).__name__}."
        )
    runtime = runtime or AnchorRuntime(
        risk_tolerance=risk_tolerance,
        warn_policy=warn_policy,
        interactive=interactive,
        isolation=isolation,
    )
    if isinstance(target, (_AnchoredAutoGenAgentProxy, _AnchoredAutoGenGroupProxy)):
        return target
    if is_autogen_group_chat(target):
        return _AnchoredAutoGenGroupProxy(target, runtime)
    return _AnchoredAutoGenAgentProxy(target, runtime)
