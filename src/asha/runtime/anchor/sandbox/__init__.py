"""Sandbox manager coordinating policy and execution modes."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .hooks import enforcement_hooks, run_in_subprocess
from .policy import SandboxPolicy, policy_from_mission


class SandboxManager:
    """
    Enforces OS-level side-effect constraints during tool execution.

    Modes:
    - off: no sandbox enforcement
    - auto: in-process hooks (default)
    - hard / subprocess: run tool callable in child process
    """

    _SUPPORTED = frozenset({"off", "none", "disabled", "auto", "hard", "subprocess"})

    def __init__(
        self, mode: str = "auto", policy: Optional[SandboxPolicy] = None
    ) -> None:
        self.mode = (mode or "auto").lower()
        if self.mode in ("docker", "bwrap", "container"):
            raise ValueError(
                f"Sandbox mode '{self.mode}' is not implemented in asha 0.4.2 "
                "(deferred to 0.5.x). Use isolation='auto' or isolation='hard'."
            )
        if self.mode not in self._SUPPORTED:
            raise ValueError(
                f"Unknown sandbox isolation mode '{self.mode}'. "
                "Supported: off, auto, hard (subprocess)."
            )
        self.policy = policy or SandboxPolicy()

    def apply_mission(
        self, *, local_only: bool, allowed_write_paths: list[str]
    ) -> None:
        self.policy = policy_from_mission(
            local_only=local_only,
            allowed_write_paths=allowed_write_paths,
        )

    def execute(
        self,
        runner: Callable[..., Any],
        *,
        tool_name: str,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        if self.mode in ("off", "none", "disabled"):
            return runner(*args, **(kwargs or {}))

        kwargs = kwargs or {}
        if self.mode in ("hard", "subprocess"):
            return run_in_subprocess(runner, args=args, kwargs=kwargs)

        with enforcement_hooks(self.policy):
            return runner(*args, **kwargs)
