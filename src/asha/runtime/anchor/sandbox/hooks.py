"""In-process OS side-effect hooks for ANCHOR sandbox enforcement."""

from __future__ import annotations

from contextlib import contextmanager
import builtins
from typing import Any, Callable, Iterator, Optional, cast

from .policy import SandboxPolicy


class SandboxViolation(RuntimeError):
    """Raised when a sandboxed operation is attempted outside policy."""


@contextmanager
def enforcement_hooks(policy: SandboxPolicy) -> Iterator[None]:
    """Patch high-risk builtins while a governed tool executes."""
    original_open = builtins.open
    blocked: list[str] = []

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        path = str(file).replace("\\", "/").lower()
        writing = any(flag in mode for flag in ("w", "a", "+", "x"))
        if writing and not policy.allow_file_write:
            raise SandboxViolation(f"File write blocked by sandbox: {file}")
        if writing and policy.allowed_write_paths:
            if not any(
                path.startswith(prefix) for prefix in policy.allowed_write_paths
            ):
                raise SandboxViolation(f"Write outside allowed paths blocked: {file}")
        return original_open(file, mode, *args, **kwargs)

    def _block(name: str) -> Callable[..., Any]:
        def _blocked(*_args: Any, **_kwargs: Any) -> Any:
            raise SandboxViolation(f"{name} blocked by ANCHOR sandbox policy.")

        return _blocked

    import sys

    patched_modules: dict[str, Any] = {}
    try:
        setattr(builtins, "open", guarded_open)
        if not policy.allow_network:
            for module_name in policy.blocked_modules:
                if module_name in sys.modules:
                    patched_modules[module_name] = sys.modules[module_name]
                    cast(Any, sys.modules)[module_name] = None
        yield
    finally:
        setattr(builtins, "open", original_open)
        for module_name, module in patched_modules.items():
            sys.modules[module_name] = module


def run_in_subprocess(
    runner: Callable[..., Any],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: Optional[float] = 30.0,
) -> Any:
    """Execute a tool callable in an isolated subprocess (hard isolation mode)."""
    import multiprocessing

    def _target(queue: Any) -> None:
        try:
            queue.put(("ok", runner(*args, **kwargs)))
        except Exception as exc:  # pragma: no cover - propagated to parent
            queue.put(("err", repr(exc)))

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_target, args=(queue,))
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        raise SandboxViolation("Sandbox subprocess timed out.")
    if queue.empty():
        raise SandboxViolation("Sandbox subprocess produced no result.")
    status, payload = queue.get()
    if status == "err":
        raise SandboxViolation(f"Sandbox subprocess failed: {payload}")
    return payload
