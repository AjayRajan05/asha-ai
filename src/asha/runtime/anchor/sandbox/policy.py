"""OS-level side-effect sandbox policies for ANCHOR tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SandboxPolicy:
    """Declarative sandbox policy derived from the mission contract."""

    allow_network: bool = False
    allow_subprocess: bool = False
    allow_file_write: bool = True
    allowed_write_paths: tuple[str, ...] = ("output/", "data/")
    blocked_modules: tuple[str, ...] = (
        "socket",
        "ftplib",
        "smtplib",
        "requests",
        "httpx",
        "urllib",
    )


def policy_from_mission(
    *, local_only: bool, allowed_write_paths: Iterable[str]
) -> SandboxPolicy:
    write_paths = tuple(allowed_write_paths) if allowed_write_paths else ("output/",)
    return SandboxPolicy(
        allow_network=not local_only,
        allow_subprocess=not local_only,
        allow_file_write=True,
        allowed_write_paths=write_paths,
    )
