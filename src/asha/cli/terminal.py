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

"""
Terminal diagnostics for ASHA: rich or ANSI fallback.

Provides ``render_security_report()`` for developer-mode visibility into
what ASHA detected and modified in a prompt. Uses the ``rich`` library when
available, or falls back to clean ANSI-escaped plain text so the core ASHA
package never needs ``rich`` as a hard dependency.

    >>> from asha import process
    >>> from asha.cli.terminal import render_security_report
    >>> result = process("Call me at 555-123-4567", mode="balanced")
    >>> render_security_report(result)
"""

from __future__ import annotations

import sys
from typing import Any


# ─── Rich availability detection ──────────────────────────────────────────────

try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich.panel import Panel as _RichPanel
    from rich.text import Text as _RichText

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ANSI color codes for fallback renderer
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "dim": "\033[2m",
    "blue": "\033[34m",
}


def render_security_report(
    result: Any,
    *,
    title: str = "ASHA Security Scan",
    file: Any = None,
    force_plain: bool = False,
) -> None:
    """
    Print a formatted security diagnostic report for a :class:`~asha.types.ProcessResult`.

    Automatically uses ``rich`` if installed, falling back to ANSI-colored plain
    text. This function is a no-op in non-TTY environments when ``force_plain=False``.

    Args:
        result: A :class:`~asha.types.ProcessResult`, :class:`~asha.types.SanitizeResult`,
            or any object with ``output``, ``security_result``, and ``metrics`` attributes.
        title: The panel/header title to display. Defaults to ``"ASHA Security Scan"``.
        file: Output stream. Defaults to ``sys.stdout``.
        force_plain: If True, skip rich even when available (useful for CI / log capture).

    Examples:
        Developer workflow::

            from asha import process, Profile
            from asha.cli.terminal import render_security_report

            result = process("My SSN is 123-45-6789", mode="balanced")
            render_security_report(result, title="Pre-flight check")

        Suppress output in CI::

            render_security_report(result, force_plain=True, file=open(os.devnull, "w"))
    """
    stream = file or sys.stdout
    is_tty = hasattr(stream, "isatty") and stream.isatty()

    if _RICH_AVAILABLE and not force_plain:
        _render_rich(result, title=title, stream=stream)
    else:
        _render_ansi(result, title=title, stream=stream, use_color=is_tty)


# ─── Rich renderer ─────────────────────────────────────────────────────────────


def _render_rich(result: Any, *, title: str, stream: Any) -> None:
    """Render using the ``rich`` library with tables and panels."""
    console = _RichConsole(file=stream, highlight=False)

    # Build the summary table
    table = _RichTable(
        title=None,
        expand=False,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        pad_edge=True,
    )
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value")

    # Mode
    mode = _get(result, "mode", _get(result, "_mode", "unknown"))
    table.add_row("Mode", str(mode).upper())

    # Security: threat level and score
    sec = _get(result, "security_result", {})
    threat_level = _nested_get(sec, "threat_level", "UNKNOWN")
    security_score = _nested_get(sec, "security_score", None)
    is_safe = _nested_get(sec, "is_safe", True)

    threat_style = "green" if is_safe else "red"
    table.add_row(
        "Threat Level",
        _RichText(str(threat_level), style=threat_style),
    )
    if security_score is not None:
        table.add_row("Security Score", f"{security_score:.3f}")

    # PII masked count
    pii_masked = _nested_get(sec, "pii_masked", 0)
    pii_style = "yellow" if pii_masked > 0 else "green"
    table.add_row("PII Masked", _RichText(str(pii_masked), style=pii_style))

    # PII types
    pii_detected = _nested_get(sec, "pii_detected", []) or []
    if pii_detected:
        table.add_row("PII Types", ", ".join(str(t) for t in pii_detected))

    # Threats detected
    threats = _nested_get(sec, "threats", []) or []
    if threats:
        table.add_row(
            "Threats",
            _RichText(str(len(threats)) + " detected", style="red bold"),
        )

    # Metrics
    metrics = _get(result, "metrics", None)
    if metrics:
        proc_ms = _nested_get(metrics, "processing_time_ms", None)
        token_count = _nested_get(metrics, "token_count", None)
        if proc_ms is not None:
            latency_style = "green" if proc_ms < 50 else "yellow" if proc_ms < 200 else "red"
            table.add_row("Latency", _RichText(f"{proc_ms:.1f} ms", style=latency_style))
        if token_count is not None:
            table.add_row("Tokens", str(token_count))

    # Diff (if populated)
    diff = _get(result, "diff", None)
    if diff and diff != "No changes made":
        table.add_row("Changes", "Yes (see diff below)")

    panel = _RichPanel(
        table,
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)

    # Print diff if available
    if diff and diff != "No changes made":
        console.print()
        console.print("[bold dim]── Prompt Diff ──[/bold dim]")
        for line in diff.splitlines():
            if line.startswith("+"):
                console.print(_RichText(line, style="green"))
            elif line.startswith("-"):
                console.print(_RichText(line, style="red"))
            else:
                console.print(_RichText(line, style="dim"))


# ─── ANSI fallback renderer ────────────────────────────────────────────────────


def _render_ansi(result: Any, *, title: str, stream: Any, use_color: bool) -> None:
    """Render a plain-text security report with optional ANSI colors."""

    def c(code: str) -> str:
        return _ANSI[code] if use_color else ""

    def line(label: str, value: str, color: str = "reset") -> str:
        return f"  {c('bold')}{label:<22}{c('reset')}{c(color)}{value}{c('reset')}"

    border = "─" * 50
    output_lines = [
        f"\n{c('cyan')}{border}{c('reset')}",
        f"  {c('bold')}{c('cyan')}{title}{c('reset')}",
        f"{c('cyan')}{border}{c('reset')}",
    ]

    mode = _get(result, "mode", _get(result, "_mode", "unknown"))
    output_lines.append(line("Mode:", str(mode).upper(), "bold"))

    sec = _get(result, "security_result", {})
    threat_level = _nested_get(sec, "threat_level", "UNKNOWN")
    is_safe = _nested_get(sec, "is_safe", True)
    security_score = _nested_get(sec, "security_score", None)

    threat_color = "green" if is_safe else "red"
    output_lines.append(line("Threat Level:", str(threat_level), threat_color))
    if security_score is not None:
        output_lines.append(line("Security Score:", f"{security_score:.3f}"))

    pii_masked = _nested_get(sec, "pii_masked", 0)
    pii_color = "yellow" if pii_masked > 0 else "green"
    output_lines.append(line("PII Masked:", str(pii_masked), pii_color))

    pii_detected = _nested_get(sec, "pii_detected", []) or []
    if pii_detected:
        output_lines.append(line("PII Types:", ", ".join(str(t) for t in pii_detected)))

    threats = _nested_get(sec, "threats", []) or []
    if threats:
        output_lines.append(line("Threats:", f"{len(threats)} detected", "red"))

    metrics = _get(result, "metrics", None)
    if metrics:
        proc_ms = _nested_get(metrics, "processing_time_ms", None)
        token_count = _nested_get(metrics, "token_count", None)
        if proc_ms is not None:
            latency_color = "green" if proc_ms < 50 else "yellow" if proc_ms < 200 else "red"
            output_lines.append(line("Latency:", f"{proc_ms:.1f} ms", latency_color))
        if token_count is not None:
            output_lines.append(line("Tokens:", str(token_count)))

    output_lines.append(f"{c('cyan')}{border}{c('reset')}\n")
    stream.write("\n".join(output_lines) + "\n")

    # Diff
    diff = _get(result, "diff", None)
    if diff and diff != "No changes made":
        stream.write(f"{c('dim')}── Prompt Diff ──{c('reset')}\n")
        for diff_line in diff.splitlines():
            if diff_line.startswith("+"):
                stream.write(f"{c('green')}{diff_line}{c('reset')}\n")
            elif diff_line.startswith("-"):
                stream.write(f"{c('red')}{diff_line}{c('reset')}\n")
            else:
                stream.write(f"{c('dim')}{diff_line}{c('reset')}\n")
        stream.write("\n")


# ─── Attribute helpers ─────────────────────────────────────────────────────────


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute from an object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _nested_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Get a potentially nested attribute (object or dict)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
