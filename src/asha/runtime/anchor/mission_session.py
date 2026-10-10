"""Session-stable mission contracts with phase overlays."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from .contracts import MissionContract
from .mission import MissionCompiler


def merge_mission_with_baseline(
    baseline: MissionContract,
    phase: MissionContract,
) -> MissionContract:
    """
    Merge a phase mission into the session baseline without weakening constraints.

    Phase missions may update goals and tools for the current step, but baseline
    security constraints (local_only, forbidden actions, path scopes) are preserved.
    """
    forbidden = list(
        dict.fromkeys(list(baseline.forbidden_actions) + list(phase.forbidden_actions))
    )
    network_tokens = list(
        dict.fromkeys(
            list(baseline.forbidden_network_tokens)
            + list(phase.forbidden_network_tokens)
        )
    )
    allowed_tools = list(
        dict.fromkeys(list(baseline.allowed_tools) + list(phase.allowed_tools))
    )
    # When baseline is low-confidence, refresh must not silently loosen restrictions
    # by unioning phase send/write permissions onto the read/list-only floor.
    if baseline.low_confidence:
        allowed_actions = list(baseline.allowed_actions)
    else:
        allowed_actions = list(
            dict.fromkeys(list(baseline.allowed_actions) + list(phase.allowed_actions))
        )
    # Baseline write/read permissions win when present
    read_paths = baseline.allowed_read_paths or phase.allowed_read_paths
    write_paths = baseline.allowed_write_paths or phase.allowed_write_paths

    return replace(
        phase,
        mission_id=baseline.mission_id,
        forbidden_actions=forbidden,
        forbidden_network_tokens=network_tokens,
        allowed_tools=allowed_tools,
        allowed_actions=allowed_actions,
        allowed_read_paths=read_paths,
        allowed_write_paths=write_paths,
        local_only=baseline.local_only or phase.local_only,
        forbid_network_exfiltration=(
            baseline.forbid_network_exfiltration or phase.forbid_network_exfiltration
        ),
        risk_tolerance=baseline.risk_tolerance,
        low_confidence=baseline.low_confidence or phase.low_confidence,
    )


class MissionSession:
    """Maintains an immutable baseline mission across task phases."""

    def __init__(self, compiler: MissionCompiler | None = None) -> None:
        self._compiler = compiler or MissionCompiler()
        self._baseline: MissionContract | None = None

    @property
    def baseline(self) -> MissionContract | None:
        return self._baseline

    def initialize(
        self, prompt: str, context: Dict[str, Any] | None = None
    ) -> MissionContract:
        mission = self._compiler.compile(prompt, context)
        self._baseline = mission
        return mission

    def refresh_phase(
        self, prompt: str, context: Dict[str, Any] | None = None
    ) -> MissionContract:
        phase = self._compiler.compile(prompt, context)
        if self._baseline is None:
            self._baseline = phase
            return phase
        return merge_mission_with_baseline(self._baseline, phase)
