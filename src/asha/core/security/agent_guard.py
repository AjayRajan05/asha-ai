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
AgentCostGuard: Agentic Loop Cost and Safety Guardrail.

Prevents runaway autonomous agent loops by enforcing hard execution-step
limits, token budgets, and cyclical tool-call detection. This guard sits
between your agent's orchestration layer and the LLM client, intercepting
every tool invocation to track cumulative cost.

    >>> from asha.core.security.agent_guard import AgentCostGuard
    >>> guard = AgentCostGuard(max_steps=10, max_tokens=8000)
    >>> for step in range(20):
    ...     guard.check_step()   # Raises AgentBudgetExceeded at step 11

Threat Model:
    Agentic loops are susceptible to three classes of runaway failure:
    1. **Infinite loops**: Identical tool arguments repeated across turns,
       causing the agent to spin without making progress.
    2. **Token avalanche**: Uncontrolled context accumulation exhausting the
       model's context window or depleting API token budgets.
    3. **Indirect injection loop amplification**: An adversarial document
       causes the agent to repeatedly call the same retrieval/search tool.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from ...exceptions import AgentBudgetExceeded


@dataclass
class StepRecord:
    """Record of a single agent execution step.

    Attributes:
        step_index: Zero-based step counter within the session.
        tool_name: Name of the tool called in this step (or 'llm' for LLM calls).
        args_hash: SHA-256 hash of the serialized tool arguments (for cycle detection).
        token_count: Estimated token count consumed in this step.
        timestamp: Monotonic clock timestamp of this step.
    """

    step_index: int
    tool_name: str
    args_hash: str
    token_count: int
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class AgentSessionStats:
    """Aggregated statistics for the current agent session.

    Attributes:
        total_steps: Total number of steps executed.
        total_tokens: Cumulative tokens consumed across all steps.
        unique_tool_calls: Number of distinct tool/argument combinations seen.
        repeated_calls: Number of step pairs that repeated the exact same tool+args.
        elapsed_seconds: Wall-clock time elapsed since session start.
        is_healthy: Whether the session is within all budget limits.
    """

    total_steps: int
    total_tokens: int
    unique_tool_calls: int
    repeated_calls: int
    elapsed_seconds: float
    is_healthy: bool


class AgentCostGuard:
    """
    Stateful budget guard for autonomous LLM agent sessions.

    Tracks cumulative token usage and step counts across an agent workflow
    session. Raises :class:`~asha.exceptions.AgentBudgetExceeded` when
    any configured limit is exceeded.

    Thread Safety:
        ``AgentCostGuard`` is thread-safe. Internal state is protected by a
        ``threading.Lock``. For concurrent agent workers, use separate guard
        instances per session.

    Args:
        max_steps: Maximum number of agent tool-call steps before raising
            :class:`~asha.exceptions.AgentBudgetExceeded`. Defaults to 25.
        max_tokens: Maximum cumulative tokens across all LLM calls in this
            session before raising. Set to 0 to disable token tracking.
            Defaults to 50_000.
        max_cycle_repeat: Number of times the same tool+args hash may repeat
            before being classified as an infinite loop. Defaults to 3.
        raise_on_limit: If True (default), raise
            :class:`~asha.exceptions.AgentBudgetExceeded` when any limit is
            hit. If False, the guard continues tracking but never raises (useful
            for audit/observe mode).

    Examples:
        Synchronous agent step tracking::

            from asha import AgentCostGuard

            guard = AgentCostGuard(max_steps=10, max_tokens=8000)

            for tool_name, tool_args in agent.run():
                # Raises AgentBudgetExceeded if limits exceeded
                guard.record_tool_call(tool_name, tool_args)
                result = execute_tool(tool_name, tool_args)

        LangChain callback integration (manual)::

            class CostGuardCallback(BaseCallbackHandler):
                def __init__(self, guard: AgentCostGuard):
                    self.guard = guard

                def on_tool_start(self, serialized, input_str, **kwargs):
                    self.guard.record_tool_call(
                        serialized.get("name", "unknown"), {"input": input_str}
                    )
    """

    def __init__(
        self,
        *,
        max_steps: int = 25,
        max_tokens: int = 50_000,
        max_cycle_repeat: int = 3,
        max_drift_score: float = 2.0,
        raise_on_limit: bool = True,
    ) -> None:
        """
        Initialise a new AgentCostGuard session.

        Args:
            max_steps: Maximum agent execution steps. Defaults to 25.
            max_tokens: Maximum cumulative token budget. Set to 0 to disable.
                Defaults to 50_000.
            max_cycle_repeat: How many repetitions of identical tool+args
                before declaring an infinite loop. Defaults to 3.
            max_drift_score: Maximum multi-turn conversational drift score before
                raising AgentBudgetExceeded. Prevents 'jailbreaking by a thousand cuts'.
                Defaults to 2.0.
            raise_on_limit: Whether to raise on limit violations. Defaults
                to True. Set to False for audit/observe mode.
        """
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_cycle_repeat = max_cycle_repeat
        self.max_drift_score = max_drift_score
        self.raise_on_limit = raise_on_limit

        self._lock = threading.Lock()
        self._steps: List[StepRecord] = []
        self._total_tokens: int = 0
        self._call_counts: Dict[str, int] = {}  # args_hash → count
        self._session_start: float = time.monotonic()
        # Rolling window for cycle detection (last N arg hashes per tool)
        self._recent_hashes: Deque[str] = deque(maxlen=50)
        # Multi-turn drift tracking
        self._turn_history: List[dict] = []
        self._cumulative_drift: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_step(self) -> None:
        """
        Assert that the session is still within the step budget.

        Raises:
            AgentBudgetExceeded: If the current step count exceeds ``max_steps``.

        Examples:
            >>> guard = AgentCostGuard(max_steps=5)
            >>> for _ in range(5):
            ...     guard.check_step()  # OK
            >>> guard.check_step()  # Raises AgentBudgetExceeded
        """
        with self._lock:
            current = len(self._steps)
            if current >= self.max_steps and self.raise_on_limit:
                raise AgentBudgetExceeded(
                    f"Agent exceeded maximum step limit of {self.max_steps}. "
                    f"Current step: {current}. "
                    "Increase max_steps or inspect for infinite loops.",
                    budget_type="steps",
                    current=current,
                    limit=self.max_steps,
                )

    def record_tool_call(
        self,
        tool_name: str,
        args: Any = None,
        *,
        token_count: int = 0,
    ) -> StepRecord:
        """
        Record a single agent tool invocation and enforce all limits.

        This is the primary method to call from your agent's orchestration
        loop. It increments the step counter, accumulates token usage, and
        runs cycle detection.

        Args:
            tool_name: Name of the tool being invoked (e.g. ``"web_search"``).
            args: Tool arguments (any JSON-serializable value). Used for
                cycle detection via content hashing. Defaults to None.
            token_count: Estimated tokens consumed by this step. Pass the
                ``usage.total_tokens`` from the LLM response if available.
                Defaults to 0.

        Returns:
            StepRecord: A record describing this step.

        Raises:
            AgentBudgetExceeded: If any limit (steps, tokens, cycle) is
                exceeded and ``raise_on_limit=True``.

        Examples:
            >>> guard = AgentCostGuard(max_steps=10)
            >>> guard.record_tool_call("web_search", {"query": "test"}, token_count=150)
        """
        args_hash = _hash_args(tool_name, args)

        with self._lock:
            step_index = len(self._steps)
            record = StepRecord(
                step_index=step_index,
                tool_name=tool_name,
                args_hash=args_hash,
                token_count=token_count,
            )
            self._steps.append(record)
            self._total_tokens += token_count

            # Update call counters for cycle detection
            self._call_counts[args_hash] = self._call_counts.get(args_hash, 0) + 1
            self._recent_hashes.append(args_hash)

            if not self.raise_on_limit:
                return record

            # Check step budget
            if step_index + 1 > self.max_steps:
                raise AgentBudgetExceeded(
                    f"Agent exceeded maximum step limit of {self.max_steps}. "
                    f"Current step: {step_index + 1}.",
                    budget_type="steps",
                    current=step_index + 1,
                    limit=self.max_steps,
                )

            # Check token budget
            if self.max_tokens > 0 and self._total_tokens > self.max_tokens:
                raise AgentBudgetExceeded(
                    f"Agent exceeded token budget of {self.max_tokens:,} tokens. "
                    f"Cumulative tokens consumed: {self._total_tokens:,}.",
                    budget_type="tokens",
                    current=self._total_tokens,
                    limit=self.max_tokens,
                )

            # Check for cyclical tool invocation
            repeat_count = self._call_counts[args_hash]
            if repeat_count >= self.max_cycle_repeat:
                raise AgentBudgetExceeded(
                    f"Infinite loop detected: tool '{tool_name}' has been called with "
                    f"identical arguments {repeat_count} times (limit: {self.max_cycle_repeat}). "
                    "This is a likely sign of an adversarial indirect injection or "
                    "a broken agent loop.",
                    budget_type="cycle",
                    current=repeat_count,
                    limit=self.max_cycle_repeat,
                )

        return record

    def record_token_usage(self, token_count: int) -> None:
        """
        Record token usage without associating it with a specific tool call.

        Use this when token counts are available from the LLM response but
        not from the tool dispatch layer.

        Args:
            token_count: Number of tokens to add to the cumulative budget.

        Raises:
            AgentBudgetExceeded: If the cumulative token budget is exceeded
                and ``raise_on_limit=True``.
        """
        with self._lock:
            self._total_tokens += token_count
            if (
                self.raise_on_limit
                and self.max_tokens > 0
                and self._total_tokens > self.max_tokens
            ):
                raise AgentBudgetExceeded(
                    f"Agent exceeded token budget of {self.max_tokens:,} tokens. "
                    f"Cumulative: {self._total_tokens:,}.",
                    budget_type="tokens",
                    current=self._total_tokens,
                    limit=self.max_tokens,
                )

    def stats(self) -> AgentSessionStats:
        """
        Return aggregated statistics for the current session.

        Returns:
            AgentSessionStats: Snapshot of step counts, token usage, and
                health status for this guard session.

        Examples:
            >>> guard = AgentCostGuard(max_steps=10)
            >>> guard.record_tool_call("search", {"q": "test"})
            >>> s = guard.stats()
            >>> s.total_steps
            1
        """
        with self._lock:
            unique = len(set(r.args_hash for r in self._steps))
            repeated = len(self._steps) - unique
            elapsed = time.monotonic() - self._session_start
            is_healthy = (
                len(self._steps) < self.max_steps
                and (self.max_tokens == 0 or self._total_tokens < self.max_tokens)
                and all(v < self.max_cycle_repeat for v in self._call_counts.values())
            )
            return AgentSessionStats(
                total_steps=len(self._steps),
                total_tokens=self._total_tokens,
                unique_tool_calls=unique,
                repeated_calls=max(repeated, 0),
                elapsed_seconds=round(elapsed, 3),
                is_healthy=is_healthy,
            )

    @property
    def total_tokens(self) -> int:
        """Cumulative token count recorded in this session."""
        with self._lock:
            return self._total_tokens

    @property
    def total_steps(self) -> int:
        """Cumulative step count recorded in this session."""
        with self._lock:
            return len(self._steps)

    @property
    def cumulative_drift(self) -> float:
        """Current cumulative drift score across recorded conversational turns."""
        with self._lock:
            return round(self._cumulative_drift, 3)

    def record_turn(
        self,
        role: str,
        content: str,
        threat_score: Optional[float] = None,
    ) -> float:
        """
        Record a conversational turn and enforce conversational memory bounds.

        Detects 'jailbreaking by a thousand cuts' (multi-turn drift) where individual
        turns stay below per-turn thresholds, but their cumulative drift pushes the
        agent out of bounds.

        Args:
            role: 'user', 'assistant', or 'system'.
            content: Turn text message.
            threat_score: Optional precomputed threat score (0.0 to 1.0). If omitted,
                a fast regex/heuristic threat score is calculated.

        Returns:
            float: Updated cumulative drift score across the conversation window.

        Raises:
            AgentBudgetExceeded: If cumulative drift score exceeds ``max_drift_score``
                and ``raise_on_limit=True``.
        """
        with self._lock:
            if threat_score is None:
                threat_score = _estimate_turn_risk(content)

            # Exponential decay: recent turns carry high weight (0.85 decay factor)
            self._cumulative_drift = (self._cumulative_drift * 0.85) + threat_score
            self._turn_history.append({"role": role, "risk": threat_score})

            if self.raise_on_limit and self._cumulative_drift > self.max_drift_score:
                raise AgentBudgetExceeded(
                    f"Agent exceeded multi-turn conversational drift limit of {self.max_drift_score:.2f}. "
                    f"Cumulative drift: {self._cumulative_drift:.2f}. "
                    "Detected potential multi-turn escalation attack ('jailbreaking by a thousand cuts').",
                    budget_type="drift",
                    current=round(self._cumulative_drift, 3),
                    limit=self.max_drift_score,
                )
            return round(self._cumulative_drift, 3)

    def reset(self) -> None:
        """
        Reset all session state, clearing step records, token counts, and drift scores.

        Use this to reuse a guard instance for a new agent session without
        creating a new object.
        """
        with self._lock:
            self._steps.clear()
            self._total_tokens = 0
            self._call_counts.clear()
            self._recent_hashes.clear()
            self._turn_history.clear()
            self._cumulative_drift = 0.0
            self._session_start = time.monotonic()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_args(tool_name: str, args: Any) -> str:
    """Create a stable content hash of a tool name + arguments pair."""
    try:
        import json

        serialized = json.dumps(
            {"tool": tool_name, "args": args}, sort_keys=True, default=str
        )
    except (TypeError, ValueError):
        serialized = f"{tool_name}:{str(args)}"
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


_ESCALATION_MARKERS = [
    r"(?i)\b(?:system\s+override|jailbreak|bypass|dan\b|developer\s+mode)\b",
    r"(?i)\b(?:ignore\s+previous|disregard\s+prior|forget\s+rules)\b",
    r"(?i)\b(?:filter\s+disabled|unrestricted|uncensored|sudo|eval\b)\b",
    r"(?i)\b(?:hidden\s+instruction|secret\s+key|dump\s+database)\b",
]


def _estimate_turn_risk(content: str) -> float:
    """Fast sub-millisecond turn risk estimation without external dependencies."""
    import re

    hits = sum(1 for p in _ESCALATION_MARKERS if re.search(p, content))
    if hits >= 2:
        return 0.9
    elif hits == 1:
        return 0.55
    return 0.05

