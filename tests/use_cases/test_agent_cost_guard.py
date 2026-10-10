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

"""Real-world use-case tests: Agent loop cost and runaway cycle prevention via AgentCostGuard."""

from __future__ import annotations

import pytest
from asha import AgentCostGuard, AgentBudgetExceeded


def test_agent_cost_guard_enforces_step_limit() -> None:
    """AgentCostGuard stops autonomous loops that exceed maximum permitted execution steps."""
    guard = AgentCostGuard(max_steps=3, raise_on_limit=True)

    # Steps 1 to 3 are permitted
    guard.record_tool_call("search_docs", {"query": "Q1 revenue"})
    guard.record_tool_call("read_table", {"table": "financials"})
    guard.record_tool_call("format_markdown", {"style": "table"})

    # Step 4 exceeds max_steps=3
    with pytest.raises(AgentBudgetExceeded, match="step limit"):
        guard.record_tool_call("send_summary", {"to": "team"})


def test_agent_cost_guard_enforces_cumulative_token_budget() -> None:
    """AgentCostGuard accumulates tokens across LLM interactions and blocks token avalanches."""
    guard = AgentCostGuard(max_tokens=1000, raise_on_limit=True)

    guard.record_token_usage(450)
    guard.record_token_usage(450)

    assert guard.total_tokens == 900

    # Adding 200 tokens pushes cumulative count to 1100 > 1000 limit
    with pytest.raises(AgentBudgetExceeded, match="token budget"):
        guard.record_token_usage(200)


def test_agent_cost_guard_detects_infinite_tool_repetition_cycle() -> None:
    """AgentCostGuard detects repetitive identical tool calls with identical arguments."""
    guard = AgentCostGuard(max_cycle_repeat=3, raise_on_limit=True)

    # Identical tool + args repeated
    guard.record_tool_call("web_search", {"q": "python tutorial"})
    guard.record_tool_call("web_search", {"q": "python tutorial"})

    # 3rd identical call triggers loop detection
    with pytest.raises(AgentBudgetExceeded, match="loop"):
        guard.record_tool_call("web_search", {"q": "python tutorial"})


def test_agent_cost_guard_audit_mode_tracks_without_raising() -> None:
    """When raise_on_limit=False, the guard logs and tracks violations without interrupting execution."""
    guard = AgentCostGuard(max_steps=2, raise_on_limit=False)

    guard.record_tool_call("tool1", {})
    guard.record_tool_call("tool2", {})
    # 3rd call exceeds limit but must not raise in audit mode
    guard.record_tool_call("tool3", {})

    assert guard.total_steps == 3
    assert guard.stats().is_healthy is False
