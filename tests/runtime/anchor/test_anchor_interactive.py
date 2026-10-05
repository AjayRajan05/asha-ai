"""Tests for interactive ANCHOR approval."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from asha.exceptions import ASHAAnchorBlocked
from asha.runtime.anchor.runtime import AnchorRuntime


def test_interactive_block_can_be_overridden_by_human() -> None:
    runtime = AnchorRuntime(warn_policy="strict", interactive=True)
    runtime.initialize_mission(
        "Analyze trends locally. Do not send data externally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    with patch("builtins.input", return_value="y"):
        allowed = runtime.evaluate_action_request(
            "tool_call",
            {"tool_name": "network_exfil", "arguments": "{}"},
        )

    assert allowed is True


def test_interactive_block_rejects_when_human_denies() -> None:
    runtime = AnchorRuntime(warn_policy="strict", interactive=True)
    runtime.initialize_mission(
        "Analyze trends locally. Do not send data externally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    with patch("builtins.input", return_value="n"):
        allowed = runtime.evaluate_action_request(
            "tool_call",
            {"tool_name": "network_exfil", "arguments": "{}"},
        )

    assert allowed is False


def test_raise_on_block_after_human_denies() -> None:
    runtime = AnchorRuntime(warn_policy="strict", interactive=True)
    runtime.initialize_mission(
        "Analyze trends locally.",
        context={"available_tools": ["load_trend_data"], "local_only": True},
    )

    with patch("builtins.input", return_value="n"):
        with pytest.raises(ASHAAnchorBlocked):
            runtime.evaluate_action_request(
                "tool_call",
                {"tool_name": "network_exfil", "arguments": "{}"},
                raise_on_block=True,
            )

