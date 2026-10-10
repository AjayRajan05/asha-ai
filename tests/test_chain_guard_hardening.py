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

"""Chain guard capability + Markov transition tests (step 5)."""

import time

import pytest

from asha.runtime.anchor.chain_guard import ChainGuard
from asha.runtime.anchor.contracts import MissionContract
from asha.runtime.anchor.tool_capabilities import (
    ToolCapabilities,
    categorize_tool,
    clear_tool_capabilities,
    register_tool,
)
from asha.runtime.anchor.transition_model import (
    TransitionModel,
    get_transition_model,
    score_category_sequence,
)
from asha.runtime.anchor.types import ActionEvent
from asha.runtime.anchor.verdicts import Verdict
from asha.core.ml.calibration import Verdict as MLVerdict


def _contract(**overrides) -> MissionContract:
    base = dict(
        mission_id="t",
        goal="t",
        intent_summary="t",
        allowed_actions=["read", "write"],
        forbidden_actions=[],
        allowed_tools=[],
        allowed_domains=[],
        allowed_memory_scopes=["session"],
        required_resources=[],
        expected_outcomes=[],
        completion_criteria=[],
        risk_tolerance="MEDIUM",
        created_at=time.time(),
        local_only=True,
        forbid_network_exfiltration=True,
    )
    base.update(overrides)
    return MissionContract(**base)  # type: ignore[arg-type]


def _action(tool_name: str) -> ActionEvent:
    return ActionEvent(
        action_id="a",
        action_type="tool_call",
        payload={"tool_name": tool_name, "arguments": "{}"},
        timestamp=0.0,
    )


def test_register_tool_capabilities_drives_category():
    clear_tool_capabilities()
    register_tool(
        "custom_blob_shipper",
        network_egress=True,
        category="network_egress",
    )
    assert categorize_tool("custom_blob_shipper") == "network_egress"
    clear_tool_capabilities()


def test_markov_blocks_read_to_network_egress():
    get_transition_model(reset=True)
    score = score_category_sequence(["read", "network_egress"])
    assert score["verdict"] in (MLVerdict.BLOCK, MLVerdict.REVIEW)
    assert score["min_transition_probability"] < 0.05


def test_markov_allows_benign_read_analyze_write():
    get_transition_model(reset=True)
    score = score_category_sequence(["read", "analyze", "write"])
    assert score["verdict"] is MLVerdict.SAFE
    assert score["min_transition_probability"] >= 0.05


def test_chain_guard_uses_capabilities_not_just_name_prefix():
    """Tool named without send_/read_ prefixes still flagged via caps."""
    clear_tool_capabilities()
    register_tool("ingest_records", reads_data=True, category="read")
    register_tool("push_elsewhere", network_egress=True, category="network_egress")
    guard = ChainGuard()
    history = [_action("ingest_records"), _action("push_elsewhere")]
    verdict = guard.evaluate_chain(history, _contract(local_only=True))
    assert verdict.verdict in (Verdict.BLOCK, Verdict.REVIEW)
    clear_tool_capabilities()


def test_chain_guard_legacy_named_exfil_still_blocked():
    guard = ChainGuard()
    history = [_action("load_trend_data"), _action("send_email")]
    verdict = guard.evaluate_chain(history, _contract(local_only=True))
    assert verdict.verdict in (Verdict.BLOCK, Verdict.REVIEW)


def test_benign_analyze_chain_allowed():
    clear_tool_capabilities()
    register_tool("load_x", reads_data=True, category="read")
    register_tool("analyze_x", category="analyze")
    register_tool("write_x", writes_data=True, category="write")
    guard = ChainGuard()
    history = [_action("load_x"), _action("analyze_x"), _action("write_x")]
    verdict = guard.evaluate_chain(
        history, _contract(local_only=True, forbid_network_exfiltration=False)
    )
    assert verdict.verdict is Verdict.ALLOW
    clear_tool_capabilities()


def test_write_then_delete_blocked_via_capabilities():
    clear_tool_capabilities()
    register_tool("persist_doc", writes_data=True, category="write")
    register_tool("wipe_doc", destructive=True, category="delete")
    guard = ChainGuard()
    # Even if Markov is soft, capability hard-rule should fire.
    history = [_action("persist_doc"), _action("wipe_doc")]
    verdict = guard.evaluate_chain(history, _contract())
    assert verdict.verdict is Verdict.BLOCK
    clear_tool_capabilities()


def test_transition_model_loads_from_generators():
    model = get_transition_model(reset=True)
    assert isinstance(model, TransitionModel)
    # Generator matrix has read→analyze as a common transition.
    assert model.transition_probability("read", "analyze") > 0.1
