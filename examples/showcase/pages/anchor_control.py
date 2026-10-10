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

"""ANCHOR Control Room: Mission compilation and runtime governance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from asha.exceptions import ASHAAnchorBlocked
from asha.runtime.anchor.runtime import AnchorRuntime
from examples.showcase.components.action_viewer import render_action_normalization_pipeline
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import get_status_badge_html
from examples.showcase.scenarios.anchor import ANCHOR_SCENARIOS


def render_anchor_control_page() -> None:
    """Render the ANCHOR mission governance control room."""
    render_header(title="ANCHOR CONTROL ROOM", subtitle="Agent Navigation, Control, History, Oversight and Runtime (ANCHOR)")

    scen_key = st.selectbox("Select Scenario Preset", list(ANCHOR_SCENARIOS.keys()), index=0)
    scen = ANCHOR_SCENARIOS[scen_key]

    mission_text = st.text_area("Mission Definition (Immutable Contract Goal)", value=scen["mission"], height=70)

    c1, c2 = st.columns([1, 1])
    with c1:
        action_type = st.text_input("Incoming Action Type", value=scen["action_type"])
    with c2:
        payload_str = st.text_area("Action Arguments / Payload (JSON)", value=json.dumps(scen["payload"], indent=2), height=100)

    try:
        payload = json.loads(payload_str)
    except Exception:
        payload = {"raw": payload_str}

    col_btn, col_pol = st.columns([1, 1])
    with col_pol:
        warn_policy = st.selectbox("Warn Policy", ["strict", "permissive"], index=0)
    with col_btn:
        st.write("")
        st.write("")
        eval_btn = st.button("Evaluate Action Against Mission", type="primary", use_container_width=True)

    if eval_btn or "last_anchor_run" in st.session_state:
        if eval_btn:
            runtime = AnchorRuntime(warn_policy=warn_policy, interactive=False, risk_tolerance="LOW")
            runtime.initialize_mission(
                mission_text,
                context={"available_tools": ["read_file", "write_report", "calculate", "search_flights"], "local_only": ("local" in mission_text.lower())},
            )

            is_allowed = False
            error_reason = ""
            try:
                is_allowed = runtime.evaluate_action_request(
                    action_type,
                    payload,
                    raise_on_block=True,
                )
            except ASHAAnchorBlocked as blocked_err:
                is_allowed = False
                error_reason = str(blocked_err)

            st.session_state["last_anchor_run"] = {
                "mission": mission_text,
                "action_type": action_type,
                "payload": payload,
                "is_allowed": is_allowed,
                "error_reason": error_reason,
                "runtime": runtime,
            }

        res = st.session_state["last_anchor_run"]
        runtime = res["runtime"]
        state = runtime.state
        mission_obj = state.mission

        st.markdown("---")
        st.markdown("### ANCHOR Mission Contract Status")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Risk Tolerance", mission_obj.risk_tolerance if mission_obj else "LOW")
        with mc2:
            st.metric("Isolation Scope", "Local-Only" if mission_obj and mission_obj.local_only else "Standard")
        with mc3:
            st.markdown(f"**Governance Decision:**<br>{get_status_badge_html('ALLOW' if res['is_allowed'] else 'BLOCK')}", unsafe_allow_html=True)
        with mc4:
            st.metric("Execution State", "EXECUTED (Simulated)" if res["is_allowed"] else "BLOCKED (Denied)")

        # Render complete normalization pipeline
        norm_tool = payload.get("tool_name") or payload.get("tool") or payload.get("function") or res["action_type"]
        render_action_normalization_pipeline(
            raw_action_type=res["action_type"],
            raw_payload=res["payload"],
            normalized_tool_name=str(norm_tool),
            normalized_arguments=json.dumps(res["payload"]),
            risk_score=0.95 if not res["is_allowed"] else 0.05,
            verdict="ALLOW" if res["is_allowed"] else "BLOCK",
            reason=res["error_reason"] or "Action strictly aligns with mission goals and safety policies.",
            executed=res["is_allowed"],
        )

        render_copy_python(operation="anchor")


render = render_anchor_control_page

if __name__ == "__main__":
    render()

