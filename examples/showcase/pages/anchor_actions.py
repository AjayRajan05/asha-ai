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

"""Action Governance Lab demonstrating normalization and anti-evasion."""

from __future__ import annotations

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
from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import get_status_badge_html


ACTION_VARIANTS = {
    "Direct Bash rm -rf": ("bash_command", {"command": "rm -rf /var/data"}),
    "Shell Tool rm -rf": ("shell", {"cmd": "rm -rf /var/data"}),
    "Evasive 'command_execution' wrapper": ("command_execution", {"tool": "terminal", "arg": "rm -rf /var/data"}),
    "Generic 'tool_call' terminal": ("tool_call", {"tool_name": "terminal", "command": "rm -rf /var/data"}),
    "Safe tool_call file_read": ("tool_call", {"tool_name": "read_file", "path": "summary.txt"}),
}


def render_action_governance_page() -> None:
    """Render the Action Governance and Normalization Anti-Evasion Lab."""
    render_header(title="ACTION GOVERNANCE & NORMALIZATION", subtitle="Anti-evasion demonstration: action normalization prevents bypass via payload or type renaming")

    st.markdown(
        """
        Adversaries frequently attempt to bypass tool guards by invoking destructive commands under alternative action keys 
        (e.g., calling <code>bash_command</code> or <code>shell</code> instead of standard <code>tool_call</code>). 
        ANCHOR normalizes all heterogeneous actions into uniform tool evaluations before authorization checks.
        """,
        unsafe_allow_html=True,
    )

    scen_key = st.selectbox("Select Action Variant", list(ACTION_VARIANTS.keys()), index=0)
    raw_type, raw_payload = ACTION_VARIANTS[scen_key]

    c1, c2 = st.columns(2)
    with c1:
        act_type = st.text_input("Action Type Key", value=raw_type)
    with c2:
        cmd_text = st.text_input("Target Command / Path", value=str(raw_payload.get("command") or raw_payload.get("cmd") or raw_payload.get("path") or raw_payload.get("arg")))

    if st.button("Evaluate Action Invariance", type="primary", use_container_width=True):
        runtime = AnchorRuntime(warn_policy="strict", interactive=False, risk_tolerance="LOW")
        runtime.initialize_mission("Perform read-only data analysis in workspace.")

        payload = {"command": cmd_text, "tool_name": "terminal" if "rm -rf" in cmd_text else "read_file"}

        # Run normalization
        norm_event = runtime.action_guard.normalize_action(act_type, payload)

        is_allowed = False
        blocked_reason = ""
        try:
            is_allowed = runtime.evaluate_action_request(act_type, payload, raise_on_block=True)
        except ASHAAnchorBlocked as err:
            is_allowed = False
            blocked_reason = str(err)

        st.markdown("---")
        st.markdown("### Normalization & Decision Pipeline")

        render_action_normalization_pipeline(
            raw_action_type=act_type,
            raw_payload=payload,
            normalized_tool_name=norm_event.payload.get("tool_name", "unknown"),
            normalized_arguments=str(norm_event.payload),
            risk_score=0.98 if not is_allowed else 0.05,
            verdict="ALLOW" if is_allowed else "BLOCK",
            reason=blocked_reason or "Normalized action satisfies read-only mission bounds.",
            executed=is_allowed,
        )

        if not is_allowed:
            st.markdown(
                """
                <div class="console-alert-danger">
                    <strong>Bypass Thwarted:</strong> Regardless of action type naming (<code>bash_command</code> vs <code>shell</code> vs <code>command_execution</code>), 
                    destructive patterns are intercepted by ASHA's action normalizer and hard-blocked.
                </div>
                """,
                unsafe_allow_html=True,
            )
            render = render_action_governance_page

render = render_action_governance_page

if __name__ == "__main__":
    render()
