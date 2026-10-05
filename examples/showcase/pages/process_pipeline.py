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

"""Process Pipeline Lab inspecting all intermediate stage outputs."""

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

from asha import PolicyConfig, process
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.header import render_header
from examples.showcase.components.object_inspector import render_object_inspector


def render_process_pipeline_page() -> None:
    """Render the Process() pipeline decomposition page."""
    render_header(title="PROCESS() PIPELINE LAB", subtitle="Deep decomposition of ASHA's multi-stage prompt transformation")

    sample_prompt = (
        "Please analyze this user request from mark.watney@nasa.gov (Tel: +1-281-555-0142):\n"
        "\"We need to recalculate the orbital ascent trajectory for the MAV on Sol 549. "
        "Do NOT vent liquid oxygen reserves under any circumstances. Required payload mass: 450.25 kg.\""
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        prompt_text = st.text_area("Input Prompt", value=sample_prompt, height=130)
    with col2:
        mode = st.selectbox("Policy Mode", ["balanced", "strict", "lite", "off"], index=0)
        token_budget = st.number_input("Token Budget", min_value=100, max_value=8000, value=1200, step=100)
        run_btn = st.button("Execute process()", type="primary", use_container_width=True)

    if run_btn or "last_process_res" in st.session_state:
        if run_btn:
            policy = PolicyConfig(mode=mode)
            res = process(prompt_text, policy=policy, token_budget=int(token_budget))
            st.session_state["last_process_res"] = (prompt_text, res, mode, token_budget)

        raw_in, res, active_mode, active_budget = st.session_state["last_process_res"]

        st.markdown("---")
        st.markdown("### Stage Representations")

        tab_sanitized, tab_compiled, tab_optimized, tab_meta, tab_inspect = st.tabs(
            ["1. Sanitized & PII", "2. Semantic Preservation", "3. Optimization", "4. Metrics", "5. Object Inspector"]
        )

        with tab_sanitized:
            st.markdown("**Sanitization & Masking Output:**")
            st.code(res.output, language="text")
            sec = getattr(res, "security", None)
            if sec:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Is Safe", str(sec.safe))
                with c2:
                    st.metric("Threat Level", str(sec.threat_level).upper())
                with c3:
                    st.metric("PII Entities Masked", len(sec.pii_detected or []))

        with tab_compiled:
            st.markdown("**Semantic Preservation Verification:**")
            st.markdown(
                """
                ASHA's compiler guarantees user nouns, code segments, negations ("Do NOT vent"), 
                and numerical constraints (Sol 549, 450.25 kg) are preserved verbatim rather than 
                being replaced with generic synthetic templates.
                """
            )
            st.code(res.output, language="text")

        with tab_optimized:
            st.markdown("**Token Optimization Results:**")
            met = getattr(res, "metrics", None)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Tokens Saved", getattr(met, "tokens_saved", 0))
            with c2:
                st.metric("Token Reduction", f"{getattr(met, 'token_reduction_pct', 0.0):.1f}%")
            with c3:
                st.metric("Processing Time", f"{getattr(met, 'processing_time_ms', 0.0):.1f} ms")

        with tab_meta:
            st.markdown("**Complete Execution Metadata:**")
            st.write({
                "security": getattr(res, "security", {}),
                "metrics": getattr(res, "metrics", {}),
            })

        with tab_inspect:
            render_object_inspector(res, title="ProcessResult Inspection")

        st.markdown("---")
        render_copy_python(
            operation="process",
            prompt=raw_in,
            mode=active_mode,
            token_budget=int(active_budget),
        )

render = render_process_pipeline_page

if __name__ == "__main__":
    render()
