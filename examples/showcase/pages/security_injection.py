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

"""Prompt injection and sanitization security testing lab."""

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

from asha import PolicyConfig, sanitize
from asha.core.text.canonicalize import canonicalize
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.diff_view import render_diff
from examples.showcase.components.header import render_header
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.status_badge import get_status_badge_html
from examples.showcase.scenarios.prompt_injection import INJECTION_SCENARIOS


def render_injection_page() -> None:
    """Render the Prompt Injection and Attack Sanitization Lab."""
    render_header(title="PROMPT INJECTION LAB", subtitle="Adversarial prompt evaluation, jailbreak defense, and canonicalized threat scoring")

    col_scen, col_mode = st.columns([1.5, 1])
    with col_scen:
        scenario_key = st.selectbox("Select Attack Scenario", list(INJECTION_SCENARIOS.keys()), index=0)
    with col_mode:
        mode = st.selectbox("Policy Enforcement Mode", ["balanced", "strict", "lite", "off"], index=0)

    default_text = INJECTION_SCENARIOS[scenario_key]
    prompt_text = st.text_area("Attack Payload / Prompt (Editable)", value=default_text, height=130)

    run_btn = st.button("Evaluate Threat with asha.sanitize()", type="primary", use_container_width=True)

    if run_btn or "last_injection_run" in st.session_state:
        if run_btn:
            policy = PolicyConfig(mode=mode)
            san_res = sanitize(prompt_text, policy=policy)
            canon_text = canonicalize(prompt_text)
            st.session_state["last_injection_run"] = {
                "raw": prompt_text,
                "san_res": san_res,
                "canon": canon_text,
                "mode": mode,
            }

        data = st.session_state["last_injection_run"]
        san_res = data["san_res"]
        sec = getattr(san_res, "security_result", None) or getattr(san_res, "security", None)

        is_safe = getattr(sec, "safe", True) if sec else True
        if hasattr(sec, "is_safe"):
            is_safe = bool(sec.is_safe)
        threat_level = str(getattr(sec, "threat_level", "low")).upper()
        threats = list(getattr(sec, "threats", []) or [])
        score = float(getattr(sec, "security_score", 1.0) if sec else 1.0)

        decision = "BLOCK" if not is_safe else "ALLOW"
        if is_safe and threat_level in ("MEDIUM", "WARN", "REVIEW"):
            decision = "REVIEW / SANITIZE"

        st.markdown("---")
        st.markdown("### Security Evaluation Summary")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Policy Decision:**<br>{get_status_badge_html(decision)}", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Threat Level:**<br>{get_status_badge_html(threat_level)}", unsafe_allow_html=True)
        with c3:
            st.metric("Threat Count", len(threats))
        with c4:
            st.metric("Confidence / Score", f"{score:.2f}")

        if threats:
            st.markdown(
                f"""
                <div class="console-alert-danger">
                    <strong>Threat Indicators Flagged:</strong> {', '.join(str(t) for t in threats)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not is_safe:
            st.markdown(
                f"""
                <div class="console-alert-danger">
                    <strong>Verdict:</strong> Input exceeds safety threshold for policy mode '{data['mode']}'.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(63, 185, 80, 0.08); border: 1px solid rgba(63, 185, 80, 0.3); border-radius: 6px; padding: 0.8rem 1rem; color: #3fb950; font-size: 0.85rem; margin: 0.8rem 0;">
                    <strong>Safe Content:</strong> No injection patterns or malicious overrides detected.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Text Transformations")
        tab_raw, tab_canon, tab_san = st.tabs(["Raw Input", "Canonicalized Text", "Sanitized Output"])
        with tab_raw:
            st.code(data["raw"], language="text")
        with tab_canon:
            st.caption("Normalized using NFKC + confusable homoglyph unfolding + zero-width stripping:")
            st.code(data["canon"], language="text")
        with tab_san:
            st.code(san_res.output, language="text")

        render_diff(data["raw"], san_res.output, title_original="Original Attack Payload", title_modified="Sanitized Output")

        render_object_inspector(san_res, title="SanitizeResult Technical Breakdown")
        render_copy_python(operation="sanitize", prompt=data["raw"], mode=data["mode"])

render = render_injection_page

if __name__ == "__main__":
    render()
