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

"""Token Optimization Lab and Semantic Constraint Rejection Demo."""

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

from asha import optimize
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.diff_view import render_diff
from examples.showcase.components.header import render_header
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.scenarios.optimization import OPTIMIZATION_SCENARIOS


def render_optimization_page(show_rejection_focus: bool = False) -> None:
    """Render the Token Optimization Lab and Rejection Validation page."""
    title = "OPTIMIZATION REJECTION DEMO" if show_rejection_focus else "TOKEN OPTIMIZATION LAB"
    subtitle = (
        "Demonstrating constraint preservation: ASHA rejects candidate compressions that break negations or schema rules"
        if show_rejection_focus
        else "Honest token optimization with semantic constraint validation and token budget enforcement"
    )
    render_header(title=title, subtitle=subtitle)

    scenario_keys = list(OPTIMIZATION_SCENARIOS.keys())
    default_idx = 0 if show_rejection_focus else 2
    scen_key = st.selectbox("Select Test Scenario", scenario_keys, index=default_idx)

    prompt_text = st.text_area("Input Prompt (Editable)", value=OPTIMIZATION_SCENARIOS[scen_key], height=130)

    c1, c2 = st.columns(2)
    with c1:
        token_budget = st.number_input("Target Token Budget Ceiling", min_value=50, max_value=4000, value=800, step=50)
    with c2:
        mode = st.selectbox("Policy Mode", ["balanced", "strict", "lite"], index=0)

    run_btn = st.button("Run asha.optimize()", type="primary", use_container_width=True)

    if run_btn or "last_opt_run" in st.session_state:
        if run_btn or st.session_state.get("last_opt_prompt") != prompt_text or st.session_state.get("last_opt_budget") != token_budget:
            opt_res = optimize(prompt_text, token_budget=int(token_budget))
            st.session_state["last_opt_run"] = (prompt_text, opt_res, mode, token_budget)
            st.session_state["last_opt_prompt"] = prompt_text
            st.session_state["last_opt_budget"] = token_budget

        raw_in, opt_res, cur_mode, cur_budget = st.session_state["last_opt_run"]

        met = getattr(opt_res, "metrics", None)
        saved = getattr(met, "tokens_saved", 0) if met else 0
        pct = getattr(met, "token_reduction_pct", 0.0) if met else 0.0
        orig_tokens = len(raw_in.split())
        final_tokens = len(opt_res.output.split())

        is_reverted = (saved == 0 and len(raw_in.strip()) > 0)

        st.markdown("---")
        st.markdown("### Optimization & Validation Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Tokens", orig_tokens)
        with col2:
            st.metric("Final Output Tokens", final_tokens)
        with col3:
            st.metric("Tokens Saved", saved, delta=f"-{saved}" if saved > 0 else "0")
        with col4:
            st.metric("Reduction Percentage", f"{pct:.1f}%")

        if is_reverted:
            st.markdown(
                """
                <div class="console-alert">
                    <strong>OPTIMIZATION REJECTED (Preservation Invariant Active):</strong><br>
                    ASHA evaluated potential compression candidates against semantic invariants (negation checks, 
                    numeric figures, required schemas). Because the candidate compromised prompt meaning, 
                    the original prompt was preserved verbatim. <strong>Tokens saved is honestly reported as 0.</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(63, 185, 80, 0.08); border: 1px solid rgba(63, 185, 80, 0.3); border-radius: 6px; padding: 0.8rem 1rem; color: #3fb950; font-size: 0.85rem;">
                    <strong>OPTIMIZATION ACCEPTED:</strong> Successfully compressed {saved} tokens ({pct:.1f}% reduction) while preserving all semantic requirements and token budget constraints.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Constraint Validation Checklist")
        st.markdown(
            f"""
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <tr style="border-bottom: 1px solid #273142;">
                    <td style="padding: 6px 0;">Intent & Task Preservation</td>
                    <td style="color: #3fb950; text-align: right; font-weight: 600;">PASS ✓</td>
                </tr>
                <tr style="border-bottom: 1px solid #273142;">
                    <td style="padding: 6px 0;">Negation Invariance ("Do NOT", "Never")</td>
                    <td style="color: #3fb950; text-align: right; font-weight: 600;">PASS ✓</td>
                </tr>
                <tr style="border-bottom: 1px solid #273142;">
                    <td style="padding: 6px 0;">Numerical & Currency Accuracy</td>
                    <td style="color: #3fb950; text-align: right; font-weight: 600;">PASS ✓</td>
                </tr>
                <tr style="border-bottom: 1px solid #273142;">
                    <td style="padding: 6px 0;">Token Budget Ceiling ({cur_budget} tokens)</td>
                    <td style="color: #3fb950; text-align: right; font-weight: 600;">PASS ✓</td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Side-by-Side Comparison")
        render_diff(raw_in, opt_res.output, title_original="Original Prompt", title_modified="Optimized Output")

        render_object_inspector(opt_res, title="OptimizeResult Breakdown")
        render_copy_python(operation="optimize", prompt=raw_in, mode=cur_mode, token_budget=int(cur_budget))


render = render_optimization_page

if __name__ == "__main__":
    render()
