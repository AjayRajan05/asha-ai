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

"""LLM Playground with 3-panel architecture and transparency modes."""

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

from asha import PolicyConfig
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.diff_view import render_diff
from examples.showcase.components.header import render_header
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.pipeline_view import render_pipeline
from examples.showcase.components.trace_viewer import render_trace_viewer
from examples.showcase.runtime.executor import ExecutionResult, PipelineExecutor
from examples.showcase.runtime.session import add_trace_to_history, get_provider_status


def render_playground_page() -> None:
    """Render the primary 3-panel LLM Playground."""
    render_header(title="LLM PLAYGROUND", subtitle="Inspect user input, live ASHA security/optimization stages, and downstream LLM inference")

    transparency = st.session_state.get("transparency_mode", "Transparent")
    providers = get_provider_status()

    # Layout: Left (Input), Center (Pipeline), Right (Output)
    col_input, col_pipeline, col_output = st.columns([1.1, 1.2, 1.2])

    with col_input:
        st.markdown("#### 1. User Input")

        sample_prompt = (
            "Please review the customer inquiry from Jane Doe (jane.doe@example.com, phone: +1-555-0199):\n"
            "\"I received my order #48102 yesterday, but the glass frame is cracked. "
            "Can I return it for a replacement under the 30-day warranty?\""
        )
        prompt_text = st.text_area("User Prompt", value=sample_prompt, height=180)

        with st.expander("System Instruction & Controls", expanded=False):
            system_inst = st.text_input("System Instruction", value="You are a helpful customer service assistant.")
            prov_options = list(providers.keys())
            active_p = st.session_state.get("selected_provider", "local")
            p_index = prov_options.index(active_p) if active_p in prov_options else 0
            selected_provider_key = st.selectbox(
                "Inference Provider",
                options=prov_options,
                format_func=lambda k: f"{providers[k]['name']} ({'Ready' if providers[k].get('configured') else 'Not Configured'})",
                index=p_index,
            )
            provider_models = providers[selected_provider_key].get("models", ["default"])
            selected_model = st.selectbox("Model", options=provider_models, index=0)

            c_mode, c_budget = st.columns(2)
            with c_mode:
                policy_mode = st.selectbox("Policy Mode", ["balanced", "strict", "lite", "off"], index=0)
            with c_budget:
                token_budget = st.number_input("Token Budget", min_value=100, max_value=8000, value=1200, step=100)

        run_btn = st.button("Run ASHA Pipeline", type="primary", use_container_width=True)

    # Execution state in session
    if run_btn:
        policy = PolicyConfig(mode=policy_mode)
        with st.spinner("Processing through ASHA security & optimization layers..."):
            exec_res = PipelineExecutor.execute_pipeline(
                prompt=prompt_text,
                system_instruction=system_inst,
                provider_name=selected_provider_key,
                model_name=selected_model,
                policy=policy,
                token_budget=int(token_budget),
            )
            add_trace_to_history(exec_res.trace)
            st.session_state["last_playground_result"] = exec_res

    res: ExecutionResult | None = st.session_state.get("last_playground_result")

    # Center Panel: Pipeline
    with col_pipeline:
        st.markdown("#### 2. Live ASHA Pipeline")
        if res:
            stages_status = {
                "INPUT": "done",
                "SANITIZE": "blocked" if not res.is_safe else "done",
                "PII": "done",
                "COMPILE": "done",
                "OPTIMIZE": "done",
                "POLICY": "blocked" if not res.is_safe else "done",
                "LLM": "blocked" if not res.is_safe else "done",
                "RESULT": "blocked" if not res.is_safe else "done",
            }
            render_pipeline(stages_status=stages_status, blocked_stage="SANITIZE" if not res.is_safe else None)

            st.markdown(
                f"""
                <div style="background: #121820; border: 1px solid #273142; border-radius: 6px; padding: 0.8rem; font-size: 0.82rem; margin-bottom: 0.8rem;">
                    <div><strong>Verdict:</strong> <span class="pill pill-{'allow' if res.is_safe else 'block'}">{'SAFE / ALLOWED' if res.is_safe else 'BLOCKED'}</span></div>
                    <div style="margin-top: 4px;"><strong>PII Masked:</strong> {res.pii_entities_count} entities</div>
                    <div style="margin-top: 4px;"><strong>Tokens Saved:</strong> {res.tokens_saved} tokens</div>
                    <div style="margin-top: 4px;"><strong>ASHA Latency:</strong> {res.trace.asha_latency_ms:.1f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detailed intermediate representation if in Transparent or Forensic mode
            if transparency in ("Transparent", "Forensic") and res.raw_result:
                st.markdown("**Protected Prompt (Dispatched to Model):**")
                st.code(res.raw_result.output, language="text")
        else:
            render_pipeline()
            st.info("Awaiting execution. Click 'Run ASHA Pipeline' to observe the live transformation stages.")

    # Right Panel: Output
    with col_output:
        st.markdown("#### 3. Output & Response")
        if res:
            if not res.is_safe:
                st.error(f"Action Blocked by ASHA Policy: {res.blocked_reason}")
                st.markdown(
                    """
                    <div class="console-alert-danger">
                        <strong>Security Invariant Triggered:</strong> ASHA prevented dangerous or adversarial payload from reaching downstream inference.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: #121820; border: 1px solid #273142; border-radius: 6px; padding: 0.8rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.76rem; color: #8b949e; margin-bottom: 4px;">
                            Provider: <span style="color: #38bdf8; font-weight: 600;">{res.trace.provider}</span> ({res.trace.model}) | Latency: {res.trace.provider_latency_ms:.1f}ms
                        </div>
                        <div style="font-size: 0.88rem; color: #e6edf3; white-space: pre-wrap; word-break: break-word;">{res.output_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Forensic view: full trace inspectable right below
            if transparency == "Forensic":
                with st.expander("Forensic Trace & Object Inspection", expanded=True):
                    render_trace_viewer(res.trace)
                    render_object_inspector(res.raw_result, title="Raw ProcessResult Object")
        else:
            st.caption("Downstream inference response and metadata will appear here.")

    # Bottom Section: Diff & Python Integration
    if res and res.raw_result:
        st.markdown("---")
        c_diff, c_py = st.columns([1.2, 1])
        with c_diff:
            st.markdown("#### Input vs Protected Output")
            render_diff(res.trace.input_text, res.raw_result.output)
        with c_py:
            render_copy_python(
                operation="process",
                prompt=res.trace.input_text,
                mode=policy_mode,
                token_budget=int(token_budget),
            )

render = render_playground_page

if __name__ == "__main__":
    render()
