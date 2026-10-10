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

"""Interactive PII Protection Lab and reversible masking flow."""

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

from asha import PIIVault, PolicyConfig, sanitize
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.diff_view import render_diff
from examples.showcase.components.header import render_header
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.runtime.local_provider import LocalDemoProvider
from examples.showcase.scenarios.pii import PII_SCENARIOS


def render_pii_page() -> None:
    """Render the interactive PII Protection and Reversible Masking Lab."""
    render_header(title="PII PROTECTION LAB", subtitle="Real-time entity detection, reversible pseudonymization, and LLM-safe de-identification")

    c_scen, c_mode = st.columns([1.5, 1])
    with c_scen:
        scenario_key = st.selectbox("Select Test Scenario", list(PII_SCENARIOS.keys()), index=0)
    with c_mode:
        pii_mode = st.selectbox("Detection Mode", ["rule", "hybrid", "ml_only"], index=0)

    default_text = PII_SCENARIOS[scenario_key]
    prompt_text = st.text_area("Input Text with PII (Editable)", value=default_text, height=130)

    c_rev, c_btn = st.columns([1, 1])
    with c_rev:
        use_vault = st.checkbox("Enable Reversible PIIVault (Bidirectional Token Replacement)", value=True)
    with c_btn:
        run_btn = st.button("Detect and Mask PII", type="primary", use_container_width=True)

    if run_btn or "last_pii_run" in st.session_state:
        if run_btn:
            policy = PolicyConfig(mode="balanced", pii_mode=pii_mode, reversible=use_vault)
            # Execute real ASHA sanitize
            san_res = sanitize(prompt_text, policy=policy)
            vault_masked = ""
            vault = None
            if use_vault:
                vault = PIIVault()
                vault_masked = vault.mask(prompt_text)

            st.session_state["last_pii_run"] = {
                "raw": prompt_text,
                "san_res": san_res,
                "vault": vault,
                "vault_masked": vault_masked,
                "use_vault": use_vault,
                "pii_mode": pii_mode,
            }

        data = st.session_state["last_pii_run"]
        san_res = data["san_res"]
        sec = getattr(san_res, "security_result", None) or getattr(san_res, "security", None)

        st.markdown("---")
        st.markdown("### Detection & Masking Results")

        entities = getattr(sec, "masked_entities", {}) if sec else {}
        pii_list = getattr(sec, "pii_detected", []) if sec else []

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Detected Entities", len(pii_list))
        with c2:
            st.metric("Entities Masked", len(entities))
        with c3:
            st.metric("Reversible Status", "Yes (PIIVault active)" if data["use_vault"] else "No (Irreversible Hash)")

        st.markdown("#### Masked Prompt (LLM-Safe)")
        output_display = data["vault_masked"] if data["use_vault"] and data["vault_masked"] else san_res.output
        st.code(output_display, language="text")

        st.markdown("#### Side-by-Side Verification")
        render_diff(data["raw"], output_display)

        # Entity Breakdown Table
        if entities:
            st.markdown("#### Detected Entities Breakdown")
            rows = []
            for k, info in entities.items():
                t = info.get("type", "unknown") if isinstance(info, dict) else str(info)
                rows.append({"Identifier": k, "Category": t, "Severity": info.get("severity", 1.0) if isinstance(info, dict) else 1.0})
            st.table(rows)

        # Real LLM Flow with Protected Input
        st.markdown("---")
        st.markdown("### Real LLM Flow with Protected Input")
        st.caption("Send the protected text to inference and observe that the model never sees raw PII.")

        if st.button("Send Protected Input to Downstream LLM"):
            with st.spinner("Dispatching protected input to inference engine..."):
                local_p = LocalDemoProvider()
                llm_resp = local_p.generate(output_display)
                st.markdown("**LLM Completion Received:**")
                st.info(llm_resp["text"])

                if data["use_vault"] and data["vault"]:
                    st.markdown("**Re-hydrated Completion (Unmasked for Application):**")
                    unmasked = data["vault"].unmask(llm_resp["text"])
                    st.success(unmasked)

        render_object_inspector(san_res, title="SanitizeResult Details")
        render_copy_python(operation="sanitize", prompt=data["raw"], pii_mode=data["pii_mode"], reversible=data["use_vault"])

render = render_pii_page

if __name__ == "__main__":
    render()
