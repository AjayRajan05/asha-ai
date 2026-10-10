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

"""Developer sidebar navigation component for ASHA Control Center."""

from __future__ import annotations

import streamlit as st

NAVIGATION_SECTIONS = {
    "OVERVIEW": [
        "Dashboard",
        "Runtime Status",
        "Provider Status",
    ],
    "PLAYGROUND": [
        "LLM Playground",
        "Process Pipeline",
        "API Explorer",
    ],
    "SECURITY": [
        "PII Protection",
        "Prompt Injection",
        "Unicode Security",
        "Long Context",
        "RAG Security",
    ],
    "OPTIMIZATION": [
        "Token Optimization",
        "Optimization Rejection Demo",
    ],
    "AGENTS": [
        "ANCHOR Control Room",
        "Action Governance",
        "Memory Security",
        "Intent Drift",
    ],
    "PROVIDERS": [
        "Provider Playground",
        "Google Gemini",
        "OpenAI",
        "Anthropic Claude",
        "Ollama Local",
        "llama.cpp Server",
    ],
    "OBSERVABILITY": [
        "Live Trace",
        "Request History",
    ],
    "CONFIGURATION": [
        "PolicyConfig & Presets",
    ],
}


def render_sidebar() -> str:
    """Render the structured sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                    <span style="color: #38bdf8;">ASHA</span> CONTROL CENTER
                </div>
                <div style="font-size: 0.75rem; color: #8b949e;">Developer Preview Console</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Transparency mode switcher
        mode = st.selectbox(
            "Transparency Mode",
            ["Normal", "Transparent", "Forensic"],
            index=["Normal", "Transparent", "Forensic"].index(
                st.session_state.get("transparency_mode", "Transparent")
            ),
            help="Normal: clean view. Transparent: full stage visibility. Forensic: low-level object inspection.",
        )
        st.session_state["transparency_mode"] = mode

        backend_name = st.session_state.get("selected_provider", "local").upper()
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; color: #8b949e; margin-top: 6px; display: flex; justify-content: space-between;">
                <span>Backend:</span>
                <span style="font-family: 'JetBrains Mono', monospace; color: #38bdf8; font-weight: 600;">{backend_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color: #273142; margin: 0.8rem 0;'>", unsafe_allow_html=True)

        current = st.session_state.get("nav", "Dashboard")
        all_pages = [page for pages in NAVIGATION_SECTIONS.values() for page in pages]

        selected_page = current
        for section, pages in NAVIGATION_SECTIONS.items():
            st.markdown(
                f"<div style='font-size: 0.72rem; font-weight: 700; color: #6e7681; letter-spacing: 0.08em; margin: 0.6rem 0 0.2rem 0;'>{section}</div>",
                unsafe_allow_html=True,
            )
            for page in pages:
                is_active = (current == page)
                btn_type = "primary" if is_active else "secondary"
                if st.button(page, key=f"nav_btn_{page}", type=btn_type, use_container_width=True):
                    st.session_state["nav"] = page
                    selected_page = page
                    st.rerun()

        st.markdown("<hr style='border-color: #273142; margin: 1rem 0;'>", unsafe_allow_html=True)
        st.caption("Zero external telemetry transmitted. All executions evaluate locally.")

        return selected_page
