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

"""Provider status table component for ASHA Control Center."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_COMPONENTS_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _COMPONENTS_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.runtime.session import get_provider_status


def render_provider_status_table() -> None:
    """Render status indicators for all supported LLM providers."""
    providers = get_provider_status()

    st.markdown(
        """
        <div style="overflow-x: auto; -webkit-overflow-scrolling: touch; width: 100%; border: 1px solid #273142; border-radius: 8px; background: #121820; box-sizing: border-box;">
        <table style="width: 100%; min-width: 520px; border-collapse: collapse; font-size: 0.85rem;">
            <thead>
                <tr style="border-bottom: 1px solid #273142; color: #8b949e; text-align: left; background: #0e141c;">
                    <th style="padding: 10px 12px;">Provider</th>
                    <th style="padding: 10px 12px;">Status</th>
                    <th style="padding: 10px 12px;">Default / Selected Model</th>
                    <th style="padding: 10px 12px;">Notes</th>
                </tr>
            </thead>
            <tbody>
        """,
        unsafe_allow_html=True,
    )

    rows_html = []
    for key, p in providers.items():
        is_ready = p.get("configured", False)
        status_text = p.get("status_label", "Unknown")
        color = "#3fb950" if "Configured" in status_text or "Ready" in status_text or "Connected" in status_text else "#8b949e"

        notes = "Offline deterministic testing" if key == "local" else "Requires API key in environment" if not is_ready else "Ready for live inference"
        if key == "ollama":
            notes = f"Local host: {p.get('endpoint', 'localhost')}"
        elif key == "llamacpp":
            notes = f"OpenAI-compatible server: {p.get('endpoint', '')}"

        row = f"""
        <tr style="border-bottom: 1px solid #1c2430;">
            <td style="padding: 10px 12px; font-weight: 600; color: #ffffff;">{p['name']}</td>
            <td style="padding: 10px 12px;"><span style="color: {color}; font-weight: 500;">{status_text}</span></td>
            <td style="padding: 10px 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38bdf8;">{p.get('default_model', '-')}</td>
            <td style="padding: 10px 12px; color: #8b949e;">{notes}</td>
        </tr>
        """
        rows_html.append(row)

    st.markdown("".join(rows_html) + "</tbody></table></div>", unsafe_allow_html=True)
