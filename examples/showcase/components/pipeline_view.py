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

"""Pipeline node visualizer component for ASHA Control Center."""

from __future__ import annotations

from typing import Dict, List, Optional
import streamlit as st


def render_pipeline(
    stages_status: Optional[Dict[str, str]] = None,
    blocked_stage: Optional[str] = None,
) -> None:
    """
    Render horizontal execution pipeline visualization.
    stages_status: dict mapping stage name to status: 'done', 'active', 'blocked', 'pending'
    """
    stages = [
        ("INPUT", "Entry"),
        ("SANITIZE", "Security"),
        ("PII", "Masking"),
        ("COMPILE", "Grammar"),
        ("OPTIMIZE", "Pruning"),
        ("POLICY", "Guard"),
        ("LLM", "Inference"),
        ("RESULT", "Verified"),
    ]

    status_map = stages_status or {}

    nodes_html = []
    for idx, (code, label) in enumerate(stages):
        st_val = status_map.get(code, "pending")
        if blocked_stage == code:
            st_val = "blocked"

        node_class = f"node-{st_val}"
        color_dot = "#8b949e"
        if st_val == "done":
            color_dot = "#3fb950"
        elif st_val == "blocked":
            color_dot = "#f85149"
        elif st_val == "active":
            color_dot = "#38bdf8"

        node_html = f"""
        <div class="pipeline-node {node_class}">
            <div style="font-size: 0.72rem; font-weight: 700; color: {color_dot}; letter-spacing: 0.05em;">{code}</div>
            <div style="font-size: 0.68rem; color: #8b949e;">{label}</div>
        </div>
        """
        nodes_html.append(node_html)

        if idx < len(stages) - 1:
            nodes_html.append('<div class="pipeline-arrow">→</div>')

    html = f"""
    <div class="pipeline-container">
        {''.join(nodes_html)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
