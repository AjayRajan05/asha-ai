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

"""Console header component for ASHA Control Center."""

from __future__ import annotations

import streamlit as st


def render_header(title: str = "CONTROL CENTER", subtitle: str = "Adaptive Security for Human-SI Systems") -> None:
    """Render the top dark console header with system indicators."""
    mode = st.session_state.get("transparency_mode", "Transparent")

    html = f"""
    <div class="console-header">
        <div class="console-header-left">
            <div class="console-title">
                <span class="console-logo">ASHA</span>
                <span class="console-heading">{title}</span>
                <span class="console-badge badge-preview">v0.4.2 Preview</span>
            </div>
            <div class="console-subtitle">
                {subtitle}
            </div>
        </div>
        <div class="console-header-right">
            <div class="console-mode-label">
                Mode: <span class="pill pill-info">{mode}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
