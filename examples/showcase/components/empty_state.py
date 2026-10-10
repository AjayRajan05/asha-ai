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

"""Empty state component for ASHA Control Center."""

from __future__ import annotations

import streamlit as st


def render_empty_state(title: str = "No data available", message: str = "Execute an operation to view live metrics.") -> None:
    """Render empty state card."""
    st.markdown(
        f"""
        <div style="background: #121820; border: 1px dashed #273142; border-radius: 8px; padding: 2rem; text-align: center; color: #8b949e; margin: 1rem 0;">
            <div style="font-size: 1rem; font-weight: 600; color: #e6edf3; margin-bottom: 0.3rem;">{title}</div>
            <div style="font-size: 0.82rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
