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

"""Metric cards component for ASHA Control Center."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_metric_cards(metrics: Dict[str, Any]) -> None:
    """Render the 6 core session metrics cards."""
    if not metrics.get("has_data", False):
        st.markdown(
            """
            <div style="background: #121820; border: 1px dashed #273142; border-radius: 8px; padding: 2rem; text-align: center; color: #8b949e;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 0.4rem;">No executions yet</div>
                <div style="font-size: 0.85rem;">Run a prompt in the LLM Playground or trigger a Security Lab scenario to populate live session metrics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Requests</div>
                <div class="metric-value">{metrics.get('requests', 0)}</div>
                <div class="metric-sub">Session total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">PII Events</div>
                <div class="metric-value" style="color: #bc8cff;">{metrics.get('pii_events', 0)}</div>
                <div class="metric-sub">Entities masked</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Tokens Saved</div>
                <div class="metric-value" style="color: #38bdf8;">{metrics.get('tokens_saved', 0)}</div>
                <div class="metric-sub">Pruned / compressed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Blocked Actions</div>
                <div class="metric-value" style="color: #f85149;">{metrics.get('blocked_actions', 0)}</div>
                <div class="metric-sub">Policy interventions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Security Alerts</div>
                <div class="metric-value" style="color: #f0883e;">{metrics.get('security_events', 0)}</div>
                <div class="metric-sub">Injections / anomalies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value" style="color: #3fb950;">{metrics.get('avg_latency_ms', 0):.0f}ms</div>
                <div class="metric-sub">End-to-end time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
