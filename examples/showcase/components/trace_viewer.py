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

"""Reusable trace viewer component for ASHA Control Center."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Optional

_COMPONENTS_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _COMPONENTS_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.runtime.trace import Trace, TraceEvent


def render_trace_viewer(trace: Optional[Trace]) -> None:
    """Render an interactive, expandable event timeline for an execution trace."""
    if trace is None or not trace.events:
        st.info("No active execution trace. Run an operation to generate a real-time event trace.")
        return

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #38bdf8;">
                    TRACE #{trace.trace_id}
                </span>
                <span class="pill pill-{'allow' if trace.status == 'SUCCESS' else 'block' if trace.status == 'BLOCKED' else 'warn'}" style="margin-left: 8px;">
                    {trace.status}
                </span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #8b949e;">
                Total: {trace.total_latency_ms:.1f}ms (ASHA: {trace.asha_latency_ms:.1f}ms | Provider: {trace.provider_latency_ms:.1f}ms)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, event in enumerate(trace.events):
        dt_str = datetime.datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
        status_color = "#3fb950" if event.status == "SUCCESS" else "#f85149" if event.status == "BLOCKED" else "#f0883e" if event.status == "WARNING" else "#38bdf8"

        event_name = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)

        dur_text = f"[{event.duration_ms:.1f}ms]" if event.duration_ms > 0 else ""
        header_text = f"`{dt_str}`  **{event_name}**  {dur_text}  `{event.status}`"

        with st.expander(header_text, expanded=(event.status in ("BLOCKED", "ERROR") or idx == len(trace.events) - 1)):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Timestamp:** `{dt_str}`")
                st.markdown(f"**Event Type:** `{event_name}`")
                if event.reason:
                    st.markdown(f"**Reason / Verdict:** {event.reason}")
            with c2:
                if event.duration_ms > 0:
                    st.markdown(f"**Duration:** `{event.duration_ms:.2f} ms`")
                if event.token_count is not None:
                    st.markdown(f"**Tokens:** `{event.token_count}`")
                if event.error:
                    st.error(f"Error: {event.error}")

            if event.details:
                st.markdown("**Event Details:**")
                st.json(event.details)


render_trace_view = render_trace_viewer
