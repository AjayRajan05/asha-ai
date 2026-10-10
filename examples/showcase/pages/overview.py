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

"""Overview dashboard for ASHA Control Center."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

import asha
from examples.showcase.components.header import render_header
from examples.showcase.components.metric_cards import render_metric_cards
from examples.showcase.components.provider_status import render_provider_status_table
from examples.showcase.runtime.session import get_session_metrics


def render_overview_page() -> None:
    """Render the main system overview and session dashboard."""
    render_header(title="CONTROL CENTER", subtitle="System status, session metrics, and provider connectivity")

    st.markdown("### Session Metrics")
    metrics = get_session_metrics()
    render_metric_cards(metrics)

    st.markdown("---")

    st.markdown("### Runtime Status")
    st.markdown(
        f"""
        <div style="background: #121820; border: 1px solid #273142; border-radius: 8px; padding: 1.2rem; box-sizing: border-box;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; font-size: 0.85rem;">
                <div style="border-bottom: 1px solid #1c2430; padding-bottom: 8px;">
                    <div style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase;">ASHA Library Version</div>
                    <div style="font-family: 'JetBrains Mono', monospace; color: #38bdf8; font-weight: 600; margin-top: 2px;">v{getattr(asha, '__version__', '0.4.2')}</div>
                </div>
                <div style="border-bottom: 1px solid #1c2430; padding-bottom: 8px;">
                    <div style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase;">Release Tier</div>
                    <div style="font-family: 'JetBrains Mono', monospace; color: #e6edf3; font-weight: 500; margin-top: 2px;">Developer Preview</div>
                </div>
                <div style="border-bottom: 1px solid #1c2430; padding-bottom: 8px;">
                    <div style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase;">Python Environment</div>
                    <div style="font-family: 'JetBrains Mono', monospace; color: #e6edf3; font-weight: 500; margin-top: 2px;">{platform.python_version()} ({platform.system()})</div>
                </div>
                <div style="padding-bottom: 8px;">
                    <div style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase;">Core Guarantee</div>
                    <div style="color: #3fb950; font-weight: 600; margin-top: 2px;">Zero-Heavy-Dependency Core</div>
                </div>
                <div style="padding-bottom: 8px;">
                    <div style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase;">Default Execution Path</div>
                    <div style="font-family: 'JetBrains Mono', monospace; color: #e6edf3; font-weight: 500; margin-top: 2px;">Calibrated Heuristic & Ensemble</div>
                </div>
                <div style="padding-bottom: 8px;">
                    <div style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase;">Telemetry & Egress</div>
                    <div style="color: #3fb950; font-weight: 600; margin-top: 2px;">Zero External Telemetry</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Provider Status")
    render_provider_status_table()

    st.markdown("---")
    st.markdown("### Recent Execution Traces")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("No executions recorded in this session yet.")
    else:
        recent = history[:5]
        for tr in recent:
            status_color = "#3fb950" if tr.status == "SUCCESS" else "#f85149" if tr.status == "BLOCKED" else "#f0883e"
            st.markdown(
                f"""
                <div style="background: #121820; border: 1px solid #273142; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #38bdf8;">{tr.trace_id}</span>
                        <span style="margin-left: 10px; font-size: 0.85rem; color: #e6edf3;">{tr.provider or 'local'} ({tr.model or 'default'})</span>
                        <span style="margin-left: 10px; font-size: 0.78rem; color: #8b949e;">{len(tr.events)} events</span>
                    </div>
                    <div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #8b949e; margin-right: 12px;">{tr.total_latency_ms:.1f}ms</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; color: {status_color};">{tr.status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

render = render_overview_page

if __name__ == "__main__":
    render()
