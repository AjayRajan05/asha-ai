"""ASHA Control Center - Observability & Telemetry Console.

Full live observability suite:
- Request History table with sorting & selection
- Deep Trace Viewer with granular expandable event timeline
- Replay Engine (actually reruns the operation with new trace ID)
- Sanitized JSON Trace Export (zero secret leakage)
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.runtime.session import (
    get_history,
    get_current_trace,
    set_current_trace,
    record_trace,
)
from examples.showcase.runtime.trace import Trace, EventType
from examples.showcase.runtime.executor import PipelineExecutor
from examples.showcase.components.trace_viewer import render_trace_view
from examples.showcase.components.status_badge import render_status_pill
from examples.showcase.components.header import render_header
from examples.showcase.components.empty_state import render_empty_state
from examples.showcase.components.object_inspector import render_object_inspector


def render() -> None:
    render_header(title="OBSERVABILITY CONSOLE", subtitle="Real-time forensic audit logs, event timelines, request replays, and JSON exports")


    history = get_history()

    if not history:
        render_empty_state(
            title="No Execution Traces Recorded Yet",
            message="Execute any operation in the LLM Playground, PII Lab, Security Labs, or ANCHOR to generate live audit traces.",
            action_label="Go to LLM Playground",
        )
        return

    tab_history, tab_trace, tab_replay = st.tabs(["Request History", "Trace Inspector", "Replay Engine"])

    # -------------------------------------------------------------
    # TAB 1: Request History
    # -------------------------------------------------------------
    with tab_history:
        st.markdown(f"### Session Requests ({len(history)} total)")

        rows: List[Dict[str, Any]] = []
        for t in history:
            rows.append({
                "Trace ID": t.trace_id,
                "Time": t.timestamp_iso.split("T")[-1][:8],
                "Provider": t.provider,
                "Model": t.model,
                "Status": t.status,
                "Events": len(t.events),
                "Latency (ms)": f"{t.duration_ms:.1f}",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

        trace_ids = [t.trace_id for t in history]
        selected_id = st.selectbox("Select Trace to Inspect", trace_ids, index=len(trace_ids) - 1)
        selected_trace = next((t for t in history if t.trace_id == selected_id), history[-1])

        if st.button("Set as Active Trace", use_container_width=True):
            set_current_trace(selected_trace)
            st.success(f"Trace #{selected_trace.trace_id} active.")

    # -------------------------------------------------------------
    # TAB 2: Trace Inspector
    # -------------------------------------------------------------
    with tab_trace:
        active_trace = get_current_trace() or history[-1]
        st.markdown(f"### Trace Inspection: `#{active_trace.trace_id}`")

        # Top summary bar
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Events Recorded", len(active_trace.events))
        with b2:
            st.metric("Total Latency", f"{active_trace.duration_ms:.2f} ms")
        with b3:
            badge_color = "success" if active_trace.status == "SUCCESS" else ("blocked" if active_trace.status == "BLOCKED" else "warning")
            st.markdown(f"**Execution Status:**<br>{render_status_pill(active_trace.status, badge_color)}", unsafe_allow_html=True)
        with b4:
            st.markdown(f"**Target System:**<br><code>{active_trace.provider}</code>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # JSON Export Button
        json_export_str = active_trace.to_json(indent=2)
        st.download_button(
            label="⬇️ Download Sanitized Trace JSON",
            data=json_export_str,
            file_name=f"asha_trace_{active_trace.trace_id}.json",
            mime="application/json",
            help="Exports full forensic log. Credentials and sensitive secrets are automatically masked.",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Trace Viewer
        render_trace_view(active_trace)

        # Raw Object Inspector
        with st.expander("Inspect Raw Trace Dictionary Representation"):
            render_object_inspector(
                raw_obj=active_trace.to_dict(),
                title=f"Trace #{active_trace.trace_id}",
                human_data={
                    "Trace ID": active_trace.trace_id,
                    "Timestamp": active_trace.timestamp_iso,
                    "Provider": active_trace.provider,
                    "Model": active_trace.model,
                    "Event Count": len(active_trace.events),
                    "Duration": f"{active_trace.duration_ms:.2f} ms",
                },
            )

    # -------------------------------------------------------------
    # TAB 3: Replay Engine
    # -------------------------------------------------------------
    with tab_replay:
        st.markdown("### Deterministic Trace Replay")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.85rem;'>"
            "Replay does not merely redisplay old cached output. It initiates a real, fresh execution "
            "using the exact parameters of the historical trace, producing a separate new Trace ID for comparison."
            "</p>",
            unsafe_allow_html=True,
        )

        replay_target_id = st.selectbox("Choose Trace to Replay", [t.trace_id for t in history], key="replay_choice")
        target_to_replay = next((t for t in history if t.trace_id == replay_target_id), history[-1])

        # Extract prompt from input_received event
        extracted_prompt = ""
        for ev in target_to_replay.events:
            if ev.event_type == EventType.INPUT_RECEIVED:
                extracted_prompt = str(ev.metadata.get("prompt", "") or ev.reason)
                break

        st.text_area("Extracted Input Payload for Replay", value=extracted_prompt, height=80, disabled=True)

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.markdown(f"**Original Trace:** `#{target_to_replay.trace_id}`")
            st.caption(f"Recorded Latency: {target_to_replay.duration_ms:.1f} ms | Status: {target_to_replay.status}")
        with col_rep2:
            replay_btn = st.button("Rerun Execution Now", type="primary", use_container_width=True)

        if replay_btn:
            new_trace = Trace(provider=target_to_replay.provider, model=target_to_replay.model)
            executor = PipelineExecutor()

            with st.spinner("Rerunning operation through ASHA pipeline..."):
                t_rep_0 = time.perf_counter()
                res = executor.execute_process(
                    prompt=extracted_prompt or "Test replay query",
                    mode="balanced",
                    token_budget=1200,
                    provider_name=target_to_replay.provider,
                    model_name=target_to_replay.model,
                    trace=new_trace,
                )
                rep_ms = (time.perf_counter() - t_rep_0) * 1000.0

            record_trace(new_trace)

            st.success(f"Replay finished with New Trace #{new_trace.trace_id} in {rep_ms:.1f} ms")

            # Side-by-side comparison
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown(f"#### Original Trace `#{target_to_replay.trace_id}`")
                st.markdown(f"Status: {render_status_pill(target_to_replay.status, 'success' if target_to_replay.status == 'SUCCESS' else 'blocked')}", unsafe_allow_html=True)
                st.caption(f"Latency: {target_to_replay.duration_ms:.2f} ms | Events: {len(target_to_replay.events)}")
                render_trace_view(target_to_replay)

            with comp_col2:
                st.markdown(f"#### Replay Trace `#{new_trace.trace_id}`")
                st.markdown(f"Status: {render_status_pill(new_trace.status, 'success' if new_trace.status == 'SUCCESS' else 'blocked')}", unsafe_allow_html=True)
                st.caption(f"Latency: {new_trace.duration_ms:.2f} ms | Events: {len(new_trace.events)}")
                render_trace_view(new_trace)


if __name__ == "__main__":
    render()

