"""Benchmark / capability cards (demo figures for showcase)."""

from __future__ import annotations

import streamlit as st

from components.cards import section_header
from components.charts import token_savings_chart
from components.metrics import hero_metrics


def render() -> None:
    section_header(
        "Numbers",
        "Benchmarks",
        "Illustrative showcase metrics (validate on your workload before production claims).",
    )

    hero_metrics(
        [
            ("Detection coverage", "High", "PII + secrets + injection"),
            ("Threat detection", "Layered", "rules + optional ML"),
            ("Avg processing", "~45ms", "lite path demo"),
            ("Avg token reduction", "~28%", "workload dependent"),
            ("Supported models", "4+", "via adapters"),
            ("Offline mode", "Yes", "base install"),
        ]
    )

    section_header("Charts", "Token reduction by scenario", "Demo dataset for visual storytelling.")
    token_savings_chart(
        ["Support ticket", "Healthcare note", "Finance brief", "Agent prompt"],
        [42.0, 31.0, 27.0, 18.0],
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
<div class="asha-card">
  <h4>Enterprise ready</h4>
  <p>Policy modes, fail-closed strict path, ANCHOR headless CI defaults, Apache-2.0 SDK.</p>
  <span class="asha-pill pill-accent">preview 0.4.2</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="asha-card">
  <h4>Offline first</h4>
  <p>No API key required for process/sanitize/optimize. Hardened models optional.</p>
  <span class="asha-pill pill-allow">lite default</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div class="asha-card">
  <h4>Agent governance</h4>
  <p>ANCHOR blocks high-risk tools on local-only missions with clear verdicts.</p>
  <span class="asha-pill pill-warn">preview</span>
</div>
""",
            unsafe_allow_html=True,
        )
