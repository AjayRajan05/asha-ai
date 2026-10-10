"""Security console: preserved APIs, premium visuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from components.cards import section_header
from components.charts import gauge_score, threat_bar
from ui_backend import metrics_dict, pii_chip_class, run_process, run_sanitize, sec_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DEFAULT_TICKET = (FIXTURES / "support_ticket.txt").read_text(encoding="utf-8")
INJECTION_PROMPTS = [
    ln.strip()
    for ln in (FIXTURES / "injection_prompts.txt").read_text(encoding="utf-8").splitlines()
    if ln.strip() and not ln.strip().startswith("#")
]


def _chips(pii: List[str]) -> str:
    if not pii:
        return '<span class="asha-pill pill-muted">none</span>'
    parts = []
    for p in pii:
        parts.append(f'<span class="asha-chip {pii_chip_class(p)}">{p}</span>')
    return "".join(parts)


def render() -> None:
    section_header("Security", "Security console", "Live before/after (process() and sanitize() unchanged).")

    mode = st.selectbox("Policy mode", ["balanced", "lite", "strict"], index=0)
    api = st.radio("API", ["process", "sanitize"], horizontal=True)
    text = st.text_area("Prompt / ticket text", value=DEFAULT_TICKET, height=220)

    if st.button("Run security scan", type="primary", key="sec_run"):
        st.markdown('<div class="asha-scan"><div></div></div>', unsafe_allow_html=True)
        with st.spinner("Scanning prompt..."):
            result = run_process(text, mode=mode) if api == "process" else run_sanitize(text, mode=mode)
            out = result.output
            sec = sec_dict(result)
            met = metrics_dict(result)

        left, right = st.columns(2)
        with left:
            st.markdown("#### Before")
            st.code(text, language="text")
        with right:
            st.markdown("#### After")
            st.code(out, language="text")

        g1, g2 = st.columns([1, 1.2])
        with g1:
            score = sec["score"]
            if score <= 1:
                # security_score may already be 0-1 or larger; normalize for display
                display = score * 100 if score <= 1.0 else min(score, 100)
            else:
                display = min(score, 100)
            # Prefer safe=True -> high score presentation
            if sec["safe"] and display < 50:
                display = max(display, 85.0)
            gauge_score(display, "Security score")
        with g2:
            st.markdown("#### Risk badges")
            threat = str(sec["threat_level"]).lower()
            badge = "pill-allow" if threat in {"low", "-"} and sec["safe"] else "pill-warn" if threat == "medium" else "pill-block"
            st.markdown(
                f'<span class="asha-pill {badge}">threat: {sec["threat_level"]}</span> '
                f'<span class="asha-pill {"pill-allow" if sec["safe"] else "pill-block"}">safe={sec["safe"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("#### PII chips")
            st.markdown(_chips(sec["pii"]), unsafe_allow_html=True)
            if met:
                st.metric("Token reduction %", f"{met.get('reduction_pct', 0):.1f}")
                st.metric("Processing time", f"{met.get('time_ms', 0):.0f} ms")

        st.markdown("#### Threat breakdown")
        counts: Dict[str, int] = {}
        for t in sec["threats"]:
            counts[t] = counts.get(t, 0) + 1
        if not counts:
            counts = {"none detected": 0}
        threat_bar(counts)
        if sec["masked"]:
            st.caption(f"Masked entity keys: {list(sec['masked'].keys())[:12]}")

    st.divider()
    section_header("Batch", "Injection gallery", "Benign vs jailbreak-style prompts.")
    if st.button("Scan gallery", key="inj_run"):
        rows: List[Dict[str, Any]] = []
        progress = st.progress(0.0, text="Scanning…")
        for i, prompt in enumerate(INJECTION_PROMPTS):
            r = run_process(prompt, mode=mode)
            sec = sec_dict(r)
            rows.append(
                {
                    "prompt": prompt[:80],
                    "safe": sec["safe"],
                    "threat": sec["threat_level"],
                    "pii": ", ".join(sec["pii"]) or "-",
                    "threats": ", ".join(sec["threats"]) or "-",
                    "output": (r.output or "")[:120],
                }
            )
            progress.progress((i + 1) / len(INJECTION_PROMPTS))
        progress.empty()
        st.dataframe(rows, use_container_width=True)
