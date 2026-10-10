"""Document scrubber with entity highlighting."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from components.cards import section_header
from ui_backend import metrics_dict, pii_chip_class, run_process, run_sanitize, sec_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def render() -> None:
    section_header(
        "Documents",
        "Document scrubber",
        "File paths, uploads, and bytes: same sanitize/process pipeline.",
    )

    source = st.radio(
        "Input",
        ["Sample support ticket", "Upload file", "Paste text"],
        horizontal=True,
    )
    mode = st.selectbox("Mode", ["balanced", "lite", "strict"], index=0, key="doc_mode")

    content: Optional[Any] = None
    label = ""
    raw_preview = ""
    if source == "Sample support ticket":
        path = FIXTURES / "support_ticket.txt"
        content = path
        label = path.name
        raw_preview = path.read_text(encoding="utf-8")
    elif source == "Upload file":
        up = st.file_uploader("Upload .txt / .md / .pdf / .docx", type=["txt", "md", "pdf", "docx"])
        if up is not None:
            suffix = Path(up.name).suffix.lower() or ".txt"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.getvalue())
            tmp.flush()
            tmp.close()
            content = Path(tmp.name)
            label = up.name
            try:
                raw_preview = up.getvalue().decode("utf-8", errors="replace")[:4000]
            except Exception:
                raw_preview = f"(binary upload: {up.name})"
    else:
        pasted = st.text_area("Paste document text", height=200)
        if pasted.strip():
            content = pasted
            label = "pasted text"
            raw_preview = pasted

    if st.button("Scrub document", type="primary", key="doc_run") and content is not None:
        st.markdown('<div class="asha-scan"><div></div></div>', unsafe_allow_html=True)
        with st.spinner("Scrubbing..."):
            san = run_sanitize(content, mode=mode)
            proc = run_process(content, mode=mode)
        st.write(f"**Source:** {label}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Original")
            st.code(raw_preview or "(see file)", language="text")
        with right:
            st.markdown("#### Secure version (sanitize)")
            st.code(san.output, language="text")

        sec = sec_dict(san)
        st.markdown("#### Removed / masked entities")
        chips = "".join(
            f'<span class="asha-chip {pii_chip_class(p)}">{p}</span>' for p in (sec["pii"] or ["none"])
        )
        st.markdown(chips, unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        s1.metric("Detected PII types", len(sec["pii"]))
        s2.metric("Masked keys", len(sec["masked"]))
        s3.metric("Safe", str(sec["safe"]))

        with st.expander("process() optimized view", expanded=False):
            st.code(proc.output, language="text")
            met = metrics_dict(proc)
            if met:
                st.write(
                    f"tokens saved={met.get('tokens_saved')} "
                    f"reduction={met.get('reduction_pct'):.1f}% "
                    f"time={met.get('time_ms'):.0f}ms"
                )
    elif content is None and source != "Sample support ticket":
        st.info("Provide a file or pasted text to scrub.")
