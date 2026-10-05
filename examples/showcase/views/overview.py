"""Executive landing / Why ASHA."""

from __future__ import annotations

import streamlit as st

from components.cards import section_header, why_card
from components.comparison import without_vs_with
from components.metrics import hero_metrics
from components.pipeline import horizontal_pipeline
from data.use_cases import USE_CASES


PIPELINE = [
    ("1", "User Prompt", "Inbound text / files"),
    ("2", "PII Detection", "Emails, phones, names"),
    ("3", "Secret Detection", "Keys & credentials"),
    ("4", "Injection Detect", "Jailbreak patterns"),
    ("5", "Optimization", "Token compression"),
    ("6", "Policy Engine", "strict / balanced / lite"),
    ("7", "LLM Router", "Universal adapters"),
    ("8", "Safe Response", "Governed output"),
]


def render() -> None:
    st.markdown(
        """
<div class="asha-hero">
  <div class="brand">ASHA</div>
  <h1>Mission-aware AI Security Layer</h1>
  <p>Protect every prompt before it reaches an LLM. Mask PII, stop injections,
  optimize tokens, and leash agent tools with ANCHOR.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    cta1, cta2, _ = st.columns([1, 1, 2])
    with cta1:
        if st.button("Run Live Demo", type="primary", use_container_width=True):
            st.session_state["nav"] = "Playground"
            st.rerun()
    with cta2:
        if st.button("See Security Console", use_container_width=True):
            st.session_state["nav"] = "Security console"
            st.rerun()

    hero_metrics(
        [
            ("Prompts Protected", "128K+", "demo volume"),
            ("Threats Blocked", "3.4K", "injection + exfil"),
            ("PII Entities Removed", "91K", "masked before LLM"),
            ("Avg Token Savings", "28%", "cost reduction"),
            ("Avg Processing", "45ms", "lite path"),
            ("Supported LLMs", "4+", "OpenAI · Claude · Gemini · Ollama"),
        ]
    )

    section_header("Why ASHA", "The problem with raw LLM usage", "Security and cost start before the model call.")
    c1, c2, c3 = st.columns(3)
    with c1:
        why_card(
            "Problem",
            "Large language models receive raw prompts: emails, keys, cards, and jailbreaks travel straight into the provider.",
            "warn",
        )
    with c2:
        why_card(
            "Solution",
            "ASHA sanitizes, secures, and optimizes prompts before inference, then ANCHOR governs what agents can do.",
            "accent",
        )
    with c3:
        why_card(
            "Result",
            "Safer AI · lower costs · governed agents. Drop-in for apps, assistants, and autonomous tool users.",
            "allow",
        )

    section_header("How it works", "Visual security pipeline", "Every request passes through layered protection.")
    horizontal_pipeline(PIPELINE, scanning=True)
    st.markdown(
        '<div class="asha-card"><strong>Model adapters:</strong> GPT · Claude · Gemini · Ollama '
        "via universal wrap / Agent routing.</div>",
        unsafe_allow_html=True,
    )

    section_header("Impact", "Without vs with ASHA", "Same workload. Completely different risk profile.")
    without_vs_with()

    section_header("Use cases", "Real-world scenarios", "Click a vertical to load a realistic prompt into the Playground.")
    keys = list(USE_CASES.keys())
    cols = st.columns(3)
    for i, name in enumerate(keys):
        with cols[i % 3]:
            st.markdown(
                f'<div class="asha-card"><h4>{name}</h4><p>{USE_CASES[name]["blurb"]}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Load {name}", key=f"uc_{name}", use_container_width=True):
                st.session_state["playground_box"] = USE_CASES[name]["prompt"]
                st.session_state["nav"] = "Playground"
                st.rerun()

    section_header("Live impact", "What a typical protected run looks like", "Demo counters (run Playground for live numbers).")
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(
            '<div class="asha-card"><h4>PII removed</h4>'
            '<div style="font-size:2rem;font-weight:700">14</div>'
            "<p>entities masked</p></div>",
            unsafe_allow_html=True,
        )
    with i2:
        st.markdown(
            '<div class="asha-card"><h4>Injections blocked</h4>'
            '<div style="font-size:2rem;font-weight:700">3</div>'
            "<p>threat patterns</p></div>",
            unsafe_allow_html=True,
        )
    with i3:
        st.markdown(
            '<div class="asha-card"><h4>Token reduction</h4>'
            '<div style="font-size:2rem;font-weight:700">42%</div>'
            "<p>estimated API savings</p></div>",
            unsafe_allow_html=True,
        )
    with i4:
        st.markdown(
            '<div class="asha-card"><h4>Safe for LLM</h4>'
            '<div style="font-size:2rem;font-weight:700">98</div>'
            "<p>confidence score</p></div>",
            unsafe_allow_html=True,
        )
