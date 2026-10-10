"""Live LLM split-screen."""

from __future__ import annotations

import os

import streamlit as st

from _common import gemini_api_key, ollama_reachable
from components.cards import section_header
from components.pipeline import horizontal_pipeline
from ui_backend import Agent, metrics_dict, run_process, sec_dict, wrap_llm


def render() -> None:
    section_header(
        "Inference",
        "Live LLM",
        "Mock by default. Gemini/Ollama when configured (wrap_llm / Agent preserved).",
    )

    backend = st.selectbox("Backend", ["mock", "gemini", "ollama"], index=0)
    prompt = st.text_area(
        "User prompt",
        value=(
            "Please draft a short reply to customer Priya Sharma "
            "(priya.sharma@acmecorp.example, phone +1-415-555-0198). "
            "Acknowledge their ticket about a leaked key sk-1234567890abcdefghijklmnop "
            "and tell them we rotated credentials. Keep it under 80 words."
        ),
        height=140,
    )
    st.caption(
        f"Gemini key: {'yes' if gemini_api_key() else 'no'} · "
        f"Ollama: {'yes' if ollama_reachable() else 'no'}"
    )

    if st.button("Run preprocess + LLM", type="primary", key="llm_run"):
        horizontal_pipeline(
            [
                ("1", "Raw Prompt", "Inbound"),
                ("2", "ASHA Process", "Mask + optimize"),
                ("3", "Optimized", "Safe prompt"),
                ("4", "Model", backend),
            ],
            scanning=True,
        )
        with st.spinner("ASHA processing..."):
            cleaned = run_process(prompt, mode="balanced")
        sec = sec_dict(cleaned)
        met = metrics_dict(cleaned)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Raw prompt")
            st.code(prompt, language="text")
        with c2:
            st.markdown("#### Optimized prompt")
            st.code(cleaned.output, language="text")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Threat level", str(sec["threat_level"]))
        k2.metric(
            "Prompt length Δ",
            f"{len(prompt) - len(cleaned.output):+d} chars",
        )
        k3.metric(
            "Token reduction",
            f"{met.get('reduction_pct', 0):.1f}%" if met else "-",
        )
        k4.metric("Latency (ASHA)", f"{met.get('time_ms', 0):.0f} ms" if met else "-")

        reply = ""
        note = ""
        active = backend
        try:
            if active == "gemini":
                if not gemini_api_key() or wrap_llm is None:
                    st.warning("Gemini unavailable; falling back to mock.")
                    active = "mock"
                else:
                    import google.generativeai as genai

                    model_name = os.getenv("ASHA_DEMO_GEMINI_MODEL", "gemini-1.5-flash")
                    genai.configure(api_key=gemini_api_key())
                    client = wrap_llm(genai.GenerativeModel(model_name), mode="balanced")
                    response = client.generate_content(prompt)
                    reply = getattr(response, "text", None) or str(response)
                    note = f"Gemini via wrap_llm ({model_name})"
            if active == "ollama":
                if not ollama_reachable():
                    st.warning("Ollama unreachable; falling back to mock.")
                    active = "mock"
                else:
                    model = os.getenv("ASHA_DEMO_OLLAMA_MODEL", "llama3.2")
                    agent = Agent(model=model, provider="ollama", privacy=True)
                    traced = agent.run(prompt, trace=True)
                    reply = str(traced.response)
                    note = f"Ollama Agent ({model})"
            if active == "mock":
                agent = Agent(model="mock", privacy=True)
                traced = agent.run(prompt, trace=True)
                reply = str(traced.response)
                note = "Mock Agent (no network)"
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            agent = Agent(model="mock", privacy=True)
            traced = agent.run(prompt, trace=True)
            reply = str(traced.response)
            note = "Mock fallback"

        st.markdown(f"#### Model response: {note}")
        st.write(reply)
        st.caption("Estimated cost: demo-only (provider pricing varies). Threats handled pre-inference.")
