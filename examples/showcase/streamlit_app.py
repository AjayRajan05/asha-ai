#!/usr/bin/env python3
"""
ASHA product showcase (Streamlit).

Premium interactive experience over the real ASHA APIs.

    pip install -e ".[demo]"
    streamlit run examples/showcase/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import load_env  # noqa: E402

load_env()

from components.styles import inject_styles  # noqa: E402
from views import architecture, benchmarks, documents, live_llm  # noqa: E402
from views import overview, playground, rogue, security, story  # noqa: E402

st.set_page_config(
    page_title="ASHA: AI Security Layer",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

NAV = [
    "Overview",
    "Playground",
    "Security console",
    "Rogue agent",
    "Document scrubber",
    "Live LLM",
    "End-to-end story",
    "Architecture",
    "Benchmarks",
]

if "nav" not in st.session_state:
    st.session_state["nav"] = "Overview"


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("### ASHA")
        st.caption("Mission-aware AI security")
        st.markdown("---")
        try:
            from streamlit_option_menu import option_menu

            choice = option_menu(
                None,
                NAV,
                icons=[
                    "speedometer2",
                    "terminal",
                    "shield-lock",
                    "cpu",
                    "file-earmark-text",
                    "robot",
                    "film",
                    "diagram-3",
                    "bar-chart",
                ],
                menu_icon="cast",
                default_index=NAV.index(st.session_state["nav"])
                if st.session_state["nav"] in NAV
                else 0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#2dd4bf", "font-size": "16px"},
                    "nav-link": {
                        "font-size": "14px",
                        "text-align": "left",
                        "margin": "2px 0",
                        "--hover-color": "#134e4a",
                    },
                    "nav-link-selected": {"background-color": "#0f766e"},
                },
            )
        except Exception:
            choice = st.radio(
                "Navigate",
                NAV,
                index=NAV.index(st.session_state["nav"])
                if st.session_state["nav"] in NAV
                else 0,
            )
        st.session_state["nav"] = choice
        st.markdown("---")
        st.caption("Developer preview v0.4.2")
        st.caption("Backend: process · sanitize · Agent · ANCHOR · wrap_llm")
        st.caption("Keys stay in `.env` (never paste into chat).")
    return st.session_state["nav"]


def main() -> None:
    page = _sidebar()
    if page == "Overview":
        overview.render()
    elif page == "Playground":
        playground.render()
    elif page == "Security console":
        security.render()
    elif page == "Rogue agent":
        rogue.render()
    elif page == "Document scrubber":
        documents.render()
    elif page == "Live LLM":
        live_llm.render()
    elif page == "End-to-end story":
        story.render()
    elif page == "Architecture":
        architecture.render()
    else:
        benchmarks.render()


if __name__ == "__main__":
    main()
