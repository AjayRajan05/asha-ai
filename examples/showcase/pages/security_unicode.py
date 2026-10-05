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

"""Unicode security, homoglyph unfolding, and zero-width smuggling lab."""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from asha import PolicyConfig, sanitize
from asha.core.text.canonicalize import canonicalize
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import get_status_badge_html
from examples.showcase.scenarios.unicode import UNICODE_SCENARIOS


def _inspect_codepoints(text: str) -> list[dict[str, str]]:
    """Decompose text into character-by-character codepoint metadata."""
    rows = []
    for char in text[:60]:  # sample first 60 chars for display
        cp = f"U+{ord(char):04X}"
        cat = unicodedata.category(char)
        name = unicodedata.name(char, "UNKNOWN")
        is_zw = ord(char) in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x200E, 0x200F)
        rows.append({
            "Glyph": repr(char)[1:-1],
            "Codepoint": cp,
            "Category": cat,
            "Visibility": "Invisible / Zero-Width" if is_zw or cat.startswith("C") else "Visible",
            "Unicode Name": name,
        })
    return rows


def render_unicode_page() -> None:
    """Render the interactive Unicode Security and Confusable Analysis Lab."""
    render_header(title="UNICODE SECURITY LAB", subtitle="Defense against zero-width character smuggling, Cyrillic homoglyphs, and confusable bypasses")

    scen_key = st.selectbox("Select Unicode Scenario", list(UNICODE_SCENARIOS.keys()), index=0)
    prompt_text = st.text_area("Unicode Payload (Editable)", value=UNICODE_SCENARIOS[scen_key], height=100)

    analyze_btn = st.button("Analyze Unicode & Run ASHA Pipeline", type="primary", use_container_width=True)

    if analyze_btn or "last_unicode_run" in st.session_state:
        if analyze_btn or st.session_state.get("last_unicode_run_prompt") != prompt_text:
            canon = canonicalize(prompt_text)
            san_res = sanitize(prompt_text, policy=PolicyConfig(mode="balanced"))
            st.session_state["last_unicode_run"] = (prompt_text, canon, san_res)
            st.session_state["last_unicode_run_prompt"] = prompt_text

        raw_in, canon, san_res = st.session_state["last_unicode_run"]
        sec = getattr(san_res, "security_result", None) or getattr(san_res, "security", None)
        is_safe = getattr(sec, "safe", True) if sec else True
        if hasattr(sec, "is_safe"):
            is_safe = bool(sec.is_safe)
        threats = list(getattr(sec, "threats", []) or [])

        # Count zero-width characters in input
        zw_count = sum(1 for c in raw_in if ord(c) in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x200E, 0x200F))

        st.markdown("---")
        st.markdown("### Normalization & Security Decision")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Policy Decision:**<br>{get_status_badge_html('ALLOW' if is_safe else 'BLOCK')}", unsafe_allow_html=True)
        with c2:
            st.metric("Detected Zero-Width Chars", zw_count)
        with c3:
            st.metric("Canonicalized Length Change", f"{len(raw_in)} → {len(canon)}")

        st.markdown("#### Dual-View Canonicalization")
        col_raw, col_norm = st.columns(2)
        with col_raw:
            st.markdown("**Raw Input Representation:**")
            st.code(raw_in, language="text")
        with col_norm:
            st.markdown("**ASHA Canonicalized (Unfolded & Stripped):**")
            st.code(canon, language="text")

        st.markdown("#### Character & Codepoint Inspection Table")
        st.caption("Inspects individual UTF-8 code points to expose hidden glyphs and non-ASCII character spoofing:")
        st.dataframe(_inspect_codepoints(raw_in), use_container_width=True)

        render_copy_python(operation="sanitize", prompt=raw_in)

render = render_unicode_page

if __name__ == "__main__":
    render()
