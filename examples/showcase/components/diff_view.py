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

"""Diff view component for ASHA Control Center."""

from __future__ import annotations

import difflib
import html

import streamlit as st


def render_diff(
    original: str,
    modified: str,
    title_original: str = "Original Input",
    title_modified: str = "Protected / Sanitized Output",
    side_by_side: bool = True,
) -> None:
    """Render side-by-side comparison of text before and after transformation."""
    if side_by_side:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{title_original}**")
            st.markdown(
                f'<div class="diff-box">{html.escape(original)}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(f"**{title_modified}**")
            st.markdown(
                f'<div class="diff-box">{html.escape(modified)}</div>',
                unsafe_allow_html=True,
            )
    else:
        matcher = difflib.SequenceMatcher(None, original.split(), modified.split())
        highlighted = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                highlighted.append(html.escape(" ".join(original.split()[i1:i2])))
            elif tag == "replace":
                highlighted.append(
                    f'<span class="diff-del">{html.escape(" ".join(original.split()[i1:i2]))}</span> '
                    f'<span class="diff-add">{html.escape(" ".join(modified.split()[j1:j2]))}</span>'
                )
            elif tag == "delete":
                highlighted.append(f'<span class="diff-del">{html.escape(" ".join(original.split()[i1:i2]))}</span>')
            elif tag == "insert":
                highlighted.append(f'<span class="diff-add">{html.escape(" ".join(modified.split()[j1:j2]))}</span>')

        st.markdown(
            f'<div class="diff-box">{" ".join(highlighted)}</div>',
            unsafe_allow_html=True,
        )
