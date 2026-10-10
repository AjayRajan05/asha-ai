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

"""RAG indirect prompt injection defense lab."""

from __future__ import annotations

import sys
from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from asha import scan_rag_context
from examples.showcase.components.code_viewer import render_copy_python
from examples.showcase.components.header import render_header
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.status_badge import get_status_badge_html
from examples.showcase.scenarios.rag import RAG_SCENARIOS


def render_rag_page() -> None:
    """Render the RAG context guard and indirect injection scanner page."""
    render_header(title="RAG SECURITY LAB", subtitle="Indirect injection scanning and delimiter boundary verification for retrieval-augmented generation")

    scen_key = st.selectbox("Select Scenario Preset", list(RAG_SCENARIOS.keys()), index=0)
    default_q, default_docs = RAG_SCENARIOS[scen_key]

    query_text = st.text_input("User Search Query", value=default_q)

    st.markdown("#### Retrieved Document Chunks (Editable)")
    docs: list[str] = []
    for i, doc in enumerate(default_docs):
        d_val = st.text_area(f"Document Chunk #{i+1}", value=doc, height=90, key=f"rag_doc_{i}")
        docs.append(d_val)

    scan_btn = st.button("Scan RAG Context Chunks with asha.scan_rag_context()", type="primary", use_container_width=True)

    if scan_btn or "last_rag_run" in st.session_state:
        if scan_btn or st.session_state.get("last_rag_docs") != docs or st.session_state.get("last_rag_query") != query_text:
            rag_res = scan_rag_context(documents=docs, query=query_text)
            st.session_state["last_rag_run"] = (query_text, docs, rag_res)
            st.session_state["last_rag_docs"] = docs
            st.session_state["last_rag_query"] = query_text

        q_in, docs_in, rag_res = st.session_state["last_rag_run"]

        st.markdown("---")
        st.markdown("### Chunk Classification Summary")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Scanned Chunks", rag_res.total_chunks)
        with c2:
            st.metric("Poisoned / Blocked Chunks", rag_res.poisoned_count)
        with c3:
            st.metric("Clean / Safe Chunks", len(rag_res.safe_documents))
        with c4:
            has_poison = rag_res.poisoned_count > 0
            st.markdown(f"**Context Status:**<br>{get_status_badge_html('WARN' if has_poison else 'SAFE')}", unsafe_allow_html=True)

        st.markdown("#### Per-Chunk Security Table")
        chunk_data = []
        for chk in rag_res.documents:
            chunk_data.append({
                "Index": f"Chunk #{chk.index + 1}",
                "Classification": "POISONED / INJECTION" if chk.is_poisoned else "SAFE / INFORMATION",
                "Threats Found": ", ".join(chk.threats) if chk.threats else "None",
                "Confidence": f"{chk.confidence:.2f}",
                "Text Preview": chk.text[:70] + "...",
            })
        st.table(chunk_data)

        if rag_res.poisoned_indices:
            st.markdown(
                f"""
                <div class="console-alert-danger">
                    <strong>Indirect Injections Quarantined:</strong> Chunks {', '.join(f'#{i+1}' for i in rag_res.poisoned_indices)} were stripped before context window assembly.
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_object_inspector(rag_res, title="RAGScanResult Inspection")
        render_copy_python(operation="rag")

render = render_rag_page

if __name__ == "__main__":
    render()
