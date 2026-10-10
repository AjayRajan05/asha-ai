"""ASHA Control Center - wrap_llm() Proxy Lab.

Interactive laboratory demonstrating transparent client proxying via wrap_llm() and SecureLLM:
- Non-mutating wrapper contract verification
- Provider interface adapters (OpenAI, Anthropic, Gemini, Ollama, OpenAI-compatible)
- Pre-call input protection & post-call rehydration
- Dynamic verification of original client immutability
"""

import copy
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Dict

_PAGES_DIR = Path(__file__).resolve().parent
_SHOWCASE_DIR = _PAGES_DIR.parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
for _p in (str(_PROJECT_ROOT), str(_SHOWCASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from examples.showcase.runtime.trace import Trace, EventType
from examples.showcase.runtime.session import record_trace
from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import render_status_pill
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.code_viewer import render_python_code


class MockLLMClient:
    """Simulated standard client demonstrating non-mutating proxying."""

    def __init__(self, provider_name: str, model_name: str):
        self.provider = provider_name
        self.model = model_name
        self._initial_state_token = f"immutable_token_{id(self)}"
        self.call_history: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.call_history.append(prompt)
        return f"[{self.provider}:{self.model}] Received guarded prompt: '{prompt}'. Response generated successfully."

    def generate_content(self, prompt: str, **kwargs: Any) -> str:
        self.call_history.append(prompt)
        return f"[{self.provider}:{self.model}] Received guarded prompt: '{prompt}'. Response generated successfully."


def render() -> None:
    render_header(title="WRAP_LLM() PROXY LAB", subtitle="Single-line client proxying with automatic PII rehydration and guardrails")


    col_prov, col_mode, col_rehydrate = st.columns([2, 1, 1])
    with col_prov:
        provider_name = st.selectbox(
            "Target Provider Interface",
            ["Google Gemini", "OpenAI", "Anthropic", "Ollama", "OpenAI-Compatible / llama.cpp"],
            index=0,
        )
    with col_mode:
        mode = st.selectbox("Policy Mode", ["balanced", "strict", "lite", "off"], index=0)
    with col_rehydrate:
        rehydrate = st.checkbox("Rehydrate PII in Response", value=True)

    test_prompt = st.text_area(
        "Client Prompt with Sensitive Data",
        value="Contact customer representative John Smith at john.smith@company.com or +1 (555) 234-5678 regarding invoice #9821.",
        height=90,
    )

    run_btn = st.button("Wrap Client & Execute Call", type="primary", use_container_width=True)

    if run_btn:
        trace = Trace(provider=provider_name, model=f"wrapped_{mode}")
        t0 = time.perf_counter()

        trace.add_event(
            EventType.INPUT_RECEIVED,
            status="INFO",
            reason=f"Initializing mock {provider_name} client for wrapping verification",
            metadata={"provider": provider_name, "mode": mode},
        )

        try:
            from asha import wrap_llm, SecureLLM, Profile

            # Create original client instance
            raw_client = MockLLMClient(provider_name, "preview-model")
            initial_dict = copy.copy(raw_client.__dict__)
            initial_id = id(raw_client)

            # Perform real wrapping
            t_wrap_0 = time.perf_counter()
            wrapped_client = wrap_llm(
                raw_client,
                mode=mode,
                rehydrate_pii=rehydrate,
            )
            wrap_latency_ms = (time.perf_counter() - t_wrap_0) * 1000.0

            # Verify non-mutation contract
            is_same_object = (raw_client is wrapped_client)
            client_dict_changed = (raw_client.__dict__ != initial_dict)
            client_mutated = is_same_object or client_dict_changed

            trace.add_event(
                EventType.RESULT_CREATED,
                status="SUCCESS",
                duration_ms=wrap_latency_ms,
                reason="Client wrapped into SecureLLM proxy successfully.",
                metadata={"is_proxy": isinstance(wrapped_client, SecureLLM), "client_mutated": client_mutated},
            )

            # Execute intercepted call
            trace.add_event(
                EventType.PROVIDER_REQUEST,
                status="INFO",
                reason="Executing intercepted client call via SecureLLM proxy...",
            )
            t_call_0 = time.perf_counter()
            if hasattr(wrapped_client, "generate_content") and "Gemini" in provider_name:
                response = wrapped_client.generate_content(test_prompt)
            else:
                response = wrapped_client.generate(test_prompt)
            call_latency_ms = (time.perf_counter() - t_call_0) * 1000.0

            total_ms = (time.perf_counter() - t0) * 1000.0
            trace.add_event(
                EventType.PROVIDER_RESPONSE,
                status="SUCCESS",
                duration_ms=call_latency_ms,
                reason="Response returned through ASHA output guard & rehydration stage.",
            )
            trace.add_event(
                EventType.REQUEST_COMPLETE,
                status="SUCCESS",
                duration_ms=total_ms,
                reason="wrap_llm() pipeline completed.",
            )
            record_trace(trace)

            # Contract verification indicators
            st.markdown("### 1. Immutability & Proxy Contract Verification")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**Original Client Mutated:**")
                mutated_badge = render_status_pill("YES (VIOLATION)", "blocked") if client_mutated else render_status_pill("NO (PRESERVED)", "success")
                st.markdown(mutated_badge, unsafe_allow_html=True)
            with c2:
                st.markdown("**Separate Proxy Instance:**")
                st.markdown(render_status_pill("YES", "success") if not is_same_object else render_status_pill("NO", "blocked"), unsafe_allow_html=True)
            with c3:
                st.markdown("**Proxy Class:**")
                st.code(type(wrapped_client).__name__, language="text")
            with c4:
                st.markdown("**Wrap Overhead:**")
                st.caption(f"{wrap_latency_ms:.2f} ms")

            st.markdown("<br>", unsafe_allow_html=True)

            # Architecture Flow
            st.markdown("### 2. Execution Flow")
            st.markdown(
                """
                <div style="background: #121820; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; font-family: monospace; font-size: 0.85rem; color: #cbd5e1; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #1e293b; padding: 6px 12px; border-radius: 4px;">User Caller</span>
                        <span style="color: #38bdf8;">→</span>
                        <span style="background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; padding: 6px 12px; border-radius: 4px; color: #38bdf8;">SecureLLM Proxy</span>
                        <span style="color: #38bdf8;">→</span>
                        <span style="background: rgba(168, 85, 247, 0.1); border: 1px solid #a855f7; padding: 6px 12px; border-radius: 4px; color: #c084fc;">ASHA Pre-Call (PII / Guard)</span>
                        <span style="color: #38bdf8;">→</span>
                        <span style="background: #1e293b; padding: 6px 12px; border-radius: 4px;">Raw Client Instance</span>
                        <span style="color: #38bdf8;">→</span>
                        <span style="background: #1e293b; padding: 6px 12px; border-radius: 4px;">Remote LLM</span>
                        <span style="color: #38bdf8;">→</span>
                        <span style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; padding: 6px 12px; border-radius: 4px; color: #34d399;">ASHA Post-Call (Rehydrate)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Output comparison
            st.markdown("### 3. Guarded Execution Result")
            st.text_area("Returned Response (with Rehydration)", value=str(response), height=100)

            # What reached the raw client?
            raw_received = raw_client.call_history[-1] if raw_client.call_history else "N/A"
            with st.expander("Inspect Actual Payload Received by Underlying Client", expanded=True):
                st.markdown(
                    "<p style='color: #94a3b8; font-size: 0.85rem;'>Notice that if PII masking occurred, "
                    "the raw client received masked tokens rather than sensitive data.</p>",
                    unsafe_allow_html=True,
                )
                st.code(raw_received, language="text")

            # Object Inspector
            render_object_inspector(
                raw_obj=wrapped_client,
                title="SecureLLM Proxy Object Inspector",
                human_data={
                    "Proxy Type": type(wrapped_client).__name__,
                    "Target Client Type": type(raw_client).__name__,
                    "Original Object ID": hex(initial_id),
                    "Proxy Object ID": hex(id(wrapped_client)),
                    "Original Mutated": str(client_mutated),
                    "Policy Mode": mode,
                    "Rehydrate PII": str(rehydrate),
                    "Wrap Latency": f"{wrap_latency_ms:.2f} ms",
                    "Call Latency": f"{call_latency_ms:.2f} ms",
                },
            )

            # Python Code Snippet
            st.markdown("<br>", unsafe_allow_html=True)
            if "Gemini" in provider_name:
                py_code = f"""import google.genai as genai
from asha import wrap_llm

raw_client = genai.Client()

# One-line transparent wrap - raw_client remains untouched
client = wrap_llm(raw_client, mode={repr(mode)}, rehydrate_pii={rehydrate})

# Call Gemini identically as before
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents={repr(test_prompt)}
)
"""
            elif "OpenAI" in provider_name:
                py_code = f"""import openai
from asha import wrap_llm

raw_client = openai.OpenAI()

# One-line transparent wrap - raw_client remains untouched
client = wrap_llm(raw_client, mode={repr(mode)}, rehydrate_pii={rehydrate})

# Call OpenAI identically as before
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{{"role": "user", "content": {repr(test_prompt)}}}]
)
"""
            else:
                py_code = f"""from asha import wrap_llm

# Wrap any existing LLM client instance
client = wrap_llm(raw_client, mode={repr(mode)}, rehydrate_pii={rehydrate})

# All methods and attributes forward directly with ASHA security active
response = client.generate({repr(test_prompt)})
"""
            render_python_code(py_code, title=f"wrap_llm() Integration for {provider_name}")

        except Exception as ex:
            st.error(f"wrap_llm evaluation error: {ex}")
            trace.add_event(EventType.ERROR, status="ERROR", reason=str(ex))
            record_trace(trace)


if __name__ == "__main__":
    render()

