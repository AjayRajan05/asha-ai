"""ASHA Control Center - Provider Playground.

Multi-provider inspection and testing:
- Google Gemini (google-genai / legacy fallback)
- OpenAI & OpenAI-compatible / llama.cpp server
- Anthropic Claude
- Ollama local daemon
- Local Demo Provider (deterministic offline)
- Capability Matrix & credential configuration status (zero secret exposure)
"""

import os
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

import requests
import streamlit as st

from examples.showcase.runtime.trace import Trace, EventType
from examples.showcase.runtime.session import record_trace, get_provider_statuses
from examples.showcase.runtime.local_provider import LocalDemoProvider
from examples.showcase.components.header import render_header
from examples.showcase.components.status_badge import render_status_pill
from examples.showcase.components.provider_status import render_provider_status_table
from examples.showcase.components.object_inspector import render_object_inspector
from examples.showcase.components.code_viewer import render_python_code


def render() -> None:
    render_header(title="PROVIDER GATEWAY", subtitle="Multi-provider configuration, key management, and capability matrix")


    tab_matrix, tab_gemini, tab_openai, tab_ollama, tab_anthropic, tab_local = st.tabs([
        "Capability Matrix",
        "Google Gemini",
        "OpenAI & llama.cpp",
        "Ollama Local",
        "Anthropic Claude",
        "Local Demo Mode",
    ])

    # -------------------------------------------------------------
    # TAB 1: Capability Matrix
    # -------------------------------------------------------------
    with tab_matrix:
        st.markdown("### Ecosystem Capability Matrix")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.85rem;'>"
            "Features supported by ASHA adapters across underlying provider ecosystems.</p>",
            unsafe_allow_html=True,
        )

        matrix_data = [
            {"Provider": "Google Gemini", "Text": "✓", "System Prompt": "✓", "Streaming": "✓", "Async": "✓", "Tools": "✓", "Structured Output": "✓", "Usage Meta": "✓", "Config": "GOOGLE_API_KEY"},
            {"Provider": "OpenAI", "Text": "✓", "System Prompt": "✓", "Streaming": "✓", "Async": "✓", "Tools": "✓", "Structured Output": "✓", "Usage Meta": "✓", "Config": "OPENAI_API_KEY"},
            {"Provider": "Anthropic Claude", "Text": "✓", "System Prompt": "✓", "Streaming": "✓", "Async": "✓", "Tools": "✓", "Structured Output": "✓", "Usage Meta": "✓", "Config": "ANTHROPIC_API_KEY"},
            {"Provider": "Ollama", "Text": "✓", "System Prompt": "✓", "Streaming": "✓", "Async": "✓", "Tools": "Partial", "Structured Output": "✓", "Usage Meta": "✓", "Config": "OLLAMA_HOST"},
            {"Provider": "llama.cpp (OpenAI API)", "Text": "✓", "System Prompt": "✓", "Streaming": "✓", "Async": "✓", "Tools": "Partial", "Structured Output": "✓", "Usage Meta": "✓", "Config": "LLAMA_CPP_BASE_URL"},
            {"Provider": "Local Demo Provider", "Text": "✓", "System Prompt": "✓", "Streaming": "N/A", "Async": "✓", "Tools": "Simulated", "Structured Output": "✓", "Usage Meta": "Simulated", "Config": "None (Offline)"},
        ]
        st.dataframe(matrix_data, use_container_width=True, hide_index=True)

        st.markdown("### Credential & Connectivity Status")
        render_provider_status_table()

    # -------------------------------------------------------------
    # TAB 2: Google Gemini
    # -------------------------------------------------------------
    with tab_gemini:
        st.markdown("### Google Gemini Integration")
        has_gemini_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

        with st.expander("Gemini API Key Setup", expanded=not has_gemini_key):
            st.markdown(
                "Configure your Gemini API key through any of the following methods:\n"
                "- **Instant UI Configuration:** Enter key below and click **Save Key**.\n"
                "- **Local `.env` File:** Create a `.env` file at project root with `GEMINI_API_KEY=AIzaSy...`\n"
                "- **Terminal Environment Variable:** `$env:GEMINI_API_KEY=\"AIzaSy...\"` (PowerShell) or `export GEMINI_API_KEY=\"...\"` (Bash)"
            )
            raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            key_val = st.text_input(
                "Gemini API Key",
                type="password",
                value=raw_key,
                placeholder="AIzaSy...",
                key="gemini_key_direct_input",
            )
            col_k1, col_k2 = st.columns([1, 1])
            with col_k1:
                if st.button("Activate for Current Session", key="btn_save_gemini_session"):
                    if key_val.strip():
                        os.environ["GEMINI_API_KEY"] = key_val.strip()
                        os.environ["GOOGLE_API_KEY"] = key_val.strip()
                        st.success("Gemini API key activated for this session!")
                        st.rerun()
            with col_k2:
                if st.button("Save to .env (Persistent)", key="btn_save_gemini_env"):
                    if key_val.strip():
                        os.environ["GEMINI_API_KEY"] = key_val.strip()
                        os.environ["GOOGLE_API_KEY"] = key_val.strip()
                        env_file = _PROJECT_ROOT / ".env"
                        existing = []
                        if env_file.exists():
                            existing = [l for l in env_file.read_text(encoding="utf-8").splitlines() if not l.startswith("GEMINI_API_KEY=") and not l.startswith("GOOGLE_API_KEY=")]
                        existing.append(f"GEMINI_API_KEY={key_val.strip()}")
                        existing.append(f"GOOGLE_API_KEY={key_val.strip()}")
                        env_file.write_text("\n".join(existing) + "\n", encoding="utf-8")
                        st.success("Saved to .env and activated!")
                        st.rerun()

        if not has_gemini_key:
            st.info(
                "No Gemini API key is currently detected. You can enter one above to test live model inference, "
                "or test deterministic offline inference using the Local Demo Provider."
            )

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            gemini_model = st.selectbox(
                "Gemini Model",
                ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                index=0,
            )
        with col_g2:
            st.markdown("**Status:**")
            st.markdown(
                render_status_pill("CONFIGURED" if has_gemini_key else "NOT CONFIGURED", "success" if has_gemini_key else "warning"),
                unsafe_allow_html=True,
            )

        sys_prompt = st.text_input("System Instruction", value="You are a helpful and concise enterprise security assistant.")
        prompt_text = st.text_area("User Prompt", value="Explain how token optimization reduces LLM operational costs in 2 sentences.", height=90)

        run_gemini_btn = st.button("Execute with GeminiAdapter", type="primary", disabled=not has_gemini_key)

        if run_gemini_btn:
            trace = Trace(provider="Google Gemini", model=gemini_model)
            t0 = time.perf_counter()
            trace.add_event(EventType.INPUT_RECEIVED, status="INFO", reason="Dispatching request to GeminiAdapter")

            try:
                from asha.runtime.adapters.gemini_adapter import GeminiAdapter

                trace.add_event(EventType.PROVIDER_REQUEST, status="INFO", reason=f"Connecting to Gemini ({gemini_model})...")
                t_call = time.perf_counter()
                adapter = GeminiAdapter(model=gemini_model)
                res = adapter.generate(prompt_text, system_instruction=sys_prompt)
                call_ms = (time.perf_counter() - t_call) * 1000.0
                total_ms = (time.perf_counter() - t0) * 1000.0

                trace.add_event(
                    EventType.PROVIDER_RESPONSE,
                    status="SUCCESS",
                    duration_ms=call_ms,
                    reason="Received response from Google Gemini.",
                )
                trace.add_event(EventType.REQUEST_COMPLETE, status="SUCCESS", duration_ms=total_ms, reason="Completed.")
                record_trace(trace)

                st.success(f"Received response in {call_ms:.1f} ms")
                st.text_area("Gemini Output", value=str(res), height=140)

            except Exception as ex:
                st.error(f"Gemini execution error: {ex}")
                trace.add_event(EventType.ERROR, status="ERROR", reason=str(ex))
                record_trace(trace)

        # Python Code
        gemini_code = f"""import os
from asha.runtime.adapters.gemini_adapter import GeminiAdapter

adapter = GeminiAdapter(
    model={repr(gemini_model)},
    api_key=os.getenv("GOOGLE_API_KEY"),
)

response = adapter.generate(
    prompt={repr(prompt_text)},
    system_instruction={repr(sys_prompt)},
)
print(response)
"""
        st.markdown("<br>", unsafe_allow_html=True)
        render_python_code(gemini_code, title="Google Gemini Adapter Snippet")

    # -------------------------------------------------------------
    # TAB 3: OpenAI & llama.cpp
    # -------------------------------------------------------------
    with tab_openai:
        st.markdown("### OpenAI & OpenAI-Compatible / llama.cpp")
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

        with st.expander("OpenAI API Key Setup", expanded=not has_openai_key):
            st.markdown(
                "Configure your OpenAI API key through any of the following methods:\n"
                "- **Instant UI Configuration:** Enter key below and click **Activate for Current Session**.\n"
                "- **Local `.env` File:** Create a `.env` file at project root with `OPENAI_API_KEY=sk-...`\n"
                "- **Terminal Environment Variable:** `$env:OPENAI_API_KEY=\"sk-...\"` (PowerShell) or `export OPENAI_API_KEY=\"...\"` (Bash)"
            )
            raw_o_key = os.getenv("OPENAI_API_KEY") or ""
            key_o_val = st.text_input(
                "OpenAI API Key",
                type="password",
                value=raw_o_key,
                placeholder="sk-...",
                key="openai_key_direct_input",
            )
            col_ok1, col_ok2 = st.columns([1, 1])
            with col_ok1:
                if st.button("Activate for Current Session", key="btn_save_openai_session"):
                    if key_o_val.strip():
                        os.environ["OPENAI_API_KEY"] = key_o_val.strip()
                        st.success("OpenAI API key activated for this session!")
                        st.rerun()
            with col_ok2:
                if st.button("Save to .env (Persistent)", key="btn_save_openai_env"):
                    if key_o_val.strip():
                        os.environ["OPENAI_API_KEY"] = key_o_val.strip()
                        env_file = _PROJECT_ROOT / ".env"
                        existing = []
                        if env_file.exists():
                            existing = [l for l in env_file.read_text(encoding="utf-8").splitlines() if not l.startswith("OPENAI_API_KEY=")]
                        existing.append(f"OPENAI_API_KEY={key_o_val.strip()}")
                        env_file.write_text("\n".join(existing) + "\n", encoding="utf-8")
                        st.success("Saved to .env and activated!")
                        st.rerun()

        c_o1, c_o2 = st.columns([2, 1])
        with c_o1:
            target_type = st.radio(
                "Interface Mode",
                ["Official OpenAI API", "OpenAI-Compatible Server (llama.cpp / vLLM / LocalAI)"],
                index=0,
                horizontal=True,
            )
        with c_o2:
            st.markdown("**Credential Status:**")
            st.markdown(
                render_status_pill("CONFIGURED" if has_openai_key else "NOT CONFIGURED", "success" if has_openai_key else "warning"),
                unsafe_allow_html=True,
            )

        if "llama.cpp" in target_type:
            base_url = st.text_input("OpenAI-Compatible Server Base URL", value="http://localhost:8080/v1")
            o_model = st.text_input("Server Model Identifier", value="meta-llama-3.1-8b-instruct")
            st.info("llama.cpp is accessed via its standard OpenAI-compatible `/v1/chat/completions` HTTP server interface.")
        else:
            base_url = "https://api.openai.com/v1"
            o_model = st.selectbox("OpenAI Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)

        o_prompt = st.text_area("Prompt", value="Verify system boundary defense posture.", height=80, key="o_prompt")
        run_o_btn = st.button("Execute OpenAI Request", type="primary", disabled=not has_openai_key and "llama.cpp" not in target_type)

        if run_o_btn:
            trace = Trace(provider="OpenAI" if "Official" in target_type else "llama.cpp", model=o_model)
            t0 = time.perf_counter()
            trace.add_event(EventType.INPUT_RECEIVED, status="INFO", reason=f"Connecting to {o_model}")

            try:
                from asha.runtime.adapters.openai_adapter import OpenAIAdapter

                adapter = OpenAIAdapter(model=o_model, base_url=base_url if "llama.cpp" in target_type else None)
                t_c = time.perf_counter()
                res = adapter.generate(o_prompt)
                c_ms = (time.perf_counter() - t_c) * 1000.0

                trace.add_event(EventType.PROVIDER_RESPONSE, status="SUCCESS", duration_ms=c_ms, reason="Success.")
                record_trace(trace)
                st.success(f"Response received in {c_ms:.1f} ms")
                st.text_area("Output", value=str(res), height=120)

            except Exception as ex:
                st.error(f"OpenAI error: {ex}")
                trace.add_event(EventType.ERROR, status="ERROR", reason=str(ex))
                record_trace(trace)

    # -------------------------------------------------------------
    # TAB 4: Ollama Local
    # -------------------------------------------------------------
    with tab_ollama:
        st.markdown("### Ollama Local Engine")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        col_ol1, col_ol2 = st.columns([2, 1])
        with col_ol1:
            host_input = st.text_input("Ollama Daemon Endpoint", value=ollama_host)
        with col_ol2:
            st.markdown("<br>", unsafe_allow_html=True)
            check_ollama_btn = st.button("Check Connection", use_container_width=True)

        ollama_status_msg = "Not tested"
        ollama_models_found: list[str] = []
        is_ollama_live = False

        if check_ollama_btn:
            try:
                r = requests.get(f"{host_input.rstrip('/')}/api/tags", timeout=2.0)
                if r.status_code == 200:
                    is_ollama_live = True
                    data = r.json()
                    ollama_models_found = [m["name"] for m in data.get("models", [])]
                    ollama_status_msg = f"Connected ({len(ollama_models_found)} models available)"
                else:
                    ollama_status_msg = f"Error: HTTP {r.status_code}"
            except Exception as ex:
                ollama_status_msg = f"Unavailable: {ex}"

        st.markdown(f"**Connection Status:** {render_status_pill('CONNECTED' if is_ollama_live else 'UNAVAILABLE', 'success' if is_ollama_live else 'blocked')}", unsafe_allow_html=True)
        st.caption(ollama_status_msg)

        chosen_ollama_model = st.selectbox(
            "Installed Model",
            ollama_models_found if ollama_models_found else ["llama3.2:latest", "mistral:latest", "phi3:latest"],
            index=0,
        )

        ollama_prompt = st.text_area("Ollama Prompt", value="Summarize the importance of local execution for data privacy in 1 sentence.", height=80, key="ollama_prompt")
        run_ollama_btn = st.button("Execute on Local Ollama", type="primary", disabled=not is_ollama_live)

        if run_ollama_btn:
            try:
                from asha.runtime.adapters.ollama_adapter import OllamaAdapter

                t_ol = time.perf_counter()
                adapter = OllamaAdapter(model=chosen_ollama_model, base_url=host_input)
                out = adapter.generate(ollama_prompt)
                ms = (time.perf_counter() - t_ol) * 1000.0
                st.success(f"Generated locally in {ms:.1f} ms")
                st.text_area("Ollama Output", value=str(out), height=120)
            except Exception as ex:
                st.error(f"Ollama execution error: {ex}")

    # -------------------------------------------------------------
    # TAB 5: Anthropic Claude
    # -------------------------------------------------------------
    with tab_anthropic:
        st.markdown("### Anthropic Claude Integration")
        has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))

        with st.expander("Anthropic API Key Setup", expanded=not has_anthropic_key):
            st.markdown(
                "Configure your Anthropic API key through any of the following methods:\n"
                "- **Instant UI Configuration:** Enter key below and click **Activate for Current Session**.\n"
                "- **Local `.env` File:** Create a `.env` file at project root with `ANTHROPIC_API_KEY=sk-ant-...`\n"
                "- **Terminal Environment Variable:** `$env:ANTHROPIC_API_KEY=\"sk-ant-...\"` (PowerShell) or `export ANTHROPIC_API_KEY=\"...\"` (Bash)"
            )
            raw_c_key = os.getenv("ANTHROPIC_API_KEY") or ""
            key_c_val = st.text_input(
                "Anthropic API Key",
                type="password",
                value=raw_c_key,
                placeholder="sk-ant-...",
                key="anthropic_key_direct_input",
            )
            col_ck1, col_ck2 = st.columns([1, 1])
            with col_ck1:
                if st.button("Activate for Current Session", key="btn_save_anthropic_session"):
                    if key_c_val.strip():
                        os.environ["ANTHROPIC_API_KEY"] = key_c_val.strip()
                        st.success("Anthropic API key activated for this session!")
                        st.rerun()
            with col_ck2:
                if st.button("Save to .env (Persistent)", key="btn_save_anthropic_env"):
                    if key_c_val.strip():
                        os.environ["ANTHROPIC_API_KEY"] = key_c_val.strip()
                        env_file = _PROJECT_ROOT / ".env"
                        existing = []
                        if env_file.exists():
                            existing = [l for l in env_file.read_text(encoding="utf-8").splitlines() if not l.startswith("ANTHROPIC_API_KEY=")]
                        existing.append(f"ANTHROPIC_API_KEY={key_c_val.strip()}")
                        env_file.write_text("\n".join(existing) + "\n", encoding="utf-8")
                        st.success("Saved to .env and activated!")
                        st.rerun()

        st.markdown(f"**Credential Status:** {render_status_pill('CONFIGURED' if has_anthropic_key else 'NOT CONFIGURED', 'success' if has_anthropic_key else 'warning')}", unsafe_allow_html=True)

        claude_model = st.selectbox(
            "Claude Model",
            ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
            index=0,
        )
        c_prompt = st.text_area("Claude Prompt", value="Draft a security policy rule for preventing SQL injection.", height=90, key="c_prompt")
        run_c_btn = st.button("Execute Claude Request", type="primary", disabled=not has_anthropic_key)

        if run_c_btn:
            try:
                from asha.runtime.adapters.claude_adapter import ClaudeAdapter

                adapter = ClaudeAdapter(model=claude_model)
                out = adapter.generate(c_prompt)
                st.text_area("Claude Output", value=str(out), height=120)
            except Exception as ex:
                st.error(f"Claude error: {ex}")

    # -------------------------------------------------------------
    # TAB 6: Local Demo Mode
    # -------------------------------------------------------------
    with tab_local:
        st.markdown("### Deterministic Local Demo Provider")
        st.markdown(
            """
            <div style="background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 14px; margin-bottom: 16px;">
                <span style="font-weight: 700; color: #38bdf8;">100% OFFLINE CAPABLE</span><br>
                <span style="color: #cbd5e1; font-size: 0.9rem;">
                    The Local Demo Provider is a built-in deterministic engine designed for offline testing,
                    CI/CD test suites, and developer previews without requiring any remote cloud API keys or GPU hardware.
                    All ASHA security, sanitization, PII masking, and optimization steps execute completely for real.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        test_local_prompt = st.text_area(
            "Local Test Prompt",
            value="Hello! I need assistance configuring zero-trust boundaries for our internal agents.",
            height=80,
            key="local_demo_prompt",
        )

        if st.button("Generate with Local Demo Engine", type="primary"):
            p = LocalDemoProvider()
            t_loc = time.perf_counter()
            resp = p.generate(test_local_prompt)
            l_ms = (time.perf_counter() - t_loc) * 1000.0

            trace = Trace(provider="Local Demo", model=resp.get("model", "asha-local-deterministic-v1"), input_text=test_local_prompt)
            trace.add_event(EventType.INPUT_RECEIVED, status="INFO", reason="Local deterministic prompt received")
            trace.add_event(EventType.PROVIDER_RESPONSE, status="SUCCESS", duration_ms=l_ms, reason="Deterministic response generated offline")
            trace.add_event(EventType.REQUEST_COMPLETE, status="SUCCESS", duration_ms=l_ms, reason="Completed offline execution")
            record_trace(trace)

            st.success(f"Generated locally in {l_ms:.2f} ms")
            st.markdown("**Deterministic Output:**")
            st.code(resp.get("text", str(resp)), language="markdown")
            render_object_inspector(
                raw_obj=resp,
                title="LocalDemoResponse Inspector",
                human_data={
                    "Provider": resp.get("provider", "Local Demo"),
                    "Model": resp.get("model", "asha-local-deterministic-v1"),
                    "Tokens Used": resp.get("usage", {}).get("total_tokens", 0),
                    "Latency": f"{l_ms:.2f} ms",
                    "Is Synthetic": "Yes (Offline Preview)",
                },
            )


if __name__ == "__main__":
    render()


