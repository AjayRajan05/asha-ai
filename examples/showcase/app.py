"""ASHA Control Center - Main Application Entrypoint.

Adaptive Security for Human-SI Systems Developer Showcase:
- Real library execution pipeline
- Interactive transparent developer UI
- Dual-mode offline local execution and cloud provider gateway
- Forensic audit trails, token diffs, and ANCHOR runtime governance
"""

from __future__ import annotations

import sys
from pathlib import Path

import os

# Ensure project root and showcase root are in sys.path
_SHOWCASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SHOWCASE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SHOWCASE_DIR) not in sys.path:
    sys.path.insert(0, str(_SHOWCASE_DIR))

# Auto-load .env if present
for _env_path in (
    _PROJECT_ROOT / ".env",
    _PROJECT_ROOT / "examples" / ".env",
    _SHOWCASE_DIR / ".env",
):
    if _env_path.exists():
        try:
            for _line in _env_path.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
        except Exception:
            pass

import streamlit as st

from examples.showcase.components.styles import inject_styles
from examples.showcase.components.header import render_header
from examples.showcase.components.sidebar import render_sidebar
from examples.showcase.runtime.session import init_session
from examples.showcase import pages


def _dispatch_page(page_module) -> None:
    """Safely execute the page's render callable, supporting hot-reloads and fallback naming."""
    render_fn = getattr(page_module, "render", None)
    if render_fn is None:
        for attr in dir(page_module):
            if attr.startswith("render_") and callable(getattr(page_module, attr)):
                render_fn = getattr(page_module, attr)
                break
    if render_fn is None:
        import importlib
        page_module = importlib.reload(page_module)
        render_fn = getattr(page_module, "render", None)
    if render_fn and callable(render_fn):
        render_fn()
    else:
        st.error(f"Could not find a valid render() function in {getattr(page_module, '__name__', str(page_module))}")


def main() -> None:
    st.set_page_config(
        page_title="ASHA Control Center",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Inject developer-console theme
    inject_styles()

    # 2. Initialize session state
    init_session()

    # 3. Render sidebar navigation & get active route
    current_page = render_sidebar()

    # 4. Route to page component
    route_map = {
        # OVERVIEW
        "Dashboard": pages.overview,
        "Runtime Status": pages.overview,
        "Provider Status": pages.overview,
        # PLAYGROUND
        "LLM Playground": pages.playground,
        "Process Pipeline": pages.process_pipeline,
        "API Explorer": pages.api_explorer,
        # SECURITY LABS
        "PII Protection": pages.security_pii,
        "Prompt Injection": pages.security_injection,
        "Unicode Security": pages.security_unicode,
        "Long Context": pages.security_long_ctx,
        "RAG Security": pages.security_rag,
        # OPTIMIZATION
        "Token Optimization": pages.optimization,
        "Optimization Rejection Demo": pages.optimization,
        # AGENTS / ANCHOR
        "ANCHOR Control Room": pages.anchor_control,
        "Action Governance": pages.anchor_actions,
        "Memory Security": pages.anchor_memory,
        "Intent Drift": pages.anchor_drift,
        "wrap_llm() Proxy": pages.wrap_llm_lab,
        # PROVIDERS
        "Provider Playground": pages.providers,
        "Google Gemini": pages.providers,
        "OpenAI": pages.providers,
        "Anthropic Claude": pages.providers,
        "Ollama Local": pages.providers,
        "llama.cpp Server": pages.providers,
        # OBSERVABILITY
        "Live Trace": pages.observability,
        "Request History": pages.observability,
        "Observability": pages.observability,
        # CONFIGURATION
        "PolicyConfig & Presets": pages.configuration,
        "Policy Configuration": pages.configuration,
    }

    target_module = route_map.get(current_page, pages.overview)
    _dispatch_page(target_module)



if __name__ == "__main__":
    main()
