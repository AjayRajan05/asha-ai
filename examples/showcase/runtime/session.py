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

"""Session state management and history tracking for ASHA Control Center."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import streamlit as st

from .trace import Trace

_BACKING_STATE: Dict[str, Any] = {
    "history": [],
    "active_trace": None,
    "transparency_mode": "Transparent",
    "nav": "Dashboard",
    "selected_provider": "local",
}


def _get_state_dict() -> Any:
    """Return st.session_state if in active Streamlit context, otherwise fallback store."""
    try:
        if hasattr(st, "session_state") and "history" in st.session_state:
            return st.session_state
        if hasattr(st, "session_state"):
            try:
                st.session_state["_test_probe"] = True
                del st.session_state["_test_probe"]
                return st.session_state
            except Exception:
                pass
    except Exception:
        pass
    return _BACKING_STATE


def init_session() -> None:
    """Initialize session state defaults if not already present."""
    state = _get_state_dict()
    if "history" not in state:
        state["history"] = []
    if "active_trace" not in state:
        state["active_trace"] = None
    if "transparency_mode" not in state:
        state["transparency_mode"] = "Transparent"
    if "nav" not in state:
        state["nav"] = "Dashboard"
    if "selected_provider" not in state:
        backend_env = (os.getenv("ASHA_DEMO_BACKEND") or "local").strip().lower()
        if backend_env in ("gemini", "openai", "anthropic", "ollama", "llamacpp", "local"):
            state["selected_provider"] = backend_env
        else:
            state["selected_provider"] = "local"


init_session_state = init_session


def add_trace_to_history(trace: Trace) -> None:
    """Add a completed trace to the session history."""
    state = _get_state_dict()
    if "history" not in state:
        state["history"] = []
    state["history"].insert(0, trace)
    state["active_trace"] = trace
    # Keep backing store in sync
    if state is not _BACKING_STATE:
        _BACKING_STATE["history"].insert(0, trace)
        _BACKING_STATE["active_trace"] = trace


record_trace = add_trace_to_history


def get_history() -> List[Trace]:
    """Retrieve session history list."""
    state = _get_state_dict()
    hist = state.get("history", [])
    if not hist and _BACKING_STATE.get("history"):
        return _BACKING_STATE["history"]
    return hist


def clear_history() -> None:
    """Clear session history (useful for test isolation)."""
    state = _get_state_dict()
    state["history"] = []
    state["active_trace"] = None
    _BACKING_STATE["history"] = []
    _BACKING_STATE["active_trace"] = None


def get_current_trace() -> Optional[Trace]:
    """Get the currently active trace."""
    state = _get_state_dict()
    return state.get("active_trace") or _BACKING_STATE.get("active_trace")


def set_current_trace(trace: Trace) -> None:
    """Set the currently active trace."""
    state = _get_state_dict()
    state["active_trace"] = trace
    _BACKING_STATE["active_trace"] = trace


def get_session_metrics() -> Dict[str, Any]:
    """Calculate aggregate session metrics from real execution history."""
    history = get_history()
    if not history:
        return {
            "has_data": False,
            "requests": 0,
            "pii_events": 0,
            "tokens_saved": 0,
            "blocked_actions": 0,
            "security_events": 0,
            "avg_latency_ms": 0.0,
        }

    total_requests = len(history)

    total_pii = 0
    total_saved = 0
    total_blocked = 0
    security_event_count = 0

    latencies: List[float] = []

    for t in history:
        t_blocked = 0
        t_pii = 0

        if hasattr(t, "duration_ms") and t.duration_ms > 0:
            latencies.append(t.duration_ms)
        elif hasattr(t, "total_latency_ms") and t.total_latency_ms > 0:
            latencies.append(t.total_latency_ms)

        for ev in t.events:
            ev_type = getattr(ev, "event_type", None)
            ev_name = ev_type.name if hasattr(ev_type, "name") else str(ev_type)

            if "PII" in ev_name:
                masked = ev.metadata.get("masked_count", 1) if ev.metadata else 1
                t_pii += int(masked)
            if "OPTIMIZATION" in ev_name and ev.metadata:
                total_saved += int(ev.metadata.get("tokens_saved", 0))
            if "ACTION_BLOCKED" in ev_name or (ev_name == "ACTION_BLOCKED"):
                t_blocked += 1

        total_pii += t_pii
        total_blocked += t_blocked
        if t_blocked > 0 or t_pii > 0:
            security_event_count += 1

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "has_data": True,
        "requests": total_requests,
        "pii_events": total_pii,
        "tokens_saved": total_saved,
        "blocked_actions": total_blocked,
        "security_events": security_event_count,
        "avg_latency_ms": round(avg_lat, 1),
    }


calculate_session_metrics = get_session_metrics


def get_provider_status() -> Dict[str, Dict[str, Any]]:
    """
    Check configuration and availability of supported LLM providers.
    NEVER returns or logs raw API keys.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_status = {
        "name": "Google Gemini",
        "configured": bool(gemini_key),
        "status_label": "Configured ✓" if gemini_key else "Not configured",
        "models": ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "default_model": "gemini-3.8-flash",
    }

    openai_key = os.getenv("OPENAI_API_KEY")
    openai_status = {
        "name": "OpenAI",
        "configured": bool(openai_key),
        "status_label": "Configured ✓" if openai_key else "Not configured",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "default_model": "gpt-4o-mini",
    }

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_status = {
        "name": "Anthropic Claude",
        "configured": bool(anthropic_key),
        "status_label": "Configured ✓" if anthropic_key else "Not configured",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "default_model": "claude-3-5-sonnet-20241022",
    }

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_connected = False
    try:
        import requests
        resp = requests.get(f"{ollama_host}/api/tags", timeout=0.8)
        ollama_connected = (resp.status_code == 200)
    except Exception:
        ollama_connected = False

    ollama_status = {
        "name": "Ollama (Local)",
        "configured": True,
        "endpoint": ollama_host,
        "status_label": "Connected ✓" if ollama_connected else "Offline (not running)",
        "connected": ollama_connected,
        "models": ["llama3", "llama3.2", "mistral", "phi3"],
        "default_model": "llama3",
    }

    llamacpp_url = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
    llamacpp_status = {
        "name": "llama.cpp (OpenAI-compatible server)",
        "configured": True,
        "endpoint": llamacpp_url,
        "status_label": "Configured endpoint",
        "default_model": "llama-cpp",
    }

    local_status = {
        "name": "Local / Demo Provider",
        "configured": True,
        "status_label": "Ready (Offline & Deterministic)",
        "models": ["asha-local-deterministic-v1"],
        "default_model": "asha-local-deterministic-v1",
    }

    return {
        "local": local_status,
        "gemini": gemini_status,
        "openai": openai_status,
        "anthropic": anthropic_status,
        "ollama": ollama_status,
        "llamacpp": llamacpp_status,
    }


def get_provider_statuses() -> List[Dict[str, Any]]:
    """Return provider statuses as a list of dicts with no secret leakage."""
    status_map = get_provider_status()
    result = [
        {
            "provider": "Google Gemini",
            "configured": status_map["gemini"]["configured"],
            "status_text": status_map["gemini"]["status_label"],
            "details": f"Models: {', '.join(status_map['gemini']['models'])}",
        },
        {
            "provider": "OpenAI",
            "configured": status_map["openai"]["configured"],
            "status_text": status_map["openai"]["status_label"],
            "details": f"Models: {', '.join(status_map['openai']['models'])}",
        },
        {
            "provider": "Anthropic",
            "configured": status_map["anthropic"]["configured"],
            "status_text": status_map["anthropic"]["status_label"],
            "details": f"Models: {', '.join(status_map['anthropic']['models'])}",
        },
        {
            "provider": "Ollama",
            "configured": status_map["ollama"]["configured"],
            "status_text": status_map["ollama"]["status_label"],
            "details": f"Host: {status_map['ollama']['endpoint']}",
        },
        {
            "provider": "llama.cpp (OpenAI Compatible)",
            "configured": status_map["llamacpp"]["configured"],
            "status_text": status_map["llamacpp"]["status_label"],
            "details": f"Endpoint: {status_map['llamacpp']['endpoint']}",
        },
        {
            "provider": "Local Demo Provider",
            "configured": status_map["local"]["configured"],
            "status_text": status_map["local"]["status_label"],
            "details": "Offline deterministic preview provider",
        },
    ]
    return result
