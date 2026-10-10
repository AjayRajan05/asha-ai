"""Unit tests for showcase session metric calculations."""

import pytest

pytest.importorskip("streamlit", reason="showcase tests require the 'demo' extra (streamlit)")
pytest.importorskip("plotly", reason="showcase tests require the 'demo' extra (plotly)")

from examples.showcase.runtime.trace import Trace, EventType
from examples.showcase.runtime.session import (
    calculate_session_metrics,
    record_trace,
    clear_history,
)


def test_empty_metrics():
    clear_history()
    metrics = calculate_session_metrics()
    assert metrics["requests"] == 0
    assert metrics["pii_events"] == 0
    assert metrics["tokens_saved"] == 0
    assert metrics["blocked_actions"] == 0
    assert metrics["security_events"] == 0
    assert metrics["avg_latency_ms"] == 0.0


def test_accumulated_metrics():
    clear_history()

    # Trace 1: PII masking with 50 tokens saved
    t1 = Trace(provider="test", model="mock")
    t1.add_event(EventType.PII_SCAN, status="SUCCESS", metadata={"masked_count": 3})
    t1.add_event(EventType.OPTIMIZATION_ACCEPTED, status="SUCCESS", metadata={"tokens_saved": 50})
    t1.add_event(EventType.REQUEST_COMPLETE, status="SUCCESS", duration_ms=100.0)
    record_trace(t1)

    # Trace 2: Blocked injection
    t2 = Trace(provider="test", model="mock")
    t2.add_event(EventType.ACTION_BLOCKED, status="BLOCKED", reason="Injection blocked")
    t2.add_event(EventType.REQUEST_COMPLETE, status="BLOCKED", duration_ms=200.0)
    record_trace(t2)

    metrics = calculate_session_metrics()
    assert metrics["requests"] == 2
    assert metrics["pii_events"] == 3
    assert metrics["tokens_saved"] == 50
    assert metrics["blocked_actions"] == 1
    assert metrics["security_events"] == 2
    assert metrics["avg_latency_ms"] == pytest.approx(150.0, rel=1e-2)
