"""Unit tests for ASHA Control Center Trace subsystem."""

import json
from examples.showcase.runtime.trace import Trace, TraceEvent, EventType


def test_trace_creation_and_attributes():
    trace = Trace(provider="Google Gemini", model="gemini-2.5-flash")
    assert trace.trace_id is not None
    assert len(trace.trace_id) >= 8
    assert trace.provider == "Google Gemini"
    assert trace.model == "gemini-2.5-flash"
    assert trace.status in ("SUCCESS", "PENDING")
    assert len(trace.events) == 0


def test_event_ordering_and_durations():
    trace = Trace(provider="test", model="mock")
    ev1 = trace.add_event(EventType.INPUT_RECEIVED, status="INFO", reason="Input")
    ev2 = trace.add_event(EventType.PII_SCAN, status="INFO", reason="Scanning PII", duration_ms=12.5)
    ev3 = trace.add_event(EventType.REQUEST_COMPLETE, status="SUCCESS", reason="Finished")

    assert len(trace.events) == 3
    assert trace.events[0].event_type == EventType.INPUT_RECEIVED
    assert trace.events[1].event_type == EventType.PII_SCAN
    assert trace.events[2].event_type == EventType.REQUEST_COMPLETE
    assert trace.events[1].duration_ms == 12.5
    assert trace.events[0].timestamp <= trace.events[1].timestamp <= trace.events[2].timestamp


def test_secret_redaction_in_trace_export():
    trace = Trace(provider="test", model="mock")
    trace.add_event(
        EventType.PROVIDER_REQUEST,
        status="INFO",
        reason="Sending to upstream",
        metadata={
            "api_key": "sk-real_super_secret_key_12345",
            "authorization": "Bearer token_secret_9999",
            "user_data": "Normal query",
            "nested": {
                "password": "my_master_password",
                "safe_field": 42,
            },
        },
    )

    data_json = trace.to_json()
    assert "sk-real_super_secret_key_12345" not in data_json
    assert "token_secret_9999" not in data_json
    assert "my_master_password" not in data_json
    assert "[REDACTED_SECRET]" in data_json

    parsed = json.loads(data_json)
    meta = parsed["events"][0]["metadata"]
    assert meta["api_key"] == "[REDACTED_SECRET]"
    assert meta["authorization"] == "[REDACTED_SECRET]"
    assert meta["user_data"] == "Normal query"
    assert meta["nested"]["password"] == "[REDACTED_SECRET]"
    assert meta["nested"]["safe_field"] == 42


def test_trace_status_blocked_and_error():
    trace = Trace(provider="test", model="mock")
    trace.add_event(EventType.ACTION_ALLOWED, status="SUCCESS")
    assert trace.status == "SUCCESS"

    trace.add_event(EventType.ACTION_BLOCKED, status="BLOCKED")
    assert trace.status == "BLOCKED"

    trace.add_event(EventType.ERROR, status="ERROR")
    assert trace.status == "ERROR"
