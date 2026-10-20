# --- Merged from test_logging_security.py ---
"""Ensure raw secrets never appear in trace/metrics output."""

import json

from asha.types.results import ProcessResult
from asha.utils.dropin import process

from conftest import output_of


SECRET_PROMPT = (
    "Contact john@secret.com SSN 123-45-6789 "
    "key sk-abcdefghijklmnopqrstuvwxyz1234567890"
)


def test_trace_output_no_raw_pii(capsys):
    process(
        SECRET_PROMPT,
        trace=True,
        log_output="json",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "john@secret.com" not in combined
    assert "123-45-6789" not in combined
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in combined


def test_metrics_no_raw_pii_in_processed_output():
    result = process(SECRET_PROMPT)
    assert isinstance(result, ProcessResult)
    optimized = output_of(result)
    security = result.security.to_dict() if result.security else {}
    sanitized = security.get("sanitized_content", "")
    blob = json.dumps({"optimized": optimized, "sanitized": sanitized})
    assert "john@secret.com" not in blob
    assert "123-45-6789" not in blob


# --- Consolidated from tests/test_otel.py ---
"""OpenTelemetry optional integration tests."""

from asha.integrations.otel import enable_otel, get_tracer, trace_stage


def test_trace_stage_noop_when_otel_disabled():
    @trace_stage("unit_test")
    def add_one(x: int) -> int:
        return x + 1

    assert add_one(1) == 2


def test_get_tracer_none_by_default():
    assert get_tracer() is None


def test_enable_otel_returns_bool():
    result = enable_otel(service_name="asha-test")
    assert isinstance(result, bool)
    if result:
        assert get_tracer() is not None

        @trace_stage("enabled")
        def echo(value: str) -> str:
            return value

        assert echo("ok") == "ok"


def test_process_runs_with_otel_enabled_or_skipped():
    from asha import process

    enabled = enable_otel("asha-test-pipeline")
    result = process("Hello world", mode="off")
    assert result.output
    if not enabled:
        assert get_tracer() is None

