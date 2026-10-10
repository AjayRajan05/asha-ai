"""Thread-safety and concurrent access tests."""

import concurrent.futures
import threading

import pytest

from asha.core.metrics_dashboard import dashboard
from asha.types.results import ProcessResult
from asha.utils.dropin import process

from conftest import output_of


@pytest.mark.slow
def test_concurrent_process_isolation():
    prompts = [f"Email user{i}@example.com task {i}" for i in range(20)]
    errors = []

    def worker(prompt):
        try:
            result = process(prompt)
            out = output_of(result)
            if "@" in out:
                errors.append(f"PII leak in: {prompt[:30]}")
        except Exception as exc:
            errors.append(str(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(worker, prompts))

    assert not errors, errors


def test_metrics_dashboard_concurrent_recording():
    start = len(dashboard.collector.events)

    def record_one(_):
        from asha.core.metrics_dashboard import record_request

        record_request(latency_ms=1.0, tokens_processed=10, tokens_saved=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record_one, range(40)))

    assert len(dashboard.collector.events) >= start + 40


def test_security_layer_instances_thread_local():
    from asha.core.security.security_layer import SecurityLayer

    results = []

    def worker(email):
        layer = SecurityLayer()
        out = layer.process(f"Contact {email}")
        results.append(email not in out.sanitized_content)

    threads = [
        threading.Thread(target=worker, args=(f"user{i}@test.com",)) for i in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results)
