"""Benchmark for asha.wrap_llm() overhead."""

import statistics
import time
from unittest.mock import MagicMock
from asha import wrap_llm

def run_benchmark():
    print("=== Benchmarking asha.wrap_llm() Overhead ===")
    
    raw_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Mock response"))]
    raw_client.chat.completions.create.return_value = mock_resp

    # Cold wrapping
    t0 = time.perf_counter()
    secure_client = wrap_llm(raw_client)
    cold_wrap_ms = (time.perf_counter() - t0) * 1000
    print(f"Cold Wrap Latency: {cold_wrap_ms:.2f} ms")

    # Execution overhead through SecureLLM proxy
    times = []
    for _ in range(25):
        t0 = time.perf_counter()
        resp = secure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Contact alice@corp.com for report"}]
        )
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    mean = statistics.mean(times)

    print(f"Warm Iterations: 25 calls")
    print(f"Mean Overhead:   {mean:.2f} ms")
    print(f"Median (P50):    {p50:.2f} ms")
    print(f"P95 Overhead:    {p95:.2f} ms")
    print("=" * 35)

if __name__ == "__main__":
    run_benchmark()
