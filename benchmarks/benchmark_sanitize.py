"""Benchmark for asha.sanitize()."""

import statistics
import time
from asha import sanitize

PROMPTS = [
    "Contact john.doe@enterprise-corp.com or call 555-019-2831.",
    "SSN 000-12-3456 and credit card 4111-2222-3333-4444 should never be disclosed.",
    "Please send the login credentials to administrator@bank-secure.net immediately.",
]

def run_benchmark():
    print("=== Benchmarking asha.sanitize() ===")
    
    # Cold latency
    t0 = time.perf_counter()
    res0 = sanitize(PROMPTS[0])
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"Cold Start Latency: {cold_ms:.2f} ms")

    # Warm iterations
    times = []
    for i in range(25):
        prompt = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        res = sanitize(prompt)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    mean = statistics.mean(times)

    print(f"Warm Iterations: 25 calls")
    print(f"Mean Latency:    {mean:.2f} ms")
    print(f"Median (P50):    {p50:.2f} ms")
    print(f"P95 Latency:     {p95:.2f} ms")
    print("=" * 35)

if __name__ == "__main__":
    run_benchmark()
