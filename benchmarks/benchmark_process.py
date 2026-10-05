"""Benchmark for asha.process()."""

import statistics
import time
from asha import process

PROMPTS = [
    "Summarize this quarterly earnings report in three bullet points: revenue grew 14% year over year.",
    "Contact john.doe@enterprise-corp.com regarding AWS deployment keys and system configuration.",
    "Translate the following user feedback from French to English and classify sentiment.",
    "Draft a polite customer support response regarding delayed order #49281.",
]

def run_benchmark():
    print("=== Benchmarking asha.process() ===")
    
    # Cold latency (first call)
    t0 = time.perf_counter()
    res0 = process(PROMPTS[0])
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"Cold Start Latency: {cold_ms:.2f} ms")

    # Warm iterations
    times = []
    for i in range(20):
        prompt = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        res = process(prompt)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    mean = statistics.mean(times)

    print(f"Warm Iterations: 20 calls")
    print(f"Mean Latency:    {mean:.2f} ms")
    print(f"Median (P50):    {p50:.2f} ms")
    print(f"P95 Latency:     {p95:.2f} ms")
    print(f"Tokens Saved:    {res0.metrics.tokens_saved if res0.metrics else 0}")
    print("=" * 35)

if __name__ == "__main__":
    run_benchmark()
