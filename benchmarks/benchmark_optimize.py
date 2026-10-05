"""Benchmark for asha.optimize()."""

import statistics
import time
from asha import optimize

PROMPTS = [
    "Could you please be so kind as to summarize the following article in three bullet points please?",
    "I would really appreciate it if you could kindly extract all the dates and numbers from this report.",
    "Please analyze the financial results for the previous four quarters and present key trends.",
]

def run_benchmark():
    print("=== Benchmarking asha.optimize() ===")
    
    # Cold latency
    t0 = time.perf_counter()
    res0 = optimize(PROMPTS[0])
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"Cold Start Latency: {cold_ms:.2f} ms")

    # Warm iterations
    times = []
    for i in range(25):
        prompt = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        res = optimize(prompt)
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
    print(f"Tokens Saved:    {res0.metrics.tokens_saved if res0.metrics else 0}")
    print("=" * 35)

if __name__ == "__main__":
    run_benchmark()
