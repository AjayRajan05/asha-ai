# Performance

**ASHA v0.4.2**: how to measure performance, and how to tune latency vs security.

Numbers below that come from a committed baseline are labeled as such. Re-run on your hardware before treating any figure as a target.

---

## How to run benchmarks

```bash
pip install -e .
python benchmarks/run_benchmarks.py --save
```

Compare against the committed baseline:

```bash
python benchmarks/run_benchmarks.py --compare benchmarks/baseline/results.json
asha benchmark --save
```

What the harness measures (`asha.core.benchmark.BenchmarkHarness`):

- PII detection / masking coverage on sample prompts
- Token-reduction metrics
- Latency (avg, P95, P99)
- Fail-safe rate
- False-positive checks (teaching emails skipped)

Sample prompts: `benchmarks/sample_prompts.json`.

Programmatic:

```python
from asha.core.benchmark import BenchmarkHarness

harness = BenchmarkHarness()
summary = harness.run_all(mode="balanced")
print(summary)
```

### Baseline artifact

| Path | Description |
|------|-------------|
| `benchmarks/baseline/results.json` | Committed baseline (re-run after major changes) |
| `benchmarks/output/` | Timestamped local runs |

The committed baseline currently reports (rule-based / lite-style install on the machine that produced it):

| Metric | Baseline value |
|--------|----------------|
| P95 pipeline latency | ~58 ms |
| Fail-safe rate | 1.0 |
| False-positive rate | 0.0 |

These are **not** SLAs. Re-run the harness for current numbers on your stack. Do not publish or rely on token-reduction marketing figures without a fresh local run; summary reduction can be near zero while individual prompt metrics vary by input shape.

### CI gates (reference)

On push (Ubuntu, Python 3.11), CI runs architecture tests, coverage floor, benchmark smoke/regression vs baseline, and related quality gates. See repository workflows for the exact thresholds in force.

**Semantic equivalence gate:** CI requires at least **~30%** of benchmark prompt pairs to pass `check_semantic_equivalence`. That is a **smoke / regression floor**, not a production SLA and not a claim that optimization preserves meaning. Treat it as “the pipeline did not totally scramble the suite,” then validate on your own prompts.

---

## How to tune

| Priority | Settings |
|----------|----------|
| **Speed** | `mode="lite"` or `mode="off"`; `PolicyConfig(pii_mode="rule")` |
| **Security** | `mode="strict"`; `PolicyConfig(pii_mode="hybrid")` (needs `asha-si[ml]`) |
| **Token savings** | Lower `token_budget`; keep optimization enabled |
| **Semantic fidelity** | `PolicyConfig(preserve_intent=True)` |

### Fastest path

```python
from asha import process

result = process(prompt, mode="lite")
# or
result = process(prompt, mode="off")
```

Avoid on hot paths: `trace=True`, `debug=True`, and ML PII until you need them.

### Maximum security posture

```python
from asha import process
from asha.core.policy_config import PolicyConfig

result = process(
    prompt,
    mode="strict",
    policy=PolicyConfig(pii_mode="hybrid", security_level="high"),
)
```

### Async batching

```python
import asyncio
from asha.utils.dropin import process_async

async def batch(prompts):
    return await asyncio.gather(
        *[process_async(p, mode="lite") for p in prompts]
    )
```

### Per-call knobs

All on `process()` (no global config).

| Parameter | Effect |
|-----------|--------|
| `mode` | Policy preset |
| `policy` | `pii_mode`, `security_level`, `preserve_intent`, … |
| `token_budget` | Optimization target (default 1200) |
| `timeout_seconds` | Abort after N seconds |
| `max_retries` | Retry transient failures |

### Agent

```python
from asha import Agent

agent = Agent(model="mock", privacy=True, token_budget=800)
```

`privacy=True` maps to strict internal preprocessing. Or preprocess with `process()` yourself.

### Wall-clock check

```python
import time
from asha import process

start = time.perf_counter()
result = process("prompt", mode="balanced")
print(f"Wall: {(time.perf_counter() - start) * 1000:.1f} ms")
print(f"Reported: {result.metrics.processing_time_ms} ms")
```

### Production checklist

1. Default `mode="balanced"` unless you have a reason not to
2. Leave `trace` / `debug` off on every request
3. Use `pii_mode="rule"` unless ML NER is required
4. Use `mode="strict"` on regulated paths
5. Log `result.degraded`
6. Pin `asha==0.4.2`

---

## Related

- [Optimization](optimization.md)
- [Security](security.md)
- [Status](status.md)
