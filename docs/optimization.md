# Optimization

How to reduce prompt tokens with ASHA's optimizer, alone or as part of `process()`.

---

## Compress tokens only

`optimize()` runs token compression only (no PII masking, no compile stage):

```python
from asha import optimize

result = optimize("Hey bro can you please analyze this dataset thoroughly")
print(result.output)
print(result.metrics.token_reduction_pct)
```

---

## Optimize as part of process()

Full path includes security + compile + optimize:

```python
from asha import process

result = process("long verbose prompt please analyze carefully", token_budget=1200)
print(result.metrics.tokens_saved)
print(result.metrics.token_reduction_pct)
```

---

## Skip optimization when safe

```python
from asha import process
from asha.core.policy_config import PolicyConfig

process(
    prompt,
    policy=PolicyConfig(preserve_intent=True),
)
```

When no PII or threats are found, the original prompt may be returned unchanged.

---

## Modes and latency

| Mode | Optimization level |
|------|-------------------|
| `balanced` | Standard compression |
| `strict` | Aggressive + fail-closed security |
| `lite` | Reduced policy features |
| `off` | Skipped |

For minimum latency: `mode="off"` or call `optimize()` alone on already-clean prompts.

---

## Read metrics

```python
from asha import process

result = process(prompt)
m = result.metrics
print(m.token_reduction_pct)
print(m.tokens_saved)
print(m.processing_time_ms)
```

Token-reduction figures vary widely by input shape; measure with [performance.md](performance.md) rather than assuming a fixed percentage.

---

## optimize() vs process()

| Function | Security | Compile | Optimize |
|----------|----------|---------|----------|
| `optimize()` | No | No | Yes |
| `sanitize()` | Yes | No | No |
| `process()` | Yes | Yes | Yes |

---

## Related

- [Performance](performance.md)
- [Core concepts](core-concepts.md)
