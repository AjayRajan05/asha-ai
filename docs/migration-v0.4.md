# Migration Guide - v0.4.x

Upgrade from ASHA **0.3.x** (or early 0.4.0) to **0.4.2**. Also covers the **PrivySHA → ASHA** package rename.

---

## 0. Package rename (PrivySHA → ASHA)

| Before | After |
|--------|-------|
| `pip install privysha` | `pip install asha-si` (pin pilots: `asha-si==0.4.2`) |
| `from privysha import process` | `from asha import process` |
| CLI `privysha` | CLI `asha` |
| Env vars `PRIVYSHA_*` | Env vars `ASHA_*` |
| PyPI name | **`asha-si`** (`asha` on PyPI is unrelated; imports stay `asha`) |
| Repo / docs | [github.com/AjayRajan05/asha-si](https://github.com/AjayRajan05/asha-si) · [docs site](https://ajayrajan05.github.io/asha-si/) |

Uninstall the old package if both are present: `pip uninstall privysha` then `pip install asha-si==0.4.2`.

### Renamed from `asha-ai` to `asha-si`

If you installed the library under the interim name `asha-ai`, switch to `asha-si`. The import name (`asha`) and CLI (`asha`) do not change.

```bash
pip uninstall asha-ai
pip install asha-si
```

---

## 1. Return types are dataclasses

```python
# Before (0.3)
text = process("a@b.com - analyze")  # sometimes str
result = process("...", return_metrics=True)
print(result["optimized"])

# After (0.4.2)
result = process("a@b.com - analyze")
print(result.output)
print(result.metrics.token_reduction_pct)
print(result.security.pii_detected)
str(result)  # still works - equals result.output
```

| Old dict key | New attribute |
|--------------|---------------|
| `optimized` | `result.output` |
| `original` | `result.original` |
| `security_result` | `result.security` |
| `metrics` | `result.metrics` |
| `token_reduction` | `result.metrics.token_reduction_pct` |

Legacy dict: `result.to_dict()` or:

```python
from asha.compat.legacy_results import to_legacy_pipeline_dict
legacy = to_legacy_pipeline_dict(process("...", include_legacy_detail=True))
```

---

## 2. Import paths

```python
# Before
from asha import wrap_llm, Pipeline, Processor, ProcessResult

# After
from asha import process, sanitize, optimize, Agent
from asha.integrations import wrap_llm, auto_patch
from asha.runtime import PromptProcessor
from asha.types import ProcessResult
```

Removed symbols raise **`AttributeError`** at root.

---

## 3. Kwargs → mode + PolicyConfig

```python
# Before
process(x, privacy=False)
process(x, preserve_intent=True)
process(x, pii_mode="hybrid")
process(x, security_fail_closed=True)

# After
process(x, mode="off")
process(x, policy=PolicyConfig(preserve_intent=True))
process(x, policy=PolicyConfig(pii_mode="hybrid"))
process(x, mode="strict")
```

Unknown kwargs raise **`TypeError`**.

---

## 4. optimize() is tokens-only

```python
optimize(prompt)  # no PII, no compile - OptimizeResult only
process(prompt)   # full path
sanitize(prompt)  # security only
```

---

## 5. wrap_llm() - mode only

```python
# Before
wrap_llm(client, privacy=False)

# After
wrap_llm(client, mode="off")
```

---

## 6. Pipeline / Processor removed

```python
# Before
from asha import Pipeline
Pipeline().process("prompt")

# After
from asha import process
process("prompt")

# Advanced
from asha.runtime import PromptProcessor
PromptProcessor().run("prompt", mode="balanced")
```

---

## 7. Safety semantics

| Mode | process/sanitize | wrap_llm (mode != "off") |
|------|------------------|--------------------------|
| `strict` | Raises on total failure | Processing strict; infra errors raise |
| `balanced` | Degraded fallback | Processing balanced; infra errors raise |
| `off` | Passthrough | No preprocessing |

---

## Recommended upgrade steps

1. Pin `asha==0.4.2`
2. Replace `result["optimized"]` → `result.output`
3. Move `from asha import wrap_llm` → `from asha.integrations import wrap_llm`
4. Replace deprecated kwargs with `mode` / `PolicyConfig`
5. Remove `Pipeline` / `Processor` usage
6. Run tests; check `result.degraded` in balanced mode

See [deprecations.md](deprecations.md) for the full removal list.
