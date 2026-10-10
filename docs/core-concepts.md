# Core concepts

**ASHA v0.4.2**: how the pieces fit together.

ASHA preprocesses prompts (detect PII, check threats, compile structure, compress tokens) and can govern autonomous agents via ANCHOR. You call one function; engines run inside `PromptProcessor`. ANCHOR is a separate runtime path.

---

## Public API

### Root package

```python
from asha import process, sanitize, optimize, Agent, anchor
```

### Common subpackage imports

```python
from asha.integrations import wrap_llm
from asha.types import ProcessResult, SanitizeResult, OptimizeResult
from asha.core.policy_config import PolicyConfig
from asha.runtime import PromptProcessor
from asha.utils.dropin import process_async
from asha.utils.unmask import unmask
```

There is **no** global `configure()`. Pass `mode` and `policy` per call.

Experimental helpers (`recommend_local_model`, smart routing, `AdapterFactory`) live under [experimental-features.md](experimental-features.md).

---

## Processing flow

```
process(prompt)
  → resolve mode + PolicyConfig
  → PromptProcessor.run()
      → run_security()      # PII, injection, masking
      → compile_prompt()    # internal representation → structured text
      → optimize_tokens()   # token compression
  → ProcessResult
```

`sanitize()` runs security only. `optimize()` runs token compression only.

---

## Policy modes

| Mode | Security | Optimization | On total failure |
|------|----------|--------------|------------------|
| `balanced` | Standard | Yes | Fail-open + fallback |
| `strict` | Maximum | Yes | Raises `ASHAProcessingError` |
| `lite` | Minimal features | Reduced | Fail-open (same as balanced) |
| `off` | Skipped | Skipped | Passthrough |

```python
process("prompt", mode="balanced")  # default
process("prompt", mode="strict")
process("prompt", mode="off")
```

---

## PolicyConfig

Advanced knobs are **not** top-level kwargs on `process()`:

```python
from asha.core.policy_config import PolicyConfig

process(
    prompt,
    policy=PolicyConfig(
        pii_mode="rule",       # hybrid (default) | rule | lite | ml_only
        reversible=False,
        preserve_intent=False,
        security_level="medium",
    ),
)
```

| Field | Purpose |
|-------|---------|
| `pii_mode` | Detection strategy (default `"hybrid"`; soft-falls back without ML) |
| `reversible` | Store masking map for `unmask()` |
| `preserve_intent` | Skip optimization when no PII/threats |
| `security_level` | `low` / `medium` / `high` |

---

## Result types

### ProcessResult

```python
result = process("prompt")
result.output          # processed text
result.original        # input text
result.degraded        # True if fallback path used
result.security        # SecurityInfo (PII, threats)
result.metrics         # tokens, timing
result.trace           # when trace=True
result.diff            # when debug=True
str(result)            # same as result.output
```

### SanitizeResult / OptimizeResult

Same pattern: `.output`, `.security` (sanitize), `.metrics` (optimize).

Legacy dict: `result.to_dict()` or `asha.compat.legacy_results.to_legacy_pipeline_dict(result)`.

---

## Fail-open vs fail-closed

| API | Default | Strict |
|-----|---------|--------|
| `process()` / `sanitize()` | Fail-open fallback | `mode="strict"` raises |
| `wrap_llm()` | Uses caller `mode` | Transport errors raise when `mode != "off"` |
| `auto_patch()` | Default `mode="strict"` | Configurable |

---

## Reversible masking

```python
from asha import sanitize
from asha.core.policy_config import PolicyConfig
from asha.utils.unmask import unmask

result = sanitize(
    "Email alice@corp.com",
    policy=PolicyConfig(reversible=True),
)
safe = result.output
restored = unmask(safe, result.security.masking_map)
```

---

## Agent vs drop-in vs ANCHOR

| Goal | Use |
|------|-----|
| Preprocess only | `process()`, `sanitize()`, `optimize()` |
| Wrap existing SDK | `wrap_llm(client)` |
| Preprocess + LLM call | `Agent(model=...)` |
| Govern autonomous agents | `anchor(agent)` / framework adapters |
| Task-based model pick | `Agent(routing_config=...)` (**experimental**) |
| Local model advice | `recommend_local_model()` (**experimental**) |

`Agent(privacy=True)` enables preprocessing with strict internal mode. `privacy=False` disables it.

---

## Internal vs public

**Public:** `process`, modes, typed results, `anchor`, documented integrations.

**Internal (do not import in app code):** `core/_ir/`, compiler internals, detector calibration artifacts.

---

## Related

- [Architecture](architecture.md)
- [API reference](api-reference.md)
- [Status](status.md)
- [Experimental features](experimental-features.md)
