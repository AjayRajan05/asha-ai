# Integrations

How to wrap LLM clients and attach ASHA middleware to common frameworks.

---

## Wrap an LLM client

```python
from asha.integrations import wrap_llm
import openai

client = wrap_llm(openai.OpenAI(), mode="balanced")
```

Import from **`asha.integrations`**, not the package root.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mode` | `balanced` | `strict`, `lite`, `off` |
| `token_budget` | `1200` | Passed to internal `process()` |

Advanced / experimental options (`auto_select_local_model`, smart routing): [experimental-features.md](experimental-features.md).

---

## auto_patch (use with caution)

```python
from asha.integrations import auto_patch, disable_auto_patch

auto_patch(mode="strict")
# ... application code ...
disable_auto_patch()
```

Globally monkey-patches SDKs. Prefer per-client `wrap_llm()` in production.

---

## Framework middleware

```bash
pip install asha-si[integrations]
# or specific: asha-si[fastapi], asha-si[langchain], etc.
```

| Framework | Module |
|-----------|--------|
| FastAPI | `asha.integrations.fastapi` |
| Flask | `asha.integrations.flask` |
| Django | `asha.integrations.django` |
| LangChain | `asha.integrations.langchain` |
| LlamaIndex | `asha.integrations.llamaindex` |
| Instructor | `asha.integrations.composition_strategy` |
| OpenTelemetry | `asha.integrations.otel` |

Middleware uses `mode="balanced"` or `mode="off"` (not a removed `privacy=` kwarg).

---

## Composition helpers

```python
from asha.integrations.composition_strategy import (
    compose_with_instructor,
    compose_with_langchain,
)
```

Requires optional deps (`instructor`, `langchain`).

---

## Test integrations locally

Integration tests skip when optional deps are missing:

```bash
pip install asha-si[integrations]
pytest tests/test_fastapi_integration.py -q
```

---

## Related

- [Experimental features](experimental-features.md)
- [API reference](api-reference.md)
- [Tutorial](tutorial.md)
