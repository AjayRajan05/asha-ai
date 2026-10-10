# Troubleshooting

**ASHA v0.4.2**

---

## Installation

### ImportError: No module named 'asha'

```bash
pip install asha-si
# or
pip install -e .
```

### ImportError: cannot import wrap_llm from asha

Root no longer exports `wrap_llm`:

```python
from asha.integrations import wrap_llm
```

### TypeError: unexpected keyword argument

Deprecated kwargs were removed. Use `mode` and `PolicyConfig`:

```python
from asha.core.policy_config import PolicyConfig
process(prompt, policy=PolicyConfig(pii_mode="hybrid"))
```

---

## PII not masked

```python
from asha import process
from asha.core.policy_config import PolicyConfig

process(prompt, mode="strict")
process(prompt, policy=PolicyConfig(pii_mode="hybrid"))  # needs asha-si[ml]
```

Check `mode="off"` is not set. Check output via `result.output`, not raw input.

---

## ML / hybrid PII fails

```bash
pip install asha-si[ml]
```

Without ML deps, `pii_mode="hybrid"` falls back to `rule`.

---

## process() returns unexpected type

v0.4+ always returns **`ProcessResult`**:

```python
result = process("prompt")
print(result.output)   # not result["optimized"]
str(result)            # works
```

---

## wrap_llm errors

- Use `mode="off"` only when you intentionally want no preprocessing
- `mode="strict"` raises on processing failure
- Prefer `wrap_llm()` over `auto_patch()` in shared environments

---

## Agent connection errors

```python
agent = Agent(model="mock")  # no API key needed for tests
```

For real providers, set `OPENAI_API_KEY` etc. and `pip install asha-si[openai]`.

---

## Slow processing

```python
process(prompt, mode="lite")
process(prompt, mode="off")
```

Avoid `trace=True` and `debug=True` in production hot paths.

---

## AttributeError: Pipeline / Processor

Removed in v0.4.1:

```python
from asha import process
from asha.runtime import PromptProcessor
```

---

## Tests

```bash
pytest tests -q
```

---

## Still stuck?

- [FAQ](faq.md) · [Support policy](support.md)
- [GitHub Discussions](https://github.com/AjayRajan05/asha-si/discussions)
- [ASHA Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw)

## Related

- [FAQ](faq.md)
- [Migration v0.4](migration-v0.4.md)
- [Deprecations](deprecations.md)
- [Performance](performance.md)
- [Status](status.md)
