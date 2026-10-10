# FAQ

**ASHA v0.4.2**

---

## General

### What is ASHA?

A Python library for mission-aware agent governance (`anchor`) and drop-in prompt security (`process` / `wrap_llm`): mask PII, check injection patterns, and compress tokens before prompts reach a model.

### Is it production-ready?

**Developer preview.** Install with `pip install asha-si` for everyday use. For production pilots, pin the version you tested (e.g. `asha-si==0.4.2`) and monitor. A semver-stable **1.0** API guarantee is not claimed yet. Read [status.md](status.md).

### Python version?

**3.10+** (3.10, 3.11, 3.12 in CI).

---

## API

### What can I import from the root?

```python
from asha import process, sanitize, optimize, Agent, anchor
```

Nothing else: no `wrap_llm`, `Pipeline`, or `Processor` at root.

### Where is wrap_llm?

```python
from asha.integrations import wrap_llm
```

### What does process() return?

A **`ProcessResult`** dataclass. `str(result)` returns `result.output`.

### Does process() raise?

- **`mode="balanced"`** (default): fail-open with degraded fallback (`result.degraded=True`)
- **`mode="strict"`**: raises `ASHAProcessingError` on total failure

### How do I get metrics?

```python
result = process("prompt")
print(result.metrics.token_reduction_pct)
print(result.security.pii_detected)
```

No `return_metrics=True`; use typed fields.

### How do I set pii_mode or reversible?

```python
from asha.core.policy_config import PolicyConfig
process(prompt, policy=PolicyConfig(pii_mode="hybrid", reversible=True))
```

Top-level `pii_mode=` on `process()` raises `TypeError`. Default `pii_mode` is `"hybrid"` (soft-falls back without ML deps).

---

## Modes

| Mode | Description |
|------|-------------|
| `balanced` | Default: security + optimization |
| `strict` | Fail-closed |
| `lite` | Minimal policy features |
| `off` | Passthrough |

---

## Performance

### Token reduction?

Depends on the prompt. Short prompts may see little or negative reported reduction after compile restructuring. Measure with [performance.md](performance.md).

### Speed?

Rule-based installs are typically tens of milliseconds on ordinary prompts; ML paths are slower. Use `mode="lite"` or `mode="off"` for lower latency. Re-run the benchmark harness on your hardware.

---

## Security

### Does process() mask PII in attached documents?

Yes when ASHA can read the file: pass a path/bytes to `sanitize` / `process`, or use `wrap_llm` so local attachments are extracted and masked before the LLM call. Scanned/image-only PDFs need OCR (not included). Provider-only `file_id` uploads without local bytes are not readable (see [security.md](security.md)).

### What PII is detected?

Emails, phones, SSNs, credit cards, API keys, JWTs, IPs, and more. See [security.md](security.md).

### GDPR compliant?

ASHA is **privacy tooling**, not a certified compliance product. See [compliance.md](compliance.md).

---

## Agent / ANCHOR

### Agent.run() return type?

**String** by default. **`AgentResult`** when `trace=True`.

### How do I govern an agent?

```python
from asha import Agent, anchor

agent = anchor(Agent(provider="mock", model="mock"), interactive=False)
```

See [anchor.md](anchor.md) and [tutorial.md](tutorial.md).

### Is the ANCHOR sandbox a real container?

No. `isolation="auto"` uses in-process hooks; `isolation="hard"` uses a subprocess. Docker/bubblewrap are not implemented yet (see [status.md](status.md)).

---

## Community

### Where do I ask questions?

[GitHub Discussions](https://github.com/AjayRajan05/asha-si/discussions) (primary) or [ASHA Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw) for optional chat.

### How do hardened models download?

`pip install asha-si[hardened]` auto-fetches  
`https://github.com/AjayRajan05/asha-si/releases/download/asha-models-0.4.2/asha-models-0.4.2.zip`  
on first use. Override with `ASHA_MODELS_DIR` / `ASHA_MODELS_ZIP` / `ASHA_DISABLE_MODEL_DOWNLOAD=1`.

---

## Removed in v0.4.1

| Removed | Use instead |
|---------|-------------|
| `Pipeline`, `Processor` | `process()` or `PromptProcessor` |
| Root `wrap_llm` | `asha.integrations.wrap_llm` |
| `security_fail_closed=` | `mode="strict"` |
| `return_metrics=True` | `result.metrics` |
| `ModelRouter` | `Agent(routing_config=...)` (experimental) |

Full list: [deprecations.md](deprecations.md).

---

## Related

- [Tutorial](tutorial.md)
- [Troubleshooting](troubleshooting.md)
- [Migration v0.4](migration-v0.4.md)
- [Status](status.md)
