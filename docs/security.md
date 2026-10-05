# Security

How to detect and mask PII, check prompt-injection patterns, and choose fail-closed vs fail-open behavior.

Security is the first engine in `process()` and the only engine in `sanitize()`.

!!! note "Developer preview"
    Pin `asha==0.4.2`. Public docs describe *what* is defended against and the general shape of the defense (layered detection, calibrated fail-closed behavior), not internal scoring mechanics.

---

## Detect PII and threats

```python
from asha import process, sanitize

result = process("Contact john@company.com or 555-123-4567", mode="balanced")
print(result.output)
print(result.security.pii_detected)
print(result.security.threats)
print(result.security.threat_level)
```

### Common PII types (rule-based path)

| Type | Examples |
|------|----------|
| Email | `user@example.com` |
| Phone | `555-123-4567` |
| SSN | `123-45-6789` |
| Credit card | `4111-1111-1111-1111` |
| API key / secret | `sk-...` |
| JWT | `eyJ...` |
| IP address | `192.168.1.1` |
| Address / name | Heuristic with context keywords |

Teaching placeholders like `test@example.com` are skipped.

Mask tokens look like `[EMAIL_HASH]_…`, `[PHONE_HASH]_…`, or `[REDACTED]` for high-risk secrets.

---

## Document inputs (PDF / DOCX / text files)

`process()`, `sanitize()`, and `optimize()` accept a **file path**, `Path`, or **bytes** (pass `filename=` for bytes). Text is extracted (no OCR), then the normal PII/security pipeline runs.

```python
from asha import sanitize, process
from pathlib import Path

sanitize(Path("report.pdf"))          # extract + mask
process("notes.docx", mode="balanced")
sanitize(raw_bytes, filename="memo.docx")
```

Supported text documents on the base install: `.txt`, `.md`, `.csv`, `.html`, `.pdf`, `.docx`.

With **`wrap_llm`**, local file attachments in the request (message file parts, `files=` / path kwargs the wrapper can see) are extracted and PII-masked **before** the call is sent to the provider. Opaque provider `file_id` uploads without local bytes cannot be read: upload masked text instead, or pass a local path/bytes the wrapper can see.

---

## Choose a safety mode

```python
from asha import process, sanitize

process(prompt, mode="balanced")  # fail-open fallback (default)
process(prompt, mode="strict")    # raises ASHAProcessingError on total failure
sanitize(prompt, mode="strict")
```

| Mode | On total security failure |
|------|---------------------------|
| `balanced` / `lite` | Degraded result, `degraded=True` |
| `strict` | Raises |
| `off` | Passthrough |

---

## Configure PII mode

Set via `PolicyConfig` (not a top-level kwarg on `process()`):

```python
from asha import process
from asha.core.policy_config import PolicyConfig

process("Contact john@example.com", policy=PolicyConfig(pii_mode="rule"))
process("...", policy=PolicyConfig(pii_mode="lite"))     # alias of rule-style lite path
process("...", policy=PolicyConfig(pii_mode="hybrid"))   # default; needs asha-ai[ml] for full NER
```

| `pii_mode` | Description | Install |
|------------|-------------|---------|
| `hybrid` | Default: rules + ML when available; soft-falls back without ML deps | Core; full NER needs `asha-ai[ml]` |
| `rule` / `lite` | Regex + heuristic only | Core only |
| `ml_only` | Experimental ML-only path | `asha-ai[ml]` |

For stronger classical ML detectors, install `asha-ai[hardened]`. Detector weights **download automatically** on first use into a local cache (no permission prompt). Override with `ASHA_MODELS_DIR` for air-gapped installs, or set `ASHA_DISABLE_MODEL_DOWNLOAD=1` to force lite fallbacks. Base installs need no model files, which is the path meant for everyone.

---

## Sanitize only

```python
from asha import sanitize

result = sanitize("john@corp.com - summarize")
print(result.safe)
print(result.security.pii_detected)
```

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
restored = unmask(llm_output, result.security.masking_map)
```

Store `masking_map` securely: it can reverse pseudonymization.

---

## Secure an existing SDK client

```python
from asha.integrations import wrap_llm

client = wrap_llm(openai_client, mode="balanced")
```

- Uses the caller's `mode` for preprocessing
- Infrastructure/wrap failures raise when `mode != "off"` (never silently send raw prompts)

---

## Defense shape (explanation for operators)

ASHA stacks multiple checks rather than relying on a single pattern list:

1. Rule/heuristic detectors available on every install
2. Optional ML/NER paths when extras are installed
3. Calibrated fail-closed behavior on uncertain scores (escalate to REVIEW / restrictive defaults rather than silent ALLOW)

Exact thresholds, model artifacts, and feature representations are intentionally not documented in public guides.

---

## Related

- [Compliance](compliance.md)
- [Core concepts](core-concepts.md)
- [FAQ](faq.md)
- [Status](status.md)
