# Architecture

**ASHA v0.4.2**: compiler-inspired prompt processing with strict layer boundaries, plus a separate ANCHOR runtime for agent governance.

This page explains the current mental model. It is not a how-to and not an API reference.

---

## Prompt processing flow

```
User input
    │
    ▼
Security (run_security)      ← PII detection, injection checks, masking
    │
    ▼
Compilation (compile_prompt) ← internal representation → structured prompt
    │
    ▼
Optimization (optimize_tokens) ← token compression (MSDPC)
    │
    ▼
ProcessResult                ← typed output, metrics, optional trace
```

`process()` runs all three engines. `sanitize()` runs security only. `optimize()` runs token compression only. `mode="off"` skips engines via the policy gate.

You interact with this pipeline as a black box through the drop-in API. Use `trace=True` when you need stage visibility.

`Agent` adds model routing and LLM generation on top of the same preprocessing path.

---

## ANCHOR (agent governance)

ANCHOR sits beside the prompt pipeline. It does not replace `process()`; it governs autonomous agents at runtime:

```
User prompt → Mission contract (immutable)
                    ↓
        Action / memory / plan / chain guards → ALLOW | WARN | BLOCK | REVIEW
                    ↓
        Process sandbox (hooks or subprocess) → execution or halt
```

Adoption is typically `anchor(agent)` or a framework adapter. Details: [anchor.md](anchor.md).

---

## Package layout

```
src/asha/
├── __init__.py              # process, sanitize, optimize, Agent, anchor
├── core/                    # engines, policy, security, compiler (IR is internal)
│   ├── engines.py
│   ├── policy_config.py
│   ├── security/
│   ├── compiler/
│   └── _ir/                 # internal, not a public API
├── runtime/                 # orchestration
│   ├── processor.py         # PromptProcessor: sole prompt orchestrator
│   ├── agent.py
│   ├── adapters/
│   ├── routing/             # experimental smart routing
│   ├── local_advisor/       # experimental AshaFit
│   └── anchor/              # ANCHOR governance
├── integrations/            # wrap_llm, auto_patch, framework middleware
├── types/                   # ProcessResult, SanitizeResult, OptimizeResult
├── utils/                   # dropin, unmask
├── compat/                  # opt-in legacy helpers only
└── cli/
```

---

## Layer boundaries

| Layer | May import | Must not import |
|-------|------------|-----------------|
| `core/` | stdlib, third-party | `runtime`, `integrations`, `compat` |
| `runtime/` | `core`, `types` | `integrations`, `compat` |
| `types/` | `core` (minimal) | `compat`, `runtime`, `integrations` |
| `utils/` | `runtime`, `core`, `types` | `compat` |
| `integrations/` | `runtime`, `core`, `utils` | none |
| `compat/` | anything | none (legacy helpers only) |

Enforced by `tests/architecture/test_boundaries.py`.

---

## Design principles

### Drop-in first

Primary adoption is `process()`, `sanitize()`, `optimize()`. Client wrapping lives in `asha.integrations.wrap_llm`.

### Safety modes

| Mode | Behavior |
|------|----------|
| `strict` | Fail-closed: raises on total failure |
| `balanced` | Fail-open with rule-based fallback (default) |
| `lite` | Minimal policy features; fail-open like balanced |
| `off` | Passthrough |

Advanced policy uses `PolicyConfig` (`pii_mode`, `reversible`, `preserve_intent`, …). Default `pii_mode` is `"hybrid"` and soft-falls back when ML deps are missing.

### Hot path

`utils/dropin.process()` → `runtime/resolve.resolve_process_call()` → `PromptProcessor.run(...)`.

`compat/` is not on the hot path.

### Legacy dict shapes

Opt-in only:

```python
from asha.compat.legacy_results import to_legacy_pipeline_dict

legacy = to_legacy_pipeline_dict(process("prompt", include_legacy_detail=True))
```

---

## PII detection shape

1. **Rule-based** patterns in `core/security/`: always available
2. **Multi-phase pipeline** in `core/pii_pipeline/`: normalization → detection → verification → scoring → masking
3. **Hybrid ML**: optional via `pip install asha-si[ml]`

Public docs describe the defense shape (layered detection, calibrated fail-closed behavior), not internal scoring mechanics.

---

## Observability

- `process(..., trace=True)` / `debug=True`
- Optional OpenTelemetry via `pip install asha-si[otel]`

---

## Related

- [Core concepts](core-concepts.md)
- [API reference](api-reference.md)
- [Security](security.md)
- [ANCHOR](anchor.md)
- [Status](status.md)
