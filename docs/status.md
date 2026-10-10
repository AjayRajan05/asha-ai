# Status

**ASHA v0.4.2: developer preview (pre-1.0).** APIs may still change before 1.0.

```bash
pip install asha-si
```

That is the default lite path for everyone. When you run a long-lived pilot, pin the version you validated (for example `asha-si==0.4.2`) so upgrades are intentional.

## Who should use what

| You want… | Use |
|-----------|-----|
| Works for everyone, no keys, no downloads | `pip install asha-si` then `process` / `sanitize` / `optimize` |
| Drop into an existing OpenAI-style client | `wrap_llm` (`asha-si[openai]` as needed) |
| Govern an agent | `anchor(..., interactive=False)` (preview; validate on your stack) |
| Stronger ML detectors | `asha-si[hardened]`: models **auto-download** on first use (no prompt); override with `ASHA_MODELS_DIR` if needed |

## Works today

| Area | Notes |
|------|-------|
| `process()`, `sanitize()`, `optimize()` | Stable within 0.4.x: primary path for most apps |
| Typed results (`ProcessResult`, …) | Default since v0.4.0 |
| Policy modes (`balanced`, `strict`, `lite`, `off`) | Unified safety semantics |
| `wrap_llm()` (`asha.integrations`) | Prefer over `auto_patch()`; pin the version |
| Provider adapters | Optional extras (`openai`, `anthropic`, `gemini`, mock, …) |
| PII masking | Default `pii_mode="hybrid"`; soft-falls back without ML; use `"rule"` / `"lite"` for regex-only |
| Lite detectors | Always available on base install (no private artifacts) |
| Architecture + CI gates | Layer boundaries enforced in tests |

## In progress / preview

| Area | Notes |
|------|-------|
| **ANCHOR** (`anchor()`, guards, sandbox) | Preview: pin `0.4.2`; validate on your agent stack |
| Framework adapters | Golden: CrewAI / LangChain / LangGraph via `asha-si[anchor]`; LlamaIndex + MCP best-effort; see [anchor.md](anchor.md) support matrix |
| Hybrid / ML PII | Needs `asha-si[ml]` / `asha-si[hardened]` for full NER |
| Hardened ML artifacts | Auto-downloaded bundle (`asha-models-0.4.2`); disable with `ASHA_DISABLE_MODEL_DOWNLOAD=1` |
| Container sandbox (`docker`, `bwrap`) | Not implemented: use `isolation="hard"` |

## Experimental

APIs may change without a deprecation cycle. Details: [experimental-features.md](experimental-features.md).

| Feature | Entry point |
|---------|-------------|
| AshaFit / local advisor | `recommend_local_model`, `Agent(local_model=...)` |
| Smart routing | `Agent(routing_config=...)` / `SmartRoutingAdapter` |
| Model gateway extras | `AdapterFactory`, advanced `wrap_llm` options |
| `auto_patch()` | Prefer `wrap_llm()` |

## Not ready (do not claim)

- Stable **1.0** API guarantee
- Certified compliance product (tooling only, see [compliance.md](compliance.md))
- Semantic optimization guarantees
- Broad ANCHOR production certification across every framework
- Paid support / SLA (community best-effort only, see [support.md](support.md))

Community: [GitHub Discussions](https://github.com/AjayRajan05/asha-si/discussions) · [Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw).

Forward-looking detail: [roadmap.md](roadmap.md). Pilot checklist (stay on 0.4.2): [production-readiness.md](production-readiness.md).
