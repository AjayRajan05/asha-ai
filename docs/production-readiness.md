# ASHA-SI Production Readiness, Release Status & Roadmap

**Current Release:** Developer Preview (v0.4.2)  
**Distribution:** `pip install asha-si==0.4.2`  
**License:** Apache 2.0  
**Target Milestone:** v1.0.0 Stable Foundation

---

## 1. Developer Preview Status

ASHA-SI is currently in **Developer Preview (v0.4.2)**. The core public API (`process`, `sanitize`, `optimize`, `SecureLLM`, `PIIVault`, `anchor`) is actively tested in pilot environments.

### 1.1 Pilot Recommendations
- **Pin Exact Releases:** In `requirements.txt` or `pyproject.toml`, specify `asha-si==0.4.2` rather than unpinned `asha-si` to ensure zero breaking changes during your evaluation.
- **Fail-Closed Stance:** For regulated workloads (finance, healthcare, enterprise PII), use `Profile.STRICT()` to guarantee fail-closed rejection on ambiguous inputs.
- **Review Explicit Boundaries:** Review the [Security & Threat Model](security.md) before production pilots.

---

## 2. Production Pilot Checklist (0.4.2)

Before advancing an ASHA-SI integration from evaluation to production pilot, verify the following checklist:

| Check | Requirement | Verification Command / Method | Status |
|---|---|---|---|
| **Base Install** | Standard install has zero heavy dependencies (`torch`, `transformers` not required) | `python -m pytest tests/contracts/test_package_smoke.py` | Verified |
| **Provider Interception** | Requests are sanitized and original PII rehydrated on responses | `python -m pytest tests/use_cases/test_provider_workflows.py` | Verified |
| **Streaming Rehydration** | Tokens split across chunk boundaries are rehydrated without latency stalls | `python -m pytest tests/use_cases/test_streaming_and_multimodal.py` | Verified |
| **Deterministic Masking** | PII tokens maintain stable index across multi-turn sessions | `python -m pytest tests/use_cases/test_pii_vault_session.py` | Verified |
| **Layer Boundaries** | Core modules never import runtime or integration layers | `python -m pytest tests/architecture/test_boundaries.py` | Verified |
| **Public Surface** | No private weights, training data, or maintainer scratch files committed | `python scripts/check_public_surface.py` | Verified |

---

## 3. Versioning & Deprecation Policy

ASHA adheres to **Semantic Versioning (SemVer 2.0)**:

- **0.4.x (Current Developer Preview):** Public APIs in `asha/__init__.py` are stable. Internal compiler representations (`_ir`) remain private.
- **Breaking Changes in 0.4.x:**
  - Package renamed from `privysha` to `asha-si` on PyPI (`from asha import ...`).
  - Environment variables unified under `ASHA_*` prefix (`ASHA_DEBUG`, `ASHA_DISABLE_ML`).
  - Deprecated legacy `Pipeline`, `AsyncPipeline`, and `Processor` classes removed in favor of `PromptProcessor` and drop-in `process()`.
  - `wrap_llm()` moved to `asha.integrations` (prefer `from asha import SecureLLM`).
- **Path to 1.0.0:** ASHA will declare 1.0.0 when the public API is frozen, multi-platform deterministic guarantees are complete, and containerized sandbox hooks for ANCHOR are implemented.

---

## 4. Development Roadmap to v1.0.0

```
┌────────────────────────────────────────────────────────────────────────┐
│  v0.4.2: Developer Preview (Shipped)                                  │
│  - Zero-dependency base install (pydantic, tiktoken, requests)         │
│  - Provider Protection Matrix across 6 major LLM providers             │
│  - Reversible PIIVault with sliding-window stream rehydration         │
│  - ANCHOR agent mission governance and tool authorization              │
│  - Multimodal URL & OCR adversarial injection scanning                 │
├────────────────────────────────────────────────────────────────────────┤
│  v0.5.0: Ecosystem & Enterprise Hardening (Next Milestone)            │
│  - Containerized sandbox backends for ANCHOR (Docker / bubblewrap)     │
│  - OpenTelemetry semantic convention export for AI security tracing    │
│  - Extended international PII extractors (EU GDPR, APAC IDs)           │
│  - Native FastAPI, Flask, and Django middleware plugins                │
├────────────────────────────────────────────────────────────────────────┤
│  v1.0.0: Production LTS Release                                       │
│  - Frozen public API contract with strict SemVer guarantees            │
│  - Full mypy strict type safety across all internal modules            │
│  - Verified deterministic token compression across Linux/macOS/Windows │
└────────────────────────────────────────────────────────────────────────┘
```
