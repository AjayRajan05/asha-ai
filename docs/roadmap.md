# ASHA Developer Roadmap: Foundation to v1.0.0

This roadmap communicates ASHA's trajectory toward **v1.0.0**. It establishes short-term foundational priorities and defines the explicit maturity criteria required for a production-grade 1.0 release.

---

## 1. Short-Term Foundation Vision

The primary short-term objective is:

> **Make the library foundationally strong, easy for developers to understand, easy for users to adopt, properly tested through real-world use cases, architecturally coherent, lightweight, and ready for continued community improvement.**

Rather than rushing to add speculative features or inflate superficial test counts, development focuses on five core pillars:

1. **Test Integrity**: Eliminate test-to-pass evasions and replace artificial coverage tests with end-to-end user workflow suites (`tests/use_cases/`).
2. **Zero-Heavy-Dependency Core**: Maintain strict discipline so `pip install asha-ai` requires only minimal lightweight libraries (`pydantic`, `tiktoken`, `requests`, `click`, `pypdf`, `python-docx`), ensuring sub-second import times.
3. **Transparent Ergonomics**: Ensure transparent client proxying (`SecureLLM`), zero-config presets (`Profile.BALANCED()`, `Profile.STRICT()`), and multi-turn session privacy (`PIIVault`).
4. **Truthful Documentation**: Keep all docstrings, tutorials, and examples 100% faithful to the actual implementation.
5. **Community Feedback Loop**: Actively invite developer review through the interactive Control Center showcase (`streamlit run examples/showcase/app.py`).

---

## 2. v1.0.0 Readiness Criteria

ASHA will reach **v1.0.0** when all of the following architectural and operational conditions are satisfied:

* **API Stability**: The public API exported from `asha/__init__.py` is frozen. No breaking signature changes without a formal deprecation cycle.
* **Deterministic Contracts**: Given identical inputs and policy profiles, ASHA produces consistent, deterministic sanitization and optimization across all supported operating systems (Linux, macOS, Windows).
* **Fail-Closed by Default**: In production modes, security violations and anomalous payloads are reliably blocked before reaching external model providers.
* **Complete Use-Case Validation**: Every major feature (`SecureLLM`, `PIIVault`, `AgentCostGuard`, `scan_rag_context`, `ANCHOR`) has passing real-world integration tests.
* **Packaging & Distribution Hygiene**: Clean PyPI wheels, zero maintainer scratch artifacts, and complete absence of large training datasets or private weights in public releases.

---

## 3. Milestones & Progress

### Shipped in Preview (v0.4.x)
- [x] Package rename and brand consolidation to `asha-ai`
- [x] Public facade lazy loading in `src/asha/__init__.py`
- [x] High-level `SecureLLM` client proxy with automatic bidirectional PII rehydration
- [x] Reversible multi-turn `PIIVault` with session isolation and TTL cleanup
- [x] `scan_rag_context()` and `scan_rag_context_stream()` for indirect injection defense in retrieval pipelines
- [x] `AgentCostGuard` for runaway autonomous agent loop and cumulative token budget bounding
- [x] Named deployment profiles (`Profile.STRICT()`, `Profile.BALANCED()`, `Profile.PERMISSIVE()`, `Profile.DEVELOPER()`)
- [x] Interactive Streamlit showcase application (`examples/showcase/app.py`)
- [x] Elimination of redundant, artificial "test-to-pass" test suites

### Milestone 1: Foundation Hardening (In Progress)
- [x] Consolidate duplicate tests into canonical test packages (`tests/security/`, `tests/contracts/`, `tests/use_cases/`)
- [x] Expand real use-case test coverage for multi-turn conversations and agent tool loops
- [x] Align documentation and quickstart guides with live implementation
- [ ] Full typing coverage (`mypy --strict` compliance across core modules)
- [ ] Benchmarks for CPU latency overhead under high concurrency (target < 5ms for standard text)

### Milestone 2: Ecosystem & Integrations (v0.5.0)
- [ ] Containerized sandbox execution backends for ANCHOR (`docker`, `bwrap`)
- [ ] Enhanced OpenTelemetry semantic conventions for AI security tracing
- [ ] Additional international PII pattern extractors (EU GDPR, APAC identifiers)
- [ ] Production ASGI/WSGI middleware adapters for FastAPI, Flask, and Django

### Milestone 3: v1.0.0 Release Candidate
- [ ] Full public API freeze
- [ ] Security audit & fuzzing report against OWASP Top 10 for LLMs
- [ ] Long-term backwards compatibility guarantee
- [ ] Official v1.0.0 release notes and migration guides

---

## 4. What is Explicitly Out of Scope for v1.0.0

To keep ASHA lightweight, focused, and maintainable, the following will **not** be included in v1.0.0:
- Managed multi-tenant cloud service / hosted SaaS backend
- Heavy bundled deep learning models in the core wheel
- Proprietary closed-source extensions
