# ASHA Documentation

ASHA is a Python library for **mission-aware agent governance**, **Superintelligence (SI) alignment**, and **drop-in prompt security**. Use `anchor()` to keep autonomous agents aligned with the user's goal, and `process()` / `wrap_llm()` to mask PII, check injection patterns, and compress prompts before they reach a model.

Licensed under the **Apache License 2.0**.

!!! warning "Developer preview (v0.4.2)"
    Pre-1.0: APIs may change. Install with `pip install asha-ai` (lite path; no API key or model download).
    See **[status.md](status.md)** for what works vs preview vs experimental. Pin a version only when you need a frozen pilot.

---

## Tutorial

Learning-oriented: one path from install to a working result.

- **[Tutorial](tutorial.md)**: install ASHA, process a prompt, wrap a client, govern a mock agent

## How-to guides

Task-oriented: solve one problem when you already know the basics.

- [Integrations](integrations.md): FastAPI, LangChain, Instructor, and related middleware
- [Security](security.md): PII masking, threat checks, fail-closed modes
- [Optimization](optimization.md): token compression and budgets
- [Debugging](debugging.md): `trace=True`, diffs, degraded results
- [Troubleshooting](troubleshooting.md): common failures and fixes
- [Performance](performance.md): run benchmarks and tune latency vs security
- [Migration v0.4](migration-v0.4.md): upgrade from 0.3.x and PrivySHA → ASHA rename

## Reference

Lookup-oriented: signatures, examples catalog, versioning facts.

- [API reference](api-reference.md)
- [Examples](examples.md)
- [Versioning](versioning.md)
- [Deprecations](deprecations.md)

## Explanation

Understanding-oriented: design, rationale, and project direction.

- [Core concepts](core-concepts.md)
- [Architecture](architecture.md)
- [ANCHOR](anchor.md)
- [Compliance](compliance.md)
- [Roadmap](roadmap.md)
- [Production readiness (0.4.2)](production-readiness.md)
- [Experimental features](experimental-features.md)

## Support

- [Support policy](support.md): channels, expectations, what to include in reports
- [GitHub Discussions](https://github.com/AjayRajan05/ASHA/discussions) · [Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw)
- [FAQ](faq.md)
- [Contributing](contributing.md)
- [Status](status.md)
