# Versioning Policy

ASHA follows [Semantic Versioning 2.0.0](https://semver.org/).

---

## Current release

| Version | Status | Notes |
|---------|--------|-------|
| **0.4.2** | **Active: developer preview** | Package rename (`asha`); ANCHOR agent governance. Install: `pip install asha-si`. See [status.md](status.md) |
| 0.4.1 | Superseded | Architecture complete; root API frozen |
| 0.4.0 | Superseded | Typed results introduced |
| 0.3.x | Previous preview | Dict/string returns, lazy root exports |
| 1.0.x (2026-05) | Retracted track | Brief release; project returned to 0.x |

PyPI classifier: **Development Status :: 3 - Alpha**

---

## What 0.4.2 means

- PyPI package and imports renamed: `privysha` → `asha`
- **ANCHOR** runtime: `anchor()`, mission guards, sandbox, framework adapters
- Root API: `process`, `sanitize`, `optimize`, `Agent`, `anchor`
- Environment variables use `ASHA_*` prefix
- Layer boundaries enforced in CI

## What 0.4.1 introduced

- Root API frozen: `process`, `sanitize`, `optimize`, `Agent` only
- `Pipeline`, `Processor`, root `wrap_llm` **removed** (not deprecated)
- `process(mode=)` + `policy=PolicyConfig(...)` - no loose deprecated kwargs
- Layer boundaries enforced in CI

Breaking changes within 0.4.x are documented in the [CHANGELOG](https://github.com/AjayRajan05/asha-si/blob/main/CHANGELOG.md) and [migration-v0.4.md](migration-v0.4.md).

---

## API stability rules (0.4.2)

### Stand behind (core drop-in)

These are the APIs most applications should use. Behavior may improve, but
signatures and result field *names* are frozen within 0.4.2 unless a
documented breaking change lands in CHANGELOG:

| Surface | Notes |
|---------|--------|
| `from asha import process, sanitize, optimize, Agent, anchor` | Only root exports |
| `process(..., mode=..., policy=...)` → `ProcessResult` | Fields: `output`, `original`, `degraded`, `security`, `metrics`, … |
| `sanitize(...)` → `SanitizeResult` | Security-only |
| `optimize(...)` → `OptimizeResult` | Never runs security |
| `from asha.integrations import wrap_llm` | Preferred client wrapping |
| `PolicyConfig` / policy modes | `balanced`, `strict`, `lite`, `off` |

Contract tests: `tests/public_api/`, `tests/contracts/`.

### Preview (pin + validate)

| Surface | Notes |
|---------|--------|
| `anchor()` / `AnchorRuntime` / framework adapters | Mission governance (validate on your agent stack) |
| Isolation modes `auto` / `hard` / `off` | Container backends not implemented |

### Experimental (may change without deprecation)

Emits `ASHAExperimentalWarning` at call time. Prefer core APIs above.

| Surface | Notes |
|---------|--------|
| `recommend_local_model` / AshaFit | Local model advisor |
| `SmartRoutingAdapter` / `Agent(routing_config=...)` | Task-based routing |
| `auto_patch()` | Global SDK monkey-patch (prefer `wrap_llm`) |
| `AdapterFactory` advanced helpers | Beyond basic mock/openai create |

Docs: [experimental-features.md](experimental-features.md).

### What a future 1.0 would guarantee

- Semver-stable root + `wrap_llm` contracts
- Documented deprecation window before removals
- Published adapter support matrix for ANCHOR

Until then, 0.x may still break with CHANGELOG notes.

---

## Path to 1.0.0 (deferred)

Stay on **0.4.2** until the product is fully built and the maintainer is ready to acknowledge a stable API. No rush.

1. **0.4.x** - Current track (finish release leftovers + pilot quality here)
2. **0.5.x** - Optional later freeze window (only if needed)
3. **1.0.0** - Only after the above (see [production-readiness.md](production-readiness.md))

---

## Breaking changes during 0.x

Allowed in any 0.x release. Documented in:

- `CHANGELOG.md` on GitHub
- `docs/migration-v0.4.md`
- `docs/deprecations.md`

After **1.0.0**, breaking changes require a major release.

---

## Deprecation policy (post-1.0)

- Deprecated APIs get at least one minor release with warnings before removal
- v0.4.1 removed several symbols **without** a deprecation period (breaking cleanup within 0.4.x)

---

## Python support

**Python 3.10+** (3.10, 3.11, 3.12 tested in CI).

---

## Installing vs pinning

```bash
pip install asha-si          # default: everyone, lite path
```

```toml
# Optional: freeze the build you validated in a pilot
asha-si==0.4.2
```

Review the [CHANGELOG](https://github.com/AjayRajan05/asha-si/blob/main/CHANGELOG.md) before upgrading across minor versions.
