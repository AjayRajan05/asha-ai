# Contributing to ASHA

> **Road to v1.0.0: Building the Concrete Foundation First**  
> ASHA is an open-source, production-grade security, preprocessing, token optimization, and mission governance library for Python LLM applications.  
> As we march toward **v1.0.0**, our priority is establishing an **ironclad, concrete foundation grounded in real-world use cases** before chasing speculative features or scaling complexity.

Support channels and expectations: [support.md](support.md).

- **Architecture, RFCs & Ideas**: [GitHub Discussions](https://github.com/AjayRajan05/asha-si/discussions)
- **Community Chat**: [ASHA Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw)
- **Issue Tracker**: [GitHub Issues](https://github.com/AjayRajan05/asha-si/issues)
- **Security Reports**: Follow [SECURITY.md](https://github.com/AjayRajan05/asha-si/blob/main/SECURITY.md) (never open public issues for security vulnerabilities).

---

## The 5-Step Developer Roadmap

All contributors are encouraged to advance through these five progressive milestones:

1. **Step 1: Understand Deeply**: Master the architecture, public facade (`asha/__init__.py`), public contracts (`tests/contracts/`), and run the live showcase (`streamlit run examples/showcase/app.py`).
2. **Step 2: Optimize Core Drop-Ins**: Ensure `process()`, `sanitize()`, `optimize()` (and async variants) are reliable, lightweight (< 5ms), non-mutating, fail-closed, and strictly free of heavy ML dependencies in the base install.
3. **Step 3: Fortify ANCHOR (Make It Real)**: ANCHOR must never be governance "in name only." Build solid action guards, parameter normalization, chain/memory guards, sandboxing, and test all framework adapters (`CrewAI`, `LangChain`, `LangGraph`, `LlamaIndex`, `MCP`, `AutoGen`) against real adversarial threats.
4. **Step 4: Build for Real-World Use Cases and Mandatory Documentation**: Write authentic end-to-end tests in `tests/use_cases/`. Eliminate superficial "test-to-pass" assertions. **Documentation is non-negotiable**: clear docstrings, updated Markdown guides, parameter tables, and runnable examples are required for every PR.
5. **Step 5: Innovate, Scale, and Bring Ambitious Ideas**: Once the foundation is solid and documented, bring ambitious proposals! Pitch RFCs for international PII scanners, streaming token guards, containerized sandboxes, and enterprise telemetry on GitHub Discussions.

---

## Setup

```bash
git clone https://github.com/AjayRajan05/asha-si.git
cd ASHA
pip install -e ".[dev]"
```

Python **3.10+** required.

---

## Run tests

```bash
# Full suite
pytest tests -q

# Architecture boundaries only
pytest tests/architecture -q

# With coverage (CI uses --cov-fail-under=50)
pytest --cov=asha --cov-report=term -m "not slow"
```

### Framework integration tests (optional deps)

```bash
pip install -e ".[dev,integrations,flask,django,anchor]"
ASHA_ANCHOR_INTERACTIVE=0 ASHA_ANCHOR_WARN_POLICY=strict \
  pytest tests/integration/test_framework_integrations.py \
  tests/runtime/anchor/ -q
```

---

## Architecture rules

Enforced by `tests/architecture/test_boundaries.py`:

| Layer | Must not import |
|-------|-----------------|
| `core/` | `runtime`, `integrations`, `compat` |
| `runtime/` | `integrations`, `compat` |
| `types/` | `compat`, `runtime`, `integrations` |
| `utils/` | `compat` |

Do not reintroduce `Pipeline`, root lazy exports, or `compat/` on the `process()` hot path.

---

## Package layout

```
src/asha/
├── core/           # engines, policy, security, compiler
├── runtime/        # PromptProcessor, Agent, adapters
├── integrations/   # wrap_llm, auto_patch, middleware
├── types/          # ProcessResult, etc.
├── utils/          # dropin functions
├── compat/         # legacy_results only
└── cli/
```

---

## Code style

```bash
flake8 src/asha --select=F401,F824
mypy src/asha --ignore-missing-imports
```

Match existing conventions - minimal diffs, no drive-by refactors.

---

## Docs

```bash
pip install -e ".[docs]"
mkdocs serve
mkdocs build --strict
```

Update docs when changing public API. Root exports are: `process`, `sanitize`, `optimize`, `Agent`, `anchor`.

---

## Release track

Current pin: **`asha==0.4.2`** (developer preview / production-pilot build). Breaking changes allowed in 0.x (document in `CHANGELOG.md` and `docs/migration-v0.4.md`).

Before PRs that touch packaging or docs:

```bash
python scripts/check_public_surface.py
```

Maintainer release checklist: repository `maintainers/RELEASE_0.4.2.md`. Private detector artifacts: never commit here (see repository `maintainers/PRIVATE_ARTIFACTS.md`).

---

## Pull requests

1. Fork and branch
2. Add tests for behavior changes
3. Run full pytest locally
4. Update relevant docs
5. Open PR with clear description
6. No secrets, `training/`, `models/`, or `*.joblib` in the public tree

See [status.md](production-readiness.md) for preview scope.
