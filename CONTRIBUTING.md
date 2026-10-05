# Contributing to ASHA

Thank you for your interest in contributing to ASHA!

> **Mission: Road to v1.0.0 - Concrete Foundation First**  
> ASHA is an open-source, production-grade security, preprocessing, token optimization, and mission governance library for Python LLM applications.  
> As we march toward **v1.0.0**, our overriding priority is to build an **ironclad, concrete foundation grounded in real-world use cases** before expanding into speculative features or scaling complexity.

---

## 1. Contributor Philosophy & Core Tenets

ASHA is an infrastructure utility layer, not a conceptual prototype or an academic demo. Every line of code merged into this repository must reinforce our core tenets:

* **Real-World Utility Over Artificial Metrics**: We do not write superficial "test-to-pass" assertions (`assert result is not None`) just to artificially pump test coverage numbers. Every test must validate genuine user-facing behavior, contract boundaries, deterministic outputs, or security invariants.
* **Strict Zero-Heavy-Dependency Core**: The base installation (`pip install asha-ai`) must stay remarkably fast, lightweight, and dependency-minimal (requiring only `pydantic`, `tiktoken`, `requests`, `click`, `pypdf`, `python-docx`). Heavy ML frameworks (PyTorch, Transformers, Scikit-learn, LangChain, FastAPI) belong exclusively in optional install extras (`[hardened]`, `[anchor]`, `[fastapi]`). Sub-second cold import times are a core feature.
* **Non-Mutating & Safe Defaults**: Proxies and client wrappers (`SecureLLM`, `wrap_llm`, `patch_context`) must never mutate underlying client objects in place. Default profiles fail closed on critical security violations.
* **Semantic & Syntactic Fidelity**: Optimization and preprocessing must never strip code formatting, destroy user intent, or inject hallucinated assistant boilerplate into user prompts.
* **Documentation is Non-Negotiable**: Code without clear, accurate documentation is incomplete. If developers cannot clearly understand how to use a feature from the documentation, the feature is not ready to be merged.

---

## 2. The 5-Step Developer Roadmap

We ask all contributors to follow this progressive 5-step roadmap. Master the fundamentals and secure the foundation before proposing scaling layers.

```
┌────────────────────────────────────────────────────────────────────────┐
│  Step 1: Understand Deeply  ──▶  Step 2: Optimize Core Drop-Ins       │
│                                           │                            │
│  Step 4: Real-World Cases   ◀──  Step 3: Fortify ANCHOR (Make it Real) │
│          & Mandatory Docs                 │                            │
│                   │                                                    │
│                   ▼                                                    │
│  Step 5: Innovate, Scale & Drop Ambitious Ideas                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Understand the Library & Real-World Use
Before writing code or proposing changes, invest time in understanding how ASHA works and how developers use it in production:

1. **Inspect the Architecture**:
   - Public facade: [src/asha/__init__.py](file:///s:/Projects/privySHA/project/src/asha/__init__.py)
   - Core drop-in utilities: [src/asha/utils/dropin.py](file:///s:/Projects/privySHA/project/src/asha/utils/dropin.py) (`process`, `sanitize`, `optimize`)
   - Transparent client proxying: [src/asha/integrations/llm_wrap.py](file:///s:/Projects/privySHA/project/src/asha/integrations/llm_wrap.py) (`SecureLLM`)
   - Multi-turn privacy vault: [src/asha/core/security/pii_vault_session.py](file:///s:/Projects/privySHA/project/src/asha/core/security/pii_vault_session.py) (`PIIVault`)
   - RAG indirect injection scanner: [src/asha/core/security/rag_guard.py](file:///s:/Projects/privySHA/project/src/asha/core/security/rag_guard.py) (`scan_rag_context`)
   - Agent governance runtime: [src/asha/runtime/anchor/](file:///s:/Projects/privySHA/project/src/asha/runtime/anchor/) (`anchor`, `AnchorRuntime`)
2. **Experience the Live Control Center**:
   Run the interactive developer showcase to see ASHA operating on live LLMs and offline simulations:
   ```bash
   pip install -e ".[demo]"
   streamlit run examples/showcase/app.py
   ```
3. **Review Established Contracts**:
   Explore [tests/contracts/](file:///s:/Projects/privySHA/project/tests/contracts/) and [tests/use_cases/](file:///s:/Projects/privySHA/project/tests/use_cases/) to see how existing public APIs are asserted and validated.

---

### Step 2: Optimize the Core Drop-Ins
The drop-in functions (`process()`, `sanitize()`, `optimize()`, and their asynchronous equivalents `process_async()`, `sanitize_async()`, `optimize_async()`) are the primary front doors of ASHA. They must be bulletproof, lightning fast, and rock solid:

* **Execution Speed & Latency**:
  - Keep CPU overhead minimal (target < 5ms for standard prompt payloads).
  - Eliminate redundant string allocations, unneeded regex compilations, and memory leaks on high-concurrency paths.
* **Fidelity & Honest Metrics**:
  - `optimize()` must report genuine token savings. Compression algorithms must preserve prompt instructions, variable names, and code syntax without degradation.
  - `sanitize()` must accurately detect and mask PII (emails, phone numbers, credit cards, IP addresses, custom enterprise tokens) with deterministic token replacements.
* **Strict Mode Fail-Closed Guarantee**:
  - In `Profile.STRICT()` mode, inputs exhibiting critical injection attacks, delimiter breakouts, or malicious payloads must raise `ASHAProcessingError` immediately.
* **Maintain the Zero-Heavy-Dependency Guarantee**:
  - Drop-in hot paths must NEVER trigger imports of heavy frameworks (e.g. PyTorch, Transformers, spaCy, LangChain).
  - Check imports with `pytest tests/architecture/test_boundaries.py`.

---

### Step 3: Fortify ANCHOR - Make It Strong, Concrete, and Real
ANCHOR ([src/asha/runtime/anchor/](file:///s:/Projects/privySHA/project/src/asha/runtime/anchor/)) is ASHA's mission-aware autonomous agent governance runtime.

> **CRITICAL DIRECTIVE**: ANCHOR must never be an abstract promise or governance "in name only." It must be a concrete, hardened runtime shield that actively prevents autonomous agents from going rogue in production environments.

Contributors working on ANCHOR should focus on making its defenses real and resilient:

* **Action Guard & Parameter Normalization**:
  - Move beyond naive tool-name matching. Accurately deserialize, sanitize, and normalize tool parameters (`kwargs`, JSON strings, CLI arguments).
  - Ensure tool calls strictly comply with the compiled mission contract (`ALLOW`, `WARN`, `BLOCK`, `REVIEW`) and cannot be bypassed via argument smuggling.
* **Chain & Transition Guards**:
  - Detect dangerous multi-hop action sequences (e.g., `read_file` -> `write_temp` -> `send_email`) that collectively violate mission boundaries even if each individual tool call appears benign.
* **Memory & Plan Guards**:
  - Defend agent scratchpads and long-term memory against indirect prompt injection payloads aimed at subverting agent goals across turns.
  - Validate planned execution steps before execution begins.
* **Execution Sandboxing**:
  - Harden process isolation (`isolation="hard"` / subprocess) and in-process execution hooks (`isolation="auto"`). Prevent arbitrary file system escapes and unauthorized socket connections.
* **Real-World Framework Battle-Testing**:
  - Validate all adapter integrations ([CrewAI](file:///s:/Projects/privySHA/project/docs/anchor.md#crewai), [LangChain](file:///s:/Projects/privySHA/project/docs/anchor.md#langchain), [LangGraph](file:///s:/Projects/privySHA/project/docs/anchor.md#langgraph), [LlamaIndex](file:///s:/Projects/privySHA/project/docs/anchor.md#llamaindex), [MCP](file:///s:/Projects/privySHA/project/docs/anchor.md#mcp), [AutoGen](file:///s:/Projects/privySHA/project/docs/anchor.md#autogen)) against real-world adversarial prompts, unexpected schema payloads, and headless CI execution.

---

### Step 4: Build for Real-World Use Cases & Mandatory Documentation
Every enhancement, fix, or optimization must be anchored to a concrete, production-grade scenario.

#### A. Authentic Use-Case Testing
* Add end-to-end integration tests in [tests/use_cases/](file:///s:/Projects/privySHA/project/tests/use_cases/) modeling how real engineers deploy ASHA:
  - Multi-turn conversational sessions with token vaulting and rehydration (`test_pii_vault_session.py`, `test_secure_llm_workflow.py`).
  - Production RAG retrieval pipeline protection against poisoned chunks (`test_rag_context_guard.py`).
  - Autonomous agent budget bounding and infinite loop defense (`test_agent_cost_guard.py`).
  - Non-invasive context patching and client lifecycle management (`test_patch_context.py`).
* **Zero tolerance for fake tests**: Tests that only assert `assert result is not None` or mock away the entire component will be rejected. Test actual state transitions, masked values, exceptions, and security verdicts.

#### B. Documentation is Mandatory
* **A feature without documentation is treated as broken and will not be merged.**
* Every PR touching functionality must include:
  1. **Accurate Python Docstrings**: Clear Google-style docstrings with complete type hints, parameter descriptions, return value schemas, and exceptions raised.
  2. **Updated Markdown Documentation**: Update relevant documentation pages under [docs/](file:///s:/Projects/privySHA/project/docs/) (e.g. `anchor.md`, `integrations.md`, `security.md`, `optimization.md`, `api-reference.md`).
  3. **Real, Runnable Examples**: Provide practical, copy-pasteable code snippets that run without modification.
  4. **Changelog & Migration Notes**: For breaking or noteworthy behavioral changes, update [CHANGELOG.md](file:///s:/Projects/privySHA/project/CHANGELOG.md).

---

### Step 5: Innovate, Scale & Drop Ambitious Ideas
**Once the foundation is solid, concrete, and thoroughly documented, we want you to dream big.**

We actively encourage contributors to drop as many creative ideas as possible to evolve and scale ASHA:
* **International & Enterprise PII Scanners**: Patterns for EU GDPR, HIPAA, UK NHS, and APAC identity numbers.
* **Streaming Security Scanners**: Real-time token-by-token injection defense for streaming LLM responses.
* **Advanced Sandboxing**: Docker, Podman, and bubblewrap (`bwrap`) backend isolation engines for ANCHOR.
* **Observability & Enterprise Export**: OpenTelemetry AI security spans, Datadog/Splunk SIEM integrations, and structured audit logs.
* **Framework Ecosystem**: New adapters for emerging agentic runtimes and MCP server implementations.

💡 **How to propose ideas:** Open a discussion thread under [GitHub Discussions](https://github.com/AjayRajan05/ASHA/discussions) with the `[RFC]` prefix. Outline the motivation, proposed API surface, dependency impact, and real-world use case.

---

## 3. Local Development Setup

### 1. Prerequisites
- Python **3.10+**
- Git

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/AjayRajan05/ASHA.git
cd ASHA

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install editable core with dev dependencies
pip install -e ".[dev]"
```

### 3. Verify Setup with Tests
```bash
# Run the fast test suite
pytest tests -q

# Run architecture boundary tests (mandatory before any PR)
pytest tests/architecture/test_boundaries.py -q

# Run real-world use-case test suite
pytest tests/use_cases/ -q
```

### 4. Interactive Showcase (Optional)
```bash
pip install -e ".[demo]"
streamlit run examples/showcase/app.py
```

---

## 4. Architectural Boundaries

To preserve sub-second import performance and prevent spaghetti dependencies, our architecture enforces strict unidirectional boundaries (validated by `tests/architecture/test_boundaries.py`):

| Layer | Path | Must NEVER Import From |
|---|---|---|
| **Core** | `src/asha/core/` | `runtime`, `integrations`, `compat` |
| **Runtime** | `src/asha/runtime/` | `integrations`, `compat` |
| **Types** | `src/asha/types/` | `compat`, `runtime`, `integrations` |
| **Utils** | `src/asha/utils/` | `compat` |

* **Zero Legacy Code**: Never reintroduce deprecated `Pipeline` abstractions or monkey-patching hacks into core modules.
* **Clean Surface**: Keep public exports clean and organized in [src/asha/__init__.py](file:///s:/Projects/privySHA/project/src/asha/__init__.py).

---

## 5. Code Quality & Formatting

* **Formatting & Linting**:
  ```bash
  flake8 src/asha --select=F401,F824
  ```
* **Type Checking**:
  ```bash
  mypy src/asha --ignore-missing-imports
  ```
* **Documentation Build**:
  ```bash
  pip install -e ".[docs]"
  mkdocs build --strict
  ```
* **Clean Tree Hygiene**:
  Never commit model weights (`*.joblib`, `*.pkl`), training datasets, scratch logs, or local API keys. Verify using:
  ```bash
  python scripts/check_public_surface.py
  ```

---

## 6. Pull Request Checklist

Before opening a pull request, verify each of the following:

- [ ] **Real-world focus**: The PR addresses a genuine, concrete production use case.
- [ ] **Drop-in health**: Core drop-ins (`process`, `sanitize`, `optimize`) remain lightweight, fast (< 5ms text target), and non-mutating.
- [ ] **ANCHOR reliability**: Any ANCHOR changes are concrete and battle-tested (not just superficial names or stubs).
- [ ] **Authentic testing**: Includes meaningful tests in `tests/use_cases/` or the appropriate test directory (no tautological "test-to-pass" assertions).
- [ ] **Documentation is complete**: All new/updated APIs have complete docstrings and updated guides in `docs/`.
- [ ] **No dependency creep**: No heavy ML or framework dependencies added to base `dependencies` in `pyproject.toml`.
- [ ] **Boundaries validated**: `pytest tests/architecture/test_boundaries.py` passes cleanly.
- [ ] **Full test suite passes**: `pytest` succeeds locally.
- [ ] **Public surface check passes**: `python scripts/check_public_surface.py` succeeds with zero errors.
- [ ] **Clean git hygiene**: No secrets, `.env` files, or private binary artifacts committed.

---

## 7. Community & Getting Help

* **Architecture & Ideas**: [GitHub Discussions](https://github.com/AjayRajan05/ASHA/discussions): pitch RFCs, brainstorm features, and discuss v1.0.0 milestones.
* **Bug Reports**: [GitHub Issues](https://github.com/AjayRajan05/ASHA/issues): submit bugs with minimal reproducible scripts.
* **Community Chat**: [ASHA Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw): chat with fellow contributors.
* **Security Disclosures**: Please review [SECURITY.md](SECURITY.md) for responsible disclosure procedures (do not open public issues for vulnerabilities).
* **Code of Conduct**: Respectful collaboration is mandatory. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

