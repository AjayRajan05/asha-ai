# ASHA Control Center: Developer Preview Showcase

**Adaptive Security for Human-SI Systems (ASHA)**  
*Interactive Developer Laboratory, Transparent Execution Console & Observability Suite*

---

## 1. What is ASHA Control Center?

**ASHA Control Center** is a production-quality, developer-oriented Streamlit application that showcases the real capabilities of the ASHA library.

It is **NOT** a static mockup, fake dashboard, or marketing demonstration. Every pipeline stage, security check, PII masking operation, token optimization calculation, and agent governance rule executes directly against the installed `asha` Python library.

### Core Architecture Guarantee: Complete Transparency
> **See exactly what enters ASHA, what ASHA changes, why it changes it, what reaches the LLM, what the LLM returns, and what ASHA allows to happen next.**

```text
USER INPUT
    ↓
ASHA ENTRY
    ↓
SANITIZE (Prompt Injection & Delimiters)
    ↓
PII PROTECTION (Reversible Masking Vault)
    ↓
COMPILE (Format & Prompt Assembly)
    ↓
OPTIMIZE (Entropy / Compression / Preservation)
    ↓
POLICY (Deterministic Guardrails)
    ↓
LLM / PROVIDER (Gemini, OpenAI, Anthropic, Ollama, Local)
    ↓
RESPONSE VALIDATION (Output Guard & PII Rehydration)
    ↓
ASHA RESULT
    ↓
USER OUTPUT
```

---

## 2. Installation

Install the library with showcase and provider dependencies:

```bash
# Clone the repository
git clone https://github.com/AjayRajan05/asha-ai.git
cd asha-ai

# Install in editable mode with development/showcase dependencies
pip install -e .
pip install streamlit plotly requests
```

---

## 3. How to Launch

Launch the application using Streamlit:

```bash
streamlit run examples/showcase/app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 4. Provider Configuration

Configure upstream LLM provider credentials via environment variables or a `.env` file in the repository root.

| Provider | Environment Variable | Default / Notes |
| :--- | :--- | :--- |
| **Google Gemini** | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Modern `google-genai` client |
| **OpenAI** | `OPENAI_API_KEY` | Official OpenAI v1 API |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet / Haiku |
| **Ollama** | `OLLAMA_HOST` | Defaults to `http://localhost:11434` |
| **llama.cpp** | `LLAMA_CPP_BASE_URL` | Defaults to `http://localhost:8080/v1` |
| **Local Demo** | *(None)* | **100% Offline** deterministic mock provider |

> **Security Invariant:** ASHA Control Center **never displays, logs, or exports raw API keys**. All credentials are masked as `[REDACTED_SECRET]` in audit traces and configuration tables.

---

## 5. Application Navigation & Features

### 5.1 Overview
- **Dashboard**: Real session metrics aggregated from live executions (Requests, PII Events, Tokens Saved, Blocked Threats, Average Latency).
- **Runtime Status**: Active memory usage, installed version (`v0.4.2`), and execution mode.
- **Provider Status**: Connectivity table displaying configured vs unconfigured providers.

### 5.2 Playground
- **LLM Playground**: 3-panel split console (`User Input` | `Live Pipeline` | `Provider Output`).
- **Transparency Modes**:
  - `Normal`: Standard developer output and clean execution summaries.
  - `Transparent`: Step-by-step diffs and intermediate string representations at each stage.
  - `Forensic`: Low-level dataclass inspection, token distributions, and raw object metadata.
- **Process Pipeline**: Deconstructed breakdown of `asha.process()`.
- **API Explorer**: Dynamic signature introspection and live runner for all top-level ASHA APIs (`process`, `sanitize`, `optimize`, `wrap_llm`, `Agent`, `ANCHOR`, `PolicyConfig`).

### 5.3 Security Labs
- **PII Protection Lab**: Interactive reversible masking using `PIIVault`. View detected entities (`PERSON`, `EMAIL`, `PHONE`), masked tokens (`[EMAIL_001]`), and response rehydration.
- **Prompt Injection Lab**: Test direct instruction overrides, system prompt disclosure, multi-turn context hijacking, and role confusion (DAN). Inspect canonicalization and confidence scores.
- **Unicode Security Lab**: Character-by-character codepoint inspector revealing zero-width spaces (`\u200b`), zero-width joiners, and Cyrillic/homoglyph obfuscation.
- **Long Context Lab**: Configurable synthetic document generator (1K to 20K tokens) testing "needle-in-a-haystack" injection detection.
- **RAG Security Lab**: Multi-document indirect injection scanner inspecting retrieved chunks before context window assembly.

### 5.4 Optimization Lab
- **Token Optimization**: Real measurements of token reduction via ASHA optimizer.
- **Constraint Rejection Demo**: Demonstrates strict preservation of negations ("Do NOT send..."), numbers, and JSON schemas. When a candidate fails semantic invariants, ASHA cleanly rejects it and reports an honest **0 tokens saved**.

### 5.5 Agents & ANCHOR Control Room
- **ANCHOR Control Room**: Runtime agent governance firewall. Establishes immutable `MissionContract` boundaries and gates tool execution.
- **Action Governance**: Validates action type normalization (mapping `bash_command`, `shell`, and `tool_call` to canonical events to prevent evasion).
- **Memory Security**: `MemoryGuard` firewall protecting against cognitive poisoning and unprivileged scope writes.
- **Intent Drift**: Multi-step trajectory simulator monitoring alignment decay (e.g. flight search escalating to identity document exfiltration).

### 5.6 Provider Gateway
- **Google Gemini**: Modern `google-genai` integration supporting `gemini-2.5-flash` and `gemini-2.0-flash`.
- **OpenAI & llama.cpp**: Support for cloud models and local OpenAI-compatible HTTP servers.
- **Ollama**: Local daemon ping and model querying.
- **wrap_llm() Proxy**: Demonstrates programmatic verification that wrapped client instances are **never mutated** (`client is not wrapped_client`).

### 5.7 Observability & Telemetry
- **Live Trace**: Expandable forensic timeline of every execution event with microsecond timestamps and duration breakdowns.
- **Request History**: Session-level history table with sorting and filtering.
- **Replay Engine**: Re-executes historical prompts from scratch with fresh trace IDs for deterministic comparative benchmarking.
- **JSON Export**: Export sanitized audit logs with zero credential leakage.

---

## 6. What Works Offline? (No API Keys Required)

The demo is **fully functional without any third-party API keys or remote network access**:
- The **Local Demo Provider** executes realistic, deterministic responses locally.
- **PII Detection & Masking** runs 100% locally via regex and spaCy/statistical models.
- **Sanitization & Prompt Injection Detection** evaluates on the CPU in under 5 milliseconds.
- **Unicode Normalization & Inspection** runs entirely in Python standard library memory.
- **ANCHOR Mission Governance & Memory Firewall** evaluates contracts in local memory.
- **Token Optimization & Preservation Validation** executes locally using `tiktoken`.

---

## 7. Security & Developer Preview Limitations

To maintain absolute security and transparency:
1. **No Destructive Commands**: The demo will never execute destructive shell commands (`rm -rf`, `DROP TABLE`). ANCHOR runs dangerous actions through a safe simulated execution sandbox.
2. **No Secret Leakage**: API keys, auth headers, and bearer tokens are automatically redacted in traces, downloads, and UI elements.
3. **No 100% Security Claims**: ASHA provides defense-in-depth and probabilistic guardrails. It is not an OS-level kernel sandbox and does not replace transport-layer encryption or IAM controls.
4. **Developer Preview**: Features marked as experimental or in-progress are transparently labeled in the UI.

---

## 8. Running Automated Showcase Tests

Run the dedicated showcase test suite:

```bash
pytest tests/showcase/ -v
```

All 17 tests validate:
- Trace creation, event sequencing, and secret redaction
- Session metric arithmetic and accumulation
- Safe recursive object serialization (preventing circular loops)
- Preset scenario loading and execution through real ASHA functions
- Zero credential leakage in provider status checks
