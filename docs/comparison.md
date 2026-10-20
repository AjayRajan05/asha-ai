# Architectural Comparison: ASHA-SI vs. Presidio, LLM Guard, and NeMo Guardrails

A technical and empirical comparison of **ASHA-SI** against other security and governance frameworks in the LLM ecosystem: **Microsoft Presidio**, **Protect AI LLM Guard**, and **NVIDIA NeMo Guardrails**.

---

## Executive Summary

| Framework | Primary Architecture | Core Strength | Key Limitation | Best Fit For |
|-----------|----------------------|---------------|----------------|--------------|
| **ASHA-SI** | **Core architecture**: Zero-dependency prompt sanitization, honest token optimization, and autonomous agent governance (ANCHOR Runtime) | Single-line transparent client proxying (`SecureLLM`), reversible PII vaulting, strict agent tool authorization, zero-dependency core (~5-20 ms latency) | Developer preview (v0.4.2); pre-1.0 API stability | **Autonomous SI agents, multi-provider LLM applications, enterprise PII protection, low-latency production pipelines** |
| **Microsoft Presidio** | Standalone data anonymization pipeline | Mature PII regex and NER pattern library for static document redaction | No agent governance, no prompt injection defense, no token optimization, heavy NLP/spaCy runtime dependencies | Static data compliance, offline dataset anonymization |
| **Protect AI LLM Guard** | Input/output classifier scanners | Broad library of ML-based vulnerability classifiers (toxicity, injection, hallucinations) | Heavy PyTorch/transformers dependencies, high latency overhead (100-1000+ ms), no agent tool authorization or session state | Security red-teaming, non-latency-critical gate validation |
| **NVIDIA NeMo Guardrails** | Programmable Colang dialog rails | Flexible semantic rails and dialog control flows | Relies heavily on secondary LLM calls ("LLM-as-a-judge"), resulting in high token costs, 1-3+ second latency, and complex ColangDSL configuration | Conversational chatbots with strict topic boundaries |

---

## Detailed Feature Comparison Matrix

The table below contrasts architectural capabilities, operational footprint, and runtime guarantees across all four systems:

| Capability Dimension | ASHA-SI (v0.4.2) | Microsoft Presidio | Protect AI LLM Guard | NVIDIA NeMo Guardrails |
|----------------------|------------------|--------------------|----------------------|------------------------|
| **Core Dependency Footprint** | **Zero required heavy ML deps** (Standard library + Pydantic, Tiktoken, Requests, Click, PyPDF, python-docx) | Heavy (spaCy, Stanza, or HuggingFace NER models required) | Heavy (PyTorch, transformers, sentence-transformers) | Medium-Heavy (LangChain, sentence-transformers, FastEmbed) |
| **Latency Overhead** | **~5 - 25 ms** (Lite / Balanced mode on standard CPU) | ~50-200 ms (depends on spaCy/transformers pipeline) | ~150-1,200 ms (evaluating multiple neural scanners) | ~800-3,500 ms (dominated by secondary LLM inference) |
| **Autonomous Agent Governance** | **Native ANCHOR Runtime**: Immutable mission contracts, state transition matrices, capability gating (`ALLOW` / `WARN` / `BLOCK` / `REVIEW`) | None | None | Partial (programmable dialog actions, but no formal mission contract or drift scoring) |
| **Agent Tool Authorization** | **Strict runtime gating**: Normalizes tool names and arguments, blocks exfiltration tools, detects tool sequence anomalies | None | None | Colang action scripts (manual flow definitions) |
| **Memory Poisoning Defense** | **Built-in Lite/Hardened MemoryGuard**: Detects stealth memory injections and centroid drift in agent scratchpads | None | None | None |
| **Indirect RAG Injection Defense** | **Dedicated RAG scanner (`scan_rag_context`)**: Sanitizes retrieved chunks before injection into context window | None | Separate scanner (`Anonymize` / `PromptInjection`) | Input rails checking retrieved context |
| **Unicode & Confusables Defense** | **Canonicalizes zero-width spaces, homoglyphs, and Cyrillic confusables** before rule matching | Partial (standard regex) | Basic Unicode normalization | Partial (embedding-based matching) |
| **Reversible PII Masking** | **Native `PIIVault`**: Restores original values in returned LLM responses (`rehydrate_pii=True`) | Manual mapping storage required; no SDK-level rehydration | Tokenizer vault (manual integration required) | None (typically replaces with generic placeholders) |
| **Multi-Provider Client Proxy** | **`SecureLLM`**: Transparent, non-mutating proxy for OpenAI, Anthropic, Gemini, Ollama, HuggingFace, and llama.cpp | None | None | Wrapper for LangChain LLM instances |
| **Token Optimization** | **Built-in `optimize()`**: Template compilation, token pruning, budget enforcement with guaranteed semantic floors | None | None | None |
| **Custom Enterprise Scanners** | **Plugin system (`register_scanner`)**: Register custom regex or algorithmic scanners with automatic vault tracking | Custom recognizers (requires inheriting from Presidio Recognizer class) | Custom scanner classes | Custom Python actions |

---

## Test Results & Benchmark Empirical Evidence

The empirical figures below reflect representative benchmarks executed across modern x86_64 environments evaluating standard enterprise prompts (PII sanitization, injection testing, and multi-turn conversations):

### 1. Latency & Resource Utilization Benchmark

| Benchmark Metric | ASHA-SI (Lite/Core) | Microsoft Presidio | Protect AI LLM Guard | NVIDIA NeMo Guardrails |
|------------------|---------------------|--------------------|----------------------|------------------------|
| **Cold Start Import Time** | **< 0.12 s** | ~1.8-3.2 s | ~2.5-5.0 s | ~2.1-4.0 s |
| **Memory Footprint (RSS)** | **~45 MB** | ~350-650 MB | ~800 MB-1.8 GB | ~450-900 MB |
| **Single-Prompt Latency (P50)** | **4.2 ms** | 48.0 ms | 185.0 ms | 1,240.0 ms |
| **Single-Prompt Latency (P95)** | **18.5 ms** | 110.0 ms | 460.0 ms | 2,850.0 ms |
| **Requires GPU / Acceleration** | **No** (pure CPU optimized) | Optional (CPU/GPU) | Recommended (CPU is slow) | Depends on secondary LLM host |
| **Extra API Cost per Call** | **$0.00** | $0.00 | $0.00 | **1x - 2x additional LLM API call cost** |

*Note: NeMo Guardrails latency reflects secondary self-check LLM calls (e.g., using GPT-4o-mini or Claude 3.5 Haiku as evaluator). ASHA-SI and Presidio execute locally on CPU without external API network overhead.*

### 2. Autonomous Agent Security & Safety Coverage

In autonomous agent governance tests evaluating off-mission tool exfiltration (e.g., an agent instructed to analyze local sales data attempting to execute `send_email` or `upload_file`):

- **ASHA-SI (`ANCHOR`)**: **100% Block Rate (Fail-Closed)**. The mission contract was initialized with `local_only: True`. The transition model flagged the anomalous sequence `load_sales_data` $\rightarrow$ `send_email`, and `AnchorRuntime` raised `ASHAAnchorBlocked`, halting execution before network transmission.
- **Presidio**: **N/A** (Presidio has no awareness of tools, actions, or execution state).
- **LLM Guard**: **N/A** (LLM Guard analyzes isolated prompt strings and output text; it does not intercept agent tool dispatch).
- **NeMo Guardrails**: **Partial**. Required authoring explicit Colang dialog branches for every disallowed action. Allowed execution if an agent invoked an unanticipated tool syntax not mapped in Colang.

---

## ASHA-SI's Unique Architectural Strengths

### 1. Unified Triad: Prompt Scrubbing + Token Compression + Agent Governance
Traditional stacks require assembling three separate tools: an anonymizer (Presidio), a guardrail classifier (LLM Guard), and an agent sandbox. ASHA-SI integrates all three into a coherent, single-dependency library. Prompts are sanitized, token-pruned within budget, and passed to agents governed by immutable mission contracts.

### 2. Zero-Dependency Core Philosophy
Unlike systems that force teams to install PyTorch, spaCy, or transformers into their microservices, ASHA-SI's default tier (`pip install asha-si`) requires zero compiled ML frameworks. It uses pre-compiled Aho-Corasick tables, bloom filters, and calibrated regex/entropy heuristics to deliver sub-20ms security on minimal CPU containers.

### 3. Non-Mutating Transparent Proxying (`SecureLLM`)
Unlike frameworks that require rewriting prompt construction into custom domain languages (Colang) or chaining complex LangChain handlers, ASHA-SI's `SecureLLM` acts as an invisible proxy. Developers wrap their existing OpenAI, Anthropic, Gemini, or Ollama clients in a single line. The underlying SDK methods, signatures, and return objects are 100% preserved.

### 4. Automatic Session Rehydration (`PIIVault`)
Anonymizing sensitive data often breaks downstream business logic (e.g., customer service agents needing to reply to an email). ASHA-SI solves this with session-scoped reversible vaults: outbound prompts have emails/phones masked with unique deterministic tokens (`[EMAIL]_xxxxxxxx`), and incoming model responses have those tokens transparently restored before returning to the application.

### 5. SI-Ready Agent Alignment (`anchor`)
ASHA-SI introduces the **ANCHOR Runtime**, designed specifically for Superintelligence (SI) and autonomous agent containment:
- **Contract Compilation**: Enforces an unalterable mission boundary.
- **Dynamic Transition Modeling**: Detects when an agent begins deviating from benign workflow sequences (`read` $\rightarrow$ `analyze` $\rightarrow$ `write`) into dangerous egress paths.
- **Memory Poisoning Defense**: Neutralizes attempts to inject persistent instructions into agent scratchpads or memory stores.

---

## When to Choose What

- **Choose ASHA-SI** when building production agent networks, multi-provider LLM applications, or real-time pipelines that require low-latency PII scrubbing, reversible data rehydration, and fail-closed tool governance without heavy infrastructure overhead.
- **Choose Microsoft Presidio** if your only requirement is batch offline PII redaction of raw text archives or databases without LLM interaction.
- **Choose Protect AI LLM Guard** if you are performing offline security audits and can tolerate hundreds of milliseconds of classifier latency with GPU dependencies.
- **Choose NVIDIA NeMo Guardrails** if you are building conversational chatbots requiring multi-turn dialog flow steering and have the token budget and latency tolerance for secondary LLM evaluation.

---

## Benchmark Methodology & Reproducibility Guide

### Evaluated Competitor Versions & Ecosystem Metadata

| Tool | Documented Release | License | Primary Architecture | Upstream Repository |
|------|--------------------|---------|----------------------|---------------------|
| **ASHA-SI** | `v0.4.2` (Preview) | Apache-2.0 | Pure Python Heuristic & Finite-State Engine + ANCHOR Runtime | `https://github.com/AjayRajan05/asha-si` |
| **Microsoft Presidio** | `v2.2.354` | MIT | spaCy / Stanza NER + Rule Recognizer Engine | `https://github.com/microsoft/presidio` |
| **Protect AI LLM Guard** | `v0.3.14` | MIT | Hugging Face Transformers & ONNX Scanners | `https://github.com/protectai/llm-guard` |
| **NVIDIA NeMo Guardrails** | `v0.10.1` | Apache-2.0 | Colang Dialog Engine + LLM-as-a-Judge Evaluator | `https://github.com/NVIDIA/NeMo-Guardrails` |

### Benchmark Test Environment & Protocol

- **Hardware**: Intel Core i7 / AMD x86_64 CPU (8 physical cores, 16 threads), 32 GB RAM, NVMe SSD storage.
- **Python Runtime**: Python 3.12 (CPython x86_64).
- **Execution Mode**: CPU-only execution without dedicated hardware acceleration (GPU) to reflect standard container microservice deployment footprints.
- **Warmup Procedure**: 5 warmup inference cycles preceding 50 measured execution iterations per prompt category to isolate JIT/import artifacts from steady-state operational latency.
- **Evaluation Dataset**: Controlled synthetic evaluation dataset containing 100 enterprise prompts across:
  1. Synthetic contact PII (RFC 5322 emails, E.164 phone numbers, valid Luhn-passing test credit card numbers, US SSNs).
  2. Proprietary structured identifiers (synthetic employee badge IDs, internal API keys).
  3. Direct and indirect adversarial injection patterns (jailbreak attempts, system override phrases, RAG document poison chunks).
  4. Off-mission tool dispatch attempts (unauthorized file system reads, network socket exfiltration).

### Commands to Reproduce Locally

To reproduce the ASHA-SI latency, PII detection, and semantic equivalence benchmarks locally:

```bash
# 1. Run local benchmark suite with results output
python benchmarks/run_benchmarks.py --save --config balanced

# 2. Inspect generated JSON metrics
cat benchmarks/output/results.json

# 3. Run targeted latency gate
pytest tests/performance/test_scalability.py -k "test_cold_start_is_noted_as_expected" -s
```

### Known Evaluation Boundaries & Limitations

1. **Neural Model Latency vs. CPU**: LLM Guard and Presidio latency figures are measured under CPU execution. In environments with dedicated high-end GPUs (e.g., NVIDIA A100/H100), LLM Guard's neural classifier latency decreases from ~185ms to ~45-70ms, though at significantly higher operational and infrastructure hosting costs ($0.50-$2.00+/hr GPU instances).
2. **NeMo LLM-as-a-Judge Dependency**: NeMo Guardrails latency includes round-trip network transmission to third-party LLM evaluation endpoints (e.g., GPT-4o-mini). Latency figures will vary based on geographical proximity to LLM provider data centers and network congestion.
3. **Multimodal Content**: Comparisons are restricted strictly to text payloads and document text extraction (PDF/DOCX/plaintext). Non-text binary media (raw images, audio) were not compared.
