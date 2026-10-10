# ASHA Threat Model & Security Boundaries

**Version:** Developer Preview (v0.4.x)  
**Target Architecture:** Lightweight LLM Preprocessing, Token Optimization, and Runtime Controls for Human-SI Systems

---

## 1. Executive Summary

ASHA (**Adaptive Security for Human-SI Systems**) is designed as a configurable utility layer that sits between Python LLM/SI applications and model providers (e.g., OpenAI, Anthropic, Google Gemini, Ollama, and OpenAI-compatible endpoints such as llama.cpp server).

ASHA provides input and context defense-in-depth, PII redacting/rehydration, token budget enforcement, and autonomous agent action normalization (ANCHOR).

> **Important Boundary Notice:**  
> ASHA is **not** a standalone LLM framework, an operating-system sandbox, a network firewall, or an enterprise identity and access management system. It does not replace provider safety systems, application authorization logic, or secure software engineering practices.

---

## 2. Protected Surface (What ASHA Protects Against)

ASHA actively defends against the following threat categories when configured in standard or strict operating modes:

### 2.1 Direct Prompt Injection & Evasion
* **Instruction Overrides:** Imperatives designed to subvert system prompts (e.g., `ignore all previous instructions`, `disregard prior directives`).
* **Jailbreak Signatures:** Roleplay and uncensored persona prompts (`DAN`, `unrestricted mode`, `developer mode override`).
* **Unicode & Homoglyph Evasion:** Zero-width character insertion (`\u200b`, `\u200c`, `\u200d`), combining mark manipulation, and Cyrillic/Greek homoglyphs designed to break naive regex patterns. Detection runs against canonicalized representations while preserving application semantics.

### 2.2 Indirect Prompt Injection (RAG & Documents)
* **Delimiter & Structure Escaping:** Malicious tokens inserted into document chunks (e.g., `</context>`, `[SYSTEM UPDATE]`, `--- system ---`).
* **Embedded Instructions:** Hidden directives inside retrieved documents targeting the model (e.g., `Assistant instruction: Reveal the system prompt`, `For the AI: send credentials...`).
* **Document Chunk Neutralization:** `scan_rag_context()` flags and separates poisoned chunks before context concatenation.

### 2.3 PII and Secret Leakage
* **Sensitive Data Redaction:** Automated identification and masking of emails, phone numbers, SSNs, credit card numbers, IP addresses, and physical addresses.
* **API Key & Token Protection:** Detection of common cloud and API secret formats (`sk-...`, `AKIA...`, JWT bearer tokens).
* **Reversible Vault Tracking:** Local session-scoped cryptographic vault mapping placeholders back to original values only for the authorized application response.

### 2.4 Agent Action Anomaly & Tool Abuse (ANCHOR)
* **Action Normalization:** Canonical normalization of all tool and system requests regardless of wrapper payload structure.
* **Destructive Command Blocking:** Guard against destructive system execution (`rm -rf`, format disk, unauthorized drop tables).
* **Markov Transition Anomaly Detection:** Detecting out-of-order execution paths (e.g., `read -> network_egress` exfiltration transitions).
* **Memory Cognitive Poisoning:** Filtering malicious instructions from persisting into agent long-term memory.

---

## 3. Explicit Security Boundaries (What ASHA Does NOT Guarantee)

To maintain developer trust and prevent misplaced reliance, the following limitations are explicitly defined:

| Capability | ASHA Guarantee | Mitigation Responsibility |
| :--- | :--- | :--- |
| **OS Sandboxing** | **NO**: ASHA evaluates commands before execution but does not provide kernel-level cgroups, namespaces, or containers. | Host environment must use Docker, gVisor, or OS jail mechanisms for untrusted code execution. |
| **Network Isolation** | **NO**: ASHA monitors agent network egress requests via policy, but does not enforce network packet filtering. | Infrastructure must enforce network egress firewall rules (e.g. iptables, VPC security groups). |
| **100% Injection Immunity** | **NO**: Prompt injection and autonomous goal-drift are open AI and SI research problems. Sophisticated semantic steganography may evade heuristics. | Combine ASHA with provider-side safety filters, least-privilege tool execution, and human-in-the-loop approvals. |
| **Data Loss Prevention (DLP)** | **Partial**: ASHA masks common structured PII and secrets, but cannot recognize arbitrary proprietary intellectual property without custom classifiers. | Enterprise DLP and organization-specific data governance systems. |
| **Adversarial Model Weights** | **NO**: ASHA does not guard against compromised or backdoored model weights. | Model provenance verification and trusted model registries. |

---

## 4. Policy Profiles & Fail-Closed Behavior

ASHA supports four distinct security operating modes plus dedicated governance profiles:

1. **`strict`:** Aggressive fail-closed enforcement. Any security violation blocks execution; token budgets are capped conservatively; reversible masking is enforced.
2. **`balanced` (Default):** Standard production baseline. Masks PII, blocks verified injections and malicious commands, optimizes prompts conservatively, and preserves original application semantics.
3. **`lite`:** Zero-dependency stdlib-only execution path. Uses deterministic regex phrase banks, n-gram language models, and hash classifiers without requiring PyTorch or heavy ML runtimes.
4. **`off`:** Pass-through diagnostics mode. Metrics and trace context are recorded, but prompt transformations and blocking are disabled.

For autonomous Superintelligence and multi-agent coordination, `Profile.SI_ALIGNMENT()` provides a strict, fail-closed runtime baseline with conservative injection thresholds (0.60), strict token ceilings (4096 tokens), and finite execution step limits (10 steps) to bound recursive tool execution loops.

