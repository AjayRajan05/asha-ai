# ASHA-AI Developer Preview Release Checklist

**Release Target:** ASHA-AI v0.4.x → Developer Preview  
**Branch:** `release/developer-preview`

---

## 1. Release Gates Status

### P0 Requirements (Must be 100% complete before release)
- [x] **Public API exports fixed:** `from asha import wrap_llm, PolicyConfig, ANCHOR, anchor, process, sanitize, optimize, Agent, SecureLLM, Profile, scan_rag_context` all importable. Verified in `tests/public_api/test_public_api_imports.py`.
- [x] **Semantic preservation in `process()`:** Rewrote `PromptCompiler` (`src/asha/core/compiler/prompt_compiler.py`) to preserve original user nouns, instructions, code, and payloads verbatim, eliminating synthetic assistant templates. Verified in `tests/test_prompt_semantic_preservation.py`.
- [x] **Honest token optimization metrics:** Rebuilt `MSDPCOptimizer` and `OptimizerEngine` so reported `tokens_saved` and `token_reduction_percentage` accurately reflect final output tokens, reporting 0 savings when prompt is reverted.
- [x] **`token_budget` propagation:** Propagated `token_budget` explicitly through `optimize()`, `PromptOptimizer`, and `MSDPCOptimizer`, enforcing budget ceilings.
- [x] **ANCHOR action normalization:** Fixed direct-action bypass vulnerability (`bash_command`, `shell`, `exec`, `terminal`). All actions normalized to tool evaluations before authorization checks. Destructive commands (`rm -rf`, `DROP TABLE`, format disk) denied. Verified in `tests/test_anchor_action_guard_normalization.py`.
- [x] **Unicode canonicalization & zero-width bypass prevention:** Added dual-view scanning in `injection_detector.py`, `security_layer.py`, and `rag_guard.py` using NFKC + zero-width strip + confusable folding. Verified in `tests/test_unicode_canonicalization_security.py`.
- [x] **Markov baseline model recalibration:** Recalibrated `read -> network_egress` transition probability to 0.02 and rebalanced `transition_matrix.json`. Verified in `tests/test_chain_guard_hardening.py`.
- [x] **RAG indirect injection defense:** Hardened `scan_rag_context()` with patterns for system updates, assistant instruction overrides, and exfiltration commands. Verified in `tests/security/test_rag_injection.py`.
- [x] **Memory poisoning defense:** Hardened `MemoryGuard` against adversarial prompt injections and cognitive poisoning. Verified in `tests/test_memory_guard_hardening.py` and `tests/test_memory_guard_lite.py`.

### P1 Requirements
- [x] **Modern Google GenAI SDK support:** Modernized `GeminiAdapter` (`src/asha/runtime/adapters/gemini_adapter.py`) with support for `google-genai` and `gemini-3.8-flash` / `gemini-2.5-flash`, safe response handling against safety filter exceptions, and usage metadata capture. Verified in `tests/integration/test_gemini_38_flash.py`.
- [x] **Ollama parameterization:** Added `base_url` parameterization, `OLLAMA_HOST` environment resolution, configurable timeout, and system prompt support in `OllamaAdapter`.
- [x] **llama.cpp positioning:** Supported llama.cpp via OpenAI-compatible server endpoint (`/v1`) through `OpenAIAdapter(base_url=...)` and `AdapterFactory`.
- [x] **Non-mutating `wrap_llm()`:** Delegated `wrap_llm()` through transparent `SecureLLM` proxy, leaving underlying client objects completely unmutated. Verified in `tests/test_wrapper_coverage.py` and `tests/test_wrap_llm_negative.py`.
- [x] **Threat model documentation:** Added `docs/security/threat-model.md` defining protected surfaces and explicit security non-guarantees.
- [x] **Positioning & documentation:** Updated `README.md` to reflect honest capabilities as a security/optimization layer rather than an infallible boundary.

---

## 2. Verification Summary

| Verification Target | Result | Details |
| :--- | :--- | :--- |
| `tests/public_api/test_public_api_imports.py` | PASS (2/2) | Public root exports (`wrap_llm`, `PolicyConfig`, `ANCHOR`, etc.) |
| `tests/public_api/test_root_exports.py` | PASS (9/9) | Complete frozen export set and lazy resolution validation |
| `tests/test_prompt_semantic_preservation.py` | PASS (7/7) | Intent, tasks, code, negations preserved verbatim |
| `tests/test_prompt_preservation.py` | PASS (18/18) | Optimizer structure & honest token reduction metrics |
| `tests/test_anchor_action_guard_normalization.py` | PASS (5/5) | Direct actions normalized and destructive commands denied |
| `tests/test_chain_guard_hardening.py` | PASS (8/8) | Baseline Markov chain transitions recalibrated |
| `tests/test_unicode_canonicalization_security.py` | PASS (5/5) | Zero-width & Cyrillic confusable injection detection |
| `tests/test_adapters_mock.py` | PASS (12/12) | Gemini, Grok, Claude, OpenAI, Ollama, llama.cpp endpoints |
| `tests/integration/test_gemini_38_flash.py` | PASS (4/4) | Gemini 3.8 Flash SDK integration & safety error handling |
| `tests/test_wrapper_coverage.py` | PASS (24/24) | Non-mutating wrap_llm client proxying |
| `tests/test_wrap_llm_negative.py` | PASS (10/10) | Fail-closed error handling and safety invariants |
| `tests/security/test_rag_injection.py` | PASS (5/5) | RAG indirect injection detection |
| `tests/test_async_optimize_sanitize.py` | PASS (7/7) | Async optimize and sanitize pipelines |
| `tests/test_instructor_integration.py` | PASS (6/6) | Instructor integration & injection defense |
| `Packaging (uv build)` | PASS | Wheel and sdist built cleanly without errors |
| `Performance Benchmarks` | PASS | Reproducible scripts in `benchmarks/` |
| `Full Suite (pytest -m "not integration")` | PASS (676 passed) | 0 test failures across entire codebase |


