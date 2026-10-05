# API Reference

**ASHA v0.4.2** - canonical signatures and import paths.

---

## Import map

```python
# Root (only these): v0.4.2 public API
from asha import process, sanitize, optimize, Agent, anchor
from asha import process_async, sanitize_async, optimize_async
from asha import SecureLLM, Profile              # v0.4.2 client proxy + presets
from asha import PIIVault                        # v0.4.2 reversible PII masking
from asha import scan_rag_context                # v0.4.2 RAG injection scanner
from asha import AgentCostGuard, AgentBudgetExceeded  # v0.4.2 agentic guard

# ANCHOR (also available from root)
from asha.runtime.anchor import anchor_any
from asha.runtime.anchor.adapters import anchor_crewai, anchor_langchain

# Subpackages
from asha.integrations import wrap_llm, auto_patch
from asha.runtime import PromptProcessor, ExecutionProfile
from asha.types import ProcessResult, SanitizeResult, OptimizeResult, AgentResult
from asha.core.policy_config import PolicyConfig, PolicyMode
from asha.utils.dropin import process_async, optimize_async, sanitize_async
from asha.utils.unmask import unmask
from asha.runtime.local_advisor.advisor import recommend_local_model
from asha.runtime.adapters.factory import AdapterFactory
from asha.compat.legacy_results import to_legacy_pipeline_dict
from asha.cli.terminal import render_security_report  # v0.4.2 rich/ANSI diagnostics
```

---

## Result types

```python
result = process("prompt")  # ProcessResult

result.output
result.original
result.degraded
result.degraded_reason
result.security       # SecurityInfo | None
result.metrics        # MetricsInfo | None
result.trace          # dict | None (trace=True)
result.diff           # str | None (debug=True)
str(result)           # == result.output
result.to_dict()      # legacy dict shape
```

---

## process()
```python
from asha import process
from asha.core.policy_config import PolicyConfig

result = process(
    prompt="...",
    mode="balanced",           # strict | balanced | lite | off
    token_budget=1200,
    trace=False
)
```

**Parameters:**
- `prompt` (`str | bytes | Path`): The input prompt or file to process.
- `mode` (`str`, optional): Policy mode. Options:
  - `strict`: Fail-closed - raises on total failure
  - `balanced`: Fail-open fallback (default)
  - `lite`: Minimal policy; fail-open
  - `off`: Passthrough
- `policy` (`PolicyConfig`, optional): Advanced runtime configuration (fields: `pii_mode`, `reversible`, `preserve_intent`, `security_level`, stage enable flags).
- `token_budget` (`int`, optional): Maximum tokens permitted before truncation (default: 1200).
- `trace` (`bool`, optional): If true, populates `result.trace` with internal execution steps.
- `debug` (`bool`, optional): If true, populates `result.diff` with a text diff.
- `max_retries` (`int`, optional): Number of retries for detector engines.
- `timeout_seconds` (`float`, optional): Max timeout for execution.
- `verbose` (`bool`, optional): Enables verbose console logs.
- `log_level` (`str`, optional): Logging level (`INFO`, `DEBUG`).
- `log_output` (`str`, optional): Output destination (`console`, etc).
- `log_file` (`str`, optional): Path to a log file.
- `include_legacy_detail` (`bool`, optional): Includes legacy dict schema.

**Returns:**
- `ProcessResult`: Contains `.output` (scrubbed string) and `.security` (detection summary).

---

## sanitize()

```python
from asha import sanitize

result = sanitize(
    prompt: str,
    mode: str = "balanced",
    *,
    policy: PolicyConfig | None = None,
    trace: bool = False,
    debug: bool = False,
    max_retries: int = 0,
    timeout_seconds: float | None = None,
    verbose: bool = False,
) -> SanitizeResult
```

---

## optimize()

```python
from asha import optimize

result = optimize(
    prompt: str,
    *,
    token_budget: int = 1200,
    trace: bool = False,
    debug: bool = False,
    timeout_seconds: float | None = None,
) -> OptimizeResult
```

Token compression only - no security or compile stages.

---

## wrap_llm()

```python
from asha.integrations import wrap_llm

client = wrap_llm(
    client,
    mode: str = "balanced",
    token_budget: int = 1200,
    auto_select_local_model: bool = False,
    sample_prompts: list[str] | None = None,
)
```

- `mode="off"` - no preprocessing
- Processing errors: strict raises; balanced degrades
- Wrap infrastructure errors raise when `mode != "off"`

---

## auto_patch()

```python
from asha.integrations import auto_patch
from asha.integrations.auto_patch import (
    get_patch_status,
    disable_auto_patch,
    enable_auto_patch,
)

auto_patch(mode: str = "strict", enable: bool = True, verbose: bool = False)
```

Globally patches installed SDKs. **Prefer `wrap_llm()`** for production.

---

## Async

```python
from asha.utils.dropin import process_async, sanitize_async, optimize_async

result = await process_async("prompt", mode="balanced")
```

---

## unmask()

```python
from asha.utils.unmask import unmask

restored = unmask(llm_output, masking_map)
```

Requires `policy=PolicyConfig(reversible=True)` during `process()` / `sanitize()`.

---

## Agent

```python
from asha import Agent

agent = Agent(
    model: str = "gpt-4o-mini",
    privacy: bool = True,
    token_budget: int = 1200,
    provider: str | None = None,
    fallback_providers: list[dict] | None = None,
    routing_config: dict[str, str] | None = None,
    timeout: int = 10,
    retries: int = 3,
    api_key: str | None = None,
    local_model: str | None = None,
    sample_prompts: list[str] | None = None,
)

# Returns str by default
response = agent.run(prompt, trace=False, task_type="chat")

# Returns AgentResult when trace=True
result = agent.run(prompt, trace=True)
```

`privacy=True` → internal `mode="strict"`. `privacy=False` → `mode="off"`.

---

## PromptProcessor

```python
from asha.runtime import PromptProcessor, ExecutionProfile

processor = PromptProcessor()
result = processor.run("prompt", mode="balanced", profile=ExecutionProfile(...))
```

---

## recommend_local_model() (experimental)

```python
from asha.runtime.local_advisor.advisor import recommend_local_model

report = recommend_local_model(
    prompts: list[str] | None = None,
    prompts_file: str | None = None,
    mode: str = "balanced",
    top: int = 5,
    gpu: str | None = None,
    vram_gb: float | None = None,
    cpu_only: bool = False,
    refresh_catalog: bool = False,
    preferred_quant: str | None = None,
    probe: bool = False,
)
```

!!! warning "Experimental: API may change"
    See [experimental-features.md](experimental-features.md).

---

## AdapterFactory (experimental)

```python
from asha.runtime.adapters.factory import AdapterFactory

adapter = AdapterFactory.create(provider="openai", model="gpt-4o-mini")
adapter = AdapterFactory.create(provider="mock")
adapter = AdapterFactory.create_smart_routing({"chat": "gpt-4o-mini"})
```

Providers: `openai`, `anthropic`, `gemini`, `ollama`, `huggingface`, `grok`, `mock`.

Smart routing and factory helpers may change (see [experimental-features.md](experimental-features.md)).

---

## Legacy dict

```python
from asha.compat.legacy_results import to_legacy_pipeline_dict

legacy = to_legacy_pipeline_dict(
    process("prompt", include_legacy_detail=True)
)
```

---

## CLI

| Command | Description |
|---------|-------------|
| `asha "prompt"` | Demo process |
| `asha quick-test` | Built-in tests |
| `asha examples` | Sample transformations |
| `asha benchmark` | Benchmark harness |
| `asha recommend` | AshaFit advisor |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` / `GEMINI_API_KEY` / `GROK_API_KEY` | Provider adapters / live demos |
| `ASHA_PROVIDER` / `ASHA_MODEL` | `Agent.from_env()` / adapter factory |
| `ASHA_TOKEN_BUDGET` / `ASHA_TIMEOUT` / `ASHA_RETRIES` | Agent / adapter defaults |
| `ASHA_PRIVACY` | `Agent.from_env()` privacy flag (`true`/`false`) |
| `ASHA_MODELS_DIR` | Use a local models tree (skip default cache layout) |
| `ASHA_MODELS_URL` | Override hardened-models download URL |
| `ASHA_MODELS_ZIP` | Local zip fallback for the models bundle |
| `ASHA_MODELS_CACHE` | Override cache directory (`~/.cache/asha/models` default) |
| `ASHA_DISABLE_MODEL_DOWNLOAD` | `1`: never fetch models (CI / air-gap) |
| `ASHA_DISABLE_ML` | Disable ML/NER paths where checked |
| `ASHA_THRESHOLDS_PATH` | Override packaged detector thresholds YAML |
| `ASHA_INJECTION_MODE` | Injection detector mode override |
| `ASHA_ENABLE_HF_SAFETY` | Opt-in Hugging Face safety path |
| `ASHA_OPTIMIZER_EMBEDDING` | Optimizer embedding mode hint |
| `ASHA_ANCHOR_INTERACTIVE` | ANCHOR interactive approval (`0`/`1`) |
| `ASHA_ANCHOR_WARN_POLICY` | ANCHOR warn policy (`strict` / `permissive` / …) |
| `ASHA_ANCHOR_TELEMETRY` / `ASHA_ANCHOR_TELEMETRY_PATH` | Opt-in ANCHOR telemetry |
| `ASHA_CACHE_DIR` | Local advisor catalog cache |

`mode` is **not** read from env: set per call. See also [`.env.example`](https://github.com/AjayRajan05/ASHA/blob/main/.env.example).

---

## Errors

| Exception | When |
|-----------|------|
| `ASHAProcessingError` | `mode="strict"` total failure |
| `TypeError` | Unknown kwargs on `process()` / `sanitize()` |
| `AttributeError` | Removed root exports (`wrap_llm`, `Pipeline`, etc.) |
| `AgentBudgetExceeded` | Agent exceeded step, token, or cycle budget (v0.4.2) |

See [migration-v0.4.md](migration-v0.4.md) and [deprecations.md](deprecations.md).

---

## SecureLLM *(v0.4.2)*

Single-line transparent proxy for any LLM client.

```python
from asha import SecureLLM, Profile
import openai

# One line: drop-in replacement with identical API
client = SecureLLM(openai.OpenAI(), profile=Profile.BALANCED())

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Email alice@corp.com the Q3 report"}]
)
# ↑ alice@corp.com is masked before going to OpenAI, re-hydrated in the response
```

**Parameters:**
- `client`: Any LLM client (OpenAI, Anthropic, Gemini, Ollama).
- `profile` (`Profile`, optional): Zero-config security preset. Defaults to `Profile.BALANCED()`.
- `mode` (`str`, optional): Policy mode string (`'strict'`, `'balanced'`, `'lite'`, `'off'`). Ignored if `profile` is provided.
- `token_budget` (`int`, optional): Override profile token budget.
- `rehydrate_pii` (`bool`, optional): Re-hydrate masked PII in LLM responses. Defaults to `True`.
- `vault` (`PIIVault`, optional): Existing vault for multi-session PII continuity.

**Intercepted methods:** `chat.completions.create`, `messages.create`, `generate_content`, `generate`, `create`.

---

## Profile *(v0.4.2)*

Zero-config security profile presets.

```python
from asha import Profile

# Factory classmethods
Profile.STRICT()      # Fail-closed, injection threshold 0.65, max 5 agent steps
Profile.BALANCED()    # Enterprise default: threshold 0.85, max 15 steps
Profile.PERMISSIVE()  # Audit/observe mode: never blocks, logs violations
Profile.DEVELOPER()   # Verbose + debug, rich terminal output, fail-open

# Custom profile
profile = Profile(
    mode="balanced",
    fail_closed=True,
    injection_threshold=0.80,
    token_budget=2048,
    max_agent_steps=10,
)
```

**Preset comparison:**

| Preset | Mode | Fail-closed | Threshold | Token Budget | Agent Steps |
|--------|------|-------------|-----------|--------------|-------------|
| `STRICT` | strict | ✅ | 0.65 | 1000 | 5 |
| `BALANCED` | balanced | ✅ | 0.85 | 2048 | 15 |
| `PERMISSIVE` | lite | ❌ | 0.95 | 4096 | 50 |
| `DEVELOPER` | balanced | ❌ | 0.80 | 2048 | 20 |

---

## PIIVault *(v0.4.2)*

Thread-safe, session-scoped reversible PII masking.

```python
from asha import PIIVault

vault = PIIVault()  # One vault per session/conversation

# Mask: replace PII with indexed tokens
masked = vault.mask("Email alice@corp.com or call 555-123-4567")
# → "Email <EMAIL_1> or call <PHONE_1>"

# Unmask: re-hydrate tokens in LLM responses
response_text = "I'll email <EMAIL_1> the report."
final = vault.unmask(response_text)
# → "I'll email alice@corp.com the report."

vault.summary()  # {'email': 1, 'phone': 1}
vault.clear()    # Wipe all vault entries
```

**Detected PII types:** `email`, `phone`, `credit_card`, `ssn`, `api_key`, `jwt_token`, `bearer_token`, `address`.

---

## scan_rag_context() *(v0.4.2)*

Indirect injection scanner for RAG retrieved documents.

```python
from asha import scan_rag_context

docs = retriever.get_relevant_documents(query)
result = scan_rag_context(docs, query=query)

if result.poisoned_count > 0:
    print(f"⚠️  {result.poisoned_count} poisoned chunks detected at indices: {result.poisoned_indices}")

# Use only verified safe chunks
safe_context = "\n\n".join(result.safe_documents)
```

**Parameters:**
- `documents` (`list[str] | list[dict]`): Retrieved text chunks or dicts with `text`/`page_content` key.
- `query` (`str`, optional): User query for relevance scoring.
- `include_sanitized` (`bool`, optional): Include sanitized versions of poisoned chunks. Defaults to `True`.
- `confidence_threshold` (`float`, optional): Minimum confidence to classify a chunk as poisoned. Defaults to `0.6`.

**Returns:** `RAGScanResult` with `.poisoned_indices`, `.safe_documents`, `.sanitized_documents`, `.poisoned_count`.

**Detected threats:** `delimiter_escape` (`</context>`, `[SYSTEM]`), `instruction_override` (classic injection), `exfiltration` (webhook/POST patterns).

---

## AgentCostGuard *(v0.4.2)*

Stateful agentic loop budget enforcement.

```python
from asha import AgentCostGuard, AgentBudgetExceeded

guard = AgentCostGuard(
    max_steps=10,         # Hard step limit
    max_tokens=8000,      # Cumulative token budget
    max_cycle_repeat=3,   # Max identical tool+args repetitions
    raise_on_limit=True,  # Set False for audit mode
)

for tool_name, tool_args in agent.run():
    try:
        guard.record_tool_call(tool_name, tool_args, token_count=150)
    except AgentBudgetExceeded as e:
        print(f"Budget exceeded: {e.budget_type} {e.current}/{e.limit}")
        break
    result = execute_tool(tool_name, tool_args)

stats = guard.stats()
print(f"Steps: {stats.total_steps}, Tokens: {stats.total_tokens}, Healthy: {stats.is_healthy}")
```

**Parameters:**
- `max_steps` (`int`, optional): Maximum execution steps. Defaults to 25.
- `max_tokens` (`int`, optional): Cumulative token budget. Set to `0` to disable. Defaults to 50,000.
- `max_cycle_repeat` (`int`, optional): Cycle detection threshold. Defaults to 3.
- `raise_on_limit` (`bool`, optional): Raise on violation vs. audit-only. Defaults to `True`.

**`AgentBudgetExceeded` attributes:** `.budget_type` (`'steps'`, `'tokens'`, `'cycle'`), `.current`, `.limit`.

