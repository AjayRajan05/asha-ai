# Experimental features

!!! warning "Experimental: API may change"
    Everything on this page is **experimental** in ASHA v0.4.2. Expect breaking changes before 1.0 without a long deprecation cycle. Prefer the stable drop-in API (`process`, `sanitize`, `optimize`, `wrap_llm`). Install with `pip install asha-ai`. See [status.md](status.md).

    Calling these APIs emits ``ASHAExperimentalWarning`` at runtime.

---

## AshaFit / local advisor

!!! warning "Experimental: API may change"
    `recommend_local_model` and `Agent(local_model=...)` are preview APIs.

Recommends local LLMs for a prompt workload and hardware profile.

```bash
pip install asha-ai[local-advisor]
pip install asha-ai[local-advisor-gpu]   # optional NVIDIA VRAM detection
```

```python
from asha.runtime.local_advisor.advisor import recommend_local_model

report = recommend_local_model(
    prompts=[
        "My email is john@company.com - analyze this dataset.",
        "Write a Python function to validate API keys.",
    ],
    mode="strict",
    top=3,
)

print(report.top_pick.model_id)
print(report.top_pick.ollama_pull_name)
```

CLI:

```bash
asha recommend --prompt "Analyze dataset" --gpu "RTX 4090"
asha recommend --prompts ./benchmarks/sample_prompts.json --top 3
```

### Agent integration

```python
from asha import Agent

agent = Agent(
    model="llama3",
    local_model="auto",
    sample_prompts=["Summarize this report."],
)
```

### wrap_llm + local selection

```python
from asha.integrations import wrap_llm

client = wrap_llm(
    ollama_client,
    auto_select_local_model=True,
    sample_prompts=["Your typical app prompts..."],
)
```

Scoring combines hardware fit, workload fit, privacy requirements, and estimated speed. Exact weights and catalog contents may change.

---

## Smart routing (`SmartRoutingAdapter`)

!!! warning "Experimental: API may change"
    Task-based routing via `routing_config` is experimental. The standalone `ModelRouter` class was removed in v0.4.1.

```python
from asha import Agent

agent = Agent(
    routing_config={
        "chat": "gpt-4o-mini",
        "analysis": "gpt-4o",
        "code": "gpt-4o",
    },
)

response = agent.run("Analyze Q1 revenue", task_type="analysis")
```

`task_type` selects which entry in `routing_config` to use.

### Fallback providers

```python
agent = Agent(
    model="gpt-4o-mini",
    fallback_providers=[
        {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
        {"provider": "ollama", "model": "llama3"},
    ],
)

# Prefer explicit fallback when you need it:
# agent.run_with_fallback("prompt")
```

### Provider auto-detection

When `provider` is omitted, `Agent` infers from the model name (`gpt-*` → openai, `claude-*` → anthropic, and similar patterns). Treat this heuristics table as best-effort.

---

## Model gateway

!!! warning "Experimental: API may change"
    Advanced adapter construction and gateway-style options beyond basic `wrap_llm` / `Agent` usage are experimental.

### wrap_llm (recommended entry)

```python
from asha.integrations import wrap_llm
import openai

client = wrap_llm(openai.OpenAI(), mode="balanced")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Data from john@example.com"}],
)
```

Import from **`asha.integrations`**, not the package root.

### Providers

| Provider | Extra | Env var |
|----------|-------|---------|
| OpenAI | `asha-ai[openai]` | `OPENAI_API_KEY` |
| Anthropic | `asha-ai[anthropic]` | `ANTHROPIC_API_KEY` |
| Gemini | `asha-ai[gemini]` | `GOOGLE_API_KEY` |
| Grok | none | `GROK_API_KEY` |
| Ollama | none | local server |
| HuggingFace | `asha-ai[transformers]` | none |
| Mock | none | testing only |

### Agent

```python
from asha import Agent

agent = Agent(model="gpt-4o-mini", privacy=True)
response = agent.run("Analyze Q1 sales")

# No API key:
agent = Agent(model="mock")
```

### AdapterFactory

```python
from asha.runtime.adapters.factory import AdapterFactory

adapter = AdapterFactory.create(provider="openai", model="gpt-4o-mini")
adapter = AdapterFactory.create(provider="mock")
```

### auto_patch

Global SDK monkey-patching. Prefer `wrap_llm()` for new code.

```python
from asha.integrations import auto_patch

auto_patch(mode="balanced")
```

---

## Related

- [Status](status.md)
- [Tutorial](tutorial.md)
- [API reference](api-reference.md)
- [Integrations](integrations.md)
