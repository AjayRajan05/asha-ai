# Integrations

How to wrap LLM clients, protect request fields across providers, and attach ASHA middleware to common frameworks.

---

## Transparent Client Proxying (`SecureLLM` / `wrap_llm`)

ASHA provides transparent, non-mutating client proxying via `SecureLLM` (or `wrap_llm()`). It wraps any existing LLM client instance using attribute delegation and method interception. The underlying client object is **never mutated in-place**, preserving 100% API compatibility.

```python
from asha import SecureLLM, Profile
import openai

# Drop-in proxy wrapping with automatic PII scrubbing & response rehydration
raw_client = openai.OpenAI()
client = SecureLLM(raw_client, profile=Profile.BALANCED(), rehydrate_pii=True)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Email executive.board@corp.com the Q3 audit."}],
)
# 1. Provider only receives sanitized prompt (email masked)
# 2. Returned response has original email restored via session-scoped PIIVault
```

---

## Provider Protection Matrix & Protected Request Fields

The table below details exactly which client methods and request payload fields are intercepted, sanitized, and rehydrated for each supported provider:

| Provider / SDK | Methods Intercepted | Request Fields Protected | Attachment Handling | Response Fields Rehydrated (`rehydrate_pii=True`) | Status & Supporting Test ID |
|----------------|---------------------|--------------------------|---------------------|---------------------------------------------------|------------------------------|
| **OpenAI / Azure OpenAI** (`openai>=1.0.0`) | `client.chat.completions.create`<br>`client.completions.create`<br>`client.chat.completions.create` (async)<br>`client.create` | `messages` (all message dicts: `role`, `content` as string or list of text parts), `prompt`, `input`, `text`, `query`, `system` | Extracts local file paths / bytes in `attachments`, `file`, `files`, `document`, `documents`; extracts PDF/DOCX/text, masks PII into vault, and folds into messages | `response.choices[].message.content`<br>`response.choices[].text`<br>Streaming chunk generators (`stream=True`) | **Verified**<br>`test_openai_chat_completions_multi_turn_and_rehydration`<br>`test_openai_multi_part_content_scrubbing`<br>`test_openai_async_chat_completions_rehydration` |
| **Anthropic** (`anthropic>=0.25.0`) | `client.messages.create`<br>`client.messages.complete`<br>`client.messages.create` (async) | `messages` (multi-turn conversation lists, string `content` or list of content blocks), `system` (top-level system prompt), `prompt` | Multi-modal text / document blocks sanitized; raw attachment kwargs extracted and folded | `response.content[].text`<br>Streaming events | **Verified**<br>`test_anthropic_messages_create_sanitizes_user_and_system`<br>`test_anthropic_async_messages_rehydration` |
| **Google Gemini** (`google-genai` & `google-generativeai`) | `client.generate_content`<br>`client.models.generate_content`<br>`client.chats.create` / `send_message`<br>Positional prompt arg (`client.generate_content("...")`) | Positional string `prompt` (`args[0]`), `contents`, `prompt`, `messages` | Local document paths in `contents` or attachments parsed and masked | `response.text`<br>`response.candidates[].content.parts[].text` | **Verified**<br>`test_gemini_generate_content_positional_and_rehydration`<br>`test_gemini_generate_content_keyword_args` |
| **Ollama** (`ollama`) | `client.generate`<br>`client.chat`<br>`SecureLLM(client).chat(...)` | `prompt` (positional or kwarg), `messages` (list of message dicts), `system` | Multi-modal and attachment keys extracted and masked | `response["response"]`<br>`response["message"]["content"]`<br>`response.message.content` | **Verified**<br>`test_ollama_generate_keyword_prompt_and_dict_rehydration`<br>`test_ollama_chat_messages_and_dict_rehydration` |
| **Hugging Face** (`transformers` / InferenceClient) | `pipeline.__call__`<br>`client.text_generation`<br>`client.chat_completion`<br>`client.generate` | Positional prompt (`args[0]`), `prompt`, `messages`, `text_inputs` | File paths coerced to text and scrubbed | String outputs, `response[0]["generated_text"]` | **Verified**<br>`test_adapters_mock.py`<br>`test_hf_adapter.py` |
| **llama.cpp / vLLM / Local Endpoints** | Standard OpenAI-compatible client API (`client.chat.completions.create`) | `messages`, `prompt`, `input` | File attachments extracted and folded | `choices[].message.content`, `choices[].text` | **Verified**<br>Tested via OpenAI client proxy fixtures |

### Status Vocabulary & Verification Criteria

- **Verified:** Implementation and automated test suite in `tests/use_cases/test_provider_workflows.py` verify that requests are intercepted, sensitive values are scrubbed before provider dispatch, and returned responses are rehydrated without data leakage.
- **Partial:** Coverage applies to documented text/document formats; non-text multimodal streams (e.g., raw binary audio or video without transcription) require external transcription before evaluation.
- **Not covered:** Model tool execution output (data returned *by* tools executed on the client machine) is governed at dispatch time by the **ANCHOR Runtime** (`evaluate_action_request`), not by input prompt sanitizers.

---

## Request Field Scrubbing & Lifecycle

When a wrapped client makes an API call:

1. **Pre-processing**:
   - Positional prompt arguments (`args[0]`) and keyword prompt arguments (`prompt`, `contents`, `content`, `system`) are registered with the session's `PIIVault` (when `rehydrate_pii=True`), then processed through ASHA's security engine (`process()`).
   - In `messages` lists, each message dictionary is inspected. Plain string `content` and multi-part content items `[{"type": "text", "text": "..."}]` are masked.
   - Any raw file attachments (`attachments`, `file`, `document`) are extracted (supporting PDF, DOCX, and plaintext via lightweight parsers), masked, and folded into user messages so raw sensitive documents never leak to external model providers.
2. **Fail-Closed Policy Enforcement**:
   - Under `Profile.STRICT()`, any detected adversarial prompt injection (e.g., system prompt override, jailbreak pattern) immediately raises `ASHAProcessingError`. The call is blocked **before** the underlying provider SDK is called.
3. **Post-processing & Rehydration**:
   - The provider response is checked for format constraints via `OutputGuard`.
   - When `rehydrate_pii=True`, session-scoped placeholders (`[EMAIL]_xxxxxxxx`, `[PHONE]_xxxxxxxx`, etc.) are mapped back to their original values in the returned object, giving downstream code the complete data while guaranteeing zero exposure to the third-party LLM.

---

## Code Examples by Provider

### 1. OpenAI (Sync and Async)

```python
from asha import SecureLLM, Profile
from openai import OpenAI, AsyncOpenAI

# Synchronous client
client = SecureLLM(OpenAI(), profile=Profile.BALANCED(), rehydrate_pii=True)
res = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Contact jane.doe@corp.com at (555) 234-5678"}],
)
print(res.choices[0].message.content)

# Asynchronous client
async_client = SecureLLM(AsyncOpenAI(), profile=Profile.BALANCED(), rehydrate_pii=True)
async_res = await async_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Audit account #12345 for analyst@bank.com"}],
)
```

### 2. Anthropic

```python
from asha import SecureLLM, Profile
import anthropic

client = SecureLLM(anthropic.Anthropic(), profile=Profile.BALANCED(), rehydrate_pii=True)
msg = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="Internal escalation line: 555-0199",
    messages=[{"role": "user", "content": "Send Q3 results to cfo@acme.com"}],
)
print(msg.content[0].text)
```

### 3. Google Gemini

```python
from asha import SecureLLM, Profile
from google import genai

client = SecureLLM(genai.Client(), profile=Profile.BALANCED(), rehydrate_pii=True)

# Works with positional prompt arguments
resp = client.generate_content("Analyze confidential records for patient SSN 123-45-6789")
print(resp.text)
```

### 4. Ollama & Local Models

```python
from asha import SecureLLM, Profile
import ollama

client = SecureLLM(ollama, profile=Profile.BALANCED(), rehydrate_pii=True)

# Generate API
res = client.generate(model="llama3", prompt="Contact developer@local-domain.org")
print(res["response"])

# Chat API
chat_res = client.chat(
    model="llama3",
    messages=[{"role": "user", "content": "Deploy update for admin@local.net"}],
)
print(chat_res["message"]["content"])
```

---

## auto_patch (Global Monkey-Patching)

```python
from asha.integrations import auto_patch, disable_auto_patch

# Experimental global patching across installed SDKs
auto_patch(mode="strict")
# ... application code ...
disable_auto_patch()
```

> [!WARNING]
> `auto_patch()` globally monkey-patches installed SDK modules (`openai`, `anthropic`, `google.generativeai`, `transformers`). In production, prefer explicit `SecureLLM(client)` for scoped, non-mutating client control.

---

## Framework Middleware

```bash
pip install asha-si[integrations]
# or framework-specific: asha-si[fastapi], asha-si[langchain], etc.
```

| Framework | Middleware Module | Description |
|-----------|-------------------|-------------|
| **FastAPI** | `asha.integrations.fastapi` | Request/response middleware validating prompt safety and masking inbound PII |
| **Flask** | `asha.integrations.flask` | WSGI middleware for Flask route protection |
| **Django** | `asha.integrations.django` | Django middleware class sanitizing LLM endpoints |
| **LangChain** | `asha.integrations.langchain` | Runnable / Chain wrapper intercepting prompt inputs |
| **LlamaIndex** | `asha.integrations.llamaindex` | Node post-processor and query engine filter |
| **Instructor** | `asha.integrations.composition_strategy` | Composable helper for structured extraction |
| **OpenTelemetry** | `asha.integrations.otel` | Distributed tracing spans for PII detection and governance verdicts |

---

## Related

- [API reference](api-reference.md)
- [Comparison: ASHA-SI vs Presidio, LLM Guard, NeMo Guardrails](comparison.md)
- [Security & Threat Model](security.md)
- [ANCHOR Runtime](anchor.md)
- [Tutorial](index.md)
